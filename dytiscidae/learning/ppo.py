"""PPO over a policy shared by every morphology in the search.

Why shared, and not one policy per candidate
--------------------------------------------

The obvious reading of "switch the controller optimiser to PPO" is to replace
the per-candidate (1+1)-ES with per-candidate PPO.  That does not work here, and
the reason is arithmetic rather than taste.

A candidate's policy sees about 7,350 transitions in its entire existence --
1,050 per Tier-1 evaluation across seven passes (one initial, six refinement).
PPO's smallest useful update is on the order of 2,048 samples, and MuJoCo-class
locomotion is usually quoted at 1e6-1e7.  At the measured 600 transitions per
second, buying 1e5 samples for a *single* candidate costs 167 s of exclusive
machine time; the last run evaluated 5,140 candidates, so doing that for each
would be 238 hours.  A (1+1)-ES is well matched to a 7k-sample budget.  PPO is
not a drop-in replacement at the same cost, it is a different cost class.

Sharing one policy across the whole search inverts the ratio.  A generation of
16 candidates over three domains contributes roughly 50,400 transitions to one
set of parameters, so a single generation is already a respectable PPO batch.

This is only possible because the interfaces are morphology-independent, which
they were before anyone planned for this: `TriphibianEnv.observation()` returns
a fixed-width vector with no joint-count-dependent term, and a policy emits
`n_modes` coefficients.  Everything that varies with the body -- the parameter
count P, which ranges from 16 to 49 -- is absorbed by the mobility basis, whose
`modes` matrix is `(n_modes, P)`.  So one network with one input width and one
output width can drive every machine in the archive, and the basis measured on
each body is what makes the same four coefficients mean the right thing on each.

Why the reward is sparse
------------------------

`TriphibianEnv._score_segment` is deliberately non-Markovian.  It divides time
fractions by the duration the segment was *asked* for rather than by the samples
recorded, and measures sink rate over the second half of the airborne stretch.
Its docstring documents two exploits the search found against earlier, denser
formulations -- machines with no lifting surface scoring 0.9 for flight by
terminating on the first step, and a design scoring 0.75 because the beach rose
to meet it.

A dense shaped reward invented here would be a third formulation, competing with
the one that was hardened against those exploits.  So the return is the segment
competence, delivered once at the end of the segment, and PPO gets it through
GAE like any other sparse-terminal task.  Sparse costs sample efficiency, and
sample efficiency is the thing sharing the policy just bought.

Two later changes widened the signal without touching that decision.  The
transitions now contribute trajectories too, rewarded with the same graded
crossing score `finalise_tier1` already folds into mission_fraction -- the part
of the mission the policy most needs to learn was the one rollout it never saw.
And the observation now carries the quantities the scorers read (depth error to
the target, contact, battery, stroke phase), so the value function has state to
regress a sparse return against rather than having to infer the mission from
body rates.

CPU by choice
-------------

The network is `n_obs -> hidden -> n_modes`, about 1,200 weights.  A CUDA launch
costs more than that arithmetic.  The rollouts it learns from are already
batched onto the GPU; this is not the part that wants an accelerator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # pragma: no cover - torch is an optional dependency
    import torch
    import torch.nn as nn
    AVAILABLE = True
    UNAVAILABLE_REASON = ""
except Exception as exc:  # pragma: no cover
    torch = None
    nn = object
    AVAILABLE = False
    UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def _layer(n_in: int, n_out: int, gain: float):
    """A linear layer with orthogonal weights and zero bias.

    Orthogonal initialisation keeps the singular values of every layer at
    ``gain``, so activations neither vanish nor saturate as they pass through
    the stack.  With ``tanh`` units the difference is not cosmetic: the default
    Kaiming-uniform initialisation puts a two-layer tanh trunk close to its
    saturated region at the start of training, where the derivative is nearly
    zero, and the policy spends its first updates escaping its own
    initialisation rather than learning.
    """
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.constant_(lin.bias, 0.0)
    return lin


def _mlp(n_in: int, n_hidden: int, n_out: int, out_gain: float = 1.0):
    # sqrt(2) is the tanh/ReLU gain; the small output gain starts the policy
    # near zero intent ("command nothing"), which is the same starting point a
    # per-candidate ``Policy`` gets from its zeroed weights.
    return nn.Sequential(
        _layer(n_in, n_hidden, math.sqrt(2.0)), nn.Tanh(),
        _layer(n_hidden, n_hidden, math.sqrt(2.0)), nn.Tanh(),
        _layer(n_hidden, n_out, out_gain),
    )


class SharedPolicy(nn.Module if AVAILABLE else object):
    """A Gaussian policy and a value head, shared by every morphology.

    The action is `n_modes` intent coefficients, the same quantity a per-machine
    `control.cpg.Policy` produces, so the two are interchangeable at the point of
    use and can be summed.
    """

    def __init__(self, n_obs: int, n_modes: int, hidden: int = 64):
        if not AVAILABLE:
            raise RuntimeError(f"torch unavailable: {UNAVAILABLE_REASON}")
        super().__init__()
        self.n_obs, self.n_modes = int(n_obs), int(n_modes)
        self.actor = _mlp(self.n_obs, hidden, self.n_modes, out_gain=0.01)
        self.critic = _mlp(self.n_obs, hidden, 1, out_gain=1.0)
        # State-independent log-std, the usual choice for continuous control:
        # the policy learns how much to explore without having to predict it
        # from an observation that may not contain the answer.
        self.log_std = nn.Parameter(torch.full((self.n_modes,), -0.5))
        # Running observation statistics.  Registered as buffers so they travel
        # in ``state_dict`` -- a normalisation that is not checkpointed makes
        # every stored weight mean something different after a resume.
        #
        # The observation is 27 channels on wildly different scales: body rates
        # divided by 5, a domain one-hot, a battery fraction, and eight
        # morphology channels.  An unnormalised first layer sees the largest of
        # them and is nearly blind to the rest.
        self.register_buffer("obs_mean", torch.zeros(self.n_obs))
        self.register_buffer("obs_var", torch.ones(self.n_obs))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    # ---------------------------------------------------------- normalisation

    def normalise(self, obs):
        return torch.clamp(
            (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)

    def observe(self, obs_np) -> None:
        """Fold a batch of observations into the running statistics.

        Called once per update, *after* the epochs, so that within one update
        every log-probability is computed under the same normalisation the
        rollout used.  Updating mid-update would make the importance ratio a
        comparison between two different functions.
        """
        x = torch.as_tensor(np.asarray(obs_np, np.float32))
        if x.ndim != 2 or x.shape[0] < 2:
            return
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = b_mean - self.obs_mean
        total = self.obs_count + b_n
        m2 = (self.obs_var * self.obs_count + b_var * b_n
              + delta.pow(2) * self.obs_count * b_n / total)
        self.obs_mean.copy_(self.obs_mean + delta * b_n / total)
        self.obs_var.copy_(m2 / total)
        self.obs_count.copy_(total)

    # ------------------------------------------------------------- the policy

    def latent(self, obs):
        """The pre-squash Gaussian over intent.

        The action is ``tanh`` of a sample from this, not a sample from a
        Gaussian whose *mean* has been squashed.  That was the previous shape
        and it was wrong in a way that mattered: with ``log_std = -0.5`` the
        standard deviation is 0.607, so a large fraction of every sample fell
        outside [-1, 1] -- and ``MobilityBasis.coeffs_for_twist`` clips its
        input to that interval before using it.  The policy was therefore
        scored on actions it had not taken, and the gradient it computed was the
        gradient of a distribution over a region the environment could not
        reach.
        """
        return torch.distributions.Normal(
            self.actor(self.normalise(obs)), self.log_std.exp())

    def value(self, obs):
        return self.critic(self.normalise(obs)).squeeze(-1)

    def log_prob(self, obs, act):
        """Log-density of an already-squashed action, with the tanh Jacobian."""
        base = self.latent(obs)
        a = act.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)
        return (base.log_prob(u).sum(-1)
                - torch.log1p(-a.pow(2) + 1e-6).sum(-1))

    def act(self, obs_np, *, deterministic: bool = False):
        """One decision, numpy in and numpy out, for use inside a rollout."""
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(obs_np, np.float32)).unsqueeze(0)
            base = self.latent(obs)
            u = base.mean if deterministic else base.sample()
            a = torch.tanh(u)
            logp = (base.log_prob(u).sum(-1)
                    - torch.log1p(-a.pow(2) + 1e-6).sum(-1))
            v = self.value(obs)
        return (a.squeeze(0).numpy().astype(float),
                float(logp.item()), float(v.item()))


@dataclass
class Trajectory:
    """One segment's worth of decisions from one machine."""

    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)
    #: Potential of each visited state, for shaping.  See ``potential_of``.
    phi: list = field(default_factory=list)
    terminal_reward: float = 0.0
    #: Which segment this came from, so its reward can be scaled against the
    #: others of its kind rather than against a water segment.
    tag: str = ""

    def __len__(self) -> int:
        return len(self.obs)


#: Observation channel indices this file depends on.  They are the layout of
#: ``TriphibianEnv.observation`` and the morphology context is appended after
#: them, so widening the observation does not move any of these.
_GRAV_Z, _DEPTH, _WET, _DEPTH_ERR, _CONTACT = 8, 9, 10, 14, 15


def potential_of(obs, domain: str) -> float:
    """A state's potential, for potential-based reward shaping.

    The reward this learner gets is one scalar per segment: 8.95M transitions
    over arch31 carried 55,668 reward values, one per 161 decisions.  Worse,
    84.1% of the variance of that scalar is explained by the *domain one-hot
    alone* -- a channel already in the observation -- and of what is left,
    roughly nine tenths is decided by which body was rolled out, which the
    observation could not see.  Under 2% of the reward's variance was
    attributable to anything a generation of PPO changed.

    Potential-based shaping (Ng, Harada & Russell 1999) is the one way to make
    that dense without changing what is optimal: adding ``gamma*Phi(s') -
    Phi(s)`` leaves the optimal policy of the original MDP untouched, whatever
    Phi is, because the sum telescopes.  So this cannot invent a new objective
    the way a hand-weighted dense reward would; it can only redistribute credit
    in time.

    Phi is built from the *same* quantities the segment score is built from --
    height above the surface and attitude in air, distance from the mission's
    depth target in water, ground contact and attitude on land -- so a step
    that moves toward what will be scored is credited when it happens rather
    than 161 steps later.
    """
    o = obs
    upright = -float(o[_GRAV_Z])          # +1 upright, -1 inverted
    if domain == "air":
        # Clear of the surface and the right way up.  ``_DEPTH`` is negative
        # above the water, so its negation rises as the machine climbs.
        return float(np.clip(-o[_DEPTH], -1.0, 1.0) + 0.5 * upright)
    if domain == "water":
        # Zero exactly at the target depth, and being wet at all is worth
        # something to a machine that has to get under the surface first.
        return float(-abs(o[_DEPTH_ERR]) + 0.25 * float(o[_WET]))
    if domain == "land":
        return float(o[_CONTACT] + 0.5 * upright)
    # Crossings: the state that matters is arriving upright and under control.
    return float(0.5 * upright)


class RolloutBuffer:
    """Trajectories from a whole generation, flattened into a PPO batch.

    Every trajectory carries a single terminal reward -- its segment competence
    -- plus a potential-based shaping term per step, which is dense and provably
    does not change what is optimal.  See ``potential_of``.

    Terminal rewards are standardised *within their own segment kind* before
    GAE.  Measured over arch31's 28,152 segments, mean competence was 0.635 in
    water against 0.137 in air and 0.089 on land, so an unscaled batch let the
    water segments set the gradient while the mission's binding constraint is
    land (the weakest medium in 75% of designs) and air (the other 25%).
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95,
                 shaping: float = 0.2):
        self.gamma, self.lam = float(gamma), float(lam)
        self.shaping = float(shaping)
        self.trajectories: list[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        if len(traj):
            self.trajectories.append(traj)

    @property
    def n_transitions(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def _terminal_scale(self) -> dict:
        """One scale per segment kind, so each contributes comparably."""
        by: dict = {}
        for t in self.trajectories:
            by.setdefault(t.tag, []).append(t.terminal_reward)
        return {k: max(float(np.std(v)), 0.05) for k, v in by.items()}

    def build(self):
        """Flatten to (obs, act, logp, advantage, return, value) arrays.

        The old value estimates come out too, because the value loss is clipped
        against them -- see ``ppo_update``.
        """
        O, A, L, ADV, RET, VAL = [], [], [], [], [], []
        scale = self._terminal_scale()
        for t in self.trajectories:
            n = len(t)
            rew = np.zeros(n)
            rew[-1] = t.terminal_reward / scale.get(t.tag, 1.0)
            if self.shaping > 0.0 and len(t.phi) == n:
                # gamma*Phi(s_{t+1}) - Phi(s_t), with Phi(terminal) = 0, which
                # is what makes the sum telescope and the optimum survive.
                phi = np.asarray(t.phi, float)
                nxt = np.concatenate([phi[1:], [0.0]])
                rew = rew + self.shaping * (self.gamma * nxt - phi)
            val = np.asarray(t.val, float)
            # GAE with a terminal bootstrap of zero: the segment is over, there
            # is no continuation to value.
            adv = np.zeros(n)
            last = 0.0
            for i in range(n - 1, -1, -1):
                nxt = val[i + 1] if i + 1 < n else 0.0
                delta = rew[i] + self.gamma * nxt - val[i]
                last = delta + self.gamma * self.lam * last
                adv[i] = last
            O.append(np.asarray(t.obs, np.float32))
            A.append(np.asarray(t.act, np.float32))
            L.append(np.asarray(t.logp, np.float32))
            ADV.append(adv)
            RET.append(adv + val)
            VAL.append(val)
        return (np.concatenate(O), np.concatenate(A), np.concatenate(L),
                np.concatenate(ADV), np.concatenate(RET), np.concatenate(VAL))


class SegmentCollector:
    """Records one segment's decisions, per machine, into trajectories.

    Handed to `rollout_batch`, which calls `record` at every control decision.
    The reward is not known until the segment has been scored, so `finish` is
    called afterwards with one competence per machine.
    """

    def __init__(self, k: int):
        self.k = int(k)
        self.live = [Trajectory() for _ in range(self.k)]

    def record(self, m: int, obs, act, logp, val, phi: float = 0.0) -> None:
        t = self.live[m]
        t.obs.append(np.asarray(obs, float))
        t.act.append(np.asarray(act, float))
        t.logp.append(float(logp))
        t.val.append(float(val))
        t.phi.append(float(phi))

    def finish(self, buffer: RolloutBuffer, rewards, tag: str = "") -> None:
        """Attach each machine's segment competence and bank the trajectory."""
        for m, t in enumerate(self.live):
            if not len(t):
                continue
            t.terminal_reward = float(rewards[m])
            t.tag = tag
            buffer.add(t)
        self.live = [Trajectory() for _ in range(self.k)]


def ppo_update(policy, buffer, *, lr: float = 1e-3, epochs: int = 10,
               minibatch: int = 2048, clip: float = 0.2,
               vf_coef: float = 0.5, ent_coef: float = 0.01,
               max_grad_norm: float = 0.5, target_kl: float = 0.015,
               lr_fraction: float = 1.0, optimiser=None) -> dict:
    """One PPO update over everything the generation collected.

    The defaults were raised after the first full run that used this.  At
    lr=3e-4, 4 epochs and a 4096 minibatch, arch30 pushed 2.93 million
    transitions through 336 updates and reported a mean KL of 0.000818 against a
    maximum of 0.003535 -- an order of magnitude below the 0.01-0.02 a PPO update
    normally aims for.  Nine thousand samples over a 4096 minibatch is three
    minibatches, so four epochs bought twelve gradient steps per generation and
    about four thousand across the whole run.  The policy was not failing to
    learn; it was barely being asked to.

    Raising the rate without a bound on how far one update may move is the
    standard way to destroy a policy, so `target_kl` stops the epoch loop once
    the batch's mean KL exceeds it.  That makes the rate safe to raise: updates
    that would have overshot end early instead, and the diagnostics say so via
    `stopped_early`.

    Returns the diagnostics worth logging: how many transitions it saw, the
    clipped-surrogate and value losses, and the approximate KL.  A run that
    reports a rising KL with a flat loss is a run whose learning rate is wrong,
    and that is not visible from the search's own fitness numbers.

    Implementation hygiene, added as one arm rather than one at a time
    -----------------------------------------------------------------

    ``lr_fraction`` linearly anneals the rate over the run; ``ent_coef`` is no
    longer zero; the value loss is clipped against the old estimate; the
    optimiser's epsilon is 1e-5 rather than torch's 1e-8; the trunk is
    orthogonally initialised and the policy is a genuinely squashed Gaussian
    (see ``SharedPolicy.latent``); and the observation is normalised by running
    statistics the policy carries.

    These are coupled -- annealing changes the KL curve, orthogonal
    initialisation changes where the tanh units start, an entropy bonus changes
    exploration -- so measuring them one per run would cost five runs to learn
    less than one arm against arch33 does.

    The diagnostic this resolves: arch33's policy hit the KL ceiling on 88% of
    updates by generation 450, and its trajectory over 46 snapshots was
    significantly anti-correlated step to step (cosine -0.061 +/- 0.019, and
    -0.128 +/- 0.016 over the last third, with displacement growing as
    n^0.388).  A *fixed* learning rate alone produces that signature, so it was
    never evidence that the gradient direction rotates -- which is why the
    anneal is in this arm and no conclusion was drawn from the old runs.
    """
    if not AVAILABLE:
        raise RuntimeError(f"torch unavailable: {UNAVAILABLE_REASON}")
    n = buffer.n_transitions
    if n < 2:
        return {"transitions": n, "skipped": True}

    obs_np, act, logp_old, adv, ret, val_old = buffer.build()
    obs = torch.as_tensor(obs_np)
    act = torch.as_tensor(act)
    logp_old = torch.as_tensor(logp_old)
    adv = torch.as_tensor(adv.astype(np.float32))
    ret = torch.as_tensor(ret.astype(np.float32))
    val_old = torch.as_tensor(val_old.astype(np.float32))
    # Normalising advantages across the batch is what lets one policy learn from
    # morphologies whose competences live on different scales.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Adam's default epsilon is 1e-8, which is small enough relative to these
    # gradients that the effective step size varies by orders of magnitude
    # between parameters; 1e-5 is the value the PPO implementations that
    # reproduce published results use.
    opt = optimiser or torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    # Linear anneal.  A fixed rate is what produces a policy that keeps moving
    # as far as the KL bound allows and arrives nowhere.
    lr_now = float(lr) * float(np.clip(lr_fraction, 0.0, 1.0))
    for group in opt.param_groups:
        group["lr"] = lr_now
    idx = np.arange(n)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "entropy": 0.0,
             "clipfrac": 0.0, "n_batches": 0}

    stopped_early = False
    for _ in range(epochs):
        if stopped_early:
            break
        np.random.shuffle(idx)
        epoch_kl, epoch_batches = 0.0, 0
        for s in range(0, n, minibatch):
            b = torch.as_tensor(idx[s:s + minibatch].copy())
            logp = policy.log_prob(obs[b], act[b])
            ratio = (logp - logp_old[b]).exp()
            a = adv[b]
            pi_loss = -torch.min(
                ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            # Clipped value loss: the critic may not move further from its own
            # previous estimate than the policy is allowed to, which stops one
            # unusually large return from dragging the baseline the advantages
            # of every other morphology are measured against.
            v = policy.value(obs[b])
            v_clipped = val_old[b] + (v - val_old[b]).clamp(-clip, clip)
            v_loss = 0.5 * torch.max((v - ret[b]) ** 2,
                                     (v_clipped - ret[b]) ** 2).mean()
            # A squashed Gaussian has no closed-form entropy, so this is the
            # one-sample estimator -- unbiased, and the only term the bonus
            # needs a gradient through.
            ent = -logp.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

            kl = float((logp_old[b] - logp).mean().item())
            stats["pi_loss"] += float(pi_loss.item())
            stats["v_loss"] += float(v_loss.item())
            stats["kl"] += kl
            stats["entropy"] += float(ent.item())
            stats["clipfrac"] += float(
                ((ratio - 1.0).abs() > clip).float().mean().item())
            stats["n_batches"] += 1
            epoch_kl += kl
            epoch_batches += 1

        # Stop once this pass has moved the policy as far as it is allowed to.
        # Checked per epoch rather than per minibatch: a single minibatch's KL
        # is noisy enough that stopping on it would end most updates after one
        # step, which is the failure the raised rate was meant to fix.
        if target_kl and epoch_batches and epoch_kl / epoch_batches > target_kl:
            stopped_early = True

    # The normalisation moves only now, so that every ratio above was computed
    # under the statistics the rollout itself used.
    policy.observe(obs_np)

    k = max(stats["n_batches"], 1)
    return {
        "transitions": n,
        "trajectories": len(buffer.trajectories),
        "pi_loss": stats["pi_loss"] / k,
        "v_loss": stats["v_loss"] / k,
        "kl": stats["kl"] / k,
        "entropy": stats["entropy"] / k,
        "clipfrac": stats["clipfrac"] / k,
        "lr": lr_now,
        "grad_steps": stats["n_batches"],
        "stopped_early": stopped_early,
        "skipped": False,
    }
