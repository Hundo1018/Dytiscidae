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

CPU by choice
-------------

The network is `n_obs -> hidden -> n_modes`, about 1,200 weights.  A CUDA launch
costs more than that arithmetic.  The rollouts it learns from are already
batched onto the GPU; this is not the part that wants an accelerator.
"""

from __future__ import annotations

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


def _mlp(n_in: int, n_hidden: int, n_out: int):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_out),
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
        self.actor = _mlp(self.n_obs, hidden, self.n_modes)
        self.critic = _mlp(self.n_obs, hidden, 1)
        # State-independent log-std, the usual choice for continuous control:
        # the policy learns how much to explore without having to predict it
        # from an observation that may not contain the answer.
        self.log_std = nn.Parameter(torch.full((self.n_modes,), -0.5))

    def distribution(self, obs):
        mean = torch.tanh(self.actor(obs))  # intent is bounded, like Policy.act
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs_np, *, deterministic: bool = False):
        """One decision, numpy in and numpy out, for use inside a rollout."""
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(obs_np, np.float32)).unsqueeze(0)
            dist = self.distribution(obs)
            a = dist.mean if deterministic else dist.sample()
            logp = dist.log_prob(a).sum(-1)
            v = self.critic(obs).squeeze(-1)
        return (a.squeeze(0).numpy().astype(float),
                float(logp.item()), float(v.item()))


@dataclass
class Trajectory:
    """One segment's worth of decisions from one machine."""

    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)
    terminal_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.obs)


class RolloutBuffer:
    """Trajectories from a whole generation, flattened into a PPO batch.

    Every trajectory carries a single terminal reward -- its segment competence.
    Advantages come from GAE over a reward that is zero everywhere except the
    last step, which is the honest translation of a score that is only defined
    for a whole episode.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma, self.lam = float(gamma), float(lam)
        self.trajectories: list[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        if len(traj):
            self.trajectories.append(traj)

    @property
    def n_transitions(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def build(self):
        """Flatten to (obs, act, logp, advantage, return) arrays."""
        O, A, L, ADV, RET = [], [], [], [], []
        for t in self.trajectories:
            n = len(t)
            rew = np.zeros(n)
            rew[-1] = t.terminal_reward
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
        return (np.concatenate(O), np.concatenate(A), np.concatenate(L),
                np.concatenate(ADV), np.concatenate(RET))


class SegmentCollector:
    """Records one segment's decisions, per machine, into trajectories.

    Handed to `rollout_batch`, which calls `record` at every control decision.
    The reward is not known until the segment has been scored, so `finish` is
    called afterwards with one competence per machine.
    """

    def __init__(self, k: int):
        self.k = int(k)
        self.live = [Trajectory() for _ in range(self.k)]

    def record(self, m: int, obs, act, logp, val) -> None:
        t = self.live[m]
        t.obs.append(np.asarray(obs, float))
        t.act.append(np.asarray(act, float))
        t.logp.append(float(logp))
        t.val.append(float(val))

    def finish(self, buffer: RolloutBuffer, rewards) -> None:
        """Attach each machine's segment competence and bank the trajectory."""
        for m, t in enumerate(self.live):
            if not len(t):
                continue
            t.terminal_reward = float(rewards[m])
            buffer.add(t)
        self.live = [Trajectory() for _ in range(self.k)]


def ppo_update(policy, buffer, *, lr: float = 1e-3, epochs: int = 10,
               minibatch: int = 2048, clip: float = 0.2,
               vf_coef: float = 0.5, ent_coef: float = 0.0,
               max_grad_norm: float = 0.5, target_kl: float = 0.015,
               optimiser=None) -> dict:
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
    """
    if not AVAILABLE:
        raise RuntimeError(f"torch unavailable: {UNAVAILABLE_REASON}")
    n = buffer.n_transitions
    if n < 2:
        return {"transitions": n, "skipped": True}

    obs, act, logp_old, adv, ret = buffer.build()
    obs = torch.as_tensor(obs)
    act = torch.as_tensor(act)
    logp_old = torch.as_tensor(logp_old)
    adv = torch.as_tensor(adv.astype(np.float32))
    ret = torch.as_tensor(ret.astype(np.float32))
    # Normalising advantages across the batch is what lets one policy learn from
    # morphologies whose competences live on different scales.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    opt = optimiser or torch.optim.Adam(policy.parameters(), lr=lr)
    idx = np.arange(n)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "n_batches": 0}

    stopped_early = False
    for _ in range(epochs):
        if stopped_early:
            break
        np.random.shuffle(idx)
        epoch_kl, epoch_batches = 0.0, 0
        for s in range(0, n, minibatch):
            b = torch.as_tensor(idx[s:s + minibatch].copy())
            dist = policy.distribution(obs[b])
            logp = dist.log_prob(act[b]).sum(-1)
            ratio = (logp - logp_old[b]).exp()
            a = adv[b]
            pi_loss = -torch.min(
                ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = ((policy.critic(obs[b]).squeeze(-1) - ret[b]) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

            kl = float((logp_old[b] - logp).mean().item())
            stats["pi_loss"] += float(pi_loss.item())
            stats["v_loss"] += float(v_loss.item())
            stats["kl"] += kl
            stats["n_batches"] += 1
            epoch_kl += kl
            epoch_batches += 1

        # Stop once this pass has moved the policy as far as it is allowed to.
        # Checked per epoch rather than per minibatch: a single minibatch's KL
        # is noisy enough that stopping on it would end most updates after one
        # step, which is the failure the raised rate was meant to fix.
        if target_kl and epoch_batches and epoch_kl / epoch_batches > target_kl:
            stopped_early = True

    k = max(stats["n_batches"], 1)
    return {
        "transitions": n,
        "trajectories": len(buffer.trajectories),
        "pi_loss": stats["pi_loss"] / k,
        "v_loss": stats["v_loss"] / k,
        "kl": stats["kl"] / k,
        "grad_steps": stats["n_batches"],
        "stopped_early": stopped_early,
        "skipped": False,
    }
