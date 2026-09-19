"""The shared PPO learner: does it run, does it learn, and do its own numbers mean anything?

Why this file is split into four parts
--------------------------------------

It used to be one function, and the first check that needed a built MuJoCo model
took the whole suite down with it -- including every check that tests only the
learner and needs no environment at all.  That is the wrong dependency: `PPO's
update logic is independent of the environment` is a claim this project's
architecture makes, and a test suite that cannot run the algorithm checks
without an environment is not testing that claim, it is contradicting it.

So:

* **unit** -- the policy, the buffer and the update, on synthetic observations.
  Needs torch and nothing else.  Runs on a machine with no GPU, no MuJoCo and no
  Mojo extension built.
* **algorithm** -- GAE, the discount, the shaping identity and the two scalars
  the update reports about itself, each against an independent reference
  computed here rather than against the implementation's own arithmetic.
* **state** -- what survives a checkpoint.  Also environment-free: `save_state`
  and `load_state` take a `SearchState`, not a rollout.
* **integration** -- the learner wired to the real batched evaluator.  Needs the
  GPU fluid extension; **skipped with its reason printed** when that is not
  importable, never silently and never by taking the rest down.

A test that only checks shapes would pass on a policy whose gradient is
disconnected from its loss, so the update checks train against rewards with
known optima and assert the policy moves toward them.

Run:  PYTHONPATH=. python tests/test_ppo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dytiscidae.learning.ppo import (AVAILABLE, UNAVAILABLE_REASON,  # noqa: E402
                                     RolloutBuffer, SharedPolicy, Trajectory,
                                     ppo_update)

N_OBS, N_MODES = 14, 4

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    """A check that did not run, rendered as itself.

    This existed as ``check(name, True, "SKIPPED: ...")``, which printed
    ``[ok  ]`` and was counted as a pass.  Measured on the runner CI actually
    uses -- numpy and torch, no MuJoCo -- that made "shared PPO checks passed"
    the summary of a run where two test functions never executed: 48 checks ran
    there against 56 here, and the missing eight said nothing about themselves.
    A skip is a third state and prints as one, and the success line carries the
    count, so the string a reader greps for cannot appear on a run that did not
    run everything.
    """
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


def _rollout(policy, rng, n_steps, reward_fn, tag: str = "", phi=None):
    t = Trajectory()
    for i in range(n_steps):
        o = rng.normal(size=N_OBS)
        a, lp, v = policy.act(o)
        t.obs.append(o)
        t.act.append(a)
        t.logp.append(lp)
        t.val.append(v)
        t.phi.append(0.0 if phi is None else float(phi[i]))
    t.terminal_reward = float(reward_fn(np.asarray(t.act)))
    t.tag = tag
    return t


def _fabricate(vals, terminal: float, phi=None, tag: str = "") -> Trajectory:
    """A trajectory with chosen value estimates, for checking arithmetic.

    The observations and actions are placeholders: nothing downstream of
    ``build`` reads them, and hand-picked values are what makes the expected
    advantage computable on paper.
    """
    t = Trajectory()
    n = len(vals)
    for i in range(n):
        t.obs.append(np.zeros(N_OBS))
        t.act.append(np.zeros(N_MODES))
        t.logp.append(0.0)
        t.val.append(float(vals[i]))
        t.phi.append(0.0 if phi is None else float(phi[i]))
    t.terminal_reward = float(terminal)
    t.tag = tag
    return t


def _reference_gae(rewards, values, gamma: float, lam: float):
    """GAE written as its definition, not as the backward recursion.

    ``A_t = sum_l (gamma*lam)^l * delta_{t+l}`` with
    ``delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`` and ``V(s_T) = 0``.  The
    implementation under test accumulates the same quantity backwards in one
    pass; writing the double sum here is what makes this a second
    implementation rather than a copy of the first.
    """
    r = np.asarray(rewards, float)
    v = np.asarray(values, float)
    n = r.size
    delta = np.array([r[i] + gamma * (v[i + 1] if i + 1 < n else 0.0) - v[i]
                      for i in range(n)])
    adv = np.array([sum((gamma * lam) ** l * delta[i + l] for l in range(n - i))
                    for i in range(n)])
    return adv, adv + v


# ==========================================================================
# unit -- the policy
# ==========================================================================


def test_one_decision_has_the_right_shape() -> None:
    print("\npolicy: one decision")
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    a, lp, v = p.act(rng.normal(size=N_OBS))
    check("one decision has the right shape",
          a.shape == (N_MODES,) and np.isfinite(lp) and np.isfinite(v),
          f"action {a.shape}, logp {lp:.3f}, value {v:.3f}")

    many = np.array([p.act(rng.normal(size=N_OBS), deterministic=True)[0]
                     for _ in range(200)])
    check("deterministic intent stays in [-1, 1]",
          bool(np.all(np.abs(many) <= 1.0)),
          f"max |a| = {np.abs(many).max():.4f}")


def test_batched_and_single_decisions_agree_when_deterministic() -> None:
    """``act_many`` is the batched form of ``act`` and must not be a second policy.

    Only under ``deterministic``: the docstring is explicit that the sampled
    path draws in a different order, so a run using one is reproducible against
    itself and not against a run using the other.  The *mean* has no such
    excuse, and it is the mean that every scoring rollout uses.
    """
    print("\npolicy: batched and single decisions")
    import torch
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    obs = rng.normal(size=(16, N_OBS))
    a_many, lp_many, v_many = p.act_many(obs, deterministic=True)
    one = [p.act(o, deterministic=True) for o in obs]
    a_one = np.array([x[0] for x in one])
    v_one = np.array([x[2] for x in one])
    check("act_many matches act row by row on the mean",
          bool(np.allclose(a_many, a_one, atol=1e-6)),
          f"max |da| = {np.abs(a_many - a_one).max():.2e}")
    check("and so does the value head",
          bool(np.allclose(v_many, v_one, atol=1e-5)),
          f"max |dv| = {np.abs(v_many - v_one).max():.2e}")


def test_the_importance_ratio_starts_at_one() -> None:
    """PPO's derivation needs ``pi_new/pi_old == 1`` before the first step.

    The stored log-probability is computed from the pre-squash sample; recovering
    it later costs an ``atanh`` of an action that has been through ``tanh``.
    Measured (``experiments/ppo_estimators``): the round trip is exact to
    2.9e-6 at the initial ``log_std = -0.5``, and loses everything at a
    ``log_std`` of 2.0, where 60% of actions sit past |a| = 0.999.  The bound
    below is the operating one; the second half of the check is the boundary,
    recorded so that a future change that lets ``log_std`` grow is caught here
    rather than in a run.
    """
    print("\npolicy: the importance ratio before any gradient step")
    import torch
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    buf = RolloutBuffer()
    for _ in range(8):
        buf.add(_rollout(p, rng, 64, lambda a: 1.0))
    obs, act, logp_old, *_ = buf.build()
    with torch.no_grad():
        lp = p.log_prob(torch.as_tensor(obs), torch.as_tensor(act))
    err = np.abs((lp - torch.as_tensor(logp_old)).exp().numpy() - 1.0)
    check("the ratio is 1 to floating point at the operating squash",
          float(err.max()) < 1e-4, f"max |ratio-1| = {err.max():.3e}")

    with torch.no_grad():
        p.log_std.fill_(2.0)
    buf2 = RolloutBuffer()
    for _ in range(8):
        buf2.add(_rollout(p, rng, 64, lambda a: 1.0))
    o2, a2, l2, *_ = buf2.build()
    with torch.no_grad():
        lp2 = p.log_prob(torch.as_tensor(o2), torch.as_tensor(a2))
    err2 = np.abs((lp2 - torch.as_tensor(l2)).exp().numpy() - 1.0)
    check("and it is the saturating squash that breaks it, as recorded",
          float(err2.max()) > 1.0,
          f"at log_std=2.0, max |ratio-1| = {err2.max():.3e} with "
          f"{(np.abs(a2) > 0.999).mean():.0%} of actions past 0.999")


def test_the_observation_normaliser_is_associative() -> None:
    """Folding two batches must give what folding their concatenation gives.

    The running statistics are a parallel (Chan) update, and its whole point is
    that the answer does not depend on how the stream was cut up.  A resume
    folds a different sequence of batches than an uninterrupted run would, so
    an update that is not associative is a resume that silently renormalises
    every stored weight.
    """
    print("\npolicy: the observation normaliser")
    import torch
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=(500, N_OBS)) * 3.0 + 1.0
    x2 = rng.normal(size=(300, N_OBS)) * 0.5 - 2.0

    split = SharedPolicy(N_OBS, N_MODES, hidden=8)
    split.observe(x1)
    split.observe(x2)
    whole = SharedPolicy(N_OBS, N_MODES, hidden=8)
    whole.observe(np.concatenate([x1, x2]))

    dm = float((split.obs_mean - whole.obs_mean).abs().max())
    dv = float((split.obs_var - whole.obs_var).abs().max())
    check("two folds match one fold of the concatenation",
          dm < 1e-4 and dv < 1e-3, f"max |dmean| = {dm:.2e}, max |dvar| = {dv:.2e}")

    # And it must actually normalise: an unnormalised first layer sees the
    # largest of 27 channels on wildly different scales and is blind to the rest.
    z = whole.normalise(torch.as_tensor(np.concatenate([x1, x2]).astype(np.float32)))
    check("and the normalised batch is centred and scaled",
          abs(float(z.mean())) < 0.05 and abs(float(z.std()) - 1.0) < 0.05,
          f"mean {float(z.mean()):+.4f}, std {float(z.std()):.4f}")


# ==========================================================================
# algorithm -- GAE, the discount, and the shaping identity
# ==========================================================================


def test_gae_matches_an_independent_reference() -> None:
    print("\nbuffer: GAE against its definition")
    vals = [0.10, -0.30, 0.50, 0.20, -0.10, 0.40, 0.00, 0.25]
    terminal = 1.0
    buf = RolloutBuffer(gamma=0.99, lam=0.95, shaping=0.0)
    buf.add(_fabricate(vals, terminal))
    _o, _a, _l, adv, ret, _v = buf.build()

    rew = np.zeros(len(vals))
    # One trajectory, so the per-tag scale is its own std, which is 0 for a
    # single sample and therefore the 0.05 floor.
    rew[-1] = terminal / 0.05
    ref_adv, ref_ret = _reference_gae(rew, vals, 0.99, 0.95)
    check("advantages match the double-sum definition",
          bool(np.allclose(adv, ref_adv, atol=1e-9)),
          f"max |dA| = {np.abs(adv - ref_adv).max():.2e}")
    check("and the return is advantage plus value",
          bool(np.allclose(ret, ref_ret, atol=1e-9)),
          f"max |dR| = {np.abs(ret - ref_ret).max():.2e}")


def test_the_discount_behaves_at_both_ends() -> None:
    """gamma = 0, lam = 0, and gamma = lam = 1 each collapse GAE to something known."""
    print("\nbuffer: the discount at its limits")
    vals = [0.10, -0.30, 0.50, 0.20]
    scale = 0.05

    b0 = RolloutBuffer(gamma=0.0, lam=0.95, shaping=0.0)
    b0.add(_fabricate(vals, 1.0))
    _o, _a, _l, adv0, _r, _v = b0.build()
    want0 = np.array([0.0, 0.0, 0.0, 1.0 / scale]) - np.asarray(vals)
    check("gamma = 0 makes the advantage the immediate reward minus the value",
          bool(np.allclose(adv0, want0, atol=1e-9)),
          f"max |dA| = {np.abs(adv0 - want0).max():.2e}")

    bl = RolloutBuffer(gamma=0.99, lam=0.0, shaping=0.0)
    bl.add(_fabricate(vals, 1.0))
    _o, _a, _l, advl, _r, _v = bl.build()
    rew = np.array([0.0, 0.0, 0.0, 1.0 / scale])
    td0 = np.array([rew[i] + 0.99 * (vals[i + 1] if i + 1 < 4 else 0.0) - vals[i]
                    for i in range(4)])
    check("lam = 0 makes it the one-step TD error",
          bool(np.allclose(advl, td0, atol=1e-9)),
          f"max |dA| = {np.abs(advl - td0).max():.2e}")

    b1 = RolloutBuffer(gamma=1.0, lam=1.0, shaping=0.0)
    b1.add(_fabricate(vals, 1.0))
    _o, _a, _l, _adv, ret1, _v = b1.build()
    check("gamma = lam = 1 makes the return the undiscounted sum of rewards",
          bool(np.allclose(ret1, np.full(4, 1.0 / scale), atol=1e-9)),
          f"returns {np.round(ret1, 4).tolist()}")


def test_one_trajectory_never_bootstraps_off_another() -> None:
    """Credit must not cross an episode boundary.

    Two trajectories in one buffer: the second's terminal reward has to be
    invisible from inside the first.  The way this breaks in practice is a
    flattened batch whose GAE recursion is run once over the concatenation,
    which is a single line and looks right.
    """
    print("\nbuffer: the episode boundary")
    a_only = RolloutBuffer(gamma=0.99, lam=0.95, shaping=0.0)
    a_only.add(_fabricate([0.1, 0.2, 0.3], 1.0, tag="x"))
    _o, _a, _l, adv_alone, _r, _v = a_only.build()

    both = RolloutBuffer(gamma=0.99, lam=0.95, shaping=0.0)
    both.add(_fabricate([0.1, 0.2, 0.3], 1.0, tag="x"))
    both.add(_fabricate([0.4, 0.5], 1.0, tag="x"))
    _o, _a, _l, adv_both, _r, _v = both.build()

    # The two share a tag, so they share a terminal scale; with equal terminal
    # rewards the scale is the same in both buffers and the first trajectory's
    # advantages must be untouched by the second's presence.
    check("the first trajectory's advantages do not see the second",
          bool(np.allclose(adv_alone, adv_both[:3], atol=1e-12)),
          f"max |dA| = {np.abs(adv_alone - adv_both[:3]).max():.2e}")
    check("and the batch is the concatenation, in order",
          adv_both.size == 5, f"{adv_both.size} advantages from 3 + 2 steps")


def test_potential_shaping_telescopes_to_nothing() -> None:
    """Potential-based shaping must not change the total return.

    Ng, Harada & Russell 1999: adding ``gamma*Phi(s') - Phi(s)`` leaves the
    optimal policy of the original MDP untouched *because the discounted sum
    telescopes* to ``-Phi(s_0)``, a constant of the start state.  If the
    implementation's Phi(terminal) is not zero, or the discount is applied on
    the wrong side, that identity fails and the shaping becomes a second
    objective competing with a segment score hardened against three exploits.
    """
    print("\nbuffer: the shaping identity")
    gamma = 0.9
    phi = [0.7, -0.2, 0.5, 1.3, -0.9]
    n = len(phi)
    buf = RolloutBuffer(gamma=gamma, lam=0.95, shaping=1.0)
    buf.add(_fabricate([0.0] * n, 0.0, phi=phi))
    # Recover the shaping rewards: with zero terminal reward and zero values,
    # the advantage at t=0 under lam=1 is the discounted sum of them.
    flat = RolloutBuffer(gamma=gamma, lam=1.0, shaping=1.0)
    flat.add(_fabricate([0.0] * n, 0.0, phi=phi))
    _o, _a, _l, adv, _r, _v = flat.build()
    check("the discounted shaping sum is -Phi(s_0), as the theorem requires",
          abs(float(adv[0]) + phi[0]) < 1e-9,
          f"sum = {float(adv[0]):+.9f}, -Phi(s_0) = {-phi[0]:+.9f}")

    off = RolloutBuffer(gamma=gamma, lam=1.0, shaping=0.0)
    off.add(_fabricate([0.0] * n, 0.0, phi=phi))
    _o, _a, _l, adv_off, _r, _v = off.build()
    check("and switching shaping off removes it entirely",
          bool(np.allclose(adv_off, 0.0, atol=1e-12)),
          f"max |A| = {np.abs(adv_off).max():.2e}")


def test_terminal_rewards_are_scaled_within_their_own_kind() -> None:
    """Water competence averaged 0.635 against 0.089 on land; unscaled, water wins.

    The scale is per tag and floored at 0.05, so a tag whose segments all scored
    the same does not get its single reward amplified without bound.
    """
    print("\nbuffer: per-segment-kind scaling")
    # Both spreads are above the 0.05 floor, so the scale is each tag's own std
    # and the floor is not what is being tested here -- the solo check below is.
    water = (0.40, 0.55, 0.70, 0.85)
    land = (0.00, 0.06, 0.12, 0.18)
    buf = RolloutBuffer(gamma=0.99, lam=0.95, shaping=0.0)
    for r in water:
        buf.add(_fabricate([0.0, 0.0], r, tag="water"))
    for r in land:
        buf.add(_fabricate([0.0, 0.0], r, tag="land"))
    scales = buf._terminal_scale()
    _o, _a, _l, _adv, ret, _v = buf.build()
    water_last = ret[1::2][:4]
    land_last = ret[1::2][4:]
    check("each tag gets its own scale",
          set(scales) == {"water", "land"}
          and abs(scales["water"] - np.std(water)) < 1e-9
          and abs(scales["land"] - np.std(land)) < 1e-9,
          f"water {scales['water']:.4f}, land {scales['land']:.4f}")
    check("and the two kinds end up on comparable scales",
          abs(np.std(water_last) - np.std(land_last)) < 0.05,
          f"raw std: water {np.std(water):.3f}, land {np.std(land):.3f}  ->  "
          f"scaled std: water {np.std(water_last):.3f}, "
          f"land {np.std(land_last):.3f}")

    lone = RolloutBuffer()
    lone.add(_fabricate([0.0], 1.0, tag="solo"))
    check("a tag with one sample takes the floor, not a division by zero",
          abs(lone._terminal_scale()["solo"] - 0.05) < 1e-12,
          f"scale {lone._terminal_scale()['solo']}")


def test_an_empty_generation_is_skipped_not_crashed_on() -> None:
    print("\nupdate: degenerate input")
    import torch
    torch.manual_seed(4)
    p = SharedPolicy(N_OBS, N_MODES, hidden=16)
    empty = ppo_update(p, RolloutBuffer())
    check("an empty generation is skipped, not crashed on",
          empty.get("skipped") is True, str(empty))

    buf = RolloutBuffer()
    buf.add(_fabricate([0.0] * 8, 0.0))
    before = p.state_dict()["actor.0.weight"].detach().numpy().copy()
    info = ppo_update(p, buf, epochs=2, minibatch=8,
                      rng=np.random.default_rng(0))
    after = p.state_dict()["actor.0.weight"].detach().numpy()
    check("a batch of all-zero rewards runs and reports finite losses",
          info.get("skipped") is False and np.isfinite(info["pi_loss"])
          and np.isfinite(info["v_loss"]),
          f"pi_loss {info['pi_loss']:+.5f}, v_loss {info['v_loss']:.5f}, "
          f"moved ||dW|| = {np.linalg.norm(after - before):.3e}")


def test_the_batch_boundaries() -> None:
    """One step, one trajectory, a negative reward, and an enormous one.

    A generation can produce any of these: a segment that terminated on its
    first control decision, a batch where every candidate but one failed Tier 0,
    a competence of zero against a scaled mean, and -- once a new rung is added
    -- a reward larger than anything the scaler has seen.  None of them was
    covered, and each is a place where an off-by-one or an overflow lives.
    """
    print("\nupdate: the boundaries of a batch")
    import torch

    # A one-step trajectory.  GAE over a single step is the terminal delta and
    # nothing else, and the loop that builds it must not index past the end.
    one = RolloutBuffer(gamma=0.99, lam=0.95, shaping=0.0)
    one.add(_fabricate([0.25], 1.0))
    _o, _a, _l, adv, ret, val = one.build()
    want = 1.0 / 0.05 - 0.25          # reward/scale + gamma*0 - V(s_0)
    check("a one-step trajectory gives one advantage, the terminal delta",
          adv.shape == (1,) and abs(float(adv[0]) - want) < 1e-12,
          f"advantage {float(adv[0]):.6f} against {want:.6f} by hand")
    check("and its return is that plus the value it started from",
          abs(float(ret[0]) - (want + 0.25)) < 1e-12, f"{float(ret[0]):.6f}")

    # One transition in the whole batch: below the floor ppo_update refuses at,
    # because a batch of one cannot be standardised (its std is zero).
    torch.manual_seed(40)
    p = SharedPolicy(N_OBS, N_MODES, hidden=16)
    before = p.state_dict()["actor.0.weight"].detach().numpy().copy()
    tiny = RolloutBuffer()
    tiny.add(_fabricate([0.1], 1.0))
    info = ppo_update(p, tiny, rng=np.random.default_rng(0))
    after = p.state_dict()["actor.0.weight"].detach().numpy()
    check("a batch of one transition is refused, and the policy is untouched",
          info.get("skipped") is True and np.array_equal(before, after),
          f"{info}")

    # Two is the smallest batch it will act on.
    pair = RolloutBuffer()
    pair.add(_fabricate([0.1, 0.2], 1.0))
    info2 = ppo_update(p, pair, epochs=1, minibatch=2,
                       rng=np.random.default_rng(0))
    check("two is the smallest batch it acts on",
          info2.get("skipped") is False and info2["transitions"] == 2,
          f"transitions {info2['transitions']}, grad_steps {info2['grad_steps']}")

    # Negative and enormous rewards.  Advantages are standardised across the
    # batch, so the scale should not reach the loss -- but nothing checked that
    # a 1e9 reward does not simply overflow on the way there.
    for label, rewards in (("negative", (-1.0, -0.5, -2.0, -0.25)),
                           ("enormous", (1e9, 2e9, 5e8, 1.5e9)),
                           ("all zero", (0.0, 0.0, 0.0, 0.0))):
        torch.manual_seed(41)
        q = SharedPolicy(N_OBS, N_MODES, hidden=16)
        b = RolloutBuffer()
        for r in rewards:
            b.add(_fabricate([0.1, 0.2, 0.3], r, tag="x"))
        got = ppo_update(q, b, epochs=2, minibatch=6,
                         rng=np.random.default_rng(0))
        w = q.state_dict()["actor.0.weight"].detach().numpy()
        finite = (got.get("skipped") is False
                  and all(np.isfinite(got[k]) for k in
                          ("pi_loss", "v_loss", "kl", "entropy"))
                  and np.isfinite(w).all())
        check(f"a {label} reward leaves the update finite",
              finite,
              f"pi_loss {got.get('pi_loss')}, v_loss {got.get('v_loss')}, "
              f"kl {got.get('kl')}, weights finite {np.isfinite(w).all()}")


def test_a_non_finite_batch_is_refused_rather_than_consumed() -> None:
    """One NaN reward used to write NaN into every weight of the shared policy.

    The batch-wide advantage standardisation spreads a single non-finite
    terminal reward across every sample, one optimiser step turns the whole
    network to NaN, and the failure surfaces several frames later inside
    ``torch.distributions.Normal`` -- after a checkpoint may already have been
    written.  "I could not measure this" must not share a value with "I measured
    zero"; here it must not share a value with "I measured this and learned from
    it" either.
    """
    print("\nupdate: a non-finite batch")
    import torch
    for where, make in (("reward", lambda t: setattr(t, "terminal_reward", np.nan)),
                        ("value", lambda t: t.val.__setitem__(2, np.inf)),
                        ("observation", lambda t: t.obs[1].__setitem__(3, np.nan))):
        torch.manual_seed(5)
        p = SharedPolicy(N_OBS, N_MODES, hidden=16)
        before = p.state_dict()["actor.0.weight"].detach().numpy().copy()
        buf = RolloutBuffer()
        for _ in range(3):
            t = _fabricate([0.1, 0.2, 0.3, 0.4], 1.0)
            buf.add(t)
        make(buf.trajectories[0])
        info = ppo_update(p, buf, epochs=2, minibatch=4,
                          rng=np.random.default_rng(0))
        after = p.state_dict()["actor.0.weight"].detach().numpy()
        check(f"a non-finite {where} is refused with a reason",
              info.get("skipped") is True and info.get("reason") == "non-finite batch",
              f"{ {k: v for k, v in info.items() if k != 'transitions'} }")
        check(f"and the policy is left exactly as it was ({where})",
              bool(np.array_equal(before, after)),
              f"||dW|| = {np.linalg.norm(after - before):.1e}")


def test_the_rate_anneal_reaches_the_optimiser() -> None:
    """``lr_fraction`` must move the optimiser's rate, not just the log line.

    The anneal is applied per update from ``gen / generations`` rather than by a
    torch scheduler, so there is no scheduler to be stepped the wrong number of
    times -- and correspondingly nothing that would notice if the number never
    reached ``param_groups``.
    """
    print("\nupdate: the learning-rate anneal")
    import torch
    torch.manual_seed(15)
    p = SharedPolicy(N_OBS, N_MODES, hidden=16)
    opt = torch.optim.Adam(p.parameters(), lr=1.0, eps=1e-5)
    seen = []
    for frac in (1.0, 0.5, 0.0, 2.0, -1.0):
        b = RolloutBuffer()
        b.add(_fabricate([0.1] * 8, 1.0))
        info = ppo_update(p, b, lr=1e-3, epochs=1, minibatch=8,
                          lr_fraction=frac, optimiser=opt,
                          rng=np.random.default_rng(0))
        seen.append((frac, info["lr"], opt.param_groups[0]["lr"]))
    check("the annealed rate is what the optimiser is given",
          all(abs(rep - act) < 1e-15 for _f, rep, act in seen),
          "; ".join(f"frac {f:g} -> {rep:.2e}" for f, rep, _a in seen))
    check("and the fraction is clamped to [0, 1]",
          seen[0][1] == 1e-3 and seen[2][1] == 0.0
          and seen[3][1] == 1e-3 and seen[4][1] == 0.0,
          f"frac 2.0 -> {seen[3][1]:.2e}, frac -1.0 -> {seen[4][1]:.2e}")


def test_the_clip_actually_clips() -> None:
    """The `P` in PPO. Mutation testing found this untested.

    Replacing ``min(r*A, clamp(r, 1-c, 1+c)*A)`` with ``min(r*A, r*A)`` -- the
    unclipped surrogate, i.e. vanilla policy gradient with importance weights --
    left every one of the 54 checks in this file green
    (``tools/mutate.py --only clip``).  The previous audit claimed clip was
    covered; it was not.

    The oracle is the ``clip`` argument itself, and it does not re-implement the
    objective.  If the clamp is real, a tight clip and a loose one must take the
    policy to different places; if it is gone, ``clip`` no longer reaches
    ``pi_loss`` and the actor's gradient is identical either way.  (It still
    reaches the *value* loss, but the actor and critic are separate stacks, so
    the actor's weights are the discriminating measurement.)

    ``clipfrac`` is asserted nonzero on the tight arm, because a test that never
    drove the ratio outside the band would pass whether or not the clamp exists.

    ``vf_coef=0`` is what makes the threshold below non-arbitrary rather than a
    guess.  ``clip`` has a second route into the actor -- it bounds the value
    loss too, and ``clip_grad_norm_`` normalises over *all* parameters at once,
    so a changed critic gradient rescales the actor's.  With the value loss left
    on, removing the clamp still left the two arms 0.000936 apart against a
    0.171 signal: a 180x margin, but one that a future change to ``vf_coef``
    could erode silently.  With the value loss off, ``clip`` reaches the actor
    through the clamp or not at all, so the mutated code gives **exactly** zero
    and the margin is no longer a number anyone has to keep calibrated.
    """
    print("\nupdate: the clip")
    import torch

    def run(clip):
        torch.manual_seed(31)
        rng = np.random.default_rng(31)
        p = SharedPolicy(N_OBS, N_MODES, hidden=32)
        opt = torch.optim.Adam(p.parameters(), lr=1e-2, eps=1e-5)
        b = RolloutBuffer()
        for _ in range(8):
            b.add(_rollout(p, rng, 64, lambda a: float(a[:, 0].mean())))
        # target_kl off: the epoch loop must run to the end on both arms, or the
        # difference measured below could be "one arm stopped earlier".
        info = ppo_update(p, b, epochs=6, minibatch=128, clip=clip,
                          target_kl=0.0, vf_coef=0.0, optimiser=opt,
                          rng=np.random.default_rng(32))
        return (p.state_dict()["actor.0.weight"].detach().numpy().copy(), info)

    w_tight, i_tight = run(0.05)
    w_loose, i_loose = run(10.0)

    check("the update leaves the clip band, so the clamp is exercised at all",
          i_tight["clipfrac"] > 0.05 and i_loose["clipfrac"] == 0.0,
          f"clipfrac {i_tight['clipfrac']:.4f} at clip=0.05, "
          f"{i_loose['clipfrac']:.4f} at clip=10.0")
    moved = float(np.linalg.norm(w_tight - w_loose))
    check("and a tight clip takes the actor somewhere a loose one does not",
          moved > 0.01,
          f"actor ||dW|| between clip=0.05 and clip=10.0 = {moved:.6f} "
          f"(0.000000 with the clamp removed)")
    check("the clipped surrogate is the smaller objective, as min() requires",
          i_tight["pi_loss"] > i_loose["pi_loss"],
          f"pi_loss {i_tight['pi_loss']:+.5f} clipped against "
          f"{i_loose['pi_loss']:+.5f} unclipped")


def test_the_reported_kl_cannot_be_negative() -> None:
    """A divergence that comes out negative is not a divergence.

    ``target_kl`` ends the epoch loop on this number, and the roadmap quotes it
    as evidence about the learning rate.  The plain ``log pi_old - log pi_new``
    it used to be is unbiased and high-variance enough to be negative: measured
    over 320 minibatches of real updates, negative on 20.0% of them, bottoming
    at -0.0072, and low enough on average that 6.9% of minibatches reported
    themselves inside the 0.015 bound while k3 put them at or over it.
    """
    print("\nupdate: the KL it stops on")
    import torch
    torch.manual_seed(6)
    rng = np.random.default_rng(6)
    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    opt = torch.optim.Adam(p.parameters(), lr=3e-3, eps=1e-5)
    kls = []
    for _ in range(8):
        b = RolloutBuffer()
        for _ in range(6):
            b.add(_rollout(p, rng, 48, lambda a: float(a[:, 0].mean())))
        info = ppo_update(p, b, epochs=4, minibatch=64, optimiser=opt,
                          rng=np.random.default_rng(7))
        kls.append(info["kl"])
    kls = np.asarray(kls)
    check("every reported KL is non-negative",
          bool((kls >= 0.0).all()), f"min {kls.min():+.6f}, mean {kls.mean():+.6f}")

    # And it is zero exactly when the policy did not move.
    b = RolloutBuffer()
    for _ in range(4):
        b.add(_rollout(p, rng, 32, lambda a: 0.0))
    still = ppo_update(p, b, epochs=1, minibatch=1 << 20, lr=0.0,
                       rng=np.random.default_rng(8), optimiser=None)
    check("and it is ~0 for an update that takes no step",
          abs(still["kl"]) < 1e-6, f"kl = {still['kl']:.3e}")


def test_the_entropy_bonus_has_a_gradient() -> None:
    """``ent_coef`` has to move exploration, or it is a knob wired to nothing.

    The term used to be ``-log pi_new(a)`` with ``a`` drawn from ``pi_old``,
    which is the cross entropy H(pi_old, pi_new).  At the on-policy point where
    every PPO update starts, ``E_{a~pi}[grad log pi(a)] = grad 1 = 0``, so the
    bonus contributed *no* expected gradient at any coefficient.  Measured over
    32 initialisations: d/d(log_std) was -0.00013 +/- 0.00188 (t = -0.07)
    against +0.43114 +/- 0.00106 (t = +409) for a real estimator, and sweeping
    ``ent_coef`` from 0 to 1.0 moved the learned mean ``log_std`` by -0.0063 --
    downward.  See ``experiments/ppo_estimators``.
    """
    print("\nupdate: the entropy bonus")
    import torch

    torch.manual_seed(9)
    rng = np.random.default_rng(9)
    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    obs = torch.as_tensor(rng.normal(size=(2048, N_OBS)).astype(np.float32))
    g = torch.autograd.grad(p.entropy(obs), p.log_std)[0]
    check("the entropy estimator pushes log_std up, not nowhere",
          bool((g > 0.2).all()), f"d/d(log_std) = {np.round(g.numpy(), 4).tolist()}")

    def train(coef, updates=12):
        torch.manual_seed(10)
        r = np.random.default_rng(10)
        q = SharedPolicy(N_OBS, N_MODES, hidden=32)
        o = torch.optim.Adam(q.parameters(), lr=3e-3, eps=1e-5)
        for _ in range(updates):
            b = RolloutBuffer()
            for _ in range(6):
                b.add(_rollout(q, r, 48, lambda a: float(a[:, 0].mean())))
            ppo_update(q, b, epochs=4, minibatch=128, optimiser=o, ent_coef=coef,
                       rng=np.random.default_rng(11))
        return float(q.log_std.detach().numpy().mean())

    off, on = train(0.0), train(0.5)
    check("and raising ent_coef raises the learned log_std",
          on > off + 0.02, f"ent_coef 0.0 -> {off:+.4f}, 0.5 -> {on:+.4f} "
                           f"(delta {on - off:+.4f})")


def test_it_learns_a_reward_with_a_known_optimum() -> None:
    """The check a shape test cannot make: is the gradient connected to the loss?"""
    print("\nupdate: does it learn?")
    import torch
    torch.manual_seed(1)
    rng = np.random.default_rng(0)
    learner = SharedPolicy(N_OBS, N_MODES, hidden=32)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-3, eps=1e-5)
    before = float(np.mean([learner.act(rng.normal(size=N_OBS),
                                        deterministic=True)[0][0]
                            for _ in range(100)]))
    for _ in range(12):
        b = RolloutBuffer()
        for _ in range(8):
            b.add(_rollout(learner, rng, 64, lambda acts: float(acts[:, 0].mean())))
        ppo_update(learner, b, epochs=4, minibatch=256, optimiser=opt,
                   rng=np.random.default_rng(12))
    after = float(np.mean([learner.act(rng.normal(size=N_OBS),
                                       deterministic=True)[0][0]
                           for _ in range(100)]))
    check("it learns a reward with a known optimum",
          after > before + 0.05, f"mode 0 mean {before:+.4f} -> {after:+.4f}")

    # A sparse terminal reward has to reach the steps that earned it, or a
    # segment-scored task cannot be learned at all.
    buf = RolloutBuffer()
    buf.add(_rollout(learner, rng, 32, lambda acts: 1.0))
    _o, _a, _l, adv, _r, _v = buf.build()
    check("and a sparse terminal reward credits every step",
          int(np.count_nonzero(adv)) == len(adv),
          f"{np.count_nonzero(adv)}/{len(adv)} nonzero advantages")


def test_the_update_is_reproducible_from_its_stream() -> None:
    """Same weights, same data, same stream -> same update.  And it must be *its* stream.

    The minibatch order used to come from ``np.random.shuffle``, i.e. the
    process-global legacy ``RandomState``: seeded by nothing from ``cfg.seed``,
    saved by nothing into the checkpoint, while
    ``adapters/trainers/search.py`` declares ``deterministic=True``.
    """
    print("\nupdate: reproducibility")
    import torch

    def run(stream_seed):
        torch.manual_seed(13)
        rng = np.random.default_rng(14)
        p = SharedPolicy(N_OBS, N_MODES, hidden=32)
        opt = torch.optim.Adam(p.parameters(), lr=3e-3, eps=1e-5)
        b = RolloutBuffer()
        for _ in range(8):
            b.add(_rollout(p, rng, 64, lambda a: float(a[:, 0].mean())))
        info = ppo_update(p, b, epochs=4, minibatch=64, optimiser=opt,
                          rng=np.random.default_rng(stream_seed))
        return p.state_dict()["actor.0.weight"].detach().numpy().copy(), info

    w1, i1 = run(0)
    w2, _ = run(0)
    w3, _ = run(1)
    check("the same stream gives the same update",
          float(np.linalg.norm(w1 - w2)) == 0.0,
          f"||dW|| = {np.linalg.norm(w1 - w2):.3e}")
    check("and a different stream gives a different one",
          float(np.linalg.norm(w1 - w3)) > 1e-3,
          f"||dW|| = {np.linalg.norm(w1 - w3):.3e}")
    check("an update given a stream says it was deterministic",
          i1.get("deterministic") is True, str(i1.get("deterministic")))

    torch.manual_seed(13)
    p = SharedPolicy(N_OBS, N_MODES, hidden=16)
    b = RolloutBuffer()
    b.add(_fabricate([0.1] * 8, 1.0))
    loose = ppo_update(p, b, epochs=1, minibatch=4)
    check("and one given none says it was not",
          loose.get("deterministic") is False, str(loose.get("deterministic")))


def test_the_search_seeds_torch_before_it_builds_the_policy() -> None:
    """Two runs at the same ``--seed`` must start from the same weights.

    They did not: ``run_search`` never called ``torch.manual_seed``, so the
    initial shared policy came from whatever entropy torch had picked up.  Two
    constructions sit ||dW|| = 7.4 apart in the first actor layer.  The worker
    pool seeds itself per shard, which covers the rollout's exploration noise
    and not this.

    This check used to read ``inspect.getsource(run_search)`` for the string
    ``manual_seed``.  Mutation testing killed it: commenting the call out leaves
    the string in the source as a comment, so the check passed on a
    ``run_search`` that no longer seeded anything
    (``tools/mutate.py --only torch-seed``).  A test whose oracle is the text of
    the code cannot tell a call from a mention of one.

    So it runs the real ``run_search`` and reads the weights it actually built.
    ``seed_archipelago`` is swapped for a probe that grabs them and stops the
    run before the first evaluation, which is also why this needs no MuJoCo and
    no GPU: nothing before that point compiles a model.  The ambient torch state
    is deliberately *different* before each call, so matching weights can only
    come from ``run_search`` having seeded torch itself.
    """
    print("\nsearch: the seed reaches the network")
    import shutil
    import tempfile

    import torch

    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution import loop as _loop

    class _StopBeforeEvaluating(Exception):
        pass

    def weights_at_birth(seed: int, ambient: int):
        torch.manual_seed(ambient)
        tmp = tempfile.mkdtemp(prefix="dyt-seed-")
        grabbed: dict = {}
        original = _loop.seed_archipelago

        def stop_here(state, spec):
            grabbed["w"] = (state.shared.state_dict()["actor.0.weight"]
                            .detach().numpy().copy())
            raise _StopBeforeEvaluating

        _loop.seed_archipelago = stop_here
        try:
            _loop.run_search(_loop.SearchConfig(
                generations=1, batch=1, seed=seed, run_dir=tmp, workers=1,
                islands=("generalist",), use_shared_policy=True), MissionSpec())
        except _StopBeforeEvaluating:
            pass
        finally:
            _loop.seed_archipelago = original       # never leave it patched
            shutil.rmtree(tmp, ignore_errors=True)
        return grabbed.get("w")

    a = weights_at_birth(1234, ambient=999)
    b = weights_at_birth(1234, ambient=7)
    c = weights_at_birth(4321, ambient=999)
    same = float(np.linalg.norm(a - b))
    other = float(np.linalg.norm(a - c))
    check("the same --seed builds the same policy, whatever torch was doing",
          same == 0.0, f"||dW|| = {same:.3e} across two different ambient "
                       f"torch states")
    check("and a different --seed builds a different one",
          other > 1.0, f"||dW|| = {other:.4f}")


# ==========================================================================
# state -- what survives a checkpoint
# ==========================================================================


def test_both_learner_streams_survive_a_checkpoint() -> None:
    """Weights, moments, minibatch order and exploration noise, or the resume diverges.

    The first two were already checkpointed, each after a run had shown what
    their absence costs.  The other two are the same bug in the only two places
    it was still possible: a resume that continues from the right weights and
    then takes a *different* sequence of gradient steps looks exactly like
    ordinary run-to-run noise.
    """
    print("\nstate: the learner's streams across a checkpoint")
    import shutil
    import tempfile

    import torch

    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.auditor import Auditor
    from dytiscidae.evolution.curator import Curator
    from dytiscidae.evolution.curriculum import Curriculum
    from dytiscidae.evolution.islands import Archipelago
    from dytiscidae.evolution.judge import Judge
    from dytiscidae.evolution.loop import (BD_AXES, SearchConfig, SearchState,
                                           load_state, save_state)
    from dytiscidae.ops.telemetry import Telemetry

    tmp = tempfile.mkdtemp(prefix="dyt-ppo-state-")
    try:
        cfg = SearchConfig(run_dir=tmp, seed=17, islands=("generalist",),
                           use_shared_policy=True)
        arch = Archipelago()
        a = Archive(BD_AXES)
        arch.register("generalist", a, Curator(a, seed=17))
        st = SearchState(telemetry=Telemetry(tmp), config=cfg,
                         rng=np.random.default_rng(17), archipelago=arch,
                         curricula={"generalist": Curriculum()}, judge=Judge(),
                         auditor=Auditor(), island="generalist")
        torch.manual_seed(17)
        st.shared = SharedPolicy(27, 6, hidden=16)
        st.shared_opt = torch.optim.Adam(st.shared.parameters(), lr=1e-3, eps=1e-5)
        st.learner_rng = np.random.default_rng(17 ^ 0x5EED5EED)
        st.learner_rng.integers(1 << 30, size=5)          # advance both streams
        torch.randn(4)

        want_learner = st.learner_rng.bit_generator.state
        want_torch = torch.get_rng_state().clone()
        save_state(st, 3)

        st.learner_rng = np.random.default_rng(0)
        torch.manual_seed(999)
        gen = load_state(st)
        check("the checkpoint knows which generation it stopped at",
              gen == 4, f"resumes at generation {gen}")
        check("PPO's minibatch stream is restored",
              st.learner_rng.bit_generator.state == want_learner,
              "bit generator state matches")
        check("and so is the stream the rollout explores with",
              bool((torch.get_rng_state() == want_torch).all()),
              "torch rng state matches")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# integration -- the learner against the real batched evaluator
# ==========================================================================


def test_the_shaping_reads_the_channels_it_thinks_it_does() -> None:
    """The one place the learner knows the environment's layout, gated.

    ``learning/ppo.py`` imports nothing from ``envs`` -- except that
    ``potential_of`` addresses the observation **by index**
    (``_GRAV_Z, _DEPTH, _WET, _DEPTH_ERR, _CONTACT = 8, 9, 10, 14, 15``), with a
    comment reasoning that the morphology context is appended after them so
    widening the observation cannot move them.  That reasoning is correct and it
    is not a gate: inserting or reordering a channel *before* index 15 would
    leave the shaping silently shaping on the wrong quantity, which is the
    failure mode potential-based shaping is least able to announce, since it
    provably cannot change the optimum and so cannot show up as a wrong answer.

    So each index is checked against the quantity recomputed from the
    environment's own accessors, which is not the same information twice.
    Needs a compiled model, and nothing more -- not the batched evaluator.

    The skip guard is on ``import mujoco`` rather than on the project modules,
    and that distinction is the whole of it: the project modules import without
    MuJoCo, which ``core/mjcf.compile_phenotype`` then imports *lazily* at the
    moment it compiles.  Guarding the visible imports therefore guards nothing,
    and this test raised on a runner with numpy and torch and no MuJoCo rather
    than skipping.  It is the lazy-import case ``tests/test_architecture.py``
    spawns a whole interpreter to catch, met here in a test of its own.
    Everything after the guard is outside a ``try``, so a real failure still
    fails.
    """
    print("\nintegration: the shaping's view of the observation")
    try:
        import mujoco  # noqa: F401
    except Exception as exc:                                     # noqa: BLE001
        skip("the shaping's view of the observation",
             f"no MuJoCo here: {type(exc).__name__}: {exc}")
        return

    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.learning import ppo as _ppo

    env = TriphibianEnv(build(beetle()))
    env.reset(Domain.WATER, randomise=False)
    obs = env.observation(Domain.WATER)

    check("the observation is the width the policy is built for",
          obs.shape == (TriphibianEnv.OBS_DIM,),
          f"{obs.shape[0]} channels, OBS_DIM = {TriphibianEnv.OBS_DIM}")

    R = env.data.xmat[env.root_body].reshape(3, 3)
    want = {
        "_GRAV_Z": (_ppo._GRAV_Z, float((R.T @ np.array([0.0, 0.0, -1.0]))[2])),
        "_DEPTH": (_ppo._DEPTH, float(np.tanh(env.depth() / 5.0))),
        "_WET": (_ppo._WET, float(env.solver.diag.mean_submerged)),
        "_DEPTH_ERR": (_ppo._DEPTH_ERR,
                       float(np.tanh((env.depth() - TriphibianEnv.TARGET_DEPTH) / 3.0))),
        "_CONTACT": (_ppo._CONTACT, 1.0 if env._touching_ground() else 0.0),
    }
    for name, (idx, expected) in want.items():
        check(f"{name} = {idx} is the channel it is named after",
              abs(float(obs[idx]) - expected) < 1e-9,
              f"obs[{idx}] = {float(obs[idx]):+.6f}, "
              f"recomputed {expected:+.6f}")

    # And the potential itself must respond to those channels rather than being
    # a constant: a shaping term that never varies contributes exactly nothing.
    at_target = np.array(obs, dtype=float)
    at_target[_ppo._DEPTH_ERR] = 0.0
    far = np.array(obs, dtype=float)
    far[_ppo._DEPTH_ERR] = 0.9
    p_at, p_far = (_ppo.potential_of(at_target, "water"),
                   _ppo.potential_of(far, "water"))
    check("and the water potential is highest exactly at the target depth",
          p_at > p_far + 0.8, f"error 0.0 -> {p_at:+.4f}, 0.9 -> {p_far:+.4f}")

    on_ground = np.array(obs, dtype=float)
    on_ground[_ppo._CONTACT] = 1.0
    check("and the land potential pays for being on the ground",
          _ppo.potential_of(on_ground, "land")
          > _ppo.potential_of(obs, "land") + 0.9,
          f"{_ppo.potential_of(obs, 'land'):+.4f} -> "
          f"{_ppo.potential_of(on_ground, 'land'):+.4f}")


def test_the_learner_is_wired_to_the_evaluator() -> None:
    """Transitions contribute trajectories, and a scoring rollout uses the mean.

    Needs the batched evaluator, which needs the built GPU fluid extension.
    Skipped -- with the reason -- rather than crashing the suite, because every
    check above tests the learner and none of them needs a physics model.
    """
    print("\nintegration: the learner against the batched evaluator")
    from dytiscidae.envs import batchroll as _br

    if not _br.AVAILABLE:
        skip("the learner against the batched evaluator",
             f"{_br.UNAVAILABLE_REASON} — needs `cd mojo && pixi run build-all`")
        return

    import torch

    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.batchroll import evaluate_tier1_batch
    from dytiscidae.envs.triphibian import TriphibianEnv

    torch.manual_seed(2)
    shared = SharedPolicy(TriphibianEnv.OBS_DIM, N_MODES, hidden=16)
    buf = RolloutBuffer()
    phenos = [build(beetle()), build(beetle())]
    evaluate_tier1_batch(phenos, segment_seconds=0.4, identify_axes=True,
                         seed=3, shared=shared, buffer=buf)
    n_traj = len(buf.trajectories)
    # 3 domain segments + 3 transitions per machine, minus any that recorded
    # nothing; strictly more than the 6 segment trajectories proves the
    # transitions contributed.
    check("transitions contribute trajectories beyond the segments",
          n_traj > 3 * len(phenos),
          f"{n_traj} trajectories from {len(phenos)} machines")

    # Exploration noise exists to generate on-policy data.  A rollout that banks
    # no trajectory has nothing to explore for, and its score is what the archive
    # stores -- so a scoring rollout must be the policy's mean.  Measured cost of
    # getting this wrong: crossings 0.600 -> 0.558 with sampling, 0.642 with the
    # mean, on the same eight bodies at one seed.
    torch.manual_seed(3)
    scorer = SharedPolicy(TriphibianEnv.OBS_DIM, N_MODES, hidden=16)
    calls = {"sampled": 0, "mean": 0}
    inner = scorer.act

    def counting_act(obs, *, deterministic=False):
        calls["mean" if deterministic else "sampled"] += 1
        return inner(obs, deterministic=deterministic)

    scorer.act = counting_act
    evaluate_tier1_batch([build(beetle())], segment_seconds=0.4,
                         identify_axes=True, seed=5, shared=scorer)
    check("a scoring rollout uses the policy mean",
          calls["sampled"] == 0 and calls["mean"] > 0,
          f"{calls['mean']} mean, {calls['sampled']} sampled")

    calls["sampled"] = calls["mean"] = 0
    evaluate_tier1_batch([build(beetle())], segment_seconds=0.4,
                         identify_axes=True, seed=5, shared=scorer,
                         buffer=RolloutBuffer())
    check("a learning rollout still explores",
          calls["sampled"] > 0 and calls["mean"] == 0,
          f"{calls['sampled']} sampled, {calls['mean']} mean")


# ==========================================================================


def main() -> int:
    if not AVAILABLE:
        print(f"SKIP: torch unavailable ({UNAVAILABLE_REASON})")
        return 0

    test_one_decision_has_the_right_shape()
    test_batched_and_single_decisions_agree_when_deterministic()
    test_the_importance_ratio_starts_at_one()
    test_the_observation_normaliser_is_associative()

    test_gae_matches_an_independent_reference()
    test_the_discount_behaves_at_both_ends()
    test_one_trajectory_never_bootstraps_off_another()
    test_potential_shaping_telescopes_to_nothing()
    test_terminal_rewards_are_scaled_within_their_own_kind()

    test_an_empty_generation_is_skipped_not_crashed_on()
    test_the_batch_boundaries()
    test_a_non_finite_batch_is_refused_rather_than_consumed()
    test_the_rate_anneal_reaches_the_optimiser()
    test_the_clip_actually_clips()
    test_the_reported_kl_cannot_be_negative()
    test_the_entropy_bonus_has_a_gradient()
    test_it_learns_a_reward_with_a_known_optimum()
    test_the_update_is_reproducible_from_its_stream()
    test_the_search_seeds_torch_before_it_builds_the_policy()

    test_both_learner_streams_survive_a_checkpoint()

    test_the_shaping_reads_the_channels_it_thinks_it_does()
    test_the_learner_is_wired_to_the_evaluator()

    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED, not run on this machine: "
              f"{', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    # Unqualified only when nothing was skipped: the string a reader greps for
    # must not appear on a run that did not run everything.
    print("shared PPO checks passed" if not SKIPPED
          else f"shared PPO checks passed, {len(SKIPPED)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
