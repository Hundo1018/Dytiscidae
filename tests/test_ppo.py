"""The shared PPO learner: does it run, and does it actually learn?

A test that only checks shapes would pass on a policy whose gradient is
disconnected from its loss, which is the failure mode that matters. So the last
check trains against a reward with a known optimum and asserts the policy moves
toward it.
"""
import numpy as np

from dytiscidae.learning.ppo import (AVAILABLE, UNAVAILABLE_REASON,
                                     RolloutBuffer, SharedPolicy, Trajectory,
                                     ppo_update)

N_OBS, N_MODES = 14, 4


def _rollout(policy, rng, n_steps, reward_fn):
    t = Trajectory()
    for _ in range(n_steps):
        o = rng.normal(size=N_OBS)
        a, lp, v = policy.act(o)
        t.obs.append(o)
        t.act.append(a)
        t.logp.append(lp)
        t.val.append(v)
    t.terminal_reward = reward_fn(np.asarray(t.act))
    return t


def main() -> int:
    if not AVAILABLE:
        print(f"SKIP: torch unavailable ({UNAVAILABLE_REASON})")
        return 0

    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ok = True

    p = SharedPolicy(N_OBS, N_MODES, hidden=32)
    a, lp, v = p.act(rng.normal(size=N_OBS))
    shape_ok = a.shape == (N_MODES,) and np.isfinite(lp) and np.isfinite(v)
    print(f"  [{'ok  ' if shape_ok else 'FAIL'}] one decision has the right shape"
          f"  -- action {a.shape}, logp {lp:.3f}, value {v:.3f}")
    ok &= shape_ok

    # Bounded intent: the mean is a tanh, so no coefficient can run away.
    many = np.array([p.act(rng.normal(size=N_OBS), deterministic=True)[0]
                     for _ in range(200)])
    bounded = bool(np.all(np.abs(many) <= 1.0))
    print(f"  [{'ok  ' if bounded else 'FAIL'}] deterministic intent stays in "
          f"[-1, 1]  -- max |a| = {np.abs(many).max():.4f}")
    ok &= bounded

    # GAE on a reward that is zero until the last step must still credit the
    # earlier steps, or a sparse-terminal task cannot be learned at all.
    buf = RolloutBuffer()
    buf.add(_rollout(p, rng, 32, lambda acts: 1.0))
    _o, _a, _l, adv, ret, _v = buf.build()
    credited = bool(np.count_nonzero(adv) == len(adv))
    print(f"  [{'ok  ' if credited else 'FAIL'}] sparse terminal reward reaches "
          f"every step  -- {np.count_nonzero(adv)}/{len(adv)} nonzero advantages")
    ok &= credited

    # An update on a near-empty buffer is a no-op rather than a crash: a
    # generation where every candidate failed Tier-0 produces exactly that.
    empty = ppo_update(p, RolloutBuffer())
    skipped = empty.get("skipped") is True
    print(f"  [{'ok  ' if skipped else 'FAIL'}] an empty generation is skipped, "
          f"not crashed on  -- {empty}")
    ok &= skipped

    # The one that matters: reward the policy for driving mode 0 positive and
    # check it does. If the gradient were disconnected this stays at zero.
    torch.manual_seed(1)
    learner = SharedPolicy(N_OBS, N_MODES, hidden=32)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-3)
    before = float(np.mean([learner.act(rng.normal(size=N_OBS),
                                        deterministic=True)[0][0]
                            for _ in range(100)]))
    for _ in range(12):
        b = RolloutBuffer()
        for _ in range(8):
            b.add(_rollout(learner, rng, 64, lambda acts: float(acts[:, 0].mean())))
        ppo_update(learner, b, epochs=4, minibatch=256, optimiser=opt)
    after = float(np.mean([learner.act(rng.normal(size=N_OBS),
                                       deterministic=True)[0][0]
                           for _ in range(100)]))
    learned = after > before + 0.05
    print(f"  [{'ok  ' if learned else 'FAIL'}] it learns a reward with a known "
          f"optimum  -- mode 0 mean {before:+.4f} -> {after:+.4f}")
    ok &= learned

    # Transitions feed the buffer too.  They are the part of the mission the
    # policy most needs to learn, and until this was wired every crossing
    # datum was thrown away while the buffer filled with steady swimming.
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
    wired = n_traj > 3 * len(phenos)
    print(f"  [{'ok  ' if wired else 'FAIL'}] transitions contribute "
          f"trajectories beyond the segments  -- {n_traj} trajectories from "
          f"{len(phenos)} machines")
    ok &= wired

    # Exploration noise exists to generate on-policy data.  A rollout that
    # banks no trajectory has nothing to explore for, and its score is what the
    # archive stores -- so a scoring rollout must be the policy's mean.
    # Measured cost of getting this wrong: crossings 0.600 -> 0.558 with
    # sampling, 0.642 with the mean, on the same eight bodies at one seed.
    from dytiscidae.envs.batchroll import evaluate_tier1_batch as _eval

    torch.manual_seed(3)
    scorer = SharedPolicy(TriphibianEnv.OBS_DIM, N_MODES, hidden=16)
    calls = {"sampled": 0, "mean": 0}
    inner = scorer.act

    def counting_act(obs, *, deterministic=False):
        calls["mean" if deterministic else "sampled"] += 1
        return inner(obs, deterministic=deterministic)

    scorer.act = counting_act
    _eval([build(beetle())], segment_seconds=0.4, identify_axes=True, seed=5,
          shared=scorer)          # no buffer: this is a scoring pass
    scoring_clean = calls["sampled"] == 0 and calls["mean"] > 0
    print(f"  [{'ok  ' if scoring_clean else 'FAIL'}] a scoring rollout uses the "
          f"policy mean  -- {calls['mean']} mean, {calls['sampled']} sampled")
    ok &= scoring_clean

    calls["sampled"] = calls["mean"] = 0
    _eval([build(beetle())], segment_seconds=0.4, identify_axes=True, seed=5,
          shared=scorer, buffer=RolloutBuffer())   # learning pass
    learning_explores = calls["sampled"] > 0 and calls["mean"] == 0
    print(f"  [{'ok  ' if learning_explores else 'FAIL'}] a learning rollout still "
          f"explores  -- {calls['sampled']} sampled, {calls['mean']} mean")
    ok &= learning_explores

    print()
    print("shared PPO checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
