# Does the shared policy learn on a body that does not change? (M4)

ROADMAP M4 (2026-10-08, from the MuscleMimic comparison). The search trains one
shared PPO policy on whatever bodies a generation produces, so a flat learning
curve in a run cannot say whether the learner fails or the moving population
starves it. This fixes the body.

## Method

Three arch48 elites: the best recorded water competence, the best land, the
best air. For each body and each of 3 learner seeds:

- a fresh `SharedPolicy` (the search's sizes, `torch.manual_seed(seed)`), the
  search's PPO settings (`SearchConfig` defaults: lr, epochs, minibatch,
  `target_kl`, `ent_coef`, `reward_shaping`), no per-design policy (the
  shared policy alone acts), the elite's stored mobility bases, no
  identification;
- 30 updates. Each update scores 8 copies of the body through
  `batchroll.evaluate_tier1_batch` with sampling on and a `RolloutBuffer`,
  each copy on its own training draw (the search's path with M2), then calls
  `ppo_update` exactly as the loop does;
- before update 0 and after every 10th, a deterministic evaluation at 8 fixed
  held-out draws (never trained on): mean competence per medium.

Controls at the same 8 held-out draws: the still machine
(`held_still_params`, no policy) and the body's base gait (no policy).

**Size, set before any data:** first drafted as 40 updates of 16; a 2-update
smoke run measured one 8-copy evaluation of the air elite (a rotor body) at
~75 s, which put that size at 11+ h beside a live run, so it was cut to 30 of 8
(~5 h) before the measurement started.

## Pre-registered predictions (written before any data)

- **P1.** On at least 2 of the 3 bodies, the held-out competence in the body's
  own medium after update 30 is not above update 0's by more than the spread
  of the 3 seeds (max − min at update 30). The learner does not learn a
  fixed body in 30 updates (240 rollouts).
- **P2.** Where a curve rises, it rises above the base gait's held-out score;
  a curve that only approaches the base gait has learned to stop interfering,
  not a skill.

## Decision rule (also written before the data)

- P1 holds (no learning): the learner is the wall. Skill pretraining and a
  learnability score are premature; M5 (PPO epochs 1/2/4/10 at a fixed budget)
  is the next learner experiment, on these bodies.
- P1 fails (it learns a fixed body): the coupling to a moving population is
  the wall. The next step is the timescale split the review proposed (several
  PPO updates per generation, or a policy per island).

## Run

    systemd-run --user --unit fixed-body -p MemoryMax=3000M -p Nice=10 --same-dir \
        env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/fixed_body_learning/run.py \
        --run runs/arch48 --out experiments/fixed_body_learning/results_arch48.json
