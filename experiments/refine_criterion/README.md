# What does controller refinement buy, and on which draw?

ROADMAP M1 (2026-10-08). `_refine_controllers` (`evolution/loop.py`) is a
(1+1)-ES on a candidate's own policy weights. It accepts a trial when
`mission_fraction` rises strictly, and the accepted trial's result is what the
archive receives. Two questions:

1. **Criterion.** In arch48, `mission_fraction` was above zero in 7 of 4,792
   evaluations, and refinement accepted 29 trials in 300 generations. Would a
   criterion that selection actually reads accept more?
2. **Draw.** A trial is scored on the same task draw (seed) as the result it
   replaces. A gain measured on the draw that chose it is optimistic. How much
   of an accepted trial's gain is still there when the trial is scored on a
   fresh draw?

## Method

arch48's merged archive (200 elites). Every elite is scored through the
search's batched path (`batchroll.evaluate_tier1_batch`), with the shared
network that scored it and no identification, exactly as the re-score and the
refinement steps are. arch48 ran `controller_refine_steps=1`, so one search
refinement is one perturbation:
`w + N(0, controller_refine_sigma=0.1)` on the elite's own policy weights.

For each elite, `T = 3` independent single-step trials. Each one is a replicate
of arch48's own refinement step. The base controller and all trials are scored
at two draws:

- `s`: the elite's own `eval_seed`, the draw it was recorded on;
- `s'`: a fresh seed derived from `s` and used only here.

This is the current code (antipodal heading pair), not arch48's. The elites
were selected under the old score.

Criteria, each computed from one `MissionResult`:

| name | what it reads | who uses it |
|---|---|---|
| `mission` | `mission_fraction` | refinement today |
| `island` | `islands.island_score(island, result)` | the island half of placement |
| `stage` | `curriculum.stage_score(stage, ...)` at the elite's recorded stage, under its island's curriculum | the curriculum half of placement |

A trial is *accepted* under a criterion when its score at `s` is strictly
greater than the base's score at `s`, which is the search's rule. For an
accepted trial:

- `gain_s = C(trial, s) - C(base, s)` (> 0 by construction);
- `gain_fresh = C(trial, s') - C(base, s')`.

The control is `gain_fresh` over all trials, accepted or not. If acceptance
picked out better weights, accepted trials beat the control at `s'`.

## Pre-registered predictions (written before any data)

- **P1.** Under `mission`, fewer than 2% of trials are accepted (arch48: 29
  acceptances over ~4,700 candidates).
- **P2.** Under `island` and `stage`, at least 20% of trials are accepted.
- **P3.** Under `island` and `stage`, accepted trials keep less than half of
  their gain at `s'`: `mean(gain_fresh) / mean(gain_s) < 0.5`.
- **P4.** Accepted trials' mean `gain_fresh` is no greater than the control's,
  within its 95% bootstrap interval. That would mean a one-step acceptance
  picks noise, not weights.

## Decision rule (also written before the data)

- If accepted `gain_fresh` is above the control (P4 false) and keeps at least
  half of `gain_s` (P3 false): switch the criterion to the one selection reads,
  and keep scoring on the same draw.
- If accepted `gain_fresh` is above the control but keeps less than half:
  switch the criterion, and report the pre-refinement score, or score accepted
  trials on a draw other than the one reported.
- If accepted `gain_fresh` is not above the control (P4 holds): one-step
  refinement selects noise. Turn it off (`--refine-steps 0`) and save the half
  of the re-score batch it rides in (about 20 s of a 129 s generation in
  arch48).

## By-products

- `base(s)` reproduces the archive's recorded medium scores only where the
  scoring change (antipodal pair) did not touch them: air.
- `base(s)` against `base(s')`, per medium, is one elite's spread between two
  draws. This is a first, two-draw read of C2's seed variance.

## Run

    systemd-run --user --unit refine-arch48 -p MemoryMax=3500M --same-dir \
        env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/refine_criterion/run.py \
        --run runs/arch48 --out experiments/refine_criterion/results_arch48.json

    python experiments/refine_criterion/run.py --selftest     # no simulator
