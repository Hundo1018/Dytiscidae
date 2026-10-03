# Does the shared policy carry weight? (ROADMAP R) — 2026-10-03

## Frozen before the run

**0 Question.** Take a finished run's elites, each scored with the shared network
that scored it. Remove the network and score again at the same seed. How much of
the score was the network's? Decision: whether GRPO (item N) and an entropy
sweep on the shared policy are worth building.

**1 Prior art.** arch33 measured -0.0014 +/- 0.0016 (t -0.90) in-distribution
(memory: "arch33: eight of nine fixes worked"). arch35-41 never scored the
shared policy at all (the actor pool dropped it from re-scores), so no earlier
run is evidence. arch45 and arch46 are the first two long runs with it scored
and recorded (`scored_with_shared_policy=True` on every elite).
Rivals for any difference: re-scoring noise (none expected: the evaluator is
deterministic at a fixed seed, so the paired difference has no sampling noise,
only elite-to-elite spread); a code change since the run (both arms run
today's code, so it cancels in the delta).

**3 Prediction (written before the run).** On arch46's elites, with minus
without, per medium:
- mean delta within +/-0.02 of zero in air, water and land, and |t| < 2 in each;
- mission_fraction delta 0.000 (it is ~0 for nearly every elite, `mf=0.000`
  throughout arch46, so it cannot move);
- but the network is not inert: at least 30% of elites change by more than 0.01
  in some medium, with both signs present (an unsystematic effect).
Falsified if any medium shows mean delta >= +0.02 with t >= 3 and at least 60%
of elites improved (then the network carries weight), or if fewer than 10% of
elites move by more than 0.01 (then it is inert and PPO is training a no-op).
Seen before the prediction was frozen: a 3-elite timing test (gens 8, 18, 61;
these are included in the final data). It moved water by -0.025 on one elite,
land by +0.18 on another, and nothing on the third.

**4 Measurement.** `rescore.py` re-scores every elite through
`batchroll.evaluate_tier1_batch` (the search's path), `identify_axes=False`,
the stored mobility basis and own policy, `segment_seconds` 6.0 from the run's
provenance, the elite's own `eval_seed`. Arm "with" uses
`scoring_networks/gen<gen>.npz`, arm "without" uses `shared=None`. Elites are
grouped by (gen, eval_seed) because those fix the network and the seed.
Paired statistics: `experiments/harness.py::paired_delta`. A "reproduction"
column compares the "with" arm to the competence the run recorded.

**5 Design.** One process, 4G cap, no pool. All 218 elites if the wall allows.

    systemd-run --user --scope -q -p MemoryMax=4G env MUJOCO_GL=disable PYTHONPATH=. \
        DYTISCIDAE_KERNEL_DIR=<main checkout>/mojo/build .venv/bin/python \
        experiments/shared_policy_value/rescore.py --run runs/arch46 \
        --out experiments/shared_policy_value/results_arch46.json

## Appended after

(filled in below)
