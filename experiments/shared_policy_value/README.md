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

**6 What ran.** The command in §5, commit `9ef542d`'s `rescore.py`, one process
under a 3500M cap, as a systemd user unit (`MemoryMax=3500M`; it was not killed); 218 of 218 elites in 148 (gen, seed)
groups, every elite re-scored with its own gen's network (`net: scored` on all
218; no fallback to the final network). Wall 4355 s (29 s per group late in the
run, 10 s early; cost rises with rotors). Results: `results_arch46.json`, with
the per-elite rows.

**8 Result.** Paired, with minus without the shared network, n = 218:

| medium | mean with | mean without | delta | se | t | 95% CI (bootstrap) | improved / worse / unchanged | moved > 0.01 | max abs |
|---|---|---|---|---|---|---|---|---|---|
| air | 0.0123 | 0.0062 | +0.0061 | 0.0036 | +1.69 | [-0.0001, +0.0136] | 19 / 14 / 185 | 17 | 0.49 |
| water | 0.0825 | 0.0698 | +0.0126 | 0.0049 | +2.59 | [+0.0032, +0.0224] | 73 / 53 / 92 | 82 | 0.37 |
| land | 0.0757 | 0.0510 | +0.0247 | 0.0071 | +3.50 | [+0.0115, +0.0389] | 69 / 46 / 103 | 61 | 0.54 |
| mission_fraction | 0.0000 | 0.0000 | -0.0000 | 0.0000 | -0.25 | [0, 0] | 4 / 7 | 0 | 0.0000 |

Medians are 0.000 in all three media: the effect is a minority of elites moving a
lot. Positive deltas sum to +1.79 / +4.74 / +7.12 and negative to -0.45 / -1.98 /
-1.75 (air / water / land). 128 of 218 elites (58.7%) move by more than 0.01 in at
least one medium, with both signs present.

By island (mean land delta): land +0.140 (n 24), land_air +0.041 (22), amphibian
+0.013 (49), generalist +0.009 (35), water +0.003, air +0.004, aerial_diver
-0.003. The land island alone supplies about 3.4 of the 5.4 total land delta
(62%); without it the land mean is about +0.011. Water by island: water +0.036
(29), amphibian +0.016, air +0.011, generalist +0.010. Air: aerial_diver +0.036
(26), land_air +0.021.

Reproduction of the recorded competence by the "with" arm (|gap| <= 0.005): air
215/218, water 192/218, land 173/218; 161 elites reproduce all three. The code
changed between arch46's commit (`9b81614`) and this one (`a101209` crossing
fix, `be3dbe0` queue pool), so the mismatches are not evidence of a wrong
network. Restricted to the 161 that reproduce all three: air +0.0080 (t +1.70),
water +0.0161 (t +2.97), land +0.0154 (t +2.34); mission_fraction 0.

**9 Against the frozen text.**
- "mean delta within +/-0.02 of zero in each medium": air yes, water yes (+0.0126),
  **land no (+0.0247)**.
- "|t| < 2 in each": air yes (1.69), **water no (2.59), land no (3.50)**.
- "mission_fraction delta 0.000": confirmed (max abs delta 0.0000).
- "at least 30% of elites change by more than 0.01 in some medium, both signs":
  confirmed (58.7%).
- Falsifier "mean >= +0.02 with t >= 3 and >= 60% improved": land has the first two
  and **not the third** (69/218 = 32%), so it does not fire. Falsifier "fewer than
  10% move": not met (58.7%). Neither falsifier fires, but the prediction as
  written was wrong on two of its four clauses: the network's average effect is
  distinguishable from zero in water and land.

**10 Belief.** The shared network carries weight, modestly and unevenly: about
half the elites are indifferent to it; of those that are not, improvements
outnumber losses 1.4:1 in water and 1.5:1 in land and are larger (positive mass 2.4x
negative in water, 4.1x in land). In relative terms land 0.051 -> 0.076 (+48%)
and water 0.070 -> 0.083 (+18%); in air the scores are too small to read (0.006
-> 0.012). It is concentrated where the network trains hardest: the land island.
It has no visible effect on `mission_fraction` because that is zero for these
elites with and without it.

**Caveat: selection bias, which makes this an upper bound.** These elites were
selected while scored *with* the network. A design that happened to be helped by
it was more likely to enter and stay in the archive, and removing the network
now takes each elite out of the distribution it was selected in. So the delta
is an upper bound on what the network contributes to a design drawn without
that selection, not an estimate of it. The arch33 figure (-0.0014 +/- 0.0016) was
measured on a different set and is not comparable. The top 20 per medium by the
"with" score (land 0.505 vs 0.273 without) are further inflated, because they
were chosen on the "with" column.

**Control not run.** The brief asked, if cheap, for the same paired re-score on
non-elite designs. It is not cheap here: `events.jsonl` records scores and
descriptors per evaluation but no genome and no controller policy
(`kind: evaluate` rows carry mass, span, medium scores, never weights), and
`archive_*.pkl` holds only the 218 survivors. A control needs a fresh draw of
designs *with* per-design own policies trained under the network, which is a new
search arm, not a re-score. The unbiased version of this read is a search run
with the shared policy off (arch47 vs a flag-off arm), not an offline re-score.
arch45 was not run (an R sweep search holds the machine).

**12 Recommendation (R).** The shared policy is not inert: keep it, and treat it
as worth about +0.013 water and +0.025 land (+0.006 air, not significant) of
competence at most, concentrated in land-island designs. Evidence line: `land
+0.0247 +/- 0.0071 (t +3.50), water +0.0126 +/- 0.0049 (t +2.59), air +0.0061
+/- 0.0036 (t +1.69), n = 218, 128/218 elites move > 0.01`. For the decision this
README exists for: GRPO (item N) and an entropy sweep are worth building only
with a target of land and water competence, where there is a measurable
contribution to improve; they cannot be justified by `mission_fraction`, which
the network does not move. Before building either, take the unbiased read
above (a flag-off arm), because the upper bound here could shrink to nothing.
