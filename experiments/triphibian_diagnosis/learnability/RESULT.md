# L0 result — 2026-11-12, commit e01e90e plus this directory

`PREREG.md` (sha256 627a91ac...) was frozen first. Reproduce:
`PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/triphibian_diagnosis/learnability/probe.py --rounds 3 --jobs 8`
(23 minutes on 8 processes; rows in `results.jsonl`, fit statistics in
`summary.json`; `data/` is not tracked and is rewritten by the command).

## What ran

As designed. 274,247 decisions of data over four fits; five reads of 96
validation missions (seeds 200-231, three starts) plus the teacher's. All
four flag targets are exactly 0 or 1 (means over round 0: thrust below the
horizon 0.000, rotors on 0.773, limb pose 1.000, memory 0.005).

## Missions completed, of 32 per start

| who | from air | from water | from land | failed rollouts |
|---|---|---|---|---|
| teacher | 32 | 32 | 32 | 0 |
| round 0 (imitation, 71,970 rows) | 1 | 0 | 27 | 25 |
| round 1 (135,114 rows) | 10 | 4 | 32 | 8 |
| round 2 (205,339 rows) | 21 | 20 | 32 | 12 |
| round 3 (274,247 rows) | 22 | 24 | 32 | 9 |
| arm Z, input ignored | 0 | 0 | 0 | 54 |

Every miss that was not a failed rollout is the land leg that follows the
water (round 3: 4 from the air, 5 from the water); none is an air or a water
leg. The failed rollouts in round 3 are 6 unstable, 2 diverged, 1 flat pack.
From the land start, whose water leg comes last and is never left, the
student is at 32 of 32 from round 1 on.

Fit on held-out teacher missions: flag accuracy 1.000, 0.999, 1.000, 0.999
in every round; continuous channels 0.07-0.57 of a standard deviation in
round 0 and 0.07-0.18 after. On its own states the round-0 student's memory
flag agrees with the teacher's 0.902 of the time and its rotors-on flag
0.955; by round 3, 0.999 and 0.999.

## The six predictions

1. *"The teacher completes at least 30 of 32 from each start on seeds
   200-231."* **Held**: 32, 32, 32.
2. *"Round 0 (plain imitation) is under the bar (25 of 32) from at least one
   start."* **Held**: 1 and 0.
3. *"Some round r <= 3 reaches 25 of 32 from every start."* **Refuted**:
   the best is round 3 at 22, 24, 32. It rises every round and had not
   stopped rising.
4. *"Where round 0 fails it fails mostly on the leg that leaves the water (the
   memory bit and the staged exit), not in the air."* **Held**: 41 of its 43
   judged misses are the land leg after the water, 2 a water leg, 0 an air
   leg (and 25 rollouts failed outright).
5. *"Arm Z completes 0 of 96 and makes at most 2 transitions."* **Held**: 0
   and 0. 54 of its 96 rollouts failed, which is not a zero.
6. *"Held-out fit error does not predict closed-loop success: round 0's flag
   accuracy on held-out teacher states is above 0.98 whatever its missions
   do."* **Held**: 0.999-1.000 with 1, 0 and 27 missions.

## By the frozen rule

The bar was not reached in three rounds, so run 1 is not launched on the
present design, and the design question is the exit from the water onto the
land. What this probe does and does not say about it:

- The mission is learnable in the parts that do not leave the water: land
  start 32 of 32 with 135,000 rows.
- The exit is where a small error is not forgiven. The teacher's own
  zero-memory arm loses the same leg (23-27 of 32), and E0 found the exit
  tumbles when a rotor re-enters the water.
- More of the same may be enough: completions rose 1 -> 10 -> 21 -> 22 from
  the air and 0 -> 4 -> 20 -> 24 from the water, and the last collection
  (round 3's) had the student completing 23 and 23, so the data were still
  getting closer to the states that matter. This probe stopped where it said
  it would and does not show it.
- Not separated here: whether the misses come from the memory flag, from the
  rotors-on flag at the surface, or from the continuous demand during the
  staged wait. `results.jsonl` has the per-mission flag agreement; the
  trajectories were not kept.
- One fit seed, one network size, 60 epochs, no weighting of the rare
  states (the memory flag is on in 0.5% of rows).
