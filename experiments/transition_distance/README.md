# Crossing rate by probe start distance; air competence by launch height (ROADMAP Y/O) — 2026-10-04

## Frozen before the run

**0 Question.** `DistanceCurriculum` (`evolution/curriculum.py`) moves a transition
probe's start back from the interface by `step` whenever `advance_share` of the
last `window` evaluations crossed from where it is, to `max_back`; and steps the
air launch down from 30 m by `launch_step` to `launch_floor` whenever
`advance_share` of the window's air segments reach `launch_bar` competence. Its
placeholders: step 0.5 m, advance_share 0.5, window 200, max_back 20, launch_step
2 m, launch_floor 4 m, launch_bar 0.1. Decision: what to set them to before anyone
turns `--distance-curriculum` on, and whether it can start at all.

**1 Prior art.** `experiments/still_transitions/` (2026-10-03, before and after the
CrossingTracker fix) and `back.py` there, on the six seed body plans only: the
curriculum's numbers have never been read off a searched population.
`probe_continuity` (2026-09-22): 0 of 90 continuous missions completed both
transitions. No run's elites were scored under the 2026-10-03 crossing definition
(`transitions.CrossingTracker`: a commanded hold, then a directional crossing), so
this is the first measurement of crossing rates by distance under it.
Rivals for any rate: the placement (is the probe starting in the right medium?),
the network (each elite runs under the network that scored it), the fixed 6 s
duration (a far start may not have time to cross).

**3 Prediction (written before the run).** On arch46's 218 elites, which scored
`mission_fraction` 0.000 and air competence 0.012 on average:
1. **Still machines cross 0.000 of the time in every kind at every distance.** A
   non-zero cell is a bug in the definition and that cell is void.
2. **Elites cross < 0.50 (the placeholder `advance_share`) at back = 0 in every
   kind**, so the curriculum cannot start from the typed number for those kinds.
   Expected ordering, highest first: water_to_land, land_to_water, air_to_water,
   water_to_air (a deep start is the hardest).
3. Crossed share is non-increasing in `back` for every kind (more distance, never
   an easier crossing), except through noise of one elite.
4. Air competence is not a monotone function of launch height in a way that
   helps: the share of elites at >= 0.1 (`launch_bar`) is < 10% at 30 m, and
   lowering the launch to 4 m raises it by < 10 points, because the air score
   needs lift the arch46 elites mostly do not have.
Falsified: any still cell > 0; any kind >= 0.50 at back = 0 (then the curriculum
can start there and the step rule below is read for it); a rise in crossed share
with `back` beyond one elite of noise.

**4 Measurement.** `run.py`: all 218 elites, grouped by (gen, eval_seed) so each
runs under its own scoring network (`scoring_networks/gen<gen>.npz`), own stored
policy and mobility basis; one `BatchedFluid` per group; `run_transition_batch`
for each kind in {air_to_water, water_to_air, water_to_land, land_to_water} at
back in {0, 0.5, 1, 2, 4, 8} m, duration 6 s (the search's); the air segment
(`rollout_batch`, 6 s, the run's scatter and task draw) at launch height in
{30, 20, 12, 8, 4} m. Each cell is run twice: the elite, and the same body with
the actuators held still (CPG amplitude zero, phase and offset kept, no policy, no
shared network; the construction of `experiments/still_transitions/`). Recorded
per cell: `crossed`, `hold` and the share with `hold >= HOLD_PASS` (0.5),
`started_in` (is the probe in its start medium; a start in the wrong medium
measures nothing), `failure`. Check: air at 30 m must reproduce the recorded air
competence of the same elite.

**5 Design.** One process, `MemoryMax=3500M`, no pool. Elites of one archive are
a stand-in for the candidate population the curriculum will see; they are a
selected, better-than-average sample, so a rate here is an upper bound on the
population's. `land_to_air` is not measured (it ignores `back`).

    systemd-run --user --unit trdist-1004 -p MemoryMax=3500M env MUJOCO_GL=disable \
        PYTHONPATH=. DYTISCIDAE_KERNEL_DIR=<main checkout>/mojo/build \
        .venv/bin/python experiments/transition_distance/run.py \
        --run runs/arch46 --out experiments/transition_distance/results.json

## Appended after

**6 What ran.** The command in §5 (`trdist-1004`), all 218 elites in 148
(gen, seed) groups, each under its own gen's scoring network, `MemoryMax=3500M`,
no kill. Wall 15 110 s (4.2 h; 53 s per group early, 100 s late). Cells: 4 kinds x
6 distances x 2 arms x 218 = 10 464 transitions and 5 heights x 2 arms x 218 = 2 180
air segments, in `results.json` (`transition_rows`, `air_rows`). Every probe
started in its start medium (`started_in` right in 100% of cells). Check: air at
30 m reproduces the recorded air competence on 215 of 218 elites (|gap| <= 0.005).
Other failures logged across all cells: 1189 battery exhausted, 83 diverged, 59
unstable.

**8 Result.** Crossed share and hold-pass share (`hold >= 0.5`), n = 218 per cell.
"still" is the same body with the actuators held still.

| kind | back (m) | elite crossed | still crossed | elite hold-pass | still hold-pass |
|---|---|---|---|---|---|
| air_to_water | 0 | 0.014 (3) | **0.023 (5)** | 0.037 | 0.046 |
| | 0.5 / 1 / 2 / 4 / 8 | 0.014 (3) each | 0.018 (4) each | 0.037 | 0.041-0.046 |
| water_to_air | 0 | 0.000 | **0.005 (1)** | 0.972 | 0.968 |
| | 0.5 / 1 / 2 / 4 / 8 | 0.000 each | 0.005 / 0.009 / 0 / 0 / 0 | 0.959-0.977 | 0.959-0.982 |
| water_to_land | 0 | 0.023 (5) | **0.037 (8)** | 0.844 | 0.862 |
| | 0.5 | 0.009 (2) | **0.032 (7)** | 0.858 | 0.890 |
| | 1 | 0.014 (3) | **0.037 (8)** | 0.885 | 0.890 |
| | 2 | 0.023 (5) | **0.050 (11)** | 0.913 | 0.927 |
| | 4 | 0.005 (1) | **0.023 (5)** | 0.950 | 0.959 |
| | 8 | 0.000 | 0.000 | 0.945 | 0.959 |
| land_to_water | 0 ... 8 (all six) | 0.000 | 0.000 | 0.950-0.954 | 0.954-0.959 |

Counts in brackets. Which bodies: at back 0, `air_to_water` is crossed by 3 elites
(land_air 2, aerial_diver 1) and 5 still machines (land_air 2, aerial_diver 2, air
1); 3 elites are among the 5. `water_to_land` is crossed by 5 elites and 8 still
machines, 2 bodies in both. Across all distances 10 elites and 17 still machines
cross `water_to_land`, 3 and 5 cross `air_to_water`, 0 and 2 cross `water_to_air`,
none cross `land_to_water`. Mean graded `approach` at back 0, elite / still:
air_to_water 0.014 / 0.023, water_to_air 0.013 / 0.011, water_to_land 0.041 /
0.053, land_to_water 0.000 / 0.000.

Air competence by launch height (6 s segment, the run's scatter and task):

| launch height (m) | elite mean | elite share >= 0.1 | still mean | still share >= 0.1 |
|---|---|---|---|---|
| 30 (current spawn) | 0.0123 | 0.041 | 0.0069 | 0.018 |
| 20 | 0.0077 | 0.032 | 0.0059 | 0.014 |
| 12 | 0.0047 | 0.014 | 0.0044 | 0.009 |
| 8 | 0.0026 | 0.014 | 0.0022 | 0.009 |
| 4 | 0.0017 | 0.005 | 0.0016 | 0.005 |

**9 Against the frozen text.**
1. "Still machines cross 0.000 everywhere": **falsified.** Still machines cross
   `air_to_water` at 1.8-2.3%, `water_to_air` at 0-0.9% and `water_to_land` at
   2.3-5.0% (0 at 8 m). Per the frozen rule those cells are void as evidence of
   crossing skill, and the still rate exceeds the elite rate in every
   `water_to_land` cell (0.032-0.050 against 0.009-0.023) and in `air_to_water`.
2. "Elites cross < 0.50 at back = 0 in every kind": confirmed, with a margin of
   20x (largest 0.023). The predicted ordering (water_to_land > land_to_water >
   air_to_water > water_to_air) held only partly: water_to_land 0.023 >
   air_to_water 0.014 > land_to_water = water_to_air = 0.
3. "Non-increasing in back": `air_to_water` and `land_to_water` and `water_to_air`
   are flat; `water_to_land` goes 5, 2, 3, 5, 1, 0 elites (0 to 8 m), a rise of 3
   elites from 0.5 to 2 m, past the "one elite" allowance, so strictly
   **falsified** — at counts of 2 and 5 that is inside binomial noise
   (the still column rises in the same cells, 7 to 11).
4. Air: share >= 0.1 at 30 m is 4.1%, under 10%, as predicted. The second half
   ("lowering to 4 m raises it by < 10 points") held trivially but for the wrong
   reason: lowering the launch **lowers** competence, monotonically, 0.0123 to
   0.0017 mean and 4.1% to 0.5% at the bar. The curriculum's air rule assumes the
   opposite.

**10 What the data say.**
- No kind comes near `advance_share` 0.5 at back 0 (largest 0.023), so
  `DistanceCurriculum.update` would never move any start: it is inert, not
  harmful, for these elites.
- There is no crossing *skill* to ladder. In every kind the elite rate is at or
  below the still-machine rate. Whatever crosses is a body property: `water_to_land`
  crossers are 5 aerial_diver of the 8 still crossers.
- The distance dimension is not where the crossings fail. `air_to_water`: 96% of
  elites fail the commanded hold (hold-pass 0.037; the log reason "did not hold
  before the command to cross" appears 1906 times across all cells) at every launch height, so adding
  height adds nothing; the crossed share does not vary with `back` at all.
  `water_to_air` and `land_to_water`: the hold passes (97%, 95%) and the machine
  then never leaves its medium (approach 0.013 and 0.000). `water_to_land` is the
  only kind with a distance effect, and it is a fall to 0 at 8 m on counts of 0 to 5.
- Because the placeholder `step` of 0.5 m moves the rate by at most 3 elites in
  218 across 0 to 8 m, no step size is determined by this data.

**11 Belief.** For arch46's elites the crossing mechanism the curriculum assumes
(a rate near 50% at the interface that decays with distance) does not exist: the
rate at the interface is 0 to 2% and indistinguishable from a still machine, and
two of the four kinds are exactly 0 for elites. Not replicated on another run
(arch45 at the same commit would be the replication; not run, an R sweep search
holds the machine). The elites are a selected, better-than-average sample, so the
candidate population's rate is lower still.

**12 Recommendation (Y/O): do not turn `--distance-curriculum` on; the
curriculum cannot start.** Leave the typed numbers alone — they are not wrong, they
are unreachable: with `window` 200 and a crossing rate of 0.014-0.023, a window
holds 3-5 crossings and `advance_share` 0.5 needs 100. Lowering `advance_share`
to anything under about 0.10 would fire on noise or on the still-machine rate
(0.02-0.05), and the curriculum would then advance on bodies that crossed without
being asked to. Specifically, in order:
1. **Fix the still-machine crossings first** (surprise, finding 1). A still body
   crosses `air_to_water` 2.3%, `water_to_land` 3.7% and `water_to_air` 0.5%; a
   crossing rate is evidence only above that column, and every rate here is below
   it. The CrossingTracker closed the large leak (0.834 and 0.865 to ~0.02-0.04)
   but not all of it. Cause not investigated here; `experiments/still_transitions/`
   is the harness to bisect it, restricted to the aerial_diver and land_air plans.
2. **`air_to_water` needs its hold, not distance.** 96% of elites cannot hold
   level height for the commanded 1.5 s; no start height changes that. A curriculum
   for it would be on the hold (a shorter hold, or a launch at trim speed), which
   is a different knob from `step`.
3. **Drop the launch stepping (O) as written**: it steps the air launch *down*,
   and competence falls with it (0.0123 at 30 m to 0.0017 at 4 m; the share at the
   0.1 bar 4.1% to 0.5%). `launch_bar` 0.1 is cleared by 4.1% at the 30 m spawn,
   against `advance_share` 0.5.
4. If, after 1, a rate above the still column appears at back 0, set `advance_share`
   at 3 standard errors above the still rate of that kind
   (`still + 3 sqrt(still (1 - still) / window)`, about 0.05-0.10 at window 200), and
   re-run this script on a run that scored under the fixed definition to find where
   the elite rate falls to half its back-0 value; `step` is then that distance
   divided by 4-5 so the population sits one step from the edge.

Re-run: `experiments/transition_distance/run.py` command in §5 (4.2 h on one
process; `--n 60` gives a stratified 60 in about a quarter of that).

