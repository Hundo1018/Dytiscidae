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

(filled in below)
