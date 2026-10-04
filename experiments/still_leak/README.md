# Still machines that still crossed after 2026-10-03 — the ninth instance (2026-10-04)

**Question.** `experiments/transition_distance` (2026-10-04) found arch46's elites,
held still, crossing `air_to_water` 1.8-2.3%, `water_to_land` up to 5.0% and
`water_to_air` up to 0.9% under `transitions.CrossingTracker` (ARCH46_SPEC §8).
How, and what gate refuses it?

**Method.** `trace.py` re-runs each still crosser singly on the numpy path
(`run_transition`) and prints, per step, root x and z, depth, clearance,
`medium_of`, ground contact, `ashore`, and uprightness, plus the hold, the height
and energy height lost over it, and the step the crossing registered.
`make_fixture.py` writes the 24 crossing bodies' genomes to
`tests/fixtures/arch46_still_crossers.pkl` for the test.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/still_leak/trace.py \
        --run runs/arch46 air_to_water:194:0 water_to_land:14:0 water_to_air:129:0
    # --rotors-on: the 2026-10-04 still arm, which left rotors at their throttle

## Mechanisms, with the trace numbers (before the gates)

**`air_to_water`: a glider coasting on its launch speed.** All five still
crossers (elites 194 and 6 land_air, 33 and 8 aerial_diver, 72 air). Elite 194:
z 4.00 → 4.34 (1.0 s) → 3.89 at the command (1.5 s): 0.11 m of height lost,
hold 1.00. Then z 3.30 (1.74 s), 1.79 (2.24 s), water at 2.79 s. Speed fell from
20.3 to 17.4 m/s over the hold. Energy height `z + v²/2g` lost over the hold
(least-squares slope): 194 5.71 m, 6 5.56 m, 33 3.44 m, 8 3.51 m. The hold read
height only, which a phugoid keeps for 1.5 s while the speed goes. An unpowered
body cannot hold its mechanical energy (drag only removes it), so a passive
body can satisfy "hold" only by trading speed, and the energy reading sees it.

**`water_to_land`: a float on the submerged ramp is "LAND".** `medium_of` says
LAND when the root is dry and something touches terrain. Elite 14 (aerial_diver),
placed at x = −0.50 with its root 0.15 m under: it floats up, root 0.087 m above
the surface at 3.6 s, its hull resting on the ramp with its lowest geometry
0.91 m under water, x = −0.15, **8.15 m seaward of the shoreline**. LAND
registered at step 900 and it was still there at the end. Elite 52 at back 4:
the same, root 0.000 m at 5.78 s. Elite 11: root 0.06 m up at x = 1.0, flipping
between `air` (floating, no contact) and `land` (contact) and ending on `land`.

**`water_to_air`: not still.** Both crossers (elites 129 and 137, amphibian) carry
6 and 16 rotors. A rotor's CPG channel is a *speed* held at its offset; zeroing
amplitude left every rotor at its throttle. Elite 129 rose from 2.0 m deep to the
surface in 2.7 s on rotor thrust and was 1.0-2.3 m above it from 3.5 s. With the
rotors stopped (`TriphibianEnv.held_still_params`), neither crosses.

**Found on the way.**
- Graded shore progress was measured from the start, so drift *during the hold*
  paid: elite 129, rotors stopped, capsized (upright 1.00 → −0.63) and rolled
  0.66 m shoreward in the hold, 0.136 of a graded `water_to_land`.
- An aborted probe kept its hold: elite 82 held still blew up at 1.95 s (x 0.56 →
  5.81 m in 0.03 s, battery flat) and earned 0.543 of a graded `water_to_land`;
  diverged and unstable `water_to_air` probes earned 0.33-0.54.

## Gates (`envs/transitions.py`, both evaluation paths)

1. Air hold reads `max(height lost, energy height lost)` against the same 0.5 /
   1.5 m band. The four gliders above score 0 on it (3.4-5.7 m lost).
2. A land target counts only `ashore`: LAND by `medium_of` *and* the ramp under
   the root above the water surface.
3. Shore progress counts from the go command.
4. A probe that aborted (battery flat, diverged, unstable) has hold 0, so no
   graded approach.
5. The still machine stops rotors (`held_still_params`), and the
   `transition_distance` harness keeps the old arm as `rotors_on`.

Each is held by a mutation in `tools/mutate.py` and by
`test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing` or
`test_a_crossing_is_commanded_and_a_still_machine_makes_none`. Rates before and
after are in `experiments/transition_distance/README.md` §13.
