# E0 result — one expressible body, a hand-written controller, the repo's own scores

Spec: `PREREG.md` beside this file (frozen 2026-11-10, commit a7770c2, not edited).
Run 2026-11-11 on main @ a7770c2, MuJoCo 3.15, numpy path only (no torch, no GPU
kernel, one process at a time). No learning, no search, no optimiser: gains were
set by hand from the physics and from the 156 trial rollouts in `trials.jsonl`.

## 0 Answer

**Yes, on every part of the question, on both bodies.** A quadrotor the genome
expresses, under a hand-written feedback law, scores in air, water and on land
through `evaluate.run_segment`, completes the four `run_transition` crossings,
and completes both commanded transitions of `mission.run_continuous` from each
of the three starting media. The same bodies with rotors stopped score
1.4e-17 or less in every medium and cross nothing. 320 final rollouts: no
failure string and no `bad_qacc` in the 160 hand-arm rows.

Four things qualify that yes, and they are the part the next plan has to carry
(sections 3 and 7):

1. The controller reads world position, a map of the beach, each rotor's wetness
   and the machine's model. None of them is in the policy's observation.
2. The crossing probes and the continuous mission do not depend on the seed, so
   "8 of 8" on those seven rows is one rollout eight times. From eight different
   entry states (probe X1) B1 crosses 32 of 32; B0 29 of 32.
3. The search's own aggregate still calls this machine a failure: Tier-1
   `mission_fraction` for B1 is 0.011-0.091 (probe X3), because it multiplies by
   a take-off height read from the land *segment*, where a machine that walks
   and stops as told does not leap. And the reference quad as built is rejected
   at Tier-0 before any rollout (`has_propulsor`).
4. Leaving the water is the one fragile manoeuvre: it works when staged and
   level; three tuning rollouts whose exit was not ended in a divergence (one)
   or a `bad_qacc` (two).

## 1 The table (PREREG section 3), seeds 0-7

`results_final.jsonl`, summarised by `summarize.py`. Segments are 8 s (the
search's `--segment-seconds`), wired as `evaluate_tier1` wires them: one
environment per seed, air then water then land, `_scatter_seed`, `task_seed`.
A rollout with a failure string or a `bad_qacc` would be listed as failed, not
as 0; there is none in the hand arms.

| item | B1 hand | B0 hand | B1 still | B0 still |
|---|---|---|---|---|
| air | **8/8** >= 0.5; min 0.891 med 0.911 max 0.927 | **8/8** >= 0.5; min 0.913 med 0.924 max 0.937 | 0/8 >= 0.012; max 0.000 | 0/8; max 0.000 |
| water | **8/8** >= 0.15; 2/8 >= 0.5; min 0.211 med 0.358 max 0.717 | **8/8** >= 0.15; 5/8 >= 0.5; min 0.401 med 0.788 max 0.853 | 0/8; max 0.000 | 0/8; max 0.000 |
| land | **8/8** >= 0.5; min 0.711 med 0.744 max 0.761 | **8/8** >= 0.15; 7/8 >= 0.5; min 0.188 med 0.679 max 0.744 | 0/8; max 0.000 | 0/8; max 0.000 |
| air_to_water | **8/8** crossed; hold 0.87 | **8/8** crossed; hold 0.96 | 0/8; approach 0.00 (did not hold) | 0/8; approach 0.00 |
| water_to_air | **8/8** crossed; hold 1.00 | **8/8** crossed; hold 1.00 | 0/8; approach 0.00 | 0/8; approach 0.00 |
| water_to_land | **8/8** crossed; hold 1.00 | **8/8** crossed; hold 1.00 | 0/8; approach 0.00 | 0/8; approach 0.00 |
| land_to_air | **8/8** crossed; hold 1.00 | **8/8** crossed; hold 1.00 | 0/8; approach 0.00 | 0/8; approach 0.00 |
| mission from air | **8/8** both transitions (2/2); `continuous_score` 0.704 | **8/8** (2/2); 0.700 | 0/8; **1/2**; 0.484 | 0/8; **1/2**; 0.575 |
| mission from water | **8/8** (2/2); 0.775 | **8/8** (2/2); 0.763 | 0/8; 0/2; 0.337 | 0/8; 0/2; 0.350 |
| mission from land | **8/8** (2/2); 0.792 | **8/8** (2/2); 0.795 | 0/8; 0/2; 0.148 | 0/8; 0/2; 0.148 |

Per seed, hand arm:

- B1 air 0.912 0.905 0.917 0.891 0.927 0.924 0.895 0.910
- B1 water 0.223 0.717 0.422 0.516 0.425 0.211 0.293 0.218
- B1 land 0.713 0.741 0.747 0.761 0.751 0.711 0.753 0.737
- B0 air 0.916 0.922 0.937 0.925 0.925 0.929 0.921 0.913
- B0 water 0.401 0.853 0.769 0.806 0.809 0.425 0.813 0.418
- B0 land 0.723 0.627 0.640 0.717 0.723 0.188 0.744 0.584

The seven crossing and mission rows are identical on every seed for every arm
(score, booked energy and end state of charge agree to the last digit):
`run_transition` draws the entry state from `transition_scatter_seed(kind)` and
`run_continuous` resets with `randomise=False` and no scatter. Their "8/8" is
n = 1. Still-arm maximum over the 48 still segments: 1.35e-17.

Bad qacc, battery, booked energy (every hand row; `logs/final_summary.md` has
the full per-item table):

| | B1 hand | B0 hand |
|---|---|---|
| `bad_qacc` rows, failure strings | 0 of 80, none | 0 of 80, none |
| end state of charge, segments and crossings (start 0.4-1.0 from `scatter`) | 0.433-0.955 | 0.428-0.949 |
| end state of charge, missions (start 1.0) | 0.988-0.993 | 0.974-0.986 |
| booked energy per rollout, Wh: air / water / land segment (second half) | 0.93-0.98 / 0.18-0.38 / 0.46-0.50 | 0.63-0.66 / 0.09-0.17 / 0.33-0.35 |
| booked energy, Wh: crossings | 0.43-0.60 | 0.28-0.47 |
| booked energy, Wh: the 24 s missions | 1.22-2.00 | 0.84-1.59 |
| mean power, W: air / water / land segments | 433 / 137 / 214 | 292 / 65 / 150 |
| peak draw, W; peak actuator overload | 2509; 2.86 | 2172; 2.86 |

The overload 2.86 is the rotor motor at its torque limit (copper loss over 0.35
of the continuous rating); it cannot exceed that, and `finalise_tier1` flags an
exploit above 3.0. B1's pack is rated 400 W continuous (Li-S, 2C): every air
rollout draws over it and is billed the `Battery.draw` overdraw penalty.

## 2 Verdict on each prediction

1. *"AIR, paired score through the Tier-1 path: competence >= 0.5 on >= 7 of 8
   seeds (stored unpaired 0.858-0.922). Falsified below 0.012 on >= 2 seeds."*
   **Confirmed.** B1 8/8 (0.891-0.927), B0 8/8 (0.913-0.937). Turn response
   0.93-0.95, height hold 0.92-0.98. fly.py's own stored result reproduces
   first (`probes/p01_fly_repro.py`: 0.8605, 0.8582, 0.9225 against stored
   0.8640, 0.8582, 0.9225).
2. *"WATER segment: >= 0.012 on >= 7 of 8 seeds; I expect >= 0.15 on at least
   half. Falsified if the body cannot hold a commanded depth within 1 m."*
   **Confirmed.** B1 8/8 >= 0.15 (0.211-0.717), B0 8/8 (0.401-0.853). Not
   falsified: mean depth error over the scored hold window is 0.00-0.08 m on
   B1's five cruise-first seeds and 0.00-0.01 m on B0's; on the three seeds
   whose hold comes first it is 0.34-0.96 m (B1) and 0.18-0.52 m (B0), inside
   1 m but still arriving, and the hold term is 0 there (section 3).
3. *"LAND segment: >= 0.012 on >= 7 of 8 seeds by thrust-assisted skidding.
   This is the prediction I trust least: the posture gate or the land domain
   test may refuse a body that is half carried by its rotors."* **Confirmed, and
   the fear did not happen.** The table's land mode is a taxi 3 cm clear of the
   beach: B1 8/8 (0.711-0.761), B0 8/8 >= 0.012 (0.188-0.744). Skidding on the
   feet as pre-registered, same controller with the clearance set-point at
   -0.012 m (X2, `trials.jsonl` step 27): B1 8/8 >= 0.012, 0.391-0.871, median
   0.64. The posture gate does not refuse a rotor-borne body; it charges the
   missing contact: gate 0.75-0.77 for the taxi (contact 0.52), contact
   0.73-0.86 for the skid.
4. *"Crossings through CrossingTracker: land_to_air and water_to_air crossed;
   air_to_water crossed with hold >= 0.5; water_to_land crossed by lifting off
   after the hold and landing ashore. Any bad_qacc at the surface falsifies 'the
   rotor model carries a body through the surface'."* **Confirmed for the four
   crossings** on both bodies (air_to_water hold 0.87 / 0.96, entry at 0.56 /
   0.40 m/s against a survivable 5.46; water_to_land ends ashore at x = 9.19 /
   9.20, 1.4 m past the waterline, in contact, rotors off). **The surface clause
   is refuted as a general statement:** no `bad_qacc` in the 320 final rows or
   the 64 X1 rows, but two tuning rollouts hit one and a third diverged, all at
   the surface, all on an exit that was not level (section 3). The rotor model
   carries a body through the surface when it goes through level, staged and
   once.
5. *"Continuous mission (`mission.run_continuous`, 8 s legs, one cycle), each of
   the three starting media: both commanded transitions complete on >= 7 of 8
   seeds."* **Confirmed, with n = 1 per start** (no seed dependence). Both
   bodies 2/2 from air, water and land; `continuous_score` 0.70-0.80. Time to
   enter the land leg after the command: 4.4 s (from air) and 5.3 s (from water)
   on B1, 4.7 s and 5.9 s on B0, of 8 s.
6. *"Nulls: the same body with actuators held still scores 0 on every crossing
   and mission, and below 0.012 in every medium."* **Confirmed for the media and
   the crossings, refuted for the mission.** Media: at most 1.35e-17. Crossings:
   0 of 8, graded approach 0.00. Mission: a still machine launched in the air
   falls into the sea and is credited with the air-to-water transition (1/2),
   and `continuous_score` pays a still machine 0.148 (from land), 0.34-0.35
   (from water) and 0.48-0.58 (from air): on-task time for the leg it starts in,
   the fall, and the depth term for sinking. `mission.py` says the search never
   reads this module; the still check is worth applying to it before anything
   trains on it.
7. *"Energy: the mission's booked energy fits the pack on the modified body."*
   **Confirmed for the mission that was run; refuted for the 45-minute
   `MissionSpec`.** The 24 s missions book 1.2-2.0 Wh of B1's 200 Wh. As
   `evaluate_tier1` extrapolates (segment mean power x 300 s x 3 cycles +
   `transition_energy`): B1 requires 196.1 Wh against 170.0 usable, 0.87, about
   39 of the 45 minutes; B0 requires 127.1 against 51.0, 0.40. A larger pack was
   not built; section 6 gives the scaling arithmetic (0.94-1.02 for 300-600 Wh
   of Li-S, inside its own error).

Rivals (PREREG section 1): *It can* stands. *Physics cannot* does not: holding
depth and travelling works by tilting the thrust, and the surface is passable.
*Nothing there* does not: fly.py reproduces and the paired score is higher than
the stored unpaired one. *The scores refuse* does not hold for any item, and
does hold in two places outside the table: Tier-0 for the bare quad, and the
`mission_fraction` aggregate (X3).

## 3 What limited or blocked each item

Nothing in the final table failed. What cost score, and what failed on the way:

* **Water, hold phase first (seeds 0, 5, 7): physics.** The hold is scored over
  the late half of its phase, so with the hold first the machine has 2 s to
  descend 1.0-2.5 m from the 4 m release and stop to within 0.05 m/s. Under
  water a rotor turns at tens of rad/s and its thrust is zero at an advance
  ratio of 0.243 (`probes/p06_advance.py`), the motor's torque limit caps the
  speed, and the airframe's vertical drag is large: the fastest descent or climb
  measured is about 0.9-1.2 m/s (`logs/p02_water_B1_s1.log`,
  `logs/p02_w2l_B1.log`). B1, 1.5 N heavy, sinks passively at 0.17 m/s, so it
  inverts to dive; B0, 6.8 N heavy, sinks at ~0.6 m/s. Neither arrives in time:
  hold 0 on those three seeds, 0.21-0.22 (B1) and 0.40-0.43 (B0) overall.
* **Water, near-neutral body: physics, and the PREREG's own body spec.** A
  thrust-vectoring body's horizontal authority under water is its net weight
  times the tangent of its tilt. B1 (density ratio 1.081) has to tilt about 70
  degrees to cruise at 0.30 m/s and invert to dive, and `upright` is in the
  water gate: gate 0.50-0.75. B0 (1.91) cruises nearly upright: gate 0.84-0.95,
  median score 0.79 against B1's 0.36. For this kind of machine the band
  1.02-1.10 is the wrong target; heavier is better in water.
* **Leaving the water: model and controller together.** A rotor's water thrust
  coefficient is 827 times its air one. One that has spun up in air and goes
  back under is hundreds of newtons for a few milliseconds (the model's own
  backward-Euler bound: about 2.4 N s per event, a 50 rad/s roll kick on a
  0.0086 kg m^2 airframe). First exits tumbled at 47-62 rad/s and then sat in a
  stable trap, two rotors under at 31 rad/s and two out at 600, skating sideways
  at 1.1 m/s (`logs/p02_mission_air_B1.log`, t = 17.8-24 s). An open-loop
  full-throttle burst was worse: one crossing diverged (x = -783 m) and one
  mission hit `bad_qacc` (`trials.jsonl` step 10). Relaxing the staging to 5
  degrees gave B0 a `bad_qacc` (step 24). The staged exit in the table (0.12 m
  under, level within 3 degrees, under 0.3 rad/s, then through) has not failed
  on B1 in 8 entry states; on B0 it fails 3 of 8 `water_to_air` entry states
  (X1), with attitude excursions to 90 degrees.
* **Yaw: body.** The reference quad's propellers are handed by side (the
  mirrored arms carry the other hand), so the yaw row of the mixer is -0.087
  times the roll row and the mix has rank 3 (`Hand._yaw_authority`). It cannot
  yaw itself; fly.py's least-squares mixer leaks every yaw demand into roll. The
  controller drops yaw. The cost shows on land: the root body origin (the hull's
  tail, 0.2 m behind B1's centre of mass) is the point every score reads, and
  the yaw rate `scatter` hands the machine (sd 0.25 rad/s) carries it sideways
  at 0.05 m/s against a 0.04 m/s command. One seed in four ran sideways
  (`logs/p02_land_B1_s3.log`, 0.0012) until the walk began by sitting down.
  Handing one diagonal pair the other way looks expressible from
  `phenotype.expand` (arms on a mast, the second pair's edge started from the
  other side); it was not built or tested: it is outside the pre-registered
  edits.
* **B0 on land, seed 5 (0.188): not traced.** B0 has no ground gear (it rests on
  its hull and arm tips); on that seed its across-track speed was 1.23 of the
  command and walk progress 0.25. The other seven are 0.584-0.744.
* **Controller bugs found and fixed before any of the above** (steps 1-5 of
  `trials.jsonl`): `data.subtree_com` is not from the same pass as `data.xipos`
  inside the rollout loop (lever arms 0.058 m off at 6.6 m/s, found with
  `probes/p05_dbg_mixer.py`; the machine settled 57 degrees nose-down,
  `logs/p02_air_B0_s0.log`); and fly.py's
  attitude gain is scaled by the inertia about the hull's tail end, 3.9 times
  the pitch inertia about the centre of mass (`probes/p03_inertia.py`).

## 4 Bodies

**B0** is `REFERENCE_PLANS["quad"]()` as built: 1.455 kg, density ratio 1.91
(0.74 L buoyant against 1.46 kg: 6.8 N heavy under water), 60 Wh Li-Po,
level-rig margin 0.84, no ground gear.

**B0 does not pass the Tier-0 gate the search applies.** `_structural_checks`
adds `has_propulsor` (applied 1, allowable 0, gating, margin -1.00) to any body
with no surface and no jet; rotors are not counted. `report.gate_margin` is
-1.00 against `SearchConfig.tier0_gate` -0.85, so `evolution/loop.py:520/694`
returns it unscored (`logs/facts_B0_final.log`). A bare multirotor cannot enter
a search here, seeded or not.

**B1** is B0 with these edits and no others (`triquad.B1_EDITS`, `b1_genome`):

| gene | B0 | B1 | why |
|---|---|---|---|
| hull length | 0.22 m | 0.40 m | volume; and the fore and aft discs no longer overlap (hub spacing 0.154 -> 0.28 m, disc radius 0.127) |
| hull radius | 0.050 m | 0.059 m | density ratio 1.081 (PREREG band 1.02-1.10); set by three builds (0.0655 -> 0.972, 0.0605 -> 1.053, 0.059 -> 1.081), `logs/facts_B1_v*.log` |
| battery | 60 Wh `lipo` | 200 Wh `lis` | section 6: covers 0.87 of the 45-minute extrapolation; Li-S is the densest cell in `materials.CELLS` and one `mut_global_energy` can draw; the same energy in Li-ion would weigh 1.0 kg instead of 0.57 |
| legs | none | four `FOOT` parts, no joint, CFRP, length 0.20 m, radius 6 mm, from the hull at 15% and 85% of its length, 50 degrees below horizontal, splayed 17 degrees fore and aft, mirrored | ground gear: feet 0.146 m under the rotor plane, stance 0.40 x 0.25 m |
| dorsal fin | none | one `PADDLE`, no joint, span 0.12 m, chord 0.05 m, mid-hull, pointing up | not for control: it is the surface `has_propulsor` asks for |
| `rotor_throttle` (arm part) | 0.45 | 0.555 | the base gait's rest speed at 1.05 x hover for the heavier body, so `level_margin` (the rig behind the air score's flight term, measured on the body's own gait) is 0.93 |

B1 as built: 2.045 kg (structure 0.864, battery 0.571, avionics 0.300,
actuators 0.240, seals 0.069), buoyant volume 1.845 L, density ratio 1.081 (1.5 N
heavy under water), Tier-0 gate passes (`gate_margin` +0.117, report `PASS`),
no air gates, rotor lift ratio 3.5, `flies_level` true, no self-contact in
flight.

## 5 The controller

`triquad.Hand`. One law for every item; it is not told which probe it is in.

* **Cascade** (fly.py's): velocity or position error -> desired force -> thrust
  direction -> attitude PD (per axis, on the inertia about the centre of mass)
  -> torques -> rotor thrusts -> rotor speeds. Torques are allocated before
  collective thrust, so a saturated climb does not cost attitude.
* **Medium-aware rotor law**: each rotor's coefficients are those of the medium
  its hub is in (air 1.23e-5, water 1.01e-2 N s^2), with a term for the flow
  through the disc (static below J = 0.07, zero at J = 0.243, both media) and
  for the velocity servo's droop (`kv (cmd - om) = kq om^2`).
* **Water**: net weight instead of weight; the thrust is pointed where the
  demand points; a target more than 0.2 m below is dived to with the thrust
  pointing down, left 0.08 m above it; a hold is flown upright (tilt at most
  30 degrees).
* **Land**: stop = rotors off. Walk = sit until still, then taxi 3 cm clear of
  the beach on a position reference moving at 0.07 m/s along the heading.
* **No task (crossings, mission)**, keyed to the commanded medium read from the
  observation. AIR: take off or leave the water, then fly to a station over deep
  water (x = -2) at the speed it has (at least 6 m/s) on a 45 degree glide
  slope. WATER: over deep water descend at 2.5 m/s, stop the rotors 0.45 m above
  the surface, fall in, hold 0.8 m deep; over the shelf, rest on it. LAND: leave
  the water, fly to x = 8.8, descend, rotors off on touch-down.
* **Leaving the water** is staged (section 3).

What it reads that is **not in the policy's observation**
(`TriphibianEnv.observation`: body-frame twist, gravity direction, `tanh(depth/5)`,
mean wetness, commanded one-hot, depth error, contact, battery, stroke, six task
channels, eight morphology channels):

| signal | where used | the observation has |
|---|---|---|
| world position x, y | every position reference: water hold, taxi, stations, landing point | nothing (by design: "no global position") |
| height above the ground or the sea (`env.clearance`) and world z | altitude holds, taxi height, rotor cut-off before entry, touch-down | `tanh(depth/5)` over the sea; in an air segment the height lost; nothing over the beach |
| world-frame velocity | all outer loops | body-frame velocity, and no heading outside a cruise phase |
| full attitude matrix | attitude loop | gravity direction (roll, pitch); heading error only while cruising |
| wetness of each rotor hub (`MediumField.properties`) | rotor law, entry and exit logic | one mean over the body's panels |
| the task phase itself: kind, world heading, speed, commanded depth in metres, and the instant it changes | segments (as fly.py) | cruise/hold flags, speed ratio, heading error, `tanh` of depth error |
| the clock (`data.time`) | moving references, settle time-out | nothing |
| the model: joint-space inertia (`mj_fullM`, added mass included), body masses and `xipos`, rotor positions and axes, servo gain and torque limit, dry mass and buoyant volume | force and torque scaling, mixer | eight morphology scalars |
| the rotor model: `bemt` static coefficients in air and water, the advance-ratio break points | rotor law | nothing |
| the map: `beach_surface_z`, slope, ramp extent, the waterline, three fixed way-points | terrain following, where to enter and where to land | nothing |

It does not read rotor speeds or contact forces.

What is composed or patched, all inside this directory:

* `env.cpg.command` is replaced on the instance (as fly.py does). The closure is
  the hand controller only for its own params token; for any other params — the
  base gait on the level and thrust rigs, the held-still arm — it calls the CPG's
  original method. (With fly.py's closure installed, `level_margin` is measured
  under the cascade instead of the body's gait; here it is measured on the gait,
  before installation.)
* The evaluators call `policy.act(env.observation(commanded))` at 25 Hz;
  `HandPolicy.act` reads the commanded medium from that observation and
  `HandBasis.command_policy` returns the token. No package code is changed.
* `run_transition` never reads MuJoCo's bad-qacc counter on the single-machine
  path, so the harness reads `data.warning[mjWARN_BADQACC]` after each crossing
  and reports a non-zero count as a failure.

## 6 Energy

Measured mean power, hand arm, mean over seeds: B1 433 W in the air segment, 137
in water, 214 on land; B0 292, 65, 150. Water is not free: at tens of rad/s the
motor works at its torque limit and the copper term `(tau/km)^2` dominates.
Braking a rotor is booked as consumption (`|tau om|`); it flattens nothing here
(lowest end state of charge 0.428, from a `scatter` start of 0.4-1.0).

The 45-minute `MissionSpec` (3 cycles x 3 media x 300 s), as `evaluate_tier1`
books it:

| body | pack | required Wh | usable Wh (0.85 x pack) | usable / required |
|---|---|---|---|---|
| B1 | 200 Wh Li-S, 0.571 kg | 196.1 | 170.0 | 0.87 (39 of 45 minutes) |
| B0 | 60 Wh Li-Po, 0.462 kg | 127.1 | 51.0 | 0.40 (18 minutes) |

The pack that was built covers 0.87 of it. A larger one was not built or run.
Scaling arithmetic only: hover power goes as mass^1.5, and under the hull-wall
rule each kilogram of pack brings 0.57-0.74 kg of PETG hull to stay inside the
density band (0.38-0.43 kg of hull per litre displaced on B1 and B0). With
required = 196 x (m / 2.045)^1.5 and m = 2.045 + k x (E - 200) / 350 for Li-S,
usable / required comes out 0.94-0.97 at 300 Wh, 0.96-1.01 at 400 and 0.94-1.02
at 500-600 (k = 1.74 and 1.57), on a 2.5-4.0 kg machine with a thrust-to-weight
of 2.8-1.8: at the edge either way, and inside the error of the arithmetic.
With Li-ion at 200 Wh/kg the same arithmetic gives 0.54-0.58. So the pack is
sized to the 24 s missions and to 39 minutes of the extrapolated one, not to
the 45, and whether the 45 can be closed on these rotors is not measured.

## 7 Where the code differs from the brief, and hazards for the next controller

* The four crossing probes have one entry state per kind and the continuous
  mission has one start per medium; the evaluation seed reaches neither.
* `transitions.SHORE_X` is 8.0; the ramp meets the still waterline at x = 7.80
  (`mjcf.SHORE_X` is 12.0, the ramp centre-line). B1's `water_to_land` start is
  placed at x = 3.99, 3.8 m seaward, and it sinks 0.15 m onto the ramp in the hold.
* The water task score has no depth term in its cruise half (`0.5 x progress +
  0.5 x hold`; the cruise's depth hold is published, not scored), and the hold
  needs a net vertical rate under 0.05 m/s over a 2 s window.
* The air task's flight term is gated on `level_margin >= 0.7`, measured by
  stepping the environment through `env.cpg.command`: whatever closure is
  installed is what gets measured.
* `scatter` calls `cpg.command(pose, 0.0)` and writes the result into the
  actuated joints' positions; a closure installed before it is called there.
* Mission domain tests (`mission.current_domain`): AIR is "not more than 34%
  submerged and no contact", so a body stuck at the surface read 74% on-task in
  air in one tuning rollout; LAND is any contact while not submerged.
* Tier-1's `mission_fraction` takes its take-off factor from the land segment's
  `takeoff_height` (floor 0.05, full at 0.30 m, capped at 0.29 for a wingless
  body), not from the `land_to_air` crossing (X3 below).
* `data.subtree_com` against `data.xipos`, and the mixer's rank, as in section 3.

## 8 Three probes that were not pre-registered

Kept out of the table; each is one file under `probes/`.

* **X1, entry states** (`x1_entry_states.py`, `results_x1.jsonl`): the four
  crossings with `transition_scatter_seed` replaced, in that process only, by a
  seed-dependent draw of the same distribution; seed 0 is the repo's draw. B1:
  8/8 on each kind, 32/32, air hold 0.87-1.00. B0: 8/8 on `air_to_water`,
  `water_to_land`, `land_to_air`; **5/8 on `water_to_air`** (lowest up-vector
  0.0-0.4 during the exit). No `bad_qacc` in the 64.
* **X2, skidding** (`trials.jsonl` step 27): prediction 3's mechanism on all
  eight seeds, in the table's item order: 0.391 0.662 0.648 0.689 0.564 0.521
  0.871 0.628. Before the walk began by sitting down, four skid variants scored
  0.000-0.118 on seed 3 (steps 13, 15, 17, 18).
* **X3, the search's aggregate** (`x3_mission_fraction.py`, `results_x3.jsonl`):
  `evaluate_tier1`'s body run line for line on an environment with the hand
  controller installed, then the repo's `finalise_tier1`. B1, seeds 0-7:
  `mission_fraction` 0.0107 0.0274 0.0183 0.0910 0.0378 0.0173 0.0181 0.0107.
  Its factors: competences as in the table, energy fraction 0.83-0.93,
  transition fraction 0.884, and a take-off factor of 0.05-0.24 (land-segment
  `takeoff_height` 0.000-0.071 m). Without that factor the same eight would
  read 0.21-0.55. A machine that walks and stops when told is multiplied by
  0.05; one that leapt 0.3 m in its walk phase would not be. (`mission_weight`
  has been 0.0 since 2026-10-10, so this number is reported, not selected on.)

## 9 What this does not show

* Not that a learner can do it. The teacher uses position, a map, per-rotor
  wetness and the model; a student limited to today's observation has none of
  them. The 20 trajectories under `trajectories/` carry states and rotor
  commands, not observations.
* Not robustness of the mission: one rollout per starting medium.
* Not walking. The land item is passed by flying 3 cm high, or by dragging the
  feet at 94% thrust; the land score pays either 0.4-0.9.
* Not the 45-minute mission: 24 s were flown, and the pack covers 39 minutes of
  the extrapolation.
* Not B1 on a pack one could buy today: Li-S is the cell `materials.py` itself
  labels optimistic.
* Tuning was by hand on seeds 0-3 and the final table includes them; seeds 4-7
  were first run in the final pass (B1: air 0.895-0.927, water 0.211-0.425,
  land 0.711-0.753).

## 10 Reproduce

```bash
cd /home/hundo/Projects/Dytiscidae/dytiscidae
export MUJOCO_GL=disable PYTHONPATH=.
E=experiments/triphibian_diagnosis/existence
.venv/bin/python $E/probes/p01_fly_repro.py 0 1 2           # fly.py's stored result, 7 s
.venv/bin/python $E/triquad.py facts --body B1              # build, Tier-0 gate; also --body B0
rm -f $E/results_final.jsonl; bash $E/run_final.sh          # the table: 320 rollouts, 16.5 min, one process
.venv/bin/python $E/summarize.py                            # the table and the energy rows
.venv/bin/python $E/probes/x1_entry_states.py B1 hand       # X1 (B0 likewise), 50 s each
.venv/bin/python $E/probes/x3_mission_fraction.py B1 hand 0 1 2 3 4 5 6 7     # X3, 2.5 min
.venv/bin/python $E/triquad.py run --body B1 --arm hand --items air,water,land --seeds 0-7 \
    --trial "X2 skid" --gains '{"c_ref": -0.012}'           # X2 (appends to trials.jsonl)
.venv/bin/python $E/probes/trials_digest.py                 # the 27 tuning steps, one line each
.venv/bin/python $E/probes/p02_trace.py B1 water_to_land 0 25    # a step trace of any item
```

The rollouts are deterministic: the crossing and mission rows of the last tuning
step and of the final run agree to the last digit.

## 11 Files

| | |
|---|---|
| `triquad.py` | bodies, Tier-0 check, controller, harness (sha256 prefix 1cc5f4bd199505f4; `logs/triquad_at_final_run.py.txt` is the copy the table was run with) |
| `run_final.sh`, `summarize.py` | the final pass and its table |
| `results_final.jsonl` | 320 rows: body, arm, item, seed, score, failure, bad_qacc, end state of charge, booked energy, measurements |
| `trials.jsonl` | 156 tuning rollouts in 27 steps: what changed, what it scored (`logs/trials_digest.txt`). Step traces re-ran logged configurations and are under `logs/p02_*.log` |
| `results_x1.jsonl`, `results_x3.jsonl` | probes X1 and X3 |
| `trajectories/{B0,B1}_<item>_seed0.npz` | 20 files, one per item and body, all successful: `time`, `qpos`, `qvel`, `commanded_medium` (index into air/water/land, -1 before the first observation), `rotor_cmd` (rad/s), `episode` (the two halves of a paired segment), `meta` |
| `probes/` | p00 body facts, p01 fly.py, p02 step trace, p03 inertia, p04/p05 mixer debugging, p06 rotor advance ratio, p07 mode timeline, x1, x3, trials digest |
| `logs/` | every command's output |
