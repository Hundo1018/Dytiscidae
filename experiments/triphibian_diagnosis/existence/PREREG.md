# E0 — does one expressible body complete the mission under a hand-written controller?

Frozen 2026-11-10 before any rollout, commit a7770c2. No learning, no search:
a feedback law written by hand, the search's own bodies, physics and scores.
(The user's rule of 2026-11-10 forbids new training; this is a measurement.)

## 0 Question and decision

Can a body the genome expresses today, driven by a hand-written feedback
controller, (a) score above each medium's certified floor on the Tier-1
segments, (b) complete each of the four crossing probes, and (c) complete the
continuous mission, with the same body scoring 0 when its actuators are held
still?

Decision: if yes, the binding cause of sixty zero-mission runs is the
controller path (reports A2, A3), and the training plan is "give the learner
this action space and this teacher on a fixed body". If no, the reason it
fails says what must change in the body or the model before any run.

## 1 Rivals

- **It can** (controller path is the wall).
- **Physics cannot**: forward-only rotors and fixed buoyancy leave no way to
  hold depth and travel, or the rotor model cannot carry a body through the
  surface (entry or exit diverges: `bad_qacc`).
- **The scores refuse**: a gate (posture on land, hold in water, airborne
  definition, energy booking) reads 0 for a machine that is visibly doing the
  task.
- **Nothing there**: the reference quad's stored air score (0.858-0.922,
  2026-09-26, before the paired air score) does not reproduce under today's
  scoring.

## 2 Mechanism under test

The cascade in `experiments/rotor/fly.py` (velocity -> tilt, attitude PD,
altitude PD, least-squares mixer) is the only controller that has flown
anything here. The same cascade with (i) the rotor thrust coefficient of the
medium it is in, (ii) a depth target in water, (iii) reduced thrust and a
velocity loop on the ground, and (iv) a mode switch keyed to the commanded
medium, should be enough for a quad-class body with near-neutral buoyancy.

## 3 Predictions (written before running)

1. AIR, paired score through the Tier-1 path: competence >= 0.5 on >= 7 of 8
   seeds (stored unpaired 0.858-0.922). Falsified below 0.012 on >= 2 seeds.
2. WATER segment: >= 0.012 on >= 7 of 8 seeds; I expect >= 0.15 on at least
   half. Falsified if the body cannot hold a commanded depth within 1 m.
3. LAND segment: >= 0.012 on >= 7 of 8 seeds by thrust-assisted skidding.
   This is the prediction I trust least: the posture gate or the land domain
   test may refuse a body that is half carried by its rotors.
4. Crossings through `CrossingTracker`: `land_to_air` and `water_to_air`
   crossed; `air_to_water` crossed with hold >= 0.5; `water_to_land` crossed
   by lifting off after the hold and landing ashore. Any `bad_qacc` at the
   surface falsifies "the rotor model carries a body through the surface".
5. Continuous mission (`mission.run_continuous`, 8 s legs, one cycle), each of
   the three starting media: both commanded transitions complete on >= 7 of 8
   seeds.
6. Nulls: the same body with actuators held still
   (`TriphibianEnv.held_still_params`, rotors stopped) scores 0 on every
   crossing and mission, and below 0.012 in every medium.
7. Energy: the mission's booked energy fits the pack on the modified body.

## 4 Measurement

- Scores are the repo's own, unmodified: `evaluate.run_segment` (paired air,
  antipodal water and land), `transitions.run_transition`,
  `mission.run_continuous` + `continuous_score`. Only the command source is
  substituted, as `fly.py` does through `env.cpg.command`.
- Inert probe: prediction 6. Missing versus zero: a rollout that ends in
  `bad_qacc`, divergence or a flat pack is reported as failed, never as 0.
- Seeds: 8 per item, fixed in the script (0..7), no selection among seeds.
- Bodies: B0 the reference quad as built; B1 the same with a pack sized for
  the mission energy, hull volume set for density ratio 1.02-1.10, and
  ground gear, required to pass Tier-0. Every body edit is listed.
- Controller gains may be set by hand from the physics and a few trial
  rollouts; every trial is appended to `trials.jsonl` (no silent retries, no
  automated optimisation of any kind).

## 5 Design

- Controls: held-still arm on every item; B0 beside B1.
- Stopping rule: stop when the table is filled for B1, or when an item is
  shown to be blocked by physics or by a score gate (with the trace), or after
  about three hours of work. A clean negative is a result.
- Not allowed: editing anything under `dytiscidae/`, weakening or bypassing a
  gate, changing a score, selecting seeds, any optimiser.
