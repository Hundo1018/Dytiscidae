# F3 result — 2026-11-12

`PREREG_F3.md` was frozen and committed first. Reproduce (147 rig runs,
about 3 minutes, none non-finite):
`PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/water_map.py experiments/flapper/results/water_map.jsonl --jobs 6`

The body: weight 0.721 N, buoyancy 0.748 N (it floats by 0.027 N). Wing
held still, drag 0.078, 0.25 and 0.84 N at 0.25, 0.5 and 1.0 m/s.

## Cheapest strokes that swim (thrust at least the drag), body held

| speed | stroke | power | vertical force swing | peak shoulder torque | Strouhal |
|---|---|---|---|---|---|
| 0.25 m/s | 0.5 Hz, 30 deg, feather 20 | 0.07 W | 2.8 body weights each way | 0.19 N m | 0.26 |
| 0.5 m/s | 0.5 Hz, 60 deg, feather 20 | 0.43 W | 5.4 | 0.43 N m | 0.25 |
| 1.0 m/s | 0.5 Hz, 60 deg, no feather | 2.0 W | 19 | 1.30 N m | 0.12 |
| 1.0 m/s, next | 1 Hz, 60 deg, feather 20 | 3.4 W | 21.5 | 1.64 N m | 0.25 |

The air stroke carried into water (3 Hz, 100 deg, feather 20): 13-18 N of
thrust, 138-140 body weights of vertical swing, 11.2-11.6 N m at the
shoulder, 190 W.

## The six predictions

1. *"A stroke that swims exists at each speed at 2 Hz or less."* **Held**:
   at 0.5 Hz at every speed (33, 26 and 17 of 48 strokes swim).
2. *"... under 2 W and asks a shoulder for under 0.42 N m."* Power **held**
   (0.43 W); torque **refuted by a hair** (0.43 N m).
3. *"With no feathering (amplitude 0) nothing swims at 1.0 m/s."*
   **Refuted**: one stroke does (0.5 Hz, 60 deg); a plate heaving across a
   stream makes thrust in a quasi-steady model.
4. *"... the vertical force swings by more than three body weights each
   way."* **Held**: 19.
5. *"... Strouhal number ... between 0.2 and 0.6."* **Held** at 0.25 and
   0.5 m/s (0.26, 0.25); **refuted** at 1.0 m/s (0.12).
6. *"The air stroke carried into water unchanged ... more than ten times
   the 0.42 N m of hover."* **Held**: 27 times.

## What the water asks of the design

- Thrust is not the difficulty. Half a hertz and a 30-60 deg stroke is
  enough on a held body, at a tenth of the power of flight.
- **The reaction is.** Even the gentlest stroke that swims throws 2.8 body
  weights up and down, and at 1 m/s nineteen. This machine is 73 g behind
  0.042 m2 of wing: under water the wings are the heavy thing and the body
  is the light one. Freed, an up-and-down stroke will move the hull through
  the water, not the wings, and most of this map's thrust will not be there.
  So the held rig cannot say the machine swims; a free body has to.
- What sheds the reaction, each seen in an animal or a machine (A11): less
  area in the water (alcids swim on a wing folded to about half), a stroke
  whose reaction is along the path (rowing: the blade broadside going back,
  edge-on coming forward), or a second pair beating against the first.
  Rowing and folding both need the outer wing to move fore and aft, the
  same degree of freedom A11 says a loaded step on land needs, and one the
  stroke-and-feather shoulder does not have.
- The shoulder that flies must not be driven in water as it is in air: the
  air stroke there asks 27 times the hover torque. Whatever drives the
  stroke has to be a slow, small-amplitude device in water; stage 2 has to
  carry both.

Limits: held body (the point above), quasi-steady model with its added-mass
terms as the package has them, one planform, no fold, coarse grid.
