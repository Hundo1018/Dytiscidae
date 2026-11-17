# Forces that do not come from the shape: an inventory

2026-11-17. The rule (the user, 2026-11-17): a lift, drag, thrust or torque
may only come from the simulated shape and its motion through the fluid
model. A number added to or multiplied onto a force, with no geometric
origin, is forbidden.

Method: a read-only pass over `physics/fluid.py` (all of it), the Mojo
mirror, `envs/`, `physics/jet.py`, `physics/rotor.py` (part),
`core/phenotype.py` (part) and `experiments/flapper/` by a subagent; the
rows marked (checked) were re-read by the session that wrote this file.
Not read: `amref.py`, `sections.py`, `core/sdf.body_panels`,
`envs/devroll/model_wp.py`, the rest of `rotor.py`. This is an inventory, not
a clearance: an entry's absence means it was not found, not that it is not
there.

## In one paragraph

Nothing in the solver multiplies or adds a typed number to lift directly:
`lift_scale`, `cd_scale` and `added_mass_scale` are 1.0 everywhere except in
the auditor's deliberate perturbations (checked: `auditor.py:226`). The
solver's forces come from each strip's chord, area, orientation, camber and
local flow. What is typed is (1) the separated-flow coefficients and the
rules that choose them, in which shape enters only through the chord and the
hinge radius and never through camber; (2) a handful of constants with no
geometric origin; (3) forces applied outside the strip solver; and (4) two
experiment-only devices in `experiments/flapper/rig.py` that break the rule
outright.

## 1. Empirical coefficients applied through a flow-state rule

| where | what | shape enters? |
|---|---|---|
| `fluid.py` `CN_PLATE` 1.98, `CN_LEV` 3.4 | normal-force coefficient of separated flow; cited (flat plate; Dickinson 1999). Chosen by `lev` | chord and r/c only. **No camber, cup or aspect-ratio term** (checked by measurement, RESULT_W5 P2: a Goe 417a wing and a flat one read the same to three decimals) |
| `LEV_ROSSBY` (3, 8), `LEV_TRAVEL` (2, 4) | when the vortex is on; 3 and (2, 4) cited, 8 a stated modelling choice | chord, radius, travel |
| `w = w + (1 - w) lev` | a strong vortex forces the separated branch at any angle | as above |
| `CD_CROSSFLOW` 1.1 | cylinder cross-flow drag on the projected area | area from shape |
| Wagner lag (Jones 1940), `cop_law` (Wang 2016; Han 2015), `PLATE_BY` 30 deg (RESULT_W5) | cited or from this repository's own comparison | chord |

## 2. Typed constants with no geometric origin

| where | what | note |
|---|---|---|
| `fluid.py:285-288` (checked) | stall at 11 deg, times a Reynolds law `0.55 + 0.45 log10(Re)/5` | no source in the code. RESULT_W5 P4: 3-5 deg late and 0.3-0.5 high for a flat wing |
| `SEPARATION_COMPLETE` 6 deg | width of the handover | stated as a modelling choice |
| `fluid.py:372` (checked) `oswald = 0.75` | induced-drag efficiency | inert with the live inflow; active in `steady()` (launch-speed and thrust-margin probes) and whenever inflow is off |
| `fluid.py:1624` (checked) `60 x weight` | cap on the total fluid load, all strips scaled together; sets `diag.clamped` | whether a score reads the flag was not checked |
| `phenotype.py:1127` | axial bluff Cd by part kind (hull 0.20, strut 0.9, ...) | not by fineness ratio |
| `fluid.py` directional added-mass coefficient | heuristic, exact for a sphere | geometric, not derived |
| `jet.py:114, 193` (checked) `refill_efficiency = 0.25` | a factor on jet thrust that differs by stroke direction | uncited; production path |
| `rotor.py:60-65` | the blade the rotor model sees is a fixed stand-in fitted to one propeller | whether the genome overrides it was not checked |

## 3. Forces applied outside the strip solver

Jets and rotors (`triphibian.py:1779-1785`), each with its own physics, added
after the solver and its cap. Initial-state assists: an air segment starts at
the measured trim speed; scatter widens the initial velocity (zero mean).

## 4. Experiment-only devices that break the rule

| where | what | status |
|---|---|---|
| `rig.py` `chord_law` | a typed factor on every strip's chord and area, switched by the stroke's direction; the wing's mass and geometry do not change | pass 1e only. Its results are a bound on what an in-stroke area change could do, not a mechanism |
| `rig.py` `strip_twist` (now `twist_setup` / `twist_turn`) | the fluid strips turned with no drive, no torsional inertia and no twisting rate in the flow | passes W6 and C1. Replaced by joints in pass W7 (`body.py wing_panels`, `freerig.run_free(chain=)`) |
| `freerig.py` `joint_damping = 0.0` | removes a typed bearing loss (0.0365 N m s/rad, itself typed) from the reported power | stated in its docstring |
| `l1a.py` `hypot(stroke torque, inertial torque)` | a reporting formula that assumes the two are a quarter beat apart | superseded by the torque-driven rig |

## What follows from it

- The cup: a concave-forward and a convex-forward plate get the same
  separated force. Until the separated law takes the section's shape, the
  model cannot show a membrane that bellies on the power stroke and flattens
  on recovery, in either direction (proposed as pass W8).
- The default stall law is the largest uncited input and is known to be
  wrong for a flat wing (RESULT_W5). Changing it is a comparability boundary.
- `oswald`, the load cap, Cd by kind and the jet's refill factor are each a
  small pass of their own: derive, cite or remove.
