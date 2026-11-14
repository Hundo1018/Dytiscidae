# L1 pass 1b result, water: the reference machine's wings against the torque it has (2026-11-15)

Pre-registration: `PREREG_L1b_water.md` (commit 26d5e95, before any rollout).
Data: `results/L1b_water.jsonl`, 150 points, all finite, one pass.
`MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1b_water.py experiments/flapper/results/L1b_water.jsonl`
Body held level at 0.3 m/s under water; the forward force is the net on the
whole machine (not flapping: -0.47 N).

## Against the frozen text

| | frozen | measured | |
|---|---|---|---|
| G-K0 | stroke 85 deg, net forward force >= 0.94 N, shoulder torque <= 12 N m | **0 points** | failed |
| G-K1 | any stroke, >= 0.94 N, shoulder <= 15.8 N m, crank-equivalent <= 12 N m | **0 points** | failed |
| prediction | G-K0 met at 0.1 Hz or below | not met | falsified |
| prediction | G-K1 met at more points than G-K0 | 0 and 0 | falsified |
| prediction | best force per shoulder torque 0.1-0.25 N per N m | 0.58 at the fastest, largest stroke (187 N m); 0.03-0.06 among points under the cap | falsified both ways |

**A fault in the gate, found on reading: it is recorded, not repaired.** The
force the rig reports is already net of the body's drag, so "at least 0.94 N"
asked for thrust of three times the drag, not twice as the text meant. The
verdict above is against the text as frozen.

## What the map shows (description, after the fact)

Best net forward force with the shoulder under 15.8 N m (`L0/mechanism.py`,
which now reads this map):

| stroke | frequency | feather | net force | shoulder torque | crank-equivalent | power |
|---|---|---|---|---|---|---|
| 15 deg | 0.25 Hz | 10 deg | +0.38 N | 10.2 N m | 2.0 N m | 0.36 W |
| 25 | 0.15 | 10 | +0.27 | 8.9 | 2.8 | 0.30 |
| 40 | 0.10 | 10 | +0.35 | 9.3 | 4.7 | 0.34 |
| 60 | 0.05 | 0 | +0.33 | 12.9 | 9.6 | 0.37 |
| 85 | 0.05 | 10 | +0.47 | 9.8 | 9.8 | 0.39 |

- Under the cap the wings hold 0.3 m/s with a net 0.3-0.5 N to spare at every
  stroke, for a third of a watt. They do not give more than that: every point
  with 0.94 N or more asks 16.5 N m or more (the least: 25 deg, 0.25 Hz,
  1.29 N, 16.5 N m).
- Torque rises about as frequency squared at every stroke (85 deg: 7.7, 17,
  26, 56, 122, 187 N m from 0.05 to 0.6 Hz). The wing is 0.34 m2 of plate; it
  is far too much wing for water at any frequency a person would call
  flapping. Full stroke under the cap means a beat every 20 seconds.
- **The variable throw has a water case after all, and it is not the one the
  paper rule gave.** At the same shoulder torque and about the same net force,
  a 15 deg stroke runs at 0.25 Hz with 2.0 N m at the crank, where the full
  stroke runs at 0.05 Hz with 9.8 N m: five times the frequency for a fifth of
  the motor torque. It does not raise the force. This is a reading of a map
  whose gate failed, not a passed test.
- The feather that works under the cap is 0-10 deg; the 30-60 deg used in
  pass 1a was the wrong range for water.
- The broadside added-mass bound printed beside each point (2.9-8 N m at the
  rows above) is of the size of the cap. Whether the solver's wing force
  already carries all of it was not checked; if not, the rows above are
  optimistic by up to that much.

## Belief before and after

Before: slow enough, the same wings swim with force to spare under the air
drive's torque. After: they hold 0.3 m/s with little to spare and cannot
accelerate or fight a current on the held rig's numbers; speed scales with the
square root of the torque allowed, so the air drive bounds the swim near
0.3-0.4 m/s. It moves back if a wing that sheds area in water (folded, swept
back, or bending away) gives more force per unit torque: that is now a
requirement on the wing, not an option.

## For the plan

- L0 row 2 reads: closes at about 0.3 m/s with no margin on force, on a held
  body; the wing must shed area or load in water to do better.
- The fixture for the next pass gains a wing that folds or sweeps back in
  water (RoboFalcon's wing does both on one drive, A14 D) as a design
  variable, beside its stiffness.
- Not covered: free body, buoyancy and trim, other speeds, a compliant wing,
  the motor's ability to turn at 0.05-0.25 Hz under load.
