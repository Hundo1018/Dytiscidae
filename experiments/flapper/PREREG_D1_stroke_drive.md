# Pass D1, pre-registration: the stroke driven by mechanisms, not by a prescribed angle

2026-11-17, on `2be1ecc`. Frozen before any candidate is measured. The user,
2026-11-17: the real drive (linkage, pulley, other mechanisms) is not to be
fixed before several suitable simulated candidates exist; build candidates
first. Until now no linkage, pulley, tendon or cam was in any model:
`rig.py` prescribes the angles, `freerig.py` puts an ideal torque on each
hinge, and `crank_eq_*` is a torque computed from the prescribed sinusoid.

This pass chooses nothing. It builds candidates and reports a table.

## 0. Seen before freezing (probe, disclosed)

`scratchpad probe_fourbar*.py`: a crank-rocker closed by MuJoCo's
`<equality><connect>` on the built cruise flapper (right wing, hull welded,
no fluid, no gravity, crank on a stiff velocity servo at 2.5 Hz). Against
the four-bar's closed-form angle: at dt 2.5e-4 and the connect's `solref`
0.002, 1.3 deg off and a 0.64 mm gap in the loop; at solref 0.0005 (the
least MuJoCo allows at that step) 0.15 deg unloaded, 0.35 deg under a
0.7 N m stroke load and 0.53 deg under 1.5 N m; at dt 1e-4 and solref
0.0002, 0.06 deg and 0.03 mm under 0.7 N m. Turning off MuJoCo's safety
floor on solref diverged. Hence: **every run of this pass, the baseline
included, at dt = 1e-4 s with every loop's solref = 2 dt.** Also read:
W7's stored rows (cruise 2.5 Hz, stroke 40 deg, tip feather 15 deg: 7.24 W,
stroke torque peak 4.8 N m both wings; take-off 4.5 Hz, stroke 85 deg,
feather 50 deg: 129 W, 1.86 weights, 14 N m).

## 1. What is the same for every candidate

- The wing, body, fluid and law of W7's arms "R strips" (cruise) and "K1
  one panel" (take-off), now under the table law (repair R3), on
  `freerig.run_free`, body held: cruise at 8 m/s, level, 2.5 Hz, stroke
  40 deg, tip feather 15 deg by strips, root feather held at -2.94 deg (not
  re-trimmed per candidate: lift and thrust are read at the same feather);
  take-off at rest, body at 90 deg, 4.5 Hz, stroke 85 deg, root feather
  50 deg on its ideal servo. **The feather and the strip twist stay
  prescribed in every candidate** (that is pass D2), phased to the
  candidate's own stroke: their phase is the crank angle less the offset
  at which the stroke's fundamental crosses zero rising (closed form), or
  time for a drive with no crank.
- One motor drives both wings' stroke. The motor is a witness model, not a
  part chosen: torque on its shaft `tau = tau_s (u - omega/omega_0)`, u the
  command in [-1, 1], tau_s 1.0 N m, omega_0 1000 rad/s (250 W at its best),
  rotor 2e-5 kg m^2, copper loss `tau^2 omega_0/tau_s`, an ideal gearbox of
  ratio N, regeneration allowed. Electrical power = shaft power + copper
  loss. Sensitivity: tau_s 0.5 N m (125 W) at each candidate's best gear.
- The gear: each candidate gets its own best ratio from a sweep, N in {12,
  16, 24, 32, 48, 64, 90} (only ratios whose needed motor speed is under
  omega_0), best = least mean electrical power that holds the stroke to
  within 2 deg of its intended amplitude. Not modelled, so a fairness
  sweep, not a design.
- Link mass: capsules of 15 g/m (sensitivity 0 and 40 g/m at one point);
  no material named. Links carry no fluid load. Lever on the wing: a point
  on its span line (its feather axis, so the stroke drive does not load the
  feather) 30 mm from the shoulder (cruise) or 40 mm (take-off).

## 2. The candidates (each a body-and-constraint mechanism in the model)

- **DD, direct.** The motor on the stroke hinge through N, the other wing
  by a joint coupling (a gear); command from a position loop with
  feed-forward on the prescribed sinusoid. The only one whose amplitude and
  waveform can be changed in flight.
- **FB, crank-rocker four-bar, two cranks geared counter-rotating.** Each
  wing a crank below its lever, designed for a time ratio of exactly 1: the
  crank centre on the chord through the lever's two end points, crank
  radius = half that chord, coupler length = the centre's distance from the
  chord's middle (2.0 a).
- **SY, Scotch yoke and two links.** One crank on the centre line drives a
  yoke on a vertical slide (pin in slot: its height is exactly r sin(crank
  angle)); a link from each side of the yoke to each wing's lever, vertical
  at mid-stroke, length 2.0 a.
- **CT, crank and cable against a spring.** A cable (MuJoCo tendon,
  one-sided: it can pull and not push) from a crank pin to each lever pulls
  the wing down; a spring at the shoulder, at rest at the top of the
  stroke, returns it. Spring stiffness from a sweep, k in {1, 2, 4, 8, 16}
  N m/rad per wing, with the gear; reported: the share of the beat the
  cable is slack.

Not built, stated: a cam (by contact it is fragile in this engine; by a
prescribed angle it is what the user ruled out); a geared sector or rack.

## 3. Checks and predictions

- K1. Each mechanism in a vacuum (fluid off, gravity on), steady beats:
  mean shaft power within 1% of its peak (the loop neither makes nor eats
  energy); loop gap under 0.5 mm.
- K2. FB and SY in air at cruise: the stroke against its closed-form
  kinematics within 0.5 deg.
- K3. FB and SY: the downstroke's share of the beat 0.50 +- 0.02; the
  stroke's second harmonic under 5% of its first.
- K4. FB and SY at cruise against the baseline (the prescribed sinusoid,
  ideal torque, same dt): lift within 3%, thrust within 0.05 N, aerodynamic
  power within 5%. At take-off: lift within 5%.
- K5. DD costs more electrical power than FB at cruise, by over 15%,
  because the rotor reverses every half beat and its reflected inertia
  shows in the copper loss; at take-off the gap is smaller than at cruise.
- K6. CT: the downstroke share departs from 0.5 by over 0.03 at its best
  spring, and at that spring the cable is slack for under 10% of the beat.
  Not a check of anything but the candidate; no direction predicted for
  its lift.
- K7. Peak force in a loop closure: cruise 10-80 N, take-off 50-400 N.
- K8. The motor is on its torque-speed limit for under 10% of the beat at
  each candidate's best gear (the witness has enough in hand); at the
  sensitivity motor this rises.

Falsified by the stated numbers. A failed K1 or K2 means the loop is not
modelled well enough to read the rest of that candidate; it is reported and
the candidate's other rows are labelled.

## 4. Output

`drive/drive.py` (the candidates as XML surgery on the compiled model,
through freerig's compile seam), `drive/d1_run.py`, `drive/d1_read.py`,
`results/D1_drive.jsonl`; a figure of each candidate from the built model
at four crank angles, checked against the geometry above; the result file
with a table of candidates and no pick.
