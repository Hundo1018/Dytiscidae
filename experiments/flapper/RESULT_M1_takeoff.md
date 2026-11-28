# Pass M1, result: a standing take-off from the beach

2026-11-18. Frozen text: `PREREG_M1_takeoff.md` (`e70721f`). Code
`m1_takeoff.py` (sha256 `ba0b7f69180a50ee…`, written as the first row of
`results/M1_takeoff.jsonl` before any scored row), `m1_read.py`,
`m1_draw.py`; tuning rows `results/M1_tuning.jsonl` (67); scored rows
`results/M1_takeoff.jsonl` (186, none with a QACC reset, none excluded).
Harness `mission.py` as of `e70721f`; the probe rows of section 0 of the
pre-registration reproduce on it (P2 step: every traced value equal; P1 K1:
lift 1.840449 both times). The controller reads the true state.

## Outcome

**No pre-registered procedure arm takes off.** V (vault), Vs (slower
vault), P (push-up) and VO (vault, open loop) pass at 0 of 8 scoring entries
at every rotor and gear. Every one of them leaves the ground, most climb
past 2 m, and every one fails G4 (attitude). The cause is measured: in
hover the machine has a lateral (wing-down) divergence that doubles in
about 0.25 s, and the two lateral levers the frozen law has carry no
sustained authority on that axis. The left-right stroke-amplitude
difference makes 0.87 N m on a held hull, but the free hull counter-spins
and the wings' absolute amplitudes come out nearly equal. The left-right
feather offset mostly spins the body.

**Added after data, not pre-registered** (section 0b): the same law plus a
third lateral lever, the left-right feather-amplitude difference, run as
arms V+ and P+. Its code was hashed before scoring. **V+ takes off at 8 of 8
scoring entries at every rotor and both gears, and at 2 of 2 across-slope
entries and 3 of 3 contact sensitivities.** It strikes its wings on the
beach and is off the ground for good 0.10-0.12 s after the command. It
passes 2 m at 1.1-1.6 s and hovers at 4.0 m, nose 87 deg, drifting
0.4-1.4 m in 9 s. P+ (push-up, then flapping, no deliberate strike) passes 8
of 8 at every rotor. Its flapping begins with the wings striking the beach
(48-62 weights), and it is off the ground at 1.3-3.2 s.

**What limits it is the strike and the power, not lift or torque:**
- **The strike.** The take-off rests on a rigid wing used as a leg. V+ puts
  a peak normal force of 25.5 weights (150 N) on one wing, and a moment
  about the shoulder 17 times the largest fluid moment of the hover. The
  model has no wing structure, so that number is a load to be checked
  against structure, not something the model checked. It is also a
  soft-contact event: the contact time constant is 4 ms, the box is rigid,
  and the sand does not give.
- **The power.** Peak electrical power for all motors together:
  - instantaneous: 3.6 kW at the witness rotor, 2.4-2.7 kW at the lighter
    ones;
  - largest 0.1 s mean: 1.55-1.83 kW;
  - against three cells' 462 W continuous (reported, not a gate).
  The hover after the climb takes 343 W at J_r 2e-5 and 140-176 W at the
  lighter rotors.
- **The model.** The climb and hover are at near-zero airspeed with 50 deg
  of feather: the regime where the strip solver has no dynamic stall (X2:
  OpenFOAM read 1.5-3.5 times the forces there, unqualified).

## 0b. Changes after the pre-registration, disclosed

- **Tuning:** 67 logged rows, within the frozen budget of 80 rollouts or 4
  h; about 2.2 h of wall time.
  - kind `auth`: 9, the free-body sign measurement, first over one beat
    and then over six;
  - kind `hover`: 20, released at 30 m with the loop on;
  - kind `olstep`: 6, open-loop steps;
  - kind `tune`: 32, the arms on tuning entries.
  Every one is on a tuning entry or in the air at 30 m.
- **The attitude PD's parametrisation.** Section 4 said "wanted angular
  accelerations, natural frequency w_a, gains from wanted acceleration to
  each output". The free-body rows showed the outputs do not act as
  constant accelerations (a bias step turns the body to a new angle and
  stops). So the outputs are set directly by a PD on the error rotation and
  the beat-mean rates: bias from about the left-wing axis (kpy 0.4, kdy
  0.10), left-right feather offset from about the back axis (kpz 0.4, kdz
  0.15), left-right amplitude difference on the spin rate (kdx 0.10). The
  same outputs, limits, beat means and half-beat low-pass are kept. Two
  further gains on the same outputs (kpz_a, kdx_f) were tried and left at 0.
  - **The frozen arms ran with these first values.** Their signs on the
    back axis and spin were later found to be the wrong way (kpz, kdz,
    kdx).
  - The hover rows show no setting of these levers (either sign, gains to
    saturation, both at once) changed the lateral divergence: wing 22-35
    deg down at 3.0-3.2 s in every case.
  - So the frozen arms' G4 failures do not depend on that choice. No row
    proves it for the scored entries.
- **ADDED after data: the feather-amplitude difference** `Psi_lr`. The left
  wing's feather amplitude is raised by half of it and the right's lowered,
  clipped at 0.3 rad.
  - Measured open loop: +0.15 rad turns the left wing up 30 deg in 0.6 s.
  - Law: `Psi_lr = -kpP e_z + kdP w_z`, with kpP 1.5 and kdP 0.2, tuned on
    the 5 tuning entries (all pass) plus P+ on 3.
  - Arms V+ and P+ are V and P with it; every V and P row of section 2 is
    run again as V+ and P+.
  - **This is outside the frozen structure, so it is a measured candidate,
    not a pre-registered verdict.**
- The figures' snapshot times (rest, +0.06, +0.15, +0.5, +2, +8 s) were
  picked after the run. The +0.06 s panel misses the peak contact, which
  falls between samples.

## Table (scoring entries: x 15/20/25/30 m, nose up and down the beach)

| arm | J_r (gear) | pass | G1 G2 G3 G4 | t_off s | t H>=2 m s | H end m | peak wing contact, weights | P peak W | P 0.1 s W | P last 3 s W |
|---|---|---|---|---|---|---|---|---|---|---|
| V | 2e-5 (24) | **0/8** | 3 1 4 0 | 0.12-9.0 | 1.33-1.52 | -0.2-4.9 | 25.5-211 | 3590-3657 | 1824-1826 | 224-311 |
| V | 5e-6 (32) / 1e-6 (32) | **0/8**, **0/8** | 7 3 4 0 / 8 8 8 0 | | | | | 2653 / 2370 | 1590 / 1554 | 120-256 / 109-140 |
| V | 5e-6 (24) / 1e-6 (24) | **0/8**, **0/8** | 5 5 5 0 / 6 6 6 0 | | | | | 2748 / 2310 | 1775 / 1750 | |
| Vs | 2e-5 (24) | **0/8** | 8 7 8 0 | 0.10 | 1.18-1.39 | 1.7-5.0 | 16.1-16.2 | 3470-3528 | 1690 | 250-269 |
| P | 2e-5 / 5e-6 / 1e-6 | **0/8 each** | 7 7 7 0 / 8 8 8 0 / 8 7 8 0 | 1.3-8.9 | 1.9-3.9 | | 31-176 | 3346-3756 | 1605-2169 | 116-349 |
| VO | 2e-5 (24) | **0/8** | 8 5 5 0 | 0.12 | 0.66-0.78 | -0.1-8.5 | 25.5 | 3126-3143 | 1825 | 213-230 |
| R0 | 2e-5 (24) | **0/8** | 0 0 0 4 | (on the ground) | - | | 0-43 | 2977-3926 | 1390-1756 | 518-665 |
| N0 | 2e-5 (24) | **0/8** | 0 0 0 8 | (on the ground) | - | 0.00 | 0 | 0 | 0 | 0 |
| N1 | 2e-5 (24) | **0/8** | 0 0 0 0 | (on the ground) | - | -0.1-0.5 | 68-84 | 4056-4070 | 2150 | 712-802 |
| N2 | 2e-5 (24) | **0/8** | 0 0 0 8 | (back down at 1.35-1.50 s) | 0.58-0.61 | -0.1 | 25.5-71.5 | 2905 | 1825 | 0-1 |
| **V+ (added)** | 2e-5 (24) | **8/8** | 8 8 8 8 | 0.12 | 1.33-1.52 | 3.99 | 25.5 | 3590-3657 | 1824-1826 | 343 |
| V+ | 5e-6 (32) / 1e-6 (32) | **8/8**, **8/8** | all 8 | 0.10-0.11 | 1.12-1.46 | 4.00 | 24.0 / 21.2 | 2653-2658 / 2370 | 1590 / 1554 | 166 / 140 |
| V+ | 5e-6 (24) / 1e-6 (24) | **8/8**, **8/8** | all 8 | 0.12 | 1.24-1.58 | 3.99 | 21.3 / 19.3 | 2748 / 2310 | 1775 / 1750 | 176 / 153 |
| **P+ (added)** | 2e-5 / 5e-6 / 1e-6 | **8/8 each** | all 8 | 1.3-3.2 | 1.9-3.9 | 3.96-4.00 | 31-62 | 3347-3756 | 1603-2170 | 338-349 / 163-166 / 138-140 |
| V, stress (across the slope) | 2e-5 (24) | 0/2 | 0 0 2 0 | | | | 137-169 | | | |
| V+, stress | 2e-5 (24) | **2/2** | all | 0.12 | 1.69 | 3.98 | 25.7 | 3659-3682 | 1824 | 349 |
| V / V+, contact ×0.5, ×2, friction 0.5 (entry 20, up) | 2e-5 (24) | 0/3 / **3/3** | | 0.12 (V+) | 1.44-1.54 | 3.99 (V+) | 48.1 / 14.4 / 25.5 (V+) | 3526-3619 | 1786-1826 | 343 (V+) |

Further reads:
- **Energy** from the command to the end, 9 s, at 2e-5: V+ 3311 J, P+
  2409-2924 J, N1 6768-7207 J. V+ at 5e-6 (32) 1664 J, at 1e-6 (32) 1427 J.
- **Mean power over the first 2 s** of V+: 482 W at 2e-5, 260-263 W at
  5e-6 (32), 228-230 W at 1e-6 (32).
- **Largest beat mean** of V+: 1616 W at 2e-5.
- **Peak stroke-hinge torque:** 25.3-43 N m. This is the motor's curve
  while back-driven: `tau_s (u - omega/omega_0)` with the hinge turned
  against u gives more than the stall torque, and the strike and touch-downs
  do that.
- **Strike window:** the stroke motors are on their curve for 100% of it in
  the vault arms.
- **Contact moment about the shoulder** in V+: 108 N m, against 6.3 N m of
  fluid moment in free flight.

## The drawings

`figures/M1_Vplus_20_0.png`, `M1_V_20_0.png`, `M1_N2_20_0.png` and
`M1_Pplus_20_0.png`, entry (20, up), J_r 2e-5, re-run through the scored
code. Each shows side and front views of every geom at six instants and the
traces. Opened and checked:

- **Rest.** Belly and tail on the beach, wings flat, span 1.40 m tip to tip
  in the front view; the tail on its 0.3 m boom behind. Matches P0.
- **V+.**
  - +0.06 s: the wings an inverted V, the tips at the beach line.
  - +0.15 s: the wings straight down and the body clear.
  - +0.5 s: nose about 70 deg, wings sweeping.
  - +2 s and +8 s: nose vertical, wings level in the front view, tail
    below, 3.9 and 5.5 m up.
  - Traces: centre-of-mass rise to 4.0 m and flat, nose 85-89 deg from 1 s,
    no contact after the strike, power at about 2 kW in the 0.05 s
    samples, then 250-450 W.
  - Matches the row: G 1111, t_off 0.12 s.
- **V (frozen).** The same as V+ up to 4 s. Then the wing goes down, the
  body turns over, climbs again to 6.3 m and falls onto the beach by 9 s:
  the lateral divergence of the outcome. Matches G 0000.
- **N2.** The strike throws it to 2.04 m, nose 87 deg at the top, at 0.70 s;
  it falls on its back, wings held straight. Back on the beach at 1.5 s;
  lying by 2 s.
- **P+.** It stands on its wing tips at 0.55 m, rocks back, and the first
  beats strike the beach (an 11-weight load spike in the trace). Off at
  1.77 s, then the same hover.
- **What the drawings show that the design does not have:** the wing as a
  leg. Nothing in the built wing is a structure that takes 150 N at its tip.

## The frozen checks against what was read

| | frozen | read | |
|---|---|---|---|
| C1 | N0 0/8, H < 0.01 m, load 0.98-0.99 | 0/8, H 0.0000, load 0.986 | holds |
| C2 | N2 0/8; peak H 1.5-2.6 m within 0.7 s; back down within 1.6 s | 0/8; 2.02-2.05 m at 0.70 s; back at 1.35-1.50 s | holds |
| C3 | N1 0/8, failing G1 or G4 | 0/8, failing all four: it flips (nose down to -85 deg) and never leaves the beach for good | holds |
| C4 | V at 2e-5 >= 7/8; t_off within 0.4 s; H >= 2 m within 1.2 s | 0/8 (G4 at every entry); t_off 0.12 s where it left; H >= 2 m at 1.33-1.52 s | **wrong** on the verdict and the 1.2 s; right on t_off. (V+, added, 8/8 with the same timing) |
| C5 | VO: G1 and G2 8/8; G4 fails at >= 3/8 | G1 8/8, **G2 5/8**, G4 fails 8/8 (it pitches over: nose -49 to -63 deg) | the G2 part **wrong**; the G4 part holds |
| C6 | P <= 4/8 | 0/8 | holds (P+ 8/8, added) |
| C7 | Vs passes no more than V; peak wing contact >= 30% under V's | 0 against 0; 16.1 against 25.5 weights (37% under) | holds |
| C8 | wing contact 5-30 weights; contact/fluid moment 2-10; stroke on its curve >= 90% of the strike; hinge 22-24 N m | 25.5 (V+, V where it did not fall); **17.1**; 100%; **25.3-43 N m** (back-driven) | two hold, two **wrong** |
| C9 | peak 1.8-3.0 kW at every rotor; 0.1 s 1.2-2.5 kW; last 3 s 300-550 / 150-300 / 120-250 W; over 462 W at the strike at every rotor and at 2e-5 in the last 3 s | **3.6 kW at 2e-5**, 2.4-2.7 kW lighter; 1.55-1.83 kW; V+ 343 / 166 / 140 W (V 224-311 / 120-256 / 109-140); over 462 W at the strike everywhere, **under it in the last 3 s at 2e-5** | peak at 2e-5 and the last-3 s comparison **wrong**; the rest holds |
| C10 | V at 5e-6 and 1e-6 (32) >= 7/8; at gear 24 the lighter rotors vault higher | 0/8 and 0/8; H in the first 0.5 s 1.21-1.25 m against 1.18-1.21 | the verdict **wrong**; the vault part holds, narrowly. V+ 8/8 at both gears |
| C11 | R0 0/8; moves under 1 m either way | 0/8; 0.15-0.25 m facing up, **1.6-3.0 m facing down** (it slides down the slope on its belly, still in contact) | the count holds; the motion facing down **wrong**. Its wing touched the beach (up to 43 weights) despite the 15 deg cap |
| C12 | no QACC reset | none in 186 rows | holds |
| C13 | V across-slope 2/2 (0.5); S rows 3/3 (0.6), H of the first 0.5 s within 30% | V 0/2, 0/3 (G4 again); H 1.18-1.30 against 1.18-1.21 m | the passes **wrong**; the height part holds. V+ 2/2 and 3/3 |

Seven of the thirteen frozen checks are wrong in whole or in part. They have
two things in common:
- **I expected the hover's attitude to be the easy part.** The vault, the
  lift and the motors were not the problem.
- **I sized power from M0's held rig.** The strike costs 3.6 kW for an
  instant; the hover after it costs less than I predicted.

## What this says, as ranges

- **Lift and torque are not the limit.** With the K1 stroke the machine
  climbs and hovers on the witness motor at every rotor, with the stroke on
  its curve only in the strike.
- **Two ways off the beach both use the wings on the ground.**
  - **The vault:** a stroke command stepped to -77 deg. It throws the body
    2.0 m by itself, nose up to 87 deg (N2).
  - **The push-up:** the body stands on its wing tips at 0.55 m, then the
    first beats strike the beach.
  - Flapping from the belly without either (N1) only flips it.
  - A ground run (R0) with a stroke that clears the beach does not start:
    it barely moves facing up, and slides 1.6-3.0 m on its belly facing down.
- **Hover attitude needs a lateral lever the frozen law did not have.**
  - With the feather-amplitude difference (V+, P+) it holds 4 m with the
    span axis within 0.8 deg of level from lift-off on (P+ 6.5 deg) and
    body rates under 0.16 rad/s over the last 3 s (P+ 1.5), at all rotors.
  - The machine is unstable laterally in hover without it. Whether a real
    feather drive can make a beat-to-beat amplitude difference is not asked
    here (D2's crank cannot; a motor on each feather can).
- **The strike is the weak point.**
  - V+: 25.5 weights (150 N) on one wing, 108 N m about the shoulder; with
    half the contact time constant 48 weights, with twice it 14.4.
  - Vs's slower strike, 16 weights, still lifts off and climbs (G1-G3 at 8,
    7, 8 of 8) under the frozen law.
  - So a gentler strike is available. It was not run with the added lever.
- **Power.** Instantaneous 2.3-3.7 kW at the strike for all motors together,
  0.1 s means 1.5-2.2 kW. Mean over the first 2 s of V+: 482 / 261 / 229 W
  at the three rotors. Hover 140-349 W. The pack figure of 462 W is
  exceeded only in the strike and the take-off's first second at the
  witness rotor.

## Not checked

- Modelled sensors (M5). The controller reads the true state, as a mean over
  one beat.
- Ground effect; a wake between the wings or onto the tail; flexible wings;
  a wing structure for the strike; sand that gives.
- Dynamic stall: the hover and climb are where X2 found the strip solver
  low by 1.5-3.5 times. The direction of the correction is more force, and
  no number here is corrected for it.
- V+ and P+ are post-hoc. Their scored rows are clean (hash first, tuning
  entries disjoint from the scoring ones), but the lever was chosen after
  the frozen arms' failure was seen in tuning.
- The frozen-law arms ran with lateral gains whose signs were later found
  wrong. The hover rows say this does not matter; no scored row tests it.
- The across-slope entry and the contact rows are one entry each for V+.
- The figures' trace is sampled at 0.05 s and misses the strike's peak; the
  peaks in the table are read every step.

## Reproducers

Prefix each with
`cd /home/hundo/Projects/Dytiscidae/dytiscidae && MUJOCO_GL=disable PYTHONPATH=.`

```bash
.venv/bin/python experiments/flapper/m1_takeoff.py score      # 186 rows, 4 processes, about 4 h here
.venv/bin/python experiments/flapper/m1_read.py
.venv/bin/python experiments/flapper/m1_draw.py V+ 20 0       # and V, N2, P+
```
