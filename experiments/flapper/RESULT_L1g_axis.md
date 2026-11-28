# L1 pass 1g result: the passive feather with its hinge axis moved aft (2026-11-15)

Pre-registration: `PREREG_L1g_axis.md` (commit ff8263c; the damping fix and
the probes are disclosed there). Data: `results/L1g_axis.jsonl`, 1,216 rows:
640 take-off and 576 cruise. All rows are finite and all cruise points are
trimmed. One pass. Reader: `l1g_read.py`.

**Erratum 2 (found 2026-11-15 while building `freerig.py`): the stroke's
inertial push on the free joint never acted in this pass.** The rig wrote it
to `qfrc_applied` before `env.step`; the fluid solver rewrites all of
`qfrc_applied` inside the step and wiped it. `probe_push.py`: on pass 1h's cheapest passive take-off point the push changes lift from 1.631 to 1.576 weights and the fluid stroke torque from 13.13 to 13.84 N m.
The rig is fixed (`legacy_coupling=True` reproduces this pass as run). Not re-measured: pass 1h supersedes it, and 1h is re-run as `L1h_cop_r`.

## (a) Take-off: 0 of 640 meet the gate (P1 confirmed)

| axis h | zeta | most lift | cheapest point at >= 1.6 weights |
|---|---|---|---|
| 0 | 0.3 | 2.78 weights, 29.8 N m, 219 W | 17.7 N m, 113 W |
| 0 | 1.0 | 2.53 | 16.5 N m, 113 W |
| 0.1 | 0.3 | 3.03, 32.1 N m, 258 W | 16.4 N m, 109 W |
| 0.1 | 1.0 | 2.19 | 22.2 N m, 185 W |
| 0.2 | 0.3 | 1.50 | none |
| 0.2 | 1.0 | 0.79 | none |
| 0.3 | 0.3 / 1.0 | 0.06 / 0.01 | none |

- The best passive point needs 16.4 N m. The prescribed feather (1a) needed
  10.7 N m and the gate is 12.
- With the damping as stated, the leading-edge hinge needs 16.5-17.7 N m.
  Pass 1f, run at the joint's built damping, needed 20.7 N m.
- Moving the axis to 0.1 changes little. At 0.2 and behind, lift collapses.

## (b) Cruise: 0 of 576 meet the gate (P2 confirmed)

- Axis 0 to 0.2: 0 of 432 points sustain.
  - The best net force at each axis is -0.31 to -0.49 N, always at 8 m/s,
    2.5 Hz, stroke 40 deg, 6-8 W, 4.5-5.2 N m.
  - P4 is confirmed: at the leading edge nothing sustains with the damping
    corrected either.
- Axis 0.3: 6 of 144 points sustain.
  - Net force +0.22 to +4.55 N, all at k = 0.6 and 3.5-4 Hz.
  - Each costs 77-207 W and 17.7-28.5 N m, against the gate's 41 W and
    12 N m.
  - The feather swings asymmetrically, as far as -58/+41 deg.
- **P3 is falsified**: the best force at h > 0 beats h = 0 by 4.89 N, not by
  less than 0.5 N. The whole gain comes from h = 0.3.

## Reading

- The only thrust comes from the one axis position behind the model's fixed
  load point (0.25 chord). There the strip normal force pushes the wing away
  from zero incidence, and a soft spring lets it swing wide. This is the
  region the pre-registration flagged as rival (c): it says more about a
  solver with no centre-of-pressure travel than about a wing.
- The gain is real in the model but not shown to be physics. Even if it is
  real, the cost is 2-5 times the power gate and 1.5-2.4 times the torque gate.
- Rival (a), that the axis only rescales the moment, fails at h = 0.3 and
  roughly holds from 0 to 0.2.
- Rival (d), that the damping fix moves 1f's answer, moves the take-off
  numbers by 3-4 N m but not the conclusion.
- **Decision: K1 with a passive feather on a single spring hinge does not
  meet either gate at any axis position in this model.**

## What would move this

- Centre-of-pressure travel with incidence in the fluid model would make the
  0.25-0.4 chord region physical rather than unstable. Then the h = 0.3
  result can be judged.
- Alternatively, take a driven feather: a stroke-linked cam, which still
  uses one motor, or a second actuator.
