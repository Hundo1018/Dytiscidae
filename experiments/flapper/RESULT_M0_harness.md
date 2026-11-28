# M0: one harness for the whole mission

2026-11-18. `mission.py` (class `Machine`), checks in `m0/m0_check*.py`.
Not pre-registered: it is a tool, checked against an earlier number, and
decides no gate (`TRIPHIBIAN_SPEC.md`).

What it is: the project's own `TriphibianEnv` (free six-axis body; the sea,
the beach ramp from x = 12 m at slope 0.12, the air; MuJoCo contact; the strip
solver in whichever medium each strip is) carrying the 0.6 kg reference
machine of C1 (BODY_S, the W6 wing with Goe 417a, feather hinge 0.1 chord
behind the leading edge, the 0.06 m2 tail), with each of its six hinges
(two strokes, two feathers, two tail halves) on its own witness motor: the
hinge's position actuator is an implicit PD whose force range is set every
step to the motor's band at the hinge's present speed, the rotor reflected
as armature, copper loss counted.  Motors (TYPED, stated): stroke and
feather tau_s 1.0 N m, omega_0 1000 rad/s, J_r 2e-5 kg m2, gears 24 and 16;
tail a tenth of that, gear 50. Defaults dt 1e-4, PD natural frequency 300
rad/s (wn 150 tracked the take-off stroke to 2.0 deg, 300 to 0.51, 600 to
0.16).

Checks:

| | read | against |
|---|---|---|
| on the beach (x 17 and 32 m), wings level, 2 s | contact load 0.986 of weight, slides 6 mm, rests 10.7 deg nose-up on the 6.8 deg ramp, no QACC reset | at rest |
| take-off stroke, body held at 90 deg in air (4.5 Hz, 85 deg, feather 50 a quarter beat ahead), stroke gear 24 | lift 1.840 weights | `freerig`'s prescribed baseline at dt 1e-4: 1.856 (-0.9%) |
| the same, power | shaft: stroke 122 W, feather 46 W; electrical: stroke 405 W, feather 255 W | D1/D2: one stroke motor for both wings 149 W (four-bar) to 295 W (direct) |

The electrical bill is the finding of D2c again, now with a motor per
hinge: each witness rotor reflected through gear 24 is 0.0115 kg m2, 2.5
times the wing's own stroke inertia (0.0046), so most of the copper loss
carries the rotor through each reversal, and four rotors pay it four times.
The rotor, the gear and whether one motor drives both wings are machine
parameters for M1-M3, not fixed here.

Not in it yet: modelled sensors (M5), the D-series mechanisms (after M5),
any controller beyond following joint targets.

## Addendum, same day: two harness faults and the sensor model

- `mj_setConst(m, d)` resets the data it is handed to `qpos0`, so every
  `Machine(domain)` built without `pose=` started at (-6, 0, 1.5) and fell
  into the sea (found by M2's design). Now called on a scratch `MjData`:
  land starts at (15, 0, 1.0), water at (-8, 0, -4), air at (-40, 0, 30).
  The checks above all passed `pose=` or `drop_to_ground` and are unaffected.
- `data.subtree_com` is not the centre of mass here: the strip solver writes
  added mass into `body_mass` without `mj_setConst` (found by M3's design).
  `Machine.com()` gives it from the dry masses. Dynamics are unaffected.
- `drop_to_ground` leaves the machine about 0.1 m up (it does not subtract a
  box's half-height); passes let it settle.

`sensors.py` (class `Sensors`): an IMU at a hull mount (gyro and
accelerometer at 1 kHz with white noise, wandering gyro bias, range
saturation and a 2 ms delay; the accelerometer reads the specific force at
its mount, so it feels the hull's motion at the flapping frequency), a
complementary attitude filter, a pressure sensor (depth in water, height in
air), a contact switch, joint encoders, and wetness. Noise constants are
TYPED from memory of a small MEMS IMU and a hobby pressure sensor, not
checked against a datasheet. Checks (`m0/sens_check.py`, `m0/sens_tau.py`),
tilt error between the estimate and the truth:

| case | filter time constant | tilt error mean / max |
|---|---|---|
| at rest on the beach | 1 s | 0.33 / 0.37 deg |
| flapping 20 deg on the beach | 1 / 5 / 30 s | 1.40 / 2.88; 0.65 / 1.46; 0.74 / 1.35 deg |
| free, flapping 40 deg from an 8 m/s launch, tumbling (true tilt to 84 deg), estimate armed before the launch | 1 / 5 / 30 s | 6.52 / 13.8; 1.28 / 3.00; 0.33 / 1.23 deg |

In flight the accelerometer reads the wings' force, not gravity, and the
filter's pull toward it is wrong: the 1 s filter is off by up to 14 deg.
The default is 30 s (gyro-led); whether a 30 s filter holds over 15-minute
flights with the gyro bias walking is for M5 to measure. Two faults fixed on
the way: the tilt correction's sign (it diverged), and the body acceleration
read in the inertial frame instead of the hull's; and an estimate initialised
from a reading that was not 1 g came out upside down, so it now waits for 1 g
or starts at the true attitude when armed before a launch.
