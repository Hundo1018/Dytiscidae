# L1 pass 1k: the cam drive with the body let go. Frozen 2026-11-15

Pass 1i found that one motor with a cam-driven feather meets the take-off
gate at 16 points and the cruise gate at 26, on a held body, with a thin
torque margin. Every such result so far holds the body still. Here the same
42 points are run on a new rig, with the body held and then released.
Written before the grid below was run.

## The new rig (`freerig.py`)

- `rig.run` moves the wing joints by overwriting their state. The body never
  feels the wings' reaction, and the drive torque is reconstructed from fluid
  loads and an analytic inertia.
- `freerig.run_free` tracks the same stroke and feather with a stiff servo
  (feed-forward plus PD at 600 rad/s, step 0.25 ms), applied as joint
  torque. The torque and power are those applied; the reaction on the body
  is the model's own.
- Held root axes carry a large armature and are reset each step. The
  velocity they gain in a step gives the holder's wrench, so the net force
  and the pitching moment are read directly.
- After 2 beats held, chosen axes are released: x and z, or x, z and pitch.
  The measurement is the next 4 beats.
- The model weighs 0.506 kg; ballast at the hull's centre of mass brings it
  to the fixture's 0.60 kg.
- The built model has 0.0365 N m s/rad of damping on every wing joint (a
  generic figure, TYPED). The main arms set it to zero, so the drive is
  charged what `rig.run`'s aerodynamic power was. One held arm keeps it, as a
  sensitivity.
- There is no tail and no controller. The pitch-free arm is open loop.

## Disclosed before freezing (`probe_free.py`, one take-off and one cruise point)

| arm | take-off (plate, h 0.1, 4.5 Hz, Psi 60, body 90) | cruise (plate, h 0.2, 8 m/s, 3.5 Hz, stroke 60, Psi 20) |
|---|---|---|
| pass 1i, old rig | 1.844 weights, 117.7 W, crank 11.7 N m | 1.008, +0.02 N, 27.1 W, crank 5.5 |
| new rig, held | 1.798, 115.7 W, crank 10.1; pitch moment peak 7.1 N m | 1.008, +0.02 N, 27.2 W, crank 5.3; pitch moment peak 2.4 N m |
| x and z free | climbs 1.78 m in 4 beats, 91 W, crank 7.0, bob 34 mm | 1.003, **-0.15 N**, 26.5 W, bob 52 mm |
| pitch free too | climbs 2.16 m, pitch -23 to +31 deg, crank 13.0 | tumbles: -216 deg in 4 beats |
| held, built joint damping | 163.8 W, crank 12.6 | 34.1 W, crank 5.9 |

Two faults were found and fixed while building it:
- zeroing `qfrc_applied` after the fluid solver removed the solver's own
  damping compensation (745 W read where 116 W was right);
- the wiped inertial push of passes 1f-1h and 1j (their errata;
  `PREREG_L1r_reruns.md`).

## 0 Question and decision

Does the cam drive still take off and cruise when the body is free to move?
Decision: whether K1 with a cam stays the reference, and what the tail and
the stabiliser must supply before a fully free flight can be run.

## 1 Rivals

(a) Releasing x and z changes little: the body is seven times the wings'
mass and bobs by millimetres to centimetres.
(b) The bob eats thrust: in cruise the body heaves against the wings, the
stroke's travel through the air shrinks, and thrust falls below drag at
points with no margin.
(c) Pitch decides it: the feather's reaction is a pitching moment of several
N m on a hull with about 0.005 kg m2 of pitch inertia, so without a tail the
body pitches by tens of degrees a beat and the feather the air sees is not
the cam's.
(d) The new rig differs from the old one for reasons of its own (tracking
error, the model's wing inertia against the analytic one), so held-new must
be read against 1i first.

## 2-3 Arms, gates and predictions

Points: pass 1i's 16 take-off and 26 cruise gate points. Arms: held; x and z
free; x, z and pitch free; held with the built joint damping.

- P1 (rival d, control): held, the new rig reproduces pass 1i's lift within
  5% and power within 5% at every point. The crank metric is lower by up to
  25%, because the model's wing inertia replaces the analytic one.
- P2 (take-off, held): 10 to 14 of 16 points still meet the gate; those that
  drop are at 1.60-1.65 weights in pass 1i.
- P3 (take-off, x and z free): every point climbs at least 1 m in 4 beats;
  lift over the first beat after release is 0.85-1.0 of the held lift; the
  crank metric is no higher than held at any point.
- P4 (cruise, x and z free, offset re-trimmed): Fx falls by 0.05-0.4 N at
  every point; 8 to 18 of 26 still meet the gate; the cheapest gate point
  costs 27-33 W.
- P5 (pitch, held): the holder's pitching moment peaks at 2-8 N m at every
  point, with a mean under 0.5 N m in size.
- P6 (pitch free, no tail): every cruise point pitches beyond 45 deg within
  4 beats; at least 12 of 16 take-off points stay within 45 deg and climb at
  least 1 m.
- P7 (sensitivity): the built joint damping raises take-off power by 30-50%
  and cruise power by 20-35%.

## 4-5 Measurement and design

`l1k_free.py`, read by `l1k_read.py`. 42 points x 4 arms = 168 jobs; a free
cruise job uses 2 to 6 runs to re-trim. One pass, no tuning after reading.
Predicted 20 minutes on 6 processes. A point that is not finite is excluded.

Take-off gate: at least 1.6 weights (held: mean; free: over the first beat
after release), crank metric <= 12 N m, power <= 150 W. Cruise gate: lift
within 3% of the weight, Fx >= 0, <= 41 W, crank metric <= 12 N m.

Not covered: a tail, a stabiliser, roll and yaw; a real cam profile and
crank-rocker; the motor's torque-speed line; ground contact at take-off; the
passive hinge and the flap on this rig (next); water.
