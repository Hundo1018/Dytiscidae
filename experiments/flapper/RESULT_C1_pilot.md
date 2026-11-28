# Pass C1, result: height and speed on the wing of pass W6

2026-11-17. Frozen text: `PREREG_C1_pilot.md`. Run: `c1_pilot.py` ->
`results/C1_pilot.jsonl` (235 flights of 100 beats, all finite), read by
`c1_read.py`.

## Outcome in one paragraph

Let go with the 1m tail, the W6 wing flies level and its pitch is held at
every one of the 29 points, and where the loop holds, the power is 6.5-7.4 W,
0.86-0.91 of the held rig's. But the loop as written meets the gates at only
6 of 29 points. The height loop does what it is asked in change (a 2 m step
is followed by 1.86-2.03 m up at 25 points and 1.97-2.01 m down at all 29)
with a standing offset of up to 1.3 m; the speed loop does not: a 1 m/s
command moves the speed by about 0.47 m/s, and every point with thrust in
hand flies fast, up to 12.7 m/s. And the four cheapest points, the ones the
7 W figure comes from, **cannot climb**: asked for 2 m more they pitch up,
slow, and sink into the sea. The W6 cruise point stands as a level-flight
point; it is not yet a point the machine can be flown at.

## Against the frozen predictions

| | frozen | read | |
|---|---|---|---|
| P1 | A0 holds at <= 3; A1 at < 10, failing on speed low | A0: 0 (some sink 30 m); A1: 0, failing on height at 27 (-0.9 to +41 m) and on speed at 16, on both sides (7.0-9.8 m/s) | counts hold; my reason for A1 was wrong |
| P2 | A2 holds at >= 24; there, power 0.80-1.10 of held; cheapest 6.0-8.0 W | **6 of 29**; 0.86-0.91; 6.46 W | **count falsified**; the power part holds |
| P3 | G0 (no flapping) holds at none | none: all three are in the sea by the late window | holds: the hold is flight, not a glide |
| P4 | height followed both ways at >= 20 | 2 by the gates | **falsified** |
| P5 | -1 m/s holds at >= 15; +1 m/s at <= 14 | 4; 5 | first **falsified**, second holds for the wrong reason |
| P6 | nose-up release holds at >= 24 | 6: the same six as A2, so the recovery itself costs nothing | **falsified** with P2 |
| P7 | at the points that hold, 0.05-0.5 m/s slow at more than half | 4 of 6 | holds |

What I had wrong: I took the three probed points for typical. They were
three of the low-thrust ones. Of the 29 points 23 make more thrust at 8 m/s
than the hull and tail take, and the speed loop cannot shed it.

## What happens at each kind of point

Ordered by the held rig's net thrust at 8 m/s (W6):

- Net thrust under 0.1 N (6 points; the cheapest, 6.5-7.8 W held): level
  flight holds, speed 7.76-7.98 m/s, height within 0.05 m, frequency within
  2% of nominal, tail -3 to +10 deg, pitch swing 2.1-5.5 deg (one, 2 Hz /
  50 deg, swings 5.5 and so fails the 5 deg gate). **Four of these crash on
  the +2 m step** (2.5/40/15, 3.0/30/10, 3.0/40/20, 3.5/30/15).
- Net thrust 0.1-0.7 N (23 points): stable, but 0.2-4.7 m/s fast and 0.1-1.3
  m high. The frequency comes down 2-20% (four sit on the -20% stop) and the
  pitch asked 1-7 deg nose down; power 7-26 W.

Steps, as changes from the height or speed held before the step:

| step | followed | settling |
|---|---|---|
| +2 m | 1.86-2.03 m at 25 of 29; 4 sink more than 10 m | 4-12 beats (median 7) where it settles; peak beat power 8-31 W where it does not crash |
| -2 m | 1.97-2.01 m at 29 of 29 | 7-58 beats (median 12) |
| +1 m/s | 0.02-0.51 m/s, median 0.47 | |
| -1 m/s | +0.05 to -0.66 m/s, median 0.48; 4 points on the frequency stop do not move | |

## What this means for the design

- The speed loop needs an integral term and more authority than +-20% of
  frequency: the stroke or the tip's feather as a throttle. Proportional
  frequency alone leaves half of every speed error standing.
- The height loop needs the thrust loop under it: pitch alone trades speed
  for height, and at the points with no thrust in hand that trade ends in a
  stall. A climb needs power first, then pitch; and a floor on speed.
- The 7 W cruise has no climb margin at all. A usable cruise is the cheap
  point plus a throttle that reaches a point with 0.3-0.5 N in hand (10-14 W
  held) on demand. That pair has not been flown.

## What these numbers rest on

- The tip's feather is still the hand-turned strips of W6 here (pass W7 made
  it by joints on a held body; the two have not been combined).
- The loops read the true state, low-passed over one beat: no sensor, no
  noise, no delay beyond the tail servo's rate limit.
- x, z and pitch only; no wind; the tail sees no wake of the wings.
- The tail is a flat plate on the default law, which is 0.3-0.5 high in
  maximum lift for a flat wing (RESULT_W5); its incidence stayed within -8
  to +16 deg, and above about 8 deg that matters. The Z+2 crashes run the
  tail to its 25 deg stop.
- `experiments/physics_extremes` (same day) checks the solver's laws the
  loads here rest on; its findings on low aspect ratio (the tail is 4.2)
  and on the stall law apply.

## The drawing against the design

`figures/C1_beat_down.png`, `C1_beat_up.png`, from the built model, opened
and checked: span 1.400 m, area 0.3400 m2, chord 0.258 m, mass 0.600 kg,
tail span 0.500 m, chord 0.120 m, area 0.0600 m2, all OK; the tail on its
boom, centred 0.586 m behind the centre of mass, its hinge along its span;
the tip nose-down in one and nose-up in the other, the root unmoved; the
centre of mass 0.04 m (0.15 chord) behind the wing's quarter chord, which
with this tail is a stable arrangement (pass 1m measured it). Not on the
drawing because not in the model: anything that turns the tip.
