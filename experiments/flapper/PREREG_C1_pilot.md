# Pass C1, pre-registration: height and speed held on the wing of pass W6

2026-11-17, branch `triphibian-diagnosis`, on `1645ccb` plus this commit.
Frozen before `c1_pilot.py` is run.

## 0. Question and the decision that hangs on it

Pass W6 found a cruise at 8 m/s for 7.2 W on a held body with no tail. Pass
1m found that a tail holds pitch and nothing holds height or speed. Does the
W6 wing, let go in x, z and pitch with the 1m tail, keep a commanded height
and speed, follow a change of either in both directions, and still cost what
W6 said? The decision: whether the cruise point of W6 stands as the machine's
(and the twist mechanism of the next pass is designed for it), or the free
body changes it.

## 1. What is known, and the rivals

- `freerig.run_free` (pass 1k) drives stroke and feather by torque; the
  body's reaction is the model's own. Its `pilot` (height by the pitch the
  tail is asked for, speed by the flapping frequency) was written after pass
  1m and probed only (`probe_pilot.py`), on the flat wing.
- New in this commit: `run_free(strip_twist=)` (the same strip turn as
  `rig.run`, moved to `rig.twist_setup` / `rig.twist_turn`), and the pilot's
  `z_step`, `v_step`. Check: `rig.run` reproduces W6's stored row to four
  figures (7.237 W, Fx 0.0453, Fz 5.9262); the torque-driven rig held at the
  same point reads 7.239 W, lift 1.003 weights, Fx 0.044 N.

Rivals for "it holds": (a) the loops work; (b) the body would have held
anyway (control A0, A1); (c) it is gliding on its launch energy, not flying
(control G0: the same loops with the wings not flapping); (d) the hold is
bought with power or crank torque past the gates (read both).

## 2. What was seen before freezing (a probe, not a measurement)

`scratchpad/probe_c1.py`: three of the 29 points (2.5 Hz/40/15, 3.0/40/20,
2.0/60/20), 60 beats, five arms each.

- Held with the tail on: lift 0.93-0.96 weights, pitching moment about the
  centre of mass -0.38 to -0.39 N m (nose down).
- Fixed tail: sinks 2.6-3.8 m, speed falls to 5.9-7.0 m/s, 4-5 W.
- Pitch feedback only: height drifts -0.4 to +1.4 m, speed 6.9-7.5 m/s.
- PI tail and the pilot at its defaults: height within 0.21 m, speed
  7.5-8.2 m/s, late power 6.3-6.6 W.
- PI tail and pilot `kh 6, kvz 6, pitch_lim 12`: height within 0.19 m, speed
  7.7-8.4 m/s, late power 6.5-7.2 W, tail -3 to +9 deg, pitch swing 2.5-4.8
  deg. These gains are the ones frozen below.

No step in height or speed, no nose-up release, no glider control and none of
the other 26 points has been run.

## 3. Design

Body: `BODY_S`, camber 4%, section `goe417a+lift`, tail 0.06 m2 (0.25 m each
side, chord 0.12 m, boom 0.3 m), 0.60 kg, hull inertia 0.012 kg m2. Wing: the
tip alone feathers (`strip_twist` share 1, no held twist). The twist's drive
is still uncharged (pass W7 is for that). Points: all 29 of W6's sustaining
8 m/s points for that wing. 100 beats released in x, z and pitch after 2
held; the "late" window is the last 10 beats.

| arm | tail | pilot | |
|---|---|---|---|
| A0 | fixed at 0 | none | control: nothing closes any loop |
| A1 | kp 1, kd 0.1 | none | control: pitch only (pass 1m's) |
| A2 | kp 1, kd 0.1, ki 2 | kh 6 deg/m, kvz 6 deg/(m/s), pitch +-12 deg, kf 0.08 per m/s, f +-20% | the loop |
| Z+2, Z-2 | as A2 | height asked steps by +2 / -2 m at beat 40 | |
| V+1, V-1 | as A2 | speed asked steps by +1 / -1 m/s at beat 40 | |
| D | as A2 | released 10 deg nose-up | |
| G0 | as A2 | wings held still (stroke 0, feather 0), three beat lengths | control: a glider under the same loops |

Gates, on the late window: height within 0.5 m of the height asked (beat
means and the instantaneous extremes); vertical speed within 0.1 m/s; beat-mean
speed within 0.5 m/s of the speed asked; pitch swing at most 5 deg; power at
most 41 W; crank metric at most 12 N m over the whole flight; all finite.
"Holds" = all of them. Recorded with each: whether the frequency sat on its
+-20% limit or the pitch asked on its +-12 deg limit in the late window (a
loop on its stop is not holding, it is resting).

A command answerable both ways (CLAUDE.md rule 8): each step is run up and
down, and a point counts as following height (or speed) only if both
directions hold.

## 4. Predictions

P1. A0 holds at no more than 3 of 29. A1 holds at fewer than 10 of 29 (what
    fails is speed, more than 0.5 m/s low).
P2. A2 holds at 24 or more of 29. At the points that hold, late power is
    0.80-1.10 of the held rig's W6 figure for that point, and the cheapest is
    6.0-8.0 W.
P3. G0 holds at none of its three runs: without flapping the speed leaves the
    0.5 m/s band or the height the 0.5 m band inside 100 beats.
P4. Height steps: both directions hold at 20 or more of 29.
P5. Speed steps: -1 m/s holds at 15 or more; +1 m/s at 14 or fewer (the
    frequency reaches its +20% stop, or the speed settles more than 0.5 m/s
    short: the speed loop is proportional only and the wing's thrust margin
    is thin). So speed is followed both ways at 14 or fewer.
P6. D holds at 24 or more of 29.
P7. Across the A2 points that hold, the speed settles below 8 m/s (droop of
    a proportional loop) by 0.05-0.5 m/s at more than half.

Falsified by the stated counts and ranges. If P3 fails (the glider holds),
every "holds" in this pass is void and the pass is rewritten around it.

## 5. Not in this pass

The twist's drive, rate and inertia; wind and gusts (the only disturbances
are the release attitude and the commands); roll, yaw and sideslip (the rig
releases x, z and pitch); sensors (the loops read the true state, low-passed
over one beat); the tail's flat section on the default law, which RESULT_W5
found 0.3-0.5 high in maximum lift for a flat wing (the tail's incidence is
recorded so its margin to that can be read).

## 6. Read points, stopping

One run, read once by `c1_read.py` against section 4 verbatim. Deterministic,
no seeds.

## 7. The drawing

`draw.py C1`: the machine as flown, tail on, at the two ends of the tip's
beat, from the built model; opened and checked against section 3's numbers
before the result is written.
