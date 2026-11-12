# F4 — does the two-pair machine fly when its body is free?

Frozen 2026-11-12 before any rollout, commit 6d0c43f. No learning, no
optimiser. Written by the lead; the agent that runs it may not edit this
file.

## 0 Where this stands

Land (F2) and water (F3d, F3e) are established for layout B with its body
free. Air is not: every air number so far is a held body on a rig, most of
them for one pair of 0.25 m wings on a 77 g machine (F0-F1a). The machine is
now two pairs of 0.21 m wings, a thumb under each, and the neutral hull
(`FLAPPER_BODY_B='{"hull_radius": 0.0209}'`, 105 g). F0b/F0c's rule made it
a bird-like flyer: forward flight and glide. Hover and take-off belong to
the transitions, which come after.

## 1 Question and decision

Launched as the repo's air segment launches every machine (at its own trim
speed and attitude, 30 m up), with ideal servos and a hand law on the
shoulders it has: does it hold height and speed by flapping, and does it
turn to the side it is told?

Decision. Both bars below met: the three ways of moving are established for
stage 1 on one body, and the work moves to the transitions. Only the level
bar: it flies and does not steer; the report says which axis lacks
authority, and a body change (an active tail, the wings' placement) is
measured under a new frozen file. Neither: the same, starting from why it
does not stay up.

## 2 Method

Body B with the neutral hull, avionics 5 g inside the experiment, ideal
position servos and an unlimited pack as in F2, a time step of 1 ms or less.
First, on the held rig (`rig.run`, `rig.trim`): this body's hover lift at
12 Hz and its trimmed cruise at 6 and 8 m/s, to choose a stroke. Then free.
Law and stroke tuned by hand on seeds 100-103 only, every tuning rollout
appended to `trials_F4.jsonl`. `bad_qacc` on every rollout; a failed rollout
is failed, never 0.

- **Level bar**, outside the score, seeds 0-7: launched as the air segment
  launches it and told to fly straight for 8 s. Flapping: net height change
  within 1.5 m either way, speed at the end within 30% of the launch speed,
  upright throughout (the root's up axis above 0.5), on at least 7 of 8.
  Beside it, the same launches with every actuator held (the glide): its
  height loss is the reference.
- **Turn bar**, seeds 0-15, unmodified air score through
  `evaluate.run_segment` (the mirrored pair: straight, then a turn of 45-90
  degrees to the drawn side, and to the other side in the second half):
  0.15 or more on at least 14; `bad_qacc` 0; every actuator held, and the
  command frozen at the first seen, under 0.012 on 16 of 16.
- Read beside both: the stroke used (frequency, amplitudes), the largest
  shoulder torque, the turn made against the turn asked in each half, and
  how each axis is controlled (which difference between wings makes roll,
  pitch and yaw, each sign found by measurement).

## 3 Predictions (written before running)

1. The level bar is met. I hold this at 0.7.
2. The held glide loses 2.5 m or more over the same 8 s on at least 6 of 8.
3. The turn bar is met. I hold this at 0.45: steering a tandem flapper by a
   hand law is what I trust least.
4. Null arms under 0.012 on 16 of 16 each (a glider that does not turn
   scores 0 under the pair).
5. No part is added: the law flies the body as F2 and F3 left it.
6. The flight stroke is between 6 and 12 Hz and asks a shoulder for under
   1.5 N m.

## 4 Limits fixed now

Quasi-steady model; the hind pair does not feel the fore pair's wake, which
in the air is the largest thing the model leaves out for a tandem wing;
rigid wings; still air. Stopping rule: rows filled, or a row shown blocked
with its trace, or about four hours. A clean negative is a result; no
rotors, no part added without a new frozen file, no bar loosened.
