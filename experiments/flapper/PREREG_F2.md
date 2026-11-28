# F2 — can the wings carry the body on land, and does that take one pair or two?

Frozen 2026-11-12 before any rollout, commit dd1ef0d, MuJoCo 3.15. No
learning, no optimiser. Written by the lead; the agent that runs it may not
edit this file. Stage 1 of the user's two stages: every actuator is an ideal
position servo (no torque limit), and the torque each joint asked for is
recorded for stage 2.

## 0 Question and decision

The user's requirement for land (2026-11-12): the machine moves with its
wings supporting the body. The wing count is open and is to be decided by
what works. Two layouts, each with the same stroke-and-feather shoulders the
air measurements used and a short fixed "thumb" under each wing (a strut
that points down from the wing, as a bat's thumb or a pterosaur's hand; in
flight it is a strut in the stream):

- **A, one pair and a tail.** Two wings of 0.25 m, an actuated tail at the
  rear of the hull. Three points on the ground: two thumbs and the tail.
- **B, two pairs.** Four wings of 0.21 m (by F0's scaling, the same hover
  lift at the same frequency as A's pair), fore and aft on the hull, a thumb
  under each, no tail.

Decision, fixed now, on the walk window of the land segment. **Wing share**
is the share of the ground's vertical force that enters through the wings
and their thumbs; **hull contact** is the share of time any hull geometry
touches the ground. A layout **walks on its wings** if on at least 7 of 8
seeds the unmodified land score is 0.15 or more, `ground_support` is 0.9 or
more, wing share is 0.7 or more and hull contact is 0.2 or less. If A walks
on its wings, the machine has one pair (fewer parts) whatever B does. If A
does not and B does, it has two pairs. If neither does, both sets of numbers
go to the user with what each lacks. Wing material and a wrist are not
decided here.

## 1 Rivals

- **A crutches but rests on its belly or tail in swing** (the turtle
  hatchling lifts its body 2 mm; E0b's paddle limbs rested on the belly).
- **B walks clear**, on diagonal pairs or three at a time.
- **Neither**: a wing 0.2 m long is a poor lever for a thumb, the feather
  axis cannot sweep the thumb far enough, or the wing surface strikes the
  ground before the thumb does.
- **The score is fooled**: a gait that ignores the heading, or a still body
  propped on its wings, is paid. Checked by the null arms.

## 2 Method

- Bodies from `experiments/flapper/body.py::flapper` (extended by the agent
  for the tail, the thumbs and the second pair; every added part listed),
  avionics 5 g inside the experiment as in F0. The body is free on the
  beach; nothing is held.
- Actuators: ideal position servos set on the experiment's own model (gain
  high enough to follow the gait, force limit off); the largest torque each
  joint asks for is recorded.
- Scores unmodified through `evaluate.run_segment` (antipodal pair, the
  repo's 0.04 m/s commanded speed), seeds 0-7; gait and gains tuned by hand
  on seeds 100-103 only, every tuning rollout appended to
  `trials_F2.jsonl`. `bad_qacc` read on every rollout; a failed rollout is
  failed, never 0.
- Null arms on each layout, 8 seeds: every actuator held at the stance pose
  (a body propped on its wings); the gait with its heading frozen at the
  first one seen.
- Top speed, apart from the score: the gait at its fastest hand-tuned
  setting on a straight commanded heading for 8 s on level beach, three
  seeds; metres per second and hull lengths per second.
- What the walking hardware costs in the air: `rig.run` at F0's hover point
  (12 Hz, stroke 130, feather 45, body upright) and at one cruise point
  (8 m/s, 8 Hz, stroke 80, feather 20, trimmed), for each layout against the
  bare wing: lift, power, all-up mass.

## 3 Predictions (written before running)

1. B walks on its wings by the rule above. I hold this at 0.5.
2. A reaches the score and the support (>= 0.15 and >= 0.9 on >= 7 of 8) but
   not the wing share or the hull contact: its hull and tail carry 30% or
   more, or its hull touches more than 20% of the time. I hold this at 0.6.
3. Null arms: propped and still under 0.012 on 8 of 8 for both layouts;
   heading frozen under 0.012 on 8 of 8 for both.
4. Top speed: B 0.10 m/s or more; A 0.05 m/s or more.
5. Air cost: thumbs and tail change A's hover lift by under 5% and its mass
   by under 10%; B's four wings give hover lift within 15% of A's pair at
   the same frequency and weigh 15-40% more all-up.
6. The largest torque asked of a shoulder on land is under the 0.42 N m the
   hover stroke asks for by its inertia alone.

## 4 Limits fixed now

Rigid wings and thumbs; contact as MuJoCo models it on the repo's beach.
Tandem wings in the air are read on the held rig, where the hind pair does
not feel the fore pair's wake. Stopping rule: stop when both layouts' tables
are filled, or when a row is shown blocked with its trace, or after about
three hours of work. A clean negative is a result; do not rescue a row by
adding legs, wheels, rotors or a third wing pair, or by loosening the rule.
