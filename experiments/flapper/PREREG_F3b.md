# F3b — does the two-pair machine swim when its body is free?

Frozen 2026-11-12 before any rollout, commit 1e9ee5a. No learning. Written
by the lead; the agent that runs it may not edit this file.

## 0 Where this stands

F2: both layouts walk on their wings; the frozen rule gave one pair and a
tail (fewer parts), with a margin of 0.06-0.08 on wing share and a hull that
sets down every stride. The user was shown both and chose **B, two pairs**
(2026-11-12), whose hull never touches. B is the body from here on.

F3 (held body, one pair): thrust is easy and the reaction is 2.8-19 body
weights of vertical swing, so only a free body can say the machine swims.

## 1 Question and decision

Layout B as F2 built it, free in the water, ideal servos: which stroke
swims?

- **S1, together**: all four wings up and down in phase, feathering.
- **S2, against**: fore pair and hind pair half a beat apart, so the
  vertical reactions oppose.
- **S3, scull**: stroke held, feather only.

Decision. If S2 or S3 passes the bar below, the machine swims with the
shoulders it has and no wrist is added for the water. If only S1 passes,
the same, with the body's bobbing reported. If none passes, a fore-aft
wrist on the outer wing is built and a rowing stroke is measured under a new
frozen file.

Bar: unmodified water score through `evaluate.run_segment` (antipodal pair)
0.15 or more on at least 7 of 8 seeds, `bad_qacc` 0, with both null arms
under 0.012 on 8 of 8.

## 2 Method

Body: F2's layout B, avionics 5 g inside the experiment, ideal position
servos and an unlimited pack as in F2, time step 1 ms. Stroke frequency,
amplitudes and the steering law (heading from left-right difference, depth
from fore-hind difference or feather offset) tuned by hand on seeds 100-103
only, every tuning rollout appended to `trials_F3b.jsonl`. Scored seeds 0-7.
Null arms per stroke that passes: every actuator held (still); the stroke
with its heading and depth command frozen at the first seen.

Also read, per stroke, on a straight level swim of 8 s (three seeds): speed
in m/s and hull lengths/s; the hull's heave and pitch swing over a beat;
the largest torque asked of a shoulder.

## 3 Predictions (written before running)

1. S2 passes the bar. I hold this at 0.5.
2. S1's hull heaves more than S2's by a factor of three or more at the same
   frequency and amplitude.
3. S3 does not reach 0.15 on 7 of 8.
4. The straight swim under the best stroke is 0.15 m/s or faster at 1 Hz or
   less.
5. Null arms: still and frozen under 0.012 on 8 of 8.
6. The largest shoulder torque in the passing stroke is under 0.7 N m (what
   F2's walk asked).

## 4 Limits fixed now

Quasi-steady fluid model with the package's added-mass terms; tandem wings
do not feel each other's wake; rigid wings; one buoyancy. Stopping rule:
rows filled, or a row shown blocked with its trace, or about three hours. A
clean negative is a result; no rotors, jets or extra fins, and the bar is
not loosened.
