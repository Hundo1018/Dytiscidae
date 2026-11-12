# W1 — do the water verdicts survive a fore-to-hind wake?

Frozen 2026-11-12 before the grid. No learning. Raised by the user the same
day: a model in which the hind wings do not feel the fore wings' wake may be
missing a good part of the drag, the thrust and the disturbance, and in a
fluid everything is coupled.

## What the model has and has not

Has: every wing strip sees the ambient flow plus one uniform, lagged
momentum inflow computed from the machine's total force
(`physics.fluid.InducedFlow`). So the hind pair is not in clean air: it
shares a mean downwash with the fore pair. Has not: the fore pair's shed
wake arriving at the hind pair, local, delayed by the travel between them
and changing through the beat. The package's own note on feeding a wake back
puts it at 10-30% of the forces for a flapping wing in wake interference
(`physics/wake.py`).

## The instrument

`experiments/flapper/wakefb.py`: by Kelvin's theorem each fore-wing strip
sheds the opposite of its change of bound circulation from its trailing
edge, as a spanwise vortex segment carried by the fluid; the segments induce
a velocity (regularised Biot-Savart) on the hind wings' strips only, times a
gain. Checked against Wagner's impulsive start (`wake_check.py`): the sign
is the one that holds a wing's lift down after it starts, and at gain 1 the
lift deficit is 0.12 at half a chord of travel, 0.08 at one, 0.05 at two,
against Wagner's two-dimensional 0.38, 0.31, 0.22. So **gain 1 is about a
third to a quarter of Wagner's effect, and gain 3 is about its size.** No
trailing (tip) vortices: the mean downwash stays the momentum inflow's.

It is a sensitivity instrument, not a wake to believe at any one gain.

## Method

Everything as F3d and F3e ran it (neutral hull, paired depth command,
`swim.py` unchanged), with `FLAPPER_WAKE_GAIN` 1 and 3. Gain 0 does not
attach the feedback; one seed was run at 0, 1 and 3 to check the plumbing
before this file (S2x seed 24: 0.4544, the stored value; 0.4435; 0.3423),
and is not counted as a prediction.

- S2x, seeds 24-39 (F3d's): score, against F3d's rows.
- S2d, seeds 40-55 (F3e's): score and depth response, against F3e's rows.
- S2x with the command frozen, gain 3, seeds 24-27.

## Verdict rule

A water verdict is **robust to the wake** if its bar is still met at gain 1
and at gain 3: swims (0.15 or more on at least 14 of 16), answers depth
(response 0.5 or more on at least 12 of 16), frozen arm under 0.012. A bar
met at 1 and lost at 3 is reported as holding up to a third of a
Wagner-sized interaction and not to a full one, and the stroke is re-tuned
under the wake before the water is called done again.

## Predictions (written before the grid)

1. Gain 1: S2x swims on at least 14 of 16 and its mean score is within 10%
   of gain 0's.
2. Gain 3: S2x swims on at least 14 of 16; its mean score is 15-40% lower.
3. S2d's depth response bar is met at gain 1. I hold this at 0.7.
4. S2d's depth response bar is met at gain 3. I hold this at 0.5.
5. The frozen arm at gain 3 is under 0.012 on 4 of 4 (both halves see the
   same wake, so it still cancels).
