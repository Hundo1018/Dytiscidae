# L1 pass 1d result: a fold that narrows the chord with the span (2026-11-15)

**Erratum (2026-11-15, pass S1): the stroke torque, crank metric and power
series in this pass carried the entrained weight's cancellation**, a force
the solver adds to cancel the weight MuJoCo gives the added mass; only the
net force had it removed. On pass 1l's grid the corrected crank metric is
0.36-0.96 of the one read (median 0.66). The pass also ran with the solver's
default entrainment reaction, a one-sided drag on a flapping wing (median 23%
of thrust on that grid). Torque caps here were applied to an overstated
torque and forces are understated. Mean force and mean power are otherwise
unaffected. See `RESULT_S1_entrainment.md`; not re-run unless said there.

Pre-registration: `PREREG_L1d_fold2.md` (commit 2101656, before pass 1c was
read and before any rollout). Data: `results/L1d_fold2.jsonl`, 582 points,
all finite, one pass; read by `fold_read.py`.

| wing (span, chord) | speed | fixed crank | variable throw | best net force under 15.8 N m |
|---|---|---|---|---|
| 0.4, 1.0 (pass 1c) | 0.3 m/s | met (2) | met (9) | 5.92 N at 0.5 Hz |
| 0.4, 0.7 | 0.3 | met (2) | met (15) | 13.61 N at 1.5 Hz |
| **0.4, 0.4** | 0.3 | met (4) | met (23) | **15.52 N at 2.5 Hz** |
| 0.4, 1.0 (1c) | 0.6 | met (2) | met (4) | 5.19 N |
| 0.4, 0.7 | 0.6 | no | met (9) | 10.25 N |
| 0.4, 0.4 | 0.6 | met (2) | met (12) | 15.77 N |
| 0.6, 1.0 (1c) | 0.3 | no | met (4) | 3.28 N |
| 0.6, 0.6 | 0.3 | met (3) | met (12) | 7.23 N |
| 0.6, 1.0 (1c) | 0.6 | no | no | 0.30 N |
| 0.6, 0.6 | 0.6 | no | met (2) | 5.53 N |

## Against the frozen text

- Predicted (rival a, the chord does not matter): the (0.4, 0.4) force under
  the cap within 30% of 1c's span-0.4 figure, at 1.3-2 times the frequency.
  Measured: 15.52 against 5.92 N, **2.6 times**, at 5 times the frequency.
  **Falsified; rival (b) stands: the chord matters.**
- Predicted: each wing meets the same gates as the span-only wing of its span
  fraction. Measured: (0.6, 0.6) meets two gates that (0.6, 1.0) fails.
  **Falsified.**

## Reading

- Narrowing the chord frees torque for thrust: at every span fraction the
  force under the cap is 2.2-2.6 times the full-chord figure at 0.3 m/s, and
  (0.6, 0.6) reaches 0.6 m/s where (0.6, 1.0) could not.
- The best (0.4, 0.4) points sit at the highest frequency tried (2.5 Hz), so
  the map is cut by its own edge there; the feather is at its edge (30 deg)
  throughout.
- With the chord narrowed the stub is no longer wider than long (0.26 m by
  0.10 m at 0.4, 0.4), so the tip-loss objection of 1c bears less on these
  rows; it is still unanswered.
- With the variable throw the motor sees 4.6-7.3 N m at 1-2.5 Hz; with a
  fixed crank the best points are 7-8 N m at 1 Hz. Both are inside the air
  drive.

## For the design

The fold should contract the chord as well as the span. With A15 (a
many-facet creased sheet is unprecedented as a wing, soft in bending and
twist, and limited in how far both directions contract together), the first
fold to design is a few rigid hinged panels that fold in both directions: a
spanwise fold or sweep, and a chordwise fold of the rear panel under the
front one. Each hinge line can carry a lock; lines in opposed pairs drive
them. Area in water about 0.16-0.36 of the flight area.

Not covered: the folded panels' drag and mass, a free body, tip loss.
