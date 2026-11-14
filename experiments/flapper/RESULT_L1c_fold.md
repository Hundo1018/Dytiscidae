# L1 pass 1c result: a wing that sheds length in water (2026-11-15)

Pre-registration: `PREREG_L1c_fold.md` (commit e169f1b, before any rollout).
Data: `results/L1c_fold.jsonl`, 486 points, all finite, one pass; read by
`fold_read.py`. Body held level under water; only the inner k of each wing
is present (full chord, 0.257 m); the folded part is not modelled.

## Against the frozen gates (thrust at least twice the drag, within the torque caps)

| working length | speed | drag | fixed crank (85 deg, <= 12 N m) | variable throw | best net force under 15.8 N m |
|---|---|---|---|---|---|
| 1.0 (pass 1b) | 0.3 m/s | 0.47 N | no | no | 0.47 N |
| 0.6 | 0.3 | 0.40 | no | **met** (4 points) | 3.28 N |
| 0.6 | 0.6 | 1.38 | no | no | 0.30 N |
| 0.4 | 0.3 | 0.37 | **met** (2) | **met** (9) | 5.92 N |
| 0.4 | 0.6 | 1.28 | **met** (2) | **met** (4) | 5.19 N |
| 0.25 | 0.3 | 0.35 | **met** (3) | **met** (13) | 13.85 N |
| 0.25 | 0.6 | 1.21 | **met** (1) | **met** (4) | 12.24 N |

| frozen prediction | measured | |
|---|---|---|
| 0.3 m/s: met at 0.4 and 0.25 by both drives, at 0.6 by the variable throw only | exactly that | held |
| 0.6 m/s: met at 0.25 by the variable throw only | met at 0.4 and at 0.25, by both drives | **falsified, on the favourable side** |
| force under the cap rises as the wing shortens, to 1.0-1.5 N at 0.4 | rises: 0.47, 3.28, 5.92, 13.85 N; **5.92 N at 0.4, four times the prediction** | direction held, size missed |

## Reading

- Shedding length opens the water row. With 0.4 of the length working the
  machine has 16 times its drag at 0.3 m/s and 4 times at 0.6 m/s under the
  cap, at 0.5-1 Hz, where a motor can turn, for 7-13 W. Rival (b) of the
  pre-registration (a shorter wing gains nothing) is ruled out in the model.
- The gain is larger than the 1/k the lead reasoned from: force per unit
  shoulder torque is 0.03-0.06 N per N m for the full wing under the cap,
  0.25 at 0.6, 0.52 at 0.4, 0.98 at 0.25. The full wing under the cap was
  not only long, it was forced to a frequency too low for its speed.
- The fixed crank passes from 0.4 down; the variable throw also passes at
  0.6 and gives about twice the force at 0.6 m/s (5.19 against 2.32 N at
  0.4). So with a fold, the variable throw is an improvement and not a
  necessity in water.
- **Rival (c) is not answered.** At 0.4 the working wing is 0.26 m long and
  0.257 m wide, at 0.25 it is wider than long. The strip model has no tip
  loss, so these forces are upper bounds, and the 0.25 row most of all. Only
  a wing in a tub, or a model with tip loss, settles how much is real.

## Belief before and after

Before: a fold helps by about 1/k and 0.6 m/s is out of reach except at
0.25. After: in the model a fold to 0.4-0.6 of the length is enough for
0.3-0.6 m/s with a fixed crank's torque. It moves back if tip loss on a
square stub takes most of the force.

## For the fixture

A fold to about half length in water becomes part of the reference machine.
Pass 1d (span and chord together) says whether the chord must contract too.
