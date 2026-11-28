# Re-runs of passes 1h and 1j with the stroke's inertial push acting. Frozen 2026-11-15

Passes 1f-1h (passive feather hinge) and 1j (one-way flap) ran with no
inertial push from the prescribed stroke on the free joint: the rig wrote it
before `env.step` and the fluid solver wiped it (errata in each result). The
rig now adds it after the solver has written. Both passes are re-run on
their frozen grids, gates and scripts, unchanged: `l1h_cop.py` to
`results/L1h_cop_r.jsonl`, `l1j_flap.py` to `results/L1j_flap_r.jsonl`.
Written before either re-run.

## Disclosed (`probe_push.py`, 7 points)

| point | as run | with the push |
|---|---|---|
| 1h passive take-off (plate, h 0.1, 3.5 Hz, k 0.3) | 1.631 weights, 13.13 N m | 1.576 weights, 13.84 N m |
| 1j rowing, still water, flap up, 0.5 Hz | +1.37 N, 17.9 N m | +2.33 N, 6.1 N m |
| 1j up-down, 0.3 m/s, flap up, feather 30 | -1.79 N | +0.24 N |

## Predictions

Pass 1h re-run:
- R1: take-off gate met at 0 of 800 points; the cheapest point at 1.6 weights
  needs 13-16 N m on the shoulder metric.
- R2: cruise gate met at 0 of 720 points.

Pass 1j re-run:
- R3: P1 of pass 1j still holds: for each wing the better flap arm's best
  force under the caps is less than the welded arm's (ratio under 1.0).
- R4: still-water rowing, wing 0.4, stroke 85 deg, 0.5 Hz: flap up +1.5 to
  +3.5 N, flap down the same size within 20% and opposite, welded under
  0.5 N.
- R5: negative mean power at more than 2% of flap points with a feather: the
  suspected power-accounting fault is not the push, so it remains.

One pass each, no tuning after reading.
