# Re-runs of passes 1h and 1j with the inertial push acting: result (2026-11-15)

Pre-registration: `PREREG_L1r_reruns.md` (commit 7f40010). Data:
`results/L1h_cop_r.jsonl` (1,520 rows) and `results/L1j_flap_r.jsonl` (306
rows), all finite. Same scripts, grids, gates and readers as the original
passes. These replace the numbers in `RESULT_L1h_cop.md` and
`RESULT_L1j_flap.md`.

## Pass 1h: the passive feather hinge

- **R1 confirmed.** Take-off gate met at 0 of 800 points. The cheapest point
  at 1.6 weights needs 15.0 N m and 81 W (plate) or 15.3 N m and 90 W
  (insect), on the shoulder metric. As first run: 13.5 and 14.0 N m.
- **R2 confirmed.** Cruise gate met at 0 of 720 points.
  - 22 points now sustain level flight (6 as first run), with the hinge at
    0 to 0.2 chord.
  - The cheapest costs 52 W at 10.3 N m (plate, leading-edge hinge); 4
    sustaining points are within 12 N m. The gate is 41 W.
- The conclusion stands: a single spring hinge meets neither gate. The push
  makes take-off slightly worse and cruise better: 52 W against 68 W.

## Pass 1j: the one-way flap

Up-down stroke at 0.3 m/s, best force under both caps:

| wing | welded | flap up | flap down |
|---|---|---|---|
| 0.4 | +6.71 N | +0.24 N | -0.18 N |
| 0.6 | +3.30 N | +0.55 N | +0.20 N |

- **R3 confirmed.** The better flap arm gives 0.04 and 0.17 of the welded
  wing's best. Pass 1e's gain is still not reproduced by a passive flap.

Rowing stroke in still water, stroke 85 deg:

| wing | frequency | welded | flap up | flap down |
|---|---|---|---|---|
| 0.4 | 0.5 Hz | +0.03 N | **+2.33 N, 6.1 N m, 3.9 W** | -2.22 N |
| 0.4 | 0.75 Hz | +0.05 N | **+4.53 N, 11.7 N m, 12.7 W** | -4.52 N |
| 0.4 | 1.0 Hz | +0.11 N | +7.04 N, 20.7 N m (over the cap) | -7.19 N |
| 0.6 | 0.25 Hz | +0.03 N | +2.18 N, 9.7 N m | -2.06 N |

- **R4 confirmed**: +2.33 and -2.22 N (predicted 1.5-3.5, within 20% of
  each other), welded 0.03 N.
- With the push acting the valve is cheaper and stronger than first read:
  the best point within the caps is 4.53 N at 11.7 N m (first run: 2.34 N).
  The torque spikes of the first run are gone.
- The flap itself (0.5 Hz): a cut at 0.2 chord with the stiffer spring gives
  3.62 N at 5.9 N m; a cut at 0.7 gives 0.5-0.7 N.
- **R5 confirmed**: mean power is negative at 10 of 128 flap points with a
  feather (7.8%), and at 0 of 64 rowing points. The power-accounting fault
  is not the push. It is still untested; `freerig.py` applies the drive as
  torque and would settle it.

## Reading

- Pass 1j's decision stands for the lift-based stroke: no passive per-beat
  flap there.
- It changes for a drag-based stroke at rest: the flap is a working valve
  worth 2.3-4.5 N of static thrust within the torque caps, where a rigid
  paddle gives nothing. That is a candidate for starting from rest and for
  manoeuvring in water. At 0.3 m/s in the broadside posture it still never
  overcomes the posture's drag.
