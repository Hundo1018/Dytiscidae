# Pass S1 result: the entrainment reaction under a driven wing in water (2026-11-15)

Pre-registration: `PREREG_S1_entrainment.md` (commit 6501a9a; five probe
findings disclosed there). Data: `results/S1_entrain.jsonl`, 820 rows, all
finite, one pass. Reader: `s1_read.py`. Mechanism drawn from the built model:
`figures/S1_wing0.4.png`, `figures/S1_wing0.6.png`. Body held, 0.3 m/s,
water, pass 1l's grid (192 moving points).

Control: the held rig as run reproduces `L1l_camwater.jsonl` exactly at 8 of
8 points, and the 10 flap points of `L1j_flap_r.jsonl` exactly.

## What was wrong, and what was changed

1. **The solver's entrainment reaction differentiates a quantity that depends
   on velocity.** A wing strip's added mass in the mass matrix is its tensor
   projected on the direction of the relative flow. `entrainment_reaction`
   reacts every growth of it with `-dm/dt v`, one-sided. Under a wing driven
   by torque that is an acceleration fed back explicitly, and it diverged.
   - New opt-in mode `FluidSolver.entrainment = "wetting"`: reacts only the
     growth a strip's density brings (water arriving), which is what the
     term was written for (the ray's water entry). Numpy path only. **The
     default is unchanged**, so the search and every stored score are as
     they were.
   - Test `test_wetting_entrainment_reacts_water_arriving_not_the_flow_turning`
     (5 checks), mutation `wetting-entrainment-reacts-standing-water` (caught). A first mutation, reacting shed water too, survived: `entrainment_reaction` clamps the per-body sum itself, so the per-strip one-sidedness is not held by a test.
2. **The held rig's torque and power series carried the entrained weight's
   cancellation**, an upward force the solver adds at each body's centre of
   mass to cancel the weight MuJoCo gives the added mass. Only the net force
   had it taken out. Fixed in `rig.run` (`legacy_added_weight=True`
   reproduces the old reading).
3. `amref.py`: the plate's normal added-mass force from the prescribed
   motion, as a reference for the inertia the held rig cannot feel.

## Against the frozen text

- **P1 confirmed.** Torque-driven with "wetting": 192 of 192 finite, worst
  tracking error 0.17 deg. Halving the step moves Fx by 0.0%, power by at
  most 0.1% and the crank metric by at most 0.7% (8 of 8).
- **P2 refuted as worded** (158 of 192 inside both bands; 173 predicted).
  - Mean power agrees within 5% at 174 points; median ratio 1.002.
  - Mean Fx agrees at 161; median difference 0.16 N.
  - The misses are at the large feather amplitudes: 45 of 48 agree at 20 and
    at 30 deg, 39 at 50, 29 at 60. Not explained.
- **P3 confirmed.** The corrected crank metric is below pass 1l's at 192 of
  192, median ratio 0.66 (0.36-0.96); the fluid stroke torque's median ratio
  is 0.70. The 0.6 wing now meets the gate under each centre-of-pressure law.
- **P4 confirmed.** With the solver's default reaction the held rig's Fx is
  lower at 192 of 192, median loss 22.8% of |Fx|; power is 5% higher.
- **P5 refuted** (99 of 192 within 20%; 134 predicted). The torque-driven
  rig's crank metric equals the held rig's without the reference (median
  ratio 0.99-1.01 at every frequency) and is nearer the reference at only 13
  points. The reference adds 23% (median) to the crank metric. So the
  scalar mass matrix puts almost none of the plate's normal added-mass load
  into the drive torque here.
- **P6 refuted in size.** The reference raises mean Fx at 192 of 192, by a
  median of 103% (30-80% predicted), and mean power by 22%.
- **P7 confirmed.** All 10 negative-power flap points read positive on the
  corrected held rig (+0.18 to +12.3 W) and on the torque-driven rig (+0.18
  to +19.9 W). The negative power was the rig's accounting. This arm changed
  both the weight correction and the reaction, so which of the two it was is
  not separated.

## Pass 1l re-read (a point counts only if it meets the gate without and with the reference)

Points meeting the gate, of 8 per cell, and the best force under the caps
(the lower of the two readings) with the higher crank metric:

| wing | feather | no law | plate | insect | pass 1l as run |
|---|---|---|---|---|---|
| 0.4 | 20 deg | 5; 10.3 N, 7.3 N m | 5; 6.8 N, 6.2 | 5; 6.8 N, 6.0 | 2 / 2 / 2 |
| 0.4 | 30 deg | 4; 10.7 N, 7.9 | 3; 6.5 N, 7.6 | 3; 6.5 N, 7.3 | 3 / 3 / 3 |
| 0.4 | 50 deg | 1; 7.5 N, 10.6 | 0 | 0 | 1 / 1 / 1 |
| 0.4 | 60 deg | 0 | 0 | 0 | 0 / 0 / 0 |
| 0.6 | 20 deg | 3; 5.3 N, 5.7 | 2; 4.0 N, 5.1 | 2; 4.0 N, 5.0 | 1 / 0 / 0 |
| 0.6 | 30 deg | 3; 8.7 N, 8.2 | 2; 5.8 N, 7.6 | 2; 5.8 N, 7.3 | 0 / 0 / 0 |
| 0.6 | 50 deg | 1; 2.4 N, 10.3 | 1; 0.6 N, 10.2 | 1; 0.6 N, 10.1 | 1 / 0 / 0 |
| 0.6 | 60 deg | 0 | 0 | 0 | 0 / 0 / 0 |

- Pass 1l's conclusion that the cruise amplitudes (20-30 deg) are the water
  amplitudes stands, with more margin: the forces are about twice what it
  read (it ran with the solver's default reaction and the inflated torque
  kept the faster points out).
- **Pass 1l's finding that the 0.6 wing fails under either law is
  withdrawn.** It was the rig's torque. The 0.6 wing meets the gate at 20
  and 30 deg under both laws.
- The take-off amplitudes still do not work in water: 50 deg only without a
  law, 60 deg nowhere.

## Reading

- Rival (b) holds. The instability is gone and the two rigs agree on mean
  force and power where the feather is 20-30 deg. But the mass matrix does
  not carry the wing's added-mass load into the drive.
- **Water thrust and torque now carry a band between two added-mass models**,
  neither validated here: the solver's (scalar projection in the mass
  matrix) and the plate reference (normal force only). The reference gives
  about twice the thrust, 22% more power and 23% more crank torque. The gate
  above takes the worse of the two on each quantity.
- The solver's default reaction, which every search score in water was read
  through, is a one-sided drag on a flapping wing: 23% of thrust (median) on
  this grid. Whether to change the default is a separate decision: it needs
  the Mojo kernel changed to match and makes water scores not comparable
  across the change.
- Air is not affected: on pass 1i's 42 gate points the corrected rig moves
  the crank metric by -0.7% to +0.4% and all 42 stay within 12 N m.

## What this does not establish

- Which added-mass model is right. A flapping plate in water with force and
  torque measured through the beat would settle it; the bench is the test.
- Why the rigs' mean Fx part at feather 50-60 deg.
- A released body in water (now possible on the torque-driven rig, not run).
- The fold's mechanism, and a cam whose amplitude can be changed.

## The drawing against the design (looked at, 2026-11-15)

`figures/S1_wing0.4.png`, `figures/S1_wing0.6.png`, drawn by `draw.py` from
the built model and checked by eye and by its table.

- Chord 0.258 m, wing area (0.136 and 0.204 m2) and mass 0.60 kg match the
  design. The feather axis is on the leading edge and the stroke axis along
  the hull, as intended.
- **The span differs from the design: 0.53 and 0.79 m built against 0.61 and
  0.87 m.** The wings' roots and the stroke axis are on the hull's centre
  line, not at its sides (radius 0.04 m), so the inner 0.04 m of each wing is
  inside the hull and every strip is 0.04 m nearer the stroke axis than the
  design has it. This is the fixture of every L1 pass (full wing: 1.32 m
  against 1.40). Not corrected here; its effect on force and torque is not
  measured.
