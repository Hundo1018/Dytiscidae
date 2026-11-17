# Passes D2b and D2c, pre-registration: the feather crank away from its toggle; the rotor's inertia

2026-11-17, on `53d3c76`. Frozen before any row of either. Follows D2's two
open items (`RESULT_D2_feather_drive.md`). Run beside D3 and D4 (other
workers), which touch other files. Nothing is chosen by either pass.

Code added before freezing, no run in air or vacuum with it: `drive.py`
takes `feather["arm_scale"]` (the arm's radius against the frozen rule's
`c_h sin 50`), `feather["motor"]` (the feather motor's own parameters),
reports the design's feather range and `transmission_min`, and a bug in
the rod-length bisection is fixed (it raised `KeyError` on most geometries;
D2's geometry was not affected: same arm, rod and design before and after).

## D2b. FX2's geometry swept

### Seen before freezing (kinematics only, `scratchpad/d2b_probe.py`)

`transmission_min`: over the crank's turn, the least cosine between the rod
and the direction the horn moves when the feather turns (1 square on, 0 at
toggle). Design only, no dynamics:

| horn c_h | shaft below horn | arm scale | design feather amp | range | transmission_min | rod |
|---|---|---|---|---|---|---|
| 30 | 80 | 1.00 (D2's) | 55.6 | -55.0..+56.2 | 0.14 | 83 |
| 30 | 80 | 0.85 | 42.9 | -41.7..+44.1 | 0.40 | 82 |
| 30 | 120 | 1.00 | 52.9 | -52.2..+53.5 | 0.33 | 122 |
| 30 | 120 | 0.85 | 42.1 | -41.2..+43.0 | 0.51 | 121 |
| 30 | 160 | 1.00 | 52.1 | -51.6..+52.7 | 0.42 | 161 |
| 30 | 160 | 0.85 | 41.8 | -41.1..+42.4 | 0.56 | 161 |
| 45 | 80 | 0.85 | 45.7 | -45.5..+46.0 | 0.13 | 84 |
| 45 | 120 | 1.00 | 55.6 | -55.0..+56.2 | 0.14 | 124 |
| 45 | 120 | 0.85 | 42.9 | -41.7..+44.1 | 0.40 | 122 |
| 45 | 160 | 1.00 | 53.5 | -53.1..+53.9 | 0.28 | 163 |
| 45 | 160 | 0.85 | 42.2 | -41.3..+43.2 | 0.49 | 162 |
| 60 | 80 | 0.85 | 45.5 | -40.1..+51.0 | 0.12 | 89 |
| 60 | 120 | 0.85 | 44.4 | -43.7..+45.2 | 0.24 | 124 |
| 60 | 160 | 1.00 | 55.6 | -55.0..+56.2 | 0.14 | 166 |
| 60 | 160 | 0.85 | 42.9 | -41.7..+44.1 | 0.40 | 163 |

(mm, deg.) 45/80/1.00, 60/80/1.00 and 60/120/1.00 do not close. D2's
geometry was the worst of those that close: what sets the margin is the
shaft's depth against the horn, and the arm's radius.

### Runs

Every row of the table, take-off as D2 (stroke by FB, lever 60 mm, gear
24), in air and in a vacuum: 30 rows. Read with D2's reader against D2's F0
(1.927 weights, 178.2 W) and FM best (295.5 W).

### Predictions

- B1. A geometry is **readable** if, in a vacuum, input per beat is within
  1% of the motor's peak shaft power, the largest loop gap is under 0.5 mm,
  and the feather's range as made is within 3 deg of its design's at both
  ends. Readable: all seven with transmission_min >= 0.40. Not readable:
  all five at <= 0.14. The three between (0.24, 0.28, 0.33): no prediction.
- B2. Readable geometries in air: feather amplitude within 3 deg of design,
  lead within 20 deg of a quarter beat, beat 0.98 or more, motor on its
  limit under 10% of the beat, loop gap under 0.5 mm.
- B3. Readable, arm scale 1.00 (30/160): lift within 10% of F0's; machine
  electrical power 170-215 W, i.e. under FM's best by at least 80 W.
- B4. Readable, arm scale 0.85 (feather 42-43 deg): lift 75-100% of F0's;
  machine power 160-215 W.
- B5. Peak loop force in the readable rows under 400 N at horn 30 mm and
  under 250 N at horn 60 mm (the feather's 2.9 N m over horn times
  transmission, with margin).

A geometry that is not readable by B1 is reported with its row labelled
and is not read for B2-B5.

## D2c. The rotor's inertia, as a sensitivity

D1 and D2 each ended on the same unmeasured sentence: the cost of a
reversing drive (direct drive of the stroke; a motor on the feather) is
copper loss carrying the rotor's own reflected inertia through each
reversal, and falls with a lighter rotor. The witness has J_r 2e-5 kg m^2.
Swept here: J_r in {2e-5 (stored rows), 5e-6, 1e-6}, nothing else changed.
Whether a motor of this torque and speed with such a rotor exists is not
asked; this is the size of the lever, not a part.

Runs: FM (feather motor only gets the lighter rotor; stroke by FB gear 24)
at gears {16, 24, 32}; DD at take-off gears {12, 16, 24, 32} and at cruise
{32, 48, 64, 90} (both motors' rule as D1); FB as the control whose rotor
does not reverse, cruise gear 48 and take-off gear 24 (lever 60 mm). 30
new rows. Best gear by D1's rule with the beat held.

- C1. FM's machine power falls with J_r at every gear. At 1e-6 its best is
  210 W or less (F0's 178 plus the copper of carrying the feather's torque,
  which I estimate at 10-40 W), at gear 32; at 5e-6, 215-260 W.
- C2. DD at cruise: 14.6 W at 2e-5; at 1e-6 within 15% of FB's 7.32 W. DD
  at take-off: 295 W at 2e-5; at 1e-6 under 175 W with its motor on the
  limit under 10% of the beat.
- C3. FB changes by under 2% in electrical power across the three rotors
  at both points (its rotor does not reverse). If it moves more, C1 and C2
  are not attributable to reversal and are reported as such.
- C4. Lift of every row within 3% of the same candidate's at 2e-5 (the
  rotor changes the bill, not the stroke), where amplitude and beat hold.
