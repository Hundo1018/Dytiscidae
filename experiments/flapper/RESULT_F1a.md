# F1a and F1a-2 results — 2026-11-12

Both pre-registrations were frozen and committed before their grids.
Reproduce (about 6 and 4 minutes on 8 processes; no row non-finite):

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/twist_trim.py experiments/flapper/results/twist_trim.jsonl --jobs 8
    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/passive_trim.py experiments/flapper/results/passive_trim.jsonl --jobs 8

`rig.py` reproduces a stored F0c row to the last digit (lift 0.70979 N, force
along the path 0.00689 N, power 3.9878 W).

## F1a: a prescribed wrist (the outer half of the wing may feather more)

Least aerodynamic power in trimmed level flight, 0.25 m wing in two panels:

| wrist twist amplitude | 6 m/s | 8 m/s |
|---|---|---|
| -10 deg (outer feathers less) | does not fly level | does not fly level |
| 0 (rigid) | 5.28 W | 5.05 W |
| 10 deg | 3.79 W | **2.11 W** (8 Hz, stroke 60, inner feather 10) |
| 20 deg | 4.02 W | 2.51 W (6 Hz) |
| 30 deg | **3.43 W** (8 Hz, stroke 100, inner feather 20) | 3.82 W |

Predictions: 1 *"at least 30% under the least with T = 0"* — **held**: 35%
and 58%. 2 *"the least-power point over all T has T >= 10, and T = -10 is
never the best"* — **held**. 3 *"level flight exists at 5 Hz or less at 8
m/s"* — **refuted**: 6 Hz. 4 *"At T = 0 the two-panel wing's least power is
within 15% of the one-panel wing's"* — **held at 6 m/s** (5.28 against
5.33), **refuted at 8 m/s** (5.05 against 3.99): the two wrist motors add
4 g, and the one-panel wing's cheapest point no longer holds the speed once
trimmed to the heavier weight.

## F1a-2: a passive wrist (the package's sprung pitch hinge)

| compliance | 6 m/s | 8 m/s |
|---|---|---|
| 0.5 | 5.31 W | 4.04 W |
| 1 | 5.25 W | 3.94 W |
| 2 | 5.21 W | 3.87 W |
| 4 | 5.19 W | 3.83 W |
| prescribed wrist, for reference | 3.43 W | 2.11 W |

Predictions: 1 *"within 25% of the prescribed wrist's"* — **refuted**: 1.5
and 1.8 times. 2 *"The best compliance is 1 or more"* — **held** (4, by 2%).
3 *"Every compliance has at least one point that flies level at 8 m/s"* —
**held**.

## By the frozen rules, and what they do not settle

- Twist is worth 35-58% of the cruise power when it is prescribed. The wing
  as the package can make it passive does not get it: that hinge is sprung
  to zero and critically damped against the load's own stiffness
  (`PITCH_DAMPING_RATIO` 1.0, set for stability on kilogram wings), and
  across an eightfold range of compliance the power moves 2%. So the result
  is about that hinge, not about membranes: in this simulator a passive
  outer wing is close to a rigid one.
- So in stage 1 a wrist, if there is one, is driven. Whether there should
  be one, and about which axis, is not settled by air: energy is not counted
  by the user, the rigid wing already flies level at 8 Hz and lifts 1.6
  times the weight at 12 Hz, and what twist buys is margin (more points
  that fly, 6 Hz instead of 8 at 8 m/s). The literature (A11 items 8 and 10)
  says the degree of freedom the water stroke and a loaded step on land ask
  for is a fore-aft sweep. The wrist's axis and the wing count are therefore
  decided after the land and water measurements, with twist kept as a
  known option worth 35-58% in cruise.

Limits: held body, quasi-steady model, coarse grids, one hand fraction
(0.5), twist in phase with the feather only.
