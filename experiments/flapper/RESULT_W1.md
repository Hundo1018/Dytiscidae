# W1 result — 2026-11-12

`PREREG_W1.md` was frozen and committed first. Reproduce, with
`DYTISCIDAE_DEPTH_PAIR=1 FLAPPER_BODY_B='{"hull_radius": 0.0209}'` and
`FLAPPER_WAKE_GAIN` 1 or 3:

    .venv/bin/python experiments/flapper/wake_check.py
    .venv/bin/python experiments/flapper/swim.py run --stroke S2x --arm swim --seeds 24-39 --jobs 6 --out <abs path>
    .venv/bin/python experiments/flapper/swim.py run --stroke S2d --arm swim --seeds 40-55 --jobs 6 --out <abs path>

(rows: `results/wake_W1_*.jsonl`). No `bad_qacc`, no failed rollout.

| wake gain | S2x: mean score | S2x: swims | S2d: mean score | S2d: swims | S2d: depth response | at or over 0.5 |
|---|---|---|---|---|---|---|
| 0 (F3d, F3e) | 0.424 | 16 of 16 | 0.397 | 15 of 16 | 0.54-0.84 | 16 of 16 |
| 1 | 0.400 (-6%) | 16 of 16 | 0.328 | 15 of 16 | 0.56-0.86 | 16 of 16 |
| 3 | 0.321 (-24%) | 16 of 16 | 0.425 | 16 of 16 | 0.52-0.86 | 16 of 16 |

The command-frozen arm at gain 3 reads 0 on 4 of 4.

Predictions: 1 *"Gain 1: S2x swims on at least 14 of 16 and its mean score
is within 10% of gain 0's"* — **held** (-6%). 2 *"Gain 3: S2x swims on at
least 14 of 16; its mean score is 15-40% lower"* — **held** (-24%).
3, 4 *"S2d's depth response bar is met at gain 1 ... at gain 3"* — **held**
both. 5 *"The frozen arm at gain 3 is under 0.012 on 4 of 4"* — **held**.

**By the rule, the water verdicts are robust to a fore-to-hind wake up to
about the size of Wagner's effect, with the law untouched.** The wake is not
small: at that size it takes a quarter of the plain swim's score, and it
moves the diving stroke's score by 17% one way and 7% the other. It does not
change which side of a bar anything is on.

What this does not cover: trailing vortices and so the mean downwash's shape
(still one uniform inflow); the hind wings' wake on the fore wings; wakes
meeting the body; a gain above 3.
