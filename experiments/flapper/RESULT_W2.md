# W2 result — 2026-11-12

`PREREG_W2.md` was frozen and committed before any rollout. Reproduce, with
`FLAPPER_BODY_B='{"hull_radius": 0.0209}'` and `FLAPPER_WAKE_GAIN` 1 or 3:

    .venv/bin/python experiments/flapper/fly.py level --arm fly --seeds 0-7 --jobs 6 --out <abs path>
    .venv/bin/python experiments/flapper/fly.py run --arm fly --seeds 0-15 --jobs 6 --out <abs path>

(rows: `results/wake_W2_*.jsonl`). No `bad_qacc`, no failed rollout.

| wake gain | level bar | turn bar: air score | at or over 0.15 |
|---|---|---|---|
| 0 (F4) | 8 of 8 | 0.907-0.983, mean 0.945 | 16 of 16 |
| 1 | 8 of 8 | 0.907-0.980, mean 0.947 | 16 of 16 |
| 3 | 8 of 8 | 0.909-0.985, mean 0.949 | 16 of 16 |

The command-frozen arm at gain 3 reads 0 on 4 of 4. All six predictions
held (the level bar at gain 3 was held at 0.4).

The feedback was acting: on this body it finds 20 fore and 20 hind strips,
and every launch flies a different path with it (seed 0's height change is
-1.48 m without it, -0.41 m at gain 1, -0.37 m at gain 3; end speeds move by
up to 0.2 m/s). The law, closed around attitude at every step, absorbs it.

**By the rule, the flight verdict is robust to a fore-to-hind wake up to
about the size of Wagner's effect, with the law untouched.**

The same limits as W1: no trailing vortices (the mean downwash is still one
uniform inflow for the whole machine, which for a tandem wing in forward
flight is the larger omission: the hind pair flies in the fore pair's
downwash and that changes its trim incidence), no hind-on-fore effect, no
gain above 3.
