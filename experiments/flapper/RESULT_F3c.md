# F3c result — 2026-11-12

`PREREG_F3c.md` was frozen and committed first; `swim.py` unchanged
(commit ce249df). Reproduce:
`PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/swim.py run --stroke S2x --arm swim,still,frozen --seeds 8-23 --jobs 6 --out <abs path>`
(rows: `results/swim_F3c.jsonl`). The lead also re-ran three of F3b's stored
rows (S2x swim, still, frozen, seed 4): every field equal.

| arm, seeds 8-23 | water score | progress part | at or over 0.15 | under 0.012 |
|---|---|---|---|---|
| S2x, the diagonal stroke | 0.347-0.491 | 0.71-0.94 | 16 of 16 | 0 |
| every actuator held | 0.000-0.072 | 0.00 on 16 | 0 | 14 of 16 |
| command frozen | 0.004-0.066 | 0.00 on 16 | 0 | 8 of 16 |

No `bad_qacc`, no failed rollout.

Predictions: 1 *"It holds: 14 or more of 16"* — the swim clause **held**
(16 of 16). 2 *"The still arm reads under 1e-6 on every seed"* —
**refuted**: 0.034 and 0.072 on two seeds, 0.005 and 0.007 on two more.
3 *"The frozen arm's progress part is 0.00 on every seed; its whole score is
above 0.012 on at least 4 of 16"* — **held** (8).

## By the rule as written

The rule asked the still arm for under 0.012 on 16 of 16 and it is 14. **So
the rule is not met and F3c does not supersede F3b.** What failed is not the
stroke: a still body makes no progress on any seed and is paid hold credit
on two, because this body sinks (0.958 N of weight on 0.801 N of buoyancy)
and on those draws it sinks past the commanded depth during the hold. That
is the leak CLAUDE.md's rule 3 records (a dense still body clearing water by
sinking) and DIAGNOSIS P0.2 lists as open (one depth command, answerable one
way).

What stands: the two-pair body swims at 0.33-0.34 m/s on its diagonals with
the shoulders it has, on 24 of 24 seeds across F3b and F3c, with the hull
heaving 3-4 mm where the in-phase stroke heaved 230 mm. What does not: a
clean null in water for this body, its buoyancy, and its depth (hold credit
0.01-0.10 while swimming).

The wrist is not built on this evidence, and the water is not called done.
Two things come first, neither of them a wrist: the hull's volume set so the
body is neutral, and the water task's depth command made a pair so that a
body that only sinks cannot be paid. Then this file's rows are read again.
