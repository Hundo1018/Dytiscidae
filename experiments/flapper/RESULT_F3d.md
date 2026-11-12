# F3d result — 2026-11-12

`PREREG_F3d.md` was frozen and committed first. Reproduce:

    DYTISCIDAE_DEPTH_PAIR=1 FLAPPER_BODY_B='{"hull_radius": 0.0209}' PYTHONPATH=. MUJOCO_GL=disable \
      .venv/bin/python experiments/flapper/swim.py run --stroke S2x --arm swim,still,frozen --seeds 24-39 --jobs 5 --out <abs path>

(rows: `results/swim_F3d.jsonl`; `swim.py` as at ce249df, stroke and law not
re-tuned). Neutral hull: radius 0.02086 m gives weight 1.0275 N and buoyancy
1.0275 N on the built model; 0.0209 was used.

| arm, seeds 24-39 | water score | at or over 0.15 | under 0.012 | largest |
|---|---|---|---|---|
| S2x, the diagonal stroke | 0.358-0.473 | 16 of 16 | 0 | |
| every actuator held | 0 | 0 | 16 of 16 | 3.9e-18 |
| command frozen | 0 | 0 | 16 of 16 | 5.4e-17 |

No `bad_qacc`, no failed rollout. On seed 27 the still body's own half reads
a hold of 0.13 and the frozen arm's 0.11: the same sinking-through credit as
before, now scaled by a response of zero.

**The rule is met: the two-pair machine swims.** Predictions: 1 *"All three
clauses hold"* — **held**. 2 *"Still arm: under 1e-9 on every seed"* —
**held**. 3 *"The swimmer's depth response ... is under 0.3 on most seeds"*
— **not measured directly** (`swim.py` keeps a subset of the measurements
and the response is not in it); by the arithmetic of the score it holds: the
score is `0.5 x progress x served + 0.5 x pair hold`, progress is 0.72-0.94
and the score 0.36-0.47, so the pair hold adds 0.00-0.01 on every seed.

What this is and is not. It swims: 0.72-0.94 of the commanded progress on
the drawn heading and on its opposite, on 40 seeds across F3b, F3c and F3d,
with a body that neither sinks nor floats. It does not hold or change depth
on command: asked for a depth either side of where it starts, it answers
nothing. Depth control is open, with the stroke's torque (0.93-0.97 N m at
1 Hz) and the absence of wake between the pairs in the model.
