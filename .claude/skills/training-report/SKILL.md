---
name: training-report
description: Produce the end-of-run report for a Dytiscidae search — a self-contained HTML page of charts built from the run's telemetry, plus video of the best design. Use when a search finishes, when the user asks for training charts, a run report, a dashboard, or to see the best design; also for comparing a finished run against an earlier one.
---

# Training report

Two artifacts from one finished run: a self-contained HTML page of charts, and
video of what the search actually produced. Run the report first — it tells you
which design is worth filming and whether the run is worth reporting at all.

## 1 · The page

```bash
.venv/bin/python .claude/skills/training-report/report.py runs/<run>
```

Writes `runs/<run>/report.html`. `report.py` aggregates
`generations.jsonl` and `events.jsonl`; `template.html` is the page, and the
data is injected at `__PAYLOAD__`. Neither file needs editing for a normal run.

The page has ten sections: search return, evaluation return (the Tier-1 against
Tier-2 scatter), feasibility rates, score retention, policy and value loss,
entropy/KL/clip, distributions, the archive as visited state space, what
structural evolution cost, and the best design.

**Absent data stays absent.** A run without `--shared-policy` has no PPO events,
and `report.py` says so on stdout rather than drawing empty axes. The page ends
with a section naming what this telemetry does not contain — Q/V value traces,
gradient norm, per-step action distributions, cross-seed curves — because a
report that quietly omits them reads as if they were fine.

## 2 · The video

```bash
# continuous mission, wake and stress overlays -- what it does end to end
.venv/bin/python -m dytiscidae.ops.run showcase --design runs/<run> \
    --island <island> --run runs/showcase_<run> --leg-seconds 8 --cycles 1

# per-segment clips and a turntable for the top archive elites
.venv/bin/python -m dytiscidae.ops.run render --run runs/<run> --top 3 --duration 8
```

`showcase` picks the highest **fitness** elite in the island, which is not
always the highest `mission_fraction` — read the best-design block in the report
and film the island that owns it. `render` writes
`runs/<run>/media/elite{N}_{air,water,land,turntable}.mp4`.

Both are minutes of compute. Launch them detached and keep the process count
low if anything else is running.

## 3 · Reading it honestly

The failure mode of a report is flattery. Three rules earned from arch34:

- **Film the result, not the best-looking clip.** arch34's continuous mission
  video shows 33% on-task with 0% in air and 0% in water. That is the run's
  actual result and it agrees with the Tier-1.5 measurement in section 4. Show
  it and say so.
- **Check what a number compares to before putting it beside another one.**
  `mission_fraction` levels are not comparable across a change that redefined
  the air score, and the report never draws two runs on one axis for that
  reason. Say "not comparable" rather than shrinking the difference.
- **Per-island series are not one series.** `filled`, `coverage` and `qd_score`
  in the generation line belong to whichever island that generation visited, and
  a descriptor refit merges cells every 400 evaluations. The coverage chart
  marks the refits for exactly this reason.

## 4 · Charts the source data cannot support

If asked for a standard RL dashboard, these have no counterpart here and should
be declined rather than approximated:

| asked for | what exists | verdict |
|---|---|---|
| training / evaluation return | mission fraction, fitness, Tier-2 verifications | charted |
| success rate | no binary success; energy, airworthiness and divergence gates | charted as three rates |
| episode length | segments are fixed length | charted as Tier-1.5 retention instead, labelled |
| policy / value loss, entropy, KL | logged per PPO update | charted |
| Q / V value | only the value *loss* is logged | absent |
| gradient norm | not logged | absent; clip fraction charted as the nearest proxy |
| action distribution | actions are not logged per step | absent; reward by segment tag charted |
| learning curves across seeds | one seed per run | absent unless several runs are given |
| state visitation | the MAP-Elites archive is the visited space | charted as coverage + refits |
| win rate, death cause | no win condition | the failure taxonomy is the equivalent |

## 5 · The palette is validated, not chosen

`template.html` uses four categorical slots that pass the `dataviz` skill's
validator in both modes (worst adjacent CVD ΔE 9.1 protan light, 8.4 dark;
light mode carries a contrast WARN on aqua and yellow, which is why every
series has a legend entry and a direct label). If you change a colour, re-run:

```bash
node <dataviz>/scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
```
