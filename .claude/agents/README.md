# Agent roles for the Dytiscidae development loop

Six roles, ordered. The ordering is the point: the three generative roles have
no web tools so that novelty questions cannot leak backwards into generation,
and the Historian refuses to run before there are results.

```
explorer ─┐
mutator  ─┼─> candidates ─> adversary ─> judge ─> surviving ─> historian
assumption-breaker ─┘        (attacks)   (verdict)  candidates   (novelty)
```

| role | model | web | writes code |
|---|---|---|---|
| `explorer` | opus | no | no |
| `mutator` | opus | no | no |
| `assumption-breaker` | opus | no | no |
| `adversary` | sonnet | no | yes (reproducers) |
| `judge` | opus | no | no |
| `historian` | sonnet | yes | no |

Escalation follows `~/.claude/playbooks/10-model-dispatch.md` §6: a role that
misses twice on a task with clear criteria goes up one tier, carrying both
failure traces.

## Rules every role shares

- Reports are <=15 lines (Judge and Historian: <=20). Long artifacts go to
  files; the report carries the path.
- Claims cite `path:line`, or paste the line of output they came from.
- "It works" without pasted evidence is not a claim, it is an adjective.
- An honest `UNKNOWN` / `UNVERIFIED` / `HYPOTHESIS` beats an invented answer.
- **Do not disturb a live training run.** `runs/arch34` and the PID in
  `runs/arch34.pid` belong to a multi-hour single-arm experiment; killing,
  reconfiguring or writing into it destroys the arm. Keep spawned process
  counts low - the pool-shape optimum is contention-sensitive.

## Where the ground truth lives

- `docs/ROADMAP.md` - every current design decision with the measurement behind
  it. Read before proposing anything it already explains.
- `docs/CPU_LEGACY.md` - the older backlog, superseded where they disagree.
- `runs/arch34_notes.md` - the live run's configuration and findings.
- `runs/arch34_watch.py` - what counts as a milestone or an anomaly, and why
  each threshold is where it is.
