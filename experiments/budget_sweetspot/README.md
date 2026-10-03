# Budget sweet spot: generations, refinement, pool shape — 2026-10-03, ec750f8 (+ uncommitted 10-03 crossing fix)

## Frozen before the sweep

**0 Question.** For a fixed day of compute, which `--generations`, `--refine-steps`
and pool shape (`--workers/--min-shard/--pool-per-worker/--pool-balance`) buy the
most search? Decision: the arch47 launch line.

**1 Prior art.** Pool shape 4×4 swept on *random* bodies (CLAUDE.md, 31.7 s).
"Run 900, read at 500" (memory, arch36/37, rung shares). ROADMAP AK (refine worth,
pre-registered: <5% placement change ⇒ at most one step) and AJ (idle >10% ⇒ sweep
queue/balance against the real path) — both built 09-30, neither read until now.
Rivals for any wall difference: the config; machine-load noise; batch composition.

**Read from stored telemetry (no new runs), `retro.py` and `stage_time.py`:**

| | arch45 (8×2) | arch46 (4×4) |
|---|---|---|
| s/gen, gens 6–100 → 500–700 | 111 → 227 | 114 → 230 (gen 400–499) |
| `main` stage, gen band 0 → last | 66 → 141 s | 67 → 136 s |
| evaluations / generation | 16, constant | 16, constant |
| refine: placements changed | 75 / 14 400 = **0.52%** | 43 / 8 000 = **0.54%** |
| refine: share of stage wall | 18.3% | 19.5% |
| pool idle (`idle` field) | **0.39** | **0.24** |

- Per-generation cost doubles over a run with the same 16 evaluations. Mechanism:
  mean `n_rotors` of evaluated designs 0.6 → 3.6; per-eval wall regresses on
  rotors at +1.6 s each (R² 0.42 arch45, 0.25 arch46) and on DOF at R² 0.19 / 0.07.
  `--pool-balance`'s cost model is `35 + 0.9·n_actuated` — the weaker predictor.
- AK: 0.5% ≪ 5%, refinement is not buying selection. Mechanism: the (1+1)-ES
  accepts on `mission_fraction`, which is ~0 for nearly every candidate
  (`mf=0.000` throughout), so its objective is flat. 157 accepts in 14 400 trials.
- Generations: pooled top-10 competence (all islands) in arch46 reaches water
  0.44 / land 0.56 by gen 200 (7.9 h); gens 200–450 add land +0.08, water −0.01,
  air −0.06. arch45: water 0.365@200 → 0.442@800; land peaked 0.576@200; air
  0.218@200 → 0.492@700 → 0.315@850. Snapshot-to-snapshot swing (noise proxy):
  air ±0.1, land ±0.05, water ±0.02.

**3 Prediction (pool sweep).** On late-arch46 bodies (rotors present), with the
same 16 designs per batch in every config:
- `4×2, per-worker 2` (queue of 8 shards) beats 4×4 by ≥10% mean wall.
- `--pool-balance` on 4×4 beats 4×4 by ≥5%, *less* than the queue, because its
  cost model is DOF and the cost is rotors.
- Falsified if 4×4 is within 5% of the best, or the queue loses.

**4 Measurement.** Wall of `ActorPool.evaluate_tier1` (identification on, shared
policy attached, 6 s segments = arch45/46) on 3 batches of 16 genomes drawn with a
fixed seed from arch46's final archives (all islands). Noise floor: 4×4 is run
twice per batch, first and last; the spread between them is the floor.
Pool shape does not change scores (asserted in `tests/test_search.py`), so this is
a pure-speed read; no inert probe applies.

**5 Design.** Configs interleaved within each batch; 4 workers in every config
(memory: 8×2 was OOM-killed on this machine). Stop after 3 batches. A difference
below the 4×4-vs-4×4 spread is not a difference.

## Appended after

**6 What ran.** `systemd-run --user --unit poolsweep-1003 -p MemoryMax=8G`,
`experiments/perf/pool_sweep.py --source runs/arch46 --batches 3 --seconds 6
--shapes 4x4 4x4:b 4x2:q2 4x2:q2:b 4x4`; rows in `pool_sweep_1003.jsonl`.
Batches hold 1-3 bodies of 12/21/13 rotors, like late generations.

**8 Result** (wall s; ratio to the mean of the two 4×4 runs of the same batch):

| batch | 4×4 first / last (floor) | 4×4:b | 4×2:q2 | 4×2:q2:b |
|---|---|---|---|---|
| 0 | 163.6 / 177.7 (9%) | 0.93 | **0.80** | 1.20 |
| 1 | 240.8 / 226.6 (6%) | **0.75** | 0.92 | 0.71 |
| 2 | 155.5 / 175.6 (13%) | 0.96 | **0.88** | 1.32 |
| total | | 0.866 | **0.875** | 1.031 |

**9 Against the frozen text.** "queue beats 4×4 by ≥10% mean wall": **confirmed**
(−12.5%, faster on 3/3 batches, though batch 1 only by 8%, inside its floor).
"balance ≥5% and less than the queue": the 5% holds on the total (−13.4%) but it
comes from one batch (0.75) with two inside the floor, and the ordering against
the queue is **not resolved** — the totals differ by 1%. Queue + balance is
unstable (0.71–1.32): balance sorts shards by a DOF cost that misses rotors, so
a mispredicted heavy shard can go last in the queue.

**10 Belief.** 4×4 was the optimum on random bodies; on late bodies, whose
cost is heavy-tailed in rotor count, a queue of 8 shards of 2 is ~12% faster
and never slower. Would move back if a run-length read (30+ generations, idle
machine) puts it inside the run-to-run spread.

**11** Not replicated on a live run; the arch47 `stages` events are that read
(`main` per band against arch46's 66→136 s).

**12 Decision for arch47:** `--workers 4 --min-shard 2 --pool-per-worker 2`,
no `--pool-balance` until its cost model reads rotors (R² 0.42 vs DOF 0.19).
`--refine-steps 0` (AK: 0.5% placements, 19% of wall). 600 generations: the
ARCH46 §1 critic read is at 500/600, and arch46 ran at `9b81614`, before the
§1-§3 build, so none of the ARCH46 reads has been taken yet.
