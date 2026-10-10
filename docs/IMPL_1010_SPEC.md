# IMPL_1010 — implementation spec for ROADMAP 2026-10-10 "an evaluation that never leaves the device" (option A) and "the order of the loop" (P1-P3, P5-P6), with three open reads

Written 2026-10-10 against `failure-theory-1010` at 63b7ace. It is a plan, not a measurement: every number below cites the ROADMAP line, the file, or a command whose output is quoted. Line numbers are as of 63b7ace; open the cited lines before editing. The format follows `docs/ARCH51_SPEC.md`.

Decisions this plan implements (ROADMAP, 63b7ace):
- Build A (ROADMAP:3740-3776, "The decision rule, applied"): MuJoCo Warp, one `Model` per body, K draws per body as worlds, the shared policy's control law and the fluid on the device. K is set from the wall budget: K ~ 50 holds today's wall, K = 100 costs 1.44-1.62x (ROADMAP:3753-3770). B is not built (Q8 refuted, ROADMAP:4048-4073). Q3' is not run (ROADMAP:3769).
- A takes a Warp port of `physics/fluid.py`, not Mojo-Warp interop (Q6 outcome, ROADMAP:3843-3863).
- `wp.capture_while` fails in warp-lang 1.18.0 ("Conditional body graph contains an unsupported operation (memory allocation)", ROADMAP:3727-3729). The rollout uses unrolled graphs of 2,000 steps or host-driven per-step graphs at +12 us per step (ROADMAP:3771-3776).
- P1 now, as a correctness fix for N11. P2 and P3 now. P4 is not built. The epoch scheduler (P5, P6, T11, T12) belongs to A (ROADMAP:4356-4366).
- Caveats that ROADMAP:3770-3776 carries into this spec, each a gate below: Q2 ran without contacts or gravity (gates A4a, A4b); the fixtures were 12-dof, 126-panel bodies, and a rotor body costs 1.6x per world (gate D-A2); T9 owns the cost of contacts (gate A4b).

Conventions for every lane (as in ARCH51_SPEC):
- One git worktree per lane: `git worktree add ../i10-<lane> -b i10-<lane> failure-theory-1010`. A dependent lane branches from the merged predecessor.
- One-function runner. Expected last line: `FAILURES []`.
  `PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('t','tests/<suite>.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.<fn>(); print('FAILURES', m.FAILURES)"`
- Mutation check: `PYTHONPATH=. .venv/bin/python tools/mutate.py --only <id>`. The expected status is `caught` by the named function. `MISAPPLIED`, `SURVIVED` and, after M1, `NOT RUN` all fail the lane.
- Index after any source edit: `tools/index_gen.py write`, then `tools/index_gen.py check` (must exit 0), then `tests/test_index.py`. A new `SearchConfig` field needs a row in `docs/HYPERPARAMETERS.md`, because `test_every_search_config_field_has_a_hyperparameter_row` (test_index.py:168) fails without one. A new package needs a `docs/index/FEATURES.yaml` entry, because test_index fails on any package that no feature names.
- Cheap suites: `for s in index architecture domain application search_adapter adapters worker; do PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1; done`. Every line must be its suite's success string. Predicted under 2 minutes; timeout 300 s.
- New test functions go immediately after the named anchor function, and into `main()`'s `run_all([...])` list right after the anchor's entry. New mutations go into `MUTATIONS` right after the named anchor mutation.
- Mutation `find` strings below are exact lines the lane must write. Keep the code text byte-identical, or update the mutation in the same commit.
- GPU gates run one GPU process at a time. They are never run while a search run maps `mojo/build` (`systemctl --user list-units --state=running`, and `lsof mojo/build/*.so`). Long jobs are launched with `systemd-run --user`, with a predicted runtime, a timeout and a log.
- The user is the sole author: no attribution lines in commits, docs or file headers.

---

## §0. Where the ROADMAP's theory and the code disagree (found while writing this plan)

Each item quotes both sides. The lanes below are written against the code.

**F1. P3 is not the move `MERGE_FIRST_REFINE` made.**
- ROADMAP:4134-4139: "`MERGE_FIRST_REFINE` already proves the move: a machine's score does not depend on what shares its batch (`test_search`), so one trip of 40 worlds returns exactly what two of 20 did."
- Code: the re-score uses the bases that the main call identified. `batchroll.py:1051` has `ctrls[i].bases = results[i].mobility`, and the pool copies them back at `actors.py:384` (`local.bases = remote.bases`). Sampling is decided for the whole call: `batchroll.py:1070` has `if shared is None or buffer is None:` in `_noise`, and `batchroll.py:791` has `deterministic=collector is None,`. `MERGE_FIRST_REFINE` merges two halves that both run with `identify_axes=False` (loop.py:789-797).
- Consequence: P3 needs a sampling flag per machine and a basis hand-off inside the call, and a machine and its twin must share one shard. Lane P3 is written for this. It is a cross-layer change (batchroll, ppo, actors, loop), not "hours".

**F2. P2's acceptance read cannot be computed from the promote events.**
- ROADMAP:4210-4212: "The acceptance rate of a refined controller over its parent (promotions whose refined `tier1_fraction` beats the unrefined, from the `promote` events) within +-10 points of arch50's."
- Code: the event's `tier1_fraction` is `cheap = float(elite.meta.get("mission_fraction", 0.0))`, the unrefined Tier-1 score. `_refined_controllers_for` passes no `log` to `_refine_controllers` (loop.py:2756-2759). No refined score and no acceptance is recorded anywhere, so arch50's acceptance rate does not exist. Lane P2 adds the fields. Its baseline is measured with the serial mode (D-P2).

**F3. Writing fluid forces from `callback.control` changes the order of the step.**
- ROADMAP:3880-3888: the callback "runs after `fwd_position` and `fwd_velocity` and before `fwd_actuation` ... so a force written there from the callback takes effect".
- Code today: `step_batch` (batchroll.py:604-625) sets `ctrl` and `finish()` writes `xfrc_applied` and the added mass into `model.body_mass` and `body_inertia` (batchroll.py:529-531) *before* `mj_step`. The fluid reads the kinematics the previous `mj_step` left.
- In mujoco_warp 3.15.0, `forward()` runs `fwd_position`, which calls `smooth.crb` (`_src/forward.py:1334`), before the callback (`_src/forward.py:2081`). Added mass written from the callback reaches the mass matrix one step late, and the callback reads kinematics at q(t+1) where today's fluid reads q(t).
- The plan: launch the control kernel and the fluid kernels *between* `mjw.step` calls, in the same graph. This reproduces today's order exactly, and no callback is needed. The 0.09-0.12 ms that Q2 measured for the callback (ROADMAP:3735) is the cost estimate for these launches.

**F4. Q1 and Q2 do not cost the host half of the fluid step.**
- ROADMAP:3740-3751: the table costs "fluid as a kernel" from Q1, which timed `FullPipeline.step` only.
- Code: `BatchedFluid.finish` (batchroll.py:452-571) runs per machine on the host:
  - slam mass;
  - `ImplicitAeroDamping.apply`, which projects strip damping through Jacobians into `dof_damping` and `qfrc_applied` (fluid.py:719-752);
  - `InducedFlow`;
  - the `finish_bodies` limiter;
  - entrainment;
  - jets;
  - the added mass per machine;
  - `RotorBatch`.
- Also `BatchedPower` (batchroll.py:626-628) and the CPG and servo commands.
- None of this is in Q1 or Q2. All of it must run on the device if the rollout never returns to the host. The ROADMAP table is a lower bound (lanes A1b, D-A2).

**F5. MJWarp is float32; the reference paths are float64.**
- mujoco_warp 3.15.0 `Data` annotations, read in `../mjwarp-venv`: `qpos array float32`, `xpos vec3f`, `xfrc_applied spatial_vectorf`. The Mojo kernel is "float64 throughout, to match numpy exactly during validation" (mojo/src/fluid_gpu.mojo:20).
- The two paths agree today to `w < 1e-5` (test_search.py, inside `test_the_two_evaluation_paths_score_the_same_machine_the_same`, 4334ff). N9 measured float32 qpos within 1.1e-5 of the CPU after only 300 steps (ROADMAP:3415).
- So a third path cannot match per machine over a 2,000-step segment. A5 is statistical, with tolerances set from a measured null spread.
- The film rule follows from this ("a film is evidence only if it reproduces the score printed on it", CLAUDE.md; `TOLERANCE = 0.02`, viz/film.py:53). A CPU film of a device score will not reproduce it. A6 records the device trajectory of the median draw and films that trajectory.

**F6. A brings a new MuJoCo version.**
- The main `.venv` has mujoco 3.11.0 (`mujoco.__version__`). mujoco-warp 3.15.0 declares `Requires-Dist: mujoco>=3.12.0` and ran with mujoco 3.15.0 (ROADMAP:3401).
- Installing A into `.venv` therefore upgrades the reference engine of every path. That is a comparability boundary of its own, and it is measured before anything depends on it (D-M0). The install must not happen while any run imports `.venv`.

**F7. Identification is 96 rollouts per body, and they run before the segments.**
- ROADMAP:3644: "identification is 24 probes per domain per body". ROADMAP:3742-3743: "identification probes are extra worlds and do not add steps".
- Code: `identify_batch` runs each probe at both signs (`for p in range(n_probes): for sign in (1.0, -1.0):`, batchroll.py:697-698), 300 steps each at dt 0.004. That is 28,800 steps per body (ROADMAP:4100, 460,800 / 16).
- On the device this is 96 worlds per body. They must finish before the policy can drive, because the basis is an input to it. So they add 300 sequential steps (+1.9% of 16,000) for a body that needs identification.

**F8. The segments need not run one after another, which changes the K table and T11's premise.**
- ROADMAP:3740-3743 costs 16,000 sequential steps per body. T11 (ROADMAP:4153-4166) says the device "requires a loop that *has* hundreds to thousands of worlds ready at once: all eight islands bred in one epoch".
- Code: each of the six segment halves (`PAIRED_MEDIA = ("water", "land", "air")`, tasks.py:262) and each of the four transitions starts from `reset` plus `scatter` (batchroll.py:1096-1104). The pair partner is read only in scoring (triphibian.py:2191, 2240).
- On the device these are 10 independent worlds per draw. A rollout is about 2,000 steps (one 8 s segment at 0.004 s) instead of 16,000, with W = 10 x K worlds per body. Then W >= W* (~80, ROADMAP:3760) for K >= 8 without any epoch.
- The epoch is therefore not a prerequisite. A6 builds it only if D-A5 shows that more concurrent bodies pay.
- The cost: batchroll carries `env.rng` across segments (the rewind at batchroll.py:1094 and 1124-1128). A parallel layout needs its own stream per (seed, segment). That makes it a device-path definition of a draw, and part of the A7 boundary.

**F9. `mutate.py` already exits 1 on MISAPPLIED; nothing runs it.**
- Code: `bad = [r for r in results if r.status in ("MISAPPLIED", "ERROR", "TIMEOUT")]` and `return 1 if bad else 0` (mutate.py:1966, 1988).
- A static count at 63b7ace (my command: import `tools/mutate.py`, count each `find` in its file) finds 7 of 195 mutations MISAPPLIED, and no suite notices: `seeds-scored-without-the-shared-policy` (appears 2 times), `turn-ignores-airborne` (0), `land-stop-adds-to-the-walk` (0), `evaluators-ask-different-tasks` (0), `single-path-crossing-unscattered` (0), `batched-fluid-diagnostics-stay-default` (0), `tier2-probe-legs-enter-the-mission` (0). Every named test function exists.
- A second hole: a mutation whose named function skips exits 0 and reads `SURVIVED`. The A tests will skip unless `mujoco_warp` is importable from `.venv/bin/python`, which is the interpreter `mutate.py:59` uses.

**F10. `Py_NewRef`: the cause is very likely PATH-dependent Python discovery, inherited through the C environment.**
- ROADMAP:4034-4039 calls it undiagnosed, with "depends on how the process is launched". I ran an import-only probe (no GPU construction), reading the C environment with `libc.getenv` before and after `import full_pipeline` in `.venv/bin/python`:
  ```
  before [None, None, b'.']
  after  [b'/home/hundo/.pyenv/shims/python3', b'/home/hundo/micromamba/lib/libpython3.9.so.1.0', b':.']
  os.environ sees [None, None, '.']
  ```
  The keys are `PYTHONEXECUTABLE`, `MOJO_PYTHON_LIBRARY`, `PYTHONPATH`. With `.venv/bin` first on PATH, the same import sets `MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0`. With `PATH=/usr/bin:/bin` it sets the same 3.12 library.
- `Py_NewRef` exists only from CPython 3.10. A child inherits the C environment:
  - `usable()`'s probe runs `subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout)` with no `env` (batchroll.py:143-144);
  - spawn workers start after `run_search` imports batchroll (loop.py:1613, then `ActorPool(` at loop.py:1639).
- `child_env()` (ops/run.py:338-353) covers this mechanism only for post-run.
- Lane D1 confirms this with GPU construction and proposes no fix. The fix choice goes to §D.
- Measurement (a) in ROADMAP:3484-3491 ("every seed plan scores exactly 0") was taken in this broken configuration. The parent re-ran every batch, so the scores should stand; D1 re-checks one cell.

**F11. After P1, a median swap would blank four identification keys on the elite.**
- `_meta` reads `mobility_rank`, `mobility_cond`, `mobility_underdetermined` and `mobility_axes` from `result.mobility` only (loop.py:1044-1048). That dict is empty when the call did not identify.
- `_reevaluate` copies every `SCORED_KEYS` key into the draw (loop.py:1381-1382). `mobility_basis` reads `ctrl.bases` first (loop.py:1067-1074) and is safe. P1 adds the fix.

**F12. Where draws are needed, and where the device path is hard, are different media.**
- Q5: K for reliability 0.9 is air 66, water 36, land 18 (ROADMAP:3830).
- Land and the crossings are the contact physics (T9). They are also per-elite hypersensitive: the Q4 null control (`solref[0]` x1.01) moved 20 of 58 elites by more than 50% (ROADMAP:3800-3803).
- A's payoff is in air and water, which touch nothing. A4 is split accordingly. Land and the crossings stay on the CPU batched path until the user decides (§D, question 1).

**F13. T12's learner rule is already today's behaviour.**
- ROADMAP:4180-4183: "the epoch keeps PPO's gradient steps per transition where they are (130 per 25k today -> ~1,040 per 200k epoch, in minibatches)".
- Code: `ppo_update(..., epochs: int = 10, minibatch: int = 2048, ...)` (ppo.py:547-548). Gradient steps scale with the buffer: 25,473 / 2,048 x 10 = 124, against the 130 in the ROADMAP. Holding `shared_minibatch` and the epochs fixed keeps steps per transition fixed. A6 adds only a test that nothing rescales them.

---

## §1. Lane order and dependencies

Ranked by cost first, then speed, then learning (memory: "rank work: cost, then speed, then learning").

| # | lane | what | cost | depends on | parallel with | tier |
|---|---|---|---|---|---|---|
| 1 | M1 mutate-check | static check of every mutation, a `NOT RUN` status, re-target the 7 MISAPPLIED | hours, CPU (a few GPU one-function checks for the re-targets) | none | 2-5 | sonnet |
| 2 | D1 pynewref | diagnose the worker `Py_NewRef` abort (F10); no fix | 1-2 h, one small GPU process | none | 1, 3-5 | sonnet |
| 3 | P1 stored-basis | reevals at the stored basis; keep the identification record | hours | none | 1, 2, 4, 5 | sonnet |
| 4 | P2 one-step-es | promotion refinement as one (1+6) step in one batch, plus acceptance telemetry | hours | none | 1-3, 5 | sonnet |
| 5 | R1 q4-repro | read: why the single path reproduced 31/58 land scores | 1-2 h GPU | none | 1-4 | sonnet |
| 6 | ~~P3 one-trip~~ | **withdrawn (§D answer 5)**: A6 runs both kinds of draw in one launch | none | none | none | none |
| 7 | A0 device-dep | optional dependency, guarded import, skip markers, new suite skeleton | hours | D-M0 read before any install into `.venv` | 1-6 | sonnet |
| 8 | A1a fluid-kernels | Warp port of the panel pipeline (the Mojo kernel's scope) | days | A0 | A2 | opus |
| 9 | A1b fluid-scatter | Warp port of `finish()`, jets, rotors, power | days | A1a | A2 | opus |
| 10 | A2 model-build | per-body `Model`, per-world mass, inertia and damping, world layout, graph strategy | days | A0 | A1a, A1b | opus |
| 11 | A3 control | observation, policy, basis, CPG and servo on the device | days | A1a, A1b, A2 | none | opus |
| 12 | A4a air-water | identification, air and water segments with gravity, divergence counter, still machine | days | A3 | none | opus |
| 13 | A5 third-path | agreement test against both evaluation paths (air and water) | 1-2 days | A4a | none | opus |
| 14 | A6 k-draw scoring | K draws, standard error, pooled-mean placement with re-evaluated incumbents (§D answer 3), median-draw film, epoch only if D-A5 says so | days | A5, P1 | none | opus |
| 15 | A4b land-crossings | contacts, land and four transitions on the device | days | A4a, A6 (the user said yes, §D answer 1) | none | opus |
| 16 | A7 boundary-docs | ROADMAP, CLAUDE.md, FEATURES, HYPERPARAMETERS | hours | A6 (and A4b if built) | none | sonnet |

Merge order: M1 first, so every later mutation is checked statically. D1, P1, P2 and R1 merge in any order. P3 is withdrawn. The A lanes merge in the order of the table. Conflicts in `tools/mutate.py`, `tests/test_search.py` and `docs/index/*.md` are trivial when the anchors are kept; re-run `index_gen write` after each merge.

---

## §2. Today's path

### M1 — mutate-check (tools/mutate.py; tests/test_index.py). No dependencies.

Goal: no mutation can be misapplied, name a missing test, or be skipped, without a cheap suite failing.

`tools/mutate.py`, after `apply_mutation` (1823-1833), add exactly:
```python
def static_problems(mutations=None, root: Path = ROOT) -> list:
    """(id, problem) for every mutation that cannot apply exactly once or names no test."""
    import ast
    out, defs = [], {}
    for m in (MUTATIONS if mutations is None else mutations):
        p = root / m.path
        n = p.read_text(encoding="utf-8").count(m.find) if p.exists() else -1
        if n != 1:
            out.append((m.id, f"the target text appears {n} times in {m.path}, expected 1"))
        for suite in m.suites:
            module, _, func = suite.partition("::")
            f = root / "tests" / f"{module}.py"
            if module not in defs:
                defs[module] = ({d.name for d in ast.parse(f.read_text(encoding="utf-8")).body
                                 if isinstance(d, ast.FunctionDef)} if f.exists() else None)
            if defs[module] is None or (func and func not in defs[module]):
                out.append((m.id, f"names {suite}, which does not exist"))
    return out


def suite_status(rc: int, out: str) -> str:
    """What one suite run says about a mutant: caught, passed, NOT RUN or TIMEOUT."""
    if rc == 124 and out.startswith("timed out"):
        return "TIMEOUT"
    if rc == 3:
        return "NOT RUN"
    return "caught" if rc != 0 else "passed"
```

Edits:
- `_ONE_FUNCTION` (1839-1859): after `before = len(getattr(mod, "FAILURES", []))` add `skipped0 = len(getattr(mod, "SKIPPED", []))`. Before the final `raise SystemExit(0)`, add:
  ```python
  if len(getattr(mod, "SKIPPED", [])) > skipped0:
      print("[NOT RUN] " + "; ".join(getattr(mod, "SKIPPED", [])[skipped0:]))
      raise SystemExit(3)
  ```
- `evaluate()` (1896-1924): classify each suite result with `suite_status`. `"NOT RUN"` returns `Result(m, "NOT RUN", by=suite, detail=...)`.
- `main()`: add `"NOT RUN": "NOT RUN "` to the mark dict, and `"NOT RUN"` to the `bad` tuple. Add `ap.add_argument("--check", action="store_true", help="static check only: every mutation applies once and names a test")`. When set, print each problem as `  {id}: {problem}`, then the exact line `static check: {len(chosen)} mutations, {len(problems)} problems`, and return 1 if any problem, else 0. It copies no tree.
- Re-target the 7 MISAPPLIED mutations (F9). For each one:
  - find where the guarded behaviour went: `git log -S '<old find text>' --oneline -- <path>`, then read the current code;
  - rewrite `find`/`replace` on the current text so that it models the same defect;
  - confirm `caught` by its named suite (four name `test_search::` GPU functions; run them one at a time);
  - if the behaviour no longer exists, delete the mutation, and give the id and the reason in the commit message and in docs/TEST_AUDIT.md;
  - for `seeds-scored-without-the-shared-policy` (2 occurrences of `            shared=state.shared, pool=state.pool)`), widen the `find` to include the preceding unique line in `seed_archipelago` (loop.py:2625; the call at 2654-2656).

Test `test_every_mutation_applies_exactly_once`, in tests/test_index.py. Anchor: after `test_every_search_config_field_has_a_hyperparameter_row`.
- Load `tools/mutate.py` by path. Register it in `sys.modules` **before** `exec_module`: without that, `@dataclass(frozen=True)` raises `AttributeError: 'NoneType' object has no attribute '__dict__'` (observed while writing this plan).
- (a) `static_problems() == []` at HEAD, with the detail listing the problems.
- (b) `static_problems([Mutation(id="x", path="tools/mutate.py", find="NO-SUCH-TEXT-1010", replace="", defect="", suites=("test_index::test_every_mutation_applies_exactly_once",))])` has exactly one problem, and it contains `appears 0 times`.
- (c) The same with `suites=("test_index::no_such_function_1010",)` and a `find` that occurs once (`"def static_problems("`) has exactly one problem, and it contains `does not exist`.
- (d) `suite_status(3, "") == "NOT RUN"`, `suite_status(1, "") == "caught"`, `suite_status(0, "") == "passed"`.
- `mutate.py` imports only the standard library, so the bare-interpreter CI job (`.github/workflows/checks.yml`, hexagon) can run it.

Mutations (new group `# --- the harness itself (IMPL_1010 M1) ---`). Anchor: `merged-rescore-halves-swapped`. Suite: `("test_index::test_every_mutation_applies_exactly_once",)`.
- `static-check-ignores-missing-text`: find `        if n != 1:\n            out.append((m.id, f"the target text appears` and replace `if n != 1` with `if n > 1`. Breaks (b).
- `static-check-ignores-missing-tests`: find `            if defs[module] is None or (func and func not in defs[module]):` and replace it with `            if defs[module] is None:`. Breaks (c).
- `skipped-function-reads-survived`: find `    if rc == 3:\n        return "NOT RUN"\n` and replace with empty. Breaks (d).

Runners:
- `tests/test_index.py | tail -1` gives `all index checks passed`. Predicted 10 s; timeout 120 s.
- `tools/mutate.py --check | tail -1` gives `static check: 198 mutations, 0 problems` (195 + 3 new, minus any deleted). Predicted 2 s.
- Each new mutation: about 60 s; timeout 300 s.

Done: the three new mutations are `caught`; the 7 re-targeted ones are `caught` or deleted with a reason; `--check` exits 0; cheap suites pass. docs/TEST_AUDIT.md gets one paragraph: the 7 ids, their new state, and the `NOT RUN` status. CLAUDE.md "Whether a test is worth anything" gets one line: `tools/mutate.py --check` runs inside test_index.

### D1 — pynewref, diagnosis only (experiments/pynewref/run.py, new). No dependencies. No production change.

Goal: confirm or refute F10 with GPU construction, and name the configurations that fail.

Frozen hypotheses:
- H1: importing `full_pipeline` sets `MOJO_PYTHON_LIBRARY` in the C environment to the libpython of the first `python3` on PATH.
- H2: a child process started after that import, with the C environment inherited, loads that library. When it is older than 3.10, constructing `FullPipeline` aborts with `symbol not found: Py_NewRef`.

Arms. Each runs in a fresh interpreter launched with `PATH=$HOME/.pyenv/shims:$PATH`, which gives 3.9.23 here (`python3 --version`), and again with `.venv/bin` first. "Construct" means `FullPipeline(64, 64, 4)` with `sys.path` holding `mojo/build`.

| arm | parent imports `full_pipeline` | child | child env | predicted (pyenv PATH) | predicted (venv PATH) |
|---|---|---|---|---|---|
| a | yes | `subprocess.run([sys.executable, "-c", construct])` | inherited (no `env=`) | ABORT Py_NewRef | ok |
| b | yes | same | `env=dict(os.environ)` (what `child_env` does) | ok | ok |
| c | yes, then spawn | `multiprocessing` spawn child constructs | inherited | ABORT | ok |
| d | spawn first, then import | spawn child constructs | inherited | ok | ok |
| e | `run_search` order: `batchroll` import, `usable()`, `ActorPool(2)` | one 2-shard `evaluate_tier1` of 4 seed plans | inherited | "a worker raised" | no such line |

Falsifiers:
- H2 is refuted if (a) or (c) constructs under the pyenv PATH.
- H1 is refuted if the `libc.getenv` value after import does not follow PATH.
- If H2 is refuted, capture `LD_DEBUG=libs` of the failing child and report it.

Read also:
- Which launcher contexts put pyenv shims first: an interactive shell; `systemd-run --user` (read `systemctl --user show-environment`); the Claude session shell.
- Whether `usable()` passes in a systemd-launched run (arch50's log says it ran on the GPU).

Re-check:
- Re-run one cell of `runs/promo_floor_trial3.log` (the air island, nref 8, 8 s) in the passing configuration.
- Compare the seed elites' air, water and land with the logged zeros.
- Write the result to a new file in the analysis directory, never into `runs/arch*`.

Report: `runs/analysis_1010_failure_theory/D1_result.md` in REPORT_FORMAT shape (the prediction is the table above). Predicted wall 20 min; timeout 1 h; one GPU process at a time.

Fix candidates for §D (not built here):
1. Before the first import, set `os.environ["MOJO_PYTHON_LIBRARY"]` to the running interpreter's libpython, from `sysconfig.get_config_var("LDLIBRARY")` and `LIBDIR`. Setting `os.environ` calls `putenv`, and Mojo honours a preset value (arm a).
2. Pass `env=dict(os.environ)` in `usable()`, and scrub the C keys with `os.unsetenv` before `ActorPool` spawns.

Gate for either: arm (e) prints no "a worker raised" under the pyenv PATH. Add a test in test_search after `test_a_kernel_older_than_its_source_is_not_usable`.

### P1 — stored-basis (evolution/loop.py; tests/test_search.py; tools/mutate.py). No dependencies.

Goal (ROADMAP:4195-4205; ROADMAP:4118-4128): a reeval of an elite that carries `mobility_basis` runs with `identify=False` and that basis. The draw then varies the task only, which is the experiment Q5 measured (`experiments/draw_variance/run.py:243`, `identify_axes=False`).

Reuse: `MobilityBasis.bases_from_record` (control/cpg.py:286), exactly as `_refined_controllers_for` uses it (loop.py:2732). Nothing new is created.

`dytiscidae/evolution/loop.py`:
- `evaluate_candidate` (433-458): add the keyword `bases=None`. Line 449 becomes `    ctrl = Controller(params=None, policy=policy, bases=bases or None)  # params filled by the env`.
- `evaluate_candidates` (556-568): add the keyword `bases=None`. After `seeds = seeds or [0] * k`, add `bases = bases or [None] * k`. Line 638 becomes exactly
  `        ctrls.append(Controller(params=None, policy=policy, bases=bases[i] or None))` (the loop variable is `i in passed`).
  The CPU fallback (629) passes `bases=bases[i]`.
- The reeval build line 1815, currently
  `            built.append((elite.genome.copy(), elite.meta.get("policy"), True, ["reeval"], elite, int(rng.integers(1 << 30))))`,
  becomes:
  ```python
              stored = bool(MobilityBasis.bases_from_record(elite.meta.get("mobility_basis")))
              built.append((elite.genome.copy(), elite.meta.get("policy"), not stored, ["reeval"], elite, int(rng.integers(1 << 30))))
  ```
  The draws from `rng` are unchanged, so `reeval_per_generation == 0` reproduces bit for bit.
- The `evaluate_candidates` call (1826-1834): after `seeds=[b[5] for b in built],`, add:
  ```python
                  bases=[(MobilityBasis.bases_from_record(b[4].meta.get("mobility_basis")) or None)
                         if b[3] == ["reeval"] else None for b in built],
  ```
- `_reevaluate`, after `    meta = _meta(pheno, result, ctrl)` (1358):
  ```python
      # A draw at the stored basis identified nothing, so ``result.mobility`` is
      # empty; keep the elite's own identification record (IMPL_1010 P1, F11).
      if not result.mobility:
          for key in ("mobility_rank", "mobility_cond", "mobility_underdetermined", "mobility_axes"):
              if key in elite.meta:
                  meta[key] = elite.meta[key]
  ```

Comparability: from the P1 merge, a reeval draw (N11) is a task draw at a fixed basis. A buffer that mixes draws from before and after P1 is not comparable with either. No stored run has N11 draws, because arch51 was not launched, so the boundary touches no stored archive. `steps.main.identify` falls by 28,800 per reeval with a stored basis (ROADMAP:4196-4199). No other score changes.

Test `test_a_reevaluation_runs_at_the_stored_basis_cpu`. Anchor: after `test_an_archived_elite_is_re_run_at_a_fresh_seed_and_kept_at_its_median_cpu` (3737).
- Extend `_reeval_stub_search` (3672) with keywords `basis=None, kw_log=None`, keeping the defaults for existing callers.
  - `fake_eval` appends `{"identify": kw.get("identify"), "bases": kw.get("bases")}` to `kw_log`.
  - It returns `mobility={"air": "id"} if <this slot's identify> else {}`.
  - The stub `_meta` returns `"mobility_basis": basis or {}` and `"mobility_rank": {k: 3 for k in r.mobility}`.
- The record `R = {"air": {"modes": [[1.0, 0.0]], "effects": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], "authority": [1.0], "medium": "air"}}`. It must rebuild through `bases_from_record`; adjust the shapes until it does.
- (a) With `basis=R`, `reeval=2`: in every call after the seeds, the slots whose genome came from a reeval have `identify` False and a `bases` dict whose `"air"` modes equal R's. Every child slot has `identify` True and `bases` None.
- (b) With `basis=None`: reeval slots have `identify` True and `bases` None, because an elite without a record is re-identified.
- (c) With `basis=R`: every elite with two or more draws has `meta["mobility_rank"] == {"air": 3}`, and so does every draw's `meta`. Without the fix the reeval draws carry `{}`.

Mutations. Anchor: `reeval-records-nothing`. Suite: `("test_search::test_a_reevaluation_runs_at_the_stored_basis_cpu",)`.
- `reeval-re-identifies`: find `            stored = bool(MobilityBasis.bases_from_record(elite.meta.get("mobility_basis")))\n` and replace it with `            stored = False\n`. Breaks (a).
- `reeval-drops-the-stored-basis`: find `                       if b[3] == ["reeval"] else None for b in built],` and replace it with `                       if False else None for b in built],`. Breaks (a).
- `reeval-blanks-the-identification-record`: find `    if not result.mobility:\n        for key in ("mobility_rank"` and replace it with `    if False:\n        for key in ("mobility_rank"`. Breaks (c).

GPU regression: `test_an_elite_is_re_measured_at_fresh_draws_and_kept_at_its_median` (3859) must stay green. Its check (5) films an elite with draws through the stored basis. Predicted 5 min; timeout 20 min.

Done: the one-function runner gives `FAILURES []` for both functions; the three mutations are `caught`; cheap suites pass; the FEATURES.yaml `ArchiveReevaluation` note gains "reevals run at the stored basis (IMPL_1010 P1)". The R prediction of P1 is D-P1.

### P2 — one-step-es (evolution/loop.py; ops/run.py; adapters/trainers/search.py; docs/HYPERPARAMETERS.md; tests). No dependencies.

Goal (ROADMAP:4206-4213; ROADMAP:4129-4133): a promotion's refinement becomes one (1+6) step. Every trial of every promoted elite, plus the noise-free re-score, goes into one batch: 3 + 18 = 21 worlds, one pool trip. Today it is six trips of up to 3 worlds each, which run in the parent: `plan_shards` gives one shard at 3 machines with `min_shard` 2, and `ActorPool.evaluate_tier1` then calls `evaluate_tier1_batch` directly (actors.py:302-306). The serial mode stays for reproduction and as the baseline arm.

Reuse vs create: `_refine_controllers` (699-875) is the (1+1) loop, and its per-step body cannot hold λ trials per candidate without rewriting its MERGE and funnel branches. A sibling `_refine_population` reuses `batchroll_eval`, `_copy_policy` and `Controller`, and leaves the generation's refinement path (`controller_refine_steps`) byte-identical.

`SearchConfig`, after `    promotion_refine_steps: int = 6` (202):
```python
    #: How promotion spends ``promotion_refine_steps`` trials per elite: True,
    #: one (1+lambda) step, every trial of every promoted elite in one batch
    #: (IMPL_1010 P2); False, the serial (1+1) steps every run to arch50 used.
    promotion_refine_parallel: bool = True
```

New function after `_refine_controllers`:
```python
def _refine_population(phenos, ctrls, results, cfg, *, spec, seed, population,
                       shared=None, pool=None, log=None, cost=None):
    """One (1+lambda) step on every candidate's policy weights, in one batch
    with the noise-free re-score (IMPL_1010 P2).  The best trial replaces the
    controller only if it beats the baseline, as in the (1+1) step."""
    import time as _time

    sigma = float(getattr(cfg, "controller_refine_sigma", 0.1))
    rng = np.random.default_rng(int(seed) ^ 0x9E3779B9)
    k = len(ctrls)
    owners, trials = [], []
    for i, c in enumerate(ctrls):
        w = c.policy.weights if c.policy is not None else None
        if w is None or w.size == 0:
            continue
        for _ in range(int(population)):
            trials.append(Controller(params=c.params, bases=c.bases,
                                     policy=_copy_policy(c.policy, w + rng.normal(0.0, sigma, size=w.shape))))
            owners.append(i)
    nb = k if shared is not None else 0
    t0 = _time.perf_counter()
    with _phase(cost, "evaluate.refine"):
        got = batchroll_eval(list(phenos[:nb]) + [phenos[i] for i in owners],
                             list(ctrls[:nb]) + trials, cfg, spec=spec, seed=int(seed),
                             shared=shared, pool=pool)
    if shared is not None:
        results = list(got[:k])
    base = [float(r.mission_fraction) for r in results]
    best = list(base)
    for j, i in enumerate(owners):
        tr = got[nb + j]
        if float(tr.mission_fraction) > best[i]:
            best[i] = float(tr.mission_fraction)
            ctrls[i].policy = trials[j].policy
            results[i] = tr
    if log is not None:
        log["refine_wall"] = round(_time.perf_counter() - t0, 3)
        log["refine_base"], log["refine_best"] = base, best
    return results
```

Other edits:
- `_refine_controllers`, serial path:
  - after `    best = [float(r.mission_fraction) for r in results]` (829), add `    base = list(best)`;
  - before the final `    return results` (875), add `    if log is not None:\n        log["refine_base"], log["refine_best"] = base, best`.
- `_refined_controllers_for` (2691):
  - signature `(state, elites, phenos, spec, rng, log=None)`;
  - replace the `_refine_controllers(...)` call (2756-2759) with:
    ```python
            sub = [group[i] for i in keep], [ctrls[i] for i in keep], [base[i] for i in keep]
            rlog: dict = {}
            if getattr(cfg, "promotion_refine_parallel", True):
                _refine_population(*sub, cfg, spec=spec, seed=int(rng.integers(1 << 30)),
                                   population=steps, shared=state.shared, pool=state.pool, log=rlog)
            else:
                _refine_controllers(*sub, cfg, spec=spec, seed=int(rng.integers(1 << 30)), steps=steps,
                                    shared=state.shared, pool=state.pool, log=rlog)
            if log is not None:
                log["by_elite"] = {slots[keep[j]]: (rlog["refine_base"][j], rlog["refine_best"][j])
                                   for j in range(len(keep)) if "refine_base" in rlog}
    ```
  The `rng` draw count is unchanged, so the Tier-1.5 and Tier-2 seeds drawn after it are the same in both modes.
- `_verify_and_label`:
  - line 2860 becomes `        rlog: dict = {}` plus `        refined = _refined_controllers_for(state, chosen, phenos, spec, rng, log=rlog)`;
  - the loop at 2883 iterates `enumerate(zip(...))` with index `ki`;
  - the promote event gains `"refine_parallel": bool(getattr(cfg, "promotion_refine_parallel", True)), "refine_base": (rlog.get("by_elite") or {}).get(ki, (None, None))[0], "refine_best": (rlog.get("by_elite") or {}).get(ki, (None, None))[1],` beside `**long_leg, **walls,` (2942).
- `ops/run.py`:
  - after `--promotion-refine-steps` (1060), add `p.add_argument("--promotion-refine-serial", action="store_true", help="the (1+1) promotion refinement of every run to arch50 (IMPL_1010 P2)")`;
  - in `search_config_from_args`, after line 184, add the line `        promotion_refine_parallel=not args.promotion_refine_serial,`.
- `adapters/trainers/search.py:63`: `    "promotion_refine_steps", "promotion_refine_parallel", "mission_weight", "reward_shaping",`.
- `docs/HYPERPARAMETERS.md`, after the `promotion_refine_steps` row (80): `| \`promotion_refine_parallel\` | True | False (to arch50) | typed | IMPL_1010 P2: one (1+6) batch instead of six serial (1+1) trips; read by D-P2 |`.

Comparability: no Tier-1 score definition changes. Which controller Tier-2 sees, and which `policy_promoted` weights children inherit (loop.py, in the build phase: `inherited = (parent.meta.get("policy_promoted") or parent.meta.get("policy"))`), change from the P2 merge. Tier-2 outcomes and lineages are not comparable across it.

Test `test_promotion_refinement_is_one_batch` (CPU). Anchor: after `test_promotion_spends_refinement_and_keeps_what_it_buys` (3441).
- Monkeypatch `loop.batchroll_eval` with a stub that records `len(phenos)` per call and returns `NS(mission_fraction=-abs(float(c.policy.weights.sum()) - 1.0))` for each controller.
- `state = NS(config=SearchConfig(promotion_refine_steps=6), shared=object(), pool=None)`.
- Three fake elites with `genome=None`, `meta={"policy": zeros(n_weights).tolist(), "mobility_basis": R}`, where R is P1's record. Phenos are `NS()`; `_controller_for` does not read them (loop.py:417-431).
- (a) `_refined_controllers_for(state, elites, phenos, None, np.random.default_rng(0), log=lg)` makes exactly one stub call, of 21 machines.
- (b) For each elite, the returned policy's score equals `max(baseline, best of its six trials)` recomputed from the stub's record, and `lg["by_elite"]` has three entries with `best >= base`.
- (c) With `promotion_refine_parallel=False`, the calls are `[6, 3, 3, 3, 3, 3]`: the first merges the re-score with step one.

Existing tests: `test_promotion_spends_refinement_and_keeps_what_it_buys` (GPU) must stay green. `test_physics::test_the_search_cli_defaults_are_the_stored_run_configuration` (194) gains `check("promotion refinement is one batch (IMPL_1010 P2)", a.promotion_refine_serial is False and search_config_from_args(a, True).promotion_refine_parallel is True)`. `test_search_adapter::test_the_plan_translates_into_a_search_config` (214) gains a plan with `promotion_refine_parallel=False` that gives `cfg.promotion_refine_parallel is False`.

Mutations. Anchor: `promotion-re-identifies`.
- `promotion-es-keeps-the-worse-trial`: find `        if float(tr.mission_fraction) > best[i]:` and replace `>` with `<`. Suite `("test_search::test_promotion_refinement_is_one_batch",)`. Breaks (b).
- `promotion-es-serial-again`: find `    promotion_refine_parallel: bool = True` and replace with `    promotion_refine_parallel: bool = False`. Same suite, breaks (a).
- `promotion-es-trials-on-one-elite`: find `            owners.append(i)` and replace with `            owners.append(0)`. Same suite, breaks (b).
- `cli-promotion-refine-serial-default`: find `        promotion_refine_parallel=not args.promotion_refine_serial,` and replace with `        promotion_refine_parallel=False,`. Suite `("test_physics::test_the_search_cli_defaults_are_the_stored_run_configuration",)`.
- `job-path-refuses-promotion-parallel`: find `    "promotion_refine_steps", "promotion_refine_parallel", "mission_weight", "reward_shaping",` and replace with `    "promotion_refine_steps", "mission_weight", "reward_shaping",`. Suite `("test_search_adapter::test_the_plan_translates_into_a_search_config",)`.

Runners: CPU function about 5 s; GPU regression about 10 min (timeout 30 min); mutations about 60 s each. Done: `FAILURES []`, five mutations `caught`, cheap suites and `test_physics` pass. The P2 predictions are D-P2.

### R1 — q4-repro, a read (experiments/contact_model_dependence/run.py). No dependencies.

Question: the single path reproduced the recorded land score within 0.005 for 31 of 58 elites. The batched re-score in `experiments/no_model_gate` reproduces 195 of 229 (ROADMAP:3807-3810), and the airpair rerun gives land 195/229 (ROADMAP:4754-4755).

Method:
- Add `--path {single,batched}` to `run.py`. `single` is today's (`ev.evaluate_tier1(p, controller=ctrl, segment_seconds=seg, seed=seed, identify_axes=False)`, run.py:144-145).
- `batched` scores the same 58 at their `eval_seed` through `batchroll.evaluate_tier1_batch(..., identify_axes=False, seed=<per-elite list>)`, with the same control laws (`viz.film.control_laws`, run.py:96 and 136).
- Run only the hard-contact arm (no solref change).

Frozen prediction: batched reproduces >= 49/58 (85%, the no_model_gate rate) and single reproduces 31/58 again.
- If both hold, the two-path agreement test misses a land case. Report per elite which recorded quantity differs (land competence, `eval_seed`, basis media, shared network generation), for the first five that disagree.
- Falsifier: batched <= 35/58. The artifact is then in the script's controller reconstruction, and the paths are not implicated.

Report: `runs/analysis_1010_failure_theory/R1_result.md`. Predicted 1 h GPU (58 elites, 8 s segments); timeout 3 h; `systemd-run --user --unit r1-q4repro`.

### P3 — one-trip (envs/batchroll.py; learning/ppo.py; envs/actors.py; evolution/loop.py; tests). Depends on P1. Tier: opus.

**Withdrawn 2026-10-10 (§D answer 5). Do not build. Kept for the record.**

Goal (ROADMAP:4214-4217): the main sampled pass and the noise-free re-score run in one `evaluate_tier1_batch` call per generation, and the placed scores are bit-identical. F1 explains why this is cross-layer.

Design:
- `evaluate_tier1_batch(..., sample=None, basis_from=None)` (batchroll.py:920).
  - `sample` is one bool per phenotype. The default, None, means "every machine samples when `shared` and `buffer` are given", which is today's behaviour.
  - `basis_from` holds one `int | None` per phenotype. After identification (after batchroll.py:1051) add exactly:
    ```python
        for i in live:
            if basis_from is not None and basis_from[i] is not None:
                ctrls[i].bases = ctrls[basis_from[i]].bases
    ```
  - `_noise` (1067-1073) returns, per live machine, `np.random.default_rng([seeds[i] & 0x7FFFFFFF, streams[i], tag]) if want[i] else None`, where `want = [True]*k if sample is None else list(sample)`.
- `rollout_batch` and `run_transition_batch` take a per-row `deterministic` mask: a machine is deterministic when its noise rng is None. `SharedPolicy.act_many` (ppo.py:258) accepts `deterministic` as a bool or a per-row sequence. A deterministic row takes the squashed mean, as today's `deterministic=True` does.
- The collector records sampled rows only. `SegmentCollector.finish` banks no trajectory for a machine that recorded nothing. Assert that the buffer's trajectory count equals the sampled machines times the halves and transitions.
- `ActorPool.evaluate_tier1` (actors.py:284): with `basis_from` given, plan the shards over the main machines and append each twin to its main's shard. Remap `basis_from` to shard-local indices, and slice `sample` as `identify_axes` is sliced (actors.py:343-344).
- `SearchConfig.merge_main_rescore: bool = True`. Add a HYPERPARAMETERS row, a CLI flag `--no-merge-main-rescore`, and a `_CONFIG_FIELDS` entry.
- `evaluate_candidates` merges when `merge_main_rescore and shared is not None and buffer is not None and controller_refine_steps == 0 and placement_draws == 1 and select is None`:
  - one `_batched` call over `phenos + phenos`;
  - controllers `ctrls + [Controller(params=None, policy=c.policy, bases=None) for c in ctrls]`;
  - `identify_axes = wants + [False]*k`, `sample = [True]*k + [False]*k`, `basis_from = [None]*k + list(range(k))`, `seed = draw + draw` (per candidate);
  - streams default `range(2k)`, so the main machines keep 0..k-1;
  - the placed results are `got[k:]`; `_refine_controllers` is called with a new keyword `rescored=True`, which skips its re-score block (788-811);
  - log `main_wall` (the whole call), `rescore_wall = 0.0`, `merged_main_rescore = True`; `_stage_event` adds `merged_main_rescore`.

Test `test_merging_main_and_rescore_changes_nothing` (GPU, `needs_batched_evaluator`). Anchor: after `test_merging_the_rescore_with_the_first_refinement_changes_nothing` (4211).
- 4 seed plans, a shared policy, a buffer, per-candidate seeds. Arm A has the merge off, arm B has it on, each through `pool=None` and through `ActorPool(2, min_shard=2)`.
- Checks: placed results are equal (`==`) in every medium's competence, `mission_fraction`, `eval_seed` and every transition's `crossed`. Buffer trajectory counts are equal, and `np.array_equal` holds on the stacked observations and actions. Each controller's bases equal modes for modes.

Mutations. Anchor: `merged-rescore-halves-swapped`. Suite: the test above.
- `merged-twin-keeps-no-basis`: find `                ctrls[i].bases = ctrls[basis_from[i]].bases` and replace with `                ctrls[i].bases = None`.
- `merged-twin-samples`: find `if want[i] else None` and replace with `if True else None`.
- `merged-pair-split-across-shards`: the lane writes the line that appends a twin to its main's shard and registers a mutation that appends it to shard 0. Caught by the pool arm.
- `merge-main-rescore-default-off`: find `    merge_main_rescore: bool = True` and replace with `    merge_main_rescore: bool = False`. Caught by a CPU check: count `evaluate_candidates`' `_batched` calls on a stub.

Also run `test_sharding_a_generation_does_not_change_a_score` (5799), `test_the_two_evaluation_paths_score_the_same_machine_the_same` and `test_a_film_reproduces_the_scored_experiment`. Predicted 30 min GPU in total; timeout 90 min. The wall prediction is D-P3.

Note for §D: A retires the two-trip structure. If A6 lands within weeks, P3's 3-8% applies only to runs before it.

---

## §3. Option A, staged

### A-wide layout (decided here; each lane implements its part)

Package `dytiscidae/envs/devroll/` (name parallel to `batchroll`; nothing named device, devroll or warp exists in `docs/index/SYMBOLS.md` at 63b7ace). It sits outside the hexagon. `test_architecture.py` forbids heavy imports only in domain, ports and application.

| module | lane | content |
|---|---|---|
| `__init__.py` | A0 | `AVAILABLE`, `UNAVAILABLE_REASON`, `usable()`, version pins |
| `fluid_wp.py` | A1a | panel kinematics, medium, coefficients, unsteady state, assembly to bodies (the scope of `mojo/src/full_pipeline.mojo`, "Nine kernels", full_pipeline.mojo:1-20) |
| `scatter_wp.py` | A1b | `finish()` terms, jets, rotors, power |
| `model_wp.py` | A2 | `Model` and `Data` per body, world layout, graphs |
| `control_wp.py` | A3 | observation, actor MLP, per-design `Policy`, basis maps, CPG, servo, gait gain |
| `rollout_wp.py` | A4 | `evaluate_tier1_device(phenos, *, controllers, seeds, k_mean, k_sampled, spec, segment_seconds, shared, buffer, identify_axes)` returning one result per body, with `draws` |

New suite `tests/test_device.py`, with `main()`, `check`, `skip` and `run_all` in the pattern of test_search.py:55-145. It is separate from `test_search.py`: that suite takes ~25 min and runs on the canary machine, while this one needs MJWarp, and a missing dependency must read `[skip]` (CLAUDE.md "Skips"). The three helpers are copied, as test_physics and test_ppo already do. Success string: `all device-path checks passed`, with `, N skipped` appended on a run with skips.

World layout for one body (F8):

| index | content | worlds | steps |
|---|---|---|---|
| identification (only when the body needs it) | 24 probes x 2 signs x {air, water}, as `identify_batch` (batchroll.py:650-723) | 96 | 300 |
| segments | for each draw d < K_m + K_s: air, air mirror, water, water opposite, land, land opposite, 4 transitions | 10 x (K_m + K_s) (4 x ... until A4b) | max segment length, 2,000 at 8 s |

- K_m draws run at the policy mean; their mean is the score. K_s draws sample and feed PPO.
- Today's equivalent is K_m = 1, K_s = 1: the rescore and the main pass.
- Draw seed: draw 0 is `eval_seed`; draw d >= 1 is `placement_draw_seed(eval_seed, d)` (loop.py:879, reused). Segment s's scatter and task come from `_scatter_seed(draw_seed, dom)` and `task_seed` (batchroll.py:1086-1088, reused). The initial state is built on the host by a CPU `TriphibianEnv` (`reset` plus `scatter`) and uploaded.

Launch mapping:
- Heterogeneous bodies are separate `Model`s, so there is one graph per body (ROADMAP:3904-3909: MJWarp batches worlds of one model).
- The bodies of a generation are launched round-robin on S CUDA streams from one host thread.
- W per launch is 10 x (K_m + K_s), which reaches W* ~ 80 at K_m + K_s >= 8. Below that, stream overlap is the only lever (D-A5).

Graph strategy:
- Default: host-driven per-step graphs, one with a control decision and one without (`control_every` = 10 at 25 Hz and dt 0.004, batchroll.py:756). The cost is +12 us per step (ROADMAP:3775), i.e. 2,300 x 12 us = 28 ms per body per generation.
- Unrolled 2,000-step graphs are used only if D-A4 measures their capture cost below that overhead. A captured graph is per body, so it is re-captured every generation.

Predicted wall per generation (16 bodies, no contacts, median body).
- Model: `steps x (c0 + c1 W)`, with c0 = 0.36 + 0.12 ms and c1 = 0.306 + 0.044 x 126 = 5.85 us per world (Q2 and Q1 fits, ROADMAP:3701-3709, 3731-3735).
- The F4 terms, contacts and rotor bodies are excluded, so these are lower bounds:

| K_m, K_s | sequential, 16,000 steps, W = K_m + K_s | parallel layout, 2,300 steps, W = 10 (K_m + K_s) | vs arch50's 168 s (ROADMAP:4097) |
|---|---|---|---|
| 10, 0 | 136 s | 39 s | 0.23x |
| 10, 10 | 153 s | 61 s | 0.36x |
| 50, 10 | 213 s | 147 s | 0.88x |
| 50, 50 | 273 s | 233 s | 1.39x |
| 100, 10 | 288 s | 254 s | 1.51x |

### A0 — device-dep (requirements-device.txt; dytiscidae/envs/devroll/__init__.py; tests/test_device.py; tests/test_search.py; tools/suite_probe.py; docs). Depends on reading D-M0 before installing into `.venv`. Tier: sonnet.

- `requirements-device.txt` (new):
  ```
  # Optional: the on-device evaluation path (IMPL_1010 A). Needs an NVIDIA GPU with CUDA >= 12.4.
  mujoco-warp==3.15.0
  warp-lang==1.18.0
  mujoco==3.15.0
  ```
  The mujoco pin is the same engine that ran Q2 (ROADMAP:3401). It is installed into `.venv` only after D-M0 is read and only while no run imports `.venv` (F6). The `requirements.txt` header gains one comment line pointing to the file.
- `devroll/__init__.py` follows `batchroll`'s guard (batchroll.py:61-200):
  - `try: import warp, mujoco_warp` sets `AVAILABLE`/`UNAVAILABLE_REASON`; nothing is imported at package import beyond that;
  - `usable()` probes `wp.get_cuda_device_count() > 0` and that `mujoco.__version__ == mujoco_warp.__version__`, in a subprocess started with `env=dict(os.environ)` (F10);
  - the message is exactly `"MJWarp not importable"` or `"MJWarp not usable: <reason>"`.
- Append `"MJWarp not importable"` and `"MJWarp not usable"` to `BLOCKED_MARKERS` in test_search.py (45) and in the new suite.
- `tests/test_device.py` skeleton with one function, `test_the_device_path_reports_why_it_cannot_run`:
  - with the package importable, `AVAILABLE` is a bool and `UNAVAILABLE_REASON` is a str;
  - when `AVAILABLE` is False, `usable()` returns `(False, reason)` with a non-empty reason;
  - the suite prints `[skip]` for every other function.
- `tools/suite_probe.py`: add `test_device` to the suites it probes. The CLAUDE.md "What each suite needs" table gains the row `| test_device.py | GPU, requirements-device.txt; every function skips without them |`.
- FEATURES.yaml: new feature `DeviceEvaluation`, naming `envs.devroll` and `tests.test_device`. It grows with each A lane.
- Mutation `device-path-hides-why`: find the line that sets `UNAVAILABLE_REASON` on import failure and replace it with `UNAVAILABLE_REASON = ""`. Caught by the test above. Anchor: a new group `# --- device path (IMPL_1010 A) ---` after M1's group.

Done: `tests/test_device.py | tail -1` gives `all device-path checks passed` (or `..., N skipped` where MJWarp is absent); cheap suites pass; `index_gen check` exits 0.

### A1a — fluid-kernels (devroll/fluid_wp.py; tests/test_device.py). Depends on A0. Tier: opus.

Goal: the panel pipeline in Warp, for W worlds of one body. Inputs per world: body poses (xpos, xmat), xipos, the 6-D body velocity, and t. Outputs per world: the panel forces assembled to bodies (`xfrc`, `fsum_b`), `m_body`, and the panel diagnostics (`subf`, `q`, `lift`, `drag`, `d_bluff`, `rho`, `pos_w`, `m_add`, `vn`, `force`, `buoy`). These are the fields `finish()` reads at batchroll.py:466-560.

Reuse vs create:
- The numpy `FluidSolver.apply` (fluid.py:1041-) is the reference. `mojo/src/full_pipeline.mojo` and `fluid_gpu.mojo` give the kernel decomposition (panel, medium, coefficients, unsteady state, added mass, bluff, strip, assembly, scatter); follow it.
- A third implementation is created because Q6 closed interop (ROADMAP:3858-3863).
- The kernels are generic in dtype (`wp.float64` for the parity test, `wp.float32` for production, which matches MJWarp's state, F5).
- `PanelSet` arrays are uploaded once per body (`upload_static`'s role, full_pipeline.mojo:340).
- `velocity_kernel`'s host-only `mj_objectVelocity` loop (full_pipeline.mojo:8-10) becomes a kernel over MJWarp's `cvel`, `xipos` and `subtree_com`.

The atan2 caveat:
- numpy folds the incidence with `rev = cos_a < 0.0` and `alpha = np.arctan2(sin_a, np.where(rev, -cos_a, cos_a) + 1e-12)` (fluid.py:1119-1121). The Mojo kernel uses its own `atan2d` (fluid_gpu.mojo:29; ROADMAP:4021-4024: Mojo's libm atan2 has no GPU path). The port uses `wp.atan2`, which is a builtin in both precisions.
- After the fold the second argument is positive, so there is no +-pi cut. The discontinuity is at `cos_a = 0`, where `rev` flips and the lift axis reverses. Rounding near it flips `rev` between precisions.
- The parity test counts panels with `|cos_a| < 1e-6` (float32) or `< 1e-12` (float64) separately, and excludes them from the force tolerance. If such panels exceed 0.1% of panel-steps, the count is a finding to report, not something to tolerate.

Test `test_the_device_fluid_is_the_numpy_fluid` (GPU and warp). Replay, so trajectory chaos does not enter:
- For each of the 7 seed plans (`BODY_PLANS`) and the two fixture elites (below), run 2,000 steps of the CPU `TriphibianEnv` with a fixed CPG gait at seed 0 in air and in water.
- At every step, feed the same state to `FluidSolver.apply` (on a copy of `data`) and to the Warp pipeline at W = 3 (three copies, which must agree with each other exactly).
- Tolerances per step and per machine:
  - float64: `max|F_wp - F_np| <= 1e-9 x (1 + max|F_np|)` on `xfrc`, and `<= 1e-9` relative on `m_body` and `subf`;
  - float32: `<= 2e-4 x max|F_np|` on `xfrc`, `<= 1e-4` relative on `m_body`.
- Report the worst step, body and field in the check's detail, and the near-cut count.
- The stateful parts (Wagner lag in `UnsteadyState.update`, fluid.py:440; the slam backward difference, fluid.py:926-935; the first-call flag) are covered because the replay runs the full 2,000 steps.

Fixture: if `runs/arch49` exists, the lane writes `tests/fixtures/arch49_device_bodies.pkl` with the genomes of `ai248_3991` (median dof, 126 panels; experiments/mjwarp_probe/elite_median_dof.json) and the rotor median elite (`elite_rotor_median_dof.json`), the same selection as `experiments/mjwarp_probe/export_mjcf.py`. Genomes only, so tests run in a fresh container (docs/PORTING.md).

Mutations (device group):
- `device-fluid-drops-the-fold`: the lane writes the fold line as `        alpha = wp.atan2(sin_a, wp.where(rev, -cos_a, cos_a) + eps)`; replace `wp.where(rev, -cos_a, cos_a)` with `wp.abs(cos_a)`. This is the mirror the fold replaced (fluid.py:1110-1118).
- `device-fluid-skips-the-lag`: replace the unsteady-state update call with a pass-through of the quasi-steady incidence.
- `device-fluid-world-offset`: index panel arrays with the panel index alone, not `world * n_panels + p`. Breaks the three-copies equality.

Runner: predicted 10 min; timeout 30 min. Done: both precisions pass on 9 bodies and 2 media; three mutations `caught`.

### A1b — fluid-scatter (devroll/scatter_wp.py; tests/test_device.py). Depends on A1a. Tier: opus.

Goal: everything `BatchedFluid.finish` (batchroll.py:452-571) and `step_batch` (604-628) do between two `mj_step`s, in Warp, as arrays per world:
- `slam_mass` (fluid.py:777);
- `ImplicitAeroDamping`: strip damping (fluid.py:753), its projection through body Jacobians onto `dof_damping` (`projected`, fluid.py:674-718), the `due()` cadence and the wet-set staleness (fluid.py:739-748), and `qfrc_applied = B * qvel` (fluid.py:752);
- `InducedFlow.update`, every `UPDATE_EVERY` steps (fluid.py:460-547);
- `finish_bodies` (fluid.py:791), entrainment (fluid.py:815), jets (`e.jets.apply`), added mass into `body_mass` and `body_inertia` per world (batchroll.py:529-531);
- the diagnostics `clamped`, `mean_submerged`, `added_mass`, `slam` (batchroll.py:532-545);
- `RotorBatch.apply` (physics/rotor.py), `BatchedPower.step` (physics/energy.py), `servo_command` (triphibian.py:1739).

The outputs are arrays; A2 writes them into the MJWarp `Model` and `Data`.

The jointless case: CPU MuJoCo needs `mj_setConst` when a panel body is `body_simple` (fluid.py:900-923; measured +32.20 kg bookkept against +0.00 applied). The lane records whether MJWarp reads `body_mass` for such bodies. mujoco_warp's `crb` builds `cinert` from per-world `body_inertia` (`_src/smooth.py:718`, `body_inertia[worldid % body_inertia.shape[0], bodyid]`). Test case (d) checks it.

Test `test_the_device_scatter_is_the_host_scatter` (GPU and warp). The same replay as A1a, in float64 then float32, with the same tolerances on `xfrc`, `qfrc_applied`, `dof_damping`, `body_mass`, `body_inertia`, the battery energy and `max_overload`.
- (a) A water-heavy plan (`eel`, `ray`).
- (b) The rotor fixture elite.
- (c) A jet plan, if any seed plan carries a jet; otherwise a fixture with one.
- (d) A jointless body: after one step, MJWarp's `qM` changes when `body_mass` changes (the device needs no `setConst`, or the lane adds the equivalent).

Mutations:
- `device-damping-not-compensated`: drop the `qfrc_applied = B * qvel` write.
- `device-added-mass-shared-world`: write `body_mass` at world 0 for every world.
- `device-rotor-skipped`: skip the rotor kernel.

Done: replay parity on 9 bodies; three mutations `caught`; D-A1 (the cost of this step) is queued.

### A2 — model-build (devroll/model_wp.py; tests/test_device.py). Depends on A0; parallel with A1a and A1b. Tier: opus.

Goal: a `Model` and `Data` for one body with W worlds, and the step graphs.
- Model: `xml, names = build_model_xml(p, scene=scene_xml(timestep=0.004))` (as `experiments/mjwarp_probe/export_mjcf.py`), then `mujoco.MjModel.from_xml_string`, then `mjw.put_model(mjm, batch_sizes={"body_mass": W, "body_inertia": W, "dof_damping": W})`. `put_model` accepts `batch_sizes` for fields whose spec starts with `*` (`_src/io.py:288-306`); `body_mass`, `body_inertia` and `dof_damping` are `*` fields (`_src/types.py:1686, 1688, 1721`).
- Margin: N9 found `put_model` rejects the 1 mm `geom margin` (`mjcf.py:193`) under MULTICCD and NATIVECCD (ROADMAP:3416-3419). Q2 zeroed it (`experiments/mjwarp_fused_step/run.py:249`). A2 zeroes it on the device model only, and records `margin_zeroed` on every device result. Air and water are expected to have no contacts (T9); A4a checks it.
- `njmax = 512` (N9: 64 overflowed on landing, ROADMAP:3421). Set `nconmax` explicitly, and read the overflow flags each segment (A4).
- The integrator stays the model's own (`implicitfast`). MJWarp supports it (`_src/types.py:495`).
- Step order inside one graph (F3), per step:
  1. control kernel, on decision steps only;
  2. fluid (A1a);
  3. scatter (A1b), writing `xfrc_applied`, `qfrc_applied`, the per-world `body_mass`, `body_inertia`, `dof_damping`, and `ctrl` via `servo_command`;
  4. `mjw.step`;
  5. the record kernel (A4);
  6. the divergence kernel (A4).
  No `callback.control`.
- Policy weights change every PPO update. Copy them into the captured device arrays in place (`wp.copy`) before the generation's first launch, and never re-capture for a weight change.

Test `test_the_device_model_is_the_cpu_model` (GPU and warp):
- (a) For the 9 bodies with the margin zeroed on both sides, gravity on, no fluid and zero `ctrl`, the device qpos stays within 1e-4 of CPU MuJoCo 3.15 for the first 300 steps (N9 measured 1.1e-5, ROADMAP:3415).
- (b) Writing a different `body_mass` per world gives different `qacc` per world, and world 0 is unchanged by the others' writes.
- (c) Every per-world field has leading dimension W after `put_model`.

Mutations:
- `device-model-shared-mass`: drop `"body_mass": W` from `batch_sizes`. Breaks (b) or (c).
- `device-step-callback-order`: move the fluid launch after `mjw.step`. Breaks A1b's replay step alignment, and A5.

Done: the three checks pass; two mutations `caught`; D-A3 (graph capture cost) and D-A4 (memory per world) are queued.

### A3 — control (devroll/control_wp.py; tests/test_device.py). Depends on A1a, A1b and A2. Tier: opus.

Goal: what `rollout_batch` does at a decision step (batchroll.py:771-860), per world, on the device:
- the observation, exactly `TriphibianEnv.observation` (triphibian.py:1602-1680). It is 33 floats (`OBS_DIM = 25 + MORPHOLOGY_DIM`, triphibian.py:1730; `MORPHOLOGY_DIM = 8`, triphibian.py:486):
  - body twist /5 and /4, clipped at 3;
  - body-frame gravity;
  - `tanh(depth/5)` and `mean_submerged` (from A1b);
  - the domain one-hot;
  - `tanh((depth - target)/3)`;
  - ground contact;
  - battery fraction (from A1b's power);
  - stroke and stroke rate;
  - `task_channels` (six; from a per-world phase table uploaded per segment);
  - the morphology context;
- `observation_finite`, which fails the world as `diverged` (batchroll.py:780-785, 803-809);
- the shared actor MLP 33-64-64-out with tanh (ppo.py:102-136), then the squash. Sampled worlds add noise from `wp.rand_init(seed, world)`; the K_m worlds take the mean;
- the per-design `Policy` (control/cpg.py:640), summed into the mode coefficients;
- `coeffs_for_twist`, `command_params`, `twist_of`, `gait_gain` (batchroll.py:836-850);
- `cpg.command(params, t)` every step, and `servo_command`.

Host recomputation (reuse):
- The device records obs and actions per decision for the sampled worlds.
- The host computes `logp` with `SharedPolicy.log_prob(obs, act)` (ppo.py:200), the value with the critic, and `potential_of` (ppo.py:374), in one batched torch call per segment.
- It feeds `SegmentCollector.record` and `finish` (ppo.py:508-540) unchanged. The critic MLP is not ported.

Test `test_the_device_controller_is_the_host_controller` (GPU and warp). Replay CPU states, as in A1a:
- (a) The observation matches `env.observation(dom)` within 1e-6 absolute (float32) on 2,000 steps, for the 9 bodies, in air and in water, with a task in force.
- (b) At the mean, the commanded joint angles match the CPU loop's `angles` within 1e-5 rad.
- (c) The host-recomputed `logp` of device-sampled actions equals `log_prob` within 1e-5, and the empirical action mean over 4,096 samples is within 3 standard errors of the squashed mean.

Mutations:
- `device-obs-drops-task-channels`: zero the six task channels.
- `device-obs-gravity-world-frame`: use world-frame gravity.
- `device-mean-worlds-sample`: K_m worlds take noise.

Done: three checks pass; three mutations `caught`.

### A4a — air-water rollouts (devroll/rollout_wp.py; tests/test_device.py). Depends on A3. Tier: opus.

Goal: `evaluate_tier1_device` for air and water, with gravity on.
- Identification as 96 worlds (F7). The host draws `deltas` exactly as `identify_batch` (batchroll.py:684-687), and `basis_from_probes` on the host turns the downloaded mean twists into bases (batchroll.py:722-723).
- Then the 4 x (K_m + K_s) segment worlds of the layout. Water and land use the antipodal pair; air uses the mirrored turn (`PAIRED_MEDIA`, tasks.py:262).
- Recording on the device: the per-step lists `rollout_batch` keeps (batchroll.py:874-890: depths, clears, alts, ups, contacts, spins, vzs, gains, xys, slam; the cmds and resp per decision), and the start and end positions. They are downloaded once per segment.
- Scoring on the host: `env._score_segment` of one CPU env per body (batchroll.py:906-911), with `pair_partner` set from the first half (`pair_partner_of`). The scoring code is reused, not reimplemented.

Divergence (CLAUDE.md: "Any new rollout loop must read that counter"):
- MJWarp does not reset on a bad acceleration. No `mj_checkAcc` equivalent was found in `_src/forward.py`; only the overflow warnings at 284-316.
- So the divergence kernel flags a world when any `qacc` is non-finite or above 1e10 in magnitude (MuJoCo's `mjMAXVAL`), or when the root position is non-finite or beyond 400 m (batchroll.py:870).
- It freezes the world, adds 1 to its `bad_qacc`, and sets failure `"unstable"` (batchroll.py:895-899) or `"diverged"`.
- The `nefc > njmax` and contact overflow flags per segment set failure `"contact overflow"`, which is a third failure kind. `diverged_rollouts` and `n_rollouts` are filled as in batchroll.
- Contacts in air and water: count `ncon > 0` per world. Any contact sets `notes += ["contact in a contact-free segment"]` and is reported. A4b owns contacts.

Test `test_the_device_path_fails_what_diverges_and_pays_nothing_still` (GPU and warp):
- (a) Still machine (rules 6 and 7): `TriphibianEnv.held_still_params` for every actuator kind (rotors at zero speed, triphibian.py:1377), on 7 seed plans x 4 seeds. Device air competence is 0 for every one; water competence is at most the batched path's still-machine value + 0.005. Reference: the no_model_gate still machine cleared comp:water 0.012 in 3 of 229 and air in 0 of 229 (ROADMAP:4744-4757).
- (b) A body made to explode (actuator gain x1e6, the fixture of `test_a_nan_observation_fails_the_rollout_not_the_batch`) fails as `"unstable"` or `"diverged"` with `bad_qacc > 0` or a diverged count of 1. The other worlds of that launch score the same as without it (bit-equal in float32).
- (c) A result equals that body's own launch at its seeds: two bodies launched on separate streams give the results each gives alone.
- (d) Identification: the device bases' `control_rank` equals the CPU path's on 7 seed plans, and the modes agree within subspace angle <= 5 deg.

Mutations:
- `device-divergence-ignored`: the divergence kernel never flags.
- `device-still-rotor-spins`: the still machine leaves rotor offsets.
- `device-pair-without-partner`: score the second half with `pair_partner = None`.

Runner: predicted 20 min; timeout 60 min. Done: four checks pass; three mutations `caught`; D-A2 (step cost on real bodies) can run.

### A5 — third-path agreement (tests/test_device.py; FEATURES.yaml). Depends on A4a. Tier: opus.

Goal: the rule "two evaluation paths ... `tests/test_search.py` asserts their agreement" (CLAUDE.md) extended to a third path. Per machine identity is impossible (F5), so the tolerance comes from a measured null spread, as Q4 did (ROADMAP:3800-3806).

Test `test_the_device_path_agrees_with_both_evaluation_paths` (GPU, warp and the Mojo kernel; guarded by `needs_batched_evaluator` and the device marker). Set: 7 seed plans plus the 2 fixture elites x 8 seeds, 8 s segments, a shared policy at a fixed seed.
- Reference R: `evaluate_tier1_batch` at those seeds, `identify_axes=False`, with the device-identified bases handed to it, so identification does not enter.
- Null N: R re-run with each initial qpos perturbed by 1e-6 rad. This is the chaos floor of the reference itself.
- Device D at K_m = 1 (draw 0 = `eval_seed`).
- Per medium (air, water), the checks:
  - (i) `|mean(D) - mean(R)| <= max(0.005, 2 x |mean(N) - mean(R)|)`;
  - (ii) Spearman(D, R) >= Spearman(N, R) - 0.1;
  - (iii) the share within 0.005 of R satisfies D >= N - 10 points;
  - (iv) exactly: the same task drawn (`task_seed`) and the same initial state within 1e-6;
  - (v) the first 50 steps' qpos within 1e-4.
- The detail line reads `f"{medium}: device {mD:.4f} batched {mR:.4f} null {mN:.4f}; spearman {sD:.2f} (null {sN:.2f}); within 0.005 {rD}/{n} (null {rN}/{n})"`.
- Against `evaluate_tier1` (single path), air only: it equals R to 1e-5 already (test_search.py:4384-4385), so the comparison is transitive. A spot check of 2 bodies x 2 seeds asserts it directly.
- A disagreement is reported as a `[FAIL]` with that line. It is never loosened in the test: a change of tolerance needs a ROADMAP entry with the null measurement behind it (CLAUDE.md "Designing a measurement" rule 5).

Mutations:
- `device-task-from-wrong-seed`: segment s's task from `task_seed(draw_seed)` instead of the scatter seed. Breaks (iv).
- `device-half-pair-swapped`: score the mirror half's partner from the wrong world. Breaks (i) or (ii) in air.

Runner: predicted 40 min; timeout 2 h. Done: both media pass; two mutations `caught`; FEATURES.yaml lists `tests.test_device::test_the_device_path_agrees_with_both_evaluation_paths` under `DeviceEvaluation` and under `BatchedEvaluation`.

### A6 — K-draw scoring and the scheduler (evolution/loop.py; envs/devroll/rollout_wp.py; viz/film.py; ops/run.py; adapters; docs/HYPERPARAMETERS.md; tests). Depends on A5 and P1. Tier: opus.

Goal (ROADMAP:3865-3875; P5 and P6, ROADMAP:4225-4242): the search scores through the device path at K_m and K_s.

`SearchConfig` fields, each with a HYPERPARAMETERS row:
- `evaluator: str = "batched"` (`"device"` selects A);
- `k_mean: int = 50`;
- `k_sampled: int = 10`;
- `device_streams: int = 4`, set by D-A5;
- `epoch: bool = False`.

Scoring:
- Each medium's competence is the mean over the K_m worlds.
- Each result carries `draws` with `n`, the per-medium mean, the standard error (`sd / sqrt(K_m)`), the median draw by `draw_key` (loop.py:886-889, reused) and its seed.
- The window gets one entry per candidate (the mean), as today.
- Placement (§D answer 3): `obj[0]` and `fitness` are both from the **pooled mean** of every draw the design has had. No lower bound, no `lb_z`.
- Each elite stores sufficient statistics per medium in `meta["draw_stats"]` (`n`, `sum`, `sum_sq`), not the draws themselves. A candidate enters with its K_m draws.
- N11 is kept and changes meaning under `evaluator == "device"`. A re-evaluation runs K_m more draws at fresh seeds and the stored basis (P1), adds them to the elite's `draw_stats`, and re-places the elite on the new pooled mean through the same insertion path as offspring (Extract-ME). The batched path keeps N11's median-of-draws behaviour unchanged. Defaults stay `reeval_per_generation 4` of 16 (25%, Extract-QD's share) and `reeval_depth 8` (Extract-QD's d = 8).
- M3 (`placement_draws`) refuses to combine with `evaluator == "device"`, with a `ValueError`, as `placement_draws` refuses refinement (loop.py:336-337).

Learner:
- K_s sampled worlds per body feed `RolloutBuffer` through the host recomputation (A3).
- `shared_minibatch` and the epochs are unchanged (F13). Test: on two buffers of 25,000 and 200,000 synthetic transitions, the gradient steps per transition differ by at most 10% (target-kl early stop off).

Film (F5):
- The median draw's world records `qpos` per frame (2,000 x nq floats per segment, downloaded once).
- `viz/film.py` gains `evaluate_on_film(..., trajectory=...)`, which renders the recorded device trajectory by kinematic replay (`mj_forward` per frame on the CPU) and stamps the score recomputed from the recorded arrays.
- The CPU re-run beside it is labelled "reference path".
- `film_manifest.json` records `evaluator: device`.

Epoch:
- `epoch=True` breeds all eight islands, then evaluates 128 bodies over `device_streams` streams, then places island by island in the round-robin order, then runs one PPO update.
- It is built only if D-A5 shows more than 10% wall per body saved going from 16 to 128 concurrent bodies (F8). Otherwise the flag is not added.
- Asynchronous placement is not built (T12, ROADMAP:4184-4193).

Tests:
- `test_device_scores_are_pooled_draw_means` (CPU, stub device evaluator; anchor: after `test_a_reevaluation_runs_at_the_stored_basis_cpu`):
  - (a) `fitness` and `obj[0]` equal the pooled mean recomputed from `draw_stats`;
  - (b) a re-evaluation of an elite adds exactly K_m to its `n` and re-places it on the pooled mean, so an incumbent whose first K_m draws were lucky loses its cell to a challenger once its pooled mean falls below the challenger's;
  - (c) `placement_draws=3` with `evaluator="device"` raises `ValueError`.
- `test_a_device_film_reproduces_its_score` (GPU and warp, in test_device): the stamped score equals the recorded score within `TOLERANCE` (0.02, film.py:53) for 3 bodies.

Mutations:
- `device-reeval-replaces-the-draws`: a re-evaluation overwrites `draw_stats` instead of adding to it.
- `device-places-on-the-first-draws`: `obj[0]` from the candidate's first K_m draws, ignoring later re-evaluations.
- `device-film-replays-the-cpu`: drop `trajectory=`.
- `device-minibatch-scales-with-buffer`.

Run before merging (the P6 read is the first arm, not this lane):
- `test_search.py` in full, detached (25 min), with `evaluator="batched"` by default, to show the default path unchanged;
- the canary;
- `test_device.py` in full.

### A4b — land and crossings on the device (devroll/rollout_wp.py; tests/test_device.py). Depends on A4a and the user's decision (§D, question 1). Tier: opus.

Scope:
- Land halves and the four transitions (`run_transition_batch`, batchroll.py:1196-1347, including `CrossingTracker`'s commanded hold-then-go and directional crossing, ARCH46_SPEC §8).
- Contacts with the beach and the ramp, the margin zeroed (A2).
- The contact constants are varied per draw if the user chooses Q4's suggestion ("the K-draw score ... should vary the contact seed too", ROADMAP:3812-3815).

Gates:
- A5 extended to land and to `crossed` per transition, with the null N defined as `solref[0]` x1.01, Q4's null (ROADMAP:3800-3803; Spearman 0.71 under it).
- The still machine crosses 0 of 7 x 4 in every kind (rule 7, "still machines now cross 0 of 218").
- Zero contact overflows on the fixtures.
- Mutations for the crossing direction and the hold phase, mirroring the existing `CrossingTracker` mutations.

Until A4b passes, `evaluator="device"` scores land and the crossings on the batched path at K = 1. The result records which path scored each medium.

### A7 — boundary-docs (docs/ROADMAP.md; CLAUDE.md; docs/index/FEATURES.yaml; docs/HYPERPARAMETERS.md). Last. Tier: sonnet.

Boundary sentence, for ROADMAP and CLAUDE.md "Comparability boundaries":
> From commit <A6 merge sha>, with `evaluator = "device"` (first device arm onward), every Tier-1 score is the pooled mean over every draw the design has had (`k_mean` at entry, plus `k_mean` per re-evaluation). The draws run in float32 on MuJoCo Warp 3.15 with geom margin 0, the Warp fluid port, the control and fluid kernels between steps, and per-(seed, draw, segment) streams. Nothing is comparable with any earlier run: per-medium competences, `mission_fraction`, coverage, `qd_score`, `fitness`, `objectives`, the Tier-1/Tier-2 gap, PPO `return_by_tag`, wall per generation. Tier-2, audits and the reference film stay on the MuJoCo CPU path. A device film replays the recorded trajectory of the median draw. Land and the crossings are scored on the batched path until A4b's merge, and each result says which path scored it.

Also:
- CLAUDE.md: the suite table row (A0), "Two evaluation paths" becomes three, and the `devroll` kernel facts (the float32 state, the margin, no callback).
- If D-M0 led to the `.venv` upgrade, a separate boundary sentence for mujoco 3.11 to 3.15 on the reference path, with D-M0's numbers.
- ROADMAP: the outcome lines of P1-P3 and D1, and every D-item's result under its item.

---

## §4. Measurements deferred to cheap agents

Each item is frozen: run it as written, report in the shape of `runs/analysis_1010_failure_theory/REPORT_FORMAT.md` to `runs/analysis_1010_failure_theory/IMPL_<id>_result.md`, and return `STATUS`, `FILES`, `OUTCOME: <word>` and the decisive number.
- Launch: `systemd-run --user --unit <id> --same-dir -p MemoryMax=<MB> bash -c '<cmd> > experiments/<dir>/<id>.log 2>&1 < /dev/null'`.
- One GPU process at a time, never while a run maps `mojo/build`.
- Scripts go in `experiments/<item>/run.py`, with `results_<run>.json` beside them.

| id | after | what | predicted wall / timeout | prediction (frozen) | falsified if |
|---|---|---|---|---|---|
| D-M0 | before A0 installs | Reference engine 3.11 vs 3.15. Run `experiments/no_model_gate/run.py`'s reproduction on arch49's 229 elites twice, in `.venv` (3.11) and in a copy of `.venv` with mujoco 3.15. | 2 x 80 min / 4 h | Under 3.15, reproduction within 0.005 is air >= 215, water >= 208, land >= 185 of 229 (3.11: 222 / 215 / 195, ROADMAP:4754-4755). Median abs change <= 0.002 per medium. | Any medium drops by more than 10 elites. Then the upgrade is a physics boundary that needs the user's decision. |
| D-P1 | P1 | P1's R: `experiments/draw_variance/run.py` (229 elites x 6 draws, stored bases), plus an `--identify` arm re-identifying at each draw | 2 x 90 min / 5 h | "R ... at stored bases is below the re-identified value by >= 15% in every medium" (ROADMAP:4201-4203) | That fails in any medium. Also check the step count of one generation with 4 reevals (`steps.main.identify`): exactly 16 x 28,800 = 460,800 at 20 built. |
| D-P2 | P2 | 30-generation trial on the generalist island, `tier2_every=5`, 3 arms of each mode (serial, parallel), same seeds; read `refine_wall` and `refine_best > refine_base` from the promote events | 6 x 40 min / 6 h | "`refine_wall` median 105.0 s -> <= 30 s"; acceptance within +-10 points of the serial arm (ROADMAP:4207-4212, with F2's serial baseline replacing arch50's) | `refine_wall` > 50 s, or acceptance lower by > 10 points |
| ~~D-P3~~ (withdrawn with P3) | P3 | 20 bodies (16 + 4 reevals), 3 repeats each of merge on and merge off, `--workers 4 --min-shard 2` | 1 h / 3 h | "Generation wall -3 to -8% at equal results" (ROADMAP:4214-4217) | Merged slower, or any placed score differs |
| D-A1 | A1b | Cost of the device fluid and scatter step at W = 24 / 100 / 730 / 1,600, float32 and float64, median and rotor fixtures, graph-captured | 1 h / 3 h | float32: the per-panel marginal <= 0.044 us (Q1's float64 Mojo, ROADMAP:3704) and the launch <= 117 us per kernel group; scatter adds <= 50% to the fluid step | The marginal > 0.1 us per panel, or the scatter doubles the step |
| D-A2 | A4a | Step cost with gravity, A1a + A1b + A3, air and water segment worlds, on arch49's median, p90-dof (30) and rotor bodies at W = 10 x {2, 20, 60, 110} | 2 h / 5 h | c0 <= 0.6 ms; c1 <= 8 us per world for the 126-panel body; rotor bodies <= 1.6 x the median's c1 (Q2: 1.6x, ROADMAP:3771-3773) | c0 > 1 ms or c1 > 15 us (T7's falsifier, adjusted for F4's terms) |
| D-A3 | A2 | Capture wall of an unrolled 2,000-step graph per body against host-driven per-step graphs over 2,300 steps | 30 min / 1 h | Capture >= 1 s per body, so per-step graphs (+28 ms per body) stay the default | Capture < 28 ms per body |
| D-A4 | A2 | GPU memory per world at `njmax` 512 for the median and p90 bodies; the largest W of 16 concurrent bodies on the 6 GB RTX 3060 | 30 min / 1 h | <= 1 MB per world; 16 bodies x 600 worlds fit | 16 x 600 does not fit (then K_m + K_s drops, or bodies run in waves) |
| D-A5 | A4a | Stream overlap: 16 bodies at K_m + K_s = 2, 20 and 60 on 1, 4 and 16 streams; then 128 bodies on the best stream count | 1 h / 3 h | 4 streams cut the wall by >= 20% at K_m + K_s = 2 (dispatch-bound) and by <= 10% at 60 (W = 600 > W*); 128 bodies save <= 10% per body over 16 | The 128 saving > 10% (then A6 builds the epoch) |
| D-A6 | A6 | Winner's curse, the UQD paper's corrected score: at the end of the first device arm's gen-150 read, re-evaluate every elite at 64 fresh draws and compare with its recorded pooled mean; run the same read on the batched path's arch49 archive for contrast | 2 h / 4 h | Recorded minus corrected, median over elites: device <= 0.25x the batched path's gap; Spearman(recorded, corrected) >= 0.8 per medium | Device gap > 0.5x batched, or Spearman < 0.6 in any medium |
| D-A7 | A6 | Generation wall at K_m, K_s = (10, 10), (50, 10), (100, 10) on 16 real bodies with air and water on device and land and crossings on batched, against arch50's 168 s (ROADMAP:4097) | 3 h / 6 h | Within 1.5x of §3's table plus the batched land and crossing time (P1-P3 era, measured in the same run) | > 1.5x the prediction at (50, 10) |
| D-A8 | A4a | Still machine and divergence on the device across arch49's 229 elites (no_model_gate's arms on `evaluator="device"`, air and water) | 2 h / 5 h | Still machine: air 0/229, comp:water 0.012 <= 5/229 (batched: 3); elites' device water competence Spearman >= 0.7 with batched | Still water > 10/229, or Spearman < 0.5 |

R1 (§2) and D1 (§2) are reads of the same kind and use the same reporting shape.

---

## §D. Open questions for the user

1. **Land and the crossings on the device (A4b), or kept on the batched path at small K?** Evidence: Q5 gives K for reliability 0.9 as air 66, water 36, land 18 (ROADMAP:3830). Q4's null control moved 34.5% of land elites by more than 50% (ROADMAP:3800-3803). Air and water carry most of A's payoff (F12). Default in this plan: built after A6, only on a yes.
2. **K_m and K_s.** Defaults are 50 and 10 (§3 table: 147 s against 168 s). K_s = K_m (today's main + rescore symmetry) costs 233 s at K_m = 50.
3. **The lower-bound rule.** ROADMAP:3867-3869 says "a candidate enters a cell when its lower bound beats the incumbent's". This plan reads that as both lower bounds, with z = 1 typed until D-A6. The alternative is the candidate's lower bound against the incumbent's mean.
4. **The reference engine upgrade to mujoco 3.15** (F6), after D-M0.
5. **P3 at all**, given that A retires the two-trip structure.
6. **The `Py_NewRef` fix**, after D1: set `MOJO_PYTHON_LIBRARY` before the first import, or scrub the children's environment.

### §D answers (the user, 2026-10-10; questions 2, 5 and 6 delegated, decided here)

1. **Yes.** A4b is built, after A6, in the order of §1.
2. **K_m = 50, K_s = 10 for the first device arm** (§3 table: 147 s against
   arch50's 168 s, a lower bound). Not lower: Q5's K for reliability 0.9 is
   air 66, water 36, land 18 (ROADMAP:3830), so 50 covers water and land and
   leaves air at about 0.87. Not higher: (100, 10) is 1.51x the wall before
   F4's terms and contacts. D-A7 adds the point (64, 10); if it holds within
   1.5x of its prediction and the wall stays under 1.2x arch50's, K_m becomes
   64. Whether K is fixed or adaptive per candidate is set with question 3.
3. **Pooled mean with re-evaluated incumbents, no lower bound** (literature
   read 2026-10-10, sources below; A6 rewritten to it). None of the noisy-QD
   papers places on a confidence bound. Extract-QD names confidence ordering
   only as untested future work. What works is comparing pooled means while
   incumbents keep collecting draws, so a lucky draw cannot hold a cell:
   - Flageat & Cully, *Uncertain Quality-Diversity*, arXiv 2302.00463 (2023).
     Plain MAP-Elites fills with lucky solutions. Archive-sampling and
     Parallel-Adaptive-sampling, which re-evaluate the archive, were best on
     closed-loop Ant (p < 5e-4) and most reproducible. A fixed K with no
     re-evaluation (ME-Sampling) loses on complex control.
   - Flageat et al., *Extract-QD Framework*, arXiv 2502.06585 (2025,
     preprint). 25% of evaluations re-evaluate elites, which re-enter through
     the normal insertion path, with depth d = 8 and pooled means. At least as
     good as the best existing method on the standard noisy tasks. This is
     what N11 already built, so the device path extends N11 rather than
     replacing it.
   - Flageat & Cully, deep grids, arXiv 2006.14253 (2020), averages a depth
     of D solutions per cell. Justesen, Risi & Mouret, GECCO 2019 Companion,
     uses adaptive sampling. Heidrich-Meisner & Igel, ICML 2009, and Hansen et
     al., IEEE TEVC 2009, use races and uncertainty handling, which rank
     within a population rather than per cell. These were checked at abstract
     level only.
   K stays fixed per candidate (answer 2). Adaptive K, a race per cell, is
   tried only if D-A6 shows the 25% re-evaluation budget is not enough. D-A6
   is now the winner's-curse read that the UQD paper uses: recorded against
   corrected scores on fresh draws.
4. **Upgrade to mujoco 3.15.** D-M0 still runs before A0, but it no longer
   gates anything. It measures the boundary, and its result goes into A7's
   boundary text. If any medium loses more than 10 reproductions, air, water
   and land become not comparable across the upgrade as well.
5. **P3 is not built.** It is the hardest lane on today's path (F1: a sampling
   flag per machine, a basis hand-off inside the call, twins kept in one
   shard; opus), and A6 retires the two trips anyway: K_m mean draws and K_s
   sampled draws run in one launch. D-P3 is withdrawn. A6 depends on P1 only.
6. **Corrected by D1 (2026-10-10, `D1_result.md`): fix `PYTHONEXECUTABLE`,
   not `MOJO_PYTHON_LIBRARY`.** The first answer set the library and would
   not have worked (D1 arm f3). The kernel import C-setenvs
   `PYTHONEXECUTABLE` to whatever `python` is first on PATH, which is a pyenv
   shim of Python 3.9 under the default PATH and under `systemd-run --user`'s
   default environment. A spawn worker inherits it, so its `sys.executable`
   is Python 3.9. Its `usable()` subprocess then aborts on `Py_NewRef`, which
   exists only from 3.10. The library is irrelevant (arm f5), and Mojo
   overwrites a preset value (f4), so the fix comes after the first import
   and before any spawn: `os.environ["PYTHONEXECUTABLE"] = sys.executable` in
   the one module that imports the extension (D1 arm h2, worker `usable()`
   True). Lane F1 applies it. Its test checks that a spawned worker's
   `usable()` is True and that its `sys.executable` is the parent's, and a
   mutation deletes the assignment. The gate is D1's re-check cell. D1 also
   confirmed that the seed plans score exactly 0 under the fix, so ROADMAP's
   measurement (a) stands. Still open: arch50's log has no `Py_NewRef` line
   although it was launched the same way; its launch environment was not
   recorded.

## §E. Non-goals

- arch51 is not launched (the user's decision). No lane starts a search.
- P4 is not built. It is a null that stops the epoch from being built for speed on the CPU path (ROADMAP:4219-4223).
- B is not built (Q8, ROADMAP:4048-4073). Q3' is not run.
- No Mojo-Warp interop (Q6). `mojo/src` and `mojo/build` are untouched by every lane here.
- No change to scoring semantics on today's path, except P1's correctness fix. Its comparability effect: N11 reeval draws become task-only draws at a fixed basis; no stored archive carries N11 draws (P1 §Comparability). P2 changes which controller Tier-2 sees and children inherit, and no Tier-1 score definition. P3 changes nothing placed (bit-identical, gated).
- No asynchronous or steady-state placement, no speculative evaluation, no double-buffering (ROADMAP:4184-4193, 4350-4354).
- No callback-based fluid (F3).
