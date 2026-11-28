# Port of E0s into `dytiscidae/` (P0.5, P0.6)

Written by the lead 2026-11-11 at commit 1dabe7b. No training, no fitting.
The prototype is `existence/stateless.py` (with `triquad.py` and
`landgait.py`); its measured rows are `existence/results_stateless.jsonl`.
The port moves code; it changes no number.

## What goes where

| new | holds | from |
|---|---|---|
| `dytiscidae/core/bodyplans.py`: `legged_quad()` and `ANCHOR_PLANS = {"legged_quad": ...}` | the L1 genome, every edit from the reference quad listed in its docstring | `triquad.b1_genome`, `landgait.l1_genome`, `limbs`, `_rethrottle` |
| `dytiscidae/control/innerloop.py` | the action layout (`N_ACT`, index names), `body_model(env)` (the model constants, no map), `InnerLoop(env, model).act(u)` | `triquad.Hand.__init__`, `_yaw_authority`, `_cascade`, `_rotor_cmd`; `stateless.Inner`; the limb gait |
| `dytiscidae/control/outer.py` | `OBS_CHANNELS`, `N_OBS`, `K`, `observe(env, commanded, prev_memory, since)`, `Teacher(model, map).teach(o)`, and `SplitController` (the environment's side: previous memory, held `u`, command clock, install/uninstall) | `stateless.observe`, `Teacher`, `Split`, `constants` |

`ANCHOR_PLANS` is separate from `REFERENCE_PLANS` and `BODY_PLANS` on
purpose: tests and tools iterate those two.

Rules:

- Search first (CLAUDE.md "Before adding anything"): `docs/index/FEATURES.yaml`,
  `SYMBOLS.md`. If something already does a piece (a mixer, a yaw test, a
  gait generator), reuse it and say so; if nothing does, say that.
- `TriphibianEnv.observation` is not changed: `outer.observe` calls it and
  appends. No stored checkpoint, score or run is affected; nothing in the
  search imports the new modules.
- `SplitController` installs itself explicitly (`install(env)` /
  `uninstall(env)`, or a context manager) and restores what it replaced; a
  second install on the same environment is an error, not a stack.
- No import from `experiments/` in the package. No module-level state.
  Constants carry their provenance in a comment (the trial or probe that set
  them), as the surrounding code does. Match the surrounding code's comment
  density and naming.
- Dependencies must pass `tests/test_architecture.py`; mujoco and numpy are
  allowed in `control/`, `core/` and `envs/` only as the existing modules
  there already use them (check `docs/index/MODULES.md`; if `core.bodyplans`
  may not import `physics`, compute the throttle where it may be computed and
  say how).

## Gates (all machine-checked; paste the output)

1. **Equivalence.** `experiments/triphibian_diagnosis/scripts/port_check.py`
   runs the package version on every row kind of the prototype (missions
   teach / zero / const / frozen / nomem for seeds 0-31 and three starts,
   segments seeds 0-7, gated crossings entries 0-7) and compares with
   `existence/results_stateless.jsonl`: every stored field except wall time
   equal. Report rows compared and rows that differ; 0 is the gate. The
   prototype files are not edited.
2. **Tests** in `tests/test_physics.py` (registered in `run_all`; no GPU),
   written in the file's style (`check(name, cond, detail)`):
   - `u = 0`: every rotor command is 0 and the limbs are at rest; a mission
     from the water under `u = 0` makes 0 transitions;
   - the inner loop does not read position: the same body state translated
     by (3, -2) m in the air far from terrain gives the same actuator
     targets; and `observe` is unchanged under a translation along the shore
     and changes only in the waterline-distance channel under one across it;
   - statelessness: a fresh `Teacher` at every decision and a fresh
     `InnerLoop` at every step give the same short mission, and neither
     assigns an attribute after construction;
   - the teacher completes one mission from each start (seed 0, default
     legs) and the command-frozen teacher completes none;
   - the anchor body passes Tier-0 as the search runs it and its limbs carry
     it (`ground_support` 0.95-1.05 at rest).
   Keep the added suite time under about two minutes; say what it is.
3. **Mutations** in `tools/mutate.py`, one per gate above at least (zero
   action not inert; inner loop reads absolute position; memory not fed
   back; command clock not restarted; rotors-on flag ignored), each naming
   the test that should catch it; `tools/mutate.py --check` clean; each new
   mutation run and caught (`--only` takes one id per invocation).
4. **Suites**: `tests/test_physics.py` (detached, `MUJOCO_GL=disable`): the
   line `all physics checks passed`, 0 `[fail`, 0 `[skip`; the eight cheap
   suites; `tools/index_gen.py write` then `check`; `tests/test_index.py`
   after adding a feature entry in `docs/index/FEATURES.yaml` for the split
   (implementation symbols and the tests).

## Not in this port

The batched path, any learner, P0.7 (curriculum), P0.9 (yaw-handed pair,
density sweep), the `air_to_water` hold, a docs rewrite. One paragraph for
`docs/ROADMAP.md` is the lead's to write.
