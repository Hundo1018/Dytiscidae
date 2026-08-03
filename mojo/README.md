# GPU port of the fluid solver

The search spends 84% of every simulated step in `physics/fluid.py`. This
directory is where that moves to the GPU.

```bash
cd mojo
pixi run test      # host-side accuracy checks
pixi run probe     # runs a kernel on the GPU
```

## Why this is worth doing, and what the numbers actually say

Two separate wins, and they are easy to conflate:

**Compilation.** The solver is *dispatch-bound, not arithmetic-bound*. Measured
across the seven body plans, going from 48 to 134 panels — 2.79x the work —
costs only 274 to 348 us, a factor of 1.27. The linear fit is **0.86 us/panel
plus 233 us of fixed overhead**, and that fixed part is Python and numpy
per-call cost on tiny arrays. It is 85% of a beetle's step. No GPU removes it;
only leaving Python does.

**Batching.** The arithmetic itself is 41-115 us per machine, which is far too
small to be worth a kernel launch on its own. It becomes worth it across
candidates: 60 machines at ~80 panels is ~4800 panels per launch.

So the GPU is the second win and it only pays once evaluation is restructured
so that N machines step in lockstep in one process. Ported but unbatched, the
kernel launch and the round trip will eat the gain.

### Measured, with persistent buffers (`fluid_state.GpuFluid`)

| panels | ~machines | alloc/call | persistent | numpy | speedup |
|---|---|---|---|---|---|
| 70 | 1 | 83.4 | 36.9 | 29.6 | 0.80x |
| 280 | 4 | 85.7 | 41.7 | 42.9 | **1.03x** |
| 1120 | 16 | 91.3 | 49.1 | 88.1 | 1.79x |
| 4480 | 64 | 119.3 | 100.8 | 292.3 | 2.90x |
| 17920 | 256 | 305.9 | 282.3 | 1164.4 | 4.13x |
| 71680 | 1024 | 1181.6 | 1053.6 | 5024.0 | 4.77x |

(microseconds per call)

Break-even is **~280 panels, about 4 machines** — not the ~1600 quoted earlier
in this file's history. That earlier figure came from a benchmark whose first
GPU call in the process paid CUDA initialisation and pool growth; warm, the
per-call allocation cost is 84 us rather than 520. Persisting the buffers still
roughly halves the fixed cost, and it is the difference between breaking even at
4 machines and at 12.

The allocation cost is real and worth removing — warm, one 8192-element float64
buffer costs 97.5 us to allocate against 6.0 us to create a whole
`DeviceContext`. The context is not worth persisting; the buffers are all of
it.

A note on what does *not* block this. Batching whole MuJoCo models under MJX
is blocked by morphology: an archive of 102 elites holds 52 distinct
(dof, n_parts, body_plan) combinations, and vmap needs one shared graph. The
fluid solver has no such problem — it already operates on a flat panel array
indexed by `body_id` (`fluid.py:698` is a scatter), so panels from
heterogeneous machines concatenate behind an offset table. Those are different
questions and the answer to one is not the answer to the other.

MuJoCo stays on the CPU. It is 2-3% of the step, and `fluid.apply` writes
`model.body_mass`, `model.body_inertia` and `data.xfrc_applied` which `mj_step`
reads on the very next line (`triphibian.py:733-734`), so there is a mandatory
host round trip every step regardless.

## Toolchain notes

Four things cost a compile cycle each; they are not in the docs.

| Symptom | Cause |
|---|---|
| `unable to locate module 'std'`, `print` is an "unknown declaration" | Running `.pixi/envs/default/bin/mojo` directly. It does not load the prelude. Always `pixi run mojo`. |
| `unable to locate module 'layout'` | `TileTensor` ships with `max`, not `mojo-compiler`. Both belong in `[dependencies]`. |
| `Int and UInt do not conform to DevicePassable` | Kernel scalar arguments must be fixed width — `Int32`/`Int64`. The failure surfaces inside the launch machinery, not at your signature. |
| `error: expected argument name` on `out: ...` | `out` is a reserved argument convention. Rename it. |

## `mathx.mojo`

`atan2` does not link for NVIDIA device code:

    ptxas fatal : Unresolved extern function 'atan2f'

and `atan` is CPU-only. Angle of attack needs two atan2 calls per panel, so it
is on the critical path and has to be supplied by hand. `mathx.atan2f` is a
rational minimax on the reduced interval plus quadrant reconstruction, measured
at **1.17e-05 rad (0.00067 deg)** worst case against libm over 8192 samples
covering all four quadrants and both sides of the |y|>|x| swap.

That error is ~100x float32 epsilon and irrelevant here — the leading-edge
vortex gate it feeds switches near 40 degrees, and the quasi-steady assumption
underneath is good to perhaps 30%. It is pinned by `tests/test_mathx.mojo` at a
2x envelope rather than left to drift. If a future Mojo ships a working device
`atan2`, delete `mathx` and its tests together instead of loosening them.

## Validation

`tests/test_physics.py` in the parent project is 118 checks that pin sign
conventions, buoyancy, orientation-dependent bluff drag and anisotropic added
mass. The port is finished when those pass against the Mojo solver, which makes
this a machine-checkable migration rather than a rewrite on faith.

## Running the GPU validation

```bash
cd mojo && pixi run build          # produces build/fluid_gpu.so
cd .. && PYTHONPATH=.:mojo/build .venv/bin/python mojo/tests/test_fluid_gpu.py
```

Two Pythons are in play and they are not interchangeable: the extension is
built against the pixi environment's 3.12, and the test imports `dytiscidae`
from the project venv. They happen to be ABI-compatible 3.12 builds, which is
why one `PYTHONPATH` spanning both works.

## Status

Done:

- `mathx.atan2f` — device `atan2`, accuracy pinned at 1.17e-05 rad.
- `fluid_gpu.body_to_world` — the kinematics block (`fluid.py:428-432`) on the
  GPU. Agrees with numpy to 4.4e-16 on all seven body plans, and to 4.4e-16 on
  **seven morphologies concatenated into a single 554-panel launch**, which is
  the batching premise demonstrated rather than assumed.

  It is not bit-exact and cannot be: `np.einsum` does not sum left-to-right
  (verified on numpy 2.5.1 for both `ni,ni->n` and `nij,nj->ni`), so a
  three-term dot differs from the kernel's in the last bit. `np.linalg.norm`
  *is* left-to-right, so norms do match bitwise.

- `fluid_gpu.strip_theory` — `fluid.py:462-487`. All seven outputs at machine
  epsilon (worst 7.9e-16 relative).
- `mathx.atan2d` — double-precision `atan2`, 4.4e-16 rad.

### Device math, measured

| | float32 | float64 |
|---|---|---|
| `sin` `cos` `exp` `log` `log10` `sqrt` | links | **compile error** |
| `atan` `atan2` | link error | link error |

Float64 *arithmetic* is fine — only the transcendentals are absent, which is
why `mathx` can hold double precision by using polynomials instead of calls.

This decides the shape of the next slice. `lift_coefficient` needs `log10`,
`sin` and `exp`; `drag_coefficient` needs more. Either they run in float32,
costing ~1e-7 relative where the solver itself is good to ~30%, or `mathx`
grows float64 versions of all three. The first is cheap and physically
irrelevant; the second keeps the machine-epsilon validation standard that has
caught two real bugs so far — the epsilon-placement error in `alpha`, and
before that a whole class of sign bug in the Python. The standard is worth more
than the precision.

Next, in dependency order:

1. Float64 `sin`, `exp`, `log10` in `mathx`, each pinned like `atan2d`.
2. `lift_coefficient` and `drag_coefficient`, then bluff-body drag, added mass
   and the body scatter — `np.add.at` at `fluid.py:698` becomes an atomic.
2. **The batched evaluator.** Until N machines step in lockstep in one process
   this port cannot pay: per call it allocates buffers, launches, and round
   trips for one machine's ~70 panels, which is far below break-even. This is
   the piece that turns a correct port into a fast one.
3. Persistent device buffers owned by that evaluator, replacing the
   allocate-per-call in `body_to_world`.

Not moving: MuJoCo, `mj_objectVelocity`, and the `body_mass` write-back stay on
the CPU.
