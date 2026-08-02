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
