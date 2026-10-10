# Q8 — a custom Mojo op inside `ops.while_loop`

ROADMAP 2026-10-10, "an evaluation that never leaves the device", section 8, probe Q8.
Question: does a thread-per-panel Mojo custom op placed in a 2000-iteration
MAX `ops.while_loop`, beside a `(1600, 33) @ (33, 64)` matmul, keep the loop at
<= 150 us per iteration (G measured 50.9 us/iteration for a matmul-only loop)?
Frozen prediction and thresholds are quoted verbatim in `run.py` and in the report.

Files
- `kernels/stub_panel_op.mojo` — two registered ops, `stub_panel_op` (functional) and
  `stub_panel_op_inplace` (`_MutableInputTensor`). One thread per panel row of a
  `(WB, 50, 8)` float32 state: rotate a 3-vector by R(q), cross product, `atan2f`
  (copied from `mojo/src/mathx.mojo`, attribution in the file), write back.
  8 floats per row cannot hold a vector and a 3x3, so the row is v(3) + quaternion(4) + out(1).
- `run.py` — graph builder (three variants: `base` = no custom op, `custom`, `inplace`),
  timing (1 warm-up execute, then `--reps` timed 2000-iteration executes, host wall
  time with a device synchronize), a short on-device correctness graph against a
  numpy reference, the verdict, and the report writer for
  `runs/analysis_1010_failure_theory/Q8_result.md`.

Run (from the pixi env; never builds into `mojo/build/`):

    cd mojo
    pixi run python ../experiments/max_loop_custom_op/run.py --selftest   # CPU, ~10 s warm (60 s cold), no GPU
    pixi run python ../experiments/max_loop_custom_op/run.py > ../experiments/max_loop_custom_op/run_q8.log 2>&1   # GPU

The GPU run compiles three loop graphs plus two 3-iteration check graphs
(~18 s each, G's number) and times five 2000-iteration executes of each of the
three variants: predicted ~3-5 minutes, use a 900 s timeout. It needs the GPU idle
(the loop is launch-bound, so a contended GPU would inflate the figure).

Known limits
- The CPU selftest runs the same Mojo source through the MAX CPU target, not the
  GPU kernel; the GPU code path of the kernel was only checked to compile
  (`mojo build` of a launcher with the kernel, build only), and the MAX GPU compile
  of the custom op happens at `session.load` on the first GPU run.
- Timing is host wall time around `execute()` + `synchronize()`, not CUDA events.
- The loop counter is a CPU int32 (as in G's probe), so each iteration's predicate is
  evaluated on the host; that is part of the measured per-iteration figure.
