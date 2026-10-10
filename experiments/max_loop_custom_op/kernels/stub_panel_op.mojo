"""Q8 stub: a thread-per-panel MAX custom op shaped like the fluid kernel.

Experiment: experiments/max_loop_custom_op (ROADMAP 2026-10-10, "an evaluation
that never leaves the device", section 8, probe Q8).

State layout, float32, shape (WB, P, 8) -- one row of 8 floats per panel:
    [0:3]  v   a 3-vector (panel velocity stand-in)
    [3:7]  q   a unit quaternion; the 3x3 rotation R(q) is formed in registers
               (the brief asked for "a 3x3 stored in the state"; 8 floats per
               panel cannot hold 9 + 3, so the 3x3 is built from the stored
               quaternion, which costs the same matvec and 4 loads instead of 9)
    [7]    out the stand-in "force" written by this op

Per panel (one thread each):  w = R(q) v;  c = w x v;  aoa = atan2f(w.z, w.x);
out = |c| * aoa + gain[body];  v <- w.   A rotation preserves |v| so 2000
applications stay bounded.  `gain` is (WB, 1): it is the matmul's first output
column, which chains this op behind the matmul inside the loop body.

Two registered ops, same body:
  * stub_panel_op          functional: reads `state`, writes a new `out` tensor
                           (the loop carries the result; MAX owns the buffers).
  * stub_panel_op_inplace  mutating: `state` is a mutable input and is written
                           back in place (graph side: ops.inplace_custom).

`atan2f` / `atan_unit` are copied from mojo/src/mathx.mojo (Dytiscidae, same
polynomial, 1.17e-05 rad max error) because `atan2` does not link for NVIDIA
device code (ptxas: Unresolved extern function 'atan2f'); copied rather than
imported so this directory is self-contained for `custom_extensions=[...]`.
"""
import extensibility

from max.gpu.host import DeviceContext
from std.math import abs, ceildiv, sqrt
from std.gpu import global_idx

from extensibility import InputTensor, OutputTensor, _MutableInputTensor

comptime PI: Float32 = 3.14159265358979323846
comptime HALF_PI: Float32 = PI / 2.0
comptime BLOCK = 128


# --- copied from mojo/src/mathx.mojo (atan_unit, atan2f) ---------------------
@always_inline
def atan_unit(x: Float32) -> Float32:
    """Rational minimax approximation of atan on |x| <= 1."""
    var z = x * x
    return x * (
        0.9998660
        + z * (-0.3302995 + z * (0.1801410 + z * (-0.0851330 + z * 0.0208351)))
    )


@always_inline
def atan2f(y: Float32, x: Float32) -> Float32:
    var ax = abs(x)
    var ay = abs(y)
    var swap = ay > ax
    var num = ax if swap else ay
    var den = ay if swap else ax
    var r = atan_unit(num / den) if den != 0.0 else Float32(0.0)
    if swap:
        r = HALF_PI - r
    if x < 0.0:
        r = PI - r
    if y < 0.0:
        r = -r
    return r


# --- the per-panel body, shared by the CPU loop and the GPU kernel -----------
@always_inline
def panel_step(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    gain: Float32,
    row: Int,
):
    var b = row * 8
    var vx = src[unsafe_offset=b + 0]
    var vy = src[unsafe_offset=b + 1]
    var vz = src[unsafe_offset=b + 2]
    var qw = src[unsafe_offset=b + 3]
    var qx = src[unsafe_offset=b + 4]
    var qy = src[unsafe_offset=b + 5]
    var qz = src[unsafe_offset=b + 6]
    # R(q), row-major 3x3
    var r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    var r01 = 2.0 * (qx * qy - qz * qw)
    var r02 = 2.0 * (qx * qz + qy * qw)
    var r10 = 2.0 * (qx * qy + qz * qw)
    var r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    var r12 = 2.0 * (qy * qz - qx * qw)
    var r20 = 2.0 * (qx * qz - qy * qw)
    var r21 = 2.0 * (qy * qz + qx * qw)
    var r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    var wx = r00 * vx + r01 * vy + r02 * vz
    var wy = r10 * vx + r11 * vy + r12 * vz
    var wz = r20 * vx + r21 * vy + r22 * vz
    # c = w x v
    var cx = wy * vz - wz * vy
    var cy = wz * vx - wx * vz
    var cz = wx * vy - wy * vx
    var aoa = atan2f(wz, wx)
    var out = sqrt(cx * cx + cy * cy + cz * cz) * aoa + gain
    dst[unsafe_offset=b + 0] = wx
    dst[unsafe_offset=b + 1] = wy
    dst[unsafe_offset=b + 2] = wz
    dst[unsafe_offset=b + 3] = qw
    dst[unsafe_offset=b + 4] = qx
    dst[unsafe_offset=b + 5] = qy
    dst[unsafe_offset=b + 6] = qz
    dst[unsafe_offset=b + 7] = out


def _panel_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    gain: Pointer[Float32, MutAnyOrigin],
    n_rows_dev: Int32,
    panels_dev: Int32,
):
    var row = global_idx.x
    if row < Int(n_rows_dev):
        var body = row // Int(panels_dev)
        panel_step(src, dst, gain[unsafe_offset=body], row)


def _run[
    target: StaticString
](
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    gain: Pointer[Float32, MutAnyOrigin],
    n_bodies: Int,
    panels: Int,
    ctx: DeviceContext,
) raises:
    var n_rows = n_bodies * panels
    comptime if target == "cpu":
        for row in range(n_rows):
            panel_step(src, dst, gain[unsafe_offset=row // panels], row)
    elif target == "gpu":
        ctx.enqueue_function[_panel_kernel](
            src,
            dst,
            gain,
            Int32(n_rows),
            Int32(panels),
            grid_dim=ceildiv(n_rows, BLOCK),
            block_dim=BLOCK,
        )
    else:
        raise Error("No known target:", target)


@extensibility.register("stub_panel_op")
struct StubPanelOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=3, ...],
        state: InputTensor[dtype=DType.float32, rank=3, ...],
        gain: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        _run[target](
            state.unsafe_ptr(),
            output.unsafe_ptr(),
            gain.unsafe_ptr(),
            state.dim_size(0),
            state.dim_size(1),
            ctx,
        )


@extensibility.register("stub_panel_op_inplace")
struct StubPanelOpInplace:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        state: _MutableInputTensor[dtype=DType.float32, rank=3, ...],
        gain: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        _run[target](
            state.unsafe_ptr(),
            state.unsafe_ptr(),
            gain.unsafe_ptr(),
            state.dim_size(0),
            state.dim_size(1),
            ctx,
        )
