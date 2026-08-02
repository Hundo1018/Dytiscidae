"""GPU kinematics for the fluid solver.

Ports the block at `physics/fluid.py:428-432`: rotate each panel's local span,
chord and normal into world frame and place its centroid.  Four einsums and a
gather in numpy, one thread per panel here.

This is the first slice deliberately.  It is pure -- no state, no sequential
dependency -- so it can be validated against numpy to the bit, and it has the
shape the whole port depends on: panels are a flat array indexed by `body_id`,
so panels belonging to *different machines* concatenate into one launch. That
is what makes batching across candidates possible even though their
morphologies differ.

Buffers are allocated per call here. That is the wrong thing for production --
the round trip is the cost that matters -- but it keeps this version honest and
easy to check. Persistent device buffers come once the batched evaluator exists
to own them.

float64 throughout, to match numpy exactly during validation. The RTX 3060 runs
fp64 at 1/32 rate, so this is a correctness-first choice, not a speed one.
"""

from std.os import abort
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

comptime BLOCK = 128
comptime DeviceBufferF64 = DeviceBuffer[DType.float64]


def kin_kernel(
    xpos: UnsafePointer[Float64, MutAnyOrigin],
    xmat: UnsafePointer[Float64, MutAnyOrigin],
    body_id: UnsafePointer[Int32, MutAnyOrigin],
    loc: UnsafePointer[Float64, MutAnyOrigin],
    span: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    normal: UnsafePointer[Float64, MutAnyOrigin],
    pos_w: UnsafePointer[Float64, MutAnyOrigin],
    span_w: UnsafePointer[Float64, MutAnyOrigin],
    chord_w: UnsafePointer[Float64, MutAnyOrigin],
    normal_w: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return

    var b = Int(body_id[unsafe_offset=i])
    var r = b * 9
    var m0 = xmat[unsafe_offset=r + 0]
    var m1 = xmat[unsafe_offset=r + 1]
    var m2 = xmat[unsafe_offset=r + 2]
    var m3 = xmat[unsafe_offset=r + 3]
    var m4 = xmat[unsafe_offset=r + 4]
    var m5 = xmat[unsafe_offset=r + 5]
    var m6 = xmat[unsafe_offset=r + 6]
    var m7 = xmat[unsafe_offset=r + 7]
    var m8 = xmat[unsafe_offset=r + 8]

    var j = i * 3

    # centroid: body origin + R @ pos_local
    var lx = loc[unsafe_offset=j + 0]
    var ly = loc[unsafe_offset=j + 1]
    var lz = loc[unsafe_offset=j + 2]
    pos_w[unsafe_offset=j + 0] = xpos[unsafe_offset=b * 3 + 0] + m0 * lx + m1 * ly + m2 * lz
    pos_w[unsafe_offset=j + 1] = xpos[unsafe_offset=b * 3 + 1] + m3 * lx + m4 * ly + m5 * lz
    pos_w[unsafe_offset=j + 2] = xpos[unsafe_offset=b * 3 + 2] + m6 * lx + m7 * ly + m8 * lz

    # the three axes are pure rotations, no translation
    var sx = span[unsafe_offset=j + 0]
    var sy = span[unsafe_offset=j + 1]
    var sz = span[unsafe_offset=j + 2]
    span_w[unsafe_offset=j + 0] = m0 * sx + m1 * sy + m2 * sz
    span_w[unsafe_offset=j + 1] = m3 * sx + m4 * sy + m5 * sz
    span_w[unsafe_offset=j + 2] = m6 * sx + m7 * sy + m8 * sz

    var cx = chord[unsafe_offset=j + 0]
    var cy = chord[unsafe_offset=j + 1]
    var cz = chord[unsafe_offset=j + 2]
    chord_w[unsafe_offset=j + 0] = m0 * cx + m1 * cy + m2 * cz
    chord_w[unsafe_offset=j + 1] = m3 * cx + m4 * cy + m5 * cz
    chord_w[unsafe_offset=j + 2] = m6 * cx + m7 * cy + m8 * cz

    var nx = normal[unsafe_offset=j + 0]
    var ny = normal[unsafe_offset=j + 1]
    var nz = normal[unsafe_offset=j + 2]
    normal_w[unsafe_offset=j + 0] = m0 * nx + m1 * ny + m2 * nz
    normal_w[unsafe_offset=j + 1] = m3 * nx + m4 * ny + m5 * nz
    normal_w[unsafe_offset=j + 2] = m6 * nx + m7 * ny + m8 * nz


@always_inline
def _f64(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBufferF64:
    """Upload `count` float64 from a host address."""
    var dev = ctx.enqueue_create_buffer[DType.float64](count)
    var host = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host)
    return dev^


def body_to_world(desc: PythonObject) raises -> PythonObject:
    """Run the kinematics block on the GPU.

    `desc` is one int64 numpy array carrying every pointer and size, because
    `def_function` accepts at most six arguments and this needs thirteen:

        [xpos, xmat, body_id, pos_local, span_local, chord_local, normal_local,
         pos_out, span_out, chord_out, normal_out, n_panels, n_body]
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data)
    )
    var n = Int(d[unsafe_offset=11])
    var nbody = Int(d[unsafe_offset=12])

    var ctx = DeviceContext()

    var xpos = _f64(ctx, Int(d[unsafe_offset=0]), nbody * 3)
    var xmat = _f64(ctx, Int(d[unsafe_offset=1]), nbody * 9)
    var loc = _f64(ctx, Int(d[unsafe_offset=3]), n * 3)
    var span = _f64(ctx, Int(d[unsafe_offset=4]), n * 3)
    var chord = _f64(ctx, Int(d[unsafe_offset=5]), n * 3)
    var normal = _f64(ctx, Int(d[unsafe_offset=6]), n * 3)

    var bid = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(
        dst_buf=bid,
        src_ptr=UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=2])
        ),
    )

    var pos_w = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var span_w = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var chord_w = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var normal_w = ctx.enqueue_create_buffer[DType.float64](n * 3)

    ctx.enqueue_function[kin_kernel](
        xpos.unsafe_ptr(), xmat.unsafe_ptr(), bid.unsafe_ptr(),
        loc.unsafe_ptr(), span.unsafe_ptr(), chord.unsafe_ptr(),
        normal.unsafe_ptr(),
        pos_w.unsafe_ptr(), span_w.unsafe_ptr(), chord_w.unsafe_ptr(),
        normal_w.unsafe_ptr(),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )

    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=7])
        ),
        src_buf=pos_w,
    )
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=8])
        ),
        src_buf=span_w,
    )
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=9])
        ),
        src_buf=chord_w,
    )
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=10])
        ),
        src_buf=normal_w,
    )
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit_fluid_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("fluid_gpu")
        m.def_function[body_to_world]("body_to_world")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
