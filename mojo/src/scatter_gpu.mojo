"""Force limiter and body scatter, per machine within a batch.

Ports fluid.py:680-700.  Two kernels rather than one, because the limiter needs
a reduction before it can act:

    pass 1  fmag = |F| per panel, and the per-machine maximum of it
    host    (nothing -- the max stays on the device)
    pass 2  scale F by each machine's own limit, form the torque about that
            body's CoM, and scatter both into xfrc_applied

The reason this is its own module and not a few lines appended to the assembly
kernel is the word *per-machine*.  In the Python the limiter is

    weight = float(self._dry_mass.sum()) * GRAVITY + 1.0
    limit  = 60.0 * weight

and `self._dry_mass` belongs to one machine.  Once panels from many machines
are concatenated into one array -- which is the entire premise of the batched
evaluator -- a global limit would apply the heaviest machine's allowance to the
lightest, letting a 1 kg design carry forces sized for a 40 kg one.  So the
limit arrives as a per-machine array and every panel looks up its own via a
machine index.

`clamped` is likewise per machine.  It is a diagnostic the scorer reads to
decide whether a run left the model's valid domain, so collapsing it to "some
machine somewhere clamped" would mark every design in the batch as suspect
because one of them tumbled.
"""

from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

comptime BLOCK = 128


def fmag_kernel(
    f: UnsafePointer[Float64, MutAnyOrigin],
    machine: UnsafePointer[Int32, MutAnyOrigin],
    fmag: UnsafePointer[Float64, MutAnyOrigin],
    fmax: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    """|F| per panel, and the running per-machine maximum.

    np.linalg.norm sums left to right, and so does this, so the magnitudes
    match bitwise rather than merely closely.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    var x = f[unsafe_offset=j + 0]
    var y = f[unsafe_offset=j + 1]
    var z = f[unsafe_offset=j + 2]
    var m = sqrt(x * x + y * y + z * z)
    fmag[unsafe_offset=i] = m
    _ = Atomic.max(fmax + Int(machine[unsafe_offset=i]), m)


def limit_scatter_kernel(
    f: UnsafePointer[Float64, MutAnyOrigin],
    fmag: UnsafePointer[Float64, MutAnyOrigin],
    machine: UnsafePointer[Int32, MutAnyOrigin],
    limit: UnsafePointer[Float64, MutAnyOrigin],
    fmax: UnsafePointer[Float64, MutAnyOrigin],
    pos: UnsafePointer[Float64, MutAnyOrigin],
    xipos: UnsafePointer[Float64, MutAnyOrigin],
    body_id: UnsafePointer[Int32, MutAnyOrigin],
    xfrc: UnsafePointer[Float64, MutAnyOrigin],
    clamped: UnsafePointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Scale, form the torque, and accumulate into xfrc_applied.

    xfrc_applied is (nbody, 6): force in 0..2, torque in 3..5.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    var mi = Int(machine[unsafe_offset=i])
    var lim = limit[unsafe_offset=mi]

    # The Python only scales at all if some panel of this machine exceeds the
    # limit (`if np.any(fmag > limit)`), and then scales every panel by
    # min(1, limit/max(fmag, 1e-9)).  Below the threshold nothing is touched,
    # so a machine that never leaves the valid domain gets bit-identical forces
    # and its `clamped` flag stays down.
    var scale: Float64 = 1.0
    if fmax[unsafe_offset=mi] > lim:
        var m = fmag[unsafe_offset=i]
        var denom = m if m > 1e-9 else 1e-9
        var s = lim / denom
        scale = s if s < 1.0 else 1.0
        clamped[unsafe_offset=mi] = 1

    var fx = f[unsafe_offset=j + 0] * scale
    var fy = f[unsafe_offset=j + 1] * scale
    var fz = f[unsafe_offset=j + 2] * scale

    var b = Int(body_id[unsafe_offset=i])
    var ax = pos[unsafe_offset=j + 0] - xipos[unsafe_offset=b * 3 + 0]
    var ay = pos[unsafe_offset=j + 1] - xipos[unsafe_offset=b * 3 + 1]
    var az = pos[unsafe_offset=j + 2] - xipos[unsafe_offset=b * 3 + 2]
    var tx = ay * fz - az * fy
    var ty = az * fx - ax * fz
    var tz = ax * fy - ay * fx

    var o = b * 6
    _ = Atomic.fetch_add(xfrc + (o + 0), fx)
    _ = Atomic.fetch_add(xfrc + (o + 1), fy)
    _ = Atomic.fetch_add(xfrc + (o + 2), fz)
    _ = Atomic.fetch_add(xfrc + (o + 3), tx)
    _ = Atomic.fetch_add(xfrc + (o + 4), ty)
    _ = Atomic.fetch_add(xfrc + (o + 5), tz)


def limit_and_scatter(desc: PythonObject) raises -> PythonObject:
    """`desc` (int64): F, machine, limit, pos, xipos, body_id,
    xfrc_out, clamped_out, fmag_out, fmax_out, n, nbody, nmachine
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data))
    var n = Int(d[unsafe_offset=10])
    var nb = Int(d[unsafe_offset=11])
    var nm = Int(d[unsafe_offset=12])
    var ctx = DeviceContext()

    var f = ctx.enqueue_create_buffer[DType.float64](n * 3)
    ctx.enqueue_copy(dst_buf=f, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=0])))
    var mach = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=mach, src_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=1])))
    var lim = ctx.enqueue_create_buffer[DType.float64](nm)
    ctx.enqueue_copy(dst_buf=lim, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=2])))
    var pos = ctx.enqueue_create_buffer[DType.float64](n * 3)
    ctx.enqueue_copy(dst_buf=pos, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=3])))
    var xipos = ctx.enqueue_create_buffer[DType.float64](nb * 3)
    ctx.enqueue_copy(dst_buf=xipos, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=4])))
    var bid = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=bid, src_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=5])))

    var xfrc = ctx.enqueue_create_buffer[DType.float64](nb * 6)
    var clamped = ctx.enqueue_create_buffer[DType.int32](nm)
    var fmag = ctx.enqueue_create_buffer[DType.float64](n)
    var fmax = ctx.enqueue_create_buffer[DType.float64](nm)
    xfrc.enqueue_fill(0.0)
    clamped.enqueue_fill(0)
    fmax.enqueue_fill(0.0)

    var g = ceildiv(n, BLOCK)
    ctx.enqueue_function[fmag_kernel](
        f.unsafe_ptr(), mach.unsafe_ptr(), fmag.unsafe_ptr(),
        fmax.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)
    ctx.enqueue_function[limit_scatter_kernel](
        f.unsafe_ptr(), fmag.unsafe_ptr(), mach.unsafe_ptr(), lim.unsafe_ptr(),
        fmax.unsafe_ptr(), pos.unsafe_ptr(), xipos.unsafe_ptr(),
        bid.unsafe_ptr(), xfrc.unsafe_ptr(), clamped.unsafe_ptr(),
        Int32(n), grid_dim=g, block_dim=BLOCK)

    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=6])), src_buf=xfrc)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=7])), src_buf=clamped)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=8])), src_buf=fmag)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=9])), src_buf=fmax)
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit_scatter_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("scatter_gpu")
        m.def_function[limit_and_scatter]("limit_and_scatter")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
