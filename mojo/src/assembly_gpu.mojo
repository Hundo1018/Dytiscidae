"""Force assembly: circulatory lift and drag, Kramer rotation, buoyancy.

Ports three pieces of `fluid.apply` that between them finish the force vector:

    fluid.py:494-496   L = q*area*cl*lift_scale ; D = q*area*cd*cd_scale
                       F += L*lift_axis + D*d_hat
    fluid.py:582-588   f_rot = where(is_wing, c_rot*rho*U*omega_s*chord^2*dr, 0)
                       F += f_rot*lift_axis
    fluid.py:676-677   f_buoy = water.rho*GRAVITY*volume_buoyant*subf
                       F[:,2] += f_buoy

It takes the bluff-body force and the added-mass gravity compensation as
*inputs* rather than being a separate stage, because the accumulation order in
the source is

    circulatory -> bluff -> Kramer -> added-mass gravity -> buoyancy

and floating-point addition is not associative.  Splitting it into two kernels
that each sum their own part and add the results at the end would reorder it.
That is a difference of last bits and nothing physical, but the whole port is
being checked at last-bit resolution and an unforced reordering would spend
that budget for no reason.

`c_rot` is a per-panel array, not a scalar: `pi * (0.75 - pitch_axis)`
(fluid.py:364), so a strip's rotational circulation depends on where its pitch
axis sits along the chord.  Reading it as a constant would give every panel the
value belonging to a quarter-chord hinge.

`omega_s` is recomputed here from `omega` and `s_hat` rather than being carried
over from the strip-theory kernel, which only exports the *magnitude* through
`reduced_freq`.  Kramer circulation is signed -- it reverses with the stroke --
so the sign has to come from somewhere, and recomputing one dot product is
cheaper than widening the strip kernel's signature.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

comptime BLOCK = 128
comptime GRAVITY: Float64 = 9.80665


@always_inline
def _up(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.float64]:
    var b = ctx.enqueue_create_buffer[DType.float64](count)
    ctx.enqueue_copy(dst_buf=b, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=addr))
    return b^


def assembly_kernel(
    q: UnsafePointer[Float64, MutAnyOrigin],
    area: UnsafePointer[Float64, MutAnyOrigin],
    cl: UnsafePointer[Float64, MutAnyOrigin],
    cd: UnsafePointer[Float64, MutAnyOrigin],
    lift_axis: UnsafePointer[Float64, MutAnyOrigin],
    d_hat: UnsafePointer[Float64, MutAnyOrigin],
    f_bluff: UnsafePointer[Float64, MutAnyOrigin],
    c_rot: UnsafePointer[Float64, MutAnyOrigin],
    rho: UnsafePointer[Float64, MutAnyOrigin],
    u: UnsafePointer[Float64, MutAnyOrigin],
    omega: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    dr: UnsafePointer[Float64, MutAnyOrigin],
    vol_buoy: UnsafePointer[Float64, MutAnyOrigin],
    subf: UnsafePointer[Float64, MutAnyOrigin],
    fz_added: UnsafePointer[Float64, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    f_out: UnsafePointer[Float64, MutAnyOrigin],
    l_out: UnsafePointer[Float64, MutAnyOrigin],
    d_out: UnsafePointer[Float64, MutAnyOrigin],
    buoy_out: UnsafePointer[Float64, MutAnyOrigin],
    lift_scale: Float64,
    cd_scale: Float64,
    water_rho: Float64,
    n: Int32,
):
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    var wing = is_wing[unsafe_offset=i] != 0

    # --- circulatory ------------------------------------------------------
    # cl and cd are already zero on bluff panels (the coefficient kernel gates
    # them), so L and D come out zero there without a branch here.  That
    # matters: the diagnostics sum |L| and |D| over all panels.
    var qq = q[unsafe_offset=i] * area[unsafe_offset=i]
    var lift = qq * cl[unsafe_offset=i] * lift_scale
    var drag = qq * cd[unsafe_offset=i] * cd_scale
    l_out[unsafe_offset=i] = lift
    d_out[unsafe_offset=i] = drag

    var fx = lift * lift_axis[unsafe_offset=j + 0] + drag * d_hat[unsafe_offset=j + 0]
    var fy = lift * lift_axis[unsafe_offset=j + 1] + drag * d_hat[unsafe_offset=j + 1]
    var fz = lift * lift_axis[unsafe_offset=j + 2] + drag * d_hat[unsafe_offset=j + 2]

    # --- bluff body, in source order --------------------------------------
    fx += f_bluff[unsafe_offset=j + 0]
    fy += f_bluff[unsafe_offset=j + 1]
    fz += f_bluff[unsafe_offset=j + 2]

    # --- Kramer rotational circulation ------------------------------------
    # Signed: this is what generates useful force through stroke reversal, when
    # the translational velocity is near zero, and it changes sign with the
    # stroke.  Bluff panels get nothing.
    if wing:
        var ws = (
            omega[unsafe_offset=j + 0] * s_hat[unsafe_offset=j + 0]
            + omega[unsafe_offset=j + 1] * s_hat[unsafe_offset=j + 1]
            + omega[unsafe_offset=j + 2] * s_hat[unsafe_offset=j + 2]
        )
        var ch = chord[unsafe_offset=i]
        var f_rot = (
            c_rot[unsafe_offset=i] * rho[unsafe_offset=i] * u[unsafe_offset=i]
            * ws * ch * ch * dr[unsafe_offset=i]
        )
        fx += f_rot * lift_axis[unsafe_offset=j + 0]
        fy += f_rot * lift_axis[unsafe_offset=j + 1]
        fz += f_rot * lift_axis[unsafe_offset=j + 2]

    # --- added-mass gravity compensation, then buoyancy -------------------
    # Both are vertical only, and both come after Kramer in the source.
    fz += fz_added[unsafe_offset=i]
    var fb = water_rho * GRAVITY * vol_buoy[unsafe_offset=i] * subf[unsafe_offset=i]
    buoy_out[unsafe_offset=i] = fb
    fz += fb

    f_out[unsafe_offset=j + 0] = fx
    f_out[unsafe_offset=j + 1] = fy
    f_out[unsafe_offset=j + 2] = fz


def assembly(desc: PythonObject, scalars: PythonObject) raises -> PythonObject:
    """`desc` (int64): q, area, cl, cd, lift_axis, d_hat, f_bluff, c_rot, rho,
    U, omega, s_hat, chord, dr, vol_buoy, subf, fz_added, is_wing,
    F_out, L_out, D_out, buoy_out, n

    `scalars`: lift_scale, cd_scale, water_rho
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data))
    var n = Int(d[unsafe_offset=22])
    var ctx = DeviceContext()

    # Each buffer gets its own name.  Indexing a List inline makes the
    # compiler treat the pointers as potentially aliasing and it refuses the
    # launch ("aliasing values passed mutably to 'args'").
    var q = _up(ctx, Int(d[unsafe_offset=0]), n)
    var area = _up(ctx, Int(d[unsafe_offset=1]), n)
    var cl = _up(ctx, Int(d[unsafe_offset=2]), n)
    var cd = _up(ctx, Int(d[unsafe_offset=3]), n)
    var lift_axis = _up(ctx, Int(d[unsafe_offset=4]), n * 3)
    var d_hat = _up(ctx, Int(d[unsafe_offset=5]), n * 3)
    var f_bluff = _up(ctx, Int(d[unsafe_offset=6]), n * 3)
    var c_rot = _up(ctx, Int(d[unsafe_offset=7]), n)
    var rho = _up(ctx, Int(d[unsafe_offset=8]), n)
    var uu = _up(ctx, Int(d[unsafe_offset=9]), n)
    var omega = _up(ctx, Int(d[unsafe_offset=10]), n * 3)
    var s_hat = _up(ctx, Int(d[unsafe_offset=11]), n * 3)
    var chord = _up(ctx, Int(d[unsafe_offset=12]), n)
    var dr = _up(ctx, Int(d[unsafe_offset=13]), n)
    var volb = _up(ctx, Int(d[unsafe_offset=14]), n)
    var subf = _up(ctx, Int(d[unsafe_offset=15]), n)
    var fza = _up(ctx, Int(d[unsafe_offset=16]), n)

    var wing = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=wing, src_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=17])))

    var f_o = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var l_o = ctx.enqueue_create_buffer[DType.float64](n)
    var d_o = ctx.enqueue_create_buffer[DType.float64](n)
    var b_o = ctx.enqueue_create_buffer[DType.float64](n)

    ctx.enqueue_function[assembly_kernel](
        q.unsafe_ptr(), area.unsafe_ptr(), cl.unsafe_ptr(), cd.unsafe_ptr(),
        lift_axis.unsafe_ptr(), d_hat.unsafe_ptr(), f_bluff.unsafe_ptr(),
        c_rot.unsafe_ptr(), rho.unsafe_ptr(), uu.unsafe_ptr(),
        omega.unsafe_ptr(), s_hat.unsafe_ptr(), chord.unsafe_ptr(),
        dr.unsafe_ptr(), volb.unsafe_ptr(), subf.unsafe_ptr(),
        fza.unsafe_ptr(), wing.unsafe_ptr(),
        f_o.unsafe_ptr(), l_o.unsafe_ptr(), d_o.unsafe_ptr(), b_o.unsafe_ptr(),
        Float64(py=scalars[0]), Float64(py=scalars[1]), Float64(py=scalars[2]),
        Int32(n), grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=18])), src_buf=f_o)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=19])), src_buf=l_o)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=20])), src_buf=d_o)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=21])), src_buf=b_o)
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit_assembly_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("assembly_gpu")
        m.def_function[assembly]("assembly")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
