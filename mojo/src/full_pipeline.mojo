"""The whole of fluid.apply on the device, for a batch of machines.

Nine kernels, one upload of the per-step state, one download of what the caller
needs. Everything between stays in device memory.

What crosses the boundary each step, and why:

  in   xpos, xmat   body poses, from mj_forward
       vel6         per-body 6D velocity, from the mj_objectVelocity loop.
                    That loop is CPU-only -- see velocity_kernel's docstring --
                    and is the one part of apply() that cannot move.
       xipos        body CoM, for the torque arm
  out  xfrc         (nbody, 6) force and torque, which mj_step reads next
       m_body       per-body added mass, which the host folds into
                    model.body_mass and model.body_inertia
       diagnostics  clamped, and the per-machine reductions

`upload_static` carries the panel geometry, which is fixed for as long as the
batch composition is. A machine dropping out on a flat battery re-packs the
batch and needs one more upload; that is a few hundred microseconds against a
rollout of thousands of steps.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from fluid_gpu import (
    added_mass_at, bluff_at, coeff_at, kin_at, strip_at, velocity_at,
    unsteady_at,
)
from medium_gpu import medium_at
from assembly_gpu import assembly_at
from scatter_gpu import fmag_at, gather_body_kernel

comptime BLOCK = 128
comptime F64 = DType.float64
comptime I32 = DType.int32


@always_inline
def _up_f64(ctx: DeviceContext, buf: DeviceBuffer[F64], addr: Int,
            count: Int) raises:
    ctx.enqueue_copy(
        dst_buf=buf.create_sub_buffer[F64](0, count),
        src_ptr=UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr))


@always_inline
def _up_i32(ctx: DeviceContext, buf: DeviceBuffer[I32], addr: Int,
            count: Int) raises:
    ctx.enqueue_copy(
        dst_buf=buf.create_sub_buffer[I32](0, count),
        src_ptr=UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr))


@always_inline
def _dn_f64(ctx: DeviceContext, buf: DeviceBuffer[F64], addr: Int,
            count: Int) raises:
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr),
        src_buf=buf.create_sub_buffer[F64](0, count))


@always_inline
def _dn_i32(ctx: DeviceContext, buf: DeviceBuffer[I32], addr: Int,
            count: Int) raises:
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr),
        src_buf=buf.create_sub_buffer[I32](0, count))


def panel_kernel(
    xpos: UnsafePointer[Float64, MutAnyOrigin],
    xmat: UnsafePointer[Float64, MutAnyOrigin],
    body_id: UnsafePointer[Int32, MutAnyOrigin],
    pos_local: UnsafePointer[Float64, MutAnyOrigin],
    span_local: UnsafePointer[Float64, MutAnyOrigin],
    chord_local: UnsafePointer[Float64, MutAnyOrigin],
    normal_local: UnsafePointer[Float64, MutAnyOrigin],
    pos_w: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    c_hat: UnsafePointer[Float64, MutAnyOrigin],
    n_hat: UnsafePointer[Float64, MutAnyOrigin],
    half_height: UnsafePointer[Float64, MutAnyOrigin],
    rho: UnsafePointer[Float64, MutAnyOrigin],
    mu: UnsafePointer[Float64, MutAnyOrigin],
    subf: UnsafePointer[Float64, MutAnyOrigin],
    u_flow: UnsafePointer[Float64, MutAnyOrigin],
    sc0: Float64,
    sc1: Float64,
    sc2: Float64,
    sc3: Float64,
    sc4: Float64,
    sc5: Float64,
    sc6: Float64,
    sc7: Float64,
    sc8: Float64,
    sc9: Float64,
    sc10: Float64,
    sc11: Float64,
    sc12: Float64,
    sc13: Float64,
    sc14: Float64,
    sc15: Float64,
    vel6: UnsafePointer[Float64, MutAnyOrigin],
    xipos: UnsafePointer[Float64, MutAnyOrigin],
    omega: UnsafePointer[Float64, MutAnyOrigin],
    v_rel: UnsafePointer[Float64, MutAnyOrigin],
    machine: UnsafePointer[Int32, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    v_ind: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    camber: UnsafePointer[Float64, MutAnyOrigin],
    u: UnsafePointer[Float64, MutAnyOrigin],
    d_hat: UnsafePointer[Float64, MutAnyOrigin],
    q: UnsafePointer[Float64, MutAnyOrigin],
    re: UnsafePointer[Float64, MutAnyOrigin],
    alpha: UnsafePointer[Float64, MutAnyOrigin],
    rf: UnsafePointer[Float64, MutAnyOrigin],
    lift_axis: UnsafePointer[Float64, MutAnyOrigin],
    st_x0: UnsafePointer[Float64, MutAnyOrigin],
    st_x1: UnsafePointer[Float64, MutAnyOrigin],
    st_s: UnsafePointer[Float64, MutAnyOrigin],
    st_rev: UnsafePointer[Float64, MutAnyOrigin],
    st_primed: UnsafePointer[Float64, MutAnyOrigin],
    ae: UnsafePointer[Float64, MutAnyOrigin],
    lev: UnsafePointer[Float64, MutAnyOrigin],
    sc20: Float64,
    reset_unsteady: Int32,
    unsteady_on: Int32,
    ar: UnsafePointer[Float64, MutAnyOrigin],
    cl: UnsafePointer[Float64, MutAnyOrigin],
    cd: UnsafePointer[Float64, MutAnyOrigin],
    ext: UnsafePointer[Float64, MutAnyOrigin],
    cd_bluff: UnsafePointer[Float64, MutAnyOrigin],
    f_bluff: UnsafePointer[Float64, MutAnyOrigin],
    d_bluff: UnsafePointer[Float64, MutAnyOrigin],
    d_full: UnsafePointer[Float64, MutAnyOrigin],
    sc16: Float64,
    dr: UnsafePointer[Float64, MutAnyOrigin],
    volume: UnsafePointer[Float64, MutAnyOrigin],
    m_add: UnsafePointer[Float64, MutAnyOrigin],
    vn: UnsafePointer[Float64, MutAnyOrigin],
    m_body: UnsafePointer[Float64, MutAnyOrigin],
    fz: UnsafePointer[Float64, MutAnyOrigin],
    sc17: Float64,
    has_bluff: Int32,
    wing_tensor: Int32,
    area: UnsafePointer[Float64, MutAnyOrigin],
    c_rot: UnsafePointer[Float64, MutAnyOrigin],
    vol_buoy: UnsafePointer[Float64, MutAnyOrigin],
    force: UnsafePointer[Float64, MutAnyOrigin],
    lift: UnsafePointer[Float64, MutAnyOrigin],
    drag: UnsafePointer[Float64, MutAnyOrigin],
    buoy: UnsafePointer[Float64, MutAnyOrigin],
    sc18: Float64,
    fmag: UnsafePointer[Float64, MutAnyOrigin],
    fmax: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    """Every per-panel stage of a step in one launch: kinematics, medium,
    relative flow, strip theory, unsteady history, coefficients, bluff drag,
    added mass, assembly and force magnitude, in the order the eleven separate
    launches ran them.  Each stage reads only its own panel's values and
    per-body inputs, so one thread can run them back to back with nothing
    shared across threads.  Eleven launches were 90 us of a 215 us step for a
    190-panel shard, almost all of it launch latency (2026-09-27).  The stage
    bodies are the `*_at` functions the separate kernels also call, so there
    is one copy of the physics.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    kin_at(i, xpos, xmat, body_id, pos_local, span_local, chord_local, normal_local, pos_w, s_hat, c_hat, n_hat)
    medium_at(i, pos_w, half_height, rho, mu, subf, u_flow, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14, sc15)
    velocity_at(i, vel6, xipos, pos_w, u_flow, body_id, omega, v_rel, machine, is_wing, v_ind)
    strip_at(i, v_rel, s_hat, c_hat, n_hat, omega, rho, mu, chord, camber, u, d_hat, q, re, alpha, rf, lift_axis)
    unsteady_at(i, alpha, rf, u, chord, omega, s_hat, st_x0, st_x1, st_s, st_rev, st_primed, ae, lev, sc20, reset_unsteady, unsteady_on)
    coeff_at(i, alpha, re, ar, lev, is_wing, cl, cd, ae)
    bluff_at(i, v_rel, s_hat, c_hat, n_hat, rho, mu, ext, cd_bluff, is_wing, f_bluff, d_bluff, d_full, sc16)
    added_mass_at(i, v_rel, s_hat, c_hat, n_hat, d_full, rho, ext, chord, dr, volume, is_wing, body_id, m_add, vn, m_body, fz, sc17, has_bluff, wing_tensor)
    assembly_at(i, q, area, cl, cd, lift_axis, d_hat, f_bluff, c_rot, rho, u, omega, s_hat, chord, dr, vol_buoy, subf, fz, is_wing, force, lift, drag, buoy, sc18, sc16, sc8)
    fmag_at(i, force, machine, fmag, fmax)


struct FullPipeline(Movable, Writable):
    var ctx: DeviceContext
    var cap_p: Int
    var cap_b: Int
    var cap_m: Int

    # static geometry
    var body_id: DeviceBuffer[I32]
    var body_start: DeviceBuffer[I32]
    var machine: DeviceBuffer[I32]
    var is_wing: DeviceBuffer[I32]
    var pos_local: DeviceBuffer[F64]
    var span_local: DeviceBuffer[F64]
    var chord_local: DeviceBuffer[F64]
    var normal_local: DeviceBuffer[F64]
    var ext: DeviceBuffer[F64]
    var chord: DeviceBuffer[F64]
    var camber: DeviceBuffer[F64]
    var dr: DeviceBuffer[F64]
    var area: DeviceBuffer[F64]
    var volume: DeviceBuffer[F64]
    var vol_buoy: DeviceBuffer[F64]
    var half_height: DeviceBuffer[F64]
    var cd_bluff: DeviceBuffer[F64]
    var ar: DeviceBuffer[F64]
    var c_rot: DeviceBuffer[F64]
    var limit: DeviceBuffer[F64]

    # per-step inputs

    # intermediates
    var s_hat: DeviceBuffer[F64]
    var c_hat: DeviceBuffer[F64]
    var n_hat: DeviceBuffer[F64]
    var mu: DeviceBuffer[F64]
    var u_flow: DeviceBuffer[F64]
    var omega: DeviceBuffer[F64]
    var v_rel: DeviceBuffer[F64]
    var d_full: DeviceBuffer[F64]
    var u: DeviceBuffer[F64]
    var d_hat: DeviceBuffer[F64]
    var re: DeviceBuffer[F64]
    var rf: DeviceBuffer[F64]
    var lift_axis: DeviceBuffer[F64]
    var cl: DeviceBuffer[F64]
    var cd: DeviceBuffer[F64]
    var f_bluff: DeviceBuffer[F64]
    var fz: DeviceBuffer[F64]
    var fmax: DeviceBuffer[F64]
    # 2026-09-23: unsteady history (Wagner, LEV travel) and the inflow.
    var ae: DeviceBuffer[F64]
    var lev: DeviceBuffer[F64]
    var st_x0: DeviceBuffer[F64]
    var st_x1: DeviceBuffer[F64]
    var st_s: DeviceBuffer[F64]
    var st_rev: DeviceBuffer[F64]
    var st_primed: DeviceBuffer[F64]

    # Every per-step input in one block and every output the host reads in
    # another, so a step is one upload and one download.  It was 5 uploads, 4
    # fills and 17 downloads of pageable memory: 210 of the 382 us a step took
    # for a 4-machine shard, against 101 us for all eleven kernels (measured
    # 2026-09-27).  Layout: `_in_offsets` / `_out_offsets`, and batchroll's
    # BatchedFluid packs its numpy views the same way.
    var inbuf: DeviceBuffer[F64]
    var outbuf: DeviceBuffer[F64]
    # Where `wait` downloads the output block to, and how much of it: set by
    # `launch`.  The host blocks are ordinary numpy memory, not pinned: Mojo's
    # first host buffer reserves a ~1.35 GB pinned pool per process (measured
    # 2026-09-27), five processes' worth of unswappable memory on a 16 GB
    # machine, and it tripped arch44's memory ceiling at generation 0.  A small
    # upload from pageable memory is staged and returns, so the kernels still
    # run while the host works; only the download waits, and it is in `wait`.
    var host_out: Int
    var n_out: Int
    # Written by gather_body_kernel and never read: the limiter is on the host.
    var clamped: DeviceBuffer[I32]

    def __init__(out self, cap_p: Int, cap_b: Int, cap_m: Int) raises:
        self.ctx = DeviceContext()
        self.cap_p = cap_p
        self.cap_b = cap_b
        self.cap_m = cap_m
        var c = self.ctx
        self.body_id = c.enqueue_create_buffer[I32](cap_p)
        self.body_start = c.enqueue_create_buffer[I32](cap_b + 1)
        self.machine = c.enqueue_create_buffer[I32](cap_p)
        self.is_wing = c.enqueue_create_buffer[I32](cap_p)
        self.pos_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.span_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.chord_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.normal_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.ext = c.enqueue_create_buffer[F64](cap_p * 3)
        self.chord = c.enqueue_create_buffer[F64](cap_p)
        self.camber = c.enqueue_create_buffer[F64](cap_p)
        self.dr = c.enqueue_create_buffer[F64](cap_p)
        self.area = c.enqueue_create_buffer[F64](cap_p)
        self.volume = c.enqueue_create_buffer[F64](cap_p)
        self.vol_buoy = c.enqueue_create_buffer[F64](cap_p)
        self.half_height = c.enqueue_create_buffer[F64](cap_p)
        self.cd_bluff = c.enqueue_create_buffer[F64](cap_p)
        self.ar = c.enqueue_create_buffer[F64](cap_p)
        self.c_rot = c.enqueue_create_buffer[F64](cap_p)
        self.limit = c.enqueue_create_buffer[F64](cap_m)
        self.s_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.c_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.n_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.mu = c.enqueue_create_buffer[F64](cap_p)
        self.u_flow = c.enqueue_create_buffer[F64](cap_p * 3)
        self.omega = c.enqueue_create_buffer[F64](cap_p * 3)
        self.v_rel = c.enqueue_create_buffer[F64](cap_p * 3)
        self.d_full = c.enqueue_create_buffer[F64](cap_p * 3)
        self.u = c.enqueue_create_buffer[F64](cap_p)
        self.d_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.re = c.enqueue_create_buffer[F64](cap_p)
        self.rf = c.enqueue_create_buffer[F64](cap_p)
        self.lift_axis = c.enqueue_create_buffer[F64](cap_p * 3)
        self.cl = c.enqueue_create_buffer[F64](cap_p)
        self.cd = c.enqueue_create_buffer[F64](cap_p)
        self.f_bluff = c.enqueue_create_buffer[F64](cap_p * 3)
        self.fz = c.enqueue_create_buffer[F64](cap_p)
        self.fmax = c.enqueue_create_buffer[F64](cap_m)
        self.ae = c.enqueue_create_buffer[F64](cap_p)
        self.lev = c.enqueue_create_buffer[F64](cap_p)
        self.st_x0 = c.enqueue_create_buffer[F64](cap_p)
        self.st_x1 = c.enqueue_create_buffer[F64](cap_p)
        self.st_s = c.enqueue_create_buffer[F64](cap_p)
        self.st_rev = c.enqueue_create_buffer[F64](cap_p)
        self.st_primed = c.enqueue_create_buffer[F64](cap_p)
        self.st_primed.enqueue_fill(0.0)
        self.clamped = c.enqueue_create_buffer[I32](cap_m)
        self.inbuf = c.enqueue_create_buffer[F64](cap_b * 21 + cap_m * 3)
        self.outbuf = c.enqueue_create_buffer[F64](cap_b * 8 + cap_p * 17)
        self.host_out = 0
        self.n_out = 0
        self.ctx.synchronize()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FullPipeline(panels<=", self.cap_p, ", bodies<=",
                     self.cap_b, ", machines<=", self.cap_m, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: FullPipeline, args: PythonObject,
                kwargs: PythonObject) raises:
        self = Self(Int(py=args[0]), Int(py=args[1]), Int(py=args[2]))

    @staticmethod
    def upload_static(self_ptr: UnsafePointer[Self, MutAnyOrigin],
                      desc: PythonObject) raises -> PythonObject:
        """body_id, machine, is_wing, pos_local, span_local, chord_local,
        normal_local, ext, chord, camber, dr, area, volume, vol_buoy,
        half_height, cd_bluff, ar, c_rot, limit, body_start,
        then n, nmachine, nbody"""
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data))
        var n = Int(d[unsafe_offset=20])
        var nm = Int(d[unsafe_offset=21])
        var nb_static = Int(d[unsafe_offset=22])
        ref s = self_ptr[]
        if n > s.cap_p or nm > s.cap_m:
            raise Error("FullPipeline: batch exceeds capacity")
        _up_i32(s.ctx, s.body_id, Int(d[unsafe_offset=0]), n)
        _up_i32(s.ctx, s.machine, Int(d[unsafe_offset=1]), n)
        _up_i32(s.ctx, s.is_wing, Int(d[unsafe_offset=2]), n)
        _up_f64(s.ctx, s.pos_local, Int(d[unsafe_offset=3]), n * 3)
        _up_f64(s.ctx, s.span_local, Int(d[unsafe_offset=4]), n * 3)
        _up_f64(s.ctx, s.chord_local, Int(d[unsafe_offset=5]), n * 3)
        _up_f64(s.ctx, s.normal_local, Int(d[unsafe_offset=6]), n * 3)
        _up_f64(s.ctx, s.ext, Int(d[unsafe_offset=7]), n * 3)
        _up_f64(s.ctx, s.chord, Int(d[unsafe_offset=8]), n)
        _up_f64(s.ctx, s.camber, Int(d[unsafe_offset=9]), n)
        _up_f64(s.ctx, s.dr, Int(d[unsafe_offset=10]), n)
        _up_f64(s.ctx, s.area, Int(d[unsafe_offset=11]), n)
        _up_f64(s.ctx, s.volume, Int(d[unsafe_offset=12]), n)
        _up_f64(s.ctx, s.vol_buoy, Int(d[unsafe_offset=13]), n)
        _up_f64(s.ctx, s.half_height, Int(d[unsafe_offset=14]), n)
        _up_f64(s.ctx, s.cd_bluff, Int(d[unsafe_offset=15]), n)
        _up_f64(s.ctx, s.ar, Int(d[unsafe_offset=16]), n)
        _up_f64(s.ctx, s.c_rot, Int(d[unsafe_offset=17]), n)
        _up_f64(s.ctx, s.limit, Int(d[unsafe_offset=18]), nm)
        # CSR offsets, one per body plus a terminator.  Panels are contiguous
        # per body, so this is all the structure the deterministic gather needs.
        _up_i32(s.ctx, s.body_start, Int(d[unsafe_offset=19]), nb_static + 1)
        s.ctx.synchronize()
        return PythonObject(n)

    @staticmethod
    def launch(self_ptr: UnsafePointer[Self, MutAnyOrigin], desc: PythonObject) raises -> PythonObject:
        """Descriptor (int64): 0 host input block, 1 host output block, 2 scalars
        (float64[23]), 3 n, 4 nbody, 5 nmachine, 6 has_bluff.

        input block  (float64): xpos[nb*3] xmat[nb*9] xipos[nb*3] vel6[nb*6]
                                v_ind[nm*3]
        output block (float64): xfrc[nb*6] m_body[nb] fsum_b[nb], then per
                                panel m_add subf alpha q lift drag buoy vn fmag
                                rho d_bluff [n each], pos_w[n*3] force[n*3]

        scalars (23): 0 amplitude, 1 wavelength, 2 period, 3 khat_x,
        4 khat_y, 5 t, 6 air_rho, 7 air_mu, 8 water_rho, 9 water_mu,
        10-12 current xyz, 13-15 wind xyz, 16 cd_scale, 17 am_scale,
        18 lift_scale, 19 wing added-mass tensor (0/1), 20 dt,
        21 reset the unsteady history (0/1), 22 unsteady history on (0/1)
        """
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data))
        var n = Int(d[unsafe_offset=3])
        var nb = Int(d[unsafe_offset=4])
        var nm = Int(d[unsafe_offset=5])
        var has_bluff = Int(d[unsafe_offset=6])
        var host_in = UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=0]))
        var host_out = UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=1]))
        # The scalars are read from a float64 array rather than converted from
        # a Python tuple: 23 conversions a step, for numbers the host already
        # holds in memory.
        var sc = UnsafePointer[Float64, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=2]))
        ref s = self_ptr[]
        if n > s.cap_p or nb > s.cap_b or nm > s.cap_m:
            raise Error("FullPipeline: batch exceeds capacity")
        var ctx = s.ctx

        var n_in = nb * 21 + nm * 3
        var n_out = nb * 8 + n * 17
        ctx.enqueue_copy(dst_buf=s.inbuf.create_sub_buffer[F64](0, n_in),
                         src_ptr=host_in)
        var ib = s.inbuf.unsafe_ptr()
        var p_xpos = ib
        var p_xmat = ib + nb * 3
        var p_xipos = ib + nb * 12
        var p_vel6 = ib + nb * 15
        var p_v_ind = ib + nb * 21
        var ob = s.outbuf.unsafe_ptr()
        var p_xfrc = ob
        var p_m_body = ob + nb * 6
        var p_fsum_b = ob + nb * 7
        var pb = ob + nb * 8
        var p_m_add = pb
        var p_subf = pb + n
        var p_alpha = pb + n * 2
        var p_q = pb + n * 3
        var p_lift = pb + n * 4
        var p_drag = pb + n * 5
        var p_buoy = pb + n * 6
        var p_vn = pb + n * 7
        var p_fmag = pb + n * 8
        var p_rho = pb + n * 9
        var p_d_bluff = pb + n * 10
        var p_pos_w = pb + n * 11
        var p_force = pb + n * 14

        var g = ceildiv(n, BLOCK)

        ctx.enqueue_function[panel_kernel](
            p_xpos,
            p_xmat,
            s.body_id.unsafe_ptr(),
            s.pos_local.unsafe_ptr(),
            s.span_local.unsafe_ptr(),
            s.chord_local.unsafe_ptr(),
            s.normal_local.unsafe_ptr(),
            p_pos_w,
            s.s_hat.unsafe_ptr(),
            s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(),
            s.half_height.unsafe_ptr(),
            p_rho,
            s.mu.unsafe_ptr(),
            p_subf,
            s.u_flow.unsafe_ptr(),
            sc[unsafe_offset=0],
            sc[unsafe_offset=1],
            sc[unsafe_offset=2],
            sc[unsafe_offset=3],
            sc[unsafe_offset=4],
            sc[unsafe_offset=5],
            sc[unsafe_offset=6],
            sc[unsafe_offset=7],
            sc[unsafe_offset=8],
            sc[unsafe_offset=9],
            sc[unsafe_offset=10],
            sc[unsafe_offset=11],
            sc[unsafe_offset=12],
            sc[unsafe_offset=13],
            sc[unsafe_offset=14],
            sc[unsafe_offset=15],
            p_vel6,
            p_xipos,
            s.omega.unsafe_ptr(),
            s.v_rel.unsafe_ptr(),
            s.machine.unsafe_ptr(),
            s.is_wing.unsafe_ptr(),
            p_v_ind,
            s.chord.unsafe_ptr(),
            s.camber.unsafe_ptr(),
            s.u.unsafe_ptr(),
            s.d_hat.unsafe_ptr(),
            p_q,
            s.re.unsafe_ptr(),
            p_alpha,
            s.rf.unsafe_ptr(),
            s.lift_axis.unsafe_ptr(),
            s.st_x0.unsafe_ptr(),
            s.st_x1.unsafe_ptr(),
            s.st_s.unsafe_ptr(),
            s.st_rev.unsafe_ptr(),
            s.st_primed.unsafe_ptr(),
            s.ae.unsafe_ptr(),
            s.lev.unsafe_ptr(),
            sc[unsafe_offset=20],
            Int32(Int(sc[unsafe_offset=21])),
            Int32(Int(sc[unsafe_offset=22])),
            s.ar.unsafe_ptr(),
            s.cl.unsafe_ptr(),
            s.cd.unsafe_ptr(),
            s.ext.unsafe_ptr(),
            s.cd_bluff.unsafe_ptr(),
            s.f_bluff.unsafe_ptr(),
            p_d_bluff,
            s.d_full.unsafe_ptr(),
            sc[unsafe_offset=16],
            s.dr.unsafe_ptr(),
            s.volume.unsafe_ptr(),
            p_m_add,
            p_vn,
            p_m_body,
            s.fz.unsafe_ptr(),
            sc[unsafe_offset=17],
            Int32(has_bluff),
            Int32(Int(sc[unsafe_offset=19])),
            s.area.unsafe_ptr(),
            s.c_rot.unsafe_ptr(),
            s.vol_buoy.unsafe_ptr(),
            p_force,
            p_lift,
            p_drag,
            p_buoy,
            sc[unsafe_offset=18],
            p_fmag,
            s.fmax.unsafe_ptr(),
            Int32(n), grid_dim=g, block_dim=BLOCK)
        # One thread per body, summing its own panels in index order.  Replaces
        # two atomic scatters whose accumulation order was decided by warp
        # arrival, which made a whole evaluation irreproducible.
        ctx.enqueue_function[gather_body_kernel](
            p_m_add, p_force, p_fmag,
            s.machine.unsafe_ptr(), s.limit.unsafe_ptr(), s.fmax.unsafe_ptr(),
            p_pos_w, p_xipos,
            s.body_start.unsafe_ptr(), p_m_body,
            p_xfrc, s.clamped.unsafe_ptr(),
            p_fsum_b, Int32(nb),
            grid_dim=ceildiv(nb, BLOCK), block_dim=BLOCK)

        s.host_out = Int(host_out)
        s.n_out = n_out
        return PythonObject(n)

    @staticmethod
    def wait(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> PythonObject:
        """Download the last `launch`'s outputs into the host block; returns
        once they are there."""
        ref s = self_ptr[]
        s.ctx.enqueue_copy(
            dst_ptr=UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s.host_out),
            src_buf=s.outbuf.create_sub_buffer[F64](0, s.n_out))
        s.ctx.synchronize()
        return PythonObject(None)

    @staticmethod
    def step(self_ptr: UnsafePointer[Self, MutAnyOrigin], desc: PythonObject) raises -> PythonObject:
        """`launch` then `wait`."""
        var n = Self.launch(self_ptr, desc)
        _ = Self.wait(self_ptr)
        return n


@export
def PyInit_full_pipeline() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("full_pipeline")
        _ = (
            m.add_type[FullPipeline]("FullPipeline")
            .def_py_init[FullPipeline.py_init]()
            .def_method[FullPipeline.upload_static]("upload_static")
            .def_method[FullPipeline.step]("step")
            .def_method[FullPipeline.launch]("launch")
            .def_method[FullPipeline.wait]("wait")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
