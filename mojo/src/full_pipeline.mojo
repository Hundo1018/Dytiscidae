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

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from fluid_gpu import (
    added_mass_kernel, bluff_kernel, coeff_kernel, kin_kernel, strip_kernel,
    velocity_kernel,
)
from medium_gpu import medium_kernel
from assembly_gpu import assembly_kernel
from scatter_gpu import fmag_kernel, gather_body_kernel

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
    var xpos: DeviceBuffer[F64]
    var xmat: DeviceBuffer[F64]
    var xipos: DeviceBuffer[F64]
    var vel6: DeviceBuffer[F64]

    # intermediates
    var pos_w: DeviceBuffer[F64]
    var s_hat: DeviceBuffer[F64]
    var c_hat: DeviceBuffer[F64]
    var n_hat: DeviceBuffer[F64]
    var rho: DeviceBuffer[F64]
    var mu: DeviceBuffer[F64]
    var subf: DeviceBuffer[F64]
    var u_flow: DeviceBuffer[F64]
    var omega: DeviceBuffer[F64]
    var v_rel: DeviceBuffer[F64]
    var d_full: DeviceBuffer[F64]
    var u: DeviceBuffer[F64]
    var d_hat: DeviceBuffer[F64]
    var q: DeviceBuffer[F64]
    var re: DeviceBuffer[F64]
    var alpha: DeviceBuffer[F64]
    var rf: DeviceBuffer[F64]
    var lift_axis: DeviceBuffer[F64]
    var cl: DeviceBuffer[F64]
    var cd: DeviceBuffer[F64]
    var f_bluff: DeviceBuffer[F64]
    var d_bluff: DeviceBuffer[F64]
    var m_add: DeviceBuffer[F64]
    var vn: DeviceBuffer[F64]
    var fz: DeviceBuffer[F64]
    var force: DeviceBuffer[F64]
    var lift: DeviceBuffer[F64]
    var drag: DeviceBuffer[F64]
    var buoy: DeviceBuffer[F64]
    var fmag: DeviceBuffer[F64]
    var fmax: DeviceBuffer[F64]

    # outputs
    var xfrc: DeviceBuffer[F64]
    var m_body: DeviceBuffer[F64]
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
        self.xpos = c.enqueue_create_buffer[F64](cap_b * 3)
        self.xmat = c.enqueue_create_buffer[F64](cap_b * 9)
        self.xipos = c.enqueue_create_buffer[F64](cap_b * 3)
        self.vel6 = c.enqueue_create_buffer[F64](cap_b * 6)
        self.pos_w = c.enqueue_create_buffer[F64](cap_p * 3)
        self.s_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.c_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.n_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.rho = c.enqueue_create_buffer[F64](cap_p)
        self.mu = c.enqueue_create_buffer[F64](cap_p)
        self.subf = c.enqueue_create_buffer[F64](cap_p)
        self.u_flow = c.enqueue_create_buffer[F64](cap_p * 3)
        self.omega = c.enqueue_create_buffer[F64](cap_p * 3)
        self.v_rel = c.enqueue_create_buffer[F64](cap_p * 3)
        self.d_full = c.enqueue_create_buffer[F64](cap_p * 3)
        self.u = c.enqueue_create_buffer[F64](cap_p)
        self.d_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.q = c.enqueue_create_buffer[F64](cap_p)
        self.re = c.enqueue_create_buffer[F64](cap_p)
        self.alpha = c.enqueue_create_buffer[F64](cap_p)
        self.rf = c.enqueue_create_buffer[F64](cap_p)
        self.lift_axis = c.enqueue_create_buffer[F64](cap_p * 3)
        self.cl = c.enqueue_create_buffer[F64](cap_p)
        self.cd = c.enqueue_create_buffer[F64](cap_p)
        self.f_bluff = c.enqueue_create_buffer[F64](cap_p * 3)
        self.d_bluff = c.enqueue_create_buffer[F64](cap_p)
        self.m_add = c.enqueue_create_buffer[F64](cap_p)
        self.vn = c.enqueue_create_buffer[F64](cap_p)
        self.fz = c.enqueue_create_buffer[F64](cap_p)
        self.force = c.enqueue_create_buffer[F64](cap_p * 3)
        self.lift = c.enqueue_create_buffer[F64](cap_p)
        self.drag = c.enqueue_create_buffer[F64](cap_p)
        self.buoy = c.enqueue_create_buffer[F64](cap_p)
        self.fmag = c.enqueue_create_buffer[F64](cap_p)
        self.fmax = c.enqueue_create_buffer[F64](cap_m)
        self.xfrc = c.enqueue_create_buffer[F64](cap_b * 6)
        self.m_body = c.enqueue_create_buffer[F64](cap_b)
        self.clamped = c.enqueue_create_buffer[I32](cap_m)
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
    def step(self_ptr: UnsafePointer[Self, MutAnyOrigin], desc: PythonObject,
             scalars: PythonObject) raises -> PythonObject:
        """in : xpos, xmat, xipos, vel6
        out: xfrc, m_body, clamped, m_add, subf, alpha, q, lift, drag, buoy,
             vn
        then n, nbody, nmachine, has_bluff

        scalars (19): 0 amplitude, 1 wavelength, 2 period, 3 khat_x,
        4 khat_y, 5 t, 6 air_rho, 7 air_mu, 8 water_rho, 9 water_mu,
        10-12 current xyz, 13-15 wind xyz, 16 cd_scale, 17 am_scale,
        18 lift_scale
        """
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data))
        var n = Int(d[unsafe_offset=15])
        var nb = Int(d[unsafe_offset=16])
        var nm = Int(d[unsafe_offset=17])
        var has_bluff = Int(d[unsafe_offset=18])
        ref s = self_ptr[]
        if n > s.cap_p or nb > s.cap_b or nm > s.cap_m:
            raise Error("FullPipeline: batch exceeds capacity")
        var ctx = s.ctx

        _up_f64(ctx, s.xpos, Int(d[unsafe_offset=0]), nb * 3)
        _up_f64(ctx, s.xmat, Int(d[unsafe_offset=1]), nb * 9)
        _up_f64(ctx, s.xipos, Int(d[unsafe_offset=2]), nb * 3)
        _up_f64(ctx, s.vel6, Int(d[unsafe_offset=3]), nb * 6)
        s.m_body.create_sub_buffer[F64](0, nb).enqueue_fill(0.0)
        s.xfrc.create_sub_buffer[F64](0, nb * 6).enqueue_fill(0.0)
        s.clamped.create_sub_buffer[I32](0, nm).enqueue_fill(0)
        s.fmax.create_sub_buffer[F64](0, nm).enqueue_fill(0.0)

        var g = ceildiv(n, BLOCK)

        ctx.enqueue_function[kin_kernel](
            s.xpos.unsafe_ptr(), s.xmat.unsafe_ptr(), s.body_id.unsafe_ptr(),
            s.pos_local.unsafe_ptr(), s.span_local.unsafe_ptr(),
            s.chord_local.unsafe_ptr(), s.normal_local.unsafe_ptr(),
            s.pos_w.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[medium_kernel](
            s.pos_w.unsafe_ptr(), s.half_height.unsafe_ptr(),
            s.rho.unsafe_ptr(), s.mu.unsafe_ptr(), s.subf.unsafe_ptr(),
            s.u_flow.unsafe_ptr(),
            Float64(py=scalars[0]), Float64(py=scalars[1]),
            Float64(py=scalars[2]), Float64(py=scalars[3]),
            Float64(py=scalars[4]), Float64(py=scalars[5]),
            Float64(py=scalars[6]), Float64(py=scalars[7]),
            Float64(py=scalars[8]), Float64(py=scalars[9]),
            Float64(py=scalars[10]), Float64(py=scalars[11]),
            Float64(py=scalars[12]), Float64(py=scalars[13]),
            Float64(py=scalars[14]), Float64(py=scalars[15]),
            Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[velocity_kernel](
            s.vel6.unsafe_ptr(), s.xpos.unsafe_ptr(), s.pos_w.unsafe_ptr(),
            s.u_flow.unsafe_ptr(), s.body_id.unsafe_ptr(),
            s.omega.unsafe_ptr(), s.v_rel.unsafe_ptr(),
            Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[strip_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.omega.unsafe_ptr(), s.rho.unsafe_ptr(),
            s.mu.unsafe_ptr(), s.chord.unsafe_ptr(), s.camber.unsafe_ptr(),
            s.u.unsafe_ptr(), s.d_hat.unsafe_ptr(), s.q.unsafe_ptr(),
            s.re.unsafe_ptr(), s.alpha.unsafe_ptr(), s.rf.unsafe_ptr(),
            s.lift_axis.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[coeff_kernel](
            s.alpha.unsafe_ptr(), s.re.unsafe_ptr(), s.ar.unsafe_ptr(),
            s.rf.unsafe_ptr(), s.is_wing.unsafe_ptr(), s.cl.unsafe_ptr(),
            s.cd.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[bluff_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.rho.unsafe_ptr(), s.mu.unsafe_ptr(),
            s.ext.unsafe_ptr(), s.cd_bluff.unsafe_ptr(), s.is_wing.unsafe_ptr(),
            s.f_bluff.unsafe_ptr(), s.d_bluff.unsafe_ptr(),
            s.d_full.unsafe_ptr(), Float64(py=scalars[16]),
            Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[added_mass_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.d_full.unsafe_ptr(), s.rho.unsafe_ptr(),
            s.ext.unsafe_ptr(), s.chord.unsafe_ptr(), s.dr.unsafe_ptr(),
            s.volume.unsafe_ptr(), s.is_wing.unsafe_ptr(),
            s.body_id.unsafe_ptr(), s.m_add.unsafe_ptr(), s.vn.unsafe_ptr(),
            s.m_body.unsafe_ptr(), s.fz.unsafe_ptr(),
            Float64(py=scalars[17]), Int32(has_bluff),
            Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[assembly_kernel](
            s.q.unsafe_ptr(), s.area.unsafe_ptr(), s.cl.unsafe_ptr(),
            s.cd.unsafe_ptr(), s.lift_axis.unsafe_ptr(), s.d_hat.unsafe_ptr(),
            s.f_bluff.unsafe_ptr(), s.c_rot.unsafe_ptr(), s.rho.unsafe_ptr(),
            s.u.unsafe_ptr(), s.omega.unsafe_ptr(), s.s_hat.unsafe_ptr(),
            s.chord.unsafe_ptr(), s.dr.unsafe_ptr(), s.vol_buoy.unsafe_ptr(),
            s.subf.unsafe_ptr(), s.fz.unsafe_ptr(), s.is_wing.unsafe_ptr(),
            s.force.unsafe_ptr(), s.lift.unsafe_ptr(), s.drag.unsafe_ptr(),
            s.buoy.unsafe_ptr(),
            Float64(py=scalars[18]), Float64(py=scalars[16]),
            Float64(py=scalars[8]), Int32(n), grid_dim=g, block_dim=BLOCK)

        ctx.enqueue_function[fmag_kernel](
            s.force.unsafe_ptr(), s.machine.unsafe_ptr(), s.fmag.unsafe_ptr(),
            s.fmax.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)
        # One thread per body, summing its own panels in index order.  Replaces
        # two atomic scatters whose accumulation order was decided by warp
        # arrival, which made a whole evaluation irreproducible.
        ctx.enqueue_function[gather_body_kernel](
            s.m_add.unsafe_ptr(), s.force.unsafe_ptr(), s.fmag.unsafe_ptr(),
            s.machine.unsafe_ptr(), s.limit.unsafe_ptr(), s.fmax.unsafe_ptr(),
            s.pos_w.unsafe_ptr(), s.xipos.unsafe_ptr(),
            s.body_start.unsafe_ptr(), s.m_body.unsafe_ptr(),
            s.xfrc.unsafe_ptr(), s.clamped.unsafe_ptr(), Int32(nb),
            grid_dim=ceildiv(nb, BLOCK), block_dim=BLOCK)

        _dn_f64(ctx, s.xfrc, Int(d[unsafe_offset=4]), nb * 6)
        _dn_f64(ctx, s.m_body, Int(d[unsafe_offset=5]), nb)
        _dn_i32(ctx, s.clamped, Int(d[unsafe_offset=6]), nm)
        _dn_f64(ctx, s.m_add, Int(d[unsafe_offset=7]), n)
        _dn_f64(ctx, s.subf, Int(d[unsafe_offset=8]), n)
        _dn_f64(ctx, s.alpha, Int(d[unsafe_offset=9]), n)
        _dn_f64(ctx, s.q, Int(d[unsafe_offset=10]), n)
        _dn_f64(ctx, s.lift, Int(d[unsafe_offset=11]), n)
        _dn_f64(ctx, s.drag, Int(d[unsafe_offset=12]), n)
        _dn_f64(ctx, s.buoy, Int(d[unsafe_offset=13]), n)
        # vn is the normal-velocity component the slam diagnostic differences
        # against.  It is cheap to carry and expensive to be without: the
        # transition score reads slam as the entry load, so omitting it zeroes
        # every crossing's shock term without any error being raised.
        _dn_f64(ctx, s.vn, Int(d[unsafe_offset=14]), n)
        ctx.synchronize()
        return PythonObject(n)


@export
def PyInit_full_pipeline() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("full_pipeline")
        _ = (
            m.add_type[FullPipeline]("FullPipeline")
            .def_py_init[FullPipeline.py_init]()
            .def_method[FullPipeline.upload_static]("upload_static")
            .def_method[FullPipeline.step]("step")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
