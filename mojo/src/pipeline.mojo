"""The whole fluid pipeline in one call, intermediates never leaving the device.

Batching the panels was necessary and not sufficient.  Running the five ported
stages as five separate entry points measured *slower* than numpy up to ~100
machines -- 0.31x at 554 panels, 1.27x at 8864 -- because each stage allocated
its own buffers and did its own host round trip, so `s_hat` and friends were
copied to the host by one stage and straight back by the next.  Five round
trips and roughly thirty allocations per step, to do work the GPU finishes in
tens of microseconds.

So: static geometry uploads once at construction, per-step state uploads once,
five kernels run back to back over device-resident intermediates, and only the
results the caller actually reads come back.

`upload_static` is separate from the constructor because the panel geometry is
fixed for a rollout but the *batch* is not -- a machine whose battery runs flat
drops out, and the survivors re-pack.  Re-uploading geometry is a few hundred
microseconds against a rollout of thousands of steps.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from fluid_gpu import (
    added_mass_kernel, bluff_kernel, coeff_kernel, kin_kernel, strip_kernel,
)

comptime BLOCK = 128
comptime F64 = DType.float64
comptime I32 = DType.int32


struct Pipeline(Movable, Writable):
    var ctx: DeviceContext
    var cap_p: Int
    var cap_b: Int

    # static, uploaded once per batch composition
    var body_id: DeviceBuffer[I32]
    var pos_local: DeviceBuffer[F64]
    var span_local: DeviceBuffer[F64]
    var chord_local: DeviceBuffer[F64]
    var normal_local: DeviceBuffer[F64]
    var chord: DeviceBuffer[F64]
    var camber: DeviceBuffer[F64]
    var dr: DeviceBuffer[F64]
    var volume: DeviceBuffer[F64]
    var ext: DeviceBuffer[F64]
    var cd_bluff: DeviceBuffer[F64]
    var ar: DeviceBuffer[F64]
    var is_wing: DeviceBuffer[I32]

    # per-step inputs
    var xpos: DeviceBuffer[F64]
    var xmat: DeviceBuffer[F64]
    var v_rel: DeviceBuffer[F64]
    var omega: DeviceBuffer[F64]
    var rho: DeviceBuffer[F64]
    var mu: DeviceBuffer[F64]

    # device-resident intermediates: these are the ones that used to round trip
    var pos_w: DeviceBuffer[F64]
    var s_hat: DeviceBuffer[F64]
    var c_hat: DeviceBuffer[F64]
    var n_hat: DeviceBuffer[F64]
    var d_full: DeviceBuffer[F64]
    var u: DeviceBuffer[F64]
    var d_hat: DeviceBuffer[F64]
    var q: DeviceBuffer[F64]
    var re: DeviceBuffer[F64]
    var alpha: DeviceBuffer[F64]
    var rf: DeviceBuffer[F64]
    var lift_axis: DeviceBuffer[F64]
    var vn: DeviceBuffer[F64]

    # outputs the caller reads
    var cl: DeviceBuffer[F64]
    var cd: DeviceBuffer[F64]
    var f_bluff: DeviceBuffer[F64]
    var d_bluff: DeviceBuffer[F64]
    var m_add: DeviceBuffer[F64]
    var m_body: DeviceBuffer[F64]
    var fz: DeviceBuffer[F64]

    def __init__(out self, cap_p: Int, cap_b: Int) raises:
        self.ctx = DeviceContext()
        self.cap_p = cap_p
        self.cap_b = cap_b
        var c = self.ctx
        self.body_id = c.enqueue_create_buffer[I32](cap_p)
        self.is_wing = c.enqueue_create_buffer[I32](cap_p)
        self.pos_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.span_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.chord_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.normal_local = c.enqueue_create_buffer[F64](cap_p * 3)
        self.ext = c.enqueue_create_buffer[F64](cap_p * 3)
        self.chord = c.enqueue_create_buffer[F64](cap_p)
        self.camber = c.enqueue_create_buffer[F64](cap_p)
        self.dr = c.enqueue_create_buffer[F64](cap_p)
        self.volume = c.enqueue_create_buffer[F64](cap_p)
        self.cd_bluff = c.enqueue_create_buffer[F64](cap_p)
        self.ar = c.enqueue_create_buffer[F64](cap_p)
        self.xpos = c.enqueue_create_buffer[F64](cap_b * 3)
        self.xmat = c.enqueue_create_buffer[F64](cap_b * 9)
        self.v_rel = c.enqueue_create_buffer[F64](cap_p * 3)
        self.omega = c.enqueue_create_buffer[F64](cap_p * 3)
        self.rho = c.enqueue_create_buffer[F64](cap_p)
        self.mu = c.enqueue_create_buffer[F64](cap_p)
        self.pos_w = c.enqueue_create_buffer[F64](cap_p * 3)
        self.s_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.c_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.n_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.d_full = c.enqueue_create_buffer[F64](cap_p * 3)
        self.d_hat = c.enqueue_create_buffer[F64](cap_p * 3)
        self.lift_axis = c.enqueue_create_buffer[F64](cap_p * 3)
        self.f_bluff = c.enqueue_create_buffer[F64](cap_p * 3)
        self.u = c.enqueue_create_buffer[F64](cap_p)
        self.q = c.enqueue_create_buffer[F64](cap_p)
        self.re = c.enqueue_create_buffer[F64](cap_p)
        self.alpha = c.enqueue_create_buffer[F64](cap_p)
        self.rf = c.enqueue_create_buffer[F64](cap_p)
        self.vn = c.enqueue_create_buffer[F64](cap_p)
        self.cl = c.enqueue_create_buffer[F64](cap_p)
        self.cd = c.enqueue_create_buffer[F64](cap_p)
        self.d_bluff = c.enqueue_create_buffer[F64](cap_p)
        self.m_add = c.enqueue_create_buffer[F64](cap_p)
        self.fz = c.enqueue_create_buffer[F64](cap_p)
        self.m_body = c.enqueue_create_buffer[F64](cap_b)
        self.ctx.synchronize()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Pipeline(panels<=", self.cap_p, ", bodies<=", self.cap_b, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Pipeline(panels<=", self.cap_p, ", bodies<=", self.cap_b, ")")

    @staticmethod
    def py_init(out self: Pipeline, args: PythonObject,
                kwargs: PythonObject) raises:
        self = Self(Int(py=args[0]), Int(py=args[1]))

    @staticmethod
    def upload_static(self_ptr: UnsafePointer[Self, MutAnyOrigin],
                      desc: PythonObject) raises -> PythonObject:
        """Panel geometry, which is fixed until the batch composition changes.

        `desc`: body_id, is_wing, pos_local, span_local, chord_local,
        normal_local, ext, chord, camber, dr, volume, cd_bluff, ar, n
        """
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data))
        var n = Int(d[unsafe_offset=13])
        ref s = self_ptr[]
        if n > s.cap_p:
            raise Error("Pipeline: ", n, " panels exceeds capacity ", s.cap_p)
        _up_i32(s.ctx, s.body_id, Int(d[unsafe_offset=0]), n)
        _up_i32(s.ctx, s.is_wing, Int(d[unsafe_offset=1]), n)
        _up_f64(s.ctx, s.pos_local, Int(d[unsafe_offset=2]), n * 3)
        _up_f64(s.ctx, s.span_local, Int(d[unsafe_offset=3]), n * 3)
        _up_f64(s.ctx, s.chord_local, Int(d[unsafe_offset=4]), n * 3)
        _up_f64(s.ctx, s.normal_local, Int(d[unsafe_offset=5]), n * 3)
        _up_f64(s.ctx, s.ext, Int(d[unsafe_offset=6]), n * 3)
        _up_f64(s.ctx, s.chord, Int(d[unsafe_offset=7]), n)
        _up_f64(s.ctx, s.camber, Int(d[unsafe_offset=8]), n)
        _up_f64(s.ctx, s.dr, Int(d[unsafe_offset=9]), n)
        _up_f64(s.ctx, s.volume, Int(d[unsafe_offset=10]), n)
        _up_f64(s.ctx, s.cd_bluff, Int(d[unsafe_offset=11]), n)
        _up_f64(s.ctx, s.ar, Int(d[unsafe_offset=12]), n)
        s.ctx.synchronize()
        return PythonObject(n)

    @staticmethod
    def step(self_ptr: UnsafePointer[Self, MutAnyOrigin],
             desc: PythonObject, scales: PythonObject) raises -> PythonObject:
        """One step for the whole batch: upload, five kernels, download.

        `desc` in : xpos, xmat, v_rel, omega, rho, mu
        `desc` out: cl, cd, F_bluff, D_bluff, m_add, m_body, fz, alpha, q,
                    lift_axis, d_hat, U, vn, pos_w, s_hat, c_hat, n_hat
        then n, nbody.  `scales`: (cd_scale, added_mass_scale, has_bluff).
        """
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data))
        var n = Int(d[unsafe_offset=23])
        var nb = Int(d[unsafe_offset=24])
        ref s = self_ptr[]
        if n > s.cap_p or nb > s.cap_b:
            raise Error("Pipeline: batch exceeds capacity")
        var ctx = s.ctx

        _up_f64(ctx, s.xpos, Int(d[unsafe_offset=0]), nb * 3)
        _up_f64(ctx, s.xmat, Int(d[unsafe_offset=1]), nb * 9)
        _up_f64(ctx, s.v_rel, Int(d[unsafe_offset=2]), n * 3)
        _up_f64(ctx, s.omega, Int(d[unsafe_offset=3]), n * 3)
        _up_f64(ctx, s.rho, Int(d[unsafe_offset=4]), n)
        _up_f64(ctx, s.mu, Int(d[unsafe_offset=5]), n)
        s.m_body.create_sub_buffer[F64](0, nb).enqueue_fill(0.0)

        var g = ceildiv(n, BLOCK)
        ctx.enqueue_function[kin_kernel](
            s.xpos.unsafe_ptr(), s.xmat.unsafe_ptr(), s.body_id.unsafe_ptr(),
            s.pos_local.unsafe_ptr(), s.span_local.unsafe_ptr(),
            s.chord_local.unsafe_ptr(), s.normal_local.unsafe_ptr(),
            s.pos_w.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)
        ctx.enqueue_function[strip_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.omega.unsafe_ptr(), s.rho.unsafe_ptr(),
            s.mu.unsafe_ptr(), s.chord.unsafe_ptr(), s.camber.unsafe_ptr(),
            s.u.unsafe_ptr(), s.d_hat.unsafe_ptr(), s.q.unsafe_ptr(),
            s.re.unsafe_ptr(), s.alpha.unsafe_ptr(), s.rf.unsafe_ptr(),
            s.lift_axis.unsafe_ptr(), Int32(n), grid_dim=g, block_dim=BLOCK)
        ctx.enqueue_function[coeff_kernel](
            s.alpha.unsafe_ptr(), s.re.unsafe_ptr(), s.ar.unsafe_ptr(),
            s.rf.unsafe_ptr(), s.is_wing.unsafe_ptr(),
            s.cl.unsafe_ptr(), s.cd.unsafe_ptr(), Int32(n),
            grid_dim=g, block_dim=BLOCK)
        ctx.enqueue_function[bluff_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.rho.unsafe_ptr(), s.mu.unsafe_ptr(),
            s.ext.unsafe_ptr(), s.cd_bluff.unsafe_ptr(), s.is_wing.unsafe_ptr(),
            s.f_bluff.unsafe_ptr(), s.d_bluff.unsafe_ptr(), s.d_full.unsafe_ptr(),
            Float64(py=scales[0]), Int32(n), grid_dim=g, block_dim=BLOCK)
        ctx.enqueue_function[added_mass_kernel](
            s.v_rel.unsafe_ptr(), s.s_hat.unsafe_ptr(), s.c_hat.unsafe_ptr(),
            s.n_hat.unsafe_ptr(), s.d_full.unsafe_ptr(), s.rho.unsafe_ptr(),
            s.ext.unsafe_ptr(), s.chord.unsafe_ptr(), s.dr.unsafe_ptr(),
            s.volume.unsafe_ptr(), s.is_wing.unsafe_ptr(), s.body_id.unsafe_ptr(),
            s.m_add.unsafe_ptr(), s.vn.unsafe_ptr(), s.m_body.unsafe_ptr(),
            s.fz.unsafe_ptr(),
            Float64(py=scales[1]), Int32(Int(py=scales[2])), Int32(n),
            grid_dim=g, block_dim=BLOCK)

        _dn_f64(ctx, s.cl, Int(d[unsafe_offset=6]), n)
        _dn_f64(ctx, s.cd, Int(d[unsafe_offset=7]), n)
        _dn_f64(ctx, s.f_bluff, Int(d[unsafe_offset=8]), n * 3)
        _dn_f64(ctx, s.d_bluff, Int(d[unsafe_offset=9]), n)
        _dn_f64(ctx, s.m_add, Int(d[unsafe_offset=10]), n)
        _dn_f64(ctx, s.m_body, Int(d[unsafe_offset=11]), nb)
        _dn_f64(ctx, s.fz, Int(d[unsafe_offset=12]), n)
        _dn_f64(ctx, s.alpha, Int(d[unsafe_offset=13]), n)
        _dn_f64(ctx, s.q, Int(d[unsafe_offset=14]), n)
        _dn_f64(ctx, s.lift_axis, Int(d[unsafe_offset=15]), n * 3)
        _dn_f64(ctx, s.d_hat, Int(d[unsafe_offset=16]), n * 3)
        _dn_f64(ctx, s.u, Int(d[unsafe_offset=17]), n)
        _dn_f64(ctx, s.vn, Int(d[unsafe_offset=18]), n)
        _dn_f64(ctx, s.pos_w, Int(d[unsafe_offset=19]), n * 3)
        _dn_f64(ctx, s.s_hat, Int(d[unsafe_offset=20]), n * 3)
        _dn_f64(ctx, s.c_hat, Int(d[unsafe_offset=21]), n * 3)
        _dn_f64(ctx, s.n_hat, Int(d[unsafe_offset=22]), n * 3)
        ctx.synchronize()
        return PythonObject(n)


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


@export
def PyInit_pipeline() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("pipeline")
        _ = (
            m.add_type[Pipeline]("Pipeline")
            .def_py_init[Pipeline.py_init]()
            .def_method[Pipeline.upload_static]("upload_static")
            .def_method[Pipeline.step]("step")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
