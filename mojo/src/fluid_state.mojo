"""A persistent GPU solver: allocate once, step many times.

The per-call functions in `fluid_gpu` allocate their device buffers on every
invocation, which was deliberate while the kernels were being checked and is
the dominant cost now that they are.  Measured warm on this card:

    DeviceContext()          6.0 us
    allocate 8192 f64       97.5 us      <-- per buffer, per call
    reuse an existing one   29.7 us

`coefficients` allocates six buffers, so ~585 us of its measured ~520 us fixed
cost is allocation.  The context is not worth persisting; the buffers are all
of it.

That number is also what decides how large a batch has to be.  At 520 us fixed
the GPU did not beat numpy until ~1600 panels, about 23 machines.  Removing the
allocation moves the crossover down, and the batched evaluator should be sized
against the new one rather than the old.

Capacity is fixed at construction and the caller may not exceed it.  Growing on
demand would reintroduce exactly the allocation this exists to avoid, at an
unpredictable moment, so it raises instead.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from fluid_gpu import coeff_kernel

comptime BLOCK = 128


struct GpuFluid(Movable, Writable):
    """Device buffers sized once, reused every step."""

    var ctx: DeviceContext
    var capacity: Int
    var alpha: DeviceBuffer[DType.float64]
    var re: DeviceBuffer[DType.float64]
    var ar: DeviceBuffer[DType.float64]
    var rf: DeviceBuffer[DType.float64]
    var wing: DeviceBuffer[DType.int32]
    var cl: DeviceBuffer[DType.float64]
    var cd: DeviceBuffer[DType.float64]

    def __init__(out self, capacity: Int) raises:
        self.ctx = DeviceContext()
        self.capacity = capacity
        self.alpha = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.re = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.ar = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.rf = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.wing = self.ctx.enqueue_create_buffer[DType.int32](capacity)
        self.cl = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.cd = self.ctx.enqueue_create_buffer[DType.float64](capacity)
        self.ctx.synchronize()

    @staticmethod
    def py_init(out self: GpuFluid, args: PythonObject,
                kwargs: PythonObject) raises:
        self = Self(Int(py=args[0]))

    @staticmethod
    def coefficients(self_ptr: UnsafePointer[Self, MutAnyOrigin],
                     desc: PythonObject) raises -> PythonObject:
        """Same maths as fluid_gpu.coefficients, without the allocation.

        `desc` (int64): alpha, re, ar, rf, is_wing, cl_out, cd_out, n
        """
        var d = UnsafePointer[Int64, MutAnyOrigin](
            unsafe_from_address=Int(py=desc.ctypes.data)
        )
        var n = Int(d[unsafe_offset=7])
        if n > self_ptr[].capacity:
            raise Error(
                "GpuFluid: ", n, " panels exceeds capacity ",
                self_ptr[].capacity,
                " -- construct it larger rather than growing here, or the"
                " allocation this exists to avoid comes back mid-run",
            )
        var ctx = self_ptr[].ctx

        # Sub-buffers of exactly n, not the whole capacity.  enqueue_copy moves
        # len(dst_buf) elements, so copying into a capacity-sized buffer from an
        # n-element host array reads past the end of the host array and CUDA
        # rejects it with CUDA_ERROR_INVALID_VALUE -- after corrupting the heap
        # on the way, which is what turns it into a segfault rather than an
        # exception.
        ctx.enqueue_copy(
            dst_buf=self_ptr[].alpha.create_sub_buffer[DType.float64](0, n),
            src_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=0])))
        ctx.enqueue_copy(
            dst_buf=self_ptr[].re.create_sub_buffer[DType.float64](0, n),
            src_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=1])))
        ctx.enqueue_copy(
            dst_buf=self_ptr[].ar.create_sub_buffer[DType.float64](0, n),
            src_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=2])))
        ctx.enqueue_copy(
            dst_buf=self_ptr[].rf.create_sub_buffer[DType.float64](0, n),
            src_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=3])))
        ctx.enqueue_copy(
            dst_buf=self_ptr[].wing.create_sub_buffer[DType.int32](0, n),
            src_ptr=UnsafePointer[Int32, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=4])))

        ctx.enqueue_function[coeff_kernel](
            self_ptr[].alpha.unsafe_ptr(), self_ptr[].re.unsafe_ptr(),
            self_ptr[].ar.unsafe_ptr(), self_ptr[].rf.unsafe_ptr(),
            self_ptr[].wing.unsafe_ptr(),
            self_ptr[].cl.unsafe_ptr(), self_ptr[].cd.unsafe_ptr(),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
        ctx.enqueue_copy(
            dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=5])),
            src_buf=self_ptr[].cl.create_sub_buffer[DType.float64](0, n))
        ctx.enqueue_copy(
            dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
                unsafe_from_address=Int(d[unsafe_offset=6])),
            src_buf=self_ptr[].cd.create_sub_buffer[DType.float64](0, n))
        ctx.synchronize()
        return PythonObject(n)

    # Both are supplied explicitly.  The default implementations are derived by
    # reflection over the fields, and DeviceContext is not Writable, so the
    # derivation fails rather than falling back.
    def write_to(self, mut writer: Some[Writer]):
        writer.write("GpuFluid(capacity=", self.capacity, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("GpuFluid(capacity=", self.capacity, ")")

    @staticmethod
    def get_capacity(self_ptr: UnsafePointer[Self, MutAnyOrigin]) -> PythonObject:
        return PythonObject(self_ptr[].capacity)


@export
def PyInit_fluid_state() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("fluid_state")
        _ = (
            m.add_type[GpuFluid]("GpuFluid")
            .def_py_init[GpuFluid.py_init]()
            .def_method[GpuFluid.coefficients]("coefficients")
            .def_method[GpuFluid.get_capacity]("get_capacity")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
