"""Q6 probe: can a Mojo DeviceBuffer be read and written by Warp with no copy?

ROADMAP 2026-10-10 "an evaluation that never leaves the device", probe Q6.
A Mojo Python-extension module that owns one float32 device buffer and exposes
what Warp needs to alias it:

    p = interop_probe.Probe(n)   # allocate n float32 on cuda:0, kernel writes i * 0.5
    p.ptr()                      # int: raw device address (DeviceBuffer.unsafe_ptr())
    p.stream_handle()            # int: the CUstream behind ctx.stream(), 0 if None
    p.context_handle()           # int: the CUcontext behind the DeviceContext
    p.checksum()                 # float: sum of the buffer, computed on device
    p.size()                     # int: n

`checksum()` re-reads the buffer on device on every call, so it sees whatever
Python (a Warp kernel) wrote through the aliased pointer.  The sum is exact: every
element is a multiple of 0.5 below 2**23 and the accumulator is float64, so the
order of the atomic adds does not matter.

Build (does NOT touch mojo/build/, which a live run may have mapped):

    cd mojo && pixi run mojo build --emit shared-lib -I src \
        ../experiments/mojo_warp_interop/interop_probe.mojo \
        -o ../experiments/mojo_warp_interop/build/.interop_probe.so.new \
      && mv ../experiments/mojo_warp_interop/build/.interop_probe.so.new \
            ../experiments/mojo_warp_interop/build/interop_probe.so

Constructing a Probe creates a CUDA context (and so touches the GPU); merely
importing the module does not.
"""

from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._nvidia_cuda import CUDA
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

comptime BLOCK = 256
comptime CHUNK = 64  # elements summed serially per thread before one atomic


def fill_half_index(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if Int32(i) < n:
        buf[unsafe_offset=i] = Float32(i) * 0.5


def checksum_kernel(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    acc: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    var start = Int(global_idx.x) * CHUNK
    var s = Float64(0.0)
    for k in range(CHUNK):
        var idx = start + k
        if Int32(idx) < n:
            s += Float64(buf[unsafe_offset=idx])
    _ = Atomic.fetch_add(acc, s)


struct Probe(Movable, Writable):
    var ctx: DeviceContext
    var n: Int
    var buf: DeviceBuffer[DType.float32]
    var acc: DeviceBuffer[DType.float64]

    def __init__(out self, n: Int) raises:
        self.ctx = DeviceContext()
        self.n = n
        self.buf = self.ctx.enqueue_create_buffer[DType.float32](n)
        self.acc = self.ctx.enqueue_create_buffer[DType.float64](1)
        self.ctx.enqueue_function[fill_half_index](
            self.buf.unsafe_ptr(),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
        self.ctx.synchronize()

    @staticmethod
    def py_init(out self: Probe, args: PythonObject,
                kwargs: PythonObject) raises:
        self = Self(Int(py=args[0]))

    @staticmethod
    def ptr(self_ptr: UnsafePointer[Self, MutAnyOrigin]) -> PythonObject:
        return PythonObject(Int(self_ptr[].buf.unsafe_ptr()))

    @staticmethod
    def size(self_ptr: UnsafePointer[Self, MutAnyOrigin]) -> PythonObject:
        return PythonObject(self_ptr[].n)

    @staticmethod
    def stream_handle(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """The CUstream under ctx.stream(); 0 when the runtime reports none."""
        var h = CUDA(self_ptr[].ctx.stream())
        if h:
            return PythonObject(Int(h.value()))
        return PythonObject(0)

    @staticmethod
    def context_handle(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """The CUcontext under the DeviceContext; 0 when none is reported."""
        var h = CUDA(self_ptr[].ctx)
        if h:
            return PythonObject(Int(h.value()))
        return PythonObject(0)

    @staticmethod
    def checksum(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        """Sum of the buffer, recomputed on device now (float64, exact here)."""
        var ctx = self_ptr[].ctx
        var n = self_ptr[].n
        # Do not trust ordering against another library's stream.
        ctx.synchronize()
        self_ptr[].acc.enqueue_fill(0.0)
        var threads = ceildiv(n, CHUNK)
        ctx.enqueue_function[checksum_kernel](
            self_ptr[].buf.unsafe_ptr(),
            self_ptr[].acc.unsafe_ptr(),
            Int32(n),
            grid_dim=ceildiv(threads, BLOCK),
            block_dim=BLOCK,
        )
        var host = ctx.enqueue_create_host_buffer[DType.float64](1)
        ctx.enqueue_copy(dst_buf=host, src_buf=self_ptr[].acc)
        ctx.synchronize()
        return PythonObject(Float64(host.unsafe_ptr()[unsafe_offset=0]))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Probe(n=", self.n, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Probe(n=", self.n, ")")


@export
def PyInit_interop_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("interop_probe")
        _ = (
            m.add_type[Probe]("Probe")
            .def_py_init[Probe.py_init]()
            .def_method[Probe.ptr]("ptr")
            .def_method[Probe.size]("size")
            .def_method[Probe.stream_handle]("stream_handle")
            .def_method[Probe.context_handle]("context_handle")
            .def_method[Probe.checksum]("checksum")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
