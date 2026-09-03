"""Smallest kernel that proves the GPU path this project needs actually works.

Computes angle of attack from a velocity pair on the device -- the operation
the fluid port hinges on, and the one the GPU stdlib cannot do unaided.  Prints
the raw triples so a host-side checker can compare against numpy.

Run with `pixi run probe`.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mathx import atan2f

comptime dtype = DType.float32
comptime N = 8192
comptime BLOCK = 256
comptime layout = row_major[N]()


def aoa_kernel(
    vx: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    vz: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    res: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    # Fixed width, not `Int`: Int and UInt do not conform to DevicePassable,
    # and passing one fails deep inside the launch machinery rather than here.
    size: Int32,
):
    var i = global_idx.x
    if Int32(i) < size:
        var y = rebind[Scalar[dtype]](vz[i])
        var x = rebind[Scalar[dtype]](vx[i])
        res[i] = rebind[res.ElementType](atan2f(y, x))


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var vx_buf = ctx.enqueue_create_buffer[dtype](N)
    var vz_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)

    # Coprime strides so the sample sweeps all four quadrants and both sides of
    # the |y| > |x| swap, including the axes where the quadrant fixups bite.
    with vx_buf.map_to_host() as hx:
        with vz_buf.map_to_host() as hz:
            var tx = TileTensor(hx, layout)
            var tz = TileTensor(hz, layout)
            for i in range(N):
                tx[i] = rebind[tx.ElementType](Float32(i % 127) - 63.0)
                tz[i] = rebind[tz.ElementType](Float32((i * 7) % 131) - 65.0)
    out_buf.enqueue_fill(0.0)

    ctx.enqueue_function[aoa_kernel](
        TileTensor(vx_buf, layout),
        TileTensor(vz_buf, layout),
        TileTensor(out_buf, layout),
        Int32(N),
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as ho:
        with vx_buf.map_to_host() as hx:
            with vz_buf.map_to_host() as hz:
                var tr = TileTensor(ho, layout)
                var tx = TileTensor(hx, layout)
                var tz = TileTensor(hz, layout)
                for i in range(N):
                    print(tx[i], tz[i], tr[i])
