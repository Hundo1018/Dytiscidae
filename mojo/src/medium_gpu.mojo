"""The medium: density, viscosity, submerged fraction and ambient flow.

Ports `MediumField.properties` and `MediumField.flow_velocity`
(`physics/medium.py:128-176`), which fluid.apply calls at fluid.py:454-455 for
every panel every step.

The free surface is never a mode switch.  Every element carries a submerged
fraction in [0,1] and the properties blend through it, which is what keeps
water entry integrable and stops the optimiser finding a discontinuity to
exploit.  Density blends linearly (a volume average, exact for buoyancy) and
viscosity blends *geometrically*, because it spans five orders of magnitude and
a linear blend would be pinned to the water value the instant an element
touched the surface.

One faithful oddity, reproduced rather than corrected.  `surface_z` builds its
phase as `k*(xy.khat) - omega*t` (medium.py:69) but `orbital_velocity` builds
its as `xy.khat - omega*t` (medium.py:87) -- no wavenumber.  The two therefore
disagree about the wave's spatial period, the orbitals being stretched by a
factor of k.  That is what the Python does, so it is what this does; changing
it here would silently alter every existing score.  It is written up separately
as a question for the physics, not smuggled in as a port fix.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mathx import sind, cosd, expd, logd, clampd

comptime BLOCK = 128


def medium_kernel(
    pos: UnsafePointer[Float64, MutAnyOrigin],
    half_height: UnsafePointer[Float64, MutAnyOrigin],
    rho_out: UnsafePointer[Float64, MutAnyOrigin],
    mu_out: UnsafePointer[Float64, MutAnyOrigin],
    subf_out: UnsafePointer[Float64, MutAnyOrigin],
    uflow_out: UnsafePointer[Float64, MutAnyOrigin],
    amplitude: Float64,
    wavelength: Float64,
    period: Float64,
    kx: Float64,
    ky: Float64,
    t: Float64,
    air_rho: Float64,
    air_mu: Float64,
    water_rho: Float64,
    water_mu: Float64,
    cur_x: Float64,
    cur_y: Float64,
    cur_z: Float64,
    wind_x: Float64,
    wind_y: Float64,
    wind_z: Float64,
    n: Int32,
):
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    var px = pos[unsafe_offset=j + 0]
    var py = pos[unsafe_offset=j + 1]
    var pz = pos[unsafe_offset=j + 2]

    var k = 6.283185307179586 / wavelength
    var omega = 6.283185307179586 / period

    # --- surface elevation and submerged fraction -------------------------
    var z_surf: Float64 = 0.0
    if amplitude > 0.0:
        z_surf = amplitude * sind(k * (px * kx + py * ky) - omega * t)
    var depth = z_surf - pz

    var h = half_height[unsafe_offset=i]
    if h < 1e-3:
        h = 1e-3
    var f = clampd(0.5 + 0.5 * depth / h, 0.0, 1.0)
    subf_out[unsafe_offset=i] = f

    rho_out[unsafe_offset=i] = air_rho + f * (water_rho - air_rho)
    # air_mu**(1-f) * water_mu**f, as exp of the log blend.  Both are strictly
    # positive so there is no domain case to guard.
    mu_out[unsafe_offset=i] = expd((1.0 - f) * logd(air_mu) + f * logd(water_mu))

    # --- ambient flow -----------------------------------------------------
    # flow_velocity re-derives the fraction with a FIXED half-height of 1e-2,
    # not the panel's own (medium.py:173).  A wing panel and a hull slice at the
    # same depth therefore see the same wind/water blend even though they see
    # different densities, and that is deliberate in the original.
    var f2 = clampd(0.5 + 0.5 * depth / 1e-2, 0.0, 1.0)

    var ox: Float64 = 0.0
    var oy: Float64 = 0.0
    var oz: Float64 = 0.0
    if amplitude > 0.0:
        var zc = pz if pz < 0.0 else 0.0
        var decay = expd(k * zc)
        # No wavenumber on this phase; see the module docstring.
        var phase = (px * kx + py * ky) - omega * t
        var u_mag = amplitude * omega * decay
        ox = u_mag * cosd(phase) * kx
        oy = u_mag * cosd(phase) * ky
        oz = u_mag * sind(phase)

    uflow_out[unsafe_offset=j + 0] = (1.0 - f2) * wind_x + f2 * (cur_x + ox)
    uflow_out[unsafe_offset=j + 1] = (1.0 - f2) * wind_y + f2 * (cur_y + oy)
    uflow_out[unsafe_offset=j + 2] = (1.0 - f2) * wind_z + f2 * (cur_z + oz)


def medium(desc: PythonObject, scalars: PythonObject) raises -> PythonObject:
    """`desc` (int64): pos, half_height, rho_out, mu_out, subf_out, uflow_out, n

    `scalars`: amplitude, wavelength, period, khat_x, khat_y, t, air_rho,
    air_mu, water_rho, water_mu, current xyz, wind xyz

    khat comes in already evaluated rather than as `direction`.  Computing
    cos/sin of the direction on the device costs a last-bit disagreement with
    numpy's, and the wave phase multiplies it by positions of tens of metres --
    so a 1e-16 difference in the cosine became 4e-15 in the phase, and a panel
    with a 1e-3 half-height divides by that, reaching 2.1e-12 in the submerged
    fraction.  It is one scalar per medium; there is no reason to recompute it
    per panel, let alone at a different precision.
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data))
    var n = Int(d[unsafe_offset=6])
    var ctx = DeviceContext()

    var pos = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var hh = ctx.enqueue_create_buffer[DType.float64](n)
    ctx.enqueue_copy(dst_buf=pos, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=0])))
    ctx.enqueue_copy(dst_buf=hh, src_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=1])))

    var rho = ctx.enqueue_create_buffer[DType.float64](n)
    var mu = ctx.enqueue_create_buffer[DType.float64](n)
    var sf = ctx.enqueue_create_buffer[DType.float64](n)
    var uf = ctx.enqueue_create_buffer[DType.float64](n * 3)

    ctx.enqueue_function[medium_kernel](
        pos.unsafe_ptr(), hh.unsafe_ptr(),
        rho.unsafe_ptr(), mu.unsafe_ptr(), sf.unsafe_ptr(), uf.unsafe_ptr(),
        Float64(py=scalars[0]), Float64(py=scalars[1]), Float64(py=scalars[2]),
        Float64(py=scalars[3]), Float64(py=scalars[4]), Float64(py=scalars[5]),
        Float64(py=scalars[6]), Float64(py=scalars[7]), Float64(py=scalars[8]),
        Float64(py=scalars[9]), Float64(py=scalars[10]), Float64(py=scalars[11]),
        Float64(py=scalars[12]), Float64(py=scalars[13]), Float64(py=scalars[14]),
        Float64(py=scalars[15]),
        Int32(n), grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=2])), src_buf=rho)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=3])), src_buf=mu)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=4])), src_buf=sf)
    ctx.enqueue_copy(dst_ptr=UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=5])), src_buf=uf)
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit_medium_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("medium_gpu")
        m.def_function[medium]("medium")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
