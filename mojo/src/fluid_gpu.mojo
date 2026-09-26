"""GPU kinematics for the fluid solver.

Ports the block marked `--- kinematics` in `physics/fluid.py`: rotate each
panel's local span,
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
from std.math import ceildiv, sqrt, abs
from std.python import PythonObject
from mathx import atan2d, sind, cosd, expd, logd, log10d, powd, clampd
from std.python.bindings import PythonModuleBuilder
from std.atomic import Atomic

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



comptime PI_D: Float64 = 3.14159265358979323846
comptime DEG: Float64 = PI_D / 180.0


@always_inline
def _skin_friction_cd(re_in: Float64) -> Float64:
    """Mirrors `fluid.skin_friction_cd`.  Blasius blended into the 1/7-power law."""
    var re = re_in if re_in > 1.0 else 1.0
    var lam = 1.328 / sqrt(re)
    var turb = 0.074 / powd(re, 0.2)
    var w = 1.0 / (1.0 + expd(-(log10d(re) - 5.7) * 4.0))
    return 2.0 * ((1.0 - w) * lam + w * turb)


@always_inline
def _stall_weight(alpha: Float64, re: Float64, lev: Float64) -> Float64:
    """Mirrors `fluid._stall` plus the LEV's pull toward separation."""
    var re_c = re if re > 10.0 else 10.0
    var stall = 11.0 * DEG
    stall = stall * clampd(0.55 + 0.45 * log10d(re_c) / 5.0, 0.5, 1.0)
    var t = clampd((abs(alpha) - stall) / (6.0 * DEG), 0.0, 1.0)
    var w = t * t * (3.0 - 2.0 * t)
    return w + (1.0 - w) * lev


@always_inline
def _lift_coefficient(
    alpha: Float64, re: Float64, ar: Float64, lev_in: Float64, alpha_e: Float64
) -> Float64:
    """Mirrors `fluid.lift_coefficient`: attached below stall on the Wagner-
    lagged incidence, separated normal force above, LEV raising CN."""
    var lev = clampd(lev_in, 0.0, 1.0)
    var ar_c = ar if ar > 0.5 else 0.5
    var cl_att = (2.0 * PI_D / (1.0 + 2.0 / ar_c)) * alpha_e
    var cn = 1.98 + (3.4 - 1.98) * lev
    var cl_sep = cn * sind(alpha) * cosd(alpha)
    var w = _stall_weight(alpha, re, lev)
    return (1.0 - w) * cl_att + w * cl_sep


@always_inline
def _drag_coefficient(
    alpha: Float64, re: Float64, ar: Float64, cl: Float64, lev_in: Float64
) -> Float64:
    """Mirrors `fluid.drag_coefficient`: friction, induced, separated pressure."""
    var lev = clampd(lev_in, 0.0, 1.0)
    var ar_c = ar if ar > 0.5 else 0.5
    var cd_i = cl * cl / (PI_D * 0.75 * ar_c)
    var w = _stall_weight(alpha, re, lev)
    var cn = 1.98 + (3.4 - 1.98) * lev
    var sa = sind(alpha)
    return _skin_friction_cd(re) + cd_i + w * cn * sa * sa


def unsteady_kernel(
    alpha: UnsafePointer[Float64, MutAnyOrigin],
    rev: UnsafePointer[Float64, MutAnyOrigin],
    u: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    omega: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    x0: UnsafePointer[Float64, MutAnyOrigin],
    x1: UnsafePointer[Float64, MutAnyOrigin],
    trav: UnsafePointer[Float64, MutAnyOrigin],
    rev_prev: UnsafePointer[Float64, MutAnyOrigin],
    primed: UnsafePointer[Float64, MutAnyOrigin],
    ae_out: UnsafePointer[Float64, MutAnyOrigin],
    lev_out: UnsafePointer[Float64, MutAnyOrigin],
    dt: Float64,
    reset: Int32,
    enabled: Int32,
    n: Int32,
):
    """Mirrors `fluid.UnsteadyState.update`: Wagner lag (Jones's two terms) and
    the LEV's travel-since-reversal and Rossby mechanisms, one strip a thread,
    with the history in persistent buffers."""
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    if enabled == 0:
        # `FluidSolver.unsteady` off: no lag, and the LEV from the Rossby
        # number alone (`rossby_lev`), exactly as the Python's else-branch.
        var c0 = chord[unsafe_offset=i]
        c0 = c0 if c0 > 1e-6 else 1e-6
        var u0 = u[unsafe_offset=i]
        var qx = omega[unsafe_offset=j + 0]
        var qy = omega[unsafe_offset=j + 1]
        var qz = omega[unsafe_offset=j + 2]
        var hx = s_hat[unsafe_offset=j + 0]
        var hy = s_hat[unsafe_offset=j + 1]
        var hz = s_hat[unsafe_offset=j + 2]
        var q_s = qx * hx + qy * hy + qz * hz
        var rx = qx - q_s * hx
        var ry = qy - q_s * hy
        var rz = qz - q_s * hz
        var wp0 = sqrt(rx * rx + ry * ry + rz * rz) * c0
        var ro0 = u0 / (wp0 if wp0 > 1e-9 else 1e-9)
        var t0 = clampd((ro0 - 3.0) / (8.0 - 3.0), 0.0, 1.0)
        ae_out[unsafe_offset=i] = alpha[unsafe_offset=i]
        lev_out[unsafe_offset=i] = 1.0 - t0 * t0 * (3.0 - 2.0 * t0)
        return
    if reset != 0:
        x0[unsafe_offset=i] = 0.0
        x1[unsafe_offset=i] = 0.0
        trav[unsafe_offset=i] = 0.0
        rev_prev[unsafe_offset=i] = 0.0
        primed[unsafe_offset=i] = 0.0
    var r = rev[unsafe_offset=i]
    if primed[unsafe_offset=i] != 0.0 and r != rev_prev[unsafe_offset=i]:
        x0[unsafe_offset=i] = 0.0
        x1[unsafe_offset=i] = 0.0
        trav[unsafe_offset=i] = 0.0
    rev_prev[unsafe_offset=i] = r
    primed[unsafe_offset=i] = 1.0
    var c = chord[unsafe_offset=i]
    c = c if c > 1e-6 else 1e-6
    var uu = u[unsafe_offset=i]
    var ds = uu * dt / c
    var s = trav[unsafe_offset=i] + ds
    trav[unsafe_offset=i] = s
    var a = alpha[unsafe_offset=i]
    var y0 = x0[unsafe_offset=i]
    var y1 = x1[unsafe_offset=i]
    y0 = y0 + (a - y0) * (1.0 - expd(-0.0455 * 2.0 * ds))
    y1 = y1 + (a - y1) * (1.0 - expd(-0.3 * 2.0 * ds))
    x0[unsafe_offset=i] = y0
    x1[unsafe_offset=i] = y1
    ae_out[unsafe_offset=i] = a * (1.0 - 0.165 - 0.335) + 0.165 * y0 + 0.335 * y1
    var tt = clampd((s - 2.0) / (4.0 - 2.0), 0.0, 1.0)
    var lev_t = 1.0 - tt * tt * (3.0 - 2.0 * tt)
    var wx = omega[unsafe_offset=j + 0]
    var wy = omega[unsafe_offset=j + 1]
    var wz = omega[unsafe_offset=j + 2]
    var sx = s_hat[unsafe_offset=j + 0]
    var sy = s_hat[unsafe_offset=j + 1]
    var sz = s_hat[unsafe_offset=j + 2]
    var ws = wx * sx + wy * sy + wz * sz
    var px = wx - ws * sx
    var py = wy - ws * sy
    var pz = wz - ws * sz
    var wp = sqrt(px * px + py * py + pz * pz) * c
    var ro = uu / (wp if wp > 1e-9 else 1e-9)
    var tr = clampd((ro - 3.0) / (8.0 - 3.0), 0.0, 1.0)
    var lev_r = 1.0 - tr * tr * (3.0 - 2.0 * tr)
    lev_out[unsafe_offset=i] = lev_r if lev_r > lev_t else lev_t


comptime CD_CROSSFLOW: Float64 = 1.1


def bluff_kernel(
    v_rel: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    c_hat: UnsafePointer[Float64, MutAnyOrigin],
    n_hat: UnsafePointer[Float64, MutAnyOrigin],
    rho: UnsafePointer[Float64, MutAnyOrigin],
    mu: UnsafePointer[Float64, MutAnyOrigin],
    ext: UnsafePointer[Float64, MutAnyOrigin],
    cd_bluff: UnsafePointer[Float64, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    f_out: UnsafePointer[Float64, MutAnyOrigin],
    d_out: UnsafePointer[Float64, MutAnyOrigin],
    dfull_out: UnsafePointer[Float64, MutAnyOrigin],
    cd_scale: Float64,
    n: Int32,
):
    """Munk slender-body plus Allen-Perkins cross-flow.

    Mirrors the block marked `--- bluff-body drag` in `fluid.py`.

    `d_full` is written for every panel, wing or not: it is computed over all N
    in the Python (the `~is_wing` mask is only applied at the accumulate), and
    the added-mass block downstream reads it for wing panels too.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3

    var vx = v_rel[unsafe_offset=j + 0]
    var vy = v_rel[unsafe_offset=j + 1]
    var vz = v_rel[unsafe_offset=j + 2]

    # np.linalg.norm IS plain left-to-right, unlike einsum.
    var u_full = sqrt(vx * vx + vy * vy + vz * vz)
    var u_full_safe = u_full if u_full > 1e-6 else 1e-6
    dfull_out[unsafe_offset=j + 0] = vx / u_full_safe
    dfull_out[unsafe_offset=j + 1] = vy / u_full_safe
    dfull_out[unsafe_offset=j + 2] = vz / u_full_safe

    if is_wing[unsafe_offset=i] != 0:
        f_out[unsafe_offset=j + 0] = 0.0
        f_out[unsafe_offset=j + 1] = 0.0
        f_out[unsafe_offset=j + 2] = 0.0
        d_out[unsafe_offset=i] = 0.0
        return

    var sx = s_hat[unsafe_offset=j + 0]
    var sy = s_hat[unsafe_offset=j + 1]
    var sz = s_hat[unsafe_offset=j + 2]

    # ext_local is (span, chord, normal) extents, in that order.
    var ex = ext[unsafe_offset=j + 0]
    var ey = ext[unsafe_offset=j + 1]
    var ez = ext[unsafe_offset=j + 2]

    var v_ax = vx * sx + vy * sy + vz * sz          # signed
    var cxv = vx - v_ax * sx
    var cyv = vy - v_ax * sy
    var czv = vz - v_ax * sz
    var u_cross = sqrt(cxv * cxv + cyv * cyv + czv * czv)
    var u_cross_safe = u_cross if u_cross > 1e-9 else 1e-9
    var dcx = cxv / u_cross_safe
    var dcy = cyv / u_cross_safe
    var dcz = czv / u_cross_safe

    var r = rho[unsafe_offset=i]
    var m = mu[unsafe_offset=i]
    var wetted = 2.0 * (ex * ey + ey * ez + ex * ez)
    # Length scale is ex, the span-axis extent, not the chord; velocity scale is
    # the axial component, not the full speed.
    var re_b = r * abs(v_ax) * ex / (m if m > 1e-12 else 1e-12)
    var f_axial = cd_scale * (
        0.5 * r * abs(v_ax) * v_ax
        * (cd_bluff[unsafe_offset=i] * ey * ez + _skin_friction_cd(re_b) * wetted)
    )

    # pc weights ex*ez and pn weights ex*ey: each cosine multiplies the area of
    # the face whose normal it belongs to.  Swapping them is the easy bug.
    var pc = abs(dcx * c_hat[unsafe_offset=j + 0]
                 + dcy * c_hat[unsafe_offset=j + 1]
                 + dcz * c_hat[unsafe_offset=j + 2])
    var pn = abs(dcx * n_hat[unsafe_offset=j + 0]
                 + dcy * n_hat[unsafe_offset=j + 1]
                 + dcz * n_hat[unsafe_offset=j + 2])
    var a_side = pc * ex * ez + pn * ex * ey
    # u_cross unclamped here, deliberately: the 1e-9 floor is only for the
    # direction, and squaring the floored value would invent a force at rest.
    var f_cross = cd_scale * 0.5 * r * u_cross * u_cross * CD_CROSSFLOW * a_side

    f_out[unsafe_offset=j + 0] = f_axial * sx + f_cross * dcx
    f_out[unsafe_offset=j + 1] = f_axial * sy + f_cross * dcy
    f_out[unsafe_offset=j + 2] = f_axial * sz + f_cross * dcz
    d_out[unsafe_offset=i] = abs(f_axial) + f_cross


comptime GRAVITY: Float64 = 9.80665


def added_mass_kernel(
    v_rel: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    c_hat: UnsafePointer[Float64, MutAnyOrigin],
    n_hat: UnsafePointer[Float64, MutAnyOrigin],
    d_full: UnsafePointer[Float64, MutAnyOrigin],
    rho: UnsafePointer[Float64, MutAnyOrigin],
    ext: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    dr: UnsafePointer[Float64, MutAnyOrigin],
    volume: UnsafePointer[Float64, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    body_id: UnsafePointer[Int32, MutAnyOrigin],
    m_add: UnsafePointer[Float64, MutAnyOrigin],
    vn_out: UnsafePointer[Float64, MutAnyOrigin],
    m_body: UnsafePointer[Float64, MutAnyOrigin],
    fz: UnsafePointer[Float64, MutAnyOrigin],
    scale: Float64,
    has_bluff: Int32,
    wing_tensor: Int32,
    n: Int32,
):
    """Anisotropic added mass and its scatter.

    Mirrors the block marked `--- added mass` in `fluid.py`.

    The scatter is `np.add.at(m_body, body_id, m_add)`: many panels share a
    body, so the collisions are the rule and not the exception.  Float64
    `Atomic.fetch_add` handles it directly -- verified on this card at 100000
    additions into 16 buckets, exact.

    What this deliberately does NOT do is write `model.body_mass`.  That is a
    MuJoCo model array read by `mj_step` on the very next line, so it stays on
    the host; this kernel only produces the per-body sum it is built from.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3

    var vx = v_rel[unsafe_offset=j + 0]
    var vy = v_rel[unsafe_offset=j + 1]
    var vz = v_rel[unsafe_offset=j + 2]
    var nx = n_hat[unsafe_offset=j + 0]
    var ny = n_hat[unsafe_offset=j + 1]
    var nz = n_hat[unsafe_offset=j + 2]
    vn_out[unsafe_offset=i] = vx * nx + vy * ny + vz * nz

    # Ca_i = 0.5 * (e_j + e_k) / (2 e_i): exact for a sphere, within 18% of
    # Lamb's disc, and correctly small for a slender body moving end-on.
    var e0 = ext[unsafe_offset=j + 0]
    var e1 = ext[unsafe_offset=j + 1]
    var e2 = ext[unsafe_offset=j + 2]
    e0 = e0 if e0 > 1e-4 else 1e-4
    e1 = e1 if e1 > 1e-4 else 1e-4
    e2 = e2 if e2 > 1e-4 else 1e-4
    var ca0 = clampd(0.5 * (e1 + e2) / (2.0 * e0), 0.05, 10.0)
    var ca1 = clampd(0.5 * (e2 + e0) / (2.0 * e1), 0.05, 10.0)
    var ca2 = clampd(0.5 * (e0 + e1) / (2.0 * e2), 0.05, 10.0)

    var ca_eff: Float64
    if has_bluff != 0:
        var dx = d_full[unsafe_offset=j + 0]
        var dy = d_full[unsafe_offset=j + 1]
        var dz = d_full[unsafe_offset=j + 2]
        var ds = dx * s_hat[unsafe_offset=j + 0] + dy * s_hat[unsafe_offset=j + 1] + dz * s_hat[unsafe_offset=j + 2]
        var dcc = dx * c_hat[unsafe_offset=j + 0] + dy * c_hat[unsafe_offset=j + 1] + dz * c_hat[unsafe_offset=j + 2]
        var dn = dx * nx + dy * ny + dz * nz
        var q0 = ds * ds
        var q1 = dcc * dcc
        var q2 = dn * dn
        # A body momentarily at rest has no direction to project onto; fall
        # back to the isotropic mean rather than to zero.
        if q0 + q1 + q2 > 1e-6:
            ca_eff = q0 * ca0 + q1 * ca1 + q2 * ca2
        else:
            ca_eff = (ca0 + ca1 + ca2) / 3.0
    else:
        # dc is all zeros, so the sum is 0 and the fallback always fires.
        ca_eff = (ca0 + ca1 + ca2) / 3.0

    var r = rho[unsafe_offset=i]
    var ma: Float64
    if is_wing[unsafe_offset=i] != 0:
        var ch = chord[unsafe_offset=i]
        if wing_tensor != 0:
            # The plate's tensor projected on the flow direction, as fluid.py:
            # span 0, chord rho pi t^2/4, normal rho pi c^2/4 per unit span.
            var tw = ext[unsafe_offset=j + 2]
            tw = tw if tw > 1e-5 else 1e-5
            var k = r * dr[unsafe_offset=i] * (PI_D * 0.25)
            var m1 = k * (tw * tw)
            var m2 = k * (ch * ch)
            var wx = d_full[unsafe_offset=j + 0]
            var wy = d_full[unsafe_offset=j + 1]
            var wz = d_full[unsafe_offset=j + 2]
            var ws = wx * s_hat[unsafe_offset=j + 0] + wy * s_hat[unsafe_offset=j + 1] + wz * s_hat[unsafe_offset=j + 2]
            var wc = wx * c_hat[unsafe_offset=j + 0] + wy * c_hat[unsafe_offset=j + 1] + wz * c_hat[unsafe_offset=j + 2]
            var wn = wx * nx + wy * ny + wz * nz
            var p0 = ws * ws
            var p1 = wc * wc
            var p2 = wn * wn
            if p0 + p1 + p2 > 1e-6:
                ma = p0 * 0.0 + p1 * m1 + p2 * m2
            else:
                ma = (0.0 + m1 + m2) / 3.0
        else:
            ma = r * PI_D * ch * ch * 0.25 * dr[unsafe_offset=i]
    else:
        ma = ca_eff * r * volume[unsafe_offset=i]
    ma = ma * scale
    m_add[unsafe_offset=i] = ma

    # Cancel the weight MuJoCo will apply to the entrained fluid: added mass has
    # inertia but no weight.
    fz[unsafe_offset=i] = ma * GRAVITY

    # The per-body sum is formed by gather_body_kernel instead, in index
    # order.  Doing it with an atomic here made the result depend on warp
    # arrival, which propagated into every score.
    _ = m_body
    _ = body_id


def coeff_kernel(
    alpha: UnsafePointer[Float64, MutAnyOrigin],
    re: UnsafePointer[Float64, MutAnyOrigin],
    ar: UnsafePointer[Float64, MutAnyOrigin],
    lev: UnsafePointer[Float64, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    cl_out: UnsafePointer[Float64, MutAnyOrigin],
    cd_out: UnsafePointer[Float64, MutAnyOrigin],
    alpha_e: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    # np.where(is_wing, ..., 0.0): bluff elements get their drag from the
    # cross-flow model instead, not from strip theory.
    if is_wing[unsafe_offset=i] == 0:
        cl_out[unsafe_offset=i] = 0.0
        cd_out[unsafe_offset=i] = 0.0
        return
    var a = alpha[unsafe_offset=i]
    var r = re[unsafe_offset=i]
    var arv = ar[unsafe_offset=i]
    var lv = lev[unsafe_offset=i]
    var cl = _lift_coefficient(a, r, arv, lv, alpha_e[unsafe_offset=i])
    cl_out[unsafe_offset=i] = cl
    cd_out[unsafe_offset=i] = _drag_coefficient(a, r, arv, cl, lv)


@always_inline
def _dl(ctx: DeviceContext, buf: DeviceBufferF64, addr: Int) raises:
    """Download a device buffer straight into a host address."""
    ctx.enqueue_copy(
        dst_ptr=UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr),
        src_buf=buf,
    )


def strip_kernel(
    v_rel: UnsafePointer[Float64, MutAnyOrigin],
    s_hat: UnsafePointer[Float64, MutAnyOrigin],
    c_hat: UnsafePointer[Float64, MutAnyOrigin],
    n_hat: UnsafePointer[Float64, MutAnyOrigin],
    omega: UnsafePointer[Float64, MutAnyOrigin],
    rho: UnsafePointer[Float64, MutAnyOrigin],
    mu: UnsafePointer[Float64, MutAnyOrigin],
    chord: UnsafePointer[Float64, MutAnyOrigin],
    camber: UnsafePointer[Float64, MutAnyOrigin],
    u_out: UnsafePointer[Float64, MutAnyOrigin],
    d_hat: UnsafePointer[Float64, MutAnyOrigin],
    q_out: UnsafePointer[Float64, MutAnyOrigin],
    re_out: UnsafePointer[Float64, MutAnyOrigin],
    alpha_out: UnsafePointer[Float64, MutAnyOrigin],
    rf_out: UnsafePointer[Float64, MutAnyOrigin],
    lift_axis: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    """Strip theory, the block marked `--- strip theory` in `fluid.py`.

    One thread per panel."""
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3

    var vx = v_rel[unsafe_offset=j + 0]
    var vy = v_rel[unsafe_offset=j + 1]
    var vz = v_rel[unsafe_offset=j + 2]
    var sx = s_hat[unsafe_offset=j + 0]
    var sy = s_hat[unsafe_offset=j + 1]
    var sz = s_hat[unsafe_offset=j + 2]

    # Project the spanwise component out: strip theory is a 2D section, and
    # flow along the span does not turn the section.
    var vs = vx * sx + vy * sy + vz * sz
    var ax = vx - vs * sx
    var ay = vy - vs * sy
    var az = vz - vs * sz

    var uu = sqrt(ax * ax + ay * ay + az * az)
    var u_safe = uu if uu > 1e-6 else 1e-6
    u_out[unsafe_offset=i] = uu

    var dx = ax / u_safe
    var dy = ay / u_safe
    var dz = az / u_safe
    d_hat[unsafe_offset=j + 0] = dx
    d_hat[unsafe_offset=j + 1] = dy
    d_hat[unsafe_offset=j + 2] = dz

    var r = rho[unsafe_offset=i]
    var m = mu[unsafe_offset=i]
    var ch = chord[unsafe_offset=i]
    q_out[unsafe_offset=i] = 0.5 * r * uu * uu
    re_out[unsafe_offset=i] = r * uu * ch / (m if m > 1e-12 else 1e-12)

    var cos_a = (
        ax * c_hat[unsafe_offset=j + 0]
        + ay * c_hat[unsafe_offset=j + 1]
        + az * c_hat[unsafe_offset=j + 2]
    ) / u_safe
    var sin_a = (
        ax * n_hat[unsafe_offset=j + 0]
        + ay * n_hat[unsafe_offset=j + 1]
        + az * n_hat[unsafe_offset=j + 2]
    ) / u_safe

    # The Python folds incidence into [-pi/2, pi/2] as
    #     atan2(sin(atan2(s,c)), |cos(atan2(s,c))| + eps)
    # which collapses exactly to atan2(s, |c| + eps): atan2 is invariant under
    # positive scaling, and sin/cos of an atan2 are just its arguments over
    # their hypotenuse.  Worth collapsing rather than transcribing -- sin and
    # cos do not link in float64 on device, and this needs neither.
    # ...but the epsilon has to be added to the *normalised* cosine, which is
    # what sin()/cos() of the first atan2 return.  Adding it to the raw dot
    # product instead scales it by the hypotenuse -- atan2(s/r, |c|/r + e) is
    # atan2(s, |c| + e*r), not atan2(s, |c| + e) -- and near 90 degrees of
    # incidence, where the cosine vanishes, that moved alpha by up to 1e-9 rad.
    # Harmless physically, but it is a difference with a cause, so it is
    # removed rather than absorbed into a tolerance.
    var hyp = sqrt(sin_a * sin_a + cos_a * cos_a)
    var hyp_safe = hyp if hyp > 1e-300 else 1e-300
    # 2026-09-23: a shift of pi, not a mirror -- see fluid.py.  Reversed flow
    # negates the folded angle and the camber term.
    var rev = cos_a < 0.0
    var cpos = -cos_a if rev else cos_a
    var alpha = atan2d(sin_a / hyp_safe, cpos / hyp_safe + 1e-12)
    var cam = camber[unsafe_offset=i]
    if rev:
        alpha = -alpha
        cam = -cam
    alpha_out[unsafe_offset=i] = alpha + 2.0 * cam

    var ws = (
        omega[unsafe_offset=j + 0] * sx
        + omega[unsafe_offset=j + 1] * sy
        + omega[unsafe_offset=j + 2] * sz
    )
    # `rf_out` carries the reversal flag now (1 when the flow arrives over the
    # trailing edge): the LEV and the Wagner lag reset on its changes
    # (`unsteady_kernel`).  The reduced pitch rate it used to carry keyed the
    # LEV wrongly, MATH_AUDIT F-02.
    _ = ws
    _ = ch
    rf_out[unsafe_offset=i] = 1.0 if rev else 0.0

    # lift acts normal to both the span and the flow
    var lx = sy * dz - sz * dy
    var ly = sz * dx - sx * dz
    var lz = sx * dy - sy * dx
    var ln = sqrt(lx * lx + ly * ly + lz * lz)
    var ln_safe = ln if ln > 1e-12 else 1e-12
    lift_axis[unsafe_offset=j + 0] = lx / ln_safe
    lift_axis[unsafe_offset=j + 1] = ly / ln_safe
    lift_axis[unsafe_offset=j + 2] = lz / ln_safe


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


def strip_theory(desc: PythonObject) raises -> PythonObject:
    """Run strip theory on the GPU.

    `desc` layout (int64):
        0..8   v_rel, s_hat, c_hat, n_hat, omega, rho, mu, chord, camber
        9..15  U, d_hat, q, re, alpha, reduced_freq, lift_axis
        16     n
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data)
    )
    var n = Int(d[unsafe_offset=16])
    var ctx = DeviceContext()

    var v_rel = _f64(ctx, Int(d[unsafe_offset=0]), n * 3)
    var s_hat = _f64(ctx, Int(d[unsafe_offset=1]), n * 3)
    var c_hat = _f64(ctx, Int(d[unsafe_offset=2]), n * 3)
    var n_hat = _f64(ctx, Int(d[unsafe_offset=3]), n * 3)
    var omega = _f64(ctx, Int(d[unsafe_offset=4]), n * 3)
    var rho = _f64(ctx, Int(d[unsafe_offset=5]), n)
    var mu = _f64(ctx, Int(d[unsafe_offset=6]), n)
    var chord = _f64(ctx, Int(d[unsafe_offset=7]), n)
    var camber = _f64(ctx, Int(d[unsafe_offset=8]), n)

    var u_o = ctx.enqueue_create_buffer[DType.float64](n)
    var d_o = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var q_o = ctx.enqueue_create_buffer[DType.float64](n)
    var re_o = ctx.enqueue_create_buffer[DType.float64](n)
    var a_o = ctx.enqueue_create_buffer[DType.float64](n)
    var rf_o = ctx.enqueue_create_buffer[DType.float64](n)
    var la_o = ctx.enqueue_create_buffer[DType.float64](n * 3)

    ctx.enqueue_function[strip_kernel](
        v_rel.unsafe_ptr(), s_hat.unsafe_ptr(), c_hat.unsafe_ptr(),
        n_hat.unsafe_ptr(), omega.unsafe_ptr(), rho.unsafe_ptr(),
        mu.unsafe_ptr(), chord.unsafe_ptr(), camber.unsafe_ptr(),
        u_o.unsafe_ptr(), d_o.unsafe_ptr(), q_o.unsafe_ptr(),
        re_o.unsafe_ptr(), a_o.unsafe_ptr(), rf_o.unsafe_ptr(),
        la_o.unsafe_ptr(),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )

    _dl(ctx, u_o, Int(d[unsafe_offset=9]))
    _dl(ctx, d_o, Int(d[unsafe_offset=10]))
    _dl(ctx, q_o, Int(d[unsafe_offset=11]))
    _dl(ctx, re_o, Int(d[unsafe_offset=12]))
    _dl(ctx, a_o, Int(d[unsafe_offset=13]))
    _dl(ctx, rf_o, Int(d[unsafe_offset=14]))
    _dl(ctx, la_o, Int(d[unsafe_offset=15]))
    ctx.synchronize()
    return PythonObject(n)


def coefficients(desc: PythonObject) raises -> PythonObject:
    """Lift and drag coefficients on the GPU.

    `desc` (int64): alpha, re, ar, reduced_freq, is_wing, cl_out, cd_out, n
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data)
    )
    var n = Int(d[unsafe_offset=7])
    var ctx = DeviceContext()

    var alpha = _f64(ctx, Int(d[unsafe_offset=0]), n)
    var re = _f64(ctx, Int(d[unsafe_offset=1]), n)
    var ar = _f64(ctx, Int(d[unsafe_offset=2]), n)
    var rf = _f64(ctx, Int(d[unsafe_offset=3]), n)
    # The standalone call has no history: alpha_e is alpha, as a separate
    # buffer because one may not be passed mutably twice.
    var ae = _f64(ctx, Int(d[unsafe_offset=0]), n)

    var wing = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(
        dst_buf=wing,
        src_ptr=UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=4])
        ),
    )

    var cl = ctx.enqueue_create_buffer[DType.float64](n)
    var cd = ctx.enqueue_create_buffer[DType.float64](n)

    ctx.enqueue_function[coeff_kernel](
        alpha.unsafe_ptr(), re.unsafe_ptr(), ar.unsafe_ptr(), rf.unsafe_ptr(),
        wing.unsafe_ptr(), cl.unsafe_ptr(), cd.unsafe_ptr(), ae.unsafe_ptr(),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    _dl(ctx, cl, Int(d[unsafe_offset=5]))
    _dl(ctx, cd, Int(d[unsafe_offset=6]))
    ctx.synchronize()
    return PythonObject(n)


def bluff_drag(desc: PythonObject, cd_scale: PythonObject) raises -> PythonObject:
    """Bluff-body drag on the GPU.

    `desc` (int64): v_rel, s_hat, c_hat, n_hat, rho, mu, ext_local, cd_bluff,
    is_wing, F_out, D_out, d_full_out, n
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data)
    )
    var n = Int(d[unsafe_offset=12])
    var ctx = DeviceContext()

    var v_rel = _f64(ctx, Int(d[unsafe_offset=0]), n * 3)
    var s_hat = _f64(ctx, Int(d[unsafe_offset=1]), n * 3)
    var c_hat = _f64(ctx, Int(d[unsafe_offset=2]), n * 3)
    var n_hat = _f64(ctx, Int(d[unsafe_offset=3]), n * 3)
    var rho = _f64(ctx, Int(d[unsafe_offset=4]), n)
    var mu = _f64(ctx, Int(d[unsafe_offset=5]), n)
    var ext = _f64(ctx, Int(d[unsafe_offset=6]), n * 3)
    var cdb = _f64(ctx, Int(d[unsafe_offset=7]), n)

    var wing = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(
        dst_buf=wing,
        src_ptr=UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(d[unsafe_offset=8])
        ),
    )

    var f_o = ctx.enqueue_create_buffer[DType.float64](n * 3)
    var d_o = ctx.enqueue_create_buffer[DType.float64](n)
    var df_o = ctx.enqueue_create_buffer[DType.float64](n * 3)

    ctx.enqueue_function[bluff_kernel](
        v_rel.unsafe_ptr(), s_hat.unsafe_ptr(), c_hat.unsafe_ptr(),
        n_hat.unsafe_ptr(), rho.unsafe_ptr(), mu.unsafe_ptr(),
        ext.unsafe_ptr(), cdb.unsafe_ptr(), wing.unsafe_ptr(),
        f_o.unsafe_ptr(), d_o.unsafe_ptr(), df_o.unsafe_ptr(),
        Float64(py=cd_scale),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    _dl(ctx, f_o, Int(d[unsafe_offset=9]))
    _dl(ctx, d_o, Int(d[unsafe_offset=10]))
    _dl(ctx, df_o, Int(d[unsafe_offset=11]))
    ctx.synchronize()
    return PythonObject(n)


def added_mass(desc: PythonObject, scale: PythonObject,
               has_bluff: PythonObject) raises -> PythonObject:
    """Added mass and its per-body scatter on the GPU.

    `desc` (int64): v_rel, s_hat, c_hat, n_hat, d_full, rho, ext, chord, dr,
    volume, is_wing, body_id, m_add_out, vn_out, m_body_out, fz_out, n, nbody
    """
    var d = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=desc.ctypes.data)
    )
    var n = Int(d[unsafe_offset=16])
    var nbody = Int(d[unsafe_offset=17])
    var ctx = DeviceContext()

    var v_rel = _f64(ctx, Int(d[unsafe_offset=0]), n * 3)
    var s_hat = _f64(ctx, Int(d[unsafe_offset=1]), n * 3)
    var c_hat = _f64(ctx, Int(d[unsafe_offset=2]), n * 3)
    var n_hat = _f64(ctx, Int(d[unsafe_offset=3]), n * 3)
    var dfull = _f64(ctx, Int(d[unsafe_offset=4]), n * 3)
    var rho = _f64(ctx, Int(d[unsafe_offset=5]), n)
    var ext = _f64(ctx, Int(d[unsafe_offset=6]), n * 3)
    var chord = _f64(ctx, Int(d[unsafe_offset=7]), n)
    var dr = _f64(ctx, Int(d[unsafe_offset=8]), n)
    var vol = _f64(ctx, Int(d[unsafe_offset=9]), n)

    var wing = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=wing, src_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=10])))
    var bid = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=bid, src_ptr=UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(d[unsafe_offset=11])))

    var ma_o = ctx.enqueue_create_buffer[DType.float64](n)
    var vn_o = ctx.enqueue_create_buffer[DType.float64](n)
    var mb_o = ctx.enqueue_create_buffer[DType.float64](nbody)
    var fz_o = ctx.enqueue_create_buffer[DType.float64](n)
    # The scatter accumulates, so the destination has to start at zero -- the
    # Python allocates a fresh np.zeros(nbody) every step for the same reason.
    mb_o.enqueue_fill(0.0)

    ctx.enqueue_function[added_mass_kernel](
        v_rel.unsafe_ptr(), s_hat.unsafe_ptr(), c_hat.unsafe_ptr(),
        n_hat.unsafe_ptr(), dfull.unsafe_ptr(), rho.unsafe_ptr(),
        ext.unsafe_ptr(), chord.unsafe_ptr(), dr.unsafe_ptr(),
        vol.unsafe_ptr(), wing.unsafe_ptr(), bid.unsafe_ptr(),
        ma_o.unsafe_ptr(), vn_o.unsafe_ptr(), mb_o.unsafe_ptr(),
        fz_o.unsafe_ptr(),
        Float64(py=scale), Int32(Int(py=has_bluff)), Int32(0), Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    _dl(ctx, ma_o, Int(d[unsafe_offset=12]))
    _dl(ctx, vn_o, Int(d[unsafe_offset=13]))
    _dl(ctx, mb_o, Int(d[unsafe_offset=14]))
    _dl(ctx, fz_o, Int(d[unsafe_offset=15]))
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit_fluid_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("fluid_gpu")
        m.def_function[body_to_world]("body_to_world")
        m.def_function[strip_theory]("strip_theory")
        m.def_function[coefficients]("coefficients")
        m.def_function[bluff_drag]("bluff_drag")
        m.def_function[added_mass]("added_mass")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))


def velocity_kernel(
    vel6: UnsafePointer[Float64, MutAnyOrigin],
    xpos: UnsafePointer[Float64, MutAnyOrigin],
    pos_w: UnsafePointer[Float64, MutAnyOrigin],
    u_flow: UnsafePointer[Float64, MutAnyOrigin],
    body_id: UnsafePointer[Int32, MutAnyOrigin],
    omega_out: UnsafePointer[Float64, MutAnyOrigin],
    vrel_out: UnsafePointer[Float64, MutAnyOrigin],
    machine: UnsafePointer[Int32, MutAnyOrigin],
    is_wing: UnsafePointer[Int32, MutAnyOrigin],
    v_ind: UnsafePointer[Float64, MutAnyOrigin],
    n: Int32,
):
    """Panel relative flow, the block marked `--- kinematics` in `fluid.py`.

    Lifting strips also see their machine's momentum-theory downwash `v_ind`
    (`fluid.InducedFlow`, MATH_AUDIT F-13), added to the ambient flow first as
    the Python does.


    `vel6` is one 6-vector per body from mj_objectVelocity -- angular in 0..2,
    linear in 3..5, world frame, the linear part at the body's centre of mass.
    `xpos` is therefore the caller's *xipos* buffer: the lever arm has to start
    where the velocity was measured (fixed 2026-09-23).  That call is a
    per-body CPU loop and stays on the host; everything downstream of it is
    per-panel and lands here.

    The Python's comment in that block explains why data.cvel is not used
    instead: its linear part is referenced to a com-based frame whose origin is
    not the body frame origin, and reconstructing element velocity from it
    disagreed with mj_objectVelocity by ~0.5 m/s.  Same reasoning applies here;
    this kernel consumes what that loop produced rather than trying to be
    clever with cvel on the device.
    """
    var i = Int(global_idx.x)
    if Int32(i) >= n:
        return
    var j = i * 3
    var b = Int(body_id[unsafe_offset=i])
    var v6 = b * 6

    var wx = vel6[unsafe_offset=v6 + 0]
    var wy = vel6[unsafe_offset=v6 + 1]
    var wz = vel6[unsafe_offset=v6 + 2]
    omega_out[unsafe_offset=j + 0] = wx
    omega_out[unsafe_offset=j + 1] = wy
    omega_out[unsafe_offset=j + 2] = wz

    # r = element centroid minus the point vel6 is measured at (xipos)
    var rx = pos_w[unsafe_offset=j + 0] - xpos[unsafe_offset=b * 3 + 0]
    var ry = pos_w[unsafe_offset=j + 1] - xpos[unsafe_offset=b * 3 + 1]
    var rz = pos_w[unsafe_offset=j + 2] - xpos[unsafe_offset=b * 3 + 2]

    # v_elem = v_org + omega x r
    var ex = vel6[unsafe_offset=v6 + 3] + (wy * rz - wz * ry)
    var ey = vel6[unsafe_offset=v6 + 4] + (wz * rx - wx * rz)
    var ez = vel6[unsafe_offset=v6 + 5] + (wx * ry - wy * rx)

    # v_rel is the fluid seen from the element, so the sign is flow minus body.
    var ux = u_flow[unsafe_offset=j + 0]
    var uy = u_flow[unsafe_offset=j + 1]
    var uz = u_flow[unsafe_offset=j + 2]
    if is_wing[unsafe_offset=i] != 0:
        var m3 = Int(machine[unsafe_offset=i]) * 3
        ux = ux + v_ind[unsafe_offset=m3 + 0]
        uy = uy + v_ind[unsafe_offset=m3 + 1]
        uz = uz + v_ind[unsafe_offset=m3 + 2]
    vrel_out[unsafe_offset=j + 0] = ux - ex
    vrel_out[unsafe_offset=j + 1] = uy - ey
    vrel_out[unsafe_offset=j + 2] = uz - ez
