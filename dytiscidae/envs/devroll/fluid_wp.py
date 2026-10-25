"""Strip-theory loads on the device (IMPL_1010 A1a, first slice).

Per element: the incidence and relative speed, the lift and drag of a lifting
strip, the bluff-body drag of a volume and the Kramer rotational lift.  This is
``FluidSolver.apply`` (physics/fluid.py) from the kinematics to the strip
forces, in quasi-static form: the Wagner lag, the inflow model, the medium
and the added-mass tensor are later slices of A1a and A1b.

The kernels are built per precision by :func:`make_kernels`, because a Warp
kernel cannot take a float32 literal in a float64 expression.  Constants come
from ``physics.fluid`` so the numbers have one source.
"""

import numpy as _np

import warp as _wp

from dytiscidae.physics.fluid import (
    CD_CROSSFLOW,
    CN_LEV,
    CN_PLATE,
    SEPARATION_COMPLETE,
)

_KERNELS: dict = {}

# Python floats, so Warp sees plain constants (numpy scalars do not cast in kernels).
_SEP = float(SEPARATION_COMPLETE)
_CN_PLATE = float(CN_PLATE)
_CN_LEV = float(CN_LEV)
_CD_CROSSFLOW = float(CD_CROSSFLOW)


def make_kernels(T):
    """Return ``(strip_loads, V)`` for precision ``T`` (``wp.float64`` or ``wp.float32``).

    ``V`` is the matching 3-vector type.  Built once per precision and cached.
    """
    if T in _KERNELS:
        return _KERNELS[T]
    V = _wp.vec3d if T == _wp.float64 else _wp.vec3

    @_wp.func
    def skin_friction(re: T) -> T:
        r = _wp.max(re, T(1.0))
        lam = T(1.328) / _wp.sqrt(r)
        turb = T(0.074) / _wp.pow(r, T(0.2))
        w = T(1.0) / (T(1.0) + _wp.exp(-(_wp.log10(r) - T(5.7)) * T(4.0)))
        return T(2.0) * ((T(1.0) - w) * lam + w * turb)

    @_wp.func
    def stall_weight(alpha: T, re: T) -> T:
        # Static stall angle, then the attached-to-separated smoothstep.
        a_stall = T(0.19198621771937624)  # radians(11)
        a_stall = a_stall * _wp.clamp(
            T(0.55) + T(0.45) * _wp.log10(_wp.max(re, T(10.0))) / T(5.0), T(0.5), T(1.0))
        t = _wp.clamp((_wp.abs(alpha) - a_stall) / T(_SEP), T(0.0), T(1.0))
        return t * t * (T(3.0) - T(2.0) * t)

    @_wp.func
    def lift_coefficient(alpha: T, re: T, ar: T, lev: T) -> T:
        lv = _wp.clamp(lev, T(0.0), T(1.0))
        cl_alpha = T(2.0) * T(3.141592653589793) / (T(1.0) + T(2.0) / _wp.max(ar, T(0.5)))
        cl_att = cl_alpha * alpha
        cn = T(_CN_PLATE) + (T(_CN_LEV) - T(_CN_PLATE)) * lv
        cl_sep = cn * _wp.sin(alpha) * _wp.cos(alpha)
        w = stall_weight(alpha, re)
        w = w + (T(1.0) - w) * lv
        return (T(1.0) - w) * cl_att + w * cl_sep

    @_wp.func
    def drag_coefficient(alpha: T, re: T, ar: T, cl: T, lev: T) -> T:
        lv = _wp.clamp(lev, T(0.0), T(1.0))
        cd_f = skin_friction(re)
        cd_i = cl * cl / (T(3.141592653589793) * T(0.75) * _wp.max(ar, T(0.5)))
        w = stall_weight(alpha, re)
        w = w + (T(1.0) - w) * lv
        cn = T(_CN_PLATE) + (T(_CN_LEV) - T(_CN_PLATE)) * lv
        sa = _wp.sin(alpha)
        return cd_f + cd_i + w * cn * sa * sa

    @_wp.func
    def rossby_lev(U: T, chord: T, om: V, s: V) -> T:
        c = _wp.max(chord, T(1e-6))
        w_perp = om - _wp.dot(om, s) * s
        ro = U / _wp.max(_wp.length(w_perp) * c, T(1e-9))
        t = _wp.clamp((ro - T(3.0)) / T(5.0), T(0.0), T(1.0))  # LEV_ROSSBY = (3, 8)
        return T(1.0) - t * t * (T(3.0) - T(2.0) * t)

    @_wp.kernel
    def strip_loads(
        rel: _wp.array(dtype=V), s_hat: _wp.array(dtype=V), c_hat: _wp.array(dtype=V),
        n_hat: _wp.array(dtype=V), omega: _wp.array(dtype=V), ext: _wp.array(dtype=V),
        chord: _wp.array(dtype=T), area: _wp.array(dtype=T), dr: _wp.array(dtype=T),
        cd_bluff: _wp.array(dtype=T), camber: _wp.array(dtype=T), ar: _wp.array(dtype=T),
        is_wing: _wp.array(dtype=_wp.int32), rho: _wp.array(dtype=T), mu: _wp.array(dtype=T),
        lift_scale: T, cd_scale: T, c_rot: T,
        F: _wp.array(dtype=V), D: _wp.array(dtype=T), cl_out: _wp.array(dtype=T),
        alpha_out: _wp.array(dtype=T), U_out: _wp.array(dtype=T), lev_out: _wp.array(dtype=T),
    ):
        i = _wp.tid()
        s = s_hat[i]
        c = c_hat[i]
        n = n_hat[i]
        r = rho[i]
        m = _wp.max(mu[i], T(1e-12))
        ch = chord[i]
        ar_i = ar[i]
        omg = omega[i]

        # Relative flow, projected out of the span.
        v_rel = rel[i]
        v_2d = v_rel - _wp.dot(v_rel, s) * s
        U = _wp.length(v_2d)
        U_safe = _wp.max(U, T(1e-6))
        d_hat = v_2d / U_safe
        q = T(0.5) * r * U * U
        re = r * U * ch / m

        cos_a = _wp.dot(v_2d, c) / U_safe
        sin_a = _wp.dot(v_2d, n) / U_safe
        rev = cos_a < T(0.0)
        cos_f = cos_a
        if rev:
            cos_f = -cos_a
        alpha = _wp.atan2(sin_a, cos_f + T(1e-12))
        if rev:
            alpha = -alpha
        # Camber shifts the zero-lift angle; reversed flow sees the other end.
        cam = camber[i]
        if rev:
            alpha = alpha + T(2.0) * (-cam)
        else:
            alpha = alpha + T(2.0) * cam

        omega_s = _wp.dot(omg, s)
        lev = rossby_lev(U, ch, omg, s)

        lift_axis = _wp.cross(s, d_hat)
        lift_axis = lift_axis / _wp.max(_wp.length(lift_axis), T(1e-12))

        is_w = is_wing[i] != 0
        F_i = V(T(0.0), T(0.0), T(0.0))
        D_i = T(0.0)
        cl = T(0.0)
        if is_w:
            cl = lift_coefficient(alpha, re, ar_i, lev)
            cd = drag_coefficient(alpha, re, ar_i, cl, lev)
            L = q * area[i] * cl * lift_scale
            D_i = q * area[i] * cd * cd_scale
            F_i = F_i + L * lift_axis + D_i * d_hat

            # Kramer rotational lift on the lifting strips.
            f_rot = -c_rot * r * U * omega_s * ch * ch * dr[i]
            F_i = F_i + f_rot * lift_axis
        else:
            # Bluff body: axial pressure plus friction, cross-flow drag.
            ex = ext[i][0]
            ey = ext[i][1]
            ez = ext[i][2]
            v_ax = _wp.dot(v_rel, s)
            v_cross = v_rel - v_ax * s
            u_cross = _wp.length(v_cross)
            d_cross = v_cross / _wp.max(u_cross, T(1e-9))

            wetted = T(2.0) * (ex * ey + ey * ez + ex * ez)
            re_b = r * _wp.abs(v_ax) * ex / m
            f_axial = cd_scale * (
                T(0.5) * r * _wp.abs(v_ax) * v_ax
                * (cd_bluff[i] * ey * ez + skin_friction(re_b) * wetted))

            pc = _wp.abs(_wp.dot(d_cross, c))
            pn = _wp.abs(_wp.dot(d_cross, n))
            a_side = pc * ex * ez + pn * ex * ey
            f_cross = cd_scale * T(0.5) * r * u_cross * u_cross * T(_CD_CROSSFLOW) * a_side

            F_i = f_axial * s + f_cross * d_cross
            D_i = _wp.abs(f_axial) + f_cross

        F[i] = F_i
        D[i] = D_i
        cl_out[i] = cl
        alpha_out[i] = alpha
        U_out[i] = U
        lev_out[i] = lev

    _KERNELS[T] = (strip_loads, V)
    return _KERNELS[T]


def strip_loads(rel, s_hat, c_hat, n_hat, omega, ext, chord, area, dr, cd_bluff, camber,
                ar, is_wing, rho, mu, lift_scale, cd_scale, c_rot, *, dtype=_wp.float64):
    """Run the strip kernel on host arrays; return the outputs as numpy arrays.

    Inputs are numpy arrays: vectors ``(N, 3)``, scalars ``(N,)``, ``is_wing`` as
    integers (``WING`` = 0, ``BLUFF`` = 1 in physics.fluid, so ``is_wing`` here
    is 1 for a lifting strip).  Returns ``F (N,3), D, cl, alpha, U, lev``.
    """
    kern, V = make_kernels(dtype)
    npdt = _np.float64 if dtype == _wp.float64 else _np.float32

    def vec(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=V)

    def sc(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

    n = len(chord)
    F = _wp.zeros(n, dtype=V)
    D = _wp.zeros(n, dtype=dtype)
    cl = _wp.zeros(n, dtype=dtype)
    al = _wp.zeros(n, dtype=dtype)
    U = _wp.zeros(n, dtype=dtype)
    lev = _wp.zeros(n, dtype=dtype)
    _wp.launch(kern, dim=n, inputs=[
        vec(rel), vec(s_hat), vec(c_hat), vec(n_hat), vec(omega), vec(ext),
        sc(chord), sc(area), sc(dr), sc(cd_bluff), sc(camber), sc(ar),
        _wp.array(_np.asarray(is_wing, dtype=_np.int32), dtype=_wp.int32),
        sc(rho), sc(mu),
        dtype(lift_scale), dtype(cd_scale), dtype(c_rot),
        F, D, cl, al, U, lev,
    ])
    return {"F": F.numpy(), "D": D.numpy(), "cl": cl.numpy(), "alpha": al.numpy(),
            "U": U.numpy(), "lev": lev.numpy()}
