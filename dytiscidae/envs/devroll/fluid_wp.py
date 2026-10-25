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
    WAGNER_A,
    WAGNER_B,
)

_KERNELS: dict = {}

# Python floats, so Warp sees plain constants (numpy scalars do not cast in kernels).
_SEP = float(SEPARATION_COMPLETE)
_CN_PLATE = float(CN_PLATE)
_CN_LEV = float(CN_LEV)
_CD_CROSSFLOW = float(CD_CROSSFLOW)
_WAGNER_A = tuple(float(x) for x in WAGNER_A)
_WAGNER_B = tuple(float(x) for x in WAGNER_B)


def make_kernels(T):
    """Return the kernels for precision ``T`` as a dict (``strip``, ``unsteady``,
    ``medium``, ``added_mass``, ``assemble``, ``finish``) and the 3-vector type ``V``.

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


    @_wp.kernel
    def unsteady_update(
        alpha: _wp.array(dtype=T), rev: _wp.array(dtype=_wp.int32), U: _wp.array(dtype=T),
        chord: _wp.array(dtype=T), omega: _wp.array(dtype=V), s_hat: _wp.array(dtype=V),
        x0: _wp.array(dtype=T), x1: _wp.array(dtype=T), s_trav: _wp.array(dtype=T),
        rev_prev: _wp.array(dtype=_wp.int32), primed: _wp.int32, dt: T,
        a1: T, a2: T, b1: T, b2: T, travel_lo: T, travel_hi: T,
        alpha_e: _wp.array(dtype=T), lev: _wp.array(dtype=T),
    ):
        i = _wp.tid()
        a = alpha[i]
        r = rev[i]
        if primed != 0:
            if r != rev_prev[i]:
                x0[i] = T(0.0)
                x1[i] = T(0.0)
                s_trav[i] = T(0.0)
        rev_prev[i] = r
        c = _wp.max(chord[i], T(1e-6))
        ds = U[i] * dt / c
        st = s_trav[i] + ds
        s_trav[i] = st
        # Exact for a step held over dt: semichords = 2 * chords.
        x0[i] = x0[i] + (a - x0[i]) * (T(1.0) - _wp.exp(-b1 * T(2.0) * ds))
        x1[i] = x1[i] + (a - x1[i]) * (T(1.0) - _wp.exp(-b2 * T(2.0) * ds))
        alpha_e[i] = a * (T(1.0) - a1 - a2) + a1 * x0[i] + a2 * x1[i]
        t = _wp.clamp((st - travel_lo) / (travel_hi - travel_lo), T(0.0), T(1.0))
        lev_travel = T(1.0) - t * t * (T(3.0) - T(2.0) * t)
        lev[i] = _wp.max(rossby_lev(U[i], chord[i], omega[i], s_hat[i]), lev_travel)

    @_wp.kernel
    def medium_flat(
        pos: _wp.array(dtype=V), half_h: _wp.array(dtype=T), wind: V, current: V,
        rho_air: T, rho_w: T, mu_air: T, mu_w: T,
        rho: _wp.array(dtype=T), mu: _wp.array(dtype=T), subf: _wp.array(dtype=T),
        flow: _wp.array(dtype=V),
    ):
        # Flat calm surface at z = 0 (SeaState amplitude 0): depth is -z.
        i = _wp.tid()
        h = _wp.max(half_h[i], T(1e-3))
        f = _wp.clamp(T(0.5) + T(0.5) * (T(0.0) - pos[i][2]) / h, T(0.0), T(1.0))
        rho[i] = rho_air + f * (rho_w - rho_air)
        mu[i] = _wp.pow(mu_air, T(1.0) - f) * _wp.pow(mu_w, f)
        subf[i] = f
        # Ambient flow: wind above, current below.  MediumField.flow_velocity blends
        # with a fixed half-height of 1e-2, not the element's own (medium.py).
        g = _wp.clamp(T(0.5) + T(0.5) * (T(0.0) - pos[i][2]) / T(1e-2), T(0.0), T(1.0))
        flow[i] = (T(1.0) - g) * wind + g * current

    @_wp.kernel
    def added_mass(
        rel: _wp.array(dtype=V), s_hat: _wp.array(dtype=V), c_hat: _wp.array(dtype=V),
        n_hat: _wp.array(dtype=V), ext: _wp.array(dtype=V), chord: _wp.array(dtype=T),
        dr: _wp.array(dtype=T), volume: _wp.array(dtype=T), rho: _wp.array(dtype=T),
        is_wing: _wp.array(dtype=_wp.int32), scale: T,
        m_add: _wp.array(dtype=T), m_slam: _wp.array(dtype=T), vn: _wp.array(dtype=T),
    ):
        # Wing: tensor added mass (chordwise t^2, normal c^2 per unit span), projected
        # on the direction of motion.  Bluff: directional Ca from the element's extents.
        i = _wp.tid()
        s = s_hat[i]
        c = c_hat[i]
        n = n_hat[i]
        v = rel[i]
        r = rho[i]
        vn[i] = _wp.dot(v, n)
        e = ext[i]
        e0 = _wp.max(e[0], T(1e-4))
        e1 = _wp.max(e[1], T(1e-4))
        e2 = _wp.max(e[2], T(1e-4))
        ca0 = _wp.clamp(T(0.5) * (e1 + e2) / (T(2.0) * e0), T(0.05), T(10.0))
        ca1 = _wp.clamp(T(0.5) * (e2 + e0) / (T(2.0) * e1), T(0.05), T(10.0))
        ca2 = _wp.clamp(T(0.5) * (e0 + e1) / (T(2.0) * e2), T(0.05), T(10.0))
        vl = _wp.length(v)
        d = v / _wp.max(vl, T(1e-6))
        ds = _wp.dot(d, s)
        dcs = ds * ds
        dcc = _wp.dot(d, c)
        dcc = dcc * dcc
        dcn = _wp.dot(d, n)
        dcn = dcn * dcn
        moving = dcs + dcc + dcn > T(1e-6)
        ca_eff = dcs * ca0 + dcc * ca1 + dcn * ca2
        ca_mean = (ca0 + ca1 + ca2) / T(3.0)
        if not moving:
            ca_eff = ca_mean
        t_wing = _wp.max(e2, T(1e-5))
        k = r * dr[i] * T(0.7853981633974483)  # rho dr pi/4
        m0 = T(0.0)
        m1 = k * t_wing * t_wing
        m2 = k * chord[i] * chord[i]
        m_wing = dcs * m0 + dcc * m1 + dcn * m2
        if not moving:
            m_wing = (m0 + m1 + m2) / T(3.0)
        normal_slam = r * T(3.141592653589793) * chord[i] * chord[i] * T(0.25) * dr[i] * scale
        if is_wing[i] != 0:
            m_add[i] = m_wing * scale
            m_slam[i] = normal_slam
        else:
            m_add[i] = ca_eff * r * volume[i] * scale
            m_slam[i] = ca_eff * r * volume[i] * scale

    @_wp.kernel
    def assemble_panels(
        pos: _wp.array(dtype=V), F_in: _wp.array(dtype=V), body_id: _wp.array(dtype=_wp.int32),
        xipos: _wp.array(dtype=V), subf: _wp.array(dtype=T), vb: _wp.array(dtype=T), buoy: T,
        F_out: _wp.array(dtype=V), fb_f: _wp.array(dtype=V), fb_t: _wp.array(dtype=V),
        total: _wp.array(dtype=T),
    ):
        # Buoyancy into the panel force, then the panel's load onto its body: force, and
        # torque about the body's centre of mass.  ``total`` is sum |F| over the machine's
        # panels, which the limiter reads.  Atomic adds reorder against numpy's add.at,
        # so the parity test is at rounding, not bitwise.
        i = _wp.tid()
        b = body_id[i]
        f = F_in[i] + V(T(0.0), T(0.0), buoy * vb[i] * subf[i])
        F_out[i] = f
        arm = pos[i] - xipos[b]
        _wp.atomic_add(fb_f, b, f)
        _wp.atomic_add(fb_t, b, _wp.cross(arm, f))
        _wp.atomic_add(total, 0, _wp.length(f))

    @_wp.kernel
    def finish_bodies_wp(
        fb_f: _wp.array(dtype=V), fb_t: _wp.array(dtype=V), m_body: _wp.array(dtype=T),
        total: _wp.array(dtype=T), limit: T, gravity: T,
    ):
        # finish_bodies: scale all six components by limit/total if the machine's total
        # exceeds the limit, then cancel the weight of the entrained fluid at the CoM.
        b = _wp.tid()
        s = T(1.0)
        if total[0] > limit:
            s = limit / total[0]
        fb_f[b] = fb_f[b] * s + V(T(0.0), T(0.0), m_body[b] * gravity)
        fb_t[b] = fb_t[b] * s

    _KERNELS[T] = {"strip": strip_loads, "unsteady": unsteady_update,
                   "medium": medium_flat, "added_mass": added_mass,
                   "assemble": assemble_panels, "finish": finish_bodies_wp, "V": V}
    return _KERNELS[T]


def strip_loads(rel, s_hat, c_hat, n_hat, omega, ext, chord, area, dr, cd_bluff, camber,
                ar, is_wing, rho, mu, lift_scale, cd_scale, c_rot, *, dtype=_wp.float64):
    """Run the strip kernel on host arrays; return the outputs as numpy arrays.

    Inputs are numpy arrays: vectors ``(N, 3)``, scalars ``(N,)``, ``is_wing`` as
    integers (``WING`` = 0, ``BLUFF`` = 1 in physics.fluid, so ``is_wing`` here
    is 1 for a lifting strip).  Returns ``F (N,3), D, cl, alpha, U, lev``.
    """
    K = make_kernels(dtype)
    kern, V = K["strip"], K["V"]
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


class WagnerLag:
    """Device history of ``physics.fluid.UnsteadyState``: Wagner lag and LEV travel.

    One instance per body's strips.  ``update`` takes the quasi-static incidence
    of this step and returns ``(alpha_e, lev)`` as numpy arrays, the same pair
    the numpy class returns.  The state lives on the device between calls.
    """

    def __init__(self, n: int, dtype=_wp.float64) -> None:
        from dytiscidae.physics.fluid import LEV_TRAVEL

        K = make_kernels(dtype)
        self._kern, V = K["unsteady"], K["V"]
        self._dtype = dtype
        self._V = V
        self._n = n
        self._x0 = _wp.zeros(n, dtype=dtype)
        self._x1 = _wp.zeros(n, dtype=dtype)
        self._s = _wp.zeros(n, dtype=dtype)
        self._rev_prev = _wp.zeros(n, dtype=_wp.int32)
        self._primed = 0
        self._travel = tuple(float(x) for x in LEV_TRAVEL)

    def update(self, alpha, rev, U, chord, omega, s_hat, dt: float):
        npdt = _np.float64 if self._dtype == _wp.float64 else _np.float32

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=self._dtype)

        def vec(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=self._V)

        n = self._n
        alpha_e = _wp.zeros(n, dtype=self._dtype)
        lev = _wp.zeros(n, dtype=self._dtype)
        _wp.launch(self._kern, dim=n, inputs=[
            sc(alpha),
            _wp.array(_np.asarray(rev, dtype=_np.int32), dtype=_wp.int32),
            sc(U), sc(chord), vec(omega), vec(s_hat),
            self._x0, self._x1, self._s, self._rev_prev,
            self._primed, self._dtype(dt),
            self._dtype(_WAGNER_A[0]), self._dtype(_WAGNER_A[1]),
            self._dtype(_WAGNER_B[0]), self._dtype(_WAGNER_B[1]),
            self._dtype(self._travel[0]), self._dtype(self._travel[1]),
            alpha_e, lev,
        ])
        self._primed = 1
        return alpha_e.numpy(), lev.numpy()


def medium_and_added_mass(pos, half_h, wind, current, rel, s_hat, c_hat, n_hat, ext, chord,
                          dr, volume, is_wing, scale, *, dtype=_wp.float64):
    """Flat-calm medium and the strip added mass, on the device; numpy out.

    The medium is ``MediumField`` with a calm sea (``SeaState`` amplitude 0): the
    surface is z = 0, so the submerged fraction is the ramp through it.  Returns
    ``rho, mu, subf, flow, m_add, m_slam, vn`` as numpy arrays.  The caller
    passes the element's own relative velocity ``rel``.
    """
    from dytiscidae.physics.medium import AIR, SEAWATER

    K = make_kernels(dtype)
    V = K["V"]
    npdt = _np.float64 if dtype == _wp.float64 else _np.float32
    n = len(chord)

    def vec(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=V)

    def sc(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

    rho = _wp.zeros(n, dtype=dtype)
    mu = _wp.zeros(n, dtype=dtype)
    subf = _wp.zeros(n, dtype=dtype)
    flow = _wp.zeros(n, dtype=V)
    _wp.launch(K["medium"], dim=n, inputs=[
        vec(pos), sc(half_h), V(*[dtype(x) for x in wind]), V(*[dtype(x) for x in current]),
        dtype(AIR.rho), dtype(SEAWATER.rho), dtype(AIR.mu), dtype(SEAWATER.mu),
        rho, mu, subf, flow])
    m_add = _wp.zeros(n, dtype=dtype)
    m_slam = _wp.zeros(n, dtype=dtype)
    vn = _wp.zeros(n, dtype=dtype)
    _wp.launch(K["added_mass"], dim=n, inputs=[
        vec(rel), vec(s_hat), vec(c_hat), vec(n_hat), vec(ext), sc(chord), sc(dr), sc(volume),
        rho, _wp.array(_np.asarray(is_wing, dtype=_np.int32), dtype=_wp.int32), dtype(scale),
        m_add, m_slam, vn])
    return {"rho": rho.numpy(), "mu": mu.numpy(), "subf": subf.numpy(), "flow": flow.numpy(),
            "m_add": m_add.numpy(), "m_slam": m_slam.numpy(), "vn": vn.numpy()}


def assemble_bodies(pos, F, body_id, xipos, subf, vb, rho_w, m_body, limit, *,
                    dtype=_wp.float64):
    """Buoyancy, panel-to-body assembly and ``finish_bodies``, on the device; numpy out.

    ``F (n, 3)`` is the strip load before buoyancy; ``body_id (n,)`` the panel's body;
    ``xipos (nb, 3)`` each body's centre of mass; ``subf``, ``vb`` per panel;
    ``m_body (nb,)`` the body masses with the added mass.  Returns ``F`` (the panel
    force with buoyancy), ``fb (nb, 6)`` (world force then torque, as
    ``xfrc_applied``) and ``clamped`` (whether the limiter bound).
    """
    from dytiscidae.physics.fluid import GRAVITY

    K = make_kernels(dtype)
    V = K["V"]
    npdt = _np.float64 if dtype == _wp.float64 else _np.float32
    n, nb = len(F), len(m_body)

    def vec(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=V)

    def sc(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

    F_out = _wp.zeros(n, dtype=V)
    fb_f = _wp.zeros(nb, dtype=V)
    fb_t = _wp.zeros(nb, dtype=V)
    total = _wp.zeros(1, dtype=dtype)
    _wp.launch(K["assemble"], dim=n, inputs=[
        vec(pos), vec(F), _wp.array(_np.asarray(body_id, dtype=_np.int32), dtype=_wp.int32),
        vec(xipos), sc(subf), sc(vb), dtype(rho_w * GRAVITY),
        F_out, fb_f, fb_t, total])
    _wp.launch(K["finish"], dim=nb, inputs=[
        fb_f, fb_t, sc(m_body), total, dtype(limit), dtype(GRAVITY)])
    fb = _np.concatenate([fb_f.numpy(), fb_t.numpy()], axis=1)
    return {"F": F_out.numpy(), "fb": fb, "clamped": float(total.numpy()[0]) > limit}
