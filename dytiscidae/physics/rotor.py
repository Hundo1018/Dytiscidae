"""Propellers: blade-element momentum theory, the comparison to flapping.

Added 2026-09-23 (ROADMAP "Why nothing flies, measured a third time") so that
the question "can anything fly in this simulator?" has a control: a machine
that makes thrust the way most small aircraft do, through the same airfoil
coefficients (`fluid.lift_coefficient`, `fluid.drag_coefficient`), the same
media and the same energy accounting as a flapping wing.

The model
---------
A fixed-pitch rotor of ``B`` blades, radius ``R``, hub ``r0``, chord ``c(r)``
and geometric pitch ``P`` (blade angle ``theta = atan(P / 2 pi r)``), spinning
at ``Omega`` with the flow meeting it at ``V_ax`` along the axis and ``V_ip``
in the disc plane.  For each annulus the induced velocity ``v`` balances the
two statements of its thrust:

    blade element:  dT = 1/2 rho W^2 B c (Cl cos phi - Cd sin phi) dr
    momentum:       dT = 4 pi r rho sqrt(V_ip^2 + (V_ax + v)^2) v F dr

with ``phi = atan((V_ax + v) / (Omega r))``, ``W^2 = (V_ax + v)^2 +
(Omega r)^2`` and Prandtl's tip loss ``F = (2/pi) acos(exp(-B (R - r) /
(2 r sin phi)))``.  ``sqrt(V_ip^2 + ...)`` is Glauert's forward-flight form, so
an edgewise rotor sees the mass flow of its forward speed.  Swirl is neglected
(small for propellers).  Torque is ``dQ = 1/2 rho W^2 B c (Cl sin phi + Cd cos
phi) r dr``.  Standard; Glauert 1935 (Durand, Aerodynamic Theory IV-L), as
restated in any rotor text.  ``experiments/rotor`` checks it against the UIUC
propeller database (APC 10x4.7 SF).

In MuJoCo a rotor is a body on a spin hinge driven by a velocity servo, so the
motor torque, its power and its reaction on the airframe go through the same
actuator and battery accounting as every joint.  `RotorSet.apply` adds the
aerodynamic thrust along the spin axis and the aerodynamic torque against the
spin, on the rotor body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fluid import drag_coefficient, lift_coefficient

#: Annuli per blade.  Converged: 16 against 64 moves static CT by <0.5%
#: (`experiments/rotor`).
N_ANNULI = 16


@dataclass
class RotorSpec:
    """Geometry of one fixed-pitch rotor."""

    radius: float
    pitch: float                 # geometric pitch, m per revolution
    blades: int = 2
    #: Constant-chord stand-in for a tapered blade; with `camber` below it
    #: reproduces the APC 10x4.7 SF to CP rms 0.003 and CT rms 0.012 over
    #: J = 0.1-0.6, static CT 19% low (`experiments/rotor`).
    chord_ratio: float = 0.12    # mean chord / R
    hub_ratio: float = 0.15      # hub radius / R
    #: Section camber as a fraction of chord.  Propeller sections are cambered,
    #: and the zero-lift angle it sets (thin-airfoil, ``-2 f/c``) is what makes
    #: the aerodynamic pitch larger than the geometric one.
    camber: float = 0.065        # fitted to the APC 10x4.7 SF static CT, the one fit
    #: +1: positive spin about the shaft axis pushes along it.  -1 is the mirror
    #: image, which a reflected part carries (a mirrored CCW propeller is a CW
    #: one).
    handed: int = 1

    def stations(self):
        r0 = self.hub_ratio * self.radius
        dr = (self.radius - r0) / N_ANNULI
        r = r0 + dr * (np.arange(N_ANNULI) + 0.5)
        c = np.full(N_ANNULI, self.chord_ratio * self.radius)
        theta = np.arctan2(self.pitch, 2.0 * np.pi * r)
        return r, dr, c, theta


def bemt(spec: RotorSpec, omega: float, v_ax: float, v_ip: float,
         rho: float, mu: float) -> tuple[float, float]:
    """Thrust (N, along the spin axis) and aerodynamic torque (N m, resisting
    the spin) of one rotor.  ``omega`` in rad/s, its sign ignored."""
    om = abs(float(omega))
    if om < 1e-6 or spec.radius <= 0.0:
        return 0.0, 0.0
    r, dr, c, theta = spec.stations()
    B = spec.blades
    R = spec.radius

    def blade(v):
        ua = v_ax + v
        ut = om * r
        phi = np.arctan2(ua, ut)
        W2 = ua * ua + ut * ut
        alpha = theta - phi + 2.0 * spec.camber
        re = rho * np.sqrt(W2) * c / max(mu, 1e-12)
        cl = lift_coefficient(alpha, re, np.full_like(r, 1e6), np.zeros_like(r))
        cd = drag_coefficient(alpha, re, np.full_like(r, 1e6), cl, 0.0)
        q = 0.5 * rho * W2 * B * c
        dT = q * (cl * np.cos(phi) - cd * np.sin(phi))
        dQ = q * (cl * np.sin(phi) + cd * np.cos(phi)) * r
        sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)
        f = B * (R - r) / (2.0 * r * sphi)
        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        return dT, dQ, np.maximum(F, 1e-3)

    def residual(v):
        dT, _, F = blade(v)
        mom = 4.0 * np.pi * r * rho * np.sqrt(v_ip * v_ip + (v_ax + v) ** 2) * v * F
        return dT - mom

    # Bisection per annulus on v >= 0.  Where the blade makes no positive thrust
    # at v = 0 (windmilling, or past zero-thrust advance ratio) v is 0.
    lo = np.zeros_like(r)
    hi = np.full_like(r, max(om * R, abs(v_ax), 1.0))
    g0 = residual(lo)
    active = g0 > 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        g = residual(mid)
        pos = g > 0.0
        lo = np.where(active & pos, mid, lo)
        hi = np.where(active & ~pos, mid, hi)
    v = np.where(active, 0.5 * (lo + hi), 0.0)
    dT, dQ, _ = blade(v)
    return float(np.sum(dT * dr)), float(np.sum(dQ * dr))


def bemt_many(spec: RotorSpec, omega, v_ax, v_ip, rho: float, mu: float):
    """`bemt` at many operating points of one rotor at once: ``omega``,
    ``v_ax``, ``v_ip`` are arrays of one length, every ``|omega|`` at least
    1e-6.  Returns thrust and torque arrays.

    The same arithmetic in the same order, one row per operating point, so
    each row is `bemt`'s answer to the bit (`tests/test_physics.py`).  Built
    2026-10-03 for `rotor_table`: its 510 scalar calls took ~2.5 s per rotor
    spec and medium, and once the per-step lookup was vectorised (`RotorBatch`)
    table building was the larger part of what a rotor cost a worker -- every
    new design brings new radius and pitch genes, so new tables, in every
    worker that evaluates it.
    """
    om = np.abs(np.asarray(omega, float))[:, None]
    vax = np.asarray(v_ax, float)[:, None]
    vip = np.asarray(v_ip, float)[:, None]
    r, dr, c, theta = spec.stations()
    B = spec.blades
    R = spec.radius
    shape = (len(om), len(r))

    def blade(v):
        ua = vax + v
        ut = om * r
        phi = np.arctan2(ua, ut)
        W2 = ua * ua + ut * ut
        alpha = theta - phi + 2.0 * spec.camber
        re = rho * np.sqrt(W2) * c / max(mu, 1e-12)
        cl = lift_coefficient(alpha, re, np.full(shape, 1e6), np.zeros(shape))
        cd = drag_coefficient(alpha, re, np.full(shape, 1e6), cl, 0.0)
        q = 0.5 * rho * W2 * B * c
        dT = q * (cl * np.cos(phi) - cd * np.sin(phi))
        dQ = q * (cl * np.sin(phi) + cd * np.cos(phi)) * r
        sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)
        f = B * (R - r) / (2.0 * r * sphi)
        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        return dT, dQ, np.maximum(F, 1e-3)

    def residual(v):
        dT, _, F = blade(v)
        mom = 4.0 * np.pi * r * rho * np.sqrt(vip * vip + (vax + v) ** 2) * v * F
        return dT - mom

    lo = np.zeros(shape)
    hi = np.broadcast_to(np.maximum(np.maximum(om * R, np.abs(vax)), 1.0), shape).copy()
    g0 = residual(lo)
    active = g0 > 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        g = residual(mid)
        pos = g > 0.0
        lo = np.where(active & pos, mid, lo)
        hi = np.where(active & ~pos, mid, hi)
    v = np.where(active, 0.5 * (lo + hi), 0.0)
    dT, dQ, _ = blade(v)
    # Row by row, as `bemt` sums one rotor's annuli: numpy's pairwise sum of a
    # 16-element row is the same whether the row stands alone or not, but that
    # is an implementation detail this does not lean on.
    T = np.array([np.sum(row) for row in dT * dr])
    Q = np.array([np.sum(row) for row in dQ * dr])
    return T, Q


#: Advance-ratio grids for the lookup table: axial ``J = V_ax / (Omega R)``
#: and in-plane ``mu = V_ip / (Omega R)``.
#: Denser near zero, where Glauert's term and the inflow change fastest.
TABLE_J = np.array([-0.3, -0.15, -0.05, 0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.27,
                    0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.2])
TABLE_MU = np.array([0.0, 0.03, 0.07, 0.12, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5])
#: Tip speeds the tables are built at, m/s, interpolated in log tip speed: the
#: Reynolds number is what the nondimensional coefficients still depend on, and
#: one table at 100 m/s was 12% off at 38 m/s.
TABLE_TIP_SPEEDS = (30.0, 80.0, 200.0)
_TABLES: dict = {}


def rotor_table(spec: RotorSpec, rho: float, mu_visc: float):
    """``(CT, CQ)`` over ``TABLE_J x TABLE_MU``, with ``T = rho Om^2 R^4 CT``
    and ``Q = rho Om^2 R^5 CQ``.  Built once per rotor and medium.

    `bemt` bisects every annulus every call -- 3.5 ms a rotor, so a quadrotor's
    8 s segment cost 28 s.  Thrust and torque are ``rho Om^2`` times functions
    of the two advance ratios (and weakly of Reynolds number), so a table built
    once is exact to its interpolation: within 2% of `bemt` over the UIUC points
    (`experiments/rotor`).
    """
    key = (round(spec.radius, 6), round(spec.pitch, 6), spec.blades, spec.chord_ratio,
           spec.hub_ratio, spec.camber, round(rho, 3), round(mu_visc, 9))
    if key in _TABLES:
        return _TABLES[key]
    ct, cq = _build_table(spec, rho, mu_visc)
    _TABLES[key] = (ct, cq)
    return ct, cq


def _build_table(spec: RotorSpec, rho: float, mu_visc: float, *, scalar: bool = False):
    """`rotor_table`'s grid, every operating point through `bemt_many` in one
    pass (``scalar``: one `bemt` call per point, the original loop, which
    `tests/test_physics.py` holds it equal to)."""
    R = spec.radius
    ct = np.zeros((len(TABLE_TIP_SPEEDS), len(TABLE_J), len(TABLE_MU)))
    cq = np.zeros_like(ct)
    pts = []
    for a, tip in enumerate(TABLE_TIP_SPEEDS):
        om = tip / max(R, 1e-6)
        norm_t = rho * om**2 * R**4
        for i, J in enumerate(TABLE_J):
            for k, mu in enumerate(TABLE_MU):
                pts.append((a, i, k, om, J * om * R, mu * om * R, norm_t))
    if scalar or R <= 0.0:
        TQ = [bemt(spec, p[3], p[4], p[5], rho, mu_visc) for p in pts]
    else:
        T, Q = bemt_many(spec, np.array([p[3] for p in pts]), np.array([p[4] for p in pts]),
                         np.array([p[5] for p in pts]), rho, mu_visc)
        TQ = list(zip(T.tolist(), Q.tolist()))
    for (a, i, k, _, _, _, norm_t), (T_, Q_) in zip(pts, TQ):
        ct[a, i, k] = T_ / norm_t
        cq[a, i, k] = Q_ / (norm_t * R)
    return ct, cq


def _bilinear(tab, J, mu):
    x = np.clip(J, TABLE_J[0], TABLE_J[-1])
    y = np.clip(mu, TABLE_MU[0], TABLE_MU[-1])
    i = min(int(np.searchsorted(TABLE_J, x, "right") - 1), len(TABLE_J) - 2)
    k = min(int(np.searchsorted(TABLE_MU, y, "right") - 1), len(TABLE_MU) - 2)
    tx = (x - TABLE_J[i]) / (TABLE_J[i + 1] - TABLE_J[i])
    ty = (y - TABLE_MU[k]) / (TABLE_MU[k + 1] - TABLE_MU[k])
    return ((1 - tx) * (1 - ty) * tab[i, k] + tx * (1 - ty) * tab[i + 1, k]
            + (1 - tx) * ty * tab[i, k + 1] + tx * ty * tab[i + 1, k + 1])


def rotor_forces(spec: RotorSpec, omega: float, v_ax: float, v_ip: float, medium_frac: float,
                 air, water) -> tuple[float, float]:
    """Thrust and torque from the tables, blended across the free surface by
    ``medium_frac`` (0 in air, 1 submerged) -- a propeller swims too."""
    om = abs(float(omega))
    if om < 1e-6 or spec.radius <= 0.0:
        return 0.0, 0.0
    R = spec.radius
    J, mu = v_ax / (om * R), v_ip / (om * R)
    out = np.zeros(2)
    for frac, fl in ((1.0 - medium_frac, air), (medium_frac, water)):
        if frac <= 0.0:
            continue
        ct, cq = rotor_table(spec, fl.rho, fl.mu)
        n = fl.rho * om**2 * R**4
        lt = np.log(np.clip(om * R, TABLE_TIP_SPEEDS[0], TABLE_TIP_SPEEDS[-1]))
        grid = np.log(TABLE_TIP_SPEEDS)
        a = min(int(np.searchsorted(grid, lt, "right") - 1), len(grid) - 2)
        w = (lt - grid[a]) / (grid[a + 1] - grid[a])
        c_t = (1 - w) * _bilinear(ct[a], J, mu) + w * _bilinear(ct[a + 1], J, mu)
        c_q = (1 - w) * _bilinear(cq[a], J, mu) + w * _bilinear(cq[a + 1], J, mu)
        out += frac * np.array([n * c_t, n * R * c_q])
    return float(out[0]), float(out[1])


_LOG_TIPS = np.log(TABLE_TIP_SPEEDS)


def rotor_forces_many(omega, v_ax, v_ip, frac_w, R, R4, tabs, rhos, rows=None):
    """`rotor_forces` for many rotors at once, each with its own table.

    ``tabs[m]`` is ``(n, len(TABLE_TIP_SPEEDS), len(TABLE_J), len(TABLE_MU),
    2)`` -- every rotor's ``(CT, CQ)`` table in medium ``m`` (0 air, 1 water),
    stacked (or more rotors' tables, with ``rows`` naming which row of ``tabs``
    belongs to each rotor); ``rhos[m]`` that medium's density; ``R4`` is
    ``R**4``.  A row whose
    medium fraction is zero may hold any table: it is never read into the
    result.  Elementwise the same arithmetic, in the same order, as the scalar
    `rotor_forces`, which stays the definition: `tests/test_physics.py` holds the
    two equal to the bit (2026-10-03).  It exists because the scalar version,
    called twice per rotor per step, was 82% of a rotor-heavy evaluation.
    """
    om = np.abs(omega)
    live = (om >= 1e-6) & (R > 0.0)
    om = np.where(live, om, 1.0)
    omR = om * R
    J, mu = v_ax / omR, v_ip / omR
    # ``om**2`` on a Python float is C ``pow``; numpy's ``**2`` is ``x*x``,
    # which differs in the last bit about once in a thousand.  An array
    # exponent keeps ``pow``.
    om2 = np.power(om, np.full_like(om, 2.0))
    lt = np.log(np.clip(omR, TABLE_TIP_SPEEDS[0], TABLE_TIP_SPEEDS[-1]))
    a = np.minimum(np.searchsorted(_LOG_TIPS, lt, "right") - 1, len(_LOG_TIPS) - 2)
    w = (lt - _LOG_TIPS[a]) / (_LOG_TIPS[a + 1] - _LOG_TIPS[a])
    # `_bilinear`'s corner and weights, the same for every table of a rotor.
    x = np.clip(J, TABLE_J[0], TABLE_J[-1])
    y = np.clip(mu, TABLE_MU[0], TABLE_MU[-1])
    i = np.minimum(np.searchsorted(TABLE_J, x, "right") - 1, len(TABLE_J) - 2)
    k = np.minimum(np.searchsorted(TABLE_MU, y, "right") - 1, len(TABLE_MU) - 2)
    tx = ((x - TABLE_J[i]) / (TABLE_J[i + 1] - TABLE_J[i]))[:, None]
    ty = ((y - TABLE_MU[k]) / (TABLE_MU[k + 1] - TABLE_MU[k]))[:, None]
    w = w[:, None]
    r = np.arange(len(om)) if rows is None else rows
    T = np.zeros(len(om))
    Q = np.zeros(len(om))
    for m, frac in ((0, 1.0 - frac_w), (1, frac_w)):
        on = live & (frac > 0.0)
        if not on.any():
            continue
        tab = tabs[m]

        def bil(a_):
            return ((1 - tx) * (1 - ty) * tab[r, a_, i, k] + tx * (1 - ty) * tab[r, a_, i + 1, k]
                    + (1 - tx) * ty * tab[r, a_, i, k + 1] + tx * ty * tab[r, a_, i + 1, k + 1])
        c = (1 - w) * bil(a) + w * bil(a + 1)            # (n, 2): c_t, c_q
        n_ = rhos[m] * om2 * R4
        T = np.where(on, T + frac * (n_ * c[:, 0]), T)
        Q = np.where(on, Q + frac * (n_ * R * c[:, 1]), Q)
    return T, Q


class RotorBatch:
    """The rotors of several machines, stepped as one array computation.

    `RotorSet.apply` (the single-machine path) is a batch of one, and
    `BatchedFluid.finish` (the search's path) is a batch of every machine in the
    shard, so both paths run this one implementation.  It replaced a Python
    loop over rotors that called `rotor_forces` twice per rotor per step, each
    call doing ~4 scalar table lookups per medium: 198 us per rotor-step, 82% of
    the wall of a rotor-heavy evaluation (`experiments/perf/rotor_cost.py`).
    """

    def __init__(self, sets) -> None:
        self.sets = [s for s in sets]
        self.off = np.cumsum([0] + [s.n for s in self.sets])
        self.n = int(self.off[-1])
        n = self.n
        flat = lambda name: [x for s in self.sets for x in getattr(s, name)]  # noqa: E731
        self.spec = flat("spec")
        self.axis = np.array(flat("axis_local"), float).reshape(n, 3)
        self.inertia = np.array(flat("inertia"), float)
        self.handed = np.array([float(sp.handed) for sp in self.spec])
        self.R = np.array([float(sp.radius) for sp in self.spec])
        self.R4 = np.array([float(sp.radius) ** 4 for sp in self.spec])
        shape = (n, len(TABLE_TIP_SPEEDS), len(TABLE_J), len(TABLE_MU), 2)
        self._tabs = [np.zeros(shape), np.zeros(shape)]
        self._have = [np.zeros(n, bool), np.zeros(n, bool)]
        self._fluid = [None, None]

    def _tables(self, m: int, fl, need) -> None:
        """Fill medium ``m``'s table for the rotors in ``need`` from the cache
        `rotor_table` keeps -- built when first used, as the scalar path did."""
        key = (fl.rho, fl.mu)
        if self._fluid[m] != key:
            self._fluid[m] = key
            self._have[m][:] = False
        for j in np.flatnonzero(need & ~self._have[m]):
            ct, cq = rotor_table(self.spec[j], fl.rho, fl.mu)
            self._tabs[m][j, ..., 0] = ct
            self._tabs[m][j, ..., 1] = cq
            self._have[m][j] = True

    def apply(self, items, t: float) -> np.ndarray:
        """``items``: ``(index of the set, model, data, medium)`` per machine to
        step.  Adds every rotor's thrust and torque to ``data.xfrc_applied`` and
        its drag damping split to ``dof_damping`` / ``qfrc_applied``, exactly as
        the per-rotor loop did.  Returns the total thrust per item, N."""
        groups: dict = {}
        for it in items:
            med = it[3]
            ss = med.sea_state
            sig = (med.air, med.water, tuple(med.current), tuple(med.wind),
                   ss.amplitude, ss.period, ss.wavelength, ss.direction)
            groups.setdefault(sig, []).append(it)
        totals = {}
        for group in groups.values():
            totals.update(self._apply_group(group, t))
        return np.array([totals[it[0]] for it in items])

    def _apply_group(self, items, t: float) -> dict:
        medium = items[0][3]
        sl, xm, om, cv, xi, com, dts = [], [], [], [], [], [], []
        for s_i, model, data, _ in items:
            s = self.sets[s_i]
            sl.append(slice(int(self.off[s_i]), int(self.off[s_i + 1])))
            dts.append(np.full(s.n, float(model.opt.timestep)))
            xm.append(data.xmat[s.body_arr])
            om.append(data.qvel[s.dof_arr])
            cv.append(data.cvel[s.body_arr])
            xi.append(data.xipos[s.body_arr])
            com.append(data.subtree_com[s.root_arr])
        rows = np.concatenate([np.arange(q.start, q.stop) for q in sl]) if len(sl) > 1 \
            else np.arange(sl[0].start, sl[0].stop)
        cat = (lambda v: np.concatenate(v)) if len(sl) > 1 else (lambda v: v[0])  # noqa: E731
        xm, omega, cv, pos, com, dt = cat(xm), cat(om), cat(cv), cat(xi), cat(com), cat(dts)
        n = len(rows)
        # The spin axis in the world: per row the same matrix-vector product
        # the loop did (a stacked matmul takes the same BLAS kernel; the
        # elementwise expansion does not, it misses the fused multiply-add).
        ax = (xm.reshape(n, 3, 3) @ self.axis[rows][:, :, None])[:, :, 0]
        # Hub velocity: `mj_objectVelocity` for mjOBJ_BODY in the world frame,
        # written out as `BatchedFluid.launch` does it (bit-exact to the call).
        off = pos - com
        ang = cv[:, 0:3]
        lin = cv[:, 3:6]
        v = np.empty((n, 3))
        v[:, 0] = lin[:, 0] - (off[:, 1] * ang[:, 2] - off[:, 2] * ang[:, 1])
        v[:, 1] = lin[:, 1] - (off[:, 2] * ang[:, 0] - off[:, 0] * ang[:, 2])
        v[:, 2] = lin[:, 2] - (off[:, 0] * ang[:, 1] - off[:, 1] * ang[:, 0])
        if medium.sea_state.amplitude <= 0.0:
            # A calm surface: every term of the two queries is elementwise.
            _, _, subf = medium.properties(pos, np.full(n, 0.01), t)
            flow = np.asarray(medium.flow_velocity(pos, t), float).reshape(n, 3)
        else:
            # Waves take a dot product whose BLAS kernel depends on how many
            # rows it is given, so the rows go one at a time, as before.
            subf, flow = np.empty(n), np.empty((n, 3))
            for j in range(n):
                subf[j] = medium.properties(pos[j][None, :], np.array([0.01]), t)[2][0]
                flow[j] = np.asarray(medium.flow_velocity(pos[j][None, :], t), float).reshape(3)
        rel = v - flow                                  # hub velocity through the fluid
        # With positive spin the blades push along +axis (see `apply` below).
        sgn = np.where(omega >= 0.0, 1.0, -1.0)
        td = (self.handed[rows] * sgn)[:, None] * ax
        v_ax = (rel * td).sum(axis=1)
        d = rel - v_ax[:, None] * td
        v_ip = np.sqrt((d * d).sum(axis=1))
        for m, fl, on in ((0, medium.air, (1.0 - subf) > 0.0), (1, medium.water, subf > 0.0)):
            need = on & (np.abs(omega) >= 1e-6)
            if self._fluid[m] != (fl.rho, fl.mu) or (need & ~self._have[m][rows]).any():
                full = np.zeros(self.n, bool)
                full[rows] = need
                self._tables(m, fl, full)
        tabs = self._tabs
        rhos = (medium.air.rho, medium.water.rho)
        R, R4 = self.R[rows], self.R4[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            T0, Q0 = rotor_forces_many(omega, v_ax, v_ip, subf, R, R4, tabs, rhos, rows)
            # `RotorSet._end_of_step`, row by row: backward Euler on the
            # rotor's own drag for the end-of-step speed.
            w_ = np.abs(omega)
            c = np.abs(Q0) / (w_ * w_)
            a = self.inertia[rows] / dt
            w2 = (-a + np.sqrt(a * a + 4.0 * c * a * w_)) / (2.0 * c)
            omega_e = np.where((w_ < 1e-6) | (c < 1e-12), omega, sgn * w2)
            second = omega_e != omega
            if second.any():
                T1, Q1 = rotor_forces_many(omega_e, v_ax, v_ip, subf, R, R4, tabs, rhos, rows)
                T = np.where(second, T1, T0)
                Q = np.where(second, Q1, Q0)
            else:
                T, Q = T0, Q0
            dq = 2.0 * np.abs(Q) / np.maximum(np.abs(omega_e), 1e-6)
        f_lin = T[:, None] * td
        f_ang = ((-Q) * sgn)[:, None] * ax
        damp = dq * omega
        out = {}
        at = 0
        for (s_i, model, data, _), q in zip(items, sl):
            s = self.sets[s_i]
            b = at + (q.stop - q.start)
            data.xfrc_applied[s.body_arr, :3] += f_lin[at:b]
            data.xfrc_applied[s.body_arr, 3:] += f_ang[at:b]
            model.dof_damping[s.dof_arr] += dq[at:b]
            data.qfrc_applied[s.dof_arr] += damp[at:b]
            s.last_thrust[:] = T[at:b]
            s.last_torque[:] = Q[at:b]
            total = 0.0
            for x in T[at:b].tolist():                  # the loop's own order
                total += x
            out[s_i] = total
            at = b
        return out


class RotorSet:
    """Every rotor of one machine, applied each step like `JetSet`."""

    def __init__(self, model, specs: dict) -> None:
        """``specs`` maps a rotor body name to its `RotorSpec`."""
        import mujoco

        self.body = []
        self.inertia = []
        self.dof = []
        self.spec = []
        self.axis_local = []
        for name, spec in specs.items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                continue
            j = int(model.body_jntadr[b])
            self.body.append(b)
            self.dof.append(int(model.jnt_dofadr[j]))
            ax = np.asarray(model.jnt_axis[j], float)
            self.axis_local.append(ax)
            self.spec.append(spec)
            # Inertia about the spin axis: the body's principal inertia, turned
            # into the body frame by `body_iquat`, projected on the axis, plus
            # the joint's armature.  Constant, so computed once.
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, model.body_iquat[b])
            Rq = Rq.reshape(3, 3)
            I_body = Rq @ np.diag(model.body_inertia[b]) @ Rq.T
            self.inertia.append(float(ax @ I_body @ ax) + float(model.dof_armature[model.jnt_dofadr[j]]))
        self.n = len(self.body)
        self.last_thrust = np.zeros(self.n)
        self.last_torque = np.zeros(self.n)
        self.body_arr = np.array(self.body, dtype=int)
        self.dof_arr = np.array(self.dof, dtype=int)
        self.root_arr = np.array([int(model.body_rootid[b]) for b in self.body], dtype=int)
        # The batch writes with fancy-indexed `+=`, which is the loop's
        # arithmetic only while no two rotors share a body or a DOF.
        assert len(set(self.body)) == self.n and len(set(self.dof)) == self.n
        self._batch = None

    def _end_of_step(self, model, data, k: int, omega: float, Q: float) -> float:
        dof = self.dof[k]
        w = abs(float(omega))
        if w < 1e-6:
            return omega
        inertia = self.inertia[k]                          # rotor + armature
        dt = float(model.opt.timestep)
        # The drag alone: it is the stiff part.  The motor's torque is left to
        # MuJoCo's own implicit velocity servo -- predicting it from last step's
        # actuator force led the servo by a step and cost the reference
        # quadrotor's controller 0.1 of its air score (0.96 -> 0.86).
        s = 1.0 if omega >= 0.0 else -1.0
        c = abs(Q) / (w * w)
        a = inertia / dt
        if c < 1e-12:
            return omega
        w2 = (-a + math.sqrt(a * a + 4.0 * c * a * w)) / (2.0 * c)
        return s * w2

    def apply(self, model, data, medium, t: float) -> float:
        """Add rotor thrust and aerodynamic torque to ``data.xfrc_applied``.
        Returns total thrust, N.  A `RotorBatch` of one machine: the batched
        path steps every machine of a shard through the same code."""
        if self.n == 0:
            return 0.0
        if self._batch is None:
            self._batch = RotorBatch([self])
        return float(self._batch.apply([(0, model, data, medium)], t)[0])

    def apply_per_rotor(self, model, data, medium, t: float) -> float:
        """The per-rotor loop `RotorBatch` replaced, kept as the readable
        statement of the physics and as the reference
        `tests/test_physics.py` holds the batch to, bit for bit.  No
        evaluation path calls it: at 198 us per rotor-step it was 82% of a
        rotor-heavy evaluation's wall (2026-10-03)."""
        if self.n == 0:
            return 0.0
        import mujoco

        v6 = np.zeros(6)
        total = 0.0
        for k in range(self.n):
            b = self.body[k]
            R = data.xmat[b].reshape(3, 3)
            ax = R @ self.axis_local[k]
            omega = float(data.qvel[self.dof[k]])
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            pos = data.xipos[b]
            _, _, subf = medium.properties(pos[None, :], np.array([0.01]), t)
            flow = np.asarray(medium.flow_velocity(pos[None, :], t), float).reshape(3)
            rel = v6[3:] - flow                     # hub velocity through the fluid
            # The rotor pushes fluid against the direction its thrust acts in:
            # with positive spin the blades' pitch makes thrust along +axis,
            # and the flow meets the disc from the +axis side as the hub moves.
            sgn = 1.0 if omega >= 0.0 else -1.0
            thrust_dir = self.spec[k].handed * sgn * ax
            v_ax = float(rel @ thrust_dir)
            v_ip = float(np.linalg.norm(rel - v_ax * thrust_dir))
            T0, Q0 = rotor_forces(self.spec[k], omega, v_ax, v_ip, float(subf[0]),
                                  medium.air, medium.water)
            # Backward Euler for the spin: in water a rotor brakes in about a
            # millisecond, far inside a 4 ms step, and thrust taken at the
            # start-of-step speed gave a plunging quadrotor ~40 N s of impulse
            # per rotor where momentum theory and its 13 J of stored energy
            # allow ~3 -- the machine was fired out of the water.  Solve the
            # rotor's own drag equation for the end-of-step speed,
            #     I (w' - w) / dt = - c w'|w'|,   c = |Q| / w^2,
            # and take thrust and torque there.  In air at hover that is ~1%
            # below w; in water it is the braking.
            omega_e = self._end_of_step(model, data, k, omega, Q0)
            if omega_e != omega:
                T, Q = rotor_forces(self.spec[k], omega_e, v_ax, v_ip, float(subf[0]),
                                    medium.air, medium.water)
            else:
                T, Q = T0, Q0
            data.xfrc_applied[b, :3] += T * thrust_dir
            data.xfrc_applied[b, 3:] += -Q * sgn * ax
            # The drag torque grows as Omega^2 and meets a rotor's tiny inertia:
            # a propeller spinning at 550 rad/s put under water, where the torque
            # is ~1000x, reached |qvel| 787,158 and a bad-qacc reset
            # (2026-09-26, arch43 stopped at gen 3).  The F-03 cure: split its
            # damping dQ/dOmega = 2Q/|Omega| into MuJoCo's implicit
            # `dof_damping` and add it back explicitly, so the torque at this
            # state is unchanged and the stiff part is integrated implicitly.
            # Added to what the fluid solver wrote this step, as jets add.
            # |Q|, not Q: a rotor slower than the flow through it windmills,
            # Q turns negative, and a negative dof_damping is an instability of
            # its own (the first version of this oscillated +-100 rad/s).
            dq = 2.0 * abs(Q) / max(abs(omega_e), 1e-6)
            dof = self.dof[k]
            model.dof_damping[dof] += dq
            data.qfrc_applied[dof] += dq * omega
            self.last_thrust[k] = T
            self.last_torque[k] = Q
            total += T
        return total
