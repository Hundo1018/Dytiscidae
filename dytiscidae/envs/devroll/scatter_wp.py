"""The scatter between two ``mj_step``s on the device (IMPL_1010 A1b).

``RotorBatch.apply`` (physics/rotor.py), ``BatchedPower.step`` (physics/energy.py),
``TriphibianEnv.servo_command`` (envs/triphibian.py), ``JetSet.apply``
(physics/jet.py), ``InducedFlow.update``, ``ImplicitAeroDamping`` (its strip
damping, projection, cadence, wet-set staleness and compensation) and the added
mass into ``body_mass``/``body_inertia`` (batchroll.py:538-540), as Warp kernels
over flat arrays.  A row is one rotor, bell, strip, body, dof, actuator or one
world's machine; the caller lays worlds out back to back and passes each row's
index into its world's arrays.  The kernels return loads and model fields;
``model_wp.DeviceBody`` writes them into the MJWarp ``Model`` and ``Data``
(``xfrc_applied``, ``dof_damping``, ``qfrc_applied``, ``body_mass``,
``body_inertia``).  Each class has a ``launch`` on device arrays, which reads
nothing back to the host, and a numpy-in, numpy-out wrapper around it that the
parity tests call.

What stays on the host: the rotor tables (``rotor_table`` builds them once per
rotor and medium, they are constants on the device), and whether the CPG left a
rate for the servo (``CPG.pop_rate`` is host state until A3).  The medium is the
flat calm sea of ``fluid_wp``; waves are not ported.

Kernels are built per precision by :func:`make_kernels`, as in ``fluid_wp``.
"""

import numpy as _np

import warp as _wp

from dytiscidae.envs.devroll import fluid_wp as _fluid_wp
from dytiscidae.physics.fluid import ImplicitAeroDamping as _ImplicitAeroDamping
from dytiscidae.physics.fluid import InducedFlow as _InducedFlow
from dytiscidae.physics.rotor import TABLE_J, TABLE_MU, TABLE_TIP_SPEEDS, rotor_table

_KERNELS: dict = {}

_NA = len(TABLE_TIP_SPEEDS)
_NJ = len(TABLE_J)
_NM = len(TABLE_MU)
_TIP_LO = float(TABLE_TIP_SPEEDS[0])
_TIP_HI = float(TABLE_TIP_SPEEDS[-1])
_REFRESH = _wp.constant(int(_ImplicitAeroDamping.REFRESH_EVERY))


def make_kernels(T):
    """Return the kernels for precision ``T`` (``rotor``, ``power_terms``,
    ``power_charge``, ``servo``) and the 3-vector type ``V``, cached."""
    if T in _KERNELS:
        return _KERNELS[T]
    V = _wp.vec3d if T == _wp.float64 else _wp.vec3
    M = _wp.mat33d if T == _wp.float64 else _wp.mat33
    V2 = _wp.vec2d if T == _wp.float64 else _wp.vec2

    @_wp.func
    def lower_index(grid: _wp.array(dtype=T), x: T, n: int) -> int:
        # searchsorted(grid, x, "right") - 1, capped at n - 2; x is already clipped.
        i = int(0)
        for j in range(1, n):
            if x >= grid[j]:
                i = j
        return _wp.min(i, n - 2)

    @_wp.func
    def table_at(tab: _wp.array(dtype=T), r: int, a: int, i: int, k: int, c: int) -> T:
        return tab[(((r * _NA + a) * _NJ + i) * _NM + k) * 2 + c]

    @_wp.func
    def bilinear(tab: _wp.array(dtype=T), r: int, a: int, i: int, k: int, c: int,
                 tx: T, ty: T) -> T:
        one = T(1.0)
        return ((one - tx) * (one - ty) * table_at(tab, r, a, i, k, c)
                + tx * (one - ty) * table_at(tab, r, a, i + 1, k, c)
                + (one - tx) * ty * table_at(tab, r, a, i, k + 1, c)
                + tx * ty * table_at(tab, r, a, i + 1, k + 1, c))

    @_wp.func
    def forces(r: int, omega: T, v_ax: T, v_ip: T, frac_w: T, R: T, R4: T,
               tab_air: _wp.array(dtype=T), tab_w: _wp.array(dtype=T),
               rho_air: T, rho_w: T, grid_j: _wp.array(dtype=T),
               grid_mu: _wp.array(dtype=T), grid_tip: _wp.array(dtype=T)) -> V2:
        # rotor_forces_many for one rotor: (thrust, torque).
        om = _wp.abs(omega)
        live = om >= T(1e-6) and R > T(0.0)
        if not live:
            om = T(1.0)
        omR = om * R
        J = v_ax / omR
        mu = v_ip / omR
        om2 = om * om
        lt = _wp.log(_wp.clamp(omR, T(_TIP_LO), T(_TIP_HI)))
        a = lower_index(grid_tip, lt, _NA)
        w = (lt - grid_tip[a]) / (grid_tip[a + 1] - grid_tip[a])
        x = _wp.clamp(J, grid_j[0], grid_j[_NJ - 1])
        y = _wp.clamp(mu, grid_mu[0], grid_mu[_NM - 1])
        i = lower_index(grid_j, x, _NJ)
        k = lower_index(grid_mu, y, _NM)
        tx = (x - grid_j[i]) / (grid_j[i + 1] - grid_j[i])
        ty = (y - grid_mu[k]) / (grid_mu[k + 1] - grid_mu[k])
        thrust = T(0.0)
        torque = T(0.0)
        if live:
            f_air = T(1.0) - frac_w
            if f_air > T(0.0):
                ct = (T(1.0) - w) * bilinear(tab_air, r, a, i, k, 0, tx, ty) \
                    + w * bilinear(tab_air, r, a + 1, i, k, 0, tx, ty)
                cq = (T(1.0) - w) * bilinear(tab_air, r, a, i, k, 1, tx, ty) \
                    + w * bilinear(tab_air, r, a + 1, i, k, 1, tx, ty)
                n_ = rho_air * om2 * R4
                thrust = thrust + f_air * (n_ * ct)
                torque = torque + f_air * (n_ * R * cq)
            if frac_w > T(0.0):
                ct = (T(1.0) - w) * bilinear(tab_w, r, a, i, k, 0, tx, ty) \
                    + w * bilinear(tab_w, r, a + 1, i, k, 0, tx, ty)
                cq = (T(1.0) - w) * bilinear(tab_w, r, a, i, k, 1, tx, ty) \
                    + w * bilinear(tab_w, r, a + 1, i, k, 1, tx, ty)
                n_ = rho_w * om2 * R4
                thrust = thrust + frac_w * (n_ * ct)
                torque = torque + frac_w * (n_ * R * cq)
        return V2(thrust, torque)

    @_wp.kernel
    def rotor_wp(
        xmat: _wp.array(dtype=M), axis: _wp.array(dtype=V), omega: _wp.array(dtype=T),
        ang: _wp.array(dtype=V), lin: _wp.array(dtype=V), xipos: _wp.array(dtype=V),
        com: _wp.array(dtype=V), subf: _wp.array(dtype=T), flow: _wp.array(dtype=V),
        inertia: _wp.array(dtype=T), handed: _wp.array(dtype=T), radius: _wp.array(dtype=T),
        radius4: _wp.array(dtype=T), dt: _wp.array(dtype=T), table_row: _wp.array(dtype=int),
        tab_air: _wp.array(dtype=T), tab_w: _wp.array(dtype=T), rho_air: T, rho_w: T,
        grid_j: _wp.array(dtype=T), grid_mu: _wp.array(dtype=T), grid_tip: _wp.array(dtype=T),
        f_lin: _wp.array(dtype=V), f_ang: _wp.array(dtype=V), dq_out: _wp.array(dtype=T),
        damp_out: _wp.array(dtype=T), thrust_out: _wp.array(dtype=T),
        torque_out: _wp.array(dtype=T),
    ):
        j = _wp.tid()
        r = table_row[j]
        ax = xmat[j] * axis[j]
        # Hub velocity: mj_objectVelocity for the body, in the world frame.
        off = xipos[j] - com[j]
        rel = lin[j] - _wp.cross(off, ang[j]) - flow[j]
        om = omega[j]
        sgn = T(-1.0)
        if om >= T(0.0):
            sgn = T(1.0)
        td = (handed[j] * sgn) * ax
        v_ax = _wp.dot(rel, td)
        d = rel - v_ax * td
        v_ip = _wp.sqrt(_wp.dot(d, d))
        R = radius[j]
        R4 = radius4[j]
        tq = forces(r, om, v_ax, v_ip, subf[j], R, R4, tab_air, tab_w, rho_air, rho_w,
                    grid_j, grid_mu, grid_tip)
        thrust = tq[0]
        torque = tq[1]
        # RotorSet._end_of_step: backward Euler on the rotor's own drag.
        w_ = _wp.abs(om)
        om_e = om
        if w_ >= T(1e-6):
            c = _wp.abs(torque) / (w_ * w_)
            if c >= T(1e-12):
                a = inertia[j] / dt[j]
                w2 = (-a + _wp.sqrt(a * a + T(4.0) * c * a * w_)) / (T(2.0) * c)
                om_e = sgn * w2
        if om_e != om:
            tq = forces(r, om_e, v_ax, v_ip, subf[j], R, R4, tab_air, tab_w, rho_air, rho_w,
                        grid_j, grid_mu, grid_tip)
            thrust = tq[0]
            torque = tq[1]
        dq = T(2.0) * _wp.abs(torque) / _wp.max(_wp.abs(om_e), T(1e-6))
        f_lin[j] = thrust * td
        f_ang[j] = (-torque * sgn) * ax
        dq_out[j] = dq
        damp_out[j] = dq * om
        thrust_out[j] = thrust
        torque_out[j] = torque

    @_wp.kernel
    def power_terms(
        force: _wp.array(dtype=T), velocity: _wp.array(dtype=T), seal: _wp.array(dtype=T),
        ratio: _wp.array(dtype=T), gear: _wp.array(dtype=T), km: _wp.array(dtype=T),
        k_iron: _wp.array(dtype=T), k_visc: _wp.array(dtype=T), rating: _wp.array(dtype=T),
        terms: _wp.array(dtype=T), over: _wp.array(dtype=T),
    ):
        # BatchedPower.step's elementwise loss terms for one actuator.
        i = _wp.tid()
        raw = _wp.abs(force[i])
        om = _wp.abs(velocity[i])
        # np.sign(|om|): 0 at rest, so a still sealed shaft pays no seal power.
        moving = T(0.0)
        if om > T(0.0):
            moving = T(1.0)
        tau = raw + moving * seal[i]
        om_m = om * ratio[i]
        cp = _wp.abs(tau) / gear[i] / km[i]
        cr = raw / gear[i] / km[i]
        terms[i] = _wp.abs(tau * om) + cp * cp + k_iron[i] * om_m + k_visc[i] * om_m * om_m
        over[i] = cr * cr / rating[i]

    @_wp.kernel
    def power_charge(
        off: _wp.array(dtype=int), terms: _wp.array(dtype=T), over: _wp.array(dtype=T),
        avionics_w: _wp.array(dtype=T), p_max: _wp.array(dtype=T), dt: _wp.array(dtype=T),
        active: _wp.array(dtype=int), energy_j: _wp.array(dtype=_wp.float64),
        total_j: _wp.array(dtype=_wp.float64), actuator_j: _wp.array(dtype=_wp.float64),
        avionics_j: _wp.array(dtype=_wp.float64), t: _wp.array(dtype=_wp.float64),
        max_overload: _wp.array(dtype=T), peak_draw: _wp.array(dtype=T),
        overdraw_time: _wp.array(dtype=T),
    ):
        # One world's machine: the sum and max over its own actuators, then
        # PowerBudget.charge and Battery.draw. The energy ledger is float64 at
        # every precision: at 9e5 J a float32 ledger resolves 0.0625 J, against
        # about 1.3 J drawn per step, and drifted 0.4% high over 50 steps.
        m = _wp.tid()
        if active[m] == 0:
            return
        a = off[m]
        b = off[m + 1]
        p = avionics_w[m]
        if b > a:
            worst = over[a]
            for i in range(a, b):
                p = p + terms[i]
                worst = _wp.max(worst, over[i])
            max_overload[m] = _wp.max(max_overload[m], worst)
        h = _wp.float64(dt[m])
        avionics_j[m] = avionics_j[m] + _wp.float64(avionics_w[m]) * h
        actuator_j[m] = actuator_j[m] + _wp.float64(p - avionics_w[m]) * h
        total_j[m] = total_j[m] + _wp.float64(p) * h
        t[m] = t[m] + h
        peak_draw[m] = _wp.max(peak_draw[m], p)
        overdraw = p / _wp.max(p_max[m], T(1e-9))
        if overdraw > T(1.0):
            overdraw_time[m] = overdraw_time[m] + dt[m]
            p = p * (T(1.0) + T(0.25) * (overdraw - T(1.0)))
        energy_j[m] = energy_j[m] - _wp.float64(p) * h
        if not (energy_j[m] > T(0.0)):
            active[m] = 0

    @_wp.kernel
    def servo_wp(
        target: _wp.array(dtype=T), rate: _wp.array(dtype=T), lead: _wp.array(dtype=T),
        world: _wp.array(dtype=int), has_rate: _wp.array(dtype=int), ctrl: _wp.array(dtype=T),
    ):
        # The target plus the servo's velocity feed-forward, where the CPG left a rate.
        i = _wp.tid()
        if has_rate[world[i]] != 0:
            ctrl[i] = target[i] + lead[i] * rate[i]
        else:
            ctrl[i] = target[i]

    @_wp.kernel
    def jets_wp(
        omega: _wp.array(dtype=T), lo: _wp.array(dtype=T),
        hi: _wp.array(dtype=T), volume: _wp.array(dtype=T), stroke: _wp.array(dtype=T),
        orifice: _wp.array(dtype=T), refill: _wp.array(dtype=T), xmat: _wp.array(dtype=M),
        axis: _wp.array(dtype=V), rho: _wp.array(dtype=T), subf: _wp.array(dtype=T),
        f_out: _wp.array(dtype=V), thrust_out: _wp.array(dtype=T), c_pump_out: _wp.array(dtype=T),
    ):
        # JetSet.apply for one bell: the thrust along -axis and the pumping
        # load as a damping coefficient on the driving joint.
        j = _wp.tid()
        span = _wp.max(hi[j] - lo[j], T(1e-6))
        q = volume[j] * stroke[j] * (omega[j] / span)        # -dV/dt, + while expelling
        coeff = refill[j]
        if q > T(0.0):
            coeff = T(1.0)
        area = _wp.max(orifice[j], T(1e-6))
        thrust = coeff * rho[j] * q * _wp.abs(q) / area * subf[j]
        f_out[j] = -thrust * (xmat[j] * axis[j])
        thrust_out[j] = thrust
        dv = volume[j] * stroke[j] / span
        c_pump_out[j] = (coeff * T(0.5) * rho[j] * dv * dv * dv * _wp.abs(omega[j]) * subf[j]
                         / (area * area))

    @_wp.kernel
    def jet_sums(
        off: _wp.array(dtype=int), thrust: _wp.array(dtype=T), c_pump: _wp.array(dtype=T),
        omega: _wp.array(dtype=T), driven: _wp.array(dtype=int),
        total_out: _wp.array(dtype=T), pump_out: _wp.array(dtype=T),
    ):
        # Per world: apply's return value (sum |thrust|) and last_pump_power.
        m = _wp.tid()
        tot = T(0.0)
        pump = T(0.0)
        for j in range(off[m], off[m + 1]):
            tot = tot + _wp.abs(thrust[j])
            if driven[j] != 0:
                pump = pump + c_pump[j] * omega[j] * omega[j]
        total_out[m] = tot
        pump_out[m] = pump

    @_wp.func
    def disc_g(x: T, vp2: T, v_par: T, k: T) -> V2:
        # InducedFlow.update's g: the residual and its derivative.
        sq = _wp.sqrt(vp2 + (v_par - x) * (v_par - x))
        return V2(x * sq - k, sq + x * (x - v_par) / _wp.max(sq, T(1e-12)))

    @_wp.kernel
    def inflow_wp(
        F: _wp.array(dtype=V), Vf: _wp.array(dtype=V), rho: _wp.array(dtype=T),
        dt: _wp.array(dtype=T), span: _wp.array(dtype=T), area: _wp.array(dtype=T),
        due: _wp.array(dtype=int), w: _wp.array(dtype=_wp.vec3d),
    ):
        # InducedFlow.update for one world's machine, on the steps the host's
        # cadence marks due.  The lagged state w is float64 at every precision.
        m = _wp.tid()
        if due[m] == 0:
            return
        zero = T(0.0)
        Tm = _wp.length(F[m])
        target = V(zero, zero, zero)
        if not (area[m] <= T(1e-9) or Tm < T(1e-9) or rho[m] <= zero):
            f = F[m] / Tm
            k = Tm / (T(2.0) * rho[m] * area[m])
            v_par = _wp.dot(Vf[m], f)
            vp2 = _wp.max(_wp.dot(Vf[m], Vf[m]) - v_par * v_par, zero)
            lo = zero
            hi = _wp.max(T(1.0), _wp.sqrt(k)) * T(4.0) + _wp.abs(v_par) + _wp.sqrt(vp2)
            while disc_g(hi, vp2, v_par, k)[0] < zero:
                hi = hi * T(2.0)
            wd = w[m]
            x = _wp.min(_wp.max(T(_wp.length(wd)), lo), hi)
            if x == zero:
                x = T(0.5) * hi
            for _it in range(60):
                g = disc_g(x, vp2, v_par, k)
                if g[0] < zero:
                    lo = x
                else:
                    hi = x
                if hi - lo < T(1e-12) * _wp.max(hi, T(1.0)):
                    break
                xn = T(0.5) * (lo + hi)
                if g[1] > zero:
                    xn = x - g[0] / g[1]
                if lo < xn and xn < hi:
                    x = xn
                else:
                    x = T(0.5) * (lo + hi)
            target = -f * x
        through = _wp.length(Vf[m] + target)
        tau = T(0.5) * span[m] / _wp.max(through, T(0.05))
        a = _wp.float64(T(1.0) - _wp.exp(-dt[m] / _wp.max(tau, T(1e-6))))
        tgt = _wp.vec3d(_wp.float64(target[0]), _wp.float64(target[1]), _wp.float64(target[2]))
        w[m] = w[m] + (tgt - w[m]) * a

    # --- ImplicitAeroDamping -------------------------------------------
    # Worlds back to back: strips [poff[w], poff[w+1]), bodies [boff..), dofs
    # [voff..); a strip's body and root are global body indices; ``anc`` is each
    # world's (nbody, nv) mask flattened from aoff[w].  The moments and the
    # projection run in float64 at every precision: the moment form expands
    # sum b |lin + rot x r|^2, which cancels when a strip sits near a dof's axis
    # far from its root's centre of mass, and in float32 it lost 1% of B on a
    # medusa bell joint (tests/test_devroll.py, the step replay).
    D3 = _wp.vec3d
    DM = _wp.mat33d
    F64 = _wp.float64

    @_wp.func
    def d3(x: V) -> _wp.vec3d:
        return _wp.vec3d(_wp.float64(x[0]), _wp.float64(x[1]), _wp.float64(x[2]))

    @_wp.kernel
    def damp_due(count: _wp.array(dtype=int), stale: _wp.array(dtype=int),
                 due: _wp.array(dtype=int)):
        # ImplicitAeroDamping.due, read before apply as the batched path reads it.
        w = _wp.tid()
        due[w] = 0
        if stale[w] != 0 or count[w] % _REFRESH == 0:
            due[w] = 1

    @_wp.kernel
    def strip_b(q: _wp.array(dtype=T), area: _wp.array(dtype=T), lift: _wp.array(dtype=T),
                drag: _wp.array(dtype=T), rho: _wp.array(dtype=T), ar: _wp.array(dtype=T),
                is_wing: _wp.array(dtype=int), lift_scale: _wp.array(dtype=T),
                world: _wp.array(dtype=int), b: _wp.array(dtype=T)):
        # strip_damping: 2|F|/U + q S CL_alpha / U, zero below U = 1e-3.
        i = _wp.tid()
        U = _wp.sqrt(T(2.0) * _wp.max(q[i], T(0.0)) / _wp.max(rho[i], T(1e-9)))
        slope = T(0.0)
        if is_wing[i] != 0:
            slope = T(2.0) * T(_wp.pi) / (T(1.0) + T(2.0) / _wp.max(ar[i], T(0.1)))
        slope = slope * lift_scale[world[i]]
        fa = _wp.sqrt(lift[i] * lift[i] + drag[i] * drag[i])
        b[i] = T(0.0)
        if U > T(1e-3):
            b[i] = (T(2.0) * fa + q[i] * area[i] * slope) / _wp.max(U, T(1e-3))

    @_wp.kernel
    def body_moments(poff: _wp.array(dtype=int), body_world: _wp.array(dtype=int),
                     due: _wp.array(dtype=int), strip_body: _wp.array(dtype=int),
                     strip_root: _wp.array(dtype=int), pos: _wp.array(dtype=V),
                     com: _wp.array(dtype=V), b: _wp.array(dtype=T),
                     B0: _wp.array(dtype=F64), B1: _wp.array(dtype=D3),
                     B2: _wp.array(dtype=DM)):
        # projected's per-body moments: sum b, sum b r, sum b (|r|^2 I - r r^T).
        g = _wp.tid()
        w = body_world[g]
        if due[w] == 0:
            return
        s0 = F64(0.0)
        s1 = D3(F64(0.0), F64(0.0), F64(0.0))
        s2 = DM(F64(0.0), F64(0.0), F64(0.0), F64(0.0), F64(0.0), F64(0.0), F64(0.0),
                F64(0.0), F64(0.0))
        for i in range(poff[w], poff[w + 1]):
            if strip_body[i] == g:
                r = d3(pos[i]) - d3(com[strip_root[i]])
                bi = F64(b[i])
                s0 = s0 + bi
                s1 = s1 + bi * r
                s2 = s2 + bi * (_wp.dot(r, r) * _wp.identity(n=3, dtype=F64)
                                - _wp.outer(r, r))
        B0[g] = s0
        B1[g] = s1
        B2[g] = s2

    @_wp.kernel
    def dof_project(boff: _wp.array(dtype=int), voff: _wp.array(dtype=int),
                    aoff: _wp.array(dtype=int), dof_world: _wp.array(dtype=int),
                    due: _wp.array(dtype=int), anc: _wp.array(dtype=int),
                    rot: _wp.array(dtype=V), lin: _wp.array(dtype=V),
                    B0: _wp.array(dtype=F64), B1: _wp.array(dtype=D3), B2: _wp.array(dtype=DM),
                    last_b: _wp.array(dtype=F64)):
        # B_k = sum over bodies dof k moves of B0|lin|^2 + 2 B1.(lin x rot) + rot.B2 rot.
        k = _wp.tid()
        w = dof_world[k]
        if due[w] == 0:
            return
        nv = voff[w + 1] - voff[w]
        kl = k - voff[w]
        nb = boff[w + 1] - boff[w]
        rt = d3(rot[k])
        ln = d3(lin[k])
        lxw = _wp.cross(ln, rt)
        val = F64(0.0)
        for j in range(nb):
            if anc[aoff[w] + j * nv + kl] != 0:
                g = boff[w] + j
                val = val + (B0[g] * _wp.dot(ln, ln) + F64(2.0) * _wp.dot(B1[g], lxw)
                             + _wp.dot(rt, B2[g] * rt))
        last_b[k] = val

    @_wp.func
    def wet_at(p: V, h: T) -> int:
        # MediumField.depth on the flat calm sea: depth = -z; wet if depth > -half.
        if -p[2] > -h:
            return 1
        return 0

    @_wp.kernel
    def wet_stale(poff: _wp.array(dtype=int), due: _wp.array(dtype=int),
                  pos: _wp.array(dtype=V), half: _wp.array(dtype=T),
                  wet_ref: _wp.array(dtype=int), prev_pos: _wp.array(dtype=V),
                  have_ref: _wp.array(dtype=int), have_prev: _wp.array(dtype=int),
                  stale: _wp.array(dtype=int), count: _wp.array(dtype=int)):
        # The wet set B was formed on, and whether this step or the next one on
        # this step's motion leaves it (apply, fluid.py:727-743); then the count.
        w = _wp.tid()
        st = int(0)
        for i in range(poff[w], poff[w + 1]):
            wet = wet_at(pos[i], half[i])
            if due[w] != 0:
                wet_ref[i] = wet
            ahead = wet
            if have_prev[w] != 0:
                ahead = wet_at(T(2.0) * pos[i] - prev_pos[i], half[i])
            if wet != wet_ref[i] or ahead != wet_ref[i]:
                st = 1
            prev_pos[i] = pos[i]
        if due[w] != 0:
            have_ref[w] = 1
        if have_ref[w] == 0:
            st = 0
        stale[w] = st
        have_prev[w] = 1
        count[w] = count[w] + 1

    @_wp.kernel
    def damp_out(dof_world: _wp.array(dtype=int), enabled: _wp.array(dtype=int),
                 base: _wp.array(dtype=T), last_b: _wp.array(dtype=_wp.float64),
                 qvel: _wp.array(dtype=T), dof_damping: _wp.array(dtype=T),
                 qfrc: _wp.array(dtype=T)):
        # dof_damping = base + B, and the compensation qfrc_applied = B qvel that
        # makes the split force-neutral at the current state; ``clear`` when off.
        k = _wp.tid()
        lb = T(0.0)
        if enabled[dof_world[k]] != 0:
            lb = T(last_b[k])
        dof_damping[k] = base[k] + lb
        qfrc[k] = lb * qvel[k]

    @_wp.kernel
    def added_mass_into_model(
        boff: _wp.array(dtype=int), body_world: _wp.array(dtype=int),
        dry_mass: _wp.array(dtype=T), dry_inertia: _wp.array(dtype=V),
        m_body: _wp.array(dtype=T), lever2: _wp.array(dtype=T),
        body_mass: _wp.array(dtype=T), body_inertia: _wp.array(dtype=V),
    ):
        # batchroll.finish:538-540, per world: the added mass into the model's
        # mass and, at its lever arm, into the principal inertia.
        g = _wp.tid()
        body_mass[g] = dry_mass[g] + m_body[g]
        body_inertia[g] = dry_inertia[g] + (m_body[g] * lever2[g]) * V(T(1.0), T(1.0), T(1.0))

    _KERNELS[T] = {"rotor": rotor_wp, "power_terms": power_terms,
                   "power_charge": power_charge, "servo": servo_wp,
                   "jets": jets_wp, "jet_sums": jet_sums, "inflow": inflow_wp,
                   "damp_due": damp_due, "strip_b": strip_b, "body_moments": body_moments,
                   "dof_project": dof_project, "wet_stale": wet_stale, "damp_out": damp_out,
                   "added_mass": added_mass_into_model, "V": V, "M": M}
    return _KERNELS[T]


def _np_dtype(dtype):
    return _np.float64 if dtype == _wp.float64 else _np.float32


class DeviceRotors:
    """The rotor tables of several ``RotorSet``s on the device, and their kernel.

    ``sets`` are the machines' ``RotorSet``s in world order (as ``RotorBatch``
    takes them); ``air`` and ``water`` the two ``Fluid``s.  Every rotor's table
    is built on the host for both media and uploaded once.
    """

    def __init__(self, sets, air, water, *, dtype=_wp.float64) -> None:
        npdt = _np_dtype(dtype)
        self._dtype = dtype
        self.sets = list(sets)
        self.off = _np.cumsum([0] + [s.n for s in self.sets])
        self.n = int(self.off[-1])
        specs = [sp for s in self.sets for sp in s.spec]
        self.air, self.water = air, water
        tabs = []
        for fl in (air, water):
            tab = _np.zeros((self.n, _NA, _NJ, _NM, 2))
            for j, sp in enumerate(specs):
                ct, cq = rotor_table(sp, fl.rho, fl.mu)
                tab[j, ..., 0] = ct
                tab[j, ..., 1] = cq
            tabs.append(_wp.array(tab.ravel().astype(npdt), dtype=dtype))
        self._tab_air, self._tab_w = tabs

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

        self._sc = sc
        self._axis = _np.array([x for s in self.sets for x in s.axis_local], float).reshape(-1, 3)
        self._axis_dev = _wp.array(_np.ascontiguousarray(self._axis, dtype=npdt),
                                   dtype=make_kernels(dtype)["V"])
        self._inertia = sc([x for s in self.sets for x in s.inertia])
        self._handed = sc([float(sp.handed) for sp in specs])
        self._R = sc([float(sp.radius) for sp in specs])
        self._R4 = sc([float(sp.radius) ** 4 for sp in specs])
        self._rows = _wp.array(_np.arange(self.n, dtype=_np.int32), dtype=int)
        self._grid_j = sc(TABLE_J)
        self._grid_mu = sc(TABLE_MU)
        self._grid_tip = sc(_np.log(TABLE_TIP_SPEEDS))

    def loads(self, xmat, omega, cvel, xipos, com, dt, wind, current):
        """One step of every rotor; numpy in, numpy out.

        Per rotor row, in the sets' order: ``xmat (n, 9)`` the rotor body's
        orientation, ``omega (n,)`` its joint speed, ``cvel (n, 6)`` the body's
        ``cvel``, ``xipos (n, 3)``, ``com (n, 3)`` its root's ``subtree_com``,
        ``dt (n,)`` its model's timestep.  ``wind`` and ``current`` are the
        calm sea's.  Returns ``f_lin, f_ang`` (world force and torque to add to
        the body's ``xfrc_applied``), ``dq`` (to add to ``dof_damping``),
        ``damp`` (to add to ``qfrc_applied``), ``thrust``, ``torque``, ``subf``.
        """
        dtype = self._dtype
        K = make_kernels(dtype)
        Km = _fluid_wp.make_kernels(dtype)
        V, M = K["V"], K["M"]
        npdt = _np_dtype(dtype)
        n = self.n

        def vec(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=V)

        sc = self._sc
        cvel = _np.asarray(cvel, float)
        pos = vec(xipos)
        rho = _wp.zeros(n, dtype=dtype)
        mu = _wp.zeros(n, dtype=dtype)
        subf = _wp.zeros(n, dtype=dtype)
        flow = _wp.zeros(n, dtype=V)
        _wp.launch(Km["medium"], dim=n, inputs=[
            pos, sc(_np.full(n, 0.01)), V(*[dtype(x) for x in wind]),
            V(*[dtype(x) for x in current]), dtype(self.air.rho), dtype(self.water.rho),
            dtype(self.air.mu), dtype(self.water.mu), rho, mu, subf, flow])
        out = self.outputs()
        self.launch(_wp.array(_np.ascontiguousarray(xmat, dtype=npdt).reshape(n, 3, 3), dtype=M),
                    sc(omega), vec(cvel[:, 0:3]), vec(cvel[:, 3:6]), pos, vec(com), subf, flow,
                    sc(dt), out)
        res = {k: v.numpy() for k, v in out.items()}
        res["subf"] = subf.numpy()
        return res

    def outputs(self) -> dict:
        """Device arrays for :meth:`launch` to write, one row per rotor."""
        V = make_kernels(self._dtype)["V"]
        out = {k: _wp.zeros(max(self.n, 1), dtype=V) for k in ("f_lin", "f_ang")}
        out.update({k: _wp.zeros(max(self.n, 1), dtype=self._dtype)
                    for k in ("dq", "damp", "thrust", "torque")})
        return out

    def launch(self, xmat, omega, ang, lin, xipos, com, subf, flow, dt, out) -> None:
        """One step on device arrays, no host transfer.  Per rotor row: ``xmat``
        (3x3), ``omega``, ``ang`` and ``lin`` (the body's ``cvel`` halves),
        ``xipos``, ``com`` (its root's ``subtree_com``), ``subf`` and ``flow``
        (the medium at ``xipos``, half-height 0.01), ``dt``.  Writes ``out``
        (:meth:`outputs`)."""
        K = make_kernels(self._dtype)
        dtype, n = self._dtype, self.n
        if n == 0:
            return
        args = [
            xmat, self._axis_dev, omega, ang, lin, xipos, com,
            subf, flow, self._inertia, self._handed, self._R, self._R4, dt, self._rows,
            self._tab_air, self._tab_w, dtype(self.air.rho), dtype(self.water.rho),
            self._grid_j, self._grid_mu, self._grid_tip,
            out["f_lin"], out["f_ang"], out["dq"], out["damp"], out["thrust"], out["torque"]]
        _wp.launch(K["rotor"], dim=n, inputs=args)


class DevicePower:
    """``BatchedPower`` with its state on the device: one row per world's machine.

    Built from the same ``envs`` as ``BatchedPower``; copies each budget's
    actuator constants and its battery and ledger state at construction.
    ``step`` charges one step from the actuator forces and velocities and
    clears ``active`` for a machine whose pack went flat; ``state()`` reads the
    ledger back.
    """

    FIELDS = ("energy_j", "total_j", "actuator_j", "avionics_j", "t", "max_overload",
              "peak_draw", "overdraw_time")

    def __init__(self, envs, *, dtype=_wp.float64) -> None:
        from dytiscidae.physics.energy import BatchedPower

        npdt = _np_dtype(dtype)
        self._dtype = dtype
        host = BatchedPower(envs)
        self.n = list(host.n)
        self.off_np = _np.asarray(host.off, dtype=_np.int32)
        self._na = int(self.off_np[-1])

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

        self._sc = sc
        self._off = _wp.array(self.off_np, dtype=int)
        self._const = [sc(getattr(host, k)) for k in
                       ("seal", "ratio", "gear", "km", "k_iron", "k_visc", "rating")]
        bud = [e.budget for e in envs]
        self._avionics = sc([b.avionics_w for b in bud])
        self._p_max = sc([b.battery.p_max for b in bud])
        self._dt = sc([e.timestep for e in envs])
        ledger = lambda xs: _wp.array(_np.asarray(xs, dtype=_np.float64), dtype=_wp.float64)
        self._state = {
            "energy_j": ledger([b.battery.energy_j for b in bud]),
            "total_j": ledger([b.total_j for b in bud]),
            "actuator_j": ledger([b.actuator_j for b in bud]),
            "avionics_j": ledger([b.avionics_j for b in bud]),
            "t": ledger([b.t for b in bud]),
            "max_overload": sc([b.max_overload for b in bud]),
            "peak_draw": sc([b.battery.peak_draw for b in bud]),
            "overdraw_time": sc([b.battery.overdraw_time for b in bud]),
        }
        self.active = _wp.array(_np.ones(len(envs), dtype=_np.int32), dtype=int)

    def step(self, force, velocity) -> None:
        """``force``, ``velocity``: every machine's first ``n`` actuator forces and
        velocities, concatenated in world order (``BatchedPower``'s layout)."""
        self.launch(self._sc(force), self._sc(velocity))

    def launch(self, force, velocity) -> None:
        """:meth:`step` on device arrays, no host transfer."""
        K = make_kernels(self._dtype)
        if not hasattr(self, "_terms"):
            na = max(self._na, 1)
            self._terms = _wp.zeros(na, dtype=self._dtype)
            self._over = _wp.zeros(na, dtype=self._dtype)
        terms, over = self._terms, self._over
        if self._na:
            _wp.launch(K["power_terms"], dim=self._na, inputs=[
                force, velocity, *self._const, terms, over])
        s = self._state
        _wp.launch(K["power_charge"], dim=len(self.n), inputs=[
            self._off, terms, over, self._avionics, self._p_max, self._dt, self.active,
            s["energy_j"], s["total_j"], s["actuator_j"], s["avionics_j"], s["t"],
            s["max_overload"], s["peak_draw"], s["overdraw_time"]])

    def state(self) -> dict:
        out = {k: v.numpy().astype(float) for k, v in self._state.items()}
        out["active"] = self.active.numpy().astype(bool)
        return out


class DeviceJets:
    """``JetSet.apply`` for several machines' bells on the device.

    ``sets`` are ``(JetSet, model)`` pairs in world order.  Per bell row it
    returns the world force to add to the bell body's ``xfrc_applied[:3]``
    (``f``; jets add no torque) and the pumping load ``c_pump`` to add to
    ``dof_damping`` at the driving dof (``dofadr``, -1 for an undriven bell);
    per world, ``JetSet.apply``'s return value (``total``, sum of |thrust|) and
    ``last_pump_power`` (``pump``).  The add-on-top-and-take-back bookkeeping of
    ``JetSet._unapply`` is host array ownership and stays out: on the device
    ``dof_damping`` is rebuilt every step from base, damping split, rotors and
    jets, so nothing accumulates to take back.
    """

    def __init__(self, sets, air, water, *, dtype=_wp.float64) -> None:
        npdt = _np_dtype(dtype)
        self._dtype = dtype
        self.sets = [s for s, _ in sets]
        self.off_np = _np.asarray(_np.cumsum([0] + [s.n for s in self.sets]), dtype=_np.int32)
        self.n = int(self.off_np[-1])
        self.air, self.water = air, water

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

        self._sc = sc
        cat = lambda f: _np.concatenate([_np.asarray(f(s), float) for s in self.sets]) \
            if self.n else _np.zeros(0)
        self.dofadr = _np.concatenate(
            [[int(m.jnt_dofadr[j]) if j >= 0 else -1 for j in s.joint_id] for s, m in sets]
        ).astype(int) if self.n else _np.zeros(0, int)
        self.body_id = _np.concatenate([s.body_id for s in self.sets]).astype(int) \
            if self.n else _np.zeros(0, int)
        rng = _np.concatenate([s.joint_range for s in self.sets]).reshape(-1, 2) \
            if self.n else _np.zeros((0, 2))
        self._lo, self._hi = sc(rng[:, 0]), sc(rng[:, 1])
        vol = cat(lambda s: s.volume)
        self._volume = sc(vol)
        self._stroke = sc(cat(lambda s: s.stroke_fraction))
        self._orifice = sc(cat(lambda s: s.orifice_area))
        self._refill = sc(cat(lambda s: _np.full(s.n, s.refill_efficiency)))
        self._axis = cat(lambda s: s.axis_local).reshape(-1, 3)
        self._half = sc(_np.cbrt(_np.maximum(vol, 1e-9)) * 0.5)
        self._driven = _wp.array((self.dofadr >= 0).astype(_np.int32), dtype=int)
        self._off = _wp.array(self.off_np, dtype=int)
        self._axis_dev = _wp.array(_np.ascontiguousarray(self._axis.reshape(-1, 3), dtype=npdt),
                                   dtype=make_kernels(dtype)["V"]) if self.n else None

    def host_inputs(self, datas) -> dict:
        """``omega``, ``xmat``, ``xpos`` per bell row, read from each world's
        ``MjData`` as ``JetSet.apply`` reads them (0 for an undriven bell).  The
        joint angle is not an input: ``JetSet.apply`` reads it into a volume
        that nothing downstream uses, and dV/dt comes from the rate alone."""
        om, xm, xp = [], [], []
        for w, d in enumerate(datas):
            a, b = self.off_np[w], self.off_np[w + 1]
            va, bid = self.dofadr[a:b], self.body_id[a:b]
            om.append(_np.where(va >= 0, d.qvel[_np.maximum(va, 0)], 0.0))
            xm.append(d.xmat.reshape(-1, 9)[bid])
            xp.append(d.xpos[bid])
        return {"omega": _np.concatenate(om),
                "xmat": _np.concatenate(xm), "xpos": _np.concatenate(xp)}

    def loads(self, omega, xmat, xpos, wind, current) -> dict:
        """One step of every bell; numpy in, numpy out (see the class docstring)."""
        dtype = self._dtype
        K = make_kernels(dtype)
        Km = _fluid_wp.make_kernels(dtype)
        V, M = K["V"], K["M"]
        npdt = _np_dtype(dtype)
        n, sc = self.n, self._sc
        out = self.outputs()
        if n:
            def vec(a):
                return _wp.array(_np.ascontiguousarray(a, dtype=npdt).reshape(-1, 3), dtype=V)

            pos = vec(xpos)
            rho = _wp.zeros(n, dtype=dtype)
            mu = _wp.zeros(n, dtype=dtype)
            subf = _wp.zeros(n, dtype=dtype)
            flow = _wp.zeros(n, dtype=V)
            _wp.launch(Km["medium"], dim=n, inputs=[
                pos, self._half, V(*[dtype(x) for x in wind]), V(*[dtype(x) for x in current]),
                dtype(self.air.rho), dtype(self.water.rho), dtype(self.air.mu),
                dtype(self.water.mu), rho, mu, subf, flow])
            self.launch(sc(omega),
                        _wp.array(_np.ascontiguousarray(xmat, dtype=npdt).reshape(n, 3, 3),
                                  dtype=M), rho, subf, out)
        res = {k: v.numpy()[:n] if k in ("f", "thrust", "c_pump") else v.numpy()
               for k, v in out.items()}
        return res

    def outputs(self) -> dict:
        """Device arrays for :meth:`launch` to write: per bell ``f``, ``thrust``,
        ``c_pump``; per world ``total`` and ``pump``."""
        V = make_kernels(self._dtype)["V"]
        n, nw, dtype = self.n, len(self.sets), self._dtype
        return {"f": _wp.zeros(max(n, 1), dtype=V), "thrust": _wp.zeros(max(n, 1), dtype=dtype),
                "c_pump": _wp.zeros(max(n, 1), dtype=dtype),
                "total": _wp.zeros(nw, dtype=dtype), "pump": _wp.zeros(nw, dtype=dtype)}

    def launch(self, omega, xmat, rho, subf, out) -> None:
        """One step on device arrays, no host transfer: per bell row ``omega``,
        ``xmat`` (3x3), and ``rho``, ``subf`` (the medium at the bell body's
        ``xpos``, half-height ``self._half``).  Writes ``out`` (:meth:`outputs`)."""
        K = make_kernels(self._dtype)
        n = self.n
        if n:
            _wp.launch(K["jets"], dim=n, inputs=[
                omega, self._lo, self._hi, self._volume, self._stroke, self._orifice,
                self._refill, xmat, self._axis_dev, rho, subf,
                out["f"], out["thrust"], out["c_pump"]])
            _wp.launch(K["jet_sums"], dim=len(self.sets), inputs=[
                self._off, out["thrust"], out["c_pump"], omega, self._driven,
                out["total"], out["pump"]])


class DeviceInflow:
    """``InducedFlow`` for one machine per world, its state ``w`` on the device.

    ``spans`` are the machines' disc spans (``InducedFlow.span``).  The cadence
    stays on the host, as ``InducedFlow`` keeps it: ``due()`` is the per-world
    mask of steps that update, ``tick()`` advances it.  ``update`` takes per
    world the wing force (minus buoyancy), the machine's flow, the mean wing
    density and the interval, and updates only the worlds marked due.  ``w`` is
    float64 at every precision: it is a lagged state integrated over the whole
    rollout, and the solve that feeds it runs in the kernel's precision.
    """

    UPDATE_EVERY = _InducedFlow.UPDATE_EVERY

    def __init__(self, spans, *, dtype=_wp.float64) -> None:
        npdt = _np_dtype(dtype)
        self._dtype = dtype
        spans = _np.maximum(_np.asarray(spans, float), 0.0)
        self.nw = len(spans)

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

        self._sc = sc
        self._span = sc(spans)
        self._area = sc(_np.pi * spans**2 / 4.0)
        self._w = _wp.zeros(self.nw, dtype=_wp.vec3d)
        self._count = _np.zeros(self.nw, dtype=_np.int64)

    def due(self):
        return self._count % self.UPDATE_EVERY == 0

    def tick(self, mask=None) -> None:
        self._count += 1 if mask is None else _np.asarray(mask, _np.int64)

    def reset(self, mask=None) -> None:
        m = _np.ones(self.nw, bool) if mask is None else _np.asarray(mask, bool)
        w = self._w.numpy()
        w[m] = 0.0
        # In place: a captured graph holds this array's address.
        self._w.assign(w)
        self._count[m] = 0

    def update(self, F, Vf, rho, dt, due=None):
        """Update the worlds in ``due`` (default: ``self.due()``); returns ``w``."""
        K = make_kernels(self._dtype)
        npdt = _np_dtype(self._dtype)
        V = K["V"]

        def vec(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt).reshape(-1, 3), dtype=V)

        due = self.due() if due is None else _np.asarray(due, bool)
        self.launch(vec(F), vec(Vf), self._sc(rho), self._sc(_np.broadcast_to(dt, (self.nw,))),
                    _wp.array(due.astype(_np.int32), dtype=int))
        return self.w

    def launch(self, F, Vf, rho, dt, due) -> None:
        """:meth:`update` on device arrays (``due`` an int array per world), no
        host transfer and no host cadence: the caller keeps the cadence."""
        K = make_kernels(self._dtype)
        _wp.launch(K["inflow"], dim=self.nw, inputs=[
            F, Vf, rho, dt, self._span, self._area, due, self._w])

    @property
    def w(self):
        return self._w.numpy().astype(float)


def _ints(a):
    return _wp.array(_np.ascontiguousarray(a, dtype=_np.int32), dtype=int)


class DeviceDamping:
    """``ImplicitAeroDamping`` for several machines, its state on the device.

    ``dampers`` are the machines' ``ImplicitAeroDamping`` in world order (its
    ``anc``, ``base`` and strip half-heights are read from them, not rebuilt);
    ``body_ids`` each machine's strip-to-body map (``panels.body_id``).  One
    ``step`` is what the batched path does between two ``mj_step``s
    (batchroll.py:490-500): ``due()`` read before ``apply``, the strip damping
    ``b`` formed, B projected onto the dofs on due worlds only, the wet set and
    its staleness, the count, and the outputs ``dof_damping = base + B`` and
    ``qfrc_applied = B qvel`` (``clear`` on a world whose ``enabled`` is 0).
    Jets and rotors add to both after it, as on the host.

    The sea is the flat calm one of ``fluid_wp``: ``depth = -z``, so the wet
    test does not read the time.  ``last_b`` is recomputed, not accumulated;
    it and the moments it is projected from are float64 at every precision.
    """

    def __init__(self, dampers, body_ids, *, dtype=_wp.float64) -> None:
        npdt = _np_dtype(dtype)
        self._dtype = dtype
        self.dampers = list(dampers)
        nw = self.nw = len(self.dampers)
        models = [d.model for d in self.dampers]
        body_ids = [_np.asarray(b, int) for b in body_ids]
        self.poff = _np.cumsum([0] + [len(b) for b in body_ids]).astype(_np.int32)
        self.boff = _np.cumsum([0] + [m.nbody for m in models]).astype(_np.int32)
        self.voff = _np.cumsum([0] + [m.nv for m in models]).astype(_np.int32)
        self.aoff = _np.cumsum([0] + [m.nbody * m.nv for m in models]).astype(_np.int32)
        self.n, self.nb, self.nv = int(self.poff[-1]), int(self.boff[-1]), int(self.voff[-1])

        def sc(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

        self._sc = sc
        world = _np.concatenate([_np.full(len(b), w) for w, b in enumerate(body_ids)])
        sbody = _np.concatenate([self.boff[w] + b for w, b in enumerate(body_ids)])
        sroot = _np.concatenate([self.boff[w] + models[w].body_rootid[b]
                                 for w, b in enumerate(body_ids)])
        for d, b in zip(self.dampers, body_ids):
            if d._medium is None or d._half is None or len(d._half) != len(b):
                raise ValueError("DeviceDamping needs each damper's medium and one "
                                 "half-height per strip, as FluidSolver builds it")
        self._world = _ints(world)
        self._sbody, self._sroot = _ints(sbody), _ints(sroot)
        self._body_world = _ints(_np.repeat(_np.arange(nw), _np.diff(self.boff)))
        self._dof_world = _ints(_np.repeat(_np.arange(nw), _np.diff(self.voff)))
        self._poff, self._boff = _ints(self.poff), _ints(self.boff)
        self._voff, self._aoff = _ints(self.voff), _ints(self.aoff)
        self._anc = _ints(_np.concatenate([d.anc.astype(_np.int32).ravel()
                                           for d in self.dampers]))
        self._base = sc(_np.concatenate([d.base for d in self.dampers]))
        self._half = sc(_np.concatenate([d._half for d in self.dampers]))
        K = make_kernels(dtype)
        V, M = K["V"], K["M"]
        self._V = V
        self._B0 = _wp.zeros(max(self.nb, 1), dtype=_wp.float64)
        self._B1 = _wp.zeros(max(self.nb, 1), dtype=_wp.vec3d)
        self._B2 = _wp.zeros(max(self.nb, 1), dtype=_wp.mat33d)
        self._b = _wp.zeros(max(self.n, 1), dtype=dtype)
        self.reset()

    def reset(self) -> None:
        """``ImplicitAeroDamping.reset`` for every world, in place (a captured
        graph holds these arrays' addresses)."""
        if hasattr(self, "_count"):
            for a in (self._count, self._stale, self._have_ref, self._have_prev, self._due,
                      self._wet_ref, self._prev_pos, self._last_b):
                a.zero_()
            return
        nw = self.nw
        self._count = _wp.zeros(nw, dtype=int)
        self._stale = _wp.zeros(nw, dtype=int)
        self._have_ref = _wp.zeros(nw, dtype=int)
        self._have_prev = _wp.zeros(nw, dtype=int)
        self._due = _wp.zeros(nw, dtype=int)
        self._wet_ref = _wp.zeros(max(self.n, 1), dtype=int)
        self._prev_pos = _wp.zeros(max(self.n, 1), dtype=self._V)
        self._last_b = _wp.zeros(max(self.nv, 1), dtype=_wp.float64)

    def step(self, pos, q, area, lift, drag, rho, ar, is_wing, lift_scale,
             subtree_com, cdof, qvel, enabled=None) -> dict:
        """One step; numpy in, numpy out.

        Per strip (worlds back to back): ``pos`` world position, ``q``,
        ``area``, ``lift``, ``drag`` (``fluid_wp.strip_loads``' ``D``, which already
        carries the bluff term: the Mojo path's ``drag + d_bluff``), ``rho``,
        ``ar``, ``is_wing``.  Per world: ``lift_scale``,
        ``enabled`` (``implicit_damping``; default all on).  Per body:
        ``subtree_com``; per dof: ``cdof (nv, 6)``, ``qvel``.  Returns
        ``dof_damping``, ``qfrc_applied``, ``due`` (whether each world formed B
        this step) and ``last_b``.
        """
        dtype = self._dtype
        K = make_kernels(dtype)
        V = self._V
        npdt = _np_dtype(dtype)
        sc = self._sc

        def vec(a):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt).reshape(-1, 3), dtype=V)

        nw, nv = self.nw, self.nv
        en = _np.ones(nw, _np.int32) if enabled is None else _np.asarray(enabled, _np.int32)
        cdof = _np.asarray(cdof, float).reshape(-1, 6)
        dd = _wp.zeros(max(nv, 1), dtype=dtype)
        qf = _wp.zeros(max(nv, 1), dtype=dtype)
        self.launch(vec(pos), sc(q), sc(area), sc(lift), sc(drag), sc(rho), sc(ar),
                    _ints(_np.asarray(is_wing) != 0), sc(_np.broadcast_to(lift_scale, (nw,))),
                    vec(subtree_com), vec(cdof[:, :3]), vec(cdof[:, 3:]), sc(qvel), _ints(en),
                    dd, qf)
        # ``due`` is read back here, after the step, for the parity test's due
        # sequence: wet_stale reads it and nothing rewrites it until the next
        # step's damp_due.  :meth:`launch` itself reads nothing back.
        return {"dof_damping": dd.numpy()[:nv], "qfrc_applied": qf.numpy()[:nv],
                "due": self._due.numpy().astype(bool), "last_b": self._last_b.numpy()[:nv]}

    def launch(self, pos, q, area, lift, drag, rho, ar, is_wing, lift_scale, subtree_com,
               cdof_rot, cdof_lin, qvel, enabled, dof_damping, qfrc_applied) -> None:
        """:meth:`step` on device arrays, no host transfer: ``is_wing`` and
        ``enabled`` int arrays, ``lift_scale`` one per world, ``cdof_rot`` and
        ``cdof_lin`` the two halves of ``cdof``.  Writes ``dof_damping`` and
        ``qfrc_applied`` (one row per dof, worlds back to back)."""
        K = make_kernels(self._dtype)
        nw, n, nb, nv = self.nw, self.n, self.nb, self.nv
        _wp.launch(K["damp_due"], dim=nw, inputs=[self._count, self._stale, self._due])
        if n:
            _wp.launch(K["strip_b"], dim=n, inputs=[
                q, area, lift, drag, rho, ar, is_wing, lift_scale, self._world, self._b])
        if nb:
            _wp.launch(K["body_moments"], dim=nb, inputs=[
                self._poff, self._body_world, self._due, self._sbody, self._sroot, pos,
                subtree_com, self._b, self._B0, self._B1, self._B2])
        if nv:
            _wp.launch(K["dof_project"], dim=nv, inputs=[
                self._boff, self._voff, self._aoff, self._dof_world, self._due, self._anc,
                cdof_rot, cdof_lin, self._B0, self._B1, self._B2, self._last_b])
        _wp.launch(K["wet_stale"], dim=nw, inputs=[
            self._poff, self._due, pos, self._half, self._wet_ref, self._prev_pos,
            self._have_ref, self._have_prev, self._stale, self._count])
        if nv:
            _wp.launch(K["damp_out"], dim=nv, inputs=[
                self._dof_world, enabled, self._base, self._last_b, qvel,
                dof_damping, qfrc_applied])


def added_mass_into_model(dry_mass, dry_inertia, m_body, lever2, boff, *, dtype=_wp.float64):
    """batchroll.finish's added-mass write for many worlds; numpy in, numpy out.

    Per body, worlds back to back from ``boff``: ``dry_mass``, ``dry_inertia
    (nb, 3)``, ``m_body`` (the step's added mass) and ``lever2``.  Returns
    ``body_mass`` and ``body_inertia``, each world's own: ``model_wp.DeviceBody``
    writes them into the per-world (batched) ``Model`` fields.
    """
    K = make_kernels(dtype)
    V = K["V"]
    npdt = _np_dtype(dtype)

    def sc(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

    boff = _np.asarray(boff, _np.int32)
    nb = int(boff[-1])
    bm = _wp.zeros(nb, dtype=dtype)
    bi = _wp.zeros(nb, dtype=V)
    _wp.launch(K["added_mass"], dim=nb, inputs=[
        _ints(boff), _ints(_np.repeat(_np.arange(len(boff) - 1), _np.diff(boff))),
        sc(dry_mass), _wp.array(_np.ascontiguousarray(dry_inertia, dtype=npdt).reshape(-1, 3),
                                dtype=V),
        sc(m_body), sc(lever2), bm, bi])
    return bm.numpy(), bi.numpy()


def servo_command(target, rate, lead, world, has_rate, *, dtype=_wp.float64):
    """``TriphibianEnv.servo_command`` for many worlds; numpy in, numpy out.

    ``target``, ``rate`` and ``lead`` are per actuator row, ``world`` the row's
    world and ``has_rate`` per world: 1 where the CPG left a rate of the
    target's length (``pop_rate`` is host state until A3), else 0.
    """
    K = make_kernels(dtype)
    npdt = _np_dtype(dtype)

    def sc(a):
        return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dtype)

    n = len(target)
    ctrl = _wp.zeros(n, dtype=dtype)
    _wp.launch(K["servo"], dim=n, inputs=[
        sc(target), sc(rate), sc(lead), _wp.array(_np.asarray(world, dtype=_np.int32), dtype=int),
        _wp.array(_np.asarray(has_rate, dtype=_np.int32), dtype=int), ctrl])
    return ctrl.numpy()
