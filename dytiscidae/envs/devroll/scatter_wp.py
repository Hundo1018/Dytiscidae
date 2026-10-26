"""Rotors, the power budget and the servo command on the device (IMPL_1010 A1b).

This slice of A1b: ``RotorBatch.apply`` (physics/rotor.py), ``BatchedPower.step``
(physics/energy.py) and ``TriphibianEnv.servo_command`` (envs/triphibian.py), as
Warp kernels over flat arrays.  A row is one rotor, one actuator or one world's
machine; the caller lays worlds out back to back and passes each row's index into
its world's arrays.  The rotor kernel returns per-rotor loads; writing them into
``xfrc_applied``, ``dof_damping`` and ``qfrc_applied`` is A2's job.

What stays on the host: the rotor tables (``rotor_table`` builds them once per
rotor and medium, they are constants on the device), and whether the CPG left a
rate for the servo (``CPG.pop_rate`` is host state until A3).  The medium is the
flat calm sea of ``fluid_wp``; waves are not ported.

Kernels are built per precision by :func:`make_kernels`, as in ``fluid_wp``.
"""

import numpy as _np

import warp as _wp

from dytiscidae.envs.devroll import fluid_wp as _fluid_wp
from dytiscidae.physics.rotor import TABLE_J, TABLE_MU, TABLE_TIP_SPEEDS, rotor_table

_KERNELS: dict = {}

_NA = len(TABLE_TIP_SPEEDS)
_NJ = len(TABLE_J)
_NM = len(TABLE_MU)
_TIP_LO = float(TABLE_TIP_SPEEDS[0])
_TIP_HI = float(TABLE_TIP_SPEEDS[-1])


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

    _KERNELS[T] = {"rotor": rotor_wp, "power_terms": power_terms,
                   "power_charge": power_charge, "servo": servo_wp, "V": V, "M": M}
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
        out = {k: _wp.zeros(n, dtype=V) for k in ("f_lin", "f_ang")}
        out.update({k: _wp.zeros(n, dtype=dtype) for k in ("dq", "damp", "thrust", "torque")})
        args = [
            _wp.array(_np.ascontiguousarray(xmat, dtype=npdt).reshape(n, 3, 3), dtype=M),
            vec(self._axis), sc(omega), vec(cvel[:, 0:3]), vec(cvel[:, 3:6]), pos, vec(com),
            subf, flow, self._inertia, self._handed, self._R, self._R4, sc(dt), self._rows,
            self._tab_air, self._tab_w, dtype(self.air.rho), dtype(self.water.rho),
            self._grid_j, self._grid_mu, self._grid_tip,
            out["f_lin"], out["f_ang"], out["dq"], out["damp"], out["thrust"], out["torque"]]
        _wp.launch(K["rotor"], dim=n, inputs=args)
        res = {k: v.numpy() for k, v in out.items()}
        res["subf"] = subf.numpy()
        return res


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
        K = make_kernels(self._dtype)
        na = max(self._na, 1)
        terms = _wp.zeros(na, dtype=self._dtype)
        over = _wp.zeros(na, dtype=self._dtype)
        if self._na:
            _wp.launch(K["power_terms"], dim=self._na, inputs=[
                self._sc(force), self._sc(velocity), *self._const, terms, over])
        s = self._state
        _wp.launch(K["power_charge"], dim=len(self.n), inputs=[
            self._off, terms, over, self._avionics, self._p_max, self._dt, self.active,
            s["energy_j"], s["total_j"], s["actuator_j"], s["avionics_j"], s["t"],
            s["max_overload"], s["peak_draw"], s["overdraw_time"]])

    def state(self) -> dict:
        out = {k: v.numpy().astype(float) for k, v in self._state.items()}
        out["active"] = self.active.numpy().astype(bool)
        return out


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
