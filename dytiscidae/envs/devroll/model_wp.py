"""One body on the device: its MJWarp ``Model`` and ``Data`` over W worlds, and the
step that drives them (IMPL_1010 A2).

``DeviceBody(envs)`` takes W ``TriphibianEnv`` of the same phenotype, one per
world, and builds:

* ``m``: ``mjw.put_model`` of a copy of ``envs[0].model`` at its dry mass, inertia
  and damping, with ``body_mass``, ``body_inertia`` and ``dof_damping`` batched
  per world (``batch_sizes``), so each world carries its own added mass and its
  own damping split.  ``body_subtreemass`` is batched too and recomputed from
  ``body_mass`` after every write of it (``subtree_mass``): put_model fixes it
  from the dry mass, and a stale subtree mass put subtree_com far from the body
  and made MJWarp's float32 qM factorisation NaN in water (2026-10-26).  The host
  keeps its stale one, harmless in float64; ``subtree_com`` differs between the
  two paths by that, a change of reference point, not of the dynamics.  MJWarp reads the per-world ``body_mass`` for a welded body
  too (``experiments/mjwarp_probe/probe_jointless_mass.py``), so nothing stands
  in for ``mj_setConst``.  The geom margin is kept.  MJWarp 3.15 refuses a
  non-zero margin while MULTICCD or NATIVECCD is on (N9), so the device copy turns
  those two off instead.  That is a contact-model boundary against the host,
  which keeps both on; it is recorded, not hidden by zeroing the margin.
* ``d``: ``mjw.put_data`` with ``nworld = W``, ``njmax`` 512 and ``nconmax`` set.

World layout: every per-body, per-dof and per-panel array of the step is flat
with worlds back to back (``g = w * nbody + b``, ``k = w * nv + j``,
``i = w * npan + p``), the layout ``scatter_wp`` and ``fluid_wp`` take.  MJWarp's
own arrays are ``(W, n)``; the ``write`` kernels map one onto the other.

:meth:`DeviceBody.step` is ``batchroll.step_batch`` with ``HostFluid`` for these
W machines, in its order: the servo command into ``ctrl``; ``FluidSolver.apply``
(kinematics, medium, induced flow, strip loads on the Wagner-lagged incidence,
the inflow update and its per-world reduction, added mass, buoyancy and
assembly, the damping split, the limiter and weight cancellation, the
entrainment reaction); jets; rotors; the added mass into ``body_mass`` and
``body_inertia``, and ``body_subtreemass`` from it; the four diagnostics batchroll keeps (``clamped``,
``mean_submerged``, ``added_mass``, ``slam``); then ``mjw.step``; then the power
budget.  No kernel launch in it reads anything back to the host, and nothing in
it is host state baked into a launch: each world's step count since its reset
(``count``) is a device array, and the inflow cadence and whether the Wagner lag
and the slam difference are primed are read from it on the device, so one
captured graph serves the first step of a segment and every later one
(IMPL_1010 A4a).  ``reset(mask)`` resets the fluid state of the worlds in
``mask`` only; ``step(alive=...)`` holds a dead world's loads, mass and damping
at the dry, unloaded values so a world frozen after a failure stays finite.

Precision: the fluid and scatter run in ``dtype`` (float64 or float32); MJWarp
is float32 only, so its state is read in float32 and every field written into
it is rounded to float32.  The staged ``dtype`` copies of what is written
(``staged()``) are what a float64 parity test can hold at 1e-9.  The induced
flow, the power ledger and the chords travelled since reversal are float64 at
every precision.
"""

# No ``from __future__ import annotations``: Warp reads the kernel annotations
# (``V``, ``M``) as objects when it builds the kernels.
import copy as _copy

import numpy as _np

import mujoco as _mujoco
import mujoco_warp as _mjw
import warp as _wp

from dytiscidae.envs.devroll import fluid_wp as _fluid_wp
from dytiscidae.envs.devroll import scatter_wp as _scatter_wp
from dytiscidae.physics.fluid import GRAVITY, INFLOW_AR, WAGNER_A, WAGNER_B, InducedFlow

#: Constraints per world.  N9: 64 overflowed on landing (ROADMAP:3421).
NJMAX = 512
#: Contacts per world.  Air and water are expected to have none (T9).
NCONMAX = 128
#: The previous added mass a reset leaves: no growth can be measured against it,
#: so the first step after a reset reacts nothing, as ``_prev_mbody = None`` does.
_NO_PREVIOUS_MASS = 1.0e30

_KERNELS: dict = {}


def make_kernels(T):
    """The A2 gather, reduction, diagnostic and write kernels for precision ``T``."""
    if T in _KERNELS:
        return _KERNELS[T]
    V = _wp.vec3d if T == _wp.float64 else _wp.vec3
    M = _wp.mat33d if T == _wp.float64 else _wp.mat33
    f32 = _wp.float32

    @_wp.func
    def v3(x: _wp.vec3) -> V:
        return V(T(x[0]), T(x[1]), T(x[2]))

    @_wp.func
    def m3(x: _wp.mat33) -> M:
        return M(T(x[0, 0]), T(x[0, 1]), T(x[0, 2]), T(x[1, 0]), T(x[1, 1]), T(x[1, 2]),
                 T(x[2, 0]), T(x[2, 1]), T(x[2, 2]))

    @_wp.kernel
    def gather_bodies(
        nb: int, rootid: _wp.array(dtype=int), xpos_d: _wp.array2d(dtype=_wp.vec3),
        xmat_d: _wp.array2d(dtype=_wp.mat33), xipos_d: _wp.array2d(dtype=_wp.vec3),
        com_d: _wp.array2d(dtype=_wp.vec3), cvel_d: _wp.array2d(dtype=_wp.spatial_vector),
        xpos: _wp.array(dtype=V), xmat: _wp.array(dtype=M), xipos: _wp.array(dtype=V),
        com: _wp.array(dtype=V), ang: _wp.array(dtype=V), clin: _wp.array(dtype=V),
        lin_com: _wp.array(dtype=V),
    ):
        # The MJWarp state into the step's arrays, and the body velocity at its
        # centre of mass exactly as batchroll.launch shifts cvel (mjOBJ_BODY).
        g = _wp.tid()
        w = g // nb
        b = g - w * nb
        xpos[g] = v3(xpos_d[w, b])
        xmat[g] = m3(xmat_d[w, b])
        xi = v3(xipos_d[w, b])
        xipos[g] = xi
        com[g] = v3(com_d[w, b])
        c = cvel_d[w, b]
        a = V(T(c[0]), T(c[1]), T(c[2]))
        lv = V(T(c[3]), T(c[4]), T(c[5]))
        ang[g] = a
        clin[g] = lv
        off = xi - v3(com_d[w, rootid[b]])
        lin_com[g] = lv - _wp.cross(off, a)

    @_wp.kernel
    def gather_dofs(
        nv: int, cdof_d: _wp.array2d(dtype=_wp.spatial_vector), qvel_d: _wp.array2d(dtype=f32),
        rot: _wp.array(dtype=V), lin: _wp.array(dtype=V), qvel: _wp.array(dtype=T),
    ):
        k = _wp.tid()
        w = k // nv
        j = k - w * nv
        c = cdof_d[w, j]
        rot[k] = V(T(c[0]), T(c[1]), T(c[2]))
        lin[k] = V(T(c[3]), T(c[4]), T(c[5]))
        qvel[k] = T(qvel_d[w, j])

    @_wp.kernel
    def gather_rows(
        nb: int, nv: int, row_world: _wp.array(dtype=int), row_body: _wp.array(dtype=int),
        row_dof: _wp.array(dtype=int), row_root: _wp.array(dtype=int),
        xmat: _wp.array(dtype=M), xpos: _wp.array(dtype=V), xipos: _wp.array(dtype=V),
        com: _wp.array(dtype=V), ang: _wp.array(dtype=V), clin: _wp.array(dtype=V),
        qvel: _wp.array(dtype=T),
        r_xmat: _wp.array(dtype=M), r_xpos: _wp.array(dtype=V), r_xipos: _wp.array(dtype=V),
        r_com: _wp.array(dtype=V), r_ang: _wp.array(dtype=V), r_lin: _wp.array(dtype=V),
        r_omega: _wp.array(dtype=T),
    ):
        # A rotor or bell row's inputs: its body's frame and velocity, its root's
        # subtree_com and its joint speed (0 for an undriven bell, dof -1).
        j = _wp.tid()
        w = row_world[j]
        g = w * nb + row_body[j]
        r_xmat[j] = xmat[g]
        r_xpos[j] = xpos[g]
        r_xipos[j] = xipos[g]
        r_com[j] = com[w * nb + row_root[j]]
        r_ang[j] = ang[g]
        r_lin[j] = clin[g]
        r_omega[j] = T(0.0)
        if row_dof[j] >= 0:
            r_omega[j] = qvel[w * nv + row_dof[j]]

    @_wp.kernel
    def m_body_sum(body_id: _wp.array(dtype=_wp.int32), m_add: _wp.array(dtype=T),
                   m_body: _wp.array(dtype=T)):
        # np.add.at(m_body, body_id, m_add); body_id is global.
        i = _wp.tid()
        _wp.atomic_add(m_body, body_id[i], m_add[i])

    @_wp.kernel
    def pick_root(nb: int, xipos: _wp.array(dtype=V), out: _wp.array(dtype=V)):
        w = _wp.tid()
        out[w] = xipos[w * nb + 1]

    @_wp.kernel
    def inflow_reduce(
        npan: int, nb: int, is_wing: _wp.array(dtype=_wp.int32), F: _wp.array(dtype=V),
        rho: _wp.array(dtype=T), root_flow: _wp.array(dtype=V), lin_com: _wp.array(dtype=V),
        Fw: _wp.array(dtype=V), Vf: _wp.array(dtype=V), rho_w: _wp.array(dtype=T),
    ):
        # FluidSolver.apply's inputs to InducedFlow.update, per world: the lifting
        # strips' force (before buoyancy, which is the same as wing force minus
        # wing buoyancy, batchroll.py:506-507), their mean density, and
        # machine_flow (the medium at the root body minus mj_objectVelocity's
        # linear velocity of body 1).
        w = _wp.tid()
        s = V(T(0.0), T(0.0), T(0.0))
        r = T(0.0)
        nwing = int(0)
        for p in range(w * npan, (w + 1) * npan):
            if is_wing[p] != 0:
                s = s + F[p]
                r = r + rho[p]
                nwing = nwing + 1
        Fw[w] = s
        rho_w[w] = T(0.0)
        if nwing > 0:
            rho_w[w] = r / T(nwing)
        Vf[w] = root_flow[w] - lin_com[w * nb + 1]

    @_wp.kernel
    def diagnostics(
        npan: int, nb: int, limit: T, dt: T, subf: _wp.array(dtype=T),
        m_body: _wp.array(dtype=T), m_slam: _wp.array(dtype=T), vn: _wp.array(dtype=T),
        total: _wp.array(dtype=T), prev_ms: _wp.array(dtype=T), primed: _wp.array(dtype=int),
        clamped: _wp.array(dtype=int), mean_submerged: _wp.array(dtype=T),
        added_mass: _wp.array(dtype=T), slam: _wp.array(dtype=T),
    ):
        # batchroll.finish:532-545 per world: whether the limiter bound this step,
        # the mean submerged fraction (the policy's observation), the added mass,
        # and slam, the one-step difference of the entrained mass times the normal
        # velocity, zero on the first step after a reset.
        w = _wp.tid()
        clamped[w] = 0
        if total[w] > limit:
            clamped[w] = 1
        sf = T(0.0)
        worst = T(0.0)
        for p in range(w * npan, (w + 1) * npan):
            sf = sf + subf[p]
            if primed[w] != 0:
                worst = _wp.max(worst, _wp.abs((m_slam[p] - prev_ms[p]) / dt * vn[p]))
            prev_ms[p] = m_slam[p]
        mean_submerged[w] = T(0.0)
        if npan > 0:
            mean_submerged[w] = sf / T(npan)
        am = T(0.0)
        for b in range(w * nb, (w + 1) * nb):
            am = am + m_body[b]
        added_mass[w] = am
        slam[w] = worst
        primed[w] = 1

    @_wp.kernel
    def scatter_rows(
        nb: int, nv: int, row_world: _wp.array(dtype=int), row_body: _wp.array(dtype=int),
        row_dof: _wp.array(dtype=int), f_lin: _wp.array(dtype=V), f_ang: _wp.array(dtype=V),
        dq: _wp.array(dtype=T), damp: _wp.array(dtype=T), with_torque: int, with_qfrc: int,
        xf_f: _wp.array(dtype=V), xf_t: _wp.array(dtype=V), dd: _wp.array(dtype=T),
        qf: _wp.array(dtype=T),
    ):
        # A rotor's or bell's load onto its body and driving dof, added on top of
        # the fluid's, as RotorBatch.apply and JetSet.apply add theirs.
        j = _wp.tid()
        w = row_world[j]
        g = w * nb + row_body[j]
        _wp.atomic_add(xf_f, g, f_lin[j])
        if with_torque != 0:
            _wp.atomic_add(xf_t, g, f_ang[j])
        if row_dof[j] >= 0:
            k = w * nv + row_dof[j]
            _wp.atomic_add(dd, k, dq[j])
            if with_qfrc != 0:
                _wp.atomic_add(qf, k, damp[j])

    @_wp.kernel
    def write_bodies(
        nb: int, fb_f: _wp.array(dtype=V), fb_t: _wp.array(dtype=V), xf_f: _wp.array(dtype=V),
        xf_t: _wp.array(dtype=V), bm: _wp.array(dtype=T), bi: _wp.array(dtype=V),
        s_f: _wp.array(dtype=V), s_t: _wp.array(dtype=V),
        xfrc_applied: _wp.array2d(dtype=_wp.spatial_vector),
        body_mass: _wp.array2d(dtype=f32), body_inertia: _wp.array2d(dtype=_wp.vec3),
    ):
        # The step's body loads and inertia into world w's Data and Model rows.
        g = _wp.tid()
        w = g // nb
        b = g - w * nb
        f = fb_f[g] + xf_f[g]
        t = fb_t[g] + xf_t[g]
        s_f[g] = f
        s_t[g] = t
        xfrc_applied[w, b] = _wp.spatial_vector(f32(f[0]), f32(f[1]), f32(f[2]),
                                                f32(t[0]), f32(t[1]), f32(t[2]))
        body_mass[w, b] = f32(bm[g])
        i = bi[g]
        body_inertia[w, b] = _wp.vec3(f32(i[0]), f32(i[1]), f32(i[2]))

    @_wp.kernel
    def write_dofs(
        nv: int, dd: _wp.array(dtype=T), dd_x: _wp.array(dtype=T), qf: _wp.array(dtype=T),
        qf_x: _wp.array(dtype=T), s_dd: _wp.array(dtype=T), s_qf: _wp.array(dtype=T),
        dof_damping: _wp.array2d(dtype=f32), qfrc_applied: _wp.array2d(dtype=f32),
    ):
        k = _wp.tid()
        w = k // nv
        j = k - w * nv
        a = dd[k] + dd_x[k]
        q = qf[k] + qf_x[k]
        s_dd[k] = a
        s_qf[k] = q
        dof_damping[w, j] = f32(a)
        qfrc_applied[w, j] = f32(q)

    @_wp.kernel
    def write_ctrl(nact: int, ctrl_t: _wp.array(dtype=T), ctrl: _wp.array2d(dtype=f32)):
        i = _wp.tid()
        w = i // nact
        ctrl[w, i - w * nact] = f32(ctrl_t[i])

    @_wp.kernel
    def gather_power(
        row_world: _wp.array(dtype=int), row_act: _wp.array(dtype=int),
        force_d: _wp.array2d(dtype=f32), vel_d: _wp.array2d(dtype=f32),
        force: _wp.array(dtype=T), vel: _wp.array(dtype=T),
    ):
        # BatchedPower's layout: each machine's first n actuators, back to back.
        i = _wp.tid()
        force[i] = T(force_d[row_world[i], row_act[i]])
        vel[i] = T(vel_d[row_world[i], row_act[i]])

    @_wp.kernel
    def cadence(every: int, count: _wp.array(dtype=int), due: _wp.array(dtype=int),
                primed: _wp.array(dtype=int)):
        # InducedFlow's cadence (update on the reset step and every ``every``
        # after it) and UnsteadyState's first-call flag, per world, from the
        # world's own step count.
        w = _wp.tid()
        due[w] = 0
        if count[w] % every == 0:
            due[w] = 1
        primed[w] = 0
        if count[w] > 0:
            primed[w] = 1

    @_wp.kernel
    def count_tick(count: _wp.array(dtype=int)):
        w = _wp.tid()
        count[w] = count[w] + 1

    @_wp.kernel
    def hold_bodies(
        nb: int, alive: _wp.array(dtype=int), dry_mass: _wp.array(dtype=T),
        dry_inertia: _wp.array(dtype=V), xfrc_applied: _wp.array2d(dtype=_wp.spatial_vector),
        body_mass: _wp.array2d(dtype=f32), body_inertia: _wp.array2d(dtype=_wp.vec3),
    ):
        # A world that failed is not stepped on the host.  On the device it is
        # stepped unloaded at its dry mass, so a NaN in its fluid state cannot
        # reach MJWarp's solve (and the contact pool every world shares).
        g = _wp.tid()
        w = g // nb
        if alive[w] != 0:
            return
        b = g - w * nb
        unloaded = _wp.spatial_vector(f32(0.0), f32(0.0), f32(0.0), f32(0.0), f32(0.0),
                                      f32(0.0))
        xfrc_applied[w, b] = unloaded
        body_mass[w, b] = f32(dry_mass[g])
        i = dry_inertia[g]
        body_inertia[w, b] = _wp.vec3(f32(i[0]), f32(i[1]), f32(i[2]))

    @_wp.kernel
    def subtree_mass(nb: int, parent: _wp.array(dtype=int), body_mass: _wp.array2d(dtype=f32),
                     body_subtreemass: _wp.array2d(dtype=f32)):
        # Each world's subtree masses from the mass it carries now (dry plus
        # added, or dry when held).  MJWarp's put_model fixes body_subtreemass
        # from the dry mass, and subtree_com divides the mass-weighted sum of the
        # bodies' positions by it: with water's added mass it landed far from the
        # body, cinert reached ~2e6 and the float32 factorisation of qM went NaN on
        # the first step (ray in water, seed 21, 2026-10-26).  Bodies are in
        # topological order (parent < child), so one reverse pass adds every
        # child into its parent.  One thread per world, no atomics.
        w = _wp.tid()
        for b in range(nb):
            body_subtreemass[w, b] = body_mass[w, b]
        for i in range(nb - 1):
            b = nb - 1 - i
            pb = parent[b]
            body_subtreemass[w, pb] = body_subtreemass[w, pb] + body_subtreemass[w, b]

    @_wp.kernel
    def hold_dofs(
        nv: int, alive: _wp.array(dtype=int), base: _wp.array(dtype=T),
        dof_damping: _wp.array2d(dtype=f32), qfrc_applied: _wp.array2d(dtype=f32),
    ):
        k = _wp.tid()
        w = k // nv
        if alive[w] != 0:
            return
        j = k - w * nv
        dof_damping[w, j] = f32(base[k])
        qfrc_applied[w, j] = f32(0.0)

    @_wp.kernel
    def hold_ctrl(nact: int, alive: _wp.array(dtype=int), ctrl: _wp.array2d(dtype=f32)):
        i = _wp.tid()
        w = i // nact
        if alive[w] != 0:
            return
        ctrl[w, i - w * nact] = f32(0.0)

    _KERNELS[T] = {"cadence": cadence, "count_tick": count_tick, "hold_bodies": hold_bodies,
                   "subtree_mass": subtree_mass,
                   "hold_dofs": hold_dofs, "hold_ctrl": hold_ctrl,
                   "gather_bodies": gather_bodies, "gather_dofs": gather_dofs,
                   "gather_rows": gather_rows, "m_body": m_body_sum, "pick_root": pick_root,
                   "inflow_reduce": inflow_reduce, "diagnostics": diagnostics,
                   "scatter_rows": scatter_rows, "write_bodies": write_bodies,
                   "write_dofs": write_dofs, "write_ctrl": write_ctrl,
                   "gather_power": gather_power, "V": V, "M": M}
    return _KERNELS[T]


def _masked_fill(arr, per: int, mask, value=0) -> None:
    """``arr``'s ``per`` rows of every world in ``mask`` set to ``value``, in place
    (a captured graph keeps the array's address).  Host-side: a reset runs between
    segments, outside any graph."""
    if per <= 0:
        return
    a = arr.numpy()
    m = _np.repeat(_np.asarray(mask, bool), per)
    a[:len(m)][m] = value
    arr.assign(a)


def device_model(mjm, nworld: int):
    """``mjw.put_model`` of a copy of ``mjm`` with MULTICCD and NATIVECCD disabled
    (the geom margin is kept) and the four per-world fields batched: the three the
    step writes and ``body_subtreemass``, which :meth:`DeviceBody.subtree_mass`
    keeps equal to the subtree sums of ``body_mass``."""
    mjm = _copy.deepcopy(mjm)
    mjm.opt.disableflags |= (_mujoco.mjtDisableBit.mjDSBL_MULTICCD
                             | _mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    m = _mjw.put_model(mjm, batch_sizes={"body_mass": nworld, "body_inertia": nworld,
                                         "dof_damping": nworld, "body_subtreemass": nworld})
    return m, mjm


class DeviceBody:
    """W worlds of one body on the device; see the module docstring.

    ``envs`` are the worlds' machines (the same phenotype each).  Their solvers,
    rotor and jet sets, dampers and budgets supply the constants, and their
    budgets the starting battery state.  The machines themselves are not
    stepped: ``load_state`` copies their MuJoCo state into the worlds.
    """

    def __init__(self, envs, *, dtype=_wp.float64, njmax: int = NJMAX,
                 nconmax: int = NCONMAX) -> None:
        envs = list(envs)
        self.envs, self.W, self.dtype = envs, len(envs), dtype
        e0 = envs[0]
        mj0, sol0 = e0.model, e0.solver
        npdt = _np.float64 if dtype == _wp.float64 else _np.float32
        for e in envs:
            m_, s_ = e.model, e.solver
            if (m_.nbody, m_.nv, m_.nu, m_.nq, s_.panels.n) != (
                    mj0.nbody, mj0.nv, mj0.nu, mj0.nq, sol0.panels.n):
                raise ValueError("DeviceBody takes W machines of one body")
            if (s_.inflow, s_.unsteady, s_.entrainment) != (
                    sol0.inflow, sol0.unsteady, sol0.entrainment):
                raise ValueError("DeviceBody's worlds share inflow, unsteady and entrainment")
            if s_.quasi_static or not s_.wing_added_mass_tensor:
                raise NotImplementedError("the device fluid is the unsteady, tensor added-mass "
                                          "solver; quasi-static probes stay on the host")
            if s_.medium.sea_state.amplitude != 0.0:
                raise NotImplementedError("the device medium is the flat calm sea")
        if sol0.panels.n == 0:
            raise NotImplementedError("a body with no panels has no device fluid step")
        W, nb, nv, npan = self.W, mj0.nbody, mj0.nv, sol0.panels.n
        self.nb, self.nv, self.npan = nb, nv, npan
        self.nact = len(e0.act_names)
        self.timestep = float(mj0.opt.timestep)

        # --- the MJWarp model and data ------------------------------------
        mjm = _copy.deepcopy(mj0)
        mjm.body_mass[:] = sol0._dry_mass
        mjm.body_inertia[:] = sol0._dry_inertia
        mjm.dof_damping[:] = sol0._damping.base
        self.m, self.mjm = device_model(mjm, W)
        self.njmax, self.nconmax = njmax, nconmax
        self.d = _mjw.put_data(self.mjm, _mujoco.MjData(self.mjm), nworld=W, nconmax=nconmax,
                               njmax=njmax)

        K, Kf = make_kernels(dtype), _fluid_wp.make_kernels(dtype)
        V, M = K["V"], K["M"]
        self._V, self._M = V, M

        def sc(a, dt_=dtype):
            return _wp.array(_np.ascontiguousarray(a, dtype=npdt), dtype=dt_)

        def vec(a):
            return _wp.array(_np.ascontiguousarray(_np.asarray(a, float).reshape(-1, 3),
                                                   dtype=npdt), dtype=V)

        ints = _scatter_wp._ints
        tile = lambda a: _np.concatenate([_np.asarray(a)] * W)  # noqa: E731
        self._sc, self._vec = sc, vec

        # --- panels, worlds back to back ----------------------------------
        p = sol0.panels
        bid = _np.asarray(p.body_id, int)
        self._body_id = ints(_np.concatenate([bid + w * nb for w in range(W)]))
        self._machine = ints(_np.repeat(_np.arange(W), npan))
        is_wing = (_np.asarray(p.kind) == 0).astype(_np.int32)
        self._is_wing = ints(tile(is_wing))
        for k in ("pos_local", "span_local", "chord_local", "normal_local", "ext_local"):
            setattr(self, "_" + k, vec(tile(getattr(p, k))))
        for k in ("chord", "camber", "dr", "area", "volume", "volume_buoyant", "half_height",
                  "cd_bluff"):
            setattr(self, "_" + k, sc(tile(_np.asarray(getattr(p, k), float))))
        self.live_inflow = bool(sol0.inflow)
        if self.live_inflow:
            ar = _np.full(npan, INFLOW_AR)
        elif sol0._machine_ar is not None:
            ar = _np.full(npan, sol0._machine_ar)
        else:
            ar = _np.asarray(p.aspect_ratio, float)
        self._ar = sc(tile(ar))
        self._c_rot = sc(_np.concatenate([_np.broadcast_to(e.solver.c_rot, (npan,))
                                          for e in envs]))
        self.cd_scale, self.lift_scale = float(sol0.cd_scale), float(sol0.lift_scale)
        self.added_mass_scale = float(sol0.added_mass_scale)
        for e in envs:
            if (e.solver.cd_scale, e.solver.lift_scale, e.solver.added_mass_scale) != (
                    self.cd_scale, self.lift_scale, self.added_mass_scale):
                raise ValueError("DeviceBody's worlds share the auditor's coefficient scales")
        self.limit = 60.0 * (float(sol0._dry_mass.sum()) * GRAVITY + 1.0)
        med = sol0.medium
        self.medium = med
        self._wind = V(*[dtype(x) for x in med.wind])
        self._current = V(*[dtype(x) for x in med.current])
        self._fl = (dtype(med.air.rho), dtype(med.water.rho), dtype(med.air.mu),
                    dtype(med.water.mu))
        self._buoy = dtype(med.water.rho * GRAVITY)

        # --- bodies and dofs ----------------------------------------------
        self._rootid = ints(mj0.body_rootid)
        self._body_world = ints(_np.repeat(_np.arange(W), nb))
        self._dry_mass = sc(_np.concatenate([e.solver._dry_mass for e in envs]))
        self._dry_inertia = vec(_np.concatenate([e.solver._dry_inertia for e in envs]))
        self._lever2 = sc(_np.concatenate([e.solver._lever2 for e in envs]))
        self._boff = ints(_np.arange(W + 1) * nb)

        # --- the A1b pieces -----------------------------------------------
        self.damping = _scatter_wp.DeviceDamping([e.solver._damping for e in envs],
                                                 [p.body_id] * W, dtype=dtype)
        self._enabled = ints([int(bool(e.solver.implicit_damping)) for e in envs])
        self._lift_scale_w = sc(_np.full(W, self.lift_scale))
        self.inflow = _scatter_wp.DeviceInflow([e.solver._inflow.span for e in envs],
                                               dtype=dtype)
        self.rotors = (_scatter_wp.DeviceRotors([e.rotors for e in envs], med.air, med.water,
                                                dtype=dtype) if e0.rotors.n else None)
        self.jets = (_scatter_wp.DeviceJets([(e.jets, e.model) for e in envs], med.air,
                                            med.water, dtype=dtype) if e0.jets.n else None)
        self.power = _scatter_wp.DevicePower(envs, dtype=dtype)
        pw_world = _np.repeat(_np.arange(W), self.power.n)
        pw_act = _np.concatenate([_np.arange(k) for k in self.power.n]) if sum(self.power.n) \
            else _np.zeros(0, int)
        self._pw_world, self._pw_act = ints(pw_world), ints(pw_act)

        def rows(bodies, dofs, roots):
            world = _np.concatenate([_np.full(len(b), w) for w, b in enumerate(bodies)])
            return {"n": int(len(world)), "world": ints(world),
                    "body": ints(_np.concatenate(bodies)), "dof": ints(_np.concatenate(dofs)),
                    "root": ints(_np.concatenate(roots))}

        if self.rotors is not None:
            self._rot_rows = rows([e.rotors.body_arr for e in envs],
                                  [e.rotors.dof_arr for e in envs],
                                  [e.rotors.root_arr for e in envs])
            self._rot_dt = sc(_np.full(self._rot_rows["n"], self.timestep))
            self._rot_half = sc(_np.full(self._rot_rows["n"], 0.01))
        if self.jets is not None:
            jb = self.jets.body_id
            off = self.jets.off_np
            self._jet_rows = rows([jb[off[w]:off[w + 1]] for w in range(W)],
                                  [self.jets.dofadr[off[w]:off[w + 1]] for w in range(W)],
                                  [mj0.body_rootid[jb[off[w]:off[w + 1]]] for w in range(W)])

        # --- servo ----------------------------------------------------------
        na = W * self.nact
        self._act_world = ints(_np.repeat(_np.arange(W), self.nact))
        self._lead = sc(_np.concatenate([e.servo_lead[:self.nact] for e in envs])
                        if na else _np.zeros(0))
        self.targets = _wp.zeros(max(na, 1), dtype=dtype)
        self.rate = _wp.zeros(max(na, 1), dtype=dtype)
        self.has_rate = _wp.zeros(W, dtype=int)
        self._ctrl = _wp.zeros(max(na, 1), dtype=dtype)

        # --- step buffers ---------------------------------------------------
        z = lambda n, t=dtype: _wp.zeros(max(n, 1), dtype=t)  # noqa: E731
        P, B, D_ = W * npan, W * nb, W * nv
        self.state = {k: z(B, V) for k in ("xpos", "xipos", "com", "ang", "clin", "lin_com")}
        self.state["xmat"] = z(B, M)
        self.state.update({"cdof_rot": z(D_, V), "cdof_lin": z(D_, V), "qvel": z(D_)})
        self.buf = {k: z(P, V) for k in ("pos", "s", "c", "n", "omega", "v_elem", "flow", "rel",
                                          "F", "F_out")}
        self.buf.update({k: z(P) for k in (
            "rho", "mu", "subf", "D", "cl", "alpha", "U", "lev", "q", "L", "alpha_e", "lev_u",
            "m_add", "m_slam", "vn")})
        self.buf["rev"] = z(P, _wp.int32)
        self.buf.update({k: z(B, V) for k in ("fb_f", "fb_t", "xf_f", "xf_t", "bi", "s_f",
                                              "s_t")})
        self.buf.update({k: z(B) for k in ("m_body", "fsum_b", "bm", "prev_mbody")})
        self.buf.update({k: z(D_) for k in ("dd", "qf", "dd_x", "qf_x", "s_dd", "s_qf")})
        self.buf.update({"total": z(W), "root_pos": z(W, V), "root_flow": z(W, V),
                         "root_rho": z(W), "root_mu": z(W), "root_subf": z(W),
                         "Fw": z(W, V), "Vf": z(W, V), "rho_w": z(W),
                         "inflow_dt": sc(_np.full(W, self.timestep * InducedFlow.UPDATE_EVERY)),
                         "inflow_due": z(W, int)})
        # machine_flow's medium query: MediumField.flow_velocity blends with a fixed
        # 1e-2 whatever half-height it is given; only the flow output is read.
        self._root_half = sc(_np.full(W, 0.01))
        self.diag = {"clamped": z(W, int), "mean_submerged": z(W), "added_mass": z(W),
                     "slam": z(W)}
        self._prev_ms = z(P)
        self._slam_primed = z(W, int)
        # Each world's steps since its reset, and the Wagner lag's primed flag
        # derived from it: device state, so a graph needs no host counter.
        self.count = z(W, int)
        self._unsteady_primed = z(W, int)
        self._all_held = z(W, int)
        # Wagner lag and LEV travel (UnsteadyState), per panel; the chords
        # travelled are float64 at every precision.
        self._x0, self._x1 = z(P), z(P)
        self._s_trav = z(P, _wp.float64)
        self._rev_prev = z(P, _wp.int32)
        from dytiscidae.physics.fluid import LEV_TRAVEL
        self._travel = tuple(float(x) for x in LEV_TRAVEL)
        if self.rotors is not None:
            self._rot_out = self.rotors.outputs()
            self._rot_in = self._row_buffers(self._rot_rows["n"])
        if self.jets is not None:
            self._jet_out = self.jets.outputs()
            self._jet_in = self._row_buffers(self._jet_rows["n"])
        self._pw_force = z(int(sum(self.power.n)))
        self._pw_vel = z(int(sum(self.power.n)))
        self.reset()

    def _row_buffers(self, n):
        V, M, T = self._V, self._M, self.dtype
        b = {k: _wp.zeros(max(n, 1), dtype=V) for k in ("xpos", "xipos", "com", "ang", "lin",
                                                          "flow")}
        b["xmat"] = _wp.zeros(max(n, 1), dtype=M)
        b.update({k: _wp.zeros(max(n, 1), dtype=T) for k in ("omega", "rho", "mu", "subf")})
        return b

    # ------------------------------------------------------------------ state

    def reset(self, mask=None) -> None:
        """``FluidSolver.reset`` for every world, or for the worlds in ``mask``
        (one bool per world); the others keep their state.  The MuJoCo state is
        not touched."""
        if mask is None:
            self._x0.zero_()
            self._x1.zero_()
            self._s_trav.zero_()
            self._rev_prev.zero_()
            self.inflow.reset()
            self.damping.reset()
            self.buf["prev_mbody"].fill_(_NO_PREVIOUS_MASS)
            self._prev_ms.zero_()
            self._slam_primed.zero_()
            self.count.zero_()
            self._unsteady_primed.zero_()
            return
        mask = _np.asarray(mask, bool)
        if mask.shape != (self.W,):
            raise ValueError(f"reset takes one flag per world ({self.W})")
        for a in (self._x0, self._x1, self._s_trav, self._rev_prev, self._prev_ms):
            _masked_fill(a, self.npan, mask)
        _masked_fill(self.buf["prev_mbody"], self.nb, mask, _NO_PREVIOUS_MASS)
        for a in (self._slam_primed, self.count, self._unsteady_primed):
            _masked_fill(a, 1, mask)
        self.inflow.reset(mask)
        self.damping.reset(mask)

    def load_state(self, datas) -> None:
        """Each world's ``qpos``, ``qvel``, ``time`` and the solver's warm start
        from its ``MjData``, then ``mjw.forward``: the worlds start where the host
        machines are.  The warm start matters: left from a previous segment it
        moved the first step's qpos by up to 3e-4 m against a fresh one (eel,
        2026-10-26); the host's ``reset`` zeroes it (``mj_resetData``)."""
        f32 = _np.float32
        self.d.qpos.assign(_np.stack([d.qpos for d in datas]).astype(f32))
        self.d.qvel.assign(_np.stack([d.qvel for d in datas]).astype(f32))
        self.d.time.assign(_np.array([d.time for d in datas], dtype=f32))
        self.d.qacc_warmstart.assign(_np.stack([d.qacc_warmstart for d in datas]).astype(f32))
        # The dry mass, inertia and damping and no load, as ``FluidSolver.reset``
        # leaves the host model before its ``mj_forward``: the forward's
        # subtree_com (the fluid's and the twist's reference point) is
        # mass-weighted, and a previous segment's last added mass moved the first
        # step of the next by 2e-4 rad/s (eel, 2026-10-26).
        self.hold(self._all_held)
        _mjw.forward(self.m, self.d)

    def gather(self) -> None:
        """The step's kinematic inputs from MJWarp's state (float32) into ``state``."""
        K = make_kernels(self.dtype)
        st, d = self.state, self.d
        _wp.launch(K["gather_bodies"], dim=self.W * self.nb, inputs=[
            self.nb, self._rootid, d.xpos, d.xmat, d.xipos, d.subtree_com, d.cvel,
            st["xpos"], st["xmat"], st["xipos"], st["com"], st["ang"], st["clin"],
            st["lin_com"]])
        if self.nv:
            _wp.launch(K["gather_dofs"], dim=self.W * self.nv, inputs=[
                self.nv, d.cdof, d.qvel, st["cdof_rot"], st["cdof_lin"], st["qvel"]])

    def host_state(self, datas) -> None:
        """``state`` from host ``MjData`` in this precision, for the parity tests:
        the same quantities ``gather`` reads from MJWarp, at the host's precision."""
        npdt = _np.float64 if self.dtype == _wp.float64 else _np.float32
        cat = lambda f: _np.ascontiguousarray(  # noqa: E731
            _np.concatenate([f(d) for d in datas]), dtype=npdt)
        rid = self.mjm.body_rootid
        st = self.state
        st["xpos"].assign(cat(lambda d: d.xpos))
        st["xmat"].assign(cat(lambda d: d.xmat.reshape(-1, 3, 3)))
        st["xipos"].assign(cat(lambda d: d.xipos))
        st["com"].assign(cat(lambda d: d.subtree_com))
        st["ang"].assign(cat(lambda d: d.cvel[:, 0:3]))
        st["clin"].assign(cat(lambda d: d.cvel[:, 3:6]))

        def lin_com(d):
            off = d.xipos - d.subtree_com[rid]
            ang, lin = d.cvel[:, 0:3], d.cvel[:, 3:6]
            v = _np.empty_like(lin)
            v[:, 0] = lin[:, 0] - (off[:, 1] * ang[:, 2] - off[:, 2] * ang[:, 1])
            v[:, 1] = lin[:, 1] - (off[:, 2] * ang[:, 0] - off[:, 0] * ang[:, 2])
            v[:, 2] = lin[:, 2] - (off[:, 0] * ang[:, 1] - off[:, 1] * ang[:, 0])
            return v
        st["lin_com"].assign(cat(lin_com))
        if self.nv:
            st["cdof_rot"].assign(cat(lambda d: d.cdof[:, 0:3]))
            st["cdof_lin"].assign(cat(lambda d: d.cdof[:, 3:6]))
            st["qvel"].assign(cat(lambda d: d.qvel))

    # ------------------------------------------------------------------- step

    def step(self, *, gather: bool = True, alive=None) -> None:
        """One ``step_batch`` step for every world (see the module docstring).

        ``targets`` (and ``rate``, ``has_rate``) must already hold this step's
        servo targets.  ``gather`` False keeps the ``state`` the caller loaded
        (``host_state``) instead of reading MJWarp's.  ``alive`` (a device int
        array, one per world) holds the worlds marked 0 unloaded at their dry
        mass and damping, with no control, before ``mjw.step``."""
        if gather:
            self.gather()
        self.fluid_and_scatter()
        if alive is not None:
            self.hold(alive)
        _mjw.step(self.m, self.d)
        self._charge()
        _wp.launch(make_kernels(self.dtype)["count_tick"], dim=self.W, inputs=[self.count])

    def hold(self, alive) -> None:
        """The dead worlds' loads, mass, damping and control at their dry,
        unloaded values (see :meth:`step`)."""
        K = make_kernels(self.dtype)
        _wp.launch(K["hold_bodies"], dim=self.W * self.nb, inputs=[
            self.nb, alive, self._dry_mass, self._dry_inertia, self.d.xfrc_applied,
            self.m.body_mass, self.m.body_inertia])
        if self.nv:
            _wp.launch(K["hold_dofs"], dim=self.W * self.nv, inputs=[
                self.nv, alive, self.damping._base, self.m.dof_damping, self.d.qfrc_applied])
        if self.nact:
            _wp.launch(K["hold_ctrl"], dim=self.W * self.nact, inputs=[
                self.nact, alive, self.d.ctrl])
        self.subtree_mass()

    def subtree_mass(self) -> None:
        """``body_subtreemass`` from the ``body_mass`` just written, every world
        (see ``make_kernels``' ``subtree_mass``).  Run after ``write_bodies`` and
        after ``hold_bodies``, so ``load_state``'s forward and every step see it."""
        _wp.launch(make_kernels(self.dtype)["subtree_mass"], dim=self.W, inputs=[
            self.nb, self.m.body_parentid, self.m.body_mass, self.m.body_subtreemass])

    def fluid_and_scatter(self) -> None:
        """Everything before ``mjw.step``: servo, fluid, damping, jets, rotors,
        added mass, diagnostics, and the writes into ``m`` and ``d``."""
        T, K, Kf, Ks = self.dtype, make_kernels(self.dtype), _fluid_wp.make_kernels(
            self.dtype), _scatter_wp.make_kernels(self.dtype)
        W, nb, nv, npan = self.W, self.nb, self.nv, self.npan
        P, B, D_ = W * npan, W * nb, W * nv
        st, bf = self.state, self.buf
        h = T(self.timestep)

        # Each world's cadence from its own step count (device state).
        _wp.launch(K["cadence"], dim=W, inputs=[
            int(InducedFlow.UPDATE_EVERY), self.count, bf["inflow_due"],
            self._unsteady_primed])

        # Servo: ctrl for the targets (TriphibianEnv.servo_command).
        if self.nact:
            _wp.launch(Ks["servo"], dim=W * self.nact, inputs=[
                self.targets, self.rate, self._lead, self._act_world, self.has_rate,
                self._ctrl])
            _wp.launch(K["write_ctrl"], dim=W * self.nact, inputs=[
                self.nact, self._ctrl, self.d.ctrl])

        # FluidSolver.apply: kinematics, medium, relative flow.
        _wp.launch(Kf["kinematics"], dim=P, inputs=[
            self._body_id, st["xpos"], st["xmat"], st["xipos"], st["ang"], st["lin_com"],
            self._pos_local, self._span_local, self._chord_local, self._normal_local,
            bf["pos"], bf["s"], bf["c"], bf["n"], bf["omega"], bf["v_elem"]])
        _wp.launch(Kf["medium"], dim=P, inputs=[
            bf["pos"], self._half_height, self._wind, self._current, *self._fl,
            bf["rho"], bf["mu"], bf["subf"], bf["flow"]])
        _wp.launch(Kf["rel"], dim=P, inputs=[
            bf["flow"], bf["v_elem"], self._is_wing, self._machine, int(self.live_inflow),
            self.inflow._w, bf["rel"]])

        # Strip loads: the incidence pass, the Wagner lag and LEV, the lagged pass.
        def strip(lagged):
            _wp.launch(Kf["strip"], dim=P, inputs=[
                bf["rel"], bf["s"], bf["c"], bf["n"], bf["omega"], self._ext_local,
                self._chord, self._area, self._dr, self._cd_bluff, self._camber, self._ar,
                self._is_wing, bf["rho"], bf["mu"], T(self.lift_scale), T(self.cd_scale),
                self._c_rot, lagged, bf["alpha_e"], bf["lev_u"],
                bf["F"], bf["D"], bf["cl"], bf["alpha"], bf["U"], bf["lev"], bf["rev"],
                bf["q"], bf["L"]])
        strip(0)
        if self.envs[0].solver.unsteady:
            a1, a2 = (float(x) for x in WAGNER_A)
            b1, b2 = (float(x) for x in WAGNER_B)
            _wp.launch(Kf["unsteady"], dim=P, inputs=[
                bf["alpha"], bf["rev"], bf["U"], self._chord, bf["omega"], bf["s"],
                self._x0, self._x1, self._s_trav, self._rev_prev, self._machine,
                self._unsteady_primed, h, T(a1), T(a2), T(b1), T(b2), T(self._travel[0]), T(self._travel[1]),
                bf["alpha_e"], bf["lev_u"]])
            strip(1)

        # The induced flow, every UPDATE_EVERY steps of each world's own count
        # (``cadence`` above), on this step's lifting force.
        if self.live_inflow:
            _wp.launch(K["pick_root"], dim=W, inputs=[nb, st["xipos"], bf["root_pos"]])
            _wp.launch(Kf["medium"], dim=W, inputs=[
                bf["root_pos"], self._root_half, self._wind, self._current, *self._fl,
                bf["root_rho"], bf["root_mu"], bf["root_subf"], bf["root_flow"]])
            _wp.launch(K["inflow_reduce"], dim=W, inputs=[
                npan, nb, self._is_wing, bf["F"], bf["rho"], bf["root_flow"], st["lin_com"],
                bf["Fw"], bf["Vf"], bf["rho_w"]])
            self.inflow.launch(bf["Fw"], bf["Vf"], bf["rho_w"], bf["inflow_dt"],
                               bf["inflow_due"])

        # Added mass, buoyancy and assembly onto the bodies.
        _wp.launch(Kf["added_mass"], dim=P, inputs=[
            bf["rel"], bf["s"], bf["c"], bf["n"], self._ext_local, self._chord, self._dr,
            self._volume, bf["rho"], self._is_wing, T(self.added_mass_scale),
            bf["m_add"], bf["m_slam"], bf["vn"]])
        for k in ("m_body", "fb_f", "fb_t", "fsum_b", "total", "xf_f", "xf_t", "dd_x", "qf_x"):
            bf[k].zero_()
        _wp.launch(K["m_body"], dim=P, inputs=[self._body_id, bf["m_add"], bf["m_body"]])
        _wp.launch(Kf["assemble"], dim=P, inputs=[
            bf["pos"], bf["F"], self._body_id, st["xipos"], bf["subf"], self._volume_buoyant,
            self._buoy, self._machine, bf["F_out"], bf["fb_f"], bf["fb_t"], bf["fsum_b"],
            bf["total"]])

        # The damping split (before the limiter, as apply runs it).
        self.damping.launch(bf["pos"], bf["q"], self._area, bf["L"], bf["D"], bf["rho"],
                            self._ar, self._is_wing, self._lift_scale_w, st["com"],
                            st["cdof_rot"], st["cdof_lin"], st["qvel"], self._enabled,
                            bf["dd"], bf["qf"])
        # Limiter and weight cancellation, then the entrainment reaction.
        _wp.launch(Kf["finish"], dim=B, inputs=[
            bf["fb_f"], bf["fb_t"], bf["m_body"], bf["total"], self._body_world, T(self.limit),
            T(GRAVITY)])
        if self.envs[0].solver.entrainment:
            _wp.launch(Kf["entrain"], dim=B, inputs=[
                bf["fb_f"], bf["fb_t"], bf["m_body"], bf["prev_mbody"], self._lever2,
                st["ang"], st["lin_com"], h])
            _wp.copy(bf["prev_mbody"], bf["m_body"])

        # Jets, then rotors, on top of the fluid.
        if self.jets is not None:
            r, x = self._jet_rows, self._jet_in
            self._gather_rows(r, x)
            _wp.launch(Kf["medium"], dim=r["n"], inputs=[
                x["xpos"], self.jets._half, self._wind, self._current, *self._fl,
                x["rho"], x["mu"], x["subf"], x["flow"]])
            o = self._jet_out
            self.jets.launch(x["omega"], x["xmat"], x["rho"], x["subf"], o)
            _wp.launch(K["scatter_rows"], dim=r["n"], inputs=[
                nb, nv, r["world"], r["body"], r["dof"], o["f"], o["f"], o["c_pump"],
                o["c_pump"], 0, 0, bf["xf_f"], bf["xf_t"], bf["dd_x"], bf["qf_x"]])
        if self.rotors is not None:
            r, x = self._rot_rows, self._rot_in
            self._gather_rows(r, x)
            _wp.launch(Kf["medium"], dim=r["n"], inputs=[
                x["xipos"], self._rot_half, self._wind, self._current, *self._fl,
                x["rho"], x["mu"], x["subf"], x["flow"]])
            o = self._rot_out
            self.rotors.launch(x["xmat"], x["omega"], x["ang"], x["lin"], x["xipos"], x["com"],
                               x["subf"], x["flow"], self._rot_dt, o)
            _wp.launch(K["scatter_rows"], dim=r["n"], inputs=[
                nb, nv, r["world"], r["body"], r["dof"], o["f_lin"], o["f_ang"], o["dq"],
                o["damp"], 1, 1, bf["xf_f"], bf["xf_t"], bf["dd_x"], bf["qf_x"]])

        # The added mass into each world's mass and inertia, and the diagnostics.
        _wp.launch(Ks["added_mass"], dim=B, inputs=[
            self._boff, self._body_world, self._dry_mass, self._dry_inertia, bf["m_body"],
            self._lever2, bf["bm"], bf["bi"]])
        dg = self.diag
        _wp.launch(K["diagnostics"], dim=W, inputs=[
            npan, nb, T(self.limit), h, bf["subf"], bf["m_body"], bf["m_slam"], bf["vn"],
            bf["total"], self._prev_ms, self._slam_primed, dg["clamped"],
            dg["mean_submerged"], dg["added_mass"], dg["slam"]])

        # Into the MJWarp Model and Data, world by world.
        _wp.launch(K["write_bodies"], dim=B, inputs=[
            nb, bf["fb_f"], bf["fb_t"], bf["xf_f"], bf["xf_t"], bf["bm"], bf["bi"],
            bf["s_f"], bf["s_t"], self.d.xfrc_applied, self.m.body_mass, self.m.body_inertia])
        if nv:
            _wp.launch(K["write_dofs"], dim=D_, inputs=[
                nv, bf["dd"], bf["dd_x"], bf["qf"], bf["qf_x"], bf["s_dd"], bf["s_qf"],
                self.m.dof_damping, self.d.qfrc_applied])
        self.subtree_mass()

    def _gather_rows(self, r, x) -> None:
        K, st = make_kernels(self.dtype), self.state
        _wp.launch(K["gather_rows"], dim=r["n"], inputs=[
            self.nb, self.nv, r["world"], r["body"], r["dof"], r["root"], st["xmat"],
            st["xpos"], st["xipos"], st["com"], st["ang"], st["clin"], st["qvel"],
            x["xmat"], x["xpos"], x["xipos"], x["com"], x["ang"], x["lin"], x["omega"]])

    def _charge(self) -> None:
        """BatchedPower.step on the actuator forces mjw.step left."""
        n = int(sum(self.power.n))
        if n:
            K = make_kernels(self.dtype)
            _wp.launch(K["gather_power"], dim=n, inputs=[
                self._pw_world, self._pw_act, self.d.actuator_force, self.d.actuator_velocity,
                self._pw_force, self._pw_vel])
        self.power.launch(self._pw_force, self._pw_vel)

    # -------------------------------------------------------------- read back

    def set_targets(self, angles_list, rates=None) -> None:
        """Host targets (one array per world) into ``targets``: a test and
        bring-up convenience; A3 writes ``targets`` on the device."""
        if not self.nact:
            return
        npdt = _np.float64 if self.dtype == _wp.float64 else _np.float32
        self.targets.assign(_np.concatenate(
            [_np.asarray(a, float)[:self.nact] for a in angles_list]).astype(npdt))
        if rates is not None:
            has = [int(r is not None and len(r) == self.nact) for r in rates]
            self.rate.assign(_np.concatenate([
                _np.asarray(r, float) if h else _np.zeros(self.nact)
                for r, h in zip(rates, has)]).astype(npdt))
            self.has_rate.assign(_np.asarray(has, _np.int32))

    def staged(self) -> dict:
        """What the last step wrote, at the step's precision, per world:
        ``xfrc (W, nb, 6)``, ``qfrc_applied``, ``dof_damping`` ``(W, nv)``,
        ``body_mass (W, nb)``, ``body_inertia (W, nb, 3)``."""
        bf, W, nb, nv = self.buf, self.W, self.nb, self.nv
        xfrc = _np.concatenate([bf["s_f"].numpy()[:W * nb], bf["s_t"].numpy()[:W * nb]], axis=1)
        return {"xfrc": xfrc.reshape(W, nb, 6),
                "qfrc_applied": bf["s_qf"].numpy()[:W * nv].reshape(W, nv),
                "dof_damping": bf["s_dd"].numpy()[:W * nv].reshape(W, nv),
                "body_mass": bf["bm"].numpy()[:W * nb].reshape(W, nb),
                "body_inertia": bf["bi"].numpy()[:W * nb].reshape(W, nb, 3)}

    def resident(self) -> dict:
        """The same five fields read from the MJWarp ``Model`` and ``Data``."""
        return {"xfrc": self.d.xfrc_applied.numpy().astype(float),
                "qfrc_applied": self.d.qfrc_applied.numpy().astype(float),
                "dof_damping": self.m.dof_damping.numpy().astype(float),
                "body_mass": self.m.body_mass.numpy().astype(float),
                "body_inertia": self.m.body_inertia.numpy().astype(float)}

    def diagnostics(self) -> dict:
        out = {k: v.numpy()[:self.W].astype(float) for k, v in self.diag.items()}
        out["clamped"] = out["clamped"].astype(bool)
        return out
