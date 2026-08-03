"""Step several candidates together so their panels share one GPU launch.

A candidate's Tier-1 evaluation is about ten thousand sequential steps, and at
each one the fluid solver does the same work for that one machine's fifty-odd
panels. That is far too little to be worth a kernel launch: measured, a single
machine on the GPU is barely faster than numpy. Across candidates it is worth a
great deal -- seven machines run 7.7x and a hundred and twelve run 18.5x
against `FluidSolver.apply`, which works out at 3.7x to 4.9x on `env.step`
after Amdahl.

So the axis being exploited here is *candidates*, not timesteps. The timesteps
remain strictly sequential, as they must.

Import is guarded: the GPU extension is optional and this module degrades to
telling the caller it is unavailable rather than breaking a CPU-only install.
"""
from __future__ import annotations

import numpy as np

from ..physics.medium import GRAVITY
from .triphibian import Domain

try:  # pragma: no cover - depends on a built extension
    import full_pipeline as _fp
    import mujoco as _mj
    AVAILABLE = True
    UNAVAILABLE_REASON = ""
except Exception as exc:  # pragma: no cover
    _fp = None
    AVAILABLE = False
    UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


class BatchedFluid:
    """One GPU pipeline serving N environments stepped in lockstep.

    Body indices are rebased as environments are appended, so a single
    `body_id` array addresses every body in the batch and the added-mass
    scatter lands in the right place without knowing which machine a panel came
    from. The force limiter is per-machine for the same reason it is in
    `scatter_gpu`: the limit is that machine's own dry weight, and a shared one
    would hand the heaviest machine's allowance to the lightest.
    """

    def __init__(self, envs):
        if not AVAILABLE:
            raise RuntimeError(
                f"GPU fluid extension not importable ({UNAVAILABLE_REASON}). "
                "Build it with `cd mojo && pixi run build-all`, and put "
                "mojo/build on PYTHONPATH.")
        self.envs = list(envs)
        C = np.ascontiguousarray
        P = [e.solver.panels for e in self.envs]
        self.npan = [len(p.body_id) for p in P]
        self.nbod = [e.model.nbody for e in self.envs]
        self.poff = np.cumsum([0] + self.npan)
        self.boff = np.cumsum([0] + self.nbod)
        n, nb, nm = int(self.poff[-1]), int(self.boff[-1]), len(self.envs)
        self.n, self.nb, self.nm = n, nb, nm

        self.body_id = C(np.concatenate(
            [np.asarray(p.body_id, np.int32) + self.boff[i]
             for i, p in enumerate(P)]).astype(np.int32))
        self.machine = C(np.concatenate(
            [np.full(self.npan[i], i, np.int32) for i in range(nm)]))
        # PanelSet stores kind with WING == 0; the kernels want a flag.
        self.is_wing = C(np.concatenate(
            [(np.asarray(p.kind) == 0).astype(np.int32) for p in P]))
        g = lambda nm_: C(np.concatenate(
            [np.asarray(getattr(p, nm_), float) for p in P]))
        self.pos_local, self.span_local = g("pos_local"), g("span_local")
        self.chord_local, self.normal_local = g("chord_local"), g("normal_local")
        self.ext, self.chord, self.camber = g("ext_local"), g("chord"), g("camber")
        self.dr, self.area, self.volume = g("dr"), g("area"), g("volume")
        self.vol_buoy, self.half_height = g("volume_buoyant"), g("half_height")
        self.cd_bluff, self.ar = g("cd_bluff"), g("aspect_ratio")
        self.c_rot = C(np.concatenate(
            [np.asarray(e.solver.c_rot, float) for e in self.envs]))
        self.limit = C(np.array([
            60.0 * (float(e.solver._dry_mass.sum()) * GRAVITY + 1.0)
            for e in self.envs]))
        self.dry_mass = [e.solver._dry_mass.copy() for e in self.envs]
        self.dry_inertia = [e.solver._dry_inertia.copy() for e in self.envs]
        self.lever2 = [e.solver._lever2.copy() for e in self.envs]

        self.pipe = _fp.FullPipeline(n, nb, nm)
        self.pipe.upload_static(np.array(
            [a.ctypes.data for a in (
                self.body_id, self.machine, self.is_wing, self.pos_local,
                self.span_local, self.chord_local, self.normal_local, self.ext,
                self.chord, self.camber, self.dr, self.area, self.volume,
                self.vol_buoy, self.half_height, self.cd_bluff, self.ar,
                self.c_rot, self.limit)] + [n, nm], dtype=np.int64))

        self.xpos = np.zeros((nb, 3)); self.xmat = np.zeros((nb, 9))
        self.xipos = np.zeros((nb, 3)); self.vel6 = np.zeros((nb, 6))
        self.out = {k: np.zeros(s) for k, s in (
            ("xfrc", (nb, 6)), ("m_body", nb), ("m_add", n), ("subf", n),
            ("alpha", n), ("q", n), ("lift", n), ("drag", n), ("buoy", n))}
        self.clamped = np.zeros(nm, dtype=np.int32)
        self._has_bluff = int((self.is_wing == 0).any())
        self._v6 = np.zeros(6)

    def apply(self, t: float, active=None) -> None:
        """Run the fluid for every environment and write their xfrc_applied.

        `active` is an optional boolean mask; a machine whose battery has gone
        flat stops being stepped but stays in the batch. Dropping it would mean
        re-packing the panel arrays and re-uploading the geometry, which costs
        more than letting a few dead machines ride along in a kernel that is
        launch-bound rather than compute-bound.
        """
        for i, e in enumerate(self.envs):
            if active is not None and not active[i]:
                continue
            a, b = self.boff[i], self.boff[i + 1]
            self.xpos[a:b] = e.data.xpos
            self.xmat[a:b] = e.data.xmat.reshape(-1, 9)
            self.xipos[a:b] = e.data.xipos
            for bi in range(e.model.nbody):
                _mj.mj_objectVelocity(e.model, e.data,
                                      _mj.mjtObj.mjOBJ_BODY, bi, self._v6, 0)
                self.vel6[a + bi] = self._v6

        o = self.out
        e0 = self.envs[0]
        med, sol, s = e0.solver.medium, e0.solver, e0.solver.medium.sea_state
        desc = np.array(
            [a.ctypes.data for a in (self.xpos, self.xmat, self.xipos, self.vel6)]
            + [o["xfrc"].ctypes.data, o["m_body"].ctypes.data,
               self.clamped.ctypes.data]
            + [o[k].ctypes.data for k in
               ("m_add", "subf", "alpha", "q", "lift", "drag", "buoy")]
            + [self.n, self.nb, self.nm, self._has_bluff], dtype=np.int64)
        self.pipe.step(desc, (
            s.amplitude, s.wavelength, s.period,
            float(np.cos(s.direction)), float(np.sin(s.direction)), t,
            med.air.rho, med.air.mu, med.water.rho, med.water.mu,
            *[float(x) for x in med.current], *[float(x) for x in med.wind],
            sol.cd_scale, sol.added_mass_scale, sol.lift_scale))

        # Scatter back into each machine, and fold the added mass into its mass
        # matrix.  That write stays here rather than in a kernel because these
        # are MuJoCo model arrays and mj_step reads them on the next line.
        for i, e in enumerate(self.envs):
            if active is not None and not active[i]:
                continue
            a, b = self.boff[i], self.boff[i + 1]
            e.data.xfrc_applied[:] = o["xfrc"][a:b]
            mb = o["m_body"][a:b]
            e.model.body_mass[:] = self.dry_mass[i] + mb
            e.model.body_inertia[:] = (
                self.dry_inertia[i] + (mb * self.lever2[i])[:, None])
            e.solver.diag.clamped = bool(self.clamped[i])


def step_batch(envs, angles_list, bf: BatchedFluid, active=None):
    """One timestep for the whole batch.  Returns the updated active mask.

    Mirrors `TriphibianEnv.step`: set the control, zero xfrc, apply the fluid,
    integrate, charge the battery.  The only difference is that the fluid is
    computed for every machine at once.
    """
    if active is None:
        active = np.ones(len(envs), dtype=bool)
    for i, e in enumerate(envs):
        if not active[i]:
            continue
        if len(e.act_names):
            e.data.ctrl[: len(angles_list[i])] = angles_list[i]
        e.data.xfrc_applied[:] = 0.0

    bf.apply(envs[0].data.time, active=active)

    for i, e in enumerate(envs):
        if not active[i]:
            continue
        e._mj.mj_step(e.model, e.data)
        alive = e.budget.step(np.abs(e.data.actuator_force),
                              np.abs(e.data.actuator_velocity), e.timestep)
        if not alive:
            active[i] = False
    return active
