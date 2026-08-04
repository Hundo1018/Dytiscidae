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

Not bit-reproducible
--------------------
This path gives slightly different numbers on every run, including with the same
seed. The added-mass and force scatters use `Atomic.fetch_add`, which sums in
whatever order the warps arrive, so the last bits move; over a few hundred steps
of a chaotic simulation that reaches around 1e-9 relative. Measured across five
identical runs: 6.0e-10, 6.7e-10, 6.9e-10, 1.2e-9, 2.4e-9.

The CPU path does not behave this way, so this is a real change and not merely a
tolerance. What it costs: a run can no longer be replayed exactly from its seed,
and a resumed run will not retrace the trajectory the original took. What it does
not cost: anything physical -- 1e-9 is nine orders below the quasi-steady model's
own accuracy, and the search is already stochastic in its sampling.

If exact replay is ever wanted back, the fix is to make the scatter
deterministic -- sort panels by body and use a segmented reduction instead of
atomics -- at some cost in speed. That has not been done because nothing has
asked for it yet.
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


def rollout_batch(envs, bf: BatchedFluid, duration: float, params_list,
                  domain, control_hz: float = 25.0, policies=None,
                  bases=None):
    """`TriphibianEnv.rollout` for a whole batch, one GPU call per timestep.

    Mirrors the single-machine version step for step, including which sample
    goes into which list, because `_score_segment` divides its time fractions by
    the length the segment was *asked* for rather than by the number of samples
    recorded -- that is the defence against the truncated-episode exploit and it
    only works if the recording matches.

    Returns one SegmentResult per environment, scored by that environment's own
    `_score_segment`, so the scoring is untouched by batching.
    """
    from .triphibian import SegmentResult

    k = len(envs)
    res = [SegmentResult(domain=domain, duration=duration) for _ in envs]
    n_steps = int(duration / envs[0].timestep)
    control_every = max(1, int(1.0 / (control_hz * envs[0].timestep)))

    starts = [e.root_pos().copy() for e in envs]
    rec = [dict(depths=[], alts=[], ups=[], contacts=[], clears=[], slam=0.0)
           for _ in envs]
    cur = list(params_list)
    active = np.ones(k, dtype=bool)

    for i in range(n_steps):
        angles = []
        for m, e in enumerate(envs):
            if active[m]:
                if (policies is not None and policies[m] is not None
                        and bases is not None and bases[m] is not None
                        and i % control_every == 0):
                    coeffs = policies[m].act(e.observation(domain))
                    cur[m] = bases[m].command_params(
                        params_list[m], coeffs, e.cpg.n)
                angles.append(e.cpg.command(cur[m], e.data.time))
            else:
                angles.append(None)

        was = active.copy()
        active = step_batch(envs, angles, bf, active)
        for m in range(k):
            if was[m] and not active[m]:
                res[m].failure = "battery exhausted"

        for m, e in enumerate(envs):
            if not active[m]:
                continue
            pos = e.root_pos()
            if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 400.0:
                res[m].survived = False
                res[m].failure = "diverged"
                active[m] = False
                continue
            r = rec[m]
            r["depths"].append(e.medium.depth(pos[None, :], e.data.time)[0])
            r["clears"].append(e.clearance())
            r["alts"].append(pos[2])
            R = e.data.xmat[e.root_body].reshape(3, 3)
            r["ups"].append(float(R[2, 2]))
            r["contacts"].append(1.0 if e._touching_ground() else 0.0)
            r["slam"] = max(r["slam"], e.solver.diag.slam)

        if not active.any():
            break

    for m, e in enumerate(envs):
        r = rec[m]
        end = e.root_pos().copy()
        res[m].distance = float(np.linalg.norm((end - starts[m])[:2]))
        res[m].mean_speed = res[m].distance / max(duration, 1e-6)
        res[m].mean_power = e.budget.mean_power
        res[m].peak_slam = r["slam"]
        res[m].max_actuator_overload = e.budget.max_overload
        res[m].attitude_rms = float(np.std(r["ups"])) if r["ups"] else 1.0
        res[m].ground_contact_fraction = (
            float(np.mean(r["contacts"])) if r["contacts"] else 0.0)
        res[m].max_depth = float(max(r["depths"])) if r["depths"] else 0.0
        res[m].competence = e._score_segment(
            domain, res[m], np.array(r["depths"]), np.array(r["alts"]),
            np.array(r["ups"]), np.array(r["contacts"]), np.array(r["clears"]))
    return res


def evaluate_tier1_batch(phenos, *, spec=None, controllers=None,
                         segment_seconds: float = 10.0,
                         identify_axes: bool = False, seed: int = 0,
                         sea_state=None, perturb: dict | None = None):
    """`evaluate_tier1` for a whole generation, sharing one GPU pipeline.

    The three domain segments are batched -- they are the bulk of the work and
    all three run the same kernels.  The transitions are still one machine at a
    time: `run_transition` has its own stepping loop and its own placement per
    crossing, and batching it is a separate piece of work.  That caps the gain
    here at roughly 1.6x rather than the 3.6x the segments alone reach, which
    is worth knowing before reading the numbers.

    Falls back to nothing: a phenotype that fails to compile is returned as a
    dead MissionResult in its slot, exactly as the unbatched version does, so
    the caller's indexing is never disturbed.
    """
    import time as _time

    from .evaluate import (
        Controller, evaluate_tier0, finalise_tier1, transition_energy)
    from .transitions import run_transition
    from .triphibian import DOMAIN_CYCLE, MissionResult, MissionSpec, TriphibianEnv

    spec = spec or MissionSpec()
    t0 = _time.time()
    k = len(phenos)
    results = [MissionResult(tier=1) for _ in range(k)]
    envs, live = [None] * k, []

    for i, p in enumerate(phenos):
        r = results[i]
        t0_r = evaluate_tier0(p, spec)
        r.structural_margin = t0_r.structural_margin
        r.feasible = t0_r.feasible
        if not p.segments:
            r.notes.append("empty phenotype")
            continue
        try:
            envs[i] = TriphibianEnv(p, seed=seed, sea_state=sea_state,
                                    perturb=perturb)
            live.append(i)
        except Exception as exc:
            r.notes.append(f"compile failed: {type(exc).__name__}: {exc}")

    if not live:
        for r in results:
            r.wall_time = _time.time() - t0
        return results

    ctrls = [None] * k
    clamped = [False] * k
    for i in live:
        ctrls[i] = (controllers[i] if controllers and controllers[i] is not None
                    else Controller(params=envs[i].cpg.base))

    # Mobility identification, if asked for.  Per machine and sequential: it
    # drives the CPG with random parameter perturbations and fits a Jacobian
    # from the resulting body twist, so every machine is running a *different*
    # experiment and there is no shared timestep to batch across.  Left on the
    # CPU rather than pretended away, and it is a real part of the cost --
    # identify_axes_every is 1 in every run config so far, meaning every
    # candidate pays it.
    if identify_axes:
        from .triphibian import Domain as _D
        for i in live:
            for dom in (_D.AIR, _D.WATER):
                try:
                    results[i].mobility[dom.value] = envs[i].identify(dom, seed=seed)
                except Exception as exc:
                    results[i].notes.append(
                        f"mobility id failed in {dom.value}: {exc}")
            if ctrls[i].bases is None:
                ctrls[i].bases = results[i].mobility

    group = [envs[i] for i in live]
    bf = BatchedFluid(group)

    for dom in DOMAIN_CYCLE:
        for i in live:
            envs[i].reset(dom)
        segs = rollout_batch(
            group, bf, segment_seconds, [ctrls[i].params for i in live], dom,
            policies=[ctrls[i].policy for i in live],
            bases=[ctrls[i].basis_for(dom) for i in live])
        for slot, i in enumerate(live):
            results[i].segments[dom.value] = segs[slot]
            clamped[i] = clamped[i] or bool(envs[i].solver.diag.clamped)

    # Transitions stay per-machine.  See the docstring.
    for i in live:
        for kind in ("air_to_water", "water_to_air", "water_to_land"):
            tr = run_transition(envs[i], kind, ctrls[i])
            results[i].transitions.results[kind] = tr
            results[i].transition_ok[kind] = tr.crossed
            if tr.failure:
                results[i].notes.append(f"{kind}: {tr.failure}")

    for i in live:
        r, p = results[i], phenos[i]
        cruise_j = sum(s.mean_power * spec.seconds_per_domain * spec.cycles
                       for s in r.segments.values())
        trans_j = sum(transition_energy(p.mass, kk) for kk in spec.transitions)
        r.energy_required_wh = (cruise_j + trans_j) / 3600.0
        r.energy_available_wh = p.genome.battery_wh * 0.85
        # Shared with evaluate_tier1 rather than reimplemented: two copies of a
        # scoring rule is how the copies stop agreeing.
        finalise_tier1(r, clamped[i])

    for r in results:
        r.wall_time = _time.time() - t0
    return results
