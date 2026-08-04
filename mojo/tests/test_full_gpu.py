"""The whole of fluid.apply on the GPU, against FluidSolver itself.

The reference is the real solver running the real MuJoCo state, not a
re-implementation, so a shared misreading cannot pass. What is compared is what
mj_step actually consumes: data.xfrc_applied, and model.body_mass.
"""
import numpy as np
import mujoco
import full_pipeline

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv
from dytiscidae.physics.medium import GRAVITY


def pose(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(Domain.WATER, randomise=False)
    nq = env.model.nq
    if nq >= 7:
        q = rng.normal(size=4)
        env.data.qpos[3:7] = q / np.linalg.norm(q)
        env.data.qpos[0:3] = rng.normal(scale=1.5, size=3)
        env.data.qpos[2] = rng.uniform(-3.0, 1.0)      # straddle the surface
    if nq > 7:
        env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
    nv = env.model.nv
    env.data.qvel[:] = rng.normal(scale=2.0, size=nv)
    mujoco.mj_forward(env.model, env.data)
    return env


class Batch:
    def __init__(self, envs):
        self.envs = envs
        C = np.ascontiguousarray
        P = [e.solver.panels for e in envs]
        self.npan = [len(p.body_id) for p in P]
        self.nbod = [e.model.nbody for e in envs]
        self.poff = np.cumsum([0] + self.npan)
        self.boff = np.cumsum([0] + self.nbod)
        n, nb, nm = int(self.poff[-1]), int(self.boff[-1]), len(envs)
        self.n, self.nb, self.nm = n, nb, nm

        cat = lambda f: C(np.concatenate([f(p, i) for i, p in enumerate(P)]))
        self.body_id = C(np.concatenate(
            [np.asarray(p.body_id, np.int32) + self.boff[i]
             for i, p in enumerate(P)]).astype(np.int32))
        self.machine = C(np.concatenate(
            [np.full(self.npan[i], i, np.int32) for i in range(nm)]))
        self.is_wing = C(np.concatenate(
            [(np.asarray(p.kind) == 0).astype(np.int32) for p in P]))
        g = lambda name: cat(lambda p, i: np.asarray(getattr(p, name), float))
        self.pos_local = g("pos_local"); self.span_local = g("span_local")
        self.chord_local = g("chord_local"); self.normal_local = g("normal_local")
        self.ext = g("ext_local"); self.chord = g("chord"); self.camber = g("camber")
        self.dr = g("dr"); self.area = g("area"); self.volume = g("volume")
        self.vol_buoy = g("volume_buoyant"); self.half_height = g("half_height")
        self.cd_bluff = g("cd_bluff"); self.ar = g("aspect_ratio")
        self.c_rot = C(np.concatenate(
            [np.asarray(e.solver.c_rot, float) for e in envs]))
        self.limit = C(np.array([
            60.0 * (float(e.solver._dry_mass.sum()) * GRAVITY + 1.0)
            for e in envs]))

        self.body_start = C(np.searchsorted(
            self.body_id, np.arange(nb + 1), side="left").astype(np.int32))
        self.p = full_pipeline.FullPipeline(n, nb, nm)
        self.p.upload_static(np.array(
            [a.ctypes.data for a in (
                self.body_id, self.machine, self.is_wing, self.pos_local,
                self.span_local, self.chord_local, self.normal_local, self.ext,
                self.chord, self.camber, self.dr, self.area, self.volume,
                self.vol_buoy, self.half_height, self.cd_bluff, self.ar,
                self.c_rot, self.limit, self.body_start)]
            + [n, nm, nb], dtype=np.int64))

        self.xpos = np.zeros((nb, 3)); self.xmat = np.zeros((nb, 9))
        self.xipos = np.zeros((nb, 3)); self.vel6 = np.zeros((nb, 6))
        self.out = {k: np.zeros(s) for k, s in (
            ("xfrc", (nb, 6)), ("m_body", nb), ("m_add", n), ("subf", n),
            ("alpha", n), ("q", n), ("lift", n), ("drag", n), ("buoy", n),
            ("vn", n))}
        self.clamped = np.zeros(nm, dtype=np.int32)

    def step(self, t):
        v6 = np.zeros(6)
        for i, e in enumerate(self.envs):
            a, b = self.boff[i], self.boff[i + 1]
            self.xpos[a:b] = e.data.xpos
            self.xmat[a:b] = e.data.xmat.reshape(-1, 9)
            self.xipos[a:b] = e.data.xipos
            for bi in range(e.model.nbody):
                mujoco.mj_objectVelocity(e.model, e.data,
                                         mujoco.mjtObj.mjOBJ_BODY, bi, v6, 0)
                self.vel6[a + bi] = v6
        o = self.out
        med = self.envs[0].solver.medium
        s = med.sea_state
        sol = self.envs[0].solver
        desc = np.array(
            [a.ctypes.data for a in (self.xpos, self.xmat, self.xipos, self.vel6)]
            + [o[k].ctypes.data for k in ("xfrc", "m_body")]
            + [self.clamped.ctypes.data]
            + [o[k].ctypes.data for k in
               ("m_add", "subf", "alpha", "q", "lift", "drag", "buoy", "vn")]
            + [self.n, self.nb, self.nm, int((self.is_wing == 0).any())],
            dtype=np.int64)
        self.p.step(desc, (
            s.amplitude, s.wavelength, s.period,
            float(np.cos(s.direction)), float(np.sin(s.direction)), t,
            med.air.rho, med.air.mu, med.water.rho, med.water.mu,
            *[float(x) for x in med.current], *[float(x) for x in med.wind],
            sol.cd_scale, sol.added_mass_scale, sol.lift_scale))


def main():
    envs = [pose(TriphibianEnv(build(p())), hash(k) % 9973)
            for k, p in BODY_PLANS.items()]
    b = Batch(envs)
    print(f"{len(envs)} machines, {b.n} panels, {b.nb} bodies")

    t = 0.0
    b.step(t)

    worst_f, worst_m = 0.0, 0.0
    for i, e in enumerate(envs):
        e.data.xfrc_applied[:] = 0.0
        e.solver.apply(e.data, t)
        a, bb = b.boff[i], b.boff[i + 1]
        ref_f = e.data.xfrc_applied
        got_f = b.out["xfrc"][a:bb]
        sc = max(float(np.abs(ref_f).max()), 1e-12)
        worst_f = max(worst_f, float(np.abs(ref_f - got_f).max()) / sc)
        ref_m = e.model.body_mass - e.solver._dry_mass
        got_m = b.out["m_body"][a:bb]
        sm = max(float(np.abs(ref_m).max()), 1e-12)
        worst_m = max(worst_m, float(np.abs(ref_m - got_m).max()) / sm)

    print(f"  xfrc_applied  max relative error {worst_f:.3e}")
    print(f"  body added mass                  {worst_m:.3e}")
    ok = worst_f < 1e-9 and worst_m < 1e-12
    print()
    print("full-pipeline GPU checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
