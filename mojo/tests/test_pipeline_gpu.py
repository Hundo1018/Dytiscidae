"""The fused pipeline: same numbers as the stage-by-stage path, far fewer trips."""
import time
import numpy as np
import mujoco
import pipeline as pipe_mod

from batched import BatchedPanels
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv


def pose(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(Domain.WATER, randomise=False)
    nq = env.model.nq
    if nq >= 7:
        q = rng.normal(size=4)
        env.data.qpos[3:7] = q / np.linalg.norm(q)
        env.data.qpos[0:3] = rng.normal(scale=2.0, size=3)
    if nq > 7:
        env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
    mujoco.mj_forward(env.model, env.data)
    return env


def group(reps):
    return [pose(TriphibianEnv(build(p())), hash((k, r)) % 99991)
            for r in range(reps) for k, p in BODY_PLANS.items()]


class Fused:
    """Python side of the fused pipeline: one upload, one download per step."""

    def __init__(self, bp):
        self.bp = bp
        n, nb = bp.total_panels, bp.total_bodies
        self.p = pipe_mod.Pipeline(n, nb)
        self.p.upload_static(np.array(
            [a.ctypes.data for a in (
                bp.body_id, bp.is_wing, bp.pos_local, bp.span_local,
                bp.chord_local, bp.normal_local, bp.ext_local, bp.chord,
                bp.camber, bp.dr, bp.volume, bp.cd_bluff, bp.aspect_ratio)]
            + [n], dtype=np.int64))
        self.out = {k: np.zeros(s) for k, s in (
            ("cl", n), ("cd", n), ("F_bluff", (n, 3)), ("D_bluff", n),
            ("m_add", n), ("m_body", nb), ("fz", n), ("alpha", n), ("q", n),
            ("lift_axis", (n, 3)), ("d_hat", (n, 3)), ("U", n), ("vn", n),
            ("pos_w", (n, 3)), ("s_hat", (n, 3)), ("c_hat", (n, 3)),
            ("n_hat", (n, 3)))}

    def step(self, v, om, rho, mu):
        bp = self.bp
        bp.gather_kinematics()
        o = self.out
        desc = np.array(
            [a.ctypes.data for a in (bp.xpos, bp.xmat, v, om, rho, mu)]
            + [o[k].ctypes.data for k in (
                "cl", "cd", "F_bluff", "D_bluff", "m_add", "m_body", "fz",
                "alpha", "q", "lift_axis", "d_hat", "U", "vn", "pos_w",
                "s_hat", "c_hat", "n_hat")]
            + [bp.total_panels, bp.total_bodies], dtype=np.int64)
        self.p.step(desc, (1.0, 1.0, int((bp.is_wing == 0).any())))


def flow(n, seed):
    rng = np.random.default_rng(seed); C = np.ascontiguousarray
    v = C(rng.normal(scale=4.0, size=(n, 3)))
    om = C(rng.normal(scale=6.0, size=(n, 3)))
    rho = C(np.where(rng.random(n) < 0.5, 1.225, 1025.0))
    mu = C(np.where(rho > 500, 1.08e-3, 1.81e-5))
    return v, om, rho, mu


def main():
    envs = group(2)
    bp = BatchedPanels(envs)
    n = bp.total_panels
    v, om, rho, mu = flow(n, 5)

    bp.run(v, om, rho, mu)                       # stage-by-stage reference
    f = Fused(bp)
    f.step(v, om, rho, mu)

    print(f"{len(envs)} machines, {n} panels: fused vs stage-by-stage")
    worst = 0.0
    for k in ("cl", "cd", "F_bluff", "m_add", "m_body", "alpha", "s_hat"):
        a = getattr(bp, {"F_bluff": "F_bluff"}.get(k, k))
        b = f.out[k]
        sc = max(float(np.abs(a).max()), 1e-12)
        e = float(np.abs(a - b).max()) / sc
        worst = max(worst, e)
        print(f"  {k:10s} {e:.3e}")
    ok = worst < 1e-15

    print()
    print(f"{'machines':>9s} {'panels':>7s} {'numpy':>10s} {'staged GPU':>11s} {'fused GPU':>10s} {'vs numpy':>9s}")
    # Built once and sliced.  Constructing a TriphibianEnv compiles MJCF, so
    # rebuilding the group per sweep point cost more than every measurement in
    # this file put together -- it was the whole runtime.
    pool = group(16)
    for take in (7, 28, 112):
        g = pool[:take]
        b2 = BatchedPanels(g)
        n2 = b2.total_panels
        vv, oo, rr, mm = flow(n2, 9)
        f2 = Fused(b2)
        f2.step(vv, oo, rr, mm)
        t0 = time.perf_counter()
        for _ in range(20): f2.step(vv, oo, rr, mm)
        fused = (time.perf_counter() - t0) / 20 * 1e6
        b2.run(vv, oo, rr, mm)
        t0 = time.perf_counter()
        for _ in range(20): b2.run(vv, oo, rr, mm)
        staged = (time.perf_counter() - t0) / 20 * 1e6
        # numpy equivalent of the same five stages
        from dytiscidae.physics.fluid import (
            lift_coefficient, drag_coefficient, skin_friction_cd)
        def cpu():
            R = b2.xmat.reshape(-1, 3, 3)[b2.body_id]
            s = np.einsum('nij,nj->ni', R, b2.span_local)
            c = np.einsum('nij,nj->ni', R, b2.chord_local)
            nh = np.einsum('nij,nj->ni', R, b2.normal_local)
            _ = b2.xpos[b2.body_id] + np.einsum('nij,nj->ni', R, b2.pos_local)
            v2 = vv - np.einsum('ni,ni->n', vv, s)[:, None] * s
            U = np.linalg.norm(v2, axis=1); Us = np.maximum(U, 1e-6)
            re = rr * U * b2.chord / np.maximum(mm, 1e-12)
            ca = np.einsum('ni,ni->n', v2, c) / Us
            sa = np.einsum('ni,ni->n', v2, nh) / Us
            al = np.arctan2(sa, np.abs(ca) + 1e-12) + 2 * b2.camber
            rf = np.abs(np.einsum('ni,ni->n', oo, s)) * b2.chord / (2 * Us)
            w = b2.is_wing.astype(bool)
            cl = np.where(w, lift_coefficient(al, re, b2.aspect_ratio, rf), 0.0)
            cd = np.where(w, drag_coefficient(al, re, b2.aspect_ratio, cl), 0.0)
            vax = np.einsum('ni,ni->n', vv, s); vc = vv - vax[:, None] * s
            uc = np.linalg.norm(vc, axis=1)
            ex, ey, ez = b2.ext_local[:, 0], b2.ext_local[:, 1], b2.ext_local[:, 2]
            reb = rr * np.abs(vax) * ex / np.maximum(mm, 1e-12)
            _ = 0.5 * rr * np.abs(vax) * vax * (
                b2.cd_bluff * ey * ez + skin_friction_cd(reb) * 2 * (ex*ey+ey*ez+ex*ez))
            e = np.maximum(b2.ext_local, 1e-4)
            cax = np.clip(0.5*(e[:, [1,2,0]]+e[:, [2,0,1]])/(2*e), 0.05, 10.0)
            ma = np.where(w, rr*np.pi*b2.chord**2*0.25*b2.dr, cax.mean(1)*rr*b2.volume)
            mb = np.zeros(b2.total_bodies); np.add.at(mb, b2.body_id, ma)
            return mb
        cpu()
        t0 = time.perf_counter()
        for _ in range(20): cpu()
        cput = (time.perf_counter() - t0) / 20 * 1e6
        print(f"{len(g):9d} {n2:7d} {cput:10.1f} {staged:11.1f} {fused:10.1f} {cput/fused:8.2f}x")

    print()
    print("fused pipeline checks passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
