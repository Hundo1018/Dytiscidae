"""Check the GPU kinematics against the numpy block it replaces.

Runs on the real body plans, not synthetic data: the panel counts, the
body_id gather pattern and the rotation matrices are all whatever the actual
phenotypes produce.  Then does it again on several machines concatenated into
one launch, which is the case the whole batching argument rests on.

Needs the parent project's venv on sys.path for dytiscidae, and the built
fluid_gpu.so importable.  See mojo/README.md.
"""
import sys

import numpy as np

import fluid_gpu

from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv


def numpy_reference(xpos, xmat, body_id, loc, span, chord, normal):
    """Exactly fluid.py:428-432."""
    R = xmat.reshape(-1, 3, 3)[body_id]
    pos = xpos[body_id] + np.einsum("nij,nj->ni", R, loc)
    s_hat = np.einsum("nij,nj->ni", R, span)
    c_hat = np.einsum("nij,nj->ni", R, chord)
    n_hat = np.einsum("nij,nj->ni", R, normal)
    return pos, s_hat, c_hat, n_hat


def run_gpu(xpos, xmat, body_id, loc, span, chord, normal):
    n = len(body_id)
    nbody = len(xpos)
    outs = [np.zeros((n, 3), dtype=np.float64) for _ in range(4)]
    ins = [xpos, xmat, body_id, loc, span, chord, normal]
    desc = np.array(
        [a.ctypes.data for a in ins] + [a.ctypes.data for a in outs] + [n, nbody],
        dtype=np.int64,
    )
    got = fluid_gpu.body_to_world(desc)
    assert got == n, f"kernel reported {got} panels, expected {n}"
    return outs


def pose(env, seed):
    """Put the machine in a real, non-degenerate attitude.

    Without this the test is worthless: a freshly constructed TriphibianEnv has
    never had mj_forward called, so `data.xmat` is all zeros and every rotation
    in the comparison is a multiplication by zero.  The first version of this
    test reported bit-exact agreement on all seven plans while proving nothing
    -- it was comparing zeros to zeros.  Joint angles are randomised too, so the
    bodies do not all share the root's orientation.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    env.reset(Domain.WATER, randomise=False)
    nq = env.model.nq
    if nq >= 7:
        # random unit quaternion for the free joint, random angles below it
        q = rng.normal(size=4)
        env.data.qpos[3:7] = q / np.linalg.norm(q)
        env.data.qpos[0:3] = rng.normal(scale=2.0, size=3)
    if nq > 7:
        env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
    mujoco.mj_forward(env.model, env.data)
    assert np.abs(env.data.xmat).max() > 0.5, "xmat still degenerate"
    return env


def case(name, env):
    p = env.solver.panels
    body_id = np.ascontiguousarray(p.body_id, dtype=np.int32)
    xpos = np.ascontiguousarray(env.data.xpos, dtype=np.float64)
    xmat = np.ascontiguousarray(env.data.xmat, dtype=np.float64)
    loc = np.ascontiguousarray(p.pos_local, dtype=np.float64)
    span = np.ascontiguousarray(p.span_local, dtype=np.float64)
    chord = np.ascontiguousarray(p.chord_local, dtype=np.float64)
    normal = np.ascontiguousarray(p.normal_local, dtype=np.float64)

    ref = numpy_reference(xpos, xmat, body_id, loc, span, chord, normal)
    got = run_gpu(xpos, xmat, body_id, loc, span, chord, normal)

    worst = 0.0
    for r, g in zip(ref, got):
        worst = max(worst, float(np.abs(r - g).max()))
    return len(body_id), worst


def main():
    failures = []
    print(f"{'plan':10s} {'panels':>7s} {'max abs err':>12s}")
    envs = {}
    for nm, plan in BODY_PLANS.items():
        env = pose(TriphibianEnv(build(plan())), seed=hash(nm) % 10000)
        envs[nm] = env
        n, worst = case(nm, env)
        flag = "" if worst < 1e-13 else "   <-- MISMATCH"
        if worst >= 1e-13:
            failures.append(nm)
        print(f"{nm:10s} {n:7d} {worst:12.3e}{flag}")

    # The batching claim: panels from different morphologies, one launch.
    print()
    names = list(envs)
    xposs, xmats, bids, locs, spans, chords, normals = [], [], [], [], [], [], []
    body_off = 0
    for nm in names:
        env = envs[nm]
        p = env.solver.panels
        xposs.append(np.ascontiguousarray(env.data.xpos, dtype=np.float64))
        xmats.append(np.ascontiguousarray(env.data.xmat, dtype=np.float64))
        bids.append(np.asarray(p.body_id, dtype=np.int32) + body_off)
        body_off += len(env.data.xpos)
        locs.append(np.ascontiguousarray(p.pos_local, dtype=np.float64))
        spans.append(np.ascontiguousarray(p.span_local, dtype=np.float64))
        chords.append(np.ascontiguousarray(p.chord_local, dtype=np.float64))
        normals.append(np.ascontiguousarray(p.normal_local, dtype=np.float64))

    XP = np.ascontiguousarray(np.concatenate(xposs))
    XM = np.ascontiguousarray(np.concatenate(xmats))
    BI = np.ascontiguousarray(np.concatenate(bids))
    LO = np.ascontiguousarray(np.concatenate(locs))
    SP = np.ascontiguousarray(np.concatenate(spans))
    CH = np.ascontiguousarray(np.concatenate(chords))
    NO = np.ascontiguousarray(np.concatenate(normals))

    ref = numpy_reference(XP, XM, BI, LO, SP, CH, NO)
    got = run_gpu(XP, XM, BI, LO, SP, CH, NO)
    worst = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    print(f"{len(names)} morphologies concatenated: {len(BI)} panels in one "
          f"launch, max abs err {worst:.3e}")
    if worst >= 1e-13:
        failures.append("batched")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all GPU kinematics checks passed (< 1e-13 vs numpy einsum)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
