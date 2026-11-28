"""M3 phase-1 probes (disclosed in PREREG_M3_land.md section 0): the built
machine's geometry, what touches the beach at rest, hinge ranges and signs,
and the cost of a step.  Decides nothing.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m3_probe.py geom|rest|signs
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mission import Machine  # noqa: E402


def contacts_by_body(mc):
    m, d, mj = mc.m, mc.d, mc.mj
    out, c = {}, np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        if (b1 == 0) == (b2 == 0):
            continue
        b = b2 if b1 == 0 else b1
        mj.mj_contactForce(m, d, i, c)
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
        out[nm] = out.get(nm, 0.0) + abs(c[0])
    return out


def geom():
    mc = Machine("land", pose=dict(x=20, z=5))
    m, mj = mc.m, mc.mj
    print("mass", mc.mass, "acts", list(mc.env.act_names), "roles", list(mc.role))
    for n, j in enumerate(mc.jid):
        print("joint", mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j), "limited", m.jnt_limited[j],
              "range", np.degrees(m.jnt_range[j]).round(1), "axis", m.jnt_axis[j].round(3),
              "damping", m.dof_damping[mc.dof[n]], "armature", round(float(m.dof_armature[mc.dof[n]]), 5))
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        print("geom", g, mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]), "type", m.geom_type[g],
              "size", m.geom_size[g].round(4), "contype", m.geom_contype[g], "conaff", m.geom_conaffinity[g],
              "pos", mc.d.geom_xpos[g].round(3))
    for b in range(1, m.nbody):
        print("body", b, mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b), "mass", round(float(m.body_mass[b]), 4),
              "xpos", mc.d.xpos[b].round(3))


def rest(x=20.0, seconds=2.0):
    mc = Machine("land", pose=dict(x=x, z=5))
    mc.drop_to_ground()
    p0 = mc.d.qpos[:3].copy()
    q0 = np.zeros(len(mc.aid))
    t0 = time.time()
    for k in range(int(seconds / mc.dt)):
        mc.step(q0)
    wall = time.time() - t0
    print("rest x=%.0f: drift %s m, pitch %.2f, load %.3f, qacc %d, wall %.1f s for %.1f s (%.0f us/step)" % (
        x, (mc.d.qpos[:3] - p0).round(4), mc.pitch_deg(), mc.contact_load(), mc.bad_qacc, wall, seconds,
        1e6 * wall / mc.n))
    print("  contact normal force by body (N):", {k: round(v, 3) for k, v in contacts_by_body(mc).items()})


def signs():
    """A small step on each hinge, from rest held in the air: which way does a wing point move."""
    mc = Machine("air", pose=dict(x=20, z=30))
    m, d, mj = mc.m, mc.d, mc.mj
    for n, a in enumerate(mc.env.act_names):
        b = int(m.jnt_bodyid[mc.jid[n]])
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        q = np.zeros(len(mc.aid))
        mc.place(q=q)
        before = np.array([d.geom_xpos[g].copy() for g in geoms])
        q[n] = 0.3
        mc.place(q=q)
        after = np.array([d.geom_xpos[g].copy() for g in geoms])
        print(a, mc.role[n], "geom centre moves (world, +0.3 rad):", (after - before).mean(0).round(4))
        mc.place(q=np.zeros(len(mc.aid)))


def stand(stroke_deg=-12.0, feather_deg=0.0, tail_deg=-30.0, ramp_s=1.5, hold_s=1.0, x=20.0):
    """Lower both wings and the tail TE onto the beach slowly, hold, read who carries the weight."""
    mc = Machine("land", pose=dict(x=x, z=5))
    mc.drop_to_ground()
    for k in range(int(0.5 / mc.dt)):            # settle from the drop
        mc.step(np.zeros(len(mc.aid)))
    goal = np.array([{"stroke": stroke_deg, "feather": feather_deg, "tail": tail_deg}[r] for r in mc.role])
    goal = np.radians(goal)
    p0 = mc.d.qpos[:3].copy()
    peak = np.zeros(len(mc.aid))
    n_ramp, n_hold = int(ramp_s / mc.dt), int(hold_s / mc.dt)
    for k in range(n_ramp + n_hold):
        s = min(1.0, k / n_ramp)
        mc.step(goal * s, goal / ramp_s * (k < n_ramp))
        peak = np.maximum(peak, np.abs(mc.d.actuator_force[mc.aid]))
    cb = contacts_by_body(mc)
    hull_z = mc.d.xpos[mc.hull][2] - mc.env.ground_height(mc.d.xpos[mc.hull][0])
    print("stand stroke %.0f feather %.0f tail %.0f: drift %s pitch %.1f load %.3f qacc %d hull origin %.3f m over ground" % (
        stroke_deg, feather_deg, tail_deg, (mc.d.qpos[:3] - p0).round(4), mc.pitch_deg(), mc.contact_load(),
        mc.bad_qacc, hull_z))
    print("  normal force by body (N):", {k: round(float(v), 3) for k, v in cb.items()})
    print("  peak hinge torque (N m) by actuator:", dict(zip(mc.env.act_names, peak.round(3))))
    print("  final hinge angles (deg):", dict(zip(mc.env.act_names, np.degrees(mc.d.qpos[mc.qad]).round(1))))
    print("  elec W:", {k: round(v, 2) for k, v in mc.summary()["elec_W"].items()})


def crutch(keys=None):
    """One stance of a both-wing crutch on the trailing edges, keyframed (stroke, feather, tail) deg, linear."""
    keys = keys or [(0.0, (0, 0, 0)), (1.0, (-12, -50, -30)), (1.5, (-12, -50, -30)),
            (2.5, (-22, -15, -30)), (2.9, (5, -15, -30)), (3.3, (5, -50, -30)), (3.7, (-12, -50, -30)),
            (4.7, (-22, -15, -30)), (5.0, (-22, -15, -30))]
    mc = Machine("land", pose=dict(x=20, z=5))
    mc.drop_to_ground()
    for k in range(int(0.5 / mc.dt)):
        mc.step(np.zeros(len(mc.aid)))
    col = {"stroke": 0, "feather": 1, "tail": 2}
    ix = [col[r] for r in mc.role]
    ts = np.array([k[0] for k in keys])
    vs = np.radians(np.array([k[1] for k in keys], float))
    w = mc.env.solver._dry_mass[1:]          # d.subtree_com is not the CoM here (stale subtree mass)
    com = lambda: (w[:, None] * mc.d.xipos[1:]).sum(0) / w.sum()
    c0, last = com(), -1.0
    for k in range(int(ts[-1] / mc.dt)):
        t = k * mc.dt
        i = min(np.searchsorted(ts, t, side="right") - 1, len(ts) - 2)
        s = (t - ts[i]) / (ts[i + 1] - ts[i])
        v = vs[i] + s * (vs[i + 1] - vs[i])
        r = (vs[i + 1] - vs[i]) / (ts[i + 1] - ts[i])
        mc.step(v[ix], r[ix])
        if t - last >= 0.5 - 1e-9:
            last = t
            cb = contacts_by_body(mc)
            print("t %.1f com dx %+.3f dz %+.3f pitch %+.1f  N by body %s" % (
                t, *(com() - c0)[[0, 2]], mc.pitch_deg(), {k[5:]: round(float(x), 2) for k, x in cb.items()}))
    print("qacc", mc.bad_qacc, "elec", {k: round(v, 2) for k, v in mc.summary()["elec_W"].items()})


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "crutch":
        crutch()
        raise SystemExit
    if a == "vault":                              # place the trailing edge first, then sweep it back
        k, t = [(0.0, (0, 0, 0)), (0.4, (8, -55, -30))], 0.4
        for i in range(3):
            k += [(t + 0.4, (-10, -55, -30)), (t + 1.4, (-24, -5, -30)), (t + 1.8, (8, -5, -30)), (t + 2.2, (8, -55, -30))]
            t += 2.2
        crutch(k)
        raise SystemExit
    if a == "stand":
        stand(*[float(v) for v in sys.argv[2:5]])
    else:
        {"geom": geom, "rest": lambda: (rest(17.0), rest(25.0)), "signs": signs}[a]()
