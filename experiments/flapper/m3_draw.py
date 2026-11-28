"""Pass M3 figures: the built model posed at recorded states of a run (its trace),
side (x-z) and front (y-z) views in the world frame, the beach line, the CoM,
the contact points MuJoCo finds at that pose, and the recorded vertical ground
force per group at that instant.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m3_draw.py ID OUT.png t1,t2,... [title]
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def surface(m, g, n=10):
    """Points on a geom's surface, local frame."""
    t, s = m.geom_type[g], m.geom_size[g]
    if t == 6:                                           # box: the corners
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return c * s
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n), np.linspace(0, np.pi, n))
    sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1).reshape(-1, 3)
    if t == 7:                                           # mesh: its vertices
        a, k = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
        return np.array(m.mesh_vert[a:a + k], float)
    if t == 4:                                           # ellipsoid
        return sph * s[:3]
    if t == 3:                                           # capsule along z
        r, h = s[0], s[1]
        p = sph * r
        return np.concatenate([p + [0, 0, h], p - [0, 0, h]])
    return sph * s[0]


def draw(rid, out, times, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    from mission import Machine
    tr = np.load(HERE / "results" / "M3_traces" / f"{rid}.npz")
    mc = Machine("land", pose=dict(x=20, z=5))
    m, d, mj = mc.m, mc.d, mc.mj
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    col = {"seg0_hull": "0.55", "seg1_wing": "tab:blue", "seg2_wing": "tab:orange", "seg3_strut": "0.3",
           "seg4_wing": "tab:green", "seg5_wing": "tab:green"}
    fig, axes = plt.subplots(len(times), 2, figsize=(11, 3.2 * len(times)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    for row, t in enumerate(times):
        i = int(np.argmin(np.abs(tr["t"] - t)))
        d.qpos[:] = tr["qpos"][i]
        d.qvel[:] = 0
        mj.mj_forward(m, d)
        com = mc.com()
        att = tr["att"][i]
        psi = float(att[0])
        e1 = np.array([math.cos(psi), math.sin(psi), 0.0])      # body heading, horizontal
        e2 = np.array([-math.sin(psi), math.cos(psi), 0.0])     # body left, horizontal
        for col_i, (e, lab) in enumerate(((e1, "side: along the body heading (right = nose)"),
                                          (e2, "front: across the body (right = left wing side)"))):
            ax = axes[row, col_i]
            proj = lambda P: np.stack([P @ e, P[..., 2]], -1)
            for g in range(m.ngeom):
                bid = int(m.geom_bodyid[g])
                if bid == 0:
                    continue
                P = d.geom_xpos[g] + surface(m, g) @ d.geom_xmat[g].reshape(3, 3).T
                pts = proj(P)
                try:
                    hull = ConvexHull(pts)
                    ax.add_patch(Polygon(pts[hull.vertices], closed=True, fc=col.get(names[bid], "k"),
                                         ec="k", lw=0.5, alpha=0.55))
                except Exception:
                    ax.plot(pts[:, 0], pts[:, 1], color=col.get(names[bid], "k"), lw=1)
            ss = np.linspace(-1.0, 1.0, 50)
            line = com[None, :] + ss[:, None] * e[None, :]
            gz = np.array([mc.env.ground_height(p[0]) for p in line])
            gx = line @ e
            ax.plot(gx, gz, color="saddlebrown", lw=1.5)
            c2 = proj(com)
            ax.plot(c2[0], c2[1], "r+", ms=12, mew=2)
            for k in range(d.ncon):
                p2 = proj(np.array(d.contact[k].pos))
                ax.plot(p2[0], p2[1], "kx", ms=6)
            ax.set_aspect("equal")
            ax.set_xlim(c2[0] - 0.75, c2[0] + 0.75)
            ax.set_ylim(gz.min() - 0.05, max(com[2] + 0.45, gz.max() + 0.1))
            v = tr["vert"][i]
            ax.set_title("%s  t=%.2f s  %s\nground N: wingL %.1f wingR %.1f tail %.1f body %.1f  yaw %.0f pitch %.0f roll %.0f deg" % (
                rid, tr["t"][i], lab, v[0], v[1], v[2], v[3], *np.degrees(att)), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title or rid, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    draw(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")], sys.argv[4] if len(sys.argv) > 4 else "")
