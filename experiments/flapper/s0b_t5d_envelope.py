"""T5d, probe S support: the lower envelope of the main wings over every stroke and pose the machine uses, from the built T5b
(kinematic, hinge targets tracked exactly).  zmin(x, y) = the lowest height (hull frame) any main-wing surface point reaches above
the hull-frame cell (x, y), 1 cm cells.  A fixed part at (x, y) clears the wings by ``c`` only if its top is below zmin - c over its
footprint grown by c.  Saved to results/S0b_T5d_envelope.npz; `free_top(x, y, r)` reads it.
    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5d_envelope.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os
NOMINAL = bool(os.environ.get("T5D_ENV_NOMINAL"))      # nominal: the strokes without the differential (control) inputs
NPZ = HERE / "results" / ("S0b_T5d_envelope_nominal.npz" if NOMINAL else "S0b_T5d_envelope.npz")
X0, X1, Y0, Y1, DX = -0.75, 1.05, 0.0, 0.80, 0.01
SETS = ("hover", "swim", "surface", "all")


def compute(nph=120):
    import s0b_laws as LW
    import s0b_t5b_probe as Q
    mc = Q._make("water", 0.0, "x4", 1e-4, z=0.2, mod="s0b_t5b_machine")
    m, d, mj = mc.m, mc.d, mc.mj
    gid = [g for g in range(m.ngeom) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "") in ("seg1_wing_g", "seg3_wing_g")]   # left wings; right by mirror
    u = np.linspace(-1, 1, 41)
    v = np.linspace(-1, 1, 81)
    nx, ny = int(round((X1 - X0) / DX)), int(round((Y1 - Y0) / DX))
    zmin = {k: np.full((nx, ny), 9.0) for k in SETS}
    zmax = {k: np.full((nx, ny), -9.0) for k in SETS}
    Pl = dict(LW.LEVEL, **Q.law_cfg_b()["P"])

    def add(q, key):
        d.qpos[mc.qad] = q
        mj.mj_forward(m, d)
        o, R = d.xpos[mc.hull], d.xmat[mc.hull].reshape(3, 3)
        for g in gid:
            s = m.geom_size[g]
            c, Rg = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            for sz in (-1.0, 1.0):                               # the two large faces (the box is 2 cm thick)
                P = np.stack(np.meshgrid(u * s[0], v * s[1], indexing="ij"), -1).reshape(-1, 2)
                P = np.concatenate([P, np.full((len(P), 1), sz * s[2])], 1)
                W = ((c + P @ Rg.T) - o) @ R
                ix = np.floor((W[:, 0] - X0) / DX).astype(int)
                iy = np.floor((np.abs(W[:, 1]) - Y0) / DX).astype(int)
                ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                for k in (key, "all"):
                    np.minimum.at(zmin[k], (ix[ok], iy[ok]), W[ok, 2])
                    np.maximum.at(zmax[k], (ix[ok], iy[ok]), W[ok, 2])
    n = len(mc.aid)
    for amp in (0.5, 0.713, 0.9):
        for sw0 in ((Pl["sw0"],) if NOMINAL else (Pl["sw0"], Pl["sw0"] + 0.35, Pl["sw0"] - 0.35)):
            for alr in ((0.0,) if NOMINAL else (0.0, 0.4, -0.4)):
                for i in range(nph):
                    add(LW.targets_level(mc, 2 * math.pi * i / nph, 1.0, dict(Pl, amp=amp, sw0=sw0, amp_lr=alr))[0], "hover")
    for osc in np.linspace(0, 1, 6):                             # the ramp from the raised pose into the stroke
        for i in range(0, nph, 4):
            add(LW.targets_level(mc, 2 * math.pi * i / nph, 1.0, dict(Pl, amp=0.713), osc=osc)[0], "hover")
    for bias in ((0.0,) if NOMINAL else (0.0, 0.12, -0.12)):
        for a in (0.3, 0.6, 1.0):
            for plr in ((0.0,) if NOMINAL else (0.0, 0.6, -0.6)):
                Pw = dict(LW.ZERO, Phi=math.radians(20.0), Psi=math.radians(50.0) * a, lr_phase=math.pi, amp=a, bias=bias, Psi_lr=plr)
                for i in range(nph):
                    add(LW.targets(mc, 2 * math.pi * i / nph, 1.0, Pw)[0], "swim")
    for sw in (np.linspace(0.0, 0.45, 10) if NOMINAL else np.linspace(-0.5, 0.9, 29)):                        # the surfacing poses: level wings at any mean sweep, one side raised
        for dih in ((0.0,) if NOMINAL else (0.0, 0.3, 0.6)):
            q = np.zeros(n)
            for tag, L in (("FL", 1.0), ("FR", -1.0), ("HL", 1.0), ("HR", -1.0)):
                q[mc.ix[(tag, "sweep")]] = -L * sw
                q[mc.ix[(tag, "stroke")]] = dih
            add(q, "surface")
    np.savez(NPZ, **{f"zmin_{k}": zmin[k] for k in SETS}, **{f"zmax_{k}": zmax[k] for k in SETS})
    return zmin, zmax


def free_top(x, y, r, c=0.03, key="all", z=None):
    """Highest top (hull z) of a vertical part of radius r at (x, |y|) rising from below that keeps ``c`` from the wings."""
    z = np.load(NPZ)[f"zmin_{key}"] if z is None else z
    nx, ny = z.shape
    xs = X0 + DX * (np.arange(nx) + 0.5)
    ys = Y0 + DX * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    sel = (X - x) ** 2 + (Y - abs(y)) ** 2 <= (r + c + 0.7 * DX) ** 2
    if abs(y) < r + c:                                           # the footprint reaches over the centre line: mirror
        sel |= (X - x) ** 2 + (Y + abs(y)) ** 2 <= (r + c + 0.7 * DX) ** 2
    return float(z[sel].min() - c) if sel.any() else 9.0


if __name__ == "__main__":
    zmin, zmax = compute()
    z = zmin["all"]
    print("lowest wing point anywhere", round(float(z[z < 8].min()), 3))
    for y in (0.0, 0.04, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
        print(f"y {y:.2f}: free top (r 0.025, c 0.03) at x = " + "  ".join(f"{x:+.2f}:{min(free_top(x, y, 0.025, z=z), 0.99):+.3f}" for x in np.arange(-0.25, 0.71, 0.08)), flush=True)
