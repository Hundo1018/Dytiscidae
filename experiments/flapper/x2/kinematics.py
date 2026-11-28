"""Pass X2, stage B: the motion of the station at 70% of the wing's length, read
off the rig as pass W8 ran it (w8_cruise.py: l1a.BODY_S, camber 0.04, section
goe417a+lift+extend, strip_twist tip 0 share 1, l1i_cam's cruise job with
feather axis 0.1), not assumed.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/x2/kinematics.py

For each cruise point the rig is run for one beat with `TriphibianEnv.step`
wrapped to record every right-wing strip's world position and chord after each
step.  For the strip nearest 70% of the wing's length from the stroke axis
(and interpolated to exactly 70% between the two neighbours) it prints:
radius from the stroke axis, plunge (z), pitch of the chord about the strip's
span against the flight direction (nose up positive), each fitted by
offset + a cos(wt) + b sin(wt).  Writes kinematics.json beside this file.
"""
import functools
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FLAP = HERE.parent
sys.path.insert(0, str(FLAP))
sys.path.insert(0, str(FLAP.parents[1]))

POINTS = {"B8": dict(V=8.0, f=2.5, Phi=40.0, Psi=15.0, psi0=-2.9557053438451977),
          "B6": dict(V=6.0, f=3.5, Phi=60.0, Psi=40.0, psi0=-11.181753128708566)}
STATION = 0.70


def record(pt):
    import l1a
    import rig
    from dytiscidae.envs import triphibian as TR
    from dytiscidae.physics.fluid import WING
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift+extend")
    rows = []
    orig = TR.TriphibianEnv.step

    def step(self, tgt):
        out = orig(self, tgt)
        d, m, mj = self.data, self.model, self._mj
        pan = self.solver.panels
        R = d.xmat.reshape(-1, 3, 3)[pan.body_id]
        pos = d.xpos[pan.body_id] + np.einsum("nij,nj->ni", R, pan.pos_local)
        c = np.einsum("nij,nj->ni", R, pan.chord_local)
        s = np.einsum("nij,nj->ni", R, pan.span_local)
        # stroke axis: the hinge joints whose body's parent is the hull
        hull = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "seg0_hull")
        ax = [(d.xanchor[j].copy(), d.xaxis[j].copy(), int(m.jnt_bodyid[j]))
              for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
              and int(m.body_parentid[int(m.jnt_bodyid[j])]) == hull
              and not (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith("_jf")]
        rows.append(dict(t=float(d.time), pos=pos.copy(), c=c.copy(), s=s.copy(), wing=pan.kind == WING,
                         ax=ax, x_body=float(d.qpos[0])))
        return out
    TR.TriphibianEnv.step = step
    try:
        r = rig.run(body, pt["V"], 0.0, pt["f"], pt["Phi"], pt["Psi"], pt["psi0"], beats=1, skip=0,
                    feather_axis=0.1, cop_law="plate", strip_twist=dict(tip_deg=0.0, share=1.0))
    finally:
        TR.TriphibianEnv.step = orig
    return rows, r


def derive(name, pt):
    import l1a
    rows, r = record(pt)
    w = 2 * math.pi * pt["f"]
    t = np.array([q["t"] for q in rows])
    wing = rows[0]["wing"]
    # right wing: strips with y > 0
    right = wing & (rows[0]["pos"][:, 1] > 0)
    idx = np.flatnonzero(right)
    # the stroke axis of the right wing: the candidate whose anchor has y > 0
    anchor, axis = None, None
    for a, e, b in rows[0]["ax"]:
        if a[1] > 0:
            anchor, axis = a, e
    if anchor is None:
        raise SystemExit("no right stroke axis")
    # anchor and axis in the body-following frame: the body translates at V along x
    radius = []
    for q in rows[:1]:
        p = q["pos"][idx]
        rel = p - q["ax"][[k for k, (a, e, b) in enumerate(q["ax"]) if a[1] > 0][0]][0]
        e = axis / np.linalg.norm(axis)
        radius = np.linalg.norm(rel - np.outer(rel @ e, e), axis=1)
    order = np.argsort(radius)
    idx, radius = idx[order], radius[order]
    L = l1a.BODY_S["wing_length"]
    r_st = STATION * L
    k = int(np.searchsorted(radius, r_st))
    k0, k1 = max(k - 1, 0), min(k, len(idx) - 1)
    wgt = 0.0 if k1 == k0 else (r_st - radius[k0]) / (radius[k1] - radius[k0])
    ex = np.array([1.0, 0.0, 0.0])

    def pitch_of(c, s):
        s = s / np.linalg.norm(s)
        e1 = ex - (ex @ s) * s
        e1 /= np.linalg.norm(e1)
        n = np.cross(s, e1)
        if n[2] < 0:
            n = -n
        c = c / np.linalg.norm(c)
        return math.atan2(-(c @ n), -(c @ e1))     # chord LE->TE aft: TE below the line = nose up

    z, th, rr = [], [], []
    for q in rows:
        zz, tt = [], []
        for j in (idx[k0], idx[k1]):
            zz.append(q["pos"][j, 2])
            tt.append(pitch_of(q["c"][j], q["s"][j]))
        z.append((1 - wgt) * zz[0] + wgt * zz[1])
        th.append((1 - wgt) * tt[0] + wgt * tt[1])
    z, th = np.array(z), np.array(th)
    # chord direction check: aft?
    c0 = rows[0]["c"][idx[k0]]
    A = np.stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)], 1)
    fz, *_ = np.linalg.lstsq(A, z, rcond=None)
    fth, *_ = np.linalg.lstsq(A, th, rcond=None)
    rz = z - A @ fz
    rth = th - A @ fth
    # the stroke: (Phi/2) sin(w t) about the shoulder; the spec's straight plunge
    h_amp_spec = r_st * math.radians(pt["Phi"]) / 2
    out = dict(name=name, **pt, wing_length=L, station_radius=r_st,
               strips_used=[float(radius[k0]), float(radius[k1])], strip_weight=float(wgt),
               n_right_strips=int(len(idx)), chord_world_at_t0=c0.tolist(),
               z_fit=dict(mean=float(fz[0]), cos=float(fz[1]), sin=float(fz[2]),
                          amp=float(math.hypot(fz[1], fz[2])), phase_deg=float(math.degrees(math.atan2(fz[1], fz[2]))),
                          resid_rms=float(np.sqrt(np.mean(rz ** 2)))),
               pitch_fit_deg=dict(mean=float(math.degrees(fth[0])), cos=float(math.degrees(fth[1])),
                                  sin=float(math.degrees(fth[2])),
                                  amp=float(math.degrees(math.hypot(fth[1], fth[2]))),
                                  resid_rms=float(math.degrees(np.sqrt(np.mean(rth ** 2))))),
               plunge_amp_spec=h_amp_spec,
               t=t.tolist(), z=z.tolist(), pitch_deg=np.degrees(th).tolist(),
               rig_Fz=r["Fz"], rig_Fx=r["Fx"], rig_power=r["aero_power_W"])
    return out


def main():
    res = {}
    for name, pt in POINTS.items():
        o = derive(name, pt)
        res[name] = o
        print(name, json.dumps({k: v for k, v in o.items() if k not in ("t", "z", "pitch_deg")}))
    (HERE / "kinematics.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
