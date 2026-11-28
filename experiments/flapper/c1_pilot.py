"""Pass C1: height and speed held by the tail and the flapping frequency, on
the wing of pass W6 (PREREG_C1_pilot.md).  The Goe 417a rectangle whose tip
alone feathers, the 0.06 m2 all-moving tail of pass 1m, torque-driven rig,
released in x, z and pitch for 100 beats.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/c1_pilot.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAIL = dict(stab_length=0.25, stab_chord=0.12, stab_boom=0.3)
FB = dict(kp=1.0, kd=0.1)
PI = dict(FB, ki=2.0)
PILOT = dict(kh=6.0, kvz=6.0, pitch_lim_deg=12.0, kf=0.08, f_lim=0.2)
I0 = 0.012          # kg m2, as pass 1m
BEATS, STEP_AT = 100, 40
V = 8.0


def body():
    import l1a
    return dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift", **TAIL)


def points():
    """W6's sustaining 8 m/s points with the tip alone feathering and no held twist."""
    rows = [json.loads(x) for x in open(HERE / "results" / "W6_twist.jsonl")]
    return [(r["f"], r["Phi_deg"], r["Psi_deg"], r["psi0_deg"], r["aero_power_W"]) for r in rows
            if r.get("kind") == "cruise" and r["V"] == V and r["twist"] == 0.0 and r["share"] == 1.0
            and r["sustains"] and r["crank_eq_peak"] <= 12]


def job(a):
    import numpy as np
    from freerig import run_free
    arm, pt, kw = a
    f, Phi, Psi, psi0, p_held = pt
    kw = dict(kw)
    th = kw.pop("theta", 0.0)
    o = run_free(body(), V, th, f, Phi, Psi, psi0, feather_axis=0.1, cop_law="plate",
                 strip_twist=dict(tip_deg=0.0, share=1.0), hull_inertia=I0, free="xzp",
                 beats=BEATS, keep_series=True, **kw)
    s = o.pop("series")
    o.pop("com0", None)
    nb = int(round(1.0 / (f * 2.5e-4)))
    per = lambda x: [float(v) for v in np.asarray(x)[: BEATS * nb].reshape(BEATS, nb).mean(1)]
    pos, vel = np.array(s["pos"]), np.array(s["vel"])
    o.update(arm=arm, point=list(pt[:4]), held_power_W=p_held,
             z_beat=per(pos[:, 2] - pos[0, 2]), vx_beat=per(vel[:, 0]), vz_beat=per(vel[:, 2]),
             pitch_beat=per(np.degrees(s["pitch"])), P_beat=per(s["P"]),
             tail_beat=per(np.degrees(s["tail"])), f_beat=per(np.array(s["w"]) / (2 * np.pi)))
    o.pop("pitch_beat_mean_deg", None)
    return o


def main() -> int:
    grid = []
    full = dict(tail=PI, pilot=PILOT)
    for pt in points():
        grid.append(("A0 fixed tail", pt, {}))
        grid.append(("A1 pitch loop", pt, dict(tail=FB)))
        grid.append(("A2 pilot", pt, full))
        grid.append(("Z+2", pt, dict(tail=PI, pilot=dict(PILOT, z_step=(STEP_AT, 2.0)))))
        grid.append(("Z-2", pt, dict(tail=PI, pilot=dict(PILOT, z_step=(STEP_AT, -2.0)))))
        grid.append(("V+1", pt, dict(tail=PI, pilot=dict(PILOT, v_step=(STEP_AT, 1.0)))))
        grid.append(("V-1", pt, dict(tail=PI, pilot=dict(PILOT, v_step=(STEP_AT, -1.0)))))
        grid.append(("D nose-up 10", pt, dict(full, theta=10.0, tail=dict(PI, ref_deg=0.0))))
    for f in (2.0, 2.5, 3.0):                    # the wings held still: a glider under the same loops
        grid.append(("G0 no flapping", (f, 0.0, 0.0, -2.955, 0.0), full))
    with mp.get_context("spawn").Pool(14) as pool:
        with open(Path(sys.argv[1]).resolve(), "w") as fh:
            for r in pool.imap_unordered(job, grid, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
    print("rows", len(grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
