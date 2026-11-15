"""L1 pass 1m: a tail and pitch (PREREG_L1m_tail.md).  Pass 1k's released
cruise points and pass 1i's take-off points on the torque-driven rig, with an
all-moving tail on a boom, fixed and under pitch feedback.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1m_tail.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAILS = {"none": {}, "0.06": dict(stab_length=0.25, stab_chord=0.12, stab_boom=0.3),
         "0.03": dict(stab_length=0.18, stab_chord=0.085, stab_boom=0.3)}
FB = dict(kp=1.0, kd=0.1)
I0 = 0.012          # kg m2: 0.56 kg spread evenly along the 0.5 m hull


def cruise_points():
    rows = [json.loads(x) for x in open(HERE / "results" / "L1k_free.jsonl")]
    return [r for r in rows if r["kind"] == "cruise" and r["free"] == "xz"
            and abs(r["lift_over_weight"] - 1) <= 0.01 and r["Fx"] >= 0 and r["power_W"] <= 41
            and r["crank_eq_peak"] <= 12]


def job(a):
    from freerig import run_free
    from l1a import BODY
    arm, pt, tail, kw = a
    V, th, f, Phi, Psi, psi0, h, law = pt
    kw = dict(kw)
    th = kw.pop("theta", th)
    r = run_free(dict(BODY, **TAILS[tail]), V, th, f, Phi, Psi, psi0, feather_axis=h,
                 cop_law=law, **kw)
    r.update(arm=arm, tail_area=tail, axis=h, law=law, point=list(pt))
    return r


def main() -> int:
    from l1k_free import points
    to, _ = points()
    cr = [(r["V"], r["theta_deg"], r["f"], r["Phi_deg"], r["Psi_deg"], r["psi0_deg"], r["axis"], r["law"])
          for r in cruise_points()]
    grid = []
    for pt in cr:
        for tail in ("none", "0.06"):                       # held: the derivatives
            for th in (-5.0, 0.0, 5.0):
                grid.append(("held th %+.0f" % th, pt, tail, dict(theta=th, hull_inertia=I0)))
        for i0 in (-4.0, 4.0):
            grid.append(("held i0 %+.0f" % i0, pt, "0.06", dict(hull_inertia=I0, tail=dict(i0_deg=i0))))
        fly = dict(free="xzp", beats=40, hull_inertia=I0)
        grid.append(("T0", pt, "none", fly))
        grid.append(("T1", pt, "0.06", fly))
        grid.append(("T2", pt, "0.06", dict(fly, tail=FB)))
        grid.append(("T3", pt, "0.03", fly))
        grid.append(("T4", pt, "0.03", dict(fly, tail=FB)))
        grid.append(("T5", pt, "0.06", dict(fly, hull_inertia=None)))
        grid.append(("T6", pt, "0.06", dict(fly, hull_inertia=0.03)))
        grid.append(("D1", pt, "0.06", dict(fly, theta=10.0, tail=dict(ref_deg=0.0))))
        grid.append(("D2", pt, "0.06", dict(fly, theta=10.0, tail=dict(FB, ref_deg=0.0))))
    for pt in to:
        grid.append(("K0", pt, "none", dict(free="xzp", hull_inertia=I0)))
        grid.append(("K1", pt, "0.06", dict(free="xzp", hull_inertia=I0, tail=FB)))
    with mp.get_context("spawn").Pool(8) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows), "cruise points", len(cr), "take-off points", len(to))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
