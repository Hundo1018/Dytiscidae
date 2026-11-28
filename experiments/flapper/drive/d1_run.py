"""Pass D1: the stroke candidates against the prescribed stroke
(`PREREG_D1_stroke_drive.md`).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d1_run.py sweep  OUT.jsonl
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d1_run.py checks SWEEP.jsonl OUT.jsonl

``sweep``: every candidate over its gears (and the cable's springs) at both
points, and the baseline.  ``checks``: at each candidate's best gear, the
vacuum (K1), the half-size motor (K8), and the link mass at 0 and 40 g/m
(the four-bar, cruise).
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
DT = 1e-4
GEARS = (12, 16, 24, 32, 48, 64, 90)
SPRINGS = (1.0, 2.0, 4.0, 8.0, 16.0)
POINTS = {
    # V, theta, f, stroke, feather amplitude, offset, lever a, extra run_free args
    "cruise": (8.0, 0.0, 2.5, 40.0, 15.0, -2.94, 0.03,
               dict(strip_twist=dict(tip_deg=0.0, share=1.0))),
    "takeoff": (0.0, 90.0, 4.5, 85.0, 50.0, 0.0, 0.04, dict(solver_flags=dict(camber_moment=True))),
}


def gears(kind, point):
    V, th, f, Phi, *_ = POINTS[point]
    w = 2 * math.pi * f
    top = (math.radians(Phi) / 2 * w) if kind == "DD" else w      # the input's peak speed
    return [N for N in GEARS if N * top < 1000.0]


def job(a):
    import l1a
    from drive import Drive
    from freerig import run_free
    from dytiscidae.physics.medium import Fluid
    point, kind, N, k, extra = a
    V, th, f, Phi, Psi, psi0, lev, kw = POINTS[point]
    lev = extra.get("a", lev)
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift")
    kw = dict(kw, feather_axis=0.1, cop_law="plate", dt=DT)
    if extra.get("vacuum"):
        kw["fluid"] = Fluid("vacuum", 1e-9, 1.81e-5)
    dv = None
    if kind != "base":
        dv = Drive(kind, Phi, f, N, a=lev, k_spring=k, dt=DT, feather_axis=0.1,
                   lam=extra.get("lam", 0.015), motor=extra.get("motor"))
    try:
        r = run_free(body, V, th, f, Phi, Psi, psi0, drive=dv, **kw)
    except Exception as e:                       # a row, not a crash
        return dict(point=point, kind=kind, N=N, k=k, extra=extra, error=repr(e))
    r.update(point=point, kind=kind, N=N, k=k, extra=extra)
    return r


def best(rows, point, kind):
    Phi = POINTS[point][3]
    ok = [r for r in rows if r["point"] == point and r["kind"] == kind and not r.get("extra")
          and "error" not in r and r["finite"] and abs(r["drive"]["amp_deg"] - Phi) <= 2.0]
    return min(ok, key=lambda r: r["drive"]["elec_W"]) if ok else None


def main() -> int:
    stage = sys.argv[1]
    if stage == "sweep":
        grid = []
        for point in POINTS:
            grid.append((point, "base", 0, 0.0, {}))
            for kind in ("FB", "SY", "DD"):
                grid += [(point, kind, N, 0.0, {}) for N in gears(kind, point)]
            grid += [(point, "CT", N, k, {}) for N in gears("CT", point) for k in SPRINGS]
    elif stage == "lever":                       # addendum D1b
        grid = []
        for a in (0.06, 0.08, 0.12):
            for kind in ("FB", "SY"):
                grid += [("takeoff", kind, N, 0.0, dict(a=a)) for N in (16, 24, 32)]
            grid += [("takeoff", "CT", N, k, dict(a=a)) for N in (24, 32) for k in (8.0, 16.0)]
    else:
        rows = [json.loads(x) for x in open(sys.argv[2])]
        grid = []
        for point in POINTS:
            grid.append((point, "base", 0, 0.0, dict(vacuum=True)))
            for kind in ("FB", "SY", "DD", "CT"):
                b = best(rows, point, kind)
                if b is None:
                    continue
                grid.append((point, kind, b["N"], b["k"], dict(vacuum=True)))
                grid.append((point, kind, b["N"], b["k"], dict(motor=dict(tau_s=0.5))))
        b = best(rows, "cruise", "FB")
        if b is not None:
            grid += [("cruise", "FB", b["N"], 0.0, dict(lam=lam)) for lam in (0.0, 0.04)]
    out = Path(sys.argv[-1]).resolve()
    with mp.get_context("spawn").Pool(6) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(job, grid):
            r.pop("series", None)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print("rows", len(grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
