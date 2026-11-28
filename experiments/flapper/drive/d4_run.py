"""Pass D4: the variable-stroke candidates at both points
(`PREREG_D4_variable_stroke.md`).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d4_run.py sweep  OUT.jsonl
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d4_run.py checks SWEEP.jsonl OUT.jsonl

``sweep``: the baseline, the dedicated four-bar (`drive.Drive` "FB", lever
60 mm) and VL, VR, VP over the frozen gears at both points.  ``checks``:
each at its best gear in a vacuum (V1).  Three worker processes.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import d1_run  # noqa: E402

DT = 1e-4
POINTS = d1_run.POINTS
GEARS = {"cruise": (16, 24, 32, 48), "takeoff": (16, 24, 32)}
KINDS = ("FB", "VL", "VR", "VP")
LEVER_FB = 0.06
# Added after the first vacuum smoke rows, before any air row (RESULT_D4): inertia
# on the adjuster's own locked coordinate, kg (slides) or kg m^2 (VP's carrier),
# so the lock is as stiff as the loops.  Both variants are run.
STIFF = {"VL": 1.0, "VR": 1.0, "VP": 0.1}
VARIANTS = ("frozen", "stiff")


def extra_of(kind, variant):
    return dict(va=dict(lock_armature=STIFF[kind])) if (variant == "stiff" and kind in STIFF) else {}
WORKERS = 3


def job(a):
    import l1a
    from drive import Drive
    from freerig import run_free
    from va import VA
    from dytiscidae.physics.medium import Fluid
    point, kind, N, extra = a
    V, th, f, Phi, Psi, psi0, _, kw = POINTS[point]
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift")
    kw = dict(kw, feather_axis=0.1, cop_law="plate", dt=DT)
    if extra.get("vacuum"):
        kw["fluid"] = Fluid("vacuum", 1e-9, 1.81e-5)
    for k_ in ("beats", "hold_beats"):
        if k_ in extra:
            kw[k_] = extra[k_]
    dv = None
    if kind == "FB":
        dv = Drive("FB", Phi, f, N, a=LEVER_FB, dt=DT, feather_axis=0.1)
    elif kind != "base":
        dv = VA(kind, point, f, N, dt=DT, feather_axis=0.1, **extra.get("va", {}))
    try:
        r = run_free(body, V, th, f, Phi, Psi, psi0, drive=dv, **kw)
    except Exception as e:                       # a row, not a crash
        return dict(point=point, kind=kind, N=N, extra=extra, error=repr(e))
    r.update(point=point, kind=kind, N=N, extra=extra)
    return r


def best(rows, point, kind, variant="frozen"):
    """Least electrical power with the stroke within 2 deg and the beat at
    least 0.98 of the planned (the frozen rule)."""
    Phi = POINTS[point][3]
    ok = [r for r in rows if r["point"] == point and r["kind"] == kind and r.get("extra") == extra_of(kind, variant)
          and "error" not in r and r["finite"] and abs(r["drive"]["amp_deg"] - Phi) <= 2.0
          and r["drive"]["freq_ratio"] >= 0.98]
    return min(ok, key=lambda r: r["drive"]["elec_W"]) if ok else None


def main() -> int:
    stage = sys.argv[1]
    if stage == "one":                           # a smoke row: point kind N [json of extra]
        r = job((sys.argv[2], sys.argv[3], int(sys.argv[4]), json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}))
        r.pop("series", None)
        print(json.dumps(r))
        return 0
    if stage == "sweep":
        grid = []
        for point in POINTS:
            grid.append((point, "base", 0, {}))
            for kind in KINDS:
                for variant in (VARIANTS if kind in STIFF else VARIANTS[:1]):
                    grid += [(point, kind, N, extra_of(kind, variant)) for N in GEARS[point]]
    else:
        rows = [json.loads(x) for x in open(sys.argv[2])]
        grid = []
        for point in POINTS:
            for kind in KINDS:
                for variant in (VARIANTS if kind in STIFF else VARIANTS[:1]):
                    b = best(rows, point, kind, variant)
                    # no gear passes the rule: the vacuum row at the expected gear, labelled by the reader
                    N = b["N"] if b is not None else {"cruise": 48, "takeoff": 24}[point]
                    grid.append((point, kind, N, dict(extra_of(kind, variant), vacuum=True)))
    import va
    va.designs()                                 # solved once, before the workers read it
    out = Path(sys.argv[-1]).resolve()
    with mp.get_context("spawn").Pool(WORKERS) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(job, grid):
            r.pop("series", None)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("done", r["point"], r["kind"], r["N"], r.get("error", ""), flush=True)
    print("rows", len(grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
