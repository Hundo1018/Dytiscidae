"""L1 pass 1n: the gates again on the fixture with its shoulders on the hull's
sides (PREREG_L1n_refit.md).  Pass 1i's air grid, pass S1's water arms, then
pass 1k's held and x-z-released arms on the new gate points.  Every job runs
the earlier pass's own code with `l1a.BODY` replaced by `l1a.BODY_S`.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1n_refit.py OUTDIR
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def job(a):
    import l1a
    l1a.BODY = l1a.BODY_S
    which, arg = a
    if which == "air":
        from l1i_cam import job as f
    elif which == "water":
        from s1_entrain import job as f
    else:
        from l1k_free import job as f
    r = f(arg)
    r["seated"] = True
    return r


def gate_points(rows):
    W = rows[0]["real_weight"]
    to = [r for r in rows if r["kind"] == "rest" and r["Fz"] / W >= 1.6 and r["crank_eq_peak"] <= 12
          and r["aero_power_W"] <= 150]
    cr = [r for r in rows if r["kind"] == "cruise" and r["sustains"] and r["aero_power_W"] <= 41
          and r["crank_eq_peak"] <= 12]
    key = ("V", "theta_deg", "f", "Phi_deg", "Psi_deg", "psi0_deg", "axis", "law")
    return [tuple(r[k] for k in key) for r in to], [tuple(r[k] for k in key) for r in cr]


def save(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(path.name, len(rows), flush=True)


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    AX, LAW = (0.0, 0.1, 0.2), ("plate", "insect")
    air = [("rest", th, f, Psi, h, z) for th in (45.0, 60.0, 75.0, 90.0)
           for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)
           for h in AX for z in LAW]
    air += [("cruise", V, f, Phi, Psi, h, z) for V in (6.0, 8.0) for f in (2.5, 3.0, 3.5, 4.0)
            for Phi in (40.0, 60.0, 85.0) for Psi in (0.0, 10.0, 20.0, 30.0)
            for h in AX for z in LAW]
    pts = [(ks, Phi, f, Psi, law, h) for ks in (0.4, 0.6) for Phi in (40.0, 85.0)
           for f in (0.25, 0.5, 0.75, 1.0) for Psi in (20.0, 30.0, 50.0, 60.0)
           for law, h in ((None, 0.0), ("plate", 0.1), ("insect", 0.1))]
    rest = [(ks, 40.0, 0.0, 0.0, None, 0.0) for ks in (0.4, 0.6)]
    water = [(arm, *p, None) for arm in ("held", "heldref") for p in rest + pts]
    water += [("free", *p, None) for p in pts]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(job, [("air", a) for a in air], chunksize=1)
        save(out / "L1n_air.jsonl", rows)
        to, cr = gate_points(rows)
        free = [(k, pt, fr, 0.0) for k, ps, frs in (("takeoff", to, ("",)), ("cruise", cr, ("", "xz")))
                for pt in ps for fr in frs]
        save(out / "L1n_free.jsonl", pool.map(job, [("free", a) for a in free], chunksize=1))
        save(out / "L1n_water.jsonl", pool.map(job, [("water", a) for a in water], chunksize=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
