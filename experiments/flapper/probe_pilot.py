"""Probe for pass 1o: the height and speed loop on three of pass 1m's cruise
points (the centre-line fixture), tail 0.06 m2 with pitch feedback.  Not a
measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_pilot.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def job(a):
    from freerig import run_free
    from l1a import BODY
    name, pt, pilot, tail = a
    V, th, f, Phi, Psi, psi0, h, law = pt
    o = run_free(dict(BODY, stab_length=0.25, stab_chord=0.12, stab_boom=0.3), V, th, f, Phi, Psi, psi0,
                 feather_axis=h, cop_law=law, free="xzp", beats=40, hull_inertia=0.012,
                 tail=dict(kp=1.0, kd=0.1, **tail), pilot=pilot)
    return "%-26s V %.0f f %.1f | dz %+.2f..%+.2f (late %+.2f..%+.2f) vz late %+.2f vx late %.2f..%.2f | P %.1f (late %.1f) crank %.1f | f late %s pitch asked late %s got %+.1f | swing %.1f tail %.0f..%.0f" % (
        name, V, f, o["dz_min"], o["dz_max"], o["dz_late_min"], o["dz_late_max"], o["vz_late"],
        o["vx_late_min"], o["vx_late_max"], o["power_W"], o["power_late_W"], o["crank_eq_peak"],
        "%.2f" % o["f_late"] if pilot is not None else "-",
        "%+.1f" % o["pitch_ask_late_deg"] if pilot is not None else "-", o["pitch_late_deg"],
        o["pitch_swing_late_deg"], o["tail_min_deg"], o["tail_max_deg"])


def main():
    rows = [json.loads(x) for x in open(HERE / "results" / "L1m_tail.jsonl")]
    t2 = sorted((r for r in rows if r["arm"] == "T2"), key=lambda r: r["dz_min"])
    pts = [tuple(t2[i]["point"]) for i in (0, len(t2) // 2, len(t2) - 1)]
    grid = []
    for pt in pts:
        grid.append(("pilot, P tail", pt, {}, {}))
        grid.append(("pilot, PI tail ki 2", pt, {}, dict(ki=2.0)))
        grid.append(("pilot lim 12, PI ki 2", pt, dict(pitch_lim_deg=12.0), dict(ki=2.0)))
        grid.append(("pilot kh 6 kvz 6, PI ki 2", pt, dict(kh=6.0, kvz=6.0, pitch_lim_deg=12.0), dict(ki=2.0)))
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(row)


if __name__ == "__main__":
    main()
