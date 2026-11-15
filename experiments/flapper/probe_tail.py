"""Probe for pass 1m: an all-moving tail on a boom, on pass 1k's cheapest
released cruise point.  Held: the pitching moment about the centre of mass
against tail incidence and body pitch.  Released: 12 beats with the tail
fixed and under feedback.  Not a measurement.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/probe_tail.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAIL = dict(stab_length=0.25, stab_chord=0.12, stab_boom=0.3)


def point():
    rows = [json.loads(x) for x in open(HERE / "results" / "L1k_free.jsonl")]
    ok = [r for r in rows if r["kind"] == "cruise" and r["free"] == "xz"
          and abs(r["lift_over_weight"] - 1) <= 0.01 and r["Fx"] >= 0 and r["power_W"] <= 41]
    return min(ok, key=lambda r: r["power_W"])


def job(a):
    from freerig import run_free
    from l1a import BODY
    name, r, tailed, kw = a
    body = dict(BODY, **TAIL) if tailed else BODY
    th = kw.pop("theta", 0.0)
    o = run_free(body, r["V"], th, r["f"], r["Phi_deg"], r["Psi_deg"], r["psi0_deg"],
                 feather_axis=r["axis"], cop_law=r["law"], **kw)
    if "pitch_end_deg" in o:
        return "%-34s pitch beat means %s; swing %.1f deg; dz %+.2f m, vx %.2f, vz %+.2f; P %.1f W; tail %.1f..%.1f deg" % (
            name, " ".join("%+.0f" % x for x in o["pitch_beat_mean_deg"]), o["pitch_swing_deg"], o["dz"],
            o["vx_end"], o["vz_end"], o["power_W"], o.get("tail_min_deg", 0), o.get("tail_max_deg", 0))
    return "%-34s M_cg mean %+.3f swing %.2f N m; L/W %.3f Fx %+.2f; P %.1f W" % (
        name, o["pitch_moment_cg_mean"], o["pitch_moment_cg_swing"], o["lift_over_weight"], o["Fx"],
        o["power_W"])


def main():
    r = point()
    print({k: r[k] for k in ("V", "f", "Phi_deg", "Psi_deg", "psi0_deg", "axis", "law", "power_W")})
    grid = [("held no tail th %+d" % th, r, False, dict(theta=float(th))) for th in (-5, 0, 5)]
    grid += [("held tail i0 %+d th %+d" % (i0, th), r, True, dict(theta=float(th), tail=dict(i0_deg=float(i0))))
             for i0, th in ((-4, 0), (0, 0), (4, 0), (0, -5), (0, 5))]
    for I in (None, 0.012, 0.03):
        grid.append(("free no tail I %s" % I, r, False, dict(free="xzp", beats=12, hull_inertia=I)))
        for kp, kd in ((0.0, 0.0), (1.0, 0.1), (2.0, 0.2)):
            grid.append(("free tail I %s kp %.0f kd %.1f" % (I, kp, kd), r, True,
                         dict(free="xzp", beats=12, hull_inertia=I, tail=dict(i0_deg=0.0, kp=kp, kd=kd))))
    with mp.get_context("spawn").Pool(6) as pool:
        for row in pool.map(job, grid, chunksize=1):
            print(row)


if __name__ == "__main__":
    main()
