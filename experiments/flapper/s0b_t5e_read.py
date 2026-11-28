"""Read rows of results/S0b_probes7.jsonl: `python s0b_t5e_read.py <id prefix> [...]` prints one block per mission row."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rows(path=None):
    return [json.loads(x) for x in open(path or HERE / "results" / "S0b_probes7.jsonl")]


def surf_metrics(r):
    """Per surface phase: the largest |roll|, |pitch| from the first dry main-wing strip to 2 s after afloat (the phase's end)."""
    a = np.array([x[:14] for x in r["trace"]], dtype=float)
    out = []
    for p in r["phases_out"]:
        if p["kind"] != "surface":
            continue
        s = a[(a[:, 0] >= p["t0"]) & (a[:, 0] <= p["t1"] + 2.0)]
        dry = s[s[:, 10] < 32]
        if len(dry) == 0:
            out.append(dict(reached=False, z_end=p["com1"][2]))
            continue
        w = s[s[:, 0] >= dry[0, 0]]
        vz = np.abs(s[:, 4])
        out.append(dict(reached=not p.get("surface_timeout", False), t_first_dry=float(dry[0, 0] - p["t0"]), dur=float(p["t1"] - p["t0"]),
                        peak_roll=float(np.abs(w[:, 5]).max()), peak_pitch=float(np.abs(w[:, 6]).max()), vz_max=float(s[:, 4].max()),
                        afloat_att=p.get("afloat_att"), fill_end=p["fill1"]))
    return out


def brief(r):
    o = dict(id=r["id"], abort=r["abort"] and r["abort"]["why"], lim=r["clamped_steps"], qacc=r["bad_qacc"], ne=r["not_evidence"],
             x2=r.get("x2_share") and round(r["x2_share"], 2), tq={k: round(v) for k, v in r["tq_peak"].items()},
             contact_wing=round(r.get("contact_peak_W_wing", 0.0), 2), wall=r.get("wall_s"))
    ph = []
    for p in r["phases_out"]:
        e = dict(k=p["kind"], t=round(p["t1"] - p["t0"], 1), z=round(p["com1"][2], 2), att=[round(v, 1) for v in p["att1"]], fill=round(p["fill1"], 2),
                 wet=p["wet_main_max"], cl=round(min(p.get("clear_min", 9), p.get("clear_hull_min", 9)), 3))
        if p["kind"] == "fly":
            e.update(sub=p.get("sub_end"), hold=p.get("hold_z_rel") and [round(v, 2) for v in p["hold_z_rel"]], drift=p.get("hold_drift_m") and round(p["hold_drift_m"], 2),
                     tilt=p.get("hold_tilt_max_deg") and round(p["hold_tilt_max_deg"], 1), sat=round(p.get("law_sat", 0), 2), clip=round(p.get("law_clip", 0), 3),
                     rng=p.get("law_rng") and {k: [round(v, 2) for v in vv] for k, vv in p["law_rng"].items()})
        if "v_mean_along_cmd" in p:
            T = p["swim_trace"][-1][0]
            e.update(v_whole=round(p["disp_along_cmd"] / max(T, 1e-9), 3), v_last=round(p["v_mean_along_cmd"], 3), ey=round(p["yaw_err_end_deg"], 1),
                     zr=[round(v, 2) for v in p["z_range"]], turn_s=p.get("turn_done_s") and round(p["turn_done_s"], 1))
        ph.append(e)
    o["phases"] = ph
    sm = surf_metrics(r)
    if sm:
        o["surface"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()} for m in sm]
    return o


if __name__ == "__main__":
    for r in rows():
        if r.get("kind") == "mission" and any(r["id"].startswith(p) for p in sys.argv[1:]):
            b = brief(r)
            print(json.dumps({k: v for k, v in b.items() if k not in ("phases", "surface")}))
            for p in b["phases"]:
                print("   ", json.dumps(p))
            for m in b.get("surface", []):
                print("    surface:", json.dumps(m, default=lambda v: [round(x, 1) for x in v]))


def kick_env(r):
    """Largest |roll| and |pitch| (beat-smoothed over 0.22 s) in 2 s windows after the hold starts (the kick instant)."""
    a = np.array([x[:14] for x in r["trace"]], dtype=float)
    p = [q for q in r["phases_out"] if q["kind"] == "fly"][0]
    t0 = p["t0"] + p.get("at_height_s", 0.0)
    out = []
    k = 11
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
    rr, pp = sm(a[:, 5]), sm(a[:, 6])
    for i in range(5):
        s = (a[:, 0] >= t0 + 2 * i) & (a[:, 0] < t0 + 2 * i + 2)
        if s.any():
            out.append((round(float(np.abs(rr[s]).max()), 1), round(float(np.abs(pp[s]).max()), 1)))
    return out
