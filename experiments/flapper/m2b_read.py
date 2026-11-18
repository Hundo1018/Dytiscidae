"""Pass M2b: the gates per entrainment mode and the co-primary verdict
(PREREG_M2b_water.md section 3), from results/M2b_rows.jsonl.

    m2b_read.py              # every gate, both modes, and the verdict
    m2b_read.py trials       # GX30 / GX on the stay-up tuning trials
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m2_read import arr, gate_gs, gate_gw, gate_gx  # noqa: E402

ROWS = HERE / "results" / "M2b_rows.jsonl"
THREE_CELLS_W = 462.0


def mode_name(m):
    return "wetting" if m == "wetting" else "default"


def gate_gx30(r):
    tr = r["trace"]
    t = arr(tr, "t")
    wet = arr(tr, "wet")
    z = np.array([x["com"][2] for x in tr])
    nose = np.array([x["nose"] for x in tr])
    dors = np.array([x["dors"] for x in tr])
    side = np.cross(dors, nose)
    ne = np.degrees(np.arcsin(np.clip(nose[:, 2], -1, 1)))
    se = np.degrees(np.arcsin(np.clip(side[:, 2], -1, 1)))
    wv = np.array([x["w"] for x in tr])        # body rate vector; M1's G4 reads the norm of its beat mean
    out = dict(met=False, t_star=None, clear_s=0.0, why=None)
    first = np.nonzero((wet == 0) & (t <= 5.0 + 1e-9))[0]
    # longest dry run from the first dry sample, for the report
    if len(first) == 0:
        out["why"] = "never clear by 5 s"
        return out
    i0 = int(first[0])
    ts = t[i0]
    out["t_star"] = float(ts)
    j = i0
    while j < len(tr) and wet[j] == 0:
        j += 1
    out["clear_s"] = round(float(t[j - 1] - ts), 1)
    end = ts + 30.0
    if t[-1] < end - 1e-6:
        out["why"] = "row too short"
        return out
    win = (t >= ts - 1e-9) & (t <= end + 1e-9)
    if not np.all(wet[win] == 0):
        out["why"] = "wet again at %.1f s" % float(t[win][np.argmax(wet[win] > 0)])
        return out
    hw = (t >= ts + 2.0 - 1e-9) & (t <= end + 1e-9)
    if np.any(z[hw] < 1.0):
        out["why"] = "below 1 m at %.1f s" % float(t[hw][np.argmax(z[hw] < 1.0)])
        return out

    def means(x, a, b):
        sel = np.nonzero((t >= a - 1e-9) & (t <= b + 1e-9))[0]
        return np.array([x[sel[k:k + 5]].mean() for k in range(0, len(sel) - 4)])
    nm, sm = means(ne, ts + 1.0, end), means(se, ts + 1.0, end)
    if nm.min() < -10.0 or np.abs(sm).max() > 35.0:
        out["why"] = "attitude: nose %.0f, span %.0f deg" % (nm.min(), np.abs(sm).max())
        return out
    sel = np.nonzero((t >= end - 3.0 - 1e-9) & (t <= end + 1e-9))[0]
    rate = max(float(np.linalg.norm(wv[sel[k:k + 5]].mean(0))) for k in range(0, len(sel) - 4))
    if rate > 1.5:
        out["why"] = "rate over 1.5 rad/s in the last 3 s"
        return out
    out["met"] = True
    return out


def load(path=ROWS):
    R = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "row" or r.get("extra"):
            continue
        R[(r["arm"], r["seed"], r["half"], mode_name(r["mode"]))] = r
    return R


def both_halves(R, arm, s, m, gs=False):
    rA = R[(arm, s, "A", m)]
    if (arm, s, "B", m) in R:
        return rA, R[(arm, s, "B", m)]
    cmd = dict(h=rA["cmd"]["h"] + math.pi) if gs else dict(h=rA["cmd"]["h"] + math.pi, z=3.0 - rA["draw"]["delta"])
    return rA, dict(rA, _as_A=True, cmd=cmd)


def verdict(counts, bar):
    a, b = counts["wetting"], counts["default"]
    if a >= bar and b >= bar:
        return "met in both modes"
    if a < bar and b < bar:
        return "not met in both modes"
    return "not established, the modes disagree"


def main(path=ROWS):
    R = load(path)
    seeds = sorted({k[1] for k in R})
    print("rows:", len(R), "QACC:", [k for k, r in R.items() if r["summary"]["bad_qacc"]])
    res = {}
    for arm in ("XS", "X0f"):            # XSt = XS (no tuned setting was better; PREREG_M2b section 2)
        cnt = {}
        for m in ("wetting", "default"):
            n = 0
            for s in seeds:
                if (arm, s, "X", m) not in R:
                    continue
                r = R[(arm, s, "X", m)]
                g, g3 = gate_gx30(r), gate_gx(r)
                ok = g["met"] and r["summary"]["bad_qacc"] == 0
                n += ok
                sm = r["summary"]
                print("GX30 %s %s seed %d: met %s (GX3 %s) t* %s clear %.1f s z_max %.2f %s | P %.0f W peak %.0f W lim %s clamp %d" % (
                    arm, m, s, ok, g3["met"], g["t_star"], g["clear_s"], g3["z_max"], g["why"] or "",
                    sm["P_elec_mean_W"], sm["P_peak_50ms_all_W"], {k: round(v, 2) for k, v in sm["at_limit"].items()},
                    sm["clamped_steps"]))
            cnt[m] = n
        res[arm] = cnt
        print("GX30 %s: wetting %d, default %d -> %s\n" % (arm, cnt["wetting"], cnt["default"], verdict(cnt, 4)))
    for arm, gs in (("WT", False), ("WT-VB", False), ("W0", False), ("VB-W0", False), ("WTF", False), ("WTF-VB", False),
                    ("ST", True), ("S0", True), ("SF", True)):
        cnt = {}
        for m in ("wetting", "default"):
            n = 0
            for s in seeds:
                if (arm, s, "A", m) not in R:
                    continue
                rA, rB = both_halves(R, arm, s, m, gs)
                g = gate_gs(rA, rB) if gs else gate_gw(rA, rB, rA["draw"]["delta"])
                ok = g["met"] and rA["summary"]["bad_qacc"] + rB["summary"]["bad_qacc"] == 0
                n += ok
                extra = "" if gs else " R %+.2f depth A %.2f/%.2f B %.2f/%.2f" % (
                    g["R"], g["A"]["depth_end"], g["A"]["target"], g["B"]["depth_end"], g["B"]["target"])
                print("%s %s %s seed %d: met %s P %+.3f (A %+.3f B %+.3f)%s clauses %s | P_el %.0f/%.0f W peak %.0f W" % (
                    "GS" if gs else "GW", arm, m, s, ok, g["P"], g["A"]["prog"], g["B"]["prog"], extra,
                    "".join("+" if v else "-" for v in g["clauses"].values()), rA["summary"]["P_elec_mean_W"],
                    rB["summary"]["P_elec_mean_W"], max(rA["summary"]["P_peak_50ms_all_W"], rB["summary"]["P_peak_50ms_all_W"])))
            cnt[m] = n
        res[arm] = cnt
        print("%s %s: wetting %d, default %d -> %s\n" % ("GS" if gs else "GW", arm, cnt["wetting"], cnt["default"], verdict(cnt, 5)))
    return res


def trials():
    for line in open(HERE / "results" / "M2b_trials.jsonl"):
        r = json.loads(line)
        if r["spec"].get("law") == "stay" and r["spec"].get("keep_every") == 1:
            print(r["spec"]["note"][:50], r["spec"].get("entrainment", "wetting"), gate_gx30(r), gate_gx(r))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trials":
        trials()
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROWS)
