"""Pass M2: the gates and checks of PREREG_M2_water.md section 3-4, read from
results/M2_rows.jsonl (last row per (arm, seed, half) wins).

    m2_read.py [rows file]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "results" / "M2_rows.jsonl"
THREE_CELLS_W = 462.0


def load(path=ROWS):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "row" or r.get("extra"):
            continue
        rows[(r["arm"], r["seed"], r["half"])] = r
    return rows


def arr(tr, key):
    return np.array([x[key] for x in tr], float)


def window(tr, a, b):
    t = arr(tr, "t")
    return [x for x, tt in zip(tr, t) if a - 1e-6 <= tt <= b + 1e-6]


def at(tr, t0):
    t = arr(tr, "t")
    return tr[int(np.argmin(np.abs(t - t0)))]


def tumbling(tr, a, b):
    w = window(tr, a, b)
    dz = min(x["dors"][2] for x in w)
    om = float(np.mean([np.linalg.norm(x["w"]) for x in w]))
    return dict(dors_min=dz, omega_mean=om, ok=bool(dz >= -0.25 and om <= 1.0))


def progress(tr, h, a=15.0, b=40.0):
    p0, p1 = np.array(at(tr, a)["com"]), np.array(at(tr, b)["com"])
    d = p1 - p0
    return float((d[0] * math.cos(h) + d[1] * math.sin(h)) / (b - a))


def half_gw(r):
    tr = r["trace"]
    tgt = r["cmd"]["z"] if "z" in r["cmd"] else None
    out = dict(prog=progress(tr, r["cmd"]["h"]), tumb=tumbling(tr, 15.0, 40.0))
    if tgt is not None:
        w = window(tr, 35.0, 40.0)
        dm = float(np.mean([-x["com"][2] for x in w]))
        out.update(depth_end=dm, target=tgt, err=dm - tgt,
                   min_depth=float(min(-x["com"][2] for x in window(tr, 15.0, 40.0))))
    else:
        out.update(max_depth=float(max(-x["com"][2] for x in window(tr, 15.0, 40.0))))
    return out


def gate_gw(rA, rB, delta):
    """GW for one seed; rA told deeper (h), rB shallower (h + pi)."""
    a, b = half_gw(rA), half_gw(rB)
    hB = rB["cmd"]["h"]
    if rB is rA or rB.get("_as_A"):
        # one rollout for both halves: half B's progress is along h + pi, its depth told 3 - delta
        b = dict(a)
        b["prog"] = progress(rA["trace"], hB)
        b["target"] = 3.0 - delta
        b["err"] = a["depth_end"] - b["target"]
    P = 0.5 * (a["prog"] + b["prog"])
    R = 0.5 * ((a["depth_end"] - 3.0) / delta + (3.0 - b["depth_end"]) / delta)
    held = abs(a["err"]) <= 0.5 and abs(b["err"]) <= 0.5 and a["min_depth"] >= 0.3 and b["min_depth"] >= 0.3
    c = dict(progress=P >= 0.05 and a["prog"] > 0 and b["prog"] > 0, response=R >= 0.5, held=held,
             upright=a["tumb"]["ok"] and b["tumb"]["ok"])
    return dict(P=P, R=R, A=a, B=b, clauses=c, met=all(c.values()))


def gate_gs(rA, rB):
    a, b = half_gw(rA), half_gw(rB)
    if rB is rA or rB.get("_as_A"):
        b = dict(a)
        b["prog"] = progress(rA["trace"], rB["cmd"]["h"])
    P = 0.5 * (a["prog"] + b["prog"])
    c = dict(progress=P >= 0.05 and a["prog"] > 0 and b["prog"] > 0,
             surface=a["max_depth"] <= 0.5 and b["max_depth"] <= 0.5,
             upright=a["tumb"]["ok"] and b["tumb"]["ok"])
    return dict(P=P, A=a, B=b, clauses=c, met=all(c.values()))


def gate_gx(r):
    tr = r["trace"]
    t = arr(tr, "t")
    z = np.array([x["com"][2] for x in tr])
    wet = arr(tr, "wet")
    n3 = int(round(3.0 / 0.1))
    best = None
    met = False
    for i in range(len(tr) - n3):
        if np.all(wet[i:i + n3 + 1] == 0):
            j, j2 = i + n3, i + int(round(2.0 / 0.1))
            ok = z[j] >= 0.5 and z[j] >= z[i] and (z[j] - z[j2]) >= 0.0
            if ok:
                met = True
                best = float(t[i])
                break
    # longest clear run
    run, longest = 0, 0
    for w_ in wet:
        run = run + 1 if w_ == 0 else 0
        longest = max(longest, run)
    return dict(met=met, t_star=best, clear_longest_s=round(longest * 0.1, 1), z_max=float(z.max()),
                wet_min=int(wet.min()))


def main(path=ROWS):
    R = load(path)
    seeds = sorted({k[1] for k in R})
    print("rows:", len(R))
    qx = [(k, r["summary"]["bad_qacc"]) for k, r in R.items() if r["summary"]["bad_qacc"] > 0]
    print("rows with a QACC reset (excluded, counted not met):", qx)
    out = {}
    for arm in ("WV", "WK", "W0", "WF", "VB", "VB-W0"):
        res = []
        for s in seeds:
            if (arm, s, "A") not in R:
                continue
            rA = R[(arm, s, "A")]
            rB = R.get((arm, s, "B"), dict(rA, _as_A=True, cmd=dict(h=rA["cmd"]["h"] + math.pi, z=3.0 - rA["draw"]["delta"])))
            g = gate_gw(rA, rB, rA["draw"]["delta"])
            q = rA["summary"]["bad_qacc"] + rB["summary"]["bad_qacc"]
            g["met"] = g["met"] and q == 0
            res.append((s, g, rA, rB))
        if not res:
            continue
        out[arm] = res
        print("\nGW %s: met %d of %d" % (arm, sum(g["met"] for _, g, _, _ in res), len(res)))
        for s, g, rA, rB in res:
            print("  seed %d P %+.3f m/s (A %+.3f B %+.3f) R %+.2f  depth A %.2f/%.2f B %.2f/%.2f  min %.2f/%.2f  dors_min %.2f/%.2f  |w| %.2f/%.2f  %s  P_el %.0f/%.0f W peak_all %.0f W"
                  % (s, g["P"], g["A"]["prog"], g["B"]["prog"], g["R"], g["A"]["depth_end"], g["A"]["target"],
                     g["B"]["depth_end"], g["B"]["target"], g["A"]["min_depth"], g["B"]["min_depth"],
                     g["A"]["tumb"]["dors_min"], g["B"]["tumb"]["dors_min"], g["A"]["tumb"]["omega_mean"],
                     g["B"]["tumb"]["omega_mean"], "".join("+" if v else "-" for v in g["clauses"].values()),
                     rA["summary"]["P_elec_mean_W"], rB["summary"]["P_elec_mean_W"],
                     max(rA["summary"]["P_peak_50ms_all_W"], rB["summary"]["P_peak_50ms_all_W"])))
    for arm in ("WS", "S0", "SF"):
        res = []
        for s in seeds:
            if (arm, s, "A") not in R:
                continue
            rA = R[(arm, s, "A")]
            rB = R.get((arm, s, "B"), dict(rA, _as_A=True, cmd=dict(h=rA["cmd"]["h"] + math.pi)))
            g = gate_gs(rA, rB)
            g["met"] = g["met"] and rA["summary"]["bad_qacc"] + rB["summary"]["bad_qacc"] == 0
            res.append((s, g, rA, rB))
        if not res:
            continue
        print("\nGS %s: met %d of %d" % (arm, sum(g["met"] for _, g, _, _ in res), len(res)))
        for s, g, rA, rB in res:
            print("  seed %d P %+.3f m/s (A %+.3f B %+.3f) max depth %.2f/%.2f dors_min %.2f/%.2f |w| %.2f/%.2f %s P_el %.0f/%.0f W peak_all %.0f W"
                  % (s, g["P"], g["A"]["prog"], g["B"]["prog"], g["A"]["max_depth"], g["B"]["max_depth"],
                     g["A"]["tumb"]["dors_min"], g["B"]["tumb"]["dors_min"], g["A"]["tumb"]["omega_mean"],
                     g["B"]["tumb"]["omega_mean"], "".join("+" if v else "-" for v in g["clauses"].values()),
                     rA["summary"]["P_elec_mean_W"], rB["summary"]["P_elec_mean_W"],
                     max(rA["summary"]["P_peak_50ms_all_W"], rB["summary"]["P_peak_50ms_all_W"])))
    for arm in ("XB", "XD", "XV", "XDV", "X0f", "X0d", "X1f", "X1d"):
        res = [(s, gate_gx(R[(arm, s, "X")]), R[(arm, s, "X")]) for s in seeds if (arm, s, "X") in R]
        if not res:
            continue
        for s, g, r in res:
            g["met"] = g["met"] and r["summary"]["bad_qacc"] == 0
        print("\nGX %s: met %d of %d" % (arm, sum(g["met"] for _, g, _ in res), len(res)))
        for s, g, r in res:
            sm = r["summary"]
            print("  seed %d met %s clear_longest %.1f s z_max %.2f m wet_min %d  P_el %.0f W peak_all %.0f W (%s three cells) lim %s log %s"
                  % (s, g["met"], g["clear_longest_s"], g["z_max"], g["wet_min"], sm["P_elec_mean_W"],
                     sm["P_peak_50ms_all_W"], "over" if sm["P_peak_50ms_all_W"] > THREE_CELLS_W else "under",
                     {k: round(v, 2) for k, v in sm["at_limit"].items()}, r.get("exit_log")))


def flights(path=ROWS):
    """The exit flights (RESULT table 2): first clear, the longest clear run,
    height, the climb over its first 2 s, mean electrical power over it, and
    the energy spent from the arm's start to the first clear sample."""
    R = load(path)
    for arm in ("XV", "XDV"):
        for s in range(6):
            if (arm, s, "X") not in R:
                continue
            tr = R[(arm, s, "X")]["trace"]
            t = arr(tr, "t")
            wet = arr(tr, "wet")
            E = arr(tr, "E")
            z = np.array([x["com"][2] for x in tr])
            if not (wet == 0).any():
                print(arm, s, "never clear")
                continue
            best, run = (0, 0, 0), None
            for i, w_ in enumerate(wet):
                if w_ == 0:
                    run = i if run is None else run
                    if i - run > best[0]:
                        best = (i - run, run, i)
                else:
                    run = None
            _, i0, i1 = best
            f = int(np.argmax(wet == 0))
            print("%s %d first clear %.1f s (energy to there %.0f J); clear %.1f-%.1f s; max height %.2f m; "
                  "climb first 2 s %.2f m/s; flight power %.0f W" % (
                      arm, s, t[f], E[f] - E[0], t[i0], t[i1], z.max(), (z[min(i0 + 20, i1)] - z[i0]) / 2,
                      (E[i1] - E[i0]) / max(t[i1] - t[i0], 1e-9)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "flights":
        flights()
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROWS)
