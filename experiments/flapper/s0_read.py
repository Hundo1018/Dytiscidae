"""Pass S0: the frozen gates read on the S0 rows (PREREG_S0 section 3; results/S0_plan.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/s0_read.py [ROWS.jsonl] [RUNG]

Gates are imported from the M series, not rewritten: `m4_read.cruise_half`,
`m4_read.entry_half`, `m3_read.walk_pairs`, `m3b_read.turn_pairs`,
`m2_read.gate_gw` / `gate_gs` / `half_gw`, `m2b_read.gate_gx30`,
`m1_takeoff.passed`.  Two leg reads are the M2 depth clauses applied to the
dive and surface legs (`dive_clause`, `surface_clause`); the stand clause is
`m3b_read.main`'s, copied (it is inline there).  Added: section 4's flags.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TR = HERE / "results" / "S0_traces"
ROWS = HERE / "results" / "S0_rows.jsonl"
BAND = (2.5, 6.0)


def load(path=ROWS, rung=None):
    R = {}
    for x in open(path):
        if not x.strip():
            continue
        r = json.loads(x)
        if r.get("kind") == "hash" or "error" in r:
            continue
        if rung and r["spec"]["rung"] != rung:
            continue
        R[r["id"]] = r
    return R


def trace(rid):
    z = np.load(TR / f"{rid}.npz")
    return {k: z[k] for k in z.files}


def flags(r, freq=None):
    """Section 4: contact over 10 weights, X2 share over 0.5, frequency outside the band."""
    out = dict(contact_W=round(r["peaks"]["contact_max_W"], 1))
    out["rigid"] = r["peaks"]["contact_max_W"] > 10.0
    x2 = r.get("x2") or (r.get("again") or {}).get("x2")
    if x2:
        out["x2"] = round(x2["share"], 2)
        out["X2"] = x2["share"] > 0.5
    if freq is not None:
        out["f_Hz"] = freq
        out["band"] = BAND[0] <= freq <= BAND[1]
    return out


# --------------------------------------------------------------- air (M4 H, T)
def air(R, rung, tag="air"):
    from m4_read import cruise_half
    res = {}
    starts = sorted({r["spec"]["half"]["start_name"] for r in R.values() if r["spec"]["scenario"] == tag
                     and r["spec"]["leg"] == "cruise" and r["spec"]["rung"] == rung})
    for s in starts:
        ids = {h: f"{rung}_{tag}_{s}_L_{h}" for h in ("H", "Tp", "Tm")}
        nid = f"{rung}_{tag}_{s}_N_all"
        psi0 = R[nid]["spec"]["half"]["start"][1] if nid in R else None
        out = {}
        if ids["H"] in R:
            ok, q = cruise_half(R[ids["H"]]["result"], trace(ids["H"]), psi0 if psi0 is not None else
                                R[ids["H"]]["spec"]["half"]["cmd"], "H")
            out["H"] = ok
            out["Hq"] = {k: q[k] for k in ("trk_err_max", "roll_max_hold", "dz_hold", "V_hold")}
            out["flags_H"] = flags(R[ids["H"]], 3.5)
        if all(i in R for i in ids.values()):
            p0 = R[ids["H"]]["spec"]["half"]["cmd"]
            a_ok, aq = cruise_half(R[ids["Tp"]]["result"], trace(ids["Tp"]), p0 + 90.0, "T")
            b_ok, bq = cruise_half(R[ids["Tm"]]["result"], trace(ids["Tm"]), p0 - 90.0, "T")
            P_t = 0.5 * (aq["dpsi"] - bq["dpsi"])
            hashes = len({R[i]["result"]["hash"] for i in ids.values()}) == 1
            out["T"] = bool(a_ok and b_ok and P_t >= 60.0 and hashes)
            out["P_t"] = P_t
        if nid in R:
            p0 = R[nid]["spec"]["half"]["cmd"]
            T = trace(nid)
            nh, _ = cruise_half(R[nid]["result"], T, p0, "H")
            na, aq = cruise_half(R[nid]["result"], T, p0 + 90.0, "T")
            nb, bq = cruise_half(R[nid]["result"], T, p0 - 90.0, "T")
            out["null_H"] = nh
            out["null_T"] = bool(na and nb and 0.5 * (aq["dpsi"] - bq["dpsi"]) >= 60.0)
        res[s] = out
    return res


# --------------------------------------------------------------- land (M3, M3b)
def stand_clause(r):
    """m3b_read.main's stand clause (copied: it is inline there)."""
    why = []
    if r["bad_qacc"]:
        why.append("qacc")
    if r["share"]["body"] > 0.02 or r["body_contact_frac"] > 0.05:
        why.append("body %.3f/%.3f" % (r["share"]["body"], r["body_contact_frac"]))
    if r["drift_m"] > 0.02:
        why.append("drift %.3f" % r["drift_m"])
    if abs(r["dyaw_deg"]) > 2:
        why.append("yaw %.2f" % r["dyaw_deg"])
    return not why, why


def land(R, rung, tag="land"):
    import m3b_read
    from m3_read import walk_pairs
    m3b_read.TRACES = TR / "land"
    rows = [r for r in R.values() if r["spec"]["scenario"] == tag and r["spec"]["rung"] == rung and r["spec"]["kind"] == "land"]
    res = dict(walk={}, turn={}, stand={})
    W = 12.0

    def as_walk(r):
        x = r["result"]
        return dict(x, along=x["along_12"], across=x["across_12"], window_s=W)
    for arm in ("K", "U", "N0"):
        ws = [as_walk(r) for r in rows if r["spec"]["leg"] == "walk" and r["result"]["arm"] == arm]
        un = {}
        if arm == "K":
            for p in walk_pairs([as_walk(r) for r in rows if r["spec"]["leg"] == "walk" and r["result"]["arm"] == "U"]):
                if p[2] is not None:
                    un[p[0]] = p[2]
        for pk, ok, pm, why, hs in walk_pairs(ws, un if arm == "K" else None):
            res["walk"][(arm, pk)] = dict(ok=ok, pm=pm, why=why)
    for arm in ("K", "N0"):
        ts = [r["result"] for r in rows if r["spec"]["leg"] == "turn" and r["result"]["arm"] == arm]
        for pk, ok, halves, why in m3b_read.turn_pairs(ts, 14.0):
            res["turn"][(arm, pk)] = dict(ok=ok, why=why, dyaw=[h[1] for h in halves], err=[h[2] for h in halves])
    for r in rows:
        if r["spec"]["leg"] == "stand":
            ok, why = stand_clause(r["result"])
            res["stand"][(r["result"]["arm"], r["result"]["pair"])] = dict(ok=ok, why=why, P=r["result"]["P_window_W"])
    res["flags"] = {r["id"]: flags(r) for r in rows}
    return res


# --------------------------------------------------------------- water (M2, M2b)
def wrow(r):
    x = r["result"]
    return dict(trace=x["trace"], cmd=x["cmd"], summary=x["summary"], draw=x["draw"])


def dive_clause(rr, target=3.0):
    """M2's depth clauses (GW's 'held') on the dive leg: mean depth over 35-40 s within 0.5 m of the
    target, and depth >= 0.3 m at every sample of 15-40 s."""
    from m2_read import half_gw
    h = half_gw(rr)
    ok = abs(h["depth_end"] - target) <= 0.5 and h["min_depth"] >= 0.3
    return ok, dict(depth_end=h["depth_end"], min_depth=h["min_depth"], tumb=h["tumb"])


def surface_clause(rr):
    """M2/GS's surface clause on the surface leg: depth <= 0.5 m at every sample of 35-40 s."""
    from m2_read import window, tumbling
    w = window(rr["trace"], 35.0, 40.0)
    dmax = float(max(-x["com"][2] for x in w))
    return dmax <= 0.5, dict(max_depth_35_40=dmax, tumb=tumbling(rr["trace"], 15.0, 40.0))


def water(R, rung, tag="water"):
    from m2_read import gate_gw, gate_gs
    rows = {k: r for k, r in R.items() if r["spec"]["scenario"] == tag and r["spec"]["rung"] == rung}
    res = defaultdict(dict)
    seeds = sorted({r["spec"]["seed"] for r in rows.values()})
    for s in seeds:
        for m in ("wet", "def"):
            g = lambda leg: f"{rung}_{tag}_{leg}_s{s}_{m}"
            if g("D") in R:
                ok, q = dive_clause(wrow(R[g("D")]))
                res[("D", m)][s] = dict(ok=ok and R[g("D")]["bad_qacc"] == 0, **q)
            if g("D0") in R:
                ok, q = dive_clause(wrow(R[g("D0")]))
                res[("D0", m)][s] = dict(ok=ok, **q)
            if g("U") in R:
                ok, q = surface_clause(wrow(R[g("U")]))
                res[("U", m)][s] = dict(ok=ok and R[g("U")]["bad_qacc"] == 0, **q)
            if g("U0") in R:
                ok, q = surface_clause(wrow(R[g("U0")]))
                res[("U0", m)][s] = dict(ok=ok, **q)
            a, b = f"{rung}_{tag}_GW_s{s}_A_{m}", f"{rung}_{tag}_GW_s{s}_B_{m}"
            if a in R and b in R:
                rA, rB = wrow(R[a]), wrow(R[b])
                q = gate_gw(rA, rB, rA["draw"]["delta"])
                res[("GW", m)][s] = dict(ok=q["met"] and R[a]["bad_qacc"] + R[b]["bad_qacc"] == 0, P=q["P"], R=q["R"],
                                         clauses=q["clauses"], depth=[q["A"]["depth_end"], q["B"]["depth_end"]],
                                         target=[q["A"]["target"], q["B"]["target"]])
            if g("W0") in R:
                rA = wrow(R[g("W0")])
                rB = dict(rA, _as_A=True, cmd=dict(h=rA["cmd"]["h"] + math.pi, z=3.0 - rA["draw"]["delta"]))
                q = gate_gw(rA, rB, rA["draw"]["delta"])
                res[("W0", m)][s] = dict(ok=q["met"], P=q["P"], R=q["R"], clauses=q["clauses"])
            a, b = f"{rung}_{tag}_GS_s{s}_A_{m}", f"{rung}_{tag}_GS_s{s}_B_{m}"
            if a in R and b in R:
                q = gate_gs(wrow(R[a]), wrow(R[b]))
                res[("GS", m)][s] = dict(ok=q["met"] and R[a]["bad_qacc"] + R[b]["bad_qacc"] == 0, P=q["P"],
                                         clauses=q["clauses"], dors_min=[q["A"]["tumb"]["dors_min"], q["B"]["tumb"]["dors_min"]],
                                         max_depth=[q["A"]["max_depth"], q["B"]["max_depth"]])
            if g("S0") in R:
                rA = wrow(R[g("S0")])
                rB = dict(rA, _as_A=True, cmd=dict(h=rA["cmd"]["h"] + math.pi))
                q = gate_gs(rA, rB)
                res[("S0", m)][s] = dict(ok=q["met"], P=q["P"], clauses=q["clauses"])
    res["flags"] = {k: flags(r, (r["spec"]["water"]["law"].get("g") or {}).get("f", {"WT": 0.25, "ST": 0.3}.get(
        r["spec"]["water"]["law"].get("law"))) if r["spec"]["water"]["law"].get("law") != "still" else None)
        for k, r in rows.items()}
    return dict(res)


# --------------------------------------------------------------- crossings
def gx30(r_or_again):
    from m2b_read import gate_gx30
    return gate_gx30(dict(trace=r_or_again["trace"]))


def exit_leg(R, rung, tag="waterair"):
    res = defaultdict(dict)
    for k, r in R.items():
        sp = r["spec"]
        if sp["scenario"] != tag or sp["rung"] != rung or sp["leg"] != "exit":
            continue
        g = gx30(r["result"])
        m = "wet" if sp["mode"] == "wetting" else "def"
        res[(sp["arm"], m)][sp["seed"]] = dict(ok=g["met"] and r["bad_qacc"] == 0, why=g["why"], t_star=g["t_star"],
                                               clear_s=g["clear_s"], flags=flags(r, 4.5),
                                               at_limit=r["result"]["summary"]["at_limit"])
    return dict(res)


def entry_leg(R, rung, tag, medium):
    from m4_read import entry_half
    from m4_read import XE as M4XE
    near, far = M4XE[medium]
    res = {}
    groups = defaultdict(dict)
    for k, r in R.items():
        sp = r["spec"]
        if sp["scenario"] != tag or sp["rung"] != rung or sp["kind"] != "entry":
            continue
        m = "wet" if sp.get("mode", "wetting") == "wetting" else "def"
        groups[(sp["half"]["start_name"], m)][sp["half"]["half"] if sp["half"]["arm"] not in ("NW", "NL") else "null"] = r
    for (s, m), g in sorted(groups.items()):
        out = {}
        if "near" in g and "far" in g:
            a_ok, aq = entry_half(g["near"]["result"], trace(g["near"]["id"]), near, medium)
            b_ok, bq = entry_half(g["far"]["result"], trace(g["far"]["id"]), far, medium)
            hashes = g["near"]["result"]["hash"] == g["far"]["result"]["hash"]
            out.update(ok=bool(a_ok and b_ok and hashes), A=_slim(aq), B=_slim(bq), hash_eq=hashes,
                       P_e=(bq["x"] - aq["x"]) / (far - near) if "x" in aq and "x" in bq else None,
                       flags=[flags(g["near"]), flags(g["far"])])
            out["again"] = [_again(g[h], medium) for h in ("near", "far")]
        if "null" in g:
            n_ok_a, na = entry_half(g["null"]["result"], trace(g["null"]["id"]), near, medium)
            n_ok_b, nb = entry_half(g["null"]["result"], trace(g["null"]["id"]), far, medium)
            out["null"] = bool(n_ok_a and n_ok_b)
            out["null_q"] = _slim(na)
            out["null_again"] = _again(g["null"], medium)
        res[(s, m)] = out
    return res


def _slim(q):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items() if k not in ("loads", "loads_10ms", "win")}


def _again(r, medium):
    a = r.get("again")
    if not a:
        return None
    if medium == "land":
        return dict(ok=bool(a["passed"]), G=[a["G1"], a["G2"], a["G3"], a["G4"]], t_off=a["t_off"], H_end=a["H_end"],
                    x2=a.get("x2"))
    g = gx30(a)
    return dict(ok=g["met"], why=g["why"], t_star=g["t_star"], clear_s=g["clear_s"], x2=a.get("x2"))


def takeoff_leg(R, rung, tag="landair"):
    res = {}
    for k, r in R.items():
        sp = r["spec"]
        if sp["scenario"] != tag or sp["rung"] != rung or sp["kind"] != "takeoff":
            continue
        x = r["result"]
        res[(sp["arm"], tuple(sp["entry"]))] = dict(ok=bool(x["passed"]), G=[x["G1"], x["G2"], x["G3"], x["G4"]],
                                                    t_off=x["t_off"], H_end=x["H_end"],
                                                    wing_contact_W=x["peak_wing_contact_W"],
                                                    stroke_tq=x["peak_stroke_hinge_Nm"], P_peak_W=x["P_peak_W"],
                                                    flags=flags(r, 4.5))
    return res


def everything(path=ROWS, rung="A0"):
    R = load(path)
    out = dict(rung=rung)
    out["air"] = air(R, rung, "air")
    out["land"] = land(R, rung, "land")
    out["water"] = water(R, rung, "water")
    out["landair_takeoff"] = takeoff_leg(R, rung)
    out["landair_cruise"] = air(R, rung, "landair")
    out["landair_landing"] = entry_leg(R, rung, "landair", "land")
    out["waterair_exit"] = exit_leg(R, rung)
    out["waterair_entry"] = entry_leg(R, rung, "waterair", "water")
    return out


def strkeys(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): strkeys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strkeys(v) for v in o]
    return o


def show(o, indent=""):
    o = strkeys(o)
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 160:
                print(f"{indent}{k}:")
                show(v, indent + "  ")
            else:
                print(f"{indent}{k}: {json.dumps(v, default=str)}")
    else:
        print(indent + json.dumps(o, default=str))


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROWS
    rung = sys.argv[2] if len(sys.argv) > 2 else "A0"
    show(everything(path, rung))
