"""Pass M3: the frozen gates of PREREG_M3_land.md section 3, read from the rows.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m3_read.py results/M3_land.jsonl
"""
import json
import sys
from collections import defaultdict

BAR, HALF_MIN, CROSS = 0.020, 0.06, 0.5


def stand_pass(r):
    why = []
    if r["bad_qacc"]:
        why.append("qacc")
    if r["share"]["body"] > 0.02 or r["body_contact_frac"] > 0.05:
        why.append("body %.3f/%.3f" % (r["share"]["body"], r["body_contact_frac"]))
    drift = (r["dx"] ** 2 + r["dy"] ** 2) ** 0.5
    if drift > 0.02:
        why.append("drift %.3f" % drift)
    if abs(r["dyaw_deg"]) > 2.0:
        why.append("yaw %.2f" % r["dyaw_deg"])
    return not why, drift, why


def walk_pairs(rows, untimed=None):
    """rows: walk halves of one arm. Pairs by 'pair' key. untimed: {pair: pair mean} of U."""
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    out = []
    for pk, hs in sorted(by.items()):
        why = []
        if len(hs) == 1 and hs[0]["arm"] in ("N0", "N1"):       # the null never reads the command
            a = hs[0]
            b = dict(a, along=-a["along"], across=-a["across"], heading_deg=(a["heading_deg"] + 180) % 360)
            hs = [a, b]
        if len(hs) != 2:
            out.append((pk, False, None, ["incomplete"], hs))
            continue
        a, b = hs
        if a["hash_c"] != b["hash_c"]:
            why.append("hash")
        if a["bad_qacc"] or b["bad_qacc"]:
            why.append("qacc")
        win = a["window_s"]
        pm = (a["along"] + b["along"]) / (2 * win)
        if pm < BAR:
            why.append("pair mean %.4f" % pm)
        for h in (a, b):
            if h["along"] < HALF_MIN:
                why.append("half %s along %.3f" % (h["heading_deg"], h["along"]))
            elif abs(h["across"]) > CROSS * h["along"]:
                why.append("half %s across %.3f" % (h["heading_deg"], h["across"]))
        if untimed and untimed.get(pk, 0) > 0 and pm < 2 * untimed[pk]:
            why.append("U %.4f" % untimed[pk])
        out.append((pk, not why, pm, why, hs))
    return out


def null_err(r, turn_deg):
    import math
    import numpy as np
    from pathlib import Path
    tr = np.load(Path(__file__).resolve().parent / "results" / "M3_traces" / f"{r['id']}.npz")
    tgt = math.radians(r["yaw0_deg"] + turn_deg)
    err = np.degrees(np.abs((tr["att"][:, 0] - tgt + np.pi) % (2 * np.pi) - np.pi))
    last = tr["t"] >= tr["t"][-1] - 2.0
    return float(err[last].mean()), float(err[last].max())


def turn_pairs(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    out = []
    for pk, hs in sorted(by.items()):
        why = []
        if len(hs) == 1 and hs[0]["arm"] in ("N0", "N1"):      # the null never reads the command:
            a = hs[0]                                          # its other half is the same run, read
            b = dict(a, turn_deg=-a["turn_deg"])               # against the other command from the trace
            b["err_last2_mean_deg"], b["err_last2_max_deg"] = null_err(a, -a["turn_deg"])
            hs = [a, b]
        for h in hs:
            if h["bad_qacc"]:
                why.append("qacc")
            em, ex = h["err_last2_mean_deg"], h["err_last2_max_deg"]
            if em > 15 or ex > 20:
                why.append("turn %+d err %.1f/%.1f" % (h["turn_deg"], em, ex))
            if h["dyaw_deg"] * h["turn_deg"] <= 0:
                why.append("turn %+d wrong sense %.1f" % (h["turn_deg"], h["dyaw_deg"]))
        if len(hs) == 2 and hs[0]["hash_c"] != hs[1]["hash_c"]:
            why.append("hash")
        out.append((pk, not why, None, why, hs))
    return out


def main(path):
    rows = [json.loads(x) for x in open(path)]
    arms = defaultdict(list)
    for r in rows:
        arms[(r["arm"], r.get("family", ""), r["kind"], r.get("J_r", 2e-5))].append(r)
    for key in sorted(arms, key=str):
        arm, fam, kind, jr = key
        rs = arms[key]
        print(f"\n== {arm} {fam} {kind} J_r {jr}: {len(rs)} runs, qacc runs {sum(r['bad_qacc'] > 0 for r in rs)}")
        if kind == "stand":
            for r in rs:
                ok, drift, why = stand_pass(r)
                print("  yaw0 %4.0f %s drift %.4f dyaw %.3f body %.3f/%.3f P %.1f W peakP window %.0f W, run %.0f W  %s" % (
                    r["yaw0_deg"], "PASS" if ok else "fail", drift, r["dyaw_deg"], r["share"]["body"],
                    r["body_contact_frac"], r["P_window_W"], r["peak_P_window_W"], r["peak_P_run_W"], ",".join(why)))
        elif kind == "walk" and arm == "F":                      # a pairing check, not a walk gate
            a, b = rs if len(rs) == 2 else (rs + [None])[:2]
            pm = (a["along"] + b["along"]) / (2 * a["window_s"]) if b else float("nan")
            print("  F pair mean %.3e m/s (must be |.| < 1e-9), hashes equal %s -> %s" % (
                pm, b and a["hash_c"] == b["hash_c"], "PASS" if b and abs(pm) < 1e-9 and a["hash_c"] == b["hash_c"] else "fail"))
        elif kind == "walk":
            un = {}
            if arm in ("G1", "G2", "F"):
                for p in walk_pairs([r for r in rows if r["arm"] == "U" and r.get("family") == fam and r["kind"] == "walk"
                                     and r.get("J_r", 2e-5) == jr]):
                    if p[2] is not None:
                        un[p[0]] = p[2]
            for pk, ok, pm, why, hs in walk_pairs(rs, un):
                print("  %-10s %s pair %s halves %s  %s" % (pk, "PASS" if ok else "fail",
                      "%.4f" % pm if pm is not None else "-",
                      " ".join("h%s:%.3f/%.3f" % (h["heading_deg"], h["along"], h["across"]) for h in hs), ",".join(why)))
        elif kind == "turn":
            for pk, ok, _, why, hs in turn_pairs(rs):
                print("  %-10s %s %s  %s" % (pk, "PASS" if ok else "fail",
                      " ".join("%+d:dyaw %.1f" % (h["turn_deg"], h["dyaw_deg"]) for h in hs), ",".join(why)))


if __name__ == "__main__":
    main(sys.argv[1])
