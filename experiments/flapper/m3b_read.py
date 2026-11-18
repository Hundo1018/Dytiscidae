"""Pass M3b: the frozen gates (M3's, PREREG_M3b section 3) on the 12/14 s reads, and the
secondary 24 s read, from the rows.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m3b_read.py results/M3b_land.jsonl [tuning]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3_read import walk_pairs  # noqa: E402

TRACES = HERE / "results" / "M3b_traces"


def as_walk(r, W):
    return dict(r, along=r[f"along_{W:g}"], across=r[f"across_{W:g}"], window_s=W)


def turn_err(r, target_turn, W):
    tr = np.load(TRACES / f"{r['id']}.npz")
    tgt = math.radians(r["yaw0_deg"] + target_turn)
    err = np.degrees(np.abs((tr["att"][:, 0] - tgt + np.pi) % (2 * np.pi) - np.pi))
    t_c = 2.0
    last = (tr["t"] >= t_c + W - 2.0) & (tr["t"] <= t_c + W + 1e-9)
    return float(err[last].mean()), float(err[last].max())


def turn_pairs(rows, W):
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    out = []
    for pk, hs in sorted(by.items()):
        why = []
        halves = []
        if len(hs) == 1 and hs[0]["arm"] in ("N0", "N1"):      # one run read against both commands
            a = hs[0]
            for td in (a["turn_deg"], -a["turn_deg"]):
                em, ex = turn_err(a, td, W)
                halves.append((td, a[f"dyaw_{W:g}"], em, ex, a))
        else:
            for h in hs:
                halves.append((h["turn_deg"], h[f"dyaw_{W:g}"], h[f"err_mean_{W:g}"], h[f"err_max_{W:g}"], h))
        if len(halves) != 2:
            why.append("incomplete")
        for td, dy, em, ex, h in halves:
            if h["bad_qacc"]:
                why.append("qacc")
            if em > 15 or ex > 20:
                why.append("turn %+d err %.1f/%.1f" % (td, em, ex))
            if dy * td <= 0:
                why.append("turn %+d wrong sense %.1f" % (td, dy))
        if len(hs) == 2 and hs[0]["hash_c"] != hs[1]["hash_c"]:
            why.append("hash")
        out.append((pk, not why, halves, why))
    return out


def main(path, tuning=False):
    rows = [json.loads(x) for x in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["machine"], r["arm"], r["kind"])].append(r)
    for key in sorted(groups):
        mach, arm, kind = key
        rs = groups[key]
        print(f"\n== {mach} {arm} {kind}: {len(rs)} runs, qacc runs {sum(r['bad_qacc'] > 0 for r in rs)}")
        if kind == "stand":
            for r in rs:
                why = []
                if r["bad_qacc"]:
                    why.append("qacc")
                if r["share"]["body"] > 0.02 or r["body_contact_frac"] > 0.05:
                    why.append("body %.3f/%.3f" % (r["share"]["body"], r["body_contact_frac"]))
                if r["drift_m"] > 0.02:
                    why.append("drift %.3f" % r["drift_m"])
                if abs(r["dyaw_deg"]) > 2:
                    why.append("yaw %.2f" % r["dyaw_deg"])
                print("  yaw0 %4.0f %s drift %.4f dyaw %.3f body %.3f/%.3f P %.1f W  %s" % (
                    r["yaw0_deg"], "fail" if why else "PASS", r["drift_m"], r["dyaw_deg"], r["share"]["body"],
                    r["body_contact_frac"], r["P_window_W"], ",".join(why)))
            continue
        for W, lab in ((12.0, "PRIMARY 12 s") if kind == "walk" else (14.0, "PRIMARY 14 s"), (24.0, "secondary 24 s")):
            if not all(f"along_{W:g}" in r for r in rs):
                continue
            print("  --", lab)
            if kind == "walk":
                un = {}
                if arm in ("T", "K"):
                    urs = [as_walk(r, W) for r in rows if r["machine"] == mach and r["arm"] == "U" and r["kind"] == "walk"]
                    for p in walk_pairs(urs):
                        if p[2] is not None:
                            un[p[0]] = p[2]
                ws = [as_walk(r, W) for r in rs]
                if tuning:                                     # tuning: pair by (gait, yaw) instead of a pair key
                    for r in ws:
                        r["pair"] = json.dumps(r.get("gait", {}), sort_keys=True)[:60] + "@%g" % r["yaw0_deg"]
                n = 0
                for pk, ok, pm, why, hs in walk_pairs(ws, un if not tuning else None):
                    n += ok
                    print("  %-10s %s pair %s halves %s  %s" % (pk, "PASS" if ok else "fail",
                          "%.4f" % pm if pm is not None else "-",
                          " ".join("h%s:%.3f/%.3f" % (h["heading_deg"], h["along"], h["across"]) for h in hs), ",".join(why)))
                print("  pairs passed:", n)
            else:
                n = 0
                for pk, ok, halves, why in turn_pairs(rs, W):
                    n += ok
                    print("  %-10s %s %s  %s" % (pk, "PASS" if ok else "fail",
                          " ".join("%+d:dyaw %.1f err %.1f" % (td, dy, em) for td, dy, em, ex, h in halves), ",".join(why)))
                print("  pairs passed:", n)


if __name__ == "__main__":
    main(sys.argv[1], "tuning" in sys.argv[2:])
