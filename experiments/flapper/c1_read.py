"""Read pass C1 against its gates (PREREG_C1_pilot.md section 3).

    PYTHONPATH=. .venv/bin/python experiments/flapper/c1_read.py results/C1_pilot.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
LATE = 10
ASK = {"Z+2": (2.0, 0.0), "Z-2": (-2.0, 0.0), "V+1": (0.0, 1.0), "V-1": (0.0, -1.0)}


def gates(r):
    dz, dv = ASK.get(r["arm"], (0.0, 0.0))
    z, vx = r["z_beat"][-LATE:], r["vx_beat"][-LATE:]
    g = dict(
        height=max(abs(x - dz) for x in z) <= 0.5 and abs(r["dz_late_min"] - dz) <= 0.5
        and abs(r["dz_late_max"] - dz) <= 0.5,
        climb=abs(r["vz_late"]) <= 0.1,
        speed=max(abs(x - (8.0 + dv)) for x in vx) <= 0.5,
        swing=r["pitch_swing_late_deg"] <= 5.0,
        power=r["power_late_W"] <= 41.0,
        crank=r["crank_eq_peak"] <= 12.0,
        finite=bool(r["finite"]))
    return g


def stops(r):
    out = []
    if "f_late" in r:
        f0 = r["point"][0]
        if abs(r["f_late"] / f0 - 1.0) >= 0.195:
            out.append("f")
        if abs(r["pitch_ask_late_deg"]) >= 11.5:
            out.append("pitch")
    return out


for r in rows:
    r["g"] = gates(r)
    r["holds"] = all(r["g"].values())
    r["stops"] = stops(r)
arms = ["A0 fixed tail", "A1 pitch loop", "A2 pilot", "Z+2", "Z-2", "V+1", "V-1", "D nose-up 10", "G0 no flapping"]
print("arm               n  holds | fails by gate (height climb speed swing power crank) | on a stop | late: z m, vx m/s, P W, tail deg")
for a in arms:
    q = [r for r in rows if r["arm"] == a]
    if not q:
        continue
    fails = [sum(1 for r in q if not r["g"][k]) for k in ("height", "climb", "speed", "swing", "power", "crank")]
    dz = ASK.get(a, (0.0, 0.0))[0]
    zl = [st.mean(r["z_beat"][-LATE:]) for r in q]
    vl = [st.mean(r["vx_beat"][-LATE:]) for r in q]
    print("%-16s %2d  %2d    | %s | %d | z %+.2f..%+.2f  vx %.2f..%.2f  P %.1f..%.1f  tail %.0f..%.0f" % (
        a, len(q), sum(r["holds"] for r in q), " ".join("%2d" % x for x in fails),
        sum(1 for r in q if r["stops"]), min(zl), max(zl), min(vl), max(vl),
        min(r["power_late_W"] for r in q), max(r["power_late_W"] for r in q),
        min(r.get("tail_min_deg", 0.0) for r in q), max(r.get("tail_max_deg", 0.0) for r in q)))

key = lambda r: tuple(r["point"])
by = {a: {key(r): r for r in rows if r["arm"] == a} for a in arms}
a2 = [r for r in by["A2 pilot"].values() if r["holds"]]
if a2:
    ratio = sorted(r["power_late_W"] / r["held_power_W"] for r in a2)
    print("\nA2 holding points: late power / W6 held power: %.2f..%.2f (median %.2f); cheapest %.2f W, dearest %.2f W" % (
        ratio[0], ratio[-1], st.median(ratio), min(r["power_late_W"] for r in a2), max(r["power_late_W"] for r in a2)))
    droop = [8.0 - st.mean(r["vx_beat"][-LATE:]) for r in a2]
    print("   speed below 8 m/s by %.2f..%.2f (median %.2f); droop in 0.05-0.5 at %d of %d" % (
        min(droop), max(droop), st.median(droop), sum(0.05 <= x <= 0.5 for x in droop), len(droop)))
    print("   height offset %.2f..%.2f m; pitch late %.1f..%.1f deg; swing %.1f..%.1f; f late/f0 %.2f..%.2f" % (
        min(st.mean(r["z_beat"][-LATE:]) for r in a2), max(st.mean(r["z_beat"][-LATE:]) for r in a2),
        min(r["pitch_late_deg"] for r in a2), max(r["pitch_late_deg"] for r in a2),
        min(r["pitch_swing_late_deg"] for r in a2), max(r["pitch_swing_late_deg"] for r in a2),
        min(r["f_late"] / r["point"][0] for r in a2), max(r["f_late"] / r["point"][0] for r in a2)))
both_z = sum(1 for k in by["Z+2"] if by["Z+2"][k]["holds"] and by["Z-2"].get(k, {}).get("holds"))
both_v = sum(1 for k in by["V+1"] if by["V+1"][k]["holds"] and by["V-1"].get(k, {}).get("holds"))
print("\nheight followed both ways: %d of %d; speed followed both ways: %d of %d" % (both_z, len(by["Z+2"]), both_v, len(by["V+1"])))


def settle(r, target, band, series):
    """Beats after the step until the beat mean enters the band and stays."""
    x = r[series][40:]
    for i in range(len(x)):
        if all(abs(v - target) <= band for v in x[i:]):
            return i
    return None


for a, series, tgt, band in (("Z+2", "z_beat", 2.0, 0.5), ("Z-2", "z_beat", -2.0, 0.5),
                             ("V+1", "vx_beat", 9.0, 0.5), ("V-1", "vx_beat", 7.0, 0.5)):
    s = [settle(r, tgt, band, series) for r in by[a].values()]
    ok = sorted(x for x in s if x is not None)
    pk = [max(r["P_beat"][40:]) for r in by[a].values()]
    print("%s: settles (stays in band) at %d of %d; beats to settle %s; peak beat power %.1f..%.1f W; late power %.1f..%.1f" % (
        a, len(ok), len(s), ("%d..%d (median %d)" % (ok[0], ok[-1], ok[len(ok) // 2])) if ok else "-",
        min(pk), max(pk), min(r["power_late_W"] for r in by[a].values()), max(r["power_late_W"] for r in by[a].values())))
print("\nG0 (no flapping):")
for r in by["G0 no flapping"].values():
    print("   z late %+.2f, vx late %.2f, vz late %+.2f, pitch late %+.1f, holds %s" % (
        st.mean(r["z_beat"][-LATE:]), st.mean(r["vx_beat"][-LATE:]), r["vz_late"], r["pitch_late_deg"], r["holds"]))
print("\npoints that fail A2, and why:")
for r in by["A2 pilot"].values():
    if not r["holds"]:
        print("  ", r["point"], [k for k, v in r["g"].items() if not v], "stops", r["stops"],
              "z %+.2f vx %.2f P %.1f" % (st.mean(r["z_beat"][-LATE:]), st.mean(r["vx_beat"][-LATE:]), r["power_late_W"]))
