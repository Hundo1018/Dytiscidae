"""Read pass 1m against its frozen text (PREREG_L1m_tail.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/l1m_read.py results/L1m_tail.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
arm = {}
for r in rows:
    arm.setdefault(r["arm"], {}).setdefault(r["tail_area"], {})[tuple(r["point"])] = r
pts = sorted(arm["T1"]["0.06"])
print("rows", len(rows), "not finite", sum(not r["finite"] for r in rows), "cruise points", len(pts))
med = st.median


def pitch_abs(r):
    return [x + r["theta_deg"] for x in r["pitch_beat_mean_deg"]]


def bounded(r, lim=15.0):
    return r["finite"] and all(abs(x) <= lim for x in pitch_abs(r))


def flies(r):
    return (bounded(r) and r["dz_min"] >= -3.0 and r["dz_max"] <= 3.0 and abs(r["vz_late"]) <= 0.3
            and r["vx_min"] >= 5.0 and r["vx_max"] <= 11.0
            and r["power_W"] + r.get("tail_power_W", 0.0) <= 41.0 and r["crank_eq_peak"] <= 12.0)


print("\nheld: pitching moment about the centre of mass (nose-down positive), N m")
for tail in ("none", "0.06"):
    k, sw, m0 = [], [], []
    for p in pts:
        lo, mid, hi = (arm["held th %+.0f" % t][tail][p] for t in (-5, 0, 5))
        k.append((hi["pitch_moment_cg_mean"] - lo["pitch_moment_cg_mean"]) / 10.0)
        sw.append(mid["pitch_moment_cg_swing"])
        m0.append(mid["pitch_moment_cg_mean"])
    print("  tail %-5s restoring stiffness %.4f..%.4f N m/deg (median %.4f); mean at 0 deg %+.2f..%+.2f; swing %.2f..%.2f" % (
        tail, min(k), max(k), med(k), min(m0), max(m0), min(sw), max(sw)))
e = [(arm["held i0 +4"]["0.06"][p]["pitch_moment_cg_mean"] - arm["held i0 -4"]["0.06"][p]["pitch_moment_cg_mean"]) / 8.0
     for p in pts]
print("  tail authority %.3f..%.3f N m per deg of incidence" % (min(e), max(e)))

print("\nreleased, 40 beats")
names = {"T0": ("none", "no tail"), "T1": ("0.06", "tail 0.06 fixed"), "T2": ("0.06", "tail 0.06 feedback"),
         "T3": ("0.03", "tail 0.03 fixed"), "T4": ("0.03", "tail 0.03 feedback"),
         "T5": ("0.06", "tail 0.06 fixed, built inertia"), "T6": ("0.06", "tail 0.06 fixed, inertia 0.03"),
         "D1": ("0.06", "start +10 deg, fixed"), "D2": ("0.06", "start +10 deg, feedback")}
for a, (tail, label) in names.items():
    rs = [arm[a][tail][p] for p in pts]
    b = [r for r in rs if bounded(r)]
    fl = [r for r in rs if flies(r)]
    why = dict(height=sum(bounded(r) and not (r["dz_min"] >= -3 and r["dz_max"] <= 3) for r in rs),
               vz=sum(bounded(r) and abs(r["vz_late"]) > 0.3 for r in rs),
               speed=sum(bounded(r) and not (r["vx_min"] >= 5 and r["vx_max"] <= 11) for r in rs),
               power=sum(bounded(r) and r["power_W"] + r.get("tail_power_W", 0.0) > 41 for r in rs),
               crank=sum(bounded(r) and r["crank_eq_peak"] > 12 for r in rs))
    line = "  %-3s %-32s pitch within 15 deg: %2d of %d; flies: %2d" % (a, label, len(b), len(rs), len(fl))
    if b:
        line += "; of the bounded: swing %.1f..%.1f deg, end pitch %+.1f..%+.1f, dz %+.2f..%+.2f m, vz late %+.2f..%+.2f, vx %.1f..%.1f, power %.1f..%.1f W, crank %.1f..%.1f" % (
            min(r["pitch_swing_late_deg"] for r in b), max(r["pitch_swing_late_deg"] for r in b),
            min(pitch_abs(r)[-1] for r in b), max(pitch_abs(r)[-1] for r in b),
            min(r["dz_min"] for r in b), max(r["dz_max"] for r in b),
            min(r["vz_late"] for r in b), max(r["vz_late"] for r in b),
            min(r["vx_min"] for r in b), max(r["vx_max"] for r in b),
            min(r["power_W"] for r in b), max(r["power_W"] for r in b),
            min(r["crank_eq_peak"] for r in b), max(r["crank_eq_peak"] for r in b))
        if "tail_min_deg" in b[0]:
            line += "; tail %.1f..%.1f deg, servo %.2f W" % (min(r["tail_min_deg"] for r in b),
                                                             max(r["tail_max_deg"] for r in b),
                                                             max(r["tail_power_W"] for r in b))
    print(line)
    print("       bounded but failing:", why)
    if a == "T0":
        gone = [next((i for i, x in enumerate(pitch_abs(r)) if abs(x) > 15), None) for r in rs]
        print("       first beat beyond 15 deg:", gone)
    if a in ("D1", "D2"):
        ref = "T1" if a == "D1" else "T2"
        rec = []
        for p in pts:
            r, base = arm[a]["0.06"][p], arm[ref]["0.06"][p]
            end = pitch_abs(base)[-1]
            rec.append(next((i for i, x in enumerate(pitch_abs(r))
                             if all(abs(y - end) <= 5 for y in pitch_abs(r)[i:])), None))
        print("       beats until within 5 deg of the undisturbed run's end pitch for good:", rec)

print("\ntake-off points, 4 beats")
for a, tail, label in (("K0", "none", "no tail, inertia 0.012"), ("K1", "0.06", "tail 0.06 feedback, inertia 0.012")):
    rs = list(arm[a][tail].values())
    ok = [r for r in rs if r["finite"] and r["pitch_dev_max_deg"] <= 45 and r["dz"] >= 1.0]
    print("  %-34s within 45 deg and climbs 1 m: %d of %d; largest pitch excursion %.0f..%.0f deg; climb %.2f..%.2f m" % (
        label, len(ok), len(rs), min(r["pitch_dev_max_deg"] for r in rs), max(r["pitch_dev_max_deg"] for r in rs),
        min(r["dz"] for r in rs), max(r["dz"] for r in rs)))
