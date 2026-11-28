"""Read pass D4 against its frozen checks (`PREREG_D4_variable_stroke.md`).

    .venv/bin/python experiments/flapper/drive/d4_read.py results/D4_sweep.jsonl results/D4_checks.jsonl results/D4_geometry.jsonl
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d4_run  # noqa: E402

K85 = math.sin(math.radians(42.5))               # freerig's constant behind crank_eq_peak
load = lambda p: [json.loads(x) for x in open(p)]
sweep = load(sys.argv[1])
checks = load(sys.argv[2]) if len(sys.argv) > 2 else []
geom = load(sys.argv[3]) if len(sys.argv) > 3 else []
err = [r for r in sweep + checks if "error" in r]
print("rows %d sweep, %d checks, %d raised" % (len(sweep), len(checks), len(err)))
for r in err:
    print("  raised:", r["point"], r["kind"], r["N"], r["extra"], r["error"][:200])
good = [r for r in sweep if "error" not in r]
POINTS = d4_run.POINTS
base = {p: next(r for r in good if r["point"] == p and r["kind"] == "base") for p in POINTS}
variants = lambda kind: d4_run.VARIANTS if kind in d4_run.STIFF else d4_run.VARIANTS[:1]

# V10 first: the dedicated four-bar against the stored D1b rows
print("\nV10, session guard: dedicated FB, take-off, lever 60 mm, against results/D1b_lever.jsonl")
old = [r for r in load(HERE.parent / "results" / "D1b_lever.jsonl")
       if r["kind"] == "FB" and r["extra"].get("a") == 0.06 and "error" not in r]
v10 = True
for o in sorted(old, key=lambda r: r["N"]):
    n = next((r for r in good if r["point"] == "takeoff" and r["kind"] == "FB" and r["N"] == o["N"]), None)
    if n is None:
        print("  N %d: not re-run" % o["N"])
        v10 = False
        continue
    de, dl = n["drive"]["elec_W"] / o["drive"]["elec_W"] - 1, n["Fz"] / o["Fz"] - 1
    v10 &= abs(de) <= 0.005 and abs(dl) <= 0.005
    print("  N %d: electrical %.3f W against %.3f (%+.3f%%), lift %.4f N against %.4f (%+.3f%%)" % (
        o["N"], n["drive"]["elec_W"], o["drive"]["elec_W"], 100 * de, n["Fz"], o["Fz"], 100 * dl))
print("  V10", "holds" if v10 else "FAILS")

bests = {}
for p in POINTS:
    Phi, f = POINTS[p][3], POINTS[p][2]
    b0 = base[p]
    fb = d4_run.best(sweep, p, "FB")
    print("\n%s: baseline %.2f W, lift %.3f N, thrust %+.3f N; dedicated FB best gear %s: %.2f W electrical, lift %.3f N" % (
        p, b0["power_W"], b0["Fz"], b0["Fx"], fb and fb["N"], fb["drive"]["elec_W"] if fb else float("nan"),
        fb["Fz"] if fb else float("nan")))
    print("  %-4s %-6s %3s %6s %6s %6s %6s %6s %6s %7s %7s %7s %7s %6s %6s %8s %8s  %s" % (
        "kind", "lock", "N", "amp", "mean", "down", "kinerr", "gapmm", "freq", "limit", "mechW", "elecW", "lift%b",
        "lift%F", "dFx", "hold", "slip", "peak force per loop, N"))
    for kind in d4_run.KINDS:
        for variant in variants(kind):
            ex = d4_run.extra_of(kind, variant)
            b = d4_run.best(sweep, p, kind, variant)
            bests[(p, kind, variant)] = b
            for r in sorted((r for r in good if r["point"] == p and r["kind"] == kind and r["extra"] == ex),
                            key=lambda r: r["N"]):
                dv = r["drive"]
                forces = dv.get("loop_force_N") or {"all": dv["loop_force_peak_N"]}
                gap = max(dv["loop_gap_mm"].values()) if "loop_gap_mm" in dv else dv["loop_gap_max_mm"]
                slip = max(dv["lock_slip_max"].values()) if "lock_slip_max" in dv else float("nan")
                slip = slip * (57.29578 if kind == "VP" else 1e3)
                print("  %-4s %-6s %3d %6.2f %+6.2f %6.3f %6.2f %6.3f %6.3f %7.3f %7.2f %7.2f %+7.1f %+6.1f %+6.3f %8.2f %8.3f  %s%s" % (
                    kind, variant, r["N"], dv["amp_deg"], dv["mean_deg"], dv["down_share"], dv["kin_err_deg"], gap,
                    dv["freq_ratio"], dv["at_limit"], r["power_W"], dv["elec_W"], 100 * (r["Fz"] / b0["Fz"] - 1),
                    100 * (r["Fz"] / fb["Fz"] - 1) if fb else float("nan"), r["Fx"] - b0["Fx"],
                    dv.get("hold_peak", float("nan")), slip,
                    " ".join("%s %.0f" % (k, v) for k, v in forces.items()),
                    "  <- best by the frozen rule" if r is b else ""))
            if b is None:
                print("  %-4s %-6s no gear holds the stroke within 2 deg and the beat within 0.98" % (kind, variant))

print("\nAgainst the frozen checks (each candidate at its best gear; none: at the gear in brackets, labelled)")
want_down = {"VL": (0.517, 0.483), "VR": (0.506, 0.497), "VP": (0.542, 0.458)}
want_force = {"VL": ((300, 600), (20, 45)), "VR": ((250, 350), (30, 60)), "VP": ((130, 260), (15, 40))}
want_hold = {"VL": ((100, 400), (5, 25)), "VR": ((100, 300), (10, 50)), "VP": ((1.5, 5), (0.4, 1.5))}
for kind in d4_run.KINDS[1:]:
    for variant in variants(kind):
        for ip, p in enumerate(("takeoff", "cruise")):
            Phi, f = POINTS[p][3], POINTS[p][2]
            ex = d4_run.extra_of(kind, variant)
            b, label = bests[(p, kind, variant)], ""
            if b is None:
                N0 = {"cruise": 48, "takeoff": 24}[p]
                b = next((r for r in good if r["point"] == p and r["kind"] == kind and r["extra"] == ex and r["N"] == N0), None)
                label = " [no best; gear %d]" % N0
            if b is None:
                print("  %s %s %s: no row" % (kind, variant, p))
                continue
            dv, b0, fb = b["drive"], base[p], bests.get((p, "FB", "frozen")) or d4_run.best(sweep, p, "FB")
            print("  %s %-6s %-7s N %d%s" % (kind, variant, p, b["N"], label))
            vac = next((r for r in checks if r["point"] == p and r["kind"] == kind and "error" not in r
                        and r["extra"] == dict(ex, vacuum=True)), None)
            if vac:
                peak = vac["crank_eq_peak"] * K85 * 2 * math.pi * f
                gapv = max(vac["drive"]["loop_gap_mm"].values())
                print("     V1 vacuum (N %d): mean of motor+servo power %.4f W = %.4f of its peak %.1f W; gap %.3f mm; kin err %.2f deg" % (
                    vac["N"], vac["power_W"], abs(vac["power_W"]) / peak, peak, gapv, vac["drive"]["kin_err_deg"]))
            gap = max(dv["loop_gap_mm"].values())
            print("     V2 kin err %.3f deg (gate 0.5); V3 amp %.2f (%.0f), mid %+.2f, down %.3f (closed form %.3f, design %.3f), h2 closed form %.3f" % (
                dv["kin_err_deg"], dv["amp_deg"], Phi, dv["mean_deg"], dv["down_share"], want_down[kind][ip],
                dv["design_down_share"], dv["design_h2"]))
            print("     V4 lift %+.2f%% of baseline, thrust %+.3f N, mech power %+.1f%%; lift %+.2f%% of dedicated FB" % (
                100 * (b["Fz"] / b0["Fz"] - 1), b["Fx"] - b0["Fx"], 100 * (b["power_W"] / b0["power_W"] - 1),
                100 * (b["Fz"] / fb["Fz"] - 1)))
            print("     V5 electrical %.2f W, %+.1f%% of dedicated FB's %.2f (gear %d)" % (
                dv["elec_W"], 100 * (dv["elec_W"] / fb["drive"]["elec_W"] - 1), fb["drive"]["elec_W"], fb["N"]))
            print("     V6 loop force peak %s N (predicted %d-%d); gap %.3f mm" % (
                ", ".join("%s %.0f" % kv for kv in dv["loop_force_N"].items()), *want_force[kind][ip], gap))
            line = "     V7 holds %.2f %s peak (predicted %g-%g); per wing %s; lock moved %.4g %s" % (
                dv["hold_peak"], dv["hold_unit"], *want_hold[kind][ip],
                ", ".join("%.2f" % v for v in dv["lock_peak"].values()),
                max(dv["lock_slip_max"].values()) * (57.29578 if kind == "VP" else 1e3), "deg" if kind == "VP" else "mm")
            if kind == "VP":
                quarter = dv["shaft_W"] * dv["N"] / dv["motor_speed_max"] / 4.0 if False else \
                    dv["shaft_W"] / (2 * math.pi * f * dv["freq_ratio"]) / 4.0
                line += "; a quarter of the mean crank torque %.3f N m, ratio %.1f; gear mesh peak %.2f N m, slip %.3f deg" % (
                    quarter, dv["hold_peak"] / quarter, dv["gear_torque_peak"], math.degrees(dv["gear_slip_max_rad"]))
            print(line)
            print("     V8 on the limit %.3f, beat %.3f" % (dv["at_limit"], dv["freq_ratio"]))
        r32 = next((r for r in good if r["point"] == "takeoff" and r["kind"] == kind and r["N"] == 32
                    and r["extra"] == d4_run.extra_of(kind, variant)), None)
        if r32:
            print("  %s %-6s take-off gear 32 (V8): on the limit %.3f, beat %.3f" % (
                kind, variant, r32["drive"]["at_limit"], r32["drive"]["freq_ratio"]))
for p in POINTS:
    fb = d4_run.best(sweep, p, "FB")
    if fb:
        dv = fb["drive"]
        print("  FB dedicated %s N %d: loop force %.0f N, gap %.3f mm, kin err %.3f deg, down %.3f, elec %.2f W, lift %+.2f%% of baseline" % (
            p, fb["N"], dv["loop_force_peak_N"], dv["loop_gap_max_mm"], dv["kin_err_deg"], dv["down_share"], dv["elec_W"],
            100 * (fb["Fz"] / base[p]["Fz"] - 1)))
        vac = next((r for r in checks if r["point"] == p and r["kind"] == "FB" and "error" not in r), None)
        if vac:
            peak = vac["crank_eq_peak"] * K85 * 2 * math.pi * POINTS[p][2]
            print("     vacuum: %.4f of peak, gap %.3f mm" % (abs(vac["power_W"]) / peak, vac["drive"]["loop_gap_max_mm"]))

print("\nV0 and V9, from the built models (draw_d4.py)")
for r in geom:
    if r.get("check") == "V0":
        print("  V0 %s: adjuster moved %.5g; loops within %.2e mm; stroke against the cruise model within %.2e deg" % (
            r["cand"], r["travel"], r["loop_worst_mm"], r["stroke_diff_deg"]))
    else:
        print("  V9 %s %-7s: lowest %.1f mm below the shoulder, outermost %.1f mm; dims %s" % (
            r["cand"], r["point"], r["lowest_mm"], r["outermost_mm"], r["dims"]))
