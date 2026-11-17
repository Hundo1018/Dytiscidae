"""Read pass D1 against its frozen checks.

    .venv/bin/python experiments/flapper/drive/d1_read.py results/D1_sweep.jsonl results/D1_checks.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_run  # noqa: E402

sweep = [json.loads(x) for x in open(sys.argv[1])]
checks = [json.loads(x) for x in open(sys.argv[2])] if len(sys.argv) > 2 else []
err = [r for r in sweep + checks if "error" in r]
print("rows %d sweep, %d checks, %d raised" % (len(sweep), len(checks), len(err)))
for r in err:
    print("  raised:", r["point"], r["kind"], r["N"], r["k"], r["extra"], r["error"][:160])

base = {p: next(r for r in sweep if r["point"] == p and r["kind"] == "base") for p in d1_run.POINTS}
bests = {}
for p in d1_run.POINTS:
    Phi = d1_run.POINTS[p][3]
    b0 = base[p]
    print("\n%s: baseline (prescribed stroke, ideal torque): %.2f W, lift %.3f N (%.3f of weight), thrust %+.3f N"
          % (p, b0["power_W"], b0["Fz"], b0["lift_over_weight"], b0["Fx"]))
    print("  %-4s %4s %5s %6s %7s %7s %7s %6s %6s %6s %6s %6s %6s %7s %6s %5s %5s" % (
        "kind", "N", "k", "amp", "mean", "down", "kinerr", "mechW", "elecW", "Cu W", "lift%", "dFx", "limit",
        "loopN", "gapmm", "freq", "slack"))
    for kind in ("DD", "FB", "SY", "CT"):
        rows = sorted((r for r in sweep if r["point"] == p and r["kind"] == kind and "error" not in r),
                      key=lambda r: (r["k"], r["N"]))
        b = d1_run.best(sweep, p, kind)
        bests[(p, kind)] = b
        for r in rows:
            dv = r["drive"]
            mark = " <- least electrical power within 2 deg" if r is b else ""
            print("  %-4s %4d %5.1f %6.2f %+7.2f %7.3f %7s %6.2f %6.2f %6.2f %+6.1f %+6.3f %6.3f %7.1f %6.3f %5.3f %5s%s" % (
                kind, r["N"], r["k"], dv["amp_deg"], dv["mean_deg"], dv["down_share"],
                "%.2f" % dv["kin_err_deg"] if "kin_err_deg" in dv else "-", r["power_W"], dv["elec_W"],
                dv["copper_W"], 100 * (r["Fz"] / b0["Fz"] - 1), r["Fx"] - b0["Fx"], dv["at_limit"],
                dv["loop_force_peak_N"], dv["loop_gap_max_mm"], dv.get("freq_ratio", float("nan")),
                "%.2f" % dv["slack_share"] if "slack_share" in dv else "-", mark))
        if b is None:
            print("  %-4s none holds the stroke within 2 deg" % kind)

print("\nAgainst the frozen checks")
for p in d1_run.POINTS:
    for kind in ("DD", "FB", "SY", "CT"):
        b = bests[(p, kind)]
        if b is None:
            continue
        vac = next((r for r in checks if r["point"] == p and r["kind"] == kind and r["extra"].get("vacuum")), None)
        half = next((r for r in checks if r["point"] == p and r["kind"] == kind and r["extra"].get("motor")), None)
        dv = b["drive"]
        line = "  %-7s %-3s N %d k %g:" % (p, kind, b["N"], b["k"])
        if vac and "error" not in vac:
            vd = vac["drive"]
            line += " K1 vacuum mean shaft %.4f of peak, gap %.3f mm;" % (abs(vd["shaft_W"]) / max(vd["shaft_peak_W"], 1e-12),
                                                                     vd["loop_gap_max_mm"])
        if kind in ("FB", "SY"):
            line += " K2 kin err %.2f deg; K3 down %.3f, h2 (closed form) %.3f;" % (dv["kin_err_deg"], dv["down_share"], dv["design_h2"])
            b0 = base[p]
            line += " K4 lift %+.1f%%, dFx %+.3f N, power %+.1f%%;" % (100 * (b["Fz"] / b0["Fz"] - 1), b["Fx"] - b0["Fx"],
                                                                      100 * (b["power_W"] / b0["power_W"] - 1))
        if kind == "CT":
            line += " K6 down %.3f, slack %.3f;" % (dv["down_share"], dv.get("slack_share", float("nan")))
        line += " K7 loop force %.0f N; K8 at limit %.3f" % (dv["loop_force_peak_N"], dv["at_limit"])
        if half and "error" not in half:
            line += " (half motor %.3f, amp %.1f deg, elec %.2f W)" % (half["drive"]["at_limit"], half["drive"]["amp_deg"],
                                                                      half["drive"]["elec_W"])
        print(line)
    dd, fb = bests[(p, "DD")], bests[(p, "FB")]
    if dd and fb:
        print("  %-7s K5 DD electrical %.2f W against FB %.2f W: %+.0f%%" % (
            p, dd["drive"]["elec_W"], fb["drive"]["elec_W"], 100 * (dd["drive"]["elec_W"] / fb["drive"]["elec_W"] - 1)))
for r in checks:
    if r["extra"].get("lam") is not None and "error" not in r:
        print("  link mass %g kg/m (FB cruise): elec %.3f W, lift %.3f N, loop force %.0f N" % (
            r["extra"]["lam"], r["drive"]["elec_W"], r["Fz"], r["drive"]["loop_force_peak_N"]))
