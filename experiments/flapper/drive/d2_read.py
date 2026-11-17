"""Pass D2's reader: the table and the checks G1-G4 of
`PREREG_D2_feather_drive.md` (with addendum D2a) from `results/D2_grid.jsonl`.

    PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d2_read.py experiments/flapper/results/D2_grid.jsonl
"""
import json
import sys


def label(r):
    fe = r["feather"] or {}
    if r["cand"] == "FM":
        return f"FM gear {fe['N']}"
    if r["cand"] == "FP":
        return f"FP k {fe['k']:g}"
    if r["cand"].startswith("FX2"):
        return f"{r['cand']} stroke gear {r['N']}"
    return r["cand"]


def mech_in(r):
    """Every motor's shaft power plus the servo's net power (G1's input)."""
    d = r["drive"]
    return d["shaft_W"] + d.get("feather_motor_shaft_W", 0.0) + (r["feather_power_W"] if r["feather"] is None else 0.0)


def main() -> int:
    # argv: the grid, then any re-run rows that replace the grid's of the same candidate
    rows = [json.loads(x) for x in open(sys.argv[1])]
    for extra in sys.argv[2:]:
        new = [json.loads(x) for x in open(extra)]
        rows = [r for r in rows if r["cand"] not in {n["cand"] for n in new}] + new
    order = {"F0": 0, "FM": 1, "FX": 2, "FX2": 3, "FX2 stop 85": 4, "FP": 5}
    rows.sort(key=lambda r: (r["vacuum"], order[r["cand"]], r["N"], json.dumps(r["feather"])))
    f0 = next(r for r in rows if r["cand"] == "F0" and not r["vacuum"])
    print("| candidate | medium | lift (weights) | lift vs F0 | machine elec W | stroke motor elec W | feather motor elec W "
          "| servo net W | feather amp | mean | lead | stroke amp | beat | stroke motor on limit | feather motor on limit "
          "| peak loop force N | max gap mm |")
    print("|" + "---|" * 17)
    for r in rows:
        if "error" in r:
            print(f"| {label(r)} | {'vacuum' if r['vacuum'] else 'air'} | {r['error']} |" + " |" * 15)
            continue
        d = r["drive"]
        print(f"| {label(r)} | {'vacuum' if r['vacuum'] else 'air'} | {r['lift_over_weight']:.3f} | "
              f"{100 * (r['lift_over_weight'] / f0['lift_over_weight'] - 1):+.1f}% | {r['machine_W']:.1f} | "
              f"{d['elec_W']:.1f} | {d.get('feather_motor_elec_W', float('nan')):.1f} | "
              f"{r['feather_power_W'] if r['feather'] is None else 0.0:.1f} | {d['feather_amp_deg']:.1f} | "
              f"{d['feather_mean_deg']:+.1f} | {d['feather_lead_deg']:+.1f} | {d['amp_deg']:.1f} | {d['freq_ratio']:.3f} | "
              f"{d['at_limit']:.2f} | {d.get('feather_motor_at_limit', float('nan')):.2f} | {d['loop_force_peak_N']:.0f} | "
              f"{d['loop_gap_max_mm']:.3f} |")
    ok = lambda r: "error" not in r and not r["vacuum"]
    print("\nG1 (vacuum): mechanical input per beat / largest motor's peak shaft power")
    for r in rows:
        if r["vacuum"] and "error" not in r:
            d = r["drive"]
            peak = max(d["shaft_peak_W"], d.get("feather_motor_shaft_peak_W", 0.0))
            print(f"  {label(r)}: {mech_in(r):+.2f} W / {peak:.1f} W = {100 * mech_in(r) / peak:+.2f}%  "
                  f"(stroke shaft {d['shaft_W']:+.2f}, feather motor shaft {d.get('feather_motor_shaft_W', 0.0):+.2f}, "
                  f"servo {r['feather_power_W'] if r['feather'] is None else 0.0:+.2f})")
    fm = [r for r in rows if ok(r) and r["cand"] == "FM"
          and abs(r["drive"]["feather_amp_deg"] - 50.0) <= 3.0 and r["drive"]["freq_ratio"] >= 0.98]
    fmb = min(fm, key=lambda r: r["machine_W"]) if fm else None
    print("\nG2: FM best by D1's rule:", label(fmb) if fmb else None,
          f"machine {fmb['machine_W']:.1f} W, feather motor elec {fmb['drive']['feather_motor_elec_W']:.1f} W" if fmb else "")
    for r in rows:
        if ok(r) and r["cand"].startswith("FX2"):
            d = r["drive"]
            print(f"G3 {label(r)}: amp {d['feather_amp_deg']:.1f}, lead {d['feather_lead_deg']:+.1f}, "
                  f"lift vs F0 {100 * (r['lift_over_weight'] / f0['lift_over_weight'] - 1):+.1f}%, "
                  f"machine {r['machine_W']:.1f} W vs FM best {fmb['machine_W'] if fmb else float('nan'):.1f} "
                  f"({100 * (r['machine_W'] / fmb['machine_W'] - 1) if fmb else float('nan'):+.1f}%), "
                  f"limit {d['at_limit']:.2f}, beat {d['freq_ratio']:.3f}")
    for r in rows:
        if ok(r) and r["cand"] == "FP":
            d = r["drive"]
            print(f"G4 {label(r)}: amp {d['feather_amp_deg']:.1f}, lead {d['feather_lead_deg']:+.1f} "
                  f"(a quarter beat is -90 here), lift vs F0 {100 * r['lift_over_weight'] / f0['lift_over_weight']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
