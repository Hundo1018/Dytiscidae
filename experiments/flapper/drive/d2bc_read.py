"""Readers for `PREREG_D2bc_crank_and_motor.md`.

    PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d2bc_read.py b results/D2b_geometry.jsonl results/D2_grid.jsonl
    PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d2bc_read.py c results/D2c_rotor.jsonl results/D2_grid.jsonl results/D1_sweep.jsonl results/D1b_lever.jsonl
"""
import json
import sys

LOAD = lambda p: [json.loads(x) for x in open(p)]


def read_b(rows, d2):
    f0 = next(r for r in d2 if r["cand"] == "F0" and not r["vacuum"])
    fm = min(r["machine_W"] for r in d2 if r["cand"] == "FM" and not r["vacuum"])
    key = lambda r: (round(r["feather"]["c_h"] * 1e3), round(r["feather"]["drop"] * 1e3), r["feather"]["arm_scale"])
    air = {key(r): r for r in rows if not r["vacuum"]}
    vac = {key(r): r for r in rows if r["vacuum"]}
    print("| horn | drop | scale | trans_min | vacuum: input/peak | gap mm | range made (design) | readable | "
          "air: amp | lead | beat | on limit | gap mm | lift vs F0 | machine W | peak force N |")
    print("|" + "---|" * 16)
    for k in sorted(air):
        a, v = air[k], vac[k]
        if "error" in a or "error" in v:
            print(f"| {k[0]} | {k[1]} | {k[2]} | error: {(a.get('error') or v.get('error'))[:80]} |")
            continue
        da, dv, arm = a["drive"], v["drive"], a["drive"]["arm"]
        e = 100 * dv["shaft_W"] / dv["shaft_peak_W"]
        lo, hi = dv["feather_mean_deg"] - dv["feather_amp_deg"], dv["feather_mean_deg"] + dv["feather_amp_deg"]
        ok = (abs(e) <= 1.0 and dv["loop_gap_max_mm"] < 0.5 and abs(lo - arm["design_min_deg"]) <= 3.0
              and abs(hi - arm["design_max_deg"]) <= 3.0)
        print(f"| {k[0]} | {k[1]} | {k[2]} | {arm['transmission_min']:.2f} | {e:+.2f}% | {dv['loop_gap_max_mm']:.3f} | "
              f"{lo:+.1f}..{hi:+.1f} ({arm['design_min_deg']:+.1f}..{arm['design_max_deg']:+.1f}) | {'yes' if ok else 'NO'} | "
              f"{da['feather_amp_deg']:.1f} | {da['feather_lead_deg']:+.1f} | {da['freq_ratio']:.3f} | {da['at_limit']:.2f} | "
              f"{da['loop_gap_max_mm']:.3f} | {100 * (a['lift_over_weight'] / f0['lift_over_weight'] - 1):+.1f}% | "
              f"{a['machine_W']:.1f} | {da['loop_force_peak_N']:.0f} |")
    print(f"\nF0: {f0['lift_over_weight']:.3f} weights, {f0['machine_W']:.1f} W; FM best {fm:.1f} W")


def read_c(rows, d2, d1, d1b):
    J0 = 2e-5
    fm = [dict(J=J0, **r) for r in d2 if r["cand"] == "FM" and not r["vacuum"]]
    fm += [dict(J=r["feather"]["motor"]["J_r"], **r) for r in rows if r["cand"] == "FM"]
    print("FM (machine W; feather motor W; on limit; amp; lift)")
    for r in sorted(fm, key=lambda r: (-r["J"], r["feather"]["N"])):
        if "error" in r:
            print("  ", r["J"], r["feather"]["N"], r["error"][:80]); continue
        d = r["drive"]
        print(f"  J {r['J']:.0e} gear {r['feather']['N']}: {r['machine_W']:.1f} W; {d['feather_motor_elec_W']:.1f}; "
              f"{d['feather_motor_at_limit']:.2f}; amp {d['feather_amp_deg']:.1f}; lift {r['lift_over_weight']:.3f}")
    old = [dict(J=J0, **r) for r in d1 if not r.get("extra") and r["kind"] in ("DD", "FB")]
    old += [dict(J=J0, **r) for r in d1b if r["kind"] == "FB" and r["extra"].get("a") == 0.06]
    new = [dict(J=r["extra"]["motor"]["J_r"], **r) for r in rows if r["cand"] == "D1"]
    print("DD and FB (elec W; on limit; amp; beat; lift)")
    for r in sorted(old + new, key=lambda r: (r["point"], r["kind"], -r["J"], r["N"])):
        if "error" in r:
            print("  ", r["point"], r["kind"], r["J"], r["N"], r["error"][:80]); continue
        if r["kind"] == "FB" and not ((r["point"] == "cruise" and r["N"] == 48) or
                                      (r["point"] == "takeoff" and r["N"] == 24 and r["extra"].get("a") == 0.06)):
            continue
        d = r["drive"]
        print(f"  {r['point']} {r['kind']} J {r['J']:.0e} gear {r['N']}: {d['elec_W']:.2f} W; {d['at_limit']:.2f}; "
              f"amp {d['amp_deg']:.1f}; beat {d['freq_ratio']:.3f}; lift {r['lift_over_weight']:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "b":
        read_b(LOAD(sys.argv[2]), LOAD(sys.argv[3]))
    else:
        read_c(*(LOAD(p) for p in sys.argv[2:6]))
