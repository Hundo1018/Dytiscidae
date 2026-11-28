"""Read a fold water map (L1 pass 1c or 1d) against its frozen gates.

    .venv/bin/python experiments/flapper/fold_read.py experiments/flapper/results/L1c_fold.jsonl
"""
import json
import sys

CAP_K0, CAP_SHOULDER, CAP_CRANK = 12.0, 15.8, 12.0


def show(r):
    return (f"stroke {r['Phi_deg']:.0f} f {r['f']} feather {r['Psi_deg']:.0f}: net {r['Fx']:.2f} N, "
            f"shoulder {r['stroke_torque_peak']:.1f}, crank {r['crank_torque']:.1f} N m, "
            f"{r['aero_power_W']:.2f} W")


def main() -> int:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    print(len(rows), "rows; finite", sum(r["finite"] for r in rows))
    wings = sorted({(r["fold"], r.get("chord_fold", 1.0)) for r in rows}, reverse=True)
    for ks, kc in wings:
        for V in sorted({r["V"] for r in rows}):
            S = [r for r in rows if r["fold"] == ks and r.get("chord_fold", 1.0) == kc and r["V"] == V]
            drag = -next(r["Fx"] for r in S if r["f"] == 0.0)
            S = [r for r in S if r["f"] > 0.0]
            k0 = [r for r in S if r["Phi_deg"] == 85.0 and r["Fx"] >= drag
                  and r["stroke_torque_peak"] <= CAP_K0]
            k1 = [r for r in S if r["Fx"] >= drag and r["stroke_torque_peak"] <= CAP_SHOULDER
                  and r["crank_torque"] <= CAP_CRANK]
            cap = [r for r in S if r["stroke_torque_peak"] <= CAP_SHOULDER]
            print(f"\nspan {ks} chord {kc} V {V}: drag {drag:.2f} N; fixed crank {len(k0)} points, "
                  f"variable throw {len(k1)} points")
            if cap:
                print("  best under the shoulder cap:", show(max(cap, key=lambda r: r["Fx"])))
            else:
                print("  no point under the shoulder cap")
            if k0:
                print("  best fixed crank:          ", show(max(k0, key=lambda r: r["Fx"])))
            if k1:
                print("  best variable throw:       ", show(max(k1, key=lambda r: r["Fx"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
