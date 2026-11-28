"""Pass X2: one solver-health line per case in results.jsonl (the last row per case).

    python3 experiments/flapper/x2/summary.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    last = {}
    for line in (HERE / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        last[(r.get("stage"), r["case"], r.get("mode"))] = r
    for (st, case, mode), r in last.items():
        if st == "A":
            if r.get("mode") == "steady":
                rs = r.get("steady_res", {})
                h = (f"steady {r['steady_iters']} it, CL range {r.get('steady_CL_range', float('nan')):.1e} (last 1000), "
                     f"residuals p {rs.get('p', float('nan')):.1e} Ux {rs.get('Ux', float('nan')):.1e}"
                     f"{', residualControl' if r.get('stopped_by_residualControl') else ''}")
            elif r.get("mode") == "transient":
                h = (f"transient {r['window']}, CL in [{r['CL_min']:.3f}, {r['CL_max']:.3f}], "
                     f"CD in [{r['CD_min']:.4f}, {r['CD_max']:.4f}], stop {r.get('transient_rc')}")
            else:
                h = f"FAILED ({r.get('transient_rc')}, t {r.get('t_reached')})"
            y = r.get("yplus") or r.get("yplus_steady") or {}
            print(f"{case}: CL {r.get('CL', float('nan')):.4f} CD {r.get('CD', float('nan')):.5f} | {h} | "
                  f"y+ max {y.get('max', float('nan')):.2f} mean {y.get('mean', float('nan')):.2f} | cells {r.get('cells')}")
        elif st == "B" and r.get("station"):
            y = r.get("yplus") or {}
            print(f"{case}: CN {r['CN']:.4f} CX {r['CX']:.4f} CP {r['CP']:.4f} | beat CN means "
                  f"{' '.join('%.3f' % b for b in r['beat_means_CN'])} | {r['window']} | y+ max {y.get('max', float('nan')):.2f}")
        elif st == "B":
            print(f"{case}: added mass ratio {r['ratio']:.3f}, fit residual {r['fit_resid_rms_over_amp']:.3f} of peak")


if __name__ == "__main__":
    main()
