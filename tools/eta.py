"""ETA of a running search from its own generation times, as a range.

    PYTHONPATH=. .venv/bin/python tools/eta.py runs/arch44 [--total 600]

Per-generation cost drifts upward (designs grow, the laptop is shared) and is
noisy, so one flat rate is always wrong.  Four estimators bracket it:
flat at the last-100 mean, the last-200 linear trend, a linear fit over the
steady state, and a quadratic fit (optimistic where cost is concave).  Report
the spread, not one number.
"""
import argparse, json, time
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--total", type=int, default=600)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.run}/generations.jsonl")]
    rows = [r for r in rows if "generation" in r]
    g = np.array([r["generation"] for r in rows]); t = np.array([r["t"] for r in rows])
    dt, gg = np.diff(t), g[1:]
    n = len(rows); ahead = np.arange(n, a.total)
    steady = gg >= 7                      # skip the six-island verification burst
    est = {
        "flat, last 100 mean": len(ahead) * dt[-100:].mean(),
        "trend, last 200": np.polyval(np.polyfit(gg[-200:], dt[-200:], 1), ahead).sum(),
        "linear, steady state": np.polyval(np.polyfit(gg[steady], dt[steady], 1), ahead).sum(),
        "quadratic, steady state": np.polyval(np.polyfit(gg[steady], dt[steady], 2), ahead).sum(),
    }
    now = time.time()
    print(f"{a.run}: gen {n}/{a.total}, {t[-1]/3600:.1f} h elapsed, last-100 {dt[-100:].mean():.0f} s/gen")
    for k, v in est.items():
        print(f"  {k:26s} {v/3600:5.1f} h left  -> {time.strftime('%a %m-%d %H:%M', time.localtime(now+v))}")
    v = sorted(est.values())
    print(f"  range {v[0]/3600:.0f}-{v[-1]/3600:.0f} h; middle two average {np.mean(v[1:3])/3600:.0f} h")


main()
