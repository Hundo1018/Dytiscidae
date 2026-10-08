"""What share of a run's elites flaps in the regime the fluid model extrapolates?

ROADMAP D2 (2026-10-06 external review).  ``docs/model_validity.md`` labels
flapping at reduced frequency ``k > 0.3`` as extrapolating (the LEV term is a
fit and wake feedback is absent).  Nominal ``k = pi f c / U`` per elite:
``f`` the gait's frequency (``flap_hz``), ``c = wing_area / span`` the mean
chord, ``U`` the air segment's launch speed (``TriphibianEnv.launch_speed``,
the measured trim speed).  Elites with no lifting surface or no flapping are
counted separately.  Decision rule (written before the data, in the ROADMAP
D2 row): a small share keeps D2 documented-only.

    PYTHONPATH=. .venv/bin/python experiments/reduced_frequency_share/run.py \
        --run runs/arch48 --out experiments/reduced_frequency_share/results_arch48.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch48")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv

    rows = []
    for i, e in enumerate(load_elites(args.run)):
        m = e.meta or {}
        area, span, f = float(m.get("wing_area") or 0), float(m.get("span") or 0), float(m.get("flap_hz") or 0)
        row = {"index": i, "plan": m.get("body_plan"), "flap_hz": f, "wing_area": area,
               "span": span, "air": m.get("air"), "n_rotors": m.get("n_rotors", 0)}
        if area <= 0 or span <= 0:
            row["class"] = "no lifting surface"
        elif f <= 0:
            row["class"] = "not flapping"
        else:
            try:
                env = TriphibianEnv(build(e.genome), seed=0)
                U = float(env.launch_speed)
            except Exception as exc:          # noqa: BLE001
                row["class"], row["error"] = "could not measure", str(exc)
                rows.append(row)
                continue
            c = area / span
            k = math.pi * f * c / max(U, 1e-6)
            row.update({"chord": c, "U": U, "k": k,
                        "class": "extrapolating (k > 0.3)" if k > 0.3 else "quasi-steady (k <= 0.3)"})
        rows.append(row)
        print(f"{i:3d} {row['class']:26s} k={row.get('k', float('nan')):.3f}", flush=True)
    ks = np.array([r["k"] for r in rows if "k" in r])
    out = {"run": args.run, "elites": len(rows),
           "classes": {c: sum(r["class"] == c for r in rows) for c in sorted({r["class"] for r in rows})},
           "k_quantiles": {q: float(np.percentile(ks, q)) for q in (10, 50, 90)} if ks.size else {},
           "share_extrapolating_of_flappers": float((ks > 0.3).mean()) if ks.size else None,
           "rows": rows}
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
