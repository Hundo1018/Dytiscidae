#!/usr/bin/env python3
"""Probe: reproduce experiments/rotor/results/fly.json (stored 2026-09-26).

Calls `experiments/rotor/fly.py`'s own `fly(seed)` -- imported, not run as a
script, because its `main()` rewrites `experiments/rotor/results/fly.json` and
this experiment may not touch existing experiments' files.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/triphibian_diagnosis/existence/probes/p01_fly_repro.py 0 1 2
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rotor_fly", ROOT / "experiments/rotor/fly.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

stored = {r["seed"]: r for r in json.loads((ROOT / "experiments/rotor/results/fly.json").read_text())}
rows = []
for s in [int(x) for x in sys.argv[1:]] or [0, 1, 2]:
    t0 = time.time()
    r = mod.fly(s)
    r["wall_s"] = round(time.time() - t0, 1)
    r["stored_competence"] = stored.get(s, {}).get("competence")
    rows.append(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
(Path(__file__).resolve().parents[1] / "logs" / "p01_fly_repro.json").write_text(json.dumps(rows, indent=1))
