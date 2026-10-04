"""Write tests/fixtures/arch46_still_crossers.pkl: every body that crossed held still in
experiments/transition_distance/results.json (2026-10-04), with its eval seed and the
first start distance it crossed from.  Usage: make_fixture.py [runs/arch46]
"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, "experiments/shared_policy_value")
from rescore import load_elites

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/arch46"
# (kind, index, backs) -- every still crossing body of experiments/transition_distance (2026-10-04)
CASES = {
    "air_to_water": [194, 6, 33, 8, 72],
    "water_to_air": [129, 137],
    "water_to_land": [14, 48, 25, 52, 50, 11, 129, 146, 177, 75, 186, 112, 116, 39, 180, 28, 24],
}
import json
d = json.load(open("experiments/transition_distance/results.json"))
first = {}
for r in d["transition_rows"]:
    if r["arm"] == "still" and r["crossed"]:
        k = (r["kind"], r["index"])
        first[k] = min(first.get(k, 99.0), r["back"])
el = load_elites(RUN)
out = []
for kind, ids in CASES.items():
    for i in ids:
        m = el[i].meta or {}
        out.append({"kind": kind, "index": i, "island": m.get("island"), "back": first[(kind, i)],
                    "eval_seed": int(m.get("eval_seed") or 0), "genome": el[i].genome})
Path("tests/fixtures").mkdir(exist_ok=True)
Path("tests/fixtures/arch46_still_crossers.pkl").write_bytes(pickle.dumps(out))
print(len(out), Path("tests/fixtures/arch46_still_crossers.pkl").stat().st_size)
