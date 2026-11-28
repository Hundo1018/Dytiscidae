"""Pass M2b: GW / GS read on tuning-trial pairs (same seed, halves told h / h + pi).

    m2b_pairs.py PREFIX_A PREFIX_B [gs]     # note prefixes of the two halves
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m2_read import gate_gw, gate_gs
rows = {json.loads(l)["spec"]["note"].split()[0]: json.loads(l) for l in open(HERE / "results" / "M2b_trials.jsonl")}
a, b = rows[sys.argv[1]], rows[sys.argv[2]]
g = gate_gs(a, b) if len(sys.argv) > 3 else gate_gw(a, b, a["draw"]["delta"])
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k in ("P", "R", "clauses", "met")},
      "A", {k: round(v, 3) for k, v in g["A"].items() if isinstance(v, float)}, "B", {k: round(v, 3) for k, v in g["B"].items() if isinstance(v, float)})
