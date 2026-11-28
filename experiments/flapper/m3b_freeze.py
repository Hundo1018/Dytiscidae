"""Pass M3b: M3's freeze rule (m3_freeze.margins and its eligibility) on the 12/14 s reads.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m3b_freeze.py [--write]
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3_freeze import margins  # noqa: E402


def conv(r):
    if r["kind"] == "walk":
        return dict(r, along=r["along_12"], across=r["across_12"], window_s=12.0)
    return dict(r, err_last2_mean_deg=r["err_mean_14"], err_last2_max_deg=r["err_max_14"], dyaw_deg=r["dyaw_14"])


def main(write=False):
    rows = [conv(json.loads(x)) for x in open(HERE / "results" / "M3b_tuning.jsonl")]
    sets = defaultdict(list)
    for r in rows:
        if r["arm"] in ("T", "K"):
            sets[(r["arm"], json.dumps(r.get("gait", {}), sort_keys=True))].append(r)
    best = {}
    need = {"pair@7", "pair@-25", "pair@97", "turn+90", "turn-90"}
    for (arm, k), rs in sorted(sets.items()):
        m, notes = margins(rs)
        ok = need <= set(notes)
        print("%s margin %8.3f n %2d %s %s %s" % (arm, m, len(rs), "eligible" if ok else "partial  ", " ".join(sorted(notes)), k))
        if ok and (arm not in best or m > best[arm][0]):
            best[arm] = (m, k)
    print("best:", best)
    if write:
        F = {a: json.loads(v[1]) for a, v in best.items()}
        F["margin"] = {a: v[0] for a, v in best.items()}
        out = HERE / "results" / "M3b_frozen.json"
        out.write_text(json.dumps(F, indent=1, sort_keys=True))
        print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main("--write" in sys.argv)
