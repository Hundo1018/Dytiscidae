"""A8: which measurement keys each segment of `evaluate.ladder_measurements` carries, per run.

Streams events.jsonl.  Usage: python a8_ladder_keys.py arch47 arch51 ...
Writes results/ladder_keys.json.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results" / "ladder_keys.json"


def main():
    out = json.load(open(OUT)) if OUT.exists() else {}
    for run in sys.argv[1:]:
        keys = collections.defaultdict(collections.Counter)
        promote_notes = collections.Counter()
        reeval_status = collections.Counter()
        n = 0
        with open(ROOT / "runs" / run / "events.jsonl") as f:
            for line in f:
                if '"kind": "evaluate"' in line[:40]:
                    r = json.loads(line)
                    n += 1
                    for seg, d in (r.get("ladder_measurements") or {}).items():
                        for k in d:
                            keys[seg][k] += 1
                elif '"kind": "promote"' in line[:40]:
                    r = json.loads(line)
                    for x in r.get("notes") or []:
                        promote_notes[x.split(":")[0][:60]] += 1
                elif '"kind": "reevaluate"' in line[:40]:
                    r = json.loads(line)
                    reeval_status[str(r.get("status"))] += 1
        out[run] = {"evaluate_events": n, "segments": {s: dict(c) for s, c in keys.items()},
                    "promote_note_heads": dict(promote_notes.most_common(40)),
                    "reevaluate_status": dict(reeval_status)}
        print(run, n, {s: len(c) for s, c in keys.items()}, flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
