#!/usr/bin/env python3
"""One line per tuning step in trials.jsonl: what changed, and what it scored.

    .venv/bin/python experiments/triphibian_diagnosis/existence/probes/trials_digest.py
"""
import json
from pathlib import Path

rows = [json.loads(l) for l in open(Path(__file__).resolve().parents[1] / "trials.jsonl")]
groups, last = [], None
for r in rows:
    key = (r["trial"], json.dumps(r.get("gains", {}), sort_keys=True), r["body"])
    if key != last:
        groups.append((key, []))
        last = key
    groups[-1][1].append(r)
print(f"{len(rows)} trial rollouts in {len(groups)} steps")
for i, ((note, gains, body), rs) in enumerate(groups, 1):
    parts = []
    for r in rs:
        tag = f"{r['item']}"
        if r["item"] in ("air", "water", "land"):
            tag += f"[{r['seed']}]"
        val = f"{r['score']:.3f}"
        if "crossed" in r:
            val = ("crossed" if r["crossed"] else "no") + f" hold {r['hold']:.2f}"
        if "transitions" in r:
            val = f"{r['transitions'][0]}/{r['transitions'][1]} score {r['score']:.2f}"
        if r.get("failure"):
            val += f" ({r['failure']})"
        if int(r.get("bad_qacc", 0)) > 0:
            val += " BAD_QACC"
        parts.append(f"{tag} {val}")
    g = "" if gains == "{}" else f" gains={gains}"
    print(f"{i:2d}. {body}: {note}{g}\n      " + "; ".join(parts))
