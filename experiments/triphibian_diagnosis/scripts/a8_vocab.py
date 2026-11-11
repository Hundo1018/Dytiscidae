"""A8 pass 1: vocabulary of every stored run, streamed line by line.

For each run in arch43..arch60 it records, without holding parsed rows:
  - events.jsonl: count per `kind`; top-level keys per kind (with counts);
    `status` values of `evaluate`; the `notes` vocabulary of `evaluate`
    (digits replaced by '#'); keys of `ppo` events; segment keys in
    `ladder_measurements`; `scored_by` values.
  - generations.jsonl: top-level key union (with counts) and the run_start config.
Output: results/vocab.json (+ a log on stdout).

Usage: nice -n 10 .venv/bin/python experiments/triphibian_diagnosis/scripts/a8_vocab.py [run ...]
"""
import collections
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parents[1] / "results" / "vocab.json"
DEFAULT = [f"arch{n}" for n in range(43, 61)]
DIG = re.compile(r"-?\d+(\.\d+)?(e-?\d+)?")


def norm(note):
    return DIG.sub("#", note)


def scan_events(path):
    kinds = collections.Counter()
    keys = collections.defaultdict(collections.Counter)
    status = collections.Counter()
    notes = collections.Counter()
    ppo_keys = collections.Counter()
    lm_keys = collections.Counter()
    scored_by = collections.Counter()
    bad = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            k = r.get("kind", "?")
            kinds[k] += 1
            for kk in r:
                keys[k][kk] += 1
            if k == "evaluate":
                status[str(r.get("status"))] += 1
                for n in r.get("notes") or []:
                    notes[norm(str(n))] += 1
                lm = r.get("ladder_measurements") or {}
                for seg in lm:
                    lm_keys[seg] += 1
                sb = r.get("scored_by")
                if isinstance(sb, dict):
                    for m, v in sb.items():
                        scored_by[f"{m}:{v}"] += 1
            elif k == "ppo":
                for kk, v in r.items():
                    ppo_keys[kk] += 1
                    if kk in ("reward_by_tag", "return_by_tag") and isinstance(v, dict):
                        for tag in v:
                            ppo_keys[f"{kk}.{tag}"] += 1
    return {
        "bad_lines": bad,
        "kinds": dict(kinds),
        "keys": {k: dict(c) for k, c in keys.items()},
        "evaluate_status": dict(status),
        "evaluate_notes": dict(notes.most_common(200)),
        "evaluate_notes_distinct": len(notes),
        "ppo_keys": dict(ppo_keys),
        "ladder_segments": dict(lm_keys),
        "scored_by": dict(scored_by),
    }


def scan_generations(path):
    keys = collections.Counter()
    cfg = None
    n = 0
    islands = collections.Counter()
    gens = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "run_start":
                if cfg is None:
                    cfg = r.get("config")
                keys["(run_start)"] += 1
                continue
            n += 1
            for kk in r:
                keys[kk] += 1
            islands[str(r.get("island"))] += 1
            g = r.get("generation")
            if isinstance(g, int):
                gens.append(g)
    starts = keys.get("(run_start)", 0)
    return {
        "lines": n, "run_starts": starts, "keys": dict(keys), "islands": dict(islands),
        "gen_min": min(gens) if gens else None, "gen_max": max(gens) if gens else None,
        "gen_distinct": len(set(gens)),
        "config": cfg,
    }


def main():
    runs = sys.argv[1:] or DEFAULT
    out = json.load(open(OUT)) if OUT.exists() else {}
    for run in runs:
        d = RUNS / run
        if not d.is_dir():
            print(run, "missing")
            continue
        t0 = time.time()
        rec = {}
        if (d / "generations.jsonl").exists():
            rec["generations"] = scan_generations(d / "generations.jsonl")
        if (d / "events.jsonl").exists():
            rec["events"] = scan_events(d / "events.jsonl")
        rec["files"] = sorted(p.name for p in d.iterdir() if p.is_file())
        out[run] = rec
        print(run, f"{time.time() - t0:.1f}s", rec.get("events", {}).get("kinds"), flush=True)
        OUT.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
