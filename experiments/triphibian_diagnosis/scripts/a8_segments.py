"""A8 pass A: the surviving timeline of every run.

A resumed run restarts from its last checkpoint, so generations between that
checkpoint and the stop were run twice and the first copy was discarded.  Both
streams are append-only:
  - generations.jsonl marks each process with a `{"kind": "run_start"}` line;
  - events.jsonl has no marker, but its `t` (seconds since that process's
    telemetry was created, ops/telemetry.py `t0 = time.time()`) restarts near 0.
A process segment s keeps generation g iff g < min(first generation of every
later segment).  Everything downstream counts the kept lines only and reports
the discarded ones separately.

Output: results/segments.json  {run: {"gen_segments": [...], "event_segments": [...],
         "gen_keep_below": [...], "event_keep_below": [...], ...}}
Usage: nice -n 10 .venv/bin/python experiments/triphibian_diagnosis/scripts/a8_segments.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parents[1] / "results" / "segments.json"
DEFAULT = [f"arch{n}" for n in range(43, 61)]
INF = 10 ** 9


def keep_below(segments):
    """For each segment, the generation below which its lines survive."""
    out = []
    for i in range(len(segments)):
        later = [s["gen_min"] for s in segments[i + 1:] if s["gen_min"] is not None]
        out.append(min(later) if later else INF)
    return out


def gen_segments(path):
    segs = []
    cur = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "run_start":
                cur = {"gen_min": None, "gen_max": None, "lines": 0, "t_max": 0.0,
                       "config_generations": (r.get("config") or {}).get("generations"),
                       "resume": (r.get("config") or {}).get("resume")}
                segs.append(cur)
                continue
            if cur is None:  # lines before any run_start
                cur = {"gen_min": None, "gen_max": None, "lines": 0, "t_max": 0.0}
                segs.append(cur)
            g = r.get("generation")
            cur["lines"] += 1
            cur["t_max"] = max(cur["t_max"], float(r.get("t") or 0.0))
            if isinstance(g, int):
                cur["gen_min"] = g if cur["gen_min"] is None else min(cur["gen_min"], g)
                cur["gen_max"] = g if cur["gen_max"] is None else max(cur["gen_max"], g)
    return segs


def event_segments(path):
    segs = []
    cur = None
    t_prev = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = float(r.get("t") or 0.0)
            if cur is None or (t_prev is not None and t < t_prev - 1.0):
                cur = {"gen_min": None, "gen_max": None, "lines": 0, "t_first": t, "t_max": t}
                segs.append(cur)
            t_prev = t
            cur["lines"] += 1
            cur["t_max"] = max(cur["t_max"], t)
            g = r.get("gen")
            if isinstance(g, int):
                cur["gen_min"] = g if cur["gen_min"] is None else min(cur["gen_min"], g)
                cur["gen_max"] = g if cur["gen_max"] is None else max(cur["gen_max"], g)
    return segs


def main():
    runs = sys.argv[1:] or DEFAULT
    out = json.load(open(OUT)) if OUT.exists() else {}
    for run in runs:
        t0 = time.time()
        d = RUNS / run
        gs = gen_segments(d / "generations.jsonl")
        es = event_segments(d / "events.jsonl")
        rec = {"gen_segments": gs, "gen_keep_below": keep_below(gs),
               "event_segments": es, "event_keep_below": keep_below(es)}
        out[run] = rec
        print(run, f"{time.time() - t0:.1f}s",
              "gen segs", [(s["gen_min"], s["gen_max"], s["lines"]) for s in gs],
              "event segs", [(s["gen_min"], s["gen_max"], s["lines"]) for s in es], flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
