"""P0.2 measurement: hold drift, end dwell and attitude of every crossing arm.

No learning.  Runs the four crossings from eight entry states on B0, B1 (arms
hand, base, blind, still) and L1 (arms walk, still) through the repo's
``run_transition`` and writes the raw ``TransitionResult`` beside the row, so
the P0.2 thresholds are set from what the arms measure (CLAUDE.md rule 5, 9).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/scripts/crossing_telemetry.py OUT.jsonl
"""
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "existence"))

import landgait as lg  # noqa: E402
import nulls as nl  # noqa: E402
import triquad as tq  # noqa: E402

KEEP = ("crossed", "failure", "bad_qacc", "hold", "hold_drift", "end_dwell", "min_upright",
        "approach", "shore_progress", "go_peak_clearance", "aloft_fraction", "exit_upright",
        "started_in")
last = {}
_run = tq.run_transition


def capture(*a, **k):
    r = _run(*a, **k)
    last["r"] = r
    return r


tq.run_transition = capture


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    plan = [(b, a) for b in ("B1", "B0") for a in ("hand", "base", "blind", "still")]
    plan += [("L1", "walk"), ("L1", "still")]
    for body, arm in plan:
        for kind in tq.CROSSINGS:
            for entry in range(8):
                _, env, _ = tq.make_env(body, 0)
                last.clear()
                try:
                    if body == "L1":
                        row = lg.run_item(env, body, arm, kind, 0, entry=entry)
                    else:
                        row = nl.run_row(env, body, arm, kind, entry)
                except Exception as exc:  # a broken rollout is a row, not a crash
                    row = {"failure": f"exception: {exc!r}"}
                rec = {"body": body, "arm": arm, "kind": kind, "entry": entry,
                       "row_failure": row.get("failure", ""), "row_bad_qacc": row.get("bad_qacc")}
                if "r" in last:
                    d = dataclasses.asdict(last["r"])
                    rec.update({k: d[k] for k in KEEP})
                with open(out, "a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                print(json.dumps(rec, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
