#!/usr/bin/env python3
"""X1 (not pre-registered): the four crossings from eight different entry states.

`transitions.run_transition` draws every machine's entry state for a crossing
kind from one fixed seed (`transition_scatter_seed(kind)`), so the PREREG's
"8 seeds" are eight copies of one rollout.  This probe asks the question the
seeds were meant to ask: it replaces that one function, in this process only,
by a seed-dependent draw of the same distribution (`env.scatter`, unchanged),
and runs the same crossing, the same `CrossingTracker` and the same controller.
Seed 0 is the repo's own draw.  Nothing else is touched; rows go to
results_x1.jsonl and are never mixed into the table.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/triphibian_diagnosis/existence/probes/x1_entry_states.py B1 hand
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import triquad as tq  # noqa: E402
import dytiscidae.envs.transitions as T  # noqa: E402

body = sys.argv[1] if len(sys.argv) > 1 else "B1"
arm = sys.argv[2] if len(sys.argv) > 2 else "hand"
orig = T.transition_scatter_seed
out = HERE / "results_x1.jsonl"
for seed in tq.SEEDS:
    T.transition_scatter_seed = (lambda kind, _s=seed: (orig(kind) + 7919 * _s) & 0x7FFFFFFF)
    p, env, facts = tq.make_env(body, seed)
    for kind in tq.CROSSINGS:
        ctrl, hand = tq.controller_for(env, arm, False)
        row = tq.run_cross_item(env, ctrl, kind)
        if hand is not None:
            env.cpg.command = hand._orig
        row = {"probe": "x1_entry_states", "body": body, "arm": arm, "item": kind, "seed": seed, **row}
        with open(out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({k: row[k] for k in ("body", "arm", "item", "seed", "crossed", "hold",
                                              "failure", "bad_qacc")}
                         | {"min_upright": row["m"]["min_upright"],
                            "entry_speed": row["m"]["peak_entry_speed"]}), flush=True)
T.transition_scatter_seed = orig
