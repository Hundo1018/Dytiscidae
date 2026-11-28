#!/usr/bin/env python3
"""Probe: the controller's mode changes over one item (time, mode, position).

    ... probes/p07_modes.py B1 mission_water 0 [gains-json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triquad as tq  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

body, item, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
gains = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
p, env, facts = tq.make_env(body, seed)
ctrl, hand = tq.controller_for(env, "hand", False, gains)
act, last = hand._act, [None]


def traced():
    out = act()
    d = env.data
    cm = hand.policy.commanded
    key = (hand.mode, None if cm is None else cm.value)
    if key != last[0]:
        last[0] = key
        print(f"t={d.time:7.3f} {hand.mode:>16s} cmd={'-' if cm is None else cm.value:5s} "
              f"p=({d.qpos[0]:8.2f},{d.qpos[1]:6.2f},{d.qpos[2]:6.2f}) "
              f"v=({d.qvel[0]:5.2f},{d.qvel[1]:5.2f},{d.qvel[2]:5.2f}) clr={env.clearance():6.2f}")
    return out


hand._act = traced
if item in ("air", "water", "land"):
    row = tq.run_seg_item(env, ctrl, Domain(item), seed)
elif item in tq.CROSSINGS:
    row = tq.run_cross_item(env, ctrl, item)
else:
    row = tq.run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
print(json.dumps({k: row[k] for k in ("score", "failure", "bad_qacc", "transitions", "crossed", "hold", "legs") if k in row}))
