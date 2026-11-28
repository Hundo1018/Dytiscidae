#!/usr/bin/env python3
"""Probe: step-by-step trace of one item under the hand controller.

    ... probes/p02_trace.py B0 air 0 [every_n_steps] [t_max] [gains-json]

Prints time, mode, position, velocity, up-vector z, body rates, rotor speeds
and wetness.  Uses the same harness functions as `triquad.py run`; the trace is
taken from inside the controller's own call, so it adds no rollout.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triquad as tq  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

body, item, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 25
t_max = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
gains = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None

p, env, facts = tq.make_env(body, seed)
ctrl, hand = tq.controller_for(env, "hand", False, gains)
act = hand._act
count = [0]


def traced():
    out = act()
    d = env.data
    if count[0] % every == 0 and d.time <= t_max:
        R = d.xmat[env.root_body].reshape(3, 3)
        s = env.medium.properties(d.xipos[hand.rot], np.full(hand.n, 0.01), d.time)[2]
        cm = hand.policy.commanded
        print(f"t={d.time:6.3f} {hand.mode:>16s} cmd={'-' if cm is None else cm.value[:1]} "
              f"p=({d.qpos[0]:8.3f},{d.qpos[1]:7.3f},{d.qpos[2]:7.3f}) "
              f"v=({d.qvel[0]:6.2f},{d.qvel[1]:6.2f},{d.qvel[2]:6.2f}) up={R[2,2]:5.2f} "
              f"w=({d.qvel[3]:6.2f},{d.qvel[4]:6.2f},{d.qvel[5]:6.2f}) "
              f"om_cmd=[{' '.join(f'{x:5.0f}' for x in out)}] "
              f"om=[{' '.join(f'{d.qvel[k]:5.0f}' for k in env.rotors.dof)}] "
              f"wet={s.mean():.2f} clr={env.clearance():6.3f} touch={int(env._touching_ground())} "
              f"soc={env.budget.battery.soc:.3f}")
    count[0] += 1
    return out


hand._act = traced
if item in ("air", "water", "land"):
    row = tq.run_seg_item(env, ctrl, Domain(item), seed)
elif item in tq.CROSSINGS:
    row = tq.run_cross_item(env, ctrl, item)
else:
    row = tq.run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
print(json.dumps({k: v for k, v in row.items() if k != "facts"}, default=float))
