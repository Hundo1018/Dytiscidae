"""Why do gannet elites report "battery exhausted" in 51% of fresh air draws (arch54 screen)?
Replays an exhausted elite draw on the single-machine path and logs power, joint speeds, altitude."""
import json, sys
import numpy as np
sys.path.insert(0, "experiments/shared_policy_value")
from rescore import load_elites
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv
from dytiscidae.ops.run import controller_for_elite

run, rows_path, n_cases = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(rows_path)]
cases = [(r["index"], r["seed"]) for r in rows if r["arm"] == "elite" and r["failure"] == "battery exhausted" and r["plan"] == "gannet"]
seen, picks = set(), []
for i, s in cases:
    if i not in seen: seen.add(i); picks.append((i, s))
elites = load_elites(run)
for idx, seed in picks[:n_cases]:
    e = elites[idx]; p = build(e.genome)
    ctrl = controller_for_elite(run, e, p, seed, log=lambda *a, **k: None)
    env = TriphibianEnv(p, seed=seed)
    tr = []
    def on_step(env_, i):
        if i % 50: return
        b = env_.budget
        tr.append((round(float(env_.data.time), 2), round(float(b.battery.soc), 3), round(float(b.mean_power), 0),
                   round(float(np.abs(env_.data.actuator_velocity).max()), 1) if len(env_.data.actuator_velocity) else 0.0,
                   round(float(env_.root_pos()[2]), 1)))
    seg = env.rollout(8.0, params=ctrl.params, policy=ctrl.policy, basis=ctrl.basis_for(Domain.AIR), domain=Domain.AIR, on_step=on_step)
    m = e.meta or {}
    print(f"idx {idx} seed {seed} rotors {m.get('n_rotors')} mass {m.get('mass')} battery_wh {m.get('battery_wh')} "
          f"actuators {len(env.data.actuator_force)} -> '{seg.failure}' ; avionics {env.budget.avionics_w} W, p_max {round(env.budget.battery.p_max)} W")
    print("   (t, soc, mean_power W, max|joint speed| rad/s, altitude m):")
    for t in tr[::max(1, len(tr)//10)]: print("   ", t)
