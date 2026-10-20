"""Air list item 3 (ROADMAP 2026-10-20): why does a held-still machine report
"battery exhausted" inside an 8 s air segment?  Replays still-arm draws that
did (from screen.rows.jsonl) on the single-machine numpy path and logs when the
pack goes flat and what the actuators were doing."""
import json, sys
import numpy as np
sys.path.insert(0, "experiments/shared_policy_value")
from rescore import load_elites
from dytiscidae.core.phenotype import build
from dytiscidae.envs.evaluate import Controller, run_segment
from dytiscidae.envs.triphibian import Domain, TriphibianEnv

run = sys.argv[1]; rows_path = sys.argv[2]; n_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = [json.loads(l) for l in open(rows_path)]
cases = [(r["index"], r["seed"], r["plan"]) for r in rows if r["arm"] == "still" and r["failure"] == "battery exhausted"][:n_cases]
elites = load_elites(run)
for idx, seed, plan in cases:
    e = elites[idx]; p = build(e.genome)
    env = TriphibianEnv(p, seed=seed)
    ctrl = Controller(params=env.held_still_params())
    trace = []
    def on_step(env_, i):
        b = env_.budget
        trace.append((round(env_.data.time, 3), b.battery.soc, float(np.abs(env_.data.actuator_force).max()) if len(env_.data.actuator_force) else 0.0,
                      float(np.abs(env_.data.actuator_velocity).max()) if len(env_.data.actuator_velocity) else 0.0,
                      float(env_.root_pos()[2]), str(env_.medium)))
    seg = run_segment(env, Domain.AIR, 8.0, ctrl, on_step=on_step) if "on_step" in run_segment.__code__.co_varnames else None
    if seg is None:
        seg = env.rollout(8.0, params=ctrl.params, domain=Domain.AIR, on_step=on_step)
    meta = e.meta or {}
    print(f"idx {idx} seed {seed} plan {plan} rotors {meta.get('n_rotors')} mass {meta.get('mass')} battery_wh {meta.get('battery_wh')} "
          f"actuators {len(env.data.actuator_force)} -> failure '{seg.failure}' duration {seg.duration} steps {len(trace)}")
    if trace:
        flat = next((t for t in trace if t[1] <= 0.0), None)
        print("   first step:", trace[0]); print("   last step: ", trace[-1])
        if flat: print("   flat at:", flat)
        peak = max(trace, key=lambda t: t[2]); print("   peak |actuator_force|:", peak)
        print("   soc at 1,2,4 s:", [round(next((t[1] for t in trace if t[0] >= s), -1), 3) for s in (1, 2, 4)])
        print("   mean power W:", round(env.budget.mean_power, 1), "avionics W:", env.budget.avionics_w, "p_max W:", round(env.budget.battery.p_max, 1))
