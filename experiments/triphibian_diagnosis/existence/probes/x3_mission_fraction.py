#!/usr/bin/env python3
"""X3 (not pre-registered): what the search's own aggregate says of this machine.

`evaluate.evaluate_tier1` builds its own environment, so a command source cannot
be handed to it.  This probe is its body, line for line (segments in
DOMAIN_CYCLE order with the same seeds and tasks, the four transitions, the
energy extrapolation), on an environment that has the hand controller installed,
and then calls the repo's own `finalise_tier1` on the result.  It prints
`mission_fraction` and each factor it is made of.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/triphibian_diagnosis/existence/probes/x3_mission_fraction.py B1 hand 0 1 2 3
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import triquad as tq  # noqa: E402
from dytiscidae.envs.evaluate import (  # noqa: E402
    TAKEOFF_FLOOR, TAKEOFF_FULL, _scatter_seed, finalise_tier1, run_segment)
from dytiscidae.envs.tasks import schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.transitions import run_transition  # noqa: E402
from dytiscidae.envs.triphibian import (  # noqa: E402
    DOMAIN_CYCLE, Domain, MissionResult, MissionSpec, evaluate_tier0)
from dytiscidae.physics.energy import transition_energy  # noqa: E402

body, arm = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]] or [0]
spec = MissionSpec()
out = HERE / "results_x3.jsonl"
for seed in seeds:
    p, env, facts = tq.make_env(body, seed)
    env.air_launch_height = spec.air_launch_height
    ctrl, hand = tq.controller_for(env, arm, False)
    tier0 = evaluate_tier0(p, spec)
    r = MissionResult(tier=1)
    r.structural_margin, r.feasible, r.air_gates = tier0.structural_margin, tier0.feasible, list(tier0.air_gates)
    clamped = False
    for dom in DOMAIN_CYCLE:
        sseed = _scatter_seed(seed, dom)
        seg = run_segment(env, dom, tq.SEGMENT_SECONDS, ctrl, scatter_seed=sseed,
                          task=schedule_for(dom, np.random.default_rng(task_seed(sseed))))
        clamped |= env.solver.diag.clamped
        r.segments[dom.value] = seg
    for kind in tq.CROSSINGS:
        tr = run_transition(env, kind, ctrl)
        r.transitions.results[kind] = tr
        r.transition_ok[kind] = tr.crossed
    cruise_j = sum(s.mean_power * spec.seconds_per_domain * spec.cycles for s in r.segments.values())
    trans_j = sum(transition_energy(p.mass, k) for k in spec.transitions)
    r.energy_required_wh = (cruise_j + trans_j) / 3600.0
    r.energy_available_wh = p.genome.battery_wh * 0.85
    finalise_tier1(r, clamped)
    comp = {k: float(s.competence) for k, s in r.segments.items()}
    tc = r.transitions.component_means()
    tf = float(tc["crossed"] * (0.40 + 0.60 * np.mean([tc[k] for k in ("shock", "control", "settle", "economy", "exit_state")])))
    th = float(r.segments["land"].measurements.get("takeoff_height", 0.0) or 0.0)
    row = {"probe": "x3_mission_fraction", "body": body, "arm": arm, "seed": seed,
           "mission_fraction": float(r.mission_fraction), "competence": comp,
           "energy_fraction": float(np.clip(r.energy_available_wh / max(r.energy_required_wh, 1e-6), 0, 1)),
           "energy_required_wh": float(r.energy_required_wh), "energy_available_wh": float(r.energy_available_wh),
           "transition_fraction": tf, "transition_components": {k: float(v) for k, v in tc.items()},
           "land_takeoff_height": th,
           "takeoff_factor": float(max(np.clip(th / TAKEOFF_FULL, 0.0, 1.0), TAKEOFF_FLOOR)),
           "clamped": bool(clamped), "exploit": r.exploit, "notes": list(r.notes),
           "diverged_rollouts": int(r.diverged_rollouts),
           "max_overload": float(max(s.max_actuator_overload for s in r.segments.values()))}
    with open(out, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                      if k not in ("transition_components", "probe")}, default=float), flush=True)
    if hand is not None:
        env.cpg.command = hand._orig
