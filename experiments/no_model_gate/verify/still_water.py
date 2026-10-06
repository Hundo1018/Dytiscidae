"""Judge's check of the still arm's water passers (offline numpy TriphibianEnv path).

Water segment only, reproducing the batched path's RNG order: fresh env,
reset(AIR) (consumes env.rng exactly as the batched path does before water),
then reset(WATER), scatter(_scatter_seed(seed, WATER)), task from task_seed.

Variants:
  asis       scatter as the search does it
  noscatter  scatter skipped entirely
  rest       scatter kept, then all qvel zeroed (machine starts at rest)
  rest_pose  as rest, and actuated joints placed at the held-still command
             (removes the initial servo transient from scatter's gait pose)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/shared_policy_value")


def late_cruise_v(pos, env, res, dt):
    t = env.task
    n = len(pos)
    for (lo, hi), ph in zip(t.bounds(max(int(round(env._seg_T / dt)), 1)), t.phases):
        if ph.kind == "cruise":
            mid = lo + (hi - lo) // 2
            top = min(hi, n)
            v = (pos[top - 1, :2] - pos[mid, :2]) / ((top - 1 - mid) * dt)
            h = np.array([np.cos(ph.heading), np.sin(ph.heading)])
            return {"window_s": [round(mid * dt, 2), round(top * dt, 2)], "v_xy": v.round(4).tolist(),
                    "along": round(float(v @ h), 4), "vz": round(float((pos[top - 1, 2] - pos[mid, 2]) / ((top - 1 - mid) * dt)), 4),
                    "depth_window": [round(float(-pos[mid, 2]), 2), round(float(-pos[top - 1, 2]), 2)]}


def main():
    idxs = [int(x) for x in sys.argv[1].split(",")]
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["asis"]
    from rescore import load_elites
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import _scatter_seed
    from dytiscidae.envs.tasks import schedule_for, task_seed
    from dytiscidae.envs.triphibian import Domain, MissionSpec, TriphibianEnv
    from dytiscidae.viz.film import run_provenance

    run = Path("runs/arch48")
    cfg = run_provenance(run).get("config") or {}
    seg_s = float(cfg.get("segment_seconds") or 8.0)
    spec = MissionSpec()
    elites = load_elites(run)
    rows = {}
    for ln in open("experiments/no_model_gate/results_arch48.json.rows.jsonl"):
        r = json.loads(ln)
        rows[(r["index"], r["arm"])] = r
    for i in idxs:
        e_ = elites[i]
        m = e_.meta or {}
        seed = int(m.get("eval_seed") or 0)
        p = build(e_.genome)
        rec = rows[(i, "still")]
        assert rec["plan"] == (m.get("body_plan") or "?") and rec["gen"] == int(m.get("gen") or 0)
        for var in variants:
            env = TriphibianEnv(p, seed=seed)
            env.air_launch_height = spec.air_launch_height
            still = env.held_still_params()
            env.reset(Domain.AIR)                      # burn env.rng as the batched path does
            env.reset(Domain.WATER)
            ss = _scatter_seed(seed, Domain.WATER)
            if var != "noscatter":
                env.scatter(np.random.default_rng(ss))
            if var in ("rest", "rest_pose"):
                env.data.qvel[:] = 0.0
            if var == "rest_pose" and len(env._act_qadr):
                env.data.qpos[env._act_qadr] = np.clip(still.offset, env.cpg.lo, env.cpg.hi)
            if var in ("rest", "rest_pose"):
                env._mj.mj_forward(env.model, env.data)
            env.task = schedule_for(Domain.WATER, np.random.default_rng(task_seed(ss)))
            log = dict(ctrl=[], jv=[], pos=[], dep=[])

            def hook(en, k):
                if var == "rotorlock" and en.rotors.n:   # judge's probe: hold every rotor at zero spin
                    for d_ in en.rotors.dof:
                        en.data.qvel[d_] = 0.0
                log["ctrl"].append(en.data.ctrl.copy())
                log["jv"].append(en.data.qvel[en._act_vadr].copy() if len(en._act_vadr) else np.zeros(0))
                log["pos"].append(en.root_pos().copy())
                log.setdefault("ncon", []).append(int(en.data.ncon))
                log.setdefault("ground", []).append(bool(en._touching_ground()))
                log.setdefault("ang", []).append(float(np.linalg.norm(en.data.qvel[3:6])))
                if en.rotors.n:
                    log.setdefault("thr", []).append(np.asarray(en.rotors.last_thrust, float).copy())

            res = env.rollout(seg_s, params=still, policy=None, basis=None,
                              domain=Domain.WATER, on_step=hook)
            ctrl = np.array(log["ctrl"])
            jv = np.abs(np.array(log["jv"]))
            pos = np.array(log["pos"])
            dt = env.timestep
            n05 = int(0.5 / dt)
            dctrl = float(np.abs(np.diff(ctrl, axis=0)).max()) if len(ctrl) > 1 and ctrl.shape[1] else 0.0
            ph = env.task.phases
            cr = next(q for q in ph if q.kind == "cruise")
            meas = res.measurements
            out = {
                "index": i, "variant": var, "plan": rec["plan"], "seed": seed, "seg_s": seg_s,
                "n_act": int(len(env.act_names)),
                "rotors": sum(n.endswith("_r") for n in env.act_names),
                "density_ratio": round(float(p.density_ratio), 4),
                "mass": round(float(p.mass), 3),
                "recorded_still_water": rec["comp"]["water"],
                "reproduced_water": float(res.competence),
                "parts": {k: round(float(v), 4) for k, v in (res.parts or {}).items()},
                "task_score": meas.get("task_score"),
                "cruise_progress": meas.get("cruise_progress"),
                "cruise_depth_hold": meas.get("cruise_depth_hold"),
                "cruise_tracking": meas.get("cruise_tracking"),
                "hold_score": meas.get("hold_score"),
                "hold_depth_error": meas.get("hold_depth_error"),
                "hold_drift": meas.get("hold_drift"),
                "task_hold_first": meas.get("task_hold_first"),
                "cmd_depth": meas.get("cmd_depth"),
                "cmd_heading_deg": round(float(np.degrees(cr.heading)), 1),
                "max_abs_dctrl": dctrl,
                "max_joint_vel_all": float(jv.max()) if jv.size else 0.0,
                "max_joint_vel_after_0.5s": float(jv[n05:].max()) if jv.size else 0.0,
                "ctrl_unique_rows": int(len(np.unique(np.round(ctrl, 9), axis=0))) if ctrl.size else 0,
                "start_pos": pos[0].round(3).tolist(), "end_pos": pos[-1].round(3).tolist(),
                "horiz_disp": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
                "disp_heading_deg": float(np.degrees(np.arctan2(*(pos[-1, :2] - pos[0, :2])[::-1]))),
                "depth_start_end": [round(float(-pos[0, 2]), 3), round(float(-pos[-1, 2]), 3)],
                "bad_qacc": int(res.bad_qacc),
                "joint_maxvel_after_0.5s": {n: round(float(jv[n05:, k].max()), 3) for k, n in enumerate(env.act_names)} if jv.size else {},
                "ground_contact_frac": float(np.mean(log["ground"])),
                "first_ground_t": (float(np.argmax(log["ground"]) * dt) if any(log["ground"]) else None),
                "max_body_angvel_after_0.5s": float(np.max(log["ang"][n05:])),
                "rotor_thrust_N": ({"max_abs": round(float(np.abs(log["thr"]).max()), 4),
                                    "mean": round(float(np.mean(log["thr"])), 4)} if "thr" in log else None),
                "late_cruise_v": late_cruise_v(pos, env, res, dt),
            }
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
