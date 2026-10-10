"""Q4 (ROADMAP 2026-10-10, "an evaluation that never leaves the device" §4):
how much of land competence and of the crossings is the contact model?

Each land-competent elite of a finished run (``meta.land >= 0.012``) is scored
through the single-machine path ``evaluate.evaluate_tier1`` (numpy fluid), at its
own ``eval_seed`` and ``segment_seconds``, under its own control law (own policy
+ the network that scored it), three times:

  hard   the MJCF as built (``core/mjcf.py:191-193``: default geom
         solref = [max(2*timestep, 0.004), 1] = [0.008, 1], solimp = MuJoCo
         default [0.9, 0.95, 0.001, 0.5, 2], margin 0.001, condim 4, elliptic cone)
  x2     soft contacts, dose 2: every geom's solref[0] *= 2, solimp -> [0.7, 0.875, 0.005, 0.5, 2]
  x4     soft contacts, dose 4: every geom's solref[0] *= 4, solimp -> [0.5, 0.80, 0.010, 0.5, 2]

The change is applied to the compiled ``mujoco.MjModel`` right after the env is
built and before any stepping (``geom_solref``, ``geom_solimp`` of ALL geoms,
machine and terrain alike, so the pair mix -- solmix 1 averages the two geoms --
is scaled by the same factor).  Nothing else differs between arms: same body,
same stored policy / basis / scoring network, same scatter and task draws.

Why this approximates a penalty / soft-contact engine: MuJoCo's soft constraint
is a reference acceleration a_ref = -b v - k r with k = d(r) / (dmax^2 tc^2 dr^2),
b = 2 / (dmax tc).  Raising tc by f and lowering d (the impedance, the share of the
error the solver may correct) turns the contact into a compliant spring-damper
with a stiffness that falls ~ d / (dmax^2 tc^2) and a penetration that is
millimetres rather than tenths of one: the behaviour of a penalty engine
(contact force = spring on penetration depth, + damper), which is what a
GPU-native engine would use.  ``stiffness_ratio`` below prints the factor.

    systemd-run/setsid ... env MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/contact_model_dependence/run.py --run runs/arch49 \
        --out experiments/contact_model_dependence/results_arch49.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/shared_policy_value")
MEDIA = ("air", "water", "land")
LAND_MIN = 0.012
FLOOR = 0.005          # absolute floor under the relative-change denominator
BASE_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)     # MuJoCo default, mjcf.py sets none
ARMS = {
    "hard": None,
    "x2": dict(f=2.0, solimp=(0.7, 0.875, 0.005, 0.5, 2.0)),
    "x4": dict(f=4.0, solimp=(0.5, 0.80, 0.010, 0.5, 2.0)),
    # Null control (added after the first full run showed hard did not reproduce the
    # recorded score for 27/58): a 1% change of the time constant, solimp untouched.
    # Any movement here is sensitivity of the score to a perturbation, not softness.
    "eps": dict(f=1.01, solimp=BASE_SOLIMP),
}
KINDS = ("water_to_land", "land_to_air")


def stiffness_ratio(arm):
    """k = d / (dmax^2 tc^2) of the soft arm over the hard one (per unit mass)."""
    tc0 = 0.008
    k0 = BASE_SOLIMP[1] / (BASE_SOLIMP[1] ** 2 * tc0 ** 2)
    a = ARMS[arm]
    k = a["solimp"][1] / (a["solimp"][1] ** 2 * (tc0 * a["f"]) ** 2)
    return k / k0


def soften(model, arm):
    a = ARMS[arm]
    if a is None:
        return
    model.geom_solref[:, 0] *= a["f"]
    model.geom_solimp[:, :] = np.asarray(a["solimp"])[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="first n land-competent elites (smoke)")
    ap.add_argument("--arms", default="hard,x4,x2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise-only", action="store_true",
                    help="re-aggregate <out>.rows.jsonl plus <out stem>_eps.json.rows.jsonl")
    args = ap.parse_args()
    if args.summarise_only:
        summarise(Path(str(args.out) + ".rows.jsonl"), Path(args.out), float("nan"))
        return

    import torch
    torch.set_num_threads(1)
    from rescore import load_elites
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import evaluate as ev
    from dytiscidae.ops.run import controller_for_elite
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    arms = args.arms.split(",")
    seg = float((run_provenance(run).get("config") or {}).get("segment_seconds") or 8.0)
    elites = load_elites(run)
    pick = [i for i, e in enumerate(elites) if float((e.meta or {}).get("land") or 0.0) >= LAND_MIN]
    print(f"{len(elites)} elites, {len(pick)} with meta.land >= {LAND_MIN}; segment_seconds {seg}; "
          f"stiffness ratio x2 {stiffness_ratio('x2'):.3f} x4 {stiffness_ratio('x4'):.3f}", flush=True)
    if args.n:
        pick = pick[: args.n]

    # The one place the model is touched: right after the env is built.
    current = {"arm": "hard"}
    Base = ev.TriphibianEnv

    class SoftEnv(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            soften(self.model, current["arm"])

    ev.TriphibianEnv = SoftEnv

    rows_path = Path(str(args.out) + ".rows.jsonl")
    done = set()
    if args.resume and rows_path.exists():
        done = {(r["index"], r["arm"]) for r in map(json.loads, rows_path.read_text().splitlines())}
    fh = open(rows_path, "a" if args.resume else "w")
    t0 = time.time()
    for n, i in enumerate(pick):
        el = elites[i]
        m = el.meta or {}
        seed = int(m.get("eval_seed") or 0)
        p = build(el.genome)
        ctrl = controller_for_elite(str(run), el, p, seed, log=lambda *a, **k: None)
        if ctrl is None:
            print(f"elite {i}: no controller, skipped", flush=True)
            continue
        summed = getattr(ctrl, "policy", None)
        shared = getattr(summed, "shared", None)
        desc, net = next(iter(control_laws(el, run, shared)))
        if summed is not None and hasattr(summed, "shared"):
            summed.shared = net
        for arm in arms:
            if (i, arm) in done:
                continue
            current["arm"] = arm
            t1 = time.time()
            res = ev.evaluate_tier1(p, controller=ctrl, segment_seconds=seg, seed=seed,
                                    identify_axes=False)
            land = res.segments.get("land")
            meas = dict(getattr(land, "measurements", {}) or {})
            rec = {"index": int(i), "arm": arm, "island": m.get("island"), "gen": m.get("gen"),
                   "plan": m.get("body_plan"), "seed": seed, "law": desc,
                   "recorded_land": float(m["land"]),
                   "comp": {d: float(res.segments[d].competence) for d in MEDIA if d in res.segments},
                   "land": {k: float(meas.get(k, float("nan")))
                            for k in ("land_speed", "contact_fraction", "upright",
                                      "measured_land_speed", "measured_contact_fraction")},
                   "bad_qacc": {d: int(res.segments[d].bad_qacc) for d in MEDIA if d in res.segments},
                   "crossed": {k: bool(res.transitions.results[k].crossed) for k in res.transitions.results},
                   "crossed_fraction": float(res.transitions.crossed_fraction),
                   "secs": time.time() - t1}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{n + 1}/{len(pick)}] elite {i} {arm}: land {rec['comp'].get('land', float('nan')):.4f} "
                  f"(recorded {rec['recorded_land']:.4f}) crossed {rec['crossed_fraction']:.2f} "
                  f"{rec['secs']:.0f}s total {time.time() - t0:.0f}s", flush=True)
    fh.close()
    ev.TriphibianEnv = Base
    summarise(rows_path, Path(args.out), time.time() - t0, arms)


def summarise(rows_path, out, wall, arms=("hard", "x4", "x2")):
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    eps_path = Path(str(out).replace(".json", "_eps.json") + ".rows.jsonl")
    if eps_path.exists():
        rows += [json.loads(ln) for ln in eps_path.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["index"], {})[r["arm"]] = r
    idx = sorted(i for i, a in by.items() if all(x in a for x in ("hard", "x4", "x2")))
    hard = np.array([by[i]["hard"]["comp"]["land"] for i in idx])
    rec = np.array([by[i]["hard"]["recorded_land"] for i in idx])
    repro = int((np.abs(hard - rec) <= 0.005).sum())
    out_d = {"n": len(idx), "wall_s": wall, "floor": FLOOR,
             "reproduction_hard_within_0.005": {"k": repro, "n": len(idx)},
             "stiffness_ratio": {a: stiffness_ratio(a) for a in ("x2", "x4")}, "arms": {}}
    for a in ("x4", "x2"):
        soft = np.array([by[i][a]["comp"]["land"] for i in idx])
        rel = np.abs(soft - hard) / np.maximum(hard, FLOOR)
        rel_rec = np.abs(soft - rec) / np.maximum(rec, FLOOR)
        d = {"share_lt20": float((rel < 0.2).mean()), "k_lt20": int((rel < 0.2).sum()),
             "share_gt50": float((rel > 0.5).mean()), "k_gt50": int((rel > 0.5).sum()),
             "median_abs_rel": float(np.median(rel)), "iqr_abs_rel": [float(np.percentile(rel, 25)),
                                                                        float(np.percentile(rel, 75))],
             "mean_signed_rel": float(np.mean((soft - hard) / np.maximum(hard, FLOOR))),
             "mean_comp_hard": float(hard.mean()), "mean_comp_soft": float(soft.mean()),
             "share_lt20_vs_recorded": float((rel_rec < 0.2).mean()),
             "share_gt50_vs_recorded": float((rel_rec > 0.5).mean()),
             "spearman_hard_soft": _spearman(hard, soft)}
        for k in ("land_speed", "contact_fraction", "upright"):
            h = np.array([by[i]["hard"]["land"][k] for i in idx])
            s = np.array([by[i][a]["land"][k] for i in idx])
            d[k] = {"hard_mean": float(h.mean()), "soft_mean": float(s.mean()),
                    "median_abs_change": float(np.median(np.abs(s - h)))}
        for kind in KINDS:
            h = np.array([by[i]["hard"]["crossed"].get(kind, False) for i in idx])
            s = np.array([by[i][a]["crossed"].get(kind, False) for i in idx])
            d[f"crossed_{kind}"] = {"hard": int(h.sum()), "soft": int(s.sum()),
                                    "changed": int((h != s).sum())}
        d["crossed_fraction"] = {"hard_mean": float(np.mean([by[i]["hard"]["crossed_fraction"] for i in idx])),
                                 "soft_mean": float(np.mean([by[i][a]["crossed_fraction"] for i in idx])),
                                 "changed": int(sum(by[i]["hard"]["crossed_fraction"] != by[i][a]["crossed_fraction"]
                                                    for i in idx))}
        d["bad_qacc_land"] = {"hard": int(sum(by[i]["hard"]["bad_qacc"].get("land", 0) for i in idx)),
                              "soft": int(sum(by[i][a]["bad_qacc"].get("land", 0) for i in idx))}
        out_d["arms"][a] = d
    # ---- the null control and the subsets (added after the first full run) ----
    def shares(sel, arm):
        sel = [i for i in sel if arm in by[i]]
        if not sel:
            return None
        rel = np.array([abs(by[i][arm]["comp"]["land"] - by[i]["hard"]["comp"]["land"])
                        / max(by[i]["hard"]["comp"]["land"], FLOOR) for i in sel])
        return {"n": len(sel), "share_lt20": float((rel < 0.2).mean()),
                "share_gt50": float((rel > 0.5).mean()), "median_abs_rel": float(np.median(rel))}
    repro_set = [i for i in idx if abs(by[i]["hard"]["comp"]["land"] - by[i]["hard"]["recorded_land"]) <= 0.005]
    gate = Path("experiments/no_model_gate/results_arch49.json.rows.jsonl")
    s51 = set()
    if gate.exists():
        s51 = {r["index"] for r in map(json.loads, gate.read_text().splitlines())
               if r["arm"] == "elite" and r["comp"].get("land", 0) >= 0.012}
    out_d["subsets"] = {
        name: {a: shares(sel, a) for a in ("eps", "x2", "x4")}
        for name, sel in (("all_58", idx), ("hard_reproduces_recorded", repro_set),
                          ("batched_land_ge_0.012_(the_ROADMAP_51)", [i for i in idx if i in s51]))}
    out_d["null_control_eps_solref_x1.01"] = out_d["subsets"]["all_58"]["eps"]
    out_d["per_elite"] = [{"index": i, "recorded": by[i]["hard"]["recorded_land"],
                           "hard": by[i]["hard"]["comp"]["land"], "x2": by[i]["x2"]["comp"]["land"],
                           "x4": by[i]["x4"]["comp"]["land"],
                           "eps": by[i].get("eps", {}).get("comp", {}).get("land")} for i in idx]
    out.write_text(json.dumps(out_d, indent=1))
    print(json.dumps({k: v for k, v in out_d.items() if k != "per_elite"}, indent=1))


def _spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)[0]
    return float(r)


if __name__ == "__main__":
    main()
