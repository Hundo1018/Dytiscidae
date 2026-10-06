"""Does a first-order response model say anything `_turn_authority` does not?  (papers-2610, ReCo §3.2)

ReCo fits, per command channel, v' = (k*c - v)/tau.  The discrete form fitted
here, per angular channel of one segment's control decisions, is

    r[t+1] = a * r[t] + b * c[t] + e          (least squares, no intercept)

    tau = -dt / ln(a)     (0 < a < 1, else undefined and recorded as NaN)
    k   = b / (1 - a)     (same condition)
    R2_full  = 1 - SSE(a, b) / SST        SST centred on mean(r[1:])
    R2_ar    = 1 - SSE(a)    / SST        autoregression only, "momentum"
    dR2      = R2_full - R2_ar            how much the command explains beyond momentum

``c`` is the commanded body angular rate (``basis.twist_of(coeffs)[3:]``) and ``r`` the
realised one (``env.body_twist()[3:]``), both recorded per control decision by
``TriphibianEnv.rollout`` (dt = control_every * timestep, read from the env, not assumed).
``r[t]`` is measured at decision ``t`` *before* command ``t`` acts, so ``c[t]`` drives
``r[t+1]``: the same pairing ``_turn_authority`` uses.

Capture (no library change): the offline path -- ``TriphibianEnv.rollout`` under
``evaluate_tier1``'s own per-medium initial conditions (reset, scatter from the shared
seed, task from the shared seed, ``air_launch_height`` from MissionSpec) -- with
``env._score_segment`` wrapped on the *instance* to copy the ``commands``/``responses``
lists it is handed.  This is NOT the search's path (``batchroll.rollout_batch``); the
segment competence it reproduces is checked against the one the run recorded.

Still-machine rule: a machine with no controller issues no commands, so there is nothing to
fit and the measurement is NaN (not 0, which would be "measured zero").  Beside the real
dR2 each segment gets SHUFFLES copies with the command sequence permuted in time (all three
channels permuted together), the shuffled-command control.  A measurement is worth something
only to the extent it beats that.

    # stage 1: simulate and capture (one process, ~3.5 GB cap on the shared box)
    systemd-run --user --scope -q -p MemoryMax=3000M env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/step_response/run.py capture --run runs/arch48 --n 10 \
        --out experiments/_cache/step_response/captures_arch48.npz
    # stage 2: fit and correlate (no simulation, seconds)
    PYTHONPATH=. .venv/bin/python experiments/step_response/run.py analyze \
        --captures experiments/_cache/step_response/captures_arch48.npz \
        --out experiments/step_response/results_arch48.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MEDIA = ("air", "water", "land")
SHUFFLES = 50


# --------------------------------------------------------------------------- fitting


def fit_channel(c, r):
    """First-order fit of one channel.  ``c``, ``r``: (T,) command and response."""
    c, r = np.asarray(c, float), np.asarray(r, float)
    n = min(len(c), len(r)) - 1
    out = dict(a=np.nan, b=np.nan, tau=np.nan, k=np.nan, r2_full=np.nan, r2_ar=np.nan,
               dr2=np.nan, n=int(max(n, 0)))
    if n < 8:
        return out
    x_r, x_c, y = r[:n], c[:n], r[1:n + 1]
    sst = float(np.sum((y - y.mean()) ** 2))
    # No variance in the realised rate, or none in the command: nothing to explain / nothing
    # to explain it with.  NaN, not 0 -- "could not measure" is not "measured zero".
    if sst < 1e-12 or float(np.std(x_c)) < 1e-9:
        return out
    X = np.stack([x_r, x_c], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse_full = float(np.sum((y - X @ coef) ** 2))
    denom = float(x_r @ x_r)
    a_ar = float(x_r @ y / denom) if denom > 1e-12 else 0.0
    sse_ar = float(np.sum((y - a_ar * x_r) ** 2))
    a, b = float(coef[0]), float(coef[1])
    out.update(a=a, b=b, r2_full=1 - sse_full / sst, r2_ar=1 - sse_ar / sst,
               dr2=(sse_ar - sse_full) / sst)
    return out


def fit_segment(cmds, resp, dt):
    """Per-channel fits plus the segment summaries (median and max over channels)."""
    c, r = np.asarray(cmds, float), np.asarray(resp, float)
    if c.ndim != 2 or r.ndim != 2 or len(c) < 9:
        return None
    ch = [fit_channel(c[:, j], r[:, j]) for j in range(min(c.shape[1], r.shape[1]))]
    for f in ch:
        a = f["a"]
        if np.isfinite(a) and 0.0 < a < 1.0:
            f["tau"] = float(-dt / np.log(a))
            f["k"] = float(f["b"] / (1.0 - a))
    dr = np.array([f["dr2"] for f in ch], float)
    if not np.isfinite(dr).any():
        return {"channels": ch, "dr2_median": np.nan, "dr2_max": np.nan,
                "k": np.nan, "tau": np.nan, "a": np.nan, "b": np.nan, "r2_full": np.nan}
    best = int(np.nanargmax(dr))
    return {"channels": ch, "dr2_median": float(np.nanmedian(dr)), "dr2_max": float(np.nanmax(dr)),
            # k, tau, a, b of the best-explained channel (the one the command moves most)
            "k": ch[best]["k"], "tau": ch[best]["tau"], "a": ch[best]["a"], "b": ch[best]["b"],
            "r2_full": ch[best]["r2_full"]}


def shuffled_dr2(cmds, resp, dt, rng, n_shuffles):
    """dR2 (median over channels) with the command sequence permuted in time."""
    c = np.asarray(cmds, float)
    vals = []
    for _ in range(n_shuffles):
        f = fit_segment(c[rng.permutation(len(c))], resp, dt)
        if f is not None and np.isfinite(f["dr2_median"]):
            vals.append(f["dr2_median"])
    return np.array(vals, float)


# --------------------------------------------------------------------------- capture


def stage_capture(args):
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, stratified
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import evaluate as ev
    from dytiscidae.envs.tasks import schedule_for, task_seed
    from dytiscidae.envs.triphibian import DOMAIN_CYCLE, Domain, MissionSpec, TriphibianEnv, _turn_authority
    from dytiscidae.ops.run import controller_for_elite
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    seg_s = float((run_provenance(run).get("config") or {}).get("segment_seconds") or 8.0)
    elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))
    spec = MissionSpec()
    print(f"{len(elites)} elites; capturing {len(pick)}; segment_seconds {seg_s}", flush=True)

    rows, arrays, t0 = [], {}, time.time()
    for ordinal, i in enumerate(pick):
        elite = elites[i]
        meta = elite.meta or {}
        seed = int(meta.get("eval_seed") or 0)
        row = {"index": int(i), "island": meta.get("island"), "gen": meta.get("gen"),
               "born_at": getattr(elite, "born_at", None), "cell": list(elite.cell),
               "seed": seed, "recorded": {m: meta.get(m) for m in MEDIA}, "media": {}}
        try:
            p = build(elite.genome)
            ctrl = controller_for_elite(str(run), elite, p, seed, log=lambda *a, **k: None)
            summed = getattr(ctrl, "policy", None) if ctrl is not None else None
            shared = getattr(summed, "shared", None)
            laws = list(control_laws(elite, run, shared))
            desc, net = laws[0]
            if summed is not None and hasattr(summed, "shared"):
                summed.shared = net
            row["control_law"] = desc
            row["has_controller"] = ctrl is not None
            env = TriphibianEnv(p, seed=seed)
            env.air_launch_height = spec.air_launch_height
            if ctrl is None:
                ctrl = ev.Controller(params=env.cpg.base)
            if ctrl.params is None:
                ctrl.params = env.cpg.base
            # control interval, from the code rather than assumed (rollout: control_hz=25.0)
            control_hz = 25.0
            every = max(1, int(1.0 / (control_hz * env.timestep)))
            row["dt"] = float(every * env.timestep)

            cap = {}
            real_score = env._score_segment

            def grab(*a, _real=real_score, _cap=cap, **kw):
                _cap["commands"] = [np.array(x, float) for x in (kw.get("commands") or [])]
                _cap["responses"] = [np.array(x, float) for x in (kw.get("responses") or [])]
                return _real(*a, **kw)
            env._score_segment = grab

            for dom in DOMAIN_CYCLE:
                cap.clear()
                env.reset(dom)
                env.scatter(np.random.default_rng(ev._scatter_seed(seed, dom)))
                env.task = schedule_for(dom, np.random.default_rng(task_seed(ev._scatter_seed(seed, dom))))
                seg = env.rollout(seg_s, params=ctrl.params, policy=ctrl.policy,
                                  basis=ctrl.basis_for(dom), domain=dom)
                c = np.array(cap.get("commands", []), float).reshape(-1, 3)
                r = np.array(cap.get("responses", []), float).reshape(-1, 3)
                ta, rate = _turn_authority(cap.get("commands"), cap.get("responses"))
                row["media"][dom.value] = {
                    "competence": float(seg.competence), "survived": bool(seg.survived),
                    "bad_qacc": int(seg.bad_qacc), "failure": seg.failure,
                    "n_decisions": int(len(c)), "turn_authority": float(ta), "mean_rate": float(rate)}
                arrays[f"{i}|{dom.value}|c"] = c
                arrays[f"{i}|{dom.value}|r"] = r
        except Exception as exc:                                 # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"elite {ordinal + 1}/{len(pick)} idx {i} {row.get('island')} "
              f"{'ERR ' + row['error'] if 'error' in row else ''} wall {time.time() - t0:.0f}s", flush=True)
        if (ordinal + 1) % 5 == 0 or ordinal + 1 == len(pick):          # checkpoint partial work
            np.savez_compressed(args.out, meta=json.dumps({"run": str(run), "segment_seconds": seg_s,
                                "wall_s": time.time() - t0, "rows": rows}), **arrays)
    print(f"capture done: {len(rows)} elites, wall {time.time() - t0:.0f}s -> {args.out}")


# --------------------------------------------------------------------------- analysis


def _ranks(x):
    from scipy.stats import rankdata
    return rankdata(np.asarray(x, float))


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    rho, p = spearmanr(x[ok], y[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def partial_spearman(x, y, z):
    """Spearman of x's rank-residual after rank-regressing on z, against y."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5 or np.ptp(z[ok]) == 0 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    rx, ry, rz = _ranks(x[ok]), _ranks(y[ok]), _ranks(z[ok])
    Z = np.stack([np.ones_like(rz), rz], axis=1)
    res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    from scipy.stats import pearsonr
    rho, p = pearsonr(res_x, res_y)            # partial rank correlation; p ignores the 1 df spent
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def loo_r2(features, y):
    """Leave-one-out R^2 of rank-OLS of rank(y) on the rank-features (higher = predicts better)."""
    F = np.stack([_ranks(f) for f in features], axis=1)
    t = _ranks(y)
    n = len(t)
    X = np.concatenate([np.ones((n, 1)), F], axis=1)
    err = 0.0
    for k in range(n):
        m = np.arange(n) != k
        w = np.linalg.lstsq(X[m], t[m], rcond=None)[0]
        err += float((t[k] - X[k] @ w) ** 2)
    return float(1.0 - err / np.sum((t - t.mean()) ** 2))


def stage_analyze(args):
    z = np.load(args.captures, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    rng = np.random.default_rng(args.seed)
    rows = [r for r in meta["rows"] if "error" not in r]
    errors = [r for r in meta["rows"] if "error" in r]

    # Tier-2 join: the archive holds the *final* occupant of each cell.  A promote event is for
    # this elite when it names the same island and cell and was written at or after the elite
    # was born; earlier events belong to a previous occupant.
    ev_path = Path(meta["run"]) / "events.jsonl"
    promos = [json.loads(l) for l in ev_path.read_text().splitlines() if '"promote"' in l] if ev_path.exists() else []
    promos = [e for e in promos if e.get("kind") == "promote"]

    table = []
    for r in rows:
        t2 = None
        for e in sorted(promos, key=lambda e: e["gen"]):
            if e["island"] == r["island"] and list(e["cell"]) == list(r["cell"]) and r["born_at"] is not None \
                    and e["gen"] >= r["born_at"]:
                t2 = e["tier2_media"]
                break
        for m in MEDIA:
            md = r["media"].get(m)
            if md is None:
                continue
            c, resp = z[f"{r['index']}|{m}|c"], z[f"{r['index']}|{m}|r"]
            ok_seg = md["survived"] and md["bad_qacc"] == 0
            f = fit_segment(c, resp, r["dt"]) if (ok_seg and len(c) >= 9) else None
            sh = shuffled_dr2(c, resp, r["dt"], rng, args.shuffles) if f is not None and np.isfinite(f["dr2_median"]) else np.array([])
            real = f["dr2_median"] if f else np.nan
            rep = r["recorded"].get(m)
            table.append({
                "index": r["index"], "island": r["island"], "medium": m,
                "competence": md["competence"], "recorded": rep,
                "reproduced": (None if rep is None else bool(abs(md["competence"] - rep) <= 0.005)),
                "n_decisions": md["n_decisions"], "usable": f is not None, "fit_ok": bool(f and np.isfinite(f["dr2_median"])),
                "excluded_reason": None if ok_seg else ("diverged/unstable" if not md["survived"] or md["bad_qacc"] else ""),
                "turn_authority": md["turn_authority"], "mean_rate": md["mean_rate"],
                "dr2_median": real, "dr2_max": f["dr2_max"] if f else np.nan,
                "k": f["k"] if f else np.nan, "tau": f["tau"] if f else np.nan,
                "a": f["a"] if f else np.nan, "b": f["b"] if f else np.nan,
                "r2_full": f["r2_full"] if f else np.nan,
                "dr2_shuf_mean": float(sh.mean()) if len(sh) else np.nan,
                "dr2_shuf_p95": float(np.percentile(sh, 95)) if len(sh) else np.nan,
                "dr2_excess": float(real - sh.mean()) if len(sh) and np.isfinite(real) else np.nan,
                "shuf_p": float((1 + np.sum(sh >= real)) / (1 + len(sh))) if len(sh) and np.isfinite(real) else np.nan,
                "tier2_media": t2,
            })

    def col(m, key, sel=None):
        return np.array([(np.nan if x[key] is None else x[key]) for x in table if x["medium"] == m
                         and (sel is None or sel(x))], float)

    summary = {}
    for m in MEDIA:
        sel = lambda x: x["usable"]                                  # noqa: E731
        comp = col(m, "competence", sel)
        ta, dr, dx = col(m, "turn_authority", sel), col(m, "dr2_median", sel), col(m, "dr2_excess", sel)
        k, tau = col(m, "k", sel), col(m, "tau", sel)
        n_all = int(sum(1 for x in table if x["medium"] == m))
        s = {"n_elites": n_all, "n_usable": int(sel and sum(1 for x in table if x["medium"] == m and x["usable"])),
             "n_with_commands_fit": int(np.isfinite(dr).sum()),
             "n_reproduced": int(sum(1 for x in table if x["medium"] == m and x["reproduced"])),
             "n_with_recorded": int(sum(1 for x in table if x["medium"] == m and x["reproduced"] is not None)),
             "competence_ge_0.05": int((comp >= 0.05).sum()),
             "spearman_dr2_vs_competence": spearman(dr, comp),
             "spearman_dr2_excess_vs_competence": spearman(dx, comp),
             "spearman_turn_authority_vs_competence": spearman(ta, comp),
             "spearman_dr2_vs_turn_authority": spearman(dr, ta),
             "partial_dr2_given_turn_authority_vs_competence": partial_spearman(dr, comp, ta),
             "partial_turn_authority_given_dr2_vs_competence": partial_spearman(ta, comp, dr),
             "spearman_k_vs_competence": spearman(k, comp),
             "spearman_tau_vs_competence": spearman(tau, comp),
             "real_dr2_median": float(np.nanmedian(dr)) if np.isfinite(dr).any() else None,
             "shuffled_dr2_mean_of_means": float(np.nanmean(col(m, "dr2_shuf_mean", sel))) if np.isfinite(col(m, "dr2_shuf_mean", sel)).any() else None,
             "real_dr2_beats_shuffle_p<0.05": int(np.sum(col(m, "shuf_p", sel) < 0.05)),
             "k_range": [float(np.nanmin(k)), float(np.nanmax(k))] if np.isfinite(k).any() else None,
             "tau_range_s": [float(np.nanmin(tau)), float(np.nanmax(tau))] if np.isfinite(tau).any() else None,
             "tau_median_s": float(np.nanmedian(tau)) if np.isfinite(tau).any() else None,
             "share_a_in_0_1": float(np.mean([(0 < x["a"] < 1) for x in table if x["medium"] == m and np.isfinite(x["a"])]) ) if any(np.isfinite(x["a"]) for x in table if x["medium"] == m) else None}
        # Robustness cuts, same three correlations: only the segments whose offline re-score
        # reproduced the recorded (search-path) competence within 0.005, and only those with
        # competence >= 0.05 (the zero-competence ties dominate the other rows).
        rep = np.array([bool(x["reproduced"]) for x in table if x["medium"] == m and x["usable"]])
        for name, cut in (("reproduced_only", rep), ("competence_ge_0.05", comp >= 0.05)):
            s[name] = {"n": int(cut.sum()),
                       "dr2_vs_competence": spearman(dr[cut], comp[cut]),
                       "turn_authority_vs_competence": spearman(ta[cut], comp[cut]),
                       "partial_dr2_given_turn_authority": partial_spearman(dr[cut], comp[cut], ta[cut])}
        s["spearman_dr2_max_vs_competence"] = spearman(col(m, "dr2_max", sel), comp)
        s["n_no_fit"] = int(sum(1 for x in table if x["medium"] == m and not x["fit_ok"]))
        s["real_dr2_p95"] = float(np.nanpercentile(dr, 95)) if np.isfinite(dr).any() else None
        s["shuffled_dr2_p95_of_p95"] = float(np.nanpercentile(col(m, "dr2_shuf_p95", sel), 95)) if np.isfinite(col(m, "dr2_shuf_p95", sel)).any() else None
        shm = col(m, "dr2_shuf_mean", sel)
        s["median_shuffled_dr2"] = float(np.nanmedian(shm)) if np.isfinite(shm).any() else None
        s["median_dr2_excess"] = float(np.nanmedian(dx)) if np.isfinite(dx).any() else None
        both = np.isfinite(dr) & np.isfinite(ta) & np.isfinite(comp)
        if both.sum() >= 8:
            s["loo_r2_rank"] = {"turn_authority": loo_r2([ta[both]], comp[both]),
                                "dr2": loo_r2([dr[both]], comp[both]),
                                "turn_authority+dr2": loo_r2([ta[both], dr[both]], comp[both]),
                                "n": int(both.sum())}
        summary[m] = s
    # All three media pooled (segments, not independent elites; for scale only)
    out = {"captures": str(args.captures), "run": meta["run"], "capture_wall_s": meta.get("wall_s"),
           "segment_seconds": meta["segment_seconds"], "elites_captured": len(rows), "capture_errors": errors,
           "shuffles": args.shuffles, "summary": summary,
           "tier2_joined": [x for x in table if x["tier2_media"] is not None], "rows": table}
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: None))

    f = lambda d: "nan" if d["rho"] != d["rho"] else f"{d['rho']:+.2f} (p={d['p']:.3f}, n={d['n']})"   # noqa: E731
    print(f"elites {len(rows)} (errors {len(errors)}); capture wall {meta.get('wall_s', 0):.0f}s")
    hdr = ("medium", "n", "repro", "dR2 vs comp", "dR2-excess vs comp", "turn_auth vs comp", "dR2 vs turn_auth",
           "partial dR2|TA", "partial TA|dR2")
    print(" | ".join(hdr))
    for m, s in summary.items():
        print(" | ".join([m, f"{s['n_with_commands_fit']}/{s['n_elites']}", f"{s['n_reproduced']}/{s['n_with_recorded']}",
                          f(s["spearman_dr2_vs_competence"]), f(s["spearman_dr2_excess_vs_competence"]),
                          f(s["spearman_turn_authority_vs_competence"]), f(s["spearman_dr2_vs_turn_authority"]),
                          f(s["partial_dr2_given_turn_authority_vs_competence"]),
                          f(s["partial_turn_authority_given_dr2_vs_competence"])]))
    for m, s in summary.items():
        print(f"{m} cuts: reproduced-only n={s['reproduced_only']['n']} dR2 {f(s['reproduced_only']['dr2_vs_competence'])} "
              f"TA {f(s['reproduced_only']['turn_authority_vs_competence'])}; competence>=0.05 n={s['competence_ge_0.05']['n']} "
              f"dR2 {f(s['competence_ge_0.05']['dr2_vs_competence'])} TA {f(s['competence_ge_0.05']['turn_authority_vs_competence'])} "
              f"partial {f(s['competence_ge_0.05']['partial_dr2_given_turn_authority'])}; dR2max {f(s['spearman_dr2_max_vs_competence'])}; "
              f"k {f(s['spearman_k_vs_competence'])} tau {f(s['spearman_tau_vs_competence'])}; no-fit {s['n_no_fit']}; "
              f"real dR2 p95 {s['real_dr2_p95']:.4f}; median real {s['real_dr2_median']:.4f} vs median shuffled {s['median_shuffled_dr2']:.4f}")
    for m, s in summary.items():
        print(f"{m}: real dR2 median {s['real_dr2_median']}  shuffled mean {s['shuffled_dr2_mean_of_means']}  "
              f"beats shuffle p<.05 in {s['real_dr2_beats_shuffle_p<0.05']}/{s['n_with_commands_fit']}  "
              f"k {s['k_range']}  tau_s {s['tau_range_s']} (median {s['tau_median_s']})  a in (0,1) {s['share_a_in_0_1']}  "
              f"LOO {s.get('loo_r2_rank')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--run", required=True)
    c.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    c.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--captures", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--shuffles", type=int, default=SHUFFLES)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (stage_capture if args.stage == "capture" else stage_analyze)(args)


if __name__ == "__main__":
    main()
