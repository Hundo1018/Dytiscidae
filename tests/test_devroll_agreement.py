"""A5 score-agreement suite (IMPL_1010 A5): the device path's scores over draws
against both host evaluation paths, with the pre-registered bar in the ``A5_*``
constants.  Split out of ``test_devroll.py`` because it takes about 9 minutes on
a GPU where the rest of that suite takes seconds.  Skips with a reason where the
device path is not available.

Run with:  PYTHONPATH=. .venv/bin/python tests/test_devroll_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


# Copied unchanged from test_devroll.py (importing it is avoided: keep in step).
def _controllers(envs, seed, *, gain=True):
    """Per world a random ``MobilityBasis`` (orthonormal modes, unit effects) and a
    per-design ``Policy`` (world 0 linear, world 1 one hidden layer), and a
    ``SharedPolicy`` whose normalisation and output layer are not the identity
    and the near-zero start, so every term reaches the command."""
    import torch

    from dytiscidae.control.cpg import MobilityBasis, Policy
    from dytiscidae.learning.ppo import SharedPolicy

    rng = np.random.default_rng(seed)
    bases, pols = [], []
    for w, e in enumerate(envs):
        P = e.cpg.n_params
        r = min(4, P)
        q, _ = np.linalg.qr(rng.normal(size=(P, r)))
        eff = rng.normal(size=(r, 6))
        eff /= np.linalg.norm(eff, axis=1, keepdims=True)
        bases.append(MobilityBasis(modes=0.2 * q.T, effects=eff,
                                   authority=rng.uniform(0.3, 3.0, r), medium="water"))
        p = Policy(n_obs=e.OBS_DIM, n_modes=r, hidden=(0 if w % 2 == 0 else 8), gain=gain)
        p.weights = rng.normal(0.0, 0.3, p.n_weights)
        pols.append(p)
    torch.manual_seed(seed)
    shared = SharedPolicy(envs[0].OBS_DIM, 6 + int(gain), hidden=16)
    with torch.no_grad():
        shared.actor[4].weight.mul_(80.0)
        shared.obs_mean.copy_(torch.as_tensor(rng.normal(0.0, 0.2, envs[0].OBS_DIM)))
        shared.obs_var.copy_(torch.as_tensor(rng.uniform(0.3, 2.0, envs[0].OBS_DIM)))
    return bases, pols, shared


def _a4_controller(p, seed):
    """A Controller for body ``p`` with one random basis per medium and a per-design
    policy (``_controllers``), and the shared policy."""
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import TriphibianEnv

    probe = [TriphibianEnv(p, seed=0) for _ in range(2)]
    # level_margin's rig leaves its machine's stroke phase offset at 0 and caches
    # its result on the phenotype: measured once here, so neither scoring of a
    # replayed machine runs it and shifts the 16-phase flap_travel the other reads.
    probe[0].level_margin()
    bases, pols, shared = _controllers(probe, seed)
    return Controller(params=None, policy=pols[1], bases={"air": bases[0], "water": bases[1]}), \
        shared


#: A5's pre-registered bar (written before the first run, 2026-10-26; a change
#: needs a ROADMAP entry with the null measurement behind it, CLAUDE.md
#: "Designing a measurement" rule 5).
A5_SECONDS = 8.0
A5_SEEDS = 8                    # draws per body: eval_seed and 7 placement draws
A5_EPS = 1e-6                   # rad on every hinge angle: the null's nudge
A5_MEAN_FLOOR = 0.005           # (i) |mean D - mean R| <= max(this, 2 |mean N - mean R|)
A5_RHO_SLACK = 0.1              # (ii) rho(D, R) >= rho(N, R) - this
A5_RHO_DISTINCT = 3             # (ii) undefined (reported) when R has fewer distinct values
A5_NEAR = 0.005                 # (iii) a draw "agrees" when |x - R| <= this
A5_SHARE_SLACK = 0.10           # (iii) share(D) >= share(N) - this
A5_START = 1e-6                 # (iv) start qpos and qvel
A5_STEP1, A5_STEP25 = 2e-4, 1e-3  # (v) A4a's measured bar, relative to the maxima
A5_SINGLE = 1e-5                # single path, air: |S - R| <= max(this, 2 |N - R|)


def _a5_spearman(a, b) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _a5_batched(phenos, ctrls, seeds, shared, *, eps=None):
    """``evaluate_tier1_batch`` at ``seeds`` (one per phenotype), identification
    off (each controller's bases are used as handed), with every air and water
    half's start state, task and first 50 recorded steps captured from
    ``rollout_batch`` and ``_record_step`` as they run.  ``eps``: every machine's
    hinge angles nudged by ``eps`` after each ``scatter`` (the same fixed nudge for
    both halves of a pair, so the pair's cancellation is untouched)."""
    import dataclasses

    import mujoco

    from dytiscidae.envs import batchroll as br
    from dytiscidae.envs.triphibian import TriphibianEnv

    calls, live = [], {}
    real_rb, real_rec, real_sc = br.rollout_batch, br._record_step, TriphibianEnv.scatter

    def rec(e, r, pos):
        live[id(e)] = r
        return real_rec(e, r, pos)

    def rb(envs, bf, duration, params_list, domain, *a, **k):
        snap = [(e.data.qpos.copy(), e.data.qvel.copy(),
                 dataclasses.astuple(e.task) if e.task is not None else None) for e in envs]
        out = real_rb(envs, bf, duration, params_list, domain, *a, **k)
        first = []
        for e in envs:
            r = live.get(id(e)) or {}
            first.append({ch: np.asarray(r.get(ch, [])[:50], float) for ch in ("alts", "ups", "xys")})
        calls.append({"dom": domain.value, "n": len(envs), "starts": snap, "first": first})
        return out

    hinged = []

    def scatter(self, rng, *a, **k):
        out = real_sc(self, rng, *a, **k)
        m = self.model
        h = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.data.qpos[h] += eps
        mujoco.mj_forward(m, self.data)
        hinged.append(len(h))
        return out

    br.rollout_batch, br._record_step = rb, rec
    if eps is not None:
        TriphibianEnv.scatter = scatter
    try:
        res = br.evaluate_tier1_batch(phenos, controllers=ctrls, seed=list(seeds),
                                      segment_seconds=A5_SECONDS, identify_axes=False,
                                      shared=shared)
    finally:
        br.rollout_batch, br._record_step, TriphibianEnv.scatter = real_rb, real_rec, real_sc
    halves = {}
    for med in ("air", "water"):
        mine = [c for c in calls if c["dom"] == med][:2]
        for h, c in enumerate(mine):
            halves[(med, h)] = c
    return res, halves, hinged


def test_the_device_path_agrees_with_both_evaluation_paths() -> None:
    """IMPL_1010 A5: the device path is a third evaluation path, and agrees with
    the batched and single-machine paths as **scores over draws**, not as
    trajectories.  A4a replays 150 steps to 1e-9 in float64, but a free-running
    float32 MJWarp segment of 2,000 steps does not track a float64 host
    trajectory (A4a: 25 steps within 2.7e-4, 200 steps up to 0.76), so the bar
    is referenced to the host's own chaos floor, as Q4 did.

    Set: the seven seed plans and the rotor quad (the spec's two arch49
    fixture elites are not in tests/fixtures, so they are not here), x 8 draws
    (``eval_seed`` 9100 + i and ``placement_draw_seed`` 1-7), 8 s, each body with
    a per-design policy, one random basis per medium and the shared policy of
    ``_a4_controller`` (seed 171).  Identification is off on every arm and every
    arm drives the same handed bases: it is not on the device (A4a), so water
    with identification, the known batched-vs-single residual, is excluded.

    Arms.  D: ``evaluate_tier1_device`` (float32, the draws at the policy mean).
    R: ``evaluate_tier1_batch`` at the same seeds.  N: R with every hinge angle
    nudged by 1e-6 rad after each scatter (the chaos floor of R itself).  S:
    ``evaluate_tier1`` on 2 bodies x 2 draws.

    Pre-registered (constants ``A5_*``), per medium (air, water) and for the
    pooled air + water competence (``loop.draw_key`` restricted to the two media
    the device runs; land and the crossings are A4b's):
      (i)   |mean D - mean R| <= max(0.005, 2 |mean N - mean R|);
      (ii)  Spearman(D, R) >= Spearman(N, R) - 0.1, undefined and reported when
            R has fewer than 3 distinct values;
      (iii) draws within 0.005 of R: D's share >= N's share - 10 points;
      (iv)  exact: every half faces the batched path's task, from its start
            qpos and qvel within 1e-6;
      (v)   the first recorded step within 2e-4 and the first 25 within 1e-3 of
            the batched record (alts, ups, xys, relative to their maxima: A4a's
            measured bar; the spec's "50 steps within 1e-4" is below what A4a
            measured at 25 steps, so 50 steps is reported, not gated);
      single path, air: |S - R| <= max(1e-5, 2 |N - R|) per draw (test_search's
            own bar between the host paths), water reported.
    """
    from dytiscidae.envs import devroll

    ok, reason = devroll.usable()
    if not ok:
        skip("the device path agrees with both evaluation paths", reason)
        return
    from dytiscidae.envs import batchroll as br

    ok, reason = br.usable()
    if not ok:
        skip("the device path agrees with both evaluation paths",
             f"the batched evaluator's GPU kernel: {reason}")
        return
    import copy
    import dataclasses
    import time

    import warp as wp

    from dytiscidae.core.bodyplans import BODY_PLANS, REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.devroll import rollout_wp as R
    from dytiscidae.envs.evaluate import SharedController, SummedPolicy, evaluate_tier1
    from dytiscidae.evolution.loop import placement_draw_seed

    names = list(BODY_PLANS) + ["quad"]
    bodies, ctrls, shared = [], [], None
    for n in names:
        p = build((BODY_PLANS.get(n) or REFERENCE_PLANS[n])())
        c, sh = _a4_controller(p, 171)          # primes level_margin on p
        shared = sh if shared is None else shared
        bodies.append(p)
        ctrls.append(c)
    draws = {i: [9100 + i] + [placement_draw_seed(9100 + i, d) for d in range(1, A5_SEEDS)]
             for i in range(len(bodies))}

    # D, one body at a time.
    t0 = time.time()
    dev = []
    for i, p in enumerate(bodies):
        dev.append(R.evaluate_tier1_device([p], controllers=[ctrls[i]], seeds=[draws[i][0]],
                                           k_mean=A5_SEEDS, segment_seconds=A5_SECONDS,
                                           shared=shared, dtype=wp.float32)[0])
    t_dev = time.time() - t0
    keys = [(i, k) for i in range(len(bodies)) for k in range(A5_SEEDS)]
    check("D draws the batched path's seeds (draw 0 = eval_seed, then placement_draw_seed)",
          all(dev[i].draws[k]["seed"] == draws[i][k] for i, k in keys))

    # R and N: one batch of every (body, draw), each machine its own phenotype copy.
    flat_p = [copy.deepcopy(bodies[i]) for i, _ in keys]
    flat_c = [ctrls[i] for i, _ in keys]
    flat_s = [draws[i][k] for i, k in keys]
    t0 = time.time()
    rres, rhalves, _ = _a5_batched(flat_p, flat_c, flat_s, shared)
    t_r = time.time() - t0
    t0 = time.time()
    nres, _, hinged = _a5_batched([copy.deepcopy(bodies[i]) for i, _ in keys], flat_c, flat_s,
                                  shared, eps=A5_EPS)
    t_n = time.time() - t0
    print(f"    D {t_dev:.0f} s, R {t_r:.0f} s, N {t_n:.0f} s; the null nudged "
          f"{min(hinged) if hinged else 0}-{max(hinged) if hinged else 0} hinges per machine")

    def comp(res, med):
        seg = (res.segments or {}).get(med)
        return float(seg.competence) if seg is not None else float("nan")

    scored = [j for j, _ in enumerate(keys) if (rres[j].segments or {}).get("air") is not None]
    check("every (body, draw) passed Tier-0 on the batched path and was scored",
          len(scored) == len(keys), f"{len(scored)}/{len(keys)}")
    cols = {}
    for med in ("air", "water"):
        cols[med] = (np.array([float(dev[keys[j][0]].draws[keys[j][1]][med].competence)
                               for j in scored]),
                     np.array([comp(rres[j], med) for j in scored]),
                     np.array([comp(nres[j], med) for j in scored]))
    cols["air+water"] = tuple(cols["air"][a] + cols["water"][a] for a in range(3))
    for med, (D, Rr, N) in cols.items():
        print(f"    {med} D " + " ".join(f"{x:.4f}" for x in D))
        print(f"    {med} R " + " ".join(f"{x:.4f}" for x in Rr))
        print(f"    {med} N " + " ".join(f"{x:.4f}" for x in N))
    fails = {"D": sum(not dev[i].draws[k][m].survived for i, k in keys for m in ("air", "water")),
             "R": sum(not rres[j].segments[m].survived for j in scored for m in ("air", "water"))}
    which = [f"{names[i]}:{k}:{m} {dev[i].draws[k][m].failure!r}" for i, k in keys
             for m in ("air", "water") if not dev[i].draws[k][m].survived]
    which_r = [f"{names[keys[j][0]]}:{keys[j][1]}:{m} {rres[j].segments[m].failure!r}"
               for j in scored for m in ("air", "water") if not rres[j].segments[m].survived]
    print(f"    failed segments (paired result): device {fails['D']} {which[:8]}, batched "
          f"{fails['R']} {which_r[:8]}; device notes: {[d.notes for d in dev if d.notes][:3]}")

    for med, (D, Rr, N) in cols.items():
        n = len(D)
        mD, mR, mN = float(D.mean()), float(Rr.mean()), float(N.mean())
        defined = len(np.unique(np.round(Rr, 9))) >= A5_RHO_DISTINCT
        sD, sN = (_a5_spearman(D, Rr), _a5_spearman(N, Rr)) if defined else (np.nan, np.nan)
        rD = int((np.abs(D - Rr) <= A5_NEAR).sum())
        rN = int((np.abs(N - Rr) <= A5_NEAR).sum())
        line = (f"{med}: device {mD:.4f} batched {mR:.4f} null {mN:.4f}; spearman {sD:.2f} "
                f"(null {sN:.2f}); within 0.005 {rD}/{n} (null {rN}/{n}); "
                f"nonzero R {int((Rr > 0).sum())}/{n}")
        bar = max(A5_MEAN_FLOOR, 2.0 * abs(mN - mR))
        check(f"(i) {med}: the device mean within max(0.005, 2 x the null's) of the batched "
              f"mean (bar {bar:.4f})", abs(mD - mR) <= bar, line)
        if defined:
            check(f"(ii) {med}: the device ranks the draws as the null does, within 0.1",
                  bool(sD >= sN - A5_RHO_SLACK), line)
        else:
            print(f"  [n/a ] (ii) {med}: Spearman undefined, R has fewer than "
                  f"{A5_RHO_DISTINCT} distinct values (reported, not gated) -- {line}")
        check(f"(iii) {med}: the share of draws within 0.005 of R at least the null's - 10 points",
              rD >= rN - A5_SHARE_SLACK * n, line)

    # (iv) and (v): the device's starts and records against the batched path's.
    start_err, task_bad, e1, e25, e50, missing = 0.0, [], ("", 0.0), ("", 0.0), 0.0, 0
    for i, p in enumerate(bodies):
        envs, _, _, _, _, index = R.body_worlds(p, ctrls[i], draws[i])
        for (k, med, half), w in index.items():
            c = rhalves.get((med, half))
            j = keys.index((i, k))
            if c is None or c["n"] != len(keys):
                missing += 1
                continue
            q, v, task = c["starts"][j]
            e = envs[w]
            start_err = max(start_err, float(np.abs(e.data.qpos - q).max()),
                            float(np.abs(e.data.qvel - v).max()))
            if dataclasses.astuple(e.task) != task:
                task_bad.append(f"{names[i]}:{k}:{med}{half}")
            rd = dev[i].recorded[(k, med, half)]
            for ch in ("alts", "ups", "xys"):
                a = np.asarray(rd[ch], float)
                b = c["first"][j][ch]
                for nn, slot in ((1, 1), (25, 25), (50, 50)):
                    if len(b) < nn or len(a) < nn:
                        continue
                    err = float(np.abs(a[:nn] - b[:nn]).max()) / max(float(np.abs(b[:nn]).max()),
                                                                    1e-9)
                    tag = f"{names[i]}:{k} {med}{half} {ch}"
                    if slot == 1 and err > e1[1]:
                        e1 = (tag, err)
                    elif slot == 25 and err > e25[1]:
                        e25 = (tag, err)
                    elif slot == 50:
                        e50 = max(e50, err)
    check("(iv) every device half faces the batched path's task from its start state within "
          "1e-6", not missing and not task_bad and start_err <= A5_START,
          f"start {start_err:.1e}; tasks differ {len(task_bad)} {task_bad[:4]}; "
          f"halves not captured {missing}")
    check("(v) the first recorded step within 2e-4 and the first 25 within 1e-3 of the batched "
          "record (relative to the maxima)", e1[1] <= A5_STEP1 and e25[1] <= A5_STEP25,
          f"step 1 worst {e1[0]} {e1[1]:.1e}; 25 steps worst {e25[0]} {e25[1]:.1e}; "
          f"50 steps {e50:.1e} (not gated)")

    # The single path, air gated, water reported: 2 bodies x 2 draws.
    for name in ("teal", "eel"):
        i = names.index(name)
        r_ = len(ctrls[i].bases["air"].modes)
        for k in (0, 1):
            j = keys.index((i, k))
            law = SharedController(params=None, bases=ctrls[i].bases,
                                   policy=SummedPolicy(own=ctrls[i].policy, shared=shared,
                                                       n_modes=r_))
            rs = evaluate_tier1(copy.deepcopy(bodies[i]), controller=law,
                                segment_seconds=A5_SECONDS, identify_axes=False,
                                seed=draws[i][k])
            sa, ra, na = comp(rs, "air"), comp(rres[j], "air"), comp(nres[j], "air")
            da = float(dev[i].draws[k]["air"].competence)
            sw, rw = comp(rs, "water"), comp(rres[j], "water")
            dw = float(dev[i].draws[k]["water"].competence)
            bar = max(A5_SINGLE, 2.0 * abs(na - ra))
            check(f"single path, {name} draw {k}: air within max(1e-5, 2 x the null's) of the "
                  f"batched path", abs(sa - ra) <= bar,
                  f"air single {sa:.5f} batched {ra:.5f} device {da:.5f} (bar {bar:.1e}); "
                  f"water single {sw:.5f} batched {rw:.5f} device {dw:.5f} (not gated)")


def main() -> int:
    functions = [test_the_device_path_agrees_with_both_evaluation_paths]
    for fn in functions:
        print(f"{fn.__name__}")
        fn()
    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED: not a defect; the device path is not built yet")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all device-agreement checks passed" + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
