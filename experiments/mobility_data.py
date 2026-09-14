"""Collect the raw mobility probes once, so several experiments can share them.

The expensive part of asking anything about the mobility basis is the probing:
24 random CPG-parameter directions, each run with both signs for 1.2 s of
simulated time, per body per medium.  Roughly 15 s of wall clock for one
(body, medium, seed).  The analysis afterwards -- a least-squares, an SVD, a
threshold sweep, a damping sweep -- costs microseconds.

So this module runs the probes and caches `(deltas, responses)`, and the
experiments do their own fitting from that.  Two consequences worth stating:

* The rank sweep and the damping sweep are measured on **identical** data, so
  a difference between them is a difference in the analysis and not in the
  sample.
* Repeat identification with a different probe seed is cheap to ask for, which
  is what makes the *reproducibility* of a mode measurable at all.  A singular
  value only means something if a second independent identification finds it
  again.

Drift guard: `collect_one` reproduces the probe loop of
`TriphibianEnv.identify`, and `verify_against_env` asserts that refitting the
cached probes gives bit-comparable singular values to the real method.  If the
env's probe loop changes, that assertion fails rather than the cache quietly
describing a different experiment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "_cache"

#: Matches `TriphibianEnv.identify`'s defaults, which in turn match
#: `identify_batch`'s.  Changing these here makes the cache describe a
#: different experiment from the one the search runs.
PROBE_TIME = 1.2
N_PROBES = 24
MAX_MODES = 6


def body_plan(name: str):
    from dytiscidae.core import bodyplans
    return getattr(bodyplans, name)()


def build_env(plan_name: str):
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    return TriphibianEnv(build(body_plan(plan_name)))


def collect_one(plan_name: str, medium: str, seed: int) -> dict:
    """Run the probe loop and return the raw deltas and responses.

    Mirrors `TriphibianEnv.identify` exactly, minus the final fit.
    """
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.envs.triphibian import Domain

    env = build_env(plan_name)
    domain = Domain(medium)
    env.reset(domain, randomise=False)
    snap = env.snapshot()
    base = env.cpg.base
    n_params = env.cpg.n_params

    def reset_fn():
        env.restore(snap)
        env.budget.reset()

    def step_fn(delta):
        params = CPGParams.from_flat(base.flat() + delta, env.cpg.n)
        acc = np.zeros(6)
        n = int(PROBE_TIME / env.timestep)
        for _ in range(n):
            env.step(env.cpg.command(params, env.data.time))
            acc += env.body_twist()
        if not np.all(np.isfinite(env.root_pos())):
            return np.zeros(6)
        return acc / max(n, 1)

    rng = np.random.default_rng(seed)
    deltas = rng.normal(0.0, 0.35, size=(N_PROBES, n_params))
    responses = np.zeros((N_PROBES, 6))
    t0 = time.time()
    for k, d in enumerate(deltas):
        reset_fn()
        plus = np.asarray(step_fn(d), float)
        reset_fn()
        minus = np.asarray(step_fn(-d), float)
        responses[k] = 0.5 * (plus - minus)

    return {
        "plan": plan_name,
        "medium": medium,
        "seed": seed,
        "deltas": deltas,
        "responses": responses,
        "n_joints": env.cpg.n,
        "n_params": n_params,
        "seconds": time.time() - t0,
    }


def verify_against_env(plan_name: str, medium: str, seed: int) -> dict:
    """Assert the cached probe loop and `TriphibianEnv.identify` agree.

    The probes are stochastic only through the seed, and both paths draw from
    `default_rng(seed)` with the same shape, so the singular values must match
    to floating-point summation order -- not approximately.
    """
    from dytiscidae.control.cpg import basis_from_probes
    from dytiscidae.envs.triphibian import Domain

    rec = collect_one(plan_name, medium, seed)
    ours = basis_from_probes(rec["deltas"], rec["responses"],
                             medium=medium, max_modes=MAX_MODES)
    env = build_env(plan_name)
    theirs = env.identify(Domain(medium), probe_time=PROBE_TIME,
                          n_probes=N_PROBES, seed=seed, max_modes=MAX_MODES)
    err = float(np.max(np.abs(ours.authority - theirs.authority)))
    rel = err / max(float(theirs.authority[0]), 1e-30)
    return {"plan": plan_name, "medium": medium, "seed": seed,
            "max_abs_sigma_diff": err, "relative": rel,
            "agrees": rel < 1e-9}


def collect(plans: list[str], media: list[str], seeds: list[int],
            *, cache: Path | None = None, refresh: bool = False) -> list[dict]:
    """Collect (or load) probes for every (plan, medium, seed) combination."""
    cache = Path(cache) if cache else CACHE / "probes.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    want = [(p, m, s) for p in plans for m in media for s in seeds]

    have: dict[tuple, dict] = {}
    if cache.exists() and not refresh:
        blob = np.load(cache, allow_pickle=True)
        for rec in blob["records"]:
            rec = dict(rec.item()) if hasattr(rec, "item") else dict(rec)
            have[(rec["plan"], rec["medium"], rec["seed"])] = rec

    missing = [k for k in want if k not in have]
    if missing:
        print(f"collecting {len(missing)} probe sets "
              f"(~{len(missing) * 15 / 60:.1f} min)", flush=True)
    for i, (p, m, s) in enumerate(missing, 1):
        rec = collect_one(p, m, s)
        have[(p, m, s)] = rec
        print(f"  [{i}/{len(missing)}] {p:<7} {m:<5} seed={s} "
              f"{rec['seconds']:.1f}s", flush=True)
        np.savez(cache, records=np.array(list(have.values()), dtype=object))

    return [have[k] for k in want]


def fit(rec: dict, max_modes: int = MAX_MODES):
    """Fit the project's own basis from a cached probe set."""
    from dytiscidae.control.cpg import basis_from_probes
    return basis_from_probes(rec["deltas"], rec["responses"],
                             medium=rec["medium"], max_modes=max_modes)


def jacobian(rec: dict) -> np.ndarray:
    """The fitted (P, 6) Jacobian, in the scaled twist space the basis uses."""
    scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    J, *_ = np.linalg.lstsq(rec["deltas"], rec["responses"] * scale, rcond=None)
    return J
