"""What a rotor costs a worker, and whether a change to the rotor path moved
any number.

    # where a rotor's evaluation wall goes (counts, per-rotor-step cost, table builds)
    python experiments/perf/rotor_cost.py --count
    # the reference and the check, on the rotor-heavy batch
    python experiments/perf/rotor_cost.py --dump ref.json
    python experiments/perf/rotor_cost.py --compare ref.json
    # RotorSet.apply alone, per rotor-step, on the batch's machines mid-flight
    python experiments/perf/rotor_cost.py --micro

The batch is drawn from a run's final archives (default arch46, in the main
checkout), only elites carrying at least ``--min-rotors`` rotors, in a seeded
order, so the same designs come back every time.  The evaluation is the
generation's main one: batched, identification on, shared policy attached,
6 s segments (arch45/46/47).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

RUN = "/home/hundo/Projects/Dytiscidae/dytiscidae/runs/arch46"


def rotor_designs(run=RUN, n=4, seed=1, min_rotors=4, max_rotors=99):
    """``n`` elites of ``run`` carrying ``min_rotors``-``max_rotors`` rotors,
    built, in an order fixed by ``seed``."""
    from dytiscidae.core.phenotype import build
    from dytiscidae.evolution.archive import Archive
    elites = []
    for p in sorted(Path(run).glob("archive_*.pkl")):
        a = Archive.load(p)
        cells = a.cells.values() if isinstance(a.cells, dict) else a.cells
        elites.extend(e for e in cells if e is not None
                      and min_rotors <= int(e.meta.get("n_rotors", 0)) <= max_rotors)
    elites.sort(key=lambda e: (int(e.meta.get("n_rotors", 0)), float(e.fitness)))
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(elites)):
        try:
            out.append(build(elites[i].genome))
        except Exception:
            continue
        if len(out) == n:
            break
    return out


def n_rotors(p):
    return sum(1 for s in p.segments if getattr(s, "rotor", None) is not None)


def evaluate(phenos, seconds, identify=True, shared=True):
    import torch
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy
    torch.manual_seed(0)
    torch.set_num_threads(1)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1)
    net.eval()
    return batchroll.evaluate_tier1_batch(
        phenos, spec=MissionSpec(), segment_seconds=seconds, seed=5,
        identify_axes=identify, shared=net if shared else None)


def count(a):
    """Instrument `RotorSet.apply`, `rotor_forces` and `rotor_table`."""
    from dytiscidae.physics import rotor as R
    st = dict(apply_calls=0, rotor_steps=0, spinning=0, apply_s=0.0,
              forces_calls=0, forces_s=0.0, table_builds=0, table_s=0.0)
    o_apply, o_forces, o_table = R.RotorSet.apply, R.rotor_forces, R.rotor_table

    def apply(self, model, data, medium, t):
        st["apply_calls"] += 1
        st["rotor_steps"] += self.n
        st["spinning"] += int(np.sum(np.abs(data.qvel[self.dof]) >= 1e-6)) if self.n else 0
        t0 = time.perf_counter()
        out = o_apply(self, model, data, medium, t)
        st["apply_s"] += time.perf_counter() - t0
        return out

    def forces(*args, **kw):
        st["forces_calls"] += 1
        t0 = time.perf_counter()
        out = o_forces(*args, **kw)
        st["forces_s"] += time.perf_counter() - t0
        return out

    def table(spec, rho, mu):
        key = (round(spec.radius, 6), round(spec.pitch, 6), spec.blades, spec.chord_ratio,
               spec.hub_ratio, spec.camber, round(rho, 3), round(mu, 9))
        miss = key not in R._TABLES
        t0 = time.perf_counter()
        out = o_table(spec, rho, mu)
        if miss:
            st["table_builds"] += 1
            st["table_s"] += time.perf_counter() - t0
        return out

    R.RotorSet.apply, R.rotor_forces, R.rotor_table = apply, forces, table
    phenos = rotor_designs(n=a.n, seed=a.seed, min_rotors=a.min_rotors)
    print("rotors per design", [n_rotors(p) for p in phenos],
          "distinct specs", [len({(round(s.rotor.radius, 6), round(s.rotor.pitch, 6), s.rotor.blades)
                                  for s in p.segments if getattr(s, 'rotor', None) is not None})
                             for p in phenos])
    t0 = time.perf_counter()
    evaluate(phenos, a.seconds, identify=bool(a.identify))
    wall = time.perf_counter() - t0
    R.RotorSet.apply, R.rotor_forces, R.rotor_table = o_apply, o_forces, o_table
    rs = max(st["rotor_steps"], 1)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()}))
    print(f"wall {wall:.1f} s; apply {st['apply_s']:.1f} s = {st['apply_s'] / wall:.0%}; "
          f"per rotor-step {1e6 * (st['apply_s'] - st['table_s']) / rs:.1f} us excluding table builds; "
          f"spinning share {st['spinning'] / rs:.1%}; forces calls per rotor-step "
          f"{st['forces_calls'] / rs:.2f}; table builds {st['table_builds']} "
          f"costing {st['table_s']:.1f} s")


def micro(a):
    """`RotorSet.apply` alone, on states taken from a short batched rollout."""
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    phenos = rotor_designs(n=a.n, seed=a.seed, min_rotors=a.min_rotors)
    for dom in (Domain.AIR, Domain.WATER):
        tot_s, tot_rs = 0.0, 0
        for p in phenos:
            env = TriphibianEnv(p, seed=5)
            env.reset(dom)
            for _ in range(50):           # spin up, so the rotors are turning
                env.step(env.cpg.command(env.cpg.base, env.data.time))
            snap = env.snapshot()
            damping = env.model.dof_damping.copy()
            reps = a.reps
            t0 = time.perf_counter()
            for _ in range(reps):
                env.rotors.apply(env.model, env.data, env.medium, env.data.time)
            dt = time.perf_counter() - t0
            env.restore(snap)
            env.model.dof_damping[:] = damping
            tot_s += dt
            tot_rs += reps * env.rotors.n
        print(f"{dom.value}: {1e6 * tot_s / tot_rs:.1f} us per rotor-step "
              f"({tot_rs} rotor-steps)")


def dump_or_compare(a):
    from profile_shard import flatten
    phenos = rotor_designs(n=a.n, seed=a.seed, min_rotors=a.min_rotors)
    t0 = time.perf_counter()
    res = evaluate(phenos, a.seconds, identify=bool(a.identify))
    print(f"wall {time.perf_counter() - t0:.1f} s, rotors {[n_rotors(p) for p in phenos]}")
    flat = flatten(res)
    if a.dump:
        json.dump(flat, open(a.dump, "w"))
        print(f"dumped {len(flat)} numbers")
    if a.compare:
        ref = json.load(open(a.compare))
        keys = sorted(set(ref) & set(flat))
        diffs = sorted(((abs(ref[k] - flat[k]), k) for k in keys), reverse=True)
        print(f"compared {len(keys)} numbers ({len(set(ref) ^ set(flat))} keys differ); "
              f"worst {diffs[0][0]:.3g} at {diffs[0][1]}; "
              f"{sum(d > 0 for d, _ in diffs)} not bit-identical")


def floor(a):
    """The path-agreement noise floor on rotor-bearing machines, measured the
    way `tests/test_search.py` measures it (`_nudged`, and the bar in
    `test_the_two_evaluation_paths_score_the_same_machine_the_same`): the
    single path against itself under 1e-15 relative dither of the fluid
    forces, the largest of three seeds; bar = max(1e-5, 2 x floor).  Also the
    batched path against the single one, which the bar exists to judge."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.test_search import _nudged
    from dytiscidae.control.cpg import Policy
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller, evaluate_tier1
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv

    def worst_of(x, y):
        worst, key, n = 0.0, "", 0
        for dom in ("land", "air", "water"):
            sx, sy = (x.segments or {}).get(dom), (y.segments or {}).get(dom)
            if sx is None or sy is None:
                continue
            for k in sorted(set(sx.measurements or {}) & set(sy.measurements or {})):
                u, v = sx.measurements[k], sy.measurements[k]
                if isinstance(u, (int, float)) and isinstance(v, (int, float)):
                    n += 1
                    if abs(u - v) > worst:
                        worst, key = abs(u - v), f"{dom}.{k}"
        return worst, key, n

    spec, seed = MissionSpec(), 7
    genomes = [p.genome for p in rotor_designs(n=a.n, seed=a.seed, min_rotors=a.min_rotors,
                                               max_rotors=a.max_rotors)]
    rows = []
    for g in genomes:
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0)
        pol.weights = np.random.default_rng(3).normal(0.0, 0.2, pol.n_weights)

        def single():
            return evaluate_tier1(build(g), spec=spec,
                                  controller=Controller(params=None, policy=pol),
                                  segment_seconds=a.seconds, identify_axes=False, seed=seed)
        rs0 = single()
        rb0 = batchroll.evaluate_tier1_batch(
            [build(g)], spec=spec, controllers=[Controller(params=None, policy=pol)],
            segment_seconds=a.seconds, identify_axes=False, seed=seed)[0]
        fl = 0.0
        for ds in range(3):
            fl = max(fl, worst_of(rs0, _nudged(single, seed=1000 + ds))[0])
        w0, key, n = worst_of(rb0, rs0)
        row = {"rotors": n_rotors(build(g)), "floor": fl, "bar": max(1e-5, 2.0 * fl),
               "paths": w0, "paths_key": key, "n": n}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(f"floor (max over designs) {max(r['floor'] for r in rows):.3e}; "
          f"bar {max(1e-5, 2 * max(r['floor'] for r in rows)):.3e}; "
          f"paths worst {max(r['paths'] for r in rows):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--max-rotors", type=int, default=99)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-rotors", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--identify", type=int, default=1)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    if a.floor:
        floor(a)
    elif a.count:
        count(a)
    elif a.micro:
        micro(a)
    else:
        dump_or_compare(a)


if __name__ == "__main__":
    main()
