"""Where a Tier-1 evaluation's physics steps and wall time go, by phase.

    python experiments/perf/phase_budget.py [--n 4] [--seconds 8] [--refine 2]
                                            [--shared 1] [--out budget.json]

Runs the single-machine path (`evaluate.evaluate_tier1`, numpy fluid, no GPU)
on the same designs `profile_shard.py` builds, and attributes every
`mujoco.mj_step` call and every second of wall time to the innermost phase it
ran in:

    identify       TriphibianEnv.identify (both media)
    segment        TriphibianEnv.rollout outside identification
    transition     evaluate.run_transition
    level_margin   TriphibianEnv.level_margin (rig, cached per phenotype)
    thrust_margin  TriphibianEnv.thrust_margin
    other          everything else inside evaluate_tier1

Step counts are path-independent: the batched path runs the same experiments
(`batchroll.identify_batch` mirrors `TriphibianEnv.identify`, probe for probe).
Wall time is the numpy path's and is not the GPU path's cost per step; use it
only for the ratio between phases on this path.

Then it scales the measured steps to one search generation per design:
the main evaluation (with identification), the noise-free re-score when a
shared policy is on (`_refine_controllers`), and `--refine` (1+1)-ES steps,
each of which is a full evaluation without identification.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class PhaseMeter:
    """Exclusive wall time and mj_step counts per phase, via a phase stack."""

    def __init__(self):
        self.stack = ["other"]
        self.steps = defaultdict(int)
        self.wall = defaultdict(float)
        self._t = time.perf_counter()

    def _flush(self):
        now = time.perf_counter()
        self.wall[self.stack[-1]] += now - self._t
        self._t = now

    def enter(self, name):
        self._flush()
        self.stack.append(name)

    def leave(self):
        self._flush()
        self.stack.pop()

    def wrap(self, owner, attr, name):
        fn = getattr(owner, attr)
        meter = self

        def wrapped(*a, **kw):
            meter.enter(name)
            try:
                return fn(*a, **kw)
            finally:
                meter.leave()

        setattr(owner, attr, wrapped)
        return fn

    def count_steps(self, module):
        fn = module.mj_step
        meter = self

        def wrapped(m, d, *a, **kw):
            meter.steps[meter.stack[-1]] += 1
            return fn(m, d, *a, **kw)

        module.mj_step = wrapped
        return fn


def install(meter):
    import mujoco

    from dytiscidae.envs import evaluate
    from dytiscidae.envs.triphibian import TriphibianEnv

    meter.count_steps(mujoco)
    meter.wrap(TriphibianEnv, "identify", "identify")
    meter.wrap(TriphibianEnv, "rollout", "segment")
    meter.wrap(TriphibianEnv, "level_margin", "level_margin")
    meter.wrap(TriphibianEnv, "thrust_margin", "thrust_margin")
    meter.wrap(evaluate, "run_transition", "transition")


def generation_budget(per_design, *, identify_steps, shared, refine):
    """Steps per design per generation, from one measured evaluation.

    ``per_design`` maps phase -> steps of one evaluation with identification.
    An evaluation without identification costs the same minus ``identify``.
    """
    without = {k: v for k, v in per_design.items() if k != "identify"}
    repeats = (1 if shared else 0) + refine
    out = {"main.identify": identify_steps}
    for k, v in without.items():
        out[f"main.{k}"] = v
    if repeats:
        out["rescore_and_refine"] = repeats * sum(without.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--refine", type=int, default=2,
                    help="controller_refine_steps of the run being modelled")
    ap.add_argument("--shared", type=int, default=1,
                    help="1 if the run uses a shared policy (adds one re-score)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from profile_shard import designs

    from dytiscidae.envs.evaluate import evaluate_tier1
    from dytiscidae.envs.triphibian import MissionSpec

    phenos = designs(a.n, a.seed)
    meter = PhaseMeter()
    install(meter)

    per_design = []
    for p in phenos:
        before_s, before_w = dict(meter.steps), dict(meter.wall)
        t = time.perf_counter()
        evaluate_tier1(p, spec=MissionSpec(), segment_seconds=a.seconds,
                       identify_axes=True, seed=a.eval_seed)
        meter._flush()
        row = {"wall": time.perf_counter() - t,
               "steps": {k: meter.steps[k] - before_s.get(k, 0) for k in meter.steps},
               "phase_wall": {k: meter.wall[k] - before_w.get(k, 0.0) for k in meter.wall}}
        per_design.append(row)
        print(f"design {len(per_design)}: wall {row['wall']:.1f} s, steps "
              + ", ".join(f"{k} {v}" for k, v in sorted(row["steps"].items())))

    total_steps = sum(meter.steps.values())
    total_wall = sum(meter.wall.values())
    print("\nphase          steps    share   wall s   share   us/step")
    for k in sorted(meter.steps, key=lambda k: -meter.steps[k]):
        s, w = meter.steps[k], meter.wall[k]
        print(f"{k:13s} {s:7d}  {s / total_steps:6.1%}  {w:7.1f}  {w / total_wall:6.1%}"
              f"  {1e6 * w / max(s, 1):8.0f}")
    for k in sorted(set(meter.wall) - set(meter.steps)):
        print(f"{k:13s} {0:7d}  {0:6.1%}  {meter.wall[k]:7.1f}  "
              f"{meter.wall[k] / total_wall:6.1%}")

    mean = {k: meter.steps[k] / a.n for k in meter.steps}
    gen = generation_budget(mean, identify_steps=mean.get("identify", 0),
                            shared=bool(a.shared), refine=a.refine)
    g_total = sum(gen.values())
    print(f"\none generation, per design (shared={a.shared}, refine={a.refine}):")
    for k, v in sorted(gen.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:9.0f} steps  {v / g_total:6.1%}")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "args": vars(a), "per_design": per_design,
            "steps": dict(meter.steps), "wall": dict(meter.wall),
            "generation_per_design": gen}, indent=2))


if __name__ == "__main__":
    main()
