"""Trace a still machine's crossing step by step, on the numpy path (``run_transition``).

The still crossers of ``experiments/transition_distance`` (2026-10-04), traced
singly to classify how a body with its actuators held still crosses: root
position, depth, clearance, medium and ground contact per step, the hold, the
energy height lost over it, and when the crossing registered.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/still_leak/trace.py \
        --run runs/arch46 air_to_water:194:0 water_to_land:14:0 water_to_air:129:0

Each case is ``kind:elite_index:back``.  ``--rotors-on`` reproduces the
2026-10-04 still arm, which left rotors at their throttle; ``--quiet`` prints
only the summary line.
"""
import argparse
import sys

sys.path.insert(0, "experiments/shared_policy_value")
sys.path.insert(0, "experiments/transition_distance")

from rescore import load_elites  # noqa: E402
from run import still_params  # noqa: E402

from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs import transitions as T  # noqa: E402
from dytiscidae.envs.evaluate import Controller  # noqa: E402
from dytiscidae.envs.triphibian import TriphibianEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rotors-on", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("cases", nargs="+")
    args = ap.parse_args()
    elites = load_elites(args.run)

    log, made = [], {}
    observe, init = T.CrossingTracker.observe, T.CrossingTracker.__init__

    def traced_observe(self, env, i):
        observe(self, env, i)
        p = env.root_pos()
        log.append((i, float(env.data.time), p[0], p[2], env.depth(), env.clearance(),
                    T.medium_of(env).value, env._touching_ground(), T.ashore(env),
                    float(env.data.xmat[env.root_body].reshape(3, 3)[2, 2])))

    def traced_init(self, env, kind):
        init(self, env, kind)
        made["t"] = self

    T.CrossingTracker.observe, T.CrossingTracker.__init__ = traced_observe, traced_init
    for case in args.cases:
        kind, idx, back = case.split(":")
        el = elites[int(idx)]
        m = el.meta or {}
        e = TriphibianEnv(build(el.genome), seed=int(m.get("eval_seed") or 0))
        log.clear()
        prm = still_params(e) if args.rotors_on else e.held_still_params()
        r = T.run_transition(e, kind, Controller(params=prm), back=float(back))
        t = made["t"]
        eloss = t.energy_height_loss() if t.start.value == "air" else None
        print(f"== {kind} elite {idx} ({m.get('island')}) back {back}: crossed {r.crossed} "
              f"hold {r.hold:.2f} height lost {t.z0 - t.z_min:.2f} m energy height lost "
              f"{'-' if eloss is None else f'{eloss:.2f} m'} approach {r.approach:.3f} "
              f"cross_step {t.cross_step} (hold ends {t.hold_steps}) x0 {t.x0:.2f} z0 {t.z0:.2f} "
              f"'{r.failure}'")
        if args.quiet:
            continue
        n = len(log)
        picks = sorted({*range(0, n, max(n // 24, 1)), t.hold_steps - 1, t.hold_steps,
                        max(t.cross_step, 0), n - 1})
        print("   step     t        x        z   depth   clear  medium ground ashore    up")
        for j in picks:
            if j < n:
                a = log[j]
                print(f"  {a[0]:5d} {a[1]:5.2f} {a[2]:8.2f} {a[3]:8.3f} {a[4]:7.3f} {a[5]:7.3f} "
                      f"{a[6]:>6s} {int(a[7]):6d} {int(a[8]):6d} {a[9]:5.2f}")


if __name__ == "__main__":
    main()
