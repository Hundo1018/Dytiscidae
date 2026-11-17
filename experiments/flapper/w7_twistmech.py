"""Pass W7: the tip's twist as a mechanism (PREREG_W7_twistmech.md).  The
wing of pass W6 built as a chain of panels on span hinges, on the
torque-driven rig with the body held: the hinges driven one by one, driven
by one torque from root to tip through a torsionally soft wing, or left to
springs alone; in cruise and at take-off.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w7_twistmech.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CM = dict(camber_moment=True)
KS_TUBE = (0.3, 1.0, 3.0, 10.0, 30.0)
KS_PASSIVE = (1.0, 3.0, 10.0, 30.0, 100.0)
CRUISE = ((2.5, 40.0, 15.0), (2.0, 50.0, 15.0), (3.0, 40.0, 20.0))       # f, stroke, tip feather
TAKEOFF = ((4.5, 50.0), (4.0, 50.0))                                       # f, feather; body pitch 90, stroke 85


def job(a):
    import l1a
    from freerig import run_free
    kind, arm, pt, bkw, kw = a
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift", **bkw)
    kw = dict(kw)
    if kind == "takeoff":
        f, Psi = pt
        r = run_free(body, 0.0, 90.0, f, 85.0, Psi, 0.0, feather_axis=0.1, cop_law="plate", **kw)
        r.update(kind=kind, arm=arm, point=list(pt))
        return r
    f, Phi, tipA = pt
    Psi = kw.pop("Psi_hinge", 0.0)

    def go(x):
        return run_free(body, 8.0, 0.0, f, Phi, Psi, x, feather_axis=0.1, cop_law="plate", **kw)
    # the cam offset trimmed to lift = weight, as pass 1i's cruise job does
    x0, x1 = -3.0, -1.0
    r0, r1 = go(x0), go(x1)
    best, evals = min((r0, r1), key=lambda r: abs(r["lift_over_weight"] - 1.0)), 2
    while abs(best["lift_over_weight"] - 1.0) > 0.01 and evals < 8:
        y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
        if abs(y1 - y0) < 1e-9:
            break
        x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), -30.0), 25.0)
        r2 = go(x2)
        evals += 1
        x0, r0, x1, r1 = x1, r1, x2, r2
        if abs(r2["lift_over_weight"] - 1.0) < abs(best["lift_over_weight"] - 1.0):
            best = r2
    best = dict(best)
    best.update(kind=kind, arm=arm, point=list(pt), evals=evals,
                trimmed=bool(abs(best["lift_over_weight"] - 1.0) <= 0.03))
    best["sustains"] = bool(best["trimmed"] and best["Fx"] >= 0.0 and best["finite"])
    return best


def main() -> int:
    grid = []
    for pt in CRUISE:
        A = pt[2]
        st = dict(strip_twist=dict(tip_deg=0.0, share=1.0), Psi_hinge=A)
        grid.append(("cruise", "R strips", pt, {}, dict(st)))
        grid.append(("cruise", "Rm strips +Cm", pt, {}, dict(st, solver_flags=CM)))
        for n in (3, 6):
            grid.append(("cruise", "S%d servo" % n, pt, dict(wing_panels=n),
                         dict(chain=dict(mode="servo", tip_deg=A), solver_flags=CM)))
        grid.append(("cruise", "S6 servo, no Cm", pt, dict(wing_panels=6), dict(chain=dict(mode="servo", tip_deg=A))))
        for k in KS_TUBE:
            grid.append(("cruise", "T6 tube k %g" % k, pt, dict(wing_panels=6),
                         dict(chain=dict(mode="tube", tip_deg=A, k=k), solver_flags=CM)))
        for k in KS_PASSIVE:
            grid.append(("cruise", "P6 passive k %g" % k, pt, dict(wing_panels=6),
                         dict(chain=dict(mode="passive", k=k), solver_flags=CM)))
    for pt in TAKEOFF:
        grid.append(("takeoff", "K1 one panel", pt, {}, dict(solver_flags=CM)))
        grid.append(("takeoff", "L6 servo at 0", pt, dict(wing_panels=6),
                     dict(chain=dict(mode="servo", tip_deg=0.0), solver_flags=CM)))
        for k in KS_TUBE:
            grid.append(("takeoff", "T6 tube k %g" % k, pt, dict(wing_panels=6),
                         dict(chain=dict(mode="tube", tip_deg=0.0, k=k), solver_flags=CM)))
        for k in KS_PASSIVE:
            grid.append(("takeoff", "P6 passive k %g" % k, pt, dict(wing_panels=6),
                         dict(chain=dict(mode="passive", k=k), solver_flags=CM)))
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
