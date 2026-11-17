"""Pass D2: what turns the root feather at take-off
(`PREREG_D2_feather_drive.md`, addendum D2a).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d2_run.py all OUT.jsonl
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d2_run.py fx2 OUT.jsonl

Every row: take-off, the stroke by D1's four-bar at lever 60 mm, gear 24
unless stated.  F0 the ideal servo; FM a second motor at gears {12, 16, 24,
32}; FX the arm on the stroke crank (as frozen); FX2 the feather crank geared
1:1 (stroke gears 16, 24, 32); FP a spring at {0.5, 1, 2, 4, 8} N m/rad.  In
a vacuum (G1): F0, every FM gear, FX2 at gear 24.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
LEVER = 0.06
FM_GEARS = (12, 16, 24, 32)
SPRINGS = (0.5, 1.0, 2.0, 4.0, 8.0)


def job(a):
    import d1_run as R
    import l1a
    from drive import Drive
    from freerig import run_free
    from dytiscidae.physics.medium import Fluid
    cand, N, fe, vacuum = a
    V, th, f, Phi, Psi, psi0, _, kw = R.POINTS["takeoff"]
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift")
    kw = dict(kw, feather_axis=0.1, cop_law="plate", dt=R.DT)
    if vacuum:
        kw["fluid"] = Fluid("vacuum", 1e-9, 1.81e-5)
    head = dict(cand=cand, N=N, feather=fe, vacuum=vacuum)
    try:
        dv = Drive("FB", Phi, f, N, a=LEVER, dt=R.DT, feather_axis=0.1, feather=fe)
        r = run_free(body, V, th, f, Phi, Psi, psi0, drive=dv, **kw)
    except Exception as e:                       # a row, not a crash
        return dict(head, error=repr(e))
    r.update(head)
    # the machine's electrical power: every motor plus the ideal servo's net
    # power where the feather is on the servo (PREREG_D2 1)
    r["machine_W"] = r["drive"]["machine_elec_W"] + (r["feather_power_W"] if fe is None else 0.0)
    return r


def any_job(a):
    """D2c mixes D2's rows (the feather motor) with D1's (DD, FB)."""
    if a[0] == "d1":
        import d1_run
        return dict(d1_run.job(a[1]), cand="D1")
    return job(a)


D2B = [(30, 80, 1.0), (30, 80, .85), (30, 120, 1.0), (30, 120, .85), (30, 160, 1.0), (30, 160, .85),
       (45, 80, .85), (45, 120, 1.0), (45, 120, .85), (45, 160, 1.0), (45, 160, .85),
       (60, 80, .85), (60, 120, .85), (60, 160, 1.0), (60, 160, .85)]          # horn mm, drop mm, arm scale


def main() -> int:
    grid = [("F0", 24, None, False), ("F0", 24, None, True)]
    grid += [("FM", 24, dict(mode="motor", N=n), v) for n in FM_GEARS for v in (False, True)]
    grid += [("FX", 24, dict(mode="crank", c_h=0.03), False)]
    grid += [("FX2", n, dict(mode="crank2", c_h=0.03), False) for n in (16, 24, 32)]
    grid += [("FX2", 24, dict(mode="crank2", c_h=0.03), True)]
    grid += [("FP", 24, dict(mode="passive", k=k), False) for k in SPRINGS]
    if sys.argv[1] == "fx2":     # FX2's rows again, after the horn anchor fix (RESULT_D2)
        grid = [g for g in grid if g[0] == "FX2"]
        grid += [("FX2 stop 85", 24, dict(mode="crank2", c_h=0.03, range_deg=85.0), v) for v in (False, True)]
    if sys.argv[1] == "d2b":     # PREREG_D2bc: FX2's geometry
        grid = [("FX2", 24, dict(mode="crank2", c_h=c * 1e-3, drop=dr * 1e-3, arm_scale=sc), v)
                for c, dr, sc in D2B for v in (False, True)]
    if sys.argv[1] == "d2c":     # PREREG_D2bc: the rotor's inertia
        grid = []
        for J in (5e-6, 1e-6):
            mo = dict(J_r=J)
            grid += [("FM", 24, dict(mode="motor", N=n, motor=mo), False) for n in (16, 24, 32)]
            grid += [("d1", ("takeoff", "DD", n, 0.0, dict(motor=mo))) for n in (12, 16, 24, 32)]
            grid += [("d1", ("cruise", "DD", n, 0.0, dict(motor=mo))) for n in (32, 48, 64, 90)]
            grid += [("d1", ("takeoff", "FB", 24, 0.0, dict(motor=mo, a=LEVER))),
                     ("d1", ("cruise", "FB", 48, 0.0, dict(motor=mo)))]
    out = Path(sys.argv[-1]).resolve()
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if sys.argv[2].isdigit() else 6) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(any_job, grid):
            r.pop("series", None)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print("rows", len(grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
