"""Pass S1: the entrainment reaction under a driven wing in water
(PREREG_S1_entrainment.md).  Pass 1l's grid on the held rig as run, corrected,
with the solver's default reaction, with the reference added-mass load, and on
the torque-driven rig with `FluidSolver.entrainment = "wetting"`; then pass
1j's negative-power flap points.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s1_entrain.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SERIES = ("time_series", "power_series", "stroke_power_series", "stroke_torque_series")
SUB = [(0.4, 40.0, 0.75, 30.0, None, 0.0), (0.4, 40.0, 0.75, 30.0, "plate", 0.1),
       (0.4, 40.0, 0.75, 50.0, None, 0.0), (0.4, 40.0, 0.75, 50.0, "plate", 0.1),
       (0.4, 85.0, 0.5, 30.0, None, 0.0), (0.4, 85.0, 0.5, 60.0, None, 0.0),
       (0.6, 40.0, 0.5, 30.0, "plate", 0.1), (0.6, 85.0, 0.25, 60.0, None, 0.0)]
HELD = {"ctrl": dict(legacy_added_weight=True),
        "held": dict(solver_flags={"entrainment": False}),
        "heldent": dict(),
        "heldref": dict(added_mass_ref=True)}


def job(a):
    from l1a import BODY, LENGTH, finish
    from l1i_cam import cam
    arm, ks, Phi, f, Psi, law, h, flap = a
    body = dict(BODY, wing_length=ks * LENGTH)
    if arm.startswith("free") or arm == "flapfree":
        from freerig import run_free
        r = run_free(body, 0.3, 0.0, f, Phi, Psi, 0.0, medium="water", hold_beats=1, beats=2,
                     cop_law=law, feather_axis=h, one_way_flap=flap,
                     dt=1.25e-4 if arm == "freedt" else 2.5e-4,
                     solver_flags={"entrainment": "wetting"})
        r["aero_power_W"] = r["power_W"]
    else:
        from rig import run
        kw = HELD[{"flapctrl": "ctrl", "flapheld": "held"}.get(arm, arm)]
        r = run(body, 0.3, 0.0, f, Phi, Psi, 0.0, beats=3, skip=1, medium="water", cop_law=law,
                feather_axis=h, one_way_flap=flap, **kw)
        if f > 0 and flap is None:
            r = cam(finish(r, "water"))
        for key in SERIES:
            r.pop(key, None)
    r.update(arm=arm, fold=ks, law=law, axis=h)
    return r


def main() -> int:
    pts = [(ks, Phi, f, Psi, law, h) for ks in (0.4, 0.6) for Phi in (40.0, 85.0)
           for f in (0.25, 0.5, 0.75, 1.0) for Psi in (20.0, 30.0, 50.0, 60.0)
           for law, h in ((None, 0.0), ("plate", 0.1), ("insect", 0.1))]
    rest = [(ks, 40.0, 0.0, 0.0, None, 0.0) for ks in (0.4, 0.6)]
    grid = [("ctrl", *p, None) for p in SUB] + [("freedt", *p, None) for p in SUB]
    for arm in ("held", "heldent", "heldref"):
        grid += [(arm, *p, None) for p in rest + pts]
    grid += [("free", *p, None) for p in pts]
    old = [json.loads(x) for x in open(HERE / "results" / "L1j_flap_r.jsonl")]
    for r in old:
        if r["f"] > 0 and r["aero_power_W"] < 0:
            fl = dict(x_h=r["x_h"], open_deg=80.0, k_f=r["k_f"], damp_f=0.002, sign=r["sign"])
            assert r["V"] == 0.3 and r["theta_deg"] == 0.0
            for arm in ("flapctrl", "flapheld", "flapfree"):
                grid.append((arm, r["fold"], r["Phi_deg"], r["f"], r["Psi_deg"], None, 0.0, fl))
    with mp.get_context("spawn").Pool(8) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
