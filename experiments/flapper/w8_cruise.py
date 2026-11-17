"""Pass W8, section 3: W4's cruise grid at 6 and 8 m/s with the tip alone
feathering, under the three treatments of the section's drag between its
drag run's end and its lift's peak (PREREG_W8_section_to_stall.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w8_cruise.py OUT.jsonl
"""
import functools
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TREAT = {"normal": "goe417a+lift", "extend": "goe417a+lift+extend", "hold": "goe417a+lift+hold"}


def job(a):
    import l1a
    import rig
    treat, arg = a
    l1a.BODY = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section=TREAT[treat])
    if not hasattr(rig, "_run0"):
        rig._run0 = rig.run
    rig.run = functools.partial(rig._run0, strip_twist=dict(tip_deg=0.0, share=1.0))
    from l1i_cam import job as f
    r = f(arg)
    r["treat"] = treat
    for k in [k for k, v in r.items() if isinstance(v, list) and len(v) > 20]:
        r.pop(k)
    return r


def main() -> int:
    cruise = [("cruise", V, f, Phi, Psi, 0.1, "plate") for V in (6.0, 8.0) for f in (2.0, 2.5, 3.0, 3.5)
              for Phi in (30.0, 40.0, 50.0, 60.0) for Psi in (0.0, 5.0, 10.0, 15.0, 20.0)]
    grid = [(t, a) for t in TREAT for a in cruise]
    if len(sys.argv) > 2 and sys.argv[2] == "b":            # addendum W8b: a larger tip feather at 6 m/s
        wide = [("cruise", 6.0, f, Phi, Psi, 0.1, "plate") for f in (2.0, 2.5, 3.0, 3.5)
                for Phi in (30.0, 40.0, 50.0, 60.0) for Psi in (25.0, 30.0, 35.0, 40.0)]
        grid = [(t, a) for t in ("normal", "extend") for a in wide]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
