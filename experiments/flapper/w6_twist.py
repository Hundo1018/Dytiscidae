"""Pass W6: a spanwise twist that is a control (PREREG_W6_twist.md).  The
Goe 417a rectangle on the seated fixture, held rig, W4's cruise and take-off
grids, with the tip's held twist and the feather's spanwise share varied
(`rig.run(strip_twist=...)`).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w6_twist.py OUT.jsonl
"""
import functools
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def job(a):
    import l1a
    import rig
    tw, share, arg = a
    l1a.BODY = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift")
    if not hasattr(rig, "_run0"):
        rig._run0 = rig.run
    rig.run = functools.partial(rig._run0, strip_twist=dict(tip_deg=tw, share=share))
    from l1i_cam import job as f
    r = f(arg)
    r.update(twist=tw, share=share)
    for k in [k for k, v in r.items() if isinstance(v, list) and len(v) > 20]:
        r.pop(k)
    return r


def main() -> int:
    cruise = [("cruise", V, f, Phi, Psi, 0.1, "plate") for V in (6.0, 8.0) for f in (2.0, 2.5, 3.0, 3.5)
              for Phi in (30.0, 40.0, 50.0, 60.0) for Psi in (0.0, 5.0, 10.0, 15.0, 20.0)]
    rest = [("rest", th, f, Psi, 0.1, "plate") for th in (45.0, 60.0, 75.0, 90.0)
            for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)]
    grid = [(tw, sh, a) for tw in (0.0, -4.0, -8.0, -12.0, -16.0) for sh in (0.0, 1.0) for a in cruise]
    grid += [(tw, sh, a) for tw in (8.0, 4.0, 0.0, -4.0, -8.0) for sh in (0.0, 0.5, 1.0) for a in rest]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
