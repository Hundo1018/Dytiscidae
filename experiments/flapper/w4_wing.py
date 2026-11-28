"""Pass W4: the wing's planform, twist and section, with gentler strokes
(PREREG_W4_wing.md).  Seated fixture, held rig, pass 1i's cruise and take-off
jobs with the body replaced.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w4_wing.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SHAPES = [(lam, tw) for lam in (1.0, 0.6, 0.4) for tw in (0.0, -8.0)]


def job(a):
    import l1a
    section, lam, tw, arg = a
    if section is None:
        l1a.BODY = dict(l1a.BODY_S)
    else:
        l1a.BODY = dict(l1a.BODY_S, wing_shape=dict(camber=0.04, taper=lam, twist_tip_deg=tw),
                        wing_section=section)
    from l1i_cam import job as f
    r = f(arg)
    r.update(section=section, taper=lam, twist=tw)
    for k in [k for k, v in r.items() if isinstance(v, list) and len(v) > 20]:
        r.pop(k)
    return r


def main() -> int:
    cruise = [("cruise", V, f, Phi, Psi, 0.1, "plate") for V in (6.0, 8.0) for f in (2.0, 2.5, 3.0, 3.5)
              for Phi in (30.0, 40.0, 50.0, 60.0) for Psi in (0.0, 5.0, 10.0, 15.0, 20.0)]
    rest = [("rest", th, f, Psi, 0.1, "plate") for th in (45.0, 60.0, 75.0, 90.0)
            for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)]
    grid = [(None, 1.0, 0.0, a) for a in cruise]
    grid += [("goe417a", lam, tw, a) for lam, tw in SHAPES for a in cruise + rest]
    grid += [("e387a", lam, tw, a) for lam, tw in SHAPES if lam > 0.5 for a in cruise]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
