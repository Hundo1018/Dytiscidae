"""Pass W3: measured wing sections in the fluid model (PREREG_W3_section.md).
check: the model against the separate lift runs, no simulation.
sweep: pass 1i's air grid on the seated fixture with a section.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w3_section.py check
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w3_section.py sweep OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SECTIONS = ("goe417a", "bw3", "cr001", "sd7003", "sd7037b", "e387a")


def check() -> None:
    import sections
    from dytiscidae.physics.fluid import SEPARATION_COMPLETE
    big, zero = np.array([1e6]), np.array([0.0])
    for name in SECTIONS + ("e387c", "e387e"):
        sec = sections.load(name)
        tabs = {int(round(r / 1e4)): (a, l, d) for r, a, l, d in sec.tables}
        for re_, al, cl in sections.lift_runs(name):
            key = min(tabs, key=lambda k: abs(k * 1e4 - re_))
            if abs(key * 1e4 - re_) > 1.5e4:
                continue
            a_t = tabs[key][0]
            model = np.array([sec.coefficients(np.array([x]), np.array([re_]), big, zero)[0][0] for x in al])
            inside = (al >= a_t[0]) & (al <= a_t[-1])
            past = (al > a_t[-1]) & (al <= a_t[-1] + SEPARATION_COMPLETE)
            far = al > a_t[-1] + SEPARATION_COMPLETE
            f = lambda m: "%+.3f (n %d)" % (float(np.mean(model[m] - cl[m])), int(m.sum())) if m.any() else "-"
            print("%-8s Re %6.0f: inside the drag run's range rms %.3f (n %d); model minus measured: up to 6 deg past it %s, beyond that %s" % (
                name, re_, float(np.sqrt(np.mean((model[inside] - cl[inside]) ** 2))), int(inside.sum()), f(past), f(far)))
    print("\nthree E387 models, drag runs: cd min and best L/D")
    for name in ("e387a", "e387c", "e387e"):
        for r, a, l, d in sections.load(name).tables:
            print("  %s Re %6.0f: cd min %.4f, best L/D %.1f" % (name, r, d.min(), (l / d).max()))


def job(a):
    import l1a
    section, arg = a
    l1a.BODY = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section=section)
    from l1i_cam import job as f
    r = f(arg)
    r["section"] = section
    for k in [k for k, v in r.items() if isinstance(v, list) and len(v) > 20]:
        r.pop(k)
    return r


def sweep(out) -> None:
    AX, LAW = (0.0, 0.1, 0.2), ("plate", "insect")
    rest = [("rest", th, f, Psi, h, z) for th in (45.0, 60.0, 75.0, 90.0)
            for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)
            for h in AX for z in LAW]
    cruise = [("cruise", V, f, Phi, Psi, h, z) for V in (6.0, 8.0) for f in (2.5, 3.0, 3.5, 4.0)
              for Phi in (40.0, 60.0, 85.0) for Psi in (0.0, 10.0, 20.0, 30.0)
              for h in AX for z in LAW]
    grid = [("goe417a", a) for a in rest]
    grid += [(s, a) for s in ("goe417a", "sd7003", "e387a") for a in cruise]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(out).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else sweep(sys.argv[2])
