"""M4 design probe 3 (disclosed in PREREG_M4 section 0): does the lateral law's
structure fly at all, and what does the do-nothing machine do when let go over
the sea and toward the beach.  Six short runs, one set of hand-picked gains,
no tuning.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m4_smoke.py OUT.jsonl
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LAW = dict(kphi=0.1, kp=0.01, ff=-0.09, kpsi=0.5, phimax=15.0, fmax=6.0, kbeta=0.4, kr=0.15, ymax=8.0)


def run(a):
    name, pt, kw = a
    import m4_common as C
    try:
        mc = C.build(x0=kw.get("x0", -40.0), z0=kw.get("z0", 30.0), roll0=kw.get("roll0", 0.0))
        fl = C.Flight(mc, pt)
        fl.law = LAW
        fl.psi_c = 0.0
        T = kw["T"]
        n = int(round(T / mc.dt))
        every = int(round(0.05 / mc.dt))
        rec = []
        E0 = 0.0
        for k in range(n):
            t = k * mc.dt
            if "turn" in kw and abs(t - kw["turn"][0]) < 0.5 * mc.dt:
                fl.psi_c = kw["turn"][1]
            if "hold" in kw and abs(t - kw["hold"]) < 0.5 * mc.dt and fl.hold is None:
                fl.do_nothing()
            fl.step()
            if k % every == every - 1:
                s = fl.state()
                E = float(mc.E_elec.sum())
                rec.append([round(s[q], 4) for q in ("t", "x", "y", "z", "pitch", "roll", "yaw", "V", "vz", "tail")]
                           + [round(fl.track_deg(), 3), round(fl.lat["feather"], 3), round(fl.lat["twamp"], 3),
                              round(float(getattr(mc.env.solver.diag, "mean_submerged", 0.0)), 4),
                              round(mc.contact_load(), 3), round((E - E0) / 0.05, 2)])
                E0 = E
            if not np.all(np.isfinite(mc.d.qpos)):
                break
        return dict(name=name, point=list(pt[:4]), kw=kw, law=LAW, bad_qacc=mc.bad_qacc,
                    cols="t x y z pitch roll yaw V vz tail trk u_feather u_twamp wet contact P_elec".split(),
                    trace=rec)
    except Exception as e:
        import traceback
        return dict(name=name, error=traceback.format_exc()[-800:])


def main() -> int:
    import c1_pilot
    pts = sorted(c1_pilot.points(), key=lambda p: p[4])
    cheap, thrusty = pts[0], [p for p in pts if p[4] > 10.0][0]
    grid = [("H35", thrusty, dict(T=15.0, roll0=5.0)),
            ("T35+", thrusty, dict(T=15.0, turn=(3.0, 90.0))),
            ("T35-", thrusty, dict(T=15.0, turn=(3.0, -90.0))),
            ("H25", cheap, dict(T=15.0, roll0=5.0)),
            ("Nwater", thrusty, dict(T=8.0, x0=-30.0, z0=3.0, hold=2.0)),
            ("Nland", thrusty, dict(T=8.0, x0=-5.0, z0=4.5, hold=1.5))]
    with mp.get_context("spawn").Pool(4) as pool, open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in pool.imap_unordered(run, grid):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["name"], r.get("error", ""), "qacc", r.get("bad_qacc"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
