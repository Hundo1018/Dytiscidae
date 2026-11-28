"""M4 design probe 2 (disclosed in PREREG_M4 section 0): what each lateral
input does in the first beats, free flight, all six axes.

Launch level at 8 m/s, C1's longitudinal pilot, no lateral loop, 8 beats;
then one differential input stepped on (m4_common.LAT_KEYS) and held 3 beats.
Read against the baseline (no step) from the same launch: the change in body
roll rate p (about the hull's x) and in world yaw rate r, beat means, and in
roll and yaw angle, 1, 2 and 3 beats after the step.  Plus one copy check:
the lead's probe run (3.5/30/10, roll0 5, no loop) re-flown through
m4_common for 11 beats.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m4_derivs.py OUT.jsonl [2]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEP_BEAT, AFTER = 8, 3
STEPS = dict(amp=4.0, bias=4.0, twist=3.0, feather=2.0, tail=5.0)
STEPS2 = dict(twamp=3.0, tail=5.0)       # second batch: tail re-run after the side fix (the first batch's
                                         # "tail" rows drove both halves together: a common step)


def run(a):
    pt, kind, mag, roll0, beats = a
    import m4_common as C
    try:
        mc = C.build(roll0=roll0)
        fl = C.Flight(mc, pt)
        right_y = None
        rows = []
        acc = []
        for k in range(beats * fl.n_beat):
            if kind and k == STEP_BEAT * fl.n_beat:
                fl.lat[kind] = mag
            fl.step()
            s = fl.state()
            acc.append([s["p"], s["r_world"]])
            if k % fl.n_beat == fl.n_beat - 1:
                pr = np.mean(acc, 0)
                acc = []
                rows.append([s["t"], s["x"], s["y"], s["z"], s["pitch"], s["roll"], s["yaw"], s["V"], s["tail"],
                             float(pr[0]), float(pr[1])])
            if right_y is None:
                b = mc.mj.mj_name2id(mc.m, mc.mj.mjtObj.mjOBJ_BODY, "seg1_wing")
                right_y = float(mc.d.xipos[b][1] - mc.d.xipos[mc.hull][1])
        return dict(point=list(pt[:4]), kind=kind, mag=mag, roll0=roll0, seg1_y=right_y, bad_qacc=mc.bad_qacc,
                    trace=rows)
    except Exception as e:
        return dict(point=list(pt[:4]), kind=kind, mag=mag, error=repr(e))


def main() -> int:
    import c1_pilot
    pts = sorted(c1_pilot.points(), key=lambda p: p[4])
    cheap, thrusty = pts[0], [p for p in pts if p[4] > 10.0][0]
    grid = []
    if len(sys.argv) > 2 and sys.argv[2] == "2":
        for pt in (thrusty, cheap):
            for k, v in STEPS2.items():
                grid.append((pt, k, v, 0.0, STEP_BEAT + AFTER))
    else:
        grid = [(thrusty, None, 0.0, 5.0, 11)]
        for pt in (thrusty, cheap):
            grid.append((pt, None, 0.0, 0.0, STEP_BEAT + AFTER))
            for k, v in STEPS.items():
                grid.append((pt, k, v, 0.0, STEP_BEAT + AFTER))
    with mp.get_context("spawn").Pool(4) as pool, open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in pool.imap_unordered(run, grid):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["point"][:3], r["kind"], r.get("error", "")[:200], "qacc", r.get("bad_qacc"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
