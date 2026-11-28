"""Repair R3's checks T1-T5 and T7 (PREREG_R3_table.md): the table law
against the runs it is built from, the section it must equal, Ananda's
flat-plate wings, and a section it does not use.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/physics_extremes/r3_check.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import sections  # noqa: E402  (run.py puts experiments/flapper on the path)
from dytiscidae.physics import thin_plate_tables as T  # noqa: E402
from dytiscidae.physics.fluid import skin_friction_cd, table_coefficients  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent / "tools"))
import bake_thin_plate as B  # noqa: E402


def law(a_rad, re, camber):
    """Steady, two-dimensional, no vortex."""
    a = np.atleast_1d(np.asarray(a_rad, float))
    cl, cd, _ = table_coefficients(a, np.full(a.shape, float(re)), np.full(a.shape, 1e12), 0.0, None, camber)
    return cl, cd


def t1():
    print("T1 the anchors reproduce their runs")
    worst = 0.0
    for re_, a, cl in B.flat_runs(sections):
        worst = max(worst, float(np.max(np.abs(law(a, re_, 0.0)[0] - cl))))
    R.record("T1", "flat plate, lift runs (odd, zero-shifted)", worst, "< 0.01", 0.01, worst < 0.01)
    w_in = w_d = w_past = 0.0
    lifts = {round(r, -4): (a, c) for r, a, c in sections.lift_runs("goe417a")}
    for re_, a, cl, cd in sections.tables("goe417a"):
        m_cl, m_cd = law(a, re_, T.CAMBER)
        w_in = max(w_in, float(np.max(np.abs(m_cl - cl))))
        w_d = max(w_d, float(np.max(np.abs(m_cd - cd))))
        la, lc = lifts[round(re_, -4)]
        keep = la > a[-1]
        w_past = max(w_past, float(np.max(np.abs(law(la[keep], re_, T.CAMBER)[0] - lc[keep]))))
    R.record("T1", "Goe 417a, lift inside the drag runs", w_in, "< 0.01", 0.01, w_in < 0.01)
    R.record("T1", "Goe 417a, drag runs", w_d, "< 0.002", 0.002, w_d < 0.002)
    R.record("T1", "Goe 417a, lift runs past the drag runs", w_past, "< 0.01", 0.01, w_past < 0.01)


def t2():
    print("T2 at the arc's camber the law is the section")
    sec = sections.load("goe417a+lift+extend")
    a = np.radians(np.arange(-10.0, 40.01, 0.25))
    for re_, tol in ((6e4, (0.02, 0.005)), (1e5, (0.02, 0.005)), (2e5, (0.02, 0.005)),
                     (8e4, (0.03, 0.008)), (1.5e5, (0.03, 0.008))):
        cl, cd = law(a, re_, T.CAMBER)
        s_cl, s_cd = sec.coefficients(a, np.full(a.shape, re_), np.full(a.shape, 1e12), np.zeros(a.shape))
        d_cl, d_cd = float(np.max(np.abs(cl - s_cl))), float(np.max(np.abs(cd - s_cd)))
        R.record("T2", "Re %g: largest |dcl|" % re_, d_cl, "< %g" % tol[0], tol[0], d_cl < tol[0])
        R.record("T2", "Re %g: largest |dcd|" % re_, d_cd, "< %g" % tol[1], tol[1], d_cd < tol[1])


def t3():
    print("T3 Ananda's flat-plate wings (read by eye, +-0.03 in CL, +-0.02 in CD)")
    rows = []
    for line in open(R.FLAP / "sections" / "separated" / "ananda2012_flatplate_rect_wings.csv"):
        if line[0] in "#a":
            continue
        ar, a, c, d = (line.strip().split(",") + [""])[:4]
        new = R.coeff_translate(float(a), ar=float(ar), flags=dict(stall_law="table"))
        old = R.coeff_translate(float(a), ar=float(ar), flags=dict(stall_law="legacy"))
        rows.append((float(ar), float(a), float(c), float(d) if d else None, new, old))
        print("  AR %g %4g deg  CL table %.3f legacy %.3f data %.2f   CD table %.3f legacy %.3f data %s" % (
            float(ar), float(a), new["CL"], old["CL"], float(c), new["CD"], old["CD"], d or "-"))
    rms = lambda k, i: math.sqrt(sum((r[k]["CL"] - r[2]) ** 2 for r in rows) / len(rows)) if i == "CL" else \
        math.sqrt(sum((r[k]["CD"] - r[3]) ** 2 for r in rows if r[3] is not None)
                  / sum(r[3] is not None for r in rows))
    new, old = rms(4, "CL"), rms(5, "CL")
    R.record("T3", "rms CL error, %d points" % len(rows), new, "< 0.14", 0.14, new < 0.14)
    R.record("T3", "the gate: below the legacy law's", new, "< %.3f" % old, "-", new < old, "legacy %.3f" % old)
    worst = max(rows, key=lambda r: abs(r[4]["CL"] - r[2]))
    R.record("T3", "largest |CL error|", abs(worst[4]["CL"] - worst[2]), "<= 0.25, at AR 2", 0.25,
             abs(worst[4]["CL"] - worst[2]) <= 0.25, "AR %g, %g deg" % worst[:2])
    R.record("T3", "rms CD error where given", rms(4, "CD"), "< 0.05", 0.05, rms(4, "CD") < 0.05,
             "legacy %.3f" % rms(5, "CD"))


def t4():
    print("T4 BW-3 (camber %.4f), which the law does not use" % B.camber_of(sections.coordinates("bw3")))
    f = B.camber_of(sections.coordinates("bw3"))
    for re_, a, cl in sections.lift_runs("bw3"):
        d = np.degrees(a)
        m = law(a, re_, f)[0]
        for lo, hi, tol in ((0.0, 5.0, 0.10 if re_ > 7e4 else 0.20), (8.0, 19.0, 0.10)):
            k = (d >= lo) & (d <= hi)
            w = float(np.max(np.abs(m[k] - cl[k])))
            R.record("T4", "lift, Re %dk, %g-%g deg" % (re_ / 1e3, lo, hi), w, "<= %g" % tol, tol, w <= tol)
        k = (d > 5.0) & (d < 8.0)
        print("     5-8 deg: %.3f" % float(np.max(np.abs(m[k] - cl[k]))))
    for re_, a, cl, cd in sections.tables("bw3"):
        d = np.degrees(a)
        k = (d >= 0.0) & (d <= 10.0)
        w = float(np.max(np.abs(law(a[k], re_, f)[1] - cd[k])))
        ok = w <= 0.008 or re_ < 7e4
        R.record("T4", "drag, Re %dk, 0-10 deg" % (re_ / 1e3), w, "<= 0.008 (100k, 200k)", 0.008, ok)


def t5():
    print("T5 a 4% cambered plate's maximum (second-hand tables: 1.0-1.1)")
    a = np.radians(np.arange(0.0, 20.01, 0.25))
    for re_ in (6e4, 1e5):
        cl = law(a, re_, 0.04)[0]
        R.record("T5", "Re %g: maximum CL" % re_, float(cl.max()), "1.17-1.22 (stated disagreement)", "-",
                 1.17 <= cl.max() <= 1.22, "at %.1f deg" % np.degrees(a[int(cl.argmax())]))


def t7():
    print("T7 sanity")
    a = np.radians(np.arange(-90.0, 90.001, 0.1))
    step = odd = under = broad = 0.0
    for f in (0.0, 0.02, 0.04, T.CAMBER, 0.09):
        for re_ in (1e4, 1e5, 1e6):
            cl, cd = law(a, re_, f)
            cf = float(skin_friction_cd(np.array([re_]))[0])
            step = max(step, float(np.max(np.abs(np.diff(cl)))))
            under = max(under, float(np.max(cf - cd)))
            broad = max(broad, abs(float(cd[-1]) - 1.98 - cf))
            if f == 0.0:
                odd = max(odd, float(np.max(np.abs(cl + cl[::-1]))))
    R.record("T7", "largest step of cl over 0.1 deg", step, "<= 0.02", 0.02, step <= 0.02)
    R.record("T7", "flat plate: |cl(a) + cl(-a)|", odd, "<= 1e-12", 1e-12, odd <= 1e-12)
    R.record("T7", "largest cf - cd", under, "<= 0", 0.0, under <= 1e-12)
    R.record("T7", "|cd(90 deg) - 1.98 - cf|", broad, "<= 1e-9", 1e-9, broad <= 1e-9)
    re_, ta, tcl, tcd = sections.tables("goe417a")[1]
    cl, cd = law(a, re_, T.CAMBER)
    got, want = float(np.max(cl / cd)), float(np.max(tcl / tcd))
    R.record("T7", "Goe 417a camber, Re %dk: best cl/cd" % (re_ / 1e3), got, "%.2f within 2%%" % want, 0.02,
             abs(got / want - 1.0) <= 0.02)


def main() -> int:
    for t in (t1, t2, t7, t4, t5, t3):
        t()
    json.dump(R.OUT, open(HERE / "r3_results.json", "w"), indent=1)
    bad = [r for r in R.OUT if not r["ok"]]
    print("\n%d rows, %d failed" % (len(R.OUT), len(bad)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
