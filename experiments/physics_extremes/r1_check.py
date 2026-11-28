"""Repair R1's checks V1-V5 (PREREG_R1_stall.md): the default law against
thin-plate data.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/physics_extremes/r1_check.py
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

PLATE = dict(stall_law="plate")
TWO_D = dict(flags=dict(PLATE, inflow=False, _machine_ar=None), strip_ar=1e4)


def cl(a, **kw):
    return R.coeff_translate(float(a), **kw)["CL"]


def main() -> int:
    print("V1 Ananda's flat-plate wings (by eye, +-0.03)")
    err = []
    for line in open(R.FLAP / "sections" / "separated" / "ananda2012_flatplate_rect_wings.csv"):
        if line[0] in "#a":
            continue
        ar, a, c, _ = (line.strip().split(",") + [""])[:4]
        m = cl(a, ar=float(ar), flags=PLATE)
        err.append(m - float(c))
        print("  AR %s %5s deg  model %.3f  data %.2f  (%+.2f)" % (ar, a, m, float(c), m - float(c)))
    rms = math.sqrt(sum(e * e for e in err) / len(err))
    R.record("V1", "rms CL error over %d points" % len(err), rms, "< 0.08", 0.08, rms < 0.08)
    R.record("V1", "largest |error|", max(abs(e) for e in err), "<= 0.15", 0.15, max(abs(e) for e in err) <= 0.15)

    print("V2 two-dimensional flat plate")
    for re in (2e4, 8e4):
        c = {a: cl(a, re=re, **TWO_D) for a in range(0, 31, 1)}
        print("  Re %g:" % re, " ".join("%d:%.2f" % (a, c[a]) for a in (2, 4, 6, 7, 8, 10, 12, 15, 19, 22, 25, 30)))
        for a in (10, 15, 19):
            R.record("V2", "Re %g, %d deg" % (re, a), c[a], "0.60-0.76 +-0.10", 0.10, 0.50 <= c[a] <= 0.86)
        peak = max(c.values())
        a_pk = max(c, key=c.get)
        # the first maximum (the bend), not the plate branch's rise past 20 deg
        a_pk = max((a for a in c if a <= 12), key=c.get)
        peak = c[a_pk]
        after = min(v for a, v in c.items() if a_pk < a <= 20)
        R.record("V2", "Re %g: drop after the maximum (to 20 deg)" % re, peak - after, "<= 0.06", 0.06, peak - after <= 0.06,
                 "max %.3f at %d deg" % (peak, a_pk))

    print("V3 two-dimensional 5% arc, Re 2e4")
    c = {a: cl(a, re=2e4, camber=0.05, **TWO_D) for a in range(-4, 21)}
    print("  ", " ".join("%d:%.2f" % (a, c[a]) for a in (-4, -2, 0, 2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20)))
    R.record("V3", "CL at zero incidence", c[0], "0.15-0.25", "-", 0.15 <= c[0] <= 0.25)
    pk = max(c, key=c.get)
    R.record("V3", "maximum CL", c[pk], "0.95-1.10", "-", 0.95 <= c[pk] <= 1.10, "at %d deg" % pk)
    R.record("V3", "angle of the maximum, deg", pk, "8-12", "-", 8 <= pk <= 12)
    for a in (14, 16, 18):
        R.record("V3", "CL at %d deg" % a, c[a], "0.82-0.92", "-", 0.82 <= c[a] <= 0.92)

    print("V4 AR 6, 5% arc, Re 2e4")
    c = {a: cl(a, re=2e4, camber=0.05, ar=6.0, flags=PLATE) for a in range(0, 21)}
    pk = max(c, key=c.get)
    print("  ", " ".join("%d:%.2f" % (a, c[a]) for a in (0, 4, 8, 10, 11, 12, 14, 16, 18, 20)))
    R.record("V4", "maximum CL", c[pk], "0.90-1.05", "-", 0.90 <= c[pk] <= 1.05, "at %d deg" % pk)

    import sections
    SEC = sections.load("goe417a+lift")
    print("V5 a 5.8% plate on the law against Goe 417a's measured table, Re 1e5 (consistency only)")
    worst = 0.0
    row = []
    for a in range(0, 21, 2):
        law = cl(a, re=1e5, camber=0.058, **TWO_D)
        sec = R.coeff_translate(float(a), re=1e5, camber=0.058, **dict(TWO_D, flags=dict(TWO_D["flags"], section=SEC)))["CL"]
        worst = max(worst, abs(law - sec))
        row.append("%d:%.2f/%.2f" % (a, law, sec))
    print("   law/table ", " ".join(row))
    R.record("V5", "largest |law - table|, 0-20 deg", worst, "<= 0.15", 0.15, worst <= 0.15)
    print("\n%d rows, %d failed" % (len(R.OUT), sum(not r["ok"] for r in R.OUT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
