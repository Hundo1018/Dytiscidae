"""Pass X2, stage A: the comparison table and A1-A5 against the frozen wording.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/x2/compare.py [MESH]

MESH (default L0) is the mesh the grid is read on.  Measured CL: the lift run's
rising sweep (`sections.lift_runs`); measured CD: the drag run
(`goe417a_Re100k.csv`, to 11.65 deg); both interpolated linearly at the CFD
angle.  For each (section, mesh, closure, alpha) the last row in
results.jsonl is used.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sections  # noqa: E402


def rows():
    out = {}
    for line in (HERE / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("stage") == "A" and r.get("mode") in ("steady", "transient"):
            out[(r["section"], r["mesh"], r["closure"], float(r["alpha"]))] = r
    return out


def measured():
    lr = [x for x in sections.lift_runs("goe417a") if abs(x[0] - 1e5) < 2e4][0]
    a_l, cl_l = np.degrees(np.asarray(lr[1])), np.asarray(lr[2])
    rows_ = [l.split(",") for l in (HERE.parent / "sections" / "goe417a_Re100k.csv").read_text().splitlines()
             if l and not l.startswith("#") and not l.startswith("alpha")]
    d = np.array(rows_, float)
    return a_l, cl_l, d[:, 0], d[:, 1], d[:, 2]


def main():
    mesh = sys.argv[1] if len(sys.argv) > 1 else "L0"
    R = rows()
    a_l, cl_l, a_d, cl_d, cd_d = measured()
    m_cl = lambda a: float(np.interp(a, a_l, cl_l))
    m_cd = lambda a: float(np.interp(a, a_d, cd_d)) if a_d.min() <= a <= a_d.max() else None
    m_cld = lambda a: float(np.interp(a, a_d, cl_d)) if a_d.min() <= a <= a_d.max() else None
    alphas = [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
    print(f"Goe 417a, Re 1e5, mesh {mesh}")
    print("| alpha | meas CL (lift run) | meas CL (drag run) | meas CD | SST CL | SST CD | SST mode | LM CL | LM CD | LM mode |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for a in alphas:
        s = R.get(("goe417a", mesh, "kOmegaSST", float(a)))
        t = R.get(("goe417a", mesh, "kOmegaSSTLM", float(a)))
        print(f"| {a} | {m_cl(a):.3f} | {f(m_cld(a))} | {f(m_cd(a), 4)} | {f(s and s['CL'])} | {f(s and s['CD'], 4)} | "
              f"{s['mode'] if s else 'not run'} | {f(t and t['CL'])} | {f(t and t['CD'], 4)} | {t['mode'] if t else 'not run'} |")
    verdict = {}
    # A1
    print("\nA1 mesh (L0 -> L1): lift under 3%, drag under 8%")
    a1 = {}
    for cl in ("kOmegaSST", "kOmegaSSTLM"):
        ok = True
        for a in (4.0, 12.0):
            r0, r1 = R.get(("goe417a", "L0", cl, a)), R.get(("goe417a", "L1", cl, a))
            if not (r0 and r1):
                print(f"  {cl} {a}: not available")
                ok = None
                continue
            dl = abs(r1["CL"] - r0["CL"]) / abs(r0["CL"])
            dd = abs(r1["CD"] - r0["CD"]) / abs(r0["CD"])
            good = dl < 0.03 and dd < 0.08
            ok = ok and good if ok is not None else ok
            print(f"  {cl} {a:4.0f} deg: CL {r0['CL']:.4f} -> {r1['CL']:.4f} ({100 * dl:.1f}%), "
                  f"CD {r0['CD']:.5f} -> {r1['CD']:.5f} ({100 * dd:.1f}%): {'holds' if good else 'fails'}")
        a1[cl] = ok
    print("  A1:", a1)
    # A2, A3 per closure (A2/A3 are worded for the transition closure; the other is read alongside)
    res = {}
    for cl in ("kOmegaSSTLM", "kOmegaSST"):
        ok_l, ok_d, worst_l, worst_d, have = True, True, 0.0, 0.0, True
        for a in (2, 4, 6, 8, 10):
            r = R.get(("goe417a", mesh, cl, float(a)))
            if not r:
                have = False
                continue
            el = abs(r["CL"] - m_cl(a))
            ed = abs(r["CD"] - m_cd(a)) / m_cd(a)
            worst_l, worst_d = max(worst_l, el), max(worst_d, ed)
            ok_l &= el <= 0.10
            ok_d &= ed <= 0.40
        g = [(a, R[("goe417a", mesh, cl, float(a))]["CL"]) for a in alphas if ("goe417a", mesh, cl, float(a)) in R]
        if g:
            k = int(np.argmax([c for _, c in g]))
            amax, cmax = g[k]
            if 0 < k < len(g) - 1:
                (x0, y0), (x1, y1), (x2, y2) = g[k - 1], g[k], g[k + 1]
                A = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
                a_fit = -A[1] / (2 * A[0]) if A[0] < 0 else x1
                c_fit = float(np.polyval(A, a_fit))
            else:
                a_fit, c_fit = amax, cmax
        else:
            amax = cmax = a_fit = c_fit = None
        res[cl] = dict(A2_lift=ok_l, A2_drag=ok_d, worst_dCL=worst_l, worst_dCD_rel=worst_d, complete=have,
                       clmax_grid=cmax, a_clmax_grid=amax, clmax_fit=c_fit, a_clmax_fit=a_fit)
        a3 = cmax is not None and abs(cmax - 1.355) <= 0.15 and abs(a_fit - 13.5) <= 2.0
        res[cl]["A3"] = a3
        print(f"\n{cl}: A2 lift (|dCL| <= 0.10 at 2-10): {'holds' if ok_l else 'fails'} (worst {worst_l:.3f}); "
              f"A2 drag (within 40%): {'holds' if ok_d else 'fails'} (worst {100 * worst_d:.0f}%); complete {have}")
        print(f"   max CL on the grid {f(cmax)} at {amax} deg; parabola {f(c_fit)} at {f(a_fit, 2)} deg; "
              f"A3 (within 0.15 of 1.355 and 2 deg of 13.5): {'holds' if a3 else 'fails'}")
    # A4
    print("\nA4: at 0 deg (measured %.3f) either closure off by up to 0.2" % m_cl(0.0))
    for cl in ("kOmegaSST", "kOmegaSSTLM"):
        r = R.get(("goe417a", mesh, cl, 0.0))
        if r:
            print(f"  {cl}: CL {r['CL']:.3f}, off by {r['CL'] - m_cl(0.0):+.3f}; worst point of 2-10? "
                  f"{abs(r['CL'] - m_cl(0.0)) >= res[cl]['worst_dCL']}")
    # A5
    print("\nA5: SST drag too high below 6 deg, lift held")
    for a in (-2, 0, 2, 4):
        r = R.get(("goe417a", mesh, "kOmegaSST", float(a)))
        if r and m_cd(a):
            print(f"  {a} deg: SST CD {r['CD']:.4f} vs measured {m_cd(a):.4f} ({100 * (r['CD'] / m_cd(a) - 1):+.0f}%)")
    print(f"  SST lift at 2-10 within 0.10: {res['kOmegaSST']['A2_lift']} (worst {res['kOmegaSST']['worst_dCL']:.3f})")
    # plate
    print("\nFlat plate (3.1%, semicircular edges), Re 1e5, mesh L0")
    lp = [x for x in sections.lift_runs("flatplate_uiuc") if abs(x[0] - 1e5) < 1e4]
    if lp:
        ap, cp = np.degrees(np.asarray(lp[0][1])), np.asarray(lp[0][2])
    print("| alpha | tunnel CL (rising) | 2 pi alpha | SST CL | SST CD | LM CL | LM CD |")
    print("|---|---|---|---|---|---|---|")
    for a in (2, 5, 8, 10, 15, 20, 24):
        s = R.get(("plate", "L0", "kOmegaSST", float(a)))
        t = R.get(("plate", "L0", "kOmegaSSTLM", float(a)))
        tun = float(np.interp(a, ap, cp)) if lp and a <= ap.max() else None
        print(f"| {a} | {f(tun)} | {2 * math.pi * math.radians(a):.3f} | {f(s and s['CL'])} | {f(s and s['CD'], 4)} | "
              f"{f(t and t['CL'])} | {f(t and t['CD'], 4)} |")
    json.dump(dict(A1=a1, per_closure=res), open(HERE / "stageA_verdict.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
