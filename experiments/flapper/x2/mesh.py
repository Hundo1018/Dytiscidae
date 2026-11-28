"""Pass X2: a two-dimensional hybrid mesh around a section, with gmsh.

    cd /home/hundo/Projects/Dytiscidae/.cfd && pixi run python \
        <repo>/experiments/flapper/x2/mesh.py SECTION LEVEL OUT.msh

SECTION: goe417a | plate.  LEVEL: 0 (the mesh) or 1 (every spacing halved).

Unit chord, leading edge at the origin, chord along +x, one cell thick
(z = 0 .. DZ).  Wall-normal quads from gmsh's boundary-layer field (first
cell H1, growth RATIO), quad-dominant unstructured cells outside, a wake box,
far field a circle of radius R_FAR chords about the mid chord.  Physical
names: wall, farfield, front, back, fluid.  Written as msh 2.2 for gmshToFoam.

Goe 417a: the UIUC outline (`sections.coordinates('goe417a')`, identical to
`ptera/goe417a_coords.txt`) closes to a point at the trailing edge.  It is
opened to a blunt base GAP chords thick by adding +-GAP/2 * x to the upper and
lower surfaces (XFOIL's TGAP with a linear blend), so the chord stays 1 and
the base can be meshed.  The plate: 3.1% thick, semicircular ends, length 1.
"""
import math
import sys
from pathlib import Path

import gmsh
import numpy as np

HERE = Path(__file__).resolve().parent
DZ = 0.01
R_FAR = 25.0
GAP = 0.0025
# Boundary-layer thickness by section: gmsh left one degenerate or inverted
# element in the layer's outer row at other values (checkMesh failed: goe L1
# at 0.03 and 0.035, plate at 0.02 and 0.03), see NOTES.md.
THICK = {"goe417a": 0.02, "plate": 0.025}
OPTS = {}                     # extra gmsh options (set per mesh in MESH_OPTS)
BASE = dict(h1=1.0e-4, ratio=1.10, thick=0.02, ds_max=0.006, ds_le=0.0008, ds_te=0.0012,
            near=0.003, far_slope=2.0 / 25.0, wake=0.015, n_base=4)


def goe417a_outline():
    pts = np.loadtxt(HERE.parent / "ptera" / "goe417a_coords.txt")
    i = int(np.argmin(pts[:, 0]))
    up = pts[: i + 1][::-1].copy()        # LE -> TE
    lo = pts[i:].copy()                   # LE -> TE
    le = pts[i].copy()
    # shift so the leading edge is at the origin and the chord is exactly 1
    for a in (up, lo):
        a[:, 0] -= le[0]
        a[:, 1] -= le[1]
    sc = 1.0 / (pts[0, 0] - le[0])
    up *= sc
    lo *= sc
    up[:, 1] += 0.5 * GAP * up[:, 0]
    lo[:, 1] -= 0.5 * GAP * lo[:, 0]
    return up, lo


def _cubic(t, y):
    """Natural cubic spline through (t, y); returns an evaluator."""
    n = len(t)
    h = np.diff(t)
    A = np.zeros((n, n))
    r = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for k in range(1, n - 1):
        A[k, k - 1], A[k, k], A[k, k + 1] = h[k - 1], 2 * (h[k - 1] + h[k]), h[k]
        r[k] = 3 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
    c = np.linalg.solve(A, r)
    b = (np.diff(y) / h) - h * (2 * c[:-1] + c[1:]) / 3
    d = np.diff(c) / (3 * h)

    def f(s):
        k = np.clip(np.searchsorted(t, s) - 1, 0, n - 2)
        x = s - t[k]
        return y[k] + b[k] * x + c[k] * x * x + d[k] * x ** 3
    return f


def resample_closed(up, lo, p):
    """One spline through TE_lo -> LE -> TE_up (parameter: chord-length), so the
    leading edge is smooth; each side redistributed with spacing ds_le at the
    leading edge, ds_te at the trailing edge, ds_max between."""
    xy = np.vstack([lo[::-1], up[1:]])                     # TE_lo .. LE .. TE_up
    seg = np.hypot(*np.diff(xy, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    fx, fy = _cubic(t, xy[:, 0]), _cubic(t, xy[:, 1])
    t_le = t[len(lo) - 1]
    # dense arc length
    tt = np.linspace(0, t[-1], 20001)
    xx, yy = fx(tt), fy(tt)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))])
    s_le = float(np.interp(t_le, tt, s))

    def side(L):
        # march from the trailing edge (u=0) to the leading edge (u=1)
        out = [0.0]
        while out[-1] < L:
            u = out[-1] / L
            ds = (p["ds_max"] - (p["ds_max"] - p["ds_te"]) * math.exp(-u / 0.04)
                  - (p["ds_max"] - p["ds_le"]) * math.exp(-(1 - u) / 0.03))
            out.append(out[-1] + ds)
        a = np.array(out)
        return a * (L / a[-1])                               # stretch to land on the LE
    s_lo = side(s_le)                                        # TE_lo (s=0) .. LE (s=s_le)
    s_up = np.sort(s[-1] - side(s[-1] - s_le))               # LE .. TE_up (s=s_end)
    ss = np.unique(np.concatenate([s_lo, s_up]))
    tq = np.interp(ss, s, tt)
    P = np.stack([fx(tq), fy(tq)], 1)
    P[0] = lo[-1]
    P[-1] = up[-1]
    return P                                                 # TE_lo .. LE .. TE_up


def plate_outline(p, t=0.031):
    r = t / 2
    # straight part from x=r to 1-r on y=+-r, semicircles at both ends
    pts = []
    n_arc = max(int(math.pi * r / p["ds_le"]), 24)
    # the flat parts: ds_le at both ends (matching the arcs), ds_max between
    L = 1 - 2 * r
    out = [0.0]
    while out[-1] < L:
        u = out[-1] / L
        out.append(out[-1] + p["ds_max"] - (p["ds_max"] - p["ds_le"]) * (
            math.exp(-u / 0.03) + math.exp(-(1 - u) / 0.03)))
    xs = r + np.array(out) * (L / out[-1])
    for x in xs[:-1]:                                     # upper, LE -> TE
        pts.append((x, r))
    for a in np.linspace(math.pi / 2, -math.pi / 2, n_arc + 1)[:-1]:   # TE arc
        pts.append((1 - r + r * math.cos(a), r * math.sin(a)))
    for x in xs[::-1][:-1]:                                # lower, TE -> LE
        pts.append((x, -r))
    for a in np.linspace(-math.pi / 2, -3 * math.pi / 2, n_arc + 1)[:-1]:  # LE arc
        pts.append((r + r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


def build(section: str, level: int, out: str):
    p = dict(BASE)
    p["thick"] = THICK.get(section, p["thick"])
    f = 0.5 ** level
    for k in ("h1", "ds_max", "ds_le", "ds_te", "near", "wake"):
        p[k] *= f
    p["ratio"] = BASE["ratio"] ** f
    p["far_slope"] *= 1.0          # far field: unchanged growth slope, halved base size via 'near'
    p["n_base"] = int(BASE["n_base"] / f)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("x2")
    geo = gmsh.model.geo
    if section == "goe417a":
        up, lo = goe417a_outline()
        P = resample_closed(up, lo, p)                       # TE_lo .. LE .. TE_up (open at the base)
        ids = [geo.addPoint(x, y, 0) for x, y in P]
        lines = [geo.addLine(ids[k], ids[k + 1]) for k in range(len(ids) - 1)]
        base = geo.addLine(ids[-1], ids[0])                  # TE_up -> TE_lo, the blunt base
        geo.mesh.setTransfiniteCurve(base, p["n_base"] + 1)
        wall = lines + [base]
        fans = [ids[0], ids[-1]]
    elif section == "plate":
        P = plate_outline(p)
        ids = [geo.addPoint(x, y, 0) for x, y in P]
        wall = [geo.addLine(ids[k], ids[(k + 1) % len(ids)]) for k in range(len(ids))]
        fans = []
    else:
        raise SystemExit("section: goe417a | plate")
    for c in wall[: len(wall) - (1 if section == "goe417a" else 0)]:
        geo.mesh.setTransfiniteCurve(c, 2)
    loop_in = geo.addCurveLoop(wall)
    cx = 0.5
    c0 = geo.addPoint(cx, 0, 0)
    q = [geo.addPoint(cx + R_FAR * math.cos(a), R_FAR * math.sin(a), 0)
         for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    arcs = [geo.addCircleArc(q[k], c0, q[(k + 1) % 4]) for k in range(4)]
    loop_out = geo.addCurveLoop(arcs)
    surf = geo.addPlaneSurface([loop_out, loop_in])
    geo.synchronize()

    fld = gmsh.model.mesh.field
    bl = fld.add("BoundaryLayer")
    fld.setNumbers(bl, "CurvesList", wall)
    fld.setNumber(bl, "Size", p["h1"])
    fld.setNumber(bl, "Ratio", p["ratio"])
    fld.setNumber(bl, "Thickness", p["thick"])
    fld.setNumber(bl, "Quads", 1)
    if fans:
        fld.setNumbers(bl, "FanPointsList", fans)
        fld.setNumbers(bl, "FanPointsSizesList", [5, 5])
    fld.setAsBoundaryLayer(bl)

    dist = fld.add("Distance")
    fld.setNumbers(dist, "CurvesList", wall)
    fld.setNumber(dist, "Sampling", 4000)
    th = fld.add("MathEval")
    fld.setString(th, "F", "%g + %g*F%d" % (p["near"], p["far_slope"], dist))
    box = fld.add("Box")
    fld.setNumber(box, "VIn", p["wake"])
    fld.setNumber(box, "VOut", 10.0)
    fld.setNumber(box, "XMin", 0.9)
    fld.setNumber(box, "XMax", 3.0)
    fld.setNumber(box, "YMin", -0.4)
    fld.setNumber(box, "YMax", 1.3)
    fld.setNumber(box, "Thickness", 0.5)
    mn = fld.add("Min")
    fld.setNumbers(mn, "FieldsList", [th, box])
    fld.setAsBackgroundMesh(mn)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)

    ext = geo.extrude([(2, surf)], 0, 0, DZ, [1], [1], recombine=True)
    geo.synchronize()
    top = ext[0][1]
    vol = ext[1][1]
    sides = [e[1] for e in ext[2:]]
    n_out = len(arcs)
    # extrude returns the sides in the order of the surface's boundary: outer loop first
    far_s, wall_s = sides[:n_out], sides[n_out:]
    gmsh.model.addPhysicalGroup(2, wall_s, name="wall")
    gmsh.model.addPhysicalGroup(2, far_s, name="farfield")
    gmsh.model.addPhysicalGroup(2, [surf], name="back")
    gmsh.model.addPhysicalGroup(2, [top], name="front")
    gmsh.model.addPhysicalGroup(3, [vol], name="fluid")
    for k, v in OPTS.items():
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(out)
    n3 = len(gmsh.model.mesh.getElements(3)[1][0]) if gmsh.model.mesh.getElements(3)[1] else 0
    gmsh.finalize()
    print("section %s level %d: h1 %.3g ratio %.4f near %.3g wall nodes %d" % (
        section, level, p["h1"], p["ratio"], p["near"], len(P)))
    return n3


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), sys.argv[3])
