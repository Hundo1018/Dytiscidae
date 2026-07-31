"""Free-form body volumes from an implicit occupancy field.

Why this exists
---------------
Every non-surface part in this project was a capsule: a rod parameterised by
length and radius.  The CPPN could shape a wing's chord distribution but had no
way to express a bell, a teardrop, a lumpy fuselage, or anything that is not a
solid of revolution about a single axis.  The result was that every design the
search produced looked like jointed sticks, and calling that generative design
was overstating what the representation could do.

Here the body geometry is a *field*.  A CPPN is queried over a local voxel grid
and its output thresholded into occupancy, following the encoding used for
voxel-based soft robots (Cheney et al. 2013, *Unshackling Evolution*; Hiller &
Lipson 2012).  Mass, volume, centre of mass and inertia all come from the voxels
that survive, so the physics reads the shape that was actually generated rather
than a capsule standing in for it.

Why only bodies, and not the lifting surfaces
---------------------------------------------
Flight at this scale needs large, stiff, high-aspect-ratio surfaces, and a voxel
lattice represents those badly -- which is precisely why every published
voxel-evolved robot crawls or swims and none of them fly.  So load-bearing
surfaces keep their spar-and-membrane parameterisation, where a beam model is
both accurate and cheap, and everything else -- hulls, cavities, fairings,
joint housings -- becomes free-form.

The split is a deliberate engineering choice, not a claim that beams are the
only answer.  It is recorded here so it can be revisited by experiment rather
than inherited as an assumption.

Collision representation
------------------------
MuJoCo needs convex geometry.  A bell is concave, so the occupied voxels are
clustered and each cluster's convex hull becomes one mesh geom -- a crude convex
decomposition that keeps concavity where it matters (a cavity stays a cavity)
without pulling in a decomposition library.  Fluid forces do not depend on this
at all: they are computed per voxel cluster from the real occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cppn import CPPN

#: Inputs a body CPPN receives, in the part's own normalised frame.
#:   x   along the part axis, -1 at base to +1 at tip
#:   y,z lateral, -1..1
#:   r   sqrt(y^2+z^2), so radial symmetry is one weight away
#:   d   distance from the axis midpoint, for fore-aft tapering
BODY_INPUTS = ["x", "y", "z", "r", "d", "bias"]

#: Outputs.  ``solid`` is thresholded into occupancy; ``shell`` biases material
#: toward the boundary, which is what makes a hollow bell rather than a lump.
BODY_OUTPUTS = ["solid", "shell"]


def new_body_cppn(rng: np.random.Generator) -> CPPN:
    return CPPN.minimal(BODY_INPUTS, BODY_OUTPUTS, rng)


@dataclass(eq=False)
class BodyField:
    """A free-form volume, discretised."""

    centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    voxel: float = 0.02
    hulls: list = field(default_factory=list)  # (verts, faces) per convex chunk
    volume: float = 0.0
    #: Void that is completely surrounded by material, m^3.  Water cannot reach
    #: it, so it displaces and (if the part is sealed) it floats.  This is what
    #: makes "enclose a bubble" a move the search can discover rather than a
    #: property only a hand-written hull was allowed to have.
    enclosed: float = 0.0
    #: Open concavity, m^3: the difference between the convex hull and what is
    #: actually there.  A bell's cavity is open to the water, so it is not
    #: buoyant and not enclosed -- but it is the volume a pulsed jet expels.
    cavity: float = 0.0
    surface_area: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.ones(3) * 1e-6
                                )
    extent: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.01)

    @property
    def n_voxels(self) -> int:
        return len(self.centres)

    @property
    def is_empty(self) -> bool:
        return self.n_voxels < 4

    @property
    def elongation(self) -> float:
        """Longest extent over shortest.  1 is a ball, high is a rod.

        Reported so a run can say whether the search actually left the
        stick-shaped corner of the space, rather than being asked to take it
        on trust.
        """
        e = np.sort(self.extent)
        return float(e[-1] / max(e[0], 1e-9))


def sample_body(
    cppn: CPPN | None,
    *,
    length: float,
    radius: float,
    resolution: int = 17,
    threshold: float = 0.0,
    max_clusters: int = 4,
    rng: np.random.Generator | None = None,
) -> BodyField:
    """Query a CPPN over a voxel grid and build the resulting volume.

    ``resolution`` is per axis, so 11 is 1331 samples -- a few hundred
    microseconds, negligible beside the dynamics.  Higher resolutions make
    prettier shapes and cost the convex hulls more, which is the real expense.
    """
    nx, ny = resolution, max(5, resolution // 2 + 1)
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    zs = np.linspace(-1.0, 1.0, ny)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    R = np.sqrt(Y**2 + Z**2)
    D = np.abs(X)

    if cppn is None:
        occ = (R <= 1.0)  # a plain cylinder, the degenerate case
        shell = np.zeros_like(R)
    else:
        out = cppn.query(x=X, y=Y, z=Z, r=R, d=D, bias=np.ones_like(X))
        solid = np.asarray(out["solid"], float)
        shell = np.asarray(out["shell"], float)
        # Always confine to the part's own envelope: without this the field can
        # spill outside the bounding box the mass budget and the joints assume.
        occ = (solid > threshold) & (R <= 1.0)

    if occ.sum() < 4:
        # An empty field is a dead genome. Fall back to a thin cylinder so the
        # design is still evaluable and can be scored as poor rather than
        # crashing the run.
        occ = (R <= 0.5)

    # Physical scale: x spans the part length, y/z span its diameter.
    half = np.array([length * 0.5, radius, radius])
    pts = np.stack([X[occ], Y[occ], Z[occ]], axis=1) * half
    # Shift so the part still runs from 0 to length along +X, matching the
    # capsule convention every joint and attachment already uses.
    pts[:, 0] += length * 0.5

    vx = float(2.0 * half[0] / max(nx - 1, 1))
    vy = float(2.0 * half[1] / max(ny - 1, 1))
    vz = float(2.0 * half[2] / max(ny - 1, 1))
    cell_vol = vx * vy * vz

    f = BodyField(centres=pts, voxel=float(np.cbrt(cell_vol)))
    f.volume = float(len(pts) * cell_vol)
    f.extent = np.array([
        max(pts[:, i].max() - pts[:, i].min(), 1e-3) for i in range(3)
    ])
    f.com = pts.mean(axis=0)

    # Boundary voxels: those with a missing neighbour.  Their count times the
    # cell face area is the wetted surface, which is what a shell's mass and the
    # form drag both scale with.
    idx = np.stack(np.nonzero(occ), axis=1)
    key = {tuple(i) for i in idx}
    face = (vy * vz + vx * vz + vx * vy) * 2.0 / 3.0
    exposed = 0
    for i in idx:
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            if (i[0] + d[0], i[1] + d[1], i[2] + d[2]) not in key:
                exposed += 1
    f.surface_area = float(exposed * face / 2.0)

    # Diagonal inertia of the voxel cloud about its own centre of mass, per unit
    # mass; the caller scales by the mass it decides this part has.
    rel = pts - f.com
    f.inertia = np.array([
        float(np.mean(rel[:, 1] ** 2 + rel[:, 2] ** 2)),
        float(np.mean(rel[:, 0] ** 2 + rel[:, 2] ** 2)),
        float(np.mean(rel[:, 0] ** 2 + rel[:, 1] ** 2)),
    ])

    f.enclosed = _enclosed_volume(occ) * cell_vol
    f.hulls = _convex_chunks(pts, max_clusters=max_clusters, pad=0.5 * f.voxel)
    f.cavity = max(_hull_volume(pts, pad=0.5 * f.voxel) - f.volume - f.enclosed, 0.0)
    return f


def _enclosed_volume(occ: np.ndarray) -> float:
    """Empty cells that no path reaches from outside, in cells.

    A flood fill of the empty space from the grid boundary; whatever it does not
    reach is sealed inside the material.  Without this a generated body could
    only ever float on the material it was made of, so the search had no way to
    invent a swim bladder -- and buoyancy closure is one of the three binding
    constraints on this mission.
    """
    try:
        from scipy import ndimage
    except Exception:
        return 0.0
    empty = ~occ
    if not empty.any():
        return 0.0
    # Pad by one cell so the flood always has an outside to start from, even
    # when the body touches the edge of its own envelope.
    padded = np.pad(empty, 1, mode="constant", constant_values=True)
    lab, n = ndimage.label(padded)
    if n == 0:
        return 0.0
    outside = lab[0, 0, 0]
    sealed = (lab != outside) & padded
    return float(sealed[1:-1, 1:-1, 1:-1].sum())


def _hull_volume(pts: np.ndarray, *, pad: float) -> float:
    """Volume of the convex hull of the occupied cells."""
    if len(pts) < 8:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        corners = pts[:, None, :] + pad * np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )[None, :, :]
        return float(ConvexHull(corners.reshape(-1, 3), qhull_options="QJ").volume)
    except Exception:
        return 0.0


def _convex_chunks(pts: np.ndarray, *, max_clusters: int, pad: float) -> list:
    """Split a point cloud into a few convex hulls.

    A crude convex decomposition.  Clustering along the longest axis rather than
    by k-means is deliberate: it is deterministic, it costs nothing, and for the
    shapes that matter here -- a bell, a tapered hull, a lobed fairing -- the
    concavity that needs preserving is almost always along the part's own axis.
    """
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return []

    if len(pts) < 8:
        return []
    axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    order = np.argsort(pts[:, axis])
    k = int(np.clip(max_clusters, 1, max(1, len(pts) // 8)))
    chunks = np.array_split(order, k)

    hulls = []
    for c in chunks:
        p = pts[c]
        if len(p) < 6:
            continue
        # Inflate each point to a cube so a planar cluster still hulls.
        corners = p[:, None, :] + pad * np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )[None, :, :]
        corners = corners.reshape(-1, 3)
        try:
            h = ConvexHull(corners, qhull_options="QJ")
        except Exception:
            continue
        verts = corners[h.vertices]
        remap = {v: i for i, v in enumerate(h.vertices)}
        faces = np.array([[remap[i] for i in s] for s in h.simplices if all(i in remap for i in s)])
        if len(verts) >= 4 and len(faces) >= 4:
            hulls.append((verts, faces))
    return hulls


def body_panels(f: BodyField, *, cd: float, n_max: int = 6):
    """Bluff fluid elements for a free-form body.

    Slices the occupancy along the part's own axis and returns one element per
    slice, rather than one element per part.  Three things follow that a single
    capsule element could not express:

    * **Buoyancy is distributed.**  A hull that is fat forward and fine aft has
      its centre of buoyancy forward of its mid-length, so it trims bow-up.
      With one element the buoyant force acted at the geometric centre no matter
      what shape the CPPN drew, and pitch trim was therefore blind to the shape.
    * **Drag is distributed.**  A slice far from the centre of mass produces a
      moment as well as a force, which is how a tapered tail stabilises and a
      blunt one does not.
    * **Area is honest per slice.**  Each slice reports its own three extents,
      so the projected-area drag in the solver sees the local cross-section
      rather than the bounding box of the whole part.

    Slices are cut along the longest extent -- the same axis ``_convex_chunks``
    uses -- because that is where a body's section actually varies.  Extents are
    reported in the part's local (span, chord, normal) order, which for a body
    is (x, y, z): the frame every joint and attachment already uses.
    """
    if f.is_empty:
        return []
    axis = int(np.argmax(f.extent))
    order = np.argsort(f.centres[:, axis])
    k = int(np.clip(n_max, 1, max(1, len(order) // 6)))
    out = []
    cell = f.volume / max(f.n_voxels, 1)
    for c in np.array_split(order, k):
        if len(c) == 0:
            continue
        p = f.centres[c]
        # Extents of this slice, floored at one voxel so a single-voxel-thick
        # slice still has area and does not silently drag-free.
        span = np.maximum(p.max(axis=0) - p.min(axis=0), f.voxel)
        out.append({
            "pos": p.mean(axis=0),
            "volume": float(len(c) * cell),
            "ext": span.astype(float),
            "half_height": float(0.5 * max(span[1], span[2])),
            "cd": cd,
        })
    return out
