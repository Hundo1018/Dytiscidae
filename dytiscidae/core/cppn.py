"""Compositional Pattern Producing Networks -- the implicit shape representation.

A CPPN is a small heterogeneous network that maps a coordinate to a value.
Queried over a surface's parametric domain it produces a continuous field:
chord distribution, twist, camber, thickness.  The point is that it is *not*
parameterised by a fixed list of control points, so evolution can discover
structure at whatever spatial frequency turns out to matter -- a wing that needs
a sharp chord discontinuity two thirds out can grow one by adding a single
gaussian node, and a wing that wants a smooth ellipse stays smooth for free.

This is a NEAT-style encoding: topology and weights evolve together, and the
activation function is itself a gene.  The activation set is chosen for shape
generation rather than for classification -- ``sin`` gives repetition, ``gauss``
gives localised bumps, ``abs`` gives creases, ``tanh`` gives saturation.

Everything is vectorised: ``query`` evaluates the whole network over an array of
coordinates at once, so sampling a 24 x 8 surface grid is one pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------


def _gauss(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.clip(x, -6.0, 6.0) ** 2)


def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _square(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -3.0, 3.0) ** 2


def _step(x: np.ndarray) -> np.ndarray:
    """A soft step.  Lets the network place an edge without a discontinuity."""
    return np.tanh(4.0 * x)


ACTIVATIONS: dict[str, callable] = {
    "sin": _sin,
    "gauss": _gauss,
    "tanh": _tanh,
    "sigmoid": _sigmoid,
    "abs": _abs,
    "identity": _identity,
    "square": _square,
    "step": _step,
}
ACTIVATION_NAMES = list(ACTIVATIONS)


@dataclass
class Node:
    id: int
    activation: str
    bias: float = 0.0
    #: 0 = input, 1 = hidden, 2 = output
    layer: int = 1


@dataclass
class Connection:
    src: int
    dst: int
    weight: float
    enabled: bool = True


@dataclass
class CPPN:
    """A queryable implicit field.

    Inputs and outputs are named so that the phenotype code can ask for what it
    means rather than for an index, and so that mutation can add an output
    without renumbering anything.
    """

    inputs: list[str]
    outputs: list[str]
    nodes: list[Node] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    _next_id: int = 0

    # ------------------------------------------------------------ construction

    @staticmethod
    def minimal(inputs: list[str], outputs: list[str], rng: np.random.Generator) -> "CPPN":
        """A fully connected input->output network with random weights."""
        c = CPPN(inputs=list(inputs), outputs=list(outputs))
        for name in inputs:
            c.nodes.append(Node(id=c._new_id(), activation="identity", layer=0))
        for name in outputs:
            c.nodes.append(
                Node(
                    id=c._new_id(),
                    activation=str(rng.choice(["tanh", "sin", "gauss", "identity"])),
                    bias=float(rng.normal(0.0, 0.3)),
                    layer=2,
                )
            )
        in_ids = [n.id for n in c.nodes if n.layer == 0]
        out_ids = [n.id for n in c.nodes if n.layer == 2]
        for i in in_ids:
            for o in out_ids:
                c.connections.append(Connection(i, o, float(rng.normal(0.0, 1.0))))
        return c

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def copy(self) -> "CPPN":
        return CPPN(
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            nodes=[replace(n) for n in self.nodes],
            connections=[replace(c) for c in self.connections],
            _next_id=self._next_id,
        )

    # -------------------------------------------------------------- evaluation

    def _order(self) -> list[int]:
        """Topological order of node ids.  Cycles are broken by dropping the
        offending edge for this evaluation, so a malformed genome degrades
        rather than hanging."""
        incoming: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for c in self.connections:
            if c.enabled and c.dst in incoming and c.src in incoming:
                incoming[c.dst].append(c.src)
        order: list[int] = []
        seen: set[int] = set()
        temp: set[int] = set()

        def visit(nid: int) -> None:
            if nid in seen or nid in temp:
                return
            temp.add(nid)
            for s in incoming[nid]:
                visit(s)
            temp.discard(nid)
            seen.add(nid)
            order.append(nid)

        for n in self.nodes:
            visit(n.id)
        return order

    def query(self, **coords: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the network.

        Each keyword must name one of ``self.inputs`` and carry an array of the
        same shape.  Returns one array per output name.
        """
        shape = None
        for v in coords.values():
            shape = np.shape(np.asarray(v))
            break
        if shape is None:
            raise ValueError("query needs at least one coordinate array")

        by_id = {n.id: n for n in self.nodes}
        in_ids = [n.id for n in self.nodes if n.layer == 0]
        out_ids = [n.id for n in self.nodes if n.layer == 2]

        values: dict[int, np.ndarray] = {}
        for name, nid in zip(self.inputs, in_ids):
            values[nid] = np.asarray(coords.get(name, np.zeros(shape)), dtype=float)

        by_dst: dict[int, list[Connection]] = {}
        for c in self.connections:
            if c.enabled:
                by_dst.setdefault(c.dst, []).append(c)

        for nid in self._order():
            if nid in values:
                continue
            node = by_id.get(nid)
            if node is None:
                continue
            acc = np.full(shape, node.bias, dtype=float)
            for c in by_dst.get(nid, ()):
                src = values.get(c.src)
                if src is not None:
                    acc = acc + c.weight * src
            values[nid] = ACTIVATIONS[node.activation](acc)

        return {name: values.get(nid, np.zeros(shape)) for name, nid in zip(self.outputs, out_ids)}

    # ---------------------------------------------------------------- mutation

    def mutate_weights(self, rng: np.random.Generator, sigma: float = 0.3, rate: float = 0.8) -> None:
        for c in self.connections:
            if rng.random() < rate:
                if rng.random() < 0.1:
                    c.weight = float(rng.normal(0.0, 1.0))  # occasional full reset
                else:
                    c.weight += float(rng.normal(0.0, sigma))
                c.weight = float(np.clip(c.weight, -8.0, 8.0))
        for n in self.nodes:
            if n.layer != 0 and rng.random() < rate * 0.5:
                n.bias = float(np.clip(n.bias + rng.normal(0.0, sigma * 0.5), -4.0, 4.0))

    def mutate_add_node(self, rng: np.random.Generator) -> bool:
        """Split an existing connection with a new node (NEAT's structural add).

        The incoming weight is 1 and the outgoing weight inherits the original,
        so the new node is behaviourally near-neutral at birth and only becomes
        useful once its weights drift.  Neutral-at-birth structural mutation is
        what stops topology growth from being uniformly lethal.
        """
        live = [c for c in self.connections if c.enabled]
        if not live:
            return False
        c = live[int(rng.integers(len(live)))]
        c.enabled = False
        n = Node(
            id=self._new_id(),
            activation=str(rng.choice(ACTIVATION_NAMES)),
            bias=0.0,
            layer=1,
        )
        self.nodes.append(n)
        self.connections.append(Connection(c.src, n.id, 1.0))
        self.connections.append(Connection(n.id, c.dst, c.weight))
        return True

    def mutate_add_connection(self, rng: np.random.Generator, tries: int = 12) -> bool:
        ids_by_layer = {0: [], 1: [], 2: []}
        for n in self.nodes:
            ids_by_layer[n.layer].append(n.id)
        sources = ids_by_layer[0] + ids_by_layer[1]
        targets = ids_by_layer[1] + ids_by_layer[2]
        if not sources or not targets:
            return False
        existing = {(c.src, c.dst) for c in self.connections}
        for _ in range(tries):
            s = sources[int(rng.integers(len(sources)))]
            t = targets[int(rng.integers(len(targets)))]
            if s == t or (s, t) in existing:
                continue
            if self._creates_cycle(s, t):
                continue
            self.connections.append(Connection(s, t, float(rng.normal(0.0, 1.0))))
            return True
        return False

    def _creates_cycle(self, src: int, dst: int) -> bool:
        """True if adding src->dst would make dst an ancestor of itself."""
        adj: dict[int, list[int]] = {}
        for c in self.connections:
            if c.enabled:
                adj.setdefault(c.src, []).append(c.dst)
        stack = [dst]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, ()))
        return False

    def mutate_activation(self, rng: np.random.Generator) -> bool:
        cand = [n for n in self.nodes if n.layer != 0]
        if not cand:
            return False
        n = cand[int(rng.integers(len(cand)))]
        n.activation = str(rng.choice(ACTIVATION_NAMES))
        return True

    def mutate_toggle(self, rng: np.random.Generator) -> bool:
        if not self.connections:
            return False
        c = self.connections[int(rng.integers(len(self.connections)))]
        # Never disable the last surviving path to an output.
        if c.enabled and sum(1 for x in self.connections if x.enabled and x.dst == c.dst) <= 1:
            return False
        c.enabled = not c.enabled
        return True

    # --------------------------------------------------------------- crossover

    @staticmethod
    def crossover(a: "CPPN", b: "CPPN", rng: np.random.Generator) -> "CPPN":
        """Structural crossover by matching (src, dst) pairs.

        Without global innovation numbers this is approximate, but for shape
        fields the competing-conventions problem matters much less than it does
        for control networks -- two CPPNs that both describe a tapered wing are
        genuinely interchangeable in the region where they agree.
        """
        child = a.copy()
        b_by_edge = {(c.src, c.dst): c for c in b.connections}
        for c in child.connections:
            other = b_by_edge.get((c.src, c.dst))
            if other is not None and rng.random() < 0.5:
                c.weight = other.weight
                c.enabled = other.enabled
        return child

    # ------------------------------------------------------------- descriptors

    @property
    def complexity(self) -> int:
        return sum(1 for c in self.connections if c.enabled) + len(self.nodes)

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [{"id": n.id, "act": n.activation, "bias": n.bias, "layer": n.layer} for n in self.nodes],
            "connections": [
                {"src": c.src, "dst": c.dst, "w": c.weight, "on": c.enabled} for c in self.connections
            ],
            "next_id": self._next_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "CPPN":
        return CPPN(
            inputs=list(d["inputs"]),
            outputs=list(d["outputs"]),
            nodes=[Node(n["id"], n["act"], n["bias"], n["layer"]) for n in d["nodes"]],
            connections=[Connection(c["src"], c["dst"], c["w"], c["on"]) for c in d["connections"]],
            _next_id=d["next_id"],
        )


# --------------------------------------------------------------------------
# Surface fields
# --------------------------------------------------------------------------

#: Inputs every surface CPPN receives.
#:   u   spanwise station, 0 at root, 1 at tip
#:   d   distance from the body centreline, normalised -- lets one CPPN encode a
#:       left/right symmetric pattern without being told about symmetry
#:   r   radial coordinate sqrt(u^2 + d^2), a useful primitive for smooth taper
#:   bias constant 1
SURFACE_INPUTS = ["u", "d", "r", "bias"]

#: Outputs, all in [-1, 1] before being mapped to physical ranges.
#:   chord      chord multiplier along the span
#:   twist      geometric twist (washout is negative)
#:   camber     section camber, drives the zero-lift angle
#:   thickness  section thickness ratio, drives spar depth and buoyancy
#:   dihedral   out-of-plane bend of the surface
SURFACE_OUTPUTS = ["chord", "twist", "camber", "thickness", "dihedral"]


def new_surface_cppn(rng: np.random.Generator) -> CPPN:
    return CPPN.minimal(SURFACE_INPUTS, SURFACE_OUTPUTS, rng)


@dataclass(eq=False)
class SurfaceField:
    """A CPPN evaluated over one surface, mapped into physical units."""

    u: np.ndarray  # (n,) spanwise stations, 0..1
    chord: np.ndarray  # (n,) m
    twist: np.ndarray  # (n,) rad
    camber: np.ndarray  # (n,) fraction of chord
    thickness: np.ndarray  # (n,) fraction of chord
    dihedral: np.ndarray  # (n,) rad

    @property
    def area(self) -> float:
        """Planform area by the trapezoid rule over the span stations."""
        return float(np.trapezoid(self.chord, self.u))

    def mean_chord(self) -> float:
        return float(np.mean(self.chord))


def sample_surface(
    cppn: CPPN,
    *,
    span: float,
    root_chord: float,
    stations: int = 12,
    lateral_offset: float = 0.0,
    max_twist_deg: float = 35.0,
) -> SurfaceField:
    """Evaluate ``cppn`` over a surface and map outputs to physical quantities.

    The chord map is deliberately floored at 12% of root chord: a CPPN that
    outputs -1 everywhere would otherwise produce a zero-area wing, which is
    both a degenerate phenotype and a division-by-zero waiting to happen in the
    aspect-ratio calculation.
    """
    u = np.linspace(0.0, 1.0, stations)
    d = np.full_like(u, lateral_offset)
    r = np.sqrt(u**2 + d**2)
    out = cppn.query(u=u, d=d, r=r, bias=np.ones_like(u))

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, float), -1.0, 1.0)

    chord = root_chord * (0.56 + 0.44 * norm(out["chord"]))
    chord = np.maximum(chord, 0.12 * root_chord)
    twist = np.radians(max_twist_deg) * norm(out["twist"])
    camber = 0.09 * norm(out["camber"])
    thickness = 0.04 + 0.10 * (0.5 + 0.5 * norm(out["thickness"]))
    dihedral = np.radians(30.0) * norm(out["dihedral"])
    return SurfaceField(u=u * span, chord=chord, twist=twist, camber=camber,
                        thickness=thickness, dihedral=dihedral)
