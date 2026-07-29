"""Continuous morphological traits, replacing the hand-written part taxonomy.

The problem with categories
---------------------------
The previous representation gave every part a ``kind`` drawn from a list I
wrote: hull, strut, wing, paddle, foot, ballast, membrane, bell, fin.  Adding
the last three looked like widening the search space, but it only replaced a
short list of categories with a longer one, each carrying parameters I had
tuned.  A part that is 60% wing and 40% bell -- or something with no name --
remained unreachable at any mutation distance, and the space *between* the
categories is exactly where a triphibian compromise would have to live.

The alternative, following voxel-based soft-robot evolution (Hiller & Lipson
2012; Cheney et al. 2013), is to make morphology a continuous field of physical
properties and let "what kind of thing is this" be an emergent region of that
space rather than a declaration.

The traits
----------
Five continuous properties, each read *directly* by the physics rather than
switched on by a label:

    flatness          rod  <-> thin plate      lift slope, added mass, drag
    enclosure         open <-> sealed cavity   buoyancy, pressure loading
    compliance        rigid spar <-> membrane  which structural check applies
    cavity_actuation  fixed <-> full jet       momentum-flux thrust
    contact_affinity  slick <-> gripping foot  ground contact and walking

A wing is flatness 1.0, compliance 0.2, enclosure 0.3.  A medusa bell is
flatness 0.2, enclosure 0.9, cavity_actuation 0.5.  Neither is declared;
both are points, and everything between them is reachable by interpolation.

Legacy body plans keep working: ``traits_for_kind`` maps each old category to
the point in trait space it always implicitly occupied, so the archetypes remain
valid seeds while ceasing to be the only expressible designs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: Order used when a trait vector is flattened for mutation or for descriptors.
TRAIT_NAMES = ("flatness", "enclosure", "compliance", "cavity_actuation",
               "contact_affinity")


@dataclass
class Traits:
    """Where a part sits in morphological property space.

    Every field is in [0, 1] and every field is read by at least one physical
    model.  A trait nothing reads would be a decoration, and would let the
    search waste mutations on a gene with no consequence.
    """

    #: 0 = slender rod (a strut, a bone), 1 = thin plate (a wing, a fin).
    #: Sets how much circulatory lift the element develops, how much added mass
    #: it carries normal to its surface, and its form drag.
    flatness: float = 0.5
    #: 0 = open frame or solid, 1 = sealed gas-filled cavity.  Sets net buoyancy
    #: and whether the part is loaded by ambient pressure at depth.
    enclosure: float = 0.2
    #: 0 = rigid spar carrying bending, 1 = tension-only membrane that sheds
    #: load by deflecting.  Selects which structural idealisation applies and
    #: how much load reaches the member that carries it.
    compliance: float = 0.2
    #: 0 = fixed volume, 1 = the cavity strokes fully each cycle, ejecting
    #: fluid.  Multiplied by ``enclosure`` to give usable jet thrust: a jet
    #: needs something to squeeze.
    cavity_actuation: float = 0.0
    #: 0 = slick fairing, 1 = a foot.  Scales contact friction and the credit
    #: a part gets for supporting the machine on ground.
    contact_affinity: float = 0.2

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in TRAIT_NAMES], dtype=float)

    @staticmethod
    def from_array(v: np.ndarray) -> "Traits":
        return Traits(**{n: float(np.clip(x, 0.0, 1.0)) for n, x in zip(TRAIT_NAMES, v)})

    def copy(self) -> "Traits":
        return replace(self)

    def clipped(self) -> "Traits":
        return Traits.from_array(self.as_array())

    # ------------------------------------------------------------- physics

    @property
    def is_lifting(self) -> bool:
        """Whether this part is flat enough to develop useful circulation."""
        return self.flatness > 0.45

    @property
    def lift_efficiency(self) -> float:
        """Fraction of ideal circulatory lift this shape achieves.

        A rod develops essentially none; a thin plate develops nearly all.  The
        curve is deliberately smooth so that a part can evolve from strut to
        surface through states that are partially useful, rather than having to
        cross a valley where it is neither.
        """
        return float(np.clip((self.flatness - 0.25) / 0.6, 0.0, 1.0)) ** 1.5

    @property
    def drag_coefficient(self) -> float:
        """Bluff-body drag referenced to frontal area.

        Flat plates broadside are the draggiest thing in the model (~1.9); a
        streamlined rod is ~0.3.  A paddle is a plate used deliberately.
        """
        return float(0.30 + 1.60 * self.flatness)

    @property
    def jet_strength(self) -> float:
        """How much of a pulsed jet this part can produce.

        The product, not either alone: a sealed cavity that never contracts
        makes no jet, and a vigorous stroke of an open frame moves no fluid.
        """
        return float(self.enclosure * self.cavity_actuation)

    @property
    def carries_bending(self) -> bool:
        """Whether the part has a bending member of its own.

        Below this the loads pass into whatever carries the part -- the bat's
        digit rather than its membrane -- which is a real difference in load
        path, not a modelling convenience.
        """
        return self.compliance < 0.55

    @property
    def load_shed(self) -> float:
        """Fraction of hydrodynamic load a compliant surface deflects away.

        A rigid plate transmits all of it; a fish fin or a jellyfish tentacle
        bends out of the flow and transmits far less, which is why neither is
        built like a spar.
        """
        return float(np.clip(self.compliance, 0.0, 1.0)) ** 0.7 * 0.75

    @property
    def friction(self) -> float:
        """Contact friction coefficient for MJCF."""
        return float(0.25 + 1.15 * self.contact_affinity)

    def describe(self) -> str:
        """A readable name for where this part sits, for telemetry only.

        Nothing in the physics reads this.  It exists so that a human looking at
        an archive can tell at a glance what the search found, without the label
        ever constraining what the search can find.
        """
        if self.jet_strength > 0.25:
            return "jet-cavity"
        if self.flatness > 0.7:
            return "membrane-surface" if self.compliance > 0.55 else "stiff-surface"
        if self.flatness > 0.45:
            return "paddle-ish"
        if self.enclosure > 0.6:
            return "float-body"
        if self.contact_affinity > 0.6:
            return "foot"
        return "strut"


# --------------------------------------------------------------------------
# Legacy bridge
# --------------------------------------------------------------------------

#: Where each old hand-written category always implicitly sat.  Keeping this
#: lets the five archetype body plans remain valid seeds while the taxonomy
#: itself stops being the only thing expressible.
_KIND_TRAITS = {
    "hull":     Traits(flatness=0.10, enclosure=0.90, compliance=0.05,
                       cavity_actuation=0.0, contact_affinity=0.15),
    "strut":    Traits(flatness=0.05, enclosure=0.10, compliance=0.05,
                       cavity_actuation=0.0, contact_affinity=0.20),
    "wing":     Traits(flatness=0.95, enclosure=0.30, compliance=0.20,
                       cavity_actuation=0.0, contact_affinity=0.10),
    "paddle":   Traits(flatness=0.80, enclosure=0.10, compliance=0.30,
                       cavity_actuation=0.0, contact_affinity=0.25),
    "foot":     Traits(flatness=0.15, enclosure=0.05, compliance=0.10,
                       cavity_actuation=0.0, contact_affinity=0.90),
    "ballast":  Traits(flatness=0.10, enclosure=0.75, compliance=0.05,
                       cavity_actuation=0.10, contact_affinity=0.15),
    "membrane": Traits(flatness=0.98, enclosure=0.05, compliance=0.85,
                       cavity_actuation=0.0, contact_affinity=0.10),
    "bell":     Traits(flatness=0.20, enclosure=0.85, compliance=0.45,
                       cavity_actuation=0.55, contact_affinity=0.10),
    "fin":      Traits(flatness=0.90, enclosure=0.15, compliance=0.65,
                       cavity_actuation=0.0, contact_affinity=0.15),
}


def traits_for_kind(kind: str) -> Traits:
    """The point in trait space an old category occupied."""
    return _KIND_TRAITS.get(kind, Traits()).copy()


def nearest_kind(t: Traits) -> str:
    """Closest legacy category, for reading archives written before traits.

    Used only for display and for backward-compatible MJCF colouring.  Nothing
    downstream branches on the result.
    """
    v = t.as_array()
    return min(_KIND_TRAITS, key=lambda k: float(np.linalg.norm(_KIND_TRAITS[k].as_array() - v)))


def mutate_traits(t: Traits, rng: np.random.Generator, sigma: float = 0.12) -> Traits:
    """Perturb a part's position in trait space.

    Continuous and unconstrained by category boundaries, so a strut can drift
    into being a paddle by passing through the states in between -- each of
    which is a real design the physics can evaluate, rather than an invalid
    intermediate that has to be jumped over.
    """
    v = t.as_array()
    # Move one trait decisively rather than all of them slightly: a diffuse
    # perturbation of five traits mostly produces a part that is marginally
    # different in every way and meaningfully different in none.
    i = int(rng.integers(len(v)))
    v[i] += rng.normal(0.0, sigma * 3.0)
    v += rng.normal(0.0, sigma * 0.3, len(v))
    return Traits.from_array(v)


def random_traits(rng: np.random.Generator) -> Traits:
    """A part from anywhere in the space, with no category prior at all."""
    return Traits.from_array(rng.random(len(TRAIT_NAMES)))


def interpolate(a: Traits, b: Traits, w: float) -> Traits:
    """Blend two parts.  The states no taxonomy contains live here."""
    return Traits.from_array((1.0 - w) * a.as_array() + w * b.as_array())
