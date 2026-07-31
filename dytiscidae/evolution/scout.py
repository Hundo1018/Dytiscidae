"""The scout: a network that predicts a lineage's potential, not its score.

The problem with selecting on score
-----------------------------------
Every selection pressure in this project so far reads the design in front of it.
The curator prefers high fitness, the judge scores what was measured, the critic
discounts what will not survive.  All of them are asking "how good is this",
and none of them can ask "what will this become".

Those are different questions and the second one has a different answer.  The
history of quality-diversity search is largely the history of noticing that the
stepping stones to a good solution usually look bad -- Lehman and Stanley's
result that abandoning the objective entirely and selecting on novelty alone
beats objective-driven search on deceptive problems is uncomfortable precisely
because it is not a tuning detail.  A design with a superb body and a controller
that has not learned to use it scores near zero, and it is worth more than a
mediocre design that is already at its ceiling.  Greedy selection cannot see the
difference, so it breeds the second and discards the first.

The trap has a particular shape here.  Six islands, a curriculum that promotes
on measured performance, and a judge whose bar ratchets: all three sharpen
selection, and sharper selection is exactly what kills a dark horse faster.

What potential can be trained on
--------------------------------
The useful thing about this being an evolutionary run is that potential is not
a matter of opinion.  It is *observable in arrears*: for any design that has been
in the archive long enough, the best score achieved by any of its descendants is
a fact.  So the label is

    lift = (best score any descendant reached within H generations) - (own score)

and the model predicts that lift from what was knowable about the design at
birth.  A design whose lineage went on to triple is a dark horse whether or not
anyone recognised it at the time, and there are hundreds of such labels in any
long run.

What the model gets to see
--------------------------
Deliberately *not* mostly the score.  The features are the things that tend to
precede improvement rather than accompany it:

``control authority``
    The singular values of the body's mobility basis: how much twist the body
    can produce per unit of control input, measured by probing, independent of
    whether the controller has learned to use any of it.  A body with authority
    and no policy is the canonical dark horse -- and the gap between authority
    and achieved motion is directly measurable.
``novelty``
    Distance to the nearest occupants of behaviour space.  The classic
    non-learned answer to this question, kept as a feature so the scout can
    learn how much to trust it rather than being told.
``headroom``
    Structural and energy margin.  A design running at the edge of its structure
    has nowhere to go; one with three times the margin it needs can afford to
    grow.
``distance to the next rung``
    How close the design is to a qualitative threshold.  Just below a rung is a
    different situation from far below it, and the total score cannot say which.
``complexity``
    Parts, joints, network size.  Room to grow, and its cost.

Why a network here and a linear model for the critic
-----------------------------------------------------
The critic predicts a smooth quantity -- how much of a score survives -- and
does it from features that relate to it fairly directly.  Potential does not
behave that way: authority matters only when the controller is *not* already
using it, novelty matters more early than late, and headroom matters only if
something else is binding.  Those are interactions, and a linear model cannot
represent an interaction.  So this is a small MLP, trained with Adam, on a few
hundred to a few thousand labels.

The guard rails are the same as everywhere else, for the same reason.  The scout
may **protect** a design from pruning and may **add selection weight**.  It may
never change a score.  A signal that can raise a score is a signal the population
will optimise against, and a potential predictor that could be farmed would
select for designs that look promising forever and never deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: What the scout sees.  Score is included so the model learns the *residual* --
#: the lift beyond what the score already implies -- rather than rediscovering
#: that good designs have good descendants.
SCOUT_FEATURES = (
    "fitness", "novelty",
    "authority_max", "authority_sum", "authority_unused",
    "structural_headroom", "energy_headroom",
    "rung_min", "rung_max", "rung_spread", "distance_to_next_rung",
    "log_mass", "n_parts", "n_actuated", "cppn_complexity",
    "stage", "improvement_over_parent",
)
SCOUT_DIM = len(SCOUT_FEATURES)


def scout_features(
    meta: dict,
    *,
    novelty: float = 0.0,
    parent_fitness: float | None = None,
    mobility=None,
) -> np.ndarray:
    """Assemble what was knowable about a design at birth."""

    def g(k, d=0.0):
        v = meta.get(k, d)
        return float(v) if isinstance(v, (int, float)) else d

    rungs = meta.get("rungs") or {}
    vals = [float(v) for v in rungs.values()] or [0.0]

    # Control authority: the singular values of the identified mobility basis.
    # This is the body's capability, measured by probing, with the controller
    # taken out of it.
    auth = []
    for b in (mobility or {}).values():
        a = getattr(b, "authority", None)
        if a is not None and len(a):
            auth.extend(float(x) for x in a)
    auth = auth or [0.0]
    a_max, a_sum = max(auth), float(np.sum(auth))

    fit = g("fitness", g("mission_fraction"))
    # How much of the available authority is *not* showing up in performance.
    # A body that can produce a lot of motion and is not producing it is the
    # case this whole module exists for.
    unused = float(np.clip(a_max, 0.0, 5.0) * (1.0 - float(np.clip(fit, 0.0, 1.0))))

    return np.array([
        fit,
        float(novelty),
        a_max, a_sum, unused,
        float(np.clip(g("margin"), -1.0, 3.0)),
        float(np.clip(g("energy_margin"), -1.0, 3.0)),
        min(vals), max(vals), max(vals) - min(vals),
        float(g("distance_to_next_rung", 1.0)),
        float(np.log10(max(g("mass", 1.0), 1e-3))),
        g("n_parts"), g("dof"), g("cppn_complexity"),
        g("stage"),
        float(fit - parent_fitness) if parent_fitness is not None else 0.0,
    ], dtype=float)


# --------------------------------------------------------------------------
# A small MLP
# --------------------------------------------------------------------------


@dataclass(eq=False)
class MLP:
    """One hidden layer, tanh, Adam.  Small on purpose.

    Potential is an interaction problem -- authority matters only when it is
    unused, novelty matters more early than late -- and an interaction is the
    one thing a linear model cannot represent.  It is still only a few hundred
    parameters, because the label count is in the hundreds and a bigger network
    would memorise the run rather than learn from it.
    """

    n_in: int
    hidden: int = 24
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0, 1.0 / np.sqrt(self.n_in), (self.n_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, 1.0 / np.sqrt(self.hidden), self.hidden)
        self.b2 = 0.0
        self._m = [np.zeros_like(x) for x in (self.w1, self.b1, self.w2)]
        self._v = [np.zeros_like(x) for x in (self.w1, self.b1, self.w2)]
        self._mb = self._vb = 0.0
        self._t = 0

    def forward(self, X: np.ndarray) -> tuple:
        h = np.tanh(X @ self.w1 + self.b1)
        return h @ self.w2 + self.b2, h

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(X))[0]

    def step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01,
             weight_decay: float = 1e-4) -> float:
        n = len(X)
        out, h = self.forward(X)
        err = out - y
        loss = float(np.mean(err**2))

        gw2 = h.T @ err / n + weight_decay * self.w2
        gb2 = float(np.mean(err))
        dh = np.outer(err, self.w2) * (1.0 - h**2) / n
        gw1 = X.T @ dh + weight_decay * self.w1
        gb1 = dh.sum(axis=0)

        self._t += 1
        b1c, b2c = 1 - 0.9**self._t, 1 - 0.999**self._t
        for i, (p, g) in enumerate(((self.w1, gw1), (self.b1, gb1), (self.w2, gw2))):
            self._m[i] = 0.9 * self._m[i] + 0.1 * g
            self._v[i] = 0.999 * self._v[i] + 0.001 * g * g
            p -= lr * (self._m[i] / b1c) / (np.sqrt(self._v[i] / b2c) + 1e-8)
        self._mb = 0.9 * self._mb + 0.1 * gb2
        self._vb = 0.999 * self._vb + 0.001 * gb2 * gb2
        self.b2 -= lr * (self._mb / b1c) / (np.sqrt(self._vb / b2c) + 1e-8)
        return loss


# --------------------------------------------------------------------------
# Lineage bookkeeping
# --------------------------------------------------------------------------


@dataclass(eq=False)
class _Node:
    design_id: str
    parent_id: str | None
    generation: int
    fitness: float
    island: str
    features: np.ndarray
    #: Best score reached by this design or anything descended from it.  Seeded
    #: with the design's *own* score, not zero: a design with no descendants yet
    #: has achieved no lift, and labelling it with ``0 - fitness`` makes every
    #: childless node a negative example proportional to how good it was, which
    #: is the opposite of the truth and swamped the signal.
    best_descendant: float = 0.0
    labelled: bool = False


@dataclass(eq=False)
class Scout:
    """Predicts how much better a lineage will get, and protects the ones that
    look like they will.

    Parameters
    ----------
    horizon:
        Generations to wait before a design's realised lift is considered known.
        Too short and every label reads zero because nothing has had time to
        improve; too long and the scout learns from a run that has ended.
    reserve:
        Fraction of an archive protected from pruning on predicted potential
        rather than on score.  This is the actual dark-horse quota.
    """

    horizon: int = 40
    reserve: float = 0.15
    min_samples: int = 80
    refit_every: int = 60
    hidden: int = 24
    seed: int = 0

    nodes: dict = field(default_factory=dict)
    _x: list = field(default_factory=list)
    _y: list = field(default_factory=list)
    net: MLP | None = None
    _mean: np.ndarray | None = None
    _scale: np.ndarray | None = None
    seen_since_fit: int = 0
    fits: int = 0
    calibration: float = 0.0
    protected: int = 0

    # ------------------------------------------------------------- recording

    def record(self, design_id: str, parent_id, generation: int, fitness: float,
               island: str, features: np.ndarray) -> None:
        """Note a design's birth and propagate its score up its ancestry."""
        self.nodes[design_id] = _Node(
            design_id=design_id, parent_id=parent_id, generation=generation,
            fitness=float(fitness), island=island,
            features=np.asarray(features, float),
            best_descendant=float(fitness),
        )
        # Walk up: every ancestor's best-descendant score may have just improved.
        # Bounded depth so a long lineage cannot make this quadratic.
        pid, depth = parent_id, 0
        while pid is not None and depth < 200:
            node = self.nodes.get(pid)
            if node is None:
                break
            if fitness > node.best_descendant:
                node.best_descendant = float(fitness)
            pid, depth = node.parent_id, depth + 1
        if len(self.nodes) > 20000:
            self._forget(generation)

    def _forget(self, generation: int) -> None:
        cutoff = generation - 4 * self.horizon
        for k in [k for k, n in self.nodes.items()
                  if n.labelled and n.generation < cutoff]:
            del self.nodes[k]

    def harvest(self, generation: int) -> int:
        """Turn matured lineages into training labels.

        The label is what the lineage *actually did*: the best score any
        descendant reached, minus the design's own.  Nobody has to decide what
        counts as promise; the run answers it.
        """
        new = 0
        for n in self.nodes.values():
            if n.labelled or generation - n.generation < self.horizon:
                continue
            lift = float(np.clip(n.best_descendant - n.fitness, 0.0, 2.0))
            self._x.append(n.features)
            self._y.append(lift)
            n.labelled = True
            new += 1
            self.seen_since_fit += 1
        if len(self._x) > 6000:
            self._x, self._y = self._x[-4000:], self._y[-4000:]
        return new

    def due(self) -> bool:
        return len(self._x) >= self.min_samples and self.seen_since_fit >= self.refit_every

    # ------------------------------------------------------------------- fit

    def fit(self, epochs: int = 300) -> bool:
        if len(self._x) < self.min_samples:
            return False
        X = np.asarray(self._x, float)
        y = np.asarray(self._y, float)
        X = np.where(np.isfinite(X), X, 0.0)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._scale = np.where(sd > 1e-9, sd, 1.0)
        Z = (X - self._mean) / self._scale

        # Held out, so calibration is a claim about unseen designs rather than
        # about how well the network memorised the ones it was given.
        n_hold = max(len(Z) // 5, 10)
        rng = np.random.default_rng(self.seed + self.fits)
        idx = rng.permutation(len(Z))
        tr, ho = idx[n_hold:], idx[:n_hold]

        self.net = MLP(n_in=Z.shape[1], hidden=self.hidden, seed=self.seed + self.fits)
        for _ in range(epochs):
            batch = rng.choice(tr, size=min(64, len(tr)), replace=False)
            self.net.step(Z[batch], y[batch])

        pred = self.net.predict(Z[ho])
        if float(np.std(pred)) > 1e-9 and float(np.std(y[ho])) > 1e-9:
            self.calibration = float(np.clip(np.corrcoef(pred, y[ho])[0, 1], 0.0, 1.0))
        else:
            self.calibration = 0.0
        self.fits += 1
        self.seen_since_fit = 0
        return True

    @property
    def fitted(self) -> bool:
        return self.net is not None

    # ------------------------------------------------------------ prediction

    def potential(self, features: np.ndarray) -> float:
        """Predicted lift this lineage will achieve, scaled by calibration.

        Before the network has learned anything, this falls back to novelty --
        the classic non-learned answer to the same question, and the right prior
        to hold while waiting for evidence.
        """
        f = np.asarray(features, float)
        if not np.all(np.isfinite(f)):
            return 0.0
        if not self.fitted or self.calibration <= 0.1:
            return float(np.clip(f[1], 0.0, 1.0)) * 0.5  # novelty
        z = (f - self._mean) / self._scale
        return float(np.clip(self.net.predict(z)[0], 0.0, 2.0)) * self.calibration

    def selection_weight(self, features: np.ndarray, strength: float = 2.0) -> float:
        """Multiplier on a design's chance of being bred from.

        Additive in effect, never subtractive: the scout can argue for a design
        nobody else wants, and cannot argue against one.  Something that only
        adds cannot be used to suppress a rival, which matters when the thing
        making the argument is a learned model.
        """
        return float(1.0 + strength * self.potential(features))

    def reserve_ids(self, elites: list, key=lambda e: e.meta.get("scout_features")) -> set:
        """Which designs the scout is protecting from pruning.

        A quota, not a threshold: a fixed fraction of the archive is held on
        potential regardless of how the scores are distributed.  A threshold
        would protect everything early and nothing late, which is the opposite
        of what is wanted -- dark horses matter more once selection has
        sharpened, not less.
        """
        if not elites:
            return set()
        n = max(1, int(round(self.reserve * len(elites))))
        scored = []
        for e in elites:
            f = key(e)
            if f is None:
                continue
            scored.append((self.potential(np.asarray(f, float)), id(e)))
        if not scored:
            return set()
        scored.sort(reverse=True)
        keep = {i for _, i in scored[:n]}
        self.protected = len(keep)
        return keep

    def explains(self, top: int = 4) -> list:
        """Which features drive the prediction, by gradient at the mean design.

        A network is harder to read than a linear model, so it has to be asked
        rather than inspected: perturb each input at the population's own centre
        and see what moves.  Without this the scout is an unexplained preference,
        and this project has enough of those behind it to know how that ends.
        """
        if not self.fitted:
            return [{"feature": "novelty", "effect": 1.0, "note": "unfitted: novelty prior"}]
        z0 = np.zeros((1, SCOUT_DIM))
        base = float(self.net.predict(z0)[0])
        out = []
        for i in range(SCOUT_DIM):
            z = z0.copy()
            z[0, i] = 1.0
            out.append((abs(float(self.net.predict(z)[0]) - base), i,
                        float(self.net.predict(z)[0]) - base))
        out.sort(reverse=True)
        return [
            {"feature": SCOUT_FEATURES[i], "effect": round(d, 3)}
            for _, i, d in out[:top]
        ]

    def report(self) -> dict:
        return {
            "fitted": self.fitted,
            "labels": len(self._x),
            "tracked": len(self.nodes),
            "fits": self.fits,
            "calibration": round(self.calibration, 3),
            "protected": self.protected,
            "drivers": self.explains(),
        }


def novelty_of(descriptor, archive, k: int = 5) -> float:
    """Mean distance to the k nearest occupied cells, normalised.

    The non-learned half of the answer, and the scout's prior before it has
    learned anything.  Kept as a plain function because it is also worth
    reporting on its own: a population whose novelty has collapsed is one that
    has stopped exploring, whatever its scores are doing.
    """
    if not archive.cells:
        return 1.0
    d = np.asarray(descriptor, float)
    pts = np.array([e.descriptor for e in archive.cells.values()])
    if pts.ndim != 2 or pts.shape[1] != d.shape[0]:
        return 0.0
    span = np.maximum(archive.hi - archive.lo, 1e-9)
    dist = np.linalg.norm((pts - d) / span, axis=1)
    kk = min(k, len(dist))
    return float(np.clip(np.mean(np.sort(dist)[:kk]), 0.0, 1.0))
