"""The critic: a learned adversary trained to catch what cheap evaluation misses.

The relationship being built
----------------------------
Actor and critic, or generator and discriminator -- the structure is the same
one, and it fits this problem better than it fits most, because there is a real
asymmetry to exploit.

The population (the actor) is scored on Tier 1: eight-second episodes and an
extrapolation.  That is what it optimises against, because that is what it is
told.  But Tier 1 is cheap precisely because it is a *proxy*, and every proxy
has a gap between what it measures and what it stands for.  The whole history of
this project is that gap being found: a battery drained on the first step to
truncate the episode, a hillside skimmed to fake sustained flight, a coefficient
value the design silently depended on.

Each of those was eventually caught by something expensive -- a Tier-2 mission,
a perturbation audit -- and each was caught *after* the population had already
spent hundreds of generations exploiting it.  The expensive checks cannot run on
every candidate; that is why they are expensive.

So: train a model to predict what the expensive checks would have said.

    actor    the evolving population, maximising the score it is given
    critic   a model of the gap, trained on every (cheap measurement, expensive
             verdict) pair the run produces
    signal   designs whose cheap score the critic predicts will not survive are
             discounted, and are prioritised for actual expensive checking

The adversarial loop is genuine.  When the population finds a new way to look
good cheaply and fail expensively, the audits that catch it become training data,
the critic learns the signature, and the discount closes that route.  The
population must then find a way of looking good that the critic cannot
distinguish from being good -- and the only reliable such way is being good.

Why this cannot run away
------------------------
A learned critic is itself a model, and a model can be wrong or gamed.  Three
things bound it.

It is trained on labels from something that does not learn.  The auditor's
verdicts and the Tier-2 results are ground truth produced by physics and by
held-out re-measurement, not by another network's opinion, so the critic is
anchored to something outside the loop.

Its influence is bounded and one-directional.  It can discount a score, never
raise one -- the same asymmetry the auditor has, and for the same reason: a
signal that can award points is a signal that can be optimised against.

And it reports its own calibration.  A critic whose predictions do not correlate
with the outcomes it is predicting has its influence automatically reduced to
nothing, which is the honest response to a critic that has stopped knowing
anything.

Why ridge regression rather than a network
------------------------------------------
Four CPU cores, shared with the evaluations that generate the training data, and
a few thousand labelled examples over a multi-day run.  A network is the wrong
tool at that data volume and that compute budget; a regularised linear model on
engineered features fits in milliseconds, is refit from scratch every time
rather than drifting, and -- the part that matters here -- can be *read*, so the
run can say which measurement the critic has learned to distrust.  The interface
takes any object with ``fit`` and ``predict``, so this is a starting point and
not a commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: What the critic sees.  Cheap Tier-1 quantities only -- if it could see the
#: expensive verdict it would have nothing to predict.
CRITIC_FEATURES = (
    "mission_fraction", "air", "water", "land",
    "transition_crossed", "transition_shock", "transition_control",
    "transition_exit", "energy_margin", "structural_margin",
    "log_mass", "log_wing_loading", "aspect_ratio",
    "clamped", "actuator_overload", "n_actuated",
)


def critic_features(meta: dict, result) -> np.ndarray:
    """Assemble the cheap feature vector for one candidate."""
    tc = {}
    tr = getattr(result, "transitions", None)
    if tr is not None:
        try:
            tc = tr.component_means()
        except Exception:
            tc = {}

    def g(k, default=0.0):
        v = meta.get(k, default)
        return float(v) if isinstance(v, (int, float)) else default

    mass = max(g("mass", 1.0), 1e-3)
    ws = max(g("wing_loading", 1.0), 1e-3)
    return np.array([
        float(getattr(result, "mission_fraction", 0.0)),
        g("air"), g("water"), g("land"),
        float(tc.get("crossed", 0.0)),
        float(tc.get("shock", 0.0)),
        float(tc.get("control", 0.0)),
        float(tc.get("exit_state", 0.0)),
        float(np.clip(getattr(result, "energy_margin", -1.0), -1.0, 5.0)),
        float(np.clip(getattr(result, "structural_margin", 0.0), -1.0, 3.0)),
        float(np.log10(mass)),
        float(np.log10(ws)),
        g("aspect_ratio"),
        1.0 if getattr(result, "exploit", "") else 0.0,
        g("max_actuator_overload"),
        g("dof"),
    ], dtype=float)


@dataclass(eq=False)
class Critic:
    """Predicts how much of a cheap score survives expensive checking.

    Parameters
    ----------
    min_samples:
        Below this the critic abstains entirely.  A model fitted on a handful of
        labels is worse than no model, because it is confident.
    max_discount:
        The most it may ever take off a score.  Bounded so that a critic which
        has learned something wrong slows the search down rather than
        redirecting it into a wall.
    ridge:
        Regularisation.  High on purpose: the features are correlated, the
        labels are noisy, and a critic that fits the noise will discount honest
        designs for resembling dishonest ones.
    """

    min_samples: int = 60
    max_discount: float = 0.5
    ridge: float = 1.0
    refit_every: int = 40

    _x: list = field(default_factory=list)
    _y: list = field(default_factory=list)
    _w: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _scale: np.ndarray | None = None
    _bias: float = 0.0
    seen_since_fit: int = 0
    fits: int = 0
    #: Correlation between prediction and outcome on the labels seen so far.
    #: The critic's influence is scaled by this, so a critic that has stopped
    #: predicting anything stops mattering without anyone intervening.
    calibration: float = 0.0

    # ------------------------------------------------------------- labelling

    def label(self, features: np.ndarray, retained: float) -> None:
        """Record one (cheap features, expensive outcome) pair.

        ``retained`` is the fraction of the Tier-1 score that survived: Tier-2
        mission fraction over Tier-1 mission fraction, or 0 for a design the
        auditor invalidated.  It is a *ratio* rather than an absolute score so
        the critic learns about the gap rather than about performance, which is
        what makes it an adversary and not a second opinion.
        """
        f = np.asarray(features, float)
        if not np.all(np.isfinite(f)) or not np.isfinite(retained):
            return
        self._x.append(f)
        self._y.append(float(np.clip(retained, 0.0, 2.0)))
        self.seen_since_fit += 1
        if len(self._x) > 4000:
            self._x = self._x[-3000:]
            self._y = self._y[-3000:]

    @property
    def fitted(self) -> bool:
        return self._w is not None

    def due(self) -> bool:
        return (
            len(self._x) >= self.min_samples
            and self.seen_since_fit >= self.refit_every
        )

    # ------------------------------------------------------------------- fit

    def fit(self) -> bool:
        if len(self._x) < self.min_samples:
            return False
        X = np.asarray(self._x, float)
        y = np.asarray(self._y, float)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._scale = np.where(sd > 1e-9, sd, 1.0)
        Z = (X - self._mean) / self._scale
        self._bias = float(y.mean())
        yc = y - self._bias

        # Ridge, solved directly.  Sixteen features is nothing to invert, and a
        # closed-form refit from scratch each time is what stops the critic from
        # accumulating drift the way an incrementally-updated one would.
        n_feat = Z.shape[1]
        A = Z.T @ Z + self.ridge * np.eye(n_feat)
        try:
            self._w = np.linalg.solve(A, Z.T @ yc)
        except np.linalg.LinAlgError:
            return False

        pred = Z @ self._w + self._bias
        if float(np.std(pred)) > 1e-9 and float(np.std(y)) > 1e-9:
            self.calibration = float(np.clip(np.corrcoef(pred, y)[0, 1], 0.0, 1.0))
        else:
            self.calibration = 0.0
        self.fits += 1
        self.seen_since_fit = 0
        return True

    # ----------------------------------------------------------- prediction

    def predict(self, features: np.ndarray) -> float:
        """Predicted fraction of the cheap score that would survive."""
        if not self.fitted:
            return 1.0
        f = np.asarray(features, float)
        if not np.all(np.isfinite(f)):
            return 1.0
        z = (f - self._mean) / self._scale
        return float(np.clip(z @ self._w + self._bias, 0.0, 2.0))

    def discount(self, features: np.ndarray) -> float:
        """Multiplier in [1 - max_discount, 1] to apply to a cheap score.

        Never above 1.  A critic that could raise a score would be a second
        objective for the population to optimise against, and the population is
        very good at optimising against things.
        """
        if not self.fitted or self.calibration <= 0.05:
            return 1.0
        predicted = self.predict(features)
        shortfall = float(np.clip(1.0 - predicted, 0.0, 1.0))
        # Scaled by calibration: a critic that does not predict its own labels
        # has no business moving anyone's score.
        return float(1.0 - self.max_discount * shortfall * self.calibration)

    def suspicion(self, features: np.ndarray) -> float:
        """How badly this design is expected to fail expensive checking.

        Used to *prioritise* auditing rather than to punish: the point of a
        critic that can smell an exploit is to spend the expensive checks where
        they will find something.
        """
        if not self.fitted:
            return 0.0
        return float(np.clip(1.0 - self.predict(features), 0.0, 1.0)) * self.calibration

    # -------------------------------------------------------------- reading

    def distrusts(self, top: int = 3) -> list:
        """Which measurements the critic has learned to read as warning signs.

        The reason for a linear model: a run can say *what* the critic learned,
        and a critic whose top weights are nonsense is visible as nonsense
        rather than as an unexplained drop in everyone's score.
        """
        if not self.fitted:
            return []
        order = np.argsort(self._w)
        return [
            {"feature": CRITIC_FEATURES[i], "weight": round(float(self._w[i]), 3)}
            for i in order[:top]
        ]

    def report(self) -> dict:
        return {
            "fitted": self.fitted,
            "labels": len(self._x),
            "fits": self.fits,
            "calibration": round(self.calibration, 3),
            "distrusts": self.distrusts(),
        }
