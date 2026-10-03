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

What it predicts: a gap, never a ratio
--------------------------------------
Until 2026-10-03 the label was ``Tier-2 mission / Tier-1 mission``, and a pair
was recorded only when the Tier-1 mission exceeded 1e-4, because a ratio has no
value at zero.  arch45 measured what that cost: 147 promotions, 128 dropped by
the gate, and the 19 kept were all 0.0 -- so the critic never fitted once in 900
generations.  The four promotions where Tier-2 *succeeded* (0.1667 each) all had
a Tier-1 mission of exactly zero, and were exactly the ones thrown away.

Those four were not the cheap tier missing a success either.  Tier-2's mission
is ``completed legs / 6`` from a *random* start domain, stopping at the first
leg under 0.15 competence, while Tier-1's is gated on the weakest medium: 0.1667
is "the first leg drawn was a medium this body can do".  The two mission
fractions are different quantities, so their gap is not a gap.

So the target is now the residual ``expensive - cheap`` of each medium's
competence, ``CRITIC_TARGETS`` -- the same scorer at both tiers, defined at
zero, and with variance even where the mission is zero almost everywhere.  A
medium Tier-2 never reached (the legs after a failure are not run) is *not
measured*, not zero: it is NaN and that target is fitted without the row.  The
score the critic acts on is the mean residual; the discount reads only its
negative part, so the one-directional bound below is unchanged.

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

#: What the critic predicts the expensive tier will do to, as residuals: each
#: medium's competence, in [0, 1] at both tiers, and each the feature of the same
#: name in ``CRITIC_FEATURES`` -- which is what lets the critic compute the gap
#: from the features it was given rather than trusting a caller to.  Not the
#: mission: the two tiers define it differently (see the module docstring).
CRITIC_TARGETS = ("air", "water", "land")
_TARGET_IDX = np.array([CRITIC_FEATURES.index(k) for k in CRITIC_TARGETS])


def expensive_outcome(result) -> np.ndarray:
    """The expensive tier's values for ``CRITIC_TARGETS``, from a MissionResult.

    A mission segment wins over a probe leg (``label_all_media``) for the same
    medium.  A medium with neither was never run -- Tier-2 stops at the first
    failed leg -- and is NaN, not 0: "could not measure" must not share a value with
    "measured zero".  A result flagged as an exploit retained nothing in any
    medium, whatever its numbers say.
    """
    if result is None or getattr(result, "exploit", ""):
        return np.zeros(len(CRITIC_TARGETS))
    seg = {**(getattr(result, "probe_segments", None) or {}),
           **(getattr(result, "segments", {}) or {})}
    return np.array(
        [float(seg[k].competence) if k in seg else np.nan for k in CRITIC_TARGETS],
        dtype=float)


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
    """Predicts how far a cheap score moves under expensive checking.

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
    _bias: np.ndarray | None = None
    seen_since_fit: int = 0
    fits: int = 0
    #: Out-of-fold skill: correlation between the expensive outcome the critic
    #: implies and the one observed, minus the cheap score's own correlation
    #: with it, per target, weighted by how many labels each target has.  The critic's influence is scaled by this, so a
    #: critic that has stopped predicting anything stops mattering without
    #: anyone intervening.
    calibration: float = 0.0
    calibration_by_target: dict = field(default_factory=dict)
    #: Labels discarded on restore because they were written as ratios.
    dropped_legacy: int = 0

    def __setstate__(self, state: dict) -> None:
        # A critic pickled before 2026-10-03 holds scalar ratio labels and a
        # scalar bias.  Those mean something else and cannot be converted (the
        # Tier-2 per-medium values were never kept), so they are dropped, and
        # said to be: arch45's whole critic was 27 such labels, all 0.0.
        state = dict(state)
        ys = state.get("_y") or []
        if ys and np.size(ys[0]) != len(CRITIC_TARGETS):
            state["dropped_legacy"] = state.get("dropped_legacy", 0) + len(ys)
            state.update(_x=[], _y=[], _w=None, _mean=None, _scale=None,
                         _bias=None, seen_since_fit=0, calibration=0.0)
        elif np.ndim(state.get("_bias")) == 0 and state.get("_w") is not None:
            state.update(_w=None, _bias=None, calibration=0.0)
        state.setdefault("calibration_by_target", {})
        state.setdefault("dropped_legacy", 0)
        self.__dict__.update(state)

    # ------------------------------------------------------------- labelling

    def label(self, features: np.ndarray, expensive) -> None:
        """Record one (cheap features, expensive outcome) pair.

        ``expensive`` is the expensive tier's value for each of
        ``CRITIC_TARGETS`` (``expensive_outcome`` builds it from a result), NaN
        where that medium was not measured, or all zeros for a design the
        auditor invalidated.  The label stored is the
        residual ``expensive - cheap``, so the critic learns about the gap
        rather than about performance -- which is what makes it an adversary
        and not a second opinion -- and, unlike a ratio, the gap is defined when
        the cheap score is zero.  No pair is ever refused for a small cheap
        score: those are the pairs where the cheap tier is most likely wrong.
        """
        f = np.asarray(features, float)
        e = np.asarray(expensive, float)
        if e.shape != (len(CRITIC_TARGETS),):
            raise ValueError(
                f"expensive outcome must have {len(CRITIC_TARGETS)} entries "
                f"{CRITIC_TARGETS}, got shape {e.shape}")
        if f.shape != (len(CRITIC_FEATURES),) or not np.all(np.isfinite(f)) \
                or not np.any(np.isfinite(e)):
            return
        cheap = np.clip(f[_TARGET_IDX], 0.0, 1.0)
        self._x.append(f)
        self._y.append(np.clip(e, 0.0, 1.0) - cheap)
        self.seen_since_fit += 1
        if len(self._x) > 4000:
            self._x = self._x[-3000:]
            self._y = self._y[-3000:]

    def observe(self, features, result) -> None:
        """Label from a Tier-2 result: the one call the verification loop makes."""
        if features:
            self.label(np.asarray(features, float), expensive_outcome(result))

    def observe_invalid(self, features) -> None:
        """Label a design the auditor invalidated: nothing survived."""
        if features:
            self.label(np.asarray(features, float), np.zeros(len(CRITIC_TARGETS)))

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
        Y = np.asarray(self._y, float)
        n_t = len(CRITIC_TARGETS)
        mean = X.mean(axis=0)
        sd = X.std(axis=0)
        scale = np.where(sd > 1e-9, sd, 1.0)
        W = np.zeros((X.shape[1], n_t))
        bias = np.zeros(n_t)
        skills, counts = np.zeros(n_t), np.zeros(n_t)
        by_target = {}

        # Each target is fitted on the rows where it was measured; a target
        # with too few labels predicts no gap and has no skill.
        #
        # Calibration is measured on the *expensive outcome* the critic implies,
        # cheap + predicted residual, against the one observed -- not on the
        # residual itself.  The residual contains -cheap and cheap is a feature,
        # so a residual fit correlates with its labels even when Tier-2 is pure
        # noise: measured 0.70 on random labels before this was changed.  It is
        # out-of-fold (five folds by index), so an overfit cannot vouch for
        # itself either.  And it is *skill over the cheap score*: how much
        # better the implied outcome ranks the observed one than the cheap
        # score alone does.  Where Tier-1 already predicts Tier-2 there is
        # nothing to correct; where neither predicts it, the critic must not
        # discount high cheap scores on regression to the mean.
        Z = (X - mean) / scale
        for j, k in enumerate(CRITIC_TARGETS):
            rows = np.isfinite(Y[:, j])
            n = int(rows.sum())
            if n < max(10, self.min_samples // 3):
                by_target[k] = {"labels": n, "skill": 0.0}
                continue
            Zj, yj = Z[rows], Y[rows, j]
            try:
                bias[j], W[:, j] = _ridge(Zj, yj, self.ridge)
                pj = np.empty(n)
                folds = np.arange(n) % 5
                for f in range(5):
                    tr, te = folds != f, folds == f
                    b, w = _ridge(Zj[tr], yj[tr], self.ridge)
                    pj[te] = Zj[te] @ w + b
            except np.linalg.LinAlgError:
                return False
            cheap = np.clip(X[rows, _TARGET_IDX[j]], 0.0, 1.0)
            skills[j] = _skill(np.clip(pj, -1.0, 1.0) + cheap, cheap, yj + cheap)
            counts[j] = n
            by_target[k] = {"labels": n, "skill": round(float(skills[j]), 3)}
        self._mean, self._scale, self._bias, self._w = mean, scale, bias, W
        self.calibration = (float(skills @ counts / counts.sum())
                            if counts.sum() > 0 else 0.0)
        self.calibration_by_target = by_target
        self.fits += 1
        self.seen_since_fit = 0
        return True

    # ----------------------------------------------------------- prediction

    def predict_components(self, features: np.ndarray) -> np.ndarray:
        """Predicted residual, expensive minus cheap, for each target."""
        zero = np.zeros(len(CRITIC_TARGETS))
        if not self.fitted:
            return zero
        f = np.asarray(features, float)
        if not np.all(np.isfinite(f)):
            return zero
        z = (f - self._mean) / self._scale
        return np.clip(z @ self._w + self._bias, -1.0, 1.0)

    def predict(self, features: np.ndarray) -> float:
        """Predicted mean residual: negative means the cheap score overstates."""
        return float(np.mean(self.predict_components(features)))

    def discount(self, features: np.ndarray) -> float:
        """Multiplier in [1 - max_discount, 1] to apply to a cheap score.

        Never above 1.  A critic that could raise a score would be a second
        objective for the population to optimise against, and the population is
        very good at optimising against things.  A predicted *positive* gap --
        the cheap tier undersells this design -- is information for choosing
        what to verify, not points.
        """
        if not self.fitted or self.calibration <= 0.05:
            return 1.0
        shortfall = float(np.clip(-self.predict(features), 0.0, 1.0))
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
        return float(np.clip(-self.predict(features), 0.0, 1.0)) * self.calibration

    # -------------------------------------------------------------- reading

    def distrusts(self, top: int = 3) -> list:
        """Which measurements the critic has learned to read as warning signs.

        The reason for a linear model: a run can say *what* the critic learned,
        and a critic whose top weights are nonsense is visible as nonsense
        rather than as an unexplained drop in everyone's score.  Read on the
        mean residual, the quantity the discount acts on.
        """
        if not self.fitted:
            return []
        w = self._w.mean(axis=1)
        order = np.argsort(w)
        return [
            {"feature": CRITIC_FEATURES[i], "weight": round(float(w[i]), 3)}
            for i in order[:top]
        ]

    def report(self) -> dict:
        return {
            "fitted": self.fitted,
            "labels": len(self._x),
            "fits": self.fits,
            "calibration": round(self.calibration, 3),
            "calibration_by_target": dict(self.calibration_by_target),
            "dropped_legacy": self.dropped_legacy,
            "distrusts": self.distrusts(),
        }


def _ridge(Z: np.ndarray, y: np.ndarray, ridge: float):
    """Ridge on standardised features, solved directly: (bias, weights).

    Sixteen features is nothing to invert, and a closed-form refit from scratch
    each time is what stops the critic from accumulating drift the way an
    incrementally-updated one would.
    """
    bias = float(y.mean())
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    return bias, np.linalg.solve(A, Z.T @ (y - bias))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Signed correlation, and 0 where either side is constant."""
    if float(np.std(a)) > 1e-9 and float(np.std(b)) > 1e-9:
        return float(np.corrcoef(a, b)[0, 1])
    return 0.0


def _skill(predicted: np.ndarray, cheap: np.ndarray, observed: np.ndarray) -> float:
    """How much better ``predicted`` ranks ``observed`` than ``cheap`` does, in [0, 1]."""
    return float(np.clip(_corr(predicted, observed) - max(_corr(cheap, observed), 0.0),
                         0.0, 1.0))
