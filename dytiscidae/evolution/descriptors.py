"""Learned behaviour descriptors, so the archive axes stop being my guesses.

The problem
-----------
``BD_AXES`` was a list I wrote: log mass, density ratio, air competence, water
competence.  Four choices, each of which decides what the search considers a
"different kind of machine", and none of which the system had any say in.  Two
designs that differ in a way none of those axes captures collide in the same
cell and one is discarded -- so the axes silently bound what can be found, in
the same way the part taxonomy bounded what could be built.

The fix, following AURORA (Cully 2019, *Autonomous skill discovery with
quality-diversity and unsupervised descriptors*): record a fixed, generic
feature vector from every episode, learn a low-dimensional projection of it from
the data itself, and use that latent as the archive descriptor.  The system then
decides what behavioural difference means, and the definition moves as the
population moves.

Why PCA rather than an autoencoder
----------------------------------
An autoencoder is the usual choice and is strictly more expressive.  On four CPU
cores it is also several minutes of training per refit, competing with the
evaluations that produce the data.  Incremental PCA costs milliseconds, is
deterministic, and captures the dominant behavioural variation, which is what
the descriptor needs.  The interface below takes a projector, so swapping in an
autoencoder later changes one class and nothing else.

The honest caveat
-----------------
Learned descriptors move.  An archive built under one projection is not directly
comparable to one built under another, so refits are deliberately infrequent and
the archive is re-binned when they happen.  A projection that never refits is
just a different set of fixed axes; one that refits constantly destroys the
archive it is supposed to organise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The raw per-episode observation the descriptor is learned from.
#:
#: Deliberately generic and behavioural rather than morphological: these are
#: things the machine *did*, not things it *is*.  Morphological axes let the
#: archive fill up with structurally distinct machines that all behave the same,
#: which is diversity in the wrong currency.
FEATURE_NAMES = (
    "air_time_fraction", "water_time_fraction", "land_time_fraction",
    "mean_speed_air", "mean_speed_water", "mean_speed_land",
    "max_depth", "net_altitude_change", "vertical_speed_rms",
    "attitude_variability", "power_air", "power_water", "power_land",
    "actuation_duty", "transition_count", "peak_stress",
)
FEATURE_DIM = len(FEATURE_NAMES)


def episode_features(result, phenotype) -> np.ndarray:
    """Extract the raw behaviour vector from a mission result.

    Missing quantities become zero rather than raising: a candidate that died in
    its first second still has a behaviour, and it is a useful one to keep in
    the archive as the boundary of what does not work.
    """
    seg = getattr(result, "segments", {}) or {}

    def g(dom, attr, default=0.0):
        s = seg.get(dom)
        return float(getattr(s, attr, default)) if s is not None else default

    total = sum(g(d, "duration") for d in ("air", "water", "land")) or 1.0
    depths = [g(d, "max_depth") for d in ("air", "water", "land")]
    return np.array([
        g("air", "duration") / total,
        g("water", "duration") / total,
        g("land", "duration") / total,
        g("air", "mean_speed"),
        g("water", "mean_speed"),
        g("land", "mean_speed"),
        max(depths),
        g("air", "altitude_held"),
        float(np.mean([g(d, "attitude_rms") for d in ("air", "water", "land")])),
        g("air", "attitude_rms"),
        g("air", "mean_power"),
        g("water", "mean_power"),
        g("land", "mean_power"),
        float(getattr(phenotype, "n_actuated", 0)),
        float(sum(1 for v in (getattr(result, "transition_ok", {}) or {}).values() if v)),
        float(max((g(d, "max_actuator_overload") for d in ("air", "water", "land")), default=0.0)),
    ], dtype=float)


@dataclass
class LearnedDescriptors:
    """Projects raw behaviour features onto a learned low-dimensional space.

    Parameters
    ----------
    n_dims:
        Archive dimensionality.  Four matches what the hand-picked axes used, so
        the archive's capacity and the curator's crowding logic are unchanged.
    refit_every:
        Episodes between refits.  Infrequent on purpose -- see the module note.
    min_samples:
        Below this the projection is not fitted at all and the raw first
        ``n_dims`` features are used, which keeps early generations working
        before there is anything to learn from.
    """

    n_dims: int = 4
    refit_every: int = 400
    min_samples: int = 60

    _buffer: list = field(default_factory=list)
    _mean: np.ndarray | None = None
    _components: np.ndarray | None = None
    _scale: np.ndarray | None = None
    seen: int = 0
    refits: int = 0

    # ---------------------------------------------------------------- record

    def observe(self, features: np.ndarray) -> None:
        self._buffer.append(np.asarray(features, float))
        self.seen += 1
        # Bounded memory, biased toward recent behaviour: the definition of
        # "different" should track where the population actually is now.
        if len(self._buffer) > 3000:
            self._buffer = self._buffer[-2000:]

    def due_for_refit(self) -> bool:
        return (
            len(self._buffer) >= self.min_samples
            and self.seen % max(self.refit_every, 1) == 0
        )

    # ------------------------------------------------------------------- fit

    def fit(self) -> bool:
        """Refit the projection.  Returns True if it changed."""
        if len(self._buffer) < self.min_samples:
            return False
        X = np.asarray(self._buffer, float)
        X = np.where(np.isfinite(X), X, 0.0)

        self._mean = X.mean(axis=0)
        # Standardise before PCA: power is in watts and time fractions are in
        # [0,1], so without this the projection is entirely about power.
        sd = X.std(axis=0)
        self._scale = np.where(sd > 1e-9, sd, 1.0)
        Z = (X - self._mean) / self._scale

        # Economy SVD rather than an eigendecomposition of the covariance: same
        # answer, better conditioned, and 16 features is nothing to factor.
        try:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            return False
        self._components = Vt[: self.n_dims]
        self.refits += 1
        return True

    @property
    def fitted(self) -> bool:
        return self._components is not None

    # --------------------------------------------------------------- project

    def project(self, features: np.ndarray) -> np.ndarray:
        """Map a behaviour vector to archive coordinates."""
        f = np.asarray(features, float)
        f = np.where(np.isfinite(f), f, 0.0)
        if not self.fitted:
            # Before there is anything to learn from, fall back to raw features.
            # Not a good descriptor, but a working one, and it is replaced as
            # soon as there is data.
            return f[: self.n_dims]
        z = (f - self._mean) / self._scale
        return self._components @ z

    def bounds(self) -> np.ndarray:
        """Empirical range of each latent axis, for binning the archive.

        Recomputed from the buffer rather than assumed, because a learned axis
        has no natural units and its range is only knowable from the data.
        """
        if not self._buffer:
            return np.tile(np.array([-1.0, 1.0]), (self.n_dims, 1))
        P = np.array([self.project(f) for f in self._buffer])
        lo = np.percentile(P, 2, axis=0)
        hi = np.percentile(P, 98, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        return np.stack([lo - 0.05 * span, hi + 0.05 * span], axis=1)

    def axis_meaning(self, i: int, top: int = 3) -> str:
        """Which raw features a learned axis is mostly made of.

        The whole point of learning the axes is that nobody chose them, so the
        run has to be able to report what they turned out to be -- otherwise the
        archive is a map with unlabelled coordinates.
        """
        if not self.fitted or i >= len(self._components):
            return f"raw:{FEATURE_NAMES[i]}" if i < FEATURE_DIM else "?"
        w = self._components[i]
        order = np.argsort(-np.abs(w))[:top]
        return " ".join(
            f"{'+' if w[j] > 0 else '-'}{FEATURE_NAMES[j]}({abs(w[j]):.2f})" for j in order
        )

    def report(self) -> dict:
        return {
            "fitted": self.fitted,
            "samples": len(self._buffer),
            "refits": self.refits,
            "axes": [self.axis_meaning(i) for i in range(self.n_dims)] if self.fitted else [],
        }
