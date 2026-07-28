"""CMA-ES for controller weights.

A compact, dependency-free implementation of the (mu/mu_w, lambda) covariance
matrix adaptation strategy.  It is here rather than pulled from a library
because the whole system is meant to be portable and CPU-first, and because the
controller vectors are small (a few hundred weights) -- well inside the range
where full-covariance CMA-ES is both affordable and clearly better than the
diagonal approximations.

It is used two ways:

* **Controller optimisation.**  Given a fixed morphology, optimise the policy
  weights and the CPG parameters that drive its discovered mobility basis.
* **Archive-guided emitters.**  A CMA-ES instance seeded at an archive elite,
  rewarded for *improving the archive* rather than for raw fitness, is the CMA-ME
  emitter idea: it lets a local search push into unexplored cells instead of
  merely refining one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CMAES:
    """Covariance matrix adaptation evolution strategy.

    Parameters
    ----------
    x0 :
        Initial mean.
    sigma0 :
        Initial step size, in the same units as ``x0``.
    popsize :
        Defaults to the standard ``4 + floor(3 ln n)``.
    """

    x0: np.ndarray
    sigma0: float = 0.3
    popsize: int | None = None
    seed: int = 0
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.x0, float).copy()
        self.n = len(self.mean)
        self.sigma = float(self.sigma0)
        self.rng = np.random.default_rng(self.seed)

        self.lam = self.popsize or (4 + int(3 * math.log(max(self.n, 2))))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)

        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.generation = 0
        self._eig_stale = 0
        self._pop: np.ndarray | None = None
        self.best_x = self.mean.copy()
        self.best_f = -np.inf

    # ------------------------------------------------------------------- ask

    def ask(self) -> np.ndarray:
        """Sample a population.  Returns ``(lam, n)``."""
        self._update_eigen()
        z = self.rng.standard_normal((self.lam, self.n))
        y = (self.B * self.D) @ z.T
        pop = self.mean[None, :] + self.sigma * y.T
        if self.bounds is not None:
            pop = np.clip(pop, self.bounds[0], self.bounds[1])
        self._pop = pop
        return pop

    # ------------------------------------------------------------------ tell

    def tell(self, pop: np.ndarray, scores: np.ndarray) -> None:
        """Update the distribution.  ``scores`` are maximised."""
        pop = np.asarray(pop, float)
        scores = np.asarray(scores, float)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        order = np.argsort(-scores)
        selected = pop[order[: self.mu]]

        if scores[order[0]] > self.best_f:
            self.best_f = float(scores[order[0]])
            self.best_x = pop[order[0]].copy()

        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        y = (self.mean - old_mean) / max(self.sigma, 1e-12)
        self._update_eigen()
        c_inv_sqrt = self.B @ np.diag(1.0 / np.maximum(self.D, 1e-12)) @ self.B.T

        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (c_inv_sqrt @ y)

        self.generation += 1
        hsig = float(
            np.linalg.norm(self.ps)
            / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
            / self.chiN
            < 1.4 + 2 / (self.n + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y

        ys = (selected - old_mean) / max(self.sigma, 1e-12)
        rank_mu = (ys * self.weights[:, None]).T @ ys
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc)
                         + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.sigma *= math.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )
        self.sigma = float(np.clip(self.sigma, 1e-8, 1e3))
        self._eig_stale += 1

    def _update_eigen(self) -> None:
        if self._eig_stale < max(1, int(self.lam / (10 * self.n * (self.c1 + self.cmu)))):
            if self._eig_stale > 0:
                return
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        try:
            d, B = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            # A degenerate covariance is recoverable: restart from isotropic
            # rather than aborting a run that may be hours old.
            self.C = np.eye(self.n)
            d, B = np.ones(self.n), np.eye(self.n)
        d = np.maximum(d, 1e-20)
        self.D = np.sqrt(d)
        self.B = B
        self._eig_stale = 0

    # ------------------------------------------------------------ diagnostics

    @property
    def converged(self) -> bool:
        """True when further sampling is unlikely to find anything."""
        return bool(self.sigma * self.D.max() < 1e-6 or self.D.max() / self.D.min() > 1e7)

    def state(self) -> dict:
        return {
            "generation": self.generation,
            "sigma": self.sigma,
            "best_f": self.best_f,
            "condition": float(self.D.max() / max(self.D.min(), 1e-20)),
        }


@dataclass
class Emitter:
    """A CMA-ES instance whose objective is *archive improvement*.

    This is the CMA-ME idea.  A plain CMA-ES seeded at an elite will refine that
    elite and stop; an emitter scored by "did this land in an empty cell, or beat
    the occupant, and by how much" will instead walk along the frontier of the
    archive, filling it.  The two behaviours are wanted at different times, so
    the curator runs a mix and adjusts the ratio from how the archive is
    actually growing.
    """

    optimiser: CMAES
    parent_cell: tuple[int, ...] | None = None
    kind: str = "improvement"  # or "optimising"
    restarts: int = 0
    placed: int = 0
    evaluated: int = 0
    history: list[float] = field(default_factory=list)

    def rank_score(self, status: str, fitness: float, previous: float) -> float:
        """Turn an archive outcome into something CMA-ES can maximise.

        A new cell is worth more than an improvement, because coverage is the
        scarce resource; an improvement is worth its size; a rejection is worth
        its shortfall, so the search still knows which direction it missed by.
        """
        if self.kind == "optimising":
            return fitness
        if status == "new":
            return 1.0 + fitness
        if status == "improved":
            return fitness - previous
        return -0.05 - max(previous - fitness, 0.0)

    @property
    def yield_rate(self) -> float:
        return self.placed / max(self.evaluated, 1)
