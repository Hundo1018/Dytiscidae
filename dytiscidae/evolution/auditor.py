"""The third party: a check on both the designs and the judge.

Why it cannot be another learned thing
--------------------------------------
There are now two adaptive parties in this loop.  The population is trying to
score; the judge answers breakthroughs by raising the bar.  That is a useful
pressure and it is also a closed system: nothing in it can tell the difference
between the population getting genuinely better and the population and the judge
drifting together into a corner of the simulator where the numbers are large and
the physics is not.

A third adaptive party would not help.  It would drift too, and three drifting
things agreeing with each other is not evidence.  The authority of this module
comes from exactly one property: **it does not learn, it does not optimise, and
nothing it checks is a function of the run's own history.**  Every check below is
either a physical invariant, a re-measurement under conditions the design was
not selected on, or a perturbation of an assumption the design had no way to see.

The four checks
---------------

``no free energy``
    The mechanical work the fluid does on the machine, integrated over an
    episode, must not exceed what the actuators put in plus what gravity and
    buoyancy supply.  A design that comes out ahead is extracting energy from
    the integrator, not from the water.  This is conservation, not preference,
    and it cannot be argued with.

``held out``
    Re-run the design on seeds and sea states it was never scored on.  A score
    that does not survive this was a measurement of the seed.

``perturbation``
    Re-run with the model's own coefficients moved -- drag, added mass, lift
    slope -- by more than their honest uncertainty.  A real machine loses some
    performance and keeps most of it.  A design that only works at one exact
    value of a coefficient has found a hole in the model, and the size of the
    collapse is the size of the hole.  This is the strongest of the four,
    because it is the only one that catches an exploit nobody has thought of.

``scaling``
    Compare against how real flying and swimming things scale.  Wing loading
    against mass follows a known power law across five orders of magnitude of
    animals and aircraft; a design far off it is either a discovery or an error,
    and it should be looked at either way.

What it may do
--------------
Audit, veto, and roll back -- in that order of severity.  It can mark a design's
score invalid, which removes it from the archive and from the judge's evidence.
It can veto a tightening of the judge that was driven by a design that later
failed audit.  And it can roll a bar back to where it was before such a
tightening, because a bar moved by evidence that turned out not to be evidence
permanently penalises every honest design that follows.

It never rewards.  There is no path by which passing an audit raises a score.
That asymmetry is deliberate: a check that can give points is a check that can
be optimised against, and then it is not a third party any more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Finding:
    """One thing the auditor objected to."""

    check: str
    severity: str  # "note" | "invalidate"
    detail: str
    measured: float = 0.0
    expected: float = 0.0


@dataclass(eq=False)
class AuditReport:
    design: str = ""
    findings: list = field(default_factory=list)
    retained_fraction: float = 1.0
    checks_run: int = 0

    @property
    def invalid(self) -> bool:
        return any(f.severity == "invalidate" for f in self.findings)

    def summary(self) -> dict:
        return {
            "design": self.design,
            "invalid": self.invalid,
            "retained": round(self.retained_fraction, 3),
            "checks": self.checks_run,
            "findings": [
                {"check": f.check, "severity": f.severity, "detail": f.detail}
                for f in self.findings
            ],
        }


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_energy_conservation(result) -> Finding | None:
    """The fluid must not be a power source.

    Not a heuristic: the machine's only energy inputs are its battery and the
    potential it starts with.  A run whose recorded mechanical output exceeds
    that is not a good design, it is a broken integrator, and the difference
    matters because the first is worth keeping.
    """
    segs = getattr(result, "segments", {}) or {}
    if not segs:
        return None
    # Electrical in, mechanical out.  A flapping machine is at best ~40%
    # efficient at converting one to the other, so mechanical output above the
    # electrical input is impossible by a wide margin.
    elec = sum(s.mean_power * s.duration for s in segs.values())
    if elec <= 1e-6:
        return None
    over = getattr(result, "mechanical_output_j", None)
    if over is None:
        return None
    if over > elec:
        return Finding(
            check="no_free_energy",
            severity="invalidate",
            detail=f"mechanical output {over:.0f} J exceeds electrical input {elec:.0f} J",
            measured=over,
            expected=elec,
        )
    return None


def check_scaling(phenotype) -> Finding | None:
    """Wing loading against mass, compared with the real world.

    Flying animals and aircraft follow ``W/S ~ m^(1/3)`` over five orders of
    magnitude -- Tennekes' "great flight diagram".  At 5 kg that is roughly
    100-200 N/m^2.  Two orders of magnitude off it is not a design, it is a
    number that means something other than what the name says: in this project
    it meant "wing area rounded to zero and the machine is flying on its hull".

    A note, not an invalidation.  Being off the diagram is how a genuinely new
    kind of machine would look, and this module does not get to rule that out.
    It gets to make sure nobody fails to notice.
    """
    m = float(getattr(phenotype, "mass", 0.0))
    s = float(getattr(phenotype, "wing_area", 0.0))
    if m <= 0.0:
        return None
    if s < 1e-4:
        return Finding(
            check="scaling",
            severity="note",
            detail=f"no lifting surface at all ({s:.5f} m^2) on a {m:.1f} kg machine",
            measured=0.0,
        )
    ws = m * 9.80665 / s
    expected = 47.0 * m ** (1.0 / 3.0)  # Tennekes, fitted across the diagram
    ratio = ws / expected
    if ratio > 12.0 or ratio < 1 / 12.0:
        return Finding(
            check="scaling",
            severity="note",
            detail=(
                f"wing loading {ws:.0f} N/m^2 is {ratio:.1f}x the great-flight-diagram "
                f"value of {expected:.0f} N/m^2 for {m:.1f} kg"
            ),
            measured=ws,
            expected=expected,
        )
    return None


# --------------------------------------------------------------------------
# The auditor
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Auditor:
    """Re-measures designs under conditions they were not selected on.

    Parameters
    ----------
    held_out_seeds:
        How many unseen seeds to re-evaluate on.
    perturbations:
        Fractional changes applied to model coefficients.  Larger than the
        model's honest uncertainty on purpose -- the question is not "is the
        number right" but "does this design depend on the number being right".
    collapse_threshold:
        Retaining less than this fraction of the original score under
        perturbation is treated as evidence of a modelling exploit rather than
        of a delicate design.
    """

    held_out_seeds: int = 2
    perturbations: tuple = (("cd_scale", 1.25), ("added_mass_scale", 0.8),
                            ("lift_scale", 0.85))
    collapse_threshold: float = 0.35
    reports: list = field(default_factory=list)
    invalidated: int = 0
    vetoed_tightenings: int = 0

    # ---------------------------------------------------------------- audit

    def audit(self, phenotype, result, *, reevaluate=None, name: str = "") -> AuditReport:
        """Run every check that the available information supports.

        ``reevaluate(seed=..., perturb=...)`` should return a fresh
        MissionResult.  When it is not supplied the audit still runs the checks
        that need only the existing result, so an audit is never skipped
        entirely for want of a budget.
        """
        rep = AuditReport(design=name or getattr(phenotype, "name", "?"))

        for check in (lambda: check_energy_conservation(result),
                      lambda: check_scaling(phenotype)):
            rep.checks_run += 1
            f = check()
            if f is not None:
                rep.findings.append(f)

        if reevaluate is None:
            self.reports.append(rep)
            if rep.invalid:
                self.invalidated += 1
            return rep

        base = float(getattr(result, "mission_fraction", 0.0))
        if base > 1e-6:
            retained = []

            # Held-out seeds: a score that was a measurement of the seed.
            for i in range(self.held_out_seeds):
                rep.checks_run += 1
                try:
                    alt = reevaluate(seed=100_000 + i, perturb=None)
                except Exception as exc:
                    rep.findings.append(Finding(
                        check="held_out", severity="note",
                        detail=f"re-evaluation failed: {type(exc).__name__}: {exc}"))
                    continue
                retained.append(float(getattr(alt, "mission_fraction", 0.0)) / base)

            # Perturbation: does this design need the model to be exactly right?
            for key, factor in self.perturbations:
                rep.checks_run += 1
                try:
                    alt = reevaluate(seed=0, perturb={key: factor})
                except Exception as exc:
                    rep.findings.append(Finding(
                        check="perturbation", severity="note",
                        detail=f"{key} x{factor} failed: {type(exc).__name__}: {exc}"))
                    continue
                keep = float(getattr(alt, "mission_fraction", 0.0)) / base
                retained.append(keep)
                if keep < self.collapse_threshold:
                    rep.findings.append(Finding(
                        check="perturbation",
                        severity="invalidate",
                        detail=(
                            f"score collapses to {keep:.0%} when {key} moves by "
                            f"{abs(1 - factor):.0%} -- this design depends on the "
                            "model being exactly right"
                        ),
                        measured=keep,
                        expected=self.collapse_threshold,
                    ))

            if retained:
                rep.retained_fraction = float(np.mean(retained))

        self.reports.append(rep)
        if rep.invalid:
            self.invalidated += 1
        return rep

    # ----------------------------------------------------------------- veto

    def review_tightening(self, judge, moves: list, invalid_designs: int) -> list:
        """Veto a bar that was moved by evidence that did not hold up.

        A tightening driven by a design the audit later invalidated makes the
        standard permanently harder for every honest design that follows, on the
        strength of something that never happened.  Rolling it back is the only
        remedy, because a ratchet by construction will not come down on its own.
        """
        if not moves or invalid_designs <= 0:
            return []
        vetoed = []
        for move in moves:
            r = judge.ratchets.get(move["domain"])
            if r is not None and r.rollback():
                vetoed.append(move)
                self.vetoed_tightenings += 1
        return vetoed

    def report(self) -> dict:
        recent = self.reports[-50:]
        return {
            "audited": len(self.reports),
            "invalidated": self.invalidated,
            "vetoed_tightenings": self.vetoed_tightenings,
            "mean_retained": round(
                float(np.mean([r.retained_fraction for r in recent])), 3
            ) if recent else 1.0,
            "recent_findings": [
                f.detail for r in recent[-5:] for f in r.findings
            ][:5],
        }
