"""A dimensional-analysis engine small enough to read in one sitting.

Why this exists
---------------
The project's failure mode is not "cannot use a formula".  It is "used a
formula whose symbols mean something slightly different here", and the cheapest
detector for that is dimensions.  `k = omega * c / (2 * U)` is dimensionless;
if the expression you wrote is not, the symbols are not what you think.

Two things are deliberately *not* here:

* No unit conversion.  Everything in this project is SI already, and a
  conversion layer would invite the "I thought it was in millimetres" bug it is
  supposed to prevent.
* No integration with numpy.  `Q` carries a single float.  Dimensional checks
  run on one representative scalar per formula, not on the arrays the solver
  actually evaluates, because dimensions do not depend on the value.

The base set is (metre, kilogram, second).  Radians, revolutions and any other
angle are dimensionless, which is the convention the SI uses and the one that
makes `omega * c / U` come out right.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Dim:
    """Exponents of metre, kilogram and second."""

    m: Fraction = Fraction(0)
    kg: Fraction = Fraction(0)
    s: Fraction = Fraction(0)

    @staticmethod
    def of(m=0, kg=0, s=0) -> "Dim":
        return Dim(Fraction(m), Fraction(kg), Fraction(s))

    def __mul__(self, o: "Dim") -> "Dim":
        return Dim(self.m + o.m, self.kg + o.kg, self.s + o.s)

    def __truediv__(self, o: "Dim") -> "Dim":
        return Dim(self.m - o.m, self.kg - o.kg, self.s - o.s)

    def __pow__(self, k) -> "Dim":
        k = Fraction(k).limit_denominator(1000)
        return Dim(self.m * k, self.kg * k, self.s * k)

    @property
    def dimensionless(self) -> bool:
        return self.m == 0 and self.kg == 0 and self.s == 0

    def __str__(self) -> str:
        if self.dimensionless:
            return "1"
        parts = []
        for sym, e in (("m", self.m), ("kg", self.kg), ("s", self.s)):
            if e == 0:
                continue
            parts.append(sym if e == 1 else f"{sym}^{e}")
        return ".".join(parts)


ONE = Dim.of()
M = Dim.of(m=1)
KG = Dim.of(kg=1)
S = Dim.of(s=1)
M2 = M**2
M3 = M**3
HZ = ONE / S
VELOCITY = M / S
ACCEL = M / S**2
DENSITY = KG / M3
VISCOSITY = KG / (M * S)  # Pa.s
FORCE = KG * M / S**2
PRESSURE = FORCE / M2
ENERGY = FORCE * M
POWER = ENERGY / S
MOMENT = FORCE * M
INERTIA = KG * M2
ANG_VEL = ONE / S  # radian is dimensionless
ANG_ACCEL = ONE / S**2
AREA_MOMENT = M**4
SECTION_MODULUS = M3
VOLUME_FLOW = M3 / S
MOTOR_CONSTANT = MOMENT / POWER ** Fraction(1, 2)


class DimensionError(AssertionError):
    """Raised when an expression's dimensions do not work out."""


@dataclass(frozen=True)
class Q:
    """A scalar with a dimension.  Arithmetic propagates the dimension."""

    value: float
    dim: Dim = ONE

    # -- construction helpers
    @staticmethod
    def of(value: float, dim: Dim = ONE) -> "Q":
        return Q(float(value), dim)

    # -- arithmetic
    def __add__(self, o: "Q") -> "Q":
        o = _as_q(o)
        if self.dim != o.dim:
            raise DimensionError(f"cannot add [{self.dim}] to [{o.dim}]")
        return Q(self.value + o.value, self.dim)

    def __sub__(self, o: "Q") -> "Q":
        o = _as_q(o)
        if self.dim != o.dim:
            raise DimensionError(f"cannot subtract [{o.dim}] from [{self.dim}]")
        return Q(self.value - o.value, self.dim)

    def __mul__(self, o) -> "Q":
        o = _as_q(o)
        return Q(self.value * o.value, self.dim * o.dim)

    def __rmul__(self, o) -> "Q":
        return self.__mul__(o)

    def __truediv__(self, o) -> "Q":
        o = _as_q(o)
        return Q(self.value / o.value, self.dim / o.dim)

    def __rtruediv__(self, o) -> "Q":
        return _as_q(o).__truediv__(self)

    def __pow__(self, k) -> "Q":
        return Q(self.value ** float(k), self.dim ** k)

    def __neg__(self) -> "Q":
        return Q(-self.value, self.dim)

    def __abs__(self) -> "Q":
        return Q(abs(self.value), self.dim)

    def sqrt(self) -> "Q":
        return self ** Fraction(1, 2)

    def __repr__(self) -> str:
        return f"{self.value:.6g} [{self.dim}]"

    def expect(self, dim: Dim, what: str = "") -> "Q":
        """Assert the dimension, returning self so this chains."""
        if self.dim != dim:
            raise DimensionError(
                f"{what or 'expression'}: got [{self.dim}], expected [{dim}]")
        return self


def _as_q(o) -> Q:
    """A bare number is dimensionless.  This is the one place that is assumed."""
    return o if isinstance(o, Q) else Q(float(o), ONE)


def dimensionless(q: Q, what: str = "") -> Q:
    return q.expect(ONE, what)
