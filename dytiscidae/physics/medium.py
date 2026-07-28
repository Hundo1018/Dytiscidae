"""Medium field: air above the free surface, water below, and the blended
transition layer in between.

The free surface is the single most important physical feature in this project.
A triphibian machine spends most of its interesting time *straddling* it, so we
never model "in air" and "in water" as two discrete cases.  Instead every
immersible element gets a continuous submerged fraction in [0, 1] and the local
density / viscosity are blended accordingly.  Everything downstream (blade
element loads, buoyancy, added mass, drag) then behaves continuously through
water entry and exit, which keeps the dynamics integrable and keeps the
optimiser from discovering discontinuity exploits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665  # m/s^2
P_ATM = 101325.0  # Pa


@dataclass(frozen=True)
class Fluid:
    """A Newtonian fluid at a reference state."""

    name: str
    rho: float  # density, kg/m^3
    mu: float  # dynamic viscosity, Pa.s

    @property
    def nu(self) -> float:
        """Kinematic viscosity, m^2/s."""
        return self.mu / self.rho


AIR = Fluid("air", 1.225, 1.81e-5)
FRESHWATER = Fluid("freshwater", 998.0, 1.002e-3)
SEAWATER = Fluid("seawater", 1025.0, 1.08e-3)

# Ratio of water to air density.  Roughly 837 for seawater.  This number is the
# reason a triphibian vehicle is hard: the same wing that must generate 15 kg of
# lift in air sees ~840x the dynamic pressure in water at the same speed.
DENSITY_RATIO = SEAWATER.rho / AIR.rho


@dataclass
class SeaState:
    """A simple linear (Airy) wave field used to perturb the free surface.

    Sea state 0 is a flat calm.  The default (amplitude 0.05 m, 2 s period)
    corresponds to a light chop, which is enough to make water entry and exit
    non-trivial without making the mission impossible.
    """

    amplitude: float = 0.0  # m, wave amplitude (half of peak-to-trough)
    period: float = 2.0  # s
    wavelength: float = 6.0  # m
    direction: float = 0.0  # rad, propagation heading in the XY plane

    def surface_z(self, xy: np.ndarray, t: float) -> np.ndarray:
        """Free-surface elevation at world XY positions ``xy`` (..., 2)."""
        if self.amplitude <= 0.0:
            return np.zeros(xy.shape[:-1])
        k = 2.0 * np.pi / self.wavelength
        omega = 2.0 * np.pi / self.period
        khat = np.array([np.cos(self.direction), np.sin(self.direction)])
        phase = k * (xy @ khat) - omega * t
        return self.amplitude * np.sin(phase)

    def orbital_velocity(self, xyz: np.ndarray, t: float) -> np.ndarray:
        """Water particle velocity of the wave field at ``xyz`` (..., 3).

        Deep-water Airy orbitals, decaying as exp(k*z) with depth.  This gives
        the vehicle something real to fight against near the surface, which is
        exactly where amphibious transitions fail in practice.
        """
        out = np.zeros_like(xyz)
        if self.amplitude <= 0.0:
            return out
        k = 2.0 * np.pi / self.wavelength
        omega = 2.0 * np.pi / self.period
        khat = np.array([np.cos(self.direction), np.sin(self.direction)])
        z = np.minimum(xyz[..., 2], 0.0)
        decay = np.exp(k * z)
        phase = xyz[..., :2] @ khat - omega * t
        u_mag = self.amplitude * omega * decay
        out[..., 0] = u_mag * np.cos(phase) * khat[0]
        out[..., 1] = u_mag * np.cos(phase) * khat[1]
        out[..., 2] = u_mag * np.sin(phase)
        return out


class MediumField:
    """Queries the local fluid state anywhere in the world.

    Parameters
    ----------
    water:
        The liquid phase.  Seawater by default -- it is denser (more buoyancy,
        more drag) and more corrosive than fresh water, so designing against it
        is the conservative choice.
    sea_state:
        Free-surface wave field.
    current:
        Constant water current in world coordinates, m/s.
    wind:
        Constant air velocity in world coordinates, m/s.
    """

    def __init__(
        self,
        water: Fluid = SEAWATER,
        air: Fluid = AIR,
        sea_state: SeaState | None = None,
        current: np.ndarray | None = None,
        wind: np.ndarray | None = None,
    ) -> None:
        self.water = water
        self.air = air
        self.sea_state = sea_state or SeaState()
        self.current = np.zeros(3) if current is None else np.asarray(current, float)
        self.wind = np.zeros(3) if wind is None else np.asarray(wind, float)

    # ---------------------------------------------------------------- geometry

    def submerged_fraction(
        self, pos: np.ndarray, half_height: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Fraction of an element that is under water, in [0, 1].

        The element is treated as a vertical extent of ``2 * half_height``
        centred on ``pos``.  The result is a linear ramp through the surface
        rather than a step, which is what keeps the transition differentiable.

        ``half_height`` is clamped away from zero so that infinitesimally thin
        elements (a wing panel edge-on to the surface) still produce a finite
        gradient instead of a numerical cliff.
        """
        pos = np.atleast_2d(pos)
        h = np.maximum(np.asarray(half_height, float), 1e-3)
        z_surf = self.sea_state.surface_z(pos[..., :2], t)
        # Depth of the element centre below the local surface.
        depth = z_surf - pos[..., 2]
        return np.clip(0.5 + 0.5 * depth / h, 0.0, 1.0)

    def depth(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Depth below the local free surface, metres.  Negative when in air."""
        pos = np.atleast_2d(pos)
        return self.sea_state.surface_z(pos[..., :2], t) - pos[..., 2]

    # --------------------------------------------------------------- properties

    def properties(
        self, pos: np.ndarray, half_height: np.ndarray, t: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(rho, mu, submerged_fraction)`` at each element.

        Density blends linearly with submerged fraction (a volume average, which
        is exact for buoyancy).  Viscosity blends geometrically, because it
        spans five orders of magnitude and a linear blend would be dominated by
        the water value the instant an element touched the surface.
        """
        f = self.submerged_fraction(pos, half_height, t)
        rho = self.air.rho + f * (self.water.rho - self.air.rho)
        mu = self.air.mu ** (1.0 - f) * self.water.mu**f
        return rho, mu, f

    def flow_velocity(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Ambient fluid velocity at ``pos``: wind in air, current + waves in water."""
        pos = np.atleast_2d(pos)
        f = self.submerged_fraction(pos, np.full(pos.shape[0], 1e-2), t)[:, None]
        water_v = self.current[None, :] + self.sea_state.orbital_velocity(pos, t)
        return (1.0 - f) * self.wind[None, :] + f * water_v

    def pressure(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Absolute ambient pressure, Pa.  Used for hull and seal checks."""
        d = np.maximum(self.depth(pos, t), 0.0)
        return P_ATM + self.water.rho * GRAVITY * d


def hydrostatic_pressure(depth_m: float, water: Fluid = SEAWATER) -> float:
    """Absolute pressure at a depth, Pa.  ``depth_m`` is positive downwards.

    At the 10 m design depth this is about 2.0 bar absolute, i.e. 1.0 bar
    gauge.  That is a mild pressure vessel problem -- the hard part of a 10 m
    dive is not crush strength but sealing every actuator penetration and
    controlling buoyancy precisely enough to hold depth.
    """
    return P_ATM + water.rho * GRAVITY * max(depth_m, 0.0)
