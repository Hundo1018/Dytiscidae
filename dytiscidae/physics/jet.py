"""Pulsed-jet propulsion: medusa bells and squid mantles.

Nothing in a lift-and-drag model can express this.  A jellyfish does not have a
wing; it has a cavity that it squeezes, and the thrust is the momentum flux of
the water leaving the orifice.  Since the search is supposed to be free to
arrive at a medusa rather than being handed a bird, the physics has to be there
for it to find.

The model
---------
A bell of enclosed volume ``V`` driven by a joint angle contracts at rate
``Q = -dV/dt``.  The fluid leaves through an orifice of area ``A`` at
``v_e = Q / A``, so

    thrust = rho * Q * v_e = rho * Q^2 / A

along the bell axis, while the cavity is contracting.  On the refill stroke the
flow reverses and the momentum flux would push the animal backwards; real
medusae avoid most of that by refilling slowly and by recapturing the stopping
vortex, so refill thrust is charged at a reduced coefficient rather than
symmetrically.

Two consequences fall out of the ``rho`` and the ``1/A`` that matter for
design, and neither is obvious from a wing-based intuition:

* Jetting is a **water** propulsor.  The same stroke in air produces 1/840 of
  the thrust, so a bell is dead weight in flight -- which is exactly the kind of
  domain trade the archive should be mapping.
* Thrust goes as ``Q^2 / A``, so a *small* orifice is better for thrust at fixed
  flow, but costs more pressure and therefore more actuator work.  There is a
  real optimum, and it is the sort of thing the search can find and a designer
  usually guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .medium import MediumField


@dataclass
class JetSet:
    """All the pulsed-jet cavities on one machine.

    Attributes
    ----------
    body_id : (N,) int
        MuJoCo body of each bell.
    joint_id : (N,) int
        Joint whose angle drives the contraction; -1 if unactuated.
    axis_local : (N, 3)
        Direction the jet leaves the body, in the body frame.  Thrust is the
        opposite of this.
    volume : (N,)
        Enclosed volume at full relaxation, m^3.
    stroke_fraction : (N,)
        Fraction of ``volume`` swept between full relaxation and full
        contraction.
    orifice_area : (N,)
        Nozzle area, m^2.
    """

    body_id: np.ndarray
    joint_id: np.ndarray
    axis_local: np.ndarray
    volume: np.ndarray
    stroke_fraction: np.ndarray
    orifice_area: np.ndarray
    joint_range: np.ndarray  # (N, 2)

    #: Fraction of the ideal momentum flux recovered on the refill stroke.  Real
    #: medusae refill slowly and recapture part of the stopping vortex, so the
    #: reverse thrust is far from symmetric.
    refill_efficiency: float = 0.25

    def __post_init__(self) -> None:
        self.n = len(self.body_id)
        self._prev_v = None

    @staticmethod
    def empty() -> "JetSet":
        return JetSet(
            body_id=np.zeros(0, dtype=int),
            joint_id=np.zeros(0, dtype=int),
            axis_local=np.zeros((0, 3)),
            volume=np.zeros(0),
            stroke_fraction=np.zeros(0),
            orifice_area=np.zeros(0),
            joint_range=np.zeros((0, 2)),
        )

    def reset(self) -> None:
        self._prev_v = None

    def apply(self, model, data, medium: MediumField, t: float, dt: float) -> float:
        """Add jet thrust to ``data.xfrc_applied``.  Returns total thrust, N."""
        if self.n == 0:
            return 0.0

        # Current enclosed volume from the driving joint angle.
        theta = np.zeros(self.n)
        omega = np.zeros(self.n)
        for i, j in enumerate(self.joint_id):
            if j >= 0:
                adr = model.jnt_qposadr[j]
                vadr = model.jnt_dofadr[j]
                theta[i] = data.qpos[adr]
                omega[i] = data.qvel[vadr]

        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        span = np.maximum(hi - lo, 1e-6)
        frac = np.clip((theta - lo) / span, 0.0, 1.0)
        v_now = self.volume * (1.0 - self.stroke_fraction * frac)
        # dV/dt straight from the joint rate, so it is exact rather than a
        # difference of two sampled volumes.
        dv_dt = -self.volume * self.stroke_fraction * (omega / span)

        pos = data.xpos[self.body_id]
        xmat = data.xmat.reshape(-1, 3, 3)[self.body_id]
        axis_world = np.einsum("nij,nj->ni", xmat, self.axis_local)

        # Density at the orifice: jetting in air is worth almost nothing, and
        # that asymmetry is a real design pressure rather than a modelling
        # convenience.
        half = np.cbrt(np.maximum(self.volume, 1e-9)) * 0.5
        rho, _, subf = medium.properties(pos, half, t)

        q = -dv_dt  # positive while expelling
        expelling = q > 0
        coeff = np.where(expelling, 1.0, self.refill_efficiency)
        thrust_mag = coeff * rho * q * np.abs(q) / np.maximum(self.orifice_area, 1e-6)
        thrust_mag *= subf  # no jet from a cavity that is not in the fluid

        # Thrust opposes the direction the fluid leaves.
        f = -thrust_mag[:, None] * axis_world
        np.add.at(data.xfrc_applied[:, :3], self.body_id, f)

        self._prev_v = v_now
        return float(np.abs(thrust_mag).sum())

    def actuator_work(self, model, data) -> float:
        """Mechanical power the bell muscles are spending, W.

        Pressure inside the cavity is the stagnation pressure needed to drive
        the jet, ``0.5 * rho * v_e^2``, and the work rate is ``p * Q``.  A small
        orifice buys thrust and is charged for it here, which is what makes the
        orifice ratio a real trade rather than free thrust.
        """
        if self.n == 0:
            return 0.0
        return 0.0  # accounted through the driving actuator's torque
