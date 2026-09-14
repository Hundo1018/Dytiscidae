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

Who pays for the water
----------------------
Thrust is momentum flux, and momentum flux is not free.  The jet leaves at
``v_e = Q / A``, so the cavity must be held at the stagnation pressure needed
to drive it, ``p = 0.5 * rho * v_e^2``, and the muscle squeezing the bell does
work at

    P_pump = p * Q = 0.5 * rho * Q^3 / A^2 = 0.5 * m_dot * v_e^2

-- the same number written two ways, as it must be.  That power has to come
through the joint that drives the contraction, so it appears as a load torque

    tau = p * |dV/dtheta|,     dV/dtheta = -V0 * stroke_fraction / span

which makes ``tau * omega = p * Q`` exactly.  Until this was applied, the
thrust existed and the work did not: a bell produced 141.61 J of jet and the
actuator was charged 0.011 J of it, measured in ``experiments/jet_energy``.
Free thrust in water is the single most exploitable thing this project can
contain, and it was in the module written so the search could find a medusa.

It is applied as **joint damping**, not as an explicit torque, and that is not
a detail.  Substituting ``Q = (dV/dtheta) omega`` gives

    tau = -c(omega) * omega,    c = 0.5 rho (dV/dtheta)^3 |omega| / A^2

-- a velocity-dependent load whose effective damping coefficient runs into the
hundreds of N.m.s/rad against a bell inertia of a few thousandths of a kg.m^2.
Applied explicitly, that is unstable for the same reason added mass is: the
force is computed from last step's velocity, so a stiff dissipative term
overshoots, changes sign and diverges.  Measured directly: the first version of
this reached a jet velocity of 7.4e6 m/s within four seconds.  Written into
``dof_damping`` instead, MuJoCo's ``implicitfast`` integrator inverts it along
with the rest of the dynamics and it is unconditionally stable however large
the coefficient gets.  Same lesson as the mass matrix, same shape of fix.

Two consequences fall out of the ``rho`` and the ``1/A`` that matter for
design, and neither is obvious from a wing-based intuition:

* Jetting is a **water** propulsor.  The same stroke in air produces 1/840 of
  the thrust, so a bell is dead weight in flight -- which is exactly the kind of
  domain trade the archive should be mapping.
* Thrust goes as ``Q^2 / A``, so a *small* orifice is better for thrust at fixed
  flow, but costs more pressure and therefore more actuator work.  There is a
  real optimum, and it is the sort of thing the search can find and a designer
  usually guesses.  That trade is only real because the work is charged:
  thrust goes as ``1/A`` and the work as ``1/A^2``, so halving the orifice
  doubles the thrust and quadruples the bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .medium import MediumField


@dataclass(eq=False)
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
        #: DOF address of each driving joint, resolved against the model on
        #: first use.  -1 for a bell whose part carries no joint.
        self._dofadr: np.ndarray | None = None
        #: The model's own joint damping, kept so the pumping load can be
        #: rebuilt from scratch each step rather than accumulating, and
        #: restored on reset.  `FluidSolver` keeps `_dry_mass` for the same
        #: reason and learned it the same way.
        self._dry_damping: np.ndarray | None = None
        #: Mechanical power the bell muscles spent last step, W.  Recorded so
        #: the energy budget and the telemetry can see it separately from the
        #: work of swinging the bell's own inertia.
        self.last_pump_power = 0.0

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

    def reset(self, model=None) -> None:
        self._prev_v = None
        self.last_pump_power = 0.0
        # Leaving a previous episode's pumping load in the model would quietly
        # make the next episode's bells stiffer.
        if model is not None and self._dry_damping is not None:
            model.dof_damping[:] = self._dry_damping

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
        area = np.maximum(self.orifice_area, 1e-6)
        thrust_mag = coeff * rho * q * np.abs(q) / area
        thrust_mag *= subf  # no jet from a cavity that is not in the fluid

        # Thrust opposes the direction the fluid leaves.
        f = -thrust_mag[:, None] * axis_world
        np.add.at(data.xfrc_applied[:, :3], self.body_id, f)

        # --- and the reaction the muscle feels ---------------------------
        #
        # The cavity has to be held at the stagnation pressure that drives the
        # jet.  That pressure acts on a moving wall, and the generalised force
        # it exerts on the driving joint is the pressure times the rate at
        # which the joint changes the enclosed volume:
        #
        #     p   = 0.5 rho v_e^2 = 0.5 rho (Q/A)^2
        #     tau = p * |dV/dtheta|,   dV/dtheta = -V0 * stroke_fraction / span
        #
        # so that tau * omega = p * Q exactly, which is the power the jet
        # carries away.
        #
        # Since Q = (dV/dtheta) omega, that torque is a pure function of the
        # joint rate and can be written as a damping coefficient,
        #
        #     tau = -c omega,   c = 0.5 rho (dV/dtheta)^3 |omega| / A^2
        #
        # which is how it is applied: into `dof_damping`, where MuJoCo's
        # implicit integrator inverts it with the rest of the dynamics.  See
        # the module docstring for why the explicit form diverges.
        #
        # `coeff` and `subf` multiply it for the same reasons they multiply the
        # thrust: a stroke whose momentum flux is only partly real does not
        # cost full price, and a cavity out of the water is not pumping
        # anything.  One coefficient, one physical story.
        dv_dtheta = self.volume * self.stroke_fraction / span
        c_pump = (coeff * 0.5 * rho * dv_dtheta**3 * np.abs(omega) * subf
                  / area**2)

        if self._dofadr is None:
            self._dofadr = np.array(
                [model.jnt_dofadr[j] if j >= 0 else -1 for j in self.joint_id],
                dtype=int)
        if self._dry_damping is None:
            self._dry_damping = model.dof_damping.copy()
        driven = self._dofadr >= 0
        if np.any(driven):
            adr = self._dofadr[driven]
            # Rebuilt from the dry value every step, never accumulated.
            model.dof_damping[adr] = self._dry_damping[adr]
            np.add.at(model.dof_damping, adr, c_pump[driven])
            self.last_pump_power = float(
                np.sum(c_pump[driven] * omega[driven] ** 2))
        else:
            self.last_pump_power = 0.0

        self._prev_v = v_now
        return float(np.abs(thrust_mag).sum())

    def actuator_work(self, model, data) -> float:
        """Mechanical power the bell muscles are spending, W.

        Pressure inside the cavity is the stagnation pressure needed to drive
        the jet, ``0.5 * rho * v_e^2``, and the work rate is ``p * Q``.  A small
        orifice buys thrust and is charged for it here, which is what makes the
        orifice ratio a real trade rather than free thrust.

        This returns the value ``apply`` recorded, and it *is* now accounted
        through the driving actuator's torque -- ``apply`` puts the reaction on
        the joint, the position servo has to overcome it, and
        ``data.actuator_force`` carries it into the energy budget.  The number
        here is the same power, reported separately so telemetry can tell the
        cost of pumping from the cost of swinging the bell.

        It used to return a bare ``0.0`` under a comment claiming exactly that
        accounting, with no torque anywhere in the module to support it.
        Measured in ``experiments/jet_energy``: the actuator was charged
        0.008% of what the jet was doing.
        """
        return self.last_pump_power
