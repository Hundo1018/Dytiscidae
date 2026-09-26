"""Propellers: blade-element momentum theory, the comparison to flapping.

Added 2026-09-23 (ROADMAP "Why nothing flies, measured a third time") so that
the question "can anything fly in this simulator?" has a control: a machine
that makes thrust the way most small aircraft do, through the same airfoil
coefficients (`fluid.lift_coefficient`, `fluid.drag_coefficient`), the same
media and the same energy accounting as a flapping wing.

The model
---------
A fixed-pitch rotor of ``B`` blades, radius ``R``, hub ``r0``, chord ``c(r)``
and geometric pitch ``P`` (blade angle ``theta = atan(P / 2 pi r)``), spinning
at ``Omega`` with the flow meeting it at ``V_ax`` along the axis and ``V_ip``
in the disc plane.  For each annulus the induced velocity ``v`` balances the
two statements of its thrust:

    blade element:  dT = 1/2 rho W^2 B c (Cl cos phi - Cd sin phi) dr
    momentum:       dT = 4 pi r rho sqrt(V_ip^2 + (V_ax + v)^2) v F dr

with ``phi = atan((V_ax + v) / (Omega r))``, ``W^2 = (V_ax + v)^2 +
(Omega r)^2`` and Prandtl's tip loss ``F = (2/pi) acos(exp(-B (R - r) /
(2 r sin phi)))``.  ``sqrt(V_ip^2 + ...)`` is Glauert's forward-flight form, so
an edgewise rotor sees the mass flow of its forward speed.  Swirl is neglected
(small for propellers).  Torque is ``dQ = 1/2 rho W^2 B c (Cl sin phi + Cd cos
phi) r dr``.  Standard; Glauert 1935 (Durand, Aerodynamic Theory IV-L), as
restated in any rotor text.  ``experiments/rotor`` checks it against the UIUC
propeller database (APC 10x4.7 SF).

In MuJoCo a rotor is a body on a spin hinge driven by a velocity servo, so the
motor torque, its power and its reaction on the airframe go through the same
actuator and battery accounting as every joint.  `RotorSet.apply` adds the
aerodynamic thrust along the spin axis and the aerodynamic torque against the
spin, on the rotor body.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fluid import drag_coefficient, lift_coefficient

#: Annuli per blade.  Converged: 16 against 64 moves static CT by <0.5%
#: (`experiments/rotor`).
N_ANNULI = 16


@dataclass
class RotorSpec:
    """Geometry of one fixed-pitch rotor."""

    radius: float
    pitch: float                 # geometric pitch, m per revolution
    blades: int = 2
    #: Constant-chord stand-in for a tapered blade; with `camber` below it
    #: reproduces the APC 10x4.7 SF to CP rms 0.003 and CT rms 0.012 over
    #: J = 0.1-0.6, static CT 19% low (`experiments/rotor`).
    chord_ratio: float = 0.12    # mean chord / R
    hub_ratio: float = 0.15      # hub radius / R
    #: Section camber as a fraction of chord.  Propeller sections are cambered,
    #: and the zero-lift angle it sets (thin-airfoil, ``-2 f/c``) is what makes
    #: the aerodynamic pitch larger than the geometric one.
    camber: float = 0.065        # fitted to the APC 10x4.7 SF static CT, the one fit
    #: +1: positive spin about the shaft axis pushes along it.  -1 is the mirror
    #: image, which a reflected part carries (a mirrored CCW propeller is a CW
    #: one).
    handed: int = 1

    def stations(self):
        r0 = self.hub_ratio * self.radius
        dr = (self.radius - r0) / N_ANNULI
        r = r0 + dr * (np.arange(N_ANNULI) + 0.5)
        c = np.full(N_ANNULI, self.chord_ratio * self.radius)
        theta = np.arctan2(self.pitch, 2.0 * np.pi * r)
        return r, dr, c, theta


def bemt(spec: RotorSpec, omega: float, v_ax: float, v_ip: float,
         rho: float, mu: float) -> tuple[float, float]:
    """Thrust (N, along the spin axis) and aerodynamic torque (N m, resisting
    the spin) of one rotor.  ``omega`` in rad/s, its sign ignored."""
    om = abs(float(omega))
    if om < 1e-6 or spec.radius <= 0.0:
        return 0.0, 0.0
    r, dr, c, theta = spec.stations()
    B = spec.blades
    R = spec.radius

    def blade(v):
        ua = v_ax + v
        ut = om * r
        phi = np.arctan2(ua, ut)
        W2 = ua * ua + ut * ut
        alpha = theta - phi + 2.0 * spec.camber
        re = rho * np.sqrt(W2) * c / max(mu, 1e-12)
        cl = lift_coefficient(alpha, re, np.full_like(r, 1e6), np.zeros_like(r))
        cd = drag_coefficient(alpha, re, np.full_like(r, 1e6), cl, 0.0)
        q = 0.5 * rho * W2 * B * c
        dT = q * (cl * np.cos(phi) - cd * np.sin(phi))
        dQ = q * (cl * np.sin(phi) + cd * np.cos(phi)) * r
        sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)
        f = B * (R - r) / (2.0 * r * sphi)
        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        return dT, dQ, np.maximum(F, 1e-3)

    def residual(v):
        dT, _, F = blade(v)
        mom = 4.0 * np.pi * r * rho * np.sqrt(v_ip * v_ip + (v_ax + v) ** 2) * v * F
        return dT - mom

    # Bisection per annulus on v >= 0.  Where the blade makes no positive thrust
    # at v = 0 (windmilling, or past zero-thrust advance ratio) v is 0.
    lo = np.zeros_like(r)
    hi = np.full_like(r, max(om * R, abs(v_ax), 1.0))
    g0 = residual(lo)
    active = g0 > 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        g = residual(mid)
        pos = g > 0.0
        lo = np.where(active & pos, mid, lo)
        hi = np.where(active & ~pos, mid, hi)
    v = np.where(active, 0.5 * (lo + hi), 0.0)
    dT, dQ, _ = blade(v)
    return float(np.sum(dT * dr)), float(np.sum(dQ * dr))


class RotorSet:
    """Every rotor of one machine, applied each step like `JetSet`."""

    def __init__(self, model, specs: dict) -> None:
        """``specs`` maps a rotor body name to its `RotorSpec`."""
        import mujoco

        self.body = []
        self.dof = []
        self.spec = []
        self.axis_local = []
        for name, spec in specs.items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                continue
            j = int(model.body_jntadr[b])
            self.body.append(b)
            self.dof.append(int(model.jnt_dofadr[j]))
            self.axis_local.append(np.asarray(model.jnt_axis[j], float))
            self.spec.append(spec)
        self.n = len(self.body)
        self.last_thrust = np.zeros(self.n)
        self.last_torque = np.zeros(self.n)

    def apply(self, model, data, medium, t: float) -> float:
        """Add rotor thrust and aerodynamic torque to ``data.xfrc_applied``.
        Returns total thrust, N."""
        if self.n == 0:
            return 0.0
        import mujoco

        v6 = np.zeros(6)
        total = 0.0
        for k in range(self.n):
            b = self.body[k]
            R = data.xmat[b].reshape(3, 3)
            ax = R @ self.axis_local[k]
            omega = float(data.qvel[self.dof[k]])
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            pos = data.xipos[b]
            rho, mu, _ = medium.properties(pos[None, :], np.array([0.01]), t)
            flow = np.asarray(medium.flow_velocity(pos[None, :], t), float).reshape(3)
            rel = v6[3:] - flow                     # hub velocity through the fluid
            # The rotor pushes fluid against the direction its thrust acts in:
            # with positive spin the blades' pitch makes thrust along +axis,
            # and the flow meets the disc from the +axis side as the hub moves.
            sgn = 1.0 if omega >= 0.0 else -1.0
            thrust_dir = self.spec[k].handed * sgn * ax
            v_ax = float(rel @ thrust_dir)
            v_ip = float(np.linalg.norm(rel - v_ax * thrust_dir))
            T, Q = bemt(self.spec[k], omega, v_ax, v_ip, float(rho[0]), float(mu[0]))
            data.xfrc_applied[b, :3] += T * thrust_dir
            data.xfrc_applied[b, 3:] += -Q * sgn * ax
            self.last_thrust[k] = T
            self.last_torque[k] = Q
            total += T
        return total
