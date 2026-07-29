"""Vortex-particle wake, for seeing the flow the solver is actually computing.

What this is, and what it is not
--------------------------------
This is **not CFD**.  There is no Navier-Stokes solve, no mesh, no pressure
Poisson equation.  Running one would cost hours per evaluation and the search
loop would be finished before the first candidate was scored.

What it is: a Lagrangian vortex method driven by the *same* bound circulation
the blade-element solver already computes.  By Kutta-Joukowski a strip carrying
lift ``L' = rho * U * Gamma`` has bound circulation ``Gamma = 0.5 * C_L * U * c``,
and by Kelvin's theorem any change in that bound circulation must be shed into
the wake as an equal and opposite vortex.  So each strip sheds
``-dGamma/dt * dt`` from its trailing edge every interval, the shed particles
advect in the local flow plus each other's induced velocity (Biot-Savart, with a
regularised core so nothing blows up at close range), and they diffuse away.

That makes the picture *derived from the forces being used*, not drawn on top of
them.  If the solver thinks a wing is generating lift, a vortex leaves its
trailing edge; if the solver thinks the wing stalled, the wake shows it.  A
flapping wing produces the reverse von Karman street that is the signature of
thrust production, and it appears here because the circulation history produces
it, not because anyone drew it.

The wake is diagnostic only: it is never fed back into the forces.  Doing that
properly (a free-wake panel method) is a real option for a future high-fidelity
tier, and would change the forces by 10-30% for a flapping wing in ground or
wake interference -- but it belongs in Tier 2+, not in the inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fluid import WING


@dataclass
class VortexWake:
    """A cloud of shed vortex particles.

    Parameters
    ----------
    max_particles:
        Hard cap.  Oldest are discarded first.  Biot-Savart is O(N^2), so this
        is the knob that trades wake length against render time.
    core_radius:
        Regularisation length for the induced velocity kernel, m.  Roughly a
        chord thickness; below it the singular 1/r^2 is smoothly capped.
    decay_time:
        e-folding time for particle strength, s.  Stands in for viscous
        diffusion and vortex breakdown, neither of which a point particle has.
    """

    max_particles: int = 420
    core_radius: float = 0.06
    decay_time: float = 2.2
    shed_interval: float = 0.02  # s between shedding events

    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vort: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    age: np.ndarray = field(default_factory=lambda: np.zeros(0))
    born_in_water: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    _prev_gamma: np.ndarray | None = None
    _last_shed: float = -1e9

    def reset(self) -> None:
        self.pos = np.zeros((0, 3))
        self.vort = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.born_in_water = np.zeros(0, dtype=bool)
        self._prev_gamma = None
        self._last_shed = -1e9

    # ------------------------------------------------------------------ shed

    def update(self, solver, data, t: float, dt: float) -> None:
        """Shed from the current circulation, then advect and decay everything."""
        state = getattr(solver, "last_state", None)
        if state is None:
            return

        gamma = state["gamma"]
        if self._prev_gamma is None or len(self._prev_gamma) != len(gamma):
            self._prev_gamma = gamma.copy()

        if t - self._last_shed >= self.shed_interval:
            self._shed(state, gamma, t)
            self._last_shed = t
            self._prev_gamma = gamma.copy()

        self._advect(state, dt)

    def _shed(self, state, gamma, t: float) -> None:
        wing = state["kind"] == WING
        if not wing.any():
            return
        dgamma = gamma - self._prev_gamma
        # Kelvin: the wake receives the negative of the bound-circulation change.
        strength = -dgamma * state["dr"]
        keep = wing & (np.abs(strength) > 1e-4)
        if not keep.any():
            return

        # Shed from the trailing edge: the strip centroid is at the quarter
        # chord, so the trailing edge is three quarters of a chord downstream.
        te = state["pos"][keep] + 0.75 * state["chord"][keep, None] * state["chord_axis"][keep]
        new_vort = strength[keep, None] * state["span_axis"][keep]

        self.pos = np.concatenate([self.pos, te])
        self.vort = np.concatenate([self.vort, new_vort])
        self.age = np.concatenate([self.age, np.zeros(len(te))])
        self.born_in_water = np.concatenate(
            [self.born_in_water, state["submerged"][keep] > 0.5]
        )

        if len(self.pos) > self.max_particles:
            cut = len(self.pos) - self.max_particles
            self.pos = self.pos[cut:]
            self.vort = self.vort[cut:]
            self.age = self.age[cut:]
            self.born_in_water = self.born_in_water[cut:]

    # --------------------------------------------------------------- advect

    def _advect(self, state, dt: float) -> None:
        n = len(self.pos)
        if n == 0:
            return

        # Ambient flow at each particle.
        vel = state["flow_at"](self.pos)

        # Mutual induction.  Regularised Biot-Savart:
        #   u = (1/4pi) * sum_j  (omega_j x r) / (|r|^2 + a^2)^(3/2)
        # The a^2 in the denominator is what keeps two coincident particles from
        # accelerating each other to infinity, which is otherwise the first
        # thing that happens.
        if n <= 260:
            d = self.pos[None, :, :] - self.pos[:, None, :]
            r2 = np.sum(d * d, axis=2) + self.core_radius**2
            inv = 1.0 / (4.0 * np.pi * r2**1.5)
            np.fill_diagonal(inv, 0.0)
            cross = np.cross(self.vort[None, :, :], d)
            vel = vel + np.einsum("ij,ijk->ik", inv, cross)

        self.pos = self.pos + vel * dt
        self.age = self.age + dt
        self.vort = self.vort * np.exp(-dt / max(self.decay_time, 1e-6))

        alive = self.age < self.decay_time * 3.0
        if not alive.all():
            self.pos = self.pos[alive]
            self.vort = self.vort[alive]
            self.age = self.age[alive]
            self.born_in_water = self.born_in_water[alive]

    # ------------------------------------------------------------- rendering

    def render_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(positions, signed_strength, fade)`` for drawing.

        ``signed_strength`` is the vorticity component along the world Y axis,
        which for a machine flying along X is the sense that distinguishes the
        upper and lower rows of a vortex street -- the thing worth seeing.
        """
        if len(self.pos) == 0:
            return self.pos, np.zeros(0), np.zeros(0)
        signed = self.vort[:, 1]
        fade = np.clip(1.0 - self.age / (self.decay_time * 3.0), 0.0, 1.0)
        return self.pos, signed, fade


def attach_wake_probe(solver) -> None:
    """Ask a ``FluidSolver`` to publish the state a wake needs.

    Recording is opt-in because the search loop calls ``apply`` millions of
    times and should not pay to store arrays nobody reads.
    """
    solver.record_state = True
