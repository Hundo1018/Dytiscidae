"""Quasi-steady blade-element fluid loads for flapping surfaces and bluff bodies.

Why this module exists
----------------------
MuJoCo ships an ellipsoid fluid model, but it does not include buoyancy (a body
of density 500 kg/m^3 still sinks in a fluid of density 1000 kg/m^3 -- verified
directly), and its drag model is not adequate for a flapping wing, where most
of the useful force comes from three effects it does not represent:

  1. a leading-edge vortex that holds lift attached to ~45 degrees of incidence,
  2. rotational (Kramer) circulation during stroke reversal,
  3. added mass -- negligible in air, dominant in water.

So the rigid-body dynamics and contacts come from MuJoCo, and the entire fluid
interaction is computed here and injected through ``xfrc_applied``.

The model
---------
Each lifting surface is discretised into spanwise strips.  For strip *i*:

    v_rel  = u_fluid - v_element                (relative flow, world frame)
    v_2d   = v_rel - (v_rel . s_hat) s_hat      (strip theory: drop spanwise flow)
    alpha  = atan2(v_2d . n_hat, v_2d . c_hat)  (angle of attack)
    q      = 0.5 * rho * |v_2d|^2

    L      = q * dS * CL(alpha, Re, AR, k)  along  (d_hat x s_hat)
    D      = q * dS * CD(alpha, Re, AR, CL)  along  d_hat
    F_rot  = C_rot * rho * |v_2d| * omega_s * c^2 * dr   along the lift axis
    F_am   = -d(m_added * v_normal)/dt                   along n_hat
    F_buoy = rho_water * g * V * submerged_fraction      along +Z

Frame convention (verified against a worked example in the tests):
    s_hat  spanwise, root -> tip
    c_hat  chordwise, leading edge -> trailing edge
    n_hat  = c_hat x s_hat   (the "upper" surface normal)
    d_hat  = v_2d / |v_2d|   (downstream direction)
    lift   acts along d_hat x s_hat, positive for positive alpha

Everything is vectorised over strips with numpy; a 200-strip machine costs
roughly 60 microseconds per step, which keeps a 20 s episode at 500 Hz well
under a second of wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .medium import GRAVITY, MediumField

# Kinds of element.
WING = 0  # a lifting strip: generates lift, induced drag, rotational lift
BLUFF = 1  # a volume: pressure drag, buoyancy, added mass, no circulation


@dataclass
class PanelSet:
    """The fluid-facing discretisation of a machine.

    All geometry is stored in the *local frame of the owning body*, so it is
    built once at compile time and transformed each step.

    Attributes
    ----------
    body_id : (N,) int
        MuJoCo body index each element is rigidly attached to.
    pos_local : (N, 3)
        Element centroid in the owning body's frame.
    span_local, chord_local : (N, 3)
        Orthonormal span and chord axes in the owning body's frame.  The normal
        is derived as ``chord x span``.
    chord : (N,)
        Chord length, m.  For bluff elements, the streamwise extent.
    dr : (N,)
        Spanwise width of the strip, m.
    volume : (N,)
        Outer envelope volume, m^3.  Drives added mass of bluff bodies -- a
        flooded fairing is exactly as hard to accelerate as a sealed one.
    volume_buoyant : (N,)
        Volume that generates net buoyancy, m^3.  Defaults to ``volume`` when
        not given, which is right for a wing strip and wrong for a flooded
        fairing, so the phenotype always sets it explicitly.
    half_height : (N,)
        Half of the vertical extent, m.  Sets the width of the free-surface
        blend for this element.
    kind : (N,) int
        WING or BLUFF.
    aspect_ratio : (N,)
        Aspect ratio of the *surface this strip belongs to* (not of the strip).
        Used for the induced-drag and lift-slope corrections.
    cd_bluff : (N,)
        Pressure drag coefficient for BLUFF elements, referenced to ``area``.
    """

    body_id: np.ndarray
    pos_local: np.ndarray
    span_local: np.ndarray
    chord_local: np.ndarray
    chord: np.ndarray
    dr: np.ndarray
    volume: np.ndarray
    half_height: np.ndarray
    kind: np.ndarray
    aspect_ratio: np.ndarray
    cd_bluff: np.ndarray
    #: Fraction of chord at which the strip pitches. 0.25 is the quarter-chord.
    pitch_axis: np.ndarray = field(default=None)  # type: ignore[assignment]
    volume_buoyant: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.body_id)
        if self.pitch_axis is None:
            self.pitch_axis = np.full(n, 0.25)
        if self.volume_buoyant is None:
            self.volume_buoyant = np.array(self.volume, dtype=float)
        self.area = self.chord * self.dr
        # Normal axis, n = c x s.
        self.normal_local = np.cross(self.chord_local, self.span_local)
        norms = np.linalg.norm(self.normal_local, axis=1, keepdims=True)
        self.normal_local = self.normal_local / np.maximum(norms, 1e-12)
        self.n = n

    @property
    def total_area(self) -> float:
        return float(self.area[self.kind == WING].sum())

    @property
    def total_volume(self) -> float:
        return float(self.volume.sum())

    @staticmethod
    def empty() -> "PanelSet":
        z3 = np.zeros((0, 3))
        z1 = np.zeros(0)
        return PanelSet(
            body_id=np.zeros(0, dtype=int),
            pos_local=z3,
            span_local=z3,
            chord_local=z3,
            chord=z1,
            dr=z1,
            volume=z1,
            half_height=z1,
            kind=np.zeros(0, dtype=int),
            aspect_ratio=z1,
            cd_bluff=z1,
            pitch_axis=z1,
            volume_buoyant=z1,
        )

    @staticmethod
    def concat(sets: list["PanelSet"]) -> "PanelSet":
        sets = [s for s in sets if s.n > 0]
        if not sets:
            return PanelSet.empty()
        cat = lambda name: np.concatenate([getattr(s, name) for s in sets])  # noqa: E731
        return PanelSet(
            body_id=cat("body_id"),
            pos_local=cat("pos_local"),
            span_local=cat("span_local"),
            chord_local=cat("chord_local"),
            chord=cat("chord"),
            dr=cat("dr"),
            volume=cat("volume"),
            half_height=cat("half_height"),
            kind=cat("kind"),
            aspect_ratio=cat("aspect_ratio"),
            cd_bluff=cat("cd_bluff"),
            pitch_axis=cat("pitch_axis"),
            volume_buoyant=cat("volume_buoyant"),
        )


# --------------------------------------------------------------------------
# Coefficient models
# --------------------------------------------------------------------------


def skin_friction_cd(re: np.ndarray) -> np.ndarray:
    """Two-sided skin friction coefficient of a flat plate.

    Blends the laminar Blasius result into the turbulent 1/7-power result
    around Re = 5e5.  Clamped at low Re so that a strip that is momentarily at
    rest (every flapping stroke has two of these per cycle) does not produce an
    infinite coefficient.
    """
    re = np.maximum(re, 1.0)
    lam = 1.328 / np.sqrt(re)
    turb = 0.074 / re**0.2
    w = 1.0 / (1.0 + np.exp(-(np.log10(re) - 5.7) * 4.0))
    return 2.0 * ((1.0 - w) * lam + w * turb)


def lift_coefficient(
    alpha: np.ndarray, re: np.ndarray, ar: np.ndarray, reduced_freq: np.ndarray
) -> np.ndarray:
    """Lift coefficient spanning attached, LEV-augmented and post-stall regimes.

    Below stall the strip behaves like a finite wing with the Helmholtz
    lift-slope correction ``2*pi / (1 + 2/AR)``.  Above stall it behaves like a
    flat plate, ``CL_max * sin(2*alpha)``.

    ``CL_max`` and the stall angle are both raised by the reduced frequency
    ``k = omega * c / (2 * U)``.  This is the leading-edge-vortex term: a wing
    that is flapping fast relative to its own translation carries a stable LEV
    and keeps generating lift far past the static stall angle, which is the
    whole reason flapping flight is competitive at this scale.  A gliding wing
    (k -> 0) gets the conventional static behaviour.
    """
    lev = np.clip(reduced_freq / 0.30, 0.0, 1.0)

    cl_max = 1.10 + 0.80 * lev  # 1.1 static flat plate -> 1.9 with a strong LEV
    alpha_stall = np.radians(11.0 + 26.0 * lev)  # 11 deg static -> 37 deg flapping
    # Very low Reynolds number wings stall early and softly.
    alpha_stall *= np.clip(0.55 + 0.45 * np.log10(np.maximum(re, 10.0)) / 5.0, 0.5, 1.0)

    cl_alpha = 2.0 * np.pi / (1.0 + 2.0 / np.maximum(ar, 0.5))
    cl_linear = cl_alpha * alpha

    cl_plate = cl_max * np.sin(2.0 * alpha)

    # Smooth handover so the optimiser never sees a kink to exploit.
    blend_width = np.radians(6.0)
    w = 1.0 / (1.0 + np.exp(-(np.abs(alpha) - alpha_stall) / blend_width))
    cl = (1.0 - w) * cl_linear + w * cl_plate
    # Never exceed the plate envelope; the linear branch is unbounded.
    return np.clip(cl, -1.2 * cl_max, 1.2 * cl_max)


def drag_coefficient(
    alpha: np.ndarray, re: np.ndarray, ar: np.ndarray, cl: np.ndarray
) -> np.ndarray:
    """Profile + induced + separated pressure drag."""
    cd_f = skin_friction_cd(re)
    # Oswald efficiency: low for the stubby, highly twisted surfaces this
    # pipeline tends to generate.
    oswald = 0.75
    cd_i = cl**2 / (np.pi * oswald * np.maximum(ar, 0.5))
    # Flat plate normal to the flow is ~1.98; this term dominates at high alpha
    # and is what makes a wing a paddle when it is in water.
    cd_p = 1.98 * (1.0 - np.cos(2.0 * alpha)) * 0.5
    return cd_f + cd_i + cd_p


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@dataclass
class FluidDiagnostics:
    """Per-step aggregates, recorded for observability and for scoring."""

    lift: float = 0.0
    drag: float = 0.0
    buoyancy: float = 0.0
    added_mass: float = 0.0
    max_submerged: float = 0.0
    mean_submerged: float = 0.0
    max_alpha: float = 0.0
    max_dynamic_pressure: float = 0.0
    #: Peak slamming force seen this step, N.  Water entry loads are a real
    #: structural sizing case and a real reason amphibious craft break.
    slam: float = 0.0


class FluidSolver:
    """Applies blade-element fluid loads to a MuJoCo model each step.

    Usage::

        solver = FluidSolver(model, panels, medium)
        while stepping:
            solver.apply(data, t)
            mujoco.mj_step(model, data)
    """

    def __init__(
        self,
        model,
        panels: PanelSet,
        medium: MediumField,
        *,
        c_rot: float | None = None,
        added_mass_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.panels = panels
        self.medium = medium
        self.added_mass_scale = added_mass_scale
        # Kramer rotational circulation coefficient, pi * (0.75 - x0).
        self.c_rot = (
            np.pi * (0.75 - panels.pitch_axis) if c_rot is None else np.full(panels.n, c_rot)
        )
        # Previous normal velocity and added mass, for the d/dt term.
        self._prev_vn = np.zeros(panels.n)
        self._prev_ma = np.zeros(panels.n)
        self._prev_t = None
        # The added-mass term is a backward difference, so it has no valid value
        # on the first call.  Without this flag the very first step reports
        # d(m*v)/dt = (m*v - 0)/dt, which for a wing already moving at 10 m/s is
        # an impulse an order of magnitude larger than the real lift -- injected
        # once at the start of every episode, exactly where it does most damage.
        self._primed = False
        self.diag = FluidDiagnostics()
        # Scratch buffers reused every step.
        self._nbody = model.nbody
        self._vel6 = np.zeros(6)

    def reset(self) -> None:
        self._prev_vn[:] = 0.0
        self._prev_ma[:] = 0.0
        self._prev_t = None
        self._primed = False
        self.diag = FluidDiagnostics()

    # ------------------------------------------------------------------ step

    def apply(self, data, t: float) -> FluidDiagnostics:
        """Compute and accumulate fluid loads into ``data.xfrc_applied``."""
        import mujoco

        p = self.panels
        if p.n == 0:
            return self.diag

        dt = self.model.opt.timestep if self._prev_t is None else max(t - self._prev_t, 1e-6)
        self._prev_t = t

        # --- kinematics ---------------------------------------------------
        xpos = data.xpos  # (nbody, 3) body frame origin
        xmat = data.xmat.reshape(-1, 3, 3)  # (nbody, 3, 3) body -> world
        xipos = data.xipos  # (nbody, 3) body CoM

        R = xmat[p.body_id]  # (N, 3, 3)
        pos = xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
        s_hat = np.einsum("nij,nj->ni", R, p.span_local)
        c_hat = np.einsum("nij,nj->ni", R, p.chord_local)
        n_hat = np.einsum("nij,nj->ni", R, p.normal_local)

        # Body 6D velocities (world frame, at the body frame origin).
        vel = np.zeros((self._nbody, 6))
        for b in np.unique(p.body_id):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), self._vel6, 0)
            vel[b] = self._vel6
        omega = vel[p.body_id, :3]
        v_org = vel[p.body_id, 3:]
        v_elem = v_org + np.cross(omega, pos - xpos[p.body_id])

        # --- medium -------------------------------------------------------
        rho, mu, subf = self.medium.properties(pos, p.half_height, t)
        u_flow = self.medium.flow_velocity(pos, t)
        v_rel = u_flow - v_elem

        # --- strip theory -------------------------------------------------
        v_span = np.einsum("ni,ni->n", v_rel, s_hat)[:, None] * s_hat
        v_2d = v_rel - v_span
        U = np.linalg.norm(v_2d, axis=1)
        U_safe = np.maximum(U, 1e-6)
        d_hat = v_2d / U_safe[:, None]

        q = 0.5 * rho * U**2
        re = rho * U * p.chord / np.maximum(mu, 1e-12)

        cos_a = np.einsum("ni,ni->n", v_2d, c_hat) / U_safe
        sin_a = np.einsum("ni,ni->n", v_2d, n_hat) / U_safe
        alpha = np.arctan2(sin_a, cos_a)
        # Fold into [-pi/2, pi/2]: a strip at 170 deg incidence is a strip at
        # -10 deg incidence with its other face to the flow.
        alpha = np.arctan2(np.sin(alpha), np.abs(np.cos(alpha)) + 1e-12)

        # Angular rate about the span axis -> reduced frequency and Kramer lift.
        omega_s = np.einsum("ni,ni->n", omega, s_hat)
        reduced_freq = np.abs(omega_s) * p.chord / (2.0 * U_safe)

        is_wing = p.kind == WING
        lift_axis = np.cross(d_hat, s_hat)
        lift_axis /= np.maximum(np.linalg.norm(lift_axis, axis=1, keepdims=True), 1e-12)

        F = np.zeros((p.n, 3))

        # Circulatory lift and drag.
        cl = np.where(is_wing, lift_coefficient(alpha, re, p.aspect_ratio, reduced_freq), 0.0)
        cd = np.where(
            is_wing,
            drag_coefficient(alpha, re, p.aspect_ratio, cl),
            p.cd_bluff + skin_friction_cd(re),
        )
        L = q * p.area * cl
        D = q * p.area * cd
        F += L[:, None] * lift_axis + D[:, None] * d_hat

        # Rotational (Kramer) circulation: the force generated by a strip that
        # is pitching while translating.  This is what lets an insect wing
        # generate useful force through stroke reversal, when U is small.
        f_rot = np.where(
            is_wing,
            self.c_rot * rho * U * omega_s * p.chord**2 * p.dr,
            0.0,
        )
        F += f_rot[:, None] * lift_axis

        # Added mass, including the d(m)/dt slamming term.  For a flat strip the
        # 2D added mass for normal acceleration is rho * pi * c^2 / 4 per unit
        # span; a bluff element uses its displaced volume with Ca = 0.5.
        vn = np.einsum("ni,ni->n", v_rel, n_hat)
        m_add = np.where(
            is_wing,
            rho * np.pi * p.chord**2 * 0.25 * p.dr,
            0.5 * rho * p.volume,
        ) * self.added_mass_scale
        # d(m_a * v_n)/dt, backward difference.  The m_dot * v_n part is the
        # water-entry slam: it fires exactly when a fast-moving surface changes
        # its submerged fraction, which is the load case that breaks wings.
        if self._primed:
            dmv = (m_add * vn - self._prev_ma * self._prev_vn) / dt
            slam = float(np.abs((m_add - self._prev_ma) / dt * vn).max())
        else:
            # First call after a reset: no previous sample, so no rate term.
            dmv = np.zeros(p.n)
            slam = 0.0
            self._primed = True
        F += dmv[:, None] * n_hat
        self._prev_vn = vn.copy()
        self._prev_ma = m_add.copy()

        # Buoyancy: only the genuinely submerged portion, at true water density,
        # and only over the volume that is actually sealed rather than flooded.
        f_buoy = self.medium.water.rho * GRAVITY * p.volume_buoyant * subf
        F[:, 2] += f_buoy

        # --- accumulate to bodies -----------------------------------------
        # xfrc_applied takes a world-frame force at the body CoM plus a torque.
        arm = pos - xipos[p.body_id]
        T = np.cross(arm, F)
        np.add.at(data.xfrc_applied[:, :3], p.body_id, F)
        np.add.at(data.xfrc_applied[:, 3:], p.body_id, T)

        # --- diagnostics ---------------------------------------------------
        d = self.diag
        d.lift = float(np.abs(L).sum())
        d.drag = float(np.abs(D).sum())
        d.buoyancy = float(f_buoy.sum())
        d.added_mass = float(np.abs(dmv).sum())
        d.max_submerged = float(subf.max())
        d.mean_submerged = float(subf.mean())
        d.max_alpha = float(np.abs(alpha).max())
        d.max_dynamic_pressure = float(q.max())
        d.slam = slam
        return d

    # -------------------------------------------------------------- analysis

    def instantaneous_power(self, data) -> float:
        """Mechanical power the machine is currently putting into the fluid, W.

        Computed as the negative of the rate of work the fluid does on the
        machine.  Positive means the machine is spending energy on the fluid,
        which is what any propulsion must do.
        """
        import mujoco

        p = self.panels
        if p.n == 0:
            return 0.0
        power = 0.0
        vel6 = np.zeros(6)
        for b in np.unique(p.body_id):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), vel6, 0)
            f = data.xfrc_applied[b]
            v_com = vel6[3:] + np.cross(vel6[:3], data.xipos[b] - data.xpos[b])
            power -= float(f[:3] @ v_com + f[3:] @ vel6[:3])
        return power
