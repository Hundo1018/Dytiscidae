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

def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of two (N, 3) arrays.

    ``np.cross`` spends most of its time in ``moveaxis`` and axis normalisation
    rather than in arithmetic; profiling the fluid step showed it accounting for
    roughly a fifth of the entire simulation cost.  Writing the three components
    out directly removes that overhead.
    """
    return np.stack(
        (
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ),
        axis=1,
    )


#: Cross-flow drag coefficient of a blunt body, referenced to the side area it
#: presents to the cross component of the stream.  1.1 is the circular-cylinder
#: value across the Reynolds range this machine operates in.  It is deliberately
#: a different number from ``cd_bluff``, which is the axial coefficient: using
#: one value for both collapses the resultant back onto the flow direction and
#: with it the body's lift.
CD_CROSSFLOW = 1.1

# Kinds of element.
WING = 0  # a lifting strip: generates lift, induced drag, rotational lift
BLUFF = 1  # a volume: pressure drag, buoyancy, added mass, no circulation


@dataclass(eq=False)
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
        Pressure drag coefficient for BLUFF elements, referenced to the
        *projected* area, not to ``area``.
    ext_local : (N, 3)
        Extent of a BLUFF element along its own span, chord and normal axes, m.
        This is what makes a bluff element's drag depend on which way round it
        is: the projected area is recomputed each step from the flow direction
        and these three numbers.  Defaults to the strip's own dimensions, which
        is right for a wing and for a capsule stand-in alike.
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
    ext_local: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.body_id)
        if self.pitch_axis is None:
            self.pitch_axis = np.full(n, 0.25)
        if self.volume_buoyant is None:
            self.volume_buoyant = np.array(self.volume, dtype=float)
        if self.ext_local is None:
            self.ext_local = np.stack(
                [self.dr, self.chord, 2.0 * self.half_height], axis=1
            ) if n else np.zeros((0, 3))
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
            ext_local=z3,
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
            ext_local=np.concatenate([s.ext_local for s in sets], axis=0),
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
    #: True when the force limiter engaged, i.e. the state was already outside
    #: the range the quasi-steady model is valid over.
    clamped: bool = False


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
        # Dry inertial properties, kept so the added-mass augmentation below can
        # be recomputed from scratch each step rather than accumulating.
        self._dry_mass = model.body_mass.copy()
        self._dry_inertia = model.body_inertia.copy()
        # Mean squared lever arm of each body's panels about its CoM, used to
        # turn translational added mass into added rotational inertia.
        self._lever2 = np.zeros(model.nbody)
        for b in np.unique(panels.body_id):
            sel = panels.body_id == b
            r = panels.pos_local[sel] - model.body_ipos[b]
            self._lever2[b] = float(np.mean(np.sum(r**2, axis=1)))

        # Previous normal velocity and added mass, for the slam *diagnostic*.
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
        # Opt-in state recording for the wake visualiser.  Off by default: the
        # search loop calls apply() millions of times and should not pay to
        # record anything nobody reads.
        self.record_state = False
        self.last_state: dict | None = None
        # Scratch buffers reused every step.
        self._nbody = model.nbody
        self._vel6 = np.zeros(6)
        self._bodies = np.unique(panels.body_id)

    def reset(self) -> None:
        self._prev_vn[:] = 0.0
        self._prev_ma[:] = 0.0
        self._prev_t = None
        self._primed = False
        # Restore dry inertia: leaving a previous episode's entrained water in
        # the mass matrix would silently make the next episode heavier.
        self.model.body_mass[:] = self._dry_mass
        self.model.body_inertia[:] = self._dry_inertia
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
        #
        # Reading ``data.cvel`` directly and vectorising this looks tempting,
        # but its linear component is referenced to a com-based frame whose
        # origin is not the body frame origin; reconstructing element velocity
        # from it disagreed with mj_objectVelocity by ~0.5 m/s in testing, and
        # profiling showed this loop is not the bottleneck anyway (the step cost
        # is dominated by mj_step itself).  Correct and adequate beats clever.
        vel = np.zeros((self._nbody, 6))
        for b in self._bodies:
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), self._vel6, 0
            )
            vel[b] = self._vel6
        omega = vel[p.body_id, :3]
        v_org = vel[p.body_id, 3:]
        v_elem = v_org + _cross3(omega, pos - xpos[p.body_id])

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
        lift_axis = _cross3(d_hat, s_hat)
        lift_axis /= np.maximum(np.linalg.norm(lift_axis, axis=1, keepdims=True), 1e-12)

        F = np.zeros((p.n, 3))

        # Circulatory lift and drag, on the lifting strips only.
        cl = np.where(is_wing, lift_coefficient(alpha, re, p.aspect_ratio, reduced_freq), 0.0)
        cd = np.where(is_wing, drag_coefficient(alpha, re, p.aspect_ratio, cl), 0.0)
        L = q * p.area * cl
        D = q * p.area * cd
        F += L[:, None] * lift_axis + D[:, None] * d_hat

        # --- bluff-body drag ------------------------------------------------
        # Strip theory is wrong for a volume, and it was wrong here in a way
        # that mattered: the spanwise component of the flow was projected out
        # before the drag was formed, so a hull travelling nose-first along its
        # own axis felt *no* pressure drag at all.  A design could therefore
        # make its body arbitrarily long and pay nothing for it, and the search
        # duly produced long thin things.
        #
        # A volume is instead given the projected area of its own bounding box
        # against the true relative flow,
        #
        #     A(d) = |d.s| ey ez + |d.c| ex ez + |d.n| ex ey
        #
        # which is exact for a box, within a few percent for an ellipsoid, and
        # -- the point of it -- orientation dependent.  A slender chunk now
        # presents little area nose-on and a lot broadside, so elongation costs
        # what it should and the shape the CPPN generated reaches the dynamics
        # instead of stopping at the mass matrix.
        # The force is *not* aligned with the flow, and that omission was the
        # bigger half.  A body at incidence is loaded mainly by the component of
        # the stream across its own axis, and that load acts normal to the axis,
        # not downstream.  Resolving it that way -- Munk's slender-body result
        # with Allen and Perkins' cross-flow correction, the standard missile
        # aerodynamics treatment -- gives a body three things it did not have:
        #
        #   * lift.  A tapered hull at 15 degrees generates a force component
        #     perpendicular to the freestream.  Without it, only the surfaces
        #     could ever hold a machine up, and the search had no reason to
        #     shape a body for flight at all -- a lifting body was unreachable.
        #   * a pitching moment.  The normal force acts at each slice, so a body
        #     fat forward and fine aft is unstable and one fat aft is stable.
        #     This is what makes a tail a tail; before, a tail was drag.
        #   * a reason to point where it is going.  Cross-flow load exceeds
        #     axial load for anything slender, so flying sideways is expensive,
        #     which is the whole basis of weathercock stability.
        n_bluff = int((~is_wing).sum())
        if n_bluff:
            b = ~is_wing
            U_full = np.linalg.norm(v_rel, axis=1)
            U_full_safe = np.maximum(U_full, 1e-6)
            d_full = v_rel / U_full_safe[:, None]
            ex, ey, ez = p.ext_local[:, 0], p.ext_local[:, 1], p.ext_local[:, 2]

            # Split the stream into flow along the element's own long axis and
            # flow across it.  ``s_hat`` is that axis: for a body slice it is
            # the part's centreline, which is the axis the shape is built about.
            v_ax = np.einsum("ni,ni->n", v_rel, s_hat)
            v_axial = v_ax[:, None] * s_hat
            v_cross = v_rel - v_axial
            u_cross = np.linalg.norm(v_cross, axis=1)
            d_cross = v_cross / np.maximum(u_cross, 1e-9)[:, None]

            # Axial: base pressure over the frontal area plus friction over the
            # wetted area.  Slender bodies are cheap this way round, which is
            # the point of being slender.
            wetted = 2.0 * (ex * ey + ey * ez + ex * ez)
            re_b = rho * np.abs(v_ax) * ex / np.maximum(mu, 1e-12)
            f_axial = (
                0.5 * rho * np.abs(v_ax) * v_ax
                * (p.cd_bluff * ey * ez + skin_friction_cd(re_b) * wetted)
            )

            # Cross-flow: the side area presented to the cross component, with a
            # blunt-body coefficient.  1.1 is the standard cross-flow drag of a
            # circular cylinder at the Reynolds numbers this machine lives at,
            # and it is a genuinely different number from the streamwise cd --
            # using one coefficient for both is what collapses the force back
            # onto the flow direction and loses the lift.
            pc = np.abs(np.einsum("ni,ni->n", d_cross, c_hat))
            pn = np.abs(np.einsum("ni,ni->n", d_cross, n_hat))
            a_side = pc * ex * ez + pn * ex * ey
            f_cross = 0.5 * rho * u_cross**2 * CD_CROSSFLOW * a_side

            # Both act *along* the relative flow, as the wing branch's drag
            # does: ``v_rel`` is the fluid's velocity seen from the body, so a
            # resistive force pushes the body the way the fluid is going.
            # ``f_axial`` carries its own sign through ``|v_ax| v_ax``.
            F_b = f_axial[:, None] * s_hat + f_cross[:, None] * d_cross
            F += np.where(b[:, None], F_b, 0.0)
            D_b = np.where(b, np.abs(f_axial) + f_cross, 0.0)
            D = D + D_b

        # Rotational (Kramer) circulation: the force generated by a strip that
        # is pitching while translating.  This is what lets an insect wing
        # generate useful force through stroke reversal, when U is small.
        f_rot = np.where(
            is_wing,
            self.c_rot * rho * U * omega_s * p.chord**2 * p.dr,
            0.0,
        )
        F += f_rot[:, None] * lift_axis

        # --- added mass ----------------------------------------------------
        # For a flat strip the 2D added mass for normal acceleration is
        # rho * pi * c^2 / 4 per unit span; a bluff element uses its displaced
        # volume with Ca = 0.5.  In water a single wing strip of this machine
        # carries tens of kilograms of added mass -- several times the mass of
        # the whole vehicle.
        #
        # That ratio is exactly why this must NOT be applied as an external
        # force.  An explicit ``F = -d(m_a v)/dt`` term is a feedback loop whose
        # gain is m_added / m_body, so above unity it diverges within a few
        # steps: the classic added-mass instability of partitioned
        # fluid-structure coupling.  Applying it explicitly here produced NaN
        # accelerations after 1.3 s of simulated water time.
        #
        # Instead the added mass is folded into the *mass matrix*, which MuJoCo
        # inverts implicitly, so it is unconditionally stable no matter how far
        # the added mass exceeds the structural mass.  Two corrections come with
        # that: MuJoCo would otherwise apply gravity to the added mass (added
        # mass has inertia but no weight), and the translational term also has
        # to appear as rotational inertia about the body's CoM.
        # Bluff added mass is *anisotropic*, and it has to be: a flat body
        # accelerating broadside entrains far more fluid than the same body
        # accelerating edge-on, and treating them alike with a flat Ca = 0.5
        # told the search that a plate and a sphere of equal volume cost the
        # same to shake.  That erases the whole reason a fin is a fin.
        #
        # Directional coefficient from the element's own three extents,
        #
        #     Ca_i = 0.5 (e_j + e_k) / (2 e_i)
        #
        # which is exact for a sphere (0.5), within 18% of Lamb's result for a
        # disc moving normal to itself, and correctly small for a slender body
        # moving along its own axis.  The mass matrix takes a scalar, so what
        # goes in is the quadratic form of that diagonal tensor along the
        # instantaneous direction of motion -- the effective entrained mass for
        # the acceleration the body is actually undergoing.  It is rebuilt every
        # step, so as the body rotates its added mass changes with it.
        vn = np.einsum("ni,ni->n", v_rel, n_hat)
        e = np.maximum(p.ext_local, 1e-4)
        ca_axis = np.clip(
            0.5 * (e[:, [1, 2, 0]] + e[:, [2, 0, 1]]) / (2.0 * e), 0.05, 10.0
        )
        if n_bluff:
            # Direction cosines in the element's own frame, from the same
            # relative-velocity direction the drag used.
            dc = np.stack([
                np.einsum("ni,ni->n", d_full, s_hat),
                np.einsum("ni,ni->n", d_full, c_hat),
                np.einsum("ni,ni->n", d_full, n_hat),
            ], axis=1) ** 2
        else:
            dc = np.zeros((p.n, 3))
        ca_eff = np.einsum("ni,ni->n", dc, ca_axis)
        # A body momentarily at rest has no direction of motion to project onto;
        # fall back to the isotropic mean rather than to zero.
        ca_eff = np.where(dc.sum(axis=1) > 1e-6, ca_eff, ca_axis.mean(axis=1))

        m_add = np.where(
            is_wing,
            rho * np.pi * p.chord**2 * 0.25 * p.dr,
            ca_eff * rho * p.volume,
        ) * self.added_mass_scale

        m_body = np.zeros(self._nbody)
        np.add.at(m_body, p.body_id, m_add)
        self.model.body_mass[:] = self._dry_mass + m_body
        self.model.body_inertia[:] = self._dry_inertia + (
            m_body * self._lever2
        )[:, None]
        # Cancel the weight MuJoCo will apply to the entrained fluid.
        F[:, 2] += m_add * GRAVITY

        # The slamming rate term is still computed, but only as a *diagnostic*:
        # the structural check needs to know the peak entry load, while the
        # dynamics get the same physics through the varying mass matrix.
        if self._primed:
            slam = float(np.abs((m_add - self._prev_ma) / dt * vn).max())
        else:
            slam = 0.0
            self._primed = True
        self._prev_vn = vn.copy()
        self._prev_ma = m_add.copy()
        dmv = np.zeros(p.n)

        # Buoyancy: only the genuinely submerged portion, at true water density,
        # and only over the volume that is actually sealed rather than flooded.
        f_buoy = self.medium.water.rho * GRAVITY * p.volume_buoyant * subf
        F[:, 2] += f_buoy

        # A last-resort limiter.  The quasi-steady model is only valid for
        # states a real machine could be in; once a candidate is tumbling at
        # 50 m/s the coefficients are extrapolation and the forces can be
        # arbitrarily large.  Clamping to a multiple of the vehicle's weight
        # keeps the integrator alive long enough to record the failure, and
        # raises the flag that tells the scorer this run left the valid domain
        # rather than discovering free thrust.
        weight = float(self._dry_mass.sum()) * GRAVITY + 1.0
        fmag = np.linalg.norm(F, axis=1)
        limit = 60.0 * weight
        if np.any(fmag > limit):
            scale_f = np.minimum(1.0, limit / np.maximum(fmag, 1e-9))
            F *= scale_f[:, None]
            self.diag.clamped = True

        # --- accumulate to bodies -----------------------------------------
        # xfrc_applied takes a world-frame force at the body CoM plus a torque.
        arm = pos - xipos[p.body_id]
        T = _cross3(arm, F)
        np.add.at(data.xfrc_applied[:, :3], p.body_id, F)
        np.add.at(data.xfrc_applied[:, 3:], p.body_id, T)

        # --- diagnostics ---------------------------------------------------
        d = self.diag
        d.lift = float(np.abs(L).sum())
        d.drag = float(np.abs(D).sum())
        d.buoyancy = float(f_buoy.sum())
        d.added_mass = float(m_body.sum())
        d.max_submerged = float(subf.max())
        d.mean_submerged = float(subf.mean())
        d.max_alpha = float(np.abs(alpha).max())
        d.max_dynamic_pressure = float(q.max())
        d.slam = slam

        if self.record_state:
            # Bound circulation, from Kutta-Joukowski: a strip carrying
            # L' = 0.5 * rho * U^2 * c * CL also satisfies L' = rho * U * Gamma,
            # so Gamma = 0.5 * CL * U * c.  The wake is shed from *this*, which
            # is what makes the flow picture derived from the forces in use
            # rather than drawn alongside them.
            self.last_state = {
                "gamma": 0.5 * cl * U * p.chord,
                "pos": pos.copy(),
                "chord": p.chord,
                "dr": p.dr,
                "kind": p.kind,
                "span_axis": s_hat.copy(),
                "chord_axis": c_hat.copy(),
                "normal_axis": n_hat.copy(),
                "alpha": alpha.copy(),
                "submerged": subf.copy(),
                # Structural force: everything the member physically carries.
                #
                # The added-mass gravity compensation is deliberately removed.
                # It exists only to cancel the weight MuJoCo would otherwise
                # apply to entrained fluid, and it is not a load any spar
                # reacts.  Leaving it in is not a small error: a submerged wing
                # strip carries ~32 kg of added mass, so the compensation is
                # ~314 N per strip and it swamped the real aerodynamic load,
                # reporting 3500% structural utilisation for a machine that was
                # merely floating.
                "force": F - np.stack(
                    [np.zeros(p.n), np.zeros(p.n), m_add * GRAVITY], axis=1
                ),
                "q": q.copy(),
                "body_id": p.body_id,
                "flow_at": lambda xyz, _t=t: self.medium.flow_velocity(np.atleast_2d(xyz), _t),
            }
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
