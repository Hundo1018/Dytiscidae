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

    L      = q * dS * CL(alpha, Re, AR, kappa)  along  (s_hat x d_hat)
    D      = q * dS * CD(alpha, Re, AR, CL)  along  d_hat
    F_rot  = C_rot * rho * |v_2d| * omega_s * c^2 * dr   along the lift axis
    F_am   = -d(m_added * v_normal)/dt                   along n_hat
    F_buoy = rho_water * g * V * submerged_fraction      along +Z

Frame convention (verified against a worked example in the tests):
    s_hat  spanwise, root -> tip
    c_hat  chordwise, leading edge -> trailing edge
    n_hat  = s_hat x c_hat   (the "upper" surface normal)
    d_hat  = v_2d / |v_2d|   (downstream direction)
    lift   acts along s_hat x d_hat, positive for positive alpha

The two cross products used to be the other way round -- ``c x s`` and
``d x s`` -- which is self-consistent (both signs flip, so an uncambered wing
lifts correctly) but describes a machine whose wings are upside down: a strip
at +5 deg of geometric incidence reported alpha = -5 deg.  Everything that
reads alpha *without* a matching sign flip was therefore backwards, and one
thing does: the camber shift.  Positive camber subtracted lift.  Stated in the
ordinary aerodynamic convention, as it now is, the sign of alpha is the sign of
incidence and there is nowhere left for that class of bug to hide.

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
    #: Section camber as a fraction of chord.  Sets the zero-lift angle: a
    #: cambered section lifts at zero incidence and stops lifting at a negative
    #: one.  This was generated by the CPPN, stored on the surface, described in
    #: the docstring as driving the zero-lift angle -- and never read by
    #: anything.  Camber is the single most important property a section has,
    #: and for the whole project it was a gene that did nothing.
    camber: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.body_id)
        if self.pitch_axis is None:
            self.pitch_axis = np.full(n, 0.25)
        if self.volume_buoyant is None:
            self.volume_buoyant = np.array(self.volume, dtype=float)
        if self.camber is None:
            self.camber = np.zeros(n)
        if self.ext_local is None:
            self.ext_local = np.stack(
                [self.dr, self.chord, 2.0 * self.half_height], axis=1
            ) if n else np.zeros((0, 3))
        self.area = self.chord * self.dr
        # Normal axis, n = s x c: for a wing with span +Y and chord aft along
        # -X, that is +Z.  Upper surface up.
        self.normal_local = np.cross(self.span_local, self.chord_local)
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
            camber=z1,
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
            camber=cat("camber"),
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


#: How far past the stall angle the attached flow takes to separate completely:
#: the handover starts *at* the stall angle and is finished this far beyond it.
#:
#: 2026-09-23 (MATH_AUDIT F-08): the handover used to run from zero incidence to
#: ``alpha_stall + 16 deg``, so CL rose monotonically all the way to 45 deg and
#: the model had no stall at all.  Thin plates at Re 1e4-1e5 lose attached flow
#: within a few degrees of stall; 6 degrees is a modelling choice, recorded as
#: such in `docs/MATH_AUDIT.md`, not a measurement.
SEPARATION_COMPLETE = np.radians(6.0)

#: Normal-force coefficient of the separated branch, ``CL = CN sin a cos a``,
#: ``CD = CN sin^2 a``.  1.98 is a flat plate broadside (the pressure-drag
#: constant this module always used); 3.4 is a wing carrying a stable
#: leading-edge vortex -- Dickinson, Lehmann & Sane 1999's robofly fit gives
#: CD(90) = 3.46 and CL(45) = 1.80, i.e. CN ~ 3.4-3.6.  One CN for both
#: coefficients, because a separated plate is loaded normal to itself.
CN_PLATE = 1.98
CN_LEV = 3.4


def _stall(alpha, re):
    """Static stall angle and the attached-to-separated weight ``w``.

    ``w`` is 0 up to the stall angle and reaches 1 ``SEPARATION_COMPLETE``
    beyond it, a smoothstep, so ``w`` and ``w'`` are exactly zero at zero
    incidence (F-05) and the attached lift peaks at stall and falls to the plate
    curve after it (F-08).
    """
    alpha_stall = np.radians(11.0)
    # Very low Reynolds number wings stall early and softly.
    alpha_stall = alpha_stall * np.clip(
        0.55 + 0.45 * np.log10(np.maximum(re, 10.0)) / 5.0, 0.5, 1.0)
    t = np.clip((np.abs(alpha) - alpha_stall) / SEPARATION_COMPLETE, 0.0, 1.0)
    return alpha_stall, t * t * (3.0 - 2.0 * t)


def lift_coefficient(
    alpha: np.ndarray, re: np.ndarray, ar: np.ndarray,
    lev: np.ndarray, alpha_e: np.ndarray | None = None,
) -> np.ndarray:
    """Lift coefficient: attached below stall, separated normal force above.

    Attached: the Helmholtz slope ``2 pi / (1 + 2/AR)`` on the *effective*
    incidence ``alpha_e`` -- the Wagner-lagged angle the circulation has had
    time to build to (``alpha`` itself when none is given).  With the solver's
    inflow on, ``AR`` is passed as effectively infinite, because the downwash
    that the Helmholtz factor stands in for is then in the flow itself.

    Separated: ``CN sin(alpha) cos(alpha)`` with ``CN`` raised from a flat
    plate's 1.98 to 3.4 by the leading-edge vortex strength ``lev`` in [0, 1].
    ``lev`` used to be the reduced *pitch rate* (F-02), which is zero on a wing
    that revolves at fixed incidence -- the robofly -- and read CL 1.10 where
    the robofly measures 1.80.  The solver now forms ``lev`` from the strip's
    Rossby number and its travel since the flow last reversed.
    """
    lev = np.clip(lev, 0.0, 1.0)
    ae = alpha if alpha_e is None else alpha_e
    cl_alpha = 2.0 * np.pi / (1.0 + 2.0 / np.maximum(ar, 0.5))
    cl_att = cl_alpha * ae
    cn = CN_PLATE + (CN_LEV - CN_PLATE) * lev
    cl_sep = cn * np.sin(alpha) * np.cos(alpha)
    _, w = _stall(alpha, re)
    # A strong LEV is separated flow from the leading edge on: the robofly's
    # CL is CN sin a cos a from a few degrees up (0.55 at 9 deg; CN 3.5 gives
    # 0.54).  So the LEV strength also moves the handover toward the separated
    # branch, all the way at lev = 1.
    w = w + (1.0 - w) * lev
    return (1.0 - w) * cl_att + w * cl_sep


def drag_coefficient(
    alpha: np.ndarray, re: np.ndarray, ar: np.ndarray, cl: np.ndarray,
    lev: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Skin friction, plus induced drag, plus separated pressure drag.

    The pressure term ``CN sin^2 a`` belongs to separated flow only.  It was
    applied at every incidence, so attached flow paid a flat plate's pressure
    drag and had no leading-edge suction; a cambered section lost L/D for being
    cambered (12.2 against 15.2 flat at AR 6.6, F-12).  Weighted by the same
    handover as the lift now.  Induced drag ``CL^2 / (pi e AR)`` vanishes when
    the solver passes an effectively infinite AR because its inflow model is
    producing the downwash itself.
    """
    lev = np.clip(lev, 0.0, 1.0)
    cd_f = skin_friction_cd(re)
    # Oswald efficiency: low for the stubby, highly twisted surfaces this
    # pipeline tends to generate.
    oswald = 0.75
    cd_i = cl * cl / (np.pi * oswald * np.maximum(ar, 0.5))
    _, w = _stall(alpha, re)
    w = w + (1.0 - w) * lev
    cn = CN_PLATE + (CN_LEV - CN_PLATE) * lev
    sa = np.sin(alpha)
    cd_p = w * cn * sa * sa
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


#: R.T. Jones's two-term approximation to Wagner's function,
#: ``phi(s) = 1 - A1 exp(-b1 s) - A2 exp(-b2 s)``, ``s`` in semichords of
#: travel (Jones 1940, NACA Report 681; constants as quoted by
#: arXiv:2104.15122).  An impulsively started plate carries half its
#: steady-state circulation at once and builds the rest over a few chords.
WAGNER_A = (0.165, 0.335)
WAGNER_B = (0.0455, 0.3)
#: The leading-edge vortex of a translating wing persists for about two chords
#: of travel after an impulsive start and is gone by about four (Dickinson &
#: Goetz 1993, J Exp Biol 174:45, as summarised in runs/_logs/
#: aero_literature_0923.md item 3: "~2 chords").  Reset at every reversal of
#: the chordwise flow, which is every half-stroke of a flapping wing.
LEV_TRAVEL = (2.0, 4.0)
#: A revolving wing keeps its LEV attached below a Rossby number of about 3
#: (Lentink & Dickinson 2009, J Exp Biol 212:2705: fly wings, Ro ~ 2.9, stable;
#: translating, Ro = infinity, not).  Where it is gone is not a published
#: threshold; 8 is a modelling choice recorded in MATH_AUDIT.
LEV_ROSSBY = (3.0, 8.0)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rossby_lev(U, chord, omega, s_hat):
    """LEV strength from the strip's own Rossby number, ``U / (|w_perp| c)``.

    ``w_perp`` is the strip's angular velocity less its pitch about the span:
    a strip revolving about a hinge at radius ``r`` has ``U = w r`` and so
    ``Ro = r / c``, the revolving-wing Rossby number; a translating strip has
    ``Ro`` infinite.
    """
    c = np.maximum(chord, 1e-6)
    w_perp = omega - np.einsum("ni,ni->n", omega, s_hat)[:, None] * s_hat
    ro = U / np.maximum(np.linalg.norm(w_perp, axis=1) * c, 1e-9)
    rlo, rhi = LEV_ROSSBY
    return 1.0 - _smoothstep((ro - rlo) / (rhi - rlo))


class UnsteadyState:
    """Per-strip history: Wagner lag and chords travelled since reversal.

    ``update`` returns ``(alpha_e, lev)``: the circulation-lagged incidence the
    attached branch uses, and the leading-edge-vortex strength -- the larger of
    the rotational (Rossby) and the delayed-stall (travel) mechanisms.
    """

    def __init__(self, n: int) -> None:
        self.x = np.zeros((2, n))
        self.s = np.zeros(n)
        self.rev = np.zeros(n, bool)
        self.primed = False

    def reset(self) -> None:
        self.x[:] = 0.0
        self.s[:] = 0.0
        self.rev[:] = False
        self.primed = False

    def update(self, alpha, rev, U, chord, omega, s_hat, dt):
        c = np.maximum(chord, 1e-6)
        if self.primed:
            flip = rev != self.rev
            self.x[:, flip] = 0.0
            self.s[flip] = 0.0
        self.rev = rev.copy()
        self.primed = True
        ds = U * dt / c                         # chords this step
        self.s += ds
        a1, a2 = WAGNER_A
        for i, b in enumerate(WAGNER_B):
            # exact for a step held constant over dt; semichords = 2 * chords
            self.x[i] += (alpha - self.x[i]) * (1.0 - np.exp(-b * 2.0 * ds))
        alpha_e = alpha * (1.0 - a1 - a2) + a1 * self.x[0] + a2 * self.x[1]
        lo, hi = LEV_TRAVEL
        lev_travel = 1.0 - _smoothstep((self.s - lo) / (hi - lo))
        return alpha_e, np.maximum(rossby_lev(U, chord, omega, s_hat), lev_travel)


class InducedFlow:
    """Momentum-theory downwash of a machine's lifting system, one vector.

    Glauert's actuator disc: the fluid through a disc of area ``A`` that
    carries force ``F`` is accelerated by ``w`` along ``-F`` with

        w |V + w| = |F| / (2 rho A)

    where ``V`` is the flow the disc sees.  In hover that is Rankine-Froude,
    ``w = sqrt(T / 2 rho A)``; in forward flight with ``A = pi b^2 / 4`` it is
    exactly lifting-line theory's induced angle ``CL / (pi AR)`` -- so one
    model covers both, and the strips then use the 2D lift slope and no
    separate induced drag (they would count it twice).  The disc spans the
    machine's tip-to-tip span, so a bilateral pair is one wing of the full
    aspect ratio (F-12) and not two halves.  MATH_AUDIT F-13.

    ``w`` follows its target with a first-order lag of one disc radius of
    travel, ``tau = (b/2) / |V + w|`` -- the wake has to convect away before
    the inflow it induces is established.  A modelling choice (Pitt-Peters is
    the rotorcraft form and its constant could not be confirmed), recorded in
    MATH_AUDIT.
    """

    def __init__(self, span: float) -> None:
        self.span = float(max(span, 0.0))
        self.area = np.pi * self.span**2 / 4.0
        self.w = np.zeros(3)

    def reset(self) -> None:
        self.w[:] = 0.0

    def update(self, F: np.ndarray, V: np.ndarray, rho: float, dt: float) -> np.ndarray:
        T = float(np.linalg.norm(F))
        if self.area <= 1e-9 or T < 1e-9 or rho <= 0.0:
            target = np.zeros(3)
        else:
            f = F / T
            k = T / (2.0 * rho * self.area)
            v_par = float(V @ f)
            v_perp2 = float(V @ V) - v_par * v_par
            # w * sqrt(v_perp^2 + (v_par - w)^2) = k: the smallest root, by
            # bisection -- fixed-point iteration oscillates in hover.
            g = lambda x: x * np.sqrt(max(v_perp2, 0.0) + (v_par - x) ** 2) - k
            lo, hi = 0.0, max(1.0, np.sqrt(k)) * 4.0 + abs(v_par) + np.sqrt(max(v_perp2, 0.0))
            while g(hi) < 0.0:
                hi *= 2.0
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if g(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            target = -f * 0.5 * (lo + hi)
        through = float(np.linalg.norm(V + target))
        tau = 0.5 * self.span / max(through, 0.05)
        self.w += (target - self.w) * (1.0 - np.exp(-dt / max(tau, 1e-6)))
        return self.w


#: The aspect ratio the strips are given when the inflow model is on: large
#: enough that ``2 pi / (1 + 2/AR)`` is the 2D slope and ``CL^2/(pi e AR)``
#: vanishes, because the downwash both stand in for is then in the flow.
INFLOW_AR = 1.0e6


def _wing_span(model, panels) -> float:
    """Tip-to-tip extent of the lifting strips at the model's rest pose."""
    import mujoco
    sel = panels.kind == WING
    if not np.any(sel):
        return 0.0
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    R = d.xmat.reshape(-1, 3, 3)[panels.body_id[sel]]
    pos = d.xpos[panels.body_id[sel]] + np.einsum("nij,nj->ni", R, panels.pos_local[sel])
    diff = pos[:, None, :] - pos[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


def machine_flow(model, data, medium, t: float) -> np.ndarray:
    """The flow the machine's root body sees: medium velocity minus its own."""
    import mujoco
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, v, 0)
    u = medium.flow_velocity(data.xipos[1:2], t)
    return np.asarray(u, float).reshape(-1, 3)[0] - v[3:]


class ImplicitAeroDamping:
    """Lift and drag made implicit by splitting their damping off.

    Lift and drag go into ``xfrc_applied`` and are integrated explicitly; added
    mass goes into the mass matrix and is implicit.  An explicit damping force
    ``-b v`` on a body of inertia ``m`` is stable only while ``b dt / m`` stays
    below about two, and a wing strip in water has ``b`` of hundreds of N s/m on
    a few grams of spar.  What kept it stable was the isotropic added mass,
    about 2.8x too large (MATH_AUDIT F-03): correct it and eel and ray run away
    at dt = 0.004 (2026-09-23: 11,110 and 27,341 rad).

    The split: each step, project each strip's linearised damping onto the
    joints, ``B_kk = sum_i b_i |J_i e_k|^2``, put ``B_kk`` into MuJoCo's own
    ``dof_damping`` -- which ``implicitfast`` integrates implicitly -- and add
    ``+B_kk qdot_k`` back to ``qfrc_applied``.  At the current state the two
    cancel exactly, so the force the machine feels is unchanged; what changes is
    that the stiff part is integrated implicitly.  The price is an O(dt) lag in
    the damping force, the same order as the explicit scheme's own error.

    ``b_i`` bounds the strip's force derivative with respect to its own
    velocity.  For ``F = q S (CL l + CD d)``, a streamwise perturbation gives
    ``rho S U sqrt(CL^2 + CD^2) = 2|F|/U`` and a normal one adds the lift slope,
    ``q S CL_alpha / U``; the diagonal only, because MuJoCo's ``dof_damping`` is
    diagonal.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.base = model.dof_damping.copy()
        nb, nv = model.nbody, model.nv
        # anc[j, k]: dof k moves body j (k's body is j or one of j's ancestors).
        anc = np.zeros((nb, nv), bool)
        for j in range(nb):
            b = j
            while b > 0:
                anc[j, model.dof_bodyid == b] = True
                b = int(model.body_parentid[b])
        # Articulated joints only.  The split adds dt*B of effective inertia
        # to first order, and on the free root that is 20-50% more pitch
        # inertia for a glider -- it broke the gannet's glide under the launch
        # scatter (sink 1.5 -> 10.5 m/s, 2026-09-26).  The instability the
        # split exists for is in light spars (b dt >> their mass), never in the
        # root, which is heavy and stable explicitly.
        free = np.zeros(nv, bool)
        for j in range(model.njnt):
            if model.jnt_type[j] == 0:                       # mjJNT_FREE
                a = model.jnt_dofadr[j]
                free[a:a + 6] = True
        anc[:, free] = False
        self.anc = anc
        self.last_b = np.zeros(nv)

    def reset(self) -> None:
        self.model.dof_damping[:] = self.base
        self.last_b[:] = 0.0

    def clear(self, data) -> None:
        """The step's damping with the split off: the dry model's own.  Called
        every step either way, because `dof_damping` has one owner that
        rewrites it and everyone else (jets) adds to what it wrote."""
        self.model.dof_damping[:] = self.base
        data.qfrc_applied[:] = 0.0

    def apply(self, data, pos: np.ndarray, body_id: np.ndarray, b: np.ndarray) -> None:
        """``pos`` (n, 3) world strip positions, ``b`` (n,) N s/m per strip."""
        m = self.model
        if m.nv == 0:
            return
        root = m.body_rootid[body_id]
        r = pos - data.subtree_com[root]                         # (n, 3)
        rot = data.cdof[:, :3]                                    # (nv, 3)
        lin = data.cdof[:, 3:]
        # velocity of each strip per unit rate of each dof: lin + rot x r
        u = lin[None, :, :] + np.cross(rot[None, :, :], r[:, None, :])
        u2 = np.einsum("nkd,nkd->nk", u, u) * self.anc[body_id]
        B = b @ u2                                                # (nv,)
        m.dof_damping[:] = self.base + B
        # It owns ``qfrc_applied`` outright -- nothing else in the project writes
        # it -- and rewrites all of it every step, so a snapshot restore or a
        # reset between steps cannot leave a stale compensation behind.
        data.qfrc_applied[:] = B * data.qvel
        self.last_b = B


def strip_damping(q, area, lift, drag, rho, aspect_ratio, is_wing,
                  lift_scale: float = 1.0) -> np.ndarray:
    """Per-strip damping bound ``b_i`` for `ImplicitAeroDamping`, N s/m.

    ``2|F_aero|/U + q S CL_alpha / U``, with ``|F_aero| = sqrt(L^2 + D^2)`` --
    the *velocity-dependent* force only (circulatory lift and drag, and a bluff
    body's drag).  Buoyancy does not depend on velocity and must not be in it:
    the first version used the whole strip force, and a floored ``U`` turned a
    resting medusa's buoyancy into 1e5 N s/m and ran it away (2026-09-23).  The
    lift slope is the attached-flow ``2 pi / (1 + 2/AR)`` on lifting strips.
    ``U`` is recovered from ``q`` and ``rho`` so both evaluation paths form it
    from what they already hold.  Every term scales as ``U``, so it goes to zero
    with the flow rather than blowing up.
    """
    U = np.sqrt(2.0 * np.maximum(q, 0.0) / np.maximum(rho, 1e-9))
    Us = np.maximum(U, 1e-3)
    # The slope term carries the same `lift_scale` the lift does: a bound on a
    # force the auditor has scaled to zero must be zero too.
    slope = np.where(is_wing, 2.0 * np.pi / (1.0 + 2.0 / np.maximum(aspect_ratio, 0.1)),
                     0.0) * lift_scale
    fa = np.sqrt(lift * lift + drag * drag)
    return np.where(U > 1e-3, (2.0 * fa + q * area * slope) / Us, 0.0)


def slam_mass(m_add, rho, chord, dr, is_wing, scale: float = 1.0) -> np.ndarray:
    """The entrained mass the slam diagnostic differences: a wing's *normal*
    value, ``rho pi c^2 / 4 dr``, whatever direction the flow comes from.

    Slam is a normal impact -- the load of a surface being wetted -- and the
    direction-dependent tensor (F-03) changes a wing's entrained mass every
    time the flow turns relative to it, i.e. every flap.  Differencing that
    read flapping in water as slamming: the ray's nose-first entry swung between
    19 and 730 kPa with configuration (2026-09-26).  Bluff elements keep their
    own value.
    """
    return np.where(is_wing, rho * np.pi * chord**2 * 0.25 * dr * scale, m_add)


def finish_bodies(fb: np.ndarray, fsum_b: np.ndarray, m_body: np.ndarray,
                  limit: float) -> bool:
    """Per-machine limiter and weight cancellation, in place on ``fb`` (nb, 6).

    The limiter is a last resort: the quasi-steady model is only valid for
    states a real machine could be in, and once a candidate is tumbling at
    50 m/s the forces can be arbitrarily large.  It bounds the machine's total
    fluid load, ``sum |F_i|``, by ``limit`` (60x its dry weight) and scales
    every strip by the same factor, so the distribution of load -- and hence
    the moments -- is kept.  It was a bound *per strip* until 2026-09-23, so a
    200-strip machine could carry 200x the stated bound and a clamped machine
    had its load redistributed toward its weakest strips (MATH_AUDIT F-04).

    Then the weight MuJoCo applies to the entrained fluid is cancelled at each
    body's centre of mass (F-14).  Returns whether the limiter bound.
    """
    total = float(fsum_b.sum())
    clamped = total > limit
    if clamped:
        fb *= limit / total
    fb[:, 2] += m_body * GRAVITY
    return clamped


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
        cd_scale: float = 1.0,
        lift_scale: float = 1.0,
        disc_span: float | None = None,
    ) -> None:
        self.model = model
        self.panels = panels
        self.medium = medium
        self.added_mass_scale = added_mass_scale
        # Deliberate handles for the auditor to move.  A design that only works
        # at one exact value of a coefficient has found a hole in the model, and
        # the only way to find that out is to move the coefficient.
        self.cd_scale = cd_scale
        self.lift_scale = lift_scale
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

        # Whether this model needs MuJoCo's derived constants rebuilt after the
        # mass edit below.
        #
        # `body_mass` is an input to `cinert`, which `mj_crb` turns into the
        # mass matrix -- except for bodies MuJoCo has marked `simple`, whose
        # DOFs take their mass from `dof_M0`, a constant computed when the
        # model was compiled.  A body is simple when nothing is jointed to it,
        # which for this project means a machine with no actuated degrees of
        # freedom: exactly the degenerate designs the search keeps producing.
        # For those, writing into `body_mass` updated `cinert` and never
        # reached `M`, so the machine swam with none of its entrained water --
        # measured at +32.20 kg bookkept against +0.00 kg applied.
        #
        # `mj_setConst` rebuilds `dof_M0` and the other derived constants, and
        # costs 1.8 us with a reused scratch.  It is gated on detection because
        # no panel-carrying body of any seed plan is simple, so a real machine
        # pays nothing: it is only the jointless case that needs it.
        #
        # The scratch `MjData` is not optional.  `mj_setConst(model, data)`
        # leaves `data` at `qpos0` -- it resets the simulation state -- so it
        # must be handed a throwaway.
        self._needs_const = bool(
            len(panels.body_id)
            and np.any(model.body_simple[np.unique(panels.body_id)])
        )
        self._const_scratch = None

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
        #: Apply the wing's full added-mass tensor rather than the plate's
        #: normal entry in every direction.  **Off**: it is correct and the
        #: solver does not survive it at this timestep.  See MATH_AUDIT F-03
        #: and `experiments/wing_added_mass`.
        #: **On since 2026-09-23**, with `implicit_damping`: the pair closes
        #: F-03.  Measured at dt = 0.004, largest joint angle over 5 s, driven:
        #: tensor with explicit lift/drag ran ray to 7,282 rad (27,341 before
        #: the lever-arm fix); with the split every plan stays under 2.8 rad in
        #: air and 2.3 in water (`experiments/flight_audit/stability_probe.py`).
        self.wing_added_mass_tensor = True
        #: Lift and drag damping integrated implicitly (`ImplicitAeroDamping`).
        self.implicit_damping = True
        self._damping = ImplicitAeroDamping(model)
        #: Wagner lag and the LEV's travel/Rossby history (F-02, F-13).
        self.unsteady = True
        self._unsteady = UnsteadyState(panels.n)
        #: Momentum-theory inflow over the machine's span (F-12, F-13).  With
        #: it on, strips use the 2D lift slope and no separate induced drag.
        self.inflow = True
        if disc_span is None:
            disc_span = _wing_span(model, panels)
        self._inflow = InducedFlow(disc_span)
        # The machine's own aspect ratio, tip-to-tip span squared over its
        # lifting area -- what a steady, lifting-line reading of the same
        # inflow gives, and what `steady()` measurements use.
        s_wing = float(panels.area[panels.kind == WING].sum()) if panels.n else 0.0
        self._machine_ar = (disc_span**2 / s_wing) if (disc_span > 0 and s_wing > 0) else None
        #: Quasi-static evaluation: see `steady`.
        self.quasi_static = False
        if self._inflow.area <= 1e-9:
            # A lifting system with no span -- a single strip, or none -- has
            # no disc to put momentum through; it keeps the finite-wing model.
            self.inflow = False
        self.record_state = False
        self.last_state: dict | None = None
        # Scratch buffers reused every step.
        self._nbody = model.nbody
        self._vel6 = np.zeros(6)
        self._bodies = np.unique(panels.body_id)

    def _publish_inertia(self, mass: np.ndarray, inertia: np.ndarray) -> None:
        """Write the augmented inertia into the model so the solver uses it.

        The write is the easy half.  The other half is making sure MuJoCo
        derives the mass matrix from it -- see `_needs_const` in `__init__`.
        """
        self.model.body_mass[:] = mass
        self.model.body_inertia[:] = inertia
        if self._needs_const:
            import mujoco

            if self._const_scratch is None:
                self._const_scratch = mujoco.MjData(self.model)
            mujoco.mj_setConst(self.model, self._const_scratch)

    def steady(self):
        """Context for quasi-static probes (trim, lift and thrust margins).

        They pose the machine, reset the solver and read one step's forces, so
        a history means nothing to them: the Wagner lag would restart at half
        circulation and every pose would be "just reversed" with a fresh LEV.
        Inside it the incidence is not lagged, the LEV comes from the steady
        (Rossby) mechanism alone, and the downwash is the steady lifting-line
        one -- the machine's tip-to-tip aspect ratio -- instead of the lagged
        inflow.
        """
        import contextlib

        @contextlib.contextmanager
        def _cm():
            was = self.quasi_static
            self.quasi_static = True
            try:
                yield self
            finally:
                self.quasi_static = was
        return _cm()

    def reset(self) -> None:
        self._prev_vn[:] = 0.0
        self._prev_ma[:] = 0.0
        self._prev_t = None
        self._primed = False
        # Restore dry inertia: leaving a previous episode's entrained water in
        # the mass matrix would silently make the next episode heavier.
        self._publish_inertia(self._dry_mass, self._dry_inertia)
        self._damping.reset()
        self._unsteady.reset()
        self._inflow.reset()
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

        # Body 6D velocities, world frame.  The linear part is the velocity of
        # the body's *centre of mass* (``xipos``), not of its frame origin:
        # mj_objectVelocity on mjOBJ_BODY reports the inertial frame, which is
        # what ``batchroll`` has said since it was written.  The lever arm
        # below was measured from ``xpos`` until 2026-09-23, so a wing hinged
        # at its root had every strip moving as if the hinge were at the wing's
        # centre of mass as well -- flapping strip speeds came out 1.8-3.6x too
        # high in U^2 S.  See ROADMAP "Why nothing flies, measured a third
        # time".
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
        v_elem = v_org + _cross3(omega, pos - xipos[p.body_id])

        # --- medium -------------------------------------------------------
        rho, mu, subf = self.medium.properties(pos, p.half_height, t)
        u_flow = self.medium.flow_velocity(pos, t)
        is_wing = p.kind == WING
        live_inflow = self.inflow and not self.quasi_static
        if live_inflow:
            u_flow = u_flow + np.where(is_wing[:, None], self._inflow.w[None, :], 0.0)
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
        # Fold into [-pi/2, pi/2] by a *shift* of pi, not a mirror.  Flow
        # arriving over the trailing edge sees the section rotated half a turn:
        # 171 deg of incidence is -9 deg with the old trailing edge leading,
        # and the lift axis ``s x d`` has flipped with the flow, so the force
        # reverses as it physically must -- a plate swept back and forth at a
        # fixed pitch pushes up one way and down the other.  The mirror this
        # replaced, ``atan2(sin, |cos|)``, sent 171 deg to +9 and gave the same
        # force in both directions (+-10 m/s at 10 deg: equal lift), which paid
        # every reversing stroke for lift it cannot make.
        rev = cos_a < 0.0
        alpha = np.arctan2(sin_a, np.where(rev, -cos_a, cos_a) + 1e-12)
        alpha = np.where(rev, -alpha, alpha)
        # Camber shifts the zero-lift angle.  Thin-airfoil theory gives
        # alpha_0 = -2 f/c for a parabolic arc, so a section with 6% camber
        # still lifts at 7 degrees *below* geometric zero.  Without this a wing
        # can only lift by being pitched, which is why every design in this
        # project trimmed at an absurd attitude or not at all.  Reversed flow
        # sees the arc from its other end: the arc is fore-aft symmetric, so
        # it still lifts toward its convex side, which is the opposite sign on
        # the flipped lift axis.
        alpha = alpha + 2.0 * np.where(rev, -p.camber, p.camber)

        # Angular rate about the span axis: the strip's *pitch* rate, which the
        # Kramer rotational force below wants.  It used to index the LEV as
        # well (F-02); the LEV now comes from `UnsteadyState`.
        omega_s = np.einsum("ni,ni->n", omega, s_hat)
        if self.unsteady and not self.quasi_static:
            alpha_e, lev = self._unsteady.update(alpha, rev, U, p.chord, omega, s_hat, dt)
        else:
            alpha_e, lev = alpha, rossby_lev(U, p.chord, omega, s_hat)
        if live_inflow:
            ar_eff = np.full(p.n, INFLOW_AR)
        elif self._machine_ar is not None:
            ar_eff = np.full(p.n, self._machine_ar)
        else:
            ar_eff = p.aspect_ratio

        lift_axis = _cross3(s_hat, d_hat)
        lift_axis /= np.maximum(np.linalg.norm(lift_axis, axis=1, keepdims=True), 1e-12)

        F = np.zeros((p.n, 3))

        # Circulatory lift and drag, on the lifting strips only.
        cl = np.where(
            is_wing,
            lift_coefficient(alpha, re, ar_eff, lev, alpha_e),
            0.0)
        cd = np.where(is_wing, drag_coefficient(alpha, re, ar_eff, cl, lev), 0.0)
        L = q * p.area * cl * self.lift_scale
        D = q * p.area * cd * self.cd_scale
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
        # The full (not strip-projected) relative flow direction, for every
        # element.  Hoisted out of the bluff branch because the added-mass
        # tensor below needs it for wings too; the values are unchanged.
        U_full = np.linalg.norm(v_rel, axis=1)
        U_full_safe = np.maximum(U_full, 1e-6)
        d_full = v_rel / U_full_safe[:, None]
        if n_bluff:
            b = ~is_wing
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
            f_axial = self.cd_scale * (
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
            f_cross = self.cd_scale * 0.5 * rho * u_cross**2 * CD_CROSSFLOW * a_side

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
        #
        # Sign: rotating a strip by +omega about s_hat turns c toward n, and
        # with alpha = atan2(v.n, v.c) that *lowers* the incidence --
        # d(alpha)/dt = -omega_s.  Kramer circulation adds lift when the wing
        # pitches nose-up (Sane & Dickinson 2002), so the force is along the
        # lift axis with -omega_s.  It was +omega_s until 2026-09-23, which
        # penalised exactly the pitch-reversal kinematics that make the term
        # worth having.
        f_rot = np.where(
            is_wing,
            -self.c_rot * rho * U * omega_s * p.chord**2 * p.dr,
            0.0,
        )
        F += f_rot[:, None] * lift_axis
        if live_inflow:
            # For the next step: the lifting system's force and the flow it
            # sees.  A one-step lag, well inside the inflow's own time constant.
            wing_any = bool(is_wing.any())
            self._inflow.update(
                F[is_wing].sum(axis=0) if wing_any else np.zeros(3),
                machine_flow(self.model, data, self.medium, t),
                float(rho[is_wing].mean()) if wing_any else 0.0, dt)

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
        # the added mass exceeds the structural mass.  Folding it in is done by
        # `_publish_inertia`, which also rebuilds MuJoCo's derived constants
        # where the model needs it -- writing `body_mass` is not the same as
        # MuJoCo using it, and for a jointless machine it was not using it.  Two corrections come with
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
        # Direction cosines in the element's own (span, chord, normal) frame,
        # from the same relative-velocity direction the drag used.  Computed
        # for every element: the wing branch below projects onto them too.
        dc = np.stack([
            np.einsum("ni,ni->n", d_full, s_hat),
            np.einsum("ni,ni->n", d_full, c_hat),
            np.einsum("ni,ni->n", d_full, n_hat),
        ], axis=1) ** 2
        moving = dc.sum(axis=1) > 1e-6
        ca_eff = np.einsum("ni,ni->n", dc, ca_axis)
        # A body momentarily at rest has no direction of motion to project onto;
        # fall back to the isotropic mean rather than to zero.
        ca_eff = np.where(moving, ca_eff, ca_axis.mean(axis=1))

        # A wing strip's added mass is a **tensor** and the default here is a
        # scalar: the plate's normal value `rho pi c^2/4 per unit span` applied
        # whichever way the strip accelerates.  Strip theory gives all three,
        #
        #     m_span = 0      m_chord = rho pi t^2/4 b      m_normal = rho pi c^2/4 b
        #
        # verified against the closed form at c/t = 10, 100 and 1000 in
        # `benchmarks/layers.py` layer 2 with no simulation in the chain.
        # `docs/MATH_AUDIT.md` **F-03**.
        #
        # The tensor is behind `wing_added_mass_tensor`, **on** since 2026-09-23.
        # Switching it on makes layer 2 hold (0.729 -> 8.3e-06) and takes a
        # gannet's wing added mass from 70.2 kg to 25.1 kg -- and the surplus
        # inertia the scalar carried had been what kept the *explicit* lift and
        # drag stable at dt = 0.004 (`experiments/wing_added_mass`).  That
        # needed an implicit treatment of those forces, not a coefficient, and
        # `ImplicitAeroDamping` is it.
        if self.wing_added_mass_tensor:
            t_wing = np.maximum(p.ext_local[:, 2], 1e-5)
            # `rho` is per element -- one straddling the free surface carries a
            # blended density -- so it joins `dr` in the column factor.  Writing
            # it as `rho * ... * p.dr[:, None]` broadcasts (N,) against (N,1)
            # into (N,N), which a single-panel benchmark does not catch.
            m_wing_axis = (rho * p.dr)[:, None] * (np.pi * 0.25) * np.stack([
                np.zeros(p.n),      # spanwise: a flat plate entrains nothing
                t_wing**2,          # chordwise
                p.chord**2,         # normal
            ], axis=1)
            m_wing = np.einsum("ni,ni->n", dc, m_wing_axis)
            # At rest the same fallback the bluff branch uses: the mean of the
            # three, not zero, and not the normal entry.
            m_wing = np.where(moving, m_wing, m_wing_axis.mean(axis=1))
        else:
            m_wing = rho * np.pi * p.chord**2 * 0.25 * p.dr

        m_add = np.where(
            is_wing,
            m_wing,
            ca_eff * rho * p.volume,
        ) * self.added_mass_scale

        m_body = np.zeros(self._nbody)
        np.add.at(m_body, p.body_id, m_add)
        self._publish_inertia(
            self._dry_mass + m_body,
            self._dry_inertia + (m_body * self._lever2)[:, None],
        )
        # The weight MuJoCo will apply to the entrained fluid is cancelled at
        # each body's *centre of mass*, where MuJoCo applies it -- see
        # `finish_bodies`.  It was added at every strip until 2026-09-23, which
        # also put a couple ``sum (r_i - com) x m_i g`` on every body: 22.8 N m
        # about the beetle's wing axis in water (MATH_AUDIT F-14).

        # The slamming rate term is still computed, but only as a *diagnostic*:
        # the structural check needs to know the peak entry load, while the
        # dynamics get the same physics through the varying mass matrix.
        m_s = slam_mass(m_add, rho, p.chord, p.dr, is_wing, self.added_mass_scale)
        if self._primed:
            slam = float(np.abs((m_s - self._prev_ma) / dt * vn).max())
        else:
            slam = 0.0
            self._primed = True
        self._prev_vn = vn.copy()
        self._prev_ma = m_s.copy()
        dmv = np.zeros(p.n)

        # Buoyancy: only the genuinely submerged portion, at true water density,
        # and only over the volume that is actually sealed rather than flooded.
        f_buoy = self.medium.water.rho * GRAVITY * p.volume_buoyant * subf
        F[:, 2] += f_buoy

        # --- accumulate to bodies -----------------------------------------
        # xfrc_applied takes a world-frame force at the body CoM plus a torque.
        # Summed per body first, unscaled, in panel order -- the batched path's
        # deterministic gather does exactly this -- and then `finish_bodies`
        # applies the machine's limiter and the weight cancellation.
        fmag = np.linalg.norm(F, axis=1)
        if self.implicit_damping:
            self._damping.apply(data, pos, p.body_id, strip_damping(
                q, p.area, L, D, rho, ar_eff, is_wing, self.lift_scale))
        else:
            self._damping.clear(data)
        nb = self._nbody
        fb = np.zeros((nb, 6))
        arm = pos - xipos[p.body_id]
        np.add.at(fb[:, :3], p.body_id, F)
        np.add.at(fb[:, 3:], p.body_id, _cross3(arm, F))
        fsum_b = np.zeros(nb)
        np.add.at(fsum_b, p.body_id, fmag)
        weight = float(self._dry_mass.sum()) * GRAVITY + 1.0
        clamped = finish_bodies(fb, fsum_b, m_body, 60.0 * weight)
        if clamped:
            self.diag.clamped = True
        data.xfrc_applied[:] += fb

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
                # Speed of the flow across each strip.  Recorded so a probe can
                # check the kinematics a strip was given against its hinge.
                "speed": U.copy(),
                # Direction cosines of the full relative flow in each
                # element's own (span, chord, normal) frame.  What the
                # added-mass tensor is projected onto, so a probe can ask
                # which way the fluid is actually being pushed.
                "flow_cosines": np.stack([
                    np.einsum("ni,ni->n", d_full, s_hat),
                    np.einsum("ni,ni->n", d_full, c_hat),
                    np.einsum("ni,ni->n", d_full, n_hat),
                ], axis=1),
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
            v_com = vel6[3:]  # already at xipos for mjOBJ_BODY
            power -= float(f[:3] @ v_com + f[3:] @ vel6[:3])
        return power
