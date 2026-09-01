"""Central pattern generator, and the *discovered* control basis.

Why not pitch/yaw/roll
----------------------
Those axes are a convention borrowed from aircraft that have a fuselage, a
recognisable nose, and control surfaces arranged to produce near-decoupled
moments about three orthogonal axes.  A generated triphibian machine has none of
that guaranteed.  Its "wings" may be paddles, it may have six limbs and no tail,
and the motion produced by beating harder on one side may be a coupled
surge-roll-heave with no name.  Imposing roll/pitch/yaw on such a machine forces
the controller to synthesise axes the body does not naturally have, wasting most
of its actuation fighting itself.

So the control axes are measured, not assumed:

  1. Drive the CPG with a set of random parameter perturbations.
  2. Record the resulting mean body twist (6-vector: linear + angular velocity,
     in the body frame) for each perturbation.
  3. Least-squares fit the Jacobian J from CPG parameters to body twist.
  4. Take the SVD, ``J = U S V^T``.  The leading columns of ``V`` are the
     parameter directions that move the machine most; the matching columns of
     ``U`` describe *what motion each one actually produces*, and the singular
     values say how much authority each has.

The controller then commands coefficients in that basis.  Mode 0 might be
"surge, with a bit of roll"; mode 1 might be "heave coupled to pitch".  They are
whatever this body can do, ranked by how well it can do them.

Crucially the basis is **measured separately per medium**.  The same wing that
is a high-authority lifting surface in air is a high-drag paddle in water, so a
machine's control axes genuinely change when it crosses the surface -- and the
number of usable axes changes too.  That is a real property of amphibious
vehicles and it falls straight out of this construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class CPGParams:
    """Open parameters of the pattern generator.

    Flattened as ``[amplitude(n), phase(n), offset(n), frequency(1)]``.
    """

    amplitude: np.ndarray
    phase: np.ndarray
    offset: np.ndarray
    frequency: float

    @property
    def n(self) -> int:
        return len(self.amplitude)

    def flat(self) -> np.ndarray:
        return np.concatenate([self.amplitude, self.phase, self.offset, [self.frequency]])

    @staticmethod
    def from_flat(v: np.ndarray, n: int) -> "CPGParams":
        v = np.asarray(v, float)
        return CPGParams(
            amplitude=v[:n],
            phase=v[n : 2 * n],
            offset=v[2 * n : 3 * n],
            frequency=float(v[3 * n]),
        )

    def clipped(self, lo: np.ndarray, hi: np.ndarray) -> "CPGParams":
        """Clamp offsets and amplitudes into the joints' physical travel."""
        span = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        off = np.clip(self.offset, lo + 0.05 * span, hi - 0.05 * span)
        amp = np.clip(self.amplitude, 0.0, np.maximum(span - np.abs(off - mid), 1e-3))
        return CPGParams(amp, self.phase, off, float(np.clip(self.frequency, 0.1, 20.0)))


class CPG:
    """A bank of phase-coupled oscillators, one per actuated joint.

    Deliberately simple -- sinusoids with per-joint amplitude, phase and offset,
    sharing one frequency.  The expressive power that matters for locomotion is
    in the *phase relationships*, and those are fully represented here.  Anything
    fancier (Hopf oscillators, coupled Matsuoka networks) mostly buys smoother
    transients, which the mobility basis handles at a higher level anyway.
    """

    def __init__(self, n_joints: int, base_frequency: float = 2.0,
                 joint_range: np.ndarray | None = None) -> None:
        self.n = n_joints
        if joint_range is None:
            joint_range = np.tile(np.array([-1.0, 1.0]), (max(n_joints, 1), 1))
        self.lo = np.asarray(joint_range, float)[:, 0]
        self.hi = np.asarray(joint_range, float)[:, 1]
        self.base = CPGParams(
            amplitude=0.45 * (self.hi - self.lo) * 0.5,
            # A travelling wave along the joint index is a much better starting
            # point than all-in-phase: it already looks like a gait, and it
            # breaks the symmetry that would otherwise make every joint fight
            # every other one.
            phase=np.linspace(0.0, np.pi, max(n_joints, 1)),
            offset=0.5 * (self.hi + self.lo),
            frequency=base_frequency,
        )
        self.t = 0.0
        #: Added to every joint's phase.  Rollouts used to begin at exactly the
        #: same point in the stroke every time, so the phase the observation
        #: reports was the same number at the same moment of every episode the
        #: policy ever saw.
        self.phase_offset = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def command(self, params: CPGParams, t: float) -> np.ndarray:
        """Target joint angles at time ``t``."""
        p = params.clipped(self.lo, self.hi)
        return p.offset + p.amplitude * np.sin(
            2.0 * np.pi * p.frequency * t + p.phase + self.phase_offset)

    @property
    def n_params(self) -> int:
        return 3 * self.n + 1


#: Degrees of freedom of a body twist: surge, sway, heave, roll, pitch, yaw.
#: This is the width a *shared* policy commands in, and unlike the mode count
#: it is a property of space rather than of a body, which is the whole point.
TWIST_DIM = 6

#: What a saturated intent asks for, as a fraction of a body's own reach.
#:
#: A policy's output is a tanh, so it saturates constantly, and reading
#: saturation as "everything this body has" leaves no headroom and makes every
#: strong opinion a full-authority command.  Measured on 48 bodies drawn at
#: random from arch30 and arch31, paired (the same body scored with the policy
#: off and on), against a policy trained 14 generations:
#:
#:     intent x1.0   delta -0.00387 +/- 0.00173 (SE)   t = -2.23   13/48 better
#:     intent x0.5   delta -0.00075 +/- 0.00170        t = -0.44   16/48 better
#:
#: So full authority is measurably harmful and half is indistinguishable from
#: doing nothing.  This is the honest reading of that: it makes an undertrained
#: policy harmless rather than useful.  Whether it becomes useful is a question
#: about a training budget nobody has spent yet, not about this constant.
#:
#: Ten-body samples gave +25.4% for this same value on one archive and +1.3% on
#: another, which is why the number above is from 48.
INTENT_AUTHORITY = 0.5


@dataclass(eq=False)
class MobilityBasis:
    """The control axes a particular body actually has, in a particular medium.

    Attributes
    ----------
    modes : (r, P)
        Rows are CPG-parameter directions.  Commanding coefficient ``c`` means
        ``params = base + modes.T @ c``.
    effects : (r, 6)
        What each mode does to the body twist, as ``[vx, vy, vz, wx, wy, wz]``
        in the body frame, unit-normalised.
    authority : (r,)
        Singular values: how much twist per unit of parameter change.  A mode
        with near-zero authority is an axis this body does not have.
    medium : str
        Which medium the probe was run in.
    """

    modes: np.ndarray
    effects: np.ndarray
    authority: np.ndarray
    medium: str = "air"
    base: np.ndarray | None = None

    @property
    def rank(self) -> int:
        """Number of axes with meaningful authority.

        Thresholded relative to the strongest mode, so it answers "how many
        genuinely independent things can this machine do" rather than counting
        numerical noise.
        """
        if len(self.authority) == 0:
            return 0
        return int(np.sum(self.authority > 0.08 * self.authority[0]))

    def describe(self) -> list[str]:
        """Human-readable names for the discovered axes.

        Named after what they *do*, since they have no reason to line up with
        any conventional axis.  This is the text that appears in telemetry when
        the system reports what a machine learned to control.
        """
        labels = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
        out = []
        for i, e in enumerate(self.effects):
            order = np.argsort(-np.abs(e))
            parts = [f"{labels[j]}{'+' if e[j] > 0 else '-'}" for j in order[:2]
                     if abs(e[j]) > 0.25]
            desc = "&".join(parts) if parts else "weak"
            out.append(f"mode{i}[{desc}] sigma={self.authority[i]:.3g}")
        return out

    def command_params(self, base: CPGParams, coeffs: np.ndarray, n: int) -> CPGParams:
        """Turn intent coefficients into concrete CPG parameters."""
        c = np.asarray(coeffs, float)
        r = min(len(c), self.modes.shape[0])
        delta = self.modes[:r].T @ c[:r] if r > 0 else np.zeros(base.flat().shape)
        return CPGParams.from_flat(base.flat() + delta, n)

    # ------------------------------------------------------- the inverse model
    #
    # A mode index is a private coordinate.  Modes come out of an SVD, so they
    # are ordered by how much twist they produce and their sign is arbitrary,
    # and what mode 0 physically *is* depends entirely on the body: measured
    # across twelve elites from arch31, the mean pairwise cosine between their
    # water mode-0 directions is +0.094, and mode 0 is yaw on four of them,
    # heave on three, roll on two.
    #
    # That is fine for a controller that lives on one body, and fatal for one
    # shared across bodies: the same output means unrelated things on different
    # machines, so the gradients average toward nothing.  The fix is to let a
    # shared policy command in the units every body has in common -- a body
    # twist -- and to make each body's basis solve for the coefficients that
    # best deliver it.  That is what the basis was always for; the mode index
    # was never meant to be an interface.

    @property
    def _inverse(self):
        """``((r,6) damped inverse, (6,) reachable magnitude)``, cached.

        Lazy so that a basis unpickled from an older run still works.
        """
        cached = getattr(self, "_inv_cache", None)
        if cached is not None:
            return cached
        r = self.modes.shape[0] if self.modes.size else 0
        if r == 0:
            cached = (np.zeros((0, 6)), np.zeros(6))
        else:
            # Twist produced per unit of coefficient.  ``effects`` are unit
            # directions and ``authority`` is the gain, both in the scaled
            # twist space `basis_from_probes` fitted in -- rotations are
            # already weighted there so they are comparable with translation,
            # and that scaling is a constant, so it stays consistent across
            # bodies.
            A = self.effects * self.authority[:, None]          # (r, 6)
            G = A @ A.T                                          # (r, r)
            # Damping sized from the problem rather than as an absolute: it is
            # what turns "asking for an axis this body does not have" into a
            # small coefficient instead of an enormous one.  Measured cond(A)
            # is 13-50 across arch31 elites, so this is well inside the regime
            # where a modest ridge is enough.
            #
            # The absolute floor is not decoration.  A body that cannot move at
            # all has zero authority, so a purely proportional ridge is also
            # zero and the solve raises on a singular matrix -- found by drawing
            # 48 bodies at random from two archives rather than the dozen best,
            # which is where such a machine actually lives.
            lam = 0.01 * float(np.trace(G)) / max(r, 1) + 1e-12
            try:
                inv = np.linalg.solve(G + lam * np.eye(r), A)    # (r, 6)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(A.T)
            cached = (inv, np.linalg.norm(A, axis=0))
        self._inv_cache = cached
        return cached

    def coeffs_for_twist(self, intent: np.ndarray) -> np.ndarray:
        """Coefficients that best deliver a commanded body twist.

        ``intent`` is six numbers in [-1, 1] -- surge, sway, heave, roll,
        pitch, yaw -- read as a *fraction of what this body can do on that
        axis*.  Absolute units would not travel: the leading authority across
        arch31's elites spans 1.3 to 22.0 in air and 0.044 to 1.24 in water, so
        a fixed twist would ask one machine for a nudge and another for
        everything it has.  As a fraction, "+1 heave" means "climb as hard as
        this body climbs" on every one of them.

        A body has three or four controllable axes, not six, so the returned
        coefficients deliver the closest reachable twist rather than the one
        asked for.  That is the honest answer: a machine with no pitch
        authority should not be able to fake pitch, and the damping means it
        does not try.
        """
        inv, reach = self._inverse
        if inv.shape[0] == 0:
            return np.zeros(0)
        w = np.zeros(6)
        v = np.clip(np.asarray(intent, float).ravel(), -1.0, 1.0)
        w[: min(6, len(v))] = v[:6]
        return inv @ (w * reach * INTENT_AUTHORITY)

    def twist_of(self, coeffs: np.ndarray) -> np.ndarray:
        """The twist a coefficient vector produces.  The forward model."""
        c = np.asarray(coeffs, float)
        r = min(len(c), self.modes.shape[0])
        if r == 0:
            return np.zeros(6)
        A = self.effects[:r] * self.authority[:r, None]
        return A.T @ c[:r]


def identify_mobility(
    step_fn,
    reset_fn,
    n_params: int,
    *,
    n_probes: int = 24,
    probe_scale: float = 0.35,
    medium: str = "air",
    rng: np.random.Generator | None = None,
    max_modes: int = 6,
) -> MobilityBasis:
    """Empirically identify a body's control axes.

    Parameters
    ----------
    step_fn : callable(delta) -> (6,) array
        Runs one probe: applies a CPG parameter offset and returns the mean body
        twist it produced, in the body frame.
    reset_fn : callable()
        Restores the initial state between probes.
    n_probes :
        Number of random directions.  Must exceed the number of modes wanted;
        more probes give a better-conditioned Jacobian at linear cost.

    Notes
    -----
    The probes are centred (each direction is run with both signs and the
    responses differenced) so that any constant drift -- sinking, gliding down,
    being pushed by a current -- cancels out.  Without that, the first
    "discovered axis" of a negatively buoyant machine is always just gravity.
    """
    rng = rng or np.random.default_rng(0)
    deltas = rng.normal(0.0, probe_scale, size=(n_probes, n_params))
    responses = np.zeros((n_probes, 6))

    for k, d in enumerate(deltas):
        reset_fn()
        plus = np.asarray(step_fn(d), float)
        reset_fn()
        minus = np.asarray(step_fn(-d), float)
        # Central difference: cancels drift that is independent of the command.
        responses[k] = 0.5 * (plus - minus)

    return basis_from_probes(deltas, responses, medium=medium, max_modes=max_modes)


def basis_from_probes(deltas, responses, *, medium: str = "air",
                      max_modes: int = 6) -> MobilityBasis:
    """Fit the mobility basis from probe deltas and their measured responses.

    Split out of ``identify_mobility`` so that the probing and the fitting can
    be driven separately.  The probing is the expensive part -- 8 directions,
    both signs, 1.2 s of simulation each -- and it batches across machines,
    because a machine's probe delta is an *input* to the rollout and not a
    branch in it.  The fitting is a least-squares and an SVD on a (P, 6) matrix
    and is not worth moving anywhere.

    ``identify_mobility`` still exists and still drives one machine, because the
    unbatched evaluation path uses it and it is the reference the batched
    version is checked against.
    """
    deltas = np.asarray(deltas, float)
    responses = np.asarray(responses, float)

    # Scale twist components so that rotation and translation are comparable;
    # without this the SVD is dominated by whichever has the larger raw units.
    scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    Y = responses * scale

    # Least squares for J such that Y ~ deltas @ J.
    J, *_ = np.linalg.lstsq(deltas, Y, rcond=None)  # (n_params, 6)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)  # U:(P,k) S:(k,) Vt:(k,6)

    r = min(max_modes, len(S))
    modes = U[:, :r].T  # (r, P) parameter-space directions
    effects = Vt[:r]  # (r, 6) twist directions
    # Normalise the effect rows so they read as directions.
    norms = np.linalg.norm(effects, axis=1, keepdims=True)
    effects = effects / np.maximum(norms, 1e-12)
    return MobilityBasis(modes=modes, effects=effects, authority=S[:r], medium=medium)


@dataclass(eq=False)
class Policy:
    """Maps observations to intent coefficients in the mobility basis.

    A deliberately tiny network.  The heavy lifting is done by the CPG (which
    supplies the rhythm) and by the mobility basis (which supplies the
    coordination), so the policy only has to decide *how much of which axis* to
    ask for.

    ``hidden = 0`` makes it linear -- a single matrix from observations to
    intent -- and that is the default, because the one hidden layer that used to
    be the default was not affordable.  At ``hidden = 16`` this is 308 weights,
    and CMA-ES on 308 dimensions wants some thousands of evaluations; the
    default training budget was 180.  Measured on a body that flies, five
    iterations moved the population best from 0.365 to 0.349 -- which is not a
    controller failing to learn, it is an optimiser that has barely been asked a
    question.  Linear is 60 weights for the same problem, which the same budget
    can genuinely move, and the capacity is still there for anything that
    saturates it.
    """

    n_obs: int
    n_modes: int
    hidden: int = 0
    weights: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros(self.n_weights)

    @property
    def n_weights(self) -> int:
        if self.hidden <= 0:
            return self.n_obs * self.n_modes + self.n_modes
        return self.n_obs * self.hidden + self.hidden + self.hidden * self.n_modes + self.n_modes

    def act(self, obs: np.ndarray) -> np.ndarray:
        w = self.weights
        n_in, h, n_out = self.n_obs, self.hidden, self.n_modes
        x = np.asarray(obs, float)
        if h <= 0:
            W = w[: n_in * n_out].reshape(n_in, n_out)
            b = w[n_in * n_out : n_in * n_out + n_out]
            return np.tanh(x @ W + b)
        i = 0
        W1 = w[i : i + n_in * h].reshape(n_in, h); i += n_in * h
        b1 = w[i : i + h]; i += h
        W2 = w[i : i + h * n_out].reshape(h, n_out); i += h * n_out
        b2 = w[i : i + n_out]
        z = np.tanh(x @ W1 + b1)
        return np.tanh(z @ W2 + b2)
