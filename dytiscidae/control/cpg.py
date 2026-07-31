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

    def reset(self) -> None:
        self.t = 0.0

    def command(self, params: CPGParams, t: float) -> np.ndarray:
        """Target joint angles at time ``t``."""
        p = params.clipped(self.lo, self.hi)
        return p.offset + p.amplitude * np.sin(2.0 * np.pi * p.frequency * t + p.phase)

    @property
    def n_params(self) -> int:
        return 3 * self.n + 1


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


def identify_mobility(
    step_fn,
    reset_fn,
    n_params: int,
    *,
    n_probes: int = 10,
    probe_scale: float = 0.35,
    medium: str = "air",
    rng: np.random.Generator | None = None,
    max_modes: int = 4,
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
