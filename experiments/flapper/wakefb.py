"""A fore-to-hind wake, fed back, for a sensitivity read (2026-11-12).

The package's fluid model gives every wing strip the ambient flow plus one
uniform, lagged momentum inflow for the whole machine
(`physics.fluid.InducedFlow`).  What it leaves out between two wing pairs is
the fore pair's *shed* wake arriving at the hind pair: local, delayed by the
travel between them, and changing through the beat.  The package has that
wake as a picture (`physics.wake.VortexWake`: by Kelvin's theorem a strip
whose bound circulation changes sheds the opposite circulation from its
trailing edge) and never feeds it back.

This module feeds it back, onto the hind wings only and from the fore wings
only, times a gain.  It is not a wake model to be believed at gain 1: point
vortices from a quasi-steady circulation, no trailing (tip) vortices, a
regularised core.  It is there so that a verdict can be read at gain 0, 1 and
2 and called robust to an interaction of that order, or not.

    FLAPPER_WAKE_GAIN=1 ...   (read by `landwalk.make_env`)
"""
import math

import numpy as np

from dytiscidae.physics.fluid import WING

#: Sign of the feedback, fixed by the check in `wake_check.py`: with this sign
#: the vortex a wing sheds when its lift starts reduces that wing's own lift
#: (Wagner's effect), which is the one thing the sign has to get right.
SIGN = 1.0


class WakeFeedback:
    def __init__(self, env, gain: float, *, core_chords: float = 0.5, self_influence: bool = False,
                 max_particles: int = 6000, shed_chords: float = 0.1, keep_chords: float = 40.0):
        self.env, self.gain, self.self_influence = env, float(gain), bool(self_influence)
        s = env.solver
        s.record_state = True
        p = s.panels
        wing = np.asarray(p.kind) == WING
        self.n_panels = int(len(p.body_id))
        m, d = env.model, env.data
        # Fore and hind by where a wing body's joint sits along the hull.
        scratch = env._mj.MjData(m)
        env._mj.mj_forward(m, scratch)
        hull = env.root_body
        Rh = scratch.xmat[hull].reshape(3, 3)
        bodies = sorted(set(int(b) for b in np.asarray(p.body_id)[wing]))
        x_of = {}
        for b in bodies:
            root = b
            while int(m.body_parentid[root]) != hull and int(m.body_parentid[root]) != 0:
                root = int(m.body_parentid[root])
            x_of[b] = float((Rh.T @ (scratch.xpos[root] - scratch.xpos[hull]))[0])
        xs = sorted(set(round(v, 6) for v in x_of.values()))
        if len(xs) < 2 and not self_influence:
            self.fore = self.hind = np.zeros(self.n_panels, bool)
        else:
            cut = 0.5 * (xs[0] + xs[-1])
            fore_b = {b for b, v in x_of.items() if v > cut} if len(xs) > 1 else set(bodies)
            hind_b = {b for b, v in x_of.items() if v <= cut} if len(xs) > 1 else set(bodies)
            bid = np.asarray(p.body_id)
            self.fore = wing & np.isin(bid, list(fore_b))
            self.hind = wing & np.isin(bid, list(fore_b if self_influence else hind_b))
        chord = float(np.median(np.asarray(p.chord)[wing])) if wing.any() else 0.05
        self.chord = chord
        self.core = core_chords * chord
        self.max_particles = max_particles
        # A segment is shed each time the fore wings have travelled
        # `shed_chords` of a chord through the fluid, and kept until they have
        # travelled `keep_chords`: the same wake in air at 8 m/s and in water
        # at 0.3 m/s.  (Shedding every step and keeping a fixed count dropped
        # the starting vortex after 0.75 chords in the first check.)
        self.shed_chords, self.keep_chords = shed_chords, keep_chords
        self._travel = 0.0
        self._pending = None
        self._prev_pos = None
        self.born = np.zeros(0)
        self.path = 0.0
        self.pos = np.zeros((0, 3))
        self.vort = np.zeros((0, 3))
        self.age = np.zeros(0)
        self._prev_gamma = None
        self._flow = env.medium.flow_velocity
        self._step = env.step
        self._reset = env.reset
        self.last_induced = np.zeros((0, 3))
        env.medium.flow_velocity = self.flow
        env.step = self.step
        env.reset = self.reset
        env.wake_feedback = self

    # -- the environment's hooks
    def reset(self, *a, **k):
        self.pos, self.vort, self.age = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        self._prev_gamma = None
        self._travel, self._pending, self._prev_pos = 0.0, None, None
        self.born, self.path = np.zeros(0), 0.0
        return self._reset(*a, **k)

    def step(self, *a, **k):
        out = self._step(*a, **k)
        self._advance()
        return out

    def flow(self, pos, t: float = 0.0):
        u = self._flow(pos, t)
        pos = np.atleast_2d(pos)
        if self.gain == 0.0 or pos.shape[0] != self.n_panels or not len(self.pos):
            return u
        tgt = pos[self.hind]
        if not len(tgt):
            return u
        r = tgt[:, None, :] - self.pos[None, :, :]                    # from the vortex to the strip
        inv = 1.0 / (4.0 * math.pi * (np.sum(r * r, axis=2) + self.core ** 2) ** 1.5)
        ind = SIGN * np.einsum("ij,ijk->ik", inv, np.cross(self.vort[None, :, :], r))
        self.last_induced = ind
        u = np.array(u, float, copy=True)
        u[self.hind] += self.gain * ind
        return u

    # -- the wake
    def _advance(self) -> None:
        st = getattr(self.env.solver, "last_state", None)
        if st is None:
            return
        dt = float(self.env.timestep)
        gamma = np.asarray(st["gamma"], float)
        if self._prev_gamma is None or len(self._prev_gamma) != len(gamma):
            self._prev_gamma = gamma.copy()
        # Kelvin: what the bound circulation gains, the wake receives with the
        # other sign, as a spanwise segment at the trailing edge.
        strength = -(gamma - self._prev_gamma) * np.asarray(st["dr"], float)
        self._prev_gamma = gamma.copy()
        self._pending = strength if self._pending is None else self._pending + strength
        pos = np.asarray(st["pos"], float)
        if self._prev_pos is not None and self.fore.any():
            step = float(np.mean(np.linalg.norm(pos[self.fore] - self._prev_pos[self.fore], axis=1)))
            self._travel += step
            self.path += step
        self._prev_pos = pos.copy()
        if self._travel >= self.shed_chords * self.chord:
            self._travel = 0.0
            keep = self.fore & (np.abs(self._pending) > 1e-12)
            if keep.any():
                te = pos[keep] + 0.75 * st["chord"][keep, None] * st["chord_axis"][keep]
                self.pos = np.concatenate([self.pos, te])
                self.vort = np.concatenate([self.vort, self._pending[keep, None] * st["span_axis"][keep]])
                self.age = np.concatenate([self.age, np.zeros(len(te))])
                self.born = np.concatenate([self.born, np.full(len(te), self.path)])
            self._pending = None
        if not len(self.pos):
            return
        # Carried by the ambient flow (the fluid is still, the machine moves).
        self.pos = self.pos + self._flow(self.pos, float(self.env.data.time)) * dt
        self.age = self.age + dt
        alive = (self.path - self.born) <= self.keep_chords * self.chord
        if len(self.pos) - int(alive.sum()) or len(self.pos) > self.max_particles:
            if len(self.pos) > self.max_particles:
                alive[: len(self.pos) - self.max_particles] = False
            self.pos, self.vort, self.age, self.born = (self.pos[alive], self.vort[alive],
                                                        self.age[alive], self.born[alive])


def attach(env, gain: float, **kw) -> WakeFeedback:
    return WakeFeedback(env, gain, **kw)
