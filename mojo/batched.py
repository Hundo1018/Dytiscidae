"""Step N machines in lockstep so their panels share one GPU launch.

This is the piece that makes the port pay. A single machine has 48-134 panels,
and at that size the GPU loses to numpy (0.80x at 70 panels, measured). The
arithmetic only becomes worth a launch across candidates: break-even is ~280
panels, about four machines, and 64 machines run 2.9x.

The premise is that panels from *different morphologies* concatenate. They do,
and it is not obvious: MJX cannot batch these machines because vmap needs one
shared kinematic tree and an archive of 102 elites holds 52 distinct ones. But
the fluid solver never sees a tree. It sees a flat panel array indexed by
`body_id`, so panels from unlike machines pack end to end behind an offset
table, exactly like a ragged batch. `mojo/tests/test_fluid_gpu.py` pins that
against numpy on seven morphologies in one 554-panel launch.

What stays per-machine and on the CPU: MuJoCo itself, the `mj_objectVelocity`
loop, and the `model.body_mass` write-back. Those are 2-3% of the step.
"""
from __future__ import annotations

import numpy as np

import fluid_gpu


class BatchedPanels:
    """Concatenated panel state for a group of machines.

    Body indices are rebased as machines are appended, so one `body_id` array
    addresses every body in the batch and the atomic scatter in the added-mass
    kernel lands in the right place without knowing which machine a panel came
    from.
    """

    def __init__(self, envs):
        self.envs = list(envs)
        self.n_panels = [len(e.solver.panels.body_id) for e in self.envs]
        self.n_bodies = [e.model.nbody for e in self.envs]
        self.panel_off = np.cumsum([0] + self.n_panels)
        self.body_off = np.cumsum([0] + self.n_bodies)
        self.total_panels = int(self.panel_off[-1])
        self.total_bodies = int(self.body_off[-1])

        C = np.ascontiguousarray
        cat = lambda parts: C(np.concatenate(parts))
        P = [e.solver.panels for e in self.envs]

        # Static per-panel geometry: built once, never changes during a rollout.
        self.body_id = C(np.concatenate(
            [np.asarray(p.body_id, np.int32) + self.body_off[i]
             for i, p in enumerate(P)]).astype(np.int32))
        self.pos_local = cat([np.asarray(p.pos_local, float) for p in P])
        self.span_local = cat([np.asarray(p.span_local, float) for p in P])
        self.chord_local = cat([np.asarray(p.chord_local, float) for p in P])
        self.normal_local = cat([np.asarray(p.normal_local, float) for p in P])
        self.chord = cat([np.asarray(p.chord, float) for p in P])
        self.dr = cat([np.asarray(p.dr, float) for p in P])
        self.volume = cat([np.asarray(p.volume, float) for p in P])
        self.camber = cat([np.asarray(p.camber, float) for p in P])
        self.ext_local = cat([np.asarray(p.ext_local, float) for p in P])
        self.cd_bluff = cat([np.asarray(p.cd_bluff, float) for p in P])
        self.aspect_ratio = cat([np.asarray(p.aspect_ratio, float) for p in P])
        # PanelSet stores kind with WING == 0; the kernels take is_wing as a
        # flag, so this is inverted here rather than in every kernel.
        self.is_wing = C(np.concatenate(
            [(np.asarray(p.kind) == 0).astype(np.int32) for p in P]))

        n, nb = self.total_panels, self.total_bodies
        self.xpos = np.zeros((nb, 3))
        self.xmat = np.zeros((nb, 9))
        self.pos_w = np.zeros((n, 3))
        self.s_hat = np.zeros((n, 3))
        self.c_hat = np.zeros((n, 3))
        self.n_hat = np.zeros((n, 3))
        self.d_full = np.zeros((n, 3))
        self.F_bluff = np.zeros((n, 3))
        self.D_bluff = np.zeros(n)
        self.m_add = np.zeros(n)
        self.vn = np.zeros(n)
        self.m_body = np.zeros(nb)
        self.fz = np.zeros(n)
        self.U = np.zeros(n)
        self.d_hat = np.zeros((n, 3))
        self.q = np.zeros(n)
        self.re = np.zeros(n)
        self.alpha = np.zeros(n)
        self.rf = np.zeros(n)
        self.lift_axis = np.zeros((n, 3))
        self.cl = np.zeros(n)
        self.cd = np.zeros(n)

    def gather_kinematics(self):
        """Pull each machine's body poses into the shared arrays."""
        for i, e in enumerate(self.envs):
            a, b = self.body_off[i], self.body_off[i + 1]
            self.xpos[a:b] = e.data.xpos
            self.xmat[a:b] = e.data.xmat.reshape(-1, 9)

    def _desc(self, arrays, tail):
        return np.array([a.ctypes.data for a in arrays] + list(tail),
                        dtype=np.int64)

    def body_to_world(self):
        fluid_gpu.body_to_world(self._desc(
            [self.xpos, self.xmat, self.body_id, self.pos_local,
             self.span_local, self.chord_local, self.normal_local,
             self.pos_w, self.s_hat, self.c_hat, self.n_hat],
            [self.total_panels, self.total_bodies]))

    def strip_theory(self, v_rel, omega, rho, mu):
        fluid_gpu.strip_theory(self._desc(
            [v_rel, self.s_hat, self.c_hat, self.n_hat, omega, rho, mu,
             self.chord, self.camber,
             self.U, self.d_hat, self.q, self.re, self.alpha, self.rf,
             self.lift_axis],
            [self.total_panels]))

    def coefficients(self):
        fluid_gpu.coefficients(self._desc(
            [self.alpha, self.re, self.aspect_ratio, self.rf, self.is_wing,
             self.cl, self.cd],
            [self.total_panels]))

    def bluff_drag(self, v_rel, rho, mu, cd_scale=1.0):
        fluid_gpu.bluff_drag(self._desc(
            [v_rel, self.s_hat, self.c_hat, self.n_hat, rho, mu,
             self.ext_local, self.cd_bluff, self.is_wing,
             self.F_bluff, self.D_bluff, self.d_full],
            [self.total_panels]), cd_scale)

    def added_mass(self, v_rel, rho, scale=1.0):
        has_bluff = int((self.is_wing == 0).any())
        fluid_gpu.added_mass(self._desc(
            [v_rel, self.s_hat, self.c_hat, self.n_hat, self.d_full, rho,
             self.ext_local, self.chord, self.dr, self.volume, self.is_wing,
             self.body_id,
             self.m_add, self.vn, self.m_body, self.fz],
            [self.total_panels, self.total_bodies]), scale, has_bluff)

    def run(self, v_rel, omega, rho, mu, cd_scale=1.0, am_scale=1.0):
        """The whole ported pipeline, one launch per stage for the batch."""
        self.gather_kinematics()
        self.body_to_world()
        self.strip_theory(v_rel, omega, rho, mu)
        self.coefficients()
        self.bluff_drag(v_rel, rho, mu, cd_scale)
        self.added_mass(v_rel, rho, am_scale)

    def slice_of(self, i):
        """Panel range belonging to machine `i`."""
        return slice(int(self.panel_off[i]), int(self.panel_off[i + 1]))
