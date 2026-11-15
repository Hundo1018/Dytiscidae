"""The added-mass load of a flapping plate from its own motion, as a reference.

The solver puts a wing strip's added mass into MuJoCo's mass matrix as a
scalar (the tensor projected on the direction of motion) and reacts its growth
with `entrainment_reaction`.  A rig that overwrites the wing's state never
feels the matrix, so its torque leaves the water's inertia out.  This computes
what strip potential flow gives for a thin plate, per strip of width dr:

    F   = -m_n d(u . n)/dt n     m_n = rho pi c^2 / 4 dr,  u at mid-chord,
                                 relative to the stream
    tau = -I_a d(w . s)/dt s     I_a = rho pi c^4 / 128 dr, about mid-chord

by a backward difference along the motion.  ``d(u . n)/dt`` is the rate of the
plate-frame normal velocity, so the turning of n is in it.  The force is
normal to the plate and acts at mid-chord: this is the added-mass term of the
quasi-steady flapping models (Sane & Dickinson 2002; Wang, Goosen & van Keulen
2016).  Left out on purpose, both being potential-flow loads a separated plate
does not carry: the chordwise force ``-m_n (u . n) dn/dt`` (leading-edge
suction) and the Munk moment ``u x P``, which for attached flow is the lift
acting at the quarter chord and is already in the solver's circulatory load.
In still water the load does no net work over a periodic motion.
"""
import numpy as np


class AddedMassRef:
    def __init__(self, solver, mj) -> None:
        from dytiscidae.physics.fluid import WING
        self.s, self.mj = solver, mj
        self.w = np.flatnonzero(solver.panels.kind == WING)
        self.prev = None

    def add(self, d, t: float) -> None:
        """Adds the wrench to ``d.xfrc_applied`` for the state the solver last
        saw (``d.xpos`` and ``d.cvel`` are those of the step just taken)."""
        s, mj, p, w = self.s, self.mj, self.s.panels, self.w
        m = s.model
        b = p.body_id[w]
        R = d.xmat.reshape(-1, 3, 3)[b]
        s_hat = np.einsum("nij,nj->ni", R, p.span_local[w])
        c_hat = np.einsum("nij,nj->ni", R, p.chord_local[w])
        n_hat = np.einsum("nij,nj->ni", R, p.normal_local[w])
        pos = d.xpos[b] + np.einsum("nij,nj->ni", R, p.pos_local[w])
        mid = pos + 0.25 * p.chord[w][:, None] * c_hat
        vel, v6 = np.zeros((m.nbody, 6)), np.zeros(6)
        for bb in np.unique(b):
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, int(bb), v6, 0)
            vel[bb] = v6
        om = vel[b, :3]
        u = vel[b, 3:] + np.cross(om, mid - d.xipos[b]) - s.medium.flow_velocity(mid, t)
        rho = s.medium.properties(pos, p.half_height[w], t)[0]
        m_n = rho * np.pi * 0.25 * p.chord[w] ** 2 * p.dr[w]
        i_a = rho * np.pi / 128.0 * p.chord[w] ** 4 * p.dr[w]
        vn = np.einsum("ni,ni->n", u, n_hat)
        ws = np.einsum("ni,ni->n", om, s_hat)
        if self.prev is not None:
            vn0, ws0, t0 = self.prev
            h = max(t - t0, 1e-9)
            F = (-m_n * (vn - vn0) / h)[:, None] * n_hat
            tau = (-i_a * (ws - ws0) / h)[:, None] * s_hat
            np.add.at(d.xfrc_applied[:, :3], b, F)
            np.add.at(d.xfrc_applied[:, 3:], b, np.cross(mid - d.xipos[b], F) + tau)
        self.prev = (vn, ws, t)
