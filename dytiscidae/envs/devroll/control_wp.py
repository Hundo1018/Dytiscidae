"""The controller at a decision step and the CPG every step, on the device (IMPL_1010 A3).

What ``batchroll.rollout_batch`` does around ``step_batch`` (batchroll.py:840-920),
per world of one ``model_wp.DeviceBody``, as Warp kernels:

* the sensing ``TriphibianEnv.observation`` reads (triphibian.py:1602-1680): the
  root body's twist exactly as ``body_twist`` (``mj_objectVelocity`` at the body's
  centre of mass, rotated into its *inertial* frame ``ximat``), the body-frame
  gravity (``xmat``), the joint encoders, ground contact from MJWarp's contact pool,
  and ``clearance`` (the lowest colliding geom above the sea or the beach ramp);
* the 33 observation channels, and ``observation_finite``: a world whose
  observation is not finite at a decision fails as ``diverged`` and stops deciding;
* the shared actor (``SharedPolicy.latent``: the running normalisation, the
  tanh MLP, then the squash ``tanh``).  A world flagged ``sampled`` adds
  ``exp(log_std) * eps`` with ``eps`` from ``wp.rand_init(seed, decision)``; the
  others take the mean.  Each decision's observation and action are recorded so
  the host can recompute ``log_prob``, the value and the potential in one batch;
* the per-design ``control.cpg.Policy`` (linear or one hidden layer), summed into
  the mode coefficients with ``MobilityBasis.coeffs_for_twist`` of the shared
  action, the gait gain (``gait_gain``) and ``MobilityBasis.command_params``, and
  ``twist_of`` the coefficients (the commanded twist the air score reads);
* ``CPG.command`` every step: ``CPGParams.clipped``, the phase continuity term
  ``_psi``, the target and its rate into the body's ``targets`` and ``rate``,
  which ``DeviceBody.fluid_and_scatter`` turns into ``ctrl`` with A1b's
  ``scatter_wp`` servo kernel (``servo_command`` is reused, not ported again).

Host state kept explicit: whether the CPG left a rate for the servo
(``CPG.pop_rate``) is the body's per-world ``has_rate`` flag, set by the host at
:meth:`DeviceControl.arm` (1 where the CPG drives every actuator, which is every
rollout step).  The task schedule, the bases and the policies are uploaded per
segment by the host.  The clock, the CPG phase state (``_psi``, the last
frequency and time) and the battery read are float64 at every precision: an
8 s clock in float32 resolves 0.5 us, and ``2 pi f t`` at 20 Hz then carries
6e-5 rad of rounding into every angle.

The segment around this -- resetting single worlds, freezing a world that
failed, the per-step record, the divergence and contact-overflow checks -- is
``rollout_wp`` (A4a).  ``arm`` takes one domain per world, so the air and water
worlds of one body share a launch.  ``records()`` refuses a world that made
more decisions than ``record`` holds instead of returning the first ``record``
of them.
"""

# No ``from __future__ import annotations``: Warp reads the kernel annotations.
import math as _math

import numpy as _np

import warp as _wp

from dytiscidae.control.cpg import GAIN_RANGE, INTENT_AUTHORITY, TWIST_DIM

#: Observation width, ``TriphibianEnv.OBS_DIM`` (25 + MORPHOLOGY_DIM).
OBS = 33
#: Phases per segment the device task table holds (``schedule_for`` makes 2).
MAX_PHASES = 4

_KERNELS: dict = {}
#: 2 pi in float64: ``wp.pi`` is a float32 constant (2.7e-8 relative), which put
#: 2e-7 into every float64 angle and 3e-9 into the stroke rate.
_TWO_PI = _wp.constant(_wp.float64(2.0 * _math.pi))


def make_kernels(T):
    """The A3 kernels for precision ``T``, cached."""
    if T in _KERNELS:
        return _KERNELS[T]
    V = _wp.vec3d if T == _wp.float64 else _wp.vec3
    M = _wp.mat33d if T == _wp.float64 else _wp.mat33
    F64 = _wp.float64
    f32 = _wp.float32

    @_wp.func
    def v3(x: _wp.vec3) -> V:
        return V(T(x[0]), T(x[1]), T(x[2]))

    @_wp.func
    def m3(x: _wp.mat33) -> M:
        return M(T(x[0, 0]), T(x[0, 1]), T(x[0, 2]), T(x[1, 0]), T(x[1, 1]), T(x[1, 2]),
                 T(x[2, 0]), T(x[2, 1]), T(x[2, 2]))

    @_wp.func
    def clip(x: T, lo: T, hi: T) -> T:
        return _wp.min(_wp.max(x, lo), hi)

    # --- sensing from MJWarp -----------------------------------------------

    @_wp.kernel
    def sense_root(
        root: int, root_root: int, xmat_d: _wp.array2d(dtype=_wp.mat33),
        ximat_d: _wp.array2d(dtype=_wp.mat33), xpos_d: _wp.array2d(dtype=_wp.vec3),
        xipos_d: _wp.array2d(dtype=_wp.vec3), com_d: _wp.array2d(dtype=_wp.vec3),
        cvel_d: _wp.array2d(dtype=_wp.spatial_vector),
        R: _wp.array(dtype=M), Ri: _wp.array(dtype=M), pos: _wp.array(dtype=V),
        xipos: _wp.array(dtype=V), com: _wp.array(dtype=V), ang: _wp.array(dtype=V),
        lin: _wp.array(dtype=V),
    ):
        # The root body's frame, inertial frame, position and cvel (about its
        # root's subtree_com), as the host reads them from MjData.
        w = _wp.tid()
        R[w] = m3(xmat_d[w, root])
        Ri[w] = m3(ximat_d[w, root])
        pos[w] = v3(xpos_d[w, root])
        xipos[w] = v3(xipos_d[w, root])
        com[w] = v3(com_d[w, root_root])
        c = cvel_d[w, root]
        ang[w] = V(T(c[0]), T(c[1]), T(c[2]))
        lin[w] = V(T(c[3]), T(c[4]), T(c[5]))

    @_wp.kernel
    def sense_joints(
        nact: int, qadr: _wp.array(dtype=int), vadr: _wp.array(dtype=int),
        qpos_d: _wp.array2d(dtype=f32), qvel_d: _wp.array2d(dtype=f32),
        qa: _wp.array(dtype=T), va: _wp.array(dtype=T),
    ):
        i = _wp.tid()
        w = i // nact
        j = i - w * nact
        qa[i] = T(qpos_d[w, qadr[j]])
        va[i] = T(qvel_d[w, vadr[j]])

    @_wp.kernel
    def sense_geoms(
        ng: int, gid: _wp.array(dtype=int), gx_d: _wp.array2d(dtype=_wp.vec3),
        gR_d: _wp.array2d(dtype=_wp.mat33), gx: _wp.array(dtype=V), gR: _wp.array(dtype=M),
    ):
        i = _wp.tid()
        w = i // ng
        g = gid[i - w * ng]
        gx[i] = v3(gx_d[w, g])
        gR[i] = m3(gR_d[w, g])

    @_wp.kernel
    def touch_clear(touching: _wp.array(dtype=int)):
        touching[_wp.tid()] = 0

    @_wp.kernel
    def touch_ground(
        nacon: _wp.array(dtype=int), cgeom: _wp.array(dtype=_wp.vec2i),
        cworld: _wp.array(dtype=int), geom_bodyid: _wp.array(dtype=int),
        touching: _wp.array(dtype=int),
    ):
        # TriphibianEnv._touching_ground: any contact against body 0 (the
        # terrain), not the machine against itself.  MJWarp's contacts are one
        # pool over all worlds, tagged with their world.
        i = _wp.tid()
        if i >= nacon[0]:
            return
        g = cgeom[i]
        hit = int(0)
        if g[0] >= 0:
            if geom_bodyid[g[0]] == 0:
                hit = 1
        if g[1] >= 0:
            if geom_bodyid[g[1]] == 0:
                hit = 1
        if hit != 0:
            touching[cworld[i]] = 1

    @_wp.kernel
    def clearance_wp(
        ng: int, gx: _wp.array(dtype=V), gR: _wp.array(dtype=M),
        centre: _wp.array(dtype=V), half: _wp.array(dtype=V),
        beach_lo: T, beach_hi: T, shore_x: T, slope: T, shore_z: T,
        clear: _wp.array(dtype=T),
    ):
        # TriphibianEnv._clearance_now on the flat calm sea: per colliding geom
        # its AABB's lowest point, above the higher of the sea surface (z = 0)
        # and the beach ramp where the ramp exists; the minimum over geoms.
        w = _wp.tid()
        best = T(1.0e30)
        for k in range(ng):
            i = w * ng + k
            R = gR[i]
            row = V(R[2, 0], R[2, 1], R[2, 2])
            cz = gx[i][2] + _wp.dot(row, centre[k])
            drop = _wp.abs(row[0]) * half[k][0] + _wp.abs(row[1]) * half[k][1] \
                + _wp.abs(row[2]) * half[k][2]
            x = gx[i][0]
            ground = T(0.0)
            if x >= beach_lo and x <= beach_hi:
                ground = _wp.max(ground, (x - shore_x) * slope + shore_z)
            best = _wp.min(best, cz - drop - ground)
        clear[w] = best

    # --- the decision cadence -----------------------------------------------

    @_wp.kernel
    def due_wp(
        every: int, count: _wp.array(dtype=int), active: _wp.array(dtype=int),
        power_active: _wp.array(dtype=int), has_ctrl: _wp.array(dtype=int),
        due: _wp.array(dtype=int),
    ):
        w = _wp.tid()
        due[w] = 0
        if active[w] != 0 and power_active[w] != 0 and has_ctrl[w] != 0:
            if count[w] % every == 0:
                due[w] = 1

    # --- the observation ----------------------------------------------------

    @_wp.kernel
    def observe(
        nact: int, R: _wp.array(dtype=M), Ri: _wp.array(dtype=M), pos: _wp.array(dtype=V),
        xipos: _wp.array(dtype=V), com: _wp.array(dtype=V), ang: _wp.array(dtype=V),
        lin: _wp.array(dtype=V), msub: _wp.array(dtype=T), touching: _wp.array(dtype=int),
        energy: _wp.array(dtype=F64), capacity: _wp.array(dtype=F64),
        qa: _wp.array(dtype=T), va: _wp.array(dtype=T), mid: _wp.array(dtype=T),
        half: _wp.array(dtype=T), base_f: _wp.array(dtype=T), cmd: _wp.array(dtype=int),
        t: _wp.array(dtype=F64), armed: _wp.array(dtype=int), nph: _wp.array(dtype=int),
        ph_start: _wp.array(dtype=F64), ph_cruise: _wp.array(dtype=int),
        ph_speed: _wp.array(dtype=T), ph_heading: _wp.array(dtype=T),
        ph_depth: _wp.array(dtype=T), seg_t0: _wp.array(dtype=F64),
        seg_T: _wp.array(dtype=F64), is_air: _wp.array(dtype=int),
        height_ref: _wp.array(dtype=T), clear: _wp.array(dtype=T),
        default_depth: T, morph: _wp.array(dtype=T),
        due: _wp.array(dtype=int), active: _wp.array(dtype=int),
        diverged: _wp.array(dtype=int), obs: _wp.array(dtype=T),
    ):
        # TriphibianEnv.observation (with task_channels), then observation_finite.
        w = _wp.tid()
        if due[w] == 0:
            return
        o = w * 33
        Rw = R[w]
        Rt = _wp.transpose(Ri[w])
        a = ang[w]
        # body_twist: mj_objectVelocity(mjOBJ_BODY, flg_local=1): the velocity at
        # xipos (shifted from the root's subtree_com), in the inertial frame.
        lc = lin[w] - _wp.cross(xipos[w] - com[w], a)
        tl = Rt * lc
        ta = Rt * a
        for k in range(3):
            obs[o + k] = clip(tl[k] / T(5.0), T(-3.0), T(3.0))
            obs[o + 3 + k] = clip(ta[k] / T(4.0), T(-3.0), T(3.0))
            # R^T (0, 0, -1): the body-frame gravity direction.
            obs[o + 6 + k] = -Rw[2, k]
        d = -pos[w][2]                       # the flat calm sea: depth = -z
        obs[o + 9] = _wp.tanh(d / T(5.0))
        obs[o + 10] = msub[w]
        for k in range(3):
            obs[o + 11 + k] = T(0.0)
        if cmd[w] >= 0:
            obs[o + 11 + cmd[w]] = T(1.0)
        # The phase in force: TaskSchedule.index_at on the float64 clock.
        k_ph = int(-1)
        if armed[w] != 0:
            frac = (t[w] - seg_t0[w]) / seg_T[w]
            k_ph = int(0)
            for p in range(nph[w]):
                if frac >= ph_start[w * 4 + p]:
                    k_ph = p
        depth_target = default_depth
        if k_ph >= 0:
            depth_target = ph_depth[w * 4 + k_ph]
        obs[o + 14] = _wp.tanh((d - depth_target) / T(3.0))
        obs[o + 15] = T(0.0)
        if touching[w] != 0:
            obs[o + 15] = T(1.0)
        obs[o + 16] = T(energy[w] / _wp.max(capacity[w], F64(1e-9)))
        stroke = T(0.0)
        srate = T(0.0)
        if nact > 0:
            sq = T(0.0)
            sv = T(0.0)
            for j in range(nact):
                i = w * nact + j
                sq = sq + (qa[i] - mid[i]) / half[i]
                sv = sv + va[i] / half[i]
            stroke = clip(sq / T(nact), T(-1.0), T(1.0))
            om = T(_TWO_PI) * _wp.max(base_f[w], T(0.1))
            srate = clip((sv / T(nact)) / om, T(-3.0), T(3.0))
        obs[o + 17] = stroke
        obs[o + 18] = srate
        # task_channels: [cruise, hold, speed, cos(err), sin(err), height lost].
        for k in range(6):
            obs[o + 19 + k] = T(0.0)
        if k_ph >= 0:
            q = w * 4 + k_ph
            if ph_cruise[q] != 0:
                obs[o + 19] = T(1.0)
                obs[o + 21] = ph_speed[q]
                yaw = _wp.atan2(Rw[1, 0], Rw[0, 0])
                err = ph_heading[q] - yaw
                obs[o + 22] = _wp.cos(err)
                obs[o + 23] = _wp.sin(err)
            else:
                obs[o + 20] = T(1.0)
            if is_air[w] != 0:
                obs[o + 24] = _wp.tanh((height_ref[w] - clear[w]) / T(3.0))
        for k in range(8):
            obs[o + 25 + k] = morph[w * 8 + k]
        # observation_finite: a non-finite observation fails the world as diverged.
        ok = int(1)
        for k in range(33):
            if not _wp.isfinite(obs[o + k]):
                ok = 0
        if ok == 0:
            diverged[w] = 1
            active[w] = 0
            due[w] = 0

    # --- the shared actor ---------------------------------------------------

    @_wp.kernel
    def actor_in(
        H: int, obs: _wp.array(dtype=T), mean: _wp.array(dtype=T), var: _wp.array(dtype=T),
        W1: _wp.array(dtype=T), b1: _wp.array(dtype=T), due: _wp.array(dtype=int),
        h1: _wp.array(dtype=T),
    ):
        # SharedPolicy.normalise, then the first Linear and Tanh.  The input is
        # the observation rounded to float32, as rollout_batch hands it to
        # act_many (np.asarray(obs_rows, np.float32)) at every precision.
        i = _wp.tid()
        w = i // H
        j = i - w * H
        if due[w] == 0:
            return
        s = b1[j]
        for k in range(33):
            x = clip((T(f32(obs[w * 33 + k])) - mean[k]) / _wp.sqrt(var[k] + T(1e-8)),
                     T(-10.0), T(10.0))
            s = s + W1[j * 33 + k] * x
        h1[i] = _wp.tanh(s)

    @_wp.kernel
    def actor_mid(
        H: int, h1: _wp.array(dtype=T), W2: _wp.array(dtype=T), b2: _wp.array(dtype=T),
        due: _wp.array(dtype=int), h2: _wp.array(dtype=T),
    ):
        i = _wp.tid()
        w = i // H
        j = i - w * H
        if due[w] == 0:
            return
        s = b2[j]
        for k in range(H):
            s = s + W2[j * H + k] * h1[w * H + k]
        h2[i] = _wp.tanh(s)

    @_wp.kernel
    def actor_out(
        H: int, NO: int, h2: _wp.array(dtype=T), W3: _wp.array(dtype=T),
        b3: _wp.array(dtype=T), log_std: _wp.array(dtype=T), sampled: _wp.array(dtype=int),
        seed: _wp.array(dtype=int), ndec: _wp.array(dtype=int), due: _wp.array(dtype=int),
        u_out: _wp.array(dtype=T), act: _wp.array(dtype=T),
    ):
        # The latent mean; a sampled world adds std * eps, then the squash.
        i = _wp.tid()
        w = i // NO
        j = i - w * NO
        if due[w] == 0:
            return
        s = b3[j]
        for k in range(H):
            s = s + W3[j * H + k] * h2[w * H + k]
        u = s
        if sampled[w] != 0:
            st = _wp.rand_init(seed[w], ndec[w] * NO + j)
            u = s + _wp.exp(log_std[j]) * T(_wp.randn(st))
        u_out[i] = u
        act[i] = _wp.tanh(u)

    # --- the decision -------------------------------------------------------

    @_wp.kernel
    def decide(
        P: int, n: int, rmax: int, NO: int, hmax: int, cap: int,
        obs: _wp.array(dtype=T), act: _wp.array(dtype=T), use_shared: int,
        has_policy: _wp.array(dtype=int), pw: _wp.array(dtype=T), pw_off: _wp.array(dtype=int),
        p_hidden: _wp.array(dtype=int), p_out: _wp.array(dtype=int),
        p_modes: _wp.array(dtype=int), r: _wp.array(dtype=int),
        modes: _wp.array(dtype=T), inv: _wp.array(dtype=T), reach: _wp.array(dtype=T),
        A: _wp.array(dtype=T), base: _wp.array(dtype=T), gain_lo: T, gain_hi: T,
        authority: T, due: _wp.array(dtype=int), ndec: _wp.array(dtype=int),
        z: _wp.array(dtype=T), c: _wp.array(dtype=T),
        cur: _wp.array(dtype=T), cmd_twist: _wp.array(dtype=T), gain_out: _wp.array(dtype=T),
        has_gain: _wp.array(dtype=int), rec_obs: _wp.array(dtype=T),
        rec_act: _wp.array(dtype=T), rec_u: _wp.array(dtype=T), u: _wp.array(dtype=T),
    ):
        # rollout_batch's decision for one world: the design's Policy.act, its
        # coefficients and gain intent, MobilityBasis.coeffs_for_twist of the
        # shared action and its gain intent, gait_gain, command_params, twist_of.
        w = _wp.tid()
        if due[w] == 0:
            return
        rw = r[w]
        for k in range(rw):
            c[w * rmax + k] = T(0.0)
        gain = T(0.0)
        hg = int(0)
        ob = w * 33
        if has_policy[w] != 0:
            off = pw_off[w]
            h = p_hidden[w]
            no = p_out[w]
            nm = p_modes[w]
            nwide = _wp.min(_wp.min(no, rw), nm)
            for k in range(no):
                s = T(0.0)
                if h <= 0:
                    s = pw[off + 33 * no + k]
                    for i in range(33):
                        s = s + obs[ob + i] * pw[off + i * no + k]
                else:
                    if k == 0:
                        for j in range(h):
                            zz = pw[off + 33 * h + j]
                            for i in range(33):
                                zz = zz + obs[ob + i] * pw[off + i * h + j]
                            z[w * hmax + j] = _wp.tanh(zz)
                    o2 = off + 33 * h + h
                    s = pw[o2 + h * no + k]
                    for j in range(h):
                        s = s + z[w * hmax + j] * pw[o2 + j * no + k]
                val = _wp.tanh(s)
                if k < nwide:
                    c[w * rmax + k] = c[w * rmax + k] + val
                if k == nm:
                    gain = val
                    hg = 1
        if use_shared != 0:
            if NO > 6:
                gain = gain + act[w * NO + 6]
                hg = 1
            for k in range(rw):
                s = T(0.0)
                for j in range(6):
                    a = clip(act[w * NO + j], T(-1.0), T(1.0))
                    s = s + inv[(w * rmax + k) * 6 + j] * (a * reach[w * 6 + j] * authority)
                c[w * rmax + k] = c[w * rmax + k] + s
        g = T(1.0)
        if hg != 0:
            g = clip(T(1.0) + gain, gain_lo, gain_hi)
        gain_out[w] = g
        has_gain[w] = hg
        for q in range(P):
            v = base[w * P + q]
            for k in range(rw):
                v = v + modes[(w * rmax + k) * P + q] * c[w * rmax + k]
            if q < n and g != T(1.0):
                v = v * g
            cur[w * P + q] = v
        for j in range(6):
            s = T(0.0)
            for k in range(rw):
                s = s + A[(w * rmax + k) * 6 + j] * c[w * rmax + k]
            cmd_twist[w * 6 + j] = s
        nd = ndec[w]
        if nd < cap:
            for k in range(33):
                rec_obs[(w * cap + nd) * 33 + k] = obs[ob + k]
            for k in range(NO):
                rec_act[(w * cap + nd) * NO + k] = act[w * NO + k]
                rec_u[(w * cap + nd) * NO + k] = u[w * NO + k]
        ndec[w] = nd + 1

    # --- the CPG ------------------------------------------------------------

    @_wp.kernel
    def cpg_wp(
        n: int, P: int, cur: _wp.array(dtype=T), lo: _wp.array(dtype=T),
        hi: _wp.array(dtype=T), margin: _wp.array(dtype=T), phase_offset: _wp.array(dtype=F64),
        t: _wp.array(dtype=F64), active: _wp.array(dtype=int),
        power_active: _wp.array(dtype=int), psi: _wp.array(dtype=F64),
        f_last: _wp.array(dtype=F64), t_last: _wp.array(dtype=F64),
        has_f: _wp.array(dtype=int), targets: _wp.array(dtype=T), rate: _wp.array(dtype=T),
    ):
        # CPG.command for one world: CPGParams.clipped, the phase continuity, the
        # target angles and their rate.  The phase is float64 at every precision.
        w = _wp.tid()
        if active[w] == 0 or power_active[w] == 0:
            return
        f = F64(cur[w * P + 3 * n])
        f = _wp.min(_wp.max(f, F64(0.1)), F64(20.0))
        tw = t[w]
        if has_f[w] == 0 or tw < t_last[w]:
            psi[w] = F64(0.0)
        elif f != f_last[w]:
            psi[w] = psi[w] + _TWO_PI * (f_last[w] - f) * tw
        f_last[w] = f
        t_last[w] = tw
        has_f[w] = 1
        two_pi_f = _TWO_PI * f
        for j in range(n):
            i = w * n + j
            l = F64(lo[i])
            h = F64(hi[i])
            mg = F64(margin[i])
            span = F64(0.5) * (h - l)
            mid = F64(0.5) * (h + l)
            off = _wp.min(_wp.max(F64(cur[w * P + 2 * n + j]), l + mg * span), h - mg * span)
            amp = _wp.min(_wp.max(F64(cur[w * P + j]), F64(0.0)),
                          _wp.max(span - _wp.abs(off - mid), F64(1e-3)))
            arg = two_pi_f * tw + psi[w] + F64(cur[w * P + n + j]) + phase_offset[w]
            rate[i] = T(amp * two_pi_f * _wp.cos(arg))
            targets[i] = T(off + amp * _wp.sin(arg))

    @_wp.kernel
    def tick_wp(dt: F64, t: _wp.array(dtype=F64), count: _wp.array(dtype=int)):
        # mj_step's d.time += timestep, on the controller's float64 clock.
        w = _wp.tid()
        t[w] = t[w] + dt
        count[w] = count[w] + 1

    _KERNELS[T] = {"sense_root": sense_root, "sense_joints": sense_joints,
                   "sense_geoms": sense_geoms, "touch_clear": touch_clear,
                   "touch_ground": touch_ground, "clearance_wp": clearance_wp, "due_wp": due_wp,
                   "observe": observe, "actor_in": actor_in, "actor_mid": actor_mid,
                   "actor_out": actor_out, "decide": decide, "cpg_wp": cpg_wp,
                   "tick_wp": tick_wp, "V": V, "M": M}
    return _KERNELS[T]


def _np_dtype(dtype):
    return _np.float64 if dtype == _wp.float64 else _np.float32


def _ints(a):
    return _wp.array(_np.ascontiguousarray(_np.asarray(a).reshape(-1), dtype=_np.int32),
                     dtype=int)


class DeviceControl:
    """The controller of the W worlds of one ``DeviceBody`` (see the module docstring).

    ``shared`` is the ``SharedPolicy`` (or None), ``policies`` and ``bases`` one
    ``control.cpg.Policy`` and one ``MobilityBasis`` (or None) per world, as
    ``rollout_batch`` takes them.  ``record`` is the number of decisions per
    world whose observation and action are kept for the host.

    A segment: :meth:`arm`, then per step :meth:`step` (or, for the parity
    tests, :meth:`sense_host` and :meth:`control`, then the body's step, then
    :meth:`tick`).  Nothing in :meth:`sense`, :meth:`control` or :meth:`tick`
    reads back to the host.
    """

    def __init__(self, body, *, shared=None, policies=None, bases=None,
                 control_hz: float = 25.0, record: int = 256) -> None:
        self.body = body
        self.dtype = T = body.dtype
        self.npdt = npdt = _np_dtype(T)
        envs = body.envs
        W, e0 = body.W, envs[0]
        self.W, self.envs = W, envs
        self.n = n = int(e0.cpg.n)
        self.P = 3 * n + 1
        if n != body.nact:
            raise ValueError("the CPG drives every actuator in a rollout; this body's do not")
        self.every = max(1, int(1.0 / (control_hz * body.timestep)))
        self.cap = int(record)
        K = make_kernels(T)
        V, M = K["V"], K["M"]

        def sc(a, t=T):
            return _wp.array(_np.ascontiguousarray(_np.asarray(a, float).reshape(-1),
                                                   dtype=_np.float64 if t == _wp.float64
                                                   else npdt), dtype=t)

        self._sc = sc
        z = lambda k, t=T: _wp.zeros(max(int(k), 1), dtype=t)  # noqa: E731

        # --- constants of the body ---------------------------------------
        m = e0.model
        self.root = int(e0.root_body)
        self.root_root = int(m.body_rootid[self.root])
        self._qadr, self._vadr = _ints(e0._act_qadr), _ints(e0._act_vadr)
        jr = _np.concatenate([_np.asarray(e.joint_range, float).reshape(-1, 2) for e in envs]) \
            if n else _np.zeros((0, 2))
        self._mid = sc(jr.mean(axis=1) if n else [0.0])
        self._half = sc(_np.maximum(0.5 * (jr[:, 1] - jr[:, 0]), 1e-6) if n else [1.0])
        self._lo = sc(_np.concatenate([e.cpg.lo[:n] for e in envs]) if n else [0.0])
        self._hi = sc(_np.concatenate([e.cpg.hi[:n] for e in envs]) if n else [0.0])
        self._margin = sc(_np.concatenate([_np.broadcast_to(e.cpg.margin, (n,)) for e in envs])
                          if n else [0.0])
        self._base_f = sc([e.cpg.base.frequency for e in envs])
        self._capacity = sc([e.budget.battery.capacity_j for e in envs], _wp.float64)
        self._morph = sc(_np.concatenate([e.morphology_context for e in envs]))
        geoms = _np.asarray(e0._machine_geoms, int)
        self.ng = len(geoms)
        self._gid = _ints(geoms if self.ng else [0])
        aabb = m.geom_aabb.reshape(-1, 6)[geoms] if self.ng else _np.zeros((1, 6))
        self._centre = _wp.array(_np.ascontiguousarray(aabb[:, :3], dtype=npdt), dtype=V)
        self._ghalf = _wp.array(_np.ascontiguousarray(aabb[:, 3:], dtype=npdt), dtype=V)
        from dytiscidae.core.mjcf import BEACH_SLOPE, SHORE_X, beach_extent, beach_surface_z
        lo, hi = beach_extent()
        self._beach = tuple(T(x) for x in (lo, hi, SHORE_X, BEACH_SLOPE,
                                           beach_surface_z(SHORE_X)))
        self._geom_bodyid = _ints(m.geom_bodyid)
        self.default_depth = float(e0.TARGET_DEPTH)

        # --- the shared actor ----------------------------------------------
        self.shared = shared
        self.NO = int(shared.n_modes) if shared is not None else 0
        self.H = int(shared.actor[0].out_features) if shared is not None else 0
        self._actor = {k: z(s) for k, s in (
            ("W1", self.H * OBS), ("b1", self.H), ("W2", self.H * self.H), ("b2", self.H),
            ("W3", self.NO * self.H), ("b3", self.NO), ("mean", OBS), ("var", OBS),
            ("log_std", self.NO))}
        if shared is not None:
            self.set_shared(shared)

        # --- per-design policies and bases ----------------------------------
        policies = list(policies) if policies is not None else [None] * W
        bases = list(bases) if bases is not None else [None] * W
        self.policies, self.bases = policies, bases
        self.rmax = max([b.modes.shape[0] for b in bases if b is not None] + [1])
        rmax, P = self.rmax, self.P
        modes = _np.zeros((W, rmax, P))
        inv = _np.zeros((W, rmax, 6))
        A = _np.zeros((W, rmax, 6))
        reach = _np.zeros((W, 6))
        r = _np.zeros(W, int)
        for w, b in enumerate(bases):
            if b is None:
                continue
            k = b.modes.shape[0]
            if k and b.modes.shape[1] != P:
                raise ValueError("a basis must span the CPG's 3n+1 parameters")
            r[w] = k
            if k:
                modes[w, :k] = b.modes
                iv, rc = b._inverse
                inv[w, :k], reach[w] = iv, rc
                A[w, :k] = b.effects * b.authority[:, None]
        self._modes, self._inv, self._A, self._reach = sc(modes), sc(inv), sc(A), sc(reach)
        self._r = _ints(r)
        has_pol = _np.array([p is not None for p in policies], _np.int32)
        pw, off, hid, nout, nmod = [], [], [], [], []
        at = 0
        for p in policies:
            off.append(at)
            if p is None:
                hid.append(0), nout.append(0), nmod.append(0)
                continue
            if int(p.n_obs) != OBS:
                raise ValueError(f"a per-design Policy reads {OBS} observations")
            wts = _np.asarray(p.weights, float)
            pw.append(wts)
            at += len(wts)
            hid.append(int(p.hidden)), nout.append(int(p.n_out))
            nmod.append(int(getattr(p, "n_modes", p.n_out)))
        self._pw = sc(_np.concatenate(pw) if pw else [0.0])
        self._pw_off, self._p_hidden = _ints(off), _ints(hid)
        self._p_out, self._p_modes = _ints(nout), _ints(nmod)
        self._has_policy = _ints(has_pol)
        self.hmax = max(hid + [1])
        self._has_ctrl = _ints([int(b is not None and (p is not None or shared is not None))
                                for b, p in zip(bases, policies)])

        # --- per-segment state ----------------------------------------------
        self.sense_buf = {"R": z(W, M), "Ri": z(W, M), "pos": z(W, V), "xipos": z(W, V),
                          "com": z(W, V), "ang": z(W, V), "lin": z(W, V), "msub": z(W),
                          "energy": z(W, _wp.float64), "touching": z(W, int),
                          "clear": z(W), "qa": z(W * n), "va": z(W * n),
                          "gx": z(W * self.ng, V), "gR": z(W * self.ng, M)}
        self.task = {"armed": z(W, int), "nph": z(W, int), "ph_start": z(W * MAX_PHASES,
                                                                         _wp.float64),
                     "ph_cruise": z(W * MAX_PHASES, int), "ph_speed": z(W * MAX_PHASES),
                     "ph_heading": z(W * MAX_PHASES), "ph_depth": z(W * MAX_PHASES),
                     "seg_t0": z(W, _wp.float64), "seg_T": z(W, _wp.float64),
                     "is_air": z(W, int), "height_ref": z(W), "cmd": z(W, int)}
        self.t = z(W, _wp.float64)
        self.count = z(W, int)
        self.active = z(W, int)
        self.diverged = z(W, int)
        self.due = z(W, int)
        self.sampled = z(W, int)
        self.seed = z(W, int)
        self.ndec = z(W, int)
        self.obs = z(W * OBS)
        self._h1, self._h2 = z(W * self.H), z(W * self.H)
        self.act, self.u = z(W * self.NO), z(W * self.NO)
        self._z, self._c = z(W * self.hmax), z(W * rmax)
        self.base = z(W * P)
        self.cur = z(W * P)
        self.cmd_twist = z(W * 6)
        self.gain = z(W)
        self.has_gain = z(W, int)
        self.rec_obs = z(W * self.cap * OBS)
        self.rec_act = z(W * self.cap * self.NO)
        self.rec_u = z(W * self.cap * self.NO)
        self._phase_offset = z(W, _wp.float64)
        self._psi, self._f_last, self._t_last = (z(W, _wp.float64) for _ in range(3))
        self._has_f = z(W, int)

    # ------------------------------------------------------------- the host

    def set_shared(self, shared) -> None:
        """The shared policy's current weights and normalisation into the device
        arrays, in place (a captured graph keeps their addresses).  Called once
        per PPO update, before the generation's first launch."""
        import torch

        a = shared.actor
        npdt = self.npdt
        with torch.no_grad():
            src = {"W1": a[0].weight, "b1": a[0].bias, "W2": a[2].weight, "b2": a[2].bias,
                   "W3": a[4].weight, "b3": a[4].bias, "mean": shared.obs_mean,
                   "var": shared.obs_var, "log_std": shared.log_std}
            for k, v in src.items():
                self._actor[k].assign(_np.ascontiguousarray(
                    v.detach().cpu().double().numpy().reshape(-1), dtype=npdt))

    def arm(self, domain, duration: float, params_list, *, sampled=None, seeds=None) -> None:
        """Start a segment on every world: ``TriphibianEnv._arm_task`` on the host
        machines (the same call ``rollout_batch`` makes) and its schedule, clock,
        height reference and command uploaded; the base parameters; the CPG's
        phase state and offset; the starting wetness and battery.  ``domain`` is
        one ``Domain`` or one per world.  ``sampled`` (per world) selects the
        worlds that explore, ``seeds`` their noise."""
        from dytiscidae.envs.tasks import CRUISE, TASK_SPEED
        from dytiscidae.envs.triphibian import DOMAIN_CYCLE

        W, P, npdt = self.W, self.P, self.npdt
        doms = list(domain) if isinstance(domain, (list, tuple)) else [domain] * W
        if len(doms) != W:
            raise ValueError(f"arm takes one domain or one per world ({W})")
        for e, dom in zip(self.envs, doms):
            e._arm_task(dom, duration)
        start = _np.zeros((W, MAX_PHASES))
        cru = _np.zeros((W, MAX_PHASES), _np.int32)
        spd, hdg, dep = (_np.zeros((W, MAX_PHASES)) for _ in range(3))
        nph = _np.zeros(W, _np.int32)
        for w, e in enumerate(self.envs):
            tk = e._active_task
            if len(tk.phases) > MAX_PHASES:
                raise ValueError(f"the device task table holds {MAX_PHASES} phases")
            nph[w] = len(tk.phases)
            ref = TASK_SPEED.get(tk.domain) or max(e.launch_speed, 1e-6)
            for p, ph in enumerate(tk.phases):
                start[w, p] = ph.start
                cru[w, p] = int(ph.kind == CRUISE)
                spd[w, p] = float(_np.clip(ph.speed / max(ref, 1e-6), 0.0, 3.0))
                hdg[w, p] = ph.heading
                dep[w, p] = ph.depth if ph.depth is not None else e.TARGET_DEPTH
        tk = self.task
        tk["armed"].assign(_np.ones(W, _np.int32))
        tk["nph"].assign(nph)
        tk["ph_start"].assign(start.ravel())
        tk["ph_cruise"].assign(cru.ravel())
        for k, v in (("ph_speed", spd), ("ph_heading", hdg), ("ph_depth", dep)):
            tk[k].assign(v.ravel().astype(npdt))
        tk["seg_t0"].assign(_np.array([e._seg_t0 for e in self.envs]))
        tk["seg_T"].assign(_np.array([e._seg_T for e in self.envs]))
        tk["is_air"].assign(_np.array([int(e._active_task.domain == "air")
                                       for e in self.envs], _np.int32))
        tk["height_ref"].assign(_np.array([e._height_ref for e in self.envs], npdt))
        tk["cmd"].assign(_np.array([DOMAIN_CYCLE.index(d) for d in doms], _np.int32))
        self.base.assign(_np.concatenate([_np.asarray(p.flat(), float)
                                          for p in params_list]).astype(npdt))
        self.cur.assign(self.base.numpy())
        self.t.assign(_np.array([float(e.data.time) for e in self.envs]))
        self.count.zero_()
        self.ndec.zero_()
        self.diverged.zero_()
        # No gain intent until this segment's first decision sets one.
        self.gain.fill_(1.0)
        self.has_gain.zero_()
        self.active.assign(_np.ones(W, _np.int32))
        self.sampled.assign(_np.zeros(W, _np.int32) if sampled is None
                            else _np.asarray(sampled, _np.int32))
        self.seed.assign(_np.zeros(W, _np.int32) if seeds is None
                         else (_np.asarray(seeds, _np.int64) % (2**31 - 1)).astype(_np.int32))
        cp = [e.cpg for e in self.envs]
        self._phase_offset.assign(_np.array([c.phase_offset for c in cp], float))
        self._psi.assign(_np.array([c._psi for c in cp], float))
        self._f_last.assign(_np.array([0.0 if c._f_last is None else c._f_last for c in cp]))
        self._t_last.assign(_np.array([c._t_last if _np.isfinite(c._t_last) else -1e300
                                       for c in cp]))
        self._has_f.assign(_np.array([int(c._f_last is not None) for c in cp], _np.int32))
        # CPG.pop_rate is host state: every rollout step commands the CPG for
        # every actuator, so the servo always has a rate to feed forward.
        self.body.has_rate.assign(_np.ones(W, _np.int32))
        self.body.diag["mean_submerged"].assign(
            _np.array([e.solver.diag.mean_submerged for e in self.envs], npdt))

    def sense_host(self, envs) -> None:
        """The sensing from host machines at this precision, for the parity tests:
        what :meth:`sense` reads from MJWarp, read from each ``MjData``."""
        npdt, b = self.npdt, self.sense_buf
        r, rr = self.root, self.root_root
        st = lambda f: _np.ascontiguousarray(_np.stack([f(e) for e in envs]), dtype=npdt)  # noqa
        b["R"].assign(st(lambda e: e.data.xmat[r].reshape(3, 3)))
        b["Ri"].assign(st(lambda e: e.data.ximat[r].reshape(3, 3)))
        b["pos"].assign(st(lambda e: e.data.xpos[r]))
        b["xipos"].assign(st(lambda e: e.data.xipos[r]))
        b["com"].assign(st(lambda e: e.data.subtree_com[rr]))
        b["ang"].assign(st(lambda e: e.data.cvel[r, 0:3]))
        b["lin"].assign(st(lambda e: e.data.cvel[r, 3:6]))
        b["msub"].assign(st(lambda e: e.solver.diag.mean_submerged))
        b["energy"].assign(_np.array([e.budget.battery.energy_j for e in envs], float))
        b["touching"].assign(_np.array([int(e._touching_ground()) for e in envs], _np.int32))
        if self.n:
            b["qa"].assign(_np.concatenate([e.data.qpos[e._act_qadr] for e in envs]).astype(npdt))
            b["va"].assign(_np.concatenate([e.data.qvel[e._act_vadr] for e in envs]).astype(npdt))
        if self.ng:
            g = _np.asarray(self.envs[0]._machine_geoms, int)
            b["gx"].assign(_np.concatenate([e.data.geom_xpos[g] for e in envs]).astype(npdt))
            b["gR"].assign(_np.concatenate([e.data.geom_xmat[g].reshape(-1, 3, 3)
                                            for e in envs]).astype(npdt))
        self._clearance()

    def sense(self) -> None:
        """The sensing from MJWarp's state (float32) and the body's own wetness
        and battery, no host transfer."""
        K, b, body = make_kernels(self.dtype), self.sense_buf, self.body
        d, W = body.d, self.W
        _wp.launch(K["sense_root"], dim=W, inputs=[
            self.root, self.root_root, d.xmat, d.ximat, d.xpos, d.xipos, d.subtree_com, d.cvel,
            b["R"], b["Ri"], b["pos"], b["xipos"], b["com"], b["ang"], b["lin"]])
        if self.n:
            _wp.launch(K["sense_joints"], dim=W * self.n, inputs=[
                self.n, self._qadr, self._vadr, d.qpos, d.qvel, b["qa"], b["va"]])
        _wp.copy(b["msub"], body.diag["mean_submerged"], count=W)
        _wp.copy(b["energy"], body.power._state["energy_j"], count=W)
        _wp.launch(K["touch_clear"], dim=W, inputs=[b["touching"]])
        _wp.launch(K["touch_ground"], dim=int(d.contact.geom.shape[0]), inputs=[
            d.nacon, d.contact.geom, d.contact.worldid, self._geom_bodyid, b["touching"]])
        if self.ng:
            _wp.launch(K["sense_geoms"], dim=W * self.ng, inputs=[
                self.ng, self._gid, d.geom_xpos, d.geom_xmat, b["gx"], b["gR"]])
        self._clearance()

    def _clearance(self) -> None:
        K, b = make_kernels(self.dtype), self.sense_buf
        if self.ng:
            _wp.launch(K["clearance_wp"], dim=self.W, inputs=[
                self.ng, b["gx"], b["gR"], self._centre, self._ghalf, *self._beach, b["clear"]])

    def control(self) -> None:
        """The decision (on decision steps) and the CPG command, no host transfer.
        Writes the body's ``targets`` and ``rate``."""
        K, T, W, b, tk = make_kernels(self.dtype), self.dtype, self.W, self.sense_buf, self.task
        body = self.body
        _wp.launch(K["due_wp"], dim=W, inputs=[
            self.every, self.count, self.active, body.power.active, self._has_ctrl, self.due])
        _wp.launch(K["observe"], dim=W, inputs=[
            self.n, b["R"], b["Ri"], b["pos"], b["xipos"], b["com"], b["ang"], b["lin"],
            b["msub"], b["touching"], b["energy"], self._capacity, b["qa"], b["va"], self._mid,
            self._half, self._base_f, tk["cmd"], self.t, tk["armed"], tk["nph"],
            tk["ph_start"], tk["ph_cruise"], tk["ph_speed"], tk["ph_heading"], tk["ph_depth"],
            tk["seg_t0"], tk["seg_T"], tk["is_air"], tk["height_ref"], b["clear"],
            T(self.default_depth), self._morph, self.due, self.active, self.diverged, self.obs])
        a = self._actor
        if self.shared is not None:
            H, NO = self.H, self.NO
            _wp.launch(K["actor_in"], dim=W * H, inputs=[
                H, self.obs, a["mean"], a["var"], a["W1"], a["b1"], self.due, self._h1])
            _wp.launch(K["actor_mid"], dim=W * H, inputs=[
                H, self._h1, a["W2"], a["b2"], self.due, self._h2])
            _wp.launch(K["actor_out"], dim=W * NO, inputs=[
                H, NO, self._h2, a["W3"], a["b3"], a["log_std"], self.sampled, self.seed,
                self.ndec, self.due, self.u, self.act])
        _wp.launch(K["decide"], dim=W, inputs=[
            self.P, self.n, self.rmax, self.NO, self.hmax, self.cap, self.obs, self.act,
            int(self.shared is not None), self._has_policy, self._pw, self._pw_off,
            self._p_hidden, self._p_out, self._p_modes, self._r, self._modes, self._inv,
            self._reach, self._A, self.base, T(GAIN_RANGE[0]), T(GAIN_RANGE[1]),
            T(INTENT_AUTHORITY), self.due, self.ndec, self._z, self._c, self.cur,
            self.cmd_twist, self.gain, self.has_gain, self.rec_obs, self.rec_act, self.rec_u,
            self.u])
        if self.n:
            _wp.launch(K["cpg_wp"], dim=W, inputs=[
                self.n, self.P, self.cur, self._lo, self._hi, self._margin, self._phase_offset,
                self.t, self.active, body.power.active, self._psi, self._f_last, self._t_last,
                self._has_f, body.targets, body.rate])

    def tick(self) -> None:
        """After the body's step: the clock and the step count."""
        K = make_kernels(self.dtype)
        _wp.launch(K["tick_wp"], dim=self.W, inputs=[
            _wp.float64(self.body.timestep), self.t, self.count])

    def step(self) -> None:
        """One rollout step on the device: sense, decide, command, the body's
        step (servo, fluid, scatter, ``mjw.step``, power), the clock."""
        self.sense()
        self.control()
        self.body.step()
        self.tick()

    # ------------------------------------------------------------- read back

    def observation(self):
        """The last decision's observation per world, ``(W, 33)``."""
        return self.obs.numpy()[:self.W * OBS].reshape(self.W, OBS).astype(float)

    def targets(self):
        nn = self.W * self.n
        return (self.body.targets.numpy()[:nn].reshape(self.W, self.n).astype(float),
                self.body.rate.numpy()[:nn].reshape(self.W, self.n).astype(float))

    def records(self) -> dict:
        """Per world, the decisions made so far (up to ``record``): ``obs``,
        ``act`` and the pre-squash ``u``, for the host's ``log_prob``, value and
        potential."""
        made = self.ndec.numpy()[:self.W]
        if int(made.max(initial=0)) > self.cap:
            raise RuntimeError(f"a world made {int(made.max())} decisions and the record "
                               f"holds {self.cap}: size record to the segment "
                               "(rollout_wp.decisions_per_segment)")
        nd = _np.minimum(made, self.cap)
        W, cap, NO = self.W, self.cap, max(self.NO, 1)
        ob = self.rec_obs.numpy()[:W * cap * OBS].reshape(W, cap, OBS)
        ac = self.rec_act.numpy()[:W * cap * NO].reshape(W, cap, NO)
        uu = self.rec_u.numpy()[:W * cap * NO].reshape(W, cap, NO)
        return {"n": nd, "obs": [ob[w, :nd[w]].astype(float) for w in range(W)],
                "act": [ac[w, :nd[w], :self.NO].astype(float) for w in range(W)],
                "u": [uu[w, :nd[w], :self.NO].astype(float) for w in range(W)]}

    def decision(self) -> dict:
        """The last decision per world: ``cur`` (the commanded CPG parameters,
        ``(W, 3n+1)``), the commanded twist, the gain and whether one was set,
        and the shared action."""
        W = self.W
        return {"cur": self.cur.numpy()[:W * self.P].reshape(W, self.P).astype(float),
                "cmd_twist": self.cmd_twist.numpy()[:W * 6].reshape(W, 6).astype(float),
                "gain": self.gain.numpy()[:W].astype(float),
                "has_gain": self.has_gain.numpy()[:W].astype(bool),
                "act": self.act.numpy()[:W * self.NO].reshape(W, self.NO).astype(float),
                "diverged": self.diverged.numpy()[:W].astype(bool),
                "clearance": self.sense_buf["clear"].numpy()[:W].astype(float),
                "touching": self.sense_buf["touching"].numpy()[:W].astype(bool)}


def sample_actions(shared, obs, seeds, decision: int = 0, *, sampled=True,
                   dtype=_wp.float32) -> tuple:
    """The device actor on ``obs (N, 33)``, one row per world; numpy in, numpy out.

    Each row draws its noise from ``wp.rand_init(seeds[row], decision * NO + j)``
    when ``sampled``, else takes the mean.  Returns ``(act, u)``: the squashed
    action and the pre-squash latent.  For the sampling check, which needs far
    more rows than a body has worlds.
    """
    K = make_kernels(dtype)
    npdt = _np_dtype(dtype)
    obs = _np.asarray(obs, float)
    N = obs.shape[0]
    H, NO = int(shared.actor[0].out_features), int(shared.n_modes)

    class _Shim:
        pass

    host = _Shim()
    host.npdt = npdt
    z = lambda k, t=dtype: _wp.zeros(max(int(k), 1), dtype=t)  # noqa: E731
    host._actor = {k: z(s) for k, s in (
        ("W1", H * OBS), ("b1", H), ("W2", H * H), ("b2", H), ("W3", NO * H), ("b3", NO),
        ("mean", OBS), ("var", OBS), ("log_std", NO))}
    DeviceControl.set_shared(host, shared)
    a = host._actor
    ob = _wp.array(_np.ascontiguousarray(obs.reshape(-1), dtype=npdt), dtype=dtype)
    due = _wp.array(_np.ones(N, _np.int32), dtype=int)
    smp = _wp.array(_np.full(N, int(bool(sampled)), _np.int32), dtype=int)
    sd = _wp.array((_np.asarray(seeds, _np.int64) % (2**31 - 1)).astype(_np.int32), dtype=int)
    nd = _wp.array(_np.full(N, int(decision), _np.int32), dtype=int)
    h1, h2, u, act = z(N * H), z(N * H), z(N * NO), z(N * NO)
    _wp.launch(K["actor_in"], dim=N * H, inputs=[H, ob, a["mean"], a["var"], a["W1"], a["b1"],
                                                 due, h1])
    _wp.launch(K["actor_mid"], dim=N * H, inputs=[H, h1, a["W2"], a["b2"], due, h2])
    _wp.launch(K["actor_out"], dim=N * NO, inputs=[H, NO, h2, a["W3"], a["b3"], a["log_std"],
                                                   smp, sd, nd, due, u, act])
    return (act.numpy().reshape(N, NO).astype(float), u.numpy().reshape(N, NO).astype(float))
