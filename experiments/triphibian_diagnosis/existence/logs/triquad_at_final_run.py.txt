#!/usr/bin/env python3
"""E0: one expressible body, a hand-written feedback controller, the repo's own scores.

Spec: PREREG.md beside this file (frozen).  No learning, no search, no optimiser.

What is substituted, and only this: the command.  `experiments/rotor/fly.py`
replaces `env.cpg.command` with a closure; this does the same, by composition:

* `Hand.command(params, t)` is installed as `env.cpg.command`.  It acts as the
  hand controller only when `params` is this controller's own token (the object
  handed to the evaluators as ``Controller.params``).  For any other params
  object -- the body's base gait on the level/thrust rigs, the held-still arm --
  it calls the CPG's original method, so every body-level gate the scores read
  (`level_margin`, `thrust_margin`, `flap_travel`, the trim) is measured on the
  body's own gait, as the search would measure it.
* The commanded medium reaches the controller through the evaluators' own
  channel: they call ``policy.act(env.observation(commanded))`` at 25 Hz, and
  `HandPolicy.act` reads the domain one-hot (observation[11:14]) from it.
  `HandBasis.command_policy` returns the token unchanged.

Scores are called unmodified: `evaluate.run_segment`, `transitions.run_transition`,
`mission.run_continuous` + `continuous_score`.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/triphibian_diagnosis/existence/triquad.py facts --body B1
    ... triquad.py run --body B1 --arm hand --items air,water,land --seeds 0-7 --out final
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import mujoco  # noqa: E402

from dytiscidae.core.bodyplans import REFERENCE_PLANS  # noqa: E402
from dytiscidae.core.genome import FOOT, PADDLE, Edge, Part  # noqa: E402
from dytiscidae.core.mjcf import BEACH_SLOPE, beach_extent, beach_surface_z  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.evaluate import Controller, _scatter_seed, run_segment  # noqa: E402
from dytiscidae.envs.mission import continuous_score, run_continuous  # noqa: E402
from dytiscidae.envs.tasks import CRUISE, HOLD, schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.transitions import run_transition  # noqa: E402
from dytiscidae.envs.triphibian import (  # noqa: E402
    DOMAIN_CYCLE, Domain, MissionSpec, TriphibianEnv, evaluate_tier0)
from dytiscidae.physics.medium import GRAVITY  # noqa: E402
from dytiscidae.physics.rotor import bemt  # noqa: E402

SEEDS = list(range(8))          # PREREG section 4: fixed, no selection
SEGMENT_SECONDS = 8.0           # the search's --segment-seconds (CLAUDE.md launch block)
LEG_SECONDS = 8.0               # PREREG prediction 5
CROSSINGS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")
_BAD = mujoco.mjtWarning.mjWARN_BADQACC

# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------

#: Every edit that turns the reference quad (B0) into B1.  Listed in RESULT.md.
B1_EDITS = dict(
    hull_length=0.40,        # m   (B0 0.22): room for non-overlapping discs, and volume
    hull_radius=0.059,       # m   (B0 0.05): sets density ratio ~1.09 (see `facts`)
    battery_wh=200.0,        # Wh  (B0 60)
    battery_chem="lis",      # (B0 "lipo"): the only cell in materials.CELLS that closes the pack
    leg_length=0.20, leg_radius=0.006, leg_azimuth_deg=-50.0, leg_elev_deg=17.0,
    fin_span=0.12, fin_chord=0.05, fin_radius=0.004,
    throttle_margin=1.05,    # base-gait rotor throttle = this x hover (level-rig gate)
)


def b0_genome():
    return REFERENCE_PLANS["quad"]()


def b1_genome(**over):
    e = dict(B1_EDITS)
    e.update(over)
    g = REFERENCE_PLANS["quad"]()
    hull, arm = g.parts
    hull.length, hull.radius = float(e["hull_length"]), float(e["hull_radius"])
    g.battery_wh, g.battery_chem = float(e["battery_wh"]), str(e["battery_chem"])
    # Ground gear: four fixed legs (FOOT, no joint) from the hull, down and outward.
    leg = Part(kind=FOOT, length=float(e["leg_length"]), radius=float(e["leg_radius"]),
               material="cfrp", joint="none", actuated=False)
    # One fixed dorsal fin (PADDLE, no joint).  Not for control: `_structural_checks`
    # fails `has_propulsor` (gating, margin -1) for a body with no surface and no
    # jet, so the search's Tier-0 gate rejects a bare multirotor.
    fin = Part(kind=PADDLE, span=float(e["fin_span"]), root_chord=float(e["fin_chord"]),
               radius=float(e["fin_radius"]), material="cfrp", joint="none", actuated=False)
    g.parts = [hull, arm, leg, fin]
    az, el = math.radians(e["leg_azimuth_deg"]), math.radians(e["leg_elev_deg"])
    g.edges = list(g.edges) + [
        Edge(parent=0, child=2, pos_u=0.85, azimuth=az, elevation=+el, roll=0.0,
             scale=1.0, reflect=True),
        Edge(parent=0, child=2, pos_u=0.15, azimuth=az, elevation=-el, roll=0.0,
             scale=1.0, reflect=True),
        Edge(parent=0, child=3, pos_u=0.5, azimuth=math.pi / 2, elevation=0.0,
             roll=math.pi / 2, scale=1.0, reflect=False),
    ]
    g.lineage = ["quad", "E0-B1"]
    # Base-gait throttle at hover for the heavier body: `level_margin` (the rig
    # behind the air score's flight term) is measured on the body's own gait.
    p = build(g)
    from dytiscidae.core.mjcf import ROTOR_TIP_SPEED
    from dytiscidae.physics.medium import AIR
    spec = next(s.rotor for s in p.segments if s.rotor is not None)
    T, _ = bemt(spec, 600.0, 0.0, 0.0, AIR.rho, AIR.mu)
    om_h = math.sqrt(e["throttle_margin"] * p.mass * GRAVITY / (4.0 * T / 600.0 ** 2))
    arm.rotor_throttle = float(np.clip(om_h / (ROTOR_TIP_SPEED / spec.radius), 0.05, 0.95))
    return g


BODIES = {"B0": b0_genome, "B1": b1_genome}


def tier0_gate(pheno) -> dict:
    """The check `evolution/loop.py` applies before any rollout (lines 519-520,
    690-694): ``gate_margin < cfg.tier0_gate or t0.mission_fraction <= 0``."""
    from dytiscidae.evolution.loop import SearchConfig
    assert "torch" not in sys.modules
    t0 = evaluate_tier0(pheno, MissionSpec())
    bar = float(SearchConfig().tier0_gate)
    rejected = bool(pheno.report.gate_margin < bar or t0.mission_fraction <= 0.0)
    return {"passes": not rejected, "gate_margin": float(pheno.report.gate_margin),
            "tier0_gate": bar, "tier0_mission_fraction": float(t0.mission_fraction),
            "checks": {c.name: [round(float(c.margin), 4), bool(c.gating)]
                       for c in pheno.report.checks},
            "notes": list(t0.notes)}


# --------------------------------------------------------------------------
# The hand controller
# --------------------------------------------------------------------------

P_STATION = np.array([-2.0, 0.0])   # water/air station: deep water, 2 m seaward of the shelf
X_SHELF = 0.086                     # seaward end of the ramp (mjcf.beach_extent)
X_LAND = 8.8                        # landing target (waterline is at x = 7.80)
X_ASHORE = 8.1


class HandPolicy:
    """Reads the commanded medium out of the observation the evaluator builds."""

    def __init__(self):
        self.commanded = None

    def act(self, obs):
        c = np.asarray(obs[11:14], float)
        self.commanded = DOMAIN_CYCLE[int(np.argmax(c))] if c.max() > 0.5 else None
        return np.zeros(1)


class HandBasis:
    def __init__(self, token):
        self.token = token

    def command_policy(self, params, action, n, policy):
        return self.token, np.zeros(6), None

    def twist_of(self, coeffs):
        return np.zeros(6)


class Hand:
    def __init__(self, env: TriphibianEnv, record: bool = False, gains: dict | None = None):
        self.env = env
        m = env.model
        assert all(a.endswith("_r") for a in env.act_names), env.act_names
        self.rot = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a[:-2] + "_rot")
                    for a in env.act_names]
        self.n = len(self.rot)
        spec = env.rotors.spec[0]
        air, water = env.medium.air, env.medium.water
        Ta, Qa = bemt(spec, 600.0, 0.0, 0.0, air.rho, air.mu)
        Tw, Qw = bemt(spec, 30.0, 0.0, 0.0, water.rho, water.mu)
        self.kt_a, self.kq_a = Ta / 600.0 ** 2, Qa / 600.0 ** 2
        self.kt_w, self.kq_w = Tw / 30.0 ** 2, Qw / 30.0 ** 2
        self.rho_w = float(water.rho)
        self.om_max = float(env.cpg.hi[0])
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in env.act_names]
        self.kv = float(m.actuator_gainprm[aid[0], 0])
        self.tau_max = float(m.actuator_forcerange[aid[0], 1])
        self.mass = float(env.p.mass)
        self.vol_b = float(env.p.buoyant_volume)
        self.M = np.zeros((m.nv, m.nv))
        self.token = env.held_still_params()        # identity marker only
        self.policy = HandPolicy()
        self.basis = HandBasis(self.token)
        self.ctrl = Controller(params=self.token, policy=self.policy,
                               bases={"air": self.basis, "water": self.basis})
        self.g = dict(
            # air: fly.py's cascade
            kv_a=1.2, kz_a=2.0, kvz_a=5.0, tilt_a=35.0, kv_nav=2.5, kpos_nav=1.5, abrake=4.0, wn=13.0, zeta=1.3, wn_yaw=9.0, zeta_yaw=1.0,
            # water
            kv_w=1.5, kpos_w=1.0, kz_w=4.0, kvz_w=8.0, adec_w=1.5, vsink=1.5, vrise=1.0,
            cruise_gain_w=1.15, dive_in=0.2, dive_out=0.08, tilt_hold_w=30.0,
            stage_depth=0.12, stage_tilt_deg=3.0, stage_rate=0.3,
            # ground skim
            kp_g=8.0, kd_g=4.0, kz_g=50.0, kdz_g=8.0, c_ref=0.03, tilt_g=12.0, v_walk=0.07,
            settle_s=1.0,
            # navigation
            kpos=0.8, v_nav=6.0, v_tr=9.0, h_air=1.8, h_tr=0.45, depth_nav=0.8,
            cut_clear=0.45, vdesc=2.5,
        )
        if gains:
            self.g.update(gains)
        self.yaw_ok = self._yaw_authority()
        self.rec = [] if record else None
        self._orig = env.cpg.command
        env.cpg.command = self.command
        self._reset()

    def _yaw_authority(self) -> bool:
        """Whether rotor reaction torque can yaw the body independently of roll
        and pitch.  On the reference quad it cannot: its propellers are handed
        by side (left +, right -), so the yaw row of the mixer is -(kappa/0.2)
        times the roll row and the 4x4 mix has rank 3.  fly.py's least-squares
        mixer then leaks every yaw demand into roll (8.6%).  Without authority
        the yaw demand is dropped and the heading is left free; the cascade is
        omnidirectional and no score reads the heading."""
        env, m, d = self.env, self.env.model, self.env.data
        mujoco.mj_forward(m, d)
        R = d.xmat[env.root_body].reshape(3, 3)
        mb = m.body_mass[1:]
        com = (mb[:, None] * d.xipos[1:]).sum(0) / float(mb.sum())
        A = np.zeros((4, self.n))
        for i, b in enumerate(self.rot):
            r = R.T @ (d.xipos[b] - com)
            ax_b = R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[m.body_jntadr[b]])
            A[:, i] = [1.0, r[1], -r[0], -float(np.sign(ax_b[2]))]
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        return int(np.linalg.matrix_rank(A, tol=1e-3)) == 4

    # ----------------------------------------------------------------- state
    def _reset(self):
        self.t_last = -1.0
        self.L = {}                 # latches of the current episode
        self.ph_idx = None
        self.nav_M = None
        self.landed = False
        self.air_up = False
        self.cut = False
        self.mode = ""
        self.diving = False
        self.policy.commanded = None

    def command(self, params, t):
        if params is not self.token:
            return self._orig(params, t)        # the body's own CPG
        d = self.env.data
        if d.time <= 0.0 or d.time < self.t_last:
            self._reset()
        self.t_last = float(d.time)
        out = self._act()
        if self.rec is not None:
            cm = self.policy.commanded
            self.rec.append((float(d.time), d.qpos.copy(), d.qvel.copy(),
                             -1 if cm is None else DOMAIN_CYCLE.index(cm), out.copy()))
        return out

    # --------------------------------------------------------------- sensing
    def _sense(self):
        env, d = self.env, self.env.data
        S = type("S", (), {})()
        S.t = float(d.time)
        S.R = d.xmat[env.root_body].reshape(3, 3).copy()
        S.p = d.xpos[env.root_body].copy()            # the point every score reads
        # Not `d.subtree_com`: in the rollout loop it is not from the same pass as
        # `xipos` (probes/p05: the lever arms came out 0.058 m off at 6.6 m/s, about
        # two steps of travel, and the mixer pitched the machine over).  The mass-
        # weighted mean of `xipos` is, by construction.
        mb = env.model.body_mass[1:]
        S.com = (mb[:, None] * d.xipos[1:]).sum(0) / float(mb.sum())
        S.v = d.qvel[:3].copy()
        S.w = d.qvel[3:6].copy()                      # body frame
        S.hubs = d.xipos[self.rot].copy()
        S.s = np.asarray(env.medium.properties(S.hubs, np.full(self.n, 0.01), S.t)[2], float)
        S.depth = float(env.depth())
        S.clr = float(env.clearance())
        S.touch = bool(env._touching_ground())
        S.sub = float(env.solver.diag.mean_submerged)
        S.yaw = math.atan2(float(S.R[1, 0]), float(S.R[0, 0]))
        return S

    @staticmethod
    def _surf(x: float) -> float:
        """Height of what is underneath: the beach, or the sea surface."""
        return max(beach_surface_z(float(x)), 0.0)

    @staticmethod
    def _slope(x: float) -> float:
        lo, hi = beach_extent()
        return BEACH_SLOPE if (lo <= x <= hi and beach_surface_z(float(x)) > 0.0) else 0.0

    # --------------------------------------------------------------- cascade
    def _off(self):
        return np.zeros(self.n)

    def _cascade(self, S, a_fb, yaw, tilt_deg, *, water=False, fz_cap=None, fz_min=None,
                 allow_down=False):
        env, m, d = self.env, self.env.model, self.env.data
        mujoco.mj_fullM(m, d, self.M)
        Mt = np.diag(self.M[0:3, 0:3]).copy()
        c = S.R.T @ (S.com - S.p)                       # COM in the root frame
        mt = float(Mt.mean())
        J = self.M[3:6, 3:6] - mt * (float(c @ c) * np.eye(3) - np.outer(c, c))
        w_net = self.mass * GRAVITY - self.rho_w * GRAVITY * self.vol_b * S.sub
        f = Mt * np.asarray(a_fb, float)
        f[2] += w_net
        if water:
            # Under water the weight is a few percent of the dry weight, so the
            # thrust vector is pointed where the demand points: sideways to
            # travel, and -- while the target is far below (`allow_down`) -- down.
            w0 = max(w_net, 0.5)
            if not allow_down:
                f[2] = max(f[2], 0.05 * w0)
                if tilt_deg > 0.0:
                    lim = math.tan(math.radians(tilt_deg))
                    h = float(np.linalg.norm(f[:2]))
                    if h > f[2] * lim:
                        f[:2] *= f[2] * lim / h
            fdir = f.copy()
            fn = float(np.linalg.norm(f))
            if fn < 0.5 * w0:
                fdir[2] += 0.5 * w0 - fn            # a small demand leans back upright
            z_des = fdir / max(float(np.linalg.norm(fdir)), 1e-9)
        else:
            lim = math.tan(math.radians(tilt_deg))
            h = float(np.linalg.norm(f[:2]))
            f[2] = max(f[2], 0.02 * self.mass * GRAVITY if fz_min is None else fz_min)
            if fz_cap is not None:
                f[2] = min(f[2], fz_cap)
            if h > f[2] * lim:
                f[:2] *= f[2] * lim / h
            z_des = f / np.linalg.norm(f)
        if self.yaw_ok:
            x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        else:
            # No yaw authority (see `_yaw_authority`): ask for the attitude that
            # points the thrust without twisting, whatever the heading is.
            x_c = S.R[:, 0] if abs(float(S.R[:, 0] @ z_des)) < 0.95 else S.R[:, 1]
        y_d = np.cross(z_des, x_c)
        y_d /= max(np.linalg.norm(y_d), 1e-9)
        x_d = np.cross(y_d, z_des)
        Rd = np.stack([x_d, y_d, z_des], 1)
        E = 0.5 * (Rd.T @ S.R - S.R.T @ Rd)
        e_r = np.array([E[2, 1], E[0, 2], E[1, 0]])
        # fly.py scales one PD by the mean diagonal of the inertia about the root
        # *origin* (the hull's tail end), which on the reference quad is 3.9x its
        # pitch inertia about the centre of mass (probes/p03): an effective 13.9
        # rad/s, zeta 1.6 in pitch.  Written here per axis on the inertia about
        # the centre of mass, so a longer hull does not change the loop.
        g = self.g
        kp = np.array([g["wn"] ** 2, g["wn"] ** 2, g["wn_yaw"] ** 2])
        kd = np.array([2.0 * g["zeta"] * g["wn"], 2.0 * g["zeta"] * g["wn"],
                       2.0 * g["zeta_yaw"] * g["wn_yaw"]])
        if not self.yaw_ok:
            kp[2] = kd[2] = 0.0
        tau = J @ (-kp * e_r - kd * S.w)
        thrust = max(float(f @ S.R[:, 2]), 0.0)
        # per-rotor coefficients of the medium each rotor is in
        k = (1.0 - S.s) * self.kt_a + S.s * self.kt_w
        kq = (1.0 - S.s) * self.kq_a + S.s * self.kq_w
        kap = kq / k
        A = np.zeros((4, self.n))
        for i, b in enumerate(self.rot):
            r = S.R.T @ (S.hubs[i] - S.com)
            ax_b = S.R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[m.body_jntadr[b]])
            sgn = float(np.sign(ax_b[2]))
            A[:, i] = [1.0, r[1], -r[0], -kap[i] * sgn]
        t_max = np.minimum(0.9 * self.tau_max / kap, k * self.om_max ** 2)
        # torques first, then as much collective as the rotors have left: the
        # same least-squares answer as fly.py's while nothing saturates
        if self.yaw_ok:
            dlt = np.linalg.lstsq(A, np.array([0.0, tau[0], tau[1], tau[2]]), rcond=None)[0]
        else:
            dlt = np.linalg.lstsq(A[:3], np.array([0.0, tau[0], tau[1]]), rcond=None)[0]
        span = float(dlt.max() - dlt.min())
        if span > float(t_max.min()):
            dlt *= float(t_max.min()) / span
        col = float(np.clip(thrust / self.n, -dlt.min(), float(np.min(t_max - dlt))))
        t_i = np.clip(col + dlt, 0.0, t_max)
        return self._rotor_cmd(S, t_i, k, kq)

    #: Axial advance ratio J = V_ax / (Omega R) below which thrust is static,
    #: and at which it is zero (probes/p06, the repo's `bemt`: T/T0 = 0.99 at
    #: 0.05, 0.86 at 0.10, 0.29 at 0.20, -0.05 at 0.25; the same in both media).
    J_FLAT, J_ZERO = 0.07, 0.243

    def _rotor_cmd(self, S, t_i, k, kq):
        """Rotor speed commands for thrusts ``t_i``: the static law, corrected
        for the flow each rotor is moving through, plus the servo's droop.

        Under water a rotor turns at tens of rad/s, so a 0.6 m/s climb is
        J = 0.17 and takes two thirds of its thrust away (the first exit from
        the water stalled at exactly its pitch speed)."""
        R_r = float(self.env.rotors.spec[0].radius)
        w_world = S.R @ S.w
        om = np.sqrt(t_i / k)
        out = np.zeros(self.n)
        for i in range(self.n):
            v_hub = S.v + np.cross(w_world, S.hubs[i] - S.p)
            v_ax = float(v_hub @ S.R[:, 2])
            o = float(om[i])
            if v_ax > 0.0 and t_i[i] > 0.0 and v_ax / max(o * R_r, 1e-9) > self.J_FLAT:
                a_ = k[i] * self.J_ZERO
                b_ = k[i] * v_ax / R_r
                c_ = t_i[i] * (self.J_ZERO - self.J_FLAT)
                o = (b_ + math.sqrt(b_ * b_ + 4.0 * a_ * c_)) / (2.0 * a_)
            out[i] = o + kq[i] * o * o / self.kv      # kv (om_cmd - om) = kq om^2
        return np.clip(out, 0.0, self.om_max)

    def _accel(self, S, *, p_ref=None, v_ff=None, v_des=None, z_ref, kpos, kv, kz, kvz,
               vmax, v_dn, v_up, vz_ff=0.0, a_dec=None, a_brake=None):
        if v_des is None:
            dvec = np.asarray(p_ref, float) - S.p[:2]
            dist = float(np.linalg.norm(dvec))
            sp = kpos * dist
            if a_brake is not None and dist > a_brake / (kpos * kpos):
                sp = math.sqrt(2.0 * a_brake * (dist - 0.5 * a_brake / (kpos * kpos)))
            v_cmd = dvec / max(dist, 1e-9) * min(sp, vmax)
            if v_ff is not None:
                v_cmd = v_cmd + np.asarray(v_ff, float)
        else:
            v_cmd = np.asarray(v_des, float)
        a = np.zeros(3)
        a[:2] = kv * (v_cmd - S.v[:2])
        e = float(z_ref - S.p[2])
        if a_dec is None:
            vz = kz * e
        else:
            # linear inside e_lin, then the speed a constant deceleration can
            # still take off before the target: arrives without creeping
            e_lin = a_dec / (kz * kz)
            vz = kz * e if abs(e) <= e_lin else math.copysign(
                math.sqrt(2.0 * a_dec * (abs(e) - 0.5 * e_lin)), e)
        vz_cmd = float(np.clip(vz, -v_dn, v_up)) + vz_ff
        a[2] = kvz * (vz_cmd - S.v[2])
        return a

    def _fly_air(self, S, *, yaw, tilt=None, **kw):
        g = self.g
        a = self._accel(S, kpos=kw.pop("kpos", g["kpos"]), kv=kw.pop("kv", g["kv_a"]),
                        kz=g["kz_a"], kvz=g["kvz_a"], **kw)
        return self._cascade(S, a, yaw, g["tilt_a"] if tilt is None else tilt)

    def _fly_water(self, S, *, yaw, dive=False, tilt=0.0, **kw):
        g = self.g
        a = self._accel(S, kpos=g["kpos_w"], kv=g["kv_w"], kz=g["kz_w"], kvz=g["kvz_w"],
                        v_dn=kw.pop("v_dn", g["vsink"]), v_up=kw.pop("v_up", g["vrise"]),
                        a_dec=g["adec_w"], **kw)
        # Thrust may point below the horizon only while the target is well below:
        # entered `dive_in` above it, left `dive_out` above it.
        above = float(S.p[2] - kw["z_ref"])
        if not dive or above < g["dive_out"]:
            self.diving = False
        elif above > g["dive_in"]:
            self.diving = True
        return self._cascade(S, a, yaw, tilt, water=True, allow_down=self.diving)

    def _latch(self, key, value):
        if key not in self.L:
            self.L[key] = value() if callable(value) else value
        return self.L[key]

    # ----------------------------------------------------------- the law
    def _act(self):
        S = self._sense()
        yaw0 = self._latch("yaw0", S.yaw)
        self._latch("z0", float(S.p[2]))
        ph = self.env._phase_now()
        if ph is not None:
            task = self.env._active_task
            idx = task.index_at((S.t - self.env._seg_t0) / self.env._seg_T)
            new = idx != self.ph_idx
            self.ph_idx = idx
            dom = task.domain
            if dom == "air":
                return self._seg_air(S, ph)
            if dom == "water":
                return self._seg_water(S, ph, new, yaw0)
            return self._seg_land(S, ph, new, yaw0)
        M = self.policy.commanded
        if M is None:
            self.mode = "no-command"
            return self._off()
        if M is not self.nav_M:
            self.nav_M = M
            for k in [k for k in self.L if k.startswith("nav_")]:
                del self.L[k]
            self.landed = self.air_up = self.cut = False
        if M is Domain.LAND:
            return self._nav_land(S, yaw0)
        if M is Domain.AIR:
            return self._nav_air(S, yaw0)
        return self._nav_water(S, yaw0)

    # ---- segments
    def _seg_air(self, S, ph):
        """`experiments/rotor/fly.py`: cruise on the commanded heading at the
        commanded speed, hold the height the segment began at, nose on heading."""
        self.mode = "seg-air"
        g = self.g
        head, speed = float(ph.heading), float(ph.speed or 0.0)
        v_des = speed * np.array([math.cos(head), math.sin(head)])
        return self._fly_air(S, v_des=v_des, z_ref=self.L["z0"], yaw=head, vmax=0.0,
                             v_dn=1e9, v_up=1e9)

    def _seg_water(self, S, ph, new, yaw0):
        g = self.g
        if not (S.s.min() > 0.5 and S.depth > 0.0):
            self.mode = "seg-water-dry"
            return self._off()
        z_ref = -float(ph.depth)
        if ph.kind == HOLD:
            self.mode = "seg-water-hold"
            if new or "w_hold" not in self.L:
                self.L["w_hold"] = S.p[:2] + 0.6 * S.v[:2]
            return self._fly_water(S, p_ref=self.L["w_hold"], z_ref=z_ref, yaw=yaw0, vmax=0.5,
                                   dive=True, tilt=g["tilt_hold_w"])
        self.mode = "seg-water-cruise"
        if new or "w_p0" not in self.L:
            self.L["w_p0"], self.L["w_t0"] = S.p[:2].copy(), S.t
        h = np.array([math.cos(ph.heading), math.sin(ph.heading)])
        v = g["cruise_gain_w"] * float(ph.speed) * h
        p_ref = self.L["w_p0"] + v * (S.t - self.L["w_t0"])
        return self._fly_water(S, p_ref=p_ref, v_ff=v, z_ref=z_ref, yaw=yaw0, vmax=0.5, dive=True)

    def _seg_land(self, S, ph, new, yaw0):
        g = self.g
        if ph.kind != CRUISE:
            self.mode = "seg-land-stop"
            return self._off()
        if new:
            self.L.pop("g_p0", None)
            self.L["g_start"] = S.t
        self.L.setdefault("g_start", S.t)
        if "g_p0" not in self.L:
            # Sit down first.  This airframe cannot yaw itself (`_yaw_authority`),
            # so the yaw rate the segment starts with (scatter: sd 0.25 rad/s)
            # would carry the root -- 0.2 m behind the centre of mass, and the
            # point the score reads -- sideways at 0.05 m/s for the whole walk.
            # Friction takes it out in a fraction of a second.
            still = (S.touch and float(np.linalg.norm(S.w)) < 0.1
                     and float(np.linalg.norm(S.v)) < 0.05)
            if not still and S.t - self.L["g_start"] < g["settle_s"]:
                self.mode = "seg-land-settle"
                return self._off()
            self.L["g_p0"], self.L["g_t0"] = S.p[:2].copy(), S.t
        # Taxi: a few centimetres clear of the beach, on a position reference
        # that moves along the commanded heading.  (Skidding on the feet was the
        # pre-registered mode and is in trials.jsonl: contact moments through
        # the roll channel spin this airframe in yaw, and one seed in four ran
        # sideways.)
        self.mode = "seg-land-taxi"
        h = np.array([math.cos(ph.heading), math.sin(ph.heading)])
        v = g["v_walk"] * h
        p_ref = self.L["g_p0"] + v * (S.t - self.L["g_t0"])
        a = np.zeros(3)
        a[:2] = g["kp_g"] * (p_ref - S.p[:2]) + g["kd_g"] * (v - S.v[:2])
        a[2] = g["kz_g"] * (g["c_ref"] - S.clr) - g["kdz_g"] * S.v[2]
        w = self.mass * GRAVITY
        return self._cascade(S, a, yaw0, g["tilt_g"], fz_cap=1.10 * w, fz_min=0.5 * w)

    # ---- no task: crossings and the continuous mission
    def _zoff(self, S):
        """Root height above the machine's lowest point, at this attitude."""
        return float(S.p[2] - (S.clr + self._clear_ref(S)))

    def _clear_ref(self, S):
        # `clearance` is the minimum over geoms of (bottom - ground under it);
        # the ground under the root is close enough for a set-point.
        return self._surf(S.p[0])

    def _exit_water(self, S, yaw0, z_abs):
        """Up through the surface: rise to `stage_depth` under it, wait there
        until level and still, then climb through on the same cascade.

        Why staged: a rotor that has spun up in air and is put back under is
        hundreds of newtons for a few milliseconds (0.0101 Om^2; ~2.4 N s per
        event, a 50 rad/s roll kick on this airframe), so the four rotors have
        to leave together and not come back.  The first exits tumbled at 60-86
        rad/s on exactly that, and one diverged."""
        g = self.g
        xy = self._latch("nav_exit_xy", lambda: S.p[:2].copy())
        hub_top = float(S.hubs[:, 2].max())
        if "nav_go" not in self.L and S.s.min() > 0.99:
            self.mode = "exit-stage"
            tilt = math.degrees(math.acos(float(np.clip(S.R[2, 2], -1.0, 1.0))))
            if (hub_top > -(g["stage_depth"] + 0.05) and tilt < g["stage_tilt_deg"]
                    and float(np.linalg.norm(S.w)) < g["stage_rate"]):
                self.L["nav_go"] = S.t
            z_stage = float(S.p[2] + (-g["stage_depth"] - hub_top))
            return self._fly_water(S, p_ref=xy, z_ref=z_stage, yaw=yaw0, vmax=0.3, v_up=1.5,
                                   tilt=20.0)
        self.L.setdefault("nav_go", S.t)
        if S.s.min() > 0.99:
            self.mode = "exit-water"
            return self._fly_water(S, p_ref=xy, z_ref=z_abs, yaw=yaw0, vmax=0.3, v_up=3.0,
                                   tilt=10.0)
        self.mode = "exit-air"
        return self._fly_air(S, p_ref=xy, z_ref=z_abs, yaw=yaw0, vmax=1.0, v_dn=2.0, v_up=3.0,
                             tilt=10.0)

    def _nav_land(self, S, yaw0):
        g = self.g
        x = float(S.p[0])
        dry = S.depth <= 0.0 and S.s.max() < 0.01
        if self.landed or (dry and x >= X_ASHORE and (S.touch or S.clr < 0.12)
                           and float(np.linalg.norm(S.v)) < 1.5):
            self.landed = True
            self.mode = "landed"
            return self._off()
        if not dry:
            return self._exit_water(S, yaw0, 0.9)
        tgt = self._latch("nav_land_xy", lambda: np.array(
            [max(X_LAND, x), float(np.clip(S.p[1], -5.0, 5.0))]))
        dist = float(np.linalg.norm(tgt - S.p[:2]))
        sp = float(np.linalg.norm(S.v[:2]))
        zoff = S.p[2] - S.clr - self._surf(x)
        nav = dict(kpos=g["kpos_nav"], kv=g["kv_nav"], a_brake=g["abrake"])
        if not (x >= X_ASHORE + 0.3 and dist < 1.0 and sp < 1.2):
            self.mode = "to-land"
            z_ref = self._surf(x) + zoff + g["h_tr"]
            return self._fly_air(S, p_ref=tgt, z_ref=z_ref, yaw=yaw0, vmax=g["v_tr"],
                                 v_dn=1.5, v_up=3.0, vz_ff=self._slope(x) * float(S.v[0]), **nav)
        self.mode = "touch-down"
        return self._fly_air(S, p_ref=tgt, z_ref=S.p[2] - 1.0, yaw=yaw0, vmax=1.0,
                             v_dn=1.0, v_up=1.0, **nav)

    def _nav_air(self, S, yaw0):
        g = self.g
        x = float(S.p[0])
        if S.depth > 0.0 or S.s.max() > 0.01:
            return self._exit_water(S, yaw0, g["h_air"])
        zoff = S.p[2] - S.clr - self._surf(x)
        if not self.air_up:
            if S.clr > 1.0:
                self.air_up = True
            else:
                self.mode = "take-off"
                xy = self._latch("nav_to_xy", lambda: S.p[:2].copy())
                return self._fly_air(S, p_ref=xy, z_ref=S.p[2] + (g["h_air"] - S.clr), yaw=yaw0,
                                     vmax=1.0, v_dn=1.0, v_up=3.0, tilt=20.0)
        self.mode = "air-nav"
        v_nav = self._latch("nav_v", lambda: float(np.clip(
            max(np.linalg.norm(S.v[:2]), g["v_nav"]), g["v_nav"], 9.0)))
        z_lat = self._latch("nav_z", float(S.p[2]))
        dvec = P_STATION - S.p[:2]
        dist = float(np.linalg.norm(dvec))
        v_cmd = dvec / max(dist, 1e-6) * min(v_nav, g["kpos"] * dist)
        z_floor = self._surf(x) + zoff + g["h_air"]
        z_ref = max(z_floor, min(z_lat, z_floor + dist))
        return self._fly_air(S, v_des=v_cmd, z_ref=z_ref, yaw=yaw0, vmax=0.0, v_dn=6.0, v_up=3.0,
                             vz_ff=self._slope(x) * float(S.v[0]))

    def _nav_water(self, S, yaw0):
        g = self.g
        x = float(S.p[0])
        wet_all, wet_any = S.s.min() > 0.99, S.s.max() > 0.01
        if wet_all and S.depth > 0.0:
            self.cut = True
            if x >= X_SHELF - 0.4:
                self.mode = "rest-on-shelf"
                return self._off()
            self.mode = "water-hold"
            near = float(np.linalg.norm(P_STATION - S.p[:2])) < 20.0
            xy = P_STATION if near else self._latch("nav_w_xy", lambda: S.p[:2].copy())
            return self._fly_water(S, p_ref=xy, z_ref=-g["depth_nav"], yaw=yaw0, vmax=0.5,
                                   tilt=45.0)
        if wet_any or S.depth > 0.0 or self.cut:
            self.mode = "sink"
            return self._off()
        zoff = S.p[2] - S.clr - self._surf(x)
        if x > -1.0:
            self.mode = "to-water"
            return self._fly_air(S, p_ref=P_STATION, z_ref=self._surf(x) + zoff + 1.5, yaw=yaw0,
                                 vmax=g["v_tr"], v_dn=2.0, v_up=3.0, kpos=g["kpos_nav"],
                                 kv=g["kv_nav"], a_brake=g["abrake"])
        if S.clr < g["cut_clear"]:
            self.cut = True
            self.mode = "cut"
            return self._off()
        self.mode = "descend"
        return self._fly_air(S, v_des=np.zeros(2), z_ref=S.p[2] - 5.0, yaw=yaw0, vmax=0.0,
                             v_dn=g["vdesc"], v_up=1.0)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def make_env(body: str, seed: int):
    p = build(BODIES[body]())
    env = TriphibianEnv(p, seed=seed)
    # Body-level facts the air score reads, measured on the body's own gait
    # before any controller is installed (they are cached on the phenotype).
    facts = {"launch_speed": float(env.launch_speed), "level_margin": env.level_margin(),
             "flies_level": bool(env.flies_level()), "air_gates": list(env.air_gates)}
    tm = env.thrust_margin()
    facts["thrust_margin"] = None if tm is None else float(tm)
    return p, env, facts


def controller_for(env, arm: str, record: bool, gains=None):
    """``hand``: the hand controller.  ``still``: `held_still_params`, rotors stopped."""
    if arm == "still":
        return Controller(params=env.held_still_params(), policy=None, bases=None), None
    h = Hand(env, record=record, gains=gains)
    return h.ctrl, h


def _energy(env) -> dict:
    b = env.budget
    return {"soc_end": round(float(b.battery.soc), 6), "booked_wh": round(float(b.total_j) / 3600.0, 6),
            "mean_power_w": round(float(b.mean_power), 3),
            "max_overload": round(float(b.max_overload), 4),
            "peak_draw_w": round(float(b.battery.peak_draw), 2)}


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 6)
        elif isinstance(v, dict):
            out[k] = _clean(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [round(float(x), 6) if isinstance(x, (float, np.floating)) else x for x in v]
        else:
            out[k] = v
    return out


def run_seg_item(env, ctrl, dom: Domain, seed: int) -> dict:
    """One scored segment, wired as `evaluate.evaluate_tier1` wires it."""
    sseed = _scatter_seed(seed, dom)
    task = schedule_for(dom, np.random.default_rng(task_seed(sseed)))
    seg = run_segment(env, dom, SEGMENT_SECONDS, ctrl, scatter_seed=sseed, task=task)
    row = {"score": float(seg.competence), "failure": seg.failure, "bad_qacc": int(seg.bad_qacc),
           "survived": bool(seg.survived), "parts": _clean(dict(seg.parts)),
           "task": [(ph.kind, round(ph.heading, 3), round(ph.speed, 3), ph.depth)
                    for ph in task.phases],
           "m": _clean({k: v for k, v in seg.measurements.items() if k in (
               "task_score", "turn_response", "pair_turn", "pair_turn_mean", "pair_turn_factor",
               "height_hold", "cruise_height_hold", "turn_height_hold", "airborne_fraction",
               "measured_sink_rate", "spin_rate", "air_gates", "level_margin",
               "pair_along", "pair_along_mean", "pair_across", "pair_second", "cruise_progress",
               "cruise_tracking", "hold_score", "hold_depth_error", "hold_drift",
               "hold_sink_rate", "cmd_depth", "cmd_heading", "task_hold_first", "max_depth",
               "walk_progress", "walk_tracking", "stop_score", "stop_drift", "upright",
               "contact_fraction", "measured_contact_fraction", "task_walk_first",
               "measured_land_speed", "measured_takeoff_height")})}
    row.update(_energy(env))
    return row


def run_cross_item(env, ctrl, kind: str) -> dict:
    tr = run_transition(env, kind, ctrl)
    bad = int(env.data.warning[_BAD].number)      # `_place_for` reset the counter
    row = {"score": float(tr.approach), "crossed": bool(tr.crossed), "failure": tr.failure,
           "bad_qacc": bad, "hold": float(tr.hold), "started_in": tr.started_in,
           "components": _clean(tr.components),
           "m": _clean({"peak_entry_speed": tr.peak_entry_speed,
                        "survivable_entry_speed": tr.survivable_entry_speed,
                        "min_upright": tr.min_upright, "exit_depth": tr.exit_depth,
                        "exit_upright": tr.exit_upright, "peak_clearance": tr.peak_clearance,
                        "go_peak_clearance": tr.go_peak_clearance,
                        "aloft_fraction": tr.aloft_fraction, "shore_progress": tr.shore_progress,
                        "slam_pressure": tr.slam_pressure, "slam_capacity": tr.slam_capacity,
                        "end_x": float(env.root_pos()[0]), "end_z": float(env.root_pos()[2]),
                        "end_clearance": float(env.clearance())})}
    if bad > 0 and not row["failure"]:
        row["failure"] = "unstable (bad_qacc; run_transition does not read the counter)"
    row.update(_energy(env))
    return row


def run_mission_item(env, ctrl, start: Domain) -> dict:
    i0 = DOMAIN_CYCLE.index(start)
    schedule = [(DOMAIN_CYCLE[(i0 + k) % 3], LEG_SECONDS) for k in range(3)]
    res = run_continuous(env, ctrl, schedule)
    row = {"score": float(continuous_score(res, MissionSpec())),
           "transitions": [int(res.transitions_completed), int(res.transitions_commanded)],
           "failure": res.failure, "survived": bool(res.survived),
           "bad_qacc": int(env.data.warning[_BAD].number),
           "legs": [{"cmd": leg.commanded.value, "entered": bool(leg.entered),
                     "entry_time": None if leg.entry_time is None else round(leg.entry_time, 3),
                     "on_task": round(leg.on_task_fraction, 4),
                     "mean_power": round(leg.mean_power, 2)} for leg in res.legs],
           "m": _clean({"on_task": res.on_task, "max_depth": res.max_depth,
                        "energy_wh": res.energy_wh, "peak_slam": res.peak_slam,
                        "end_x": float(env.root_pos()[0]), "end_z": float(env.root_pos()[2])})}
    row.update(_energy(env))
    return row


def save_traj(hand: Hand, path: Path, meta: dict) -> None:
    if hand is None or not hand.rec:
        return
    t = np.array([r[0] for r in hand.rec])
    np.savez_compressed(
        path, time=t, qpos=np.array([r[1] for r in hand.rec]),
        qvel=np.array([r[2] for r in hand.rec]),
        commanded_medium=np.array([r[3] for r in hand.rec], dtype=np.int8),
        rotor_cmd=np.array([r[4] for r in hand.rec]),
        episode=np.cumsum(np.r_[0, (np.diff(t) < 0).astype(int)]).astype(np.int16),
        meta=json.dumps(meta))


def parse_seeds(s: str) -> list:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def cmd_facts(args) -> int:
    p = build(BODIES[args.body]())
    env = TriphibianEnv(p, seed=0)
    out = {"body": args.body, "summary": p.summary(),
           "mass": p.mass, "budget": {k: round(v, 4) for k, v in p.budget.as_dict().items()},
           "density_ratio": p.density_ratio, "buoyant_L": p.buoyant_volume * 1e3,
           "battery": [p.genome.battery_wh, p.genome.battery_chem, round(p.battery.mass, 4),
                       p.battery.p_max],
           "tier0": tier0_gate(p),
           "rotor_throttle": [float(q.rotor_throttle) for q in p.genome.parts if q.rotor_radius > 0],
           "launch_speed": env.launch_speed, "level_margin": env.level_margin(),
           "flies_level": env.flies_level(), "air_gates": env.air_gates,
           "act_names": env.act_names, "max_entry_speed": p.max_entry_speed}
    env.reset(Domain.LAND, randomise=False)
    out["land_rest"] = {"root_z": float(env.data.qpos[2]), "clearance": float(env.clearance()),
                        "ncon_in_flight": int(env.data.ncon)}
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_run(args) -> int:
    items = args.items.split(",")
    seeds = parse_seeds(args.seeds)
    gains = json.loads(args.gains) if args.gains else None
    out_path = HERE / ("trials.jsonl" if args.trial else f"results_{args.out}.jsonl")
    for seed in seeds:
        p, env, facts = make_env(args.body, seed)
        for item in items:
            t0 = time.time()
            ctrl, hand = controller_for(env, args.arm, args.save_traj, gains)
            if item in ("air", "water", "land"):
                row = run_seg_item(env, ctrl, Domain(item), seed)
            elif item in CROSSINGS:
                row = run_cross_item(env, ctrl, item)
            elif item.startswith("mission_"):
                row = run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
            else:
                raise SystemExit(f"unknown item {item}")
            if hand is not None:
                env.cpg.command = hand._orig         # uninstall before the next item
            row = {"body": args.body, "arm": args.arm, "item": item, "seed": seed, **row,
                   "facts": facts, "wall_s": round(time.time() - t0, 1)}
            if args.trial:
                row = {"trial": args.trial, "gains": gains or {}, **row}
            with open(out_path, "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            brief = {k: row[k] for k in ("body", "arm", "item", "seed", "score", "failure",
                                         "bad_qacc", "soc_end", "booked_wh", "wall_s")}
            for k in ("crossed", "hold", "transitions"):
                if k in row:
                    brief[k] = row[k]
            brief["score"] = round(brief["score"], 4)
            print(json.dumps(brief), flush=True)
            if args.verbose:
                print("   ", json.dumps({k: row[k] for k in ("m", "parts", "task", "legs",
                                                             "components") if k in row}),
                      flush=True)
            if args.save_traj and hand is not None:
                (HERE / "trajectories").mkdir(exist_ok=True)
                save_traj(hand, HERE / "trajectories" / f"{args.body}_{item}_seed{seed}.npz",
                          {k: row[k] for k in ("body", "arm", "item", "seed", "score", "failure")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--body", default="B1")
    r = sub.add_parser("run")
    r.add_argument("--body", default="B1")
    r.add_argument("--arm", default="hand", choices=("hand", "still"))
    r.add_argument("--items", default="air,water,land")
    r.add_argument("--seeds", default="0-7")
    r.add_argument("--out", default="final")
    r.add_argument("--trial", default="", help="tuning rollout: what changed (-> trials.jsonl)")
    r.add_argument("--gains", default="", help="JSON overrides of Hand.g (trials only)")
    r.add_argument("--save-traj", action="store_true")
    r.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return cmd_facts(args) if args.cmd == "facts" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
