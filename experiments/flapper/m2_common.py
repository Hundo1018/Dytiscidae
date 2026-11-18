"""Pass M2 (PREREG_M2_water.md): machines, the law family, the recorder.

Builds the M0 machine (`mission.Machine`, not edited) per section 1 of the
pre-registration, runs a law on it and records what the gates read.

    from m2_common import draw, build, run_law
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

X0 = -30.0            # every start (section 1)
DEPTH0 = 3.0          # GW start depth of the CoM, m
VB_SCALE = 0.584      # VB: sealed volume factor (section 2)
CTRL_DT = 0.04        # the law is asked at 25 Hz
SAMPLE_DT = 0.1       # the gates read 10 Hz samples


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------ the draw
def draw(seed: int) -> dict:
    """What a seed sets (section 1), in a fixed order."""
    rng = np.random.default_rng(10_000 + int(seed))
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-9)
    ang = float(rng.normal(0.0, 0.18))
    dx, dy = (float(v) for v in rng.normal(0.0, 0.2, size=2))
    phase = float(rng.uniform(0.0, 2 * math.pi))
    h = float(rng.uniform(0.0, 2 * math.pi))
    delta = float(rng.uniform(0.8, 1.5))
    return dict(seed=int(seed), axis=axis.tolist(), ang=ang, dx=dx, dy=dy, phase=phase, h=h, delta=delta)


def _quat(pitch_deg: float, dr: dict) -> np.ndarray:
    a = math.radians(pitch_deg)
    q = np.array([math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0])
    half = 0.5 * dr["ang"]
    ax = np.array(dr["axis"])
    dq = np.array([math.cos(half), *(math.sin(half) * ax)])
    out = np.array([
        dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
        dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
        dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
        dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0],
    ])
    return out / np.linalg.norm(out)


# ------------------------------------------------------------------ building
def build(start: str, dr: dict, *, gains: str, vb: bool = False, entrainment="wetting",
          motors: dict | None = None, dt: float = 1e-4, float_z: float = 0.0):
    """A machine at a start state.

    start: "depth" (CoM at DEPTH0, level + scatter) or "float" (level + scatter,
    hull axis at ``float_z``, to be settled by the caller).  gains: "here"
    (built at the start, the PD sees that medium) or "air" (built at z = 5 m in
    air at the start attitude, then placed, `solver.reset()`).
    """
    from mission import Machine
    q = _quat(0.0, dr)
    x, y = X0 + dr["dx"], dr["dy"]
    if gains == "air":
        mc = Machine("air", motors=motors, dt=dt, pose=dict(x=x, y=y, z=5.0))
    else:
        mc = Machine("water", motors=motors, dt=dt, pose=dict(x=x, y=y, z=-DEPTH0))
    d = mc.d
    d.qpos[3:7] = q
    mc.mj.mj_forward(mc.m, d)
    if start == "depth":
        d.qpos[2] += -DEPTH0 - float(mc.com()[2])
    else:
        d.qpos[2] = float_z
    d.qvel[:] = 0.0
    mc.mj.mj_forward(mc.m, d)
    s = mc.env.solver
    if vb:
        s.panels.volume_buoyant = s.panels.volume_buoyant * VB_SCALE
    s.entrainment = entrainment
    if gains == "here" and start == "float":
        # rebuild the gains at the float pose (the PD sees what is wet there)
        _regain(mc)
    s.reset()
    mc.mj.mj_forward(mc.m, d)
    return mc


def _regain(mc, wn=300.0):
    m, d, mj = mc.m, mc.d, mc.mj
    mc.env.solver.reset()
    mj.mj_forward(m, d)
    mc.env.solver.apply(d, d.time)
    mj.mj_forward(m, d)
    J = np.zeros(len(mc.dof))
    for n, i in enumerate(mc.dof):
        e, col = np.zeros(m.nv), np.zeros(m.nv)
        e[i] = 1.0
        mj.mj_mulM(m, d, col, e)
        J[n] = col[i]
    mc.kp, mc.kv = J * wn * wn, 2.0 * J * wn
    for n, a in enumerate(mc.aid):
        m.actuator_gainprm[a, 0] = mc.kp[n]
        m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = -mc.kp[n], -mc.kv[n]
    d.xfrc_applied[:] = 0.0


# ------------------------------------------------------------------ state
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def panel_sub(mc) -> np.ndarray:
    s, p, d = mc.env.solver, mc.env.solver.panels, mc.d
    R = d.xmat.reshape(-1, 3, 3)[p.body_id]
    pos = d.xpos[p.body_id] + np.einsum("nij,nj->ni", R, p.pos_local)
    return mc.env.medium.submerged_fraction(pos, p.half_height, d.time), pos


def state(mc) -> dict:
    d = mc.d
    R = d.xmat[mc.hull].reshape(3, 3)
    nose, dors = R[:, 0], R[:, 2]
    com = mc.com()
    w = d.qvel[3:6].copy()          # free joint: angular velocity, body frame
    yaw = math.atan2(nose[1], nose[0])
    side = R[:, 1]
    roll = math.atan2(-side[2], dors[2])
    return dict(t=mc.t, com=com, v=d.qvel[:3].copy(), nose=nose, dors=dors, side=side, w=w,
                yaw=yaw, pitch=math.degrees(math.asin(max(-1, min(1, nose[2])))), roll=roll,
                depth=-float(com[2]), q=d.qpos[mc.qad].copy())


# ------------------------------------------------------------------ the law family
class Law:
    """stroke_i = b + s_i A/2 sin(phi + p_i); feather_i = psi0 + d_i - Psi cos(phi + chi);
    tail = e +- a.  ``update(st, cmd)`` (25 Hz) sets the slow terms from the state.
    All angles in radians.  Subclasses hold the feedback."""

    name = "base"

    def __init__(self, **g):
        self.g = dict(self.defaults(), **g)
        self.f = self.g.get("f", 0.2)
        self.A = self.g.get("A", 0.0)
        self.b = self.g.get("b", 0.0)
        self.Psi = self.g.get("Psi", 0.0)
        self.chi = self.g.get("chi", 0.0)
        self.psi0 = self.g.get("psi0", 0.0)
        self.s1 = self.s2 = 1.0
        self.p1 = self.p2 = 0.0
        self.d1 = self.d2 = 0.0
        self.e = self.a = 0.0
        self.phi = self.g.get("phase0", 0.0)
        self.ramp = self.g.get("ramp", 1.0)
        self.t0 = None

    def defaults(self):
        return {}

    def update(self, st, cmd):
        pass

    def targets(self, mc, dt):
        if self.t0 is None:
            self.t0 = mc.t
        r = min((mc.t - self.t0) / max(self.ramp, 1e-9), 1.0)
        w = 2 * math.pi * self.f
        self.phi += w * dt
        q = np.zeros(len(mc.aid))
        qd = np.zeros(len(mc.aid))
        rng = mc.m.jnt_range[mc.jid]
        k = 0
        for n, role in enumerate(mc.role):
            if role == "stroke":
                first = (k == 0)          # seg1, the +y (port) wing
                s, p = (self.s1, self.p1) if first else (self.s2, self.p2)
                q[n] = r * (self.b + s * self.A / 2 * math.sin(self.phi + p))
                qd[n] = r * s * self.A / 2 * w * math.cos(self.phi + p)
            elif role == "feather":
                first = (k == 1)
                dd = self.d1 if first else self.d2
                q[n] = r * (self.psi0 + dd - self.Psi * math.cos(self.phi + self.chi))
                qd[n] = r * self.Psi * w * math.sin(self.phi + self.chi)
            else:
                tail_r = (k == 4)
                q[n] = self.e + (self.a if tail_r else -self.a)
            k += 1
            lo, hi = rng[n]
            if q[n] <= lo or q[n] >= hi:
                q[n] = min(max(q[n], lo + 1e-3), hi - 1e-3)
                qd[n] = 0.0
        return q, qd


class Still(Law):
    name = "still"

    def targets(self, mc, dt):
        n = len(mc.aid)
        return np.zeros(n), np.zeros(n)


# ------------------------------------------------------------------ the recorder
class Recorder:
    def __init__(self, mc):
        self.mc = mc
        self.trace = []
        self.win = deque(maxlen=int(round(0.05 / mc.dt)) + 1)
        self.peak_step = 0.0
        self.peak_win = 0.0
        self.E0_elec = mc.E_elec.copy()
        self.E0_shaft = mc.E_shaft.copy()
        self.lim0 = mc.at_limit.copy()
        self.n0 = mc.n
        self.t0 = mc.t
        self._E = float(mc.E_elec.sum())
        self.win.append(self._E)
        self.clamped_steps = 0           # ADDED after data: the solver's 60-weight fluid-force limiter engaged
        self.slam_max = 0.0

    def step(self):
        mc = self.mc
        E = float(mc.E_elec.sum())
        dg = mc.env.solver.diag
        if dg.clamped:
            self.clamped_steps += 1
            dg.clamped = False
        self.slam_max = max(self.slam_max, float(dg.slam))
        self.peak_step = max(self.peak_step, (E - self._E) / mc.dt)
        self._E = E
        self.win.append(E)
        if len(self.win) == self.win.maxlen:
            self.peak_win = max(self.peak_win, (self.win[-1] - self.win[0]) / ((len(self.win) - 1) * mc.dt))

    def sample(self, extra=None):
        mc = self.mc
        st = state(mc)
        sub, pos = panel_sub(mc)
        row = dict(t=round(mc.t - self.t0, 4), com=[round(float(v), 4) for v in st["com"]],
                   nose=[round(float(v), 4) for v in st["nose"]], dors=[round(float(v), 4) for v in st["dors"]],
                   w=[round(float(v), 3) for v in st["w"]], wet=int((sub > 0).sum()),
                   wet_main=int(((sub > 0) & np.isin(mc.env.solver.panels.body_id, [2, 3])).sum()),
                   low=round(float(pos[:, 2].min()), 4), q=[round(float(v), 3) for v in st["q"]],
                   E=round(float(mc.E_elec.sum() - self.E0_elec.sum()), 3))
        if extra:
            row.update(extra)
        self.trace.append(row)

    def summary(self):
        mc = self.mc
        T = max(mc.t - self.t0, 1e-9)
        n = max(mc.n - self.n0, 1)
        roles = sorted(set(mc.role))
        dE = mc.E_elec - self.E0_elec
        dS = mc.E_shaft - self.E0_shaft
        dl = mc.at_limit - self.lim0
        return dict(T=T, E_elec_J=float(dE.sum()), P_elec_mean_W=float(dE.sum() / T),
                    P_elec_role_W={r: float(dE[mc.role == r].sum() / T) for r in roles},
                    P_shaft_role_W={r: float(dS[mc.role == r].sum() / T) for r in roles},
                    at_limit={r: float(dl[mc.role == r].mean() / n) for r in roles},
                    P_peak_50ms_all_W=self.peak_win, P_peak_step_all_W=self.peak_step,
                    bad_qacc=int(mc.bad_qacc), clamped_steps=self.clamped_steps, slam_max_N=self.slam_max)


def run_law(mc, law, T: float, cmd: dict | None = None, rec: Recorder | None = None, on_sample=None):
    """Run ``law`` for T s; law.update at 25 Hz, samples at 10 Hz."""
    rec = rec or Recorder(mc)
    cmd = cmd or {}
    nctl = int(round(CTRL_DT / mc.dt))
    nsm = int(round(SAMPLE_DT / mc.dt))
    N = int(round(T / mc.dt))
    rec.sample()
    for k in range(N):
        if k % nctl == 0:
            law.update(state(mc), cmd)
        q, qd = law.targets(mc, mc.dt)
        mc.step(q, qd)
        rec.step()
        if (k + 1) % nsm == 0:
            rec.sample()
            if on_sample is not None and on_sample(rec):
                break
        if not np.all(np.isfinite(mc.d.qpos)):
            break
    return rec


def settle(mc, T=3.0):
    """Joints held at 0 for T s (the float start; not scored)."""
    return run_law(mc, Still(), T)


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
