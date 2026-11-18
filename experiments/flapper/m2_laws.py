"""Pass M2: the laws of the arms (PREREG_M2_water.md section 2), on `m2_common.Law`.

Every law reads the true state (M5 brings modelled sensors).  Angles in
radians inside, degrees in the parameter dicts (keys ending in _deg).
"""
from __future__ import annotations

import math

import numpy as np

from m2_common import Law, Still, wrap

D = math.radians


def _rad(g):
    return {k[:-4] if k.endswith("_deg") else k: (D(v) if k.endswith("_deg") else v) for k, v in g.items()}


class Open(Law):
    """No feedback: the family's terms as given (sign checks, open-loop reads)."""
    name = "open"

    def __init__(self, **g):
        g = _rad(g)
        super().__init__(**g)
        self.e = g.get("e", 0.0)
        self.d1 = g.get("d1", g.get("dfe", 0.0))
        self.d2 = g.get("d2", g.get("dfe", 0.0))
        self.s1, self.s2 = g.get("s1", 1.0), g.get("s2", 1.0)


class Paddle(Law):
    """WS: afloat, the stroke biased down under the hull, the feather half a
    beat from the family's default (chi = pi; T5, T24), which moves the body
    tail first; heading of travel = nose yaw + pi.  Steering by the two wings'
    amplitude share (s1 > s2 turns +yaw, T15) and the feather offset
    difference (d1 - d2 > 0 turns +yaw, T13); pitch by the tail."""
    name = "WS"

    def defaults(self):
        return dict(f=0.25, A=D(40), b=D(-30), Psi=D(30), chi=math.pi, kh=1.0, smin=0.0,
                    kd=D(10), e0=0.0, kpitch=0.0, kq=0.0, ramp=3.0, tailfirst=1)

    def __init__(self, **g):
        super().__init__(**_rad(g))

    def update(self, st, cmd):
        g = self.g
        if "h" in cmd:
            travel = st["yaw"] + (math.pi if g["tailfirst"] else 0.0)
            e = wrap(cmd["h"] - travel)
            u = float(np.clip(g["kh"] * e, -1.0, 1.0))
            self.s1 = float(np.clip(1.0 + u, g["smin"], 1.0))
            self.s2 = float(np.clip(1.0 - u, g["smin"], 1.0))
            dd = float(np.clip(g["kd"] * e, -g["kd"], g["kd"]))
            self.d1, self.d2 = dd, -dd
        th = math.radians(st["pitch"])
        self.e = float(np.clip(g["e0"] + g["kpitch"] * th + g["kq"] * st["w"][1], -0.55, 0.55))


class Scull(Law):
    """The feather oscillating as the propulsor (it drives the body tail first in
    this model, trials T3, T7, T9, T12); the stroke held at a bias b, with a small
    amplitude A whose left-right share steers (WS: stroke biased down at the
    surface; WK: A = 0, under water).  Heading of travel = nose yaw + pi.
    Steering: feather offset difference (d1 - d2 > 0 turns +yaw, T13, T14) and
    the stroke share (s1 > s2 turns +yaw, T15).  Depth (WK): psi0 > 0 raises
    the body going tail first (T16).  Pitch: tail."""
    name = "scull"

    def defaults(self):
        return dict(f=0.45, A=0.0, b=D(-40), Psi=D(40), chi=0.0, kh=1.0, kd=D(15), smin=0.0,
                    kz=D(10), zlim=D(15), e0=0.0, kpitch=0.0, kq=0.0, ramp=1.0, tailfirst=1)

    def __init__(self, **g):
        super().__init__(**_rad(g))

    def update(self, st, cmd):
        g = self.g
        if "h" in cmd:
            travel = st["yaw"] + (math.pi if g["tailfirst"] else 0.0)
            e = wrap(cmd["h"] - travel)
            u = float(np.clip(g["kh"] * e, -1.0, 1.0))
            self.s1 = float(np.clip(1.0 + u, g["smin"], 1.0))
            self.s2 = float(np.clip(1.0 - u, g["smin"], 1.0))
            dd = float(np.clip(g["kd"] * e, -g["kd"], g["kd"]))
            self.d1, self.d2 = dd, -dd
        if "z" in cmd:
            ed = st["depth"] - cmd["z"]
            self.psi0 = float(np.clip(g["kz"] * ed, -g["zlim"], g["zlim"]))
        th = math.radians(st["pitch"])
        self.e = float(np.clip(g["e0"] + g["kpitch"] * th + g["kq"] * st["w"][1], -0.55, 0.55))


class Sweep(Law):
    """WV: the body pitched nose-up (it turns nose-up by itself under water,
    p2, T4), the full stroke sweeping across; depth by the stroke amplitude
    (more amplitude when the body is too shallow), heading by the amplitude
    share and the feather offset difference as in WS; pitch by the tail and the
    common feather offset."""
    name = "WV"

    def defaults(self):
        return dict(f=0.15, A=D(80), Psi=D(40), chi=math.pi, kh=1.0, kd=D(10), smin=0.0,
                    a_lo=0.3, kA=0.7, th_ref=D(60), kth=0.5, e0=0.0, ramp=2.0, tailfirst=1)

    def __init__(self, **g):
        super().__init__(**_rad(g))
        self.A0 = self.A

    def update(self, st, cmd):
        g = self.g
        if "h" in cmd:
            travel = st["yaw"] + (math.pi if g["tailfirst"] else 0.0)
            e = wrap(cmd["h"] - travel)
            u = float(np.clip(g["kh"] * e, -1.0, 1.0))
            self.s1 = float(np.clip(1.0 + u, g["smin"], 1.0))
            self.s2 = float(np.clip(1.0 - u, g["smin"], 1.0))
            dd = float(np.clip(g["kd"] * e, -g["kd"], g["kd"]))
            self.d1, self.d2 = dd, -dd
        if "z" in cmd:
            ed = cmd["z"] - st["depth"]            # + : too shallow, push down harder
            self.A = self.A0 * float(np.clip(g["a_lo"] + g["kA"] * ed, 0.0, 1.0))
        th = math.radians(st["pitch"])
        self.e = float(np.clip(g["e0"] + g["kth"] * (th - g["th_ref"]), -0.55, 0.55))


class Exit(Law):
    """Water to air.  The flying phase is M1's `Fly` (copied in `m2_fly.py`):
    K1 with its attitude and climb loop, height read as the CoM's height above
    the surface (H_t = 4 m).  What comes before it is the arm:

    XB  from float: Fly from the start with the stroke biased up by b_up
        while any main-wing strip is wet (then the bias decays over b_decay s).
    XV  from float (ADDED after data, trials T19, T20: a down stroke into the
        water threw the body clear): M1's vault, the stroke stepped to
        q_strike, Fly when the stroke passes switch_q or after switch_s.
    XD  from depth: joints held (the wings edge-on to a nose-up rise) until the
        shoulder is h_sw above the surface, then the vault (pre="vault") or
        Fly with XB's bias (pre="fly").
    """
    name = "exit"

    def defaults(self):
        return dict(mode="XV", q_strike=-1.35, switch_q=-1.2, switch_s=0.25, blend_s=0.1,
                    b_up=0.0, b_decay=0.5, h_sw=0.15, wait_max=35.0, pre="vault", fly={})

    def __init__(self, **g):
        super().__init__(**g)
        self.phase = {"XB": "fly", "XV": "proc", "XD": "wait"}[self.g["mode"]]
        self.fly = None
        self.t_ph = None
        self.q_proc = None
        self.wet_main = 99
        self.k = 0
        self.t_clear = None
        self.log = []

    def _wet(self, mc):
        from m2_common import panel_sub
        sub, _ = panel_sub(mc)
        return int(((sub > 0) & np.isin(mc.env.solver.panels.body_id, [2, 3])).sum())

    def _start_fly(self, mc):
        import m2_fly as F
        g = dict(F.GAINS, **F.PLUS)
        g.update(self.g["fly"])
        self.fly = F.Fly(mc, g, loop=True)
        self.fly.t0 = mc.t
        self.phase, self.t_ph = "fly", mc.t
        self.log.append(("fly", round(mc.t, 3)))

    def targets(self, mc, dt):
        g = self.g
        n = len(mc.aid)
        st = mc.role == "stroke"
        if self.t0 is None:
            self.t0 = mc.t
            self.t_ph = mc.t
            self.q_proc = np.zeros(n)
            if self.phase == "fly":
                self._start_fly(mc)
        if self.k % 100 == 0:
            self.wet_main = self._wet(mc)
            if self.wet_main == 0 and self.t_clear is None:
                self.t_clear = mc.t
        self.k += 1
        q, qd = np.zeros(n), np.zeros(n)
        if self.phase == "wait":
            sh = float(np.mean([mc.d.xanchor[j][2] for j, r in zip(mc.jid, mc.role) if r == "stroke"]))
            if sh >= g["h_sw"] or mc.t - self.t0 >= g["wait_max"]:
                self.log.append(("surface", round(mc.t, 3), round(sh, 3)))
                if g["pre"] == "vault":
                    self.phase, self.t_ph = "proc", mc.t
                else:
                    self._start_fly(mc)
            return q, qd
        if self.phase == "proc":
            self.q_proc = np.where(st, g["q_strike"], 0.0)
            qs = float(mc.d.qpos[mc.qad[st]].mean())
            if g.get("switch_clear"):
                go = (self.wet_main == 0) or mc.t - self.t_ph >= g["switch_s"]
            else:
                go = qs <= g["switch_q"] or mc.t - self.t_ph >= g["switch_s"]
            if go:
                self._start_fly(mc)
            else:
                return self.q_proc.copy(), qd
        H = float(mc.com()[2])
        self.fly.sample(H)
        qf, qdf = self.fly.targets(mc.t)
        b = min((mc.t - self.t_ph) / g["blend_s"], 1.0) if g["mode"] != "XB" else 1.0
        q = (1 - b) * self.q_proc + b * qf
        qd = b * qdf
        if g["b_up"]:
            if self.wet_main > 0 or self.t_clear is None:
                w = 1.0
            else:
                w = max(0.0, 1.0 - (mc.t - self.t_clear) / g["b_decay"])
            q[st] += w * g["b_up"]
        lo, hi = mc.m.jnt_range[mc.jid].T
        q = np.minimum(np.maximum(q, lo + 1e-3), hi - 1e-3)
        return q, qd


class Frozen(Law):
    """A law with its command frozen at the first one it is given (SF, WF)."""

    def __init__(self, law: Law):
        self.law = law
        self.cmd = None
        self.name = law.name + "-frozen"

    def update(self, st, cmd):
        if self.cmd is None:
            self.cmd = dict(cmd)
        self.law.update(st, self.cmd)

    def targets(self, mc, dt):
        return self.law.targets(mc, dt)


LAWS = dict(open=Open, WS=Paddle, scull=Scull, WV=Sweep, exit=Exit, still=Still)


def make(spec: dict) -> Law:
    kind = spec["law"]
    if kind == "still":
        return Still()
    law = LAWS[kind](**spec.get("g", {}))
    if spec.get("frozen"):
        return Frozen(law)
    return law
