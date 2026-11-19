"""Pass S0: the laws S0 adds on top of the frozen ones (PREREG_S0 section 2).

- `HoverEntry` (A4): M4's cruise (law L, C1's pilot) until the CoM passes the
  commanded place x_e, then V+ (`m1_takeoff.Fly`, V+ gains verbatim) with the
  height read above the surface under the CoM and its target falling at
  0.5 m/s from the switch, until first contact; then M4's frozen contact
  hold (`EntryFlight.contact_hold`).  It slots into `m4_lib.run_half` in
  place of `EntryFlight` (same interface).
- `water_row(spec)`: `m2b_run.run_row` with the law, start and command given
  in the spec (the dive and surface legs, and A5w's tuned laws), built by the
  same `m2b_laws.build2`, settled, run by `m2_common.run_law`.
- `ballast_law` (A3) and the sweep attitude feedback (A5p.1) as
  ``mc.sweep_fn`` / a per-step hook, used only if those rungs are taken.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m4_common as C  # noqa: E402
import m4_lib  # noqa: E402

DESCENT = 0.5           # m/s, A4's commanded descent


class HoverEntry(m4_lib.EntryFlight):
    def __init__(self, mc, pt, E, x_e, ground, V=8.0, z_ref=None):
        super().__init__(mc, pt, E, x_e, ground, V=V, z_ref=z_ref)
        self.fly = None
        self.t_sw = None
        self.H_sw = None
        self.q_last = np.zeros(len(mc.aid))
        self.log = []

    def H(self):
        c = self.mc.com()
        return float(c[2] - self.ground(float(c[0])))

    def step(self):
        if not self.armed or self.hold is not None:
            return C.Flight.step(self)
        mc = self.mc
        if self.fly is None:
            if float(mc.com()[0]) < self.x_e:
                self.phase = "cruise"
                return C.Flight.step(self)
            import m1_takeoff as T
            g = dict(T.GAINS, **T.PLUS)
            self.fly = T.Fly(mc, g, loop=True)
            self.fly.t0 = mc.t
            self.t_sw, self.H_sw = mc.t, self.H()
            mc.set_twist(0.0, 0.0)
            self.phase = "hover"
            self.log.append(("hover", round(mc.t, 3), round(self.H_sw, 3)))
            # the targets of the last cruise step, for the blend (M1's blend_s)
            self.q_last = np.array(mc.d.ctrl[mc.aid])
        g = self.fly.g
        g["H_t"] = self.H_sw - DESCENT * (mc.t - self.t_sw)
        self.fly.sample(self.H())
        qf, qdf = self.fly.targets(mc.t)
        b = min((mc.t - self.t_sw) / g["blend_s"], 1.0)
        q = (1 - b) * self.q_last + b * qf
        mc.step(q, b * qdf)
        self.k += 1


def water_row(spec):
    """spec: arm, seed, half, mode, start (float|depth), gains (air|here), law (m2b make2 spec), cmd, T."""
    from m2_common import draw, run_law, settle
    from m2b_laws import build2, make2
    t0 = time.time()
    dr = draw(spec["seed"])
    mc = build2(spec["start"], dr, gains=spec["gains"], vb=False, entrainment=spec["mode"])
    if spec["start"] == "float":
        settle(mc, 3.0)
    law = make2(spec["law"])
    tgt = getattr(law, "law", law)
    if hasattr(tgt, "phi"):
        tgt.phi = dr["phase"]
    cmd = dict(spec.get("cmd") or {})
    if spec.get("cmd_from_draw"):
        cmd = cmd_from_draw(spec["cmd_from_draw"], spec["half"], dr)
    hook = spec.get("hook")
    if hook:
        install_hook(mc, hook)
    t_start = mc.t
    rec = run_law(mc, law, spec["T"], cmd)
    return mc, dict(cmd=cmd, draw=dr, t_start=t_start, summary=rec.summary(), wall_s=round(time.time() - t0, 1),
                    log=getattr(tgt, "log", None), trace=rec.trace)


def cmd_from_draw(kind, half, dr):
    """The commands of each water leg from the seed's draw (m2b_run.command_for and the S0 legs)."""
    h = dr["h"] if half != "B" else dr["h"] + math.pi
    if kind == "GW":
        return dict(h=h, z=3.0 + dr["delta"]) if half != "B" else dict(h=h, z=3.0 - dr["delta"])
    if kind == "GS":
        return dict(h=h)
    if kind == "D":            # dive from the float start to 3 m
        return dict(h=dr["h"], z=3.0)
    if kind == "U":            # come up from 3 m to the surface
        return dict(h=dr["h"], z=0.0)
    return {}


# --------------------------------------------------------------- A3, A5p.1 hooks
def install_hook(mc, hook):
    """hook: dict(kind="sweep_att", kp, kd) (A5p.1) and/or dict(kind="ballast", k, z, ...) (A3)."""
    if hook.get("kind") == "sweep_att":
        kp, kd = math.radians(hook["kp"]), math.radians(hook["kd"])
        names = list(mc.env.act_names)
        sw = np.nonzero(mc.role == "sweep")[0]
        left = np.array(["seg1" in names[i] for i in sw])
        ssign = np.where(left, -1.0, 1.0)         # +sweep = tip forward (m3b_land.Controller)

        def fn(mc_):
            from m2_common import state
            st = state(mc_)
            pitch = math.radians(st["pitch"])
            w = st["w"]
            roll = st["roll"]
            com = hook.get("sp", 1.0) * (kp * pitch + kd * w[1])
            dif = hook.get("sr", 1.0) * (kp * roll + kd * w[0])
            sw_deg = np.clip(np.degrees(com + np.where(left, dif, -dif)), -32.0, 32.0)
            return np.radians(sw_deg) * ssign, np.zeros(len(sw))
        mc.sweep_fn = fn
