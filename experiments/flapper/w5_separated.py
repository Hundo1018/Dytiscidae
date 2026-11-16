"""Pass W5: the solver's separated-flow branch against measurements at
Re 1e4-2.5e5 (PREREG_W5_separated.md; data in sections/separated/).

Two rigs, both driving `FluidSolver.apply` as the flapper rig does (inflow and
unsteady state on, nothing fitted):

* `revolve`: one rectangular wing revolving at constant rate and fixed
  incidence in air.  `experiments/robofly/run.py` is the same experiment in
  oil at Re 136 with its geometry fixed in the module; this one takes the
  radius, chord, Reynolds number and section, and returns the vertical and
  horizontal force coefficients on the second moment of area (Usherwood's
  Cv, Ch).
* `translate`: a rectangular wing of a given aspect ratio at constant speed
  and incidence, CL and CD on its area.

    PYTHONPATH=. .venv/bin/python experiments/flapper/w5_separated.py > results/W5_separated.jsonl
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dytiscidae.physics.fluid import WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import AIR, MediumField  # noqa: E402

import sections  # noqa: E402

NSTRIP = 24
NU = AIR.mu / AIR.rho


def _wing(m, x, dr, chord, alpha, ar, camber):
    """Strips at spanwise stations ``x`` moving along +Y: leading edge +Y,
    the chord (LE -> TE) down and back for positive incidence, upper side +Z."""
    n = len(x)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wing")
    return bid, PanelSet(
        body_id=np.full(n, bid), pos_local=np.stack([x, np.zeros(n), np.zeros(n)], 1),
        span_local=np.tile([-1.0, 0.0, 0.0], (n, 1)),
        chord_local=np.tile([0.0, -math.cos(alpha), -math.sin(alpha)], (n, 1)),
        chord=np.full(n, chord), dr=np.full(n, dr), volume=np.zeros(n),
        half_height=np.full(n, 0.01), kind=np.full(n, WING),
        aspect_ratio=np.full(n, ar), cd_bluff=np.zeros(n), camber=np.full(n, camber))


def _solver(m, p, span, section, flags):
    sol = FluidSolver(m, p, MediumField(air=AIR), disc_span=span)
    if section:
        sol.section = sections.load(section) if isinstance(section, str) else section
    for k, v in (flags or {}).items():
        setattr(sol, k, v)
    return sol


def revolve(alpha_deg, *, R=0.33, chord=0.10, re=54000.0, section=None, camber=0.0,
            revs=3, flags=None):
    """(Cv, Ch) of a wing of radius ``R`` revolving about its root; ``re`` is
    on the chord and the speed at the radius of gyration."""
    xml = """<mujoco><option timestep="0.0005" gravity="0 0 0"/>
      <worldbody><body name="wing" pos="0 0 5">
        <joint name="h" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.1 0.05 0.001" pos="0.1 0 0" density="100"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dr = R / NSTRIP
    r = dr * (np.arange(NSTRIP) + 0.5)
    bid, p = _wing(m, r, dr, chord, math.radians(alpha_deg), R / chord, camber)
    sol = _solver(m, p, 2 * R, section, flags)
    r2 = math.sqrt(np.mean(r ** 2))
    omega = re * NU / chord / r2
    dt = m.opt.timestep
    per = int(round(2 * math.pi / omega / dt))
    qs = float(np.sum(0.5 * AIR.rho * (omega * r) ** 2 * chord * dr))
    V = H = 0.0
    for k in range(revs * per):
        d.qpos[0], d.qvel[0] = omega * k * dt, omega
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, k * dt)
        if k >= (revs - 1) * per:
            fw = d.xfrc_applied[bid, :3].copy()
            fw[2] -= (m.body_mass[bid] - sol._dry_mass[bid]) * 9.80665
            f = d.xmat[bid].reshape(3, 3).T @ fw
            V += f[2]; H += -f[1]
    return V / per / qs, H / per / qs


def translate(alpha_deg, *, ar=4.0, chord=0.10, re=100000.0, section=None, camber=0.0,
              chords=60.0, flags=None):
    """(CL, CD) of a rectangular wing of aspect ratio ``ar`` in steady flight,
    averaged over the last tenth of ``chords`` of travel."""
    xml = """<mujoco><option timestep="0.0005" gravity="0 0 0"/>
      <worldbody><body name="wing" pos="0 0 5">
        <joint name="s" type="slide" axis="0 1 0"/>
        <geom type="box" size="0.1 0.05 0.001" density="100"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    b = ar * chord
    dr = b / NSTRIP
    x = -b / 2 + dr * (np.arange(NSTRIP) + 0.5)
    bid, p = _wing(m, x, dr, chord, math.radians(alpha_deg), ar, camber)
    sol = _solver(m, p, b, section, flags)
    U = re * NU / chord
    dt = m.opt.timestep
    n = int(round(chords * chord / U / dt))
    q = 0.5 * AIR.rho * U ** 2 * chord * b
    L = D = 0.0
    last = max(n // 10, 1)
    for k in range(n):
        d.qpos[0], d.qvel[0] = U * k * dt, U
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, k * dt)
        if k >= n - last:
            fw = d.xfrc_applied[bid, :3].copy()
            fw[2] -= (m.body_mass[bid] - sol._dry_mass[bid]) * 9.80665
            L += fw[2]; D += -fw[1]
    return L / last / q, D / last / q


def main() -> int:
    out = []
    put = lambda **kw: (out.append(kw), print(json.dumps(kw), flush=True))
    for sec, cam in ((None, 0.0), ("goe417a", 0.06)):
        for re, rc in ((54000.0, 3.3), (14000.0, 4.0), (14000.0, 2.9)):
            for a in range(0, 91, 5):
                cv, ch = revolve(a, R=rc * 0.10, re=re, section=sec, camber=cam)
                put(rig="revolve", section=sec, re=re, r_over_c=rc, alpha=a, cv=cv, ch=ch)
    for ar in (2.0, 3.0, 4.0, 5.0, 6.0):
        for a in (5, 8, 10, 12, 14, 15, 18, 20, 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90):
            cl, cd = translate(a, ar=ar, re=250000.0 if ar == 6.0 else 100000.0)
            put(rig="translate", section=None, ar=ar, alpha=a, cl=cl, cd=cd)
    for a in (10, 14, 18, 20, 25, 30, 35, 40, 45, 50):
        cl, cd = translate(a, ar=6.0, re=200000.0, section="goe417a", camber=0.06)
        put(rig="translate", section="goe417a", ar=6.0, alpha=a, cl=cl, cd=cd)
    for a in (30, 35, 40, 45, 50, 60, 70, 80, 90):
        cl, cd = translate(a, ar=1e4, chord=0.01, re=100000.0, flags=dict(inflow=False))
        put(rig="translate", section=None, ar="2D", alpha=a, cl=cl, cd=cd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
