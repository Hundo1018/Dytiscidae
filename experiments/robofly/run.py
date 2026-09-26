#!/usr/bin/env python3
"""The fluid model against Dickinson's robofly -- ROADMAP item AF.

A wing revolving at constant rate and fixed incidence is the experiment behind
the most-used quasi-steady coefficients in flapping flight:

    Dickinson MH, Lehmann F-O, Sane SP (1999) Wing rotation and the aerodynamic
    basis of insect flight.  Science 284:1954-1960.
      CL(a) = 0.225 + 1.58 sin(2.13 a - 7.20 deg)
      CD(a) = 1.92  - 1.55 cos(2.04 a - 9.82 deg)       (a in degrees)

(fits as reproduced by later papers; runs/_logs/aero_literature_0923.md item 1).
The robofly was a fly-shaped wing in mineral oil at Re ~ 1e2; the wing here is
rectangular, R = 0.25 m, c = 0.086 m, so R/c = 2.9, the Rossby number Lentink &
Dickinson (2009, J Exp Biol 212:2705) give for the fly; Re = U(r2) c / nu = 136.

It drives `FluidSolver.apply` through three revolutions at each incidence and
averages the third, in the wing's own frame: lift is +Z, drag opposes the
tangential motion, both normalised by sum(q S) over the strips -- the
blade-element form of Dickinson's r2 normalisation.  Nothing is fitted here;
the model is run as the search runs it.

    PYTHONPATH=. python experiments/robofly/run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402

from dytiscidae.physics.fluid import WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import AIR, Fluid, MediumField  # noqa: E402

OIL = Fluid("mineral oil", 880.0, 880.0 * 115e-6)   # 115 cSt, the robofly's
R, C, NSTRIP, ROOT = 0.25, 0.086, 24, 0.0
DT = 0.004
REVS = 3


def dickinson(a_deg):
    a = np.asarray(a_deg, float)
    return (0.225 + 1.58 * np.sin(np.radians(2.13 * a - 7.20)),
            1.92 - 1.55 * np.cos(np.radians(2.04 * a - 9.82)))


def revolve(alpha_deg: float, *, re: float = 136.0, configure=None) -> tuple[float, float]:
    xml = """<mujoco><option timestep="0.004" gravity="0 0 0"/>
      <worldbody><body name="wing" pos="0 0 5">
        <joint name="h" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.125 0.043 0.001" pos="0.125 0 0" density="100"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    a = math.radians(alpha_deg)
    dr = (R - ROOT) / NSTRIP
    r = ROOT + dr * (np.arange(NSTRIP) + 0.5)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wing")
    # Revolving +Z moves the wing along +Y, so the leading edge faces +Y and the
    # chord (LE -> TE) points -Y, trailing edge down for positive incidence.
    chord = np.tile([0.0, -math.cos(a), -math.sin(a)], (NSTRIP, 1))
    span = np.tile([-1.0, 0.0, 0.0], (NSTRIP, 1))     # n = s x c points up
    p = PanelSet(
        body_id=np.full(NSTRIP, bid), pos_local=np.stack([r, np.zeros(NSTRIP), np.zeros(NSTRIP)], 1),
        span_local=span, chord_local=chord, chord=np.full(NSTRIP, C), dr=np.full(NSTRIP, dr),
        volume=np.zeros(NSTRIP), half_height=np.full(NSTRIP, 0.01), kind=np.full(NSTRIP, WING),
        aspect_ratio=np.full(NSTRIP, R / C), cd_bluff=np.zeros(NSTRIP))
    med = MediumField(air=OIL)
    sol = FluidSolver(m, p, med, disc_span=2 * R)   # one wing sweeps a disc of radius R
    if configure is not None:
        configure(sol)
    nu = OIL.mu / OIL.rho
    r2 = math.sqrt(np.mean(r ** 2))
    omega = re * nu / C / r2
    n = int(round(REVS * 2 * math.pi / omega / DT))
    n_last = int(round(2 * math.pi / omega / DT))
    L = Dg = QS = 0.0
    for k in range(n):
        t = k * DT
        d.qpos[0] = omega * t
        d.qvel[0] = omega
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, t)
        if k >= n - n_last:
            Rm = d.xmat[bid].reshape(3, 3)
            fw = d.xfrc_applied[bid, :3].copy()
            # Less the entrained fluid's weight cancellation (`finish_bodies`):
            # a constant vertical force MuJoCo's gravity would otherwise meet,
            # and in oil it is larger than the lift.
            fw[2] -= (m.body_mass[bid] - sol._dry_mass[bid]) * 9.80665
            f = Rm.T @ fw                                # wing frame
            u = omega * r
            qs = float(np.sum(0.5 * OIL.rho * u ** 2 * C * dr))
            L += f[2]; Dg += -f[1]; QS += qs
    return L / QS, Dg / QS


def main() -> int:
    alphas = np.arange(0, 91, 9.0)
    cl_ref, cd_ref = dickinson(alphas)
    rows = []
    print(f"{'alpha':>6} {'CL model':>9} {'CL robofly':>11} {'CD model':>9} {'CD robofly':>11}")
    for a, clr, cdr in zip(alphas, cl_ref, cd_ref):
        cl, cd = revolve(a)
        rows.append({"alpha": float(a), "cl": cl, "cd": cd, "cl_ref": float(clr), "cd_ref": float(cdr)})
        print(f"{a:6.0f} {cl:9.3f} {clr:11.3f} {cd:9.3f} {cdr:11.3f}", flush=True)
    e_cl = math.sqrt(np.mean([(x["cl"] - x["cl_ref"]) ** 2 for x in rows]))
    e_cd = math.sqrt(np.mean([(x["cd"] - x["cd_ref"]) ** 2 for x in rows]))
    pk = max(rows, key=lambda x: x["cl"])
    print(f"\nRMS error: CL {e_cl:.3f}, CD {e_cd:.3f}.  Model CL peaks {pk['cl']:.2f} at "
          f"{pk['alpha']:.0f} deg; robofly 1.80 at ~45 deg.")
    out = Path(__file__).with_name("results")
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(
        {"rows": rows, "rms_cl": e_cl, "rms_cd": e_cd}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
