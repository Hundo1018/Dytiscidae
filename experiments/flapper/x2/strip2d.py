"""Pass X2, stage B, the strip solver's side: a bare strip doing the station's
two-dimensional motion (flight at V along +Y, plunge z = h0 sin wt, pitch
theta = theta0 + A cos wt nose up about the strip's load point, the quarter
chord), built on physics_extremes/run.py's `bare` (its `coeff_heave` with a
pitch hinge added).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/x2/strip2d.py [check]

Solver flags: section = sections.load("goe417a+lift+extend"), inflow off,
unsteady on, cop_law "plate" (as the stored W8 rows ran: l1i_cam's cruise job),
strip and machine aspect ratio 1e4 (two-dimensional), air (`medium.AIR`).  Camber 0.04,
the rig's, marks the upper side (the measured section supplies the camber's
effect).

Two ways to pitch, both reported:
  hinge: a hinge joint about the span, so the solver sees the pitch rate
         (rotational lift, pitch damping) -- the two-dimensional motion;
  turn:  the strip's chord turned in place each step with no body rate, as
         rig.twist_turn turns the strips in the stored W8 rows.
Coefficients per unit span on the station's chord and q = rho V^2 / 2:
  CN  = force normal to the stream (+ up), CX = force along the flight
  direction (+ thrust), CP = -(Fz dz/dt + M dtheta/dt) / (q V c), power put
  into the fluid; beat means over the last 3 of 5 beats.
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "physics_extremes"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import run as PX  # noqa: E402  experiments/physics_extremes/run.py
import sections  # noqa: E402
from dytiscidae.physics.medium import AIR  # noqa: E402

SECTION = "goe417a+lift+extend"
CAMBER = 0.04
AR2D = 1.0e4
SPAN = 1.0


def flags():
    return dict(section=sections.load(SECTION), inflow=False, unsteady=True, cop_law="plate")


def motion(chord, V, f, h0, theta0_deg, amp_deg, *, mode="hinge", beats=5, avg=3, steps=800,
           extra_flags=None):
    w = 2 * math.pi * f
    T = 1.0 / f
    dt = T / steps
    joints = ('<joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>'
              + ('<joint type="hinge" axis="1 0 0"/>' if mode == "hinge" else ""))
    fl = flags()
    fl.update(extra_flags or {})
    # disc_span sets the solver's machine aspect ratio (disc_span^2 / area), which
    # it uses in place of the strips' own when inflow is off: make it AR2D too
    m, d, sol, bid = PX.bare(joints, SPAN, chord, 0.0, fluid=AIR, ar=AR2D, dt=dt, flags=fl, camber=CAMBER,
                             disc_span=math.sqrt(AR2D * SPAN * chord))
    assert abs(sol._machine_ar - AR2D) < 1e-6 * AR2D
    pan = sol.panels
    c0 = pan.chord_local.copy()
    th0, A = math.radians(theta0_deg), math.radians(amp_deg)
    q = 0.5 * AIR.rho * V ** 2
    ref = q * chord * SPAN
    hist = []
    for k in range(beats * steps):
        t = k * dt
        z, zd = h0 * math.sin(w * t), h0 * w * math.cos(w * t)
        th, thd = th0 + A * math.cos(w * t), -A * w * math.sin(w * t)
        if mode == "hinge":
            d.qpos[:3] = (V * t, z, th)
            d.qvel[:3] = (V, zd, thd)
        else:
            d.qpos[:2] = (V * t, z)
            d.qvel[:2] = (V, zd)
            ca, sa = math.cos(th), math.sin(th)
            # turn the chord about the span (-X): (0,-1,0) -> (0,-cos, -sin) is nose up
            pan.chord_local[:] = np.stack([c0[:, 0], c0[:, 1] * ca - c0[:, 2] * sa,
                                           c0[:, 1] * sa + c0[:, 2] * ca], 1)
            pan.normal_local[:] = np.cross(pan.span_local, pan.chord_local)
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, t)
        F = PX.fluid_force(m, d, sol, bid)
        M = float(d.xfrc_applied[bid, 3])            # about +X at the load point: + nose up
        P = -(F[2] * zd + M * thd)
        hist.append((t, F[2] / ref, F[1] / ref, P / (ref * V), M / (ref * chord), z, th))
    h = np.array(hist)
    last = h[(beats - avg) * steps:]
    one = h[(beats - 1) * steps:]
    return dict(mode=mode, CN=float(last[:, 1].mean()), CX=float(last[:, 2].mean()), CP=float(last[:, 3].mean()),
                CM=float(last[:, 4].mean()), CN_max=float(one[:, 1].max()), CN_min=float(one[:, 1].min()),
                phase=((one[:, 0] - one[0, 0]) / T).tolist(), CN_hist=one[:, 1].tolist(),
                CX_hist=one[:, 2].tolist(), CP_hist=one[:, 3].tolist(),
                beats_cmp=[float(h[b * steps:(b + 1) * steps, 1].mean()) for b in range(beats)])


def static_check():
    """Fixed strip at alpha, flying at 8 m/s: CL against the section's table."""
    out = []
    for a in (0.0, 4.0, 8.0):
        r = motion(0.25758, 8.0, 2.5, 0.0, a, 0.0, mode="hinge", beats=3, avg=1)
        out.append((a, r["CN"], r["CX"]))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "emit":
        emit_rows()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        for a, cn, cx in static_check():
            print("static alpha %.1f: CN %.4f CX %.5f" % (a, cn, cx))
        return
    kin = json.loads((HERE / "kinematics.json").read_text())
    chord = 0.34 / (2 * 0.66)
    rows = []
    for name in ("B8", "B6"):
        k = kin[name]
        h0 = k["plunge_amp_spec"]
        for mode in ("hinge", "turn"):
            r = motion(chord, k["V"], k["f"], h0, k["pitch_fit_deg"]["mean"], 0.7 * k["Psi"], mode=mode)
            r.update(case=name, chord=chord, V=k["V"], f=k["f"], h0=h0, theta0_deg=k["pitch_fit_deg"]["mean"],
                     pitch_amp_deg=0.7 * k["Psi"], k_red=2 * math.pi * k["f"] * chord / (2 * k["V"]),
                     Re=k["V"] * chord / (AIR.mu / AIR.rho))
            rows.append(r)
            print(name, mode, "CN %.4f CX %.4f CP %.4f CM %.4f  CN beat range [%.3f, %.3f]  beat means %s" % (
                r["CN"], r["CX"], r["CP"], r["CM"], r["CN_min"], r["CN_max"],
                " ".join("%.4f" % x for x in r["beats_cmp"])))
    (HERE / "strip2d.json").write_text(json.dumps(rows))
    emit_rows(rows)


def emit_rows(rows=None):
    """One results.jsonl row per strip-solver run (the CN history thinned to 100 points)."""
    rows = rows if rows is not None else json.loads((HERE / "strip2d.json").read_text())
    with open(HERE / "results.jsonl", "a") as fh:
        for r in rows:
            n = len(r["CN_hist"])
            k = max(1, n // 100)
            out = {k_: v for k_, v in r.items() if not k_.endswith("_hist") and k_ != "phase"}
            out.update(stage="B-strip", solver="strip (dytiscidae.physics.fluid via physics_extremes.bare)",
                       window="beats 3-5 of 5", history_last_beat=dict(phase=r["phase"][::k], CN=r["CN_hist"][::k]))
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
