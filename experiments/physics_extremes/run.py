"""Physics at the extremes (PREREG.md): invariances, conservation laws,
momentum-theory bounds and closed-form limits, on a bare plate and on the
flapper as the pipeline builds it.  No expected value here is a number typed
to be met: each is a second run, or a formula of the run's own quantities.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/physics_extremes/run.py [workers]
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FLAP = HERE.parent / "flapper"
sys.path.insert(0, str(FLAP))
sys.path.insert(0, str(HERE.parents[1]))

from dytiscidae.physics.fluid import WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import AIR, FRESHWATER, Fluid, MediumField  # noqa: E402

G = 9.80665
NSTRIP = 24
OUT: list = []


def record(check, case, measured, expected, tol, ok, note=""):
    r = dict(check=check, case=case, measured=measured, expected=expected, tol=tol, ok=bool(ok), note=note)
    OUT.append(r)
    f = lambda v: ("%.5g" % v) if isinstance(v, (int, float)) else str(v)
    print("  [%s] %-4s %-46s got %-24s want %-22s tol %s %s" % (
        "ok  " if ok else "FAIL", check, case, f(measured), f(expected), f(tol), note), flush=True)


# ---------------------------------------------------------------- the bare plate

def bare(joints, span, chord, alpha_deg, *, fluid=AIR, wind=None, ar=None, dt, gravity=0.0,
         mass=0.2, volume=0.0, half_height=None, flags=None, medium=None, z0=5.0, x_of=None,
         disc_span=None):
    """One rectangular plate of strips on a body with the given joints.  Span
    along X, the chord (leading edge to trailing) back along -Y and down for a
    positive incidence, upper side +Z; it flies along +Y."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 {-gravity}"/>
      <worldbody><body name="wing" pos="0 0 {z0}">{joints}
        <geom type="box" size="{span / 2} {chord / 2} {chord / 40}" mass="{mass}"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dr = span / NSTRIP
    x = (-span / 2 + dr * (np.arange(NSTRIP) + 0.5)) if x_of is None else x_of(dr)
    a = math.radians(alpha_deg)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wing")
    n = NSTRIP
    p = PanelSet(
        body_id=np.full(n, bid), pos_local=np.stack([x, np.zeros(n), np.zeros(n)], 1),
        span_local=np.tile([-1.0, 0.0, 0.0], (n, 1)),
        chord_local=np.tile([0.0, -math.cos(a), -math.sin(a)], (n, 1)),
        chord=np.full(n, chord), dr=np.full(n, dr), volume=np.full(n, volume / n),
        half_height=np.full(n, half_height if half_height is not None else chord / 20),
        kind=np.full(n, WING), aspect_ratio=np.full(n, ar if ar is not None else span / chord),
        cd_bluff=np.zeros(n), camber=np.zeros(n))
    med = medium if medium is not None else MediumField(air=fluid, wind=wind)
    sol = FluidSolver(m, p, med, disc_span=disc_span or span)
    for k, v in (flags or {}).items():
        setattr(sol, k, v)
    return m, d, sol, bid


def fluid_force(m, d, sol, bid):
    """The fluid's force on the body, world axes: what the solver wrote, less
    the weight it cancels for the fluid it folded into the body's mass."""
    f = d.xfrc_applied[bid, :3].copy()
    f[2] -= (m.body_mass[bid] - sol._dry_mass[bid]) * G
    return f


def translate(alpha, *, chord=0.1, ar=4.0, re=1e5, fluid=AIR, frame="body", chords=40.0,
              flags=None, strip_ar=None, series=False, U=None):
    """Steady flight along +Y; mean force over the last tenth, or the history."""
    U = re * fluid.nu / chord if U is None else U
    dt = chord / U / 100.0
    wind = np.array([0.0, -U, 0.0]) if frame == "wind" else None
    m, d, sol, bid = bare('<joint type="slide" axis="0 1 0"/>', ar * chord, chord, alpha, fluid=fluid,
                          wind=wind, ar=strip_ar or ar, dt=dt, flags=flags)
    n = int(round(chords * chord / U / dt))
    last = max(n // 10, 1)
    F, hist = np.zeros(3), []
    for k in range(n):
        if frame == "body":
            d.qpos[0], d.qvel[0] = U * k * dt, U
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, k * dt)
        f = fluid_force(m, d, sol, bid)
        if series:
            hist.append(f)
        if k >= n - last:
            F += f
    q = 0.5 * fluid.rho * U ** 2 * chord * ar * chord
    if series:
        return np.array(hist) / q, 2.0 * U * dt * np.arange(n) / chord      # coefficients, semi-chords
    return F / last, q


def coeff_translate(alpha, **kw):
    F, q = translate(alpha, **kw)
    return {"CL": F[2] / q, "CD": -F[1] / q}


def coeff_revolve(alpha, *, chord=0.1, rc=3.3, re=5.4e4, fluid=AIR, revs=3):
    R = rc * chord
    dr = R / NSTRIP
    r = dr * (np.arange(NSTRIP) + 0.5)
    omega = re * fluid.nu / chord / math.sqrt(np.mean(r ** 2))
    dt = 2 * math.pi / omega / 4000.0
    m, d, sol, bid = bare('<joint type="hinge" axis="0 0 1"/>', R, chord, alpha, fluid=fluid, ar=rc, dt=dt,
                          x_of=lambda dr_: dr_ * (np.arange(NSTRIP) + 0.5), disc_span=2 * R)
    per = 4000
    qs = float(np.sum(0.5 * fluid.rho * (omega * r) ** 2 * chord * dr))
    V = H = 0.0
    for k in range(revs * per):
        d.qpos[0], d.qvel[0] = omega * k * dt, omega
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, k * dt)
        if k >= (revs - 1) * per:
            f = d.xmat[bid].reshape(3, 3).T @ fluid_force(m, d, sol, bid)
            V += f[2]
            H += -f[1]
    return {"Cv": V / per / qs, "Ch": H / per / qs}


def coeff_heave(*, chord=0.1, ar=6.0, re=1e5, fluid=AIR, kr=0.25, h_over_c=0.5, alpha=10.0, periods=6):
    """A plate at fixed pitch flying at U and heaving h0 sin(wt), reduced
    frequency w c / 2U = kr."""
    U = re * fluid.nu / chord
    w = 2.0 * kr * U / chord
    T = 2 * math.pi / w
    dt = T / 800.0
    m, d, sol, bid = bare('<joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>',
                          ar * chord, chord, alpha, fluid=fluid, dt=dt)
    h0 = h_over_c * chord
    q = 0.5 * fluid.rho * U ** 2 * ar * chord * chord
    cl, cd = [], []
    for k in range(periods * 800):
        t = k * dt
        d.qpos[:2] = (U * t, h0 * math.sin(w * t))
        d.qvel[:2] = (U, h0 * w * math.cos(w * t))
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        sol.apply(d, t)
        if k >= (periods - 2) * 800:
            f = fluid_force(m, d, sol, bid)
            cl.append(f[2] / q)
            cd.append(-f[1] / q)
    cl, cd = np.array(cl), np.array(cd)
    return {"CL mean": cl.mean(), "CL swing": (cl.max() - cl.min()) / 2, "CD mean": cd.mean()}


def e1_galilean():
    print("E1 Galilean invariance")
    for fluid in (AIR, FRESHWATER):
        for U in (0.05, 8.0, 80.0):
            for a in (5.0, 45.0):
                Fb, _ = translate(a, U=U, fluid=fluid, frame="body")
                Fw, _ = translate(a, U=U, fluid=fluid, frame="wind")
                rel = float(np.linalg.norm(Fb - Fw) / max(np.linalg.norm(Fb), 1e-300))
                record("E1", "%s U %g m/s, %g deg" % (fluid.name, U, a), rel, 0.0, 1e-6, rel <= 1e-6,
                       "|F| %.3g N" % np.linalg.norm(Fb))


def _similar(check, label, ref, other):
    for k in ref:
        dev = abs(other[k] - ref[k]) / max(abs(ref[k]), 1e-12)
        record(check, "%s: %s" % (label, k), other[k], ref[k], 1e-3, dev <= 1e-3, "rel %.2e" % dev)


def e2_e3_similarity():
    print("E2 similarity in size, E3 similarity in fluid")
    cases = [("translate %g deg" % a, lambda a=a, **kw: coeff_translate(a, **kw)) for a in (5.0, 20.0, 60.0)]
    cases += [("revolve 40 deg", lambda **kw: coeff_revolve(40.0, **kw)), ("heave kr 0.25", lambda **kw: coeff_heave(**kw))]
    for name, fn in cases:
        ref = fn(chord=0.1, fluid=AIR)
        again = fn(chord=0.1, fluid=AIR)
        assert all(again[k] == ref[k] for k in ref), "a run is not equal to itself"
        for k in (0.02, 50.0):
            _similar("E2", "%s, chord x%g" % (name, k), ref, fn(chord=0.1 * k, fluid=AIR))
        _similar("E3", "%s, fresh water" % name, ref, fn(chord=0.1, fluid=FRESHWATER))


def e9_finite_wing():
    print("E9 finite wing: induced-drag bound and lift slope")
    for ar in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0):
        c = {a: coeff_translate(a, ar=ar) for a in (2.0, 5.0, 8.0)}
        worst = min(c[a]["CD"] - c[a]["CL"] ** 2 / (math.pi * ar) for a in c)
        record("E9", "AR %g: least CD - CL^2/(pi AR), 2-8 deg" % ar, worst, ">= 0", 0.0, worst >= 0.0)
        slope = (c[5.0]["CL"] - c[2.0]["CL"]) / math.radians(3.0)
        want = 2 * math.pi * ar / (2 + math.sqrt(ar * ar + 4))
        record("E9", "AR %g: lift slope per rad" % ar, slope, want, "15%", abs(slope / want - 1) <= 0.15,
               "ratio %.3f" % (slope / want))


def e10_added_mass():
    print("E10 added mass from a spring's period, in water")
    chord, span, mb, ks = 0.1, 1.0, 0.2, 800.0
    want = math.pi * FRESHWATER.rho * chord ** 2 * span / 4.0
    got = {}
    for name, axis in (("normal", "0 0 1"), ("edge-on", "0 1 0")):
        dt = 1e-4
        m, d, sol, bid = bare(f'<joint type="slide" axis="{axis}" stiffness="{ks}"/>', span, chord, 0.0,
                              fluid=FRESHWATER, dt=dt, mass=mb, gravity=G)
        # Gravity on (harness fix after the first run, RESULT.md): the solver
        # cancels the weight of the fluid it folds into the body's mass with
        # Earth's g whatever the model's gravity is, so with gravity off the
        # plate was pushed 0.097 m up its spring.  The period is read about
        # the position it settles at.
        d.qpos[0] = (-mb * G / ks if axis == "0 0 1" else 0.0) + 0.002 * chord
        qs, ts = [], []
        for k in range(int(3.0 / dt)):
            d.xfrc_applied[:] = 0.0
            sol.apply(d, k * dt)
            mujoco.mj_step(m, d)
            qs.append(float(d.qpos[0]))
            ts.append((k + 1) * dt)
        qs, ts = np.array(qs), np.array(ts)
        qs = qs - qs[len(qs) // 2:].mean()
        up = np.flatnonzero((qs[:-1] < 0) & (qs[1:] >= 0))
        tz = ts[up] + dt * (-qs[up]) / (qs[up + 1] - qs[up])
        if len(tz) < 3:
            got[name] = float("nan")
            continue
        T = float(np.mean(np.diff(tz[:4])))
        got[name] = ks * T * T / (4 * math.pi ** 2) - mb
    record("E10", "normal to the plate: added mass, kg", got["normal"], want, "15%",
           abs(got["normal"] / want - 1) <= 0.15, "pi rho c^2 b / 4; ratio %.3f" % (got["normal"] / want))
    record("E10", "edge-on: added mass, kg", got["edge-on"], "< 5%% of %.3g" % want, 0.05 * want,
           abs(got["edge-on"]) <= 0.05 * want)


def e11_archimedes():
    print("E11 Archimedes")
    chord, span, h = 0.1, 1.0, 0.05
    V = 1e-3
    rho = FRESHWATER.rho
    for name, mass, z0, secs in (("neutral, 50 m under", rho * V, -50.0, 1.0), ("half density, at the surface", 0.5 * rho * V, 0.5 * h, 30.0)):
        dt = 1e-3
        m, d, sol, bid = bare('<joint type="slide" axis="0 0 1"/>', span, chord, 0.0, dt=dt, gravity=G,
                              mass=mass, volume=V, half_height=h, medium=MediumField(water=FRESHWATER), z0=z0)
        for k in range(int(secs / dt)):
            d.xfrc_applied[:] = 0.0
            sol.apply(d, k * dt)
            mujoco.mj_step(m, d)
        z, v = z0 + float(d.qpos[0]), float(d.qvel[0])
        if z0 < -1:
            record("E11", name + ": speed after 1 s, m/s", abs(v), 0.0, 1e-4 * G, abs(v) <= 1e-4 * G)
        else:
            record("E11", name + ": centre above the waterline, m", z, 0.0, 0.02 * 2 * h, abs(z) <= 0.02 * 2 * h,
                   "speed %.2e m/s" % v)


def e12_wagner():
    print("E12 impulsive start against Wagner")
    c, s = translate(5.0, ar=8.0, chords=60.0, flags=dict(inflow=False), strip_ar=1e4, series=True)
    cl = c[:, 2]
    steady = float(cl[-len(cl) // 10:].mean())
    for s_ in (2.0, 4.0, 8.0, 16.0, 40.0):
        got = float(np.interp(s_, s, cl)) / steady
        want = 1 - 0.165 * math.exp(-0.0455 * s_) - 0.335 * math.exp(-0.3 * s_)
        record("E12", "lift / steady at s = %g semi-chords" % s_, got, want, 0.10, abs(got - want) <= 0.10,
               "steady CL %.3f (2 pi a = %.3f)" % (steady, 2 * math.pi * math.radians(5)))


def e14_broadside():
    print("E14 broadside plate (measured)")
    c1 = coeff_translate(90.0, ar=1.0)
    record("E14", "AR 1 at 90 deg: CD", c1["CD"], 1.17, "10%", abs(c1["CD"] / 1.17 - 1) <= 0.10,
           "disc and square plate, TYPED_CONSTANTS_SOURCES.md C")
    c2 = coeff_translate(90.0, ar=8.0, flags=dict(inflow=False), strip_ar=1e4)
    record("E14", "2D at 90 deg: CD", c2["CD"], 1.98, "10%", abs(c2["CD"] / 1.98 - 1) <= 0.10)


# ---------------------------------------------------------------- the machine

SECT = dict(wing_shape=dict(camber=0.04), wing_section="goe417a+lift")


def _body(kind):
    import l1a
    b = dict(l1a.BODY_S)
    if kind == "flat":
        return b
    if kind == "cambered":
        return dict(b, wing_shape=dict(camber=0.04))
    if kind == "section":
        return dict(b, **SECT)
    if kind == "chain6":
        return dict(b, wing_panels=6, **SECT)
    if kind == "stubby":                                   # each wing 0.15 m long, 0.40 m chord
        return dict(b, wing_length=0.15, wing_chord=0.40)
    if kind == "slender":                                  # each wing 0.90 m long, 0.03 m chord
        return dict(b, wing_length=0.90, wing_chord=0.03)
    raise ValueError(kind)


def free_job(a):
    from freerig import run_free
    tag, kind, args, kw = a
    kw = dict(kw)
    if kw.get("fluid") == "vacuum":
        kw["fluid"] = Fluid("vacuum", 1e-9, 1.81e-5)
    keep = kw.pop("keep", None)
    r = run_free(_body(kind), *args, feather_axis=0.1, cop_law="plate", keep_series=bool(keep), **kw)
    s = r.pop("series", None)
    r.pop("com0", None)
    if keep == "hold":
        hold = np.array(s["hold"])
        r["hold_mean"] = hold.mean(0).tolist()
        r["hold_absmax"] = np.abs(hold).max(0).tolist()
    if keep == "motion":
        r["t"], r["vel"], r["pitch"] = s["t"], s["vel"], s["pitch"]
        r["com_vel"], r["angmom"] = s["com_vel"], s["angmom"]
    r["tag"], r["body_kind"] = tag, kind
    return r


def rig_job(a):
    import rig
    tag, kind, args, kw = a
    r = rig.run(_body(kind), *args, feather_axis=0.1, cop_law="plate", **kw)
    r["tag"], r["body_kind"] = tag, kind
    for k in [k for k, v in r.items() if isinstance(v, list) and len(v) > 20]:
        r.pop(k)
    return r


def machine_grid():
    rng = np.random.default_rng(20261117)
    lu = lambda lo, hi: float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    free, held = [], []
    tube = dict(chain=dict(mode="tube", tip_deg=15.0, k=10.0), solver_flags=dict(camber_moment=True))
    # E4: (V, theta, f, Phi, Psi, psi0)
    free.append(("E4 cruise", "chain6", (8.0, 0.0, 2.5, 40.0, 0.0, -3.56), dict(tube, keep="hold")))
    free.append(("E4 take-off", "chain6", (0.0, 90.0, 4.5, 85.0, 50.0, 0.0),
                 dict(chain=dict(mode="tube", tip_deg=0.0, k=10.0), solver_flags=dict(camber_moment=True), keep="hold")))
    # E5
    for name, fl in (("vacuum", "vacuum"), ("air", None)):
        free.append(("E5 " + name, "section", (3.0, 0.0, 5.0, 40.0, 15.0, 0.0),
                     dict(free="xzp", beats=6, keep="motion", fluid=fl)))
    # E6 / E7
    kinds = ("flat", "cambered", "section", "chain6", "stubby", "slender")
    for i in range(60):
        water = i >= 40
        kind = kinds[i % len(kinds)]
        f = lu(0.3, 3.0) if water else lu(0.5, 30.0)
        args = (0.0, float(rng.uniform(0.0, 90.0)), f, float(rng.uniform(5.0, 160.0)),
                float(rng.uniform(0.0, 85.0)), float(rng.uniform(-60.0, 60.0)))
        kw = dict(mass=lu(0.005, 50.0), dt=min(2.5e-4, 1.0 / (800.0 * f)), medium="water" if water else "air")
        if kind == "chain6":
            kw["chain"] = dict(mode="servo", tip_deg=float(rng.uniform(0.0, 20.0)))
            kw["solver_flags"] = dict(camber_moment=True)
        free.append(("E6 %02d %s" % (i, "water" if water else "air"), kind, args, kw))
    # E13 (freerig)
    s6 = dict(chain=dict(mode="servo", tip_deg=15.0), solver_flags=dict(camber_moment=True))
    for dt in (2.5e-4, 1.25e-4):
        free.append(("E13 cruise dt %g" % dt, "chain6", (8.0, 0.0, 2.5, 40.0, 0.0, -3.29), dict(s6, dt=dt)))
        free.append(("E13 take-off dt %g" % dt, "section", (0.0, 90.0, 4.5, 85.0, 50.0, 0.0),
                     dict(dt=dt, solver_flags=dict(camber_moment=True))))
    # E13 (rig) and E8 (new slow and fast points)
    for spb in (50, 200, 2000):
        held.append(("E13 rig cruise spb %d" % spb, "section", (8.0, 0.0, 2.5, 40.0, 15.0, -2.955),
                     dict(strip_twist=dict(tip_deg=0.0, share=1.0), steps_per_beat=spb)))
        held.append(("E13 rig take-off spb %d" % spb, "section", (0.0, 90.0, 4.5, 85.0, 50.0, 0.0),
                     dict(steps_per_beat=spb)))
    for V in (2.0, 3.0, 20.0, 40.0):
        for f in (2.0, 3.5):
            for Phi in (40.0, 60.0):
                for Psi in (0.0, 15.0):
                    for psi0 in (-10.0, 0.0, 10.0, 25.0):
                        held.append(("E8 new", "section", (V, 0.0, f, Phi, Psi, psi0), {}))
    return free, held


def read_machine(free, held):
    import l1a
    b, rho = l1a.SPAN, AIR.rho
    A = math.pi * (b / 2) ** 2
    by = {r["tag"]: r for r in free + held}

    print("E4 mirror symmetry (six-panel wing, tube)")
    for tag in ("E4 cruise", "E4 take-off"):
        r = by[tag]
        fz = abs(r["hold_mean"][2] + r["mass_kg"] * G)
        for name, col, scale in (("side force", 1, fz), ("rolling moment", 3, fz * b / 2), ("yawing moment", 5, fz * b / 2)):
            v = r["hold_absmax"][col] / scale
            record("E4", "%s: largest %s / vertical force" % (tag[3:], name), v, 0.0, 1e-4, v <= 1e-4)

    print("E5 momentum in a vacuum (control: air)")
    # Read on the machine's centre of mass (repair after the first read,
    # RESULT.md: the frozen form read the hull, whose beat mean moves with the
    # body's attitude even when momentum is conserved).  The frozen form is
    # kept below it and labelled.
    for tag in ("E5 vacuum", "E5 air"):
        r = by[tag]
        t, vel, pitch = np.array(r["t"]), np.array(r["vel"]), np.array(r["pitch"])
        cv, am = np.array(r["com_vel"]), np.array(r["angmom"])
        nb = int(round(1.0 / (r["f"] * (t[1] - t[0]))))
        beats = len(t) // nb
        hull_swing = float((vel[: beats * nb, 2].reshape(beats, nb).max(1) - vel[: beats * nb, 2].reshape(beats, nb).min(1)).mean())
        mass = r["mass_kg"]
        lin = {"forward": cv[:, 0], "vertical + g t": cv[:, 2] + G * (t - t[0])}
        for name, x in lin.items():
            ratio = float(np.abs(x - x[0]).max()) / hull_swing
            if tag == "E5 vacuum":
                record("E5", "vacuum: change of the centre of mass's %s speed / hull's in-beat swing" % name, ratio, 0.0,
                       1e-3, ratio <= 1e-3, "swing %.3g m/s" % hull_swing)
            else:
                record("E5", "control in air: the same, %s" % name, ratio, "> 0.01", 0.01, ratio > 1e-2)
        ratio = float(np.abs(am[:, 1] - am[0, 1]).max()) / (mass * hull_swing * b / 4)
        if tag == "E5 vacuum":
            record("E5", "vacuum: change of angular momentum (pitch) / (mass x swing x span/4)", ratio, 0.0, 1e-3,
                   ratio <= 1e-3)
        else:
            record("E5", "control in air: the same, angular momentum", ratio, "> 0.01", 0.01, ratio > 1e-2)
        if tag == "E5 vacuum":
            for name, x in (("forward speed", vel[:, 0]), ("vertical speed + g t", vel[:, 2] + G * (t - t[0]))):
                xb = x[: beats * nb].reshape(beats, nb)
                record("E5", "vacuum, as frozen (the hull, not the centre of mass): %s" % name,
                       float(xb.mean(1).max() - xb.mean(1).min()) / hull_swing, 0.0, 1e-3, True, "superseded; reported")

    print("E6 still fluid gives no energy; E7 hover bound")
    e6 = [r for r in free if r["tag"].startswith("E6")]
    neg = [r for r in e6 if r["finite"] and r["power_W"] < -0.01 * abs(r["power_W"]) - 1e-12]
    bad = [r for r in e6 if not r["finite"]]
    clamp = [r for r in e6 if r.get("clamped")]
    record("E6", "runs with negative mean drive power, of %d" % len(e6), len(neg), 0, 0, not neg,
           "; ".join("%s %s P %.3g W" % (r["tag"], r["body_kind"], r["power_W"]) for r in neg[:6]))
    record("E6", "runs that did not stay finite", len(bad), 0, 0, not bad,
           "; ".join("%s %s f %.2f" % (r["tag"], r["body_kind"], r["f"]) for r in bad[:8]))
    record("E6", "runs in which the load limiter bound (reported)", len(clamp), "-", "-", True,
           "; ".join("%s %s mass %.3g kg" % (r["tag"], r["body_kind"], r["mass_kg"]) for r in clamp[:8]))
    worst_track = max((r["track_err_deg"] for r in e6 if r["finite"]), default=0.0)
    record("E6", "largest tracking error of the drives, deg (reported)", worst_track, "-", "-", True)

    def hover_ratio(P, Fx, Fz, span):
        F = math.hypot(Fx, Fz)
        ideal = F ** 1.5 / math.sqrt(2 * rho * math.pi * (span / 2) ** 2)
        return P / ideal if ideal > 1e-9 else float("inf")

    spans = {"stubby": 2 * (0.15 + l1a.HULL_R), "slender": 2 * (0.90 + l1a.HULL_R)}
    air6 = [r for r in e6 if r["medium"] == "air" and r["finite"] and not r.get("clamped")]
    ratios = [(hover_ratio(r["power_W"], r["Fx"], r["Fz"], spans.get(r["body_kind"], b)), r) for r in air6]
    viol = [x for x in ratios if x[0] < 1.0]
    lo = min(ratios, key=lambda x: x[0])
    record("E7", "E6 air runs (limiter not bound): power below the actuator-disc bound, of %d" % len(ratios),
           len(viol), 0, 0, not viol, "least power/bound %.2f (%s %s)" % (lo[0], lo[1]["tag"], lo[1]["body_kind"]))
    stored = []
    for name, pk in (("W6_twist.jsonl", "aero_power_W"), ("W7_twistmech.jsonl", "power_W")):
        for line in open(FLAP / "results" / name):
            r = json.loads(line)
            if r.get("kind") in ("rest", "takeoff") and r.get("finite", True):
                stored.append((hover_ratio(r[pk], r["Fx"], r["Fz"], b), name, r))
    viol = [x for x in stored if x[0] < 1.0]
    record("E7", "stored rows at rest (W6, W7): below the bound, of %d" % len(stored), len(viol), 0, 0, not viol,
           "least power/bound %.2f, median %.2f" % (min(x[0] for x in stored), float(np.median([x[0] for x in stored]))))

    print("E8 cruise bound")

    def cruise_margin(P, Fx, Fz, V, span):
        return P - (Fx * V + 2 * Fz ** 2 / (rho * V * math.pi * span ** 2))

    st = []
    for name, pk in (("W6_twist.jsonl", "aero_power_W"), ("W7_twistmech.jsonl", "power_W")):
        for line in open(FLAP / "results" / name):
            r = json.loads(line)
            if r.get("kind") == "cruise" and r.get("finite", True):
                st.append((cruise_margin(r[pk], r["Fx"], r["Fz"], r["V"], b), r[pk], r))
    viol = [x for x in st if x[0] < 0.0]
    record("E8", "stored cruise rows (W6, W7): power below thrust x V + induced, of %d" % len(st), len(viol), 0, 0,
           not viol, "least margin %.2f W" % min(x[0] for x in st) + ("; worst %s" % json.dumps(
               {k: viol[0][2][k] for k in ("V", "f", "Phi_deg", "Psi_deg", "psi0_deg", "Fx", "Fz")}) if viol else ""))
    new = [r for r in held if r["tag"] == "E8 new" and r["finite"]]
    for V in (2.0, 3.0, 20.0, 40.0):
        q = [(cruise_margin(r["aero_power_W"], r["Fx"], r["Fz"], V, b), r) for r in new if r["V"] == V]
        viol = [x for x in q if x[0] < 0.0]
        w = min(q, key=lambda x: x[0])
        record("E8", "new held runs at %g m/s: below the bound, of %d" % (V, len(q)), len(viol), 0, 0, not viol,
               "least margin %.3g W at f %.1f stroke %.0f feather %.0f offset %.0f: P %.3g, Fx %.3g, lift %.3g N" % (
                   w[0], w[1]["f"], w[1]["Phi_deg"], w[1]["Psi_deg"], w[1]["psi0_deg"], w[1]["aero_power_W"],
                   w[1]["Fx"], w[1]["Fz"]))

    print("E13 time step")
    for name, a_, b_, keys, tol in (
            ("freerig cruise (S6)", "E13 cruise dt 0.00025", "E13 cruise dt 0.000125", ("power_W", "lift_over_weight"), 0.02),
            ("freerig take-off", "E13 take-off dt 0.00025", "E13 take-off dt 0.000125", ("power_W", "lift_over_weight"), 0.02),
            ("rig cruise", "E13 rig cruise spb 50", "E13 rig cruise spb 200", ("aero_power_W", "Fz"), 0.03),
            ("rig take-off", "E13 rig take-off spb 50", "E13 rig take-off spb 200", ("aero_power_W", "Fz"), 0.05),
            ("rig cruise, 1 ms against 0.2 ms (repair)", "E13 rig cruise spb 50", "E13 rig cruise spb 2000", ("aero_power_W", "Fz"), 0.03),
            ("rig take-off, 1 ms against 0.11 ms (repair)", "E13 rig take-off spb 50", "E13 rig take-off spb 2000", ("aero_power_W", "Fz"), 0.05)):
        for k in keys:
            x, y = by[a_][k], by[b_][k]
            record("E13", "%s: %s, coarse against fine" % (name, k), x, y, "%g%%" % (100 * tol),
                   abs(x / y - 1) <= tol, "rel %+.2f%%" % (100 * (x / y - 1)))


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    free, held = machine_grid()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        fr = pool.map_async(free_job, free, chunksize=1)
        hr = pool.map_async(rig_job, held, chunksize=1)
        e1_galilean()
        e2_e3_similarity()
        e9_finite_wing()
        e10_added_mass()
        e11_archimedes()
        e12_wagner()
        e14_broadside()
        free_rows, held_rows = fr.get(), hr.get()
    with open(HERE / "machine_rows.jsonl", "w") as fh:
        for r in free_rows + held_rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k not in ("t", "vel", "pitch")}) + "\n")
    read_machine(free_rows, held_rows)
    json.dump(OUT, open(HERE / "results.json", "w"), indent=1)
    n_fail = sum(not r["ok"] for r in OUT)
    print("\n%d checks, %d failed" % (len(OUT), n_fail))
    for c in sorted({r["check"] for r in OUT}):
        q = [r for r in OUT if r["check"] == c]
        print("  %-4s %2d of %2d hold" % (c, sum(r["ok"] for r in q), len(q)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
