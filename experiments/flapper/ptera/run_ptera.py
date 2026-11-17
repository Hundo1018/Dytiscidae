"""Cross-check of the flapper's cruise numbers with PteraSoftware (UVLM).

Run with the PteraSoftware venv (NOT the project's .venv):

    /home/hundo/Projects/Dytiscidae/.venv-ptera/bin/python run_ptera.py probe   # one run, prints
    /home/hundo/Projects/Dytiscidae/.venv-ptera/bin/python run_ptera.py all     # every run -> results.jsonl
    /home/hundo/Projects/Dytiscidae/.venv-ptera/bin/python run_ptera.py read    # trims and controls from results.jsonl

Conventions and deviations are in NOTES.md beside this file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"

RHO = 1.225
CHORD = 0.2576
Y_ROOT = 0.04
Y_TIP = 0.70
WEIGHT = 5.884
PIVOT = 0.1 * CHORD          # feathering / incidence axis behind the leading edge
S_REF = 2.0 * (Y_TIP - Y_ROOT) * CHORD     # the two panels, hull gap excluded
B_REF = 2.0 * Y_TIP
AR_FULL = B_REF / CHORD      # 5.43, the brief's aspect ratio (gap counted as span)


def outline():
    """Goettingen 417a outline, upper trailing edge -> leading edge -> lower
    trailing edge, as dumped from experiments/flapper/sections.py."""
    return np.loadtxt(HERE / "goe417a_coords.txt")


def build(V, f, Phi, A, inc, share, sign=1.0, nc=8, nseg=6, ns_per_seg=2,
          steps_per_cycle=None, cycles=3, flat=False, y_root=Y_ROOT):
    """The UnsteadyProblem for one case and its bookkeeping.

    share: 1 = linear tip twist, 0 = rigid feathering of the whole wing.
    sign: +1 = the brief's sense (tip leading edge DOWN at mid-downstroke),
          -1 = the opposite, as a control.
    Angles in degrees. inc positive = leading edge up.
    """
    import pterasoftware as ps

    if flat:
        af = ps.geometry.airfoil.Airfoil(
            name="flat", outline_A_lp=[[1, 0.001], [0.5, 0.001], [0, 0],
                                      [0.5, -0.001], [1, -0.001]])
    else:
        af = ps.geometry.airfoil.Airfoil(name="goe417a", outline_A_lp=outline())

    seg = (Y_TIP - y_root) / nseg
    wcs = []
    for i in range(nseg + 1):
        wcs.append(ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=af,
            num_spanwise_panels=None if i == nseg else ns_per_seg,
            chord=CHORD,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0) if i == 0 else (0.0, seg, 0.0),
            spanwise_spacing=None if i == nseg else "uniform",
            control_surface_symmetry_type="symmetric",
        ))
    wing = ps.geometry.wing.Wing(
        wcs, name="flapper",
        Ler_Gs_Cgs=(0.0, y_root, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, inc, 0.0),
        symmetric=True, symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=nc, chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        [wing], name="flapper", weight=WEIGHT,
        s_ref=S_REF, c_ref=CHORD, b_ref=B_REF)

    static = (Phi == 0 and A == 0)
    T = 1.0 / f if not static else 0.0
    # Leading-edge-up angle wanted: +sign * A * cos(w t) = A sin(w t + 90 deg)
    # (leading edge up at mid-upstroke, down at mid-downstroke for sign=+1).
    ph = 90.0 if sign > 0 else -90.0
    wing_movements = []
    for w in airplane.wings:
        wcsm = []
        n = len(w.wing_cross_sections)
        for i, cs in enumerate(w.wing_cross_sections):
            if share == 1 and A != 0 and i > 0:
                d = A / (n - 1)        # each section twists d relative to its parent
                wcsm.append(ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=cs,
                    ampAngles_Wcsp_to_Wcs_ixyz=(0.0, d, 0.0),
                    periodAngles_Wcsp_to_Wcs_ixyz=(0.0, T, 0.0),
                    phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, ph, 0.0),
                    # keep the 0.1-chord line straight: the section rotates about
                    # its leading point, so lift the leading point with the twist
                    ampLp_Wcsp_Lpp=(0.0, 0.0, PIVOT * np.sin(np.radians(d))),
                    periodLp_Wcsp_Lpp=(0.0, 0.0, T),
                    phaseLp_Wcsp_Lpp=(0.0, 0.0, ph),
                ))
            else:
                wcsm.append(ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                    base_wing_cross_section=cs))
        rigid = (share == 0 and A != 0)
        wing_movements.append(ps.movements.wing_movement.WingMovement(
            base_wing=w, wing_cross_section_movements=wcsm,
            ampAngles_Gs_to_Wn_ixyz=(Phi / 2.0, A if rigid else 0.0, 0.0),
            periodAngles_Gs_to_Wn_ixyz=(T if Phi else 0.0, T if rigid else 0.0, 0.0),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, ph if rigid else 0.0, 0.0),
            rotationPointOffset_Gs_Ler=(PIVOT, 0.0, 0.0),
        ))
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=wing_movements)
    op = ps.operating_point.OperatingPoint(rho=RHO, vCg__E=V, alpha=0.0, beta=0.0)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    if static:
        dt = CHORD / nc / V
        movement = ps.movements.movement.Movement(
            [am], opm, delta_time=dt, num_chords=30)
        n_cycle = 1
    else:
        if steps_per_cycle is None:
            # wake panels about as long as the bound trailing-edge panels
            steps_per_cycle = int(round(T / (CHORD / nc / V)))
        dt = T / steps_per_cycle
        movement = ps.movements.movement.Movement(
            [am], opm, delta_time=dt, num_steps=cycles * steps_per_cycle + 1)
        n_cycle = steps_per_cycle
    problem = ps.problems.UnsteadyProblem(movement, only_final_results=False)
    return problem, dict(dt=dt, n_cycle=n_cycle, static=static)


def make_solver(problem):
    import pterasoftware as ps
    base = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver

    class Recording(base):
        """Records, at every step, the total force and the aerodynamic power.

        power = - sum over bound legs and panels of force . velocity of the
        point the force acts on (relative to the body, which is held). The
        solver's `_calculate_current_movement_velocities_at_*` return the
        APPARENT velocity due to the motion, i.e. minus the point's velocity
        (backward difference of its position), so power = + sum F . apparent.
        """

        def _load_calculation_moment_processing_hook(self, r, fr, l, b, u):
            p = 0.0
            p += np.sum(r * self._calculate_current_movement_velocities_at_right_leg_centers())
            p += np.sum(fr * self._calculate_current_movement_velocities_at_front_leg_centers())
            p += np.sum(l * self._calculate_current_movement_velocities_at_left_leg_centers())
            p += np.sum(b * self._calculate_current_movement_velocities_at_back_leg_centers())
            p += np.sum(u * self._calculate_current_movement_velocities_at_collocation_points())
            F = (r + fr + l + b + u).sum(axis=0)
            Fs = (r + fr + l + b).sum(axis=0)
            w = self.current_airplanes[0].wings[0]      # the y > 0 wing
            P = w.panels
            le_t, te_t = P[0, -1].Frpp_GP1_CgP1, P[-1, -1].Brpp_GP1_CgP1
            le_r, te_r = P[0, 0].Flpp_GP1_CgP1, P[-1, 0].Blpp_GP1_CgP1
            pitch = lambda le, te: float(np.degrees(np.arctan2(le[2] - te[2], te[0] - le[0])))
            self.rec.append(dict(
                step=int(self._current_step), Fx=float(F[0]), Fy=float(F[1]),
                Fz=float(F[2]), Fx_steady=float(Fs[0]), Fz_steady=float(Fs[2]),
                power=float(p),
                tip_z=float(0.5 * (le_t[2] + te_t[2])), tip_y=float(le_t[1]),
                tip_pitch=pitch(le_t, te_t), root_pitch=pitch(le_r, te_r),
                tip_pivot_z=float(le_t[2] + 0.1 * (te_t[2] - le_t[2])),
                root_pivot_z=float(le_r[2] + 0.1 * (te_r[2] - le_r[2])),
            ))
            return super()._load_calculation_moment_processing_hook(r, fr, l, b, u)

    s = Recording(problem)
    s.rec = []
    return s


def run_case(case):
    """case: dict of build() arguments plus 'tag', 'group', 'free_wake'."""
    kw = {k: case[k] for k in ("V", "f", "Phi", "A", "inc", "share")}
    for k in ("sign", "nc", "nseg", "ns_per_seg", "steps_per_cycle", "cycles", "flat", "y_root"):
        if k in case:
            kw[k] = case[k]
    t0 = time.time()
    out = dict(case)
    try:
        problem, meta = build(**kw)
        solver = make_solver(problem)
        solver.run(prescribed_wake=not case.get("free_wake", False),
                   calculate_streamlines=False, show_progress=False)
        rec = solver.rec
        n = meta["n_cycle"]
        arr = lambda k, sl: np.array([r[k] for r in rec[sl]])
        last = slice(len(rec) - n, len(rec))
        prev = slice(len(rec) - 2 * n, len(rec) - n) if len(rec) >= 2 * n + 1 and not meta["static"] else None
        lift = float(arr("Fz", last).mean())
        thrust = float(-arr("Fx", last).mean())     # geometry x points aft
        power = float(arr("power", last).mean())
        out.update(
            ok=True, dt=meta["dt"], steps_per_cycle_used=n, n_steps=len(rec),
            panels=int(solver.num_panels),
            lift=lift, thrust=thrust, power=power,
            side=float(arr("Fy", last).mean()),
            CL=lift / (0.5 * RHO * case["V"] ** 2 * 2.0 * (Y_TIP - case.get("y_root", Y_ROOT)) * CHORD),
            lift_min=float(arr("Fz", last).min()), lift_max=float(arr("Fz", last).max()),
            finite=bool(np.all(np.isfinite(arr("Fz", slice(None))))),
        )
        if prev is not None:
            out.update(
                lift_prev=float(arr("Fz", prev).mean()),
                thrust_prev=float(-arr("Fx", prev).mean()),
                power_prev=float(arr("power", prev).mean()))
        if not meta["static"]:
            z = arr("tip_z", last)
            tp, rp = arr("tip_pitch", last), arr("root_pitch", last)
            vz = np.gradient(z, meta["dt"])
            i_dn, i_up = int(np.argmin(vz)), int(np.argmax(vz))
            out.update(
                tip_twist_at_mid_downstroke=float(tp[i_dn] - rp[i_dn]),
                tip_twist_at_mid_upstroke=float(tp[i_up] - rp[i_up]),
                tip_pitch_at_mid_downstroke=float(tp[i_dn]),
                tip_pitch_at_mid_upstroke=float(tp[i_up]),
                root_pitch_at_mid_downstroke=float(rp[i_dn]),
                root_pitch_at_mid_upstroke=float(rp[i_up]),
                tip_z_min=float(z.min()), tip_z_max=float(z.max()),
                tip_vz_min=float(vz.min()),
            )
        if case.get("keep_series"):
            out["series"] = rec
    except Exception as e:      # a diverged or unbuildable case is a row, not a crash
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:400])
    out["wall_s"] = round(time.time() - t0, 1)
    return out


INCS = [-8, -6, -4, -2, 0, 2, 4, 6, 8]


def plan():
    cases = []

    def sweep(group, V, f, Phi, A, share, incs=INCS, **extra):
        for inc in incs:
            cases.append(dict(group=group, V=V, f=f, Phi=Phi, A=A, share=share,
                              inc=inc, **extra))

    sweep("1", 8, 2.5, 40, 15, 1)
    sweep("1-opposite-sign", 8, 2.5, 40, 15, 1, sign=-1.0)
    sweep("2", 8, 2.0, 50, 15, 1)
    sweep("2", 8, 3.0, 40, 20, 1)
    sweep("2", 8, 3.5, 50, 15, 0)
    sweep("2", 8, 2.5, 40, 5, 0)
    for f, Phi, A in [(2.0, 60, 15), (2.5, 50, 15), (3.0, 40, 15), (3.5, 60, 20), (2.5, 60, 20)]:
        sweep("3", 6, f, Phi, A, 1, incs=INCS + [10, 12])
    sweep("4a", 8, 2.5, 0, 0, 1)
    sweep("4a-flat", 8, 2.5, 0, 0, 1, incs=[0, 4], flat=True)
    # 4b: sensitivity at one incidence of run 1 (inc = 0, 2 bracket the trim)
    for inc in (0, 2):
        cases.append(dict(group="4b-half-step", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, steps_per_cycle=200))
        cases.append(dict(group="4b-nc16", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, nc=16))
        cases.append(dict(group="4b-nc16-half-step", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, nc=16, steps_per_cycle=200))
        cases.append(dict(group="4b-span24", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, ns_per_seg=4))
        cases.append(dict(group="4b-5cycles", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, cycles=5))
        cases.append(dict(group="4b-free-wake", V=8, f=2.5, Phi=40, A=15, share=1, inc=inc, free_wake=True))
    for inc in (6, 8):
        cases.append(dict(group="3-free-wake", V=6, f=2.5, Phi=50, A=15, share=1, inc=inc, free_wake=True))
    # the root gap closed to 2 mm (the hull as a perfect seal), as a bracket
    sweep("4a-sealed", 8, 2.5, 0, 0, 1, y_root=0.001)
    sweep("4a-flat-sealed", 8, 2.5, 0, 0, 1, incs=[0, 4], flat=True, y_root=0.001)
    sweep("1-sealed", 8, 2.5, 40, 15, 1, y_root=0.001)
    sweep("3-sealed", 6, 2.5, 50, 15, 1, incs=[2, 4, 6, 8, 10], y_root=0.001)
    return cases


def key(c):
    return json.dumps({k: c.get(k) for k in (
        "group", "V", "f", "Phi", "A", "share", "inc", "sign", "nc", "nseg",
        "ns_per_seg", "steps_per_cycle", "cycles", "flat", "free_wake", "y_root")}, sort_keys=True)


def load():
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def run_all(workers):
    import multiprocessing as mp
    done = {key(r) for r in load() if r.get("ok")}
    todo = [c for c in plan() if key(c) not in done]
    print(f"{len(todo)} cases to run, {len(done)} already in {RESULTS.name}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, RESULTS.open("a") as fh:
        for r in pool.imap_unordered(run_case, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["group"], r["V"], r["f"], r["Phi"], r["A"], r["share"], r["inc"],
                  "ok" if r["ok"] else r["error"], r["wall_s"], flush=True)


def trim(rows):
    """Interpolate a sweep (rows of one configuration) to lift = weight."""
    rows = sorted(rows, key=lambda r: r["inc"])
    inc = np.array([r["inc"] for r in rows], float)
    L = np.array([r["lift"] for r in rows])
    if not (L.min() <= WEIGHT <= L.max()):
        return None
    i = int(np.searchsorted(L, WEIGHT))        # lift rises with inc (checked below)
    if not np.all(np.diff(L) > 0):
        return None
    i = max(1, min(i, len(L) - 1))
    w = (WEIGHT - L[i - 1]) / (L[i] - L[i - 1])
    g = lambda k: float((1 - w) * rows[i - 1][k] + w * rows[i][k])
    return dict(inc_trim=float((1 - w) * inc[i - 1] + w * inc[i]),
                thrust_trim=g("thrust"), power_trim=g("power"),
                bracket=(float(inc[i - 1]), float(inc[i])))


def read():
    rows = [r for r in load() if r.get("ok")]
    bad = [r for r in load() if not r.get("ok")]
    print(f"{len(rows)} good rows, {len(bad)} failed")
    for r in bad:
        print("  FAILED", key(r), r.get("error"))
    groups = {}
    for r in rows:
        cfg = (r["group"], r["V"], r["f"], r["Phi"], r["A"], r["share"])
        groups.setdefault(cfg, []).append(r)
    print("\ngroup V f Phi A share | inc_trim thrust_trim power_trim | max cycle-to-cycle change (lift, thrust, power)")
    for cfg in sorted(groups):
        g = groups[cfg]
        if len(g) < 3 or cfg[0].startswith("4a"):
            continue
        t = trim(g)
        per = [(abs(r["lift"] - r["lift_prev"]), abs(r["thrust"] - r["thrust_prev"]),
                abs(r["power"] - r["power_prev"])) for r in g if "lift_prev" in r]
        per = np.max(per, axis=0) if per else (np.nan,) * 3
        if t is None:
            L = sorted((r["inc"], round(r["lift"], 3)) for r in g)
            print(cfg, "| NO TRIM in range; lift:", L)
        else:
            print(cfg, f"| {t['inc_trim']:+.2f} deg  {t['thrust_trim']:+.3f} N  {t['power_trim']:.2f} W |",
                  f"{per[0]:.3f} N {per[1]:.4f} N {per[2]:.3f} W")
    print("\nsweeps (inc: lift, thrust, power)")
    for cfg in sorted(groups):
        g = sorted(groups[cfg], key=lambda r: r["inc"])
        print(cfg)
        for r in g:
            print(f"   {r['inc']:+3d}: {r['lift']:8.3f} N {r['thrust']:+8.4f} N {r['power']:8.3f} W"
                  + (f"   twist@mid-down {r['tip_twist_at_mid_downstroke']:+.2f} @mid-up {r['tip_twist_at_mid_upstroke']:+.2f}"
                     if "tip_twist_at_mid_downstroke" in r else ""))
    # 4a
    for name in ("4a", "4a-flat", "4a-sealed", "4a-flat-sealed"):
        g = sorted([r for r in rows if r["group"] == name], key=lambda r: r["inc"])
        if len(g) < 2:
            continue
        a = np.radians([r["inc"] for r in g])
        CL = np.array([r["CL"] for r in g])
        slope, c0 = np.polyfit(a, CL, 1)
        res = CL - (slope * a + c0)
        q = 0.5 * RHO * 64
        helm = 2 * np.pi * AR_FULL / (2 + np.sqrt(AR_FULL ** 2 + 4))
        S = 2.0 * (Y_TIP - g[0].get("y_root", Y_ROOT)) * CHORD
        print(f"\n{name}: dCL/dalpha = {slope:.3f} /rad on the panel area S = {S:.4f} m2 "
              f"({slope * S / (B_REF * CHORD):.3f} on b*c = {B_REF * CHORD:.4f}); "
              f"zero-lift angle = {np.degrees(-c0 / slope):+.2f} deg; "
              f"max |residual| of the line = {np.abs(res).max():.2e} in CL; "
              f"Helmbold(AR {AR_FULL:.2f}) = {helm:.3f}; ratio {slope / helm:.3f}")
        print("   CDi/CL^2*pi*AR at the ends:",
              [round(-r["thrust"] / (q * S) / max(r["CL"] ** 2, 1e-12) * np.pi * B_REF ** 2 / S, 3) for r in (g[0], g[-1])])
    # 4b
    base = {r["inc"]: r for r in rows if r["group"] == "1"}
    print("\n4b sensitivity against run 1 (same inc): d lift, d thrust, d power")
    for r in sorted((r for r in rows if r["group"].startswith("4b")), key=lambda r: (r["group"], r["inc"])):
        b = base.get(r["inc"])
        if b:
            print(f"   {r['group']:20s} inc {r['inc']:+d}: {r['lift'] - b['lift']:+.3f} N ({100 * (r['lift'] / b['lift'] - 1):+.1f}%)  "
                  f"{r['thrust'] - b['thrust']:+.4f} N  {r['power'] - b['power']:+.3f} W ({100 * (r['power'] / b['power'] - 1):+.1f}%)"
                  f"   [{r['wall_s']} s]")
    for name, grp in (("4b-half-step", None), ("4b-nc16", None), ("4b-nc16-half-step", None),
                      ("4b-span24", None), ("4b-5cycles", None), ("4b-free-wake", None)):
        g = [r for r in rows if r["group"] == name]
        if len(g) >= 2:
            t = trim(g)
            if t:
                print(f"   trim on {name}: inc {t['inc_trim']:+.2f} thrust {t['thrust_trim']:+.3f} power {t['power_trim']:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "read"
    if cmd == "probe":
        c = dict(group="probe", V=8, f=2.5, Phi=40, A=15, share=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                 inc=2, sign=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, cycles=2, keep_series=True)
        r = run_case(c)
        s = r.pop("series", [])
        print(json.dumps(r, indent=1))
        for row in s[:: max(1, len(s) // 24)]:
            print({k: round(v, 4) for k, v in row.items()})
    elif cmd == "all":
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        read()
