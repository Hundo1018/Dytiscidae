"""Pass X2, stage B: the still-fluid plunge check, then the station B8 and B6
through the outside solver (whole mesh moved rigidly, five beats, the last
three averaged).

    cd /home/hundo/Projects/Dytiscidae/.cfd && pixi run python \
        <repo>/experiments/flapper/x2/run_b.py still|B8|B6|read [MESH_LEVEL] [CLOSURE] [NPROC]

Kinematics from kinematics.json (kinematics.py, read off the rig): chord
c = 0.34 / (2 x 0.66) m, plunge h0 = r Phi/2 at r = 0.7 x 0.66 m, pitch
theta0 + 0.7 Psi cos(wt) nose up, plunge h0 sin(wt) up; non-dimensional here
on c and V: h0/c, omega c/V, nu = (mu/rho)_air / (V c).

Coefficients (per unit span, q = 1/2 with unit stream): CN = Fy / (q dz),
CX = -Fx / (q dz) (thrust +), CP = -(Fy dh/dt + M theta') / (q dz) with M
the nose-up moment about the moving pivot (c/4) and theta' the nose-up rate.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import case as C  # noqa: E402

ROOT = Path("/home/hundo/Projects/Dytiscidae/.cfd/cases/X2")
MESH = ROOT / "meshes"
RESULTS = HERE / "results.jsonl"
NOTES = HERE / "NOTES.md"
NU_AIR = 1.81e-5 / 1.225            # dytiscidae.physics.medium.AIR
CHORD = 0.34 / (2 * 0.66)
TIMEOUT = int(os.environ.get("X2_B_TIMEOUT", 5400))
BEATS, AVG = 5, 3


def note(msg):
    with open(NOTES, "a") as fh:
        fh.write(f"- {time.strftime('%m-%d %H:%M')} {msg}\n")


def emit(row):
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def mpi(n):
    """mpiexec with the MPICH Pstream: the env's libOpenFOAM RPATH looks for lib/sys-mpich, which
    the conda package does not ship (it has lib/mpich-3.3), so the dummy Pstream was loaded and
    every parallel run failed ("The dummy Pstream library cannot be used in parallel mode").
    The solver alone gets the MPICH one preloaded; nothing in the environment is changed."""
    pre = os.path.join(os.environ["CONDA_PREFIX"], "lib", "mpich-3.3", "libPstream.so")
    return ["mpiexec", "-n", str(n), "env", f"LD_PRELOAD={pre}"]


def run(cmd, case, log, timeout):
    t0 = time.time()
    with open(case / log, "w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=case, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                env=dict(os.environ, PWD=str(case))).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, time.time() - t0


def forces(case):
    rows = {}
    base = case / "postProcessing" / "forces1"
    for d in sorted(base.glob("*"), key=lambda p: float(p.name)):
        F = [l.split() for l in (d / "force.dat").read_text().splitlines() if l and not l.startswith("#")]
        M = [l.split() for l in (d / "moment.dat").read_text().splitlines() if l and not l.startswith("#")]
        for f, m in zip(F, M):
            rows[float(f[0])] = (float(f[1]), float(f[2]), float(m[3]))       # Fx Fy Mz about the origin
    t = sorted(rows)
    return t, [rows[x] for x in t]


def solve(case, n_proc):
    if n_proc > 1:
        run(["decomposePar", "-force"], case, "log.decomposePar", 1200)
        return run(mpi(n_proc) + ["pimpleFoam", "-parallel"], case, "log.pimpleFoam", TIMEOUT)
    return run(["pimpleFoam"], case, "log.pimpleFoam", TIMEOUT)


def still(level=0, n_proc=1):
    """Laminar, still fluid, plunge h0 = 0.02 c at omega = 2 pi (period 1), zero
    mean pitch: added mass from the force's part in phase with the acceleration."""
    import numpy as np
    nm = f"B_still_L{level}"
    case = ROOT / nm
    shutil.rmtree(case, ignore_errors=True)
    h0, om = 0.02, 2 * math.pi
    # maxCo 10: the flow round the sharp base corners runs ~12x the plunge speed
    # through cells 1e-4 c thick (measured: Co 4 there at dt 1.6e-4 with h0 = 0.05)
    summ = C.moving(case, str(MESH / f"goe_L{level}.msh"), "laminar", 1e-5, 0.0, h0=h0, omega=om,
                    theta0_deg=0.0, amp_deg=0.0, t_end=4.0, n_proc=n_proc, max_co=10.0)
    rc, wall = solve(case, n_proc)
    t, F = forces(case)
    t = np.array(t)
    Fy = np.array([f[1] for f in F]) / C.DZ
    sel = t >= 2.0
    acc = -h0 * om * om * np.sin(om * t[sel])
    vel = h0 * om * np.cos(om * t[sel])
    A = np.stack([acc, vel, np.ones(sel.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, Fy[sel], rcond=None)
    m_a = -coef[0]
    want = math.pi / 4                                   # pi rho c^2 / 4, rho = 1, c = 1
    resid = Fy[sel] - A @ coef
    row = dict(stage="B", case=nm, mesh=f"L{level}", closure="laminar", motion=dict(h0=h0, omega=om, U=0.0),
               added_mass=float(m_a), added_mass_ref=want, ratio=float(m_a / want), damping_coef=float(coef[1]),
               fit_resid_rms_over_amp=float(np.sqrt(np.mean(resid ** 2)) / np.abs(Fy[sel]).max()),
               window="t 2..4 (periods 3-4)", rc=rc, wall_s=round(wall, 1), checkMesh=summ,
               cells=int([s for s in summ if "cells:" in s][0].split()[-1]))
    emit(row)
    note(f"{nm}: added mass {m_a:.4f} vs pi c^2/4 = {want:.4f} (ratio {m_a / want:.3f}); "
         f"in-quadrature coef {coef[1]:.4g}; fit residual {row['fit_resid_rms_over_amp']:.3f} of peak; rc {rc}, {wall/60:.1f} min")
    return row


def station(name, level, closure, n_proc, max_co=8.0):
    kin = json.loads((HERE / "kinematics.json").read_text())[name]
    V, f = kin["V"], kin["f"]
    h0 = kin["plunge_amp_spec"] / CHORD
    om = 2 * math.pi * f * CHORD / V
    th0 = kin["pitch_fit_deg"]["mean"]
    amp = 0.7 * kin["Psi"]
    nu = NU_AIR / (V * CHORD)
    T = 2 * math.pi / om
    nm = f"B_{name}_L{level}_{closure}"
    case = ROOT / nm
    shutil.rmtree(case, ignore_errors=True)
    summ = C.moving(case, str(MESH / f"goe_L{level}.msh"), closure, nu, 1.0, h0=h0, omega=om,
                    theta0_deg=th0, amp_deg=amp, t_end=BEATS * T, n_proc=n_proc, max_co=max_co)
    note(f"{nm}: built (h0/c {h0:.4f}, k {om / 2:.4f}, omega* {om:.4f}, T* {T:.3f}, theta {th0:.3f} + {amp:.3f} cos, "
         f"Re {1 / nu:.4g}, maxCo {max_co}, {n_proc} procs); running")
    rc, wall = solve(case, n_proc)
    return read_station(name, level, closure, rc=rc, wall=wall, summ=summ)


def read_station(name, level, closure, rc=None, wall=None, summ=None):
    import numpy as np
    kin = json.loads((HERE / "kinematics.json").read_text())[name]
    V, f = kin["V"], kin["f"]
    h0 = kin["plunge_amp_spec"] / CHORD
    om = 2 * math.pi * f * CHORD / V
    th0, amp = kin["pitch_fit_deg"]["mean"], 0.7 * kin["Psi"]
    T = 2 * math.pi / om
    nm = f"B_{name}_L{level}_{closure}"
    case = ROOT / nm
    t, F = forces(case)
    t = np.array(t)
    F = np.array(F)
    if len(t) < 10:
        row = dict(stage="B", case=nm, mode="failed", rc=rc, t_reached=float(t[-1]) if len(t) else 0.0)
        emit(row)
        note(f"{nm}: FAILED, rc {rc}")
        return row
    Fx, Fy, Mz0 = F[:, 0] / C.DZ, F[:, 1] / C.DZ, F[:, 2] / C.DZ
    h = h0 * np.sin(om * t)
    hd = h0 * om * np.cos(om * t)
    thd = -math.radians(amp) * om * np.sin(om * t)            # nose-up rate
    Mp = Mz0 - (0.25 * Fy - h * Fx)                            # about the pivot (0.25, h), +z
    Mnu = -Mp
    q = 0.5
    CN, CX, CP, CM = Fy / q, -Fx / q, -(Fy * hd + Mnu * thd) / q, Mnu / q
    t_last = float(t[-1])
    done = t_last >= BEATS * T - 1e-6
    b_end = int(math.floor(t_last / T + 1e-9))
    b0 = max(b_end - AVG, 1)
    sel = (t >= b0 * T) & (t <= b_end * T)
    dts = np.gradient(t)

    def mean(x, s=sel):
        return float(np.sum(x[s] * dts[s]) / np.sum(dts[s]))
    beats = [mean(CN, (t >= b * T) & (t < (b + 1) * T)) for b in range(b_end)]
    last = (t >= (b_end - 1) * T) & (t <= b_end * T)
    ph = ((t[last] - (b_end - 1) * T) / T)
    hist = dict(phase=ph[::max(1, last.sum() // 200)].tolist(), CN=CN[last][::max(1, last.sum() // 200)].tolist())
    # downstroke: the plunge rate is negative (h = h0 sin wt moves down for wt in (pi/2, 3pi/2))
    down = last & (hd < 0)
    row = dict(stage="B", case=nm, station=name, mesh=f"L{level}", closure=closure,
               motion=dict(h0_over_c=h0, omega_star=om, k=om / 2, T_star=T, theta0_deg=th0, amp_deg=amp,
                           Re=1 / (NU_AIR / (V * CHORD)), pivot_x=0.25),
               CN=mean(CN), CX=mean(CX), CP=mean(CP), CM=mean(CM), beat_means_CN=beats,
               CN_max_last_beat=float(CN[last].max()), CN_max_downstroke=float(CN[down].max()) if down.any() else None,
               phase_of_max=float(ph[np.argmax(CN[last])]), window=f"beats {b0 + 1}-{b_end} of {BEATS} (t {b0 * T:.2f}..{b_end * T:.2f})",
               complete=bool(done), t_reached=t_last, rc=rc, wall_s=None if wall is None else round(wall, 1),
               history_last_beat=hist, checkMesh=summ)
    try:
        sys.path.insert(0, str(HERE))
        import run_a
        row["yplus"] = run_a.yplus(case)
        row["res"] = run_a.residuals(case / "log.pimpleFoam")
    except Exception:                                   # noqa: BLE001
        pass
    emit(row)
    note(f"{nm}: CN {row['CN']:.4f} CX {row['CX']:.4f} CP {row['CP']:.4f} CM {row['CM']:.4f}, beat CN means "
         f"{' '.join('%.3f' % b for b in beats)}, CN max (last beat) {row['CN_max_last_beat']:.3f} at phase "
         f"{row['phase_of_max']:.2f}, {row['window']}, complete {done}, rc {rc}")
    return row


if __name__ == "__main__":
    what = sys.argv[1]
    lev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    clo = sys.argv[3] if len(sys.argv) > 3 else "kOmegaSSTLM"
    npr = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if what == "still":
        print(still(lev, npr))
    elif what in ("B8", "B6"):
        r = station(what, lev, clo, npr)
        print({k: v for k, v in r.items() if k not in ("history_last_beat", "checkMesh")})
    elif what.startswith("read"):
        r = read_station(what.split(":")[1], lev, clo)
        print({k: v for k, v in r.items() if k not in ("history_last_beat", "checkMesh")})
