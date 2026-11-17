"""Pass X2, stage A driver: build and run every static case, at most CORES
cores at a time, steady first and time-accurate where steady does not
converge; one row per finished case appended to results.jsonl.

    cd /home/hundo/Projects/Dytiscidae/.cfd && pixi run python \
        <repo>/experiments/flapper/x2/run_a.py [CORES] [only-substring ...]

Rule (NOTES.md, fixed before any result): steady converged if CL varies by
<= 0.005 and CD by <= 0.0005 over the last 1000 iterations (or what exists
after residualControl stopped it).  Otherwise pimpleFoam from the steady
fields to t = 40, the dt-weighted mean over t 20..40.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import case as C  # noqa: E402

ROOT = Path(os.environ.get("X2_ROOT", "/home/hundo/Projects/Dytiscidae/.cfd/cases/X2"))
MESH = Path("/home/hundo/Projects/Dytiscidae/.cfd/cases/X2/meshes")
RESULTS = Path(os.environ.get("X2_RESULTS", HERE / "results.jsonl"))
NOTES = Path(os.environ.get("X2_NOTES", HERE / "NOTES.md"))
ITERS = int(os.environ.get("X2_ITERS", 8000))            # env overrides: smoke tests only
T_END, T_AVG = float(os.environ.get("X2_TEND", 40.0)), float(os.environ.get("X2_TEND", 40.0)) / 2
TIMEOUT = 5400               # 90 min per run
GOE_ALPHAS = [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
PLATE_ALPHAS = [2, 5, 8, 10, 15, 20, 24]
CLOSURES = ["kOmegaSST", "kOmegaSSTLM"]
LOCK = threading.Lock()


def plan():
    jobs = []
    for a in (4, 12):                      # A1 first
        for cl in CLOSURES:
            for lev in (0, 1):
                jobs.append(("goe417a", lev, cl, a))
    for a in GOE_ALPHAS:
        if a in (4, 12):
            continue
        for cl in CLOSURES:
            jobs.append(("goe417a", 0, cl, a))
    for a in PLATE_ALPHAS:
        for cl in CLOSURES:
            jobs.append(("plate", 0, cl, a))
    return jobs


def name(sec, lev, cl, a):
    return f"A_{sec}_L{lev}_{cl}_a{a:+05.1f}"


def msh(sec, lev):
    return str(MESH / f"{'goe' if sec == 'goe417a' else 'plate'}_L{lev}.msh")


def read_coeffs(case):
    f = case / "postProcessing" / "forceCoeffs1"
    rows = []
    for d in sorted(f.glob("*"), key=lambda p: float(p.name)):
        # a restart into an existing output directory writes coefficient_<t>.dat beside the first file
        for line in "\n".join(f.read_text() for f in sorted(d.glob("coefficient*.dat"))).splitlines():
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            try:
                rows.append([float(v[0]), float(v[1]), float(v[4]), float(v[7])])   # t Cd Cl CmPitch
            except ValueError:
                pass
    rows.sort(key=lambda r: r[0])
    out, last = [], None
    for r in rows:                           # drop duplicated times after a restart
        if last is None or r[0] > last:
            out.append(r)
            last = r[0]
    return out


def yplus(case):
    f = case / "postProcessing" / "yPlus1"
    best = None
    for d in sorted(f.glob("*"), key=lambda p: float(p.name)):
        for line in (d / "yPlus.dat").read_text().splitlines():
            v = line.split()
            if len(v) >= 5 and v[1] == "wall":
                best = dict(t=float(v[0]), min=float(v[2]), max=float(v[3]), mean=float(v[4]))
    return best


def residuals(log):
    last = {}
    if not log.exists():
        return last
    for line in log.read_text(errors="replace").splitlines():
        if "Solving for" in line and "Initial residual" in line:
            var = line.split("Solving for")[1].split(",")[0].strip()
            r0 = float(line.split("Initial residual =")[1].split(",")[0])
            last[var] = r0
    return last


def converged(co, by_residuals=False):
    """The NOTES rule: over the last 1000 iterations CL varies by <= 0.005 and CD by <= 0.0005
    (or over the last half of what exists, when residualControl stopped the run first)."""
    tail = co[-1000:] if not by_residuals else co[-min(1000, max(len(co) // 2, 1)):]
    if (len(tail) < 1000 and not by_residuals) or not all(math.isfinite(x) for r in tail for x in r[1:]):
        return False
    cls_, cds = [r[2] for r in tail], [r[1] for r in tail]
    return (max(cls_) - min(cls_) <= 0.005) and (max(cds) - min(cds) <= 0.0005)


def mpi(n):
    """mpiexec with the MPICH Pstream: the env's libOpenFOAM RPATH looks for lib/sys-mpich, which
    the conda package does not ship (it has lib/mpich-3.3), so the dummy Pstream was loaded and
    every parallel run failed ("The dummy Pstream library cannot be used in parallel mode").
    The solver alone gets the MPICH one preloaded; nothing in the environment is changed."""
    pre = os.path.join(os.environ["CONDA_PREFIX"], "lib", "mpich-3.3", "libPstream.so")
    return ["mpiexec", "-n", str(n), "env", f"LD_PRELOAD={pre}"]


def run(cmd, case, log, timeout, cores=1, watch=False):
    """watch: a steady run is stopped (stopAt writeNow, it writes its fields) as
    soon as the convergence rule holds, instead of running out its iterations."""
    t0 = time.time()
    env = dict(os.environ, PWD=str(case))
    with open(case / log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=case, stdout=fh, stderr=subprocess.STDOUT, env=env)
        rc, stopped = None, False
        while rc is None:
            try:
                rc = p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if time.time() - t0 > timeout:
                    p.terminate()
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    rc = "timeout"
                    break
                if watch and not stopped:
                    try:
                        if converged(read_coeffs(case)):
                            cdp = case / "system" / "controlDict"
                            cdp.write_text(cdp.read_text().replace("stopAt          endTime;", "stopAt          writeNow;"))
                            stopped = True
                    except Exception:                       # noqa: BLE001
                        pass
    return rc, time.time() - t0


def note(msg):
    with LOCK:
        with open(NOTES, "a") as fh:
            fh.write(f"- {time.strftime('%m-%d %H:%M')} {msg}\n")


def emit(row):
    with LOCK:
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(row) + "\n")


def latest_time(case):
    ts = []
    for d in case.iterdir():
        try:
            if d.is_dir() and float(d.name) > 0 and (d / "U").exists():
                ts.append((float(d.name), d))
        except ValueError:
            pass
    return max(ts)[1] if ts else None


def decompose(case, n):
    C.write(case / "system" / "decomposeParDict", "dictionary", "decomposeParDict",
            f"numberOfSubdomains {n};\nmethod scotch;\n")


class Cores:
    """At most `n` cores in use; acquire k atomically."""
    def __init__(self, n):
        self.free, self.cv = n, threading.Condition()

    def take(self, k):
        with self.cv:
            self.cv.wait_for(lambda: self.free >= k)
            self.free -= k

    def give(self, k):
        with self.cv:
            self.free += k
            self.cv.notify_all()


POOL = None


def job(spec, cores_for_transient):
    sec, lev, cl, a = spec
    nm = name(sec, lev, cl, a)
    case = ROOT / nm
    if (case / "DONE").exists():
        return
    if (case / "constant" / "polyMesh" / "faces").exists() and (case / "log.checkMesh").exists():
        # resume (startFrom latestTime): an interrupted driver left this case's mesh and written iterations
        summ = [l.strip() for l in (case / "log.checkMesh").read_text().splitlines()
                if any(k in l for k in ("cells:", "Max aspect", "non-orthogonality", "skewness", "Mesh OK", "Failed", "***"))]
        if (case / "log.simpleFoam").exists():
            (case / "log.simpleFoam").rename(case / f"log.simpleFoam.{int(time.time())}")
        cdp = case / "system" / "controlDict"
        cdp.write_text(cdp.read_text().replace("stopAt          writeNow;", "stopAt          endTime;"))
        resumed = True
    else:
        shutil.rmtree(case, ignore_errors=True)
        summ = C.static(case, msh(sec, lev), cl, a, "steady", iters=ITERS)
        resumed = False
    if not any("Mesh OK" in s for s in summ):
        raise RuntimeError(f"checkMesh failed: {summ}")
    ncell = int([s for s in summ if "cells:" in s][0].split()[-1])
    rc, wall = run(["simpleFoam"], case, "log.simpleFoam", TIMEOUT, watch=True)
    co = read_coeffs(case)
    row = dict(stage="A", case=nm, section=sec, mesh=f"L{lev}", closure=cl, alpha=a, cells=ncell,
               checkMesh=summ, steady_rc=rc, steady_wall_s=round(wall, 1), steady_iters=len(co))
    tail = co[-1000:]
    ok = bool(co) and all(math.isfinite(x) for r in tail for x in r[1:])
    if ok:
        cls_ = [r[2] for r in tail]
        cds = [r[1] for r in tail]
        row.update(steady_CL_range=max(cls_) - min(cls_), steady_CD_range=max(cds) - min(cds))
        by_res = rc == 0 and "SIMPLE solution converged" in (case / "log.simpleFoam").read_text(errors="replace")
        conv = converged(co, by_res)
        row["stopped_by_residualControl"] = by_res
    else:
        conv = False
    row["resumed"] = resumed
    row["steady_res"] = residuals(case / "log.simpleFoam")
    row["yplus_steady"] = yplus(case)
    if conv:
        last = co[-1]
        row.update(mode="steady", CL=last[2], CD=last[1], Cm=last[3], yplus=row["yplus_steady"],
                   window=f"last {len(tail)} iterations (of {len(co)})", wall_s=round(wall, 1))
        emit(row)
        note(f"{nm}: steady, {len(co)} it, CL {last[2]:.4f} CD {last[1]:.5f} Cm {last[3]:.4f}, "
             f"CL range {row['steady_CL_range']:.2e}, y+max {row['yplus_steady']['max'] if row['yplus_steady'] else 'na'}, {wall/60:.1f} min")
        (case / "DONE").write_text("steady")
        return
    # time-accurate from the steady fields
    tr = ROOT / (nm + "_tr")
    if tr.exists() and ((tr / "processor0").exists() or latest_time(tr) is not None):
        # resume an interrupted transient from its last written time
        n = cores_for_transient
        if (tr / "log.pimpleFoam").exists():
            (tr / "log.pimpleFoam").rename(tr / f"log.pimpleFoam.{int(time.time())}")
        if n > 1:
            POOL.take(n - 1)
            try:
                rc2, wall2 = run(mpi(n) + ["pimpleFoam", "-parallel"], tr, "log.pimpleFoam", TIMEOUT)
            finally:
                POOL.give(n - 1)
        else:
            rc2, wall2 = run(["pimpleFoam"], tr, "log.pimpleFoam", TIMEOUT)
        row["transient_resumed"] = True
        return finish_transient(row, nm, case, tr, rc, rc2, wall, wall2, n)
    shutil.rmtree(tr, ignore_errors=True)
    (tr / "constant").mkdir(parents=True)
    shutil.copytree(case / "constant" / "polyMesh", tr / "constant" / "polyMesh")
    src = latest_time(case) if ok else None
    if src is not None:
        (tr / "0").mkdir()
        for f in src.iterdir():
            if f.is_file() and f.name not in ("yPlus", "uniform"):
                shutil.copy(f, tr / "0" / f.name)
    C.static(tr, msh(sec, lev), cl, a, "transient", t_end=T_END)
    if src is None:
        k_inf, om_inf = 1.5 * C.TU ** 2, 1.5 * C.TU ** 2 / 1e-5
        C.zero(tr, cl, math.cos(math.radians(a)), math.sin(math.radians(a)), k_inf, om_inf)
    n = cores_for_transient
    if n > 1:
        decompose(tr, n)
        run(["decomposePar", "-force"], tr, "log.decomposePar", 1200)
        POOL.take(n - 1)
        try:
            rc2, wall2 = run(mpi(n) + ["pimpleFoam", "-parallel"], tr, "log.pimpleFoam", TIMEOUT)
        finally:
            POOL.give(n - 1)
    else:
        rc2, wall2 = run(["pimpleFoam"], tr, "log.pimpleFoam", TIMEOUT)
    return finish_transient(row, nm, case, tr, rc, rc2, wall, wall2, n)


def finish_transient(row, nm, case, tr, rc, rc2, wall, wall2, n):
    co2 = read_coeffs(tr)
    t_last = co2[-1][0] if co2 else 0.0
    t0 = T_AVG if t_last >= T_END - 1e-6 else t_last / 2
    win = [r for r in co2 if r[0] >= t0]
    if len(win) > 2 and all(math.isfinite(x) for r in win for x in r[1:]):
        dts = [win[k + 1][0] - win[k][0] for k in range(len(win) - 1)]
        tot = sum(dts)
        mean = lambda j: sum(win[k][j] * dts[k] for k in range(len(dts))) / tot
        CL, CD, CM = mean(2), mean(1), mean(3)
        row.update(mode="transient", CL=CL, CD=CD, Cm=CM,
                   CL_min=min(r[2] for r in win), CL_max=max(r[2] for r in win),
                   CD_min=min(r[1] for r in win), CD_max=max(r[1] for r in win),
                   window=f"t {t0:.2f}..{t_last:.2f} (dt-weighted mean)", transient_rc=rc2,
                   transient_wall_s=round(wall2, 1), wall_s=round(wall + wall2, 1),
                   transient_res=residuals(tr / "log.pimpleFoam"), yplus=yplus(tr), transient_cores=n)
        emit(row)
        note(f"{nm}: steady not converged (CL range {row.get('steady_CL_range', float('nan')):.3g}); "
             f"transient {t0:.1f}..{t_last:.1f}: CL {CL:.4f} [{row['CL_min']:.3f},{row['CL_max']:.3f}] "
             f"CD {CD:.5f} Cm {CM:.4f}, rc {rc2}, {wall2/60:.1f} min")
    else:
        row.update(mode="failed", transient_rc=rc2, transient_wall_s=round(wall2, 1), t_reached=t_last)
        emit(row)
        note(f"{nm}: FAILED (steady rc {rc}, transient rc {rc2}, t reached {t_last:.2f})")
    (case / "DONE").write_text("transient")


def main():
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    only = sys.argv[2:]
    jobs = [j for j in plan() if not only or any(o in name(*j) for o in only)]
    global POOL
    POOL = Cores(cores)

    def go(spec):
        # one core for the steady run; a fine-mesh transient takes 3 more (4-way decomposed)
        POOL.take(1)
        try:
            job(spec, 4 if spec[1] == 1 else 1)
        except Exception as e:                       # noqa: BLE001
            note(f"{name(*spec)}: driver error {e!r}")
        finally:
            POOL.give(1)
    with ThreadPoolExecutor(max_workers=cores) as ex:
        list(ex.map(go, jobs))
    note("stage A driver finished")


if __name__ == "__main__":
    main()
