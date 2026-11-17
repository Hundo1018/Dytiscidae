"""Pass X2: write an OpenFOAM case for a static section (stage A) or a moving
one (stage B), from a gmsh mesh written by `mesh.py`.

Run from anywhere with the OpenFOAM environment loaded, e.g.

    cd /home/hundo/Projects/Dytiscidae/.cfd && pixi run python \
        <repo>/experiments/flapper/x2/case.py static CASE_DIR MESH.msh CLOSURE ALPHA [steady|transient]

Unit chord (leading edge at x=0), unit stream, nu = 1/Re.  The angle of attack
turns the stream, not the mesh.  The far field is one patch with
freestreamVelocity/freestreamPressure.  Freestream turbulence 0.1% (k = 1.5
(Tu U)^2), eddy-viscosity ratio 1, held against decay by SST's decayControl
(kInf, omegaInf), so the airfoil sees the stated level.  kOmegaSSTLM's inlet
ReThetat from Langtry-Menter's correlation at Tu = 0.1%, zero pressure
gradient.  Wall: noSlip (movingWallVelocity on a moving mesh), k = 0,
omegaWallFunction (viscous-sublayer value at y+ < 1), nutLowReWallFunction.
"""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAX_CO = 8.0                 # transient Courant limit (measured 0.31 s/step, dt 3.3e-3 on goe L0)
DZ = 0.01                    # mesh depth (mesh.py); reference area = chord x DZ
TU = 0.001
HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def re_theta_t(tu_percent: float) -> float:
    """Langtry-Menter (2009) eq. for Re_theta_t at zero pressure gradient, Tu <= 1.3%."""
    return 1173.51 - 589.428 * tu_percent + 0.2196 / tu_percent ** 2


def write(path: Path, cls: str, obj: str, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HDR.format(cls=cls, obj=obj) + body)


def field(case, name, dim, internal, wall, far, cls="volScalarField"):
    body = f"""dimensions      {dim};
internalField   {internal};
boundaryField
{{
    wall
    {{
{wall}
    }}
    farfield
    {{
{far}
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
    write(case / "0" / name, cls, name, body)


def mesh(case: Path, msh: str, log=None):
    """gmshToFoam, patch types, checkMesh; returns checkMesh's summary lines."""
    def run(cmd, logname):
        with open(case / logname, "w") as fh:
            subprocess.run(cmd, cwd=case, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=1800,
                           env=dict(os.environ, PWD=str(case.resolve())))
    run(["gmshToFoam", str(Path(msh).resolve())], "log.gmshToFoam")
    b = case / "constant" / "polyMesh" / "boundary"
    for patch, typ in (("front", "empty"), ("back", "empty"), ("wall", "wall"), ("farfield", "patch")):
        subprocess.run(["foamDictionary", "-entry", f"entry0/{patch}/type", "-set", typ, str(b)],
                       check=True, stdout=subprocess.DEVNULL, timeout=120)
        subprocess.run(["foamDictionary", "-entry", f"entry0/{patch}/physicalType", "-remove", str(b)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    run(["checkMesh"], "log.checkMesh")
    keep = [l.strip() for l in (case / "log.checkMesh").read_text().splitlines()
            if any(k in l for k in ("cells:", "Max aspect", "non-orthogonality", "skewness", "Mesh OK", "Failed", "***"))]
    return keep


def constant(case, nu, closure, moving=None):
    write(case / "constant" / "transportProperties", "dictionary", "transportProperties",
          f"transportModel  Newtonian;\nnu              {nu:.8g};\n")
    k_inf = 1.5 * TU ** 2
    om_inf = k_inf / nu                          # eddy-viscosity ratio 1
    if closure == "laminar":
        write(case / "constant" / "turbulenceProperties", "dictionary", "turbulenceProperties",
              "simulationType  laminar;\n")
    else:
        write(case / "constant" / "turbulenceProperties", "dictionary", "turbulenceProperties",
              f"""simulationType  RAS;
RAS
{{
    RASModel        {closure};
    turbulence      on;
    printCoeffs     on;
    {closure}Coeffs
    {{
        decayControl    yes;
        kInf            {k_inf:.6g};
        omegaInf        {om_inf:.6g};
    }}
}}
""")
    if moving:
        write(case / "constant" / "dynamicMeshDict", "dictionary", "dynamicMeshDict",
              f"""dynamicFvMesh   dynamicMotionSolverFvMesh;
motionSolverLibs (fvMotionSolvers);
motionSolver    solidBody;
solidBodyMotionFunction tabulated6DoFMotion;
CofG            ({moving['pivot']:.6g} 0 0);
timeDataFileName "<constant>/motion.dat";
""")
    return k_inf, om_inf


def zero(case, closure, ux, uy, k_inf, om_inf, moving=False):
    wallU = "        type            movingWallVelocity;\n        value           uniform (0 0 0);" if moving \
        else "        type            noSlip;"
    field(case, "U", "[0 1 -1 0 0 0 0]", f"uniform ({ux:.10g} {uy:.10g} 0)", wallU,
          f"        type            freestreamVelocity;\n        freestreamValue uniform ({ux:.10g} {uy:.10g} 0);\n"
          f"        value           uniform ({ux:.10g} {uy:.10g} 0);", cls="volVectorField")
    field(case, "p", "[0 2 -2 0 0 0 0]", "uniform 0", "        type            zeroGradient;",
          "        type            freestreamPressure;\n        freestreamValue uniform 0;\n        value           uniform 0;")
    if closure == "laminar":
        return
    io = lambda v: f"        type            inletOutlet;\n        inletValue      uniform {v:.6g};\n        value           uniform {v:.6g};"
    field(case, "k", "[0 2 -2 0 0 0 0]", f"uniform {k_inf:.6g}",
          "        type            fixedValue;\n        value           uniform 0;", io(k_inf))
    field(case, "omega", "[0 0 -1 0 0 0 0]", f"uniform {om_inf:.6g}",
          f"        type            omegaWallFunction;\n        value           uniform {om_inf:.6g};", io(om_inf))
    field(case, "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
          "        type            nutLowReWallFunction;\n        value           uniform 0;",
          "        type            calculated;\n        value           uniform 0;")
    if closure == "kOmegaSSTLM":
        rt = re_theta_t(100 * TU)
        field(case, "ReThetat", "[0 0 0 0 0 0 0]", f"uniform {rt:.6g}", "        type            zeroGradient;", io(rt))
        field(case, "gammaInt", "[0 0 0 0 0 0 0]", "uniform 1", "        type            zeroGradient;", io(1.0))


def schemes(case, transient):
    ddt = "backward" if transient else "steadyState"
    write(case / "system" / "fvSchemes", "dictionary", "fvSchemes", f"""ddtSchemes {{ default {ddt}; }}
gradSchemes
{{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(omega)     $limited;
}}
divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind limited;
    turbulence      bounded Gauss linearUpwind limited;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div(phi,gammaInt) $turbulence;
    div(phi,ReThetat) $turbulence;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited corrected 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default limited corrected 0.5; }}
wallDist {{ method meshWave; }}
""")


def solution(case, transient, moving=False):
    common = """solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "pcorr.*"
    {
        $p;
        tolerance       1e-5;
        relTol          0;
    }
    "(U|k|omega|gammaInt|ReThetat)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.1;
    }
    "(U|k|omega|gammaInt|ReThetat)Final"
    {
        $U;
        relTol          0;
    }
}
"""
    if moving:
        # rigid motion keeps every cell's volume: no flux correction after the move;
        # the final pressure solve to 1e-6 (1e-7 took 50-380 GAMG sweeps per step)
        common = common.replace("relTol          0;\n    }\n    \"pcorr.*\"", "relTol          0;\n        tolerance       1e-6;\n    }\n    \"pcorr.*\"")
    if transient:
        body = common + f"""PIMPLE
{{
    momentumPredictor yes;
    nOuterCorrectors 2;
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    correctPhi      no;
    moveMeshOuterCorrectors no;
}}
relaxationFactors {{ equations {{ ".*" 1; }} }}
"""
    else:
        body = common + """SIMPLE
{
    consistent      yes;
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p               1e-6;
        U               1e-7;
        "(k|omega|gammaInt|ReThetat)" 1e-6;
    }
}
relaxationFactors
{
    fields { p 1; }
    equations { U 0.9; ".*" 0.7; }
}
"""
    write(case / "system" / "fvSolution", "dictionary", "fvSolution", body)


def control(case, app, end, dt, write_every, alpha_deg, transient, max_co=2.0, forces_raw=False):
    a = math.radians(alpha_deg)
    adj = f"adjustTimeStep  yes;\nmaxCo           {max_co};\nmaxDeltaT       0.02;\n" if transient else ""
    wc = "adjustable" if transient else "timeStep"
    raw = f"""    forces1
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (wall);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        log             false;
    }}
""" if forces_raw else ""
    write(case / "system" / "controlDict", "dictionary", "controlDict", f"""application     {app};
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          {dt};
writeControl    {wc};
writeInterval   {write_every};
purgeWrite      2;
writeFormat     binary;
writePrecision  10;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
{adj}
functions
{{
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (wall);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({-math.sin(a):.10g} {math.cos(a):.10g} 0);
        dragDir         ({math.cos(a):.10g} {math.sin(a):.10g} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 -1);
        magUInf         1;
        lRef            1;
        Aref            {DZ};
        log             false;
    }}
{raw}    yPlus1
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
    }}
}}
""")


def static(case: Path, msh: str, closure: str, alpha: float, mode: str = "steady", nu: float = 1e-5,
           iters: int = 8000, t_end: float = 40.0):
    """A stage-A case.  steady: simpleFoam, `iters` iterations at most.
    transient: pimpleFoam to t_end convective times (initial fields copied from
    a steady run if the caller put them in 0/)."""
    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    k_inf, om_inf = constant(case, nu, closure)
    a = math.radians(alpha)
    tr = mode == "transient"
    schemes(case, tr)
    solution(case, tr)
    if tr:
        control(case, "pimpleFoam", t_end, 1e-4, 5.0, alpha, True, max_co=MAX_CO, forces_raw=True)
    else:
        control(case, "simpleFoam", iters, 1, 500, alpha, False, forces_raw=True)
    summary = mesh(case, msh) if not (case / "constant" / "polyMesh" / "faces").exists() else None
    if not (case / "0" / "U").exists():
        zero(case, closure, math.cos(a), math.sin(a), k_inf, om_inf)
    return summary


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "static":
        out = static(Path(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]),
                     sys.argv[6] if len(sys.argv) > 6 else "steady")
        print("\n".join(out or []))
    elif kind == "mesh":
        print("\n".join(mesh(Path(sys.argv[2]), sys.argv[3])))


def moving(case: Path, msh: str, closure: str, nu: float, U: float, *, h0: float, omega: float,
           theta0_deg: float, amp_deg: float, t_end: float, pivot: float = 0.25, max_co: float = 4.0,
           max_dt: float = 0.02, pts_per_period: int = 720, n_proc: int = 1):
    """A stage-B case: the whole mesh moves rigidly, plunge y = h0 sin(omega t)
    and pitch theta = theta0 + amp cos(omega t) nose up about (pivot, 0); the
    stream is U along +x in the inertial frame (U = 0: still fluid).  The mesh
    is first turned to theta(0) (transformPoints) so the motion starts without
    a jump; the table then carries theta(t) - theta(0).  Unit chord."""
    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    k_inf, om_inf = constant(case, nu, closure, moving=dict(pivot=pivot))
    schemes(case, True)
    # rigid motion leaves every wall distance unchanged: do not recompute it each step
    fs = case / "system" / "fvSchemes"
    fs.write_text(fs.read_text().replace("wallDist { method meshWave; }",
                                         "wallDist { method meshWave; updateInterval 1000000000; }"))
    solution(case, True, moving=True)
    T = 2 * math.pi / omega
    control(case, "pimpleFoam", t_end, 1e-4, T / 4, 0.0, True, max_co=max_co, forces_raw=True)
    # cap the step so a period has at least 400 steps
    cd = case / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("maxDeltaT       0.02;", f"maxDeltaT       {min(max_dt, T / 400):.6g};"))
    summary = mesh(case, msh) if not (case / "constant" / "polyMesh" / "faces").exists() else None
    th0 = theta0_deg + amp_deg                        # theta at t = 0
    env = dict(os.environ, PWD=str(case.resolve()))
    # v2412's transformPoints turns about the origin: move the pivot there and back
    with open(case / "log.transformPoints", "w") as fh:
        for args in (["-translate", f"({-pivot} 0 0)"], ["-rotate-angle", f"((0 0 1) {-th0:.10g})"],
                     ["-translate", f"({pivot} 0 0)"]):
            subprocess.run(["transformPoints"] + args, cwd=case, stdout=fh, stderr=subprocess.STDOUT,
                           check=True, timeout=600, env=env)
    n = int(math.ceil(t_end / T * pts_per_period)) + 2
    lines = []
    for i in range(n):
        t = i * T / pts_per_period
        y = h0 * math.sin(omega * t)
        th = theta0_deg + amp_deg * math.cos(omega * t)
        lines.append(f"({t:.10g} ((0 {y:.10g} 0) (0 0 {-(th - th0):.10g})))")
    (case / "constant" / "motion.dat").write_text(f"{n}\n(\n" + "\n".join(lines) + "\n)\n")
    zero(case, closure, U, 0.0, k_inf, om_inf, moving=True)
    if U == 0.0:
        # still fluid: freestreamVelocity divides by |U|; use a far field that lets fluid in or out at rest
        field(case, "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
              "        type            movingWallVelocity;\n        value           uniform (0 0 0);",
              "        type            pressureInletOutletVelocity;\n        value           uniform (0 0 0);",
              cls="volVectorField")
        field(case, "p", "[0 2 -2 0 0 0 0]", "uniform 0", "        type            zeroGradient;",
              "        type            totalPressure;\n        p0              uniform 0;\n        value           uniform 0;")
    if n_proc > 1:
        write(case / "system" / "decomposeParDict", "dictionary", "decomposeParDict",
              f"numberOfSubdomains {n_proc};\nmethod scotch;\n")
    return summary
