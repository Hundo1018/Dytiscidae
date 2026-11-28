"""`fluid.py` and `mojo/src/fluid_gpu.mojo` are the same physics twice.

`batchroll.evaluate_tier1_batch` scores with the Mojo kernels and
`evaluate.evaluate_tier1` scores with the numpy solver.  CLAUDE.md already
records that the two are not interchangeable and have differed by 60x.  What it
does not record is that **nothing asserted they carry the same constants.**

`tests/test_search.py` compares the two numerically, which is the real check --
and it needs the built extension, so it cannot run without a GPU toolchain.  On
a machine that has neither, an edit to `lift_coefficient` that forgets the Mojo
half is invisible.  These checks need neither: they read both files as text.

They catch one class of drift, not all of it: a constant changed on one side
and not the other, and a provenance comment that no longer points at anything.
Agreement of the *formulas* across a whole evaluation is `test_search.py`'s
job and stays there.

The table law (`FluidSolver.stall_law == "table"`, 2026-11-17) is also checked
numerically here, against the built kernel at run time: its coefficients over
the angles, Reynolds numbers, cambers, lags and vortex strengths a strip meets,
and one fused pipeline step against `FluidSolver.apply`.  Those checks print
`[skip]` without the extension.

The provenance comments were line numbers -- `fluid.py:256-289` -- and six of
the eight had drifted off their target by the time this was written.
`fluid.py:292-305` claimed to be `drag_coefficient` and pointed into the middle
of `lift_coefficient`'s docstring; `fluid.py:428-432` claimed to be the
kinematics block and pointed at two assignments in `__init__`.  A line number
cannot be kept honest by anything, so they are names and block markers now, and
the first check below is that none has come back.

Run:  PYTHONPATH=. python tests/test_gpu_mirror.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY_PATH = ROOT / "dytiscidae/physics/fluid.py"
MJ_PATH = ROOT / "mojo/src/fluid_gpu.mojo"

#: Mojo symbol -> the Python function it mirrors.  The constants of each pair
#: are compared; the pairing itself is the thing a human has to keep right.
FUNCTION_MIRRORS = {
    "_skin_friction_cd": "skin_friction_cd",
    "_lift_coefficient": "lift_coefficient",
    "_drag_coefficient": "drag_coefficient",
    "_stall_weight": "_stall",
    "_smoothstep": "_smoothstep",
    "_thin_plate": "_thin_plate",
    "_table_coefficients": "table_coefficients",
}

#: Python-only numbers a pair may carry, and why.  Only this direction: a
#: number the Mojo has and the Python lacks is always a failure.
PYTHON_ONLY = {
    # law "plate" (repair R1) is numpy only (`batchroll._law_code` refuses
    # it): its 45-degree plateau limit and its 4-degree separation width
    "drag_coefficient": {0.25},
    # ... and the 1e12 is `_stall`'s table branch, whose weight the kernel
    # takes from `_table_coefficients` directly
    "_stall": {4.0, 1e12},
    # SEPARATION_COMPLETE (np.radians(6.0)) reaches the kernel as a value in
    # the table block `batchroll._table_block` packs, not as a typed number
    "_thin_plate": {6.0},
}

#: Block markers a Mojo docstring may name.  Each must exist in `fluid.py`.
BLOCK_MARKERS = (
    "--- kinematics",
    "--- strip theory",
    "--- bluff-body drag",
    "--- added mass",
)

FAILURES: list[str] = []
SKIPS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def strip_text(text: str) -> str:
    """Drop docstrings and line comments, so prose numbers are not constants."""
    text = re.sub(r'""".*?"""', "", text, flags=re.S)
    return re.sub(r"#.*", "", text)


def mojo_body(src: str, name: str) -> str:
    """The `def name(...)` block, signature included, by indentation."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if not re.match(rf"\s*def {re.escape(name)}\b", line):
            continue
        indent = len(line) - len(line.lstrip())
        body, j = [line], i
        depth = line.count("(") - line.count(")")
        while depth > 0 or not lines[j].rstrip().endswith(":"):
            j += 1
            body.append(lines[j])
            depth += lines[j].count("(") - lines[j].count(")")
        for nxt in lines[j + 1:]:
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                break
            body.append(nxt)
        return "\n".join(body)
    raise KeyError(f"{name} is not in {MJ_PATH.name}")


def mojo_numbers(src: str, name: str) -> set[float]:
    body = strip_text(mojo_body(src, name))
    return {float(m) for m in re.findall(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])", body)}


def _consts(node) -> set[float]:
    return {float(n.value) for n in ast.walk(node)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)}


def python_numbers(src: str, name: str) -> set[float]:
    """Literals in the function, plus those of the module constants it reads.

    Following the names matters: lifting `6.0` out of the body into a
    documented `SEPARATION_COMPLETE = np.radians(16.0)` is exactly the sort of
    tidy-up this guard has to see through, because the Mojo has no module
    constant to lift it into and writes the number inline.
    """
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        raise KeyError(f"{name} is not in {PY_PATH.name}")

    module_consts: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_consts[t.id] = node.value

    out = _consts(fn)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in module_consts:
            out |= _consts(module_consts[n.id])
    return out


# --------------------------------------------------------------------------


def test_no_line_number_references_came_back() -> None:
    print("\ngpu mirror: how the Mojo file points at the Python")
    mj = MJ_PATH.read_text()
    stale = re.findall(r"fluid\.py:\d+", mj)
    check("no `fluid.py:<line>` reference remains", not stale,
          f"found {stale}" if stale else "a line number cannot be kept honest")

    named = re.findall(r"Mirrors `fluid\.([a-z_]+)`", mj)
    check("every function reference names a function that exists",
          all(f"def {n}(" in PY_PATH.read_text() for n in named),
          f"{len(named)} references: {named}")

    marked = set(re.findall(r"marked `(--- [a-z\- ]+?)` in", mj))
    check("every block reference names a marker in fluid.py",
          all(m in PY_PATH.read_text() for m in marked),
          f"{len(marked)} markers: {sorted(marked)}")


def test_each_marker_is_unique() -> None:
    """A marker that appears twice points at two places, so it points at none."""
    print("\ngpu mirror: the block markers")
    py = PY_PATH.read_text()
    for marker in BLOCK_MARKERS:
        n = py.count(marker)
        check(f"`{marker}` appears exactly once in fluid.py", n == 1,
              f"{n} occurrences")


def test_the_pairs_carry_the_same_constants() -> None:
    """The drift that a machine without the extension cannot otherwise see."""
    print("\ngpu mirror: the constants of each mirrored function")
    py_src, mj_src = PY_PATH.read_text(), MJ_PATH.read_text()
    for mojo_name, py_name in FUNCTION_MIRRORS.items():
        a = python_numbers(py_src, py_name)
        b = mojo_numbers(mj_src, mojo_name)
        only_py = sorted(a - b - PYTHON_ONLY.get(py_name, set()))
        only_mj = sorted(b - a)
        # Counts legitimately differ -- `1.0 - w` in one language may be
        # spelled with an extra literal in the other -- so this is set
        # equality, which is what catches a value changed on one side.
        check(f"{py_name} and {mojo_name} use the same numbers",
              not only_py and not only_mj,
              f"{len(a & b)} shared" if not (only_py or only_mj)
              else f"python-only {only_py}, mojo-only {only_mj}")


def test_the_mirrored_set_is_the_whole_set() -> None:
    """A Mojo function that mirrors a Python one but is not in the table."""
    print("\ngpu mirror: coverage of the table")
    mj = MJ_PATH.read_text()
    named = set(re.findall(r"Mirrors `fluid\.([a-z_]+)`", mj))
    listed = set(FUNCTION_MIRRORS.values())
    check("every 'Mirrors fluid.<f>' docstring is in FUNCTION_MIRRORS",
          named <= listed, f"unlisted: {sorted(named - listed)}"
          if named - listed else f"{sorted(named)}")
    check("and every pair in FUNCTION_MIRRORS is findable in both files",
          all(f"def {m}(" in mj and f"def {p}(" in PY_PATH.read_text()
              for m, p in FUNCTION_MIRRORS.items()),
          f"{len(FUNCTION_MIRRORS)} pairs")


def skip(name: str, why: str) -> None:
    print(f"  [skip] {name}  -- {why}")
    SKIPS.append(name)


def _fluid_gpu():
    """The built `fluid_gpu` extension, or None."""
    from dytiscidae.envs import batchroll
    if batchroll._BUILD_DIR not in sys.path:
        sys.path.insert(0, batchroll._BUILD_DIR)
    try:
        import fluid_gpu
    except Exception:                                             # noqa: BLE001
        return None
    return fluid_gpu if hasattr(fluid_gpu, "coefficients_law") else None


def _rel(a, b) -> float:
    """The existing GPU checks' measure: relative, with a floor of 1."""
    import numpy as np
    return float((np.abs(a - b) / np.maximum(np.abs(a), 1.0)).max())


def test_the_kernel_reproduces_table_coefficients() -> None:
    """`_table_coefficients` against `fluid.table_coefficients`, both run now.

    Every strip a wing can present: incidence over the whole folded range
    with both flow directions, Re 1e3..1e6 (the tables span 6e4..2e5, so
    both holds), the cambers the tables blend between and past (+-), a lag
    between the incidence and the Wagner angle, and the vortex strength from
    none through clipped.  `coefficients_law` runs `coeff_at`, the stage the
    fused pipeline calls."""
    print("\ngpu mirror: the table law, kernel against numpy")
    import numpy as np
    from dytiscidae.physics import fluid
    from dytiscidae.physics import thin_plate_tables as T
    from dytiscidae.envs import batchroll
    fg = _fluid_gpu()
    if fg is None:
        skip("table coefficients agree", "fluid_gpu with coefficients_law is not built")
        return
    # an extension can import and still not run (batchroll.usable's NVML case)
    if batchroll.usable()[0] is not True:
        skip("table coefficients agree", "the GPU kernel is not usable here")
        return
    from dytiscidae.envs.batchroll import _table_block
    rng = np.random.default_rng(20261117)
    cambers = np.array([-0.09, -0.03, 0.0, 0.02, 0.0578, 0.09])
    # a grid on every camber, then random strips; the grid sits on the
    # table's own Re nodes, its measured-angle ends and the folded range's ends
    a_grid = np.radians(np.linspace(-90.0, 90.0, 721))
    re_grid = np.concatenate([[1e3, 1e4, 1e6], T.RE_NODES, [7.3e4, 1.5e5]])
    A, R, Cm = (x.ravel() for x in np.meshgrid(a_grid, re_grid, cambers, indexing="ij"))
    m = 200000
    alpha = np.concatenate([A, rng.uniform(-np.pi / 2, np.pi / 2, m),
                            T.LO.ravel(), T.TOP.ravel(), -T.LO.ravel(), -T.TOP.ravel()])
    k = len(alpha) - len(A) - m
    re = np.concatenate([R, 10 ** rng.uniform(3.0, 6.0, m), np.tile(T.RE_NODES, 8)[:k]])
    camber = np.concatenate([Cm, rng.choice(cambers, m), rng.choice(cambers, k)])
    n = len(alpha)
    rev = (rng.uniform(size=n) < 0.5).astype(float)
    # the Wagner angle lags the incidence: from equal (a quarter of the
    # strips) to a quarter of it and a few degrees off
    lag = rng.uniform(0.25, 1.0, n) * alpha + rng.normal(0.0, 0.05, n)
    alpha_e = np.where(rng.uniform(size=n) < 0.25, alpha, lag)
    lev = rng.choice([0.0, 1.0, 1.3, -0.2], n)
    lev = np.where(rng.uniform(size=n) < 0.5, rng.uniform(0.0, 1.0, n), lev)
    ar = np.where(rng.uniform(size=n) < 0.3, fluid.INFLOW_AR, rng.uniform(0.2, 12.0, n))
    is_wing = np.ones(n, np.int32)
    tab = _table_block()
    C = np.ascontiguousarray
    alpha, alpha_e, re, ar, lev, camber, rev = (C(x, dtype=float) for x in
                                                (alpha, alpha_e, re, ar, lev, camber, rev))

    def run(law):
        cl, cd = np.zeros(n), np.zeros(n)
        got = fg.coefficients_law(np.array(
            [x.ctypes.data for x in (alpha, alpha_e, re, ar, lev, camber, rev, is_wing, cl, cd, tab)]
            + [tab.size, law, n], dtype=np.int64))
        return got, cl, cd

    # what `FluidSolver.apply` passes under "table": the unshifted angle and
    # the camber as the flow meets it
    arc = np.where(rev != 0.0, -camber, camber)
    cl_ref, cd_ref, _ = fluid.table_coefficients(alpha, re, ar, lev, alpha_e, arc)
    got, cl, cd = run(1)
    e_cl, e_cd = _rel(cl_ref, cl), _rel(cd_ref, cd)
    check("the table law's lift and drag agree with numpy",
          got == n and e_cl < 1e-12 and e_cd < 1e-12,
          f"{n} strips; cl {e_cl:.2e}, cd {e_cd:.2e} (bar 1e-12, test_coeff_gpu's)")
    w = fluid.table_coefficients(alpha, re, ar, lev, alpha_e, arc)[2]
    check("and the sample spans the attached, blended and separated strips",
          (w < 1e-9).sum() > 1000 and ((w > 0.01) & (w < 0.99)).sum() > 1000
          and (w > 1.0 - 1e-9).sum() > 1000,
          f"w=0: {(w < 1e-9).sum()}, between: {((w > 0.01) & (w < 0.99)).sum()}, "
          f"w=1: {(w > 1.0 - 1e-9).sum()}")
    # The same entry under law 0 is the legacy pair, lag included (the
    # standalone `coefficients` has no lag), and reads no camber.
    _, cl0, cd0 = run(0)
    cl0_ref = fluid.lift_coefficient(alpha, re, ar, lev, alpha_e)
    cd0_ref = fluid.drag_coefficient(alpha, re, ar, cl0_ref, lev)
    e0 = max(_rel(cl0_ref, cl0), _rel(cd0_ref, cd0))
    check("law 0 through the same stage is the legacy law", e0 < 1e-12, f"{e0:.2e}")
    check("and the two laws differ (the switch reaches the stage)",
          float(np.abs(cl0 - cl).max()) > 0.1, f"max |cl difference| {np.abs(cl0 - cl).max():.3f}")


def test_one_pipeline_step_under_the_table_law() -> None:
    """The fused pipeline (`full_pipeline`, through `batchroll.BatchedFluid`)
    against `FluidSolver.apply`, both with ``stall_law = "table"``: the strip
    stage's missing camber shift, the camber and reversal reaching the
    coefficients, and the table block's upload.  `mojo/tests/test_full_gpu.py`'s
    fixture and bar; the legacy law is run beside it on the same poses."""
    print("\ngpu mirror: one fused step under each law, against FluidSolver")
    from dytiscidae.envs import batchroll
    if not batchroll.AVAILABLE or batchroll.usable()[0] is not True:
        skip("fused step under the table law", "the GPU pipeline is not usable here")
        return
    import zlib
    import mujoco
    import numpy as np
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    def pose(env, seed, law):
        env.solver.stall_law = law
        rng = np.random.default_rng(seed)
        env.reset(Domain.WATER, randomise=False)
        nq = env.model.nq
        if nq >= 7:
            q = rng.normal(size=4)
            env.data.qpos[3:7] = q / np.linalg.norm(q)
            env.data.qpos[0:3] = rng.normal(scale=1.5, size=3)
            env.data.qpos[2] = rng.uniform(-3.0, 1.0)
        if nq > 7:
            env.data.qpos[7:] = rng.uniform(-0.7, 0.7, nq - 7)
        env.data.qvel[:] = rng.normal(scale=2.0, size=env.model.nv)
        mujoco.mj_forward(env.model, env.data)
        return env

    seeds = {k: zlib.crc32(k.encode()) % 9973 for k in BODY_PLANS}
    worst, first = {}, {}
    for law in ("legacy", "table"):
        batched = [pose(TriphibianEnv(build(p())), seeds[k], law) for k, p in BODY_PLANS.items()]
        single = [pose(TriphibianEnv(build(p())), seeds[k], law) for k, p in BODY_PLANS.items()]
        b = batchroll.BatchedFluid(batched)
        wf = wm = 0.0
        dt = batched[0].model.opt.timestep
        # three steps, so the unsteady history (lag, vortex travel) is live
        for step in range(3):
            t = step * dt
            b.apply(t)
            for eb, e in zip(batched, single):
                e.data.xfrc_applied[:] = 0.0
                e.solver.apply(e.data, t)
                if e.jets.n:
                    e.jets.apply(e.model, e.data, e.medium, t, e.timestep)
                if e.rotors.n:
                    e.rotors.apply(e.model, e.data, e.medium, t)
                ref = e.data.xfrc_applied
                wf = max(wf, float(np.abs(ref - eb.data.xfrc_applied).max())
                         / max(float(np.abs(ref).max()), 1e-12))
                sm = max(float(np.abs(e.model.body_mass - e.solver._dry_mass).max()), 1e-12)
                wm = max(wm, float(np.abs(e.model.body_mass - eb.model.body_mass).max()) / sm)
            if step == 0:
                first[law] = np.concatenate([e.data.xfrc_applied.ravel() for e in single])
                cam = np.concatenate([e.solver.panels.camber for e in single])
            for e in batched + single:
                mujoco.mj_step(e.model, e.data)
        worst[law] = (wf, wm)
    for law, (wf, wm) in worst.items():
        check(f"{law}: three fused steps agree with FluidSolver.apply",
              wf < 1e-9 and wm < 1e-12,
              f"xfrc {wf:.2e}, added mass {wm:.2e} (test_full_gpu's bars 1e-9, 1e-12)")

    gap = float(np.abs(first["legacy"] - first["table"]).max())
    check("and the two laws load the same poses differently (the law reached the solver)",
          gap > 1e-3 and np.abs(cam).max() > 0.0,
          f"max |xfrc difference| {gap:.3g} N; cambered strips {int((cam != 0).sum())}")

    def refused(law_of):
        envs = [TriphibianEnv(build(p())) for p in list(BODY_PLANS.values())[:2]]
        for i, e in enumerate(envs):
            e.solver.stall_law = law_of(i)
        try:
            batchroll.BatchedFluid(envs)
        except ValueError as exc:
            return str(exc)
        return ""
    why = refused(lambda i: "plate")
    check("a batch under the numpy-only plate law is refused", "plate" in why, why[:80])
    why = refused(lambda i: ("legacy", "table")[i])
    check("and so is a batch mixing two laws", "mixes" in why, why[:80])


def main() -> int:
    test_no_line_number_references_came_back()
    test_each_marker_is_unique()
    test_the_pairs_carry_the_same_constants()
    test_the_mirrored_set_is_the_whole_set()
    test_the_kernel_reproduces_table_coefficients()
    test_one_pipeline_step_under_the_table_law()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nThe formulas themselves are tests/test_search.py's job and "
              "need the built extension; these checks do not.")
        return 1
    print("gpu mirror checks passed"
          + (f", {len(SKIPS)} skipped" if SKIPS else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
