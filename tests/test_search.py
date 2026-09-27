"""Verification for the generative and search machinery.

The physics tests pin down whether the world is right.  These pin down whether
the search over it behaves, which is a different and easier thing to get subtly
wrong: an archive that silently rejects everything, a bandit that never learns,
or a mobility identification that returns noise all *look* like a search that is
merely having a hard time.

The most important test here is ``test_mobility_recovers_known_basis``: it runs
the axis identification on a synthetic system whose true Jacobian is known, and
checks that the discovered axes match it.  Without that, the claim that control
axes are "discovered" is unfalsifiable.

Run:  python tests/test_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.control.cpg import CPG, CPGParams, Policy, identify_mobility  # noqa: E402
from dytiscidae.core.cppn import CPPN, sample_surface, new_surface_cppn  # noqa: E402
from dytiscidae.core.genome import (  # noqa: E402
    MUTATION_OPERATORS,
    mutate,
    random_genome,
)
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.core.reference import reference_genome  # noqa: E402
from dytiscidae.evolution.archive import Archive  # noqa: E402
from dytiscidae.evolution.cmaes import CMAES  # noqa: E402
from dytiscidae.evolution.curator import Curator, OperatorBandit  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []

#: Errors that mean "this machine lacks the hardware", not "the code is wrong".
#: Explicit rather than a bare ``except``: a new kind of environmental breakage
#: should be reported as a failure, not quietly absorbed into the skip count.
BLOCKED_MARKERS = (
    "GPU fluid extension not importable",
    "No module named 'full_pipeline'",
    # The extension imports and every construction fails -- a driver whose
    # kernel module does not match its userspace library.  Measured 2026-09-19:
    # NVRM 580.173.02 loaded under libnvidia-ml.so.580.178.04 raised
    # "Failed to initialize NVML: 18" from FullPipeline() while the import
    # succeeded, so this arrived as a traceback where the contract is a skip.
    "Failed to initialize NVML",
    "cannot be constructed",
)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    """A check that did not run.  Not ``check(name, True, "skipped: ...")``.

    That form printed ``[ok  ]`` and counted as a pass, so a machine without
    torch produced the same summary as one where the checkpoint and the shared
    policy had actually been tested.
    """
    print(f"  [skip] {name}  -- {reason}")
    SKIPPED.append(name)


def needs_batched_evaluator(fn_name: str) -> bool:
    """True when the caller should return: the batched evaluator is unavailable.

    Six functions here drive a real ``run_search``, which needs the Mojo GPU
    fluid extension.  Three of them *raise* on a machine without it and
    ``run_all`` skips them; the other three do not, because ``run_search``
    catches a failed generation, records it in telemetry and carries on with
    nothing evaluated.  Those three then failed their own assertions with
    "0 promotions", "0 tensors compared" and an identification that found six
    modes instead of four -- five red checks that say nothing about the code and
    everything about the machine.

    Declaring the requirement at the top of the function is the difference
    between a suite that reports what it could not run and one that reports a
    defect it did not find.
    """
    from dytiscidae.envs import batchroll

    # ``usable()`` rather than ``AVAILABLE``: the latter is decided at import
    # time, and a driver mismatch lets the import succeed and the construction
    # fail.  Measured 2026-09-19, that difference turned the documented three
    # ``[skip]`` into seventeen ``[fail]`` -- a suite reporting defects it had
    # not found, which is the one thing this file exists to prevent.
    ok, why = batchroll.usable()
    if ok:
        return False
    skip(fn_name, f"{why} — needs `cd mojo && pixi run build-all`, "
                  f"and a driver whose kernel module matches its userspace")
    return True


def run_all(functions) -> None:
    """Run every test, and never let one of them stop the rest.

    ``main()`` used to be a flat list of calls, so the first function that
    raised ended the file and everything after it silently never ran.  On a
    machine without the Mojo GPU fluid extension that is what happens, and the
    console shows a traceback rather than a list of what was lost:
    ``tools/suite_probe.py`` is what turns that into a count.

    A blocked function is reported as ``[skip]`` and counted separately, because
    "did not run" and "passed" must not share a line in the summary.
    """
    import traceback

    for fn in functions:
        mark = len(FAILURES)
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if any(m in text for m in BLOCKED_MARKERS):
                # The function stopped partway.  Whatever it printed before it
                # stopped is not evidence about the code -- several of these
                # check a run that never got to evaluate anything -- so its
                # checks are withdrawn rather than counted as failures.  They
                # are not counted as passes either: the function is skipped.
                partial = len(FAILURES) - mark
                del FAILURES[mark:]
                SKIPPED.append(fn.__name__)
                note = f", {partial} partial checks withdrawn" if partial else ""
                print(f"  [skip] {fn.__name__}{note}  "
                      f"-- {text.splitlines()[0][:100]}")
            else:
                FAILURES.append(f"{fn.__name__} raised")
                print(f"  [FAIL] {fn.__name__} raised")
                traceback.print_exc()


def report(label: str) -> int:
    """The summary.  A skip is never folded into the success line."""
    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED — needs the Mojo GPU fluid extension "
              f"(`cd mojo && pixi run build-all`), not a defect: "
              f"{', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    # Unqualified only when nothing was skipped, so the string a reader greps
    # for cannot appear on a run that did not run everything.
    print(label if not SKIPPED else f"{label}, {len(SKIPPED)} skipped")
    return 0


# --------------------------------------------------------------------------


def test_mobility_recovers_known_basis() -> None:
    """Identification must recover the true axes of a synthetic system.

    The fake body responds to a 6-parameter command through a known Jacobian.
    Two directions have real authority, one is deliberately far too weak to be
    useful, and three do nothing at all.

    Note the rotational scaling inside ``identify_mobility``: angular twist is
    weighted at 0.3 relative to linear, so that the SVD is not dominated by
    whichever has the larger raw units.  A roll gain of 0.12 rad/s therefore
    presents as 3.6% of the dominant mode -- correctly *below* any usable
    threshold.  Both halves of that are asserted here, because "reports an axis
    the machine does not really have" and "misses one it does" are opposite
    failures and a rank check alone would only catch one of them.
    """
    print("\ncontrol: mobility identification recovers a known basis")
    rng = np.random.default_rng(0)
    n_params = 6

    def make_J(roll_gain: float) -> np.ndarray:
        J = np.zeros((n_params, 6))
        J[0, 0] = 1.0                 # surge
        J[1, 2] = 0.6; J[1, 4] = 0.6  # coupled heave + pitch
        J[2, 3] = roll_gain           # roll
        return J

    drift = np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0])  # sinking, command-independent
    J = make_J(0.12)

    def step_fn(delta):
        return drift + delta @ J + rng.normal(0, 1e-4, 6)

    basis = identify_mobility(step_fn, lambda: None, n_params, n_probes=14,
                              rng=np.random.default_rng(1))

    check("the two strong axes are found", basis.control_rank == 2,
          f"control_rank={basis.control_rank} "
          f"sigmas={np.round(basis.authority, 3)}")
    check("a 3.6%-authority axis is not claimed as usable",
          basis.authority[2] < 0.08 * basis.authority[0],
          f"sigma2/sigma0={basis.authority[2]/basis.authority[0]:.3f}")
    # The floor here is the measurement noise injected above (1e-4), not zero:
    # the identification correctly reports the noise it was given, so the
    # assertion is that a null direction sits at that floor and is negligible
    # against the dominant mode, not that it is exactly zero.
    check("the three null parameters produce nothing above the noise floor",
          basis.authority[3] < 0.01 * basis.authority[0],
          f"sigma3={basis.authority[3]:.2e} vs sigma0={basis.authority[0]:.3f}")

    # Same system with a roll gain that genuinely is usable must report rank 3.
    J_strong = make_J(1.2)

    def step_strong(delta):
        return drift + delta @ J_strong + rng.normal(0, 1e-4, 6)

    strong = identify_mobility(step_strong, lambda: None, n_params, n_probes=14,
                               rng=np.random.default_rng(1))
    check("raising that axis's gain makes it count", strong.control_rank == 3,
          f"control_rank={strong.control_rank} "
          f"sigmas={np.round(strong.authority, 3)}")

    check("constant drift is cancelled, not reported as an axis",
          not np.any(np.abs(basis.effects[:, 2]) > 0.97),
          "no mode is pure heave")

    # The leading discovered parameter direction should live in the span of
    # params 0 and 1, i.e. carry almost no weight on the null params 3-5.
    leak = float(np.linalg.norm(basis.modes[0, 3:]))
    check("leading mode ignores the null parameters", leak < 0.2, f"leakage={leak:.3f}")

    # Discovered effects must be spanned by the true row space of J.
    true_space = J[:3]
    proj = basis.effects[0] @ np.linalg.pinv(true_space) @ true_space
    align = float(abs(proj @ basis.effects[0]) / (np.linalg.norm(basis.effects[0]) ** 2))
    check("leading effect lies in the true response space", align > 0.9,
          f"alignment={align:.3f}")
    check("axes get human-readable names", len(basis.describe()) > 0,
          basis.describe()[0])


def test_archive_placement_and_improvement() -> None:
    print("\narchive: placement, improvement, projection")
    axes = [("a", 0.0, 1.0, 4), ("b", 0.0, 1.0, 4)]
    a = Archive(axes)
    check("capacity is the product of bins", a.capacity == 16, f"{a.capacity}")

    check("first insert is new", a.add("g1", 0.5, np.array([0.1, 0.1])) == "new")
    check("worse insert is rejected", a.add("g2", 0.3, np.array([0.12, 0.12])) == "rejected")
    check("better insert improves", a.add("g3", 0.9, np.array([0.12, 0.12])) == "improved")
    check("cell count is 1 after three inserts into one cell", len(a.cells) == 1)
    check("elite is the best genome", a.best.genome == "g3", str(a.best.genome))
    check("improvement counter advanced", a.best.improvements == 1)

    a.add("g4", 0.7, np.array([0.9, 0.9]))
    check("distant descriptor makes a new cell", len(a.cells) == 2)
    check("qd score sums elites", abs(a.qd_score - 1.6) < 1e-9, f"{a.qd_score}")
    check("coverage is filled/capacity", abs(a.coverage - 2 / 16) < 1e-9)

    # Out-of-range descriptors must clamp, not vanish.
    a.add("g5", 0.4, np.array([5.0, -3.0]))
    check("out-of-range descriptor clamps into an edge cell", len(a.cells) == 3)

    best, count = a.project(0, 1)
    check("projection shape matches bins", best.shape == (4, 4), str(best.shape))
    check("projection counts every elite", int(count.sum()) == 3, str(int(count.sum())))

    nd = a.neighbour_density((0, 0), radius=1)
    check("neighbour density counts the local cluster", nd >= 1, f"{nd}")


def test_bandit_learns_which_operator_pays() -> None:
    """The bandit must concentrate on a genuinely better arm."""
    print("\ncurator: operator bandit")
    names = ["good", "bad", "neutral"]
    b = OperatorBandit(names)
    rng = np.random.default_rng(0)
    picks = {n: 0 for n in names}
    payoff = {"good": 1.0, "bad": -0.2, "neutral": 0.0}
    for _ in range(300):
        n = b.select(rng)
        picks[n] += 1
        b.update([n], payoff[n] + float(rng.normal(0, 0.05)))
    check("the paying operator is sampled most", picks["good"] > picks["bad"],
          f"good={picks['good']} bad={picks['bad']} neutral={picks['neutral']}")
    top = b.report()[0]
    check("report ranks the paying operator first", top["operator"] == "good",
          f"{top}")
    # The `check(..., True)` that used to sit here said "covered by
    # construction; asserted below" in its own comment, and printed [ok  ] for
    # something it did not check.  The assertion below is the whole of it, now
    # naming both operators so a one-sided split fails.
    b2 = OperatorBandit(["x", "y"])
    b2.update(["x", "y"], 1.0)
    halves = [b2.stats[k].lifetime_mean for k in ("x", "y")]
    check("credit is split evenly across co-applied operators",
          all(abs(h - 0.5) < 1e-9 for h in halves),
          f"x={halves[0]:.9f}, y={halves[1]:.9f}")


def test_no_operator_can_go_dormant_under_the_structural_tilt() -> None:
    """A good operator outside the top slots must still be tried.

    Measured over arch39's generations 60-200: structural operators took 68.9%
    of operator slots against a uniform 36%, and ``gait`` -- the operator the
    arm existed to test -- made 0.7% of children after 83 generations with none.
    The argmax is deterministic and the regime multiplies the eight structural
    operators by up to 2.6, so a non-structural operator with a good windowed
    mean and no picks never updates its window.  Rebuilt here: the real 22
    operators, arch39's famine tilt and three slots per child, and statistics
    frozen so only the selection rule is under test.
    """
    print("\ncurator: no operator goes dormant under the structural tilt")
    from dytiscidae.core.genome import STRUCTURAL_OPERATORS

    def shares(bandit, children=700):
        rng = np.random.default_rng(5)
        for name, st in bandit.stats.items():
            st.tries = 30
            st.recent.clear()
            st.recent.extend([0.15 if name in STRUCTURAL_OPERATORS else 0.10] * 30)
        made = {n: 0 for n in bandit.stats}
        for _ in range(children):
            picked = []
            for _slot in range(3):
                picked.append(bandit.select(rng, structural_bias=2.6,
                                            exclude=tuple(picked)))
            for n in set(picked):
                made[n] += 1
        return {n: m / children for n, m in made.items()}

    floor = shares(OperatorBandit())
    frozen = OperatorBandit()
    frozen.epsilon = 0.0
    argmax = shares(frozen)
    dead = sorted(n for n, v in argmax.items() if v == 0.0)
    check("without the floor, operators outside the top slots get nothing",
          len(dead) >= 10, f"{len(dead)} of {len(argmax)} never chosen")
    worst = min(floor, key=floor.get)
    check("with it, every operator makes at least 1% of children",
          floor[worst] >= 0.01, f"lowest: {worst} {floor[worst]:.1%}")
    s_floor = sum(v for n, v in floor.items() if n in STRUCTURAL_OPERATORS)
    s_argmax = sum(v for n, v in argmax.items() if n in STRUCTURAL_OPERATORS)
    check("and the structural tilt still favours structural operators",
          s_floor > 0.5 * s_argmax and s_floor / 8 > (3 - s_floor) / 14,
          f"structural children-share {s_argmax:.2f} -> {s_floor:.2f} (of 3 slots)")

    old = OperatorBandit()
    del old.__dict__["epsilon"]            # as unpickled from an earlier run
    check("a bandit pickled before the floor keeps its old behaviour",
          shares(old) == argmax)


def test_curator_regimes_respond_to_the_run() -> None:
    print("\ncurator: regime detection")
    a = Archive([("m", 0.0, 1.0, 8), ("d", 0.0, 1.0, 8)])
    c = Curator(a, seed=0)

    r = c.update_regime()
    check("an empty archive bootstraps", r.name == "bootstrapping", r.name)
    check("bootstrapping widens the net rather than refining",
          r.feasibility_bias >= 0.7 and r.n_mutations >= 2,
          f"feasibility_bias={r.feasibility_bias} n_mutations={r.n_mutations}")

    rng = np.random.default_rng(0)
    for i in range(40):
        a.add(f"g{i}", float(rng.random()), rng.random(2), {"feasible": True})
    for _ in range(5):
        a.generation += 1
        r = c.update_regime()
    check("a populated, static archive is not bootstrapping", r.name != "bootstrapping", r.name)

    # Force stagnation: no further growth for several generations.
    for _ in range(6):
        a.generation += 1
        r = c.update_regime()
    check("a flat run is called stagnant", r.name == "stagnant", f"{r.name} ({r.note})")
    check("stagnation raises structural pressure", r.structural_bias > 1.5,
          f"bias={r.structural_bias}")
    check("stagnation mutates harder", r.n_mutations >= 3, f"n={r.n_mutations}")

    parent = c.select_parent()
    check("a parent can be selected", parent is not None)

    # Infeasible-dominated archive must raise the feasibility bias.
    a2 = Archive([("m", 0.0, 1.0, 8), ("d", 0.0, 1.0, 8)])
    c2 = Curator(a2, seed=0)
    for i in range(40):
        a2.add(f"h{i}", float(rng.random()), rng.random(2), {"feasible": False})
    r2 = c2.update_regime()
    check("mostly-infeasible archive biases toward feasible parents",
          r2.feasibility_bias > 0.8, f"{r2.feasibility_bias:.2f}")


def test_curator_quarantines_repeat_exploits() -> None:
    print("\ncurator: exploit quarantine")
    a = Archive([("m", 0.0, 1.0, 4), ("d", 0.0, 1.0, 4)])
    c = Curator(a, seed=0)
    bd = np.array([0.5, 0.5])
    a.add("cheat", 2.0, bd, {"feasible": True})
    check("exploiting elite is present", len(a.cells) == 1)
    for i in range(2):
        c.quarantine(bd, "fake thrust")
    check("two strikes taints but does not evict", len(a.cells) == 1,
          f"taint={a.tainted.get(a.cell_of(bd))}")
    c.quarantine(bd, "fake thrust")
    check("three strikes evicts the cell", len(a.cells) == 0)
    check("exploits are recorded", len(c.exploits) == 3, f"{len(c.exploits)}")
    check("tainted cells are excluded from parent selection",
          c.select_parent() is None)


def test_cmaes_optimises_a_known_function() -> None:
    """Sanity: CMA-ES must solve a shifted sphere and an ill-conditioned ellipse."""
    print("\ncmaes: convergence")
    target = np.array([0.7, -1.3, 0.25, 2.0])
    es = CMAES(np.zeros(4), sigma0=1.0, seed=0)
    for _ in range(80):
        pop = es.ask()
        es.tell(pop, -np.sum((pop - target) ** 2, axis=1))
    err = float(np.linalg.norm(es.best_x - target))
    check("solves a shifted sphere", err < 1e-3, f"error={err:.2e}")

    scale = np.array([1.0, 30.0, 900.0])
    es2 = CMAES(np.zeros(3), sigma0=1.0, seed=1)
    for _ in range(220):
        pop = es2.ask()
        es2.tell(pop, -np.sum((pop * scale) ** 2, axis=1))
    err2 = float(np.linalg.norm(es2.best_x * scale))
    check("solves an ill-conditioned ellipsoid", err2 < 1e-2, f"residual={err2:.2e}")
    check("step size shrank on convergence", es.sigma < 0.05, f"sigma={es.sigma:.2e}")


def test_cppn_fields_are_deterministic_and_bounded() -> None:
    print("\ncppn: surface fields")
    rng = np.random.default_rng(0)
    c = new_surface_cppn(rng)
    f1 = sample_surface(c, span=1.0, root_chord=0.3, stations=12)
    f2 = sample_surface(c, span=1.0, root_chord=0.3, stations=12)
    check("same CPPN gives the same field", np.allclose(f1.chord, f2.chord))
    check("chord never degenerates to zero", float(f1.chord.min()) > 0.0,
          f"min chord={f1.chord.min():.4f} m")
    check("chord stays within the mapped range", float(f1.chord.max()) <= 0.3 * 1.001,
          f"max={f1.chord.max():.4f}")
    check("thickness is a sane fraction of chord",
          0.02 < float(f1.thickness.min()) and float(f1.thickness.max()) < 0.2,
          f"{f1.thickness.min():.3f}..{f1.thickness.max():.3f}")
    check("planform area is positive", f1.area > 0, f"{f1.area:.4f} m^2")

    # Structural mutation must stay evaluable and stay finite.
    for _ in range(30):
        c.mutate_add_node(rng)
        c.mutate_add_connection(rng)
    f3 = sample_surface(c, span=1.0, root_chord=0.3, stations=12)
    check("stays finite after 30 structural mutations",
          bool(np.all(np.isfinite(f3.chord)) and np.all(np.isfinite(f3.twist))))
    check("network grew", c.complexity > 12, f"complexity={c.complexity}")

    # Round trip.
    c2 = CPPN.from_dict(c.to_dict())
    f4 = sample_surface(c2, span=1.0, root_chord=0.3, stations=12)
    check("survives a serialisation round trip", np.allclose(f3.chord, f4.chord))


def test_the_gait_operator_moves_every_coordinate_at_once() -> None:
    """``mut_gait`` is only worth anything if it is *joint*.

    The whole case for it is that no single gait coordinate produces thrust on
    two of the three seed plans and all four together produce two thirds of the
    airframe's drag.  An operator that quietly degraded to moving one
    coordinate, or to moving one part, would look identical in every other test
    in this file -- genomes still build, children still differ from parents --
    and the arch39 arm would measure nothing.  So the jointness is asserted
    directly.
    """
    print("\ngenome: the gait operator is a joint move")
    from dytiscidae.core.genome import mut_gait

    rng = np.random.default_rng(11)

    # ``reference_genome`` rather than ``random_genome``: a random draw carries
    # at most two actuated joints (measured: 2 is the maximum over 200 draws),
    # and "every part" needs more than two parts to mean anything.
    g = reference_genome()
    movable_n = len([p for p in g.parts if p.joint != "none" and p.actuated])
    check("the reference genome has at least three actuated parts",
          movable_n >= 3, f"{movable_n}")
    if movable_n < 3:
        return

    movable = [p for p in g.parts if p.joint != "none" and p.actuated]
    before_f = g.flap_frequency
    before = [(p.stroke_amplitude, p.phase_offset, p.neutral) for p in movable]

    # 30 applications, because each coordinate can redraw close to where it was.
    moved = {"frequency": 0, "amplitude": 0, "phase": 0, "neutral": 0}
    all_parts_moved = 0
    for _ in range(30):
        c = g.copy()
        ok = mut_gait(c, rng)
        if not ok:
            continue
        cm = [p for p in c.parts if p.joint != "none" and p.actuated]
        moved["frequency"] += c.flap_frequency != before_f
        touched = 0
        for k, part in enumerate(cm):
            a, ph, ne = before[k]
            moved["amplitude"] += part.stroke_amplitude != a
            moved["phase"] += part.phase_offset != ph
            moved["neutral"] += part.neutral != ne
            touched += (part.stroke_amplitude != a and part.phase_offset != ph
                        and part.neutral != ne)
        all_parts_moved += touched == len(cm)

    check("it redraws the global frequency every time",
          moved["frequency"] == 30, f"{moved['frequency']}/30")
    n = 30 * len(movable)
    check("and every actuated part's amplitude, phase and rest offset",
          moved["amplitude"] > 0.9 * n and moved["phase"] > 0.9 * n
          and moved["neutral"] > 0.9 * n,
          f"amp {moved['amplitude']}/{n}, phase {moved['phase']}/{n}, "
          f"rest {moved['neutral']}/{n} -- a redraw can land on its own value, "
          f"so this is 90%, not 100%")
    check("all four coordinates on all parts in the same application",
          all_parts_moved >= 27, f"{all_parts_moved}/30 applications")

    # The axis-aligned operators are the contrast, and they must stay that way:
    # if one of them starts moving the whole gait, the comparison arch39 is
    # judged on stops meaning anything.
    from dytiscidae.core.genome import (mut_global_energy, mut_phase_gradient,
                                        mut_stroke)
    for name, op in (("stroke", mut_stroke), ("phase_gradient",
                     mut_phase_gradient), ("global_energy", mut_global_energy)):
        joint = 0
        for _ in range(60):
            c = g.copy()
            op(c, rng)
            cm = [p for p in c.parts if p.joint != "none" and p.actuated]
            f_moved = c.flap_frequency != before_f
            per = [(part.stroke_amplitude != before[k][0],
                    part.phase_offset != before[k][1],
                    part.neutral != before[k][2]) for k, part in enumerate(cm)]
            joint += f_moved and all(all(t) for t in per)
        check(f"`{name}` is still axis-aligned and never a whole-gait move",
              joint == 0, f"moved the whole gait on {joint}/60 applications")

    # No actuated joints: nothing to resample, and it must say so rather than
    # reporting a mutation the curator will then credit or blame.
    empty = g.copy()
    for part in empty.parts:
        part.actuated = False
    check("and it declines a genome with nothing actuated",
          mut_gait(empty, rng) is False)


def test_every_mutation_operator_keeps_the_genome_buildable() -> None:
    """No operator may produce a genome that cannot be built and compiled.

    This is the test that stops a rare mutation from killing a multi-hour run.
    """
    print("\ngenome: operator robustness")
    from dytiscidae.core.mjcf import compile_phenotype

    rng = np.random.default_rng(3)
    broken = []
    for name, op in MUTATION_OPERATORS.items():
        for trial in range(12):
            g = reference_genome() if trial % 2 else random_genome(rng)
            try:
                op(g, rng)
                p = build(g)
                compile_phenotype(p)
            except Exception as exc:
                broken.append(f"{name}: {type(exc).__name__}: {exc}")
                break
    check(f"all {len(MUTATION_OPERATORS)} operators keep genomes buildable",
          not broken, "; ".join(broken[:3]))

    # Deep mutation chains must also survive.
    survived = 0
    for i in range(25):
        g = random_genome(np.random.default_rng(i))
        try:
            for _ in range(20):
                g, _ = mutate(g, rng, n_ops=2)
            compile_phenotype(build(g))
            survived += 1
        except Exception:
            pass
    check("deep mutation chains stay valid", survived >= 24, f"{survived}/25")


def test_phenotype_invariants() -> None:
    print("\nphenotype: invariants")
    rng = np.random.default_rng(11)
    bad_mass, bad_vol, bad_buoy = 0, 0, 0
    for i in range(40):
        g = random_genome(rng)
        for _ in range(int(rng.integers(0, 6))):
            g, _ = mutate(g, rng, n_ops=2)
        p = build(g)
        if not (0.0 < p.mass < 1e4):
            bad_mass += 1
        if p.displaced_volume <= 0:
            bad_vol += 1
        # Buoyant volume can never exceed the outer envelope: a body cannot
        # generate more buoyancy than the water it displaces.
        if p.buoyant_volume > p.displaced_volume * 1.001:
            bad_buoy += 1
    check("mass is always finite and positive", bad_mass == 0, f"{bad_mass} bad")
    check("displaced volume is always positive", bad_vol == 0, f"{bad_vol} bad")
    check("buoyant volume never exceeds the envelope", bad_buoy == 0, f"{bad_buoy} bad")

    p = build(reference_genome())
    check("reference is structurally feasible", p.report.ok, p.report.summary())
    check("mass budget sums to the reported total",
          abs(p.budget.total - p.mass) < 1e-9)
    check("ballast can shed more than the surface excess",
          p.ballast_volume > 0, f"{p.ballast_volume*1e3:.1f} L")


def test_cpg_respects_joint_limits() -> None:
    print("\ncpg: joint limits")
    lims = np.array([[-0.5, 0.5], [-1.2, 0.3], [0.0, 1.0]])
    cpg = CPG(3, base_frequency=3.0, joint_range=lims)
    worst = 0.0
    wild = CPGParams(amplitude=np.array([9.0, 9.0, 9.0]),
                     phase=np.zeros(3), offset=np.array([5.0, -5.0, 5.0]),
                     frequency=3.0)
    for i in range(400):
        cmd = cpg.command(wild, i * 0.004)
        worst = max(worst, float(np.max(np.maximum(lims[:, 0] - cmd, cmd - lims[:, 1]))))
    check("commands stay inside joint travel even for absurd parameters",
          worst <= 1e-9, f"worst violation={worst:.3e} rad")

    pol = Policy(n_obs=11, n_modes=4, hidden=8)
    check("zero-weight policy commands nothing",
          np.allclose(pol.act(np.ones(11)), 0.0))
    check("policy weight count matches its layers",
          pol.n_weights == 11 * 8 + 8 + 8 * 4 + 4, f"{pol.n_weights}")


def test_a_rare_capability_survives_the_learned_projection() -> None:
    """A capability two designs in a hundred have must still get its own cell.

    This is the failure mode learned descriptors are known for and the one that
    would quietly undo the rest of this project's work on flight.  The archive's
    axes are fitted by PCA over sixteen behaviour features; PCA keeps the
    directions with the most variance, and "almost nobody does this" looks a lot
    like noise.  If flight is projected away, the one design that flies shares a
    cell with the ninety-eight that fall, loses it to whichever of them has the
    better aggregate score, and every fix in this session is averaged out.

    Measured on a synthetic population -- 96 fallers, 2 flyers -- it survives,
    and the reason it survives is worth stating: the features that go with
    flying (staying up, holding height, going fast while up there) move
    *together*, so they form a real correlated direction rather than a lone
    outlier in one coordinate.  PCA finds correlated directions, so a rare
    capability that shows up in several features at once is exactly the kind it
    keeps.  A rare capability visible in only one feature would not be, which is
    the honest limit of this check.
    """
    print("\ndescriptors: a rare capability keeps its own cell")
    from dytiscidae.evolution.descriptors import FEATURE_DIM, LearnedDescriptors

    rng = np.random.default_rng(0)

    def faller():
        f = np.zeros(FEATURE_DIM)
        f[0:3] = (0.36, 0.32, 0.32)
        f[3:6] = (6.0, 0.8, 0.3)
        f[6] = 1.2
        f[7] = 0.05            # holds no altitude
        f[8:10] = (2.5, 2.0)
        f[10:13] = (40.0, 30.0, 20.0)
        f[13:16] = (0.5, 1.0, 0.2)
        return f + rng.normal(0, 0.05, FEATURE_DIM) * np.abs(f).clip(0.05)

    def flyer():
        f = faller()
        f[0], f[3], f[7], f[8] = 0.95, 13.0, 0.90, 0.4
        return f

    n_fall, n_fly = 96, 2
    pop = [faller() for _ in range(n_fall)] + [flyer() for _ in range(n_fly)]
    d = LearnedDescriptors(n_dims=4, min_samples=60)
    for f in pop:
        d.observe(f)
    check("the projection fits", d.fit() and d.fitted)

    P = np.array([d.project(f) for f in pop])
    bounds = np.asarray(d.bounds(), float)

    def cell(p, bins: int = 8):
        idx = np.floor((p - bounds[:, 0])
                       / np.maximum(bounds[:, 1] - bounds[:, 0], 1e-9) * bins)
        return tuple(int(np.clip(v, 0, bins - 1)) for v in idx)

    cells_fall = {cell(P[i]) for i in range(n_fall)}
    cells_fly = {cell(P[i]) for i in range(n_fall, n_fall + n_fly)}
    check("the flyers do not share a cell with anything that falls",
          not (cells_fly & cells_fall),
          f"{len(cells_fall)} faller cells, flyer cells {sorted(cells_fly)}")

    sig = (P[n_fall] - P[:n_fall].mean(axis=0)) / np.maximum(P[:n_fall].std(axis=0), 1e-9)
    check("and they are far out along a learned axis, not marginally off one",
          float(np.abs(sig).max()) > 5.0,
          f"{float(np.abs(sig).max()):.0f} sigma from the cloud on axis "
          f"{int(np.argmax(np.abs(sig)))}")
    check("which the run can name, so the archive is not a map with blank axes",
          "air_time_fraction" in d.axis_meaning(int(np.argmax(np.abs(sig))))
          or "net_altitude_change" in d.axis_meaning(int(np.argmax(np.abs(sig)))),
          d.axis_meaning(int(np.argmax(np.abs(sig)))))


def test_learned_descriptors_replace_the_hand_picked_axes() -> None:
    """The archive's axes must be learnable from behaviour, and re-binning must
    not silently shrink the archive.

    ``BD_AXES`` was a list I wrote -- log mass, density ratio, air competence,
    water competence -- and each entry silently decided what the search would
    call a different *kind* of machine.  Two designs differing in a way none of
    those axes captures collide in one cell and one is discarded, so the axes
    bound what can be found in the same way a fixed part taxonomy bounds what
    can be built.
    """
    print("\narchive: axes learned from behaviour")
    from dytiscidae.envs.evaluate import BD_AXES
    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.descriptors import FEATURE_DIM, LearnedDescriptors

    rng = np.random.default_rng(0)
    learner = LearnedDescriptors(n_dims=4, refit_every=100, min_samples=60)
    archive = Archive(BD_AXES)

    # Three behavioural clusters: an air specialist, a water specialist, a
    # walker.  Nothing tells the projector they exist.
    for i in range(300):
        k = i % 3
        f = np.zeros(FEATURE_DIM)
        f[0:3] = np.roll([0.7, 0.2, 0.1], k) + rng.normal(0, 0.05, 3)
        f[3 + k] = rng.uniform(1, 8)
        f[6] = rng.uniform(0, 12)
        f[10 + k] = rng.uniform(20, 300)
        learner.observe(f)
        archive.add(
            genome=f"g{i}", fitness=float(rng.random()),
            descriptor=np.array([rng.uniform(-0.4, 1.6), rng.uniform(0.15, 1.5),
                                 rng.random(), rng.random()]),
            meta={"features": [float(x) for x in f]},
        )

    before = len(archive.cells)
    check("the projection fits from the run's own data", learner.fit(), f"{learner.seen} episodes")
    check("and it is fitted", learner.fitted)

    axes = [(f"latent{i}", float(lo), float(hi), 8) for i, (lo, hi) in enumerate(learner.bounds())]
    stats = archive.rebin(axes, lambda e: learner.project(np.asarray(e.meta["features"], float)))
    check("every elite is re-projected, none is dropped by error",
          stats["before"] == before and stats["after"] > 0.5 * before,
          f"{stats['before']} -> {stats['after']} ({stats['merged']} merged)")
    check("the merge count is reported rather than hidden", stats["merged"] >= 0,
          f"{stats['merged']} designs the new axes call the same")

    # A learned axis with no label is a map with unlabelled coordinates.
    meanings = learner.report()["axes"]
    check("each learned axis reports what it is made of", len(meanings) == 4 and all(meanings),
          meanings[0])
    # The clusters differ mainly in which domain they spend time in, so at least
    # one axis must be dominated by a time-fraction feature.
    check("the axes pick up the structure that is actually in the data",
          any("time_fraction" in m for m in meanings),
          " | ".join(m.split()[0] for m in meanings))

    # Re-binning must be idempotent: projecting twice through the same fit
    # cannot keep merging.
    again = archive.rebin(axes, lambda e: learner.project(np.asarray(e.meta["features"], float)))
    check("re-binning twice through one fit changes nothing", again["merged"] == 0,
          f"{again['merged']} merged on the second pass")

    # The schedule has to fire on the cadence a real run produces.  It used to
    # test ``seen % refit_every == 0``, checked once per generation -- so it only
    # fired if the running total landed exactly on a multiple, and a seeding
    # phase contributing an odd number of episodes offset the total permanently.
    # A 25-generation run that should have refitted five times refitted zero
    # times, and looked identical to a run whose axes were never learned.
    sched = LearnedDescriptors(n_dims=4, refit_every=20, min_samples=60)
    fired = 0
    for _ in range(5):                      # an odd-sized seeding phase
        sched.observe(rng.normal(0, 1, FEATURE_DIM))
    for _ in range(40):                     # then generations of four
        for _ in range(4):
            sched.observe(rng.normal(0, 1, FEATURE_DIM))
        if sched.due_for_refit() and sched.fit():
            fired += 1
    check("refits fire on the cadence a real run produces", fired >= 5,
          f"{fired} refits over 165 episodes at one per 20")


def test_cells_hold_a_pareto_front_not_a_weighted_sum() -> None:
    """A design must not be discarded because of an exchange rate I invented.

    ``fitness`` was a weighted sum: mission fraction plus a tenth of the
    structural margin plus a tenth of the energy margin plus a tenth of the land
    competence.  Those coefficients are three numbers I typed, and the cost of
    typing them was invisible -- a design giving up a hundredth of its mission
    fraction for three times the structural margin was kept or thrown away
    purely according to them, and nothing in the run reported which.
    """
    print("\narchive: cells hold a Pareto front")
    from dataclasses import dataclass, field as dc_field

    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.curator import Curator

    axes = [("a", 0.0, 1.0, 4), ("b", 0.0, 1.0, 4)]
    a = Archive(axes)
    cell = a.cell_of([0.5, 0.5])

    a.add("fast", 0.50, [0.5, 0.5], objectives=np.array([0.50, 0.05, 0.05]))
    st = a.add("robust", 0.72, [0.5, 0.5], objectives=np.array([0.49, 2.80, 1.90]))
    names = sorted(e.genome for e in a.front(cell))
    check("a design that trades mission for margin is kept, not discarded",
          st == "improved" and names == ["fast", "robust"], f"{st}, front={names}")
    check("the representative is the one that best does the task",
          a.cells[cell].genome == "fast", a.cells[cell].genome)

    st = a.add("worse", 0.41, [0.5, 0.5], objectives=np.array([0.40, 0.02, 0.02]))
    check("a strictly dominated design is still rejected", st == "rejected", st)

    st = a.add("better", 0.90, [0.5, 0.5], objectives=np.array([0.60, 2.90, 2.00]))
    check("a strictly dominating design collapses the front to itself",
          st == "improved" and [e.genome for e in a.front(cell)] == ["better"],
          str([e.genome for e in a.front(cell)]))

    # Coverage must still mean what it meant: one cell is one cell.
    check("coverage still counts cells, not designs",
          len(a.cells) == 1 and a.coverage == 1 / a.capacity, f"{a.coverage:.4f}")

    # The front is bounded, and what survives spans the trade-off.
    b = Archive(axes)
    rng = np.random.default_rng(1)
    for i in range(40):
        t = i / 39.0            # a clean trade-off curve, nothing dominates
        b.add(f"d{i}", float(rng.random()), [0.5, 0.5],
              objectives=np.array([t, 1.0 - t, 0.5]))
    front = b.front(b.cell_of([0.5, 0.5]))
    check("the front is bounded", len(front) <= b.front_capacity, f"{len(front)} designs")
    spread = max(e.objectives[0] for e in front) - min(e.objectives[0] for e in front)
    check("and what survives spans the trade-off rather than clustering",
          spread > 0.8, f"mission spread {spread:.2f} across the kept front")

    # The overflow branch, with the candidate interior on every objective.
    #
    # This crashed a three-thousand-generation run five generations in.  The
    # branch asked ``if cand not in kept``; ``in`` falls back to ``==``, the
    # generated dataclass __eq__ compares field tuples, and the comparison
    # reaches a numpy array -- "truth value of an array with more than one
    # element is ambiguous", raised from inside an unrelated call.
    #
    # Two things had to line up for it, which is why the clean trade-off line
    # above never saw it.  The candidate must lose on crowding, so it must be
    # interior on *every* objective -- with random objectives it is almost
    # always extreme in at least one, gets infinite crowding, and is found by
    # identity before any comparison happens.  And the genome must itself be a
    # dataclass holding an array, because tuple comparison short-circuits on the
    # first unequal field and the genome comes first.  A real Genome is exactly
    # that, through Part.joint_axis.
    @dataclass
    class _ArrayGenome:
        axis: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))

    c = Archive(axes)
    for corner in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.9, 0.9, 0.0)]:
        c.add(_ArrayGenome(), 0.5, [0.5, 0.5], objectives=np.array(corner))
    st = c.add(_ArrayGenome(), 0.5, [0.5, 0.5], objectives=np.array([0.5, 0.5, 0.5]))
    check("a candidate that loses on crowding is handled, not raised on",
          st in ("improved", "rejected"), f"status {st}")
    check("and the front stays at capacity",
          len(c.front(c.cell_of([0.5, 0.5]))) == c.front_capacity,
          f"{len(c.front(c.cell_of([0.5, 0.5])))} designs")

    # Many mutually-competing designs, which is what a real run produces.
    c = Archive(axes)
    r2 = np.random.default_rng(0)
    for _ in range(60):
        t = r2.random()
        c.add(_ArrayGenome(), float(r2.random()), [0.5, 0.5],
              objectives=np.array([t, 1.0 - t, float(r2.random())]))
    cf = c.front(c.cell_of([0.5, 0.5]))
    check("sixty mutually-competing designs in one cell do not crash it",
          0 < len(cf) <= c.front_capacity, f"front holds {len(cf)}")

    # Dropping a cell must drop all of it.  Occupancy lives in two structures
    # now -- the front and the representative -- and quarantine and pruning both
    # deleted from ``cells`` alone.  The next candidate landing in that cell
    # then found a non-empty front with no representative, and the run died on a
    # KeyError sixty-eight generations in.
    c = Archive(axes)
    c.add("a", 0.5, [0.5, 0.5], objectives=np.array([0.5, 1.0, 1.0]))
    c.add("b", 0.6, [0.5, 0.5], objectives=np.array([0.4, 2.0, 1.0]))
    cell2 = c.cell_of([0.5, 0.5])
    check("a cell holds a front and a representative", len(c.front(cell2)) == 2)
    c.remove(cell2)
    check("removing a cell empties both", not c.front(cell2) and cell2 not in c.cells,
          f"front={len(c.front(cell2))} cells={cell2 in c.cells}")
    st = c.add("d", 0.5, [0.5, 0.5], objectives=np.array([0.5, 1.0, 1.0]))
    check("and the cell can be refilled afterwards", st == "new", st)

    # The two paths that used to desync it, exercised through the curator.
    for drop in ("quarantine", "prune"):
        c = Archive(axes)
        cur = Curator(c, seed=0)
        for i in range(3):
            c.add(f"x{i}", 0.5 + 0.1 * i, [0.5, 0.5],
                  objectives=np.array([0.5 + 0.1 * i, 1.0, 1.0]))
        if drop == "quarantine":
            for _ in range(3):
                cur.quarantine(np.array([0.5, 0.5]), "test", "g")
        else:
            c.remove(c.cell_of([0.5, 0.5]))
        cell3 = c.cell_of([0.5, 0.5])
        consistent = (cell3 in c.cells) == bool(c.front(cell3))
        check(f"{drop} leaves the front and the representative in step", consistent,
              f"cells={cell3 in c.cells} front={len(c.front(cell3))}")
        st = c.add("after", 0.9, [0.5, 0.5], objectives=np.array([0.9, 1.0, 1.0]))
        check(f"and a candidate can still land there after {drop}",
              st in ("new", "improved", "rejected"), st)

    # Everything kept must be reachable as a parent, or keeping it is pointless.
    cur = Curator(a, seed=0)
    a.add("robust2", 0.7, [0.5, 0.5], objectives=np.array([0.55, 3.00, 2.00]))
    picked = {cur.select_parent().genome for _ in range(200)}
    check("parents are drawn from the whole front",
          len(picked & {e.genome for e in a.front(cell)}) == len(a.front(cell)),
          f"sampled {sorted(picked)} from front {sorted(e.genome for e in a.front(cell))}")


def test_intervention_is_triggered_by_evidence_not_a_schedule() -> None:
    """When to intervene must come from the data, not from a number I typed.

    It used to be two constants: a competence floor of 0.35 below which a
    domain counted as starved, and a patience of 25 generations before acting.
    Neither had a basis.  0.35 is not a property of flight and 25 generations is
    not a property of anything -- they produced behaviour that looked reasonable
    when I watched a few runs, which is the hand-tuning this project exists to
    remove.

    Both are gone.  Mission fraction is built on ``min(competences)``, so the
    weakest domain is the binding constraint by construction and no floor is
    needed to find it.  Whether it is stalled or merely slow is answered by the
    record process: under exchangeable draws, the chance that none of the next
    m beats the best of the first n is exactly n/(n+m).
    """
    print("\ncurator: intervention follows the evidence")
    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.curator import Curator

    rng = np.random.default_rng(0)

    def fresh() -> Curator:
        return Curator(Archive([("x", 0.0, 1.0, 4), ("y", 0.0, 1.0, 4)]), seed=0)

    # Still improving: never intervene, however long the run.
    c = fresh()
    for i in range(300):
        c.observe_domains({"air": 0.002 * i, "water": 0.8, "land": 0.7})
    check("a domain still setting records is left alone",
          c.plateau_p("air") > c.PLATEAU_ALPHA, f"p={c.plateau_p('air'):.3f} after 300 draws")

    # A hard ceiling, then a drought three times as long as it took to get there.
    c = fresh()
    for i in range(100):
        c.observe_domains({"air": 0.003 * i, "water": 0.8, "land": 0.7})
    for _ in range(400):
        c.observe_domains({"air": 0.20 * rng.random(), "water": 0.8, "land": 0.7})
    check("a genuine plateau is detected", c.plateau_p("air") <= c.PLATEAU_ALPHA,
          f"p={c.plateau_p('air'):.3f} after 400 draws with no new best")

    # The same ceiling but a short drought: bad luck is not yet evidence.
    c = fresh()
    for i in range(100):
        c.observe_domains({"air": 0.003 * i, "water": 0.8, "land": 0.7})
    for _ in range(120):
        c.observe_domains({"air": 0.20 * rng.random(), "water": 0.8, "land": 0.7})
    check("a short drought does not trigger an intervention",
          c.plateau_p("air") > c.PLATEAU_ALPHA,
          f"p={c.plateau_p('air'):.3f} after only 120 draws")

    # And it acts on whichever domain is actually weakest, not one I nominated.
    c = fresh()
    for i in range(80):
        c.observe_domains({"air": 0.9, "water": 0.002 * i, "land": 0.7})
    for _ in range(400):
        c.observe_domains({"air": 0.9, "water": 0.10 * rng.random(), "land": 0.7})
    c.archive.add("e", 0.5, [0.5, 0.5], meta={"air": 0.9, "water": 0.16, "land": 0.7},
                  objectives=np.array([0.5, 1.0, 1.0]))
    acute = c.check_famine()
    check("the binding domain is identified without a threshold",
          acute == ["water"], f"{acute} from bests "
          f"{ {k: round(v, 2) for k, v in c.domain_bests().items()} }")
    check("and the run can say why it intervened",
          "p=" in c.regime.note and "water" in c.regime.note, c.regime.note[:90])

    # Too early to judge anything.
    c = fresh()
    for i in range(4):
        c.observe_domains({"air": 0.1 * i, "water": 0.8, "land": 0.7})
    check("nothing is called a drought before there is any history",
          c.plateau_p("air") == 1.0, f"p={c.plateau_p('air'):.3f} after 4 draws")


def test_no_dataclass_can_raise_on_equality() -> None:
    """No dataclass in the package may reach a numpy array through ``==``.

    This bit twice in one day, in two unrelated places, and both times the
    failure surfaced far from the cause: "the truth value of an array with more
    than one element is ambiguous", raised from inside an archive insertion that
    never mentions equality.

    The mechanism is that ``@dataclass`` generates an ``__eq__`` comparing the
    tuple of all fields, and tuple comparison walks into whatever those fields
    contain.  One numpy array anywhere in that graph -- ``Part.joint_axis``,
    four levels below the object actually being compared -- makes ``==`` and
    therefore ``in``, ``list.remove``, ``dict`` lookups by value and
    ``assertEqual`` all raise.  Nothing in the type checker or the test suite
    sees it, because it depends on the *values* lining up: comparison
    short-circuits at the first unequal field, so the array is only reached when
    everything before it happens to match.

    Every one of these classes is mutable state, not a value, so identity is the
    correct equality anyway.  ``eq=False`` also restores ``__hash__``, which
    makes them usable in sets and as dict keys.

    Scanning for it is better than remembering it: a new dataclass with an array
    field is the most ordinary thing to write in this codebase.
    """
    print("\npackage: equality never touches an array")
    import dataclasses
    import importlib
    import inspect
    import pkgutil
    import sys
    import typing

    import dytiscidae

    for m in pkgutil.walk_packages(dytiscidae.__path__, "dytiscidae."):
        try:
            importlib.import_module(m.name)
        except Exception:
            pass

    found: dict[str, type] = {}
    for mod in list(sys.modules.values()):
        if not getattr(mod, "__name__", "").startswith("dytiscidae"):
            continue
        for obj in vars(mod).values():
            if dataclasses.is_dataclass(obj) and inspect.isclass(obj):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj

    def reaches_array(cls, stack=()) -> bool:
        if cls in stack or len(stack) > 6:
            return False
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {f.name: f.type for f in dataclasses.fields(cls)}
        for f in dataclasses.fields(cls):
            t = hints.get(f.name, f.type)
            for sub in [t, *typing.get_args(t)]:
                if "ndarray" in str(sub):
                    return True
                if (dataclasses.is_dataclass(sub) and inspect.isclass(sub)
                        and reaches_array(sub, stack + (cls,))):
                    return True
        return False

    check("the scan finds dataclasses at all", len(found) > 20, f"{len(found)} dataclasses")
    array_holders = {n: c for n, c in found.items() if reaches_array(c)}
    check("and finds the ones that hold arrays", len(array_holders) > 10,
          f"{len(array_holders)} of {len(found)}")

    unsafe = sorted(n for n, c in array_holders.items() if c.__dataclass_params__.eq)
    check("none of them generates an __eq__ that walks into one", not unsafe,
          "; ".join(unsafe) if unsafe else f"all {len(array_holders)} use identity")

    # And the two objects that actually crashed must survive the operations that
    # crashed on them.
    from dytiscidae.core.genome import random_genome
    from dytiscidae.evolution.archive import Elite

    rng = np.random.default_rng(0)
    g1, g2 = random_genome(rng), random_genome(rng)
    ok = True
    try:
        _ = g1 == g2
        _ = g1 in [g2, g1]
        e1 = Elite(genome=g1, fitness=1.0, descriptor=np.zeros(4), cell=(0, 0, 0, 0))
        e2 = Elite(genome=g2, fitness=1.0, descriptor=np.zeros(4), cell=(0, 0, 0, 0))
        _ = e1 in [e2, e1]
        _ = {e1, e2}
    except ValueError:
        ok = False
    check("comparing and containment-testing genomes and elites does not raise", ok)


def test_arriving_somewhere_is_not_one_lucky_timestep() -> None:
    """A crossing has to be held, and a gait with an aerial phase is still a gait.

    ``current_domain`` is an instantaneous reading: AIR whenever nothing is
    touching and nothing is submerged.  A transition counted the first single
    step on which that reading matched the command, so a walking machine banked
    the land-to-air crossing every time it picked its feet up.  Measured on a
    land episode the beetle reads airborne for 25.7% of steps and the gannet for
    1.7%, in bursts of 0.07 to 0.14 s, and a crossing is worth 35% of the
    continuous mission score across two of them.

    Demanding an unbroken half second instead trades one error for its mirror:
    the beetle is on the ground for 74% of a land leg and never holds it
    unbroken for that long, so a real walker arrives nowhere.  Both are the same
    mistake -- reading an instantaneous signal as a state -- and the fix is to
    judge arrival over a window, which is what a majority of half a second is.
    """
    print("\nmission: arriving is held, not glimpsed")
    from dytiscidae.envs.mission import ENTRY_MAJORITY, ENTRY_WINDOW

    dt = 0.004
    n = max(int(ENTRY_WINDOW / dt), 1)

    def arrives(pattern) -> bool:
        """Would this sequence of in-domain readings count as arrival?"""
        from collections import deque

        w: deque = deque()
        for flag in pattern:
            w.append(bool(flag))
            if len(w) > n:
                w.popleft()
            if len(w) == n and sum(w) >= ENTRY_MAJORITY * n:
                return True
        return False

    rng = np.random.default_rng(0)
    steps = int(6.0 / dt)
    # Contact chatter: airborne a quarter of the time in bursts far shorter than
    # the window.  This is the beetle walking, and it must not read as flight.
    chatter = np.zeros(steps, dtype=bool)
    i = 0
    while i < steps:
        if rng.random() < 0.25:
            burst = int(rng.integers(1, max(int(0.14 / dt), 2)))
            chatter[i:i + burst] = True
            i += burst
        i += int(rng.integers(1, 40))
    check("contact chatter is not a crossing",
          not arrives(chatter),
          f"airborne {chatter.mean()*100:.0f}% of steps in bursts, still not arrived")

    # A running gait: on the ground 74% of the time, with regular aerial phases.
    gait = np.ones(steps, dtype=bool)
    period = max(int(0.20 / dt), 3)
    for k in range(0, steps, period):
        gait[k:k + max(int(0.26 * period), 1)] = False
    check("but a gait with an aerial phase still arrives",
          arrives(gait),
          f"on the ground {gait.mean()*100:.0f}% of the time")

    # And a real crossing, obviously.
    real = np.zeros(steps, dtype=bool)
    real[int(1.0 / dt):] = True
    check("and so does actually going there", arrives(real))

    # End to end, on the two plans the numbers above came from.
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.mission import build_schedule, run_continuous
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv

    spec = MissionSpec(cycles=1, seconds_per_domain=8.0)
    for name in ("gannet", "beetle"):
        env = TriphibianEnv(build(BODY_PLANS[name]()), seed=0)
        r = run_continuous(env, None,
                           build_schedule(spec, np.random.default_rng(0), leg_seconds=8.0))
        air = next((leg for leg in r.legs if leg.commanded.value == "air"), None)
        land = next((leg for leg in r.legs if leg.commanded.value == "land"), None)
        check(f"{name}: does not claim the air leg it never flew",
              air is not None and not air.entered,
              f"air on-task {air.on_task_fraction*100:.1f}%, entered={air.entered}")
        check(f"{name}: does claim the land it is standing on",
              land is not None and land.entered,
              f"land on-task {land.on_task_fraction*100:.1f}%, entered={land.entered}")


def test_a_mujoco_auto_reset_ends_the_continuous_mission() -> None:
    """A teleport is not a mission.

    On a bad acceleration MuJoCo resets the state and keeps stepping: the
    machine reappears at its default pose, which is finite and inside 400 m, so
    ``run_continuous``'s position check never fired and the mission carried on
    across the jump -- measuring leg distance through it, and able to credit a
    domain change the machine never made.  The scored segments were already
    gated on ``bad_qacc``; the continuous mission, which the showcase films and
    controller training's continuous mode scores, read nothing.

    The divergence is injected rather than waited for, because the rate on real
    designs is 0.15% of scored rollouts and a test cannot wait for that.
    """
    print("\nmission: a MuJoCo auto-reset ends the continuous mission")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.mission import build_schedule, run_continuous
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv

    spec = MissionSpec(cycles=1, seconds_per_domain=2.0)

    def mission(inject_at):
        env = TriphibianEnv(build(BODY_PLANS["gannet"]()), seed=0)
        orig, k = env.step, [0]

        def step(angles):
            k[0] += 1
            if inject_at is not None and k[0] == inject_at:
                # After the fluid solver, which owns ``qfrc_applied`` and
                # rewrites it every step (the implicit damping split): written
                # before it, the injection was erased.
                sa = env.solver.apply

                def bad(d, t):
                    out = sa(d, t)
                    d.qfrc_applied[2] = 1e300        # one bad qacc
                    return out
                env.solver.apply = bad
                try:
                    ok = orig(angles)
                finally:
                    env.solver.apply = sa
            else:
                ok = orig(angles)
            env.data.qfrc_applied[:] = 0.0
            return ok
        env.step = step
        r = run_continuous(env, None, build_schedule(
            spec, np.random.default_rng(0), leg_seconds=2.0))
        return r, k[0]

    clean, _ = mission(None)
    check("the same mission without an injected divergence survives",
          clean.survived, f"failure {clean.failure!r}")

    hit, steps = mission(400)
    check("an auto-reset mid-mission fails it, as 'unstable'",
          not hit.survived and hit.failure == "unstable",
          f"survived={hit.survived}, failure={hit.failure!r}")
    check("and it stops at the reset rather than stepping on from the teleport",
          steps == 400, f"stepped {steps} times, injected at step 400")


def test_transitions_are_graded_not_pass_fail() -> None:
    """A crossing must be scored on how it was done, not only on whether it
    happened.

    It used to return a boolean: True if the machine's depth changed sign at any
    point in six seconds.  Under that rule a machine that fell through the
    surface tumbling, at twice the speed its hull survives, scored exactly what
    a clean controlled entry scored.  The crossings are where every real
    triphibian machine spends its structure and its energy, and handing all of
    that to one bit left the search no gradient to climb toward doing it well.
    """
    print("\ntransitions: scored, not merely survived")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.transitions import (
        TRANSITION_ENDPOINTS,
        TransitionSet,
        _place_for,
        run_transition,
    )
    from dytiscidae.envs.triphibian import TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["ray"]()))
    ctrl = Controller(params=env.cpg.base)

    # Every crossing must start from a state that is physically valid: outside
    # the terrain.  Starting a water-to-land crossing at a fixed depth put the
    # machine inside the submerged ramp, which is the same mistake that made the
    # land domain unreachable for the whole project.
    from dytiscidae.core.mjcf import beach_surface_z

    for kind in TRANSITION_ENDPOINTS:
        _place_for(env, kind)
        # Solid ground only.  Being below the *water* surface is the whole point
        # of a crossing that starts submerged, so the check is against the beach
        # ramp, not against the waterline.
        g = env._machine_geoms
        aabb = env.model.geom_aabb.reshape(-1, 6)[g]
        R = env.data.geom_xmat[g].reshape(-1, 3, 3)
        centre_z = env.data.geom_xpos[g][:, 2] + np.einsum("nij,nj->ni", R, aabb[:, :3])[:, 2]
        bottom = centre_z - np.einsum("nj,nj->n", np.abs(R[:, 2, :]), aabb[:, 3:])
        rock = np.array([beach_surface_z(float(x)) for x in env.data.geom_xpos[g][:, 0]])
        into_rock = float(np.min(bottom - rock))
        check(f"{kind} starts outside solid ground", into_rock > -0.05,
              f"lowest geometry sits {into_rock:+.3f} m above the ramp")

    r = run_transition(env, "air_to_water", ctrl, duration=5.0)
    comp = r.components
    check("a crossing reports separable components", set(comp) == {
        "crossed", "shock", "control", "settle", "economy", "exit_state"},
        ", ".join(sorted(comp)))
    check("all components are normalised", all(0.0 <= v <= 1.0 for v in comp.values()),
          str({k: round(v, 2) for k, v in comp.items()}))
    check("and the raw physics is kept alongside them",
          r.survivable_entry_speed > 0.0 and r.duration > 0.0,
          f"entry {r.peak_entry_speed:.1f} m/s against a {r.survivable_entry_speed:.1f} m/s limit")

    # Entry shock has to be a slope, not a cliff: half the hull limit must beat
    # nine tenths of it.
    from dytiscidae.envs.transitions import TransitionResult

    def shock_at(speed, limit=10.0):
        t = TransitionResult(kind="air_to_water", crossed=True)
        t.peak_entry_speed, t.survivable_entry_speed = speed, limit
        t.shock = float(np.clip(1.0 - (speed / limit) ** 2, 0.0, 1.0))
        return t.shock

    check("entering slowly beats entering fast", shock_at(5.0) > shock_at(9.0) > 0.0,
          f"5 m/s -> {shock_at(5.0):.2f}, 9 m/s -> {shock_at(9.0):.2f}")
    check("and exceeding the hull limit scores nothing", shock_at(11.0) == 0.0,
          f"{shock_at(11.0):.2f}")

    # Refusing the hard crossing must not raise the average.
    ts = TransitionSet()
    ts.results["air_to_water"] = run_transition(env, "air_to_water", ctrl, duration=4.0)
    one = ts.component_means()["crossed"]
    ts.results["water_to_air"] = run_transition(env, "water_to_air", ctrl, duration=4.0)
    two = ts.component_means()["crossed"]
    check("a failed crossing drags the mean down rather than being skipped",
          two <= one, f"{one:.2f} -> {two:.2f} after adding a second crossing")


def test_an_attempted_takeoff_outscores_never_leaving_the_ground() -> None:
    """Getting partway off the ground must be worth more than not trying.

    ``land_to_air`` asks for clearance above half a metre *at the end* of the
    episode, which is sustained flight rather than a leap, and the gate it
    feeds is multiplicative.  So while the gate read ``crossed`` directly,
    every design that failed it scored the same zero: a machine that leapt
    0.566 m and came down was worth exactly what a machine that never moved
    was worth, and the search had no gradient to climb toward takeoff.

    The fix must give a gradient without giving away the crossing, so this
    pins both directions at once.
    """
    print("\nenvs: an attempted takeoff beats sitting still")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.transitions import TransitionResult, run_transition
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    scores = {}
    for name in ("gannet", "teal"):
        env = TriphibianEnv(build(BODY_PLANS[name]()))
        ctrl = Controller(params=env.cpg.base)
        scores[name] = run_transition(env, "land_to_air", ctrl, duration=6.0)

    sit, leap = scores["gannet"], scores["teal"]
    check("the leaper actually leaves the ground and the other does not",
          leap.peak_clearance > 5.0 * sit.peak_clearance,
          f"peak clearance {sit.peak_clearance:.3f} m vs {leap.peak_clearance:.3f} m")
    check("neither of them completes the crossing",
          not sit.crossed and not leap.crossed,
          f"crossed {sit.crossed} / {leap.crossed}")
    check("yet the leap scores strictly more than sitting still",
          leap.approach > sit.approach + 0.1,
          f"approach {sit.approach:.3f} vs {leap.approach:.3f} -- both were 0.000")
    check("and both stay below what completing the crossing pays",
          max(sit.approach, leap.approach) <= 0.6 + 1e-9,
          f"capped at {max(sit.approach, leap.approach):.3f} against 1.0 for a crossing")

    # Height alone must not buy it, and neither must hang time: a single
    # ballistic hop that lands at once, and a machine that never quite touches
    # while going nowhere, are both things this should refuse to pay for.
    hop = TransitionResult(kind="land_to_air")
    hop.peak_clearance, hop.airborne_fraction = 0.6, 0.02
    drift = TransitionResult(kind="land_to_air")
    drift.peak_clearance, drift.airborne_fraction = 0.02, 0.9
    from dytiscidae.envs.transitions import _score

    for res in (hop, drift):
        _score(TriphibianEnv(build(BODY_PLANS["gannet"]())), res, -1,
               np.array([1.0]), np.array([0.0]), Domain.AIR)
    check("one term alone cannot earn a full approach",
          max(hop.approach, drift.approach) < 0.35,
          f"height-only {hop.approach:.3f}, hangtime-only {drift.approach:.3f}, "
          f"both against {leap.approach:.3f} for a real leap")
    check("a completed crossing is still worth exactly one",
          TransitionResult(kind="x", crossed=True, approach=1.0).components["crossed"] == 1.0,
          "the gate is unchanged where it mattered")


def test_judge_ladder_is_fixed_and_bar_only_tightens() -> None:
    """The standard must get harder as the population improves, without making
    the record incomparable.

    Every threshold in this project began as a number I typed, and each one
    decides invisibly where the search stops trying: once a population saturates
    a threshold the gradient vanishes.  But a bar that simply tracks the
    population destroys the thing that makes a long run readable -- 0.8 at
    generation 100 and 0.8 at generation 2000 become different achievements with
    nothing in the record to say so.

    So the two are separated.  *What* is measured is a fixed ladder of
    qualitatively different capabilities, declared once.  *Where the bar sits*
    inside the current rung is a population quantile that ratchets.
    """
    if needs_batched_evaluator("test_judge_ladder_is_fixed_and_bar_only_tightens"):
        return
    print("\njudge: fixed ladder, ratcheting bar")
    from dytiscidae.evolution.judge import LADDER, Judge, rung_reached

    j = Judge(quantile=0.9, update_every=1)

    # The ladder must be a progression: more capability, more rungs.
    # ``holds_station`` sits between holding height and climbing: a machine
    # gliding down at 0.4 m/s clears ``holds_height`` for a whole segment while
    # never holding a height, so the two are separate rungs.
    # The `lift_margin` rungs sit below these and every fixture has to carry one,
    # because a body that cannot lift its own weight is not flying whatever the
    # episode looked like -- that is the whole point of them.  The offset is
    # derived rather than written out, so inserting another rung moves the
    # expectations instead of breaking them.
    LIFT = sum(1 for _n, m, _t in LADDER["air"] if m == "lift_margin")
    # The thrust rungs sit between `holds_height` and `holds_station`, because
    # everything below them can be earned by a glider and everything above them
    # needs the flapping to make net forward force.  Derived, not written out,
    # so inserting another rung moves the expectations instead of breaking them.
    THR = sum(1 for _n, m, _t in LADDER["air"] if m == "thrust_margin")
    flies = {"lift_margin": 1.5}
    pushes = {**flies, "thrust_margin": 2.0}
    seq = [
        ({**flies, "airborne_fraction": 0.05}, LIFT + 0),
        ({**flies, "airborne_fraction": 0.7, "sink_rate": 6.0}, LIFT + 2),
        ({**flies, "airborne_fraction": 0.7, "sink_rate": 2.0}, LIFT + 3),
        ({**flies, "airborne_fraction": 0.9, "sink_rate": 0.2}, LIFT + 4),
        ({**pushes, "airborne_fraction": 0.9, "sink_rate": 0.2,
          "station_keeping": 0.8}, LIFT + THR + 5),
        ({**pushes, "airborne_fraction": 0.95, "sink_rate": -1.0,
          "station_keeping": 0.8, "turn_response": 0.0}, LIFT + THR + 6),
    ]
    ok = all(rung_reached("air", m) == k for m, k in seq)
    check("the ladder orders capability", ok,
          " ".join(str(rung_reached("air", m)) for m, _ in seq))
    check("a design cannot skip a rung it failed",
          rung_reached("air", {**flies, "airborne_fraction": 0.05,
                               "sink_rate": -9.0}) == LIFT,
          "plummeting-but-never-airborne stops where it stopped being airborne")
    # And the lift rungs order the same way: 72.6% of arch36 could not lift its
    # own weight at any speed, scored the two `airborne_fraction` rungs for
    # being dropped from 30 m, and had no gradient between "makes no lift" and
    # "flies".
    perfect = {"airborne_fraction": 1.0, "sink_rate": -2.0, "thrust_margin": 2.0,
               "station_keeping": 0.9, "turn_response": 0.9}
    check("a body that makes no lift scores nothing in air",
          rung_reached("air", {**perfect, "lift_margin": 0.0}) == 0,
          "a whole segment airborne, climbing, and it still cannot fly")
    check("and one halfway there sits between that and flying",
          0 < rung_reached("air", {**perfect, "lift_margin": 0.35}) < LIFT,
          f"margin 0.35 -> rung {rung_reached('air', {**perfect, 'lift_margin': 0.35})}")

    # A glider is not a flyer, and until arch38 the ladder could not say so.
    # Everything up to `holds_height` is earnable by descending slowly;
    # `holds_station` and `climbs` need thrust, and thrust was unmeasured for
    # four runs -- median -0.0030 over arch37's 182 elites, every hand-built
    # seed at zero or negative, and `holds_station` reached once in 14,092
    # evaluations.
    glider = {"lift_margin": 3.0, "airborne_fraction": 1.0, "sink_rate": 0.2,
              "station_keeping": 0.9, "turn_response": 0.9}
    check("a glider stops where gliding stops, however well it held station",
          rung_reached("air", {**glider, "thrust_margin": -0.2}) == LIFT + 4,
          f"rung {rung_reached('air', {**glider, 'thrust_margin': -0.2})} -- "
          f"held station 0.9 and climbed, and cannot make thrust")
    check("and the same episode with thrust goes to the top",
          rung_reached("air", {**glider, "sink_rate": -1.0,
                               "thrust_margin": 2.0}) == len(LADDER["air"]),
          "thrust is what separates the two")
    # The probe values are derived from the rungs rather than written out, so
    # re-deriving the thresholds from a new population moves the expectations
    # instead of breaking the test -- which is what happened when arch38's
    # collapsed distribution moved these bars from 0.0/0.05/0.15 down to
    # 0.002/0.010/0.020.
    THR_BARS = [t for _n, m, t in LADDER["air"] if m == "thrust_margin"]
    probes = [THR_BARS[0] - 0.1] + [b + (THR_BARS[i + 1] - b) / 2
                                    if i + 1 < len(THR_BARS) else b * 2.0
                                    for i, b in enumerate(THR_BARS)]
    check("the thrust rungs are ordered and sit above holds_height",
          [rung_reached("air", {**glider, "thrust_margin": v}) for v in probes]
          == [LIFT + 4, LIFT + 5, LIFT + 6, LIFT + 4 + THR + 1],
          "margins " + " / ".join(f"{v:+.4f}" for v in probes) + " -> rungs "
          + " ".join(str(rung_reached("air", {**glider, "thrust_margin": v}))
                     for v in probes))

    # An archived elite has to carry what it takes to reproduce its own numbers.
    #
    # arch38 kept the genome, the elite's 168-weight policy and the shared
    # network's periodic snapshots -- the machine survived the run -- and did not
    # keep the per-candidate evaluation seed, which `scatter` draws the whole
    # initial condition from.  So `max_depth` re-measured 9.51-9.69 m against an
    # archived 9.60 (robust to the draw) while `takeoff_height` re-measured 0.000
    # against an archived 2.288, and nothing on disk said which draw produced it.
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build as _build
    from dytiscidae.envs import batchroll as _br
    from dytiscidae.envs.evaluate import evaluate_tier1 as _t1
    from dytiscidae.envs.triphibian import MissionSpec as _Spec

    _p = _build(BODY_PLANS["gannet"]())
    _r1 = _t1(_p, spec=_Spec(), segment_seconds=1.0, seed=4242)
    _rb = _br.evaluate_tier1_batch([_p], spec=_Spec(), segment_seconds=1.0,
                                   seed=777)
    check("a score carries the seed it was produced under",
          getattr(_r1, "eval_seed", None) == 4242
          and getattr(_rb[0], "eval_seed", None) == 777,
          f"single {getattr(_r1, 'eval_seed', None)}, "
          f"batched {getattr(_rb[0], 'eval_seed', None)}")

    # The transition ladder put 70.6% of arch37 on rung 1 and 0.3% above it,
    # because completeness was demanded before quality and two of the quality
    # rungs were set at or above the population's own maximum -- `arrives_usable`
    # asked for `exit_state >= 0.7` against a largest-ever-measured 0.689.
    t_ok = {"crossed_fraction": 0.5, "control": 0.25, "exit_state": 0.40,
            "economy": 0.75, "shock": 0.55}
    check("crossing two boundaries well now outscores crossing two badly",
          rung_reached("transition", t_ok)
          > rung_reached("transition", {**t_ok, "control": 0.0}),
          f"rung {rung_reached('transition', t_ok)} against "
          f"{rung_reached('transition', {**t_ok, 'control': 0.0})}, "
          f"same two crossings")
    check("and every transition rung is reachable",
          all(rung_reached("transition", {**t_ok, "crossed_fraction": 1.0,
                                          "exit_state": 0.69, "control": 0.68,
                                          "shock": 1.0, "economy": 1.0}) ==
              len(LADDER["transition"]) for _ in (0,)),
          "the best values arch37 ever recorded clear the whole ladder")
    check("the transition ladder is monotone in what it asks",
          [rung_reached("transition", {"crossed_fraction": c, "control": 0.0,
                                       "exit_state": 0.0, "economy": 0.0,
                                       "shock": 0.0})
           for c in (0.25, 0.5)] == [0, 1],
          "two of four is rung 1; one of four is rung 0")

    # The water ladder had the same defect the air one had, and it outlived
    # the fix by three runs: `submerges` at 0.5 m and `dives` at 3.0 m read
    # `max_depth`, while `SPAWN[Domain.WATER]` releases the machine four metres
    # under.  Over arch37's 14,058 water segments the **minimum** `max_depth` is
    # 3.34 m, so both rungs were cleared by 100% of every evaluation ever run --
    # by being dropped.  The ladder now reads the gain over the release depth,
    # the way `takeoff_height` reads a gain over resting clearance.
    GAIN = sum(1 for _n, m, _t in LADDER["water"] if m == "depth_gain")
    deep_enough = {"hold_error": 0.5, "cruise_score": 1.0}
    check("a machine that stays where it was dropped scores nothing in water",
          rung_reached("water", {**deep_enough, "max_depth": 4.0,
                                 "depth_gain": 0.0}) == 0,
          "four metres of depth, none of it earned")
    check("and one that only ever rises scores nothing either",
          rung_reached("water", {**deep_enough, "max_depth": 3.4,
                                 "depth_gain": 0.0}) == 0,
          "the gain is floored at zero, so rising is rung 0, not below it")
    check("while one that dives past the release depth climbs the ladder",
          rung_reached("water", {**deep_enough, "max_depth": 13.0,
                                 "depth_gain": 9.0}) == len(LADDER["water"]),
          f"a 9 m gain clears all {GAIN} depth rungs and the two above them")
    check("and the depth rungs are ordered",
          [rung_reached("water", {**deep_enough, "depth_gain": g})
           for g in (0.1, 1.0, 3.0, 6.0)] == [0, 1, 2, 3],
          "gains 0.1 / 1 / 3 / 6 m -> rungs "
          + " ".join(str(rung_reached("water", {**deep_enough, "depth_gain": g}))
                     for g in (0.1, 1.0, 3.0, 6.0)))

    # The within-rung bonus must never reach the next rung's score.
    # `pushes`, not `flies`: without a thrust margin both of these stop at the
    # same rung and the check ties itself, which is not a test of anything.
    below = j.score("air", {**pushes, "airborne_fraction": 0.95, "sink_rate": -1.0,
                            "station_keeping": 0.8, "turn_response": 0.0})
    top = j.score("air", {**pushes, "airborne_fraction": 0.95, "sink_rate": -1.0,
                          "station_keeping": 0.8, "turn_response": 0.9})
    check("clearing six rungs never ties with clearing seven",
          below["total"] < top["total"], f"{below['total']:.3f} < {top['total']:.3f}")

    # The bar tightens as the population improves.
    rng = np.random.default_rng(0)
    bars = []
    for gen in range(1, 7):
        for _ in range(60):
            j.observe({"air": {"sink_rate": float(rng.normal(6.0 - gen * 0.9, 0.6))}})
        j.maybe_tighten(gen)
        bars.append(j.ratchets["air"].bar)
    check("the bar follows a population that is improving", bars[-1] < bars[0] - 2.0,
          f"{bars[0]:+.2f} -> {bars[-1]:+.2f} m/s of sink for full marks")

    # And never loosens, however bad the population gets.
    before = j.ratchets["air"].bar
    for _ in range(300):
        j.observe({"air": {"sink_rate": 9.0}})
    j.maybe_tighten(7)
    check("and never loosens", j.ratchets["air"].bar == before,
          f"{before:+.2f} unchanged after 300 terrible samples")

    # The same achievement scores lower once the bar has moved: that is the
    # point, and the rung is what stays comparable.
    fresh = Judge(quantile=0.9, update_every=1)
    # Carries a lift margin like the fixtures above: without one this lands on
    # rung 0 and the "rung is unchanged" check below compares two zeros, which
    # passes while testing nothing.
    m = {**flies, "airborne_fraction": 0.9, "sink_rate": 1.0}
    early = fresh.score("air", m)
    for _ in range(200):
        fresh.observe({"air": {"sink_rate": float(rng.normal(0.2, 0.3))}})
    fresh.maybe_tighten(1)
    late = fresh.score("air", m)
    check("the same design scores lower after a breakthrough", late["within"] < early["within"],
          f"within {early['within']:.2f} -> {late['within']:.2f}")
    check("while its rung is unchanged, so the record stays comparable",
          late["rung"] == early["rung"], f"rung {early['rung']} both times")

    # Rollback restores the previous bar exactly.
    b = j.ratchets["air"]
    prev = b.history[-1]["from"] if b.history else None
    check("a tightening can be rolled back", b.rollback() and b.bar == prev,
          f"rolled back to {b.bar:+.2f}")


def test_auditor_can_invalidate_and_veto() -> None:
    """The third party must be able to act, and must never be able to reward.

    There are two adaptive parties in this loop -- the population trying to
    score and the judge answering with a higher bar -- and nothing inside that
    pair can tell genuine progress from the two of them drifting together into a
    corner of the simulator.  The auditor's authority comes from one property:
    it does not learn, and nothing it checks is a function of the run's history.
    """
    print("\nauditor: audits, vetoes, never rewards")
    from dytiscidae.evolution.auditor import Auditor, check_scaling
    from dytiscidae.evolution.judge import Judge

    class Ph:
        def __init__(self, mass, area):
            self.mass, self.wing_area = mass, area

    class Res:
        def __init__(self, mf):
            self.mission_fraction = mf
            self.segments = {}

    # The wingless "flyer" that scored 0.75 must be noticed.
    f = check_scaling(Ph(5.28, 0.0))
    check("a machine with no lifting surface is flagged", f is not None and "no lifting" in f.detail,
          f.detail if f else "not flagged")
    check("and a plausible one is not", check_scaling(Ph(7.28, 0.687)) is None)

    a = Auditor(held_out_seeds=1, perturbations=(("cd_scale", 1.25),))

    # A design that only works at one exact coefficient value is invalidated.
    def brittle(seed=0, perturb=None):
        return Res(0.02 if perturb else 0.80)

    rep = a.audit(Ph(5.0, 0.5), Res(0.80), reevaluate=brittle, name="brittle")
    check("a design that collapses under a perturbed coefficient is invalidated",
          rep.invalid, "; ".join(x.detail for x in rep.findings) or "not invalidated")

    # A robust design survives.
    def robust(seed=0, perturb=None):
        return Res(0.72 if perturb else 0.80)

    rep2 = a.audit(Ph(5.0, 0.5), Res(0.80), reevaluate=robust, name="robust")
    check("a design that degrades gracefully is not", not rep2.invalid,
          f"retained {rep2.retained_fraction:.0%}")
    check("and the audit never raises a score", not hasattr(rep2, "bonus"))

    # The veto rolls the judge back.
    j = Judge(quantile=0.9, update_every=1)
    for _ in range(200):
        j.observe({"water": {"depth_gain": 3.0}})
    moves = j.maybe_tighten(1)
    raised = j.ratchets["water"].bar
    check("the judge tightened", moves and raised > 0.0, f"bar now {raised:.2f} m")
    vetoed = a.review_tightening(j, moves, invalid_designs=1)
    check("and the auditor can veto that tightening",
          vetoed and j.ratchets["water"].bar < raised,
          f"bar rolled back to {j.ratchets['water'].bar:.2f} m")
    check("a veto with nothing invalid does nothing",
          not a.review_tightening(j, moves, invalid_designs=0))

    # And the veto has to be *causal*.  It used to take a count of failed
    # audits and roll back the most recent tightening in every domain, so one
    # invalid design anywhere undid the air, water, land and transition bars
    # together -- bars set by hundreds of samples it had no part in.  Audits
    # find something most rounds; a ratchet reset most rounds never rises, and
    # the judge would have stopped getting stricter while still reporting that
    # it had.
    def fresh_judge():
        jj = Judge(quantile=0.9, update_every=1)
        for _ in range(200):
            jj.observe({"water": {"depth_gain": 3.0}, "air": {"sink_rate": 4.0}})
        mv = jj.maybe_tighten(1)
        return jj, mv

    j2, mv2 = fresh_judge()
    water_bar, air_bar = j2.ratchets["water"].bar, j2.ratchets["air"].bar
    check("two domains tightened together", water_bar > 0.0 and mv2,
          f"water {water_bar:.2f} m, air {air_bar:.2f} m/s")
    # A design invalidated on a shallow dive cannot have set the deep-water bar.
    shallow = [{"water": {"depth_gain": 0.4}, "air": {"sink_rate": 9.0}}]
    vetoed2 = a.review_tightening(j2, mv2, shallow)
    check("a design that never reached the bar cannot pull it down",
          not vetoed2
          and j2.ratchets["water"].bar == water_bar
          and j2.ratchets["air"].bar == air_bar,
          f"water still {j2.ratchets['water'].bar:.2f} m, "
          f"air still {j2.ratchets['air'].bar:.2f} m/s")

    j3, mv3 = fresh_judge()
    w3 = j3.ratchets["water"].bar
    # One that dove past the bar is in the tail the quantile came from, so its
    # invalidation is a reason to undo the water bar -- and only that one.
    deep = [{"water": {"depth_gain": 9.0}, "air": {"sink_rate": 9.0}}]
    a3_air = j3.ratchets["air"].bar
    vetoed3 = a.review_tightening(j3, mv3, deep)
    check("but one that set it does",
          vetoed3 and j3.ratchets["water"].bar < w3,
          f"water rolled back to {j3.ratchets['water'].bar:.2f} m")
    check("and the domains it had nothing to do with are left alone",
          j3.ratchets["air"].bar == a3_air,
          f"air still {j3.ratchets['air'].bar:.2f} m/s "
          f"(sink of 9.0 never met a {a3_air:.2f} bar)")


def test_critic_learns_the_exploit_signature() -> None:
    """The critic must learn what cheap evaluation misses, and may only subtract.

    The population is scored on Tier 1 because Tier 1 is cheap, and Tier 1 is
    cheap because it is a proxy.  Every proxy has a gap, and the history of this
    project is that gap being found: a battery drained on the first step to
    truncate an episode, a hillside skimmed to fake sustained flight, a
    coefficient the design silently depended on.  Each was caught by something
    expensive, and each was caught only after hundreds of generations of
    exploitation, because the expensive checks cannot run on everything.

    So the critic is trained to predict what the expensive check would have
    said, from the cheap measurements alone.  That makes the loop adversarial in
    the useful direction: a new way of looking good cheaply and failing
    expensively becomes training data, and the route closes.
    """
    print("\ncritic: learns the gap between cheap and expensive")
    from dytiscidae.evolution.critic import CRITIC_FEATURES, Critic

    rng = np.random.default_rng(0)
    c = Critic(min_samples=60, refit_every=20)

    def make(kind):
        f = np.zeros(len(CRITIC_FEATURES))
        if kind == "honest":
            f[0] = rng.uniform(0.2, 0.6)
            f[8] = rng.uniform(0.3, 2.0)
            f[11] = np.log10(rng.uniform(60, 300))
            f[12] = rng.uniform(3, 12)
            return f, rng.uniform(0.7, 1.0)
        # The wingless / battery-death family: looks the same cheaply.
        f[0] = rng.uniform(0.3, 0.7)
        f[8] = rng.uniform(-1.0, -0.8)
        f[11] = np.log10(rng.uniform(3000, 500000))
        f[12] = rng.uniform(0.0, 0.3)
        return f, rng.uniform(0.0, 0.15)

    check("an unfitted critic abstains entirely",
          c.discount(make("exploit")[0]) == 1.0 and c.predict(np.zeros(16)) == 1.0)

    for i in range(400):
        f, retained = make("honest" if i % 2 else "exploit")
        c.label(f, retained)
        if c.due():
            c.fit()

    check("the critic fits and is calibrated", c.fitted and c.calibration > 0.5,
          f"calibration {c.calibration:.2f} over {len(c._x)} labels")

    honest = np.mean([c.predict(make("honest")[0]) for _ in range(100)])
    exploit = np.mean([c.predict(make("exploit")[0]) for _ in range(100)])
    check("it separates the two families", honest > exploit + 0.3,
          f"predicts {honest:.2f} retention for honest, {exploit:.2f} for exploits")

    d_honest = np.mean([c.discount(make("honest")[0]) for _ in range(100)])
    d_exploit = np.mean([c.discount(make("exploit")[0]) for _ in range(100)])
    check("and discounts the exploits harder", d_exploit < d_honest,
          f"x{d_exploit:.3f} against x{d_honest:.3f}")

    # It found the signature on its own, and can say what it found.
    names = [d["feature"] for d in c.distrusts(top=2)]
    check("it can report what it learned to distrust", "log_wing_loading" in names,
          ", ".join(names))

    # The two invariants that stop it running away.
    worst = min(c.discount(make("exploit")[0]) for _ in range(200))
    check("the discount is bounded", worst >= 1.0 - c.max_discount - 1e-9,
          f"worst multiplier x{worst:.3f}, bound x{1 - c.max_discount:.2f}")
    best = max(c.discount(make("honest")[0]) for _ in range(200))
    check("and it can never raise a score", best <= 1.0 + 1e-9, f"best multiplier x{best:.3f}")

    # A critic that has stopped predicting anything must stop mattering.
    blind = Critic(min_samples=30, refit_every=10)
    for _ in range(200):
        blind.label(rng.normal(0, 1, len(CRITIC_FEATURES)), float(rng.uniform(0, 1)))
    blind.fit()
    check("a critic with no signal has no influence",
          blind.calibration < 0.35 or blind.discount(np.zeros(16)) > 0.9,
          f"calibration {blind.calibration:.2f}, discount x{blind.discount(np.zeros(16)):.3f}")


def test_one_islands_archive_is_read_alone_not_through_the_merge() -> None:
    """``showcase --island air`` must choose among the air island's elites.

    It filtered the *merged* archive, and the merge keeps one occupant per cell
    across all islands -- so an air elite that lost its cell to another
    island's occupant was dropped before the filter ever ran.  On arch39's
    generation-160 checkpoint the air archive held 78 elites and the filter
    saw 34, and the best flier was not one of them.  Built here as the smallest
    case that loses: two islands, one shared cell, the air copy scoring lower.
    """
    print("\nops: one island's archive is read alone, not through the merge")
    import shutil
    import tempfile

    from dytiscidae.evolution.archive import Archive, Elite
    from dytiscidae.ops.run import load_run_archive

    axes = [("a", 0.0, 1.0, 4), ("b", 0.0, 1.0, 4)]
    tmp = Path(tempfile.mkdtemp(prefix="dyt-island-archive-"))
    try:
        air, land = Archive(axes), Archive(axes)
        shared, only_air = (1, 1), (2, 3)
        air.cells[shared] = Elite(genome="air-flier", fitness=0.30,
                                  descriptor=np.zeros(2), cell=shared)
        air.cells[only_air] = Elite(genome="air-other", fitness=0.20,
                                    descriptor=np.zeros(2), cell=only_air)
        land.cells[shared] = Elite(genome="land-walker", fitness=0.90,
                                   descriptor=np.zeros(2), cell=shared)
        air.save(tmp / "archive_air.pkl")
        land.save(tmp / "archive_land.pkl")

        merged, _ = load_run_archive(tmp)
        via_filter = [e.genome for e in merged.cells.values()
                      if (e.meta or {}).get("island") == "air"]
        check("the merge still keeps the better occupant of a shared cell",
              merged.cells[shared].genome == "land-walker")
        check("which is why filtering the merge loses an island's elite",
              "air-flier" not in via_filter, f"filter saw {via_filter}")

        alone, names = load_run_archive(tmp, island="air")
        genomes = sorted(e.genome for e in alone.cells.values())
        check("read alone, the island keeps every elite it filed",
              genomes == ["air-flier", "air-other"] and names == ["air"],
              f"{genomes}")
        check("and each is tagged with the island it was read from",
              all(e.meta.get("island") == "air" for e in alone.cells.values()))
        miss, have = load_run_archive(tmp, island="water")
        check("an island the run does not have is reported with the ones it does",
              miss is None and sorted(have) == ["air", "land"], f"{have}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_specialist_islands_curriculum_reads_only_its_own_medium() -> None:
    """A pure-habitat island must not select on another medium.

    Every curriculum stage read all three media, and ``run_search`` never told
    an island's curriculum which island it served.  Measured at arch39's
    generation-180 checkpoint: 96.4% of the air island's elites and 96.4% of
    the land island's were better in another medium than their own, nearly all
    in water, and the air island's stage-0 top ten were all water machines with
    air competence 0.00-0.03.  Water was the medium that won because a machine
    with its actuators held still scores 0.533 there -- it sinks, and depth
    gain, hold, submerged and upright all pay for sinking.
    """
    print("\ncurriculum: a specialist island reads only its own medium")
    from types import SimpleNamespace as NS

    from dytiscidae.evolution.curriculum import Curriculum, stage_score
    from dytiscidae.evolution.islands import ISLANDS, curriculum_for

    def seg(c, **m):
        return NS(competence=c, measurements=m)
    # Good at sitting underwater, poor in the air: the measured shape.
    res = NS(mission_fraction=0.01, segments={
        # airborne, as every long-exit air segment publishes (stage 1 counts air
        # progress only while airborne, ROADMAP AE)
        "air": seg(0.03, sink_rate=2.4, cruise_progress=0.3, airborne_fraction=1.0),
        "water": seg(0.90, depth_error=0.5, cruise_progress=1.0),
        "land": seg(0.05, land_speed=0.02)})

    class TS:
        def __init__(self, results):
            self.results = results
        def component_means(self):
            keys = next(iter(self.results.values())).components.keys()
            return {k: float(np.mean([r.components[k] for r in self.results.values()]))
                    for k in keys}
    good = dict(crossed=1.0, shock=1.0, control=1.0, settle=1.0, exit_state=1.0, economy=1.0)
    bad = {k: 0.0 for k in good}
    ts = TS({"water_to_land": NS(components=good), "air_to_water": NS(components=bad),
             "water_to_air": NS(components=bad), "land_to_air": NS(components=bad)})

    air = dict(domains=("air",), transition_names=tuple(ISLANDS["air"]["transitions"]))
    check("unrestricted, stage 0 pays the air island for water",
          stage_score(0, res) == 0.90)
    check("restricted, stage 0 reads air and nothing else",
          stage_score(0, res, **air) == 0.03, f"{stage_score(0, res, **air)}")
    check("stage 1 reads only the island's own directed measure",
          abs(stage_score(1, res, **air) - 0.3) < 1e-12
          and stage_score(1, res) == 1.0,          # water's tracking is full marks
          f"{stage_score(1, res, **air):.3f} (unrestricted {stage_score(1, res):.3f})")
    # The air island's stage 1 must read *going where it was told*, not
    # *losing height slowly*: a machine falling from the 30 m launch has a sink
    # rate a glide would be proud of and no heading it chose, and paying it
    # here is the defect `sink_reduction` was introduced to close.
    falling = NS(mission_fraction=0.0, segments={
        "air": seg(0.0, sink_rate=1.2, station_keeping=0.0, cruise_progress=0.0)})
    check("a machine that only falls opens nothing at stage 1",
          stage_score(1, falling, **air) == 0.0,
          f"{stage_score(1, falling, **air):.3f} on sink_rate 1.2, tracking 0.0")
    check("stage 2 reads only the island's own crossings",
          stage_score(2, res, ts, **air) == 0.0 and stage_score(2, res, ts) > 0.0,
          f"own {stage_score(2, res, ts, **air)}, all {stage_score(2, res, ts):.3f}")
    check("stage 3 pairs a single-medium island's medium with itself",
          abs(stage_score(3, res, ts, **air) - 0.03 * 0.1) < 1e-12,
          f"{stage_score(3, res, ts, **air)}")

    gen = dict(domains=tuple(ISLANDS["generalist"]["domains"]),
               transition_names=tuple(ISLANDS["generalist"]["transitions"]))
    check("the generalist's own set is every medium, so it scores exactly as before",
          all(stage_score(k, res, ts, **gen) == stage_score(k, res, ts)
              for k in range(5)))

    cell = (0, 0)
    sr = curriculum_for("air").evaluate(cell, res, ts)
    check("run_search's constructor hands the air island an air curriculum",
          sr.detail["here"] == 0.03, f"{sr.detail}")
    old = Curriculum()
    for k in ("domains", "transition_names"):
        old.__dict__.pop(k, None)          # as unpickled from an earlier run
    check("a curriculum pickled before the fix keeps reading every medium",
          old.evaluate(cell, res, ts).detail["here"] == 0.90)


def test_an_islands_best_is_judged_on_its_own_domains() -> None:
    """"The air island's best design" must be a machine that flies.

    Measured at arch39's generation-160 checkpoint: the air island's fitness
    champion scored air 0.033 and land 0.744, because stage 0 of the curriculum
    reads the best medium anywhere, while an elite with air 0.768 sat in the
    same archive.  ``own_domain_score`` is what ``showcase --by island`` and
    the per-island report rank by, so it is pinned on that exact pair.
    """
    print("\nislands: an island's best is judged on its own domains")
    from dytiscidae.evolution.islands import own_domain_score

    lander = {"air": 0.033, "water": 0.181, "land": 0.744, "mission_fraction": 0.0321}
    flier = {"air": 0.768, "water": 0.695, "land": 0.000, "mission_fraction": 0.0}
    check("on the air island the flier beats the land specialist",
          own_domain_score("air", flier) > own_domain_score("air", lander),
          f"{own_domain_score('air', flier):.3f} vs {own_domain_score('air', lander):.3f}")
    check("and on the land island the order reverses",
          own_domain_score("land", lander) > own_domain_score("land", flier))

    # A two-domain island needs both: its weakest domain gates it, exactly as
    # island_score's base term does.  Reading the best one instead is the
    # defect that makes a specialist win a pairing island.
    one_sided = {"water": 0.95, "land": 0.0}
    both = {"water": 0.5, "land": 0.5}
    check("an amphibian that cannot walk scores nothing on the amphibian island",
          own_domain_score("amphibian", one_sided) == 0.0
          and own_domain_score("amphibian", both) > 0.0,
          f"one-sided {own_domain_score('amphibian', one_sided):.3f}, "
          f"both {own_domain_score('amphibian', both):.3f}")
    check("the generalist island's own objective is the mission",
          own_domain_score("generalist", lander) == 0.0321)
    check("and missing competences read as zero rather than raising",
          own_domain_score("aerial_diver", {}) == 0.0)


def test_the_island_objective_takes_its_weight_back() -> None:
    """The blend between curriculum and island objective must not be a constant.

    It was a flat 0.5 below stage 4, and on the 500-generation runs that meant
    the *air* island was selecting almost entirely on a score that does not read
    air.  Measured: its champion had air competence 0.107, so the island
    objective -- competence**1.5 -- could contribute at most 0.035 against a
    stored fitness of 0.5019.  96% of the pressure came from the island-blind
    half, and the island filled with wingless water specialists.

    Islands are an early device: grow terrain-adapted genes fast, keep a dark
    horse alive long enough to show itself.  By the end the target is a
    triphibian again, so the island's own objective has to come back.
    """
    print("\ncurriculum: the island objective takes its weight back")
    from dytiscidae.evolution.curriculum import N_STAGES, Curriculum

    c = Curriculum()
    # The ramp now has a floor.  At ``stage/4`` alone the island half carried
    # *zero* weight for the 69.1% of arch31's evaluations that sat at stage 0,
    # so on the generalist island -- whose island objective is the mission
    # itself -- the mission contributed nothing at all to most of the run.  The
    # weighted-mean weight applied across the whole 600 generations was 0.1027.
    check("the blend still starts mostly on the curriculum, but not entirely",
          0.0 < c.handover(0) <= 0.3,
          f"stage 0 hands over {c.handover(0):.2f} of the weight")
    check("and the top of the ladder is the island's outright",
          c.handover(N_STAGES - 1) == 1.0, "stage 4 -> 1.0")
    check("the stage ramp is monotone",
          all(c.handover(s) <= c.handover(s + 1) for s in range(N_STAGES - 1)),
          " ".join(f"{c.handover(s):.2f}" for s in range(N_STAGES)))

    # The spread-based "evidence" term is gone.  It weighted whichever half
    # discriminated more, which silences a hard objective exactly when it
    # matters: mission_fraction has a small spread *because* nothing can do the
    # mission yet.  Over arch30 it averaged 0.197, below the 0.25 stage floor it
    # was meant to improve on, and selection ended up correlating with mission
    # capability at r = 0.1413.  So the ramp is now the whole rule.
    flat = Curriculum()
    for i in range(64):
        flat.observe_blend(0.035, 0.2 + 0.8 * (i % 8) / 7.0, 1)
    check("a flat island objective still gets its stage share, not less",
          abs(flat.handover(1) - 0.25) < 1e-9,
          f"stage-1 handover {flat.handover(1):.3f}")

    sharp = Curriculum()
    for i in range(64):
        sharp.observe_blend(0.05 + 0.9 * (i % 8) / 7.0, 0.98, 1)
    check("and a sharp one gets the same share: the ramp ignores spread",
          abs(sharp.handover(1) - flat.handover(1)) < 1e-9,
          f"flat {flat.handover(1):.3f} vs sharp {sharp.handover(1):.3f}")

    # The property that actually fixes the bug: the two halves reach the blend
    # on the same scale.  Raw, the island half spans 0.0437 p10-p90 against the
    # curriculum's 0.5092 -- an 11.7x mismatch that hands the decision to the
    # curriculum whatever weight is nominally applied.
    mixed = Curriculum()
    for i in range(64):
        frac = (i % 8) / 7.0
        mixed.observe_blend(0.02 + 0.04 * frac, 0.2 + 0.9 * frac, 2)
    lo_i, lo_c = mixed.standing(0.021, 0.21, 2)
    hi_i, hi_c = mixed.standing(0.059, 1.09, 2)
    check("a tiny-scale island score still spans the standing range",
          (hi_i - lo_i) > 0.7,
          f"island standing {lo_i:.2f} -> {hi_i:.2f} over a raw span of 0.04")
    check("and the large-scale curriculum score spans no more than it",
          abs((hi_c - lo_c) - (hi_i - lo_i)) < 0.2,
          f"curriculum standing {lo_c:.2f} -> {hi_c:.2f} over a raw span of 0.9")

    # Before there is a population to rank against, both halves come back
    # neutral rather than raw.  Raw would put an unrankable design onto the same
    # axis as ranked ones at whatever scale it happens to have, which is the
    # mismatch this whole mechanism exists to remove; 0.5 says "cannot rank
    # this yet" and lets the other terms decide.
    young = Curriculum()
    young.observe_blend(0.03, 0.9, 0)
    check("with too little history neither half claims to rank",
          young.standing(0.5, 0.7, 0) == (0.5, 0.5),
          f"fewer than {young.min_rank_samples} observations -> neutral")

    # The windows are per stage.  Pooling them meant a stage-2 design, whose
    # score is a smaller quantity by construction, landed in the bottom
    # quantile for having been promoted: corr(fitness, stage) = -0.35.
    split = Curriculum()
    for i in range(64):
        split.observe_blend(0.9, 0.9, 0)          # an easy stage scores high
        split.observe_blend(0.05, 0.05, 2)        # a hard one scores low
    top_of_hard = split.standing(0.06, 0.06, 2)[1]
    same_in_easy = split.standing(0.06, 0.06, 0)[1]
    check("a design is ranked against its own stage, not against an easier one",
          top_of_hard > 0.9 and same_in_easy < 0.1,
          f"0.06 ranks {top_of_hard:.2f} at stage 2 and {same_in_easy:.2f} at stage 0")

    # And the mission arrives on the same scale as the other two, because a
    # blend of raw magnitudes hands the decision to whichever is largest.
    for i in range(64):
        split.observe_blend(0.5, 0.5, 1, mission=0.001 * i)
    check("the mission enters the blend as a standing too",
          split.mission_standing(0.062, 1) > 0.9
          and split.mission_standing(0.001, 1) < 0.2,
          f"mission 0.062 -> {split.mission_standing(0.062, 1):.2f}, "
          f"0.001 -> {split.mission_standing(0.001, 1):.2f}")

    check("the blend is in the record, since it moves",
          {"handover_floor", "handover_typical", "handover_mean", "rank_windows"}
          <= set(Curriculum(stages={(0, 0, 0, 0): 1}).report()),
          "report() carries the weight actually applied, not a field nothing writes")


def test_curriculum_and_islands_give_gradient_where_the_mission_gives_none() -> None:
    """A design that is good at one thing must be distinguishable from a design
    that is good at nothing.

    Mission fraction is built on ``min(competences)`` times a transition term,
    so a superb water specialist that cannot leave the water scores essentially
    zero -- the same essentially zero as a design that is bad at everything.
    Between those two there is a gradient that matters enormously and the score
    cannot see it.  That is the sparse-reward trap, and it is also, in biology,
    why there are almost no triphibian animals: the intermediate is worse than
    either specialist, so nothing is ever carried across the valley.
    """
    print("\ncurriculum and islands: gradient where the mission has none")
    from dytiscidae.evolution.curriculum import (
        N_STAGES,
        STAGES,
        Curriculum,
        stage_score,
    )
    from dytiscidae.evolution.islands import ISLANDS, Archipelago, island_score

    class Seg:
        def __init__(self, c, meas=None):
            self.competence = c
            self.measurements = meas or {}

    class Trans:
        def __init__(self, crossed, quality):
            self._c, self._q = crossed, quality
            self.results = {}

        def component_means(self):
            return {"crossed": self._c, "shock": self._q, "control": self._q,
                    "settle": self._q, "economy": self._q, "exit_state": self._q}

    class Res:
        def __init__(self, air, water, land, mf, meas=None):
            self.segments = {"air": Seg(air, (meas or {}).get("air")),
                             "water": Seg(water, (meas or {}).get("water")),
                             "land": Seg(land, (meas or {}).get("land"))}
            self.mission_fraction = mf

    # A water specialist and a uniform failure both score ~0 on the mission.
    # Stage 1 reads the cruise phase's tracking, so a water specialist states
    # what it does in water as going where it was told rather than as depth
    # reached -- sinking is not a capability and does not open the stage.
    specialist = Res(0.02, 0.85, 0.03, 0.001,
                     {"water": {"depth_error": 0.8, "max_depth": 10.0,
                                "depth_gain": 6.0, "cruise_progress": 1.0}})
    useless = Res(0.02, 0.03, 0.03, 0.001)
    t = Trans(0.34, 0.4)

    check("the mission cannot tell them apart",
          abs(specialist.mission_fraction - useless.mission_fraction) < 1e-6,
          f"both {specialist.mission_fraction:.4f}")
    check("stage 0 can", stage_score(0, specialist, t) > 3 * stage_score(0, useless, t),
          f"{stage_score(0, specialist, t):.3f} against {stage_score(0, useless, t):.3f}")
    check("and so can the water island",
          island_score("water", specialist, t) > 3 * island_score("water", useless, t),
          f"{island_score('water', specialist, t):.3f} against "
          f"{island_score('water', useless, t):.3f}")
    check("while the generalist island still says what the mission says",
          abs(island_score("generalist", specialist, t) - specialist.mission_fraction) < 1e-9)

    # A specialist island must not punish giving up the other media.
    check("the water island ignores the domains it does not care about",
          island_score("water", specialist, t)
          > island_score("water", Res(0.9, 0.85, 0.9, 0.5), t) * 0.9,
          "a water specialist is not outscored on water by an all-rounder")

    # The curriculum promotes, and demotes when the bar it was earned under
    # is no longer met.
    c = Curriculum()
    cell = (1, 2, 3, 4)
    check("everything starts at stage 0", c.stage_of(cell) == 0)
    sr = c.evaluate(cell, specialist, t)
    check("a design is scored at its stage and shown the next one",
          "here" in sr.detail and "next" in sr.detail, str(sr.detail))
    check("and clearing the bar promotes it", sr.passed and c.update(cell, sr) == "promoted",
          f"stage {c.stage_of(cell)}")
    for _ in range(N_STAGES + 2):
        c.update(cell, c.evaluate(cell, specialist, t))
    check("promotion stops at the top", c.stage_of(cell) <= N_STAGES - 1,
          f"stage {c.stage_of(cell)} of {N_STAGES - 1}")
    collapsed = c.evaluate(cell, useless, Trans(0.0, 0.0))
    before = c.stage_of(cell)
    c.update(cell, collapsed)
    check("and a cell that can no longer earn its stage is demoted",
          c.stage_of(cell) < before, f"{before} -> {c.stage_of(cell)}")

    # What the run reports about the curriculum has to be what the archive
    # contains.  ``reached`` is a max, it was taken over every cell the
    # curriculum had ever seen including ones since removed, and it was the only
    # number the progress line printed -- so an archive of twenty cells at
    # "single" and one at "directed" was being reported as stage 2, and I read
    # it that way in my own summaries.
    c2 = Curriculum()
    for i in range(6):
        c2.stages[(i, 0, 0, 0)] = 0
    c2.stages[(9, 9, 9, 9)] = 3
    rep = c2.report()
    check("the report says where the archive typically is, not only its furthest",
          rep["typical"] == 0 and rep["reached"] == 3,
          f"typical {rep['typical']}, reached {rep['reached']}")
    c2.forget((9, 9, 9, 9))
    rep = c2.report()
    check("and a stage does not outlive the cell that earned it",
          rep["reached"] == 0, f"reached {rep['reached']} after the cell was dropped")

    # The curator is what removes cells, so it is what has to say so.
    from dytiscidae.evolution.archive import Archive as _Archive
    from dytiscidae.evolution.curator import Curator as _Curator

    a2 = _Archive([("x", 0.0, 1.0, 4), ("y", 0.0, 1.0, 4)])
    cur2 = _Curator(a2, seed=0)
    c3 = Curriculum()
    cur2.curriculum = c3
    c3.stages[(1, 1)] = 2
    cur2._forget((1, 1))
    check("dropping a cell drops its stage with it",
          c3.report()["reached"] == 0, "curator notifies the curriculum")

    # The archipelago moves genes between lineages, which biology cannot.
    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.curator import Curator

    arc = Archipelago(migrate_every=2, n_migrants=1)
    for name in ("air", "water", "generalist"):
        a = Archive([("x", 0.0, 1.0, 4), ("y", 0.0, 1.0, 4)])
        a.add(f"{name}_best", 0.9, [0.5, 0.5], objectives=np.array([0.9, 1.0, 1.0]))
        arc.register(name, a, Curator(a, seed=0))

    rng = np.random.default_rng(0)
    check("migration is periodic, not constant", not arc.due(1) and arc.due(2))
    moved = arc.migrate(2, rng, crossover=lambda a, b, r: f"({a}+{b})")
    check("designs move between islands", any(m["kind"] == "migrant" for m in moved),
          f"{sum(1 for m in moved if m['kind'] == 'migrant')} migrants")
    check("and specialists are crossed directly -- the move biology cannot make",
          any(m["kind"] == "hybrid" for m in moved),
          str([m["genome"] for m in moved if m["kind"] == "hybrid"][:1]))
    check("an immigrant is sent somewhere other than home",
          all(m["island"] != m["origin"] for m in moved if m["kind"] == "migrant"))


def test_a_run_can_be_picked_up_where_it_stopped() -> None:
    """A run interrupted halfway must continue, not start over.

    The environments this is meant to run in are reclaimed without warning, and
    a useful search here is a day long.  The archives were already being written
    every ``checkpoint_every`` generations -- and nothing ever read them back.
    So an interrupted run lost the judge's bars, the curriculum's stages, the
    critic, the scout, the operator bandit and the generation counter: it lost
    everything except the designs, and had no way to use even those.  The README
    promised resumption and the code had checkpointing, which is not the same
    thing.
    """
    if needs_batched_evaluator("test_a_run_can_be_picked_up_where_it_stopped"):
        return
    print("\nloop: a run resumes where it stopped")
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-resume-")
    try:
        base = dict(batch=1, seed=5, segment_seconds=1.0, n_reference_seeds=1,
                    n_random_seeds=0, islands=("water", "generalist"),
                    tier2_every=999, audit_every=999, migrate_every=1,
                    checkpoint_every=1, run_dir=tmp, identify_axes_every=999)

        first = run_search(SearchConfig(generations=3, **base), MissionSpec())
        n1 = sum(len(a.cells) for a in first.archipelago.archives.values())
        bars1 = {d: r.bar for d, r in first.judge.ratchets.items()}
        seen1 = first.evaluated
        check("a run leaves a checkpoint behind",
              (Path(tmp) / "search_state.pkl").exists(), "search_state.pkl written")

        again = run_search(SearchConfig(generations=6, resume=True, **base),
                           MissionSpec())
        n2 = sum(len(a.cells) for a in again.archipelago.archives.values())
        bars2 = {d: r.bar for d, r in again.judge.ratchets.items()}

        check("the archive comes back", n2 >= n1 and n1 > 0,
              f"{n1} elites before, {n2} after")
        check("and so does the evaluation count", again.evaluated >= seen1,
              f"{seen1} -> {again.evaluated}")
        check("the judge's bars are not reset by an interruption",
              all(bars2.get(d, -1.0) >= v - 1e-9 for d, v in bars1.items()),
              f"{ {k: round(v,2) for k,v in bars1.items()} } -> "
              f"{ {k: round(v,2) for k,v in bars2.items()} }")
        check("and the seeds are not evaluated a second time",
              again.evaluated - seen1 <= 3,
              f"{again.evaluated - seen1} new evaluations across 3 more generations")

        # Archives without a state file -- an older build, or a machine that
        # went away mid-write -- must still be recovered.  They are the
        # expensive part: every cell is an evaluation someone paid for.
        (Path(tmp) / "search_state.pkl").unlink()
        # Pretend the archives got a long way before the machine went away.  The
        # generation they carry is the only clue left, and the first version of
        # this recovery did not copy it across -- so a run recovered from
        # archives alone restarted near zero and redid everything.  A
        # two-generation fixture cannot tell 1 from 0, which is why this forces
        # a number that can.
        for f in sorted(Path(tmp).glob("archive_*.pkl")):
            a = Archive.load(f)
            a.generation = 30
            a.save(f)
        salvaged_gen = None
        cfg3 = SearchConfig(generations=32, resume=True, **base)
        salvaged = run_search(cfg3, MissionSpec())
        n3 = sum(len(a.cells) for a in salvaged.archipelago.archives.values())
        check("an archive with no state file beside it is still recovered",
              n3 >= n1, f"{n1} elites recovered without the checkpoint")
        check("and it resumes at the generation the archive records",
              all(a.generation >= 30
                  for a in salvaged.archipelago.archives.values() if a.cells),
              f"archives say generation "
              f"{ {n: a.generation for n, a in salvaged.archipelago.archives.items()} }")
        del salvaged_gen

        # A fresh directory with --resume is a normal run, not an error.
        fresh = tempfile.mkdtemp(prefix="dyt-resume-fresh-")
        try:
            s = run_search(SearchConfig(generations=1, resume=True,
                                        **{**base, "run_dir": fresh}), MissionSpec())
            check("resuming a run that does not exist just starts one",
                  sum(len(a.cells) for a in s.archipelago.archives.values()) >= 0)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_needs_a_nonzero_answer_to_the_next_question() -> None:
    """A cell climbs the ladder only once it has touched the next rung.

    Stage bars alone let a swimmer pass "directed" on depth-hold and be
    promoted into "crossing" without ever having crossed anything: arch30
    logged 764 promotions against 17 demotions while the typical cell sat at
    stage 1 and five cells ever reached "chain".  Zero on the next stage's
    question is not a bar to tune -- it is the difference between "has
    something to climb" and "was pushed off a cliff".
    """
    print("\ncurriculum: promotion needs a nonzero answer to the next question")
    from types import SimpleNamespace

    from dytiscidae.evolution.curriculum import Curriculum

    def result_water_specialist():
        # Stage 1 asks whether the machine is going where it was told, and in
        # water that is the cruise phase's tracking -- so the specialist has to
        # state what it does in those terms to pass the bar at all.  Depth reached does not open
        # the stage, because a machine denser than water reaches depth without
        # choosing to.
        seg = SimpleNamespace(
            competence=0.9,
            measurements={"depth_error": 0.5, "max_depth": 9.0,
                          "depth_gain": 5.0, "water_speed": 0.5,
                          "cruise_progress": 1.0},
        )
        return SimpleNamespace(segments={"water": seg}, mission_fraction=0.0)

    def transitions(crossed: float):
        comps = {k: (0.5 if crossed > 0 else 0.0)
                 for k in ("shock", "control", "settle", "economy",
                           "exit_state")}
        comps["crossed"] = crossed
        return SimpleNamespace(component_means=lambda: comps)

    cur = Curriculum()
    cell = (1, 2, 3, 4)
    cur.stages[cell] = 1

    sr = cur.evaluate(cell, result_water_specialist(), transitions(0.0))
    check("the stage bar itself is passed", sr.passed,
          f"here={sr.detail['here']} bar={sr.detail['bar']}")
    check("but a cell that never crossed is held, not promoted",
          cur.update(cell, sr) == "held" and cur.stage_of(cell) == 1)

    sr = cur.evaluate(cell, result_water_specialist(), transitions(0.5))
    check("one real crossing, however rough, earns the promotion",
          cur.update(cell, sr) == "promoted" and cur.stage_of(cell) == 2,
          f"next={sr.detail['next']}")


def test_the_headline_is_the_mission() -> None:
    """The generation report carries the generalist's mission_fraction.

    ``best_fitness`` is whichever island's champion scored highest -- for ten
    runs in a row that was the water island's wingless specialist, and reading
    it as the run's headline hid that mission_fraction never moved.
    """
    print("\nloop: the headline is the mission")
    import json
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-headline-")
    try:
        run_search(SearchConfig(
            generations=1, batch=1, seed=11, segment_seconds=1.0,
            n_reference_seeds=1, n_random_seeds=0, islands=("generalist",),
            tier2_every=999, audit_every=999, migrate_every=999,
            checkpoint_every=999, run_dir=tmp, identify_axes_every=999,
        ), MissionSpec())
        rows = [json.loads(l) for l in open(Path(tmp) / "generations.jsonl")]
        gen_rows = [r for r in rows if "mission_best" in r]
        check("the report carries mission_best", len(gen_rows) >= 1,
              f"{len(gen_rows)} rows carry it")
        check("and it is a fraction, not a fitness",
              all(0.0 <= r["mission_best"] <= 1.0 for r in gen_rows))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_audit_perturbs_the_scored_experiment_and_nothing_else() -> None:
    """The audit's ratio is the perturbation's, not a second experiment's.

    It re-ran every design with no controller at seed 0 and divided that by a
    score earned with the design's own policy plus the shared one at its own
    seed -- and since the task is drawn from the seed, under another task.
    arch40 invalidated 25 of 42 audits that way, held-out ratios up to 4.5.
    A perturbation of x1.0 changes nothing, so it must retain exactly 1.
    """
    print("\naudit: the perturbation is the only thing that differs")
    import shutil
    import tempfile

    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution import loop as loop_mod
    from dytiscidae.evolution.auditor import Auditor
    from dytiscidae.evolution.loop import SearchConfig, run_search

    tmp = tempfile.mkdtemp(prefix="dyt-audit-")
    try:
        state = run_search(SearchConfig(
            generations=1, batch=1, seed=11, segment_seconds=4.0,
            n_reference_seeds=1, n_random_seeds=0, islands=("generalist",),
            tier2_every=999, audit_every=999, migrate_every=999,
            checkpoint_every=999, run_dir=tmp, identify_axes_every=999,
        ), MissionSpec())
        check("the fixture leaves an elite to audit", bool(state.archive.cells))
        if not state.archive.cells:
            return
        # The perturbation only runs on a nonzero mission base, and a mission
        # is a needle: the drawn beetle's 0.0009 went to 0.0 when the fluid
        # model was corrected on 2026-09-23 (MATH_AUDIT F-09..F-11).  So the
        # audited elite is a gannet, whose base is nonzero at seed 0, and that
        # precondition is what the next check reads.
        from dytiscidae.core.bodyplans import BODY_PLANS
        for e in state.archive.cells.values():
            e.genome = BODY_PLANS["gannet"]()
            e.meta["policy"] = None
            e.meta["eval_seed"] = 0
        state.auditor = Auditor(held_out_seeds=0, perturbations=(("cd_scale", 1.0),))
        loop_mod._audit(state, 1, MissionSpec(), np.random.default_rng(0))
        rep = state.auditor.reports[-1]
        # Two checks need only the result; the third is the perturbation, which
        # runs only on a base above zero -- without it the 1.0 is a default.
        check("the perturbation check ran", rep.checks_run == 3, f"{rep.checks_run} checks")
        check("and a perturbation of x1.0 retains exactly what the scored experiment scored",
              abs(rep.retained_fraction - 1.0) < 1e-9 and not rep.invalid,
              f"retained {rep.retained_fraction:.4f}, invalid {rep.invalid}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_spends_refinement_and_keeps_what_it_buys() -> None:
    """Verification refines the elite's controller, and stores the result.

    ``controller_refine_steps`` defaults to zero because refining every
    candidate costs a full batched Tier-1 per step -- so the 500-generation
    runs verified designs against a controller nothing had ever optimised.
    Promotion is the affordable place: at most three per verification round,
    and the search has already decided the design is worth a Tier-2.  Keeping
    the refined weights on the elite is what makes the second payment worth
    anything to the archive rather than only to the critic.
    """
    if needs_batched_evaluator("test_promotion_spends_refinement_and_keeps_what_it_buys"):
        return
    print("\nloop: promotion spends refinement and keeps what it buys")
    import json
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-promote-")
    try:
        state = run_search(SearchConfig(
            generations=2, batch=2, seed=3, segment_seconds=0.5,
            n_reference_seeds=2, n_random_seeds=0, islands=("generalist",),
            tier2_every=1, audit_every=999, migrate_every=999,
            checkpoint_every=999, run_dir=tmp, identify_axes_every=999,
            promotion_refine_steps=2), MissionSpec())
        events = [json.loads(l) for l in open(Path(tmp) / "events.jsonl")]
        promotions = [e for e in events if e.get("kind") == "promote"]
        errors = [e for e in events if e.get("kind") == "error"]
        check("verification still promotes", len(promotions) >= 1,
              f"{len(promotions)} promotions")
        check("and refinement at promotion raises no errors", not errors,
              f"{[e.get('error') for e in errors[:2]]}")
        stored = [e for e in state.archive.cells.values()
                  if e.meta.get("policy")]
        check("elites carry policy weights forward",
              len(stored) == len(state.archive.cells) and stored,
              f"{len(stored)} of {len(state.archive.cells)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_shared_command_means_the_same_thing_on_every_body() -> None:
    """A mode index is a private coordinate; a body twist is not.

    Modes come out of an SVD, so their order is by authority and their sign is
    arbitrary, and what mode 0 physically *is* depends on the body.  Measured
    across twelve elites from arch31, the mean pairwise cosine between their
    water mode-0 directions was +0.094 -- mode 0 was yaw on four of them, heave
    on three, roll on two.  A policy shared across bodies and indexed by mode
    therefore commands unrelated things on different machines, and its gradients
    average toward nothing.  Commanding a twist and letting each body's basis
    invert for the coefficients puts every machine back in the same units.

    Two bases here describe the *same* physical machine with the modes permuted
    and sign-flipped, which is exactly what an SVD is free to do.
    """
    print("\ncpg: a shared command means the same thing on every body")
    from dytiscidae.control.cpg import MobilityBasis

    # Three clean axes: surge, heave, yaw.
    eff = np.zeros((3, 6))
    eff[0, 0] = 1.0     # surge
    eff[1, 2] = 1.0     # heave
    eff[2, 5] = 1.0     # yaw
    modes = np.eye(3, 7)
    a = MobilityBasis(modes=modes, effects=eff.copy(),
                      authority=np.array([3.0, 2.0, 1.0]), medium="water")
    # Same machine, different SVD: modes reordered and two signs flipped.
    perm = [2, 0, 1]
    b = MobilityBasis(modes=modes[perm], effects=-eff[perm],
                      authority=np.array([1.0, 3.0, 2.0])[perm], medium="water")

    heave = np.zeros(6)
    heave[2] = 1.0
    ta = a.twist_of(a.coeffs_for_twist(heave))
    tb = b.twist_of(b.coeffs_for_twist(heave))
    cos = float(np.dot(ta, tb) / max(np.linalg.norm(ta) * np.linalg.norm(tb), 1e-12))
    check("commanding heave gives the same motion through both bases",
          cos > 0.99, f"cosine {cos:+.4f}")
    check("and it really is heave", abs(ta[2]) > 10 * abs(ta[0]) + 1e-9,
          f"twist {np.round(ta, 3).tolist()}")

    # The same request indexed by mode does not survive the relabelling.
    ca = np.zeros(3); ca[1] = 1.0
    ma, mb = a.twist_of(ca), b.twist_of(ca)
    mode_cos = float(np.dot(ma, mb)
                     / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-12))
    check("where commanding mode 1 does not",
          mode_cos < 0.5, f"cosine {mode_cos:+.4f}")

    # An axis the body does not have must come back small, not enormous.
    roll = np.zeros(6); roll[3] = 1.0
    c_roll = a.coeffs_for_twist(roll)
    c_heave = a.coeffs_for_twist(heave)
    # Saturation must not spend everything the body has.
    from dytiscidae.control.cpg import INTENT_AUTHORITY
    full = a.twist_of(a.coeffs_for_twist(heave))[2]
    _inv, reach = a._inverse
    check("a saturated intent asks for a fraction of the body's reach, not all",
          abs(full - INTENT_AUTHORITY * reach[2]) < 0.05 * reach[2],
          f"delivered {full:.4f} of reach {reach[2]:.4f} "
          f"(authority {INTENT_AUTHORITY})")

    check("asking for an axis this body lacks returns almost nothing",
          np.linalg.norm(c_roll) < 0.05 * np.linalg.norm(c_heave),
          f"||c_roll|| {np.linalg.norm(c_roll):.2e} vs "
          f"||c_heave|| {np.linalg.norm(c_heave):.3f}")

    empty = MobilityBasis(modes=np.zeros((0, 7)), effects=np.zeros((0, 6)),
                          authority=np.zeros(0))
    check("a body with no identified axes commands nothing, and does not raise",
          empty.coeffs_for_twist(heave).shape == (0,))

    # A machine that cannot move has modes but no authority, so a ridge
    # proportional to the problem is zero and the solve is singular.  Such a
    # body is rare among the best elites and common in a random draw, which is
    # how this got through the first time.
    inert = MobilityBasis(modes=np.eye(3, 7), effects=eff.copy(),
                          authority=np.zeros(3), medium="water")
    got = inert.coeffs_for_twist(heave)
    check("a body with no authority commands nothing, and does not raise",
          got.shape == (3,) and np.allclose(got, 0.0), f"{got}")
    faint = MobilityBasis(modes=np.eye(3, 7), effects=eff.copy(),
                          authority=np.array([1e-9, 1e-10, 0.0]), medium="water")
    out = faint.coeffs_for_twist(heave)
    check("and a barely-mobile one does not answer with enormous coefficients",
          np.all(np.isfinite(out)) and np.linalg.norm(out) < 10.0,
          f"||c|| {np.linalg.norm(out):.3e}")


def test_the_identification_width_reaches_the_policy() -> None:
    """``n_modes`` must drive the identification, not merely the policy.

    They were two independent defaults that had to agree and nothing connected
    them: `identify_batch` took `max_modes=4` and never saw the config, so
    `--n-modes 6` raised `operands could not be broadcast together with shapes
    (4,) (6,)` in the middle of a rollout.  A config field that can only hold
    one value is worse than no field.
    """
    if needs_batched_evaluator("test_the_identification_width_reaches_the_policy"):
        return
    print("\nloop: the identification width follows the configured one")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution.loop import SearchConfig, evaluate_candidates

    for width in (4, 6):
        cfg = SearchConfig(segment_seconds=0.3, controller_refine_steps=0,
                           n_modes=width)
        out = evaluate_candidates([beetle()], cfg, identify=True,
                                  spec=MissionSpec(), seeds=[1])
        _p, result, _c = out[0]
        widths = {k: v.modes.shape[0] for k, v in result.mobility.items()}
        check(f"n_modes={width} identifies {width} modes",
              widths and all(w == width for w in widths.values()),
              f"{widths}")


def test_every_path_agrees_on_the_control_law() -> None:
    """Refinement and verification must see the policy they will be scored with.

    The two policies' intents are *summed* at the point of use, and only the
    batched evaluator knew that.  Refinement optimised the per-candidate half
    with the shared half absent and then stored the result to be scored with it
    present; Tier-2 ran the single-machine path, which takes one policy, so it
    verified a design under a control law that was not the one its Tier-1 score
    was earned with -- and the critic is trained on exactly that ratio, so the
    missing half was charged to the design.
    """
    print("\nloop: refinement and verification see the real control law")
    import inspect

    from dytiscidae.control.cpg import MobilityBasis
    from dytiscidae.envs.evaluate import SummedPolicy
    from dytiscidae.evolution import loop as loop_mod

    class Fixed:
        """Stands in for the shared policy: a twist, mean and sampled apart."""
        def __init__(self, mean, sampled):
            self.mean, self.sampled = mean, sampled

        def act(self, obs, deterministic=False):
            v = self.mean if deterministic else self.sampled
            return np.full(6, v), 0.0, 0.0

    class Own:
        def act(self, obs):
            return np.full(3, 0.25)

    # A basis with three clean, equally strong axes, so a commanded twist maps
    # back to coefficients that are easy to read.
    eff = np.zeros((3, 6))
    eff[0, 0] = eff[1, 1] = eff[2, 2] = 1.0
    basis = MobilityBasis(modes=np.eye(3, 7), effects=eff,
                          authority=np.ones(3), medium="air")

    summed = SummedPolicy(own=Own(), shared=Fixed(0.5, 99.0), n_modes=3,
                          basis=basis)
    got = summed.act(np.zeros(19))
    check("the summed policy adds both halves",
          float(np.min(got)) > 0.25 + 1e-9, f"{np.round(got, 4).tolist()}")
    check("and takes the shared half at its mean, never a sample",
          float(np.max(got)) < 1.0, f"max {float(np.max(got)):.3f}")

    only_own = SummedPolicy(own=Own(), shared=None, n_modes=3, basis=basis)
    check("with no shared policy it is just the candidate's own",
          np.allclose(only_own.act(np.zeros(19)), 0.25))

    # Without a basis the shared half cannot be honoured -- it commands a twist
    # and nothing can turn that into coefficients -- so it is dropped rather
    # than added raw, which would mean something different on every body.
    no_basis = SummedPolicy(own=Own(), shared=Fixed(0.5, 99.0), n_modes=3)
    check("and with no basis the shared half is dropped, not misread",
          np.allclose(no_basis.act(np.zeros(19)), 0.25),
          f"{np.round(no_basis.act(np.zeros(19)), 4).tolist()}")

    # The refinement path must thread the shared policy through, or it
    # optimises one half of a sum against the other half being zero.
    for fn in (loop_mod._refine_controllers, loop_mod.batchroll_eval):
        check(f"{fn.__name__} accepts the shared policy",
              "shared" in inspect.signature(fn).parameters,
              f"parameters: {list(inspect.signature(fn).parameters)}")
    src = inspect.getsource(loop_mod._refined_controller_for)
    check("promotion-time refinement passes it on",
          "shared=state.shared" in src)
    check("and Tier-2 verification is given the summed law",
          "_with_shared(state, ctrl2)"
          in inspect.getsource(loop_mod._verify_and_label))


def _nudged(fn, rel: float = 1e-15, commands: bool = False, seed: int = 20260926):
    """Run ``fn()`` with every machine's fluid forces dithered by ``rel``
    (relative, Gaussian, seeded) at every step: rounding-sized noise in the
    single path, used to measure its own noise floor.

    The two evaluation paths do the same arithmetic in a different order, so
    they differ by rounding: measured 2026-09-26 on the eel on land, the first
    difference is 2.5e-15 relative at step 2, and it grows smoothly to 1e-9 by
    step 131 -- contact-rich locomotion amplifies it about e^27 per second.  A
    bar between the paths set below what rounding alone does to the single path
    fails on noise the paths did not cause; one set against it still catches a
    path defect, which shows as tens of percent.  (A one-off 1e-12 m/s wind
    kick was tried first and is the wrong noise: a machine on land barely
    feels wind.)
    """
    import numpy as _np
    from dytiscidae.control.cpg import CPG
    from dytiscidae.physics.fluid import FluidSolver
    rng = _np.random.default_rng(seed)
    if commands:
        # Noise in what the controller commands rather than in the forces: where
        # a policy evaluated differently lands.
        orig_c = CPG.command

        def command(self, params, t):
            out = orig_c(self, params, t)
            return out * (1.0 + rel * rng.standard_normal(_np.shape(out)))
        CPG.command = command
        try:
            return fn()
        finally:
            CPG.command = orig_c
    orig = FluidSolver.apply

    def apply(self, data, t):
        out = orig(self, data, t)
        data.xfrc_applied[:] *= 1.0 + rel * rng.standard_normal(data.xfrc_applied.shape)
        return out
    FluidSolver.apply = apply
    try:
        return fn()
    finally:
        FluidSolver.apply = orig


def test_the_two_evaluation_paths_score_the_same_machine_the_same() -> None:
    """`rollout_batch` and `TriphibianEnv.rollout` are two implementations of one
    loop, kept in step by a comment that says so.

    They diverged by sixty times and nothing detected it.  `evaluate_tier1_batch`
    calls `env.scatter(rng)` after every reset and `evaluate_tier1` did not, so
    the single-machine path -- Tier-2 verification, every offline probe, and the
    showcase's own re-measurement -- ran a different experiment from the one whose
    number it was checking.  Median `land_speed` over arch38's twelve
    highest-mission elites was 0.649 m/s batched against 0.011 m/s
    single-machine, per-elite ratios from 1.7x to 438x, and the film that started
    the search for it showed a machine standing still.

    A comment cannot hold two implementations together.  This can.
    """
    if needs_batched_evaluator("test_the_two_evaluation_paths_score_the_same_machine_the_same"):
        return
    print("\nevaluation: the batched and single-machine paths agree")
    import numpy as np

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller, evaluate_tier1
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.control.cpg import Policy

    spec, seed, secs = MissionSpec(), 7, 2.0
    # Two plans rather than one: the defect was domain-specific -- it showed on
    # land and hid in water, where `max_depth` barely depends on the initial
    # condition -- so a single fixture could have agreed by luck.
    for plan in ("beetle", "eel"):
        p = build(BODY_PLANS[plan]())
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0)
        rng = np.random.default_rng(3)
        pol.weights = rng.normal(0.0, 0.2, pol.n_weights)

        rb = batchroll.evaluate_tier1_batch(
            [build(BODY_PLANS[plan]())], spec=spec,
            controllers=[Controller(params=None, policy=pol)],
            segment_seconds=secs, identify_axes=True, seed=seed)[0]
        rs = evaluate_tier1(p, spec=spec,
                            controller=Controller(params=None, policy=pol),
                            segment_seconds=secs, identify_axes=True, seed=seed)

        def worst_of(a_res, b_res, domains):
            worst, key, n = 0.0, "", 0
            for dom in domains:
                sb = (a_res.segments or {}).get(dom)
                ss = (b_res.segments or {}).get(dom)
                if sb is None or ss is None:
                    continue
                mb, ms = sb.measurements or {}, ss.measurements or {}
                for k in sorted(set(mb) & set(ms)):
                    a, b = mb[k], ms[k]
                    if not isinstance(a, (int, float)):
                        continue
                    if not isinstance(b, (int, float)):
                        continue
                    n += 1
                    if abs(a - b) > worst:
                        worst, key = abs(a - b), f"{dom}.{k} {a:.5f}/{b:.5f}"
            return worst, key, n

        # Land and air agree to floating-point summation order -- about 2e-6,
        # not bit-for-bit; a five-decimal print made it look exact and it is not.
        # 1e-5 is still five orders of magnitude below the defect this exists to
        # catch, which was 0.638 m/s of `land_speed`.
        w, key, n = worst_of(rb, rs, ("land", "air"))
        check(f"{plan}: land and air agree to floating-point between the paths",
              n > 6 and w < 1e-5,
              f"{n} measurements, worst absolute difference {w:.6f}"
              + (f" on {key}" if key else ""))

        # Water does not, and the cause is the identification rather than the
        # rollout: with `identify_axes=False` water agrees to the same 1e-5 the
        # other two do, and with it on only water moves.  `batchroll`
        # asserts in a comment that it reuses "`basis_from_probes` the unbatched
        # path uses, so the fitting is untouched" -- true of air, not of water,
        # where the solver carries slam and wake history that the two paths reset
        # differently (`bf.reset_slam()` against `solver.reset()`).  Bounded here
        # and recorded in the ROADMAP; the residual is millimetres of depth error
        # and centimetres per second of speed, well under the first water rung.
        w_w, key_w, n_w = worst_of(rb, rs, ("water",))
        check(f"{plan}: and water agrees to within the identification's residual",
              n_w > 2 and w_w < 0.05,
              f"{n_w} measurements, worst absolute difference {w_w:.5f}"
              + (f" on {key_w}" if key_w else ""))

        # The strong form, which is what makes the diagnosis above a measurement
        # rather than a story: take the identification out and water agrees too,
        # so the residual is the identification and not the rollout.
        rb0 = batchroll.evaluate_tier1_batch(
            [build(BODY_PLANS[plan]())], spec=spec,
            controllers=[Controller(params=None, policy=pol)],
            segment_seconds=secs, identify_axes=False, seed=seed)[0]
        rs0 = evaluate_tier1(build(BODY_PLANS[plan]()), spec=spec,
                             controller=Controller(params=None, policy=pol),
                             segment_seconds=secs, identify_axes=False, seed=seed)
        w0, key0, n0 = worst_of(rb0, rs0, ("land", "air", "water"))
        # As closely as the single path agrees with itself under the smallest
        # rounding-sized noise (`_nudged`), and never looser than the old 1e-5.
        # The largest of three dithered runs: one draw is a weak estimate of a
        # chaotic spread (measured 2026-09-26 on the eel, six seeds: 1.0e-4 to
        # 7.2e-4, with one draw reading 2.6e-4 and the paths 7.8e-4).
        floor = 0.0
        for ds in range(3):
            rs1 = _nudged(lambda: evaluate_tier1(
                build(BODY_PLANS[plan]()), spec=spec,
                controller=Controller(params=None, policy=pol),
                segment_seconds=secs, identify_axes=False, seed=seed), seed=1000 + ds)
            floor = max(floor, worst_of(rs0, rs1, ("land", "air", "water"))[0])
        bar = max(1e-5, 2.0 * floor)
        check(f"{plan}: without identification all three domains agree",
              n0 > 10 and w0 < bar,
              f"{n0} measurements, worst absolute difference {w0:.6f}"
              + (f" on {key0}" if key0 else "")
              + f"; bar {bar:.2e} (own noise floor {floor:.2e})")


def test_the_batched_path_tells_the_policy_it_is_wet() -> None:
    """The batched evaluator must fill the fluid diagnostics the policy reads.

    `BatchedFluid.apply` wrote only `diag.clamped` and `diag.slam`, so every
    other field stayed at FluidDiagnostics()'s default of 0.0 for the life of a
    batched rollout -- including `mean_submerged`, which
    `TriphibianEnv.observation` feeds straight into the policy's observation
    vector. Under the path the search scores with, a policy therefore perceived
    the machine as bone dry however deep it was, while every verification, probe
    and film saw the truth. That is the whole of the water disagreement the
    ROADMAP recorded as an identification residual: with it fixed the two paths
    agree to 0.00000 in water, and the divergence had appeared at the second
    control decision, with the fluid force itself still agreeing to 1e-11.
    """
    print("\nbatched: the policy is told whether it is wet")
    if needs_batched_evaluator("test_the_batched_path_tells_the_policy_it_is_wet"):
        return
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.batchroll import BatchedFluid, rollout_batch
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    plans = ("beetle", "eel")
    envs = [TriphibianEnv(build(BODY_PLANS[n]()), seed=3) for n in plans]
    for e in envs:
        e.reset(Domain.WATER)
        e.scatter(np.random.default_rng(11))
    bf = BatchedFluid(envs)
    bf.reset_slam()
    rollout_batch(envs, bf, 2.0, [e.cpg.base for e in envs], Domain.WATER)
    for name, e in zip(plans, envs):
        d = e.solver.diag
        check(f"{name}: submerged in water reads submerged, not 0.0",
              d.mean_submerged > 0.5, f"mean_submerged {d.mean_submerged:.3f}")
        check(f"{name}: and the rest of the diagnostics are live too",
              d.buoyancy != 0.0 and d.added_mass != 0.0,
              f"buoyancy {d.buoyancy:.3f}, added_mass {d.added_mass:.3f}")

    # Dry, the same channel must read ~0 -- otherwise "always wet" would pass
    # the check above just as "always dry" passed before it.
    envs = [TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=3)]
    envs[0].reset(Domain.AIR)
    bf = BatchedFluid(envs)
    bf.reset_slam()
    rollout_batch(envs, bf, 2.0, [envs[0].cpg.base], Domain.AIR)
    check("and in the air it reads dry",
          envs[0].solver.diag.mean_submerged < 0.05,
          f"mean_submerged {envs[0].solver.diag.mean_submerged:.3f}")


def test_a_film_reproduces_the_scored_experiment() -> None:
    """A film of an elite is the evaluation that scored it -- checked end to end.

    Found 2026-09-21: arch41's mission-best elite, filmed with the showcase's
    control law, scored land 0.002 against a recorded 0.283, and reproduced its
    record to the last digit only with the shared policy removed -- because the
    actor pool had dropped the shared policy from every re-score.  So this runs
    a real, sharded search with a shared policy and refinement, films its best
    elite without a camera, and asserts that the film chose the *recorded*
    control law and reproduced every raw measurement the record kept.
    """
    if needs_batched_evaluator("test_a_film_reproduces_the_scored_experiment"):
        return
    print("\nfilm: a run's best elite is filmed as it was scored")
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import Domain, MissionSpec
    from dytiscidae.viz.film import MEDIA, evaluate_on_film, pick_elite

    tmp = tempfile.mkdtemp(prefix="dyt-film-")
    try:
        cfg = SearchConfig(generations=3, batch=4, workers=2, min_shard=2, seed=5,
                           segment_seconds=2.0, n_reference_seeds=4, n_random_seeds=0,
                           islands=("generalist",), tier2_every=999, audit_every=999,
                           migrate_every=999, checkpoint_every=1, run_dir=tmp,
                           use_shared_policy=True, promotion_refine_steps=0,
                           controller_refine_steps=1)
        run_search(cfg, MissionSpec())
        e = pick_elite(tmp, by="fitness")
        meta = e.meta or {}
        check("the elite records that the shared policy scored it",
              meta.get("scored_with_shared_policy") is True,
              f"{meta.get('scored_with_shared_policy')!r}")
        net = Path(tmp) / "scoring_networks" / f"gen{int(meta.get('gen', -1)):05d}.npz"
        check("and the network that scored its generation was kept", net.exists(), str(net))
        ev = evaluate_on_film(e, tmp, film=False, log=lambda *a, **k: None)
        check("the film drives it with the recorded control law, not a fallback",
              ev["control_law"].startswith("own + shared policy, the network that scored"),
              ev["control_law"])
        # Competences alone can match vacuously -- zero against zero -- so the
        # raw measurements the record kept are compared too, and at least one
        # of them has to be non-zero for the check to mean anything.
        compared, nonzero, worst = 0, 0, (0.0, "")
        for m in MEDIA:
            check(f"{m}: the film reproduces the recorded competence",
                  ev["media"][m]["match"],
                  f"recorded {ev['media'][m]['recorded']} film {ev['media'][m]['reproduced']}")
            rec_m = (meta.get("ladder_measurements") or {}).get(m) or {}
            seg = ev["result"].segments.get(Domain(m))
            for k, v in rec_m.items():
                got = (seg.measurements or {}).get(k) if seg is not None else None
                if not isinstance(v, (int, float)) or not isinstance(got, (int, float)):
                    continue
                compared += 1
                nonzero += abs(v) > 1e-9
                d = abs(float(v) - float(got)) / max(1.0, abs(float(v)))
                if d > worst[0]:
                    worst = (d, f"{m}.{k} {v:.6g}/{got:.6g}")
        # 2%, not bitwise: the record came through the batched path -- the GPU
        # fluid kernel, and the shared policy evaluated for a whole shard in one
        # matmul -- and the film through the single path, one row at a time.
        # Those differ at 1e-7 (see the sharding test), and a contact-rich land
        # segment amplifies that over 8 s: measured worst 0.5%, on
        # `land.stop_score`.  A different control law or a different start
        # shows as tens of percent.
        # Against the film's own noise floor too (`_nudged`): the same
        # evaluation with rounding-sized noise.  2% stays the least it may be.
        # The record and the film differ by the shared policy evaluated in one
        # shard-wide matmul against one row at a time, ~1e-7 (the comment
        # below), not by rounding -- so the film's floor is measured at that size.
        ev1 = _nudged(lambda: evaluate_on_film(e, tmp, film=False, log=lambda *a, **k: None),
                      rel=1e-7, commands=True)
        floor = 0.0
        for m in MEDIA:
            a_s = ev["result"].segments.get(Domain(m))
            b_s = ev1["result"].segments.get(Domain(m))
            if a_s is None or b_s is None:
                continue
            for k, v in (a_s.measurements or {}).items():
                w1 = (b_s.measurements or {}).get(k)
                if isinstance(v, (int, float)) and isinstance(w1, (int, float)):
                    floor = max(floor, abs(float(v) - float(w1)) / max(1.0, abs(float(v))))
        bar = max(0.02, 2.0 * floor)
        check("and every raw measurement the record kept, none of them vacuously",
              compared >= 10 and nonzero >= 3 and worst[0] < bar,
              f"{compared} compared, {nonzero} non-zero, worst {worst[1] or '-'}; "
              f"bar {bar:.4f} (the film's own floor at 1e-7 noise {floor:.4f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_kernel_older_than_its_source_is_not_usable() -> None:
    """The GPU kernel is a compiled mirror of the numpy solver; nothing compared them.

    Measured 2026-09-20: `mojo/build` held a kernel built 2026-08-04 while
    `mojo/src` was changed 2026-09-15 by F-05, which replaced the stall blend --
    a logistic in the binary, a smoothstep in the source and in `fluid.py`. The
    search scored with one physics and every verification, probe and film used
    the other. With the stale binary the two paths' air and land measurements
    differed by up to 2.0; rebuilt, they agree to 0.000002.
    """
    print("\nkernel: a build older than its source is not usable")
    import shutil
    import tempfile

    from dytiscidae.envs import kernel

    tmp = Path(tempfile.mkdtemp(prefix="dyt-kernel-"))
    try:
        src, build = tmp / "src", tmp / "build"
        src.mkdir(); build.mkdir()
        (src / "fluid_gpu.mojo").write_text("var blend = 6.0 * DEG\n")
        check("with no manifest the kernel is unverified, not fresh and not stale",
              kernel.freshness(src, build)[0] == "unverified")
        kernel.write_manifest(src, build)
        check("after a build it is fresh", kernel.freshness(src, build)[0] == "fresh")
        (src / "fluid_gpu.mojo").write_text("var sep = 16.0 * DEG\n")   # F-05
        state, why = kernel.freshness(src, build)
        check("an edited source makes it stale, and says which file",
              state == "stale" and "fluid_gpu.mojo" in why, f"{state}: {why[:60]}")
        (src / "medium_gpu.mojo").write_text("new file\n")
        check("so does a source the build never saw",
              "medium_gpu.mojo" in kernel.freshness(src, build)[1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # And a stale kernel must not be *used*: the run takes the CPU path and says
    # why, rather than scoring with physics that is not the project's any more.
    # The extension itself is not needed for this: the construction probe is
    # stubbed, so the check runs on a machine with no GPU and inside the
    # mutation harness, which copies only tracked files and so has no build.
    from dytiscidae.envs import batchroll

    class Ok:
        returncode, stdout, stderr = 0, "ok", ""

    saved = (kernel.freshness, batchroll._USABLE, batchroll.AVAILABLE,
             batchroll.subprocess.run if hasattr(batchroll, "subprocess") else None)
    import subprocess as _sp
    saved_run = _sp.run
    try:
        batchroll.AVAILABLE = True
        _sp.run = lambda *a, **k: Ok()
        kernel.freshness = lambda *a, **k: ("stale", "stale for the test")
        batchroll._USABLE = None
        ok, why = batchroll.usable()
        check("a stale kernel is not usable, and says so",
              ok is False and "stale" in why, f"{ok}, {why[:60]}")
        kernel.freshness = lambda *a, **k: ("fresh", "")
        batchroll._USABLE = None
        check("a fresh one is usable", batchroll.usable() == (True, ""))
    finally:
        _sp.run = saved_run
        kernel.freshness, batchroll._USABLE, batchroll.AVAILABLE = saved[:3]


def test_a_checkpoint_names_the_commit_the_process_started_from() -> None:
    """A commit made during a run must not be stamped on that run's checkpoints.

    ``_git_sha`` asked git at every checkpoint write, so arch39 -- launched from
    09bab3b -- wrote a generation-200 checkpoint claiming ca05812, a commit made
    five hours in.  A checkpoint names its code so that its numbers can be
    re-derived; naming the wrong code is worse than naming none.
    """
    print("\ncheckpoint: it names the commit the process started from")
    from dytiscidae.ops import checkpoint as ck

    saved, orig = ck._PROCESS_SHA, ck.subprocess.run

    class Out:
        returncode = 0
        stdout = "launch-sha\n"
    try:
        ck._PROCESS_SHA = None
        ck.subprocess.run = lambda *a, **k: Out()
        first = ck._git_sha()
        Out.stdout = "mid-run-sha\n"            # HEAD moved during the run
        later = ck._git_sha()
    finally:
        ck.subprocess.run, ck._PROCESS_SHA = orig, saved
    check("a checkpoint written after a mid-run commit still names the launch commit",
          first == later == "launch-sha", f"first {first!r}, later {later!r}")


def test_a_finished_run_is_a_checkpoint() -> None:
    """A run has to preserve the experiment, not only the machine.

    The genome, each elite's policy and the shared network's weights were all on
    disk -- and Adam's moments were not, so a resumed run continued with a warm
    network and a cold optimiser; the run's rng stream was not, so every
    evaluation seed after a resume came from a different stream and no score
    survived the boundary; and the mobility basis was not, so every consumer
    re-identified it with a seed nobody had kept.  arch38's archived
    `takeoff_height` of 2.288 m re-measured as 0.000 for that last reason, while
    `max_depth` -- which barely depends on the initial condition -- came back at
    9.51-9.69 against 9.60.
    """
    if needs_batched_evaluator("test_a_finished_run_is_a_checkpoint"):
        return
    print("\nloop: a finished run is a checkpoint")
    import shutil
    import tempfile

    import numpy as np

    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.evolution.loop import SearchConfig, run_search, save_state
    from dytiscidae.learning.ppo import AVAILABLE
    from dytiscidae.ops import checkpoint as ck

    if not AVAILABLE:
        skip("torch is available to test the checkpoint",
             "torch is not importable here")
        return
    import torch

    tmp = tempfile.mkdtemp(prefix="dyt-ckpt-")
    try:
        st = run_search(
            SearchConfig(generations=2, batch=2, seed=11, segment_seconds=1.0,
                         n_reference_seeds=2, n_random_seeds=0,
                         islands=("water", "generalist"), tier2_every=999,
                         audit_every=999, migrate_every=1, checkpoint_every=1,
                         run_dir=tmp, identify_axes_every=1,
                         use_shared_policy=True),
            MissionSpec())
        save_state(st, 2)
        c = ck.read(tmp)

        # The network, including the observation normaliser -- which is a
        # `register_buffer` precisely so that it travels, and a normalisation
        # that does not travel makes stored weights mean something else.
        net2, missing, unexpected = ck.load_network(c)
        obs = np.linspace(-1.0, 1.0, st.shared.n_obs).astype(np.float32)
        with torch.no_grad():
            a1 = st.shared.latent(torch.as_tensor(obs).unsqueeze(0)).mean.numpy()
            a2 = net2.latent(torch.as_tensor(obs).unsqueeze(0)).mean.numpy()
        check("the network reloads to the same outputs",
              not missing and not unexpected
              and float(np.max(np.abs(a1 - a2))) < 1e-12,
              f"max|diff| {float(np.max(np.abs(a1 - a2))):.3e}, "
              f"missing {list(missing)}, unexpected {list(unexpected)}")
        check("and its observation normaliser travels with it",
              float((st.shared.obs_mean - net2.obs_mean).abs().max()) < 1e-12
              and float(st.shared.obs_count) == float(net2.obs_count),
              f"obs_count {float(st.shared.obs_count):.1f}")

        # Adam.  This is what "continue training" rather than "start training
        # with a warm network" turns on.
        back = ck.Checkpoint.optimiser_state(c)
        live = st.shared_opt.state_dict()["state"]
        worst, n = 0.0, 0
        for pid, sd in live.items():
            for k, v in sd.items():
                if not hasattr(v, "detach"):
                    continue
                b = (back or {}).get("state", {}).get(int(pid), {}).get(k)
                if b is None:
                    continue
                worst = max(worst,
                            float(np.max(np.abs(v.detach().cpu().numpy() - b))))
                n += 1
        check("Adam's moments come back equal", n > 0 and worst < 1e-12,
              f"{n} tensors compared, max|diff| {worst:.3e}")

        # The seed stream.  Every candidate's evaluation seed is drawn from it.
        want = int(st.rng.integers(1 << 30))
        r2 = np.random.default_rng(0)
        r2.bit_generator.state = c.rng_state()
        check("and the rng resumes the same stream",
              int(r2.integers(1 << 30)) == want, f"next draw {want}")

        e = c.elite("mission")
        check("the stored elite carries a genome, its policy and its bases",
              e is not None and e.genome is not None and e.policy is not None
              and sorted(e.bases) == ["air", "water"],
              "" if e is None else
              f"policy {None if e.policy is None else e.policy.shape}, "
              f"bases {sorted(e.bases)}, eval_seed {e.eval_seed}")
        check("and the ladder it was scored under",
              bool((c.meta.get("provenance") or {}).get("ladder")),
              "so a reader can tell whether a stored number still means what "
              "its name means")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_shared_policy_survives_a_resume() -> None:
    """The most expensive learned thing in the run must outlive an interruption.

    Every generation's transitions go into the shared policy, and it was the one
    piece of learned state ``save_state`` did not carry -- so an interrupted
    ``--shared-policy`` run resumed with a randomly initialised network while the
    archive kept the scores the trained one had earned, and nothing said so.
    """
    print("\nloop: the shared policy survives a resume")
    import shutil
    import tempfile

    from dytiscidae.learning.ppo import AVAILABLE, SharedPolicy
    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    if not AVAILABLE:
        skip("torch is available to test the shared policy",
             "torch is not importable here")
        return

    tmp = tempfile.mkdtemp(prefix="dyt-shared-resume-")
    try:
        base = dict(batch=1, seed=6, segment_seconds=0.4, n_reference_seeds=1,
                    n_random_seeds=0, islands=("generalist",), tier2_every=999,
                    audit_every=999, migrate_every=999, checkpoint_every=1,
                    run_dir=tmp, identify_axes_every=999,
                    use_shared_policy=True, promotion_refine_steps=0)
        first = run_search(SearchConfig(generations=1, **base), MissionSpec())
        w1 = first.shared.state_dict()["actor.0.weight"].detach().numpy().copy()

        again = run_search(SearchConfig(generations=2, resume=True, **base),
                           MissionSpec())
        w2 = again.shared.state_dict()["actor.0.weight"].detach().numpy()

        fresh = SharedPolicy(first.shared.n_obs, first.shared.n_modes,
                             hidden=64)
        wf = fresh.state_dict()["actor.0.weight"].detach().numpy()
        d_resumed = float(np.linalg.norm(w2 - w1))
        d_fresh = float(np.linalg.norm(wf - w1))
        check("the resumed policy starts from the checkpointed weights",
              d_resumed < 1e-6,
              f"distance to checkpoint {d_resumed:.6f}")
        check("which a fresh network would not",
              d_fresh > 1.0, f"a fresh network sits {d_fresh:.3f} away")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_resume_says_when_controllers_cannot_be_inherited() -> None:
    """Widening the observation orphans every stored controller; say so.

    ``_controller_for`` starts from zeros when a stored weight vector does not
    fit, which is right per candidate -- weights fitted against a different
    observation space mean nothing.  Across a whole archive it is a different
    event: the 14-to-19 channel widening made every policy in every earlier run
    untransferable, and 240 discarded controllers must not look like a normal
    resume.
    """
    print("\nloop: a resume says when controllers cannot be inherited")
    import json
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-orphan-")
    try:
        base = dict(batch=1, seed=4, segment_seconds=0.5, n_reference_seeds=1,
                    n_random_seeds=0, islands=("generalist",), tier2_every=999,
                    audit_every=999, migrate_every=999, checkpoint_every=1,
                    run_dir=tmp, identify_axes_every=999,
                    promotion_refine_steps=0)
        first = run_search(SearchConfig(generations=1, **base), MissionSpec())
        arch = first.archipelago.archives["generalist"]
        check("the run stored controllers to orphan",
              any(e.meta.get("policy") for e in arch.cells.values()))

        # Corrupt the stored width the way a change of observation space does.
        for e in arch.cells.values():
            if e.meta.get("policy"):
                e.meta["policy"] = list(e.meta["policy"]) + [0.0]
        arch.save(Path(tmp) / "archive_generalist.pkl")

        run_search(SearchConfig(generations=2, resume=True, **base),
                   MissionSpec())
        events = [json.loads(l) for l in open(Path(tmp) / "events.jsonl")]
        warned = [e for e in events
                  if e.get("kind") == "policy_shape_mismatch"]
        check("the resume reports the orphaned controllers", warned,
              f"{warned[-1] if warned else 'no event'}")
        check("and counts them rather than merely flagging",
              warned and warned[-1]["mismatched"] >= 1,
              f"{warned[-1]['mismatched']} of {warned[-1]['stored']}"
              if warned else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_refit_that_changes_nothing_can_be_skipped() -> None:
    """A refit that reproduces the axes need not re-bin, and must say so.

    ROADMAP item P, measured on arch40: each refit erased 12-16 cells per island
    while islands grew 9.5-11.5 between refits, and the axes' leading features
    stayed the same refit to refit (mean Jaccard 0.71).  So the subspace
    overlap between the old and new projection is recorded on every refit, and
    ``keep_if_overlap`` (0 = off) keeps the old axes above it.
    """
    print("\ndescriptors: a refit that changes nothing can be skipped")
    from dytiscidae.evolution.descriptors import LearnedDescriptors

    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 16)) * np.linspace(3.0, 0.2, 16)
    turned = base @ (np.eye(16) + 0.8 * rng.normal(size=(16, 16)))

    def refit_twice(threshold, second):
        d = LearnedDescriptors(refit_every=1, keep_if_overlap=threshold)
        for f in base:
            d.observe(f)
        d.fit()
        d._buffer = list(second)
        return d, d.fit()

    d, replaced = refit_twice(0.0, base)
    check("the overlap is recorded on a refit, and identical data overlaps fully",
          d.last_overlap is not None and abs(d.last_overlap - 1.0) < 1e-9,
          f"{d.last_overlap}")
    check("with the gate off, a due refit still replaces the axes", replaced and d.skipped == 0)
    d, replaced = refit_twice(0.95, base)
    check("with it on, identical axes are kept and nothing is re-binned",
          not replaced and d.skipped == 1 and d.report()["skipped"] == 1)
    d, replaced = refit_twice(0.95, turned)
    check("and axes that really moved are still replaced",
          replaced and d.last_overlap < 0.95, f"overlap {d.last_overlap:.3f}")


def test_learned_axes_survive_resume() -> None:
    """A refit that moved the archive onto latent axes must survive ``--resume``.

    Two silent failures, found by forensics on arch24's checkpoint rather than
    by any test: the restored archive kept its cells but woke up on the
    constructor's hand-picked axes, and the refit trigger read an alias of the
    descriptors captured before ``load_state`` replaced them -- so the restored
    object accumulated samples (seen 3049) while the alias it checked stayed
    empty, and a 500-generation run refit zero times after its first resume.
    """
    print("\nloop: learned axes survive a resume")
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search, save_state
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-axes-resume-")
    try:
        base = dict(batch=1, seed=7, segment_seconds=1.0, n_reference_seeds=1,
                    n_random_seeds=0, islands=("generalist",),
                    tier2_every=999, audit_every=999, migrate_every=999,
                    checkpoint_every=1, run_dir=tmp, identify_axes_every=999)
        first = run_search(SearchConfig(generations=2, **base), MissionSpec())

        # Stand in for the long run this fixture cannot afford: enough observed
        # behaviour to fit, one refit, and the rebin the loop performs after one.
        rng = np.random.default_rng(0)
        d = first.descriptors
        for _ in range(d.min_samples):
            d.observe(rng.normal(size=16))
        check("the projection fits once fed", d.fit())
        d.refit_every = 1  # the next refit falls due inside the resumed run
        refits_before = d.refits
        latent = [(f"latent{i}", float(lo), float(hi), 8)
                  for i, (lo, hi) in enumerate(d.bounds())]
        n_latent = len(latent)
        fronts_saved = 0
        for name, a in first.archipelago.archives.items():
            a.rebin(latent, lambda e: np.asarray(e.descriptor, float)[:n_latent])
            fronts_saved += sum(len(f) for f in a.fronts.values())
            a.save(Path(tmp) / f"archive_{name}.pkl")
        save_state(first, first.archipelago.archives["generalist"].generation)

        again = run_search(SearchConfig(generations=5, resume=True, **base),
                           MissionSpec())
        a2 = again.archipelago.archives["generalist"]
        check("the archive wakes up on the axes it was saved with",
              [n for n, *_ in a2.axes][:n_latent]
              == [f"latent{i}" for i in range(n_latent)],
              f"axes after resume: {[n for n, *_ in a2.axes]}")
        check("the fronts come back with it",
              sum(len(f) for f in a2.fronts.values()) >= min(fronts_saved, 1),
              f"{fronts_saved} front members saved")
        check("the restored descriptors keep their memory",
              again.descriptors.seen >= d.min_samples,
              f"seen={again.descriptors.seen}")
        check("and the refit trigger reads the restored object, not a stale alias",
              again.descriptors.refits > refits_before,
              f"refits {refits_before} -> {again.descriptors.refits}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_loop_wires_every_layer_together() -> None:
    """The judge, the auditor, the critic, the curriculum and the islands must
    all be connected to the search, not merely present in the tree.

    Each of them was built and tested on its own before this, and a module that
    passes its own tests while being reachable from nothing is a thing this
    project has shipped before.  So this runs the real loop, briefly, and looks
    at what came out.
    """
    print("\nloop: every layer is actually connected")
    import shutil
    import tempfile

    from dytiscidae.evolution.loop import SearchConfig, run_search
    from dytiscidae.envs.triphibian import MissionSpec

    tmp = tempfile.mkdtemp(prefix="dyt-loop-")
    try:
        cfg = SearchConfig(
            generations=2, batch=1, seed=5, segment_seconds=1.0,
            n_reference_seeds=1, n_random_seeds=0,
            islands=("water", "generalist"), tier2_every=999, audit_every=999,
            migrate_every=1, checkpoint_every=999, run_dir=tmp,
            identify_axes_every=999,
        )
        state = run_search(cfg, MissionSpec())

        check("both islands exist and are separate",
              set(state.archipelago.names) == {"water", "generalist"}
              and state.archipelago.archives["water"]
              is not state.archipelago.archives["generalist"],
              ", ".join(state.archipelago.names))
        check("each island has its own curriculum",
              set(state.curricula) == {"water", "generalist"})
        check("the judge, auditor and critic are attached",
              state.judge is not None and state.auditor is not None
              and state.critic is not None)

        elites = [e for a in state.archipelago.archives.values()
                  for e in a.cells.values()]
        check("something was filed", elites, f"{len(elites)} elites")
        if elites:
            m = elites[0].meta
            for key in ("island", "stage", "stage_name", "rungs", "judged",
                        "critic_discount", "critic_features"):
                check(f"elites carry {key}", key in m,
                      str(m.get(key))[:60])
            check("the judge's rungs are recorded per domain",
                  set(m["rungs"]) >= {"air", "water", "land"},
                  str(m["rungs"]))
            check("and the critic abstains before it has been fitted",
                  m["critic_discount"] == 1.0, str(m["critic_discount"]))

        # The same design is filed on both islands and scored differently,
        # because that is the entire point of having islands.
        w = state.archipelago.archives["water"].best
        g = state.archipelago.archives["generalist"].best
        if w is not None and g is not None:
            check("the islands score their occupants on different terms",
                  abs(w.fitness - g.fitness) > 1e-9 or w.genome is not g.genome,
                  f"water best {w.fitness:.4f}, generalist best {g.fitness:.4f}")

        check("migration ran", state.archipelago.migrations >= 0,
              f"{state.archipelago.migrations} migrations, "
              f"{state.archipelago.hybrids} hybrids")

        # And what it wrote must be readable by the commands that consume it.
        # An archipelago writes one archive per island and no combined file,
        # while render, cohort and showcase all opened ``archive.pkl`` -- so
        # every one of them exited with "no archive" for every run since the
        # islands landed.  The search was reachable; its output was not.
        from dytiscidae.ops.run import load_run_archive

        merged, islands = load_run_archive(tmp)
        check("the run's output is loadable by the tools that consume it",
              merged is not None and len(merged.cells) > 0,
              f"{0 if merged is None else len(merged.cells)} elites "
              f"from {len(islands)} island(s): {', '.join(islands)}")
        check("and every elite says which island it came from",
              merged is not None
              and all((e.meta or {}).get("island") in islands
                      for e in merged.cells.values()),
              f"islands present: "
              f"{sorted({(e.meta or {}).get('island') for e in merged.cells.values()})}")
        check("a run directory with no archive is reported, not crashed on",
              load_run_archive(Path(tmp) / "nothing-here") == (None, []))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scout_finds_dark_horses_and_may_only_protect() -> None:
    """A design that scores badly now but is going somewhere must survive.

    Every other selection pressure here reads the design in front of it: the
    curator prefers fitness, the judge scores what was measured, the critic
    discounts what will not survive.  All of them ask "how good is this" and
    none can ask "what will this become".  A design with a superb body and a
    controller that has not learned to use it scores near zero and is worth more
    than a mediocre design already at its ceiling, and greedy selection breeds
    the second.

    Six islands, a promoting curriculum and a ratcheting bar all sharpen
    selection, which is exactly what kills a dark horse faster -- so the
    counterweight has to be explicit.

    Potential is trainable because it is observable in arrears: the best score
    any descendant reached is a fact.
    """
    print("\nscout: potential, not score")
    from dytiscidae.evolution.scout import SCOUT_DIM, Scout, novelty_of

    rng = np.random.default_rng(0)
    scout = Scout(horizon=5, min_samples=80, refit_every=40, hidden=24, seed=0)

    # Two families with *identical* score distributions.  The dark horse has
    # unused control authority and structural headroom and sits just below a
    # rung; the dead end is at its ceiling.
    def feat(kind):
        f = np.zeros(SCOUT_DIM)
        f[0] = rng.uniform(0.05, 0.20)   # fitness: the same for both
        f[1] = rng.uniform(0.1, 0.6)     # novelty
        if kind == "darkhorse":
            f[2] = rng.uniform(0.8, 1.5)
            f[4] = f[2] * (1 - f[0])
            f[5] = rng.uniform(0.8, 2.5)
            f[10] = rng.uniform(0.0, 0.2)
        else:
            f[2] = rng.uniform(0.0, 0.15)
            f[4] = f[2] * (1 - f[0])
            f[5] = rng.uniform(-0.2, 0.3)
            f[10] = rng.uniform(0.7, 1.0)
        return f

    check("an unfitted scout falls back to novelty, not to nothing",
          scout.potential(np.array([0.1, 0.9] + [0.0] * (SCOUT_DIM - 2))) > 0.0,
          "novelty prior")

    gen = 0
    for r in range(30):
        for i in range(20):
            kind = "darkhorse" if i % 2 else "deadend"
            f = feat(kind)
            pid = f"p{r}_{i}"
            scout.record(pid, None, gen, float(f[0]), "t", f)
            lift = rng.uniform(0.35, 0.7) if kind == "darkhorse" else rng.uniform(0.0, 0.05)
            scout.record(f"c{r}_{i}", pid, gen + 1, float(f[0] + lift), "t", f)
        gen += 6
        scout.harvest(gen)
        if scout.due():
            scout.fit()

    check("the scout trains on realised lift", scout.fitted and len(scout._x) > 500,
          f"{len(scout._x)} labels from lineages that matured")
    check("and is calibrated on held-out designs", scout.calibration > 0.3,
          f"calibration {scout.calibration:.2f}")

    dh = float(np.mean([scout.potential(feat("darkhorse")) for _ in range(300)]))
    de = float(np.mean([scout.potential(feat("deadend")) for _ in range(300)]))
    check("it separates two families with identical scores", dh > 5 * de,
          f"predicted lift {dh:.3f} against {de:.3f}")
    check("and turns that into selection weight",
          scout.selection_weight(feat("darkhorse")) > 1.0,
          f"x{np.mean([scout.selection_weight(feat('darkhorse')) for _ in range(50)]):.2f}")

    # The invariant that keeps it safe: it may only ever argue *for*.
    worst = min(scout.selection_weight(feat("deadend")) for _ in range(200))
    check("the scout can never argue against a design", worst >= 1.0 - 1e-9,
          f"lowest weight x{worst:.3f}")

    # A childless node must not be labelled as a negative example.  Seeding
    # best_descendant at zero made every childless design a negative label
    # proportional to how good it was, which swamped the signal and left
    # calibration at exactly 0.
    s2 = Scout(horizon=1, min_samples=10, refit_every=5)
    s2.record("lonely", None, 0, 0.8, "t", np.zeros(SCOUT_DIM))
    s2.harvest(5)
    check("a design with no descendants has no lift, not negative lift",
          s2._y and s2._y[0] == 0.0, f"label {s2._y[0] if s2._y else None}")

    # The reserve is a quota, so it protects something whatever the scores are.
    class E:
        def __init__(self, f):
            self.meta = {"scout_features": [float(x) for x in f]}

    elites = [E(feat("deadend")) for _ in range(18)] + [E(feat("darkhorse")) for _ in range(2)]
    reserved = scout.reserve_ids(elites)
    check("the reserve holds a fixed share of the archive",
          len(reserved) == max(1, int(round(scout.reserve * len(elites)))),
          f"{len(reserved)} of {len(elites)} protected")
    dark_ids = {id(e) for e in elites[18:]}
    check("and it spends that share on the dark horses",
          len(reserved & dark_ids) >= 1,
          f"{len(reserved & dark_ids)} of the 2 dark horses protected")

    # Novelty, the non-learned half, must behave.
    from dytiscidae.evolution.archive import Archive

    a = Archive([("x", 0.0, 1.0, 8), ("y", 0.0, 1.0, 8)])
    check("an empty archive makes everything novel",
          novelty_of([0.5, 0.5], a) == 1.0)
    a.add("g", 0.5, [0.5, 0.5], objectives=np.array([0.5, 1.0, 1.0]))
    near = novelty_of([0.52, 0.52], a)
    far = novelty_of([0.95, 0.05], a)
    check("and distance from what exists is what novelty means", far > near,
          f"far {far:.3f} against near {near:.3f}")


def test_the_search_is_pointed_at_the_mission_and_compounds() -> None:
    """The five defects the arch31 forensics found, each pinned by a check.

    Every number quoted here was measured over that run's 9,384 evaluations and
    1,451 surviving elites, and every one of them is a property of the machinery
    rather than of the physics, so it belongs in a test rather than in a
    postmortem.
    """
    print("\nsearch: pointed at the mission, and compounding")
    from dytiscidae.core.genome import MUTATION_OPERATORS, mutate, random_genome
    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.curator import Curator, OperatorBandit
    from dytiscidae.evolution.curriculum import Curriculum

    # 1. A new cell starts where its parent got to.  58.9% of arch31's children
    #    landed in an unoccupied cell and every one of them restarted at stage
    #    0, which is why 69.1% of all evaluations were asked the easiest
    #    question no matter what the lineage could already do.
    c = Curriculum()
    c.stages[(1, 1, 1, 1)] = 3
    check("an unvisited cell inherits its parent's stage",
          c.seed_stage((2, 2, 2, 2), c.stage_of((1, 1, 1, 1))) == 3,
          "parent at 'chain' -> child's cell starts at 'chain'")
    check("and a cell that already has a stage keeps it",
          c.seed_stage((1, 1, 1, 1), 0) == 3, "seeding never demotes")

    # 2. A multi-mutation child gets different operators.  6,198 of arch31's
    #    6,284 multi-mutation children were one operator applied two or three
    #    times, because the bandit's argmax was called n times with no state
    #    change in between.
    b = OperatorBandit()
    rng = np.random.default_rng(7)
    picked = []
    for _ in range(3):
        picked.append(b.select(rng, structural_bias=2.2, exclude=tuple(picked)))
    check("three mutations means three different operators",
          len(set(picked)) == 3, ", ".join(picked))

    # 3. And the plan the bandit made is the plan that gets applied, rather than
    #    a fresh uniform draw from it.
    g = random_genome(np.random.default_rng(3))
    plan = ["global_energy", "global_energy", "global_energy"]
    _, applied = mutate(g, np.random.default_rng(4), operators=plan, n_ops=3)
    check("mutate applies the plan it was handed",
          applied == plan, f"{applied}")

    # 4. Credit is improvement, not arrival.  Paying a flat 1.0 for landing in
    #    an empty cell left all twenty-four operators scoring 0.60-0.73 -- a
    #    bandit choosing between arms it cannot tell apart, which drifted into
    #    growth until 31.4% of the population sat on the eight-part cap.
    a = Archive([("x", 0.0, 1.0, 5), ("y", 0.0, 1.0, 5)])
    cur = Curator(a, seed=0)
    for i in range(40):
        cur.credit(["cppn_weights"], "improved", 0.002)
    for i in range(40):
        cur.credit(["add_part"], "new", 0.30)
    weak = cur.bandit.stats["cppn_weights"].mean_reward
    strong = cur.bandit.stats["add_part"].mean_reward
    check("an operator that improves outscores one that merely arrives",
          strong > weak + 0.2, f"add_part {strong:.3f} against cppn_weights {weak:.3f}")
    cur2 = Curator(a, seed=0)
    for i in range(40):
        cur2.credit(["scale"], "new", 0.0)
    check("and arriving with no improvement is worth only the novelty share",
          abs(cur2.bandit.stats["scale"].mean_reward - cur2.novelty_credit) < 1e-9,
          f"{cur2.bandit.stats['scale'].mean_reward:.3f} == novelty_credit")

    # 5. Pruning compares designs asked the same question.  Ranking a promoted
    #    design against its unpromoted neighbours culled the ones that had
    #    advanced: crossing-stage designs were 8.5% of arch31's evaluations and
    #    3.4% of its surviving elites.
    a2 = Archive([("x", 0.0, 1.0, 8), ("y", 0.0, 1.0, 8)])
    cur3 = Curator(a2, seed=0, crowding_limit=0)
    obj = np.array([0.5, 1.0, 1.0])
    for i in range(6):
        a2.add(f"easy{i}", 0.90, [0.50 + 0.02 * i, 0.50], {"stage": 0}, objectives=obj)
    # one advanced design, weak *only* by the standard of the easy question
    a2.add("hard", 0.30, [0.52, 0.52], {"stage": 2}, objectives=obj)
    hard_cell = a2.cell_of(np.array([0.52, 0.52]))
    cur3.prune(max_prunes=3)
    check("a promoted design is not culled for being ranked against easier ones",
          hard_cell in a2.cells,
          "the only stage-2 occupant has no same-stage neighbourhood to lose to")


def test_the_body_plan_outlives_the_lineage_window() -> None:
    """``meta["body_plan"]`` has to say which archetype a design descends from.

    It used to read ``genome.lineage[0]``, and ``lineage`` is a rolling window of
    the last 24 *mutation operator* names.  So the field reported a plan for a
    design's first 24 mutations and an operator name for the rest of its life,
    and every diversity claim made from it in arch30, arch31 and arch33 was
    reading a mixture of the two.
    """
    print("\ntelemetry: the body plan is not the first mutation operator")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.genome import crossover, mutate, random_genome
    from dytiscidae.core.reference import reference_genome

    rng = np.random.default_rng(0)
    g = BODY_PLANS["bat"]()
    check("an archetype knows what it is", g.body_plan == "bat", g.body_plan)
    for _ in range(30):
        g, _ = mutate(g, rng, n_ops=2)
    check("and still does after thirty mutations", g.body_plan == "bat",
          f"body_plan={g.body_plan!r} while lineage[0]={g.lineage[0]!r}")
    check("which is exactly when the old field stopped saying so",
          g.lineage[0] != "bat",
          f"lineage has rolled to {g.lineage[0]!r}")
    other = random_genome(rng)
    check("a random graph is not an archetype", other.body_plan == "random",
          other.body_plan)
    check("the reference is its own plan",
          reference_genome().body_plan == "reference")
    # Crossover takes one parent's graph wholesale, so the plan is that
    # parent's and not a blend of two.
    check("crossover keeps the graph donor's plan",
          crossover(g, other, rng).body_plan == "bat")


def test_every_island_is_reached_by_verification_and_audit() -> None:
    """The island rotates once per generation, so ``gen % N`` aliases.

    With six islands, ``tier2_every = 15`` fires only on islands 0 and 3 and
    ``audit_every = 30`` only on island 0.  Four islands had never had a design
    verified at full fidelity in any run, and every audit in arch30, arch31 and
    arch33 landed on ``air`` -- which is visible in arch33's archives, where the
    only cells carrying a refined controller belong to two islands.
    """
    print("\nschedule: every island is verified and audited")
    import inspect

    from dytiscidae.evolution import loop as loop_mod

    src = inspect.getsource(loop_mod.run_search)
    check("verification counts island visits, not generations",
          "if visits % max(cfg.tier2_every, 1) == 0" in src)
    check("and so does the audit",
          "if visits % max(cfg.audit_every, 1) == 0" in src)

    islands, gens = 6, 900
    for every in (15, 30):
        old, new = set(), set()
        visits = {}
        fires_old = fires_new = 0
        for gen in range(gens):
            isl = gen % islands
            v = visits.get(isl, 0)
            visits[isl] = v + 1
            if gen % every == 0:
                old.add(isl)
                fires_old += 1
            if v % every == 0:
                new.add(isl)
                fires_new += 1
        check(f"every {every} generations: the old rule reached "
              f"{len(old)}/{islands} islands",
              len(old) < islands, f"islands {sorted(old)}")
        check(f"the new rule reaches all {islands}", len(new) == islands,
              f"islands {sorted(new)}")
        check("at the same total cost", abs(fires_new - fires_old) <= 1,
              f"{fires_old} firings before, {fires_new} after")

    # And the counters survive an interruption, or a resumed run re-fires
    # everything at once.
    check("the visit counts are checkpointed",
          '"island_visits": dict(state.island_visits)'
          in inspect.getsource(loop_mod.save_state))
    check("and reconstructed for a checkpoint written before they existed",
          "island_visits" in inspect.getsource(loop_mod.load_state))


def test_a_long_leg_runs_on_promotion_candidates_only() -> None:
    """Eight seconds against a 300 s mission leg is 2.7% of it.

    Nothing that accumulates -- a draining battery, a drifting controller, an
    attitude diverging slowly -- is visible in that window, and over 180
    promotions in arch33 corr(tier1_fraction, tier2_fraction) was +0.077.
    """
    print("\nfidelity: a sixty-second leg where it is affordable")
    import inspect

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import evaluate_tier1_5, weakest_domain
    from dytiscidae.envs.triphibian import Domain
    from dytiscidae.evolution import loop as loop_mod

    check("the weak leg is the one the mission turns on",
          weakest_domain({"air": 0.5, "water": 0.1, "land": 0.4}) is Domain.WATER,
          "mission_fraction is gated on the minimum competence")
    check("and a missing competence counts as zero, not as absent",
          weakest_domain({"air": 0.5}) in (Domain.WATER, Domain.LAND))

    seg = evaluate_tier1_5(build(BODY_PLANS["beetle"]()), seconds=2.0,
                           competences={"air": 0.9, "water": 0.9, "land": 0.0})
    check("it runs the weakest leg and scores it",
          seg.domain is Domain.LAND and seg.duration == 2.0,
          f"{seg.domain.value} for {seg.duration:.0f} s, "
          f"competence {seg.competence:.3f}")

    src = inspect.getsource(loop_mod._verify_and_label)
    check("and it runs at promotion, where the cost is per promotion",
          "_tier1_5(state, elite, p2, ctrl2, spec, rng)" in src)
    check("before the Tier-2 mission, not instead of it",
          src.index("_tier1_5(") < src.index("evaluate_tier2("))
    check("the retention is recorded so the correlation is measurable",
          '"tier1_5_retention"' in inspect.getsource(loop_mod._tier1_5))


def test_the_shared_controller_question_is_answered_with_a_number() -> None:
    """Before a fifth shared policy, ask whether one is representable at all.

    The instrument has to be able to find a shared controller when one exists,
    or a null result from it means nothing.  So it is run twice on synthetic
    teachers: once where every body's optimum genuinely is a smooth function of
    its morphology, and once where the same teachers are paired to the wrong
    bodies.
    """
    print("\ndistillation: the instrument, checked against a known answer")
    from dytiscidae.envs.triphibian import MORPHOLOGY_DIM, TriphibianEnv
    from dytiscidae.learning import distill as D

    rng = np.random.default_rng(0)
    n_modes = 3
    n_obs = TriphibianEnv.OBS_DIM

    # Teachers whose weights are an affine function of the morphology: a
    # conditioned network can represent this family exactly.
    basis = rng.normal(0, 0.25, (MORPHOLOGY_DIM, n_obs * n_modes + n_modes))
    off = rng.normal(0, 0.05, n_obs * n_modes + n_modes)
    teachers = []
    for i in range(40):
        m = rng.uniform(-1, 1, MORPHOLOGY_DIM)
        teachers.append(D.Teacher(name=f"b{i}", island="synthetic",
                                  weights=m @ basis + off, morph=m))

    states = D.sample_states(96, rng)
    check("sampled states have the environment's own width",
          states.shape[1] == n_obs - MORPHOLOGY_DIM,
          f"{states.shape[1]} state channels + {MORPHOLOGY_DIM} morphology")
    targets = D.teacher_targets(teachers, states, n_modes)
    check("and every teacher's output is a bounded intent",
          bool(np.all(np.abs(targets) <= 1.0)),
          f"max |a| = {np.abs(targets).max():.4f}")

    order = rng.permutation(len(teachers))
    test_i, train_i = order[:12], order[12:]

    def fit(perm_targets):
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(n_obs, 128), nn.Tanh(),
                            nn.Linear(128, 128), nn.Tanh(),
                            nn.Linear(128, n_modes), nn.Tanh())
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        morph = np.stack([t.morph for t in teachers])

        def block(idx):
            s = np.repeat(states[None], len(idx), 0)
            m = np.repeat(morph[idx][:, None, :], len(states), 1)
            x = np.concatenate([s, m], 2).reshape(-1, n_obs)
            return (torch.as_tensor(x, dtype=torch.float32),
                    torch.as_tensor(perm_targets[idx].reshape(-1, n_modes),
                                    dtype=torch.float32))

        xa, ya = block(train_i)
        xb, yb = block(test_i)
        for _ in range(5000):
            i = torch.randperm(xa.shape[0])[:2048]
            loss = ((net(xa[i]) - ya[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return D._r2(net(xb).numpy(), yb.numpy())

    honest = fit(targets)
    shuffled = fit(targets[rng.permutation(len(teachers))])
    check("when a shared controller exists the study finds it",
          honest > 0.6, f"held-out R2 {honest:+.3f} on 12 unseen bodies")
    check("and when the morphology means nothing it does not",
          shuffled < 0.2 and honest - shuffled > 0.4,
          f"shuffled floor {shuffled:+.3f} against {honest:+.3f}")
    check("R2 is measured against the constant baseline, so 0.0 is 'no better "
          "than commanding the population mean'",
          abs(D._r2(np.full((64, 2), 0.5), np.full((64, 2), 0.5))) < 1e-6
          or D._r2(np.zeros((64, 2)), rng.normal(size=(64, 2))) < 0.2)


def test_sharding_a_generation_does_not_change_a_score() -> None:
    """Worker processes may only make the search faster, never different.

    Nothing per machine depends on which other machines share its batch: the
    scatter draw and the identification deltas are each ``default_rng`` built
    fresh per machine, the force limiter is per machine, and the fluid kernel
    has no cross-machine reduction.  That is what makes an actor pool safe, and
    it is worth asserting rather than assuming -- a shared generator anywhere in
    that chain would make a design's score depend on its position in the batch.
    """
    if needs_batched_evaluator("test_sharding_a_generation_does_not_change_a_score"):
        return
    print("\nactors: a shard boundary is not visible in a score")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.actors import ActorPool, split
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec

    # The shard arithmetic first, which needs no simulation.
    check("shards cover the batch exactly and in order",
          split(16, 4, 4) == [(0, 4), (4, 8), (8, 12), (12, 16)],
          f"{split(16, 4, 4)}")
    check("a shard is never smaller than min_shard",
          split(16, 16, 4) == split(16, 4, 4),
          "sixteen workers on sixteen machines still make four shards of four")
    check("and a batch smaller than one shard is not split at all",
          split(3, 8, 4) == [(0, 3)], f"{split(3, 8, 4)}")
    check("a pool of one runs in this process",
          ActorPool(1)._pool is None)

    plans = list(BODY_PLANS.values())
    phenos = [build(plans[i % len(plans)]()) for i in range(6)]
    kw = dict(spec=MissionSpec(), segment_seconds=1.0, identify_axes=True,
              seed=11, n_modes=6)

    def run(workers):
        pool = ActorPool(workers, min_shard=2)
        ctrls = [Controller(params=None, policy=None) for _ in phenos]
        try:
            res = pool.evaluate_tier1(phenos, controllers=ctrls, **kw)
        finally:
            pool.close()
        return ([r.mission_fraction for r in res],
                [sorted(c.bases or {}) for c in ctrls],
                [r.mobility["air"].control_rank for r in res])

    one = run(1)
    many = run(3)
    check("the same six machines score the same in one shard or three",
          one[0] == many[0],
          f"max difference {max(abs(a - b) for a, b in zip(one[0], many[0])):.3g}")
    check("and their measured mobility comes back with them",
          one[1] == many[1] and one[2] == many[2],
          f"air ranks {one[2]}")

    # The shared policy has to cross the process boundary too, and until
    # 2026-09-21 it only did when a rollout *buffer* came with it -- i.e. on the
    # generation's learning rollout.  The noise-free re-score and every
    # refinement trial pass no buffer, so on any generation split into more
    # than one shard the workers scored without the shared policy, and those
    # are the numbers the archive keeps: arch41's mission-best elite reproduces
    # its recorded air/water/land 0.034/0.090/0.283 exactly with the shared
    # policy removed, and 0.033/0.113/0.002 with it.  The film, Tier-2 and the
    # showcase all drove it *with* it.  Checked here with the fixture that makes
    # the difference visible: bases identified, then a re-score with a shared
    # network and no buffer, in one shard and in three.
    import torch

    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, 6, hidden=16)
    with torch.no_grad():
        for prm in net.parameters():
            if prm.ndim == 2:
                prm.mul_(4.0)          # a policy that commands something
    net.eval()
    ctrls0 = [Controller(params=None, policy=None) for _ in phenos]
    ActorPool(1).evaluate_tier1(phenos, controllers=ctrls0, **kw)   # bases
    rescore = dict(kw, identify_axes=False)

    def score(workers, shared):
        pool = ActorPool(workers, min_shard=2)
        ctrls = [Controller(params=None, policy=None, bases=dict(c.bases or {}))
                 for c in ctrls0]
        try:
            res = pool.evaluate_tier1(phenos, controllers=ctrls, shared=shared,
                                      **rescore)
        finally:
            pool.close()
        # Where each machine went, not what it scored: at a 1 s segment the
        # seed plans score zero with or without a controller, and a check that
        # compares zeros with zeros passes whatever the pool does.
        return [round(float(seg.distance), 9) for r in res
                for _d, seg in sorted(r.segments.items(), key=lambda kv: str(kv[0]))]

    alone, sharded, without = score(1, net), score(3, net), score(1, None)
    check("the shared policy changes the fixture's scores, so this can fail",
          alone != without, f"with {alone[:3]} without {without[:3]}")
    # To 1e-5, not bitwise: the shared network's forward pass runs at a batch
    # width of six in one shard and two in three, and a matmul at a different
    # width is not bitwise the same sum -- measured 1.6e-7 after a 1 s segment.
    # The defect this guards against shows as 0.226.
    check("a re-score in three shards drives the shared policy as one shard does",
          max(abs(a - b) for a, b in zip(alone, sharded)) < 1e-5,
          f"max difference {max(abs(a - b) for a, b in zip(alone, sharded)):.3g}"
          f"; without the shared policy it would be "
          f"{max(abs(a - b) for a, b in zip(without, sharded)):.3g}")


def test_the_gait_gain_drives_the_same_on_every_path() -> None:
    """The gain channel is honoured by the batched pool and the single path alike.

    Both policies carry it -- one more output on the per-candidate policy, one
    more action on the shared one -- and the two paths sum them in different
    code (``batchroll`` sums in place, ``SummedPolicy`` for everything else).
    A path that dropped either half's gain would score, and film, a different
    machine from the one the archive recorded.
    """
    print("\ncontrol: the gait gain on every path")
    import torch

    from dytiscidae.control.cpg import TWIST_DIM, Policy
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.actors import ActorPool
    from dytiscidae.envs.evaluate import Controller, SharedController, SummedPolicy, evaluate_tier1
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    pheno = build(BODY_PLANS["beetle"]())
    kw = dict(spec=MissionSpec(), segment_seconds=1.0, seed=5)
    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1, hidden=16)
    with torch.no_grad():
        for prm in net.parameters():
            if prm.ndim == 2:
                prm.mul_(4.0)
    net.eval()
    own = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0, gain=True)
    own.weights = np.random.default_rng(1).normal(0.0, 0.3, own.n_weights)

    base = Controller(params=None, policy=None)
    ActorPool(1).evaluate_tier1([pheno], controllers=[base], identify_axes=True, **kw)

    def batched(policy, shared):
        pool = ActorPool(1)
        try:
            r = pool.evaluate_tier1(
                [pheno], controllers=[Controller(params=None, policy=policy,
                                                 bases=dict(base.bases))],
                shared=shared, identify_axes=False, **kw)[0]
        finally:
            pool.close()
        GAINS.append({str(d): {k: v for k, v in seg.measurements.items() if k.startswith("gain_")}
                      for d, seg in r.segments.items()})
        return [round(float(seg.distance), 9) for _d, seg in sorted(r.segments.items(), key=lambda kv: str(kv[0]))]

    GAINS: list = []

    def single(policy, shared):
        law = SharedController(params=None, bases=dict(base.bases),
                               policy=SummedPolicy(own=policy, shared=shared, n_modes=6))
        r = evaluate_tier1(pheno, controller=law, identify_axes=False, **kw)
        GAINS.append({str(d): {k: v for k, v in seg.measurements.items() if k.startswith("gain_")}
                      for d, seg in r.segments.items()})
        return [round(float(seg.distance), 9) for _d, seg in sorted(r.segments.items(), key=lambda kv: str(kv[0]))]

    # The same weights with the gain output removed, so the gain is the only
    # difference: the fixture must move when it is on, or nothing here can fail.
    blind = Policy(n_obs=own.n_obs, n_modes=6, hidden=0, gain=False)
    k = own.n_obs * 7
    blind.weights = np.concatenate([own.weights[:k].reshape(own.n_obs, 7)[:, :6].ravel(),
                                    own.weights[k:k + 6]])
    # And the shared network without its gain action: every tensor whose
    # leading axis is the action width loses its last row.
    net6 = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM, hidden=16)
    sd6 = net6.state_dict()
    net6.load_state_dict({key: (v[:TWIST_DIM] if v.shape != sd6[key].shape else v)
                          for key, v in net.state_dict().items()})
    net6.eval()
    with_gain, without = batched(own, net), batched(blind, net6)
    check("the gain alone changes where the fixture goes, so this can fail",
          with_gain != without, f"with {with_gain} without {without}")
    s_gain = single(own, net)
    check("the single path drives both halves' gain as the batched pool does",
          max(abs(a - b) for a, b in zip(with_gain, s_gain)) < 1e-5,
          f"batched {with_gain} single {s_gain}")
    # GAINS: [batched with gain, batched without, single with gain].
    g_b, g_none, g_s = GAINS
    check("both paths publish the gain they commanded, and the same gain",
          all(g_b[d] and set(g_b[d]) == set(g_s[d]) and
              all(abs(g_b[d][k] - g_s[d][k]) < 1e-6 for k in g_b[d]) for d in g_b),
          f"batched {g_b} single {g_s}")
    check("and a controller without the channel publishes none",
          not any(g_none.values()), f"{g_none}")


def test_a_nan_observation_fails_the_rollout_not_the_batch() -> None:
    """arch43 ran single-process from generation 0 (2026-09-26): one machine's
    observation went NaN after MuJoCo's auto-reset, the shared policy raised
    inside ``torch.distributions.Normal`` in a worker, and the pool degraded for
    the rest of the run at four times the cost.  A non-finite observation now
    fails that rollout as diverged; the batch, and the other machine, go on."""
    print("\ncontrol: a NaN observation fails its own rollout")
    import torch

    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.actors import ActorPool
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    phenos = [build(BODY_PLANS["beetle"]()), build(BODY_PLANS["beetle"]())]
    kw = dict(spec=MissionSpec(), segment_seconds=1.0, seed=5)
    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1, hidden=16)
    net.eval()
    bases = Controller(params=None, policy=None)
    ActorPool(1).evaluate_tier1(phenos[:1], controllers=[bases], identify_axes=True, **kw)

    real = TriphibianEnv.observation
    order: list = []
    calls: dict = {}

    def poisoned(self, *a, **k):
        if id(self) not in calls:
            order.append(id(self))
            calls[id(self)] = 0
        calls[id(self)] += 1
        obs = real(self, *a, **k)
        # every other environment is the first machine's (two per batch)
        if order.index(id(self)) % 2 == 0 and calls[id(self)] > 3:
            obs = np.full_like(np.asarray(obs, float), np.nan)
        return obs

    TriphibianEnv.observation = poisoned
    raised = None
    try:
        res = ActorPool(1).evaluate_tier1(
            phenos, controllers=[Controller(params=None, policy=None, bases=dict(bases.bases))
                                 for _ in phenos],
            shared=net, identify_axes=False, **kw)
    except Exception as exc:          # noqa: BLE001 -- the failure under test
        raised, res = exc, None
    finally:
        TriphibianEnv.observation = real
    check("a NaN observation does not raise out of the batch",
          raised is None, f"{type(raised).__name__}: {raised}")
    if res is None:
        return
    bad = [str(d) for d, seg in res[0].segments.items() if seg.failure == "diverged"]
    check("the poisoned machine's rollouts fail as diverged and score nothing",
          bad and all(res[0].segments[d].competence == 0.0
                      for d in res[0].segments if str(d) in bad),
          f"diverged {bad} of {[str(d) for d in res[0].segments]}")
    good = res[1].segments.values()
    check("the other machine is scored as usual",
          all(not seg.failure == "diverged" and np.isfinite(seg.competence) for seg in good),
          f"{[(seg.failure, seg.competence) for seg in good]}")


def test_the_early_fluid_launch_changes_nothing() -> None:
    """`step_batch` starts the next step's fluid on the device before the caller
    records this step and runs the policy (2026-09-27): the wait was a busy
    spin of ~100 us a step.  That is only an optimisation if nothing the caller
    does in between feeds the fluid, so the whole evaluation must come out the
    same to the bit with it on and off."""
    print("\nbatched: the early fluid launch changes nothing")
    import torch

    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import SharedPolicy

    phenos = [build(BODY_PLANS[k]()) for k in ("beetle", "teal")]
    torch.manual_seed(0)
    net = SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM + 1, hidden=16)
    net.eval()

    def run():
        res = batchroll.evaluate_tier1_batch(
            phenos, spec=MissionSpec(), segment_seconds=1.0, seed=5,
            identify_axes=True, shared=net)
        out = {}
        for i, r in enumerate(res):
            for d, seg in r.segments.items():
                for k, v in list(vars(seg).items()) + list(seg.measurements.items()):
                    if isinstance(v, float):
                        out[f"{i}.{d}.{k}"] = v
            for d, tr in r.transitions.results.items():
                for k, v in vars(tr).items():
                    if isinstance(v, float):
                        out[f"{i}.T{d}.{k}"] = v
        return out

    was = batchroll.PRELAUNCH
    try:
        batchroll.PRELAUNCH = True
        early = run()
        batchroll.PRELAUNCH = False
        late = run()
    finally:
        batchroll.PRELAUNCH = was
    moved = [k for k in early if not (early[k] == late.get(k)
                                      or (early[k] != early[k] and late.get(k) != late.get(k)))]
    check("every measurement bit-identical with the launch early and in place",
          not moved and len(early) > 50, f"{len(early)} numbers, moved {moved[:5]}")


def test_structure_can_be_recombined_and_duplicated() -> None:
    """Structure was asexual: every graph descended from one seed by mutation.

    ``crossover`` took one parent's graph wholesale, so two islands that
    independently found a good wing and a good fin could not produce a design
    carrying both -- which is most of the argument for having islands.  And
    there was no way to build a graded limb series except by stumbling on each
    member from the prior, because the only additive operator draws a part from
    that prior.
    """
    print("\nstructure: grafting and duplication")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.genome import (MAX_BODIES, MAX_PARTS,
                                        MUTATION_OPERATORS, STRUCTURAL_OPERATORS,
                                        crossover, descendants, estimated_bodies,
                                        graft_subtree)
    from dytiscidae.core.phenotype import build

    rng = np.random.default_rng(3)
    check("the part cap is no longer the compute-era eight", MAX_PARTS > 8,
          f"MAX_PARTS = {MAX_PARTS}, with a body budget of {MAX_BODIES}")
    check("duplication is registered, and as a structural move",
          "duplicate_part" in MUTATION_OPERATORS
          and "duplicate_part" in STRUCTURAL_OPERATORS)

    # Duplication has to *diverge*, which means its own shape gene.
    g = BODY_PLANS["gannet"]()
    surfaces = [p for p in g.parts if p.is_surface and p.surface_cppn >= 0]
    before_parts, before_cppns = len(g.parts), len(g.cppns)
    fired = MUTATION_OPERATORS["duplicate_part"](g, rng)
    check("a duplication adds a part", fired and len(g.parts) == before_parts + 1,
          f"{before_parts} -> {len(g.parts)}")
    grew = len(g.cppns) > before_cppns
    check("and the copy gets its own shape gene rather than sharing one",
          grew or not surfaces,
          f"cppns {before_cppns} -> {len(g.cppns)}; sharing would lock the two "
          "parts to one gene that no mutation can separate")
    idx = [i for i, p in enumerate(g.parts)]
    check("the duplicate is still buildable", len(build(g).segments) > 0,
          f"{len(build(g).segments)} segments from {len(idx)} parts")

    # Grafting has to move a coherent subtree and leave the receiver buildable.
    names = list(BODY_PLANS)
    fired = built = added = 0
    for i in range(60):
        a = BODY_PLANS[names[i % len(names)]]()
        b = BODY_PLANS[names[(i // len(names) + 1) % len(names)]]()
        kid = a.copy()
        n0 = len(kid.parts)
        if not graft_subtree(kid, b, rng):
            continue
        fired += 1
        added += len(kid.parts) - n0
        try:
            build(kid)
            built += 1
        except Exception:
            pass
    check("a graft moves a limb between two graphs", fired > 40,
          f"fired {fired}/60, adding {added / max(fired, 1):.2f} parts on average")
    check("and everything it produces still builds", built == fired,
          f"{built}/{fired}")

    check("the subtree walk terminates on a cyclic graph",
          len(descendants(g, g.root)) <= len(g.parts),
          f"{len(descendants(g, g.root))} of {len(g.parts)} parts reachable")

    # And crossover reaches it, with the lineage saying which happened.
    seen = set()
    for _ in range(40):
        kid = crossover(BODY_PLANS["eel"](), BODY_PLANS["bat"](), rng)
        seen.add(kid.lineage[-1])
    check("crossover records whether structure moved or only genes did",
          seen == {"graft", "crossover"}, f"{sorted(seen)}")

    # The budget is on bodies, because a part is not a body.
    branchy = BODY_PLANS["medusa"]()
    check("the body estimate sees what a part count cannot",
          estimated_bodies(branchy) > len(branchy.parts),
          f"{len(branchy.parts)} parts expand to about "
          f"{estimated_bodies(branchy)} bodies")


def test_something_asks_a_machine_to_leave_the_ground() -> None:
    """`land_to_air` was in no island's objective, and no hybrid ever left home.

    Three runs -- arch34, arch35, arch36 -- filmed the same continuous mission
    result: 0/2 transitions and 0.0 m of depth, while arch35 and arch36 spent
    two runs building a take-off ladder, gating it, and multiplying it into
    `mission_fraction`.  The reason was not in the scoring.  Six islands covered
    three singles, water+land and air+water; **land+air was missing**, and
    `land_to_air` appeared in no island's transition tuple at all -- the land
    island's only crossing was `water_to_land`, arriving and never leaving.

    And hybridisation could not fill the gap, because the destination loop broke
    on its first existing name and `generalist` always exists: 39 of arch36's 39
    hybrids and 24 of arch35's 24 went there, and `amphibian` and `aerial_diver`
    received none.  The air x land cross -- a walking flyer -- was built every
    migration and only ever scored where water was also demanded.
    """
    print("\nislands: something has to ask a machine to leave the ground")
    import numpy as np

    from dytiscidae.evolution.archive import Archive
    from dytiscidae.evolution.curator import Curator
    from dytiscidae.evolution.islands import (
        HYBRID_HOME,
        ISLANDS,
        Archipelago,
        island_score,
    )

    asks = [k for k, v in ISLANDS.items() if "land_to_air" in v["transitions"]]
    check("at least one island's objective contains land_to_air",
          bool(asks), ", ".join(asks) or "none -- nothing asks")
    check("and one island's domains are exactly land and air",
          any(set(v["domains"]) == {"land", "air"} for v in ISLANDS.values()),
          ", ".join("+".join(v["domains"]) for v in ISLANDS.values()))

    # The pair island must read its own two domains and nothing else, or it is
    # just another generalist and the specialisation argument does not apply.
    class Seg:
        def __init__(self, c):
            self.competence = c

    class Res:
        def __init__(self, **kw):
            self.segments = {d: Seg(c) for d, c in kw.items()}
            self.mission_fraction = 0.0

    strong = island_score("land_air", Res(land=0.8, air=0.7, water=0.0))
    weak = island_score("land_air", Res(land=0.8, air=0.1, water=0.9))
    check("land_air scores land and air, and drowning in water does not help",
          strong > weak, f"{strong:.4f} with air 0.7 against {weak:.4f} with air 0.1")

    # Every specialist pair goes to the island whose objective is that pair.
    homes = {frozenset(k): v[0] for k, v in HYBRID_HOME.items()}
    check("each specialist cross has its own destination island",
          len(set(homes.values())) == 3,
          ", ".join(f"{'x'.join(sorted(k))}->{v}" for k, v in homes.items()))

    axes = [("m", 0.0, 1.0, 4), ("d", 0.0, 1.0, 4)]
    arch = Archipelago(migrate_every=1, n_migrants=1)
    for name in ISLANDS:
        a = Archive(list(axes))
        arch.register(name, a, Curator(a, seed=0))
    rng = np.random.default_rng(0)
    for i, name in enumerate(("air", "water", "land")):
        arch.archives[name].add(f"g{i}", 0.9, np.array([0.5, 0.5]), {}, tier=1)
    moved = arch.migrate(1, rng, crossover=lambda a, b, r: f"{a}+{b}")
    hyb = [m for m in moved if m["kind"] == "hybrid"]
    dests = sorted(m["island"] for m in hyb)
    check("all three crosses are made", len(hyb) == 3, f"{len(hyb)} hybrids")
    check("and they land on three different islands, not all on generalist",
          len(set(dests)) == 3, ", ".join(dests))
    check("the air x land cross goes to the land+air island",
          any(m["origin"] in ("airxland", "landxair") and m["island"] == "land_air"
              for m in hyb),
          ", ".join(f"{m['origin']}->{m['island']}" for m in hyb))

    # And the transition has to be scored, or the objective asks for something
    # the evaluator never computes.
    import inspect

    from dytiscidae.envs import batchroll, evaluate as _ev
    for mod in (batchroll, _ev):
        src = inspect.getsource(mod)
        check(f"{mod.__name__.split('.')[-1]} scores land_to_air",
              'for kind in ("air_to_water", "water_to_air", "water_to_land",\n'
              '                 "land_to_air"):' in src)


def main() -> int:
    print("=" * 68)
    print("Dytiscidae search-machinery verification")
    print("=" * 68)
    run_all([
        test_mobility_recovers_known_basis,
        test_archive_placement_and_improvement,
        test_bandit_learns_which_operator_pays,
        test_no_operator_can_go_dormant_under_the_structural_tilt,
        test_curator_regimes_respond_to_the_run,
        test_curator_quarantines_repeat_exploits,
        test_cmaes_optimises_a_known_function,
        test_cppn_fields_are_deterministic_and_bounded,
        test_the_gait_operator_moves_every_coordinate_at_once,
        test_every_mutation_operator_keeps_the_genome_buildable,
        test_phenotype_invariants,
        test_cpg_respects_joint_limits,
        test_a_rare_capability_survives_the_learned_projection,
        test_learned_descriptors_replace_the_hand_picked_axes,
        test_cells_hold_a_pareto_front_not_a_weighted_sum,
        test_intervention_is_triggered_by_evidence_not_a_schedule,
        test_no_dataclass_can_raise_on_equality,
        test_arriving_somewhere_is_not_one_lucky_timestep,
        test_a_mujoco_auto_reset_ends_the_continuous_mission,
        test_transitions_are_graded_not_pass_fail,
        test_an_attempted_takeoff_outscores_never_leaving_the_ground,
        test_judge_ladder_is_fixed_and_bar_only_tightens,
        test_auditor_can_invalidate_and_veto,
        test_critic_learns_the_exploit_signature,
        test_a_specialist_islands_curriculum_reads_only_its_own_medium,
        test_an_islands_best_is_judged_on_its_own_domains,
        test_one_islands_archive_is_read_alone_not_through_the_merge,
        test_the_island_objective_takes_its_weight_back,
        test_curriculum_and_islands_give_gradient_where_the_mission_gives_none,
        test_scout_finds_dark_horses_and_may_only_protect,
        test_a_run_can_be_picked_up_where_it_stopped,
        test_promotion_needs_a_nonzero_answer_to_the_next_question,
        test_the_headline_is_the_mission,
        test_an_audit_perturbs_the_scored_experiment_and_nothing_else,
        test_promotion_spends_refinement_and_keeps_what_it_buys,
        test_a_shared_command_means_the_same_thing_on_every_body,
        test_the_identification_width_reaches_the_policy,
        test_the_search_is_pointed_at_the_mission_and_compounds,
        test_every_path_agrees_on_the_control_law,
        test_the_shared_policy_survives_a_resume,
        test_the_batched_path_tells_the_policy_it_is_wet,
        test_a_film_reproduces_the_scored_experiment,
        test_a_kernel_older_than_its_source_is_not_usable,
        test_a_checkpoint_names_the_commit_the_process_started_from,
        test_a_finished_run_is_a_checkpoint,
        test_the_two_evaluation_paths_score_the_same_machine_the_same,
        test_a_resume_says_when_controllers_cannot_be_inherited,
        test_a_refit_that_changes_nothing_can_be_skipped,
        test_learned_axes_survive_resume,
        test_the_loop_wires_every_layer_together,
        test_the_body_plan_outlives_the_lineage_window,
        test_something_asks_a_machine_to_leave_the_ground,
        test_every_island_is_reached_by_verification_and_audit,
        test_a_long_leg_runs_on_promotion_candidates_only,
        test_the_shared_controller_question_is_answered_with_a_number,
        test_sharding_a_generation_does_not_change_a_score,
        test_the_gait_gain_drives_the_same_on_every_path,
        test_a_nan_observation_fails_the_rollout_not_the_batch,
        test_the_early_fluid_launch_changes_nothing,
        test_structure_can_be_recombined_and_duplicated,
    ])
    return report("all search-machinery checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
