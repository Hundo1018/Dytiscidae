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


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


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

    check("the two strong axes are found", basis.rank == 2,
          f"rank={basis.rank} sigmas={np.round(basis.authority, 3)}")
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
    check("raising that axis's gain makes it count", strong.rank == 3,
          f"rank={strong.rank} sigmas={np.round(strong.authority, 3)}")

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
    check("credit is split across co-applied operators",
          True)  # covered by construction; asserted below
    b2 = OperatorBandit(["x", "y"])
    b2.update(["x", "y"], 1.0)
    check("two operators each get half the reward",
          abs(b2.stats["x"].lifetime_mean - 0.5) < 1e-9,
          f"{b2.stats['x'].lifetime_mean}")


def test_curator_regimes_respond_to_the_run() -> None:
    print("\ncurator: regime detection")
    a = Archive([("m", 0.0, 1.0, 8), ("d", 0.0, 1.0, 8)])
    c = Curator(a, seed=0)

    r = c.update_regime()
    check("an empty archive bootstraps", r.name == "bootstrapping", r.name)
    check("bootstrapping runs no emitters", r.emitter_fraction == 0.0)

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
    print("\njudge: fixed ladder, ratcheting bar")
    from dytiscidae.evolution.judge import LADDER, Judge, rung_reached

    j = Judge(quantile=0.9, update_every=1)

    # The ladder must be a progression: more capability, more rungs.
    seq = [
        ({"airborne_fraction": 0.05}, 0),
        ({"airborne_fraction": 0.7, "sink_rate": 6.0}, 2),
        ({"airborne_fraction": 0.7, "sink_rate": 2.0}, 3),
        ({"airborne_fraction": 0.9, "sink_rate": 0.2}, 4),
        ({"airborne_fraction": 0.95, "sink_rate": -1.0, "turn_rate_held": 0.0}, 5),
    ]
    ok = all(rung_reached("air", m) == k for m, k in seq)
    check("the ladder orders capability", ok,
          " ".join(str(rung_reached("air", m)) for m, _ in seq))
    check("a design cannot skip a rung it failed",
          rung_reached("air", {"airborne_fraction": 0.05, "sink_rate": -9.0}) == 0,
          "plummeting-but-never-airborne stays at rung 0")

    # The within-rung bonus must never reach the next rung's score.
    below = j.score("air", {"airborne_fraction": 0.95, "sink_rate": -1.0,
                            "turn_rate_held": 0.0})
    top = j.score("air", {"airborne_fraction": 0.95, "sink_rate": -1.0,
                          "turn_rate_held": 0.5})
    check("clearing four rungs never ties with clearing five",
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
    m = {"airborne_fraction": 0.9, "sink_rate": 1.0}
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
        j.observe({"water": {"max_depth": 3.0}})
    moves = j.maybe_tighten(1)
    raised = j.ratchets["water"].bar
    check("the judge tightened", moves and raised > 0.0, f"bar now {raised:.2f} m")
    vetoed = a.review_tightening(j, moves, invalid_designs=1)
    check("and the auditor can veto that tightening",
          vetoed and j.ratchets["water"].bar < raised,
          f"bar rolled back to {j.ratchets['water'].bar:.2f} m")
    check("a veto with nothing invalid does nothing",
          not a.review_tightening(j, moves, invalid_designs=0))


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
    specialist = Res(0.02, 0.85, 0.03, 0.001,
                     {"water": {"depth_error": 0.8, "max_depth": 10.0}})
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


def main() -> int:
    print("=" * 68)
    print("Dytiscidae search-machinery verification")
    print("=" * 68)
    test_mobility_recovers_known_basis()
    test_archive_placement_and_improvement()
    test_bandit_learns_which_operator_pays()
    test_curator_regimes_respond_to_the_run()
    test_curator_quarantines_repeat_exploits()
    test_cmaes_optimises_a_known_function()
    test_cppn_fields_are_deterministic_and_bounded()
    test_every_mutation_operator_keeps_the_genome_buildable()
    test_phenotype_invariants()
    test_cpg_respects_joint_limits()
    test_learned_descriptors_replace_the_hand_picked_axes()
    test_cells_hold_a_pareto_front_not_a_weighted_sum()
    test_intervention_is_triggered_by_evidence_not_a_schedule()
    test_no_dataclass_can_raise_on_equality()
    test_transitions_are_graded_not_pass_fail()
    test_judge_ladder_is_fixed_and_bar_only_tightens()
    test_auditor_can_invalidate_and_veto()
    test_critic_learns_the_exploit_signature()
    test_curriculum_and_islands_give_gradient_where_the_mission_gives_none()
    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all search-machinery checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
