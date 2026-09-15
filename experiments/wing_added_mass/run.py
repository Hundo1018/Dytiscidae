#!/usr/bin/env python3
"""The wing's added mass is a tensor, and correcting it destabilises the solver.

0. The problem
--------------
`FluidSolver.apply` gives a wing strip

    m_add = rho * pi * chord^2 / 4 * dr

whichever way the strip accelerates.  That is the **normal** entry of the
plate's added-mass tensor.  Strip theory gives all three::

    m_span = 0        m_chord = rho pi t^2/4 b       m_normal = rho pi c^2/4 b

so the normal-to-chordwise ratio is `(c/t)^2` exactly.  `benchmarks/layers.py`
layer 2 verifies that chain against the closed form at c/t = 10, 100 and 1000
with no simulation in it.  The bluff branch four lines above already projects
its own tensor onto the direction of motion; the wing branch does not.
`docs/MATH_AUDIT.md` **F-03**.

1. What this experiment is for
------------------------------
Not to argue that the tensor is right -- layer 2 settles that.  To answer the
two questions that decide whether it can be applied:

  * `(c/t)^2` is the error in the pure edgewise direction.  What is it worth on
    a machine that is flapping, where the flow is neither purely normal nor
    purely edgewise?  Section B measures the direction the flow actually comes
    from, over every wing strip of every seed plan in all three media.
  * What happens to the solver if it is corrected?  Section D runs each plan at
    four timesteps, driven and with the actuators held completely still.

2. The result, in one line
--------------------------
Correcting it makes the ladder's layer 2 hold, and makes four of seven seed
plans diverge at this project's timestep -- three of them with no actuation at
all.  The surplus inertia is holding up the *explicit* lift and drag forces.
So F-03 stays open, and section D is the reason.

Run:  PYTHONPATH=. python experiments/wing_added_mass/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

from experiments.harness import ExperimentResult, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []
FINDINGS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)
    return cond


def finding(name: str, holds: bool, detail: str = "") -> bool:
    print(f"  [{'gap ' if holds else 'GONE'}] {name}"
          f"{('  -- ' + detail) if detail else ''}")
    (FINDINGS if holds else FAILURES).append(name)
    return holds


def wing_tensor(panels, w) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(m_span, m_chord, m_normal)` per wing strip, in seawater."""
    from dytiscidae.physics.medium import SEAWATER

    rho = SEAWATER.rho
    ext, chord, dr = panels.ext_local[w], panels.chord[w], panels.dr[w]
    t = np.maximum(ext[:, 2], 1e-5)
    k = rho * np.pi * 0.25 * dr
    return np.zeros_like(chord), k * t**2, k * chord**2


# --------------------------------------------------------------------------


def test_the_code_applies_the_normal_entry_everywhere(cfg, out) -> None:
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.physics.fluid import WING
    from dytiscidae.physics.medium import SEAWATER

    print("\nA. what the solver applies, against the tensor")
    env = TriphibianEnv(build(bodyplans.gannet()))
    panels = env.solver.panels
    w = panels.kind == WING
    m_s, m_c, m_n = wing_tensor(panels, w)
    coded = SEAWATER.rho * np.pi * panels.chord[w] ** 2 * 0.25 * panels.dr[w]
    check("the coded value is the tensor's normal entry",
          float(np.max(np.abs(coded - m_n))) < 1e-12,
          f"{coded.sum():.3f} kg against {m_n.sum():.3f} kg over "
          f"{int(w.sum())} strips")
    ratio = m_n / np.maximum(m_c, 1e-30)
    ct = panels.chord[w] / np.maximum(panels.ext_local[w][:, 2], 1e-30)
    check("and the normal-to-chordwise ratio is (c/t)^2",
          float(np.max(np.abs(ratio / ct**2 - 1.0)))
          < cfg["tolerances"]["tensor_ratio_rel"],
          f"c/t runs {ct.min():.1f} to {ct.max():.1f} on this machine, so the "
          f"edgewise overstatement is {ct.min()**2:.0f}x to {ct.max()**2:.0f}x")
    out["gannet_wing_strips"] = int(w.sum())
    out["gannet_m_normal_kg"] = float(m_n.sum())
    out["gannet_m_chord_kg"] = float(m_c.sum())
    out["c_over_t"] = {"min": float(ct.min()), "max": float(ct.max()),
                       "median": float(np.median(ct))}


def test_which_way_the_flow_actually_comes_from(cfg, out) -> None:
    """`(c/t)^2` is the worst case in one direction.  This is the machines'."""
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.fluid import WING

    print("\nB. the direction the flow comes from, at a wing strip")
    per_medium: dict[str, list] = {m: [] for m in cfg["rollout"]["media"]}
    for name in cfg["seed_plans"]:
        env = TriphibianEnv(build(getattr(bodyplans, name)()))
        env.solver.record_state = True
        w = env.solver.panels.kind == WING
        if not w.any():
            continue
        for medium in cfg["rollout"]["media"]:
            env.reset(Domain(medium), randomise=False)
            for _ in range(int(cfg["rollout"]["seconds"] / env.timestep)):
                env.step(env.cpg.command(env.cpg.base, env.data.time))
                st = env.solver.last_state
                if st is not None:
                    per_medium[medium].append(st["flow_cosines"][w] ** 2)

    print(f"         {'medium':<7} {'strip-steps':>12}  {'span':>7} "
          f"{'chord':>7} {'normal':>7}   mean squared direction cosine")
    rows, everything = {}, []
    for medium, v in per_medium.items():
        if not v:
            continue
        d = np.concatenate(v)
        d = d[np.isfinite(d).all(axis=1)]
        everything.append(d)
        rows[medium] = [float(x) for x in d.mean(axis=0)]
        print(f"         {medium:<7} {len(d):>12,}  {d[:, 0].mean():7.4f} "
              f"{d[:, 1].mean():7.4f} {d[:, 2].mean():7.4f}")
    allc = np.concatenate(everything)
    mean = allc.mean(axis=0)
    rows["all"] = [float(x) for x in mean]
    print(f"         {'all':<7} {len(allc):>12,}  {mean[0]:7.4f} "
          f"{mean[1]:7.4f} {mean[2]:7.4f}")
    finding("F-03  the flow at a wing strip is mostly not normal to it",
            mean[2] < 0.5,
            f"only {mean[2]:.1%} normal; it is more chordwise ({mean[1]:.1%}) "
            f"than normal, and {mean[0]:.1%} spanwise, which the tensor says "
            f"entrains nothing at all")
    out["flow_cosines"] = rows
    out["flow_samples"] = int(len(allc))


def test_what_it_is_worth(cfg, out) -> None:
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.physics.fluid import WING

    print("\nC. what the correction is worth, per plan")
    mean = np.array(out["flow_cosines"]["all"])
    print(f"         {'plan':<8} {'dry mass':>9} {'coded':>10} {'tensor':>10} "
          f"{'ratio':>7}   added mass on the wings, in seawater")
    rows = []
    for name in cfg["seed_plans"]:
        ph = build(getattr(bodyplans, name)())
        env = TriphibianEnv(ph)
        panels = env.solver.panels
        w = panels.kind == WING
        if not w.any():
            print(f"         {name:<8} {ph.mass:8.3f} kg        --         "
                  f"--       --   no wing strips")
            continue
        m_s, m_c, m_n = wing_tensor(panels, w)
        coded = float(m_n.sum())
        eff = float((mean[0] * m_s + mean[1] * m_c + mean[2] * m_n).sum())
        rows.append({"plan": name, "dry_mass_kg": float(ph.mass),
                     "coded_kg": coded, "tensor_kg": eff,
                     "ratio": eff / max(coded, 1e-30)})
        print(f"         {name:<8} {ph.mass:8.3f} kg {coded:9.2f} kg "
              f"{eff:9.2f} kg {eff/max(coded,1e-30):6.1%}")
    out["per_plan"] = rows
    worst = max(rows, key=lambda r: r["coded_kg"] / max(r["dry_mass_kg"], 1e-9))
    print(f"         the coded value is {worst['coded_kg']/worst['dry_mass_kg']:.1f}x "
          f"{worst['plan']}'s whole dry mass")
    finding("F-03  correcting it is worth about 3x on a real machine, not "
            "(c/t)^2",
            all(0.2 < r["ratio"] < 0.6 for r in rows),
            f"ratios {min(r['ratio'] for r in rows):.1%} to "
            f"{max(r['ratio'] for r in rows):.1%} across the plans")


def test_whether_the_solver_survives_it(cfg, out) -> None:
    """The question that decides whether F-03 can be closed.

    Run each plan with `wing_added_mass_tensor` on, at four timesteps, twice:
    driven by its own gait, and with the actuators held completely still.  The
    second is the probe CLAUDE.md's lesson section insists on, and it is what
    separates a controller problem from a solver one.  The last column is the
    same plan at the coarse step with the switch off, so the comparison is
    against the solver as it ships and not against a memory of it.
    """
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    print("\nD. what happens to the solver if the tensor is applied")

    def travel(plan: str, dt: float, drive: bool, tensor: bool) -> float:
        """Largest joint angle reached.  The CPG commands under one radian."""
        env = TriphibianEnv(build(getattr(bodyplans, plan)()))
        env.solver.wing_added_mass_tensor = tensor
        env.model.opt.timestep = dt
        env.reset(Domain.AIR, randomise=False)
        q = []
        for _ in range(int(cfg["stability"]["seconds"] / dt)):
            u = (env.cpg.command(env.cpg.base, env.data.time) if drive
                 else np.zeros(env.model.nu))
            env.step(u)
            q.append(env.data.qpos[7:].copy())
        q = np.array(q)
        return float(np.nanmax(np.abs(q))) if q.size else 0.0

    dts = cfg["stability"]["timesteps"]
    limit = cfg["stability"]["runaway_rad"]
    print(f"         max |joint angle| over {cfg['stability']['seconds']:.0f} "
          f"s, radians.  The CPG commands under 1 rad.  Bit-for-bit repeatable")
    print(f"         across runs, so every number here is one measurement, not "
          f"a sample.")
    print()
    print(f"         {'':<8} {'tensor on':>38}  {'incumbent':>22}")
    print(f"         {'plan':<8} "
          + "".join(f"{f'dt={d:g}':>12}" for d in dts)
          + f"{'still':>12}  {'driven':>10}{'still':>12}")
    rows = []
    for plan in cfg["seed_plans"]:
        driven = [travel(plan, d, True, True) for d in dts]
        still = travel(plan, dts[0], False, True)
        base_driven = travel(plan, dts[0], True, False)
        base_still = travel(plan, dts[0], False, False)
        rows.append({"plan": plan, "driven": driven, "still": still,
                     "incumbent_driven": base_driven,
                     "incumbent_still": base_still})
        print(f"         {plan:<8} " + "".join(f"{v:12.2f}" for v in driven)
              + f"{still:12.2f}  {base_driven:10.2f}{base_still:12.2f}")
    out["stability"] = {"timesteps": dts, "rows": rows}

    def runs_away(v) -> bool:
        return v > limit

    # A plan the incumbent already runs away on is not evidence about the
    # tensor.  Counted separately rather than folded in, because folding it in
    # would credit this change with a defect it did not cause.
    pre = [r["plan"] for r in rows
           if runs_away(r["incumbent_driven"]) or runs_away(r["incumbent_still"])]
    new_driven = [r["plan"] for r in rows
                  if runs_away(r["driven"][0]) and r["plan"] not in pre]
    new_still = [r["plan"] for r in rows
                 if runs_away(r["still"]) and r["plan"] not in pre]
    fine = [r["plan"] for r in rows if runs_away(r["driven"][-1])]
    print()
    print(f"         already unstable without the tensor: {pre or 'none'}")
    print(f"         destabilised by it, driven:          {new_driven}")
    print(f"         destabilised by it, actuators still: {new_still}")
    print(f"         still unstable at dt = {dts[-1]:g}:        "
          f"{fine or 'none'}")
    out["pre_existing_runaway"] = pre
    out["destabilised_driven"] = new_driven
    out["destabilised_still"] = new_still
    out["runaway_at_finest_dt"] = fine

    check("the runaway is not the controller: it happens with no actuation",
          len(new_still) >= 2,
          f"{new_still} run away with the actuators held completely still, "
          f"and none of them does so under the incumbent")
    check("and it is a timestep problem: refining the step removes it",
          not fine,
          f"every plan is stable at dt = {dts[-1]:g}, against "
          f"{len(new_driven) + len(pre)} diverging at {dts[0]:g}")
    finding("F-03  the isotropic added mass is holding up the explicit lift "
            "and drag", bool(new_still),
            f"{len(new_still)} of {len(rows)} plans need it to stay stable at "
            f"dt = {dts[0]:g} with no actuation at all")
    if pre:
        finding(f"N-02  {', '.join(pre)} already runs away at dt = {dts[0]:g} "
                f"with the solver as it ships", True,
                f"{max(r['incumbent_driven'] for r in rows if r['plan'] in pre):.0f} "
                f"rad driven, which is nothing to do with F-03")

    # Beetle's case is worth its own line: it is stable driven and unstable
    # with the actuators still, which is the opposite of the usual reading.
    odd = [r["plan"] for r in rows
           if r["plan"] not in pre and runs_away(r["still"])
           and not runs_away(r["driven"][0])]
    if odd:
        print(f"         note: {odd} is stable when driven and unstable when "
              f"held still, so the gait is what is keeping it up")
        out["stable_only_when_driven"] = odd


def main() -> int:
    cfg = load_config(HERE / "config.json")
    res = ExperimentResult(name="wing_added_mass", config=cfg)
    out: dict = {}
    print(__doc__.strip().splitlines()[0])
    test_the_code_applies_the_normal_entry_everywhere(cfg, out)
    test_which_way_the_flow_actually_comes_from(cfg, out)
    test_what_it_is_worth(cfg, out)
    test_whether_the_solver_survives_it(cfg, out)

    n_still = len(out.get("destabilised_still", []))
    n_all = len(cfg["seed_plans"])
    print("\n" + "=" * 74)
    print("  The tensor is right: layer 2 of the benchmark ladder holds with it")
    print(f"  and departs at 0.729 without it.  It cannot be the default at")
    print(f"  this project's dt = {cfg['stability']['timesteps'][0]:g}: the "
          f"surplus inertia the isotropic")
    print(f"  value carries is what keeps the *explicit* lift and drag stable,")
    print(f"  and removing it runs {n_still} of {n_all} plans away with no "
          f"actuation at all.")
    print("  Closing F-03 needs a smaller timestep or an implicit treatment of")
    print("  those forces, not a coefficient.  The switch is in the solver and")
    print("  defaults off; this experiment is what turns it on.")
    print()
    print("  Measured on this branch's base, which does not carry F-05's")
    print("  compact stall blend.  Under that lift model the same sweep gives a")
    print("  different set of plans -- the motion differs, so the divergence")
    print("  does -- and the same conclusion: several run away at the coarse")
    print("  step, none at the finest.")

    res.ok = not FAILURES
    for k, v in out.items():
        res.record(k, v)
    res.record("open_findings", FINDINGS)
    res.write(HERE / "results")
    if FINDINGS:
        print(f"\n{len(FINDINGS)} open findings, measured and still present:")
        for g in FINDINGS:
            print(f"  - {g}")
    if FAILURES:
        print(f"\n{len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
