#!/usr/bin/env python3
"""Break the code on purpose and find out which tests notice.

Why
---

A test suite's own output cannot tell you whether it is effective.  Nineteen
files, 13,308 lines and every suite green is compatible with a suite that would
stay green through a sign error in the policy loss.  The only measurement that
separates the two is to introduce the sign error and look.

So each entry below is a *defect this project could plausibly ship*: a dropped
recursion in GAE, a clip that does not clip, an optimiser step that never runs,
a reverted fix.  The harness applies one at a time to a throwaway copy of the
tree, runs the suites that ought to catch it, and reports `caught` or
`SURVIVED`.  A survivor is not a bug in the code -- the code is fine once the
mutation is reverted -- it is a **hole in the tests**, named precisely enough to
close.

What it deliberately does not do
--------------------------------

It does not mutate at random.  Random operator mutation (`+` to `-`, `<` to
`<=`) generates mostly equivalent or absurd variants and buries the signal in
triage.  Every mutation here is hand-written to model a real failure mode, and
carries the suites that *should* see it, so "caught by nothing" and "caught by
the wrong suite" are different results.

It never touches the working tree.  The tree is copied with `git ls-files`, so
uncommitted edits to tracked files are included and nothing that is not tracked
is.  A mutation that does not apply exactly once is reported as `MISAPPLIED`
rather than silently skipped -- a mutation harness whose mutations quietly fail
to apply reports a perfect score.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/mutate.py            # every mutation
    PYTHONPATH=. .venv/bin/python tools/mutate.py --only gae # by id substring
    PYTHONPATH=. .venv/bin/python tools/mutate.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():                                # pragma: no cover
    PYTHON = sys.executable


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and who ought to notice it."""

    id: str
    path: str
    find: str
    replace: str
    #: What the mutation models, in the terms a reviewer would use.
    defect: str
    #: Suites that should fail.  The first that fails ends the run for this
    #: mutation, so put the cheapest one first.
    suites: tuple
    #: Which checklist item this is evidence for.
    item: str = ""


@dataclass
class Result:
    mutation: Mutation
    status: str                 # "caught" | "SURVIVED" | "MISAPPLIED" | "ERROR"
    by: str = ""                # the suite that caught it
    detail: str = ""
    seconds: float = 0.0
    failing_checks: list = field(default_factory=list)


# --------------------------------------------------------------------------
# The mutations.  Grouped by what they model.
# --------------------------------------------------------------------------

MUTATIONS: tuple = (

    # --- the learner's arithmetic -----------------------------------------
    Mutation(
        id="gae-drops-the-recursion",
        path="dytiscidae/learning/ppo.py",
        find="                last = delta + self.gamma * self.lam * last",
        replace="                last = delta",
        defect="GAE degenerates to the one-step TD error at every lambda",
        suites=("test_ppo",), item="C/D advantage"),
    Mutation(
        id="gae-bootstraps-past-the-end",
        path="dytiscidae/learning/ppo.py",
        find="                nxt = val[i + 1] if i + 1 < n else 0.0",
        replace="                nxt = val[i + 1] if i + 1 < n else val[i]",
        defect="the terminal bootstrap is the last value instead of zero",
        suites=("test_ppo",), item="D target/bootstrap"),
    Mutation(
        id="return-loses-the-baseline",
        path="dytiscidae/learning/ppo.py",
        find="            RET.append(adv + val)",
        replace="            RET.append(adv)",
        defect="the value target is the advantage, not advantage plus value",
        suites=("test_ppo",), item="D return"),
    Mutation(
        id="gamma-ignored-in-shaping",
        path="dytiscidae/learning/ppo.py",
        find="                rew = rew + self.shaping * (self.gamma * nxt - phi)",
        replace="                rew = rew + self.shaping * (nxt - phi)",
        defect="potential shaping stops telescoping, so it changes the optimum",
        suites=("test_ppo",), item="D reward shaping"),
    Mutation(
        id="terminal-scale-disabled",
        path="dytiscidae/learning/ppo.py",
        find="        return {k: max(float(np.std(v)), 0.05) for k, v in by.items()}",
        replace="        return {k: 1.0 for k, v in by.items()}",
        defect="per-segment-kind reward scaling is off; water sets the gradient",
        suites=("test_ppo",), item="D reward"),

    # --- the update -------------------------------------------------------
    Mutation(
        id="clip-does-not-clip",
        path="dytiscidae/learning/ppo.py",
        find="                ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()",
        replace="                ratio * a, ratio * a).mean()",
        defect="PPO's clipped surrogate becomes the unclipped one",
        suites=("test_ppo",), item="J/PPO clip"),
    Mutation(
        id="policy-loss-sign-flipped",
        path="dytiscidae/learning/ppo.py",
        find="            pi_loss = -torch.min(",
        replace="            pi_loss = torch.min(",
        defect="the policy ascends away from the reward",
        suites=("test_ppo",), item="C gradient"),
    Mutation(
        id="optimiser-never-steps",
        path="dytiscidae/learning/ppo.py",
        find="            opt.step()",
        replace="            pass  # opt.step()",
        defect="the update computes a gradient and never applies it",
        suites=("test_ppo",), item="C parameter update"),
    Mutation(
        id="anneal-never-reaches-the-optimiser",
        path="dytiscidae/learning/ppo.py",
        find="        group[\"lr\"] = lr_now",
        replace="        group[\"lr\"] = float(lr)",
        defect="the annealed rate is reported but the optimiser keeps the base rate",
        suites=("test_ppo",), item="C scheduler"),
    Mutation(
        id="entropy-reverted-to-cross-entropy",
        path="dytiscidae/learning/ppo.py",
        find="        return self.entropy_of(self.latent(obs))",
        replace="        return -self._log_prob_under(\n"
                "            self.latent(obs), torch.tanh(self.latent(obs).sample())).mean()",
        defect="the entropy bonus goes back to having no expected gradient",
        suites=("test_ppo",), item="J regression / D entropy"),
    Mutation(
        id="kl-reverted-to-the-naive-estimator",
        path="dytiscidae/learning/ppo.py",
        find="                kl = float((logr.exp() - 1.0 - logr).mean().item())",
        replace="                kl = float((-logr).mean().item())",
        defect="the KL that target_kl stops on can go negative again",
        suites=("test_ppo",), item="J regression"),
    Mutation(
        id="non-finite-guard-removed",
        path="dytiscidae/learning/ppo.py",
        find="    if nonfinite:\n        return {\"transitions\": n,",
        replace="    if False:\n        return {\"transitions\": n,",
        defect="a NaN reward is consumed and poisons every weight again",
        suites=("test_ppo",), item="J regression / E NaN"),

    # --- the policy -------------------------------------------------------
    Mutation(
        id="tanh-jacobian-dropped",
        path="dytiscidae/learning/ppo.py",
        find="        return (base.log_prob(u).sum(-1)\n"
             "                - torch.log1p(-a.pow(2) + 1e-6).sum(-1))",
        replace="        return base.log_prob(u).sum(-1)",
        defect="the squashed density loses its change-of-variables term",
        suites=("test_ppo",), item="C correctness"),
    Mutation(
        id="observation-normaliser-does-not-scale",
        path="dytiscidae/learning/ppo.py",
        find="            (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)",
        replace="            (obs - self.obs_mean), -10.0, 10.0)",
        defect="observations are centred but not scaled",
        suites=("test_ppo",), item="B numerical oracle"),
    Mutation(
        id="running-variance-is-wrong",
        path="dytiscidae/learning/ppo.py",
        find="        m2 = (self.obs_var * self.obs_count + b_var * b_n\n"
             "              + delta.pow(2) * self.obs_count * b_n / total)",
        replace="        m2 = self.obs_var * self.obs_count + b_var * b_n",
        defect="the parallel variance update drops its correction term",
        suites=("test_ppo",), item="B numerical oracle"),

    # --- reproducibility --------------------------------------------------
    Mutation(
        id="torch-seed-removed-from-run-search",
        path="dytiscidae/evolution/loop.py",
        find="        _torch.manual_seed(int(cfg.seed))",
        replace="        pass  # _torch.manual_seed(int(cfg.seed))",
        defect="the shared policy's initial weights stop following --seed",
        suites=("test_ppo",), item="J regression / H seed"),
    Mutation(
        id="learner-rng-not-checkpointed",
        path="dytiscidae/evolution/loop.py",
        find="    if state.learner_rng is not None:\n"
             "        payload[\"learner_rng_state\"] = state.learner_rng.bit_generator.state",
        replace="    if False:\n"
                "        payload[\"learner_rng_state\"] = state.learner_rng.bit_generator.state",
        defect="a resume takes a different sequence of gradient steps",
        suites=("test_ppo",), item="J regression / H random state"),
    Mutation(
        id="torch-rng-not-checkpointed",
        path="dytiscidae/evolution/loop.py",
        find="            payload[\"torch_rng_state\"] = (\n                _torch.get_rng_state().numpy().copy())",
        replace="            payload[\"torch_rng_state\"] = None",
        defect="a resume explores with a different noise stream",
        suites=("test_ppo",), item="J regression / H random state"),
    Mutation(
        id="update-shuffles-from-the-global-stream",
        path="dytiscidae/learning/ppo.py",
        find="    shuffler = rng if rng is not None else np.random.default_rng()",
        replace="    shuffler = np.random.default_rng()",
        defect="the caller's stream is ignored, so the update is not reproducible",
        suites=("test_ppo",), item="H reproducibility"),

    # --- physics ----------------------------------------------------------
    Mutation(
        id="hull-buckling-exponent",
        path="dytiscidae/physics/structure.py",
        find="    return knockdown * (n**2 - 1) * e_prime * wall**3 / (",
        replace="    return knockdown * (n**2 - 1) * e_prime * wall**2 / (",
        defect="the buckling allowable loses its cubic wall-thickness dependence",
        suites=("test_hull_buckling", "test_math"), item="C formula"),
    Mutation(
        id="jet-thrust-is-linear-in-flow",
        path="dytiscidae/physics/jet.py",
        find="        thrust_mag = coeff * rho * q * np.abs(q) / area",
        replace="        thrust_mag = coeff * rho * np.abs(q) / area",
        defect="jet thrust stops being a momentum flux",
        suites=("test_jet_energy",
                "test_physics::test_jet_thrust_matches_momentum_flux"),
        item="C formula"),

    # --- the job layer ----------------------------------------------------
    Mutation(
        id="job-accepts-any-transition",
        path="dytiscidae/domain/job.py",
        find="        if to not in _ALLOWED[self.status]:",
        replace="        if False:",
        defect="the job lifecycle stops rejecting impossible transitions",
        suites=("test_domain",), item="A/D state machine"),

    # --- the index --------------------------------------------------------
    Mutation(
        id="index-generator-drifts",
        path="tools/index_gen.py",
        find="             \"is in `CLAUDE.md`.\", \"\"]",
        replace="             \"is in `CLAUDE.md`.\", \"\", \"drifted\"]",
        defect="the committed index stops matching what the source generates",
        suites=("test_index",), item="A/J index gate"),
)


# --------------------------------------------------------------------------


def copy_tree(dest: Path) -> None:
    """Every tracked file, including uncommitted edits to tracked files."""
    names = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                           capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(
        ["tar", "--null", "-T", "-", "-cf", "-"], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    untar = subprocess.Popen(["tar", "-xf", "-", "-C", str(dest)],
                             stdin=tar.stdout)
    tar.stdout.close()
    tar.stdin.write(names)
    tar.stdin.close()
    untar.communicate()
    if untar.returncode:                                     # pragma: no cover
        raise RuntimeError("copying the tree failed")


def apply_mutation(tree: Path, m: Mutation) -> str:
    """Apply it, or return why it could not be applied exactly once."""
    p = tree / m.path
    if not p.exists():
        return f"{m.path} does not exist"
    text = p.read_text(encoding="utf-8")
    n = text.count(m.find)
    if n != 1:
        return f"the target text appears {n} times in {m.path}, expected 1"
    p.write_text(text.replace(m.find, m.replace), encoding="utf-8")
    return ""


#: Driver for ``module::function``.  The suites keep their failures in a module
#: -level ``FAILURES`` list and print through ``check``, so one function can be
#: run on its own by importing the module and reading that list afterwards.
_ONE_FUNCTION = """
import importlib.util, sys, traceback
sys.path.insert(0, {tree!r})
# By path, not by package: tests/ has no __init__.py, so it is not importable
# as `tests.<module>` and pretending otherwise fails before the mutation is
# ever exercised -- which would read as a survivor.
spec = importlib.util.spec_from_file_location(
    "suite_under_test", {tree!r} + "/tests/{module}.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
before = len(getattr(mod, "FAILURES", []))
try:
    getattr(mod, "{func}")()
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
fails = getattr(mod, "FAILURES", [])[before:]
if fails:
    print("[FAIL] " + "; ".join(fails))
    raise SystemExit(1)
raise SystemExit(0)
"""


def run_suite(tree: Path, suite: str, timeout: int = 900) -> tuple:
    """(returncode, stdout+stderr).  Same invocation a contributor uses.

    ``module::function`` runs one test function instead of the whole file.
    That exists because two suites here -- ``test_physics`` and ``test_search``
    -- stop partway on a machine with no GPU fluid extension, and without this
    every mutation in the code they cover would be reported as surviving when
    what actually happened is that the check never ran.  "Not run" and "did not
    catch it" are different results and this is what keeps them apart.
    """
    env = dict(os.environ, PYTHONPATH=str(tree), MUJOCO_GL="disable")
    if "::" in suite:
        module, func = suite.split("::", 1)
        script = _ONE_FUNCTION.format(tree=str(tree), module=module, func=func)
        cmd = [PYTHON, "-c", script]
    else:
        cmd = [PYTHON, str(tree / "tests" / f"{suite}.py")]
    try:
        out = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def failing_checks(output: str) -> list:
    return [ln.strip() for ln in output.splitlines() if "[FAIL]" in ln]


def evaluate(m: Mutation, workdir: Path) -> Result:
    t0 = time.time()
    tree = workdir / m.id
    copy_tree(tree)
    problem = apply_mutation(tree, m)
    if problem:
        return Result(m, "MISAPPLIED", detail=problem,
                      seconds=time.time() - t0)

    for suite in m.suites:
        rc, out = run_suite(tree, suite)
        if rc != 0:
            fails = failing_checks(out)
            detail = fails[0] if fails else out.strip().splitlines()[-1][:160]
            return Result(m, "caught", by=suite, detail=detail,
                          seconds=time.time() - t0, failing_checks=fails)
    return Result(m, "SURVIVED",
                  detail="no named suite failed: " + ", ".join(m.suites),
                  seconds=time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="run mutations whose id contains this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default="", help="write results here")
    args = ap.parse_args(argv)

    chosen = [m for m in MUTATIONS if args.only in m.id]
    if args.list:
        for m in chosen:
            print(f"{m.id:42} {m.path:34} {m.item}")
        return 0
    if not chosen:
        print(f"no mutation matches {args.only!r}")
        return 1

    print(f"{len(chosen)} mutations, each on its own copy of the tree\n")
    workdir = Path(tempfile.mkdtemp(prefix="dyt-mutate-"))
    results = []
    try:
        for m in chosen:
            r = evaluate(m, workdir)
            results.append(r)
            mark = {"caught": "caught  ", "SURVIVED": "SURVIVED",
                    "MISAPPLIED": "MISAPPLD", "ERROR": "ERROR   "}[r.status]
            where = f" by {r.by}" if r.by else ""
            print(f"  [{mark}] {m.id:42} {r.seconds:5.1f}s{where}")
            if r.status == "caught":
                print(f"             {r.detail[:150]}")
            elif r.status != "caught":
                print(f"             {r.detail[:150]}")
            # Free the copy as we go: 24 trees is a lot of disk.
            shutil.rmtree(workdir / m.id, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    caught = sum(1 for r in results if r.status == "caught")
    survived = [r for r in results if r.status == "SURVIVED"]
    bad = [r for r in results if r.status in ("MISAPPLIED", "ERROR")]

    print("\n" + "=" * 70)
    print(f"caught {caught}/{len(results)}"
          + (f", {len(survived)} SURVIVED" if survived else "")
          + (f", {len(bad)} could not be applied" if bad else ""))
    for r in survived:
        print(f"  SURVIVED  {r.mutation.id}: {r.mutation.defect}")
    for r in bad:
        print(f"  {r.status}  {r.mutation.id}: {r.detail}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"id": r.mutation.id, "path": r.mutation.path,
              "defect": r.mutation.defect, "item": r.mutation.item,
              "status": r.status, "by": r.by, "detail": r.detail,
              "seconds": round(r.seconds, 2)} for r in results],
            indent=2) + "\n")
        print(f"\nwrote {args.json}")

    # A survivor is a reportable hole, not a crash; a misapplied mutation is a
    # broken harness and must be loud.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
