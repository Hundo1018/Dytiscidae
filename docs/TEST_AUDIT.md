# Test audit — are these tests worth anything?

18 suites, 13,308 lines, 1,114 assertions, every one of them green. This is the
audit of whether that means anything, against a twelve-section checklist.

Written 2026-09-19 against `1f8b474`. Like `docs/LEARNER_AUDIT.md` and unlike
`docs/MATH_AUDIT.md`, it **changed things**, because the central measurement came
back negative: a suite reporting 54 green checks stayed green with PPO's
clipping removed entirely.

## The instruments

Four, all in `tools/`, all runnable without a GPU. They answer different
questions and none of them substitutes for another.

| | asks | answers |
|---|---|---|
| `mutate.py` | if I break this, does anything notice? | 23 modelled defects, one throwaway tree each |
| `suite_probe.py` | which test functions can run here at all? | each one in its own interpreter |
| `assertion_audit.py` | which assertions cannot fail? | AST shape of all 1,114 `check(...)` conditions |
| `coverage_report.py` | which lines does nothing execute? | line + branch, per package, 0% modules named |

Coverage and mutation are the pair that matters: **coverage finds code nothing
touches, mutation finds code nothing judges.** A file of `check(name, True)`
reaches 100% coverage and survives every mutation.

## What could not be measured here

Same standing constraint as the previous audit: this container has no GPU and no
`pixi`, so the Mojo fluid extension is not importable. What changed is that the
cost is now *counted* rather than guessed. `tools/suite_probe.py` ran every test
function in its own interpreter:

| suite | functions | pass here | blocked | timeout |
|---|---:|---:|---:|---:|
| `test_physics.py` | 37 | **36** | 1 | 0 |
| `test_search.py` | 45 | **37** | 7 | 1 |

So `test_physics.py` never needed a GPU. It needed MuJoCo, one function skipped,
and a `main()` that does not abort. That is why it is in CI now.

---

# The findings

## 1. Removing PPO's clipping left all 54 checks green

`min(r·A, clamp(r, 1−c, 1+c)·A)` replaced by `min(r·A, r·A)` — the unclipped
surrogate, i.e. importance-weighted vanilla policy gradient — and
`tests/test_ppo.py` did not notice. The previous audit stated "ratio, clip, GAE,
entropy and value loss are all covered"; **that claim was wrong**, and only
mutation testing could have said so.

Closed by `test_the_clip_actually_clips`, whose oracle is the `clip` argument
rather than a re-implementation of the objective: a tight clip and a loose one
must take the actor to different places. Measured: ‖dW‖ = 0.171 between
`clip=0.05` and `clip=10.0`, against **exactly 0.000000** with the clamp
removed. `clipfrac` is asserted nonzero on the tight arm so the test cannot pass
without having entered the clipping regime at all.

## 2. A check on `torch.manual_seed` passed when the call was commented out

The previous audit's reproducibility check read `inspect.getsource(run_search)`
and looked for the string `manual_seed`. Commenting the call out leaves the
string in the source *as a comment*, so the check passed on a `run_search` that
seeded nothing. **A test whose oracle is the text of the code cannot tell a call
from a mention of one.**

Closed by running the real `run_search` and reading the weights it built, with
`seed_archipelago` swapped for a probe that grabs them and stops before the
first evaluation — which is also why it needs no MuJoCo. The ambient torch state
is deliberately *different* before each call, so matching weights can only come
from `run_search` having seeded torch itself: ‖dW‖ = 0.000e+00 at one seed,
10.3987 across two.

## 3. Two suites aborted on one blocked function and lost the rest

`main()` in `test_physics.py` and `test_search.py` was a flat list of calls, so
the first function that raised ended the file. Everything after it silently
never ran, and the console showed a traceback rather than a list of what was
lost.

| suite | before | now | exit code before | now |
|---|---:|---:|---|---|
| `test_physics.py` | 144 checks, then a traceback | **203 checks, 1 skipped** | 1 | **0** |
| `test_search.py` | 133 checks, then a traceback | **286 checks, 7 skipped** | 1 | **0** |

Both now run every test through `run_all`, which reports a blocked function as
`[skip]`. Getting to a clean exit took two further steps, and the second is the
more interesting one.

**A function that stops partway must withdraw what it printed.** `run_all`
snapshots the failure count before each function and truncates back to it when
the function turns out to be blocked. Without that, letting the suites continue
surfaced five `[FAIL]` lines from functions that got halfway: a run that never
evaluated anything reports "0 promotions" and "0 tensors compared". Those say
nothing about the code.

**And some functions never raise at all.** `run_search` catches a failed
generation, records it in telemetry and carries on with nothing evaluated, so
four functions ran to completion and simply failed their own assertions. No
amount of exception handling in `run_all` can classify those. They now declare
the requirement at the top, through `needs_batched_evaluator()`, which is the
difference between a suite that reports what it could not run and one that
reports a defect it did not find.

## 4. A skip was rendered as a pass, in five places

`check(name, True, "SKIPPED: ...")` prints `[ok  ]` and counts as a pass. On the
runner CI actually uses — numpy and torch, no MuJoCo — that made "shared PPO
checks passed" the summary of a run where **48 checks ran against 56 here**, and
the missing eight said nothing about themselves.

Five sites, all now `[skip]` with a separate count: `test_ppo.py` ×2,
`test_architecture.py` ×1, `test_search.py` ×2. `test_search_adapter.py`'s
whole-suite skip now ends with `ENTIRE SUITE SKIPPED, nothing was checked`
rather than a line that reads like success.

**Every success line is now qualified.** `all physics checks passed` appears
only when nothing was skipped; otherwise it is `all physics checks passed,
1 skipped`. That is what makes the bare string safe to grep for, which
`CLAUDE.md` tells the next reader to do.

## 5. Ten assertions could not fail

`tools/assertion_audit.py` found twelve conditions that are vacuous by shape.
Two were false positives of the first version of the scanner and are now
classified correctly — a scanner with false positives gets ignored along with
its true ones:

* `JobId("x") == JobId("x")` constructs two objects and is a real test of `__eq__`;
* `check(name, True)` inside an `except` handler is the "it raised, as required"
  idiom, with the failing branch in the `try`.

The other **ten were real** and are fixed:

| where | was | now |
|---|---|---|
| `test_search.py` ×2 | skip-as-pass when torch is missing | `[skip]` |
| `test_hull_buckling.py` | a body plan with no hull counted as a pass | `[skip]` |
| `test_search.py` | `check(..., True)  # covered by construction; asserted below` | the assertion, naming both operators |
| `test_hull_buckling.py`, `test_stall_blend.py` | loop with early return + trailing `check(..., True)` | one check carrying the worst point and the count |
| `test_adapters.py` ×3 | "delete is idempotent", `True` — only that it did not raise | the listing is unchanged after the second delete |
| `test_adapters.py` | a lock check that accepted all three outcomes | the two legitimate ones as a disjunction, plus "the outer holder still has its lock" |

The last one is the clearest case of checklist item B, "沒有只檢查不 crash": the
comment above it said what must not happen — a silent acquire while another
holder is inside — and the literal `True` accepted exactly that.

**After the fixes and the classifier's two corrections: 0 always-true.**

## 6. The only CPU↔GPU numeric comparison cannot run in CI, or here

This is the standing gap and it is not closed.

`tests/test_gpu_mirror.py` is honest about its own scope — its docstring says it
catches "one class of drift, not all of it" — and it compares the *text* of the
constants in `fluid.py` and `fluid_gpu.mojo`. It is not, and does not claim to
be, a check that the two compute the same thing.

The check that the two agree *numerically* is
`test_search.py::test_the_two_evaluation_paths_score_the_same_machine_the_same`,
and it needs the extension. So on every machine and every runner that does not
have a GPU, **nothing verifies that the Mojo kernels and the numpy solver
compute the same physics.** `tools/mutate.py` cannot probe the Mojo side at all.

Not fixable here. Named so it is not mistaken for covered.

---

## 7. The mutation harness never ran a batched test, then counted a timeout as a catch

Found 2026-09-21 while adding the task mutations. Two defects in
`tools/mutate.py`, one hiding the other.

**The mutant tree had no GPU kernel.** `copy_tree` copies `git ls-files`, and
`mojo/build/*.so` is untracked, so in every mutant copy the batched evaluator
was unavailable and every test that drives it *skipped*. A skip exits 0, which
the harness reads as "no suite failed". So no mutation in the batched path was
ever tested by a batched test: its verdict came from whatever unbatched checks
happened to notice. `copy_tree` now links the kernel in; its freshness is still
checked against the copy's own `mojo/src`.

**A timeout was a catch.** `evaluate` treated any non-zero exit as `caught`, and
a timeout returned 124. With the kernel present `test_search` runs ~25 min
against a flat 900 s limit, so the first three batched-path mutations run after
the fix came back `caught` at 900.2 s, 900.1 s and 900.2 s with no failing check
between them. A timeout is now its own status, counted as a broken measurement
(non-zero exit), and `test_search` gets 3600 s.

What it invalidates: any past `caught` for a mutation whose only catching suite
is `test_search` is evidence only if its failing check was named and it ran
under 900 s *without* the batched tests -- i.e. a catch by an unbatched check.
Re-run them.

**Re-run, 2026-09-30.** All 17 mutations that still named the whole
`test_search` suite now name the one function that catches them
(`suites=("test_search::<function>",)`, run by `mutate.py`'s `module::function`
driver), and each was run on its own, on a tree at `main` `1fd0a24`, with the
kernel linked in. Every one prints `caught 1/1` with a named `[FAIL]` check; no
survivor, none caught only by the whole suite, no timeout. Each function was
also run on the unmutated tree first and passes with no `[fail]` and no
`[skip]`, so a catch is a check that fails on the mutant and not on the code.

| mutation | catching function in `tests/test_search.py` | time | failing check |
|---|---|---|---|
| gait-drops-phase-and-rest | `test_the_gait_operator_moves_every_coordinate_at_once` | 0.3 s | every actuated part's amplitude, phase and rest offset (phase 0/90, rest 0/90) |
| gait-touches-one-part | same | 0.2 s | same (amp 30/90, phase 30/90, rest 30/90) |
| continuous-mission-ignores-auto-reset | `test_a_mujoco_auto_reset_ends_the_continuous_mission` | 2.1 s | an auto-reset mid-mission fails it, as 'unstable' (survived=True) |
| island-best-reads-the-best-domain | `test_an_islands_best_is_judged_on_its_own_domains` | 0.2 s | an amphibian that cannot walk scores nothing on the amphibian island (0.463 vs 0.354) |
| island-archive-read-through-the-merge | `test_one_islands_archive_is_read_alone_not_through_the_merge` | 0.2 s | read alone, the island keeps every elite it filed |
| curriculum-reads-every-medium | `test_a_specialist_islands_curriculum_reads_only_its_own_medium` | 0.3 s | restricted, stage 0 reads air and nothing else (0.9) |
| bandit-without-an-exploration-floor | `test_no_operator_can_go_dormant_under_the_structural_tilt` | 0.8 s | every operator makes at least 1% of children (phase_gradient 0.0%) |
| checkpoint-asks-git-at-every-write | `test_a_checkpoint_names_the_commit_the_process_started_from` | 0.2 s | a checkpoint written after a mid-run commit still names the launch commit |
| stale-kernel-used-anyway | `test_a_kernel_older_than_its_source_is_not_usable` | 0.3 s | a stale kernel is not usable, and says so |
| evaluators-ask-different-tasks | `test_the_two_evaluation_paths_score_the_same_machine_the_same` | 95.4 s | beetle: land and air agree to floating-point between the paths (worst 2.63 on land.cmd_heading) |
| pool-drops-the-shared-policy-from-rescores | `test_sharding_a_generation_does_not_change_a_score` | 180.1 s | a re-score in three shards drives the shared policy as one shard does (max difference 0.054) |
| film-ignores-the-evaluation-seed | `test_a_film_reproduces_the_scored_experiment` | 296.8 s | water: the film reproduces the recorded competence (recorded 0.0, film 0.082) |
| refit-gate-never-keeps | `test_a_refit_that_changes_nothing_can_be_skipped` | 0.2 s | with it on, identical axes are kept and nothing is re-binned |
| batched-fluid-diagnostics-stay-default | `test_the_batched_path_tells_the_policy_it_is_wet` | 3.4 s | beetle: submerged in water reads submerged, not 0.0 |
| stage-one-reads-gross-measurements | `test_a_specialist_islands_curriculum_reads_only_its_own_medium` | 0.2 s | stage 1 reads only the island's own directed measure (0.200 vs 1.000) |
| caller-moves-state-under-launch | `test_the_early_fluid_launch_changes_nothing` | 32.4 s | every measurement bit-identical with the launch early and in place (346 numbers moved) |
| nan-observation-reaches-policy | `test_a_nan_observation_fails_the_rollout_not_the_batch` | 10.5 s | a NaN observation does not raise out of the batch (ValueError from the policy) |

Two things the pass found. **`gait-touches-one-part` was not applying**: its
`find` text still had `rng.uniform(1.5, 8.0)` and `mut_gait` has drawn from 1.5-12
Hz since 2026-09-23 (ROADMAP AC), so the harness reported `MISAPPLIED`, a broken
measurement rather than a catch. The text is updated and it applies. **The first
function picked for `stage-one-reads-gross-measurements`,
`test_curriculum_and_islands_give_gradient_where_the_mission_gives_none`, did not
catch it**: the mutation ran to completion with no failing check, a survivor of
that function and not of the suite. The mutation edits `stage_score`'s stage-1
directed measure, and the check on it lives one function over, in the island
curriculum test, which does catch it. The suite as a whole was never run against
this mutation. Speed: the seventeen now take about 10 minutes in total against
17 x 25 minutes.

# The checklist, item by item

## A. Completeness

Coverage is in §K. Every package of `dytiscidae/` is named by at least one
feature in `docs/index/FEATURES.yaml`, and `tests/test_index.py` fails the build
if one stops being — that is the gate against "a new package nobody tested".

Normal paths, error paths and boundaries: the first two were already covered
broadly; boundaries were not, and `test_the_batch_boundaries` is new — a
one-step trajectory (advantage 19.75 against 19.75 by hand), a batch of one
transition (refused, policy untouched), a batch of two (the smallest it acts
on), and negative / 1e9 / all-zero rewards each leaving the update finite.

## B. Are the tests effective

| | |
|---|---|
| every test has a definite expected result | 1,114 assertions, 0 that cannot fail |
| not only "does not crash" | 3 fixed in `test_adapters.py`; the lock check now asserts its own comment |
| no mass of `assert x is not None` | 12 remain, each a presence check followed by a content check |
| shape without value | 5 remain, each in a place where the shape *is* the claim (an empty basis returns an empty command) |
| a reliable oracle | GAE against its double-sum definition written independently in the test; jet thrust against ρQ²/A by hand; hull buckling against the n=2 ring result |
| failures point at the problem | every `check` carries the measured number in its detail |
| the test does not re-implement what it tests | the reference GAE is the *definition*, the implementation is the backward recursion — two different computations of one quantity |

## C. Algorithm correctness

Loss, gradient, parameter update, optimiser and scheduler are each covered by a
mutation that a test catches: `policy-loss-sign-flipped`,
`optimiser-never-steps`, `anneal-never-reaches-the-optimiser`,
`clip-does-not-clip`, `tanh-jacobian-dropped`. Target and bootstrap:
`gae-bootstraps-past-the-end`, `return-loses-the-baseline`.

Tolerances are stated rather than default: 1e-12 on the reference GAE,
`rtol=1e-9` on jet thrust, 1e-12 on hull buckling, 1e-4 on the importance ratio
(measured at 2.9e-6, with the saturating boundary asserted at the other end).

## D. RL specifics

`reset`, `step`, observation, action, reward, episode boundary, rollout, buffer
insert and sample, return, advantage, policy update, value update and entropy
all have a test and, except where noted in §F, a mutation. `terminated` versus
`truncated` is **BY DESIGN**, recorded in `docs/LEARNER_AUDIT.md`: a segment is a
fixed-duration window whose entire reward arrives at its end and whose machine is
re-placed afterwards, so there is no continuation to bootstrap.

## E. Boundaries

Batch 1 and >1, a one-step episode, reward = 0, < 0 and 1e9, γ = 0 and γ → 1,
an empty buffer, NaN and Inf: all covered, most by `test_the_batch_boundaries`
and `test_the_discount_behaves_at_both_ends`. "Buffer full" is **N/A** — the
buffer is on-policy, holds one generation and is rebuilt each time, so there is
no eviction policy to test.

## F. CPU / GPU

CPU is tested. The GPU kernels are not, here — see §6 above. `PASS`, `SKIP` and
`NOT RUN` are now distinguished everywhere, which is the part that was fixable.
dtype and device transfer have no test; the learner is CPU-only by measurement,
and the evaluator's device handling is inside the part that needs the hardware.

## G. Integration

Environment → rollout → buffer → algorithm → policy, the full trainer loop,
evaluation, and checkpoint save / load / resume are covered by
`test_the_loop_wires_every_layer_together`, `test_a_run_can_be_picked_up_where_it_stopped`,
`test_a_finished_run_is_a_checkpoint` and
`test_both_learner_streams_survive_a_checkpoint`. The first three need the
extension; the last does not, and is the one that runs in CI.

## H. Reproducibility

Seed control, same-seed reproduction, saved random state and checkpoint resume
are each covered by a mutation that is caught (`torch-seed-removed-from-run-search`,
`learner-rng-not-checkpointed`, `torch-rng-not-checkpointed`,
`update-shuffles-from-the-global-stream`).

**Order independence is measured, not assumed.** `tools/suite_probe.py` runs
every test function in its own interpreter, and every function that can run here
passes there: 36/37 in `test_physics`, 37/45 in `test_search`, with the rest
blocked rather than failing. No function depends on another having run first.

## I. Quality of the tests themselves

| | |
|---|---|
| no network | zero uses of `requests`, `urllib`, `socket`, `http.client` in `tests/` |
| no wall-clock dependence | `time.sleep` appears in `test_adapters.py` and `test_worker.py` only, for process coordination; nothing asserts on a clock reading |
| no flaky tests | 5 runs each of those two suites **under load from a concurrent mutation sweep**: 123 ok / 88 ok every time, zero failures |
| no global-state pollution | the one monkeypatch (`seed_archipelago`) is restored in a `finally` |
| no arbitrary thresholds | the clip test's margin was 0.171 against a 0.000936 residual; switching the value loss off makes the mutated case **exactly zero**, so the threshold is no longer a number anyone has to keep calibrated |
| speed | one outlier: `test_the_shared_controller_question_is_answered_with_a_number` exceeded 900 s in isolation under load. It is the only one. |

## J. Mutation / failure injection

**23 modelled defects, 23 caught.** The path there is the finding:

| stage | caught | survived | harness fault |
|---|---:|---:|---:|
| first sweep | 18 | 2 | 3 |
| after fixing the harness's own targets | 20 | 3 | 0 |
| after the two new tests | **23** | **0** | 0 |

The three survivors were `clip-does-not-clip`, `torch-seed-removed-from-run-search`
and `jet-thrust-is-linear-in-flow`. The first two were holes in the tests and are
closed. The third was different: `test_physics.py::test_jet_thrust_matches_momentum_flux`
catches it (6150.000 N against 7.380 N by hand) — the suite simply could not run
end-to-end. That distinction is why `tools/mutate.py` can target a single test
function: **"caught by nothing" and "never ran" are different results**, and a
harness that conflates them reports holes that are not there.

Three mutations initially failed to apply at all, and were reported as
`MISAPPLIED` rather than skipped. A mutation harness whose mutations quietly
fail to apply reports a perfect score.

Every bug fixed in this session and the last has a mutation: the entropy
estimator, the KL estimator, the non-finite guard, both RNG streams, the clip,
the seeding.

## K. Coverage

Line and branch, every suite, `tools/coverage_report.py`:

| package | statements | line % | branches | branch % |
|---|---:|---:|---:|---:|
| `adapters` | 1480 | 71.0% | 398 | 88.9% |
| `application` | 468 | **97.6%** | 60 | 86.7% |
| `control` | 285 | 66.7% | 50 | 94.0% |
| `core` | 1762 | 95.1% | 482 | 89.2% |
| `domain` | 542 | 95.0% | 98 | 77.6% |
| `envs` | 2077 | 63.9% | 572 | 87.8% |
| `evolution` | 2225 | 86.2% | 684 | 82.2% |
| `learning` | 425 | 61.2% | 94 | 92.6% |
| `ops` | 739 | **35.6%** | 208 | 88.9% |
| `physics` | 906 | 87.6% | 114 | 88.6% |
| `ports` | 208 | **99.0%** | 0 | — |
| `viz` | 582 | **8.4%** | 158 | 100.0% |
| `worker` | 265 | 67.2% | 46 | 71.7% |
| **TOTAL** | **11,969** | **71.4%** | **2,966** | **87.3%** |

**0 modules at 0%.** Every module in the project is imported and executed by at
least one suite, which is the question worth asking — the percentage is not.

The two low numbers are honest and different:

* **`viz` at 8.4%** is rendering: `showcase`, `render`, `dashboard`. It writes
  MP4s and HTML and its output is judged by a person watching the film. Nothing
  here tests it and nothing plausibly could without a rendering oracle. It is
  also the package whose failure costs the least — a bad film wastes an
  afternoon, a bad reward function wastes a 21-hour run.
* **`ops` at 35.6%** is the CLI surface of `ops/run.py`: the `cmd_*` handlers
  for `showcase`, `render`, `distill`, `cohort`, `dashboard`. The job-layer CLI
  (`adapters/cli.py`) is separately covered at 71%. These are argument-parsing
  and dispatch, and a defect in them fails loudly on the first run.

`learning` at 61.2% is the one worth a second look: the uncovered part is the
`AVAILABLE = False` fallbacks that fire only when torch is absent, plus
`distill.py`, whose entry points are exercised only through `test_search`.
Branch coverage there is 92.6%, which is the reassuring half.

`ports` shows no branches because it is `Protocol` definitions with no logic —
the 0.0% in the raw output is 0/0, not a failure.

## L. Final review

| | |
|---|---|
| core features with no test | none: 0 modules at 0% coverage, and every package is named in `FEATURES.yaml` |
| superficial assertions | 10 found and fixed; 0 always-true remain |
| tests with no reliable oracle | the CPU↔GPU agreement, §6 — unreachable rather than absent |
| CPU-tested but GPU-unverified | the Mojo fluid kernels, §6 |
| integration gaps | the three checkpoint/resume integration tests need the extension; the learner-state half now does not |
| flaky tests | none found in 10 runs under load |
| survives mutation | nothing, of the 23 modelled |
| regression test per fixed bug | yes, and a mutation for each |
| SKIP / NOT RUN counted as PASS | fixed in 5 places; every success line now carries the skip count |

## M. The harness checks itself (IMPL_1010 M1, 2026-10-10)

`tools/mutate.py --check` applies no mutation: it counts each `find` in its file and checks that every named suite function exists (`static_problems`). `tests/test_index.py::test_every_mutation_applies_exactly_once` runs it, so a mutation that cannot apply, or names a missing test, fails a cheap suite. At 63b7ace 7 of 195 mutations were MISAPPLIED and nothing noticed: `seeds-scored-without-the-shared-policy` (2 matches), `turn-ignores-airborne`, `land-stop-adds-to-the-walk`, `evaluators-ask-different-tasks`, `single-path-crossing-unscattered`, `batched-fluid-diagnostics-stay-default`, `tier2-probe-legs-enter-the-mission` (0 matches each; the code under them had moved). Each is re-targeted on the current text and now `caught` by its named function. `seeds-scored-without-the-shared-policy` SURVIVED once re-targeted: its named test never exercised seed scoring, and the commit that added it (1929e43) added no test. `test_seeds_are_scored_with_the_shared_policy` (CPU, no GPU) now catches it. A new status, `NOT RUN`, covers a named function that skipped (it appends to `SKIPPED`; the driver exits 3). It fails the run like MISAPPLIED, where it used to read `SURVIVED`. Three mutations of the harness itself were added (`static-check-ignores-missing-text`, `static-check-ignores-missing-tests`, `skipped-function-reads-survived`). A mutation of `mutate.py` removes its own `find` from the file, so the first check in the test also fails for it; the other checks hold the intended cases.
