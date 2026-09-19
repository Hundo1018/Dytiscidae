# Learner audit — the RL half, against a standard checklist

Every claim a reinforcement-learning library normally makes about its own
structure and its own test coverage, checked against this project, with the
measurement or the command behind each verdict.

Written 2026-09-19 against `fcd6645`. Unlike `docs/MATH_AUDIT.md`, which was a
survey that changed nothing, this one **changed four things**, because four
checks came back measurably false rather than merely unproven. Each is named
below with what it was, what it is, and what it cost.

## How to read this

| status | means |
|---|---|
| `MEASURED` | an experiment in `experiments/` reproduces the number |
| `TESTED` | an invariant in `tests/` asserts it on every run |
| `HELD` | true by reading, and now gated by a test added for it |
| `BY DESIGN` | does not hold, deliberately, with the reason recorded |
| `N/A` | the checklist item does not apply and the reason is stated |
| `GAP` | does not hold, is not deliberate, and is not fixed here |

## What this machine could and could not run

`tests/test_physics.py` and `tests/test_search.py` — two of the three canaries
in `CLAUDE.md` — **did not run to completion** during this audit. Both reach
`envs/batchroll.evaluate_tier1_batch`, which requires the Mojo GPU fluid
extension (`cd mojo && pixi run build-all`) and a GPU to run it on. The container
this was done in has neither, and `pixi` is not installed:

```
RuntimeError: GPU fluid extension not importable
(ModuleNotFoundError: No module named 'full_pipeline')
```

`test_physics.py` cleared 144 checks before reaching it; `test_search.py` cleared
several hundred. Everything below that is marked `MEASURED` or `TESTED` was run
here. Nothing is claimed as verified that was not.

Two other environment notes, neither a defect in this repository: MuJoCo 3.13
imports a renderer at `import mujoco`, which fails on a headless container
without OSMesa or EGL (`MUJOCO_GL=disable` is enough to get past it), and the
PyTorch CPU wheel index was unreachable from here, so the PyPI `torch` wheel was
used instead.

---

# Part 1 — library architecture

| # | check | verdict |
|---|---|---|
| 1 | Environment: reset / step / observation / reward / termination separable | **BY DESIGN**, 3 of 5 |
| 2 | Agent: policy / value / action selection decoupled from environment | **HELD** (one index coupling, now gated) |
| 3 | Rollout: trajectory collection separable | **HELD** |
| 4 | Buffer separable | **HELD** |
| 5 | Algorithm independent of the environment | **HELD** |
| 6 | Training loop owned by a single trainer | **HELD** |
| 7 | State ownership explicit (policy, optimiser, buffer, RNG, step) | **was GAP, now TESTED** |
| 8 | Evaluation environment separated from training environment | **TESTED** |
| 9 | Vectorisation decoupled from the algorithm | **HELD** |
| 10 | New environment / agent / algorithm / buffer without editing the trainer | **GAP** for algorithm and buffer |

## 1. Environment — separable in three places, fused in two, deliberately

`TriphibianEnv` gives `reset(domain)`, `step(target_angles) -> bool` and
`observation(target) -> ndarray`, and those three are clean: `observation` is a
pure function of the current `mjData`, `step` advances exactly one timestep and
returns only whether the battery is still alive.

Reward and termination are not separable, and the reason is written into
`_score_segment`'s docstring: the segment score is **deliberately
non-Markovian**. It divides time fractions by the duration that was *asked for*
rather than by the samples recorded, and measures sink rate over the second half
of the airborne stretch — both to close exploits that a per-step reward had
handed the search (a machine with no lifting surface scoring 0.9 for flight by
terminating on the first step; a design scoring 0.75 because the beach rose to
meet it). A per-step `reward()` cannot express either, so the reward is one
scalar per segment, delivered at the end, and PPO takes it through GAE as a
sparse-terminal task.

This is the correct trade for this project and it should not be "fixed" into the
conventional five-way split. It is recorded here so that the next person to read
the checklist does not read the fusion as an oversight.

## 2. Agent — decoupled, with exactly one index-level coupling

`learning/ppo.py` imports `math`, `numpy` and `torch`, and nothing from `envs`,
`evolution` or `core`. The dependency runs the other way: `envs/batchroll.py`
lazily imports `SegmentCollector` and `potential_of` at the point of use.

The exception is `potential_of`, which addresses the observation **by index**:

```python
_GRAV_Z, _DEPTH, _WET, _DEPTH_ERR, _CONTACT = 8, 9, 10, 14, 15
```

The comment reasons that the morphology context is appended *after* these, so
widening the observation cannot move them. That reasoning is correct, and it was
not a gate: inserting or reordering a channel before index 15 would leave the
shaping reading the wrong quantity — and potential-based shaping is the least
able of any term to announce that, because it provably cannot change the optimum
and therefore cannot surface as a wrong answer. It would simply stop helping.

**Added:** `tests/test_ppo.py::test_the_shaping_reads_the_channels_it_thinks_it_does`
recomputes each of the five quantities from the environment's own accessors and
asserts the named index carries it. All five hold today (`_GRAV_Z = 8` reads
−1.000000 against −1.000000 recomputed, and so on). Needs a compiled model but
not the batched evaluator, so it runs wherever MuJoCo does.

## 3–5. Rollout, buffer, algorithm

`SegmentCollector` records one decision per machine per control interval;
`RolloutBuffer` flattens a generation into a PPO batch; `ppo_update(policy,
buffer, ...)` takes those two objects and nothing else. None of the three
mentions a morphology, a domain or a simulator. `tests/test_ppo.py` now exercises
all three on synthetic observations with no environment present at all, which is
the operational form of the claim.

## 6. Training loop

`run_search` is the one loop, and the job layer drives it through the `Trainer`
port (`adapters/trainers/search.py`), which is where pause, resume, cancel and
failure recording live. `docs/ARCHITECTURE.md` is the reference.

## 7. State ownership — this was wrong, and is the largest finding

`SearchState` owns the policy, the optimiser, the archives, the evaluation RNG
and the step counter. Two streams were missing, and
`adapters/trainers/search.py` declared `deterministic=True` regardless.

**`torch` was never seeded from `cfg.seed`.** `run_search` constructed
`SharedPolicy` before anything had called `torch.manual_seed`, so two runs at the
same `--seed` started from different initial weights: measured ‖dW‖ = **7.4** in
the first actor layer. The worker pool seeds itself per shard
(`envs/actors.py`), which covers the rollout's exploration noise and not this;
and the single-process path (`workers <= 1`, or a pool that degraded) takes no
per-shard seed at all, so it had no seeding anywhere.

**PPO's minibatch order came from the process-global legacy `RandomState`.**
`ppo_update` called `np.random.shuffle`, a stream nothing seeded from `cfg.seed`
and nothing wrote into the checkpoint. Two updates identical in every other
input differ by ‖dW‖ = **0.155** when only that stream differs, and by exactly 0
when it matches.

This is the same bug the project has already closed twice — once for the
optimiser moments ("a resumed run continued with a warm network and a cold
optimiser") and once for the evaluation seed stream ("arch38's archived
`takeoff_height` of 2.288 m re-measured as 0.000") — in the last two places it
was still possible.

**Fixed.** `run_search` calls `torch.manual_seed(cfg.seed)` before the network
exists; `SearchState.learner_rng` is a dedicated `Generator` seeded from
`cfg.seed`; `ppo_update` takes an `rng` argument and reports
`"deterministic": False` when it is not given one, rather than falling back
silently; `save_state`/`load_state` carry both streams.
`tests/test_ppo.py::test_both_learner_streams_survive_a_checkpoint` round-trips
them without needing an environment.

## 8. Evaluation separated from training

Two separations, both real and both tested.

* **Two evaluation paths.** `batchroll.evaluate_tier1_batch` is what the search
  scores with; `evaluate.evaluate_tier1` is what Tier-2 verification, every
  offline probe and the showcase use. They differed by 60x on land locomotion
  until the scatter seed was shared. `tests/test_search.py::
  test_the_two_evaluation_paths_score_the_same_machine_the_same` asserts the
  agreement.
* **Scoring uses the mean, learning samples.** A rollout that banks no
  trajectory has nothing to explore for, and its score is what the archive
  stores. Measured cost of getting it wrong: crossings 0.600 → 0.558 with
  sampling against 0.642 with the mean, on the same eight bodies at one seed.
  Asserted in the integration section of `tests/test_ppo.py`.

`policy.eval()` / `policy.train()` is **N/A**: the network is a tanh MLP with no
dropout and no batch normalisation, so it has no mode-dependent layer. The
separation that matters here is mean-versus-sample, and that is the one tested.

## 9. Vectorisation

`ActorPool` shards a generation over worker processes; each worker fills its own
`RolloutBuffer` and returns trajectories, which the parent concatenates before
running one update — so the batch the optimiser sees, `_terminal_scale`
included, is the batch it saw before. The algorithm knows nothing about the pool.

`envs/actors.py` already documents the one honest exception: a learning run's
exploration noise depends on how the work was split, so it is reproducible per
worker count rather than across worker counts. That remains true and is not a
defect. What has changed is that the *parent-side* path — `workers <= 1`, and
the degraded fallback when a worker dies — now has a seeded, checkpointed torch
stream, where before it had none.

## 10. Extensibility — a real gap, not fixed here

Adding a **trainer** is a documented extension point (`docs/ARCHITECTURE.md`,
"Adding a trainer"); `Trainer` is a `Protocol` and `adapters/composition.py` is
the only place that knows which implementation is real.

Adding an **algorithm** or a **buffer** is not. `run_search` names
`ppo_update` and `RolloutBuffer` directly, and so does the worker entry point in
`envs/actors.py`. Roadmap item **N** is "GRPO for the shared policy", which under
the current shape means editing the generation loop.

**Not fixed here, deliberately.** Introducing an injection seam for a second
algorithm that does not exist yet is a refactor with no measurement behind it,
and this project's rule is that a change to the search needs a reason that is a
number. The cheapest concrete step when GRPO is actually written is to give
`SearchConfig` a `learner` field and route the two call sites through it —
roughly fifteen lines, and best done with the second implementation in hand so
the seam is shaped by two cases rather than one.

---

# Part 2 — test coverage

`tests/test_ppo.py` went from **8 checks to 54**, and from one function that a
missing GPU extension could take down entirely to four sections of which only
the last two need anything beyond numpy and torch. `tests/test_index.py` is new
(17 checks). Both run on a machine with no GPU and no Mojo build: 47 of the 54
run with numpy and torch alone, the other 7 need a compiled MuJoCo model and
skip with their reason printed.

## 1. Unit

| item | status | where |
|---|---|---|
| Environment `reset()` | TESTED | `tests/test_physics.py`, `tests/test_search.py` |
| Environment `step()` | TESTED | `tests/test_physics.py` |
| Reward calculation | TESTED | `tests/test_search.py` (the judge ladder and segment scorer) |
| Termination / truncation | BY DESIGN | segments are fixed-duration; see Part 1 §1 and §3 below |
| Action validation | TESTED | `test_one_decision_has_the_right_shape` (intent stays in [−1, 1]), `test_cpg_respects_joint_limits` |
| Observation validation | TESTED | `test_the_shaping_reads_the_channels_it_thinks_it_does` |
| Transition | TESTED | `test_transitions_are_graded_not_pass_fail` |
| Policy | TESTED | `test_one_decision_has_the_right_shape`, `test_batched_and_single_decisions_agree_when_deterministic` |
| Value function | TESTED | value head asserted row-for-row against the batched path |
| Loss | TESTED | `test_an_empty_generation_is_skipped_not_crashed_on` (finite losses), `test_the_entropy_bonus_has_a_gradient` |
| Advantage / return | TESTED | `test_gae_matches_an_independent_reference` |
| Buffer | TESTED | `test_one_trajectory_never_bootstraps_off_another`, `test_terminal_rewards_are_scaled_within_their_own_kind` |
| Optimiser / update | TESTED | `test_it_learns_a_reward_with_a_known_optimum`, `test_the_rate_anneal_reaches_the_optimiser` |

## 2. Integration

Environment → agent → rollout → buffer → algorithm → policy, and the full
collect-then-update, are covered by `tests/test_search.py::
test_the_loop_wires_every_layer_together` and by the integration section of
`tests/test_ppo.py`. Training → checkpoint → resume is covered three ways:
`test_a_run_can_be_picked_up_where_it_stopped`, `test_a_finished_run_is_a_checkpoint`
and the new `test_both_learner_streams_survive_a_checkpoint`.

## 3. Algorithm correctness

This was the emptiest column and is where the two behavioural fixes came from.
`experiments/ppo_estimators/` is the harness; it needs no environment.

**The entropy bonus had no gradient.** `ppo_update` weighted
`ent = -logp.mean()` with `ent_coef`, where `logp` scored actions drawn from
π_old. That is the cross entropy H(π_old, π_new), and at the on-policy point
where every PPO update starts, `E_{a~π}[∇ log π(a)] = ∇ ∫ π = 0` — the expected
gradient is not small, it is zero.

| d/d(log_std), 32 initialisations, batch 4096 | mean ± SE | t |
|---|---|---|
| the term as it was used | **−0.00013 ± 0.00188** | −0.07 |
| a reparameterised estimator of the true entropy | **+0.43114 ± 0.00106** | +408.53 |

End to end, sweeping `ent_coef` from 0 to 1.0 — a hundred times the default —
moved the learned mean `log_std` by **−0.0063**, downward. After the fix
(`SharedPolicy.entropy`, sampling from the current policy through `rsample`) the
same sweep moves it by **+0.2126**. The `entropy` field in the `ppo` telemetry
line was likewise a cross entropy under a name that made it look like a collapse
detector.

**The KL could come out negative.** `target_kl` ended the epoch loop on
`(logp_old - logp).mean()`, unbiased for KL(old‖new) and variable enough to go
negative, which a divergence cannot: over 320 minibatches of real updates it was
negative on **20.0%** of them, bottomed at −0.0072, and ran ~5% below the k3
estimator on average — so on **6.9%** of minibatches it reported the update as
inside the 0.015 bound while k3 put it at or over. Now k3,
`(r − 1) − log r`, non-negative by construction.

**The importance ratio was fine, and the boundary is recorded.** max|ratio − 1|
is 2.9e−6 at the operating `log_std = −0.5`, and 1.07e+2 at `log_std = 2.0`
where 60% of actions sit past |a| = 0.999. Nothing clamps `log_std`, and the
entropy fix now pushes it up, so both halves are asserted in
`test_the_importance_ratio_starts_at_one`.

| item | status |
|---|---|
| Compared against a reference implementation | TESTED — GAE against its double-sum definition, written independently in the test |
| Forward / loss / gradient / parameter update consistent | TESTED — the update moves the policy toward a known optimum; the entropy gradient is asserted positive |
| Advantage / return | TESTED |
| Target / bootstrap | TESTED — zero terminal bootstrap, asserted |
| Discount factor γ | TESTED — γ=0, λ=0 and γ=λ=1 each collapse GAE to a hand-computable quantity |
| terminated / truncated | **BY DESIGN** — a segment is a fixed-duration window whose entire reward arrives at its end, and the machine is re-placed for the next domain rather than continuing, so there is no continuation to bootstrap. Recorded rather than changed: nothing measured here says the zero is wrong, and a `V(s_T)` bootstrap would be a different experiment. |

## 4. Environment

Observation shape and dtype, action space, state transition, reward, episode
termination, post-reset state, boundary states and seed reproducibility are
covered by `tests/test_physics.py` (which could not complete here) and
`tests/test_search.py`. The one item that had no gate and now does is
observation *layout*, above.

## 5. Buffer

| item | status |
|---|---|
| Insert | TESTED — empty trajectories are dropped, `n_transitions` counts the rest |
| Sample / batch size / sampling randomness | TESTED — minibatch order comes from an owned `Generator`; same stream, same update |
| Capacity / overflow | **N/A** — on-policy. The buffer holds one generation and is rebuilt each time; there is no eviction policy to test. |
| Episode boundary | TESTED — `test_one_trajectory_never_bootstraps_off_another` |
| Terminal transition | TESTED — zero bootstrap at the end of each trajectory |
| Prioritised sampling | **N/A** — none |
| GAE / return storage | TESTED |

## 6. Training loop

Parameters update, gradients are produced, and the optimiser step count is
reported (`grad_steps`). There is no torch scheduler to step the wrong number of
times: the linear anneal is applied per update from `gen / generations`, and
`test_the_rate_anneal_reaches_the_optimiser` asserts the annealed rate reaches
`param_groups` and is clamped to [0, 1]. Evaluation does not update the model —
`ppo_update` is called from exactly one place, and a scoring rollout passes no
buffer.

**NaN / Inf detection was absent and now exists.** A single non-finite terminal
reward — one segment scored on a rollout that diverged — becomes NaN in every
advantage in the batch through the batch-wide standardisation, and one optimiser
step writes NaN into every weight of a policy shared by every machine in the
search. The failure then surfaces several frames later inside
`torch.distributions.Normal`, after a checkpoint may already hold a dead network.
`ppo_update` now checks the built batch and refuses it with
`{"skipped": True, "reason": "non-finite batch", "non_finite": {...}}`, leaving
the policy bit-for-bit unchanged. Asserted for a non-finite reward, value and
observation.

## 7. Numerical

γ = 0, γ → 1, zero advantage, zero reward, NaN and Inf are all tested above.
Extreme observations are covered by the normaliser's ±10 clamp and by
`test_the_observation_normaliser_is_associative`, which also asserts that folding
two batches gives what folding their concatenation gives — the property a resume
depends on, since a resume folds a different cut of the same stream.
An analytic gradient check (finite differences against autograd) is **not**
present; torch's own `gradcheck` covers the primitives and the end-to-end
substitute is the known-optimum learning test. FP16 / BF16 is **N/A**: the
learner is ~1,200 weights and runs on CPU by measurement, not by accident.

## 8. Reproducibility

Same seed → same initial parameters: **was false, now tested** (Part 1 §7).
Same seed → same update: **tested**. Checkpoint resume consistency: tested for
weights, optimiser moments, evaluation seed stream, and now both learner
streams. Same config → rebuildable experiment: the job layer's whole purpose,
tested in `tests/test_application.py` and `tests/test_adapters.py`. CPU/GPU
tolerance is **N/A** for the learner (CPU only); for the *evaluator* it is the
`tests/test_gpu_mirror.py` suite, which needs the extension.

## 9. Vectorised / parallel environment

N=1 and N>1, per-environment independence and shard-invariance of a score are
covered by `tests/test_search.py::test_sharding_a_generation_does_not_change_a_score`,
which needs the extension. The batch-shape and reset-mask items have no direct
analogue: the pool shards a list of machines rather than stepping a fixed vector
of environments, and each shard is a complete batched evaluation.

## 10. Algorithm-specific

PPO only. Ratio, clip, GAE, entropy and value loss are all covered above; the
value loss is clipped against the old estimate, which `test_an_empty_generation`
exercises for finiteness and the reference-GAE check exercises for its inputs.
DQN, SAC, A2C and TD3 are **N/A** — none is implemented, and roadmap item N
(GRPO) is the only planned second algorithm.

## 11. Regression

Each of the four fixes above carries a test naming the measurement that motivated
it, which is this project's existing convention (`test_physics.py` and
`test_search.py` are written the same way). The full suite cannot be run on a
machine without the GPU extension, which is the standing constraint.

## 12. End-to-end on a tiny environment

There is no tiny known-answer environment, and `tests/test_ppo.py` now
approximates one: a synthetic observation stream with a reward whose optimum is
"drive coefficient 0 positive", through collect → buffer → update → evaluate,
with checkpoint and resume covered separately at the `SearchState` level. What is
missing is a single end-to-end script that runs all six stages in one pass on a
toy task. **GAP**, and a cheap one — the pieces all exist and are already
environment-free.

---

# Summary of changes

| what | was | is |
|---|---|---|
| entropy bonus | zero expected gradient; `ent_coef` inert at every value | `SharedPolicy.entropy`, reparameterised from the current policy |
| KL estimate | `log π_old − log π_new`, negative on 20% of minibatches | k3, `(r − 1) − log r` |
| torch seeding | absent; ‖dW‖ = 7.4 between runs at one seed | `torch.manual_seed(cfg.seed)` before the network, checkpointed |
| PPO minibatch order | process-global legacy `RandomState` | `SearchState.learner_rng`, checkpointed, reported when absent |
| non-finite batch | silently written into every weight | refused with a reason; policy untouched |
| `tests/test_ppo.py` | 8 checks, one function, dies without a GPU | 54 checks, four sections, only the last needs one |

**`shared_ent_coef` now does something, so runs before and after this commit are
not comparable on exploration.** At the default 0.01 the learned `log_std` now
sits +0.0113 above the same run with the bonus switched off, and at 0.1 it sits
+0.1048 above — where before those differences were −0.0003 and −0.0021. The
coefficient is worth re-tuning rather than inheriting, and that is a decision
for whoever configures arch39; it is roadmap item **R**.
