# Hyperparameters: every `SearchConfig` field

Written 2026-10-08 (ROADMAP B2). One row per field of `SearchConfig`
(`dytiscidae/evolution/loop.py`): its default, the value arch48 ran with, and
whether a measurement or a typed number set it.

**Defaults come from `loop.py`.** The table is a snapshot; the code wins. To
regenerate the default column, list the annotated assignments of the
`SearchConfig` class body (`ast.parse`, then every `AnnAssign` in the class).
`tests/test_index.py` is the check that every field is named in the first
column, so a field added without a row fails the build. The CLI
(`python -m dytiscidae.ops.run search`) mirrors most defaults; where it does
not, the row says so. The arch48 column is that run's recorded configuration
(`checkpoint.json` provenance, 62 keys; `runs/` is gitignored, see `docs/PORTING.md`).

**"set by" is one of four words.**

* `measured`: a sweep or a measurement chose the value. The source column
  names the ROADMAP section or `experiments/` path and the number.
* `derived`: arithmetic or physics fixes it. The source shows the arithmetic.
* `typed`: nobody measured it. The row says what the source does and does not
  establish. This is the honest majority: `docs/ROADMAP.md` §"What is set by
  measurement, and what is typed" already lists `mission_weight`,
  `reward_shaping` and `descriptor_bins` here. A `typed` value may still have a
  measured *reason to exist* (a motivating failure); the value is what is typed.
* `switch`: a feature on/off flag or a mode name. The source says which
  measurement, if any, decided its default.

**Reading it.** Where the default differs from arch48 the row says so. A
default is the dataclass default, which is not always the CLI default or what
the last run used. Nine fields differ from arch48: `generations`, `batch`,
`seed`, `workers`, `controller_refine_steps`, `use_shared_policy`,
`n_reference_seeds`, `run_dir`, `memory_ceiling_mb`; `draw_per_candidate` is
new. `shared_learner`, `grpo_group` and `grpo_bodies` were each declared
twice in the class body (identical values); the copy was removed on
2026-10-08 and `tests/test_index.py` now fails on a doubled field.
`controller_refine_steps` is 0 in the dataclass and, since 2026-10-08, in the
CLI too.

## Search size and pool shape

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `generations` | 200 | 300 | typed | A run length chosen per run; no measurement found for the default 200. |
| `batch` | 4 | 16 | typed | `loop.py` calls it "a search-design decision, not a throughput one". The pool-shape sweeps in `CLAUDE.md` (4x4 31.7 s, 1x16 72.2, 2x8 47.6, 8x2 51.8, 16x1 93.3) took batch 16 as given. |
| `seed` | 0 | 20261006 | typed | An identifier, not a tuned quantity. |
| `workers` | 1 | 4 | measured | `CLAUDE.md` "Running a search": 4 shards of 4 won the pool sweep on random bodies (4x4 31.7 s against 1x16 72.2 s); worker count is "bounded by memory before cores", so 4. The default 1 is the old single-process behaviour. |
| `min_shard` | 2 | 2 | measured | `CLAUDE.md` pool-shape sweep (8x2 51.8 s, 16x1 93.3 s, slower than one process); `experiments/budget_sweetspot/README.md` §9: queue of small shards beat 4x4 by 12.5%, faster on 3/3 batches. Not 8: at batch 16 one Tier-0 rejection gives `15 // 8 == 1` and the workers idle (`docs/ROADMAP.md` §"U. `min_shard` still defaults to 8"). |
| `pool_per_worker` | 2.0 | 2.0 | measured | `experiments/budget_sweetspot/README.md` §8: queue (4x2:q2) total 0.875 of the 4x4 wall on late rotor-heavy bodies; `docs/ROADMAP.md` §"2026-10-03 -- the eight-item pass", row AJ. Changes no score. |
| `pool_balance` | False | False | switch | Cost-balanced shard assignment, off. `experiments/budget_sweetspot/README.md` §8: balance total 1.031 with the queue, 0.866 alone, mixed across batches; ROADMAP row AJ: "balance stays off until a re-sweep with the rotor cost". |
| `controller_refine_funnel` | None | None | switch | AK's funnel (`docs/ROADMAP.md` §"3. AK."): None refines every candidate. Built 2026-09-30 off by default; no measurement turned it on. With M1's `controller_refine_steps` default of 0 it has nothing to filter. |

## Distance curriculum and command-rate penalty

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `distance_curriculum` | True | True | switch | `docs/ROADMAP.md` §"10. Y/O. Transition-distance curriculum": on by default since 2026-10-05 by the user's decision. Measured basis: crossing share at back 0 is 0% against the 0.5 advance share, so it holds at back 0 and a 30 m launch; `experiments/transition_distance/`. Changes what every transition and air score means. |
| `distance_step` | 0.5 | 0.5 | typed | Same section: "the four numbers above are placeholders". Metres a start steps back. |
| `distance_advance_share` | 0.5 | 0.5 | typed | Same section, placeholder. `experiments/transition_distance/README.md` §3 predicted the elites would sit below it at back 0. |
| `distance_window` | 200 | 200 | typed | Same section, placeholder. Evaluations in the sliding window. |
| `action_rate_penalty` | 0.0 | 0.0 | switch | `docs/ROADMAP.md` row Y ("chatter"): `command_rate` and `command_reversal` are published whether or not it is on, "flag off", the weight to be set from a measured distribution; none found. |

## Fidelity and promotion

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `segment_seconds` | 8.0 | 8.0 | typed | `loop.py` docstring calls it the main cost/fidelity dial (halving halves run time and roughly doubles Tier-1 variance); no sweep chose 8. `MEASURABLE_AIR_SECONDS` and `STATION_WINDOW` in `docs/ROADMAP.md` were set "at an 8 s segment" to keep earlier numbers, which presupposes it. |
| `identify_axes_every` | 1 | 1 | typed | Re-identify every child every generation. `docs/ROADMAP.md` §"1. AN." fixed the knob so values above 1 work; every run used 1. Identification is 67% of an evaluation (`docs/ROADMAP.md` "G. Identification is 67% of an evaluation"), which is a cost, not a reason for 1. |
| `tier0_gate` | -0.85 | -0.85 | typed | Structural-margin rejection bar. No derivation or sweep found in `docs/` or `experiments/`. |
| `tier2_every` | 5 | 5 | derived | Arithmetic on label supply: with eight islands each firing yields three critic labels per island, 15 gave 24 labels per 120 generations (arch47: 60-label minimum at gen 243); 5 reaches it near gen 87 (comment in `loop.py`; `docs/ARCH46_SPEC.md` §1, `min_samples=60`). The "13 min per round" cost figure appears only in the `loop.py` comment, not in a stored document. |
| `tier1_5_seconds` | 60.0 | 60.0 | typed | `docs/ROADMAP.md` §"1.2 Tier-1.5": 60 s is 7.5 Tier-1 segments and a fifth of a mission leg; a design argument for cost against drift visibility, not a sweep. |
| `tier2_label_all_media` | True | True | switch | `docs/ARCH46_SPEC.md` §2b: on by default; prediction (labels per medium triple) was pre-registered and the section is marked unrun. |

## Controller refinement

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `controller_refine_steps` | 0 | 1 | measured | `docs/ROADMAP.md` §"D. M1 read (2026-10-08, `experiments/refine_criterion/`)": over 600 one-step trials on arch48's 200 elites an accepted trial kept 23% (`island` criterion) or 8.5% (`stage`) of its reported gain at a fresh draw, and 2 of 600 were accepted under the criterion in use. CLI `--refine-steps` default is 0 from 2026-10-08. Every run from arch34 to arch44 passed 2; arch48 passed 1. Not comparable across this. |
| `controller_refine_sigma` | 0.1 | 0.1 | typed | Perturbation scale on policy weights. No sweep found. |
| `promotion_refine_steps` | 6 | 6 | typed | `loop.py`: affordable because bounded by promotions (at most three per round). M1 states "Promotion's 6 steps are not measured here"; no other measurement found. |
| `policy_hidden` | 0 | 0 | typed | Width of the per-candidate policy (0 = linear). No sweep found. |
| `n_modes` | 6 | 6 | typed | Number of gait modes. No sweep found. |

## Selection blend and archive

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `mission_weight` | 0.30 | 0.3 | typed | Listed as typed in `docs/ROADMAP.md` §"What is set by measurement, and what is typed". The reason for a nonzero weight was measured over arch31 (corr(archive fitness, `mission_fraction`) = 0.159 at weight 0); the 0.30 was not swept. |
| `reward_shaping` | 0.2 | 0.2 | typed | Weight of the potential-based shaping term in the PPO reward. Same ROADMAP typed list. Theory (Ng, Harada and Russell 1999) says such shaping cannot change what is optimal, only when credit arrives; the 0.2 was not swept. |
| `descriptor_bins` | 5 | 5 | typed | Same ROADMAP list. Motivation in `loop.py`: 8 bins gave 4096 cells per island, 2.6x arch31's whole evaluation budget; 5 bins gives 625. The figure 5 was not swept. |
| `learned_axes` | True | True | switch | AURORA-style learned descriptor axes instead of the hand-picked four; the reasoning is in the `loop.py` comment, and no measurement decided the default. |
| `descriptor_refit_every` | 400 | 400 | typed | Evaluations between refits. `docs/ROADMAP.md` item F measured the harm of refitting (archive size at refit fell 105 to 77 over arch34; 331 cells merged); it proposed refitting less often but 400 was not swept. |
| `descriptor_keep_if_overlap` | 0.95 | 0.95 | measured | `docs/ARCH46_SPEC.md` §3: on arch45's 33 refits, threshold 0.90 skips 25/33 (2238/3066 merges avoided), 0.95 skips 19/33 (1601/3066), 0.97 skips 17/33 (1424/3066); 0.95 chosen as the median case. arch45 and earlier used 0. |
| `gait_gain` | False | False | switch | `docs/ROADMAP.md` §"AA. The policy could not stop a machine": an oracle that zeros amplitude on stop/hold lifts land task score 0.362 to 0.501. Off by default because it changes both policies' shapes and every stored run was scored without it. |

## Shared policy (PPO)

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `use_shared_policy` | False | True | switch | Off in the dataclass, on in every run since arch29 (`CLAUDE.md` example command). `experiments/shared_policy_value/` asks whether the network carries weight; arch33 measured -0.0014 +/- 0.0016 in-distribution. No measurement decided the default. |
| `shared_hidden` | 64 | 64 | typed | `docs/ROADMAP.md` §"2026-10-08 -- the MuscleMimic comparison": "no measurement says width is the limit" (13,133 parameters). |
| `shared_lr` | 1e-3 | 0.001 | typed | No sweep found. Annealed by `shared_lr_anneal`. |
| `shared_epochs` | 10 | 10 | typed | Same section: a 1/2/4/6 sweep was proposed and not run. arch48 telemetry: `stopped_early` 0 of 300, KL median 0.0062 (max 0.0091), so the epoch loop never hit its KL bound. |
| `shared_minibatch` | 2048 | 2048 | typed | No sweep found. |
| `shared_target_kl` | 0.015 | 0.015 | typed | Same section: arch48 updates ended at KL max 0.0091, under the bound. The k3 estimator fix is in `docs/ROADMAP.md` §"R. `shared_ent_coef`". The 0.015 itself was not tuned. |
| `shared_ent_coef` | 0.01 | 0.01 | typed | `docs/ROADMAP.md` §"R. `shared_ent_coef`": "the default is still 0.01, which is the value that was measured to do nothing" before the estimator fix; after it, 0.01 buys +0.0113 on `log_std` and 0.1 buys +0.1048 (`experiments/ppo_estimators/`). "Pick it from a sweep"; none run. |
| `shared_learner` | "ppo" | "ppo" | switch | `docs/LEARNER_AUDIT.md`: "ppo" is every run to date; "grpo" and "ppo+grpo" built 2026-10-03, off. `docs/ROADMAP.md` §"11. N. GRPO" is conditional on the shared policy carrying weight. |
| `grpo_group` | 4 | 4 | typed | Rollouts of one body per GRPO group (at least 2). Inert while `shared_learner` is "ppo". |
| `grpo_bodies` | 4 | 4 | typed | Bodies given a group per generation. Inert while `shared_learner` is "ppo". |
| `shared_lr_anneal` | True | True | switch | `docs/ROADMAP.md` §"Phase 4 -- PPO implementation hygiene, as one arm -- ran, unattributable": arch33's policy hit the KL ceiling on 88% of updates past gen 450. The anneal's own effect was not isolated. |

## Seeding, output, checkpoints

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `n_reference_seeds` | 20 | 12 | typed | Reference-design seeds. The CLI default (`--reference-seeds`) is 12, which is what arch48 used; the dataclass says 20. No measurement found for either. |
| `n_random_seeds` | 8 | 8 | typed | Random seed bodies. No measurement found. |
| `run_dir` | "runs/latest" | "runs/arch48" | typed | A path. |
| `checkpoint_every` | 20 | 20 | typed | Generations between checkpoints. arch35's OOM kill (`docs/ROADMAP.md` §"What arch35 measured") lost 405 generations from a checkpoint it could have resumed; the interval was not derived from it. |
| `snapshot_every` | 50 | 50 | typed | `docs/ROADMAP.md` §"2. Telemetry bundle": added because arch44 kept only its final state and rotor against flapping could not be split by generation. The 50 is not derived. |
| `event_sample` | 1 | 1 | typed | Keep every telemetry event (`ops/telemetry.py`). |
| `resume` | False | False | switch | Continue a run in `run_dir`. |
| `memory_ceiling_mb` | 0 | 6000 | typed | 0 is off. arch35 was OOM-killed at gen 495 of 900 holding 2.7 GB across the parent and four workers (`docs/ROADMAP.md` §"What arch35 measured"). arch48's 6000 is a typed machine-specific limit. |

## Islands, judge, critic, auditor, scout

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `islands` | all eight of `ISLANDS` | the same eight | typed | The set is a design choice. `triphibian`, the eighth, was added with a measured motive: in arch46's 7,804 Tier-1 evaluations the weakest medium is 0 in 98.2% (`docs/ROADMAP.md` §"2026-10-03 -- the eight-item pass", row "island"). |
| `migrate_every` | 60 | 60 | typed | Generations between migrations. No measurement found. |
| `n_migrants` | 2 | 2 | typed | Migrants per event (`evolution/islands.py`). No measurement found. |
| `judge_quantile` | 0.9 | 0.9 | typed | Quantile argument of `Judge` (`evolution/judge.py`). No measurement found. |
| `judge_update_every` | 50 | 50 | typed | `Judge(update_every=...)`. No measurement found. |
| `use_critic` | True | True | switch | Learns the Tier-1 to Tier-2 gap (`docs/ARCH46_SPEC.md` §1). No measurement decided the default. |
| `critic_refit_every` | 40 | 40 | typed | No measurement found. |
| `audit_every` | 30 | 30 | typed | Counted per island visit since 2026-09-30, which removes the `gen % N` aliasing (`docs/ROADMAP.md` §"1.3 Schedule aliasing": 30 audits over 900 generations). The 30 itself was not measured. |
| `audits_per_review` | 2 | 2 | typed | No measurement found. |
| `use_scout` | True | True | switch | Predicts a lineage's future lift (`loop.py` comment). No measurement decided the default. |
| `scout_horizon` | 40 | 40 | typed | No measurement found. |
| `scout_reserve` | 0.15 | 0.15 | typed | Share of each archive protected on predicted potential (CLI `--scout-reserve` help text). No measurement found. |

## Evaluation draw

| field | default | arch48 | set by | source |
|---|---|---|---|---|
| `draw_per_candidate` | True | not recorded | switch | Added 2026-10-08 (`docs/ROADMAP.md` item M2). arch48 predates it and ran one task draw per generation, which is the behaviour of `False` (`eval_seed` had one distinct value in 299 of 300 generations). The default was decided by `docs/PAPERS_2610.md` §1 (a fresh seed alone took Tier-1 water passes 15/16 to 3/16 and land 13/16 to 0/16) and the M2 first read in `docs/ROADMAP.md` §"D. M1 read" (draw variance over design variance 2.16 air, 1.01 water, 1.04 land). Not comparable across it. |
| `placement_draws` | 1 | not recorded | switch | Added 2026-10-08 (`docs/ROADMAP.md` M3). Draws each candidate is scored at before placement; the median one by summed medium competence is placed. Off because C2's rows show the median buys little (Spearman vs an independent mean: water 0.36 -> 0.48, land 0.30 -> 0.36); its form is an open question. |
