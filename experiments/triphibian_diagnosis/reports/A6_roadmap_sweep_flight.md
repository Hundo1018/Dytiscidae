# A6 — evidence ledger for docs/ROADMAP.md lines 5215-7620

Investigator A6, 2026-11-10, repo at a7770c2 (`docs/ROADMAP.md`, 7,620 lines). Read-only:
no test, search, rollout or git write. The whole range was read in order; out-of-range
lines were read only where a newer dated section corrects or closes an in-range item.

Conventions
- `N` or `N-M` = `docs/ROADMAP.md` line(s). **OOR** = out of the assigned range, cited only
  as a correction or closure pointer (other investigators own those sections).
- Status words: worked / no effect / refuted / void / unrun / superseded, plus the
  ROADMAP's own words where they are more exact (built-off, deferred, open, partial).
- Numbers are copied as printed. "NOT STATED" = the ROADMAP does not say. An estimate of
  mine is labelled **ESTIMATE** with its assumptions.
- The range is newest-first: 10-08 sweep (5215) → MuscleMimic (5853) → arch48 (5965) →
  external review (6020) → eight-item pass 10-03 (6114) → flight audit 09-23..26 (6176)
  → "Why nothing flies" 09-23 (6437) → arch40's list 09-20..22 (6642) → arch34's phases
  (≤ 09-05, 7196).

---

## A. Ledger

### A1. 2026-10-08, later — the work-list sweep (5215-5852)

| id | date | what it changed or measured | result with numbers | status | refs |
|---|---|---|---|---|---|
| arch49 read | read 10-08 (ran 10-07) | 300 gens, seed 20261007, arch48's flags, antipodal pair, draw still shared per gen (pre-M2) | `mission_best` 0, crossings 0 every kind; gens with `three_media` 0 (arch48: 46 from 254); `domain_best` max air/water/land 0.422/0.294/0.430; `weakest_best` 0.001; wall/gen 133/135/143 s; divergence 0.19%. Pre-reg 1 (held-out ≥ 0.5) **failed**: land 0.151 (6 audits), air 0.328 (5), water 0.091 (1). Pre-reg 2 (T1-T2 Spearman > 0.3) **failed**: water 0.174, land 0.182 (n = 192), air 0.403 (arch48 -0.09/0.02); T1-pass/T2-fail at 0.15: water 6/7, land 21/23. Pre-reg 3 (no-model gate) **held**: water ≥ 0.012 elite 64 / still 3 / base 13; land ≥ 0.012 51/0/0; ≥ 0.055 14/0/0 and 20/0/0 | read; 2 of 3 pre-registered reads failed | 5222-5257 |
| arch49 gate side-findings | 10-08 | same gate, other bars | still clears air ≥ 0.012 on 11 bodies vs elites 6 (≥ 0.055: 9 vs 3), all 11 gannets gliding from the launch, 7 clear it held still and not under their own gait; curriculum stage 1 leaks (15/8/8); stage 4 `mission > 0` leaks (8/2/1); crossings elites air→water 1, water→land 2, land→air 1 of 229, still 0 ("underpowered"); water/land state rungs still leak; critic corr(T1,T2) 0.26/0.25, out-of-fold 0.04/0.04, air 6 non-zero labels in 194 | read | 5259-5275 |
| arch48 item 1 | 10-08 | crossing failure notes, every evaluation (arch48 4,792; arch49 4,749) | see B row 22; event kept only 3 notes, so `land_to_air` mostly cut off (8 kept from 10-08) | read → **decision: no crossing build before the air and water-cruise items; crossing curriculum stays off "because its start rate is still 0"** | 5277-5295 |
| M2 | 10-08 | one draw (seed) per candidate; `SearchConfig.draw_per_candidate` default on | arch48 had one distinct `eval_seed` in 299 of 300 gens; test + 4 mutations caught | built; comparability boundary (water, land) | 5297-5312 |
| C2 (M2 2nd half) | 10-08 | arch48 200 elites × recorded + 6 fresh draws, 1,400 evals, 5,466 s | R = draw var / design var: air 4.61 [1.85-13.8], water 3.01 [1.61-5.76], land 2.49 [1.45-5.15], `stage` 2.36 [1.32-4.77], `island` 0.83 [0.50-1.56]; one-draw reliability 0.18/0.25/0.29/0.30/0.55; draws for 0.8: 18/12/10/9/3; rank(1 draw, mean of 5) 0.41/0.33/0.30/0.48/0.55. arch49 replication (229 elites, 8,654 s): R air 5.76 [2.22, 38.0], water 3.42 [1.77, 7.87], land 1.83 [1.19, 2.96], stage 2.33, island 1.09; reliability 0.15/0.23/0.35/0.30; curse at recorded draw 52%/25%/18%/31% | read: "the draw dominates" every medium and `stage` (P1, P3, P4 held; P2 held land, failed water 0.19 < 0.20) | 5314-5348 |
| M6 / G | 10-08 | identification's wall share; probe count | 130.3 s with vs 55.6 s without → 57% of the batched evaluation's wall (74.7 s). Probes 24/12/8: wall 129.8/67.8/47.5 s, mean summed competence 0.0239/0.0144/0.0264; 12 probes: median change 0.0049, max 0.046 (> one draw's SD 0.044), mean −40% | read; **G closed for probe count, 24 stay**; cache "still not valid" | 5350-5376, 5726 |
| M3 | 10-08 | `--placement-draws` (default 1), place the median draw by summed competence | Spearman vs independent mean of 3 — one draw air .41 / water .36 / land .30 / stage .48 / island .59; median of 3: .39/.48/.36/.54/.58; mean of 3: .53/.57/.57/.65/.72; best of 3 inflates +319/+113/+55/+61/+71%; median biases single media down (land −57%, air −74%); cost ≈ +80 s/gen at n = 3 | built, off; Q3 answered 10-09: "M3 stays off" | 5378-5410, 5741-5742 |
| PAPERS_2610 item 4 | 10-08 | still machine made still | rotor "stop" was 2.5% of top speed: 27.2 rad/s commanded on elite 1, 14.6 rad/s reached in water after 2 s; servo snap: peak joint rate first 0.5 s elites 1/0/5 4.06/0.80/18.96 → 1.07/0.28/0.08 rad/s | built (fixed). "**Every still-machine read before 2026-10-08 used the leaky arm.**" Rotor bodies not comparable across it | 5412-5437 |
| arch48 item 2 | 10-08 | post-run failure is loud; `postrun_status.json` | arch49 post-run exited 0 with `report.html` (closes the second half) | built; worked | 5439-5448, 5226-5227 |
| arch48 item 3 | 10-08 | why the critic has no skill (192 labels) | Tier-2 non-zero air 1/192, water 92/192, land 94/192; corr(T1,T2) 0.126/0.041/0.165; critic out-of-fold 0.001/-0.067/0.034; ridge on all 16 features -0.08, 0.13 | read: "not broken, it is uninformed"; kept on; re-read on a per-candidate-draw run | 5450-5470 |
| C3 / arch48 item 4 | 10-08 | which air gate is closed (arch48 200 elites) | `air_gates` empty for 154; rungs cleared 0-3: 82 (stopped at `lift_margin` 0.1/0.3/0.6/1.0: 12/14/40/16); 4: 33; 5 (stop at `stays_up`): 73; 6: 10; 7: 1; 8: 1. **No elite clears a thrust rung.** Gates with a reason: 35 "no lifting surface", 11 wing loading > 937 N/m² | read; item 4 (air thrust) deferred: "nothing reaches the rung where thrust is scored" | 5472-5496, 5723 |
| D2 | 10-08 | share flapping where the model extrapolates (k = πfc/U) | 39 no lifting surface; of 161 flappers 55 (34%) above k = 0.3 (median 0.11, p90 1.50); 54 of the 73 stopped at `stays_up` flap at k ≤ 0.3 | read: air wall "closed by the bodies and their control in the regime the model handles, not by the model"; no build | 5498-5520 |
| E2 | proposed 10-08; adopted 10-12 | pass/fail definition of done, L1-L4 | see section C | adopted; air L1 bar open (10-19 cannot be set; 10-20 defined against the null, not met) | 5522-5563, 5773-5774 |
| Air on a commanded difference (Q1 → B) | built 10-09; merged 10-10 | `tasks.mirror`, `pair_turn`, `air_task(turn, height) = turn·(0.5 + 0.5·height)` | 13 arch49 gannets: held still 0/13 clear air ≥ 0.012 (was 11); own control 0/13 (one 0.011; recorded up to 0.42); mean signed turn negative in 12/13 (−0.04 to −0.60). Gate on arch49 under paired score (10-10, 4,539 s): 0/229 clear 0.012 (6 did on recorded score), 1/229 above 0 (0.0108; 12 recorded), mean 4.7e-5; still and base 0/229 on every air competence bar ("unmeasured"); rungs still leak: still `makes_lift` 203/229, `carries_itself` 121/229, `stays_up` 18 vs elites 14, `glides` 11 vs 1. Reproduction air 222/229, water 215/229, land 195/229. Mutation `air-height-counts-floating` survived the merge, then caught by the seeds test's negative control | worked as a leak closure: removes all air credit from arch49's population. Section header "item 4 owed" is stale; item 4's outcome is recorded at 5614-5633. Comparability boundary | 5565-5680 |
| M4 | 10-08/09 | shared PPO from scratch on a fixed body | see section E | **partial, 7 of 9 runs**; final read NOT STATED anywhere in the ROADMAP; `experiments/fixed_body_learning/` holds only README.md and run.py (no `results_arch48.json`; `ls`, 2026-11-10). OOR 253 hedges ("M4 so far"); OOR 3877-3878 (T6) and 5027-5028 treat it as settled ("M4 learns nothing in air"; "has learned little under any cadence") | 5682-5707, 5731 |
| state when stopped | 10-09 | — | arch50 at gen 116 of 900 (worktree 2b7bb53), reads due gen 300; M4 and `nmg-arch49-airpair` still running; arch51 not started: arch50's configuration + `--gait-gain` + paired air | record | 5709-5717 |
| D1 flexibility | 10-08 | — | "thrust is not where any elite stops (C3)" | deferred. OOR: built off by default 2026-10-12 at the user's request (3456-3458; head summary 185-192: beetle at 7 Hz gains 41% flapping thrust at compliance 0.5) | 5724 |
| D3 off-diagonal added mass | 10-08 | — | condition (D1/D2/C3 leave air short of thrust) not met: "they leave it short of staying up" | deferred | 5725 |
| B1 | partly 10-06; done 10-12 | `SearchConfig` defaults = CLI's | batch 16, min(4, cores) workers, shared policy when torch imports, 12 reference seeds; 3 mutations caught; no plan digest changed | done | 5727, 5775-5780, 6047 |
| E1/E3 | funded 10-12 | multi-seed ablations on a CPU container | CPU probe before: batched calls 0, single 10, PPO 2 events at `transitions: 0`, 600 s wall / 593 s user CPU. Option (a) `batchroll.HostFluid` built 10-12: agreement difference 0 / worst 0.000000; PPO 3000 transitions/gen (batch 4, 1 s), kl 0.0069, 0.0033; arch51 config gens 8-9 405 s and 409 s (≈ 3.7x the GPU's ~110 s) | table says "blocked" (5728), unblocked the same day by (a) (5819-5838); **no E1/E3 run is recorded anywhere** (grep: only 177, 183, 5728 and this section) → unrun; "The E1/E3 budget is owed by the user" (183) | 5728, 5781-5851, 177-183 |
| PAPERS item 2 | 10-08 | water cruise "chosen" | — | open, for the user; no later answer found by grep | 5729 |
| PAPERS item 3 | 10-08 | leaking bars | water/land competence certified on arch49; stage 1 and air leak through the same glide; stage 4 `mission > 0` "decides nothing measurable (elite missions ≤ 1e-5)" | partly read | 5730 |
| Skill pretraining, learnability score | 10-08 | — | — | waiting on M4. OOR: N12 "after M4 finishes; not scheduled" (4336-4340) | 5731, 5878-5879, 5960-5961 |
| M5 | 10-08 | PPO epochs 1/2/4/10 at fixed budget | — | unrun, waiting on M4 | 5732, 5903 |

### A2. 2026-10-08 — MuscleMimic comparison (5853-5963)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| MM claims | 10-08 | 10 claims checked against code and arch48 | 10 epochs stale: **no evidence** (arch48 300 updates: KL median 0.0062, p90 0.0078, max 0.0091; clipfrac 0.076; `stopped_early` 0 of 300; 90-100 gradient steps; `target_kl` 0.015). "~1,200 parameters": wrong, 13,133 (actor 6,726, critic 6,401; obs 33 = 25 + 8 morphology; action 6). Shaping: true (0.2). π(a\|s,m,d): already built (since arch41). Train = select: true, in the (1+1)-ES refinement (`loop.py:675`), not PPO. One draw/gen: true (→ M2). LCB: not present (C2 first). Identification: 460,800 of 657,584 main-path steps (70%), cache as proposed not valid. Fixed-body benchmark: missing (→ M4). Pretraining / learnability: deferred | read | 5865-5880 |
| MM-B refinement | 10-08 | the (1+1)-ES criterion | accepts only if `mission_fraction` strictly rises; `mission_fraction` > 0 in **7 of 4,792** arch48 evaluations (max 0.0001); 29 trials accepted, 13 placements changed in 300 gens; re-score median 40.8 s of a 129 s gen | read | 5882-5891 |
| M1 | 10-08 | criterion and its draw, arch48 200 elites, 3 one-step trials each at own draw s and fresh s', 6,266 s | `mission`: 2/600 accepted, gain 2e-7 at s, 0.0 at s', retained 0. `island`: 107/600, 0.0018 [0.0011, 0.0027] at s, 0.0004 [0.00005, 0.0008] at s', control 0.00009, retained 23%. `stage`: 155/600, 0.045 [0.034, 0.057] at s, 0.0038 [-0.0020, 0.0094] at s', control -0.0026, retained 8.5%. Direct difference vs all trials at s': island +0.0003 [-0.00005, 0.0007], stage +0.0063 [-0.0004, 0.0133] | read: P1 held, P2 failed (island 17.8% < 20%), P3 held ("reports a draw-selected gain 4-12x its real one"), P4 split. **`--refine-steps` defaults to 0 from 10-08**; criterion switch not built "until M4 shows the policy weights can be improved at all" | 5906-5939 |
| M1 by-product | 10-08 | base at s vs s' | means air 0.00095 → 0.00029, water 0.0119 → 0.0084, land 0.0188 → 0.0146; elites score 22% (land), 29% (water), 70% (air) lower at a fresh draw; draw/design 2.16/1.01/1.04 | first read; superseded by C2 proper | 5941-5954 |
| MM-E | 10-08 | larger policy / MuscleMimic scale; pretraining now; fixed seeds without rotation | — | not adopted | 5956-5963 |

### A3. 2026-10-06 — arch48 finished, and its list (5965-6018)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| arch48 | 10-06 | 300 gens at 98306a9, seed 20261006, `distance_curriculum=1`, 11.9 h, no pre-registered reads | `mission_best` 0, transitions 0 (all 4 kinds, PPO and ladder); curriculum `back` never advanced, `window_share` ~0; `weakest_best` 0.036 (arch47 0.015); `three_media` gens 46 from 254 (arch47 6); `domain_best` 0.553/0.482/0.676 (arch47 0.510/0.487/0.887); qd 472 (573); critic fitted gen 83, skill 0.0; divergence 123/32,592 (0.38%) | read (description, not a test; one seed) | 5965-5983 |
| three_media elite | 10-06 | one gannet, 4.7 kg | air 0.036, water 0.095, land 0.651 (bar 0.012); film reproduces; held still 0.0/0.068/0.0 (water 72% passive); continuous mission **0/2 transitions**: land on-task 98%, then never airborne; `thrust_margin` 0.0004, `level_margin` 0.29 (AD's gate 0.7), air task 0.0007 | read | 5985-6002 |
| PR #29 `child_env` | 10-06 | automatic post-run failed since at least arch47 (Mojo `setenv` of `PYTHONEXECUTABLE`) | fixed; test + mutation | worked | 6004-6008 |
| item 1-5 | 10-06 | 1 where crossings fail; 2 post-run checked; 3 critic skill; 4 air thrust (re-read AH vs arch48); 5 next run: notes, 900 gens read at 500, curriculum off unless 1 finds a non-zero start rate | 1 read, 2 built, 3 read, 4 deferred (A1 rows). 5: arch49 ran 300 gens with registered reads (5224-5225); arch50 900 gens (5711) | see A1 | 6010-6018 |

### A4. 2026-10-06 — external review and correction plan (6020-6112)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| review claims | 10-06 | 10 claims | controller untrained: **wrong for the runs** (true of the default); ρ ≈ +0.077: outdated (arch48 -0.09/0.02); fluid omissions, no structural dynamics: true, documented; no GPU: README stale; shared-policy generalisation: unverified; hyperparameters: partly true; no success criterion: true (→ E2); one seed per arm: true (→ E3); issue tracker: dropped | read | 6028-6041 |
| B2 | 10-06 | one hyperparameter table | "the hyperparameter table" listed as built in the 10-08 head paragraph (261); `docs/HYPERPARAMETERS.md` exists (`ls`) | built; detail NOT STATED in range | 6048, 261 |
| B3 | 10-06 | README matches code | four false sentences rewritten | done | 6049 |
| B4 | 10-06 | `runs/<run>_config.json` export | "config export" listed as built (261) | built; detail NOT STATED in range | 6050, 261 |
| C1 | 10-06 | Tier-1 → Tier-2 after the pair | arch49: water 0.174, land 0.182, air 0.403 (bar > 0.3) | read; failed | 6056, 5247-5251 |
| C2, C3, D1-D3 | 10-06 | see A1 | | | 6057-6066 |
| E1, E3 | 10-06 | ablation arms; ≥ 5 seeds + rank test | — | funded 10-12; unrun | 6072, 6074 |
| E2 | 10-06 | pass/fail "done" | — | adopted 10-12 | 6073 |
| E4 | 10-06 | each arm states its comparability boundaries | "already a rule" | in force | 6075 |
| F (not adopted) | 10-06 | refine steps 50-100; Optuna/Ray Tune; domain randomisation / conditional policy; 4-8 GPU-months | — | not adopted, each with its reason | 6077-6091 |
| G1 | 10-06 | `tools/doc_drift.py` | no status in the ROADMAP; the file is absent from `tools/` (`ls`) | unbuilt (as far as recorded) | 6104 |

### A5. 2026-10-03 — the eight-item pass, and the 10-04 ninth still leak (6114-6174)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| AJ (queue pool) | 10-03 | `--min-shard 2 --pool-per-worker 2` by default | budget sweep 0.875x; rotors R² 0.42/0.25, DOF 0.19/0.07 of shard cost | on by default; balance off | 6123 |
| rotor cost | 10-03 | `RotorBatch`, `bemt_many` | was 198 µs per rotor-step, 82% of a rotor-heavy evaluation; now 0.120-0.134x per rotor-step, 0.180-0.211x per rotor-heavy evaluation; bit-identical (CI at 1e-12) | worked | 6124 |
| AM | 10-03 | noise floor + `clearance_many` memo | floor 1.2e-9 (bar 1e-5); profile 4.36 → 1.77 s; wall on/off 0.87-1.08 (mean 0.93, inside load noise); not done: damping projection 7%, inflow ~7%, CPG 1.6%, entrainment 3.4% | partly built | 6125 |
| triphibian island | 10-03 | 8th island, `triphibian_score` = harmonic mean of `c_i + e` minus `e`, e = 0.006 | motive: arch46 7,804 Tier-1 evals, weakest medium 0 in 98.2%, c2 0 in 77.6%, none ≥ 0.15 in all three; generalist's `mission_fraction` zero for 98.8%. Still machine (7 plans × 2 seeds): max island score 0.0081, max c2 0.0199, max c3 0.0000. Geometric mean rejected | built 10-03 ("unrun" then); it ran in arch47/48 (`three_media` 6 and 46 gens: 5978) | 6126 |
| R | 10-03/04 | shared policy's weight; `shared_ent_coef` sweep | read 1 (arch46 218 elites, with − without network): land +0.0247 (t 3.50, 69/46), water +0.0126 (t 2.59), air +0.0061 (t 1.69), mission 0, median delta 0 in every medium (an upper bound: elites selected with it). Read 2 (4 arms × 60 gens): `log_std` responds (ent0 -0.50→-0.76, ent0.1 -0.47→-0.17); competence does not: top-10 air/water/land ent0 0.19/0.39/0.52, ent0.01 0.29/0.37/0.47, ent0.1 0.21/0.31/0.52, policy off 0.12/0.38/0.55, "inside one seed's swing (air ±0.1)" | read: shared policy's weight at 60 gens "not distinguishable from zero"; keep 0.01; GRPO off | 6127 |
| Y/O | 10-04 | crossing rate by start distance (arch46 218 elites) | elites cross 0-2.3% at back 0 vs `advance_share` 0.5; still machines at or above the elite rate; air competence falls as launch lowers (0.0123 at 30 m → 0.0017 at 4 m). After the ninth fix: still 0/218 every kind, elites 0/218 at 0 m; one real crossing (elite 159, `water_to_land` from 2 m) | measured: **the curriculum cannot start** (0 at back 0 vs 0.5) | 6128 |
| N (GRPO) | 10-03 | `--shared-learner ppo\|grpo\|ppo+grpo` | off bit-identical; 7/7 mutations | built, off | 6129 |
| item 6 | 10-03/04 | ray entry; whole mutation set | `test_physics` 0 `[fail]`; 116/117 → 117/117 after fixture fix | done | 6130 |
| ninth still leak | 10-04 | `CrossingTracker` gates (energy height, `ashore`, shore progress from the command, aborted probe holds nothing); rotors stopped when still | still machines cross 0 everywhere; elites 0 at 0 m; open: open-loop rotors still cross `water_to_air` (elites 129, 137) and `water_to_land` (129, 146); graded `water_to_land` pays a still glide up to 0.145 | fixed with two open items (ARCH46_SPEC §8); transitions not comparable across 10-04 | 6132-6174 |

### A6. 2026-09-23..26 flight audit (6176-6434) and 2026-09-23 "Why nothing flies" (6437-6639)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| F-02 | 09-26 | LEV keyed on pitch rate → max(Rossby, travel since reversal); CN 1.98 → 3.4 | robofly (Re 136): CL rms 0.473 → 0.159, CD 0.678 → 0.095 | fixed | 6203 |
| F-03 | 09-26 | added-mass tensor + implicit damping on articulated joints | teal in water 1718 rad explicit vs 2.29 implicit | fixed | 6204 |
| F-04, F-08, F-12, F-13, F-14, S-02, N-02 | 09-26 | limiter, stall, AR/pressure drag, inflow/Wagner, entrained weight, slam, medusa | per table (medusa 79,045 → 1.06 rad) | fixed | 6205-6211 |
| 4 defects | 09-26 | jets/damping; slam diagnostic; quasi-static probes; split on free root | gannet glide sink 1.5 → 10.5 m/s under the split, fixed → 1.48 m/s median (air 0.162) | fixed | 6213-6225 |
| Mojo mirror | 09-26 | | gannet/beetle 1e-12 over 400 steps; eel 1e-9 by step 164 | done | 6227-6230 |
| AF robofly | 09-26 | revolving wing vs Dickinson, Lehmann & Sane 1999 | gated CL rms < 0.25, CD rms < 0.20; CN_LEV = 3.4 read from the same fit, so CD at 90° not independent | built, gated | 6232-6238 |
| rotor model + quad | 09-26 | BEMT + Prandtl + Glauert; `REFERENCE_PLANS["quad"]` with a hand cascade controller | APC 10x4.7 SF: CP rms 0.003, CT rms 0.012 (J 0.1-0.6), static CT 19% low. Quad **0.963, 0.974, 0.987**; 0.86-0.93 under AE (OOR 2886-2889: "its controller sags about a metre in turns") | worked; **never seeded** | 6240-6255 |
| AB feathering wing | 09-26 | `"universal"` = stroke + feathering hinge, `feather_lead` gene | see D | built; free flight none | 6257-6289 |
| AC | 09-26 | `flap_frequency` 1.5-12 Hz + spar check | | done | 6293-6294 |
| AD | 09-26 | thrust into selection | "deliberately not done"; prerequisite AG. OOR: deferred to arch43 gen-200 (2845-2847); user 09-30: "AD's gate is `level_margin >= 0.7`" (3028) | not done in range | 6295-6298 |
| AE | 09-26 | worst one-second drop height term; stage 1 counts air progress only airborne | 5 m drop-and-recover 0.052 vs 1.000 | done | 6299-6301 |
| AG | 09-26 | level-flight margin through real actuation, published | OOR done (2810-2829): beetle 0.939, gannet 0.913, teal 0.826, quad 0.85, medusa 0.005, eel -0.07, ray -0.08, bat -1.03; glider ≈ 1 − 1/(L/D) | listed in range; OOR done | 6330-6332 |
| AH | 09-26 | actuation at 7-12 Hz (resonance co-design; feathering motor sizing; torque-feasibility term) | OOR probed (2830-2844), teal gait, free joints: own motors 0.365 (61% torque-limited), 2x 0.543 (67%), 3x 0.494 (65%), spring 0.143, compliant drive 0.153 (14%) | listed; OOR **probed, nothing built** ("Nothing clears 1") | 6333-6339 |
| AI | 09-26 | should the search build rotorcraft? | "Yes (the user, 2026-09-26, and again 2026-09-30)"; `mut_rotor` built for arch43 | done | 6340-6341 |
| AJ (path agreement) | 09-26 | | first explanation (one kick) wrong; difference is chaotic growth from 2.5e-15 at step 2 to 1e-9 by step 131; bars = max(old, 2x the path's floor) | done | 6342-6369 |
| AK (ray entry) | 09-30 | | entrainment reaction: peak speed with/without from water spawn: beetle 0.43/4.70, eel 0.63/3.78, ray 0.59/0.63, gannet 0.36/0.36 ("most of the beetle's and eel's swimming was this momentum"); damping refresh; numpy path never scattered a crossing's entry (up to 7.3 m/s gap) — fixed. Open: slam diagnostic above ~20 m/s | done, one open item | 6370-6433 |
| F-09, F-10, F-11 | 09-23 | lever arm; reversed flow; Kramer sign | strip speed 1.8-3.6x (too easy, both paths); reversed flow lifted both strokes (too easy); Kramer opposed lift (too hard) | fixed | 6455-6464 |
| servo feed-forward (item H) | 09-23 | `ctrl = q_ref + (kv/kp) dq_ref/dt` | stroke reached: gannet 7.3 Hz 0.27 (0% sat) → 0.98 (21%); beetle 11 Hz 0.17 → 0.65 (58%); cost: gannet glide seed 0 0.213 → 0.022, seed 5 0.012 → 0.001 | fixed | 6466-6502 |
| CPG phase continuity | 09-23 | | a 0.1 Hz change at t = 4 s moved the stroke 2.5 rad in one 40 ms step | fixed | 6504-6511 |
| air short branch | 09-23 | | arch42 exits: < 2.6 s n 393 median 0.030; 2.6-2.8 s 676, 0.033; longer 1,786, 0.000; air competence corr -0.44 with time aloft (audit); famine starved air in 113 of 178 gens | gated (short branch scores 0) | 6513-6529 |
| 300 random gaits | 09-23 | best `thrust_margin`, heave-only, quasi-static at trim | see D | read: "Thrust is reachable in the model" was the bugs (refuted) | 6531-6561 |
| §6 reward defects | 09-23 | thrust in no fitness term; no gradient 6-14 m/s sink; endpoint height; stage 1 `cruise_progress` ungated; PPO air reward terminal; `thrust_margin` cached | — | open at 09-23; AE (09-26) addressed height and stage-1 | 6563-6584, 6299-6301 |
| §7 fluid open items | 09-23 | F-12, F-13, F-14, F-02, F-08 | "None of them is the reason nothing flies" | superseded: closed 09-26 | 6586-6607, 6199-6211 |
| OOR corrections | 10-10 | N7, N13, N15 | N7 **refuted** (power 3.6-28x sufficient; spar band closes at 7.80 Hz; the only level gait ever found, 11.33 Hz, is outside it; there the stroke motor needs 22x its torque limit); N13 refuted the other way (St ≥ 0.25 over most of the band); N15 (heaving-pitching foil fixture) defined, no outcome anywhere (grep: 213, 4078) → unrun | correction pointers | 4014-4035, 4062, 4078 |

### A7. arch40's list, 2026-09-20..22 (6642-7195)

| id | date | what | result | status | refs |
|---|---|---|---|---|---|
| arch40 baseline | 09-20/21 | one run changing nothing further | first launch stopped gen 54 (scoring changed: passive twin gone, water gated); re-run 600 gens "at the user's instruction" | amended | 6642-6667 |
| arch40 items 1-3 | 09-20 | re-derive water ladder (`depth_gain_net`); air ladder (`sink_reduction`); S `mut_gait` re-run (Monte Carlo 80.4/94.0/69.5% vs 0-2.4%) | NOT STATED in range | unrun in range | 6671-6681 |
| item 4 V | 09-20 | replication arm | "subsumed by arch40 itself" | superseded | 6682-6683 |
| item 5 O / O | 09-20 | air spawn (−40, 0, 30) | filmed machine scoring air 0.768 spends 0% of its air leg airborne (starts on the beach); "Every air score in this project has been earned from a 30 m launch" | open: "still no safe fix" | 6684-6688, 7182-7187 |
| item 6 R | 09-20 | `shared_ent_coef` | read 10-04 (A5) | read | 6689-6690, 6127 |
| X (task, S1) | 09-21 | commanded phases; `OBS_DIM` 33 | still machine averaged over 8 draws: water 0.011/0.008, land 0.003/0.001 (additive formula had paid 0.422); command-blind machines near zero; arch41 stopped at 52 gens: pre-registered gen-50 bar failed, task medians 0 every medium | refuted at the gen-50 bar; re-derived values "measured and not applied: the user decided not to relaunch" | 6694-6801 |
| arch42 (S2) | 09-22 | re-derived values applied | stopped flat at gen 102: water median 0.000 (29% non-zero), land 0.001 (52%), air 0.000; "Tracking was the wall" | refuted | 6803-6808 |
| S3 | 09-22 | progress `clip(v·h / v_cmd, 0, 1) · exp(−(v_across / v_cmd)²)`; hold × not sinking; air `0.5 turn + 0.5 height` | 4 real-path leaks fixed (air height while floating 0.141; hold = passing through 0.63/0.51; turn = stopping 0.28/0.34; turn on water). Validation, 288 segments driven/held: water 0.133/0.018, land 0.058/0.000, air 0.000/0.000; 84% of driven air segments ballistic; hold 2% of driven water segments (one holder 0.333 vs held max 0.037). "No controller in arch42 held depth better than a body doing nothing" | water/land pass; air fails "not because of the shape"; ran in arch42 to gen 208 | 6810-6868 |
| Z | 09-22 | films reproduce scores | 4 defects: (1) pool dropped the shared policy from every re-score and refinement trial on sharded gens, arch35-arch41; (2) trim probe reset the live sim; (3) `clearance` counted camera strips; (4) promotion overwrote the scored policy. arch41's elite filmed land 0.002 vs recorded 0.283 | fixed; not comparable across | 6870-6931 |
| Z part two | 09-22 | audit = the scored experiment | arch40 invalidated 25 of 42 audits; ×1.0 perturbation read 2.2152 → 1.0000 | fixed; floor on the base "Not done" | 6933-6961 |
| AA gait gain | 09-22 | `g = clip(1 + intent, 0, 2)` × amplitude | arch42 S3 at gen 200: water 0.092, land 0.033; hold 0% > 0.1; walk-and-stop 2.0% vs 5.5% if independent. Best bounded c left median 74% (land) / 50% (water) of amplitude. Oracle: land task 0.362 → 0.501, stop 0.356 → 0.902, progress 0.567 → 0.535; water 0.231 → 0.225, hold 0.000 → 0.000. Test: walker stop 0.000 → 0.994 | built, off by default; on in arch51 (Q2); its arch51 read is NOT STATED in the ROADMAP (pre-registered in `runs/arch51_notes.md:47`) | 6963-7006 |
| Y | 09-21 | arch40 film "only trembling in place" | see B row 4 | measured, not acted on (films → Z) | 7008-7057 |
| W | 09-20 | specialist islands select water | elites better elsewhere than own medium: arch39 gen 180 air 96.4%, land 96.4%, water 7.9%; arch38 gen 899 97.1/75.9/20.3%. Still machine: water 0.533 vs own gait 0.510 (104%), air 0.032 vs 0.029, land 0.010 vs 0.048 | curriculum island-aware (b2db2e6) "unrun"; water fixed 09-20 by passive twin (6a21af6), itself replaced by gating 09-21 (6655-6657); ladder rungs read gross: open | 7059-7122 |
| L | ≤ 09-20 | aspect ratio | "arch38 rewarded thrust and the thrust distribution moved down" | unanswered; re-ask after S | 7124-7133 |
| M | ≤ 09-20 | CPPN complexity | 18 → 73, corr +0.060; `n_parts` +0.191 | unrun in range | 7135-7142 |
| N GRPO | 10-03 | partial GRPO | GRPO stage 0.29/0.43/0.42 of (main + re-score) at batch 8; within-group std 0.00 for air and every crossing, 0.02-0.07 land/water: "the signal exists only where bodies already score" | built, off | 7144-7180 |
| P | ≤ 09-20 | `descriptor_refit_every` 400 | 19.0% median merge loss per refit (11.1-31.2%) over 660 median cells | unfixed | 7189-7194 |

### A8. arch34's phases (≤ 2026-09-05; 7196-7620)

| id | what | result | status | refs |
|---|---|---|---|---|
| arch33 baseline | 9 changes, 900 gens, 40.4 h, 13,353 evals | corr(fitness, mission) +0.159 → **+0.590**; corr(fitness, min competence) +0.003 → +0.404; lineage 4.59/16 → 10.28/25; population mission 0.01856 → 0.02745 (+0.00890 ± 0.00079, t +11.3; arch31 −0.00703 ± 0.00063); lifting-surface designs only +0.00436 ± 0.00087 (t +5.0) | 8 of 9 worked | 7202-7222 |
| arch33 ninth change | shared policy with morphology, shaping, scaling | in-distribution −0.00140 ± 0.00155 (t −0.90, 13/48 improved); held-out −0.00268 ± 0.00193 (t −1.39, 17/48) | refuted: "Four architectures have now produced four indistinguishable-from-zero results. Do not build a fifth." | 7224-7234 |
| Phase 0 | fluid cost; wingless "climbers" | panels-only fit 2.67 µs/panel + 368 µs, R² 0.216 (standing figure was 0.86 µs/panel + 233 µs); 9 of 60 full-ladder designs had `wing_area` 0.0000; 13 archived wingless designs 0.427-0.853 air → 0.0000-0.0017 re-run: "The launch was the whole score" | done | 7238-7293 |
| 1.1 | air score measures flight | seed plans old → new: gannet 0.4500 → 0.1832 (airborne 1.00, sink 1.71), beetle 0.1619 → 0.0229, eel 0.1362 → 0.0001, medusa 0.0024, ray 0.0032, bat 0.0022, teal 0.0000; best-to-next 2.78 → 7.99 | done; air before arch34 not comparable | 7297-7344 |
| 1.2 | Tier-1.5, 60 s weakest leg | `tier1_5_retention` on every promote event | built, reported not enforced | 7346-7359 |
| 1.3 | schedule aliasing | 28 refined controllers all in 2 islands (air, amphibian) | fixed (per-island visits) | 7361-7375 |
| 1.4 | Tier-0 airworthiness gates | wing loading cap 992 N/m² (derived); `GATED_AIR_CREDIT` 0.05 (0.144 vs 0.007) | built | 7377-7405 |
| 1.5 | `body_plan` telemetry | every diversity claim from it in arch30/31/33 unrecoverable | fixed | 7407-7416 |
| Phase 2 | multiprocess actors | step 1179 µs (fluid 49%, mj_step 17%, power 18%, Python 16%); 16 processes 12.7x; end-to-end 4 workers 2.18x, 8 workers 2.20x; sharding bit-identical | done | 7418-7479 |
| Phase 3 | distillation test | held-out R²: constant 0.011, shuffled −0.473, unconditioned 0.024, conditioned −0.047 (train 0.664); "Capacity is not the constraint — generalisation across bodies is"; PGA-MAP-Elites "now the indicated direction" | read; whether PGA-MAP-Elites was built is NOT STATED in range | 7481-7536 |
| Phase 4 | 7 PPO hygiene changes as one arm | arch33 hit the KL ceiling on 88% of updates by gen 450 (a fixed learning rate alone produces the signature) | "ran, unattributable" | 7538-7571 |
| Phase 5 | graft, duplication, `MAX_PARTS` 24 | graft fires 60/60 pairs, +1.33 parts | "ran; see 'What Phase 5 cost' above" (outside range) | 7573-7620 |

### A9. Comparability boundaries stated in the range

M2 (10-08): water and land competence (5311-5312). Still-machine fix (10-08): rotor bodies on anything rotor thrust touches; still reads before 10-08 used the leaky arm (5434-5437). Paired air (10-10): air competence (5635, 5679-5680). M1/refine default 0: "A comparability boundary" (5899). 10-04: transition scores (6128, 6174). 09-23..26: every fluid force, actuated motion, the air score's short exit and height term, stage 1 (6324-6326). 09-23: nothing air/water-force related or actuated (6611-6614). 09-22 Z: re-scores include the shared policy (6930-6931). X: `OBS_DIM` change = new baseline (6736-6737). 09-20 W: nothing before comparable with after (7106-7109). Eight islands (6126). Air before arch34 (7342-7345).

---

## B. Mission and crossings, in date order (oldest first)

1. **arch33 (≤ 09-05).** corr(fitness, `mission_fraction`) +0.590 (arch31 +0.159); population mean `mission_fraction` 0.01856 → 0.02745, +0.00890 ± 0.00079 (t = +11.3); with a lifting surface only +0.00436 ± 0.00087 (t = +5.0) (7209, 7214-7221). This was scored under the pre-1.1 air score, where "The launch was the whole score" (7288-7289); not comparable (7342).
2. **Tier-1.5 definition.** "the mission fraction is `min(competence)**0.5 * mean(competence) * …`" (7350-7351). The Tier-0 air gate is a 0.05 multiplier, not a hard zero, to avoid flattening `mission_fraction` for every wingless machine (7398-7403).
3. **09-20, W.** The passive twin "redefines `mission_fraction`, the islands' own scores and PPO's reward in every medium at once" (7104-7109).
4. **09-21, Y, arch40's film.** Continuous mission reproduced exactly: on-task 35%, **0/2 transitions**, 0.0 m depth; net displacement 0.10, 0.16, 0.02 m over the three 8 s legs (7010-7014). The elite (beetle plan, 2 DOF, 4.71 kg, 1.92x water density) is worth something only in water (0.767); held still it scores 0.519 there (5.24 m horizontal) (7016-7027). "the fifth run whose film shows 0/2 transitions" (7036).
5. **09-21, O.** The machine scoring air 0.768 spends 0% of its air leg airborne because the mission starts it on the beach (6686-6688).
6. **09-22, Z.** arch41's mission-best elite, filmed as the showcase drove it, scored land 0.002 vs recorded 0.283 (6875-6877); `showcase` filmed only the continuous mission (6911-6913).
7. **09-22, Z part two.** The audit divided a no-controller seed-0 run by the earned `mission_fraction`; arch42's first launch invalidated none "only because every audited base was exactly zero" (6936-6948); "A design at 0.0004 of the mission has nothing to collapse" (6959-6961).
8. **09-23.** The audit fixture switched to a gannet "because the drawn beetle's mission of 0.0009 is 0.0 under the corrected fluid model" (6617-6618). PPO's air reward is terminal; the shaping potential saturates above ~10 m (6580-6581).
9. **09-30, AK.** The numpy path never scattered a crossing's entry state: "every crossing that Tier-2, a film or an offline probe measured started somewhere the scored one had not" (peak entry speed up to 7.3 m/s apart); fixed (6420-6429).
10. **10-03, triphibian island.** arch46's 7,804 Tier-1 evaluations: weakest medium 0 in 98.2%, c2 0 in 77.6%, none clears 0.15 in all three; the generalist scores `mission_fraction`, "zero for 98.8%" (6126).
11. **10-03, R.** Shared-policy delta on the mission: 0 (6127).
12. **10-04, Y/O.** Elites cross 0-2.3% at back 0 (air_to_water 1.4% flat over 0-8 m, water_to_air 0, water_to_land 2.3/0.9/1.4/2.3/0.5/0% at 0/0.5/1/2/4/8 m, land_to_water 0) vs `advance_share` 0.5; a 200-window holds 3-5 crossings; still machines cross at or above the elite rate (air_to_water 1.8-2.3%, water_to_land up to 5.0%) (6128).
13. **10-04, ninth still leak closed.** Still machines 0 of 218 every kind; elites 0 of 218 at 0 m (their earlier 3 + 5 were still-machine crossings); one real crossing, elite 159 `water_to_land` from 2 m; "The curriculum still cannot start (0 at back 0 against `advance_share` 0.5), and now for the right reason" (6128, 6162-6163). Open: open-loop rotors cross `water_to_air`/`water_to_land` after the hold; a graded still glide earns up to 0.145 (6165-6172). Caveat: these still reads predate 10-08's fix ("Every still-machine read before 2026-10-08 used the leaky arm", 5437).
14. **10-06, arch48.** `mission_best` 0, transitions crossed 0 (all 4 kinds, PPO and ladder); curriculum on, `back` never advanced, `window_share` ~0 (5975-5976). The only `three_media` elite: continuous mission 0/2, "land on-task 98%, then never airborne" (5990-5991).
15. **10-06, why the curriculum did nothing.** "A curriculum that advances on success cannot start from a success rate of zero ... the curriculum opened the door and nothing came through" (5993-5997). (OOR 3408: the user turned it on by default 2026-10-05 knowing the 0% start rate; OOR 307-308 repeats "start rate was already 0 at back 0".)
16. **10-06, air binding.** "Every crossing except `water_to_land` needs a body that can leave a surface under its own power" (6001-6002).
17. **10-06, review/item 5.** `mission_best` 0 in arch47 and arch48, no pass/fail bar (6039); next run: "crossing curriculum off unless 1 finds a nonzero start rate" (6018).
18. **10-08, MuscleMimic.** `mission_fraction` > 0 in 7 of 4,792 arch48 evaluations (max 0.0001); refinement accepted 29 trials, changed 13 placements (5884-5888). PPO's `reward_by_tag` over arch48's last 50 updates: air 0, all four crossings 0, land 0.049, water 0.037 (5869).
19. **10-08, M1.** Mission criterion: 2/600 trials accepted, gain 2e-7 at s, 0.0 at s' (5915).
20. **10-08, arch49.** `mission_best` 0, crossings 0 every kind; `three_media` 0 gens (5233-5234). No-model gate: stage 4 `mission > 0` leaks (elite 8 / still 2 / base 1); "crossings are now non-zero for elites and zero for still machines (air→water 1, water→land 2, land→air 1 of 229: underpowered)" (5266-5268).
21. **10-08, PAPERS item 3.** "stage 4's `mission > 0` decides nothing measurable (elite missions ≤ 1e-5)" (5730).
22. **10-08, arch48 item 1 — where crossings fail** (arch48 / arch49):

    | kind | did not hold before the command | never crossed the boundary | battery / diverged / unstable |
    |---|---|---|---|
    | `air_to_water` | 4,295 / 4,103 (90% / 86%) | 459 / 486 | 38 / 147 |
    | `water_to_air` | 0 / 0 | 4,736 / 4,688 (99%) | 56 / 61 |
    | `water_to_land` | 0 / 0 | 4,771 / 4,679 (99%) | 21 / 68 |

    (5283-5287). "**The wall is not the hold gate except from the air**" — from water, "machines hold and then never reach the boundary: neither the surface (to air) nor the beach (to land) is reached in 6 s" (5289-5293). `land_to_air` mostly cut off (only 3 notes kept) (5280-5281); `land_to_water` appears only in Y/O (6128).
23. **10-08, E2.** L2 crossing "0 in every kind (arch47, arch48)"; L3 mission 0 (5532-5533).
24. **10-10, paired air.** "arch51 starts with no air-competent lineage, and the first air credit it awards will be the first turn anything in this project has made on command" (5631-5633). (OOR 167: arch51, "the mission is 0/0 for a sixth run".)

**Inconsistency the text leaves open.** `mission_best` reads 0 for arch48 and arch49 (5975, 5233) while `mission_fraction` > 0 in 7 arch48 evaluations (max 0.0001; 5885-5886) and 8 arch49 elites pass `mission > 0` in the gate (5267). Which quantity `mission_best` is (rounded, Tier-2, or the continuous mission) is NOT STATED. For PREREG prediction 1 the mission is "≤ 1e-4", not identically zero, at the evaluation level.

**Explanations the text gives for "the mission is 0"** (each as stated):
- (a) **Air cannot stay up**, so the air-side crossings fail the hold: 73 of 200 stop at `stays_up` (5289-5291; C3 5481-5491); "Air is the binding medium" (5999-6002).
- (b) **From water, the boundary is never reached** in 6 s (5291-5293).
- (c) **A success-advanced curriculum cannot start from 0** (5993-5997; 6128).
- (d) **Nothing in selection asks for a crossing under the machine's own power**: every medium starts from a placed spawn; the mission starts on the beach (7031-7035); O: 30 m air launch (7182-7187).
- (e) **The criterion is flat at zero**: refinement climbs `mission_fraction`, > 0 in 7 of 4,792 (5884-5888); the generalist's `mission_fraction` is zero for 98.8% (6126).
- (f) **The learner is starved by its reward**: air 0, all crossings 0 in `reward_by_tag` (5869); GRPO within-group std 0.00 for air and every crossing (7176-7180).
- (g) **Flapping cannot fly** (09-23 kinematic ceiling, 6447-6453, 6556-6561; 09-26 actuation wall, 6193-6197). OOR newer form: T4 "mission = 0 for twenty runs is the correct answer to the question as posed, not a search failure" (3855-3863), revised by N7 to "short of a gait that makes thrust inside the band the spar and the servo allow" (4029-4031).
- (h) **Controller chatter on the beach**: bases identified in fluid driven on sand, 72% command reversals per 40 ms step (7038-7048).
- (i) **No shared controller of this class** generalises across bodies (7520-7522).
- (j) Pre-S1: "The policy is never told what to do, only where it is" (6701).

---

## C. Definition of done (E2, E1, E3) and the 2026-10-12 answers

**E2 as first written (external review plan, 10-06), 6073, verbatim:**
> E2 | **A pass/fail definition for "done"**: for each medium and each of the four crossings, the bar, the elite count required, and the held-out seed count. Take the bars from the measured still-machine and unsearched-gait distributions, as the water bar (0.012) was | closes the review's "no quantified success criterion"; reuse the certified/underpowered table from 2026-10-06

**E2 as proposed 10-08, 5524-5534, verbatim:**
> Proposed, not adopted: it defines what the project is for, so the bars are the user's call. Built from what is already certified against still machines and unsearched gaits (PAPERS_2610, 2026-10-06 re-gate), and from rule 8 (a score must survive a fresh seed).
>
> | level | passes when | today (arch48 elites, current scoring) |
> |---|---|---|
> | L1 medium | each of air, water, land has ≥ 5 distinct elites over a bar certified against both controls, retaining ≥ 50% of it on the auditor's held-out seeds | water ≥ 0.012: 41; land ≥ 0.15: 7; air: no certified behaviour bar (C3: nothing past `stays_up`) |
> | L2 crossing | each of the four crossings is crossed by ≥ 5 elites while still machines cross 0 (`CrossingTracker` gate) | 0 in every kind (arch47, arch48) |
> | L3 mission | one design completes the continuous mission (`mission_fraction > 0` with both transitions, film reproduced) on ≥ 3 of 5 held-out seeds | 0 |
> | L4 target | L3 at 15 kg, 10 m depth, 45 min (README §"On the 15 kg / 10 m / 45 min target") | — |

**Adoption, 5536-5543, verbatim:**
> **Adopted 2026-10-12 (the user, question 4).** L1-L4 as tabled. Air's L1 bar is left open: under the paired air score (2026-10-10) no bar has been certified against still machines, and a typed one would break rule 5. arch51's still-machine read on its own population sets it. arch51's notes carry L1, L2 and L3 counts on held-out seeds at gens 150 and 300 (`docs/ARCH51_SPEC.md`).
>
> Each level is read on the auditor's held-out seeds, never on the draw that selected the design.

**arch51's read, 5545-5563, verbatim (excerpt of the load-bearing sentences):**
> **arch51's still-machine read, 2026-10-19: air's L1 bar cannot be set.** ... No arch51 elite reaches air competence 0.012, so no `comp:air` bar has an elite to certify. Every air rung an elite clears is `leak` against both controls (`makes_lift` .. `leaves_surface`: the still machine clears each as often as the elites), and `stays_up` and above are `unmeasured` (0/139 elites). Rule 5 forbids typing a bar, so air L1 stays open, now for a stated reason: the search has produced no air behaviour that a still machine cannot match. **2026-10-20:** the bar is now defined against the null (a non-reader's pair mean is <= 0), and with `--floor-tiebreak` on the air island holds bodies it can be read on; see the 2026-10-20 section. Still not met: 1/46 certified-candidate, `underpowered`. Water and land, same table: `comp:water >= 0.012` 61/139 elites, `comp:water >= 0.15` 9, `comp:land >= 0.15` 9, each certified against still and base; `land >= 3:stirs` .. `climbs_slope` certified against still only. These counts are on the selecting draw; L1 also needs >= 50% retention on held-out seeds, which the gate does not measure (arch51 audits: land 0.02, water 0.06).

**E1, 6072, verbatim:**
> E1 | **Ablation arms, one change each**: refine-steps 0 vs 2 vs promotion-only (AK's two levers are built and unread), shared policy on vs off, critic on vs off | AK has no read; one arm per lever on the same seed set

**E3, 6074, verbatim:**
> E3 | **Seeds and a test**: ≥5 seeds per arm for the arms that matter (E1), a rank-based test on `weakest_best`, `domain_best` per medium and crossing count, effect sizes reported with the interval. At 12 h per 300 generations a 5-seed arm is 60 h, so run E1 on `segment_seconds`-reduced configs first and confirm one pair at full length | a difference under one seed's spread is not reported as an effect

**Questions 4-6 as asked (5761-5765):** "4. **E2's definition of done** (proposal above): adopt, change the bars, or not yet?" / "5. **B1:** make `SearchConfig`'s defaults the CLI's (batch 16, 4 workers, shared policy), accepting new plan digests for the job path?" / "6. **E1/E3 budget:** run ≥ 5 seeds for the arms that matter (60 h per arm)?"

**Answers 2026-10-12, 5771-5782, verbatim (elisions marked "..."):**
> The user accepted each recommendation and added one condition to question 6.
> - **4, E2:** adopted; air's L1 bar is set from arch51's still-machine read (E2 section above).
> - **5, B1:** done. `SearchConfig()` now equals `ops.run search` with no flags on every field (...; mutations ... all caught). ...
> - **6, E1/E3:** funded, to run on this cloud container's CPU (4 cores, 16 GB, no GPU) instead of the GPU machine.

Then: "**The user chose (a) on 2026-10-12; built the same day.**" (5819) — option (a) = a CPU batched path so E1/E3 compare the algorithm arch51 runs (5840-5846).

**What is open, as of the range:** air's L1 bar (no certified behaviour bar; 10-20 null-referenced bar not met, 1/46 `underpowered`); L1's ≥ 50% held-out retention is not measured by the gate (arch51 audits land 0.02, water 0.06); L2 = 0 in every kind; L3 = 0; L4 unread. Bars are not harmonised across reads — the E2 table quotes land ≥ 0.15 (5531), arch49's gate certified land ≥ 0.012 and ≥ 0.055 (5256-5257), arch51 `comp:land >= 0.15` 9 (5559); quoted as written, not reconciled. E1/E3: funded, unrun.

**OOR pointer, newer state of air L1** (head of the file, not re-read in depth): 2026-10-21 "E2's air L1 is not met" (64); 2026-10-23 arch56 "R2 met for the first time ... 7/111 vs 0/111 vs 0/111", R1 0 sign-test passes (53-61); 2026-10-25 the flight criterion settled as `comp:air >= 0.012` certified against still and base, "the command-reading sign test (R1) stays the definition of done for E2 air L1 and has no replicated pass" (41-51); 2026-10-26 arch58/59 not met, R1 0 on both (26-34).

---

## D. Flight: what flies in this model and what does not

**Flies.**
- **A hand-controlled quadrotor**, `REFERENCE_PLANS["quad"]`, "never seeded": textbook cascade controller through the real air segment **0.963, 0.974, 0.987**, sink about 0, height 0.975, turn 0.95-1.0 (6252-6255); **0.86-0.93** under AE's worst-second height term, "its controller sags about a metre in turns" (OOR 2886-2889). `level_margin` 0.85 (OOR 2826). Rotor model: BEMT + Prandtl tip loss + Glauert, checked on the APC 10x4.7 SF (CP rms 0.003, CT rms 0.012, static CT 19% low) (6242-6247). The score is the 09-26 air score (pre-pair). Its water and land competence: NOT STATED anywhere (grep "quad|quadrotor": 10 hits, none in water or land). Whether it was run through `CrossingTracker`: NOT STATED.
- **Rotor elites do not.** OOR N6 (2026-10-10): the hand hover law that flies the reference quad at 0.816 "flew 0 of 24 arch49 rotor bodies (median 0.000; one body 0.446, one 0.011), and 153 of 191 rotor elites cannot lift their own weight with their upward-pointing rotors at top speed (upward thrust over weight median 0.15)" (3988-3992); identified bases move rotor speed by 0.0003 of top speed per unit coefficient, "so the twist cannot reach a rotor" (3985-3987). In range, M4's rotor air elite sits at its base gait: held-out 0.125/0.126/0.133 at update 0, base gait 0.128, still 0.006, on the old (unpaired) air score (5684-5692).
- **Gliders (the gannet).** Phase 1.1: airborne 1.00, sink 1.71 m/s, air 0.1832 (7329); 09-23 open loop: sink 1.7 m/s, 0.072 (6553-6554); medians over seeds 0-2: airborne 1.00, sink 2.50 m/s, 0.074 (6500-6501); 09-26: 1.48 m/s median sink, air 0.162 (6224-6225); 10-10 seeds test: airborne 1.00 / sink 1.51 (5644). AG: a glider's `level_margin` ≈ 1 − 1/(L/D), reaching 1 only with thrust (OOR 2828-2829). On arch49 still gannets cleared air ≥ 0.012 on 11 bodies vs elites 6; "flapping destroys a glide that holding still keeps" (5259-5263). Under the paired score: 0/13 and 0/229 (5589-5597, 5614-5620).

**Flapping, heave-only (09-23).** Best `thrust_margin` over 300 random gaits per plan (amplitude, phase, offset, frequency 1.5-12 Hz, quasi-static at trim), fixed tree: gannet **+0.651** (p90 +0.287) at 11.8 Hz, share > 0 0.98; teal +0.287 (+0.108) at 10.8 Hz, 0.56; beetle +0.208 (+0.138) at 11.4 Hz, 1.00; bat +0.057 (+0.016) at 3.0 Hz, 0.24. Old (buggy) tree: gannet +1.289 (6537-6542). "nothing reaches the 1.0 that level flight at trim needs" (6546-6548). Every best gait at 10.8-11.8 Hz while `mut_gait` drew 1.5-8 Hz; the spar check fails the gannet at 10.95 Hz (1.99x allowable, audit) (6549-6551). Open loop every plan but the gannet "falls at 6-14 m/s" (6552-6554). Cause given: one-hinge joints; 54% of arch42's powered wing joints only pitch, 33% only flap (audit); no wing can flap and feather with a controlled phase (6556-6561).

**Flapping with feathering (AB, 09-26).** Quasi-static best `thrust_margin` gannet +2.50, teal +4.51 (heave-only gannet +0.65), but `thrust_margin` "is blind to lift": the gannet's best gait pulls the machine down with 0.39 of its weight (6262-6265). Level margin `min(<Fz>/W, 1 + <Fx>/W)`, heave-only / feathering: gannet 0.84 / 0.80; teal 0.73 / **1.26** (11.3 Hz, 6.8 m/s, 43°; share ≥ 1: 0.003); beetle 0.92 / 0.57 (6266-6273) — "one teal gait in about 300, at margin 1.26, with ideal kinematics" (6190-6192). Free flight: best gaits open loop sink at 9-14 m/s, competence 0 (6275-6276). Fixed rig: gannet +2.50 gait prescribed +2.62 (+2.86 with unsteady terms), free joints −1.64; teal level gait prescribed **+1.28**, free **0.42** (6279-6282).

**Torque at 7-12 Hz.** Before the 09-23 servo fix: gannet stroke reached 0.69/0.47/0.27/0.19 at 2.2/4/7.3/11 Hz with 0% saturation ("the control law threw it away"); after: 0.97/0.98/0.98/0.98 (saturation 0/6/21/32%); beetle after: 1.04/0.98/0.99/0.65 (33/23/33/58%) (6474-6486). Teal level gait on the rig: torque-limited 60% of the time, lags 30-141° (6284-6285); a tuned series spring gives margin 0.115; a compliant drive cuts saturation to 15% but not the margin (6287-6289). "**The wall has moved from the fluid model to actuation at 7-12 Hz.**" (6196-6197). AH (OOR 2830-2841): own motors 0.365 (61%), 2x 0.543 (67%), 3x 0.494 (65%); "Heavier motors stay just as saturated, so this is not simply a torque-size problem." **Correction (OOR N7, 4016-4035):** stroke power over level-flight power "never below 3.6 in nine modelling variants" (gannet 28.1 at 2.3 Hz / 12.7 at the 7.80 Hz band top); the spar band closes at 7.80 Hz, so the only level gait (11.33 Hz) is outside the allowed band; there the stroke motor needs 22x its torque limit and the servo's corner is 2.88 Hz; "**Revised T4: flapping flight is not short of actuator power; it is short of a gait that makes thrust inside the band the spar and the servo allow.**" N15 (the foil fixture that would separate "gait never sampled" from "model an order of magnitude low") has no recorded outcome (4072-4089).

**In the search's population.** arch48 C3: no elite clears a thrust rung; 73 of 200 stop at `stays_up` (5481-5487); film elite `thrust_margin` 0.0004, `level_margin` 0.29 vs AD's 0.7 (5999-6000). D2: 34% of flappers above k = 0.3, but 54 of the 73 `stays_up` stoppers are at k ≤ 0.3 (5501-5515). S3: 84% of driven air segments fall at a ballistic average (6859-6861); 86% of air segments are in the sea by 2.65 s (6798, 6846-6847). "The best flapper ever scored 0.378" (6188-6189). Every air score is earned from a 30 m launch (7184-7186); lowering it cuts elite air competence 0.0123 → 0.0017 (30 m → 4 m) (6128).

**AG-AK, how each ended.** AG: done 09-26 (OOR 2810-2829), `level_margin` published from gen 0 (2847). AH: probed, nothing built, not re-listed (OOR 2830-2844, 3053-3055). AI: done, `mut_rotor` (6340-6341; OOR 2848-2860); the search keeps building rotorcraft (OOR 3029). AJ: done; path bars = max(old, 2x own noise floor) (6342-6369). AK: done 09-30 with one open item, the slam diagnostic above ~20 m/s (6370-6433). AD: deliberately not done in range (6295-6298); OOR gate set by the user at `level_margin >= 0.7` (3028), deferred to arch43's gen-200 read (2845-2847).

---

## E. Learner

**M4, the shared PPO on a fixed body** (5682-5707; harness `experiments/fixed_body_learning/README.md`).
- Setup: three arch48 elites (best water, land, air), 3 learner seeds each = 9 runs; fresh `SharedPolicy`, `SearchConfig` PPO settings, no per-design policy, the elite's stored bases, no identification; 30 updates × 8 copies through `evaluate_tier1_batch` with sampling on, each copy on its own training draw; deterministic held-out evaluation at 8 fixed draws before update 0 and after every 10th; controls = still machine and base gait at the same draws (README). M4's definition asked for 5 seeds (5902); the README says the size was cut from 40 × 16 to 30 × 8 "before the measurement started".
- Samples: "240 rollouts" per body-seed (5701-5702). Transitions per run: NOT STATED. **ESTIMATE** (mine): at 8 s segments and 40 ms control steps (6508, 7042) one air segment is 200 policy steps, so the air elite's own-medium data is about 240 × 200 = 4.8e4 steps per seed, below PREREG §3.6's 1e5 line, if each rollout banks one air segment; totals across all segments are larger and NOT STATED. OOR, for scale: "130 per 25k today -> ~1,040 per 200k epoch" (5025), which I read as 130 PPO gradient steps per ~25k transitions in one search update as of 2026-10-10.
- Held-out own-medium competence at updates 0/10/20/30:

  | body | seed | held-out | still / base gait |
  |---|---|---|---|
  | air elite 1 (rotor) | 0 | 0.125 / 0.045 / 0.004 / 0.003 | 0.006 / 0.128 |
  | | 1 | 0.126 / 0.057 / 0.088 / 0.143 | |
  | | 2 | 0.133 / 0.070 / 0.053 / 0.127 | |
  | water elite 152 | 0 | 0.0002 / 0.120 / 0.060 / 0.069 | 0 / 0 |
  | | 1 | 0.0003 / 0.015 / 0.046 / 0.056 | |
  | | 2 | 0.0004 / 0.012 / 0.083 / 0.040 | |
  | land elite 29 | 0 | 0.002 / 0.005 / 0.028 / 0.050 | 0 / 0 |
  | | 1 | 0.006 / 0.008 / 0.003 / 0.001 | |
  | | 2 | running | |

  (5688-5698). Water: every seed rises to 0.04-0.07, above the seeds' spread (0.029) and both controls → "the shared PPO **does learn a fixed body in water** in 240 rollouts" (5700-5702). Air: starts at the base gait's score and ends there or below, mean change −0.037: "no learning" (5702-5704). Land split; P1 turns on land's third seed (5704-5706).
- Caveats from the text: the run's air is the **old (pre-pair) score** — "its process loaded the code before the air change" (5684-5685); the air body is a rotor body whose base gait already scores 0.128 (5690). So M4's air arm did not test learning a commanded turn under the paired score. The final 9-of-9 read is not in the ROADMAP and no results file is present (A1).

**M1 — refinement keeps 8-23% of its reported gain** (5913-5938): accepted trials retain 23% (`island`) and 8.5% (`stage`) at a fresh draw; the mission criterion accepted 2 of 600. Decision: `--refine-steps` 0 by default.

**KL.** arch48: KL median 0.0062 (p90 0.0078, max 0.0091), clipfrac 0.076, `stopped_early` 0 of 300 against `target_kl` 0.015, 90-100 gradient steps per update; arch30's KL 0.0008, "barely being asked to" learn (5869). Contrast, older learner: arch33 "hit the KL ceiling on 88% of updates by generation 450", attributed to the fixed learning rate (7561-7567), before the Phase 4 changes.

**What the shared policy contributes, as stated.**
- Size and inputs: 13,133 parameters (actor 6,726, critic 6,401), obs 33 (25 + 8 morphology), action 6; task-conditioned since arch41; shaping 0.2 (5870-5872).
- Its reward: over arch48's last 50 updates air 0, all four crossings 0, land 0.049, water 0.037 (5869); air reward terminal, shaping potential `-tanh(d/5)` saturates above ~10 m (6580-6581); GRPO within-group std 0.00 for air and every crossing (7176-7180).
- With vs without (R, 10-03): land +0.0247 (t 3.50), water +0.0126 (t 2.59), air +0.0061 (t 1.69), mission 0, median 0; at 60 gens "not distinguishable from zero" (6127).
- Pre-arch34: four shared-policy architectures gave four null results (arch33's: −0.00140 ± 0.00155 in-distribution) (7224-7234); distillation: no morphology-conditioned student beats the population mean on unseen bodies, "There is no shared controller of this class to find" (7484-7522).
- Pool bug (Z1): arch35-arch41 re-scores and refinement ran without the shared policy on sharded gens, so "The shared policy trained on rollouts it took no part in scoring"; "It may be part of why the shared policy 'failed' four times" (6880-6894) — the text's hedge; the arch33 nulls at 7228-7231 are offline paired reads, not pool-scored.
- Action space: the policy commands `base + modes^T c`, c ∈ [−1, 1]; the best c leaves 74% (land) / 50% (water) of amplitude, so stopping is out of reach without the gait gain (6972-6979).
- CPU path before 10-12: the shared policy received 0 transitions (5792, 5795-5803); HostFluid fixed it (5829).
- OOR T6: "PPO learns the shaping potential, not competence ... The learner is working on the reward it was given" (3873-3879).

---

## F. The seven questions of 2026-10-08 (5734-5767) and their state

Sequence in the text: 10-09 "**Answered 2026-10-09:** 1 → B ..., 2 → the gait gain is on in arch51, 3 → M3 stays off, 7 → merged ... 4, 5 and 6 are open." (5740-5745); then 10-12 "**Answered 2026-10-12:** 4 → adopt E2, air's bar after arch51; 5 → yes (B1 done); 6 → fund E1/E3, run on a CPU container, not the GPU machine." (5736-5738).

| # | question (short) | answer | state after | refs |
|---|---|---|---|---|
| 1 | Is a passive glide a capability? Keep paying a still glider in air, or score air on a commanded difference too? | B, 10-09 | built; stopped 10-09 at the user's request with 4 `test_physics` checks failing; merged 10-10 ("the user: fix it and merge it; arch51 carries it"); gate: 0/229 arch49 elites keep air credit; arch51 is the first run on it | 5747-5752, 5565-5680 |
| 2 | Turn on `--gait-gain` in arch51? | yes, 10-09 | arch51 ran with it (`runs/arch51_notes.md:34`); its pre-registered read ("share of gliders whose amplitude the policy scales below 0.5", notes line 47) has no recorded outcome in the ROADMAP: NOT STATED | 5753-5755, 5741 |
| 3 | M3's form (median / mean with full film replay / none)? | M3 stays off, 10-09 | off. (OOR, later and on the device path: K-draw pooled mean with re-evaluated incumbents, line 13) | 5756-5760, 5741 |
| 4 | Adopt E2? | adopted 10-12; air L1 bar from arch51's still read | air L1 bar could not be set 10-19; null-referenced 10-20, not met; OOR still not met through arch59 (26-34) | 5761-5762, 5773-5774, 5545-5557 |
| 5 | B1 defaults? | yes, 10-12 | done (test + 3 mutations; no digest changed) | 5763-5764, 5775-5780, 5727 |
| 6 | E1/E3 budget (≥ 5 seeds, 60 h/arm)? | funded 10-12, on the cloud container's CPU (4 cores, 16 GB, no GPU) | CPU path had no shared policy; option (a) chosen and built 10-12 (405-409 s/gen); no E1/E3 run recorded; budget "owed by the user" (183) | 5765, 5781-5851, 183 |
| 7 | PRs #37 and #38 | merged 10-09: #37 into `main`; #38 merged into `refine-criterion`, so #39 brought the sweep and the CI fix to `main` (2026-10-08 17:24 UTC, CI green on all three jobs) | closed | 5766-5767, 5741-5744 |

---

## G. Refuted or withdrawn ideas in this range

| idea | measurement that killed it | refs |
|---|---|---|
| "Thrust is reachable in the model" (flapping) | after F-09/F-10/F-11 fixes the best `thrust_margin` halved (gannet +1.289 → +0.651); nothing reaches 1.0 | 6546-6548, 6537-6542 |
| Fluid model as the reason nothing flies | "None of them is the reason nothing flies" (09-23); D2: 54 of 73 `stays_up` stoppers at k ≤ 0.3 (10-08) | 6603-6605, 5513-5517 |
| Wingless "climbers" fly by spinning | re-run at the launch floor: 0.0000-0.0017; "The launch was the whole score" | 7265-7293 |
| `thrust_margin` as a flight signal | lift-blind: gannet's best gait pulls down 0.39 W (MATH_AUDIT C-12) | 6262-6265, 6295-6298 |
| Quasi-static level margin survives real actuation | teal 1.28 prescribed → 0.42 free joints | 6279-6285 |
| Series springs rescue the stroke | margin 0.115; compliant drive cuts saturation, not margin (OOR AH: spring 0.143) | 6287-6289; 2838-2841 |
| Torque/power is the flapping wall | OOR N7: power 3.6-28x sufficient; the only level gait is outside the 7.80 Hz spar band | 4016-4035 (corrects 6196-6197) |
| "The only bodies that fly are rotorcraft" (as said of the archive) | OOR N6: true of the hand-built quad, false of the archive (153/191 rotor elites cannot lift themselves) | 3988-3995 |
| PPO's 10 epochs over-reuse data | `stopped_early` 0 of 300; KL median 0.0062 < half of 0.015 | 5869 |
| The policy has ~1,200 parameters | 13,133 | 5870 |
| A shared controller generalises across bodies | distillation held-out R² −0.047 conditioned; four nulls; "Do not build a fifth" | 7484-7522, 7233-7234 |
| The shared policy carries weight (R) | paired deltas ≤ +0.0247 (upper bound); sweep inside one seed's swing | 6127 |
| Refinement improves controllers | accepted gains keep 8.5-23% at a fresh draw; difference vs random perturbation includes 0 | 5919-5938 |
| Identification can be cached / cut to 12 or 8 probes | bases depend on the base gait; 12 probes moved one elite 0.046 and the mean −40% | 5357-5376, 5876 |
| Median-of-3 placement | small gain in water/land/stage, none in air/island, biases single media down | 5403-5410 |
| The critic is broken | "not broken, it is uninformed" (features carry no Tier-2 information) | 5461-5467 |
| Air thrust (arch48 item 4) is the next build | nothing reaches the thrust rung (C3) | 5487-5496 |
| Crossing curriculum will lift crossings | 0 at back 0 vs `advance_share` 0.5; arch48: `back` never advanced | 5993-5997, 6128, 5975-5976 |
| Elites cross at 0 m (Y/O) | all of them were still-machine crossings | 6128 |
| The air launch can step down | removes the only air signal (0.0123 → 0.0017) | 6128 |
| arch49 gannets' air credit was flight | under the paired score 0/13 and 0/229; mean signed turn negative in 12/13 | 5589-5597, 5614-5620 |
| Typing an air L1 bar | rule 5; no elite to certify (arch51 0/139 at 0.012) | 5537-5539, 5547-5553 |
| S1 task score (arch41) | gen-50 bar failed: task medians 0 every medium | 6793-6801 |
| Tracking `1 - |v - v_cmd| / |v_cmd|` (arch42) | stopped flat at gen 102: "Tracking was the wall" | 6803-6808 |
| Hold scoring before S3 | "No controller in arch42 held depth better than a body doing nothing" | 6838-6842 |
| The passive twin | "the passive twin is gone and water is gated instead" (09-21) | 6655-6657 |
| Air short branch (fall paid more than staying up) | median 0.030/0.033 vs 0.000 for the long branch; gated | 6515-6529 |
| Geometric mean for the triphibian island | puts 0.9/0.9/0 (0.164) above 0.1 in all three | 6126 |
| AJ's first explanation (one kick) | 1e-16 kick below float resolution; difference is chaotic growth | 6342-6355 |
| The ray's printed entry orderings | "The two printed orderings are not the ray's" | 6396-6402 |
| 10-04 still check | read the commanded offset, not rotor speed (2.5% of top speed); all still reads before 10-08 used the leaky arm | 5419-5421, 5437 |
| "PPO's update is on the CPU either way" | CPU probe: PPO events at `transitions: 0`; sentence corrected | 5792, 5804-5806 |
| Review: "the controller is untrained"; ρ +0.077 current | wrong for the runs; outdated | 6032-6033 |
| Review: 50-100 refine steps, Optuna/Ray, domain randomisation, 4-8 GPU-months | not adopted, each with its reason | 6077-6091 |
| MuscleMimic: larger policy / scale; fixed seeds without rotation | not adopted | 5956-5963 |
| Fluid cost "0.86 µs/panel + 233 µs fixed" | panels explain 22%; fixed dispatch dominates | 7240-7262 |
| arch33 KL ceiling as evidence that the gradient direction rotates | "A fixed learning rate alone produces that signature" | 7561-7567 |
| L: thrust fixes aspect-ratio growth as a side effect | arch38's thrust distribution moved down; L unanswered | 7127-7133 |

---

## H. User decisions and hard constraints stated in the range (quoted)

- "Proposed, not adopted: it defines what the project is for, so the bars are the user's call." (5524-5525); "**Adopted 2026-10-12 (the user, question 4).** L1-L4 as tabled." (5536)
- Air pair: "**Stopped at the user's request on 2026-10-09 with four `test_physics` checks failing**" (5567-5568); "*Merged 2026-10-10* onto `failure-theory-1010` as branch `a51-air` (the user: fix it and merge it; arch51 carries it)." (5637-5638)
- "**Answered 2026-10-09:** 1 → B (air is scored on a commanded difference too; in progress, below), 2 → the gait gain is on in arch51, 3 → M3 stays off, 7 → merged" (5740-5742)
- "The user accepted each recommendation and added one condition to question 6." (5771); "**6, E1/E3:** funded, to run on this cloud container's CPU (4 cores, 16 GB, no GPU) instead of the GPU machine." (5781-5782); "**The user chose (a) on 2026-10-12; built the same day.**" (5819); B1 "**done 2026-10-12** (the user)" (5727)
- "## 2026-10-03 — the eight-item pass (the user: do not start arch47 yet)"; "The user's list, in their words: R sweep; AM noise floor and build; AJ on by default; Y/O measured; N if it can be done; item 6 tested; one new island that covers all three media at once; the queue pool now, and keep looking for the real fix to rotor cost." (6114-6119)
- "The user asked for three things: take the AB-AF list through to verification and fixes, add a propeller model as a comparison, and close every open item in the fluid model." (6178-6180)
- **Rotors:** "**AI. Should the search build rotorcraft?** Yes (the user, 2026-09-26, and again 2026-09-30); `mut_rotor` built for arch43." (6340-6341). OOR: "**Decisions, 2026-09-30 (the user).** AD's gate is `level_margin >= 0.7`. The search keeps building rotorcraft (AI). The mission stays one triphibian machine, not a chain of pairs." (3028-3030). Whether rotor-assisted flight is accepted as *the* answer to flight is **not decided in the text**: OOR T4 calls it "a decision (rotor-assisted flight as the accepted answer, or a different actuator spec), not a build" (3862-3863); N7: "If confirmed, the decision goes to the user: accept rotor-assisted flight as the design answer ..., or change the actuator specification" (4010-4012); after N7: "The user's decision (rotor-assisted flight as the answer, or a different spec) stands, with the spec question now 'stroke bandwidth and amplitude', not 'power'." (4033-4035). No recorded answer (grep "rotor-assisted": 3862, 4010, 4034 only). Nothing in the range says the machine must flap.
- "Asked by the user: why can nothing fly, is the reward reasonable, and how rigorous is the fluid model." (6439)
- arch40: "The baseline is re-run from scratch on the new code, at 600 generations rather than 900 at the user's instruction." (6656-6658)
- X: "The user's objection: "if the operation's purpose is to go forward, forward should score; if it is to hover in a medium, it should hover then, and movement should be penalised — not one reward applied everywhere."" (6696-6699); "**Decided by the user ("照建議做"), and built**" (6741); arch41: "the user decided not to relaunch." (6800-6801); arch42: "(`runs/arch42_stopped_gen102_flat`; the user: "停掉，改任務計分的形狀")" (6804); S3: "Progress may be passive — the user's rule allows a glide, and forward motion in water through the mechanism — so it is reported, not barred." (6814-6816)
- Z: "The user, after item Y: film each medium from where it was evaluated, beside the continuous mission, "and make it happen every session, automatically, or I keep judging from the wrong video."" (6872-6874)
- AA: "Stopped at gen 208 by the user ("現在停，立刻做步態增益")." (6969-6970)
- Y: "The user, watching `runs/showcase_arch40/mission.mp4`: "it is only trembling in place."" (7010-7011)
- **What "done" is:** E2 L1-L4 adopted 10-12, with L4 = "L3 at 15 kg, 10 m depth, 45 min" (5534, 5536); each level read on held-out seeds, "never on the draw that selected the design" (5542-5543). The 15 kg / 10 m / 45 min goal is the external review's framing of the project goal (6023-6024).
- OOR: the distance curriculum was "turned on by the user's decision" on 2026-10-05 with the 0% start rate known (3408).

---

## Decision-relevant summary

1. In range the only flying machine is a hand-controlled quadrotor that was never seeded (0.963-0.987; 0.86-0.93 under AE). Its water and land scores are recorded nowhere. Newer text says rotor elites cannot lift themselves (153/191) and the hand law flies 0 of 24 arch49 rotor bodies.
2. Crossings from water hold and then never reach the boundary (99% in arch48 and arch49). From the air they fail the hold (86-90%), because the bodies cannot stay up. The crossing curriculum's start rate is 0 at back 0, so it can never advance.
3. M4 "air no" was measured on the old unpaired air score, with a rotor body whose base gait already scores 0.128. The 9-of-9 read is not in the ROADMAP and no results file exists, yet the 10-10 theory (T6, 3877-3878) treats "M4 learns nothing in air" as settled.
7. The water-side crossings read against PREREG §3.2: the hold is passed (0 "did not hold") and the go phase fails (99% never reach the boundary). Only the air side fails the hold. The range never records a purpose-built reference (quad, scripted crossing) run through `CrossingTracker` (PREREG §3.8: NOT STATED), and the quad's water and land competence is recorded nowhere in the file.
4. The flapping wall moved twice: from the kinematics (09-23) to actuation torque (09-26), then (newer N7) to "no in-band gait". N15, the fixture that would decide between those readings, was never run.
5. E1/E3 are funded and unblocked (HostFluid) but were never run. M5 waits on M4. The 10-08 sweep defaulted refinement off (it keeps 8-23% of its reported gain). The KL bound never bound in arch48 (0 of 300).
6. Rotorcraft are allowed (09-26, 09-30). The text never records whether rotor-assisted flight is accepted as the answer to flight.
