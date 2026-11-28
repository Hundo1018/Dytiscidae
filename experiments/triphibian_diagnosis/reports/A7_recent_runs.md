# A7: the frontier, arch51..arch60

Written 2026-11-10 by investigator A7, read-only, main at a7770c2. Nothing was run
except short Python reads of stored files (no rollout, no search, no import of
mujoco or torch).

Sources read in full: `runs/arch51_notes.md` .. `runs/arch60_notes.md`, `docs/ROADMAP.md`
1-640 (all of the dated head, not only 1-125), `experiments/air_turn_bar/README.md`,
`docs/PAPERS_2610.md`, `docs/AUTONOMY.md`, `docs/ARCH46_SPEC.md`. Read in part:
ROADMAP 5522-5563 (E2) and 5734-5782 (questions to the user), `docs/IMPL_1010_SPEC.md`
850-920 (§D), `experiments/no_model_gate/D_M0_read.md`, the gate tables
`experiments/air_turn_bar/gate_*_table.md` and
`experiments/no_model_gate/results_arch51_table.md` and `results_arch49_airpair_main_table.md`.

Labels:
- **[path:line]**: copied from a document.
- **[COMPUTED: file]**: tallied here from stored rows (`*.rows.jsonl`, `generations.jsonl`,
  `archive_air.json`, `checkpoint.json`). No document carries these numbers unless a
  [path:line] sits beside them.
- **[INFERRED]**: my reading. Not stated anywhere.
- **NOT STATED**: the documents do not give it. **UNMEASURED**: no measurement exists.

---

## 0. The findings that bear most on the decision

1. **Nobody has read arch60 and it is incomplete.** It stopped at gen 112 of 150. The
   journal shows `Stopping arch60loop4.service` at 2026-11-01 21:56:30, with no kernel OOM
   line between 20:41 and 06:00. A gdm greeter started at 21:56:20, so the user session
   probably ended [INFERRED]. The last checkpoint is gen 100 (19:43). There are no
   `gate_arch60*`, `screen_arch60*` or `l1_read` outputs. Its frozen reads (S1, R1-R3,
   D-A6, D-A7, D-A8) and its falsifier are UNMEASURED.
2. **The pinned worktrees do not pin when the unit's cwd is the main tree.** With
   `cwd=main` and `PYTHONPATH=<tree>`, `sys.path[0]` is `''` (the cwd), so `dytiscidae`
   resolves to the main tree (probe in §G.3). Evidence:
   - arch56's provenance is 895fe4f, against a pinned 0eddc9a. The code diff between them
     is empty, so this had no effect.
   - arch60's gen-100 provenance is a7770c2, against a pinned 25c7a70. Its gens 41-112
     (second pass) carry a `resident_mb` generation field that only main's
     `loop.py:2455` writes (e73a180).

   `CLAUDE.md:190-196` says the opposite.
3. **Water, land and the crossings are not unmeasured. They are unselected and
   un-re-measured.**
   - Every Tier-1 evaluation in every run scores all three segments and all four
     crossings (`batchroll.py:1163-1244` in the arch60 tree). The step count
     `cost.steps.main.transitions` is above zero in every generation of all ten runs
     [COMPUTED].
   - The gates on arch53-59 report water, land and crossing bars at recorded seeds.
   - Since arch51 (2026-10-19) the only island has been air, whose stored brief reads
     "owe nothing to the water", and `mission_weight` has been 0.0. Whether scout or
     curriculum terms read the other media was not checked. `mission_best`,
     `weakest_best` and `three_media` are absent from every arch52-60 generation record.
   - No fresh-seed re-measure of water, land or a crossing has been made, apart from the
     auditor's running telemetry. The screens record air only.
4. **Air.**
   - Existence (`comp:air >= 0.012`) is certified pooled over 7 populations on fresh
     seeds: 34/946 vs 0/946 [README:462-466].
   - Command reading (R1) has 0 replicated passes [README:380]. The best single pass is
     arch54 elite 0 at 7/8 [ROADMAP:81-83]. arch56 stage 2 found 0/16 at Bonferroni
     [README:336-346].
   - The best R3 is 3 of the 5 elites needed.
   - The "best configuration" (arch56) did not replicate on a second seed: arch58's R2
     was 1/114, underpowered [arch58_notes.md:20-24].
5. **No crossing has been measured at a fresh seed since 2026-10-04**
   (ARCH46_SPEC's `transition_distance` check).
   - At recorded seeds since arch49 the best counts are 2/111 (arch56 `air_to_water`,
     still 0, base 0) and 4/145 (arch55 `water_to_land`, a leak against base).
   - `water_to_air` was cleared by no elite in any gate (arch49, arch51, C-on, arch53-59).
   - E2 L2 (5 per kind) is not met for any kind.
6. **The mission's weight in fitness (`mission_weight`) has been 0.0 since arch51**
   (0.30 in arch50 [COMPUTED: config diff arch50->arch51]). In arch52-60 the only island
   is air, whose stored brief reads "fly as well as the physics allows; owe nothing to the
   water" [COMPUTED: run_start `islands`]. Its headline was last reported in
   arch51 (0.0 at gen 899 [ROADMAP:627]). At recorded seeds, 1-4 elites per air
   population have `mission_fraction > 0`, at <= 4.29e-05, and none of them crossed
   anything [COMPUTED: gate rows].
7. **Three bodies clear all three media floors (0.012) at their recorded seed**: C-on
   idx 6, arch56 idx 21 and arch59 idx 137. None of them crossed anything. At fresh seeds
   their air keeps 0.01, 0.54 and 0.42 of the recorded value, and their water and land
   were never re-measured (§H).
8. **The still machine stays aloft through the turn command more often than the
   actuated elite in 6 of 8 screened populations** (equal in arch58, lower only in
   arch57): arch56 still 23.8% vs elite 17.5% of draws. The still arm's battery exhaustion ranges from 0% to 72.1% across populations
   (arch57: 72.1% still vs 72.0% elite) [COMPUTED: screen rows; arch53, arch55 and arch57
   are in no document]. Exhaustion is therefore largely a booking artefact, which is the
   open `PowerBudget` decision.

---

## A. Run table arch51..arch60

### A1. Identity (what ran)

Flags are diffed from each run's stored `run_start` config [COMPUTED:
`runs/archNN/generations.jsonl` line 0]. Code changes that have no flag are named.

| run | dates | gens planned / ran | islands | evaluator | differs from predecessor | seed | commit: notes / provenance | refs |
|---|---|---|---|---|---|---|---|---|
| arch51 | launched 2026-10-18, done 2026-10-19 (36.6 h) | 900 / 0-899 | all eight | batched | vs arch50: `gait_gain` 0->1; `mission_weight` 0.3->0.0; `reeval_per_generation` 4, `reeval_depth` 8 (new). Code: competence floor and floor-only ranking (N5), paired (mirrored) air score, seeds scored with the shared policy | 20261018 | 443d0c6 / 443d0c6 | arch51_notes.md:1,14-18,29-48; ROADMAP:601-606 |
| arch52 | 2026-10-20; stopped at gen 100 for the arch53 handover; never resumed [INFERRED: records end at gen 99, no later note] | 300 / 0-99 (checkpoint and archive at gen 80) | air | batched | `--floor-tiebreak`; `--islands air`; generations 900 -> 300; seed 20261018 -> 20261020 | 20261020 | 2fa9b84 / 2fa9b84 | arch52_notes.md:1,11-18,33; ROADMAP:466-467 |
| arch53 | 1st launch 10-20 18:01 void (stopped ~gen 55, deleted); 2nd 10-20 ~19:50; reboot 10-21 02:04 at gen ~171, resumed from gen 160; done 10-21 | 300 / 0-299 | air | batched | `--air-turn-reward 1.0` | 20261020 | e473ed5 / e473ed5 | arch53_notes.md:9-17,32-41,73-76 |
| arch54 | launched 10-21, read 10-22 | 300 / 0-299 | air | batched | `--refine-sigma 0.02` (`controller_refine_sigma` 0.1->0.02) | 20261020 | 9a3a864 / 9a3a864 | arch54_notes.md:1-8 |
| arch55 | launched and read 10-22 | 300 / 0-299 | air | batched | `--policy-log-std -2.0` | 20261020 | 25cb6b2 / 25cb6b2 | arch55_notes.md:1-8 |
| arch56 | planned 10-22, read 10-23 | 300 / 0-299 | air | batched | vs **arch54**: code only (0eddc9a, `islands.floor_rank = 0` for a draw that ended in a failure). The config differs from arch55 only by log_std back to -0.5 | 20261020 | 0eddc9a / **895fe4f** (docs-only diff, see §G.3) | arch56_notes.md:1-6; ROADMAP:386-396 |
| arch57 | launched 10-23; reboot 10-23 20:52 at gen ~13, resumed 10-24 00:55; read 10-24 | 300 / 0-299 | air | batched | `--shared-learner ppo+grpo --grpo-group 8 --grpo-bodies 4 --grpo-measurable-first` | 20261020 | bc55751 / bc55751 | arch57_notes.md:1-10,21-23 |
| arch58 | started 10-24 16:04 (concurrent with arch59); **resumed after gen 224 (not in the notes)**; read 10-26 | 300 / 0-299 | air | batched | vs arch56: seed only | **20261024** | de427a3 / de427a3 | arch58_notes.md:1-6; [COMPUTED: run_start `resume=1` after gen 224]; runs/arch58_chain.log |
| arch59 | started 10-24 16:21; **resumed after gen 139 (not in the notes)**; read 10-26 | 300 / 0-299 | air | batched | vs arch56: `--placement-draws 4` | 20261020 | de427a3 / de427a3 | arch59_notes.md:1-9; [COMPUTED: run_start `resume=1` after gen 139] |
| arch60 | planned 10-26; dry runs 1-6 on 10-29; launched 10-29; 7 process starts across ceiling stops, outside stops and OOMs [COMPUTED: run_start lines]; OOM-killed at gen ~112 on 11-01 20:40; relaunched 20:58; **stopped 21:56:30, not resumed** | 150 / 0-112 (checkpoint gen 100) | air | **device** for air and water (K_m 50, K_s 10, pooled mean); land and crossings on batched | vs arch56: `--evaluator device --k-mean 50 --k-sampled 10`, `device_streams` 4, 150 gens, `--memory-ceiling-mb 9000`, `MemoryMax=11G`. mujoco 3.15 [INFERRED from dates] | 20261020 | 25c7a70 / **a7770c2** (gens 41-112, second pass, ran main's code, see §G.3) | arch60_notes.md:3-6,34-38,54-75; ROADMAP:11; journal of `arch60loop4` |

Reference pilot, not one of the ten: **C-on** (`runs/atb_c_on`). Air island, 40 gens,
seed 20261019, arch51's config plus `--floor-tiebreak`, run 2026-10-20 [README:133-153,
232-262].

mujoco: per `D_M0_read.md:21` the `.venv` was still 3.11.0 on 2026-10-25. Per
`ROADMAP:19-20` it was 3.15.0 from 2026-10-26. arch51-59 had finished by 10-25, so they
are 3.11; arch60 is the only 3.15 run [INFERRED from dates; checkpoint provenance carries
no mujoco version, COMPUTED]. Water and land are not comparable across that boundary
[CLAUDE.md:284-290; ROADMAP:19-24].

### A2. Hypothesis, pre-registered reads, measured values, verdicts

**arch51.** Hypothesis: a competence floor fixes the selection scalar; Extract-ME
re-evaluation fixes retention [arch51_notes.md:5-10,56-68].

| read (frozen) | predicted | measured | verdict |
|---|---|---|---|
| N5 `fitness` median, gens 0-49 | < 0.3 | 0.0 | met |
| N5 Spearman(fitness, own competence) | >= 0.7 | 0.97 | met |
| N5 elites with own-medium comp >= 0.012 | >= 40% | 22.8% (145); arch49 22.4% | **missed**: N5 falsified |
| N11 held-out retention, per medium | >= 0.5 | land 0.02 (n=9), water 0.06 (n=2), air not measurable | **missed**: N11 falsified |
| N11 Tier-1->Tier-2 Spearman | >= 0.3 | water 0.093, land 0.225, air undefined (Tier-1 air 0 in all 468) | missed |
| descriptions | | `mission_best` 0.0, `three_media` 0 at gen 899 | |
| gait-gain read (air island: share of gliders scaled below 0.5) | | NOT STATED | |
| E2 air L1 bar | settable from arch51 | no elite reaches air 0.012; 139/139 `unmeasured` at 8 fresh draws (3 of 1,112 draws airborne through the command) | cannot be set |

Refs: ROADMAP:610-627, 636-638, 5545-5553; README:157-184. [COMPUTED: arch51
generations.jsonl] `weakest_best` was 0.0 and `three_media` 0 in all 900 generations. The
air archive at gen 899 holds 69 elites and 0 gannets (eel 42, random 23).

**arch52.** Hypothesis: with the tiebreak, more generations make the reader reliable
[arch52_notes.md:4-9].

| read | predicted | measured (gen 100) | verdict |
|---|---|---|---|
| R1 8-draw sign test (seeds 2026102100..07) | >= 5 passes by gen 300 | not read | UNMEASURED (run stopped at 100) |
| R2 gate `comp:air >= 0.012` vs still and base | certified at gen 300 | not read | UNMEASURED |
| R3 retention >= 0.5 | >= 5 elites | not read | UNMEASURED |
| descriptions | | evaluations 1,164 (gannet 670, 58%); air >= 0.012 in 3 evaluations (max 0.056); clean positive pair means 3; PPO non-zero air return 0 of 100; air archive 141 elites (102 gannets), 0 with air >= 0.012; 108 s/gen | used as arch53's control |

Refs: arch52_notes.md:20-40.

**arch53.** Hypothesis: a dense, null-safe air reward (pair mean <= 0 for any non-reader)
lets PPO learn to turn the gliders the tiebreak keeps [arch53_notes.md:4-7, 15-17].

| read | predicted | measured | verdict |
|---|---|---|---|
| L1 PPO air return non-zero within 50 gens | non-zero | 25 of 50 (arch52 0 of 50) | met |
| L2 clean positive pair means by gen 100 | > arch52 | 26 (max 0.576) vs 3 | met |
| L3 air elites >= 0.012 at gen 100 | >= arch52 + 2 | 7 of 85 (max 0.202) vs 0 of 141 | met |
| R1 sign test, 8 fresh draws | >= 5 | 0 of 145 (17 measurable at >= 4 draws, 128 unmeasured; null expects 0.32) | refuted |
| R2 gate | certified | 3/145 vs 0/145 still: underpowered | refuted |
| R3 retention | >= 0.5 for >= 5 | 0.203->0.014, 0.013->0, 0.033->0 | refuted |
| failure type | | 117/145 elites aloft on 0 of 8 fresh draws; 8.3% of measurable draws positive; above-floor count fell 7 (gen 100) -> 3 (gen 300) | |

Refs: arch53_notes.md:49-91; ROADMAP:470-497. The air return was positive on 10 of the
last 50 updates [arch53_notes.md:89-90].

**arch54.** Hypothesis: promotion refinement at sigma 0.1 is B''s crash regime; 0.02
keeps gannets aloft [arch54_notes.md:4-8].

| read | predicted | measured | verdict |
|---|---|---|---|
| S1 elites measurable on >= 4/8 draws | > 12% | 25/149 = 16.8% (arch53 11.7%) | met |
| S2 promoted gannets' fresh measurable share | > arch53 | NOT STATED in the final read | UNMEASURED |
| R1 | >= 5 | **1** (elite 0, gannet, 7/8, p = 0.035; null 0.59; retention 0.43) | refuted (first pass ever) |
| R2 | certified, >= 5 | 6/149 vs 0/149 still: underpowered | refuted |
| R3 | >= 5 | 1 of 6 (0.017->0.037), corrected from 0 | refuted |
| `rung:air >= 6 stays_up` | | 46/149 vs 24/149 still: certified against still (first time) | |
| positive share of measurable draws | | 16.0% (arch53 8.3%) | |

Refs: arch54_notes.md:10-46; ROADMAP:78-83, 406; README:429-432.

**arch55.** Hypothesis: lower initial PPO exploration keeps gannets aloft. Offline read:
log_std -0.5 keeps 3.5% of rollouts aloft, -1.5 29%, -2.5 40% [arch55_notes.md:4-8;
README:293-305].

| read | predicted | measured | verdict |
|---|---|---|---|
| X1 measurable evaluations, gens 0-100 | > 6.0% | 38/1,190 = 3.2% | refuted |
| X2 positive share of measurable draws | > 16.0% | 9.9% | refuted |
| S1 | > 16.8% | 12/145 = 8.3% | refuted |
| R1 / R2 | >= 5 / certified | 0 / 1/145 underpowered | refuted |
| R3 | >= 5 | NOT STATED | |
| gannet draws "battery exhausted" | | 272/840 (32%) | |

Refs: arch55_notes.md:10-31 (not adopted; log_std stays -0.5).

**arch56.** Hypothesis: the time-aloft tiebreak favoured rotor hover, which exhausts the
pack (8-15 kW against a 5.25 kW limit, then a joint runaway booked as consumption).
Unranking failed draws removes it [ROADMAP:391-394].

| read | predicted | measured | verdict |
|---|---|---|---|
| F1 gannet fresh draws battery exhausted | < 51% | 134/832 = 16% | met |
| F2 air-archive gannets with rotors | < 59% | 25/104 = 24% | met |
| S1 | > 16.8% | 26/111 = 23.4% | met |
| R1 | >= 5 | 0 of 111 (16 elites with >= 1 positive draw; null 0.57) | refuted |
| R2 | certified | **7/111 vs 0/111 still vs 0/111 base: certified** (first on any population) | met |
| R3 | >= 5 | 3 of 8 above-floor elites | missed |
| trend | | measurable share 5.7, 5.4, 11.3, 16.0, 10.9, 9.4% per 50 gens; positive PPO air updates 5, 13, 19 per 100 gens; extension rule says no | not extended |
| stage 2 (16 elites x 24 fresh draws, Bonferroni) | >= 1 pass | 0 of 16 | falsified |
| stage 3 (idx 32, 88 on 24 more draws) | replicate | idx 32 10/23 (p 0.80); idx 88 0 of 4 | refuted |

Refs: arch56_notes.md:8-29; README:321-362; ROADMAP:398-424.

**arch57.** Hypothesis: a learner gap; GRPO over the bodies aloft gives a relative
advantage [arch57_notes.md:4-10].

| read | predicted | measured | verdict |
|---|---|---|---|
| G1 positive-air PPO updates per window; groups used | higher than arch56 | 3 / 6 / 5 per 50 gens (arch56 5 / 13 / 19 per 100); 72-81% of GRPO groups zero-variance | refuted (gen 150) |
| G2 positive share / S1 | > 21.1% / > 23.4% | 17.3% / 10/136 = 7.4% | refuted |
| G3 R1 / R2 / R3 | >= 5 / certified / >= 5 | 0 / 1/136 underpowered / 0 of 1 | refuted |
| G4 wall | <= 1.5x of 117 s | 148 s/gen (1.27x) | met |

Refs: arch57_notes.md:25-54; ROADMAP:426-438. Not adopted.

**arch58.** Hypothesis: arch56 replicates on a second seed [arch58_notes.md:2-6]. Its
decision table was frozen on 2026-10-25 [arch58_notes.md:8-17].

| read | predicted | measured | verdict |
|---|---|---|---|
| R2 | certified | elite 1/114, still 0/114, base 0/114: underpowered | **not certified**: "arch56 was seed luck; the standing best configuration is unproven" |
| S1 | 23.4 +- 5 | 21/114 = 18.4% | edge of the band |
| positive share | | 26.2% | |
| R1 / R3 | | 0 (15 elites with a positive draw; null 0.34) / 1 of 1 | |
| fresh existence (8 draws) | | 7/114 vs 0/114: certified | |

Refs: arch58_notes.md:19-24; README:443-466.

**arch59.** Hypothesis: the median of 4 draws is a less lucky label [arch59_notes.md:2-9].

| read | predicted | measured | verdict |
|---|---|---|---|
| P1 R3 | >= 5 (arch56 3 of 8) | 3 of 5 above-floor (140: 3.94, 30: 1.81, 52: 1.16) | the README applies the refuted row; see the denominator caveat in §G.2 |
| P2 R2 | certified, >= 5 | 5/146 vs 0 vs 0: underpowered | refuted |
| P3 S1 | >= 23.4% | 11/146 = 7.5% | refuted |
| R1 | >= 5 | 0 (11 elites with a positive draw; null 0.21) | refuted |
| positive share | | 49.2% (on 11 measurable elites) | |
| battery / rotor share | | 50.6% (591/1,168) / 29.5% (43/146) | |

Refs: arch59_notes.md:11-15; README:443-471; ROADMAP:26-39. [COMPUTED: arch59
archive_air.json] **The final air archive holds no gannets**: teal 66, eel 80. The R3
retainer idx 140 is a teal.

**arch60.** Hypothesis: placement noise is the wall. A pooled K = 50 mean, with
re-evaluated incumbents, corrects lucky entries [arch60_notes.md:8-10].

| read (frozen) | predicted | measured | verdict |
|---|---|---|---|
| S1 vs arch56 23.4% | | none | UNMEASURED |
| R1 >= 5; R2 certified; R3 >= 5 (arch56 3 of 8) | | none | UNMEASURED |
| D-A6: 64 fresh draws vs recorded pooled mean, also on arch56 | | none | UNMEASURED |
| D-A7: generation wall vs arch56 | | [COMPUTED: `t` deltas] median 195 s (gens 0-20), 190 s (21-40), 250 s (41-80), 281 s (81-112), against arch56's 117 s [arch57_notes.md:19] | not read formally |
| D-A8: still machine on the device | air 0/N; water <= batched + 0.005 | none | UNMEASURED |
| falsifier: R3 not better than 3 of 8 at gen 150 | | gen 150 not reached | UNREADABLE |

Refs: arch60_notes.md:12-22. [COMPUTED: arch60 archive_air.json, gen 100]: 187 elites
(184 gannets); gannets with rotors 95/184 (52%); recorded (device pooled-mean) air >= 0.012
on 1/187 (0.043, gen 55); water >= 0.012 on 31, land >= 0.012 on 25; all three on 0. Not
comparable with arch51-59 (device float32, mujoco 3.15). Also [COMPUTED: generations.jsonl]
diverged 0.18% of rollouts.

---

## B. The frontier per capability

E2 levels, adopted by the user on 2026-10-12 [ROADMAP:5529-5536]:
- **L1 medium:** at least 5 distinct elites over a bar certified against both controls,
  retaining >= 50% on held-out seeds.
- **L2 crossing:** each of the four kinds crossed by >= 5 elites while still machines
  cross 0.
- **L3 mission:** one design completes the continuous mission (`mission_fraction > 0`
  with both transitions, film reproduced) on >= 3 of 5 held-out seeds.
- **L4:** L3 at the 15 kg / 10 m / 45 min target.

### B1. Best certified numbers

| capability | best certified | run, date | null (controls) | draws | E2 status | last measured |
|---|---|---|---|---|---|---|
| **Air existence** `comp:air >= 0.012` | recorded seeds: 7/111 vs still 0/111 vs base 0/111 (arch56). Fresh seeds, mean of 8 >= 0.012: arch54 7/149, arch56 9/111, arch58 7/114 certified; pooled over 7 populations 34/946 (lower 2.7%) vs 0/946 (upper 0.3%); still reaches the bar on 0 of 7,568 draws | arch56 2026-10-23; pool 2026-10-25 | still and base (recorded); still only on fresh seeds (screens have no base arm) | 1 (recorded) / 8 (fresh) | existence criterion met for populations, not designs [README:421-427] | arch59 (gate and screen, 10-25/26); arch60 UNMEASURED |
| Air, largest values | recorded: 0.277 (arch56) [COMPUTED: gate rows]. Fresh mean of 8: 0.132 (arch56 idx 11, gannet, 6/6 measurable positive), 0.091 (arch59 idx 140, teal, 6/8 positive) [COMPUTED: screen rows] | | | | not certified per design | |
| **Air command reading R1** (8-draw sign test, p < 0.05, >= 5 elites) | 1 screen pass ever (arch54 elite 0, 7/8, p = 0.035); "0 replicated passes in every probe" [README:380]; 0 in arch53, 55-59; stage 2 0/16 Bonferroni, stage 3 0/2 (arch56) | arch54 2026-10-22; arch56 2026-10-23 | still arm <= 0 on every measurable draw, 8 populations and the stage arms (e.g. 364 draws, max -0.001) | 8; 24 (stage 2/3) | **not met** | arch59 |
| Air authority without learning (B'') | sigma 0.02 linear policy turned body 159 at 0.119 over 4/4 draws, but **0.025 on 8 fresh draws; 0 bodies > 0.1** | 2026-10-20, re-read by 2026-10-25 | still <= 0 | 4, then 8 | | `authority_s0.02_rep8.json` |
| **Air R3** (retention >= 0.5, >= 5 above-floor elites) | 3 (arch56 3 of 8; arch59 3 of 5) | | | 8 | **not met** | arch59 |
| **E2 air L1** (R1 and R2 and R3) | not met on any population | | | | **not met** | 2026-10-25/26 |
| **Water** | `comp:water >= 0.15`: 9/139 vs 0/139 still vs 1/139 base, certified. >= 0.25: 3/139, >= 0.35: 1/139, underpowered | arch51 2026-10-19 (8 islands) | still and base | 1 (recorded seed) | **L1 not met**: retention on held-out seeds 0.06 (n=2) | selected on: arch51. Gated at recorded seeds as a by-product on arch53-59: `comp:water >= 0.012` certified against both controls on arch53, 54, 55, 56, 59; against still only on arch57 (a leak vs base); underpowered on arch58. >= 0.055 only on arch53, vs still (a leak vs base) |
| **Land** | `comp:land >= 0.15`: 9/139 vs 0 vs 0, certified. >= 0.25: 5/139, >= 0.3: 1/139, underpowered | arch51 2026-10-19 | still and base | 1 | **L1 not met**: retention 0.02 (n=9) | selected on: arch51. By-product: `comp:land >= 0.055` certified on arch56 (7/111); >= 0.012 certified on arch54, 55, 56, 58, 59 |
| `air_to_water` | none certified. Best: arch56 2/111 (still 0, base 0, underpowered); arch54 1/149 (base 2/149, leak); arch49 1/229 (underpowered) | | | 1 (recorded) | **L2 not met** | arch59 gate (0 elites) |
| `water_to_air` | **0 elites in every gate read** (arch49, arch51, C-on, arch53-59) | | | | **not met** | arch59 |
| `water_to_land` | none certified. Best: arch55 4/145 eels (still 0, base 1: leak); arch49 1/229; arch46 elite 159 from 2 m back, which neither its still nor its rotors-on twin makes (a real commanded crossing, 2026-10-04) | | | | **not met** | arch59 |
| `land_to_air` | none. arch54 2/149 (base 3/149: leak); arch55 1/145 (underpowered); arch59 elite 0, base 1/146 | | | | **not met** | arch59 |
| **Mission** | `mission_best` 0.0 (arch51 gen 899). `stage4:mission>0` at recorded seeds: 1-4 elites per air population, still 0, underpowered every time; max 4.29e-05 (C-on idx 6), 1.74e-05 (arch56 idx 11, arch59 idx 137); none of these elites crossed anything [COMPUTED: gate rows]. ROADMAP:5730 says "elite missions <= 1e-5" | | | | **L3 = 0** | headline: arch51 |
| Two media / three media at 0.012 (recorded seed) | pooled arch51 + arch53..59: two 24/1,085 (2.21%), three 2/1,085 (0.18%). arch51: 11/139 two (all water+land), 0 three. C-on: 1/46 three [COMPUTED: gate rows] | | still and base clear none of the three-media bars | 1 | `weakest:c3>=0.012` underpowered wherever cleared (C-on 1/46, arch56 1/111, arch59 1/146) | arch59 |

Refs for the rows:
- `experiments/air_turn_bar/README.md:336-362, 375-411, 443-471`.
- Gate tables: `gate_arch56_table.md:24,28,30-31,34,37-38`; `gate_arch54_table.md:24,28,30,34-36`;
  `gate_arch55_table.md:24,26,28,31-32`; `gate_arch53_table.md:24,27-29,32`;
  `gate_arch57_table.md:24-27`; `gate_arch58_table.md:24-27,30`;
  `gate_arch59_table.md:24,26,28,30,32-33`; `gate_c_on_table.md:21,23,25,32,34`.
- `experiments/no_model_gate/results_arch51_table.md:22-37`;
  `results_arch49_airpair_main_table.md:24-35`.
- ROADMAP:81-83, 412-414, 5558-5563, 615, 627.
- ARCH46_SPEC.md:401-406.
- `authority_s0.02_rep8.json` [COMPUTED fields: `repl_best` 0.025, `bodies_with_replicated_mean_gt_0.1` 0].

### B2. Measured, selected and re-measured: what has not happened since when

| capability | Tier-1 score computed per evaluation? | last run with an island for it (from arch52 on, the air island alone; scout and curriculum terms not checked) | last headline telemetry | last no-model gate (recorded seed) | last fresh-seed re-measure |
|---|---|---|---|---|---|
| air | yes, every run | arch52-60 (the air island alone) | domain_best every run | arch59 | arch59 screen, 8 draws (2026-10-25) |
| water | yes, every run (`for dom in DOMAIN_CYCLE` segment loop, `batchroll.py:1163-1230` in the arch60 tree) | **arch51** (2026-10-19) | domain_best every run (not certified) | arch59 (by-product) | arch51 audits (n=2). Later only the auditor's running `mean_held_out` telemetry: final values arch53 0.074, arch54 0.224, arch57 0.084, arch58 0.092, arch59 0.454, n NOT STATED [COMPUTED: generations.jsonl]. Screens do not record water |
| land | yes, every run | **arch51** | domain_best | arch59 | arch51 audits (n=9); auditor telemetry final arch56 0.0, arch57 0.0, arch59 0.0 |
| four crossings | yes: `batchroll.py:1238-1244` runs all four kinds for every candidate; `cost.steps.main.transitions` > 0 in 100% of generations of all ten runs [COMPUTED] | **arch51** (which island weights which crossing: NOT STATED in what I read) | none: no crossing headline in the generation records; the best elite's `crossed_fraction` is 0 in every generation of every run [COMPUTED] | arch59 | **not since ARCH46_SPEC's 2026-10-04 `transition_distance` check** on arch46's elites |
| mission | yes (`mission_fraction` per evaluation) | `mission_weight` 0.0 from arch51 on (0.30 in arch50); whether any other island term reads the mission: NOT STATED | `mission_best` arch51 (computed only from the generalist island, `loop.py:2410-2418`; present in 0 generation records of arch52-60) | arch59 (`stage4:mission>0`) | NOT STATED |
| three media | yes (per-segment competences) | arch51 (triphibian island) | `three_media` and `weakest_best` arch51, 0 and 0.0 in all 900 generations (triphibian island only, `loop.py:2422-2431`) | arch59 (`weakest:c3`) | **UNMEASURED** for every three-media body (§H) |

[INFERRED] "Three media" here means three per-segment scores, each from its own placed
start (the segment loop in `batchroll.evaluate_tier1_batch`). It does not mean one
continuous run. The continuous mission is L3, and its weight in fitness has been 0.0
since arch51.

---

## C. Failure modes measured in the recent runs

### C1. Fresh-draw outcomes on the 8-draw air screens

Elite arm vs still arm, all elites in each population's final air archive.

Battery shares:
- arch54, 56, 58 and 59 match `README:485-490` exactly.
- arch53, 55, 57 and C-on are [COMPUTED: `experiments/air_turn_bar/screen_*.rows.jsonl`]
  and appear in no document.
- Gannet-only shares from the notes are given where the notes use them.

"Measurable" here is the share of *draws* with `pair_turn_factor > 0.5`. It is not S1,
which is per elite, at >= 4/8 draws; S1 is in its own column, from the notes.

| population | battery exhausted, elite | battery exhausted, still | no failure, not aloft through the command (elite / still) | measurable draws (elite / still) | diverged + unstable, elite draws | S1 [notes] |
|---|---|---|---|---|---|---|
| arch51 (8 islands) | 533/1,112 = 48% [README:161-162] | NOT STATED | NOT STATED | 3/1,112 / 3 still draws | 64 + 48 [README:161-162] | 0/139 |
| C-on | 7/368 = 1.9% | 0/368 | 91.8% / 93.2% | 6.2% / 6.8% | 0 | n/a |
| arch53 | 461/1,160 = 39.7% | **214/1,160 = 18.4%** | 50.2% / 70.7% | 8.4% / 9.8% | 1.6% | 11.7% |
| arch54 | 620/1,192 = 52.0% (gannets 604/1,176 = 51%) | 112/1,192 = 9.4% | 34.3% / 77.4% | 11.8% / 13.2% | 1.8% | 16.8% |
| arch55 | 436/1,160 = 37.6% (gannets 272/840 = 32%) | **0/1,160** | 57.8% / 95.3% | 4.5% / 4.7% | 0.2% | 8.3% |
| arch56 | 174/888 = 19.6% (gannets 134/832 = 16%) | 0/888 | 62.3% / 76.2% | **17.5% / 23.8%** | 0.7% | 23.4% |
| arch57 | 783/1,088 = 72.0% | **784/1,088 = 72.1%** | 19.1% / 21.4% | 6.9% / 6.3% | 2.0% | 7.4% |
| arch58 | 421/912 = 46.2% | 368/912 = 40.4% | 38.9% / 47.9% | 11.3% / 11.3% | 3.6% | 18.4% |
| arch59 | 591/1,168 = 50.6% | 256/1,168 = 21.9% | 42.3% / 68.8% | 6.0% / 9.2% | 1.1% | 7.5% |
| arch60 | no screen | | | | | UNMEASURED |

Readings:
- **Battery exhaustion is not mainly a property of what the elites do.** The still
  machine exhausts on 0% to 72.1% of draws, depending on the population.
  `README:492-495` calls it "a model defect for any reading that depends on the battery".
  The mechanism is a servo back-driven past `Actuator.max_speed`, booked as consumption:
  ~40 kW against a 474 W pack limit on an arch51 eel [ROADMAP:113]; 8-15 kW against
  5.25 kW, then a joint at 1,200-2,300 rad/s, on arch54 gannets [README:309-318].
- **Actuation costs time aloft.** Measurable share, still > elite: arch53, 54, 55, 56,
  59, C-on. Equal: arch58. Elite higher: arch57 only [COMPUTED]. This agrees with B'
  (held still, 5 of 8 gliders airborne; random sigma-0.5 policies 0/32 on 6 of 8 bodies
  [README:192-197]) and with the exploration read (PPO noise at log_std -0.5 keeps 3.5%
  aloft [README:295]).
- The dominant non-measurable reason in arch56, the best population, is "not aloft
  through the command" with no failure: 62.3% of elite draws.

### C2. Rotor share

[COMPUTED: final `runs/archNN/archive_air.json`; it reproduces ROADMAP:401 for arch53/54/56.]

| run | gannets with rotors / gannets | all elites with rotors | plans in the air archive |
|---|---|---|---|
| arch51 | 0/0 | 24/69 | eel 42, random 23, beetle 2, medusa 2 |
| arch52 (gen 80) | 13/102 | 32/141 | gannet 102, eel 27, random 12 |
| arch53 | 25/58 (43%) | 81/145 | gannet 58, eel 87 |
| arch54 | 87/147 (59%) | 87/149 | gannet 147, eel 2 |
| arch55 | 57/105 (54%) | 80/145 | gannet 105, eel 40 |
| arch56 | 25/104 (24%) | 25/111 | gannet 104, random 5, eel 2 |
| arch57 | 9/26 | 54/136 | gannet 26, eel 110 |
| arch58 | 3/47 | 32/114 | gannet 47, eel 67 |
| arch59 | **0/0** | 43/146 (29.5%; ROADMAP:36) | **teal 66, eel 80** |
| arch60 (gen 100) | 95/184 (52%) | 95/187 | gannet 184, random 2, eel 1 |

arch60 runs with the failed-draw rule, yet its rotor share is back near arch54's.
Whether `floor_rank`'s failure rule acts on the device path is NOT STATED (an open
question, §G.4).

### C3. Divergence

| source | arch51 | arch52 | arch53 | arch54 | arch55 | arch56 | arch57 | arch58 | arch59 | arch60 |
|---|---|---|---|---|---|---|---|---|---|---|
| `diverged_rollouts/rollouts`, pooled over generations [COMPUTED: generations.jsonl] | 1.76% | 0.18% | 0.45% | 1.82% | 0.21% | 0.08% | **6.54%** (GRPO) | 2.92% | 0.83% | 0.18% |

On the screens, diverged plus unstable is 0-3.6% of elite draws (table C1). The
documents give no divergence read for arch52-60.

### C4. Retention at fresh draws, and Tier-1 vs Tier-2

- Air R3 per run: arch53 0 of 3; arch54 1 of 6; arch55 NOT STATED; arch56 3 of 8;
  arch57 0 of 1; arch58 1 of 1; arch59 3 of 5; arch60 UNMEASURED. Refs:
  arch53_notes.md:84, ROADMAP:83, arch56_notes.md:18, arch57_notes.md:46, README:449.
- Under N11 the above-floor count fell from 7 to 3 between gens 100 and 300 in arch53
  [ROADMAP:488-489].
- The C-on gannet's in-run fresh re-evaluations were 0.5->0.0 on some and 0.0->0.5 on
  others ("a weak reader") [README:257-261].
- Water and land retention: arch51 land 0.02 (n=9), water 0.06 (n=2) [ROADMAP:615];
  arch49 below 0.35 in every medium [ROADMAP:257-258].
- Tier-1 -> Tier-2 Spearman:
  - arch51: water 0.093, land 0.225, air undefined [ROADMAP:616].
  - arch49: water 0.174, land 0.182, air 0.403 [ROADMAP:616].
  - arch48: water -0.09, land 0.02 [PAPERS_2610.md:152-155].
  - arch52-60: NOT STATED.

---

## D. The levers pulled on air, and their measured effect

| lever | where | measured effect | status | decided by |
|---|---|---|---|---|
| paired (mirrored) air score; `air_task = turn x (0.5 + 0.5 height)` | arch51 | a non-reader's pair mean is `-0.5|a-b| <= 0` (the null, rule 9). arch51: the gannet seed was rejected at gen 0 on every island; 8 gannet evaluations in 900 gens; 139/139 elites unmeasured; arch51's 14 positive pair means were rotor eels tumbling at 13-65 rad/s | kept | the user, 2026-10-09/10 [arch51_notes.md:15-17; ROADMAP:5740] |
| competence floor N5; `mission_weight` 0 | arch51 | Spearman 0.47->0.97; competent share 22.4->22.8% | kept | the user, 2026-10-10 |
| N11 re-evaluation 4 of 16, depth 8 | arch51 | retention 0.02 / 0.06 (missed) | kept | the user, 2026-10-10 |
| `--gait-gain` | arch51 | NOT STATED | on in all ten | the user, 2026-10-09, answer 2 [arch51_notes.md:38-39] |
| `--floor-tiebreak` (time aloft ranks floor members) | C pilot (40 gens), arch52 | C-on: gannet evaluations 165 vs 1; first commanded-turn air credit 0.111 (gen 27, reproduced on the single path); gate 1/46 underpowered. arch52 gen 100: 0 of 141 above floor | **default on since 2026-10-21** | the user [ROADMAP:72-74] |
| unpaired air half pays nothing (`paired_turn`) | fix 2026-10-20 | 1 of 1,112 draws affected | kept | not attributed (a rule-4 fix) [ROADMAP:583-590] |
| `--air-turn-reward 1.0` (PPO air return = competence + pair mean) | arch53 | gen 100: L1-L3 met. Gen 300: R1 0, R2 3/145, R3 0 | "stays a flag"; on in every later arm | the user, 2026-10-21 |
| `--refine-sigma 0.02` | arch54 | S1 11.7->16.8%; R1 1 (first); `stays_up` certified vs still; positive share 8.3->16.0% | kept in all later arms | the user's lever order of 10-21 [ROADMAP:74-77] |
| exploration read (offline, log_std) | 2026-10-22 | measurable share -0.5: 3.5%, -1.5: 29%, -2.5: 40%, -4.0: 38%; the 50% rule selected nothing | led to arch55 at a typed -2.0 | the user's lever order; the value was typed [README:300-305] |
| `--policy-log-std -2.0` | arch55 | worse on every read | not adopted | |
| failed draws earn no floor rank (0eddc9a) | arch56 | battery 51->16%, rotor 59->24%, S1 23.4%, R2 certified, R3 3 of 8, R1 0 | **default** | the user, 2026-10-25 [ROADMAP:457-459] |
| stage 2 / stage 3 power test | arch56 offline | 0/16, then 0/2: "the R1 miss is not a power artifact" | | |
| GRPO measurable-first | arch57 | worse on every read; groups zero-variance | not adopted | |
| second seed of arch56's config | arch58 | R2 not certified: arch56 "seed luck" | standing config unproven | frozen table [arch58_notes.md:8-17] |
| `--placement-draws 4` (M3) | arch59 | R3 3 of 5; S1 7.5%; battery back to 50.6%; gannets gone from the archive | refuted per the README | frozen table |
| device K-draw, K_m 50 / K_s 10, pooled mean with re-evaluated incumbents | arch60 | **UNMEASURED** (stopped at gen 112/150) | open | §D answer 2 was delegated (decided in the spec); §D 3 answered 2026-10-26; the launch was an advisor decision under the user's 10-26 delegation [IMPL_1010_SPEC.md:854,859-894; ROADMAP:5] |

The levers the notes name as next, and whose decision each is:
1. **The `PowerBudget.step` joint-runaway booking (lever b).** A physics-model decision
   for **the user**, "not built unasked" [ROADMAP:37-39, 445-447, 455-457;
   README:460, 494-495; arch58_notes.md:15].
2. **arch60's own fork.** If R3 does not improve on 3 of 8, "placement noise is not the
   wall (the next lever is the physics or the learner, not the score)"
   [arch60_notes.md:21-22]. Not reached.
3. **The device PPO feed (A6).** Built in f1281bd, not wired into the search, not run on
   a GPU (commit message).
4. **A4b, land and crossings on the device.** The user said yes [IMPL_1010_SPEC.md:861].
   Blocked on the device half of the settled-contact measurement [ROADMAP:9].
5. **Deferred, no owner.** Admission decoupled from the floor [ROADMAP:112] and retiring
   air rungs 1-5 [ROADMAP:115].
6. **Lever (c) from the 10-24 standing list, a longer arch56.** Only its second-seed half
   was done (arch58) [ROADMAP:445-451].

---

## E. What PAPERS_2610 and AUTONOMY say

**PAPERS_2610 (2026-10-06).** The bottleneck is the evaluation signal, not the learner
[PAPERS_2610.md:16-28]:
- Tier-1 water and land credit belongs to one task draw. A fresh seed alone takes water
  passes 15/16 -> 3/16 and land 13/16 -> 0/16: the winner's curse.
- Water competence did not separate elites from still machines (45 / 35 / 42 of 200 at
  0.15).
- The auditor's check was inert while the mission sat at 0 [:192-195].
- A side reading that bears on command reading: the policy's angular commands explain
  about 0.2% of the realised body rate [:229-235].

Its ranked list [:313-356], with status:

1. **Score Tier-1 on more than one draw.**
   - One draw per candidate: built 2026-10-08 (M2), on since.
   - Re-score incumbents: N11 (arch51).
   - k headings averaged ("the unbiased form"): A6, the device K-draw, which is arch60
     and is unread.
   - Tier-2 draws its heading: done [:268-269].
2. **Water cruise progress chosen.** Closed by the antipodal pair [:252-266].
3. **Retire or re-specify the leaking bars.** Partly read [ROADMAP:5730]. The air rungs
   are still leaking: `stays_up`, `glides` and `holds_height` leak against base in every
   population [ROADMAP:49-50]. Retiring rungs 1-5 is deferred [ROADMAP:115].
4. **Make `held_still_params` still.** Covered by the pair's cancellation [:252,
   262-265]. Rotors are stopped since 2026-10-04 [ARCH46_SPEC.md:384-388].
5. **Read arch49's `held_out`.** Done: below 0.35 in every medium [ROADMAP:257-258].

Its pre-registered reads for the next run [:305-311]:
- `mean_held_out` >= 0.5: missed in arch49 and arch51 (0.02-0.06).
- Tier-1/Tier-2 Spearman > 0.3: missed (0.093 / 0.225).
- The gate certifies every bar cleared in >= 10% of cases: not read as such anywhere I
  read.

**AUTONOMY (undated).** The direction is to remove the hand-set constants and fill in
the missing physics. "A design cannot be found if the physics that would make it work is
absent" [AUTONOMY.md:98-102]. Order of work [:136-141], with status [COMPUTED: grep of
`dytiscidae/`, so INFERRED as to "done"]:

1. **Continuous part traits replacing the kind taxonomy: not done.**
   `PART_KINDS = [HULL, ..., FIN]` is at `core/genome.py:55`. No `flatness`,
   `enclosure`, `contact_affinity` or `cavity_actuation` appears in the package.
2. **Anisotropic added mass and series elasticity: present in the code.**
   - Added mass: `physics/fluid.py:1299`; the wing added-mass tensor is on since
     2026-09-23 [`fluid.py:1345`].
   - Series elasticity: `series_stiffness` at `core/mjcf.py:656`; whether it is on by
     default is NOT STATED.
   - Passive wing pitch compliance is built and off by default [ROADMAP:185-192].
   - Free-surface deformation and wake coupling: NOT STATED.
3. **Learned descriptors: done.** AURORA at `evolution/descriptors.py:12`; arch60's
   config has `learned_axes` true and a refit every 400 evaluations. Dominance instead of
   a scalar: partial. The archive keeps a per-cell Pareto front [ARCH46_SPEC.md:194-196],
   and the N5 floor still ranks by a scalar.
4. **Evidence-based intervention: not done.** The famine weight is a fixed rule
   (`evolution/curator.py:236-350`), and there is no posterior in the package.

---

## F. ARCH46_SPEC §8 and the crossings

**Before the fix** (2026-10-03, arch45's elites, still vs base gait, 7 plans x 2 seeds):
- `air_to_water` 0.834 vs 0.693 and `water_to_land` 0.865 vs 0.861; `economy` 1.000 for a
  still machine [ARCH46_SPEC.md:49].
- `water_to_land` was credited for settling *into* the water: still machines crossed
  100% at 0, 1, 2 and 4 m back [:322-327].
- `air_to_water` was crossed by gravity: still 100% vs base 86% [:328-330].
- These were PPO's two largest tags: means 0.41 and 0.23 [:45; ROADMAP:316-319].

**Fix 1** (10-03):
- Directional crossings, starting in the start medium [:272-276].
- Commanded hold-then-go: `HOLD_SECONDS` 1.5 s, counts at hold >= 0.5; `air_to_water`
  starts level at launch speed, 4 m up [:277-282].
- Economy counts only for a completed crossing [:283-284].
- Three more leaks found on the way: a float paid 0.300 of `water_to_air`; a placement
  gap paid 0.031; the teal's `land_to_air` start sat 0.5 m off the ground [:286-297].
- After: still reward 0.000 in all four kinds; graded `water_to_land` approach 0.013;
  the base gait about 0 [:299-302].

**Fix 2** (10-04, the ninth instance). The 10-03 tracker still paid still machines at or
above the elite rate: `air_to_water` 5/218 at 0 m, `water_to_land` 8 (up to 5.0%),
`water_to_air` 1 [:363-366]. Gates added:
- an energy-height hold;
- `transitions.ashore`;
- rotors stopped in `held_still_params`;
- shore progress counted from the go command;
- an aborted probe's hold set to 0 [:369-399].

After: still machines cross 0 in every kind at every distance, and so do the elites at
0 m. One real crossing remained: arch46 elite 159 (amphibian), `water_to_land` from 2 m
back [:401-406]. Left open:
- Open-loop rotors still cross: the station gate was not built [:406-413].
- The graded `water_to_land` approach pays a still body up to 0.145 [:414-418].

**Crossings by elites since then** (all at recorded seeds; none re-measured fresh):
- arch47, arch48: 0 in every kind [ROADMAP:5532]. arch48's start rate was already 0 at
  back 0 [ROADMAP:306-309].
- arch49: `air_to_water` 1/229, `water_to_land` 1/229, both underpowered.
- arch51 (8 islands): no crossing bar cleared by any arm [results_arch51_table.md:37].
- Air-island populations:
  - arch54: `air_to_water` 1/149 and `land_to_air` 2/149, both leaks against base.
  - arch55: `water_to_land` 4/145 (leak); `land_to_air` 1/145 (underpowered).
  - arch56: `air_to_water` 2/111 (underpowered).
  - arch53, 57 and 58: none.
  - arch59: none by elites; base 1.
  - `water_to_air`: none in any population.
- Every arm since arch51 runs `--no-distance-curriculum` (stored config
  `distance_curriculum` 0 [COMPUTED]).
- arch60's crossings stayed on the batched path (A4b not built) [arch60_notes.md:28].

[COMPUTED: gate rows] Of the 10 elites that crossed anything in arch54-56 (3, 5 and 2),
only one is also above the air floor at its seed, and two have two media at floor:
- arch56 idx 72 crossed `air_to_water` at air 0.035 and land 0.013. Its fresh air is
  0.000 on 8 draws.
- arch56 idx 77 crossed `air_to_water` at air 0, water 0.035, land 0.048.
- The other eight have at most one medium at floor.

---

## G. Open questions, standing decisions, corrections, integrity

### G.1 Standing user decisions (quoted)

- E2: "**Adopted 2026-10-12 (the user, question 4).** L1-L4 as tabled." [ROADMAP:5536]
- arch51: "1. N5 + N11 in one arm"; "2. The `air-commanded-difference` branch is fixed
  and merged ... arch51 carries the paired air score." [arch51_notes.md:14-17]
- "`--gait-gain` is the user's 2026-10-09 answer 2." [arch51_notes.md:38-39]
- "**The user, 2026-10-21:** `--floor-tiebreak` is on by default from this commit ...;
  `--air-turn-reward` stays a flag; the three levers run in order: arch54 = arch53 +
  `--refine-sigma 0.02` ..., then the learner's exploration read ..., then K draws per
  body (IMPL_1010 A6)." [ROADMAP:72-77]
- "(3) `floor_rank`'s failure rule is the default: the user, 2026-10-25." [ROADMAP:457-458]
- "**2026-10-26: the user keeps the mujoco 3.15 upgrade, against D-M0.**" [ROADMAP:19]
- Arm length, on "the user, 2026-10-20: 'find a sweet spot that holds for later
  training'": mechanism arm 150, capability arm 300 read at 150/300, extended by resume
  only on a move beyond the draw spread [ROADMAP:123-143].
- IMPL_1010 §D: "§D answers (the user, 2026-10-10; questions 2, 5 and 6 delegated,
  decided here)". 1 = yes to A4b; 4 = upgrade to 3.15; 3 = "pooled mean ..., no lower
  bound", recorded as answered 2026-10-26 [IMPL_1010_SPEC.md:854, 859-898].
- "The eel `[info]` handling, the arch60 launch and the memory-ceiling settings were
  advisor decisions under the user's 10-26 delegation, not user decisions." [ROADMAP:5]
- arch60's relaunch "at original settings ... at the user's instruction" (2026-11-01)
  [arch60_notes.md:72].
- The user's goal of 2026-10-25, "找到最佳飛行判斷", answered as `comp:air >= 0.012`
  certified against still and base and confirmed on fresh seeds [README:366, 405-411].
- 10-12: E1/E3 "funded, to run on this cloud container's CPU" [ROADMAP:5781-5782].
- 2026-11-10: no new training until the analysis supports a plan that reaches normal
  triphibian locomotion within three full runs [PREREG.md:8-9].

### G.2 Open questions and corrections

**Open questions to the user, as the documents leave them:**
1. "The change to `PowerBudget.step` is a physics-model decision and is not made here;
   it is the user's to decide" [README:494-495; also ROADMAP:37-39, 455-457].
2. Standing 2026-10-24: "(a) is a build ..., (b) a physics-model decision, (c) 11 h of GPU
   for a replicate; the user picks" [ROADMAP:449-451]. (a) was taken as arch59 and
   arch60, (c) as arch58. (b) is open.
3. "The E1/E3 budget is owed by the user" [ROADMAP:183]. Whether any E1/E3 run happened:
   NOT STATED.
4. What happens to arch60: NOT STATED. Nothing was written after 2026-11-01 20:58, and
   the unit stopped at 21:56:30.
5. Open but not addressed to the user:
   - the 3.11 baseline drift between 10-08 and 10-25 [ROADMAP:22-24; D_M0_read.md:23-26];
   - the eel's water gap across the three paths [ROADMAP:3,7];
   - arch60's host memory growth, cause not located [arch60_notes.md:59,69].

**Corrections. The newer source wins:**
- arch54 R3 was 0 of 6 in the 10-22 ROADMAP table. It is 1 of 6 (0.017 -> 0.037)
  [ROADMAP:83, 406; README:429-432].
- arch59 F1/F2: "n/a" [README:473-474] is wrong. They are 50.6% (591/1,168) and 29.5%
  (43/146) [ROADMAP:34-37]. Both are all-elite denominators:
  - arch54's "51%", which ROADMAP:38 compares against, is gannet-only (604/1,176)
    [README:309-310]. Its all-elite figure is 52.0% [README:487].
  - arch59's archive has no gannets at all [COMPUTED].
- arch59's verdict: the README applies the "R3 <= 3 of 8 AND R1 0" row to a 3 of 5
  result. The denominator was 5, not 8, and the row was written for 8 [ROADMAP:29-32;
  arch59_notes.md:13-15].
- B'' authority: "body 159 replicates at 0.119 over 4/4" [README:222-230; ROADMAP:161,
  577-579] became "0.119 -> 0.025 on 8 fresh draws" [README:380;
  `authority_s0.02_rep8.json`].
- "Best configuration on record is arch56's" [ROADMAP:440-443, 10-24] became "the
  standing best configuration is unproven" [arch58_notes.md:23-24; README:454-457].
- "Not started: the first device arm" [ROADMAP:13, 10-26] is superseded by the 10-29
  launch [ROADMAP:11].
- CLAUDE.md:21-23 says "the latest finished run is arch51 ... no run is live as of
  2026-10-20". This is stale: arch52-59 finished 10-20..10-25, and arch60 is incomplete.
- PREREG.md:101-102 places the mujoco 3.15 boundary at "arch57 on?". It is arch60 only
  (§A1 note) [INFERRED from dates in D_M0_read.md:21 and ROADMAP:19-20].
- MUJOCO_GL: `egl` wrote the arch51 films on 2026-10-19 [ROADMAP:633-634;
  arch51_notes.md:101]. The newer probe (2026-10-31) says it raises `EGLError` here and
  must not be used [CLAUDE.md:80].

### G.3 Integrity of the arms (affects what any future arm is)

**1. The pin does not pin.** Probe, run here:

```
cd main; PYTHONPATH=/home/hundo/Projects/Dytiscidae/arch60-tree .venv/bin/python -c \
  "import sys, importlib.util as u; print(sys.path[:3]); print(u.find_spec('dytiscidae').origin)"
['', '/home/hundo/Projects/Dytiscidae/arch60-tree', '/usr/lib/python312.zip']
/home/hundo/Projects/Dytiscidae/dytiscidae/dytiscidae/__init__.py
```

- The chain scripts `cd /home/hundo/Projects/Dytiscidae/dytiscidae` and then run
  `PYTHONPATH=<tree> python -m dytiscidae.ops.run`. This is the case for
  `runs/arch56_chain.sh` and `runs/arch58_chain.sh`. The arch60 loops run with relative
  `tools/resume_loop.sh` paths in the main tree (journal).
- `ops/checkpoint.py:_git_sha` reads `git rev-parse HEAD` in the imported package's repo,
  so provenance names the tree that was actually imported.
- The probe and the provenance are facts about the parent process. CLAUDE.md:190-196
  phrases the pin in terms of workers. Worker import paths were not probed; spawn
  children inherit the parent's `sys.path`, so the same resolution is expected
  [INFERRED].
- A MATCH (arch51-55, 57-59) means only that main's HEAD equalled the pin when the
  process started.
- arch56: pin 0eddc9a, provenance 895fe4f. `git diff --stat 0eddc9a 895fe4f -- dytiscidae/`
  is empty, so there was no functional effect.
- arch60: pin 25c7a70, provenance a7770c2 (gen-100 checkpoint). The generation-level
  `resident_mb` field:
  - is absent in gens 0-55 (first pass; arch60r2 resumed at 17:34 on 10-31);
  - is present in gens 41-112 (second pass). That pass is arch60loop3 from 2026-11-01
    15:17, then arch60loop4; arch60loop2, at 18:42 on 10-31, wrote no generation
    [arch60_notes.md:71-75];
  - is written only by main's `loop.py:2455`, added in e73a180 at 2026-10-31 17:48:40.
- `git diff 25c7a70 a7770c2 -- dytiscidae/` touches three files:
  - `envs/devroll/rollout_wp.py` +41/-8 and `learning/ppo.py` +33 (f1281bd). Its message
    says "the search does not pass a buffer yet, so arch60 is unaffected" and "NOT run:
    it needs the GPU".
  - `evolution/loop.py` +2 (telemetry).
- `25c7a70..2c0ba4e` has no package diff, so gens 0-55 ran code identical to the pin.
  [INFERRED] The effect on arch60's scores is probably nil, but no test has verified it.

**2. Unrecorded resumes.** arch58 resumed after gen 224 and arch59 after gen 139
(`run_start` with `resume=1` in their `generations.jsonl`). Both chain logs hold only
their start lines. arch58 and arch59 ran concurrently, started 16:04 and 16:21 on
2026-10-24.

**3. How arch60 ended.**
- `arch60loop4` started 2026-11-01 21:00:37 and was stopped at 21:56:30 ("Consumed 2h
  13min" CPU) (user journal).
- The kernel journal (`journalctl _TRANSPORT=kernel`) has no OOM line between 20:41 and
  06:00.
- In the system journal, a new gdm greeter session opened at 21:56:20, 10 s before the
  stop. [INFERRED] The user's desktop session ended and took the user units with it, the
  failure mode of `runs-die-with-the-desktop-session`.
- The last record is gen 112, written twice. The checkpoint is gen 100, so gens 101-112
  would be redone on a resume.
- The earlier kill at 20:40:56 was a machine-wide OOM: `constraint=CONSTRAINT_NONE ...
  global_oom`, victim the arch60loop3 python at `anon-rss:8408888kB`. It was not the
  unit's `MemoryMax=11G`. `arch60_notes.md:75` says "killed inside the unit", which names
  the victim's cgroup, not the cause.

### G.4 Questions this read raises (not in any document) [INFERRED]

- Does the failed-draw `floor_rank` rule act on the device path? arch60's rotor share is
  52% of gannets at gen 100.
- The still arm exhausts the battery on up to 72% of draws (arch57). Are any S1 or R3
  differences between arms booking noise rather than behaviour?

---

## H. [INFERRED] The nearest thing to a three-media machine in any archive

All numbers below are measured; the selection and the reading are mine.

The only bodies in any stored gate that clear all three media floors (0.012) at their
recorded seed, with still 0/N and base 0/N on that bar [COMPUTED: `gate_*.json.rows.jsonl`,
`screen_*.rows.jsonl`]:

| body | recorded seed: air / water / land | `mission_fraction` | crossings at its seed | fresh 8 draws, air: mean, draws >= 0.012, measurable, positive | fresh water / land |
|---|---|---|---|---|---|
| C-on idx 6, gen 27, gannet, no rotors | 0.111 / 0.033 / 0.065 (reproduced on the single path [README:253-256]) | 4.29e-05 | none | 0.001, 0/8, 6, 1. ROADMAP gives airborne on 7, right way on 1, mean -0.198 [ROADMAP:563-566] | UNMEASURED |
| arch56 idx 21, gen 296, gannet | 0.114 / 0.025 / 0.016 | 1.61e-05 | none | 0.061, 2/8, 3, 2 (retains 0.54; one of arch56's three R3 retainers) | UNMEASURED |
| arch59 idx 137, gen 294, teal | 0.076 / 0.033 / 0.024 | 1.74e-05 | none | 0.032, 4/8, 8, 5 (retains 0.42) | UNMEASURED |

Neighbours:
- arch56 idx 11 (gannet, gen 253) is the best fresh-seed air elite, with 6/6 positive
  measurable draws. At its recorded seed: air 0.114, water 0.009, land 0.096
  (`weakest:c2>=0.055`).
- arch59 idx 140 (teal) reached fresh air 0.091 with 6/8 positive. It and arch54 elite 0
  are the closest to an R1 pass.
- arch56 idx 72 (air 0.035, land 0.013) and idx 77 (water 0.035, land 0.048) are the only
  bodies with two media at floor and a crossing (`air_to_water`) at one seed. idx 72 has
  no air at fresh seeds (0/8).
- The purpose-built quadrotor control flew the air segment at 0.963-0.987 on 2026-09-26
  [ROADMAP:344-346]. That is before the paired air score, so not comparable. Its water and
  land are NOT STATED in what I read.
- The scripted crossings in `test_a_crossing_is_commanded_and_a_still_machine_makes_none`
  pass the gate [ARCH46_SPEC.md:302-306]. They are unit tests, not archive bodies.

What the nearest bodies are missing, in order of distance from L3:

1. **Every crossing.** None of the three crossed any kind at its own seed. The bodies
   that did cross are different bodies, with at most two media at floor. `water_to_air`
   has never been made by an elite since arch49.
2. **One continuous run.** Their three numbers come from three placed segment starts.
   `mission_fraction` is <= 4.3e-05, and L3 needs both transitions on >= 3 of 5 held-out
   seeds.
3. **Fresh-seed reliability.**
   - Air keeps 0.01, 0.54 and 0.42 of its recorded value.
   - Water and land have never been re-measured at a fresh seed for any air-island body.
   - Those water and land values were not the selecting draw (the air island selects on
     air only), so they carry less winner's curse than the air value. They are still one
     draw each.
4. **Command reading.** None passes the 8-draw sign test. The policy's angular commands
   explain ~0.2% of body rate (arch48, PAPERS_2610.md:234-235).
5. **Magnitude.** Their water and land (0.016-0.065) are far below arch51's certified
   specialists (>= 0.15). The specialists, in turn, have no air above the floor: no arch51
   elite reaches 0.012 [ROADMAP:5548].
