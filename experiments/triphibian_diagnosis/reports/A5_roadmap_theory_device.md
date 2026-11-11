# A5: the 2026-10-10 failure theory (N1-N15), the device path (IMPL_1010), wing flexibility, the butterfly items

Written 2026-11-10 by investigator A5 at commit a7770c2. Everything here is read-only.

**What was read.**
- `docs/ROADMAP.md`, all of 3456-5214, plus 1-30. I also read three spans for context: the head summaries of the same sections (170-262), arch51's read (590-638), and the dated section headings.
- `docs/ARCH51_SPEC.md`, all of it (1-730).
- `docs/IMPL_1010_SPEC.md`: §0 (1-114), the §1 lane table (116-139), and the A-wide layout plus A1a-A7 (479-818). I also read §4, the deferred measurements (820-846), §D with its answers (850-920), and §E (922-930). Lines 145-478 (the M1, D1, P1, P2, R1 and P3 lane bodies) were **not** read.
- `runs/arch60_notes.md`, all of it (1-75).

**Numbers I computed myself (tagged A5-COMPUTED).** I read these from stored files:
- `runs/arch60/generations.jsonl` and `runs/arch56/generations.jsonl`;
- counts from `runs/arch60.log` and `runs/arch60/events.jsonl`;
- the `provenance` blocks of both runs' `checkpoint.json`;
- `runs/arch56_notes.md` lines 1 and 8;
- the user journal, `systemctl --user is-active` and `loginctl show-user`.

I used `python3 -I` with `json` only. Nothing imported mujoco, torch or warp. Nothing was written except this file.

**Keys.**
- `R:n` = `docs/ROADMAP.md` line n.
- `A51:n` = `docs/ARCH51_SPEC.md`.
- `I10:n` = `docs/IMPL_1010_SPEC.md`.
- `N60:n` = `runs/arch60_notes.md` (gitignored).

**Evidence labels.**
- **MEASURED**: a doc reports the number from a run or a probe.
- **MODELLED**: the number comes from a fit or a formula, not a run.
- **INFERRED**: reasoning, either the doc's or mine, and I say whose.
- **A5-COMPUTED**: I computed it from stored run files. It appears in no doc and nobody has reviewed it.
- **NOT STATED**: the text read does not say.

**Line-number caveat.** I10 cites ROADMAP lines as of commit 63b7ace. Today's lines are offset from those citations:
- by +739 for text before R1 (I10's "ROADMAP:3740-3776" is now R:4479-4512);
- by about +838 after R1 (I10's "ROADMAP:3830" is now R:4668), because R1-R3 (R:4555-4652) were inserted later.

A51 cites code lines at 5cc9db9. Every R line number below is today's.

---

## 0. The findings that bear most on the decision (detail and references in A-H)

1. **Random bodies beat the archive.** 300 random bodies from the genome prior were compared with arch49's gen-150 archive, about 300 evaluations each side. Random won on coverage by 1.35x and on summed competence by 1.94x, and the bootstrap 95% CIs exclude the archive's value in every medium (R:4101-4111).
   - The comparison against arch51 that R:4120-4121 announced is not recorded anywhere in R. I grepped for `random_sampling_control`, "random bodies" and "random sampl": the hits are R:218, 3624, 3868, 4092-4121 only.
2. **The competence floor (N5) fixed the ranking, not the competence.** In arch51, Spearman(`fitness`, own-medium competence) was 0.97 against arch49's 0.47. But the share of elites at or above 0.012 was 22.8% against arch49's 22.4%, where the bar was ≥ 40% (R:612-614). The ROADMAP's own reading: "the bottleneck is not the selection scalar", qualified as "a reading of one seed against one earlier run, not an effect" (R:624-626).
   - N11 retention was 0.02 on land (n=9), 0.06 in water (n=2), and air was not measurable (R:615). Both N5 and N11 are falsified (R:618).
3. **Land scores are chaotic at machine precision.** A 1e-16 change in one command moves a land score from 0.0992 to 0.0000 (R:4618-4622). Q4's 1% null control moved 34.5% of land elites by more than 50% (R:4537-4541). The ROADMAP: a land score "is one sample of a chaotic map" (R:4638-4641).
4. **Draw noise, and how many draws would fix it.** Draw variance over design variance (R) on the included elites is air 7.3, water 4.0, land 2.0. The draws needed for reliability 0.9 are K = 66 (air), 36 (water), 18 (land) (R:4667-4669). Today's path uses K_m = 1, K_s = 1 (I10:504).
5. **The class of design that can fly is neither reachable nor able.** (N6)
   - The identified bases move rotor speed by 0.0003 of top speed per unit coefficient, and 191 of 191 rotor elites are below 0.05.
   - The hand hover law that flies the reference quadrotor at 0.816 flew 0 of 24 archive rotor bodies.
   - 153 of 191 rotor elites cannot lift their own weight (R:3984-3993).
6. **Flapping is not short of power; its thrust regime is reachable, but nobody has checked the model there.** Stroke power is 3.6-28x what level flight needs (R:4014-4022). The thrust regime is reachable at about 2 Hz (St 0.2-0.4, R:4062, 4065-4068). Whether the fluid model's propulsive efficiency is right in that regime is item N15, which is **unrun**: no outcome line exists (R:4078-4090, grep).
7. **Throughput.**
   - Rollouts are 98.3% of the wall (R:4952).
   - MJWarp rigid-body alone runs 0.68x at 24 worlds (R:4142-4146).
   - The only end-to-end device-path number is arch60's (A5-COMPUTED): median 246.9 s per generation against arch56's 99.3 s, 2.49x, over gens 8-112. The batched path's step counters did not change, so the device scoring was added on top, not substituted (INFERRED). arch56 most likely ran on mujoco 3.11 and arch60 on 3.15 (INFERRED from dates).
   - arch60 stopped at gen 112 of 150 at 2026-11-01 21:56:30 and is not live.
8. **Every mission and crossing read in this range is zero.**
   - Q4's crossings: 0/58 (R:4545-4546).
   - Every seed plan scores exactly 0 in all three media (R:4224-4229).
   - arch51: `mission_best` 0.0 and `three_media` 0 at gen 899 (R:627).
   - Tier-2 completed a leg in 10 of 462 promotions (R:3830-3831).

---

## A. The theory (ROADMAP §"2026-10-10 — why the search does not progress", R:3726-4340)

### A.1 The question and its frame
- The user asked to "find why training fails or falls short, from theory first, trusting neither the repo's claims nor its fixed ideas", and to write theory and predictions before measuring anything (R:3728-3731).
- Three readers went to the code and one to the stored telemetry of arch48-50. Their reports are `runs/analysis_1010_failure_theory/{A,B,C,D}_*.md` (R:3731-3735).
- The question: "**does the selection machinery, as written, prefer a competent machine to an incompetent one at all?**" (R:3742-3744). It is posed against "the mission at 0, the best air score of any elite at 0.42-0.55, and water and land below 0.3 since the antipodal pair" (R:3739-3741).

### A.2 What was established from stored archives and code (MEASURED, R:3752-3802; code at 3d639e8)

| fact | arch48 | arch49 | arch50 (gen 120) |
|---|---|---|---|
| elites in the eight archives | 621 | 645 | 494 |
| median elite `fitness` | 0.809 | 0.828 | 0.836 |
| elites with own-medium competence < 0.012 | 82% | 91% | 91% |
| median `fitness` of those | 0.785 | 0.810 | 0.822 |
| Spearman(`fitness`, own-medium competence) | 0.27 | 0.30 | 0.30 |
| median / max evaluated `fitness`, gens 0-49 | 0.675 / 0.995 | 0.776 / 1.000 | 0.759 / 1.000 |
| elites inserted in the last 50 gens | 37% | 45% | 52% |
| Tier-2 promotions with any completed leg | 9/192 | 1/192 | 0/78 |
| median `fitness`, Tier-2 vs Tier-1 elites | 0.109 vs 0.800 | 0.095 vs 0.817 | 0.105 vs 0.829 |
| evaluations with water > 0.15 / land > 0.15 (last 50 gens) | 8.6% / 8.8% | 0.5% / 0.9% | 0.3% / 1.7% |

(R:3755-3766). One example: an air-island design scoring air 0.000, water 0.003 and land 0.000 carries `fitness` 0.945 (arch49, R:3769-3771).

What the code did (R:3777-3802):
1. **The scalar** is `0.7 * (w * isl_q + (1 - w) * cur_q) + 0.3 * mis_q`. Each term is a quantile, `mean(window <= x)`, over the island's last 256 evaluations at the cell's stage, and the window includes the candidate itself (R:3777-3782).
2. **Cells** are decided by Pareto dominance on `[blend, structural_margin, energy_margin]`, with front capacity 4 (R:3783-3785). `fitness` never enters replacement. It sets 0.35 of parent weight, and it drives migration, promotion and pruning (R:3786-3787).
3. **Tier-2** caps `fitness` at `min(fitness, f2)`; zero completed legs means `f2 <= 0.2` (R:3788-3791).
4. **Things that score nothing:**
   - The ladder enters no score.
   - The shared PPO is the controller and proposes nothing.
   - The auditor skips its invalidating check when the base mission is ≤ 1e-6, which is 98.8% of evaluations. It invalidated 2 designs in three runs.
   - The search never calls `held_still_params` (R:3792-3798).
5. **The rotor channel's offset is a speed (rad/s)** inside a basis identified with sigma-0.35 probes. The twist reaches the gait only through that basis, and the gait gain is off (R:3799-3802).

### A.3 The mechanisms as stated (R:3804-3887), with what later reads did to each

The heading says "six mechanisms". It lists T1, T1', T2, T3, T4, T5 and T6, with T1' a sub-case of T1.

- **T1. The quantile floor: no gradient on competence, by construction** (R:3806-3822).
  - With `<=`, a zero scorer stands at the zero share of the window, 0.91-0.99. The whole competent range is 0.01-0.09 of the scale, and the mission quantile is 0.99 for everyone.
  - Dominance then decides cells on margins alone, which gives 45-52% turnover with no change in competence.
  - Its extra prediction: a still machine stands near the top.
  - **Revised after N2** (R:4187-4198): "the rank transform gives full scale to sub-noise differences, and in air it also ties 97% of the population at the top". N5's floor must be a *competence* floor.
  - **N1 → neither** (R:3923-3937).
- **T1'. Tier-2 inverts selection at the top** (R:3824-3831). Promoted designs with zero legs fall to about 0.1, below every unpromoted elite, so the best designs become the least likely parents and are never migrated. **N4 → confirmed** (R:4184).
- **T2. Each cell keeps its luckiest draw** (R:3833-3843).
  - C2 measured draw variance at 2.5-4.6x design variance (one-draw reliability 0.18-0.30).
  - The best-of-k inflation is 0.56 / 0.85 / 1.03 sigma at k = 2 / 3 / 4, and the incumbent is never re-measured.
  - Prediction: retention falls as `improvements` rises.
  - **N3 → refuted.** The k-dependence is demoted to "not shown", and N11 keeps its justification from the literature (R:4199-4202).
- **T3. The controller cannot express the tasks** (R:3845-3853).
  - (a) Only rotorcraft fly (0.963-0.987 under a hand controller, 2026-09-26), and a rotor is reached only through a basis sized for angles.
  - (b) A commanded turn at 25 Hz through a 6-D twist with authority 0.5 needs yaw authority, which is unmeasured.
  - (c) A glider needs its gait stopped, and the gait gain is off.
  - **N6 → neither** (R:3993-4001): (a)'s premise "is true of the hand-built quadrotor and false of the archive". The unreachability is confirmed, but "it is not the binding gate today".
- **T4. Flapping flight is infeasible under the actuator limits, so land→air and the mission are unreachable by any flapper** (R:3855-3863).
  - Evidence (2026-09-26): level flight needs margin 1.0; the best feathering gait reaches 0.42; the motors are torque-limited 60% of the time; the best flapper scored 0.378. "No design class in the genome can both fly and be controlled."
  - Its fix is a decision for the user, not a build.
  - **N7 → refuted.** Revised: "not short of actuator power; it is short of a gait that makes thrust inside the band the spar and the servo allow" (R:4029-4031).
  - **Second revision after N13** (R:4065-4076): the thrust regime is reachable at about 2 Hz. "Either the gait searches never sampled the pitch-heave phase at `St` 0.25 and 2 Hz ... or the fluid model's propulsive force in its own validated regime is an order of magnitude low." N15 decides between the two and is **unrun**.
- **T5. Budget** (R:3865-3871).
  - About 600 evaluations per island per run at reliability 0.25 gives about 150 informative samples for a 625-cell archive. The literature spends 10^5-10^6 evaluations. Lineage depth at gen 121 is 5.
  - Prediction: the archive is indistinguishable from random sampling. Identification is 57% of the wall (M6).
  - **N8 → neither, in the unnamed direction**: random beats the archive, so selection "has subtracted" (R:4114-4121).
  - **N9 → refuted**: MJWarp does not pay at these world counts (R:4160-4167).
- **T6. PPO learns the shaping potential, not competence** (R:3873-3879).
  - The reward is `0.2 * (gamma * Phi(s') - Phi(s))` plus terminal competence over the tag's std, and competence is 0 for more than 95% of rollouts.
  - Phi in water is `-|depth error|`, the hold task, and M4 learns water. Phi in air is altitude plus upright, which is not the turn, and M4 learns nothing in air.
  - **N10 → not comparable**: the fields were missing (R:4185). L7 added them (A51:251-298), and arch51 logs `return_by_tag` (R:607-608). A re-read of N10 on arch51 is NOT STATED in the range read.
- **Rivals kept** (R:3881-3887):
  - instrument error: the fields are recorded correctly;
  - "nothing happened": every arm is one seed;
  - the user's prior (noise and the draw), which T2 keeps.
- **What T1-T6 overturn in the repo's account** (R:3889-3898):
  - "selection still has a gradient (20-23% of elites nonzero)" was reasoned, not measured;
  - the ladder, critic, auditor, M3 and E2 all read a scalar that does not rank competence;
  - the claim that "the shared PPO is the variation operator" is wrong.

### A.4 N1-N15: claim, frozen prediction, read, verdict

| item | mechanism / claim | frozen prediction (falsifier) | read, with numbers | verdict | refs |
|---|---|---|---|---|---|
| N1 | T1: a still twin scored through `_score_candidate` stands near the top | median twin `fit` ≥ 0.80 in every island, and non-dominated or dominating in ≥ 60% of cells (falsified: median < 0.5 in any specialist island, or < 30%) | **Twin median `fit`:** air 0.895, water 0.707, land 0.716, other islands 0.59-0.76, pooled 0.707 [IQR 0.667-0.756]. **Fronts:** non-dominated or dominating in 81.2% of cells, dominating outright in 30.6%; land lowest at 45%, air 96.8%; dominated 18.8%. **Run:** 229 elites × 2 arms, batched path, 2965 s. The elite arm reproduced within 0.005 in 229 / 215 / 195 of 229 (air / water / land). **Caveats:** every twin sits at stage 0; 88% land in a different cell from their elite | **neither** | R:3907-3937 |
| N2 | T1: the share of the window at exactly zero | ≥ 0.90 in air, water and land (falsified < 0.7) | **Zero share, arch49 / arch50:** air 0.965 / 0.973, aerial_diver 0.984 / 0.988, land_air 0.992 / 0.984, generalist 0.996 / 0.992; **water 0.586 / 0.464, land 0.605 / 0.601**; triphibian 0.301 / 0.256; amphibian 0.855 / 0.807. The mission's zero share is 0.973-1.000 everywhere. A design at zero in all three media gets `fit` 0.69-0.72 at stage 0 | **refuted** in water and land, **confirmed** in air | R:3939-3942, 4182 |
| N3 | T2's signature: inflation grows with `improvements` | Spearman(recorded − mean fresh, `improvements`) > 0.15 in water and in land, and retention ratio < 70% at ≥ 3 improvements (falsified: both within ±0.1) | **Spearman:** water 0.123 [−0.01, 0.25], land 0.094 [−0.05, 0.23], air 0.062. Only 9 of 229 elites have ≥ 3 improvements, so the ratios (0.685 water, 0.092 land) have CIs spanning 1. **Recorded vs fresh means:** water 0.0133 vs 0.0096, land 0.0159 vs 0.0120, air 0.0042 vs 0.0017 | **refuted** (not the reverse of the falsifier) | R:3944-3949, 4183 |
| N4 | T1': Tier-2 elites become the worst parents | bottom decile of parent weight in ≥ 80% of islands, and never a migrant (falsified: median above the island median) | Bottom decile in 24 of 24 islands (three runs); ratio 0.38-0.76; 86-90% of individual Tier-2 elites are in the bottom decile; at every migration at least 19 representatives out-rank the best Tier-2 fitness (0.12-0.25) | **confirmed** | R:3951-3956, 4184 |
| N5 | T1 / T1' fix (build): a floor that ranks | first arm: evaluated `fitness` median for gens 0-49 < 0.3; share with own competence ≥ 0.012 rises from 9% to ≥ 40% by gen 150; Spearman(`fitness`, own competence) ≥ 0.7. Any miss falsifies | **arch51:** median 0.0 (met); Spearman 0.97 against arch49's 0.47 (met); share 22.8% (145) against arch49's 22.4% (246) (**missed**). These reads use only the specialist islands, so 22.4% is a different population from the 9% (R:619-620). The generation at which the share was read is NOT STATED | built; **falsified** by arch51 | R:3958-3970; A51:24-41; R:610-618 |
| N6 | T3(a): rotor authority through the basis | (i) through the basis, air < 0.3, while (ii) direct, ≥ 0.8, on the same body and seed; the basis moves rotor speed < 5% of top per unit coefficient (falsified: (i) ≥ 0.6) | **Basis reach:** 0.0003 of top speed per unit coefficient (median, IQR 0.0002-0.0003, p90 0.0004); 191 of 191 below 0.05. **Flight:** the hand hover law flies the reference quadrotor at 0.816 direct, but flew **0 of 24** arch49 rotor bodies (median 0.000; one 0.446, one 0.011). **Lift:** 153 of 191 rotor elites cannot lift their weight (upward thrust over weight, median 0.15), and half the rotors on "eel" bodies point down. **Run:** 191 scanned, 24 rolled out, numpy path, 8 min | **neither** | R:3972-4001; A51:538-573 |
| N7 | T4: the flapping power bound (derivation) | ratio < 1 for every flapper plan at every in-band frequency, and the 0.42 margin reproduced within 30% (falsified: any ratio > 1 in band) | **Ratio, max / band top (7.80 Hz):** gannet 28.1 (at 2.3 Hz) / 12.7, teal 27.8 / 12.6, beetle 6.3 / 1.3, bat 7.7 / 4.6, feathering teal 17.6 / 7.5; never below 3.6 in 9 variants. **At 11.33 Hz:** the ratio is 3.0-7.0. **The band:** it closes at 7.80 Hz, so the only level gait ever found (11.33 Hz) is outside it. At 11.33 Hz the left stroke motor needs 22x its torque limit; the servo's corner is 2.88 Hz (80° lag); the series spring was tuned on 0.180 against MuJoCo's 0.372 kg m², so it cancels 48%. **Break-even efficiency:** 3.6% (gannet, teal) to 16% (beetle) | **refuted, both clauses** | R:4003-4036 |
| N8 | T5: random sampling against the archive | coverage and summed competence within 15% of arch49 at about 300 evaluations per island, gen 150 (falsified: arch49 leads by > 30%) | **Coverage, random / arch49:** air 55 / 36, water 55 / 41, land 55 / 32; pooled 1.35x. **Summed competence, random / arch49:** air 3.4x, water 2.3x, land 1.3x, amphibian 1.7x, triphibian 2.0x; pooled 1.94x. **Cells with competence > 0:** air 19 / 4, water 44 / 26, land 42 / 16. Bootstrap CIs exclude the archive in every medium. **Run:** 4,700 s. Caveats in B.6 | **neither: random wins** (recorded as a station-3 miss) | R:4092-4121; A51:575-606 |
| N9 | T5 / throughput: MJWarp for same-body work | rigid-body part ≥ 5x faster at 24 worlds (falsified < 2x) | 0.68x on the 12-dof elite in flight (0.537 s vs 0.367 s); full numbers in C.2 | **refuted** | R:4123-4167; A51:608-649 |
| N10 | T6: the share of shaping in the PPO return | shaping ≥ 90% in air (falsified < 60%) | The runs logged `reward_by_tag` (raw terminal competence; air is exactly 0 in 72-73% of PPO events, mean 0.0001-0.0003), not the shaping return | **not comparable** (fields added by L7) | R:4169-4174, 4185 |
| N11 | T2 fix (build): Extract-ME re-evaluation | first arm with N5 + N11: retention ≥ 0.5 in every medium (arch49: 0.09-0.33) at the same wall; Tier-1→Tier-2 Spearman ≥ 0.3 (arch49: 0.17 / 0.18). Falsified if retention < 0.35 in any medium | **arch51 retention:** land 0.02 (n=9), water 0.06 (n=2), air not measurable. **Spearman:** water 0.093, land 0.225; air undefined (Tier-1 air is 0 in all 468). **arch49 by the same method:** water 0.174, land 0.182, air 0.403 | built; **falsified** by arch51 | R:4325-4334; A51:300-460; R:615-618 |
| N12 | T5 / T6: a pretrained, frozen shared policy per arm | the read is to be written when M4's nine runs are in | none | **unrun** ("not scheduled") | R:4336-4340 |
| N13 | T4: the Strouhal gap | gannet and teal St < 0.15 at 7.80 Hz (falsified: either ≥ 0.25) | **St at the band top:** gannet 0.967, teal 0.959; beetle 0.653 and bat 0.770 at 12 Hz; feathering teal 0.881. St reaches 0.25 at 2.0 Hz. 99 of 157 arch49 flapping elites (63%) are ≥ 0.25 at their band top | **refuted, in the opposite direction** | R:4038-4047, 4062 |
| N14 | the Tier-0 spar check frequency | ≥ 10% of flapping elites commanded above their band (falsified < 3%) | **Count:** 14 of 224 (6.2%, Wilson CI 3.8-10.2). **What the check runs at:** `genome.flap_frequency`, not a fixed 5 Hz. The 14 had margins of −0.14 to −0.81 and passed the gate at −0.85. **What it misses:** the commanded CPG frequency (up to 20 Hz) is not stored, and 124 of 229 elites get no spar check at all (non-plausible flyers); forcing the check gives 9.4-12.3% over band | **neither** | R:4049-4054, 4063 |
| N15 | revised T4: the model's propulsive efficiency | η < 0.10 at St 0.30, phase 90°, α_max 20° (the experiment gives ~0.6-0.7). Falsified, which would put the gait search at fault, if η ≥ 0.30 | **no outcome recorded.** The only R lines that mention N15 are 213 and 4078-4090, and `experiments/` has no foil, heave or propulsion directory (directory names only) | **unrun** | R:4078-4090 |

### A.5 What was built from the theory, and what arch51 showed

**Built** on `failure-theory-1010` and merged (R:4205-4214; R:220-224). The full canary was green and 27 mutations were caught (R:223).
- **N5**, as lanes L1-L6:
  - L1: strict `<`; `ZERO_SCORE` 1e-9 stands at 0 (A51:43-77).
  - L2: `COMPETENCE_FLOOR` 0.012 per medium, read only for the island's own media. It is certified for water (elites 64/229 vs still 3/229) and land (51/229 vs 0/229), and **not certified for air** (elites 6/229 vs still 11/229, a leak) (A51:79-118).
  - L3: a floor design fills only an empty cell (F1), and an above-floor entry evicts floor members (F2) (A51:120-155).
  - L4: Tier-2 sets a flag (`meta["tier2_fitness"]`) instead of capping (A51:157-188).
  - L5/L6: `mission_weight` 0.0, in the scalar and the CLI (A51:190-249).
  - §D' 2: above the floor, a design is ranked only among window entries that also passed the floor (A51:711-714).
- **N10 telemetry**, as L7: the shaping/terminal split per tag (A51:251-298).
- **N11**, as L8-L11 (A51:300-460):
  - 4 of 16 evaluations per generation re-run representatives at fresh seeds;
  - a per-elite buffer of depth 8, whose score is the lower-median draw by blend;
  - no window feed, but the PPO buffer is fed.
  - As a consequence, offspring per generation fall to 12 (R:4255-4257).
- **The paired air score** (the user, 10-09/10-10) and **seeds scored with the shared policy** (R:222-223, 4215-4222). Before this, every archive held its seed elites' scores from a different controller than the film replays (R:4222-4223).
- **The auditor reports "not measurable"** for open-loop designs (0/0 ratio under paired media) (R:4238-4246).
- **Mutation `audit-base-is-the-record` was MISAPPLIED**, which is a silent hole (R:4247-4253).
- **Later:**
  - P1 (reevals at the stored basis) merged (R:4609-4611).
  - F2 (the restored-basis memory layout, bit-exact) merged 2026-10-12. Offline re-measurements of stored elites are not comparable across it (R:4631-4636).
- **The comparability boundary:** stored `fitness`, `objectives[0]` and everything built on them are not comparable with any run before arch51. Per-medium competences, `mission_fraction` and coverage are unchanged (R:4206-4214; A51:35-41).

**Measurements the GPU gate forced** (R:4223-4237):
- (a) Every seed plan scores exactly 0 in air, water and land at its draw on the batched path. So arch51 starts with nothing above the floor and nothing promotable.
- (b) The workers failed to build the GPU extension (`Py_NewRef`), which "turns 2 workers into 0". It was diagnosed later (I10:903-920; G).

**arch51** (R:599-638): 900 generations, 36.6 h (125 s/gen in the report, 146 s/gen of wall), seed 20261018, pinned at 443d0c6. arch51 = N5 + N11 + paired air, the user's decision (R:224).
- The mechanics checks passed: `at_floor`, 4 `reevaluate` events per generation, `return_by_tag` (R:607-608).
- The pre-registered table is reproduced in the N5 and N11 rows above (R:610-617).
- Retention n is 11 in total: audits ran only at gens 0-7 and 240-247, plus later rounds (R:620-622).
- `mission_best` 0.0 and `three_media` 0 at gen 899, as in arch48-50 (R:627).
- **The film:** the best-by-mission elite (triphibian, `fitness` 0.9905) reproduces air (0.000) and water (0.180) but not land (recorded 0.091, reproduced 0.029, "NOT THE SAME EXPERIMENT") (R:629-632).
- No arch51 elite reaches air 0.012, so the air L1 bar cannot be set (R:637-638).
- Open: "why the share of competent elites stays at 22%" (R:636).

### A.6 The device and loop-order theory (T7-T12), a separate block (R:4342-5212)

Here T1-T6 were about selection. T7-T12 are about where a rollout runs and what shares a launch.

- **T7. A device step's cost is nearly fixed, so draws are nearly free up to the hundreds** (R:4389-4402). Model: `T_step(W) = a + b W`. Prediction: `a` ≤ 0.5 ms and `b` ≤ 2 µs. Falsified if `a` > 1 ms or `b` > 5 µs.
  - Q1 and Q2 each confirmed their own prediction (C.2).
  - The combined fluid + rigid per-world cost on the Q1 Mojo kernel is 5.85 µs (I10:517, MODELLED), above T7's 5 µs. No explicit T7 verdict is written: NOT STATED.
  - D-A1 later measured the Warp fluid port's marginal cost at 0.0063 µs/panel in f32 (I10:833), well below Q1's 0.044.
- **T8. One launch for the whole generation wins by the number of bodies, "16x on the fixed term"** (R:4404-4409). MODELLED per step at W = 100 per body: A about 11 ms (~110 s/gen), B about 3.7 ms (~37 s/gen).
  - **Corrected after Q2:** the per-world cost is the fluid (5.5 µs for 126 panels) against 0.3 µs for the dynamics. So the 16x applies to a term that is 15% of the step, and B's whole advantage is 1.5x at K = 100: 157 s against 242 s, MODELLED (R:4495-4501).
  - Under option A the 16x is not available at all (R:4776-4778).
- **T9. Contacts are the physics that changes, and only land and the crossings read them** (R:4411-4416). Q4 and R1-R3 bear on it (B.4).
- **T10. On today's architecture, order buys little and removing redundancy buys some.** The bound is about 37 s of a ~215 s amortised generation, ~17%, MODELLED (R:4988-4997).
- **T11. On the device, the loop's order is the prerequisite** (R:4999-5012). Below W* = c0/c1 a launch is dispatch-bound.
  - W* was first written as ~200 worlds from N9, then corrected to **~80** once the fluid's 5.5 µs per world is counted (R:5001-5004, 4499-4500).
  - Today's loop offers 16-20 worlds per launch, which "would leave the device ~90% idle" (R:5004-5005).
  - **Superseded by I10 §0 F8** (I10:74-78). Each draw's 10 segment halves and transitions start from `reset` + `scatter`, so they are independent worlds: W = 10 × K per body, and W ≥ W* for K ≥ 8 "without any epoch". "The epoch is therefore not a prerequisite."
- **T12. What the epoch costs the search** (R:5014-5037):
  - eight generations' transitions arrive under one behaviour policy;
  - keep PPO's gradient steps per transition (130 per 25k → about 1,040 per 200k);
  - reject steady-state asynchrony, because evaluation cost is heritable (+1.6-1.8 s per rotor).
  - I10 F13 notes that the steps-per-transition rule is already today's behaviour: 25,473 / 2,048 × 10 = 124 (I10:110-112).

---

## B. Selection-signal facts

### B.1 The fitness scalar before and after N5
- **Before:** `0.7 * (w * isl_q + (1 - w) * cur_q) + 0.3 * mis_q`, with each term `mean(window <= x)` over 256 evaluations, the candidate included. Tier-2 capped at `min(fitness, f2)` (R:3777-3791).
- **After:**
  - strict `<`, and a raw score ≤ 1e-9 stands at 0 (A51:27, 48-58);
  - competence < 0.012 in every one of the island's media gives `isl_q = cur_q = mis_q = 0` and the flag `at_floor` (A51:28, 196-202);
  - `mission_weight` 0.0, because `mission_fraction` is zero for 97-100% of every window (A51:29, 194);
  - F1 and F2 (A51:30);
  - the Tier-2 flag (A51:31);
  - ranking among floor-passing window entries only (A51:711-714).
- **After N11:** an elite's record is its lower-median draw over up to 8 draws (A51:305).
- **The effect** (MEASURED, arch51): the evaluated `fitness` median for gens 0-49 went from 0.68-0.78 to 0.0, and Spearman with own competence from 0.47 to 0.97 (R:612-613).

### B.2 Share of elites below the competence floor
- 82% / 91% / 91% (arch48 / arch49 / arch50 at gen 120, all eight islands), with a median `fitness` of 0.785 / 0.810 / 0.822 among them (R:3759-3760).
- On the specialist islands' own medium:
  - arch51: 22.8% at or above the floor (145 elites), so 77.2% below (my arithmetic);
  - arch49 by the same method: 22.4% (246 elites) (R:614, 619-620).
- The 0.012 floor is uncertified in air (A51:84-88). No arch51 elite reaches air 0.012 (R:637-638).

### B.3 Draw variance against design variance (all MEASURED)
- C2 (2026-10-08): draw variance is 2.5-4.6x design variance; one-draw reliability 0.18-0.30 (R:3834-3835, 3651).
- Restated per medium (C2, arch48 and arch49): R = 4.6-5.8 in air, 3.0-3.4 in water, 1.8-2.5 on land. Reliability is `K/(K+R)` (MODELLED): 0.15-0.35 at K=1, 0.63-0.85 at K=10, 0.84-0.94 at K=30, 0.95-0.98 at K=100 (R:4377-4380).
- **Q5** (arch49, 229 elites × 6 fresh draws):
  - It is "refuted on rank, held on value". The rank Spearman sits below the formula by 0.15-0.24 in air and 0.12-0.18 on land; the Pearson matches within 0.02 in water and land.
  - 182 of 229 elites have six zero draws in air and are excluded.
  - R on the included elites: air 7.3, water 4.0, land 2.0, with air's CI 2.2-38.
  - K for reliability 0.9: air 66, water 36, land 18. For 0.8: 29 / 16 / 8. For 0.95: 139 / 75 / 38 (R:4659-4671).

### B.4 Land is chaotic, which bounds what any single land draw can say
- **Q4** (58 land-competent elites; contact stiffness 0.074x):
  - 40 of 58 moved by more than 50%.
  - A null control (`solref[0]` ×1.01) moved 20 of 58 (34.5%) by more than 50%.
  - Spearman(hard, soft) is 0.71 under the null, 0.23 at x2 and 0.18 at x4.
  - The population mean moved mildly (0.0485 → 0.0298 at x4; 0.0465 under the null) (R:4531-4553).
- **R2:** a freshly identified basis restored 25 of 25 non-reproducers. A 1e-9 change of one initial coordinate moves land by more than 0.005 for 20 of 25 (median 0.030, against a median record gap of 0.029) (R:4595-4611).
- **R3:**
  - The cause was the memory layout of `modes`, which rounds `command_params` differently by 1e-18 to 1e-16. Changing the layout flips the score in both directions for 8 of 8 failing elites. Elite 34: land 0.0000 against 0.0992.
  - "The archives are right ... every offline restore was wrong" (R:4613-4636).
  - The theory change: land "is one sample of a chaotic map ... its information about the design is what survives averaging over draws". A device implementation "cannot agree per draw on land" (R:4638-4652).

### B.5 Tier-1 to Tier-2 agreement, and retention at a fresh draw
- **Tier-1→Tier-2 Spearman:**
  - arch49: 0.17 / 0.18 (R:4333).
  - arch51: water 0.093, land 0.225, air undefined. arch49 by the same method: water 0.174, land 0.182, air 0.403 (R:616).
- **Tier-2 legs completed:** 9/192, 1/192, 0/78 (R:3764), i.e. 10 of 462 over three runs (R:3830-3831).
- **Auditor held-out retention:**
  - arch49: 0.09-0.33 (R:4331).
  - arch51: land 0.02 (n=9), water 0.06 (n=2), air not measurable. arch49 by the same method: land 0.06, air 0.0, water 0.09 (R:615).
- **N3's recorded-vs-fresh means** (arch49): water 0.0133 vs 0.0096, land 0.0159 vs 0.0120, air 0.0042 vs 0.0017 (R:4183).
- **Literature** (R:4278-4294):
  - At one sample per offspring, fewer than 20% of elites were in the right cell on re-evaluation (Flageat & Cully 2020).
  - With undertrained controllers, the top 5% and top 1% rank about 0 against the final ranking (Mertan & Cheney).
  - Learning speed against potential: Spearman 0.05 (Song et al.).

### B.6 Random bodies against the archive (N8)
- **The exact design** (A51:575-606; R:4101-4111):
  - **Bodies:** 300 that pass Tier-0, drawn from the run's own seeding distribution (`seed_population` + `random_genome` in 20:8, `default_rng(20261010)`).
  - **Scoring:** `loop.evaluate_candidates(..., identify=True)`, with arch49's scoring network, in batches of 16, each body at its own seed.
  - **Binning:** arch49's final descriptors and axes.
  - **Comparison:** each island's arch49 gen-150 snapshot, which is about 300 evaluations per island (the "gen-37" in the first draft was an arithmetic slip, R:4096-4097; A51:577-582).
  - **Score per cell:** the max of the island's own competence.
- **Result:** in the N8 row of A.4.
- **Caveats** (R:4111-4114):
  - one draw per body on both sides;
  - the archive's scores are arch49's own (a shared draw per generation, pre-M2), while the random side's come from the current scorer;
  - the competence definitions are shared since PR #32.
- **The control was never re-run.** It was meant as arch51's control (R:4120-4121). No later comparison is recorded in R (grep, see §0 item 1).

### B.7 Evaluations and samples per run
- About 600 evaluations per island per run, against 10^5-10^6 in the MAP-Elites literature (R:3865-3868).
- arch49: 4,820 evaluated by gen 299 (A51:506). Evaluate events over gens 0-299: water 587, land 604, air 611 (A51:580).
- arch60, A5-COMPUTED: 1,828 evaluations at gen 112 on one island (air).
- The PPO buffer gets 25,473 transitions, 144 trajectories per update and 130 gradient steps (R:4945, 4949). That is about 1.5k decisions per body per generation, against DERL's 5M steps per body and MetaMorph's 1e8 over 100 bodies (R:4297-4299, 4709).

---

## C. Throughput facts

### C.1 Where a generation's wall time goes (MEASURED, arch50, 122 generations, medians; R:4941-4960)

| phase | median s | share | what it is |
|---|---|---|---|
| `evaluate.main` | 105.1 | 63% of 168 s | 16 bodies (20 with reevals), 8 shards of 2 over 4 workers; 460,800 identification steps, 160,000 segment steps and 96,000 transition steps per generation; it fills the PPO buffer |
| `evaluate.rescore` | 43.6 | 26% | the same bodies at the same seeds, with the policy at its mean |
| `tier2` | 166.7 per round, 33 amortised | 20% of an amortised 202 s | refinement 105.0 s (six serial (1+1)-ES steps), Tier-1.5 15.2 s, Tier-2 mission 33.2 s |
| `audit` | 63.5 per round, 2 amortised | 1% | held-out re-runs |
| `ppo` | 1.4 | 0.8% | 130 gradient steps |
| host logic | 0.02-0.05 each | < 0.1% | |

- **Rollouts are 98.3% of the summed wall; decisions are 1.7%** (R:4952-4953).
- Identification is 64% of the main phase's steps and was measured at 57% of its wall (M6) (R:4955-4956).
- Per worker, 2 bodies step in lockstep on the CPU and the GPU sees a batch of 2. 4 shards of 4 beat 1 shard of 16 by 2.3x (R:4956-4960).
- The batched path is **dispatch-bound on the host, not GPU-bound** (`experiments/perf/NOTES.md`, 2026-09-27). Its split: 62% identification, 64% (cumulative) the fluid's host numpy, 18% the GPU round trip (338 µs per call, later 89 µs), 4% `mj_step`, with the GPU at 5%. That is ~0.3 ms per design-step, ~0.8 ms with identification (R:4363-4369).
- **Identification is 96 worlds per body**, not "24 probes": 24 probes × 2 signs × {air, water} × 300 steps = 28,800 steps per body, run before the segments (I10:69-72, 500).

### C.2 Device and MJWarp throughput against the batched CPU + Mojo path

| measurement | numbers | label | refs |
|---|---|---|---|
| **N9**, MJWarp rigid-body only, against 24 sequential CPU rollouts in mujoco 3.11 (one thread), 24 worlds × 2000 steps, graph captured, no fluid | 12-dof elite in flight **0.68x** (0.537 s vs 0.367 s), with contacts 0.85x; 18-dof rotor elite 1.18x / 1.09x. Without graph capture 0.04-0.08x (~5 ms per step). Scaling on the 12-dof body: 1 world 0.03x, 48 worlds 1.27x, 96 worlds 2.37x; break-even near 39 worlds. Per step ~240 µs fixed + ~1.2 µs per world, against 7.4 µs per world on the CPU, so 5x needs ~730 worlds. Identification shape (48 × 300, no fluid) 1.26x. Host copies add 113 µs per step (+42%) at 24 worlds. qpos within 1.1e-5 (f32) over 300 steps. The 1 mm geom margin is rejected under MULTICCD/NATIVECCD; `njmax` 64 overflowed on landing (512 used) | MEASURED | R:4139-4167 |
| **Q1**, the Mojo fluid kernel's marginal cost | t = 117.4 µs + 0.0440 µs per panel (R² 0.9994). 50,022 panels (397 bodies) cost 2.31 ms per step, against the ≤ 25 ms predicted. CONFIRMED | MEASURED | R:4440-4452 |
| **Q2**, MJWarp fused rigid + control (33-64-64-6 MLP + CPG in the graph), **no contacts, no gravity** | median body a = 0.361 ms, b = 0.306 µs per world; rotor body a = 0.379 ms, b = 0.487 µs. W = 24 / 100 / 730 cost 356 / 405 / 583 µs per step (median body) and 389 / 430 / 735 (rotor). The callback costs 0.09-0.12 ms fixed. `wp.capture_while` was refused. CONFIRMED | MEASURED | R:4463-4477 |
| W* = c0/c1 = (0.36 + 0.12 ms) / 5.8 µs | about **80 worlds** (not 200) | MODELLED from measured fits | R:4499-4500, 5001-5004 |
| the K decision table (16 bodies, 126 panels, 16,000 steps per generation, no contacts) | **K=1:** A-fused 94 s (0.56x arch50's 168 s), A-separate 124 s, B 9 s. **K=10:** 107 (0.64x) / 137 / 23. **K=30:** 137 (0.82x) / 167 / 53. **K=50:** 167 (1.00x) / 197 (1.17x) / 83. **K=100:** 242 (1.44x) / 272 (1.62x) / 157 (0.94x) | MODELLED | R:4479-4512 |
| I10 §3, parallel layout (2,300 steps, W = 10(K_m+K_s)) | (10,0) 39 s (0.23x); (10,10) 61 s (0.36x); **(50,10) 147 s (0.88x)**; (50,50) 233 s (1.39x); (100,10) 254 s (1.51x). Sequential layout: 136 / 153 / 213 / 273 / 288 s. These are **lower bounds**: the host-side `finish()` terms (F4), contacts and rotor bodies are excluded | MODELLED | I10:516-526, 45-57 |
| Q6, Mojo→Warp zero copy | at 1e5 floats the wrap takes 1.8 µs against an H2D copy of 1213.6 µs. The two runtimes use different streams. The 1e6 size never finished (300 s timeout) | MEASURED | R:4680-4701 |
| G/Q8, the MAX graph loop | G: 50.9 µs per iteration (4096 × 6×6 bmm). **Q8** at the real shape (1,600 × 33 @ 33×64): 974.5 µs per iteration matmul-only, 1132.8 with a functional custom op, 980.4 in place. B on MAX graph is closed | MEASURED | R:4861-4869, 4904-4919 |
| D-A1, Warp fluid + scatter graph | marginal 0.0063 µs per panel (f32), launch 11.2 µs at W=24. Scatter/fluid: medusa 4.45 / 5.27 / 1.83 / 1.26 and rotor quad 1.18 / 1.23 / 1.47 / 1.61, at W 24 / 100 / 730 / 1600. **The falsifier "scatter doubles the step" fires at every W.** Read as a launch-count effect; ~0.4-0.6 ms per step at W=60 per body | MEASURED | I10:833 |
| D-A3, graph capture | an unrolled 2000-step graph takes 74.9 s to capture (median body) and 73.3 s (rotor), against 24.0 / 45.4 ms for one step. 2,300 launches of the 1-step graph take 682 / 646 ms; the unrolled graph runs in 3,687 / 2,787 ms, so it is slower too. Per-step graphs stay the default | MEASURED | I10:840 |
| D-A4, memory | ~0.05 MiB per world (median body), ~0.11 (rotor); 16 bodies × 600 worlds fit | MEASURED | I10:839 |
| dry6 (arch60 config, 2 generations) | gen 0 478 s, gen 1 201 s | MEASURED | R:11 |
| **arch60 against arch56** (same config and seed; arch60 adds `--evaluator device --k-mean 50 --k-sampled 10`, N60:1-6); gens 8-112, the later record per generation, sum of the top-level `cost.seconds` keys | **Per-gen wall:** arch60 median 246.9 s (mean 248.5) against arch56's 99.3 s (mean 105.7), **2.49x**; over the last 20 gens, 276.6 s against 104.5 s. **arch60 split:** `evaluate.device` 115.2 s, `evaluate.main` 81.0 (arch56: 62.0), `evaluate.rescore` 39.8 (arch56: 31.7). **Batched step counters are identical in both runs:** main.identify 345,600, main.segments 192,000, main.transitions 96,000, rescore.segments 192,000, rescore.transitions 96,000 | A5-COMPUTED | `runs/arch60/generations.jsonl`, `runs/arch56/generations.jsonl` |

**Engine caveat on this comparison.** arch56 ran 2026-10-22..23 (`runs/arch56_notes.md:1, 8`), before the 2026-10-26 upgrade to mujoco 3.15 (R:19-22). arch60 launched 2026-10-29. So the two runs' batched passes most likely ran on different engines, 3.11 against 3.15. That is INFERRED from dates: neither checkpoint's `provenance` records a mujoco version, only git, numpy, torch and kernel hashes. Configuration and seed match; the engine probably does not.

The arch60 counters (192,000 = 16 × 6 halves × 2000) show the batched path still stepped all six halves in both the main and the rescore pass. That points to the device's K_m = 50 / K_s = 10 air and water scoring being **added** on top of the K=1 batched evaluation, not replacing it (INFERRED from the counters; not checked in code). Two more numbers fit the same reading:
- arch60's `evaluate.main` is *higher* than arch56's (81.0 s against 62.0 s), not lower, as it would be if the batched path ran only land and the crossings;
- the summed `rollouts` are equal (11,592 against 11,585, gens 8-112).
- Consistent with that, the device PPO feed was written on 2026-10-31 ("device half not yet run", I10:712). arch60's tree was pinned at `25c7a70` on 2026-10-29 (N60:35). So arch60's K_s worlds fed no PPO (INFERRED from the dates).
- The same §3 formula at arch60's actual device width (4 halves × 60 = 240 worlds, I10:501; R:11 "240 host envs per body") gives about 69 s per generation for 16 bodies (my arithmetic, MODELLED). The measured `evaluate.device` median is 115.2 s.
- D-A7 ("generation wall from `generations.jsonl` vs arch56's", N60:18) has not been read in any doc. The numbers above are its inputs, not its verdict.

### C.3 K draws per body; what T8, T11 and P1-P6 are
- **K_m** = draws at the policy mean; their pooled mean is the score. **K_s** = sampled draws that feed PPO. The first device arm uses K_m = 50, K_s = 10. The batched equivalent today is K_m = 1, K_s = 1 (main + rescore) (I10:503-504, 862-868; R:13).
- K_m becomes 64 if D-A7 holds within 1.5x of its prediction and the wall stays under 1.2x arch50's (I10:866-868; not read).
- **T8 / T11**: see A.6.
- **P1-P6** (R:5041-5088):
  - **P1:** reevals at the stored basis. Predicted `identify` steps 576,000 → 460,800 at 20 built and `evaluate.main` −8-14%. It is merged (R:4609). D-P1 is NOT STATED. arch56 and arch60 show main.identify 345,600 = 12 × 28,800 (A5-COMPUTED), which is consistent with reevals not re-identifying.
  - **P2:** a (1+6)-ES at promotion, predicting `refine_wall` 105.0 → ≤ 30 s and −7% amortised. I10 F2 says its acceptance read cannot be computed from promote events (I10:35-37). Status NOT STATED in the range read.
  - **P3:** main + rescore in one trip, predicting −3 to −8%. **Withdrawn**: A6 retires the two trips (I10:127, 899-902).
  - **P4:** the all-island epoch on the CPU path, a predicted null within ±10%. Not built, by design (R:5065-5069, 5206-5207).
  - **P5:** the epoch on the device; build it if the predicted wall at K=30 is ≤ 8x today's (≤ 1,600 s). The table "is filled in from Q2"; no filled table is found in R.
  - **P6:** the epoch's search efficiency at equal evaluations, within 10% of round-robin, read in the first on-device arm. Unread.

### C.4 The expected speedup from the all-island epoch: measured or modelled?
- **Modelled only.**
  - T8: B ~37 s against A ~110 s per generation (R:4404-4409). After Q2, B's whole advantage is 1.5x (R:4495-4500). B is closed (Q8, R:4919).
  - Under A, the epoch only adds stream overlap. D-A5 predicts that 128 bodies save ≤ 10% per body over 16, and the epoch flag is built only if the saving exceeds 10% (I10:771-772, 841).
- **D-A5 has not been measured**: it was "not started" on 10-26 (R:13), and the GPU was held for arch60 (N60:41).
- MAP-Elites' efficiency per evaluation is batch-size-insensitive (Lim et al., TMLR 2023, R:5092-5103). So an epoch buys wall time, not search per evaluation.

---

## D. Device-path state (newest first wins)

### D.0 What the device evaluation path is (IMPL_1010 option A, I10:479-526, 617-628, 686, 708, 811)
- **Package:** `dytiscidae/envs/devroll/`, outside the hexagon, with modules `fluid_wp`, `scatter_wp`, `model_wp`, `control_wp` and `rollout_wp`. The entry point is `evaluate_tier1_device(...)`, which returns one result per body with `draws` (I10:483-492).
- **Models:** one MuJoCo Warp `Model` per body with W worlds, because heterogeneous bodies do not batch (I10:508-510). Per-world `body_mass`, `body_inertia`, `dof_damping` and `body_subtreemass` are batched (I10:617).
- **Per step,** inside a captured graph: the control kernel (decision steps only), then the fluid (a Warp port of `physics/fluid.py`, a third implementation), then the scatter (`finish()` terms, jets, rotors, power, `servo_command`), then `mjw.step`, then the record kernel, then the divergence kernel. No `callback.control` is used (F3; I10:621-628, 39-43).
- **Precision:** float32 state (F5, I10:59-63). Host-driven per-step graphs are the default; unrolled graphs are rejected by D-A3 (I10:512-514, 840).
- **What stays on the host:** each world's start state (CPU `reset` + `scatter`, I10:505), identification (I10:711), and scoring through `batchroll.finish_segment` on the downloaded records (I10:686, 708). PPO's logp and value are recomputed on the host (I10:663-666).
- **What stays on CPU MuJoCo:** Tier-2, the audits and the reference film (I10:811). Land and the crossings stay on the batched path at K=1 until A4b (I10:806).
- **Layout:** draws are worlds. Each draw contributes its segment halves (4 until A4b, 10 after), so W = 4 or 10 × (K_m + K_s) per body (I10:496-505).

### D.1 Lane status

| lane | state | refs |
|---|---|---|
| M1 mutate-check | in the lane table. F9 found 7 of 195 mutations MISAPPLIED and noted that `mutate.py` already exits 1 on them, but nothing runs it. Its build status is NOT STATED in the range read | I10:81-84, 122 |
| D1 `Py_NewRef` | diagnosed: `PYTHONEXECUTABLE` is C-setenv'd to a pyenv 3.9 shim and spawn workers inherit it; the fix is `os.environ["PYTHONEXECUTABLE"] = sys.executable` (lane F1). Still open: why arch50's log has no `Py_NewRef` line | I10:903-920 |
| P1 | merged | R:4609 |
| P2 | NOT STATED | — |
| R1 → R2 → R3 | done; restored-basis layout fix F2 merged 2026-10-12 | R:4555-4636 |
| P3 | withdrawn | I10:127, 899-902 |
| A0, A1a, A1b, A2, A3 | built (INFERRED: A4a "as built" runs on them; `test_devroll.py` is in CLAUDE.md's suite table; D-A1 timed `fluid_and_scatter`) | I10:618, 708-714, 833 |
| A2 revisions (2026-10-26) | `body_subtreemass` batched and recomputed after every mass write (without it, the float32 `qM` went NaN: ray in water; medusa, bat, teal and gannet in air). The margin is kept, with MULTICCD and NATIVECCD off on the device copy: a contact boundary not yet measured on a settled body. The still machine's 8 s air segment falls onto the ground (air worlds touching: beetle 8/8, eel 8/8, teal 6/8; water 0/8). A contact-pool overflow fails every live world of a launch | I10:617-619 |
| A4a | built 2026-10-26 ("not committed" at the time of the note). **Agreement:** replay against `rollout_batch` within 3.4e-16 (f64) and 1.4e-5 (f32). End to end on MJWarp: step 1 within 3.0e-7, 25 steps within 2.7e-4, 200 steps up to 0.76 (quad in water, not gated). **Still machine** (8 bodies × 4 seeds): device air 0 of 32, water ≤ 0.0087. **Gaps:** device identification is not built (bases come from the host's `identify_batch`); bit-equality needs `repeatable()`, whose cost is not measured | I10:708-714 |
| A5 | **first red, then water passes.** **Red (10-26):** the device failed 54 of 128 segments against the batched path's 8; water Spearman 0.76 against a null of 0.99 (R:17). **After the subtree-mass fix `9670d85`:** water Spearman 0.97 (null 0.99); within 0.005 on 62 of 64 draws (null 59); means device 0.0065, batched 0.0072, null 0.0067. **Air is unmeasurable** on a flightless population (3 of 64 draws nonzero), so it is recorded as unmeasured (R:15) | R:15-17 |
| A6 | wired 10-26, with `evaluator` defaulting to `batched` so no run changes. **Settings:** K_m 50, K_s 10; placement on the pooled mean of all draws with re-evaluated incumbents, no lower bound. **Checks:** the `device-kdraw-*` mutations 6/6 caught; teal and quad pooled-mean parity pass; full `test_search.py` 688 checks, 0 failures. **Not started as of 10-26:** first device arm, PPO feed, device film, D-A5/D-A6/D-A7. **PPO feed** written 10-31, device half not run | R:13; I10:712, 740-791; commit f1281bd in `git log` |
| A4b (land and crossings on the device) | **blocked.** The user said yes (after A6) (I10:861). The host half of the settled-contact measurement was done 10-29 (CPU): settled z 0.8821 / 0.8819 m (margin 0.001, CCD on / off) and 0.8805 / 0.8803 (margin 0), with normal force 49.91 N in all four, so the margin moves height 1.6 mm and the CCD flags 0.2 mm. Settling convergence was not checked. The device half needs the GPU and is open. Until A4b, `evaluator="device"` scores land and the crossings on the batched path at K=1 | R:9; I10:793-806 |
| A7 boundary docs | the boundary sentence is drafted: nothing on the device path is comparable with any earlier run. Merge state NOT STATED | I10:810-816 |
| **eel water gap** | open. **Numbers on 4 draws:** device pooled mean 0.019 against batched 0.039. Per draw, device 0.021 / 0.034 / 0.021 / 0 against batched 0.092 / 0.052 / 0.013 / 0. The single path gives 0.0000 / 0.0000 / 0.0021 / 0.0000 (mean 0.0005). `setConst` is eliminated, and the host paths disagree with each other too. **Next probes:** the single path at matched placement seeds, and draw 0 step by step. The eel stays `[info]`, not gated (`A6_INFO_BODIES`) | R:3, 7, 13 |
| deferred measurements | **measured:** D-M0 (water and land drop 14 and 16 reproductions; the user kept mujoco 3.15, R:19-22), D-A1, D-A3, D-A4. **Unmeasured in any doc:** D-A2, D-A5, D-A6, D-A7, D-A8 (D-A6/7/8 are arch60's frozen reads, N60:16-19). D-P1 and D-P2: NOT STATED | I10:829-844; R:13, 19-22 |

### D.2 arch60: what it is, how far it got, and its memory growth
- **What it is:** arch56's configuration exactly (N60:1-6):
  - one island (air), sigma 0.02, `air_turn_reward` 1.0, shared policy, `log_std` −0.5;
  - floor tiebreak on, reeval 4 at depth 8, gait gain, mission weight 0;
  - seed 20261020, 150 generations (a mechanism arm);
  - the one change: `--evaluator device --k-mean 50 --k-sampled 10`.
- **Frozen reads** (N60:12-22): S1 (measurable on ≥ 4 of 8 draws, against arch56's 23.4%); R1 ≥ 5; R2; R3 ≥ 5 above-floor elites retaining ≥ 0.5 (arch56: 3 of 8); D-A6; D-A7; D-A8. **Falsifier:** if R3 does not improve on 3 of 8 at gen 150, "pooled K-draw placement is refuted; placement noise is not the wall".
- **Launch:** 2026-10-29 from the pinned tree `25c7a70`, which chunks body construction, under `MemoryMax=11G` with `--memory-ceiling-mb 9000` (N60:35-38; R:11).
  - Before the chunking fix, `body_worlds` added 526 MB per body (240 host envs), so about 0.8 GB per body stayed resident. `keep_records` was a minor share (4.5 GB for 16 bodies without it, 7.2 GB with) (R:11).
- **Memory growth in the host process:**
  - resident set 9,413 MB at gen 20 and 10,311 MB at gen 40, growing 45-65 MB per generation; cause not located (R:11; N60:47-59);
  - "module per body" ruled out: 0-10 `load on device` lines per generation, and GPU memory flat at ~1.7 GB (N60:48);
  - a static lead, untested: Warp modules recompiling per shape mid-run (N60:69).
- **How far it got** (A5-COMPUTED from `runs/arch60/generations.jsonl` and the log): **max gen 112 of 150**, with 1,828 evaluations, 149 elites and coverage 23.84% at gen 112. Gens 41-55 and 101-112 appear twice; the later line counts (N60:68, 75).
  - Unit history: cycles stopped by the ceiling at gens 20, 40 and 80 (log line 1759: "stopping at generation 80: resident set 9623 MB"). A global OOM hit on 10-31 (N60:71-72). arch60loop3 was OOM-killed mid-generation at about gen 112 (python anon-rss 8.4 GB), and arch60loop4 relaunched from the **gen-100 checkpoint** (N60:74-75).
  - **arch60loop4's end** (read by me from the user journal): at 2026-11-01 21:56:05, `org.gnome.Shell@x11.service: Failed with result 'oom-kill'`. At 21:56:30, "Stopped arch60loop4.service", having consumed 2 h 13 min of CPU time. The log's last generation line is gen112. `journalctl -k` showed no `Killed process` line to me for 21:40-22:00, so the kernel's victim is not identified.
  - Both units are `inactive` as of 2026-11-10: not live, idle about 9 days. Resuming would restart at gen 101 (INFERRED: the last checkpoint is gen 100, with `checkpoint_every` 20, N60:68, 75).
  - **The run did not die from its own memory this time** (INFERRED from the journal sequence): GNOME Shell's oom-kill at 21:56:05, then user services stopping from 21:56:17, then `Stopping arch60loop4.service` at 21:56:30. The user systemd instance stopped the unit during the session teardown; the unit was not the OOM victim. `loginctl show-user hundo -p Linger` reads `Linger=no`, so user units do not outlive the session on this machine. That is a failure mode `systemd-run --user` alone does not avoid.
  - **Provenance defect** (INFERRED): arch60's `checkpoint.json` records `provenance.git` = `a7770c2` (main HEAD, committed 2026-10-31 19:42), while the run's code came from the pinned tree at `25c7a70` (N60:35, 51). The recorded commit is therefore the main tree's HEAD, not the code that ran.
- **Data-quality counts** (A5-COMPUTED, counts only, not interpreted):
  - `runs/arch60.log` (5,681 lines, 140 generation lines) has 424 "nefc overflow - please increase njmax beyond 512" and 1,427 "solver iterations limit reached" warnings from MJWarp.
  - `runs/arch60/events.jsonl` has 0 occurrences of `"contact overflow"` and 1,219 of "contact in a contact-free segment".
  - Over gens 8-112, `diverged_rollouts` / `rollouts` sum to 22 / 11,592 (arch56: 3 / 11,585). The two runs most likely ran on different mujoco versions (C.2, engine caveat).
- **arch60's `domain_best` at gen 112:** air 0.043, water 0.063, land 0.126. These are **not comparable** with any batched run (I10:811). Air and water there are pooled K-draw means; land is batched at K=1.

### D.3 What blocks the device path as the default evaluator (each from the text)
1. **Land and the crossings never run on the device.** A4b is not built, and its device half of the settled-contact measurement is open (R:9; I10:806).
2. **The eel water gap is open**, with three paths giving three numbers (R:3, 7).
3. **Air agreement has never been measured**, because the A5 population is flightless (R:15). The device-path bases also come from host identification (I10:711).
4. **Host memory grows 45-65 MB per generation, cause unknown.** It forces ceiling-stop/resume cycles, and twice the machine-wide OOM took the arm down (R:11; N60:55-75). The chunked path is not verified by `test_devroll` (N60:45). With `Linger=no`, a desktop-session OOM also stops the user unit (D.2, INFERRED).
5. **arch60 stopped at 112 of 150, so its frozen reads are unread**: S1, R1-R3, D-A6 (winner's curse), D-A7 (wall) and D-A8 (still machine) (N60:12-21).
6. **The device film and the PPO feed have never run on the device.** The film was not started as of 10-26 (R:13); the feed was written 10-31 and not run (I10:712).
7. **The measured wall is 2.49x the batched baseline's** (A5-COMPUTED, C.2). The plan expected K≈50 to "hold today's wall" (R:4502), and §3 put (50,10) at 0.88x (I10:524).
8. **A7's boundary means nothing is comparable across the switch** (I10:811).

---

## E. Mission, crossings and multi-medium capability in this range

- The mission sat at 0 for "twenty-odd arms since arch29". The best air score of any elite is 0.42-0.55, and water and land sit below 0.3 since the antipodal pair (R:3739-3741).
- `mission_fraction` is zero for 98.8% of evaluations (R:3795-3796, 3962-3963). The mission window's zero share is 0.973-1.000 on every island (R:4182).
- T4's frame: "the mission needs land→air", and "mission = 0 for twenty runs is the correct answer to the question as posed". Both were later revised by N7 and N13 (R:3859-3861, 4029-4031, 4065-4076).
- Tier-2 (the mission legs) completed a leg in 10 of 462 promotions (R:3830-3831, 3764).
- **Q4: "Crossings carried nothing: `water_to_land` and `land_to_air` crossed 0/58 in every arm"** (R:4545-4546).
- Every seed plan scores exactly 0 in air, water and land at its draw under the paired scoring (R:4224-4229). Under three paired media an open-loop machine is credited in no medium and has mission 0 (R:4238-4241).
- **Flight capability:**
  - rotorcraft fly at 0.963-0.987 under a hand controller (R:3846-3848);
  - the reference quadrotor flies at 0.816 under the hover law, but 0 of 24 archive rotor bodies do;
  - 153 of 191 rotor elites cannot lift their weight (R:3985-3993);
  - no flapper flies in band (N7 band facts, R:4024-4031);
  - only 6 of 229 arch49 elites reach air 0.012 (R:4623-4625).
- **The still machine leaks in air:**
  - no_model_gate's air bar is a leak: elites 6/229 against still machines 11/229 (A51:87);
  - the rigid still gannet clears air on 2 of 10 seeds (R:3530-3532);
  - on the device, the still machine scores air 0 of 32 and water ≤ 0.0087 (I10:714).
- Multi-medium window shares (N2): triphibian zero share 0.301 / 0.256, amphibian 0.855 / 0.807 (R:4182). N8 random-versus-archive summed competence: triphibian 2.0x, amphibian 1.7x (R:4108-4109).
- **arch51:** `mission_best` 0.0 and `three_media` 0 at gen 899. Its best-by-mission elite reproduces air 0.000 and water 0.180, and its land does not reproduce (0.091 against 0.029) (R:627-632).
- **Not in any doc:** a purpose-built crossing reference scored through `CrossingTracker`. NOT STATED in the range read.

---

## F. Wing flexibility, and the butterfly-paper items

**Wing flexibility (2026-10-12, R:3456-3568).** Built off by default at the user's request (R:3458).
- **What it is:** one passive pitch hinge per lifting surface (`Part.pitch_compliance`). It sits about the span axis through the leading edge, with a torsion spring and damper, ±1 rad stops and no motor (R:3465-3475).
  - `c` is the deflection at 1 g, in units of 0.1 rad, over the range [0.05, 4].
  - The policy neither sees nor commands it, and the fluid model is untouched.
  - `--wing-compliance` adds an optional operator.
  - Designs without the gene compile byte-identically (R:3460-3484).
- **The two-path agreement was not run**, because that container had no Mojo kernel (R:3480-3481).
- **The first sizing failed:** compliance 0.25 limit-cycled the gannet's wing at ±0.3 rad and dropped lift from 55.9 N to 5.8 N. A 1 g reference plus damping sized on spring + aerodynamic stiffness fixed it (R:3486-3492).
- **MEASURED** on a fixed rig, open loop, numpy path (R:3494-3520):
  - The beetle at 7 Hz gains 41% of flapping thrust at c = 0.5 (ΔFx +1.37 → +1.93 N). At its own 2.2 Hz it gains nothing (+1.28 → +1.18).
  - The bat's membrane gains at c ≤ 1 and loses at c ≥ 2; at c = 4 the hinge is on its stop 67% of the time.
  - The teal is unchanged (about −2.5 N).
  - Every flexible wing loses lift at a fixed attitude: gannet held lift 55.9 / 49.1 / 39.5 / 17.7 N at c = 0 / 0.5 / 1 / 4, against a weight of 52.0. Drag falls from 4.53 to 2.19 N.
- **Still machine, gannet, 10 seeds** (R:3522-3535): rigid 2/10 (0.048), c = 1 2/10 (0.032). Own gait: rigid 2/10 (0.038), flexible 0/10.
- **Unread**, three pre-registered reads before any run uses it (R:3558-3568):
  1. the two-path agreement on a flexible design;
  2. the still-machine read on the current population;
  3. an A/B run pair.
- No run with `--wing-compliance` is recorded: grep of R finds no use beyond R:187-192 and 3456-3568.
- **What the model does not have:** spanwise bending, camber change, or twist varying along the span (R:3550-3556).
- **Correction:** the butterfly section's "The model has no structural wing compliance" (R:3713) predates this build by a day and is superseded by it.

**The butterfly paper (2026-10-11, R:3572-3722).** Huang, Chen et al., npj Robotics 4:52 (2026). Its usable part is "how they measured, not the optimiser" (R:3582-3590). "Nothing below was measured here", and **W1-W3 are pre-registered and unrun** (R:3579-3580; head R:198-202 repeats "unrun"). The order is W1, W3, W2 (R:3716-3722).
- **W1** (stored data, minutes): count elites pinned within 1% of a clip bound per gene, on the arch49-51 final archives.
  - The null is N8's random bodies, mutated for the median lineage length.
  - Bounds that can bind: span 2.5, root_chord 0.9, flap_frequency 20, scale 3.0, series_stiffness 3, drive_compliance 4, stroke_amplitude 1. No telemetry exists for this.
  - "Binding" means the elite share exceeds the null's 95% upper bound **and** the pinned elites sit above the island median. Only then does a widened-bounds probe (10/20/40% past the bound, still twin alongside) run, and a widened bound is proposed only if its gain beats the twin's (R:3604-3635).
- **W2** (CPU, hours, pilot first): one factor at a time on 8 elites per island.
  - Genes: span, root_chord, flap_frequency, stroke_amplitude, scale. 7 log-spaced values, 5 held-out draws each.
  - Still twins as the control: 1,400 evaluations per island, doubled for the twins.
  - η² against a 1,000-shuffle permutation null. Any proposal (per-gene `rel`, operator weights) returns to the user.
  - It is motivated by C2 (draw variance 2.5-4.6x design variance) and by arch37's aspect-ratio confound (R:3637-3667).
- **W3** (stored labels, offline): add r̂2 and log(S·R²·r̂2²) to the critic's features, and re-run `experiments/critic_skill` on the arch49 and arch50 labels. Adopt only if out-of-fold skill rises in ≥ 2 of 3 media by more than the fold SD. It is not a descriptor (R:3669-3701).
- **Not adopted, with reasons** (R:3703-3714):
  - the DQN, the memory reward and the bound buffer;
  - θ/f and V/f objectives (undefined at f = 0, rule 4);
  - a trend + sinusoid climb fit;
  - a fore/hind phase offset;
  - a biological r̂ regression.

---

## G. User decisions and open questions in this range

**The user's requests and decisions** (quoted):
- R:3458 — "Asked by the user: add wing-surface flexibility to the model."
- R:3574 — "Asked by the user: take what is usable from Huang, Chen et al., ..."
- R:3728-3731 — "Asked by the user: find why training fails or falls short, from theory first, trusting neither the repo's claims nor its fixed ideas; write theory and predictions here before anything is measured, and fix directions here before anything is built."
- R:4124-4125, 4167 — "The user asked whether MJWarp ... can accelerate the loop." ... "The user asked the question; the answer is no, with these numbers."
- R:4344-4347 — "The user, after N9: find the design in which each body is scored on >= 100 draws ... with the fluid solver and the control law inside the device kernel so a rollout never returns to the host; Mojo is a candidate."
- R:4785-4786 — the user's question: "`max.gpu`, TileTensor, 'some of this should parallelise as linear algebra'".
- R:4923-4925 — "The user's question (10-10): can the loop be reordered so that like work is batched (across phases, across islands, across generations), and is there a better arrangement?"
- A51:719 (§D' 7) — "**The user's:** N5 alone or N5 + N11 in arch51 (recommendation: together, ...); with or without the paired air score (recommendation: without, until its four failing checks are fixed and merged); and whether N9 may `pip install mujoco-warp` into a scratch venv."
  - Outcome: "arch51 = N5 + N11 + paired air, the user's decision" (R:224), so paired air went in against the recommendation.
- A51:689 — arch51 adds "`--gait-gain` (the user's 10-09 answer 2)". A51:706 — "Pending the user: whether the air-commanded-difference branch ... is in arch51."
- **I10 §D answers** (I10:859-920; the header says questions 2, 5 and 6 were delegated and decided there):
  1. "**Yes.** A4b is built, after A6."
  2. "K_m = 50, K_s = 10 for the first device arm."
  3. "Pooled mean with re-evaluated incumbents, no lower bound." Answered 2026-10-26 (I10:854; R:5).
  4. "Upgrade to mujoco 3.15." Then at R:19 — "the user keeps the mujoco 3.15 upgrade, against D-M0."
  5. "P3 is not built."
  6. Fix `PYTHONEXECUTABLE` (corrected by D1).
- **Superseded:** I10:924 — "arch51 is not launched (the user's decision)." arch51 later ran (seed 20261018, R:601).
- **Not the user's:** R:5 — "The eel `[info]` handling, the arch60 launch and the memory-ceiling settings were advisor decisions under the user's 10-26 delegation, not user decisions." The arch60 relaunch of 2026-11-01 was "at the user's instruction" (N60:72).

**Decisions put to the user in this text and not answered in it:**
- **T4 / N7:** "accept rotor-assisted flight as the design answer (and then N6's direction is the enabling build), or change the actuator specification" (R:4010-4012). It stands, "with the spec question now 'stroke bandwidth and amplitude', not 'power'" (R:4034-4035). No answer is found: grep of R for "rotor-assisted", "actuator spec" and "stroke bandwidth" hits only R:3862, 4010, 4034.
- **A against B:** the rule leaves 1.5x-3x of today's wall to the user (R:4834). A was modelled at 1.44x (fluid in the graph) and 1.62x (fluid launched separately), and the text says the latter "falls in the band the rule leaves to the user" (R:4492-4494). The text records the **recommendation** "build A" (R:4501-4502), which I10:6 implements as a decision. No user statement choosing A is in the range read: NOT STATED. B on MAX graph was then closed by Q8 (R:4919).
- **W1-W3 and the wing-compliance A/B:** "every adoption comes back to the user" (R:3721-3722; R:3667).

**Open reads named in the range:**
- N12 is unscheduled (R:4336-4340), and N15 is unrun (R:4078-4090).
- The arch51-against-random control is announced (R:4120-4121) and not found.
- arch51's open question: why the competent share stays at 22% (R:636).
- Q6: why 1e6 floats never finished (R:4693-4696).
- Q4's caveat: soft contacts against hypersensitivity were "not separated" (R:4544-4545).
- D1's leftover: arch50's launch environment (I10:918-920).
- The eel probes (R:3).
- The A4b device half (R:9).
- arch60's memory growth cause (R:11; N60:59, 69).
- The unread reads of the device and loop sections: D-A2, D-A5, D-A6, D-A7, D-A8, P5, P6.

---

## H. Corrections recorded (newer wins) and contradictions found

1. **W\* and the epoch.** The head says that on the device "a launch is dispatch-bound below ~200 worlds, so the all-island epoch with K draws per body is the prerequisite for T8" (R:245-247). That is stale on both counts: W\* ≈ 80 (R:4499-4500, 5001-5004), and the epoch is not a prerequisite (I10:74-78, F8).
2. **T8's 16x** became 1.5x after Q2, because the per-world cost is the fluid (R:4495-4501).
3. **Identification** is "24 probes per domain per body" (R:4384, 3870), but in fact 96 worlds per body (24 × 2 signs × 2 domains), 28,800 steps, run before the segments (I10:69-72).
4. **N9's 0.68x** is rigid-body only: MJWarp 3.15 against single-thread CPU mujoco 3.11, 24 worlds × 2000 steps, no fluid (R:4142-4146). It is not a device-path number. The only end-to-end device-path number is arch60's (C.2).
5. **The decision table and I10 §3 are lower bounds.** They omit the host `finish()` terms (F4), contacts and rotors (I10:45-57, 518). D-A1 then measured the Warp fluid port's marginal at 0.0063 µs/panel, against the 0.044 µs used in the model (I10:833). My reading, not stated in the docs: the model's fluid term is pessimistic and its scatter term was missing (INFERRED). arch60's measured `evaluate.device` (115.2 s) exceeds that formula at arch60's own width, about 69 s (C.2).
6. **`wp.capture_while`.** §6's tooling brief said it "can put a whole 2000-step rollout in one graph" (R:4735-4737). Q2 found it refused (R:4466-4468). D-A3 then found unrolled graphs slow to capture (74.9 s) and slow to run (I10:840).
7. **G's "fixed ~50 µs" MAX-graph step** (R:4864-4869, 4887-4891) is refuted by Q8 at the real shape: 974.5 µs (R:4916-4919).
8. **§7's "atan2 gap"** is closed in-project by `mojo/src/mathx.mojo` (R:4873-4876).
9. **§5's placement on a lower bound, with N11 and M3 retired** (R:4705-4713), was revised: placement is on the pooled mean, and N11 is kept (R:4715-4721; I10:869-894).
10. **N8 "gen-37"** became gen 150 at matched evaluations (R:4096-4097; A51:577-582, 718).
11. **N7's wording.** The Tier-0 spar check runs at `genome.flap_frequency`, not at 5 Hz (R:4063). N7's 0.28 St figure was at 2.3 Hz with the power-optimal amplitude, not at the band top (R:4062).
12. **T3(a)'s premise** "the only bodies that fly are rotorcraft" is false of the archive (R:3993-3996).
13. **CLAUDE.md's "variation operator" sentence** was wrong (R:3793-3794, 3897). Today's CLAUDE.md calls PPO the controller.
14. **C2's single-draw reliability** appears as 0.18-0.30 (R:3834-3835) and as 0.15-0.35 (R:4379). These are two summaries of C2's R ranges. Q5's R values (air 7.3, water 4.0, land 2.0) are the latest (R:4667-4668).
15. **"The model has no structural wing compliance"** (R:3713, 10-11) is superseded by the 10-12 build (R:3456-3568).
16. **A5's red reading** (R:17) is superseded for water by the pass after `9670d85` (R:15). Air stays unmeasured.
17. **I10 F6's "the main `.venv` has mujoco 3.11.0"** (I10:66) is out of date: `.venv` is now 3.15.0, kept against D-M0 (R:19-20).
18. **arch51's N5 share** (22.4% for arch49) is not the 9% of `arch51_notes`: a different population (R:619-620). Spearman 0.47 (R:613) and 0.30 (R:3761) are likewise different populations (specialist islands' own medium, against all eight islands).
19. **Every offline re-measurement of stored elites before F2** (2026-10-12) used the wrong basis layout. That includes Q4, R1, the reproduction counts and the films (R:4625-4636). Q4's land numbers are therefore not comparable across F2.
20. **Two counts of arch49's flapping elites disagree:** N13 says 157 (R:4062) and N14 says 224 (R:4063). Neither line says why; possibly different definitions of "flapping" (NOT STATED).
21. **arch60's checkpoint provenance names main's HEAD (`a7770c2`)**, not the pinned tree's `25c7a70` that ran (D.2). The CLAUDE.md "Where things are" row says `checkpoint.json` carries "the commit that wrote it", and here it does not.
