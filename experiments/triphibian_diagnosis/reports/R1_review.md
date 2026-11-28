# R1 — fresh-context adversarial review of DIAGNOSIS.md

Reviewer R1, 2026-11-11, main @ a7770c2. Read-only: `rg`/`sed` and small `json`
reads with `.venv/bin/python`. No mujoco or torch import, no rollout, no test, no
state-changing git, no `events.jsonl` opened. This file is the only thing written.

**Version reviewed.** I began on the file as it stood at 02:15 +0800 (209 lines,
17,560 bytes). It was replaced at 03:38:58 (274 lines, 23,443 bytes, sha256
`813a9bd46907148c…`) after E0 returned. Everything below is against the 03:38:58
text; `D:n` is its line n. §1 rows 1-5, §2, §3, §4, the run table, the fallback
paragraph, the risk table and the rejected list are word for word what they were
at 02:15. New: D:33-38, §5, the P0 table, §7 item 5. Five things I had written up
against the first version are fixed in the second and are not findings here: the
seed-independent mission and crossing rows (D:140-142, P0.2), the lenient domain
test (P0.2), a no-learning check that the observation suffices for the teacher
(P0.5), `bad_qacc` on the single path (P0.9), the land result (P0.3, §7.5).

**E0.** As briefed, I did not check E0's numbers. Where the plan now rests on
them I take them as written in DIAGNOSIS §5 and `existence/RESULT.md` and check
consistency and design only.

**What the data checks rest on.** Counts such as 108 / 57,986 are recomputed from
`results/events_archNN.json` and `results/a8_tables.json` (the forensics scripts'
extractions), not from raw `events.jsonl`. Archive, config and gate checks read
`runs/archNN/archive_*.json`, `checkpoint.json` and the `*.rows.jsonl` files
directly. The literature in §4 was not opened.

---

## 1 Facts: fifteen claims of §1-§3

Tally: **CONFIRMED 10, OVERSTATED 3, WRONG 2, CANNOT CHECK 0.**

| # | D line | claim | verdict | what I saw |
|---|---|---|---|---|
| 1 | 22 | `mission_weight` is 0.0 (`evolution/loop.py:207`); arch52-60 ran `--islands air` only | CONFIRMED | `loop.py:207`. `runs/arch{52..60}/checkpoint.json` `provenance.config.islands == ["air"]`; `mission_weight` 0.0 in arch51-60, 0.3 in arch43-50. |
| 2 | 22 | Tier-1 `mission_fraction` needs no crossing (two 0.05 floors), stored to 4 dp, "so a body at 0.08 in all three media reads 0.0000" | OVERSTATED | Floors and rounding as cited (`evaluate.py:217-223`, `TAKEOFF_FLOOR` `:148`, `loop.py:1249`). The example is wrong at its own value: 0.08^1.5 x 0.0025 = 5.66e-5, stored as 0.0001. It reads 0.0000 below 0.0737, or at 0.08 with energy fraction under 0.88. |
| 3 | 22 | Continuous mission: film only, seed 0, empty without GL, unmeasured since arch51. Tier-2 re-places legs, no crossing, 0 of 1,627 passed two legs | CONFIRMED | `showcase.py:252-253`; `film.py:438-440`; `mission.py:146-151`. `film_manifest.json`: arch51 `"0/2"`; arch53-60 `"0/0"`, `legs: []`. One more caller, `control/train.py:180-193` (offline `train_controller`), not the search. `evaluate.py:590-604`; `promote` rows of `events_arch43..60.json`: 1,583 failed leg 0 + 40 leg 1 + 4 battery = 1,627, max `t2f` 0.1667. The pool is arch43-60. |
| 4 | 23 | Basis moves rotor speed a median 0.0003 of top speed per unit coefficient (191 of 191); gain scales amplitude, a rotor's amplitude is 0 | CONFIRMED | `rotor_authority/results_arch49.json` `aggregate.air.m_frac.median` 0.000273, `share_m_frac_below_0.05` 1.0, n 191 (2026-10-10, 5fba1e5). `cpg.py:427-430`; `triphibian.py:734-739`. Same file: `verdict.outcome` "NEITHER", direct hand law 0 of 24 (`aggregate.ii.max` 0.446), `lift_excluded` 153. See C1. |
| 5 | 23 | Only machine flown under power: the reference quad (0.858-0.922, turn 0.99-1.0, 2026-09-26), by a hand cascade outside the action space; never seeded | CONFIRMED | `rotor/results/fly.json`: 0.864, 0.858, 0.922; `turn_response` 0.990, 1.0, 1.0; three seeds. `fly.py:42-96`. `bodyplans.py:696`. The number is the unpaired air score: `tasks.mirror` and `pair_turn` do not exist at 5fba1e5 (`git show`), so neither this nor N6's 0.816 was under today's score (§4 item 2). "Never run in water, on land or through a crossing" before E0: nothing found by `rg`; absence only. |
| 6 | 23 | "Three of the four crossings need powered, controlled flight" (`transitions.py:447-486`) | OVERSTATED | The gates are as cited (energy-height hold `:447-459`; terminal clearance > 0.5 m `:486`). They have been passed without it: all 46 `land_to_air` and all 6 `water_to_air` crossings of arch47-60 are by bodies whose recorded air competence is below 0.012 (51 at exactly 0); 13 of the 52 have no rotor (`events_arch*.json` `crossers`). The two known leaks: a body mid-leap at the last step (A3 §6.9) and steady open-loop thrust that outlasts the hold (`ARCH46_SPEC.md:406-414`). |
| 7 | 42-44 | "the population that scores at all is gliders and swimmers, which cannot make three of four crossings by construction" | WRONG | `air_turn_bar/gate_arch56.json.rows.jsonl`: elites 72 and 77 (gannets, `n_rotors` 0) have `crossed:air_to_water` true; their still and base arms false. 18 of the 108 crossings are by rotor-less bodies (`air_to_water` 5, `land_to_air` 12, `water_to_air` 1). |
| 8 | 23 | Own policies zero in practice (0/146, 0/69, 11/111); land through the air basis; one basis per crossing; task channels zero in crossings and the mission | CONFIRMED | `meta.policy` nonzero in 0 of 146 (arch59 `archive_air.json`), 0 of 69 (arch51 air; 0 of 359 over its eight archives), 11 of 111 (arch56). For scale, arch49: 68 of 645. `evaluate.py:54-58`; `batchroll.py:1324`; `triphibian.py:1704-1711`; `run_continuous` sets no task. |
| 9 | 23 | "no observation carries distance to a boundary" | WRONG | `triphibian.py:1641` `d = self.depth()`, `:1666` `np.tanh(d / 5.0)`; `medium.py:148-151` "Negative when in air". Signed height over the sea is observed (-0.66 at 4 m, saturated at 30 m); `ppo.py:428-432` uses it as the air potential. Missing: horizontal distance and bearing to the shore, height over ground. |
| 10 | 24 | Learner crossing reward `float(crossed) x quality`; island factor boolean; shaping telescopes; curriculum only hardens, off in arch51-60; one frozen start | CONFIRMED | `batchroll.py:1259-1264`; `islands.py:277-285`; `ppo.py:499-504`; `curriculum.py:750`; `distance_curriculum` false in arch50-60 configs, true in arch48-49; `batchroll.py:1313-1319` (one `transition_scatter_seed(kind)` for every machine). |
| 11 | 24 | arch47-60: 108 crossings in 57,986 evaluations, none with two; at most 43 of 900 updates saw crossing reward | CONFIRMED | `phys.evals` sums to 57,986; `ncross_hist` has keys 0 and 1 only, 108 ones (46 / 33 / 23 / 6 by kind). 43 of 900 is arch51. As a share the highest is arch54, 25 of 300 (8.3%). |
| 12 | 24 | `air_to_water`: "77.6% were in the water before the command"; `water_to_land` asks 0.67-1.67 m/s against "the population's p90 water speed 0.446 m/s" | OVERSTATED | 44,995 / 57,986 is the note `did not hold before the command to cross`, written iff the body reached the water and its hold was under 0.5 (`transitions.py:493-498`; `scripts/a8_events.py:60-61`). An air hold also fails on energy-height loss without touching water (`:447-459`); the share in the water before the command is not recorded. 0.67-1.67 is 3-7.5 m (`ARCH46_SPEC.md:274-275`) over 4.5 s: consistent. 0.446 is arch40's first 400 generations (`tasks.py:58-62`), before arch43's corrected fluid: not comparable with any run in the pool. |
| 13 | 25 | 13,199 parameters; about 3,500 bodies per run; at most 400 decisions per medium and 150 per crossing; 2.8e6-2.3e7 per run; return 95-100% shaping | CONFIRMED | 33-64-64-7 actor 6,791 + critic 6,401 + 7 `log_std` = 13,199. `phys.evals` 3,467-3,562 in the 300-generation runs (arch51: 10,283). PPO `transitions` sums 2.78e6 (arch52) to 2.29e7 (arch51). Terminal share recomputed from the `ret` rows: air 0.0-5.4%, crossings 0.0-0.8%. Holds for air and crossings only: water is 14-27% terminal, land 10-21%. |
| 14 | 26 | Reliability 0.18-0.30; 66 / 36 / 18 draws for 0.9; land 0.0992 to 0 on 1e-16; retention 0.02-0.06; random bodies 1.94x | CONFIRMED | ROADMAP:259, :3835; :4668; :4612-4622 (one elite); :615 (land n = 9, water n = 2, air not measurable); :4101-4111. Two R estimates are joined: 0.18-0.30 is C2's R 2.5-4.6; 66/36/18 is Q5's R 7.3/4.0/2.0, whose one-draw reliability is 0.12/0.20/0.33. |
| 15 | 58-61 | PPO learned a fixed body in water (0.0003 to 0.04-0.07 held out); the air arm had at most 48,000 decisions and no result file | CONFIRMED | ROADMAP:5682-5707 (three seeds, 240 rollouts; land split, third land seed never recorded). `experiments/fixed_body_learning/` holds `README.md` and `run.py` only. The water arm is the same unfinished experiment with the same missing file (C4). |

Further checks, not in the tally:

- D:28-31 booking and "40-72%": CONFIRMED. `energy.py:211` `np.abs(tau * om)`. Still arm "battery exhausted" 368/912 (arch58), 784/1,088 (arch57), 0/1,160 (arch55), 0/888 (arch56) in `air_turn_bar/screen_arch5N.rows.jsonl`. These are air draws of air-island populations.
- D:30-31 buoyancy genes in Tier-0 only, rotors forward only, no seed plan with a rotor: CONFIRMED (`fluid.py:1405`; `mjcf.py:796-805`; `bodyplans.py:646-657`).
- D:37-38 `has_propulsor` does not count rotors: CONFIRMED by reading (`phenotype.py:811-819`; gate at `loop.py:520`, `:694`; `tier0_gate` -0.85 at `:150`). The -1.00 margin was not re-run.
- D:52-55: 9/139 each: CONFIRMED (`no_model_gate/results_arch51_table.md:24,30`), on MuJoCo 3.11, at the recorded draw. 2 and 42 of 1,446: CONFIRMED by recount of `runs/arch5{1..9}/archive_*.json`.
- D:78 hold 0.78-0.84 on arch46: CONFIRMED (`transition_distance/README.md:207-212`: 0.844, 0.775).
- D:80 "84 of 92": matches the `a8_tables.json` headline. "Two runs exceed 2e7": WRONG against the same table: arch51 only (2.29e7); arch57 is 1.89e7.
- D:25 3.5% aloft: CONFIRMED (`air_turn_bar/README.md:291-298`: 9/256, eight gannets, one seed).
- D:67 2.49x is A5's figure from `cost.seconds`; `a8_tables.json` medians give 243.6 / 106.0 = 2.30x. arch59 (batched) ran at 218 s per generation while sharing the GPU with arch58.
- D:201 against D:266-267: `results/canary_test_search.log` ends "all search-machinery checks passed", 0 `[fail]`, 0 `[skip]`. D:266-267 is stale.

---

## 2 Causal logic

**C1. "Each of the first three is enough, alone … which is what sixty runs have
shown one lever at a time" (D:16-18) is not supported.**
- No run removed any of the three. `mission_weight` was 0.30 in arch43-50 and from 2026-09-01 (`loop.py:196-199`), with the mission at 0 throughout. The continuous mission as a signal was measured before building and dropped (0 of 90 complete, ROADMAP:2786-2791). The rotor channel was never changed. The learner's crossing reward was never graded; the curriculum was on in arch48-49 only and never advanced.
- The levers the runs did pull are others (A4 summary 2: take-off scoring, the `land_air` island, lift rungs, thrust rungs, the passive-water fix, the distance curriculum).
- Cause 2's one direct test came out "neither": the direct hand law flew 0 of 24 lift-capable arch49 rotor bodies, 153 of 191 cannot lift their weight, and the ROADMAP concludes "it is not the binding gate today" (ROADMAP:3984-3999). D:23 quotes the 0.0003 and leaves the rest out.
- E0 changed the body, the controller, what the controller reads and the Tier-0 gate together. It shows the set is jointly sufficient under a hand law. It isolates none.
- Fix: "all present at once; none was ever removed; removing them together, with a hand controller, works (E0)". Drop "sixty runs … one lever at a time".

**C2. "Control authority … first … because nothing could have used a better
signal" (D:86-89) contradicts the air record.**
- The only certified air credit and the only driven `air_to_water` crossings come from gannets the basis can steer: arch56 7/111 against 0 still and 0 base (ROADMAP:398-406), pooled 34/946 against 0/946 (ROADMAP:26-36), a gannet turned 12% by a linear policy (ROADMAP:161), elites 72 and 77 (row 7).
- Seven air-island runs on those bodies produced no replicated reader. The ROADMAP's own reading: "The gap to L1 is a learner gap" (ROADMAP:421-424). Where authority exists, cause 4 binds.
- A2 §9.5 ranks reward and credit at or above action authority. The diagnosis reverses that and does not say why.
- Fix: cause 2 binds rotor-borne flight; cause 4 binds where authority exists; rank both first. The plan already behaves that way (teacher, one fixed body).

**C3. Causes §1 does not list.**
- What the controller can see (D:164-166). After E0 it is co-equal with the rotor channel. It lives in §5 and P0.5; §1 still counts five.
- Tier-0 rejects a bare multirotor (D:148-151). Also outside the §1 table.
- The population holds no flyable rotor body even under direct command (0 of 24), cause not found (A3 §0). E0 reports the reference quad's mixer has rank 3 and flies (D:170-171), so mixer rank does not explain it (P6).
- Time against distance in the filmed mission. Legs are 8 s where `MissionSpec` has 300 s (`triphibian.py:88`); distances are not compressed (land spawn to water about 8 m, water spawn to the shore 16 m; `triphibian.py:569-573`). At arch40's p90 speeds (0.116 m/s land, 0.446 m/s water) no walker or swimmer completes a leg change under any controller. A1 §9.5 has it as INFERRED. It bears on P0.2, which keeps 8 s legs in the definition of done.
- Energy for rotor flight: arch54's rotor gannets replay at 8-15 kW against a 5.25 kW pack limit (ROADMAP:391-394); 20-72% of elite air draws end "battery exhausted" (screen rows). D:56 clears power on N7, a derivation for flapping.

**C4. §2 clears four things on thin evidence.**
- Physics (D:50-55). The water and land bars are certified at the recorded draw on MuJoCo 3.11. Held-out retention is 0.02 (n = 9) and 0.06 (n = 2) (ROADMAP:615). Water and land are not comparable across the 3.15 engine the `.venv` now runs (ROADMAP:19-24; `no_model_gate/D_M0_read.md`: 28 and 50 of 229 elites stop reproducing). After E0 the paragraph should carry D:159-163 (land locomotion not shown) and D:167-169 (the water exit diverges unless staged).
- Rotor model (D:56-57). CT rms 0.012 is a static fit in air (`rotor/results/result.json` `"0.12".rms_ct` 0.0116, camber fitted; `test_physics.py:3743-3746` allows 25% on static CT). Nothing validates a rotor in water or at the surface, the manoeuvre E0 found fragile; A3 §6.5 and A9 §0.6 list what the model lacks there.
- PPO (D:58-61). Three seeds, 240 rollouts, an experiment stopped at 7 of 9 runs with no result file: the reason D:60-61 gives for voiding its air arm. Its land arm split (0.050 and 0.001). E1 never ran. ROADMAP:7483-7510: "No fifth shared-policy variant"; distillation across bodies held-out R² -0.047.
- "The zeros are honest" (D:62-63). They are. The non-zeros are not all honest: E0 reports `continuous_score` paying a still machine 0.15-0.58 and the land score paying a hover (D:156-163), and the tracker has two open leaks (row 6).
- Fix: title the section "not shown to be wrong"; give each line its run, date, engine and n.

**C5. §3 outcomes.**
- Prediction 2 (D:78): "Refuted" on arch46's elites, outside the pre-registered pool. In arch47-60 the hold is recorded only for bodies that reached the target: bounds 0.01%-93% (A8 T2d). Write "refuted on arch46 (2026-10-04); unmeasured in arch47-60".
- Prediction 3 (D:79): 2.9% is under the 3% line because 1,087 of the 1,446 elites are air-island archives that never select for a second medium. arch51 alone: 25/359 = 7.0%; per evaluation 3.12%. That is between the two frozen lines (3% and 10%).
- Prediction 4 (D:80): the frozen statistic was `return_by_tag`. It reads 42 of 58 under 1 sd and 6 at 1.5 sd or more, up to +2.08 sd (arch54 `land_to_air`), all shaping (A8 headline). The diagnosis reports `reward_by_tag` and does not say it swapped.
- D:86-91 gives prior weights and no posterior weights.

---

## 3 The plan

Ordered by how likely each is to make "L3 within three runs" false or empty.

**P1. Run 1's gate can be met with the learned part contributing nothing, and
the fallback sentence is false as written (D:207, D:211-214).**
- P0.2's gate requires the E0 controller to pass the gated evaluator before any training (D:193). P0.4 and P0.8 put the cascade, the medium-aware rotor law and the staged water exit in the inner loop (D:195, D:199). The run gates carry two nulls: rotors stopped, and the base gait. Neither is "the same inner loop under an untrained policy".
- "which cannot score below the teacher on the evaluator" (D:212-213). A nonzero correction changes the closed loop. D:167-169 reports tumbling, two `bad_qacc` and a divergence when the exit is not staged; ROADMAP:4612-4622 has a land score going 0.0992 to 0 on a 1e-16 difference. True statement: the zero residual is in the candidate set, so the best checkpoint is at least the teacher on the validation seeds, not on the test seeds.
- "The user's condition is met at run 1 or at the fallback" (D:213-214). If the teacher passes, the condition is met by a hand controller with zero runs. If the user requires a learned controller, teacher plus a possibly null residual does not meet it. If the teacher is under the gate, so is the floor.
- Fix: (a) a §7 item: which of {hand controller; hand inner loop + learned outer loop; learned end to end} is "the agent". (b) Three more arms on every run gate, through the identical inner loop: randomly initialised policy, zero-output policy, command channels shuffled; each reads 0 of K. Report teacher, student and the difference. (c) Replace D:211-214 with the true statement.

**P2. The gates sit on an instrument with open leaks a thrust-capable body
reaches, and the nulls lack the arm that exposes them (D:193, D:197, D:207-208).**
- Steady open-loop thrust passes the water hold when the start is far enough out: "Closing it would take a station gate … Not built" (`ARCH46_SPEC.md:406-414`; `transition_distance/README.md:217-219`). P0.6 moves starts back, which widens that window.
- Terminal clearance passes a body mid-leap at the last step (`transitions.py:486`; A3 §6.9).
- The anchor's base gait is rotors at 1.05x hover (`existence/RESULT.md:239`). E0's only null was rotors stopped. The base arm on the anchor through the crossings and the mission is unmeasured, and P0.2's gate sentence names the still arm only.
- The tracker's land hold tolerates 0.10 m of clearance (`transitions.py:268`, `:432`). The 3 cm hover D:159-161 objects to counts as "on land" there. P0.3 gates the land segment, not this.
- The schedule is fixed: air, water, land, 8 s each (`triphibian.py:72`, `mission.py:167-175`). The P0.2 seed moves the entry state only. Rule 8 asks that a command be answerable both ways; a timed script is not excluded.
- The graded `approach` is 0 for an uncrossed `air_to_water` (`transitions.py:713-726` covers air and land targets only), and it has paid a still body up to 0.145 (`ARCH46_SPEC.md:414-417`). "Still earns 0" may hold on the anchor (D:137) and fail on run 2's bodies (rule 3).
- Fix, before P0.2 and P0.6 are read: build the station gate and a dwell on arrival. Add arms: open-loop rest throttle, command frozen at the first medium, command shuffled. Draw leg durations per seed. Use P0.3's weight-on-ground quantity for LAND inside the evaluator. Pay the hold for `air_to_water`. A mutation per gate in every P0 row, not only P0.2.

**P3. Run 1's gate is not yet checkable, and its power is unknown (D:207).**
- "each medium above its certified floor on 8 fresh draws": which floor (0.012 or 0.15; for air, existence or the R1 sign test, ROADMAP:26-36), which statistic over the 8, certified where. The stored floors are MuJoCo 3.11, and P0.3, P0.7 and P0.9 each move a score they were certified on.
- At least 7 of 8 on each of three starts passes with probability 0.54 at a true per-mission rate of 0.9, 0.84 at 0.95, 0.13 at 0.8. The teacher's own rate under seed-dependent entry states is unmeasured: the mission is n = 1 per start (D:140-142); on crossings B0 is 29 of 32.
- Fix: in P0.2 run teacher, still, base and the P1/P2 arms on at least 32 seeds per start. Set K and the pass count from those distributions (rules 5 and 9). Re-certify each medium's floor on the anchor after P0.3/P0.7/P0.9. Freeze all of it before run 1.

**P4. The student cannot hold the teacher's state (D:196, D:207).**
- The policy is a memoryless 33-64-64-7 MLP at 25 Hz. The E0 law has latches and timers: staged exit, "sit until still, then taxi", a settle time-out, a moving position reference (`RESULT.md:251-271`, `:286`).
- P0.5's gate is good and involves no learning. It proves the observation is enough for a stateful hand law. It does not show a feed-forward student can represent it.
- Fix: in the P0.5 gate the observation-fed controller is stateless except for state exported as an observation channel. Or state that the latches live in the inner loop, in which case P1 applies with more force.

**P5. Run 2 has no arithmetic, and the document's own evidence is against its
premise (D:208).**
- 300 generations x 16 children x 3 starts x K = 8 missions of 600 decisions = 6.9e7 decisions for evaluation alone: 12.6 h on 16 ideally scaling processes at the quad's 95 decisions per second. A fine-tune of 1e5 per child adds 4.8e8 (88 h). D:107-108 says co-design gives each body 5e5-5e6.
- D:112-113: "No report of one policy over a population that is still evolving". Run 2 is that, plus a fine-tune of unstated size.
- The same hand law already differs between two near-identical quads: 32 of 32 against 29 of 32 (D:141-142).
- Fine-tune seeds and placement seeds must be disjoint (M1: 8.5-23% of a draw-selected gain survives, ROADMAP:5906-5939). Stated for run 1, not for run 2.
- "placed by the mission evaluator": the scalar is not named. A pass/fail mission as a population's signal is the wall ROADMAP:2786-2791 measured.
- "distinct" is undefined. L2 includes `water_to_air`, which the mission never commands (`triphibian.py:74-75`).
- Fix: before run 1, a no-learning read: the teacher on at least 20 anchors mutated by the operators run 2 will use; share passing each crossing. Then write children x generations x per-child budget from a dry run.

**P6. P0.7 bundles one real gate with two changes not shown to be needed (D:198).**
- Take-off factor from `land_to_air`. `mission_weight` is 0.0, so `mission_fraction` is not selected on, and 0.011-0.091 is 37-300 times the largest value any run stored (0.0003). The change adds a comparability boundary and counts `land_to_air` twice (it is already in the transition factor, `evaluate.py:185-194`).
- Allocation check "mixer rank 4". The reference quad has rank 3, flies and turns (D:134, D:170-171). So rank does not separate the quad from the 24, and "(which also answers why the hand law flew none of them)" is unsupported. If the check gates, it rejects B0 and B1 as built and every partial rotor set, including the rotor-carrying gannets of the best air population (24% of arch56's): rule 9.
- Fix: P0.7 is `has_propulsor` only. The allocation check is telemetry until its read on the 24 bodies and on B0/B1 is in. Leave the take-off factor, or justify it with a number.

**P7. P0.3's bar is referenced to a population, and E0b is an unscoped stop (D:194).**
- "bar set from the measured distribution of the land-certified elites (rule 5)". ROADMAP:516-521 and CLAUDE.md rule 9: a bar is referenced to the null; a population quantile is not a bar. The nulls exist: the hover-taxi and the 94%-thrust skid.
- E0b (hand-driven legs on a 2 kg quad) has no pre-registration and no budget, changes mass and drag, and ends the plan if it fails.
- Fix: the bar is a weight share above the skid null's, measured on those arms. Pre-register E0b as E0 was, with its stop rule and the branch if it fails.

**P8. Boundaries and house rules the P0 list does not mention (D:190-201).**
- P0.3, P0.7, P0.9 and the observation change each move a score or the policy's input. None is declared as a comparability boundary. Stored 33-input checkpoints stop loading.
- "anything that changes selection lands behind a flag, default off" (ROADMAP:2760).
- CLAUDE.md "search before adding": P0.6 overlaps `DistanceCurriculum` (`curriculum.py:689-757`, the same 0.5 rule, `_place_for(back)`); run 1 overlaps `experiments/fixed_body_learning/run.py`; P0.2 overlaps `control/train.py:172-193`. None is named.
- E0 ran the numpy single path only. No gate checks the inner loop and the evaluator against the batched path, and none has a film reproduce the evaluator, which E2 L3 requires ("film reproduced", ROADMAP:5533).
- P0.5 adds world-frame quantities; `triphibian.py:1604-1608` records the opposite choice ("No global position, no ground-truth world velocity"). `docs/AUTONOMY.md` ("Removing the hand in the loop") and README line 3 ("flapping-wing machines") are the recorded direction §7.1-2 reverse. §7 should say so.
- Nothing is in `docs/ROADMAP.md` yet, and the directory is untracked: the pre-registrations are frozen by mtime only.

**P9. Budget for run 1 (D:176-178, D:207): consistent, with one modelled number.**
- 1e7 decisions at 95 per second is 29 process-hours: 1.8 h on 16 perfectly scaling processes, 7.3 h on four. A2 §9.1's harness figure (75 s per 12,800 decisions) gives 16 h; arch56's whole loop did 8.2e6 in 9.8 h. All fit one run.
- "Two to three hours on 16 processes" is a modelled wall on a 6P+8E machine with 4.9 GB free. CLAUDE.md records a modelled pool shape losing by 3x. D:178 says "to be confirmed by a dry run": make it a gate.
- 1e7 over eight task kinds is about 1.2e6 each, under A10's 1e7-1e8 per skill. That is acceptable only because cloning replaces RL, which is P1.

**P10. Counting to three (D:207-209, D:232-234).** A from-scratch ablation
"inside run 1" is a second training. Run 3 as "a second seed of 1+2" is two.

### The brief's list: repeats and recorded decisions

- "One triphibian machine, not a chain of pairs" (ROADMAP:3029-3030): respected.
- Rotorcraft allowed (ROADMAP:6340-6341, 3028-3029): respected. Whether rotor-borne flight is the accepted answer is open, as §7.1 says; grep finds no answer.
- Bars referenced to the null (ROADMAP:516-521): the run gates are. P0.3's bar and P0.4's "0.1 of top speed" are not.
- Repeats: none outright. Two near-repeats to name and distinguish: a continuous mission as a population's signal (ROADMAP:2786-2791), and a start-distance curriculum with a 0.5 advance share (ROADMAP:3408; start rate 0, never advanced). P0.6 differs by starting past the interface.

### P0 gates at a glance

| item | checkable as written | null arm | where a do-nothing or leaky machine gets through |
|---|---|---|---|
| P0.1 | yes, as edited | rotors stopped only | The gate text changed after the result: at 02:15 "The table in `existence/PREREG.md` §3; held-still arm 0"; now "The table in §5; held-still arm 0 on media and crossings". By the frozen text prediction 6 is refuted for the mission and prediction 4's surface clause is refuted (`RESULT.md:115-126`, `:133-142`). "Passed" should read "5 of 7 confirmed, 2 refuted in part". |
| P0.2 | once K, the seed's effect and the pass count are fixed | still in the gate; base in the text only | open-loop thrust, timed script, 10 cm hover (P2) |
| P0.3 | no: bar not set, "while moving" undefined | hover-taxi, 94% skid (good) | bar referenced to a population (P7) |
| P0.4 | partly: "moves the commanded way" has no number and no ± pair; land absent; "0.1 of top speed per unit action" is typed and undefined for a set-point | still | no zero-set-point arm |
| P0.5 | yes | not needed | passes with a stateful controller (P4) |
| P0.6 | yes | still | open-loop thrust; no signal for `air_to_water`; 0.5 is typed (A10 §4: keep starts with success 0.1-0.9) |
| P0.7 | first two yes; "classified" is not pass or fail | none | P6 |
| P0.8 | yes | none | 8 of 8 bounds the true rate only above 0.69; seeds 0-3 were also the tuning seeds (`RESULT.md:397-399`). Use at least 32 unseen entry states and report `bad_qacc`. |
| P0.9 | yes | still | name the population the "0 draws" is read on |
| P0.10 | yes | none | covers today's tests; each new gate needs its test and its mutation |

### E0's design (`existence/PREREG.md`), not its result

- Items 4-5 asked for "7 of 8 seeds" on rollouts the seed does not reach (`transitions.py:223-235`, `mission.py:204`). RESULT and D:140-142 say so; X1 was added outside the pre-registration.
- One null. No open-loop rest-throttle arm and no command-blind arm, on a body whose rest throttle is hover.
- No rule separating tuning seeds from scored seeds.
- Prediction 1 leaves 0.012-0.5 undecided. Prediction 2's falsifier (depth within 1 m) is not its prediction (a score). Prediction 3 has no falsifier. Prediction 7 is true by construction (the pack is sized to the energy).
- A three-hour stop cannot carry "no run will" (the 02:15 §5; PREREG §0 is more careful).

---

## 4 Wording

1. D:22 "`mission_weight` is 0.0": add "since arch51; 0.30 from 2026-09-01 through arch50, with the same zero". As written it reads as a lever.
2. D:23 "(0.858-0.922, turn 0.99-1.0, 2026-09-26)": three seeds, unpaired score; air is not comparable across 2026-10-10. D:59-60 voids M4's air arm for that boundary. E0 now has the paired number; cite that.
3. D:23 "the only actuator that flies": gannets earned the certified air credit. Say "the only actuator with measured thrust over weight".
4. D:23 "nonzero in 0/146, 0/69, 11/111 archives": elites of the air archives of arch59, arch51, arch56.
5. D:24 "At most 43 of 900": name arch51; as a share arch54's 25 of 300 is higher.
6. D:24 "77.6% were in the water before the command", "the population's p90": row 12.
7. D:25 "about 3,500 … per run": 300-generation runs. "Its return is 95-100% potential shaping": in air and crossings.
8. D:26 "reliability 0.18-0.30 (0.9 needs about 66 / 36 / 18)": two R estimates. "retention … 0.02-0.06": n = 9 and 2, arch51.
9. D:28-30 "up to 40-72% of draws": air draws, arch58 and arch57; 0% in arch55 and arch56.
10. D:33 "E0 … measured the first two causes directly": it measured that a hand law with privileged state suffices and that two instruments misread. The search's own controller was not run on B1.
11. D:35-36, D:144-145 "completes the continuous mission", "through all three media": D:159-163 says land locomotion is not shown; the mission rows are n = 1.
12. D:152 "calls the working machine a failure": 0.011-0.091 would be the highest `mission_fraction` on record, and nothing selects on it.
13. D:56 "Power. 3.6-28x sufficient": flapping stroke power, a derivation; the beetle at the band top is 1.3 (ROADMAP:4014-4022).
14. D:64 "by generation 150-250": coverage by 40-160 (arch48 up to 275); best per medium by 250 in 21 of 27 pairs (ROADMAP:126-133).
15. D:198 "(which also answers why …)": P6.
16. D:218-226: the risk table keeps the 02:15 numbering. "P0.3 adds them" is now P0.5; "P0.6's allocation check" is P0.7; "P0.5" for booking is P0.9; its first row is done.
17. D:254-255: decision 4 names the count as the only difference from E2 L3. P0.2 also changes the transition test and the on-task rule and drops the film.
18. D:14-18 "Five causes" after §5 added two. D:266-267 stale against D:201.
19. D:247-249 "Without it there is no evidence-backed path": with it the path also needs §7.2, §7.5 or E0b, and P0.2-P0.9.
20. Offline reads of stored elites before 2026-10-12 are "not comparable across" the basis-layout fix (ROADMAP:4634-4636). The arch46 hold-pass (10-04), C2 (10-08), Q5 and N6 part B (10-10) are cited without that label.

---

## 5 Omissions: report evidence that contradicts or qualifies the document

1. A5 §0.5 and §H.12, A3 §0: N6 was "neither"; 153 of 191 rotor elites cannot lift; the direct hand law flew 0 of 24; "T3(a)'s premise … is false of the archive". (C1)
2. A2 §9.5: reward and credit rank at or above authority. A7 §0.4: no replicated reader on steerable bodies. (C2)
3. A3 §3 and §5, A7 §0.5: driven `air_to_water` by rotor-less gannets, still 0, base 0. (row 7)
4. A1 §9.5: time against distance. A1 §9.14: `water_to_air` is not a cycle transition, yet the transition factor averages it. (C3, P5)
5. A1 §3.3, A8 §3.5: the hold is unrecorded for bodies that never crossed. (row 12, C5)
6. A8 headline: the `return_by_tag` read. (C5)
7. A4 summary 10: "making the evaluation continuous now (0 of 90)" is on its do-not-re-propose list. A4 §F1: flags default off. (P5, P8)
8. A3 §6.5 and §6.9, A9 §0.6: rotors in water are unvalidated (no cavitation or ventilation); the mid-leap leak; the solver lacks thrust collapse at the surface. (P2, C4)
9. A9 §0.2: no flying triphibian is shown crossing a shoreline on the ground; the literature's land mode is wheels pushed by rotor thrust. P0.3's gate as worded would pass that and reject a hover; §7.5 should say which the user is choosing between.
10. A10 §4: its plan spends run 1 on per-skill experts (1e8-1e9 steps), run 2 on one controller, run 3 on bodies, and records a distillation that lost over 40%. The diagnosis folds the first two into 1e7 decisions and does not say it departs from its own literature report.
11. A7 §0.8: held still, bodies stay aloft through the command more often than actuated in 6 of 8 populations. It supports causes 2 and 4 and is not used.

---

## 6 The three changes that would most raise the chance of L3 within three runs

1. **Say what "the agent" is, and gate it.** Put the question in §7. Add policy-null arms through the identical inner loop (untrained, zero output, command shuffled) to every run gate. Delete "cannot score below the teacher". Without this run 1 succeeds emptily or fails by definition (P1).
2. **Finish the instrument before anything trains against it.** Station gate and dwell in the tracker; open-loop rest-throttle and command-blind arms; seed-drawn leg durations; weight on the ground for LAND inside the evaluator. Then run the teacher and every null on at least 32 seeds per start, and set K, the pass counts and each medium's floor from those distributions, on the anchor, under MuJoCo 3.15, after P0.3/P0.7/P0.9 (P2, P3).
3. **Two no-learning reads before run 1, and a smaller P0.7.** (a) The observation-fed controller made stateless, or its state exported: the student's ceiling (P4). (b) The teacher on at least 20 mutated anchors: run 2's premise, and the number its budget is written from (P5). Cut P0.7 to `has_propulsor`; pre-register E0b with its failure branch (P6, P7).
