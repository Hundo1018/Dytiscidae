# Lead's running notes (write-as-you-go; the reports under reports/ are the evidence)

Session 2026-11-10. Cheap suites 8/8 pass at a7770c2; heavy canary NOT run
(1 GB free RAM, swap 96% used: ComfyUI 3.7 GB, win11 VM ~8 GB).
PREREG.md frozen 21:46 before any report returned.

## A5 (ROADMAP 3456-5214, ARCH51/IMPL_1010 specs) — returned first
R = docs/ROADMAP.md
- N8: 300 random prior bodies beat arch49's gen-150 archive at matched evals:
  1.35x coverage, 1.94x summed competence (R:4101-4111). arch51 re-read never recorded.
- N5 fixed ranking not competence: Spearman 0.47->0.97, competent share 22.4->22.8% (R:612-618);
  retention 0.02-0.06.
- Land is chaotic at machine precision: 1e-16 command difference moves land 0.0992 -> 0
  (R:4618-4622); a 1% null moved 34.5% of land elites by >50% (R:4537-4541).
- Draws needed for reliability 0.9: ~66 air, 36 water, 18 land (R:4667-4669); path uses 1.
- N6: identified bases move rotor speed by 0.0003 of top speed per unit coefficient;
  the hover law that flies the reference quad at 0.816 flew 0/24 archive rotor bodies;
  153/191 rotor elites cannot lift their weight (R:3984-3993).
- N7: flapping power 3.6-28x sufficient; N13 thrust regime reachable ~2 Hz; N15 (propulsive
  efficiency check) never run.
- Device path: arch60 median 246.9 s/gen vs arch56 99.3 s (2.49x SLOWER; device scoring
  added on top of batched, inferred). arch60 stopped gen 112/150 on 2026-11-01 (desktop OOM,
  Linger=no); last checkpoint gen 100; its frozen reads never read. A4b blocked, eel gap open.
- No recorded user decision on T4 (rotor-assisted flight vs new actuator spec) or A-vs-B.

## A6 (ROADMAP 5215-7620: sweep, flight audit) — returned second
- Only a hand-flown quad flies (never seeded): 0.963/0.974/0.987 (R:6252-6255), 0.86-0.93 under
  AE height term (R:2886-2889). Its water/land scores and any CrossingTracker run: nowhere.
- Crossing failures (R:5283-5293): water->air and water->land: 0 fail the hold, 99% never reach
  the boundary in 6 s. air->water fails the hold 86-90% (cannot stay up). Curriculum cannot
  advance: start rate 0 at back 0 vs advance share 0.5 (R:5993-5997, 6128).
  => PREREG prediction 2 (not started) is WRONG for the water-origin crossings: they start and
  fail in the go phase (capability/time), right only for air->water.
- Mission is <= 1e-4 not exactly 0: mission_fraction > 0 in 7 of 4,792 arch48 evals, max 0.0001
  (R:5885-5886); 8 arch49 elites pass mission > 0 (R:5267).
- M4 "air not learned" is weak: old unpaired air score, rotor body whose base gait already 0.128,
  240 rollouts, 7 of 9 runs in, final read recorded nowhere (R:5684-5692). Water did learn
  0.04-0.07 vs seed spread 0.029.
- Flapping wall moved three times: heave-only best +0.651/300 gaits (09-23); actuation 7-12 Hz,
  teal 1.28 prescribed vs 0.42 free joints (09-26); N7 refutes power (3.6-28x enough), only level
  gait 11.33 Hz lies outside the 7.80 Hz spar band; N15 never run.
- E1/E3 funded, never run. M5 waits on M4.
- Refinement keeps 8.5-23% of reported gain (M1); KL bound never binds (0/300, median 0.0062 vs
  0.015); shared policy's weight indistinguishable from zero at 60 gens (R read).
- E2 adopted, nothing in it met. Air L1 open; L2, L3 = 0.
- Rotorcraft ARE allowed by the user (09-26, 09-30: R:6340-6341, 3028-3030). Whether
  rotor-assisted flight counts as "flight solved" was asked (R:4010-4012, 4033-4035): no answer
  recorded. Nothing requires the machine to flap.

## A2 (controller and learner) — returned third
- Learner's crossing reward is BINARY float(t.crossed) (batchroll.py:1259-1264); archive's is graded
  `approach` (transitions.py:214, evaluate.py:185-194); comment at batchroll.py:1255-1258 is wrong.
  Positive crossing reward: ~47 of 55,480 crossing trajectories (arch51), ~8 of 20,080 (arch59).
  Air island reads crossings as a binary factor floored at 0.15 (islands.py:272-285).
- The policy CANNOT REACH A ROTOR: it acts through the identified basis; rotor speed moves a median
  0.0003 of top speed per unit coefficient (191/191 rotor elites, rotor_authority/results_arch49.json);
  gait gain scales amplitude only and a rotor's base amplitude is 0 (cpg.py:427-430,
  triphibian.py:734-739). The reference quad is flown by a hand mixer writing rotor speeds from world
  state (experiments/rotor/fly.py:41-96): single-path only, outside the action space, never measured
  under the paired air score.
- Own policies are all zero in practice: 0/146 nonzero (arch59 air), 0/69 (arch51), 11/111 (arch56).
  ES accepts only on mission_fraction (~0). The only controller acting: the shared PPO net (13,199
  params) at its mean, through the basis, on top of the evolved open-loop CPG, 25 Hz, open loop in 40 ms.
- Learner's return is 0.95-1.00 potential shaping in air and 0.99-1.00 in every crossing; air potential
  = altitude + upright, altitude saturated at the 30 m launch (ppo.py:428-432); nothing relates to turn.
- M4 air arm: <= 48,000 air decisions/seed, old unpaired score, rotor body the policy cannot steer;
  results file NOT FOUND. E1 never ran (0 generations in all 10 arms).
- Budget: per body evaluation <= 400 air decisions on one turn draw, 400 water, 400 land, 150 per
  crossing kind. 300-gen run banks 7.0-8.3M decisions over ~3,500 bodies (arch51 22.9M, arch57 20.0M).
  Each crossing kind uses ONE fixed entry state every generation.
- Nothing handles the medium switch: land is driven through the AIR basis (evaluate.py:54-58); each
  crossing keeps one basis for both media (batchroll.py:1324); task channels are ZERO during crossings
  and the mission (triphibian.py:1704-1711); hold->go only via domain one-hot; obs has no distance to
  any boundary.
- Exploration at log_std -0.5 keeps 3.5% of glider rollouts aloft through the turn command
  (air_turn_bar/README.md:291-298); log_std ended above its start in 8/9 runs. No scripted swimmer,
  walker or crossing teacher exists.
- Throughput through the full harness: ~1.6 h per 1e6 decisions; ~7-13 h per 1e6 AIR decisions.

## A7 (recent runs arch51-arch60) — returned fourth
- arch60 unread and incomplete: gen 112/150, last checkpoint gen 100, stopped 2026-11-01 21:56:30
  (no kernel OOM after 20:41; gdm greeter 21:56:20 => user session ended, inferred). No gate/screen output.
- PINNED WORKTREES DO NOT PIN when cwd is the main tree: sys.path[0] is the cwd, so `dytiscidae` imports
  from main whatever PYTHONPATH says. arch60 gens 41-112 ran a7770c2 (pin + f1281bd device PPO feed +
  2 telemetry lines). CLAUDE.md:190-196 is wrong (worker imports not probed separately).
- Every evaluation still scores water, land, crossings, mission. What stopped after arch51: later runs
  have ONLY the air island; mission_weight 0.0 since arch51; mission_best / three_media telemetry gone;
  nothing re-measured on fresh seeds except air.
- Air: existence certified pooled on fresh seeds 34/946 vs 0/946 still. Command reading: no replicated
  pass (best arch54 7/8 one elite; arch56 stage 2: 0/16). Best R3 3 of 5 needed. arch56 did not replicate
  on a second seed (arch58: R2 1/114).
- Crossings at recorded seeds since arch49: best clean 2/111 (arch56 air_to_water); 4/145 in arch55 is a
  leak vs base gait. No elite EVER made water_to_air. Last fresh-seed crossing read: 2026-10-04.
- Mission: 1-4 elites per population have mission_fraction > 0 (max 4.3e-05), none crossed anything. L3 = 0.
- Nearest three-media bodies: C-on idx 6, arch56 idx 21, arch59 idx 137 clear all three 0.012 floors at
  the recorded seed; fresh-draw air keeps 0.01, 0.54, 0.42; fresh water/land never measured; none crossed.
  Pooled arch51+arch53-59: 24/1,085 elites clear two media, 2/1,085 clear three.
- Actuation costs time aloft: held still, bodies stay aloft through the command MORE often than actuated
  in 6 of 8 populations. Still-arm battery exhaustion 0-72% (arch57: 72.1% still vs 72.0% elite) =>
  exhaustion is mostly booking (PowerBudget), a user decision.
- Water/land L1 fail on retention not existence: arch51 held-out retention water 0.06 (n=2), land 0.02 (n=9).
- mujoco 3.15 boundary is arch60 only. arch59 final archive has no gannets.

## A3 (body space and physics feasibility) — returned fifth
- Physics does not forbid the mission; every ingredient exists on SOME body, never all on one.
- Powered flight + turn: reference quad 0.858-0.922, turn 0.99-1.0 (experiments/rotor/results/fly.json,
  2026-09-26); never run in water, on land or through a crossing. Evolved rotor bodies: air 0, diverge in water.
- Swim/land on command: water >= 0.15 and land >= 0.15 certified on arch51 (9/139 each, still 0)
  (experiments/no_model_gate/results_arch51_table.md).
- Driven crossings (still 0, base 0): air_to_water arch56 #72/#77, arch49 #116; water_to_land arch49
  #146/#110; all underpowered at n=1-2.
- Three media >= 0.012 on one body, one draw: arch56 #21 (gannet 0.114/0.025/0.016), arch59 #137 (teal
  0.076/0.033/0.024). #137 air 5/8 fresh draws positive; fresh water/land UNMEASURED.
- Take-offs only open loop: from land arch54's 17-rotor gannet lineage (battery exhausted every gate air
  segment); from water arch46 #129 (6 rotors) rose from 2 m deep to 1.0-2.3 m above surface
  (experiments/still_leak/README.md:40-44). No body crossed water_to_air in arch49-59 gates.
- Flapping flight open, not refuted: best margin with real joints 0.42 vs 1.0 needed (R:6277-6286); power
  3.6-28x sufficient, Strouhal in band (derivations/flapping_power_bound.md:282-292, 412-425); N15 no result.
- Buoyancy cannot change during a run: gas_volume/ballast_fraction/BALLAST only in Tier-0; dynamic buoyancy
  fluid.py:1405 on fixed volumes; Mojo kernel has neither. Rotors forward-only (mjcf.py:796-805).
- Battery booking voids readings: joint back-driven by the fluid charged |tau*omega| with no speed clamp;
  still machines exhaust on up to 40% of draws (arch58) (air_turn_bar/README.md:478-495). User's decision.
- Quad ~1.41 kg (inferred, runs/physics_1020.log:76-77), ~290 W in air; mission's 900 s of air needs ~74 Wh,
  its 60 Wh pack gives 51 Wh after 0.85 margin (inferred).
- Hand hover law flew 0 of the 24 arch49 rotor bodies that can lift; why NOT FOUND (N6_result.md).
- Minimal body (inferred): quad-class multirotor 1.5-3 kg, near-neutral fixed buoyancy, legs, >= 95 Wh.
  Genome can build it; controller's basis cannot steer it (0.0003/unit). R1 ends, R2 begins here.

## A10 (literature: learning) — returned sixth
- Budget gap: project gives each body ~7e1-2e3 steps; pooled ~1e5-1e6 per skill. Cheapest single-skill
  successes: 3e5 (quad hover), 5e5-1e6 (on a hand-made stabiliser). Typical single-body skill: 3e7
  (flapping wing in MuJoCo, Cai 2024), 1e8 (Swift), 1.5e8 (Rudin), 3e8 (salamander land+water).
  => 2-4 orders below cheapest, 4-6 below typical. Three more runs add ~0.5 order: R7 ruled out.
- Chains of rare sub-goals: strongest evidence = start episodes near the goal from saved simulator
  states and move the start back (Backward Algorithm, Backplay, reverse curriculum, DeepMimic RSI:
  backflip 0.379 -> 0.791; Go-Explore: PPO no reward in 1e9 frames vs 198/200). Need reset + a
  source of intermediate states. Exploration bonuses alone failed on similar tasks.
- Multi-modal robots: learned mode switching only on bodies already competent in each mode (iRonCub
  walk-to-fly sim only; wheeled-legged on hardware; salamander land->water one sim demo). M4/LEONARDO
  switch by model-based control. No learned controller crossing air+water+land on one body found.
- One shared policy over EVOLVING bodies: no success found. Working cross-body controllers use fixed
  body sets, body-aware networks, >= 1e6-1e8 steps per body (MetaMorph 1e8/100 bodies; URMA 1e8/robot;
  ~1,000-body work: ~2e9-step expert per body then distil). SMP's plain multi-task baseline fails.
  Selection undervalues new bodies (Cheney 2018; Mertan & Cheney 2025).
- Co-design budgets: DERL fresh controller + 5e6 steps per child, ~4,000 bodies; Evolution Gym 5.12e5
  per body and still fails hardest tasks. Project 2.5-5 orders below.
- Noisy QD: with one eval per solution < 20% of elites sit in the correct cell; Extract-ME spends 25%
  of each batch re-evaluating; Deep-Grid keeps 50 per cell. FLAG: project retention 2-6% is far worse
  than any benchmark figure => either draw noise is much larger, or the metric is an artefact of the
  quantile scalar/gates (separates R4 from R6).
- Sampling MPC works in MuJoCo (10 rollouts, 1-20 ms/update) but the fluid model must run inside the
  planner's rollouts. Per-skill experts distilled into one policy is the recipe behind the strongest
  multi-skill controllers; distillation can lose > 40% (DCG-ME) => keep a hierarchical fallback.
- Its recommended plan: Step 0 offline MPC existence proof per skill and crossing from boundary states;
  Run 1 per-skill experts on one fixed body 1e7-1e8 steps/skill, crossings from saved boundary states
  moved back; Run 2 distil + chain fine-tune; Run 3 body search around the working one.

## A1 (mission and crossing mechanics) — returned seventh
- "Mission" is FOUR different numbers; the search scores only Tier-1 mission_fraction, which needs no
  crossing (crossing and take-off factors floored at 0.05 each, evaluate.py:217-223). The continuous
  mission is used only by the film and offline controller training; the search never sees it
  (mission.py:146-151). Tier-2 re-places every leg, scores no crossings, and since arch51 is only a
  flag (evaluate.py:590-639, curator.py:411-426).
- Since the paired air score (2026-10-10) Tier-1 mission is 0 because AIR is 0: arch51 4 of 10,388
  evaluations had air > 0, none had all three media > 0. Air credit needs the body aloft after 6 s from
  a 30 m launch, turning toward a commanded side (triphibian.py:2160-2203).
- arch52-60 never asked for the mission: --islands air, mission_weight 0, curriculum off; mission_best
  reads only the generalist archive (loop.py:2412-2418) which those runs lack. Stored mission rounded to
  4 dp (loop.py:1249); two floors multiply to 0.0025, so a body at ~0.08 in all three media reads 0.0000.
  arch56/59/60 had 21/5/48 such bodies. "Exactly 0" cannot be tested below 5e-5.
- The continuous mission has NOT RUN since arch54: film skips it without a GL context
  (showcase.py:252-253); arch54/56/58/59 manifests say "no GL context", "0/0" with no legs = not measured.
  When it ran: order land -> air -> water, 8 s legs, seed 0 (film.py:438-440); arch47 1/2; arch48/49/51 0/2.
- Each crossing = 6 s probe: 1.5 s hold then 4.5 s to cross, ONE fixed start state per kind for every
  candidate and run (transitions.py:223-235, 254, 597). air_to_water must hold energy height within 1.0 m
  for 1.5 s at 4 m: 0/218 elites passed. water_to_air and land_to_air must be > 0.5 m clear at t=6 s.
  water_to_land must end on dry ground from 3-7.5 m out (>= 0.5-1.7 m/s); population water p90 0.446 m/s.
  INFERRED: air phases of both air/water crossings run through the WATER control basis (transitions.py:615).
- Crossings rare not zero in noise-free scores: arch47 2/9,439; arch51 36/10,388 (all rotor bodies with
  air 0: open-loop rotor leak, inferred); arch56 6/3,549; arch59 4/3,761.
- Nothing turns a partial crossing into gradient: PPO crossing reward boolean (batchroll.py:1259-1264);
  shaping return collapses to a placement constant (ppo.py:499-504); specialist islands boolean factor
  floored 0.15 (islands.py:272-285); graded approach moves mission only above mean 0.125 while its p99
  <= 0.094; distance curriculum only makes crossings HARDER and was off in arch51-60 (curriculum.py:736-757).
- E2 adopted 2026-10-12 (R:5529-5543): L1 air not met, water/land miss retention; L3 = Tier-1 mission > 0
  AND film 2/2; no code runs the continuous mission on held-out seeds.

## A4 (ROADMAP 386-3455 ledger) — returned eighth
- Every mission film reads zero crossings: arch34/35/36/37 0/2 (R:710, 870, 1118, 1467), arch40 0/2
  (R:2747), arch44 0/2 (R:3013), arch51 three_media 0 (R:627). After arch51 nothing measures the mission.
- Every explanation moved a SEGMENT-level number while the mission stayed 0/2: take-off ladder (clears
  0.9->3.3%), land_air island (+0.0025, R:1451), lift rungs (27.4->58.0%), depth gain (4.0->30.7%).
- Only direct continuous-mission probe: arch41 top 30, 90 continuous missions, 0 completed both
  transitions, 25 completed one, mostly air->water by falling (R:2786-2791). Y/O curriculum ON since
  10-05 with placeholder numbers, "does not yet apply pressure"; land_to_air ignores it (R:3408, 3414-3415).
- Already failed/withdrawn (do not re-propose): thrust rungs (7.7->1.8%, R:2138), flap_frequency
  searchability, actuator-model rebuild, 24 s window (void twice), mass cap/battery, contact-fraction
  gate, grounded spawn, passive twin, GRPO groups of 8 and log_std -2.0 (R:426-443), scoring-only fixes
  ("Neither is a scoring question any more", R:495-497).
- User constraints: the mission stays ONE triphibian machine, not a chain of pairs (R:3029-3030);
  forward distance carries 0.6 of every medium; sinking must be chosen; active = max(headway,
  depth_station_keeping); rotorcraft allowed; rank cost, then loop speed, then learning; bars referenced
  to the null (R:516-521).
- Lessons counted by the text: a score paid for absent/uncontrolled motion (many instances); "the
  ceiling was the measurement, not the machine" (R:2026); selection pressure "shared and finite" (R:2190).

## A9 (literature: hardware) — returned ninth
- Every verified air+water+land robot is a 1.1-4.8 kg ROTOR platform with passive wheels or a rolling
  hull (TriphiBot 1.1 kg, CUHK triphibian 4.76 kg, Wukong-Omni 4.4 kg). None walks; none reports
  transition success rates; all use hand-designed mode logic with PID/MPC, none learned.
  M4: air+land only. Nezha, Mirs-X/TJ-FlyingFish: air+water only. No flier found crossing water->land
  on the ground.
- Water exit, four working methods: impulse (T/W 27 for 0.2 s .. 19,000+ for 1 ms, RoboBee); powered
  breach from depth (surface at 0.5-2.7 m/s, pitched 70-90 deg, rides 0.2-0.6 s of near-zero thrust);
  take-off from floating (air T/W 1.41-1.56); staggered propellers (Naviator).
  Submerged swimmers sit within -8% .. +3% of neutral buoyancy. Auks flap 3.1-3.6x faster in air than water.
- Flapping-only triphibian: nothing flies and swims by flapping above ~0.3 kg (MIT/EPFL FAAV ~250 g,
  Science 2026; has not closed air-water-air, cannot turn, timed open-loop). No flapping robot that
  flies, swims AND walks at any scale. Birds using the same wing in both media "are flightless above
  ~1 kg" (Elliott et al., PNAS 2013); murre flight cost 146 W/kg, highest in any vertebrate.
- Servos: ~20-60 W/kg, at most ~1.6 Hz at +-30 deg (agent's estimate from spec sheets); every working
  flapping flier uses a geared motor and crank. Solver extrapolates above reduced frequency k = 0.3;
  bird-scale underwater flapping k ~ 0.3-1.6.
- Control: every hardware transition = hand-designed mode switch with explicit medium sensing. Learned
  crossing control only in simulation on top of a classical controller, did not beat it on the easy case
  (Grando et al. 2021).
- Missing from the solver: thrust collapse as propellers cross the surface, water-surface ground
  effect, entry slamming into the dynamics, passive wing bending; jet thrust does no pumping work (J-01).
- Anchor family per literature: near-neutrally buoyant multirotor with wheels/rolling hull, air T/W 1.5-2.5.

## Lead's spot checks (2026-11-10, by grep/sed at a7770c2) — all six confirmed
loop.py:207 mission_weight 0.0 | batchroll.py:1254-1264 crossing reward = float(crossed) x quality |
triphibian.py:1704-1711 task channels zero outside a segment | cpg.py:427-430 gain scales amplitude,
triphibian.py:734-739 rotor amplitude 0 ("the policy moves it through the offset") |
showcase.py:252-253 no GL -> empty ContinuousResult | mission.py:146-151 "The search never saw it".

## E0 launched (existence/PREREG.md frozen first): hand cascade on quad-class body through the
unmodified scores: segments, four crossings, continuous mission, held-still arm. No learning.

## A8 (raw-data forensics, arch43-60) — report complete on disk (agent cut by the usage limit while polishing)
- X1 (arch47-60): 108 crossings in 57,986 Tier-1 evaluations (0.186%), 0 evaluations with >= 2;
  by kind air_to_water 23, water_to_air 6, water_to_land 33, land_to_air 46.
- air_to_water: in the water before the command 77.6%; never crossed 13.5%; aborted 8.8%.
- Tier-1 mission_fraction > 0: 23/57,986 (0.040%), max 0.0003.
- Tier-2: 1,627 promotions arch43-60; 1,583 failed leg 0, 40 failed leg 1, 4 battery; >= 2 legs passed: 0.
- Films: 0/2 in arch44-46, 48, 49, 51; 1/2 arch47; arch53-60 "no GL context" (not measured).
- Co-occurrence arch51-59: evaluations >= 2 media 1,126/36,095 (3.12%), all three 6/36,095 (0.017%);
  final archives >= 2 media 42/1,446 (2.9%), all three 2/1,446 (0.14%).
- Learner arch51-60: terminal share of return magnitude air 0-5.4%, crossings 0-0.8%; updates with any
  nonzero crossing reward at most 43/900 (arch51). 84/92 (run, tag) reward means move < 1 sd first->last fifth.
- Scale: 89,464 Tier-1 evaluations + 12,852 re-evaluations over 18 runs, 328 process-hours.
  Learner transitions per run 2.8e6-2.3e7.

## 2026-11-11 02:12 resumed after the usage limit. E0 agent resumed (nothing was on disk). Canary
test_ppo and test_physics started in the background (memory now 4.9 GB available).

## 2026-11-11 02:3x
- Canary test_ppo.py: "shared PPO checks passed", 0 [fail], 0 [skip] (results/canary_test_ppo.log).
- DIAGNOSIS.md rewritten in full (verdict, innocents, PREREG outcomes, literature, plan P0 + three runs,
  user decisions). E0 running (agent resumed). Fresh-context reviewer launched -> reports/R1_review.md.
- Canary test_physics.py: "all physics checks passed", 0 [fail], 0 [skip] (results/canary_test_physics.log).
- Canary test_search.py (GPU, ~20 min): "all search-machinery checks passed" (results/canary_test_search.log).
  Full canary green at a7770c2 on 2026-11-11: test_ppo, test_physics, test_search, 8 cheap suites.

## E0 RESULT (existence/RESULT.md), 2026-11-11 — "It can" stands
- B1 (quad + 200 Wh Li-S, hull 0.40 x 0.059, four fixed FOOT legs, token PADDLE fin; 2.045 kg, density
  ratio 1.081) and B0 (reference quad as built, 1.455 kg, ratio 1.91) under ONE hand-written feedback law,
  through unmodified run_segment / run_transition / run_continuous, seeds 0-7, 320 rollouts, 0 bad_qacc:
  air 8/8 (B1 0.891-0.927); water 8/8 >= 0.15 (B1 0.211-0.717, B0 0.401-0.853); land 8/8 (B1 0.711-0.761,
  a taxi 3 cm CLEAR of the beach; skid X2 0.391-0.871 at 94% thrust); all four crossings crossed (hold
  0.87-1.0); continuous mission 2/2 from each start, continuous_score 0.70-0.79.
- Held-still arm: media <= 1.35e-17, crossings 0. BUT continuous_score pays a still machine 0.148 /
  0.34-0.35 / 0.48-0.58 (land/water/air start) and credits 1/2 transitions from the air start (it falls in).
- Crossing probes and run_continuous are seed-independent (n = 1). X1 with 8 entry states: B1 32/32, B0 29/32.
- Tier-0 `has_propulsor` (core/phenotype.py:811-819) does not count rotors: the bare quad is rejected
  (gate_margin -1.00 vs -0.85). A bare multirotor cannot enter a search, seeded or not.
- Tier-1 mission_fraction of the hand-flown B1: 0.011-0.091, because the take-off factor is read from the
  LAND segment's takeoff_height (0.05-0.24); without it 0.21-0.55.
- Teacher reads what the policy cannot: world x,y; height over ground/sea; world-frame velocity; full
  attitude; per-rotor wetness; task phase and clock; the model (inertia, rotor geometry, bemt); the beach map.
- Quad props are handed by side: mixer rank 3, no yaw authority. Leaving the water is the fragile manoeuvre
  (2 bad_qacc + 1 divergence in tuning, unstaged exits). Heavier is better in water for a tilt-thrust body.
- run_transition never reads bad_qacc on the single-machine path. Energy: 45-min spec 0.87 (B1), 0.40 (B0).
- LEAD'S INDEPENDENT RE-RUN: B1 seed 5, hand + still arms, 20 rollouts, 38 s for the 10 hand rollouts:
  0 mismatches digit-for-digit against results_final.jsonl (results_leadverify.jsonl).
  Throughput: ~144 s simulated in 38 s wall, one process (~95 decisions/s incl. start-up).
- Lead's grep: part hinges are range-limited (mjcf.py:649-655), so a free-rolling passive wheel is not
  expressible; rotor joints are unlimited (mjcf.py:748-751).

## 2026-11-11 (later) — review R1 returned; what the lead re-checked at source

`reports/R1_review.md` (249 lines): 15 claims of sections 1-3 checked, 10
confirmed, 3 overstated, 2 wrong; ten plan findings P1-P10. Re-checked by the
lead before changing DIAGNOSIS.md:

- `triphibian.py:1641,1666`: the observation carries `tanh(depth/5)`, signed
  (negative in air), so height over the sea IS observed. "No observation
  carries distance to a boundary" was wrong; missing are shore distance and
  bearing, heading, height over ground. Docstring :1604-1608 records the choice
  "No global position, no ground-truth world velocity".
- `transitions.py:493-498`: the note "did not hold before the command to
  cross" is written iff the body reached the target AND hold < 0.5. So 77.6%
  is "reached the water with a failed hold", not "in the water before the
  command".
- `transitions.py:713-726`: graded approach exists for AIR and LAND targets
  only; an uncrossed `air_to_water` gets none. `:432`: land hold counts
  `medium_of(env, GROUND_TOL)`.
- Recount from `results/events_arch47..60.json` (lead's own script): 57,986
  evals, 108 crossings = land_to_air 46, water_to_land 33, air_to_water 23,
  water_to_air 6. Rotor-less crossers 18 (land_to_air 12: eel 8, teal 3,
  gannet 1; air_to_water 5 gannets; water_to_air 1 eel). All 52 into-air
  crossers have air < 0.012, 51 at exactly 0. PPO transitions per run: only
  arch51 exceeds 2e7 (2.292e7); arch57 1.892e7. Updates with any crossing
  reward: arch51 43/900, arch54 25/300.
- `air_turn_bar/gate_arch56.json.rows.jsonl`: elites 72 and 77 (gannets) have
  `crossed:air_to_water` in the elite arm, not in still or base.
- ROADMAP:3980-4001 (N6 "neither": 153/191 cannot lift, hand law 0 of 24,
  "not the binding gate today"); :419-424 ("The gap to L1 is a learner gap");
  :2784-2792 (0 of 90 continuous missions, 25 one transition, "would zero every
  leg"); :5527-5545 (E2: L3 = one design, both transitions, film reproduced,
  >= 3 of 5 held-out seeds); :2758-2761 (selection changes behind a flag,
  default off); :3026-3031 (one triphibian machine; the search keeps building
  rotorcraft); :124-134 (coverage 90% by gen 40-160; best per medium by 250 in
  21 of 27); :389-394 (rotor hover 8-15 kW against a 5.25 kW limit);
  :4632-4637 (offline reads not comparable across the 2026-10-12 basis fix).
- `triphibian.py:569-573` spawns: AIR (-40, 0, 30), WATER (-8, 0, -4), LAND
  (15, 0, 0.9); `:88` `seconds_per_domain` 300; the film uses 8 s legs.
  Waterline about x = 7.8 (E0: ashore at 9.19 is 1.4 m past it). So water
  spawn to shore is about 16 m and land spawn to water about 7 m, in 8 s.
- mission_fraction rounding: 0.08^1.5 x 0.0025 = 5.66e-5 -> stored 0.0001;
  it reads 0.0000 below 0.0737 in all three media. The 0.08 example was wrong.
- Gate arithmetic (lead): >= 7 of 8 on three starts passes with p = 0.54 at a
  true rate 0.9. K = 32 per start with >= 25 rejects a true 0.6 at 5% and
  passes a true 0.9 with 0.988 per start, 0.965 for three; 0 of 32 bounds a
  null rate below 0.089.

Acted on: E0n (command-ignoring arms) and E0m (hand law on 32 mutated
anchors) pre-registered in `existence/PREREG_nulls_mutants.md` and handed to
the E0 agent (resumed, id ade3bcec1817431e2). No training.

## 2026-11-11 (later) — E0n and E0m returned; lead's verification

`existence/RESULT_nulls_mutants.md` (agent ade3bcec, resumed). 236 + 528
rollouts, no training, `triquad.py` sha256 unchanged (1cc5f4bd...ee67).

- Lead read `nulls.py:60-234` (the three arms). base = `Controller(params=
  env.cpg.base, policy=None, bases=None)`; blind = `HandPolicy` subclass that
  hands the law the first medium seen; shuffled = a view of the env whose
  `_phase_now` / `_active_task` answer from seed (s+1)%8's first-half task in
  both halves. Identity tests 1-4 pass (`logs/nulls_ident.log`).
- Lead re-ran, different rows from the agent's own re-check: B1 base
  water_to_land entry 2 (score 1.0, crossed), B1 shuffled water seed 3
  (0.3177236863074825), B0 blind air_to_water entry 4 (0.0, not crossed):
  every printed field equals the stored row. Mutant M02: 17 rows, 0 rows with
  any difference (all fields but `wall_s`). Logs `existence/logs/leadverify2_*`.
- E0n: blind 0 of 64 crossings, approach 0.000. Shuffled: air 0.0 16/16, land
  < 1e-16 16/16, water 9 zeros, 7 >= 0.012, 4 >= 0.15, max 0.383 (the hold
  pays when the fed depth is near the commanded one; progress cancels to the
  sign bit). Base: 2 crossings credited in 35 measured rows (B0 water_to_air
  entry 1; B1 water_to_land entry 2; both tumbling), 4 of 64 as
  `run_transition` counts (no bad_qacc read); 49 of 118 base rollouts failed
  (bad_qacc 34, diverged 13, flat pack 2). `continuous_score` pays every arm
  0.11-0.49. All five frozen predictions N1-N5 held.
- E0m: 31 of 32 pass Tier-0 (M03 loses the fin's surface). Of 29 changed,
  Tier-0-passing children: mission from all three starts 15 (0.517, Wilson
  0.34-0.69), all four crossings 12, both 9. Lost by kind: water_to_land 16,
  water_to_air 8, air_to_water 7, land_to_air 6. Outside density 1.02-1.10:
  11 of 14 lose a crossing; inside 6 of 15. Rotor count changed (M09 8, M18
  3, M25 5): all lost. Law not applicable: M05, M07. Still: 0 of 124.
  Frozen band 0.3-0.7. M2's "all four crossings on at least half" and
  "water_to_air goes first" are refuted.

## 2026-11-11 (later) — decisions answered, E0b

- The user answered the six decisions (DIAGNOSIS §7a, quoted there). Done is D1-D5 of §7a, written by the lead on the user's delegation.
- E0b frozen in `existence/PREREG_land.md` (sha256 3d5e5598...), run by the E0 agent in about 20 min, 280 rows, 0 failed. Predictions 1, 2, 3, 5 held, 4 refuted (walker paid more than the taxi).
- Lead verification: `existence/logs/leadverify3_{walk5,frozen2,mis}.jsonl`, 8 rows, every numeric field equal to `results_land.jsonl` except `wall_s`. First attempt failed: `landgait.py --out` resolves against the script's directory; pass an absolute path.
- Found: the land command speed is 0.04 m/s (`dytiscidae/envs/tasks.py:63`), so a land score of 0.94 is a crawl (measured_land_speed 0.0087 on seed 5). Goes to P0.11.
- Not run: L1 entry-state rows, L1 crossing nulls (P0.9).

## 2026-11-11 (later still) — P0.2 first part

- Committed the record as b652788 on branch `triphibian-diagnosis` (trajectories left out, regenerable).
- `scripts/crossing_telemetry.py` -> `results/crossing_telemetry.jsonl` (gates off) and `_gates_on.jsonl`. Row-for-row: hand arms identical; base B1 water_to_land entry 2 and B0 water_to_air entry 1 lose the crossing; base holds go to 0 on 25 more rows that had not crossed.
- Dwell gate dropped on evidence (see ROADMAP). L1 walk loses water_to_land from 3 of 8 entry states with or without gates: a P0.9 item.
- `tools/mutate.py --only` takes one id per invocation (the last wins).

## 2026-11-11 — P0.3 mission verdict

- `scripts/mission_eval.py` -> `results/mission_eval.jsonl` (1,152 rows, legs 7-13 s) and `mission_eval_wide.jsonl` (288 rows, L1, legs 7-19 s).
- Clock arms (timed, otherdraw) complete 8-17 of 32: my estimate before measuring was under 0.3. D3 amended in DIAGNOSIS §7a.
- B1 from air: 25 completed, 4 failed (diverged 2, unstable 2), 3 missed the land leg. L1 has none of these.
- Launcher mistake: `( cmd & )` inside a run_in_background shell detaches, so the task "completed" at once; use plain `cmd &` then `wait`.

## 2026-11-11 — E0s (P0.5/P0.6 prototype)

- Agent commit 88359d7. Lead verification: `existence/logs/leadverify4_{m,s,c}.jsonl`, 25 rows (18 missions incl. zero and frozen arms, 3 segments, 4 gated crossings), 0 differ from `results_stateless.jsonl`; `stateless.py tests` re-run, all PASS.
- Read at source: `Teacher.teach` takes only `o`; `Inner.act` reads attitude, rates, world velocity, hub wetness, submerged fraction, mass matrix; positions only as differences on the body. The harness (`Split`) holds the previous memory, the held `u` and the command clock.
- Open: air_to_water holds at entries 3 and 4 (0.406, 0.371); observation velocity frame; 3x cruise over-ask.

## 2026-11-11 — port of E0s (5045265)

- Lead check, own script, 27 missions through `teacher_controller(env).installed(env)`: first attempt forgot to install (controller None runs the base gait, silently); second: 10 of 27 differ from the prototype rows (energy at 1e-12; nomem verdicts 2); third, with the prototype's preamble (`launch_speed`, `level_margin()`, `flies_level()`, `thrust_margin()` before installing): 8 of 8 re-run rows identical. So the port is exact and the environment carries history across `reset`.
- To trace: what `reset` leaves (candidates: fluid solver's published mass/inertia, rotor state, cached facts).
