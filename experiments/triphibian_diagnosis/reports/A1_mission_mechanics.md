# A1 — What the "mission" and each "crossing" are in code, and what a nonzero score needs

Investigator A1, 2026-11-10, repo at `a7770c2` (main). Read-only. Nothing was simulated:
no MuJoCo, torch, PPO or search was imported or run. Code is cited as `path:line`
(paths relative to the repo root; `dytiscidae/` is the package). Stored data was read by
streaming `runs/<run>/events.jsonl` (`evaluate` and `ppo` lines only) and
`runs/<run>/media/film_manifest.json`; the crossing-failure note counts come from
`experiments/triphibian_diagnosis/results/vocab.json` (another investigator's extraction
of the same events). Geometry numbers were computed from the formulas in
`dytiscidae/core/mjcf.py:86-113` with plain numpy.

Tags: **MEASURED** = read from code or stored data; **INFERRED** = derived by me from code
plus geometry/arithmetic, not run; **UNMEASURED** / **NOT FOUND** where nothing exists.

---

## 0. Summary for the lead

1. **"Mission" names four different quantities. Only one of them is scored by the search, and it does not need a crossing.**
   Tier-1 `mission_fraction` = `min(c)^0.5 · mean(c) · energy · max(transition, 0.05) · max(takeoff, 0.05)`
   (`envs/evaluate.py:217-223`). It is nonzero whenever all three per-medium competences
   are nonzero. Crossings and take-off only scale it, and each has a 0.05 floor. The continuous
   mission (`envs/mission.py`) is a different quantity that the search never sees
   (`envs/mission.py:146-151`). MEASURED.
2. **Since the paired air score (2026-10-10), Tier-1 mission = 0 because air competence = 0.**
   In arch51, 4 of 10,388 evaluations had air > 0 (max 0.002) and none had all three media
   > 0, so mission was exactly 0 in every evaluation. Air credit needs the body to be aloft
   between 6 and 8 s after a 30 m launch *and* to turn toward a commanded side
   (`envs/triphibian.py:2160-2203`, `:3042-3072`). MEASURED.
3. **arch52-arch60 never asked for the mission, and the stored "0" is partly rounding.**
   All of them ran `--islands air`, `--mission-weight 0.0` and `--no-distance-curriculum`
   (`runs/arch5x_notes.md`, `runs/arch60_notes.md:51`). `mission_best` reads only the
   generalist archive (`evolution/loop.py:2412-2418`), and that archive does not exist in
   those runs. The stored mission is rounded to 4 dp (`loop.py:1249`). With both 0.05 floors,
   three competences of about 0.08 give 5.7e-5, so most values are written as 0.0000.
   arch56, arch59 and arch60 had 21, 5 and 48 evaluations with all three media > 0, and all
   but one of them were stored as mission 0.0000. MEASURED.
4. **The continuous mission is a film-only check, and it has not been measured since arch54.**
   It runs inside `render_mission`, which returns an empty result when there is no GL
   context (`viz/showcase.py:252-253`). The manifests for arch54, 56, 58 and 59 say
   `film_error: "no GL context"` with transitions "0/0" and no legs. That "0" means not
   measured. When it did run, the film's seed was fixed at 0, so the schedule was always
   LAND → AIR → WATER with 8 s legs (`viz/film.py:438-440`; `default_rng(0).integers(3)` = 2).
   Results: arch47 1/2, arch48 0/2, arch49 0/2, arch51 0/2. MEASURED.
5. **Every crossing needs a powered capability that the population does not have.**
   Each crossing is a 6 s probe: 1.5 s of commanded hold, then 4.5 s of go
   (`envs/transitions.py:254, 597`). It starts from one fixed entry state per kind for every
   candidate in every run (`transitions.py:223-235`). What each kind needs:
   - `air_to_water`: hold energy height within 1.0 m for 1.5 s at 4 m altitude, which is
     powered level flight. 0 of 218 arch46 elites pass this hold (transition_distance §13).
   - `water_to_air` and `land_to_air`: be more than 0.5 m clear at t = 6 s.
   - `water_to_land`: end on dry ground from 3-7.5 m out. That is ≥ 0.5-1.7 m/s, against a
     population water p90 of 0.446 m/s (arch40).

   MEASURED code; speeds INFERRED.
6. **Crossings are rare, but not literally zero.** In the noise-free evaluations a single
   crossing kind was crossed in 2 of 9,439 evaluations (arch47), 36 of 10,388 (arch51), 6 of
   3,549 (arch56) and 4 of 3,761 (arch59). None crossed two kinds. All 36 in arch51 were
   rotor bodies with air = 0, which fits the open-loop rotor leak documented in
   `docs/ARCH46_SPEC.md:404-414` (INFERRED). The PPO sampling pass also scored completed
   crossings in 46 of arch51's 900 generations. MEASURED.
7. **Nothing turns a partial crossing into gradient.**
   - The PPO crossing reward is boolean `crossed × quality` (`envs/batchroll.py:1259-1264`).
   - The shaping term telescopes to a constant set by the placement (`learning/ppo.py:499-504`).
   - The specialist islands use a boolean crossing factor floored at 0.15 (`evolution/islands.py:272-285`).
   - The graded "approach" moves Tier-1 mission only when its mean exceeds 0.125. Its p99 is
     ≤ 0.094 in every run checked.
   - The distance curriculum can only make crossings harder, and it was off in arch51-60.

   MEASURED code and data; the policy-gradient claim is INFERRED from the math.

What this means for the lead's framing question: the evidence supports four layers at
once, not one of "definition / start state / capability".
- **Definition:** the paired air score; the gates in each crossing.
- **Start state:** one fixed entry draw per crossing; no start easier than the interface;
  a curriculum that only hardens.
- **Capability:** powered flight, hover take-off, and travel at about 1 m/s, none of which
  is in the population.
- **Instrument:** the floors and rounding; the film's mission not run; the lenient
  domain classifier in the continuous mission.

---

## 1. World: geometry, spawns, launch, placement draws (Q1)

### 1.1 Geometry (MEASURED from code; numbers computed from those formulas)
- **Water.** `z = 0` is the still waterline (`core/mjcf.py:126-127`).
  - Tier-1 is calm water: `SeaState.amplitude` defaults to 0.0 (`physics/medium.py:57`),
    and Tier-1 passes `sea_state=None` (`envs/evaluate.py:345`, `envs/batchroll.py:1077`).
  - Tier-2 draws waves with amplitude U(0, 0.25) m and period U(1.6, 3.2) s, a current
    N(0, 0.15)·(1, 1, 0.2) m/s and wind N(0, 1.2)·(1, 1, 0.3) m/s (`evaluate.py:564-571`).
- **Seabed.** A plane at z = −18 m (`mjcf.py:83, 233-238`). Its 60×60 size is the render
  extent. The claim that a MuJoCo plane collides without limit comes from MuJoCo semantics
  and is not checked here.
- **Beach.** One box: 40 m long, half-thickness 0.5 m, y ∈ ±30 m, slope 0.12 (6.84°), with
  12 m of its length below the waterline (`mjcf.py:74-82, 250-275`). The top surface is
  `z(x) = (x − 12)·0.12 + 0.5036` and exists only for x ∈ [0.085, 39.80] m
  (`mjcf.py:86-113`). Computed:
  - z = −0.93 m at the seaward end;
  - it crosses the waterline at **x = 7.80 m**;
  - z = 0.86 m at x = 15;
  - z = 3.84 m at the inland end.

  Beyond either end, "ground" is the water surface (`envs/triphibian.py:1448-1451`). The
  world is open sea with one emergent ramp. Two constants share a name: `transitions.SHORE_X
  = 8.0` (`envs/transitions.py:248`) and `mjcf.SHORE_X = 12.0` (`mjcf.py:74`, where the ramp's
  centre-line crosses z = 0).
- **Domain tests differ by evaluator.**
  - Tier-1 segments: airborne means clearance > 0.3 m (lowest geometry above
    max(water, beach)) and no world contact (`triphibian.py:2353`). Submerged means root
    depth > 0.2 m (`:2696`).
  - Crossings use `medium_of`: WATER if root depth > 0; LAND if touching the world or within
    a tolerance; otherwise AIR (`transitions.py:273-280`). They also use `ashore`
    (`:287-305`).
  - The continuous mission uses `current_domain`: WATER if mean submerged fraction > 0.34;
    LAND if any world contact; otherwise AIR (`envs/mission.py:156-164`). Self-collision is
    off, so a contact means a world contact (`mjcf.py:168-194`).

### 1.2 Spawns for segments (MEASURED)
`TriphibianEnv.SPAWN` (`triphibian.py:569-573`):
- **AIR** (−40, 0, 30).
- **WATER** (−8, 0, −4): 4 m deep over 18 m of water, 8 m seaward of the ramp's end.
- **LAND** (15, 0, ·): z is set so the lowest geometry sits 0.05 m above the ramp
  (`:772-773`, `:1319-1338`). This is 7.2 m inland of the waterline.

`reset(randomise=True)` adds N(0, 0.4) m to x and y and N(0, 0.2) m to z (`:765-768`).

**Air launch** (`:774-780`):
- The body's +x points along world +x, at `launch_speed` and nose-up `launch_pitch`.
- `launch_speed` comes from bisecting for the lowest speed in [6, 30] m/s at which some
  pitch in −4°..44° carries the weight (`:1243, 1281-1315`). The launch is at 1.2× that
  speed (`:1203, 1316`), at the lowest pitch that carries the weight there.
- A body that cannot carry its weight even at 30 m/s is released at 6 m/s
  (`:1290-1302`). A rotor body with `rotor_lift_ratio ≥ 1` is released level at 6 m/s
  (`:1187-1192`).
- Launch height is 30 m unless `MissionSpec.air_launch_height` is set (`:763-764`). Only the
  distance curriculum would set it (§5), and it never did.
- Free fall from 30 m takes 2.47 s; from 4 m, 0.90 s (computed).

**`scatter`** runs after every segment reset (`:791-885`) and applies:
- an attitude kick about a random axis, N(0, 0.18 rad);
- a velocity scaled by (1 + N(0, 0.15)) plus N(0, 0.35·min(|v|, 1)) per axis;
- an angular velocity of N(0, 0.25) rad/s;
- a battery at U(0.4, 1.0) of capacity;
- a CPG phase of U(0, 2π), with the joints posed at that phase.

### 1.3 Crossing placements (MEASURED)
Every crossing calls `reset(start, randomise=False)`, then `scatter` with a **fixed seed
per kind**, `transition_scatter_seed(kind)` (`transitions.py:223-235`), then
`reseat_after_scatter` (`:565-589`). Both paths do this: `run_transition` (`:607-612`) and
`run_transition_batch` (`batchroll.py:1313-1320`).

| kind | start (`_place_for`, `transitions.py:502-562`) |
|---|---|
| `air_to_water` | x = −200, z = 4.0 m + back, at the air launch speed and pitch (`:515-526`) |
| `water_to_air` | (−8, 0, −2 − back): root 2 m deep, at rest (`:527-528`) |
| `water_to_land` | root 0.15 m under, at the most shoreward x ≤ 6.0 where the geometry clears the ramp by 0.05 m, stepping 0.5 m seaward up to 12 m; then x − back (`:529-555`). "3–7.5 m from the shore, depending on the body" (`docs/ARCH46_SPEC.md:274-276`) |
| `land_to_air` | the land spawn; `back` ignored; reseated to a 0.02 m gap (`:581-583`) |
| `land_to_water` | x = 14 + back (`:556-561`). Defined but **never run in Tier-1** |
| `air_to_land` | endpoints only (`:122`); no placement branch. **Never run** (NOT FOUND in any caller; `rg` over `dytiscidae/`) |

### 1.4 What "placement draws" are (MEASURED)
- A **draw** is one evaluation seed. Each candidate has its own draw (`draw_per_candidate`,
  `evolution/loop.py:172, 717-720`). The seed sets, per medium, the scatter seed
  (`evaluate.py:252-260`) and the task seed (`envs/tasks.py:295-301`). The task draw is:
  - water: phase order, heading U(−π, π), hold depth U(5.0, 6.5) m;
  - air: turn U(45°, 90°) with a random sign;
  - land: phase order and heading.

  It also seeds identification and the env stream.
- **`SearchConfig.placement_draws`** (`loop.py:165-168`, default 1) scores each candidate
  at that many seeds and places the median draw by summed competence (`loop.py:970-983`).
  arch59 ran 4 (`runs/arch59_notes.md:2-4`).
- The device evaluator (`evaluator="device"`, `k_mean` 50 + `k_sampled` 10; `loop.py:179-191`)
  draws K seeds for air and water only (arch60).
- **No draw reaches the crossings.** Their entry state is fixed per kind (§1.3). INFERRED
  consequence: a crossing score cannot vary with the evaluation seed, except through the
  identified basis and through PPO sampling noise.

---

## 2. One Tier-1 evaluation (Q2)

### 2.1 What runs (MEASURED)
Every candidate runs the same evaluation on every island (`evaluate_tier1_batch`,
`batchroll.py:1007-1281`; the single path is `evaluate.py:307-431`). There is no island
argument. In order:
- mobility identification in air and water, when asked (`batchroll.py:1107-1138`);
- three segments, AIR, WATER, LAND (`triphibian.py:72`; `batchroll.py:1163-1229`);
- four crossings: `air_to_water`, `water_to_air`, `water_to_land`, `land_to_air`
  (`batchroll.py:1238-1264`; `evaluate.py:410-417`).

**Timing:**
- Segment length is `segment_seconds` = 8 s (`SearchConfig`, `loop.py:148`; every run
  arch51-60 passed `--segment-seconds 8`).
- `dt` = 0.004 s (`triphibian.py:613`), so a half-segment is 2,000 physics steps. Control
  runs at 25 Hz, one decision every 10 steps, so 200 decisions per half (`:1797-1798`;
  computed).
- **Every medium is a pair** (`tasks.PAIRED_MEDIA = ("water", "land", "air")`,
  `tasks.py:262`). A pair is two halves from the same initial state, with the env stream
  rewound between them (`evaluate.py:263-304`; `batchroll.py:1181-1216`). That makes 16 s
  and 4,000 steps per medium.
- Each crossing is 6 s = 1,500 steps (hold 375, go 1,125) (`transitions.py:597`;
  `batchroll.py:1285`).
- Total without identification: 48 s of segments plus 24 s of crossings (computed).

### 2.2 Commanded phases and pairing (MEASURED, `envs/tasks.py:189-284`)
Each half has two phases, starting at fractions 0 and 0.5 (4 s each at 8 s).

| medium | phases | pairing |
|---|---|---|
| water | hold at depth d ∈ U(5.0, 6.5) m; cruise at 0.30 m/s on heading U(−π, π) at d; order 50/50 (`tasks.py:73, 84, 201-211`) | antipode: heading + π (`:265-284`) |
| air | cruise on the launch heading (0 = +x) at trim speed; then cruise on a heading turned ±U(45°, 90°) (`:89, 212-219`; speed filled in `triphibian.py:1881-1883`) | mirror: same straight leg, turn to the other side (`tasks.py:234-250`) |
| land | walk at 0.04 m/s on heading U(−π, π); stop; order 50/50 (`:73, 220-228`) | antipode |

**The controller's observation** (`triphibian.py:1602-1678`):
- body twist, gravity direction, depth, wetness, a domain one-hot (always the segment's
  medium, `:1817`), depth error, contact, battery, stroke phase;
- six task channels: phase one-hot, commanded speed, the heading error as a unit vector,
  and height lost in air (`:1687-1725`);
- eight morphology channels.

### 2.3 Per-medium competence (MEASURED, `triphibian.py:2255-2998`)
Every time fraction is divided by the samples the segment *asked* for (`served`,
`:2315-2318`). Phases are scored on the late half of each phase (`:1915-1918, 1961-1969`).
- **Air.** `competence = credit · frac · task · chatter` (`:2678`).
  - `frac` is the airborne share: clearance > 0.3 m and no contact (`:2353-2354`).
  - Less than 0.4 s aloft scores 0 (`:2365-2382`). Less than 2.8 s aloft also scores 0
    (`MEASURABLE_AIR_SECONDS`, `:567`, `:2406-2455`).
  - `credit` is 0.05 if `airworthiness` fails or the spin exceeds 2π rad/s, else 1
    (`:311, 283, 2394-2397`).
  - `task = air_task(turn, height) = turn·(0.5 + 0.5·height)` (`:3042-3050`).
  - `turn` = clip(mean over the mirrored pair of the signed response, 0, 1) × factor
    (`:2160-2195`, `paired_turn` `:3053-3072`). The signed response is
    min(v_b·n, (v_b − v_a)·n) / (v·sin turn), where v_a and v_b are the late-half
    velocities of the two phases and n is the normal toward the commanded side.
  - `factor` = min(served) × min(late-half airborne share), over both phases and both halves.
  - An unpaired or unmeasured first half gives a factor of 0 (`:3069-3070`).
  - `height` = mean over the two phases of `vertical · served`, where `vertical` =
    ((0.55·flight + 0.25·glide)/0.80) × (share of the phase still aloft) (`:2017-2045`).
    `flight` is forced to 0 unless `level_margin ≥ 0.7` (`:2039-2040`, `:1099-1105`, `:327`).
- **Water.** `competence = served · submerged · (0.5 + 0.5·upright) · task · chatter`
  (`:2781-2783`). Submerged is the share of steps with depth > 0.2 m.
  - `task = 0.5·prog·served + 0.5·hold` (`:2142`).
  - `prog` = clip(mean of the pair's signed along-heading speeds / commanded speed, 0, 1)
    × exp(−across²) (`:2227-2253`).
  - The hold score of a half is `at · still · level · served` (`:2065-2075`):
    - `at` = 1 − |depth − d| / max(|d − d_release|, 1 m);
    - `still` = 1 − drift / (0.30 m/s × half the window);
    - `level` = 1 − net vertical rate / 0.05 m/s (`tasks.py:112, 128`).
- **Land.** `competence = served_seg · (0.5·contact + 0.5·upright) · prog · served_walk · (0.5 + 0.5·stop) · chatter`
  (`:2988, 2995-2998, 2126`). Here `served_seg` is the share of the segment that ran, and
  `served_walk` is the share of the walk phase's late half that was recorded (pair minimum).
  - `contact` counts steps in contact *while upright* (`:2798-2803`).
  - `stop` = 1 − drift / (0.04 m/s × half the window) (`:2077-2080`).
  - `takeoff_height` is published for the mission factor (`:2840-2897`). It is an apex
    estimate, gated per sample on being free and level, zeroed if mean upright < 0.7, and
    capped at 0.29 m for a wingless body.

### 2.4 What comes out (MEASURED)
- Per-medium competence, `measurements` (the ladder metrics), the four `TransitionResult`s,
  and `mission_fraction`, all written by `finalise_tier1` (`evaluate.py:168-248`).
- The **transition factor** is `mean_k(approach_k) · (0.40 + 0.60·mean(shock, control, settle, economy, exit_state))`
  over the four kinds (`evaluate.py:185-194`; `component_means`, `transitions.py:810-823`).
  The quality terms are 0 for any kind that did not cross (they keep their dataclass
  defaults, `transitions.py:202-206, 696-724`).
- The **take-off factor** is clip(`takeoff_height` / 0.30, 0, 1), floored at 0.05
  (`evaluate.py:143-148, 213-222`).
- **Energy** is `clip(0.85·battery_wh / (Σ_media P_seg·300 s·3 + Σ 9 cycle transitions), 0, 1)`
  (`evaluate.py:422-427`; `triphibian.py:105-111`).
- **Clamping** multiplies mission by 0.7. It is flagged as an exploit if mission > 0.35 or
  any actuator overload > 3 (`evaluate.py:233-240`).
- **Ladder rungs** (`evolution/judge.py:69-291`) are telemetry and a feature for the scout.
  They are not in the fitness scalar (`loop.py:1354-1361, 1485-1486, 1515`). The first
  transition rung, `crosses`, needs `crossed_fraction ≥ 0.34`, i.e. two of four kinds
  (`judge.py:283`). The largest `crossed_fraction` in any evaluate event I tallied is 0.25
  (arch47, 51, 56, 59). MEASURED.
- **What the archive stores is the noise-free re-score.** The main pass samples the shared
  policy because it feeds PPO (`batchroll.py:857-861, 893-897`). The re-score then runs at
  the policy mean before anything is kept (`loop.py:826-892`). A code comment records "a
  33% relative standard deviation on mission_fraction and 28% on the crossing fraction"
  under sampling (`loop.py:829-835`). That comment is **undated** and may predate the
  2026-10-03/04 crossing changes.
- **Rounding** (`loop.py:1249-1254`): the stored `mission_fraction` is rounded to 4 dp and
  the competences to 3 dp. Selection uses the unrounded values (`loop.py:1396-1405`).

---

## 3. Each crossing kind (Q3)

### 3.1 The common machinery (MEASURED, `transitions.py:325-499, 592-795`)
- **Duration.** 6 s, hard-coded as the default and not passed by either evaluator
  (`:597`; `batchroll.py:1285`).
- **Command.** For `HOLD_SECONDS` = 1.5 s the observation's domain one-hot is the start
  medium; after that it is the target (`commanded`, `:398-400`; used at `:630-636` and
  `batchroll.py:1351-1352`). The six task channels are all zero during a crossing, because
  no segment task is active (`triphibian.py:1704-1711`). The domain flip is the only
  command. INFERRED: in every segment the one-hot equals the current medium
  (`triphibian.py:1817`), so the "mismatch" input occurs only in crossings.
- **Controller basis.** `WATER if "water" in kind else AIR` (`transitions.py:615`;
  `batchroll.py:1324`). So **`air_to_water` flies its 1.5 s air hold through the water
  mobility basis**. INFERRED risk, untested by any experiment (NOT FOUND in
  `experiments/transition_distance`, `still_leak` or `still_transitions`). The same rule
  drives `water_to_air`'s exit into air through the water basis. Land has no basis of its
  own; it uses the air basis (`evaluate.py:51-58`).
- **`started_in`.** `medium_of(env, GROUND_TOL = 0.10)` must equal the start medium at
  placement (`:368`). Otherwise "did not start in …" (`:495-496`).
- **Hold score** (`:447-459`): 0 if the start check failed, if the machine left early, or
  if nothing was recorded.
  - **Air start:** must stay AIR (root above the water, no contact) every hold step
    (`:408-423`). Score = `stayed × clip((1.5 − loss)/1.0, 0, 1)`, where
    `loss = max(z0 − z_min, energy-height loss)` and the energy-height loss is the
    least-squares slope of z + |v|²/2g over the hold × 1.5 s (`:387-396, 452-458`;
    `HOLD_ALT_FULL/ZERO` 0.5 / 1.5 m, `:261-262`).
  - **Water start:** must not be LAND and clearance ≤ 0.5 m every step (`:424-430`).
  - **Land start:** steps with `medium_of(GROUND_TOL)` = LAND count. Clearance > 0.5 m or
    WATER ends the hold early (`:431-435`).
- **Arrival.**
  - Water or land target: the first go step in the target medium, and for land it must be
    `ashore` (root dry, touching terrain, ramp under the root above the water surface,
    i.e. root x ≥ 7.80 m in calm water) (`:442-445, 287-305`). The machine must **still be
    there at t = 6 s** (`:487-490`).
  - Air target: terminal only. Clearance > 0.5 m at the last step (`:483-486`).
- **Crossed** = target condition met, no failure (battery flat, diverged, or `bad_qacc` >
  0), and hold ≥ `HOLD_PASS` = 0.5 (`:256, 493`; `batchroll.py:1422-1426`).
- **Failure-string precedence.** This matters for reading the stored notes:
  - "did not hold before the command to cross" is written **only when the target condition
    was met** (`cross ≥ 0`) with hold < 0.5 (`:497-498`);
  - "never crossed the boundary" means the target condition was not met at the end,
    whatever the hold was (`:682-683`; `batchroll.py:1433-1434`).
- **Graded `approach`** (feeds `component_means["crossed"]`):
  - if crossed, approach = hold (`:726`);
  - if not crossed and the target is air: `0.6·(0.5·min(go_peak/0.5, 1) + 0.5·aloft)·hold` (`:713-717`);
  - if not crossed and the target is land: `0.6·shore_progress·hold`, where progress is
    counted from the go command toward x = 8 (`:718-723, 474-481`);
  - if not crossed and the target is **water: 0**. There is no graded approach to water.
- **Quality**, only for a completed crossing (`:728-795`):
  - shock: slam pressure against hull capacity;
  - control: minimum upright;
  - settle: time to steady attitude;
  - economy: energy against 1-10× idle;
  - exit_state: upright plus placement.

### 3.2 Per kind: what must be true (MEASURED code; feats INFERRED)

| kind | start | hold (first 1.5 s) | go (4.5 s) success | implied feat (INFERRED) |
|---|---|---|---|---|
| `air_to_water` | 4 m over open sea, at trim speed | stay airborne and lose ≤ 1.0 m of height or energy height | root under water at the end | **powered level flight for 1.5 s**: ≤ 0.67 m/s energy loss at trim. A non-flier hits the water in 0.90 s. Then enter the water (passive) |
| `water_to_air` | root 2 m deep, at rest, 8 m from the ramp | stay wet, clearance ≤ 0.5 m | lowest geometry > 0.5 m above the water **at t = 6 s** | breach from 2 m and be airborne ≥ 0.5 m at the final step: T/W > 1 in air, or a leap timed to the last step |
| `water_to_land` | root 0.15 m under, 3-7.5 m seaward of the shore | stay wet, not LAND | ashore at the end, arriving after 1.5 s | 3-7.5 m in ≤ 6 s (≥ 0.5-1.25 m/s), or in 4.5 s (0.67-1.67 m/s) if it holds still during the hold. Then climb onto dry ramp |
| `land_to_air` | beach, x = 15, at rest | on the ground ≥ 50% of the hold, never > 0.5 m clear | clearance > 0.5 m above the beach **at t = 6 s** | take-off from rest: hover (T/W > 1) or a ground run to trim speed. The repo's table quotes 7-46 m at T/W = 1 for the seed plans, against 24.8 m of beach inland of x = 15 (`transitions.py:81-92`) |

**Directional:** a start in the wrong medium scores nothing. Only a change *into* the
target, after the command, counts, and it must persist to the end
(`transitions.py:325-363`).

**What zeroes a crossing:**
- not starting in the start medium;
- leaving the start medium during the hold;
- hold < 0.5;
- any abort, which also sets hold to 0 (`:463-470`);
- not being in the target at t = 6 s.

**Which kinds are scored, and where (MEASURED):**
- **All four** are run for **every candidate on every island** (§2.1).
- The Tier-1 transition factor averages all four, including `water_to_air`. The mission
  cycle never uses `water_to_air`: `CYCLE_TRANSITIONS` are `air_to_water`,
  `water_to_land`, `land_to_air` (`triphibian.py:74-75`).
- Island scores read subsets (`islands.py:70-128`):

| island | crossings it reads |
|---|---|
| air | `air_to_water`, `water_to_air` |
| water | `air_to_water`, `water_to_land` |
| land | `water_to_land`, `land_to_air` |
| amphibian | `water_to_land` |
| aerial_diver | `air_to_water`, `water_to_air` |
| land_air | `land_to_air` |
| triphibian | all four, curriculum stages only |
| generalist | all four, via mission and its curriculum |

### 3.3 Where crossings fail, in stored data (MEASURED)
- **Crossing-failure notes** per evaluate event, from `vocab.json`. Caveat: events keep the
  first 8 notes (`loop.py:1562`), and only the first 3 before `eb9b79d` (2026-10-08), so
  arch47-49 lack `land_to_air`. arch51 (n = 10,388):

| kind | outcomes |
|---|---|
| `air_to_water` | "did not hold" 8,466 (81.5%: it **fell in**), battery 1,042, "never crossed" 578, diverged 182, unstable 120 |
| `water_to_air` | never 9,747, battery 533 |
| `water_to_land` | never 9,601, battery 685, "did not hold" 11 |
| `land_to_air` | never 9,352, battery 949, "did not hold" 7 |

  Share of `air_to_water` evaluations ending "did not hold": arch47 85%, arch56 73%,
  arch60 67%.
- **Hold-pass rates.** Pass rates for the hold are measured only offline, on arch46's 218
  elites after the 2026-10-04 gates (`experiments/transition_distance/README.md:207-212`):

| kind | hold-pass |
|---|---|
| `air_to_water` | **0.000** |
| `water_to_air` | 0.844 |
| `water_to_land` | 0.775 |
| `land_to_water` | 0.876 |

  INFERRED, because the hold-pass rates come from a different population and date
  (arch46, 2026-10-04): for the water starts, the "never crossed" machines in arch51 mostly
  held and then never left the water.
- **Noise-free crossings** (evaluate events, `ladder_measurements.transition.crossed_fraction > 0`):

| run | crossings / evaluations | kinds and bodies |
|---|---|---|
| arch47 | 2 / 9,439 | `land_to_air` gannet without rotors; `air_to_water` eel with 21 rotors |
| arch51 | 36 / 10,388 | all rotor bodies (2-21 rotors): `land_to_air` 21, `water_to_land` 10, `water_to_air` 5. Air 0 in all 36; water ≥ 0.012 in 7; land ≥ 0.012 in 1. 22 were placed, 14 rejected; spread over 8 islands |
| arch56 | 6 / 3,549 | `air_to_water` 4 (gannets without rotors); `water_to_land` 2 (rotor gannet) |
| arch59 | 4 / 3,761 | `land_to_air` 3 (teal without rotors); `water_to_air` 1 (eel without rotors) |

  No evaluation crossed more than one kind. Whether any of these is commanded rather than
  the open-loop rotor leak (`docs/ARCH46_SPEC.md:404-414`, "not built: station gate") is
  **UNMEASURED**: no still-machine or base-gait arm was run on them.
- **PPO sampled pass** (`ppo` events, `reward_by_tag` transition mean > 0). Each case is
  about 1 crossing among 16 trajectories, reward 0.66-0.98:

| run | generations with a crossing reward |
|---|---|
| arch47 | 3 |
| arch48 | 0 |
| arch49 | 4 |
| arch51 | 46 (about 47 trajectories of 55,480) |
| arch56 | 10 |
| arch59 | 8 |
| arch60 | 0 |

- **Correction to two earlier statements.**
  - "Transitions crossed 0, 0 … PPO and ladder" (`docs/ROADMAP.md:5975`): for arch47 the
    "ladder" half holds, because the first transition rung needs 2 of 4 kinds. The "PPO"
    half does not: arch47's PPO events show 3 crossing rewards, against arch48's 0.
  - E2's "0 in every kind (arch47, arch48)" (`ROADMAP.md:5532`): arch47's two noise-free
    crossers were both placed (`status: new`), so at the evaluation level that statement
    does not hold for arch47.

---

## 4. The "mission": four quantities, where each lives, what nonzero needs (Q4)

### 4.1 The four quantities (MEASURED)

| name | formula | evaluated | read by |
|---|---|---|---|
| Tier-0 `mission_fraction` | clip(available / required energy) from closed-form powers (`triphibian.py:456-469`) | every candidate | the Tier-0 gate (`loop.py:694`) |
| **Tier-1 `mission_fraction`** | `min(c)^.5·mean(c)·energy·max(T, .05)·max(takeoff, .05)` (`evaluate.py:217-223`) | **every Tier-1 evaluation**: main pass and noise-free re-score | generalist island score (`islands.py:258-259`); curriculum stage 4 and stage 3's "next" (`curriculum.py:286, 138, 553-560`); `mission_standing` × `mission_weight` (0.0 since arch51, `loop.py:207, 1416-1417`); `mission_best` (`loop.py:2412-2418`); critic and scout features |
| Tier-2 `mission_fraction` | completed legs / 9 (`evaluate.py:639`) | promoted elites only: up to 3 per verification round, every 5 visits of an island (`loop.py:156, 2275-2277, 3224-3343`) | a flag on the elite since ARCH51 L4; it does not cap fitness (`evolution/curator.py:411-426`); critic labels; an exploit check only if Tier-1 > 0.05 (`evaluate.py:644-651`) |
| Continuous mission (`run_continuous`, `continuous_score`) | `(0.45·on_task + 0.35·transition_rate + 0.20·min(depth/10, 1))·survive` (`mission.py:309-336`) | **film only** (`viz/film.py:438-440` → `viz/showcase.py:324-327`) and offline `train_controller(continuous=True)` (`control/train.py:172-193`, from `ops/run.py` `train`/`showcase`) | `film_manifest.json` only. "The search never saw it" (`mission.py:146-151`) |

**Which controller drives each path** (MEASURED):
- **Tier-1:**
  - The candidate's own `Policy` (linear, since `policy_hidden` = 0; `loop.py:240, 500-504`)
    plus the shared PPO policy. The shared policy commands a body twist, which the
    medium's mobility basis turns into CPG coefficients (`batchroll.py:880-915`).
  - In the main pass the shared policy is sampled; in the stored re-score it acts at its
    mean (`batchroll.py:857-861, 892-897`; `loop.py:826-892`).
  - Crossings use the same law with the basis rule in §3.1.
- **Tier-1.5 and Tier-2:**
  - The own policy after promotion refinement (6 steps by default; `loop.py:235,
    3249, 3261-3265`).
  - Plus the shared policy at its mean, through `SummedPolicy`/`SharedController`
    (`loop.py:3013-3028`; `evaluate.py:62-127`).
- **Film continuous mission:**
  - `controller_for_elite` (`ops/run.py:657-663`), with whichever control law reproduced
    the evaluated clips (`film.py:430-437`).
  - The basis is rebound per leg by `basis_for(commanded)`; LAND uses the air basis
    (`mission.py:224`; `evaluate.py:51-58`).
  - The task channels are zero throughout, because no segment task is active
    (`triphibian.py:1704-1711`).

**Tier-2 mechanics** (`evaluate.py:523-654`):
- a random start medium, then the cycle; 3 cycles = 9 legs;
- each leg is **re-placed** through `run_segment` (`:590-597`): reset, a drawn task, no
  scatter; 300/12 = 25 s per leg;
- the mission stops at the first leg with competence < 0.15 (`LEG_COMPETENCE_BAR`, `:442`),
  a failure, or a flat battery (`:601-607`);
- no crossings are scored;
- energy is charged at the full 300 s per leg.

**Continuous mission mechanics** (`mission.py:178-306`):
- one placement only (`reset(schedule[0][0], randomise=False)`, `:204`);
- the commanded medium changes on a timer and reaches the controller only as the domain
  one-hot (`:240-244`);
- a transition is complete when the machine is in the commanded medium (by
  `current_domain`) for ≥ 60% of a 0.5 s window during the leg (`:152-153, 266-274`);
- the mission ends on a QACC reset, divergence or a flat battery (`:245-257`);
- the film uses `cycles=1` with `leg_seconds` = the run's segment length (8 s), so 3 legs
  and 2 commanded transitions (`film.py:438-440`), and seed 0 (`film.py:440`). With
  `default_rng(0).integers(3)` = 2 (computed), the order is **LAND → AIR → WATER** on every
  film (`mission.py:170-175`);
- energy is billed at ×300/8 (`showcase.py:326`).

### 4.2 Conditions for each to be > 0 (MEASURED from code)
**Tier-1 `mission_fraction` > 0.** Crossings and take-off are **not** needed.
1. The design passes Tier-0: `gate_margin ≥ −0.85` and Tier-0 mission > 0
   (`loop.py:150, 694`).
2. Air competence > 0 (§2.3):
   - aloft ≥ 2.8 s;
   - aloft in the late halves of both phases of both mirrored halves, i.e. after t = 6 s
     from the 30 m launch;
   - a mean signed turn toward the commanded side > 0.
3. Water competence > 0: submerged; and either a pair-mean signed progress along the drawn
   heading > 0, or a hold score > 0.
4. Land competence > 0: some upright posture, and a pair-mean signed walk progress > 0.
5. Battery > 0, so energy > 0.
6. To be **visible** in stored data: ≥ 5e-5 (4 dp). With no crossing and no take-off this
   needs `min^.5·mean·energy ≥ 0.02`, about all three competences ≥ 0.08 (computed). The
   archive's 0.012 floor in all three gives 3.3e-6.

**Generalist island score > 0** is the same as above. It is also zeroed if the design is
below 0.012 in all three media (`islands.py:153-160`; `loop.py:1394-1408`).

**Tier-2 > 0:** the first, randomly chosen leg reaches competence ≥ 0.15 over 25 s.
INFERRED for an air leg: it must still be aloft and turning in t ≈ 18.75-25 s after a
30 m launch, i.e. sustained flight.

**Continuous mission (film) transitions 2/2**, in the order LAND → AIR → WATER:
1. In the 8 s AIR leg, no world contact and ≤ 34% submerged for ≥ 0.3 s of a 0.5 s window.
2. In the next 8 s WATER leg, > 34% submerged, starting from wherever the air leg left it.
3. No QACC reset or flat battery.

**Feats implied for the film** (INFERRED from the geometry in §1.1): the machine starts on
the beach at x = 15. The waterline is at x = 7.80, and water deep enough to submerge a
body some tenths of a metre tall is at x ≲ 7. So after taking off it must move about
8 m seaward within 16 s: either flying seaward, or covering ground at about 1 m/s. For
scale (arch40, `tasks.py:59-73`): land `land_speed` p90 is 0.116 m/s, water p90 0.446 m/s,
and the commanded land speed is 0.04 m/s.

**E2 L3 is a conjunction of two different quantities.** It reads "`mission_fraction > 0`
with both transitions, film reproduced, on ≥ 3 of 5 held-out seeds" (`ROADMAP.md:5533`):
Tier-1 mission > 0 **and** the film's 2/2. On held-out seeds the start medium changes
(`mission.py:170`). Succeeding on 3 of 5 seeds would in practice span at least two leg
orders, and so all three cycle transitions (INFERRED). **No code path runs the continuous
mission on held-out seeds**: the film hard-codes seed 0, and `cmd_showcase` takes one
`--seed` (NOT FOUND after checking `viz/film.py`, `viz/showcase.py`, `ops/run.py`,
`evolution/`).

### 4.3 What the stored data say (MEASURED)
- Tier-1 mission ≥ 5e-5 (from evaluate events):

| run | evaluations with mission ≥ 5e-5 | max | context |
|---|---|---|---|
| arch47 | 13 of 9,439 | 0.0003 | land_air design, air 0.181 / water 0.283 / land 0.243 |
| arch48 | 7 of 4,792 | 0.0001 | |
| arch51 | 0 | | 0 evaluations with all three media > 0; air > 0 in only 4 |
| arch56 | 1 | | |
| arch59 | 0 | | |
| arch60 | 0 | | |

- Evaluations with all three media > 0. Note the floor: competences < 5e-4 are stored as 0.

| run | all three > 0 |
|---|---|
| arch47 | 107 |
| arch48 | 54 |
| arch51 | 0 |
| arch56 | 21 |
| arch59 | 5 |
| arch60 | 48 |

  Example: arch59 gen 222 had air 0.080, water 0.051, land 0.076, which gives mission
  3.9e-5 (computed), stored as 0.0000.
- Film continuous mission, from the manifests:

| run | transitions | detail |
|---|---|---|
| arch47 | 1/2 | land 30% / air 71% / water 0% on-task. Unverified as a real take-off: `current_domain` counts any no-contact, ≤ 34%-submerged state as AIR |
| arch48 | 0/2 | |
| arch49 | 0/2 | |
| arch51 | 0/2 | |
| arch54, 56, 58, 59 | "0/0", legs [] | `film_error: "no GL context"`: **not measured** |

  The manifest's mission block reports "0/0" for "not run": could-not-measure shares a
  value with measured-zero. That is the case CLAUDE.md rule 4 forbids.

---

## 5. Crossing curriculum: "back", start rate (Q5)

**Mechanism** (MEASURED, `evolution/curriculum.py:688-769`):
- `DistanceCurriculum.kinds` = (`air_to_water`, `water_to_air`, `water_to_land`,
  `land_to_water`) (`:708`).
- `observe` pushes `tr.crossed` for crossings run at the current `back`, plus
  `air competence ≥ 0.1` for the launch (`:720-728`).
- `update` moves a start `back` by +0.5 m (to at most 20 m), or the launch down by 2 m (to
  at least 4 m), once ≥ 50% of the last 200 observations succeeded. The window then clears
  (`:736-757`).
- `apply` writes `MissionSpec.transition_back` and `air_launch_height` (`:759-763`).
- The loop observes the placed, noise-free result after each placement
  (`loop.py:2209-2216`).

**Properties** (MEASURED code):
- It only makes crossings **harder**: starts move away from the interface, and the launch
  moves lower.
- `back` is clipped at ≥ 0 (`transitions.py:514`), so no start is easier than the interface.
- `land_to_air` ignores `back` (`:502-509`).
- `land_to_water` is in `kinds` but is never run in Tier-1, so its window never fills.

**Defaults and runs:**
- `SearchConfig.distance_curriculum = True`, the user's call on 2026-10-05
  (`loop.py:127-139`; `ROADMAP.md:3406-3408`).
- arch48 ran with it on: `back` never advanced, `window_share` about 0 (`ROADMAP.md:5976`).
- **Every run arch51-arch60 passed `--no-distance-curriculum`** (`runs/arch51_notes.md:33`;
  `arch52_notes.md`; `arch60_notes.md:51`).

**"Start rate 0 at back 0"** (`ROADMAP.md:304-309, 5993-5997`) is the share of evaluations
that crossed from the current start (back 0, the interface), measured against
`advance_share` 0.5:
- arch46's 218 elites before the 2026-10-04 gates: 0-2.3% per kind, at or below still
  machines (`experiments/transition_distance/README.md:82-94`).
- After the gates: **0 of 218** in every kind at 0 m, still machines also 0. One real
  crossing remained, elite 159's `water_to_land` from 2 m back (`README.md:205-219`).
- A window of 200 holds 3-5 crossings at 1.4-2.3%, where 100 are needed (`README.md:161-165`).
- The air launch rule runs backwards: elite air competence *falls* as the launch is
  lowered, mean 0.0123 at 30 m to 0.0017 at 4 m (`README.md:105-113, 178-181`).
- From the event tallies in §3.3 (single-kind crossings in 0.02-0.35% of evaluations), a
  200-window would hold about 0-1 crossings (INFERRED).

---

## 6. Selection today (Q6)

**The scalar** (MEASURED, `loop.py:1317-1428`):

`fit = discount × [(1 − mw)·(w·isl_q + (1 − w)·cur_q) + mw·mis_q]`, where:
- `isl_q`, `cur_q` and `mis_q` are standings: the share of the per-stage window strictly
  below, ranked only against designs not at the floor; a zero stands at 0
  (`curriculum.py:290-300, 459-539`);
- `w = max(stage/4, 0.25)`, and 1.0 at stage 4 (`curriculum.py:412-457`);
- `mw` = `mission_weight` = **0.0** from ARCH51 (0.30 through arch50; `loop.py:196-207`;
  `docs/ARCH51_SPEC.md:29, 38`);
- a design below 0.012 in **all** of its island's media is zeroed and flagged `at_floor`
  (`loop.py:1394-1408`; `islands.py:144-160`). A floor design fills only an empty cell
  (`ARCH51_SPEC.md:30`);
- `floor_tiebreak` decides which of two zeros keeps a cell, by time aloft
  (`islands.py:163-196`; `loop.py:340, 1419`);
- Tier-2 no longer caps fitness (`curator.py:411-426`).

**Island scores** (`islands.py:246-285`):
- **Specialist and pair islands:** `min(own c)^.5·mean(own c)·max(quality, 0.15)`, where
  `quality = mean(crossed ? 1 : 0 over own crossings)·(0.4 + 0.6·mean(shock, control, settle, exit_state))`.
  The crossed term is **boolean**, so every non-crosser gets the same 0.15. No partial
  credit, and no ranking from crossings (INFERRED).
- **Triphibian island:** `triphibian_score = 3/Σ 1/(c_i + 0.006) − 0.006`, a harmonic
  soft-minimum with **no** transition, energy or take-off factor (`islands.py:199-237`).
- **Generalist island:** Tier-1 `mission_fraction`.
- "**three_media**" is not an island. It is a telemetry count of triphibian-archive elites
  with min(air, water, land) ≥ 0.012 (`loop.py:2425-2431`). An island of that name is NOT
  FOUND in `ISLANDS` (`islands.py:70-128`).

**Curriculum stages** (`curriculum.py:66-72, 165-286`). The score is `here + 0.25·next`
(`:552-560`). Promotion needs the bar *and* next > 0 (`:565-603`).

| stage | shared ladder | bar | triphibian ladder (`:87-138`) | bar |
|---|---|---|---|---|
| 0 | best own medium | 0.25 | c2 | 0.055 |
| 1 | best cruise progress | 0.35 | c3 | 0.012 |
| 2 | mean approach × quality: **zero without a completed crossing** | 0.35 | approach × quality × min(1, c3/0.012) | 0.35 |
| 3 | √(c1·c2) × max(mean approach, 0.1) | 0.30 | ∛(c1c2c3) × max(approach, 0.1) | 0.30 |
| 4 | mission | 0 | mission | 0 |

Promotion from stage 1 needs a stage-2 score > 0 (`:600`), or ≥ 0.14 on the triphibian
ladder (`:590-591`). So a cell can only reach the crossing stage through an actual
crossing.

**Does the mission or any crossing enter selection today?** (MEASURED code)
- **Mission:** only on the generalist island, as `isl_q` with weight `w` ≥ 0.25, and
  through stage 3's "next" and stage 4 on any island. arch52-arch60 ran `--islands air`,
  so there was **no generalist or triphibian island**, and `mission_weight` was 0. The
  mission entered no selection in those runs.
- **Crossings:** through the boolean, 0.15-floored factor on specialist and pair islands;
  through stages 2 and 3; and through Tier-1 mission's transition factor. That factor is
  `0.4·mean(approach)` with a 0.05 floor, so the graded approach matters only above a
  **mean of 0.125**. Mean approach per evaluation is p99 ≤ 0.094 in arch47, 51, 56 and 59
  (max 0.34 in arch51, crossers included). MEASURED. So the graded approach, built to make
  partial crossings "visible to selection" (`transitions.py:180-199`), is invisible in
  practice (INFERRED).
- **On the air island** (arch52-60): `min(air)^.5·mean(air)·max(quality(air_to_water, water_to_air), 0.15)`.

---

## 7. The learner's reward for crossings and the mission (Q7)

**Terminal reward** (MEASURED):
- **Segments:** competence, plus `air_turn_reward × pair_turn_mean` in air when that is
  set (1.0 in arch53-60) (`learning/ppo.py:349-371`; `batchroll.py:1227-1229`).
- **Crossings:** `float(t.crossed)·(0.40 + 0.60·mean(shock, control, settle, economy, exit_state))`
  (`batchroll.py:1259-1264`). This is **binary in `crossed`**. The comment above it says
  it is "the same graded quantity `finalise_tier1` folds into mission_fraction"
  (`:1255-1258`), but `finalise_tier1` uses the *graded* `approach` (`evaluate.py:185-194`),
  so the comment does not match the code.
- **Continuous mission:** no PPO reward. It never runs in the search.

**Shaping** (MEASURED):
- weight 0.2 (`loop.py:213, 2155`); the shaping term is `0.2·(γΦ(s') − Φ(s))` with
  Φ(terminal) = 0 (`ppo.py:499-504`);
- during a crossing, Φ is the **commanded** medium's potential (`told.value`,
  `batchroll.py:1376-1379` → `ppo.py:403-440`):
  - air: −tanh(depth/5) + 0.5·upright, i.e. height above the *water*, not above the ground;
  - water: −|tanh((d − 10)/3)| + 0.25·wet;
  - land: contact + 0.5·upright, which a touch on the submerged ramp also satisfies;
- `potential_of`'s "crossings" branch (`ppo.py:439-440`) is dead on the batched path,
  because `told` always has a `.value` (INFERRED from the code).

**Is there a dense signal toward a boundary?**
- **Per step, yes:** the shaped reward points toward the target medium's potential once
  the one-hot flips.
- **Per episode, no:** the discounted shaped return telescopes to `−0.2·Φ(s0)`, which the
  fixed placement sets.
- arch51's gen-0 `ppo` event shows exactly such constants: shaping return −0.214
  (`air_to_water`), +0.198 (`water_to_air`), +0.199 (`water_to_land`), −0.100
  (`land_to_air`), with terminal return 0.0 for every crossing tag. MEASURED.
- With the crossing reward identically 0, the true advantage of every action is 0. The
  expected policy gradient from crossing data is then zero; any push comes from GAE and
  critic bias toward raising Φ quickly (INFERRED, standard potential-shaping math).
- **Completed crossings that did reach the learner** (§3.3): arch51 had about 47 crossing
  trajectories in 46 of 900 generations; arch60 had none.

---

## 8. E2, the adopted definition of done (Q8)

Quoted from `docs/ROADMAP.md:5529-5534`:

| level | passes when |
|---|---|
| L1 medium | each of air, water, land has ≥ 5 distinct elites over a bar certified against both controls, retaining ≥ 50% of it on the auditor's held-out seeds |
| L2 crossing | each of the four crossings is crossed by ≥ 5 elites while still machines cross 0 (`CrossingTracker` gate) |
| L3 mission | one design completes the continuous mission (`mission_fraction > 0` with both transitions, film reproduced) on ≥ 3 of 5 held-out seeds |
| L4 target | L3 at 15 kg, 10 m depth, 45 min |

Adopted: "**Adopted 2026-10-12 (the user, question 4).** L1-L4 as tabled. Air's L1 bar is
left open" (`ROADMAP.md:5536-5540, 5736-5738, 5773-5774`). "Each level is read on the
auditor's held-out seeds, never on the draw that selected the design" (`:5542-5543`).

**Status as recorded:**
- **L1 air: not met.**
  - arch56 met R2: `comp:air ≥ 0.012` certified, 7/111 elites against 0 still and 0 base.
    It missed R1 (0 sign-test passes) and R3 (3 of 5 elites retaining ≥ 0.5)
    (`ROADMAP.md:398-411, 440-451`).
  - arch58 and arch59: not met (`ROADMAP.md:26-39`).
  - The existence criterion is certified when pooled over seven populations (34/946 vs
    0/946). R1 has never passed (`:32-35`).
- **L1 water and land:** the bars are certified on arch51 (water ≥ 0.012: 61/139; land
  ≥ 0.15: 9/139). Retention is unmet: arch51 audits measured land 0.02 and water 0.06
  (`ROADMAP.md:5558-5563`).
- **L2:** recorded as "0 in every kind (arch47, arch48)" (`:5532`). By my tallies (§3.3)
  there are single-kind crossings at the evaluation level in arch47, 51, 56 and 59. None
  has a still-machine check (UNMEASURED), and no kind is shown crossed by ≥ 5 certified
  elites.
- **L3:** 0. For arch54 onward it is not measured (§4.3).
- **L4:** —.
- E2 does not say which code path runs L3 on held-out seeds, and none exists (§4.2).

---

## 9. Structural blockers (Q9)

Each blocker is tagged MEASURED or INFERRED.

1. **Air is gated on reading a turn command, and the mission is gated on air.**
   *MEASURED.* `mission_fraction` = 0 whenever any medium is 0 (`evaluate.py:217-218`).
   Air credit needs a commanded-turn response that is positive over a mirrored pair
   (`triphibian.py:2160-2203`). Result in arch51: air > 0 in 4 of 10,388 evaluations and
   all three media > 0 in 0.
2. **Floors and rounding hide nonzero missions.** *MEASURED.* The 0.05 × 0.05 floors
   multiply any design without a crossing or take-off by 0.0025. Storage at 4 dp writes
   such designs as 0.0000 unless all three competences are ≳ 0.08 (`evaluate.py:221-222`;
   `loop.py:1249`). PREREG prediction 1 ("exactly 0, not small") **cannot be tested from
   stored events below 5e-5**. The substitute is the all-three-positive counts in §4.3,
   whose own floor is 5e-4 per medium (3-dp competences, `loop.py:1252-1254`).
3. **Configuration: recent runs did not ask.** *MEASURED.* arch52-60 ran `--islands air`;
   `mission_weight` has been 0 since arch51; the distance curriculum was off in arch51-60.
4. **Instrument: the continuous mission.** *MEASURED, with INFERRED leniency.*
   - It runs only inside the film, and needs GL (`showcase.py:252-253`). It has been
     unmeasured since arch54.
   - It is LAND-first on every film (seed 0).
   - Its domain test is lenient (`mission.py:156-164`): a floater ≤ 34% submerged with no
     contact reads AIR, and a body touching the submerged ramp reads LAND. No still-machine
     read of the continuous mission exists (NOT FOUND in `experiments/`).
5. **Time limit against distance.** *INFERRED, from geometry and the measured speed
   distributions.*
   - `water_to_land`: 3-7.5 m in ≤ 6 s, i.e. ≥ 0.5-1.25 m/s.
   - Film LAND → WATER: about 8 m in ≤ 16 s, including a take-off.
   - Against: water p90 0.446 m/s, land p90 0.116 m/s (arch40, `tasks.py:59-73`); commanded
     speeds 0.30 / 0.04 m/s.
   - Not measured on the current population: UNMEASURED.
6. **`air_to_water` needs powered level flight that no Tier-1 segment requires.**
   *MEASURED.* The hold allows ≤ 1.0 m of energy-height loss in 1.5 s at 4 m
   (`transitions.py:452-458`). Still seed plans lose 2.6-9.5 m (`transitions.py:259-260`).
   arch46 elites: hold-pass 0/218 after the gate. arch51: 81.5% fell in during or after a
   failed hold.
7. **The air parts of `air_to_water` and `water_to_air` are driven through the water
   basis.** *INFERRED risk, untested.* This covers `air_to_water`'s air hold and
   `water_to_air`'s exit (`transitions.py:615`; `batchroll.py:1324`).
8. **Terminal-clearance targets need T/W > 1 or a take-off run.** *INFERRED.* This applies
   to `water_to_air` and `land_to_air`: > 0.5 m clear at t = 6 s (`transitions.py:483-486`).
   The repo's T/W and ground-run table (`transitions.py:81-107`) is **undated**. It may
   predate the 2026-09-23 fluid-model fixes (`ROADMAP.md:348-354`).
   - Rotor bodies are the obvious class that could do it.
   - arch54 had 51% battery exhaustion in gannet fresh draws; arch59 had 50.6% across all
     fresh draws (`ROADMAP.md:34-38, 391-394`).
9. **One frozen entry state per crossing kind.** *MEASURED; consequence INFERRED.*
   - Each crossing kind starts from the same entry state for every candidate, generation
     and run (`transitions.py:223-235`), so rule 8 ("a score must survive a fresh seed")
     never applies to crossings.
   - N11 re-evaluation of a stored-basis elite reproduces the same crossing outcome.
   - Policy sampling noise does move crossings (28% RSD, `loop.py:829-835`, an undated
     comment): crossings show up in the PPO pass that the noise-free re-score does not keep.
10. **No learning gradient toward crossing.** *MEASURED code; gradient claim INFERRED.*
    The terminal reward is binary (`batchroll.py:1259-1264`), and the shaping return is a
    placement constant (`ppo.py:499-504`; arch51 values in §7).
11. **The curriculum cannot ease a crossing and never started.** *MEASURED.* It only
    hardens; back ≥ 0; `land_to_water` is never observed; the start rate was 0/218.
12. **Selection is blind to partial crossings.** *MEASURED code; INFERRED effect.* The
    specialist-island factor is boolean with a 0.15 floor. Stage 2 needs a completed
    crossing. The graded approach enters mission only above a mean of 0.125, which is
    above p99.
13. **Tier-2 cannot count crossings.** *MEASURED.* It re-places every leg, scores no
    crossing, and its outcome is only a flag. An air leg there needs about 19-25 s aloft
    (INFERRED, from 25 s legs whose second phase starts at 12.5 s).
14. **Minor inconsistencies.** *MEASURED.*
    - The transition factor averages `water_to_air`, which is not a cycle transition, while
      energy is charged for the 9 cycle transitions.
    - Two `SHORE_X` constants: 8.0 and 12.0.
    - The comment at `batchroll.py:1255-1258` contradicts the code at `:1259-1264`.
    - No segment ever starts a machine at a boundary. Only the 6 s crossing probes do.

---

## 10. Method notes and what I did not do

- **Stored-data tallies.**
  - Read with `grep '"kind": "evaluate"' | python` and `grep '"kind": "ppo"'` over
    `runs/{arch47,48,49,51,56,59,60}/events.jsonl`, plus `runs/*/media/film_manifest.json`.
  - "Noise-free crossings" means `ladder_measurements.transition.crossed_fraction > 0`. The
    crossed kind is inferred as the one transition missing a failure note in that event.
    Safe for arch51 onward, where 8 notes are kept.
  - PPO crossings are inferred from `reward_by_tag` (mean, std, n), assuming one nonzero
    value per batch.
- **Not run:** the still-machine and base-gait arms on any crossing body; any re-scoring;
  any physics. Whether the stored crossers are commanded is UNMEASURED.
- **Comparability.**
  - Crossings are comparable only from 2026-10-04 (ARCH46 §8), i.e. arch47 onward.
  - Air is not comparable across 2026-10-10: arch51 onward vs earlier.
  - Water and land are not comparable across 2026-10-06 and the MuJoCo 3.15 upgrade
    (2026-10-26).
  - The tallies above are per run and are not pooled across those boundaries.
