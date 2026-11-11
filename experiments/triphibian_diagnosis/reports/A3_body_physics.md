# A3 — Body physics: can any body the genome expresses do three media and the crossings?

Investigator A3, 2026-11-10, tree at `a7770c2` (read-only; no rollout, no import of
mujoco/torch). Every factual claim carries `path:line` or `path` + JSON key. Tags:
**[M]** measured and stored in the repo; **[C]** read from code (what the model *is*);
**[I]** inferred by me, not measured; **UNMEASURED** / **NOT FOUND** as stated.

Scoring regimes (a number is only comparable inside its regime; CLAUDE.md
"Comparability boundaries"):
`P0` pre-2026-09-23 fluid/actuator physics; `P1` corrected fluid (09-23..26) and
entrainment reaction (09-30); `X` crossing gates (CrossingTracker 10-03, energy hold /
`ashore` 10-04); `A` antipodal water/land (10-06); `AP` paired (mirrored) air (10-10);
`MJ15` mujoco 3.15 (10-26). Gate files for arch53-59 carry no mujoco version
(`gate_arch5x.json` top-level keys: run, n_elites, of_archive, wall_s); they were run
2026-10-25 (`experiments/air_turn_bar/README.md:434-476`), before the 3.15 switch
(`docs/ROADMAP.md:19-24`), so they are 3.11 **[I]**.

---

## 0. Answer

**The physics does not forbid the mission, but no stored measurement shows one body doing
it. The only body that flies the air segment under power — the hand-flown reference quad —
has never been run in water, on land or through a crossing; the evolved rotor bodies that
were run through them (under the shared policy) do not fly the air segment (air 0, battery
exhausted) and diverge in water.**

- Every *ingredient* has been physically produced by some expressible body in this
  simulator: powered level flight and commanded turns (the hand-flown reference quad),
  gliding with above-null air credit (gannet/teal elites), swimming and depth holding on
  command (water ≥ 0.15 certified), commanded land locomotion (land ≥ 0.15 certified),
  take-off from land (17-rotor gannet lineage, open loop), take-off from water (a 6-rotor
  amphibian, open loop), air→water and water→land crossings made only when driven
  (still and base arms 0). Two single bodies scored ≥ 0.012 in all three media at one
  draw under the current scoring (arch56 #21 gannet, arch59 #137 teal). [§3, §5]
- **No body has been measured doing powered flight *and* water *and* land.** The bodies
  that fly under power cannot be steered by the search's controller (the identified basis
  moves rotor speed by 0.0003 of top speed per unit coefficient), most evolved rotor bodies
  cannot lift themselves (153/191 below T/W 1.2 upward), the lift-capable ones were not
  flyable even by the hand law that flies the quad (0/24 ≥ 0.8), and the heavy 17-rotor
  lineage that does lift off the beach ends every gate air segment "battery exhausted"
  (arch54's rotor gannets replay at 8-15 kW against a 5.25 kW pack limit). [§3.1, §6]
- **Flapping level flight is still undecided, not refuted.** Power is 3.6-28x sufficient
  (N7), Strouhal is in band (N13), but with real joints the best measured margin is 0.42
  (needs 1.0), and the fixture that would separate "the gait search never sampled it" from
  "the fluid model under-predicts thrust" (N15) was never run (NOT FOUND). [§3.1]
- The **minimal spec** [I, §7] is a ~1.5-3 kg multirotor with T/W ≥ ~1.5 in air,
  near-neutral fixed buoyancy (rotors cannot reverse; ballast and gas genes are not in the
  dynamics), a sealed hull that survives a ~9 m/s entry, ground contact gear, and ≥ ~95 Wh
  lipo — expressible by the genome today (the quad *is* one, minus legs and battery), but
  not steerable through the current control path.

---

## 1. Design space (Q1)

### 1.1 Part kinds, joints, materials, motors [C]

- Part kinds: `hull, strut, wing, paddle, foot, ballast, membrane, bell, fin`
  (`dytiscidae/core/genome.py:35-55`); surfaces are `wing, paddle, membrane, fin`
  (`genome.py:58`); membrane has no spar (`genome.py:208-215`).
- Joint kinds: `none, hinge, universal` (`genome.py:60`). `universal` = stroke hinge +
  feathering hinge about the span, each with its own motor (`dytiscidae/core/mjcf.py:682-708`;
  second motor mass `dytiscidae/core/phenotype.py:660-675`).
- Materials `petg, petg_cf, cfrp, alu, membrane, foam` (`genome.py:62`;
  `dytiscidae/physics/materials.py:125-132`). Motors `bldc, geared, coreless`
  (`genome.py:63`; classes `materials.py:207-239`).
- Expansion: module graph with per-edge `reflect`, `radial` 1-8, `recursion` 1-4
  (`genome.py:219-234, 533, 686`); `MAX_PARTS = 24`, `MAX_BODIES = 64` estimated bodies
  (`genome.py:710, 725`).

### 1.2 Actuator / propulsor types that exist [C]

| type | how | where |
|---|---|---|
| position-servo hinge | MuJoCo `position`, `kp = 2·τ·g`, `kv = 0.15·τ·g`, `forcerange ±τ`, `ctrlrange ±3.2` | `mjcf.py:807-828` |
| universal (flap + feather) | two hinges, two motors, `feather_lead` gene | `mjcf.py:682-708`, `genome.py:163` |
| series spring / soft drive | `series_stiffness` (k = ratio·I·ω²), `drive_compliance` gain | `mjcf.py:668-673, 816`; `genome.py:108, 125` |
| passive pitch hinge (flex) | `pitch_compliance`, optional operator only | `genome.py:180, 1028-1052, 1213-1215` |
| rotor (propeller) | spinning body on a velocity servo about the part's local +Z, BEMT forces | `mjcf.py:731-761, 795-805`; `physics/rotor.py` |
| pulsed jet | `BELL` cavity, thrust `ρQ²/A` | `genome.py:47-50`; `phenotype.py:1190-1245` |
| legs / feet | `STRUT`/`FOOT` hinged parts (contact via geoms) | `genome.py:36, 39` |

**Not expressible** [C]: wheels or tracks (no part kind; `genome.py:55`); reversible rotors
(`ctrlrange "0 om_max"`, "Forward only", `mjcf.py:796-805`); **runtime buoyancy control** —
`gas_volume`, `ballast_fraction` and the `BALLAST` kind change mass (`phenotype.py:714-715`)
and the Tier-0 buoyancy checks (`phenotype.py:973-1001`) and an analytic pump-power term
(`dytiscidae/envs/triphibian.py:436-437`), but the dynamic buoyancy is
`ρ g · volume_buoyant · f_sub` (`dytiscidae/physics/fluid.py:1405`) with `volume_buoyant` fixed
per panel at build (`phenotype.py:1117-1118, 1155, 1174`); `rg gas_volume|ballast` over
`physics/`, `envs/`, `control/` finds no dynamic use (only `physics/structure.py:505-537` static
helpers and the skills bench `envs/skills.py:74-106`), and `rg "gas|ballast" mojo/src` (the
kernel the search scores on) finds nothing. So the bladder's lift is not simulated at all
and nothing can change buoyancy during a run.

### 1.3 Genome bounds [C]

| gene | bound | where |
|---|---|---|
| global `scale` | 0.25-3.0 | `genome.py:1144` |
| `battery_wh` | 10-2000 Wh; chem `liion/lipo/lis` | `genome.py:1118-1120` |
| `flap_frequency` | jitter 0.3-20 Hz; `mut_gait`/`random_genome` draw 1.5-12 Hz | `genome.py:1122, 642, 401` |
| `gas_volume` / `ballast_fraction` / `deadrise_deg` | 0-0.15 m³ / 0-0.95 / 2-65° | `genome.py:1129-1133` |
| part length / radius / span / chord | 0.02-2.0 / 0.002-0.30 / 0.03-2.5 / 0.02-0.9 m | `genome.py:436-440` |
| `motor_mass` / `gear_ratio` | 0.001-3.0 kg (then clipped to the class range) / 1-200 | `genome.py:495-497`; `physics/energy.py:44` |
| joint range | lo −2.6..−0.05, hi 0.05..2.6 rad | `genome.py:482-483` |
| `stroke_amplitude` / `neutral` | 0-1 of half travel / 0-1 | `genome.py:584-587` |
| rotor radius / pitch ratio / throttle | 0.03-0.35 m / 0.2-1.2 / 0.05-1.0 (new rotor 0.06-0.2 m) | `genome.py:1178-1180, 1149-1151` |
| jet orifice ratio / stroke fraction | 0.02-0.9 / 0.05-0.8 | `genome.py:665-668` |
| `pitch_compliance` | 0.05-4.0 when on | `genome.py:425` |

Rotor arrival: one fresh random genome in ten carries a rotor (`genome.py:402-404`);
`mut_rotor` adds/removes/retunes on any part (`genome.py:1154-1181`). Rotor motor mass
`0.06·(R/0.127)²` kg, blades `0.011·(R/0.127)³` kg (`phenotype.py:693, 698`); top speed
`150/R` rad/s (`mjcf.py:561, 738`). Mass is an outcome, not a gene (`genome.py:245-249`).

### 1.4 Seed plans and the reference [C for genes, M for built stats]

Built stats are the untouched archetypes as the current code builds them: first-pass
`gen 0, operators ['seed']` evaluate events of `runs/arch60/events.jsonl` (keys `mass,
wing_area, span, wing_loading, density_ratio, dof, n_rotors, battery_wh, flap_hz`; first
occurrence per plan in file order). `density_ratio = m / (ρ_sw · buoyant volume)`
(`phenotype.py:330-333`), so > 1 sinks.

| plan | genes (`bodyplans.py`) | m kg | S m² | span m | W/S N/m² | ρ_rel | dof | rotors | Wh | Hz | meant to |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beetle | `:235-281` | 5.129 | 0.261 | 1.816 | 192.7 | 1.957 | 6 | 0 | 260 lipo | 2.2 | flap + paddle + walk |
| medusa | `:284-323` | 10.447 | 0.122 | 0.657 | 842.4 | 0.757 | 16 | 0 | 120 liion | 1.1 | jet swimmer, no flight |
| bat | `:326-372` | 3.960 | 0.411 | 1.649 | 94.5 | 3.199 | 8 | 0 | 200 lipo | 3.1 | membrane wing/paddle |
| eel | `:375-408` | 2.259 | 0.075 | 0.001 | 297.2 | 1.364 | 5 | 0 | 150 liion | 1.8 | anguilliform swimmer |
| ray | `:411-454` | 5.445 | 0.916 | 2.524 | 58.3 | 1.641 | 9 | 0 | 230 lipo | 1.6 | "most likely to close all three" |
| gannet | `:457-539` | 5.309 | 0.514 | 2.400 | 101.3 | 1.297 | 8 | 0 | 210 lipo | 0.6 | fixed-wing glider, folding diver |
| teal | `:542-622` | 5.345 | 0.514 | 2.400 | 101.9 | 1.372 | 10 | 0 | 150 lipo | 5.0 | gannet + leaping legs |
| quad (reference, never seeded) | `:660-696` | ~1.41 [I] | 0 | — | — | NOT FOUND | 4 rotor channels | 4 (R 0.127) | 60 lipo | — | "can anything fly here" control |

- The seven seeds are `BODY_PLANS` (`bodyplans.py:646-657`); `quad` is in
  `REFERENCE_PLANS` only (`bodyplans.py:695-696`) and is never seeded (`tests/test_physics.py:3749`).
- No seed plan carries a rotor; all but medusa are denser than seawater; bat fails its own
  `positive_buoyancy_available` check (margin −0.04, arch60 event; MATH_AUDIT S-01,
  `docs/MATH_AUDIT.md:215-222`).
- Quad mass is NOT FOUND in any stored artifact. [I] 1.41 kg from two logged numbers:
  four rotors at 600 rad/s give 17.46 N (`runs/physics_1020.log:77`), top speed is
  1181 rad/s (150/0.127), so ~67.6 N at top speed, and the logged rotor lift ratio is 4.90
  (`runs/physics_1020.log:76`) → m ≈ 67.6/(4.90·9.81) ≈ 1.41 kg.
- Minor [C]: the quad's arms say `material="cfrp_tube"` (`bodyplans.py:679`), which is not a
  key of `STRUCTURAL_MATERIALS` (`materials.py:125-132`), so `Segment.material` falls back to
  PETG (`phenotype.py:218-220`).
- Gen-0 competence of every untouched seed under the current scoring is ~0 (arch60 events,
  air island; air and water are the device path's pooled mean over 50 draws, land one
  batched draw, keys `scored_by`, `draw_stats`): air 0 for all seven (teal 2.1e-5), water
  ≤ 0.0042, land ≤ 0.0028; "every seed plan scores exactly 0 in air, water and land at its
  draw" under the paired scoring, 2026-10-10 (`docs/ROADMAP.md:4224-4228`). The table's
  columns are physical stats, not capability.

---

## 2. Actuator and power model (Q2)

### 2.1 Joint servos [C]
- Continuous power `p_cont = specific_power · mass` (bldc 3000, geared 1200, coreless 150
  W/kg; mass ranges 0.02-2, 0.05-3, 0.001-0.02 kg) (`energy.py:44-46`; `materials.py:207-233`).
- Stall torque `km·√p_cont·G·η_G`, `km = 0.05(m/0.1)^0.75·η_peak/0.88`, `η_G = 0.97^(log₅ G)`
  (`energy.py:48-61`); rated speed `p_cont/τ_stall` (`energy.py:63-66`).
- In MuJoCo: `forcerange = ±clip(τ_stall, 0.02, 400)` N·m, `kp = 2τg`, `kv = 0.15τg`
  (`mjcf.py:794, 816-828`); reference-rate feed-forward `ctrl = q_ref + (kv/kp)·dq_ref`
  (`triphibian.py:647-660`; A-01 closed, `docs/MATH_AUDIT.md:53`). Joint damping
  `0.02 + 0.5·m_seg`, armature `max(1e-4, 0.01·m_seg)` (`mjcf.py:653-654`).
- **No speed limit is enforced**: "Nothing in the dynamics enforces the rated speed"
  (`derivations/flapping_power_bound.md:59-61`); gear ratio is a near-free torque multiplier
  up to 200 (`derivations/flapping_power_bound.md:186-188`).
- Seed stroke motors: gannet/teal 60.8 N·m (x2), beetle 24.3, bat 11.2/8.2/6.0
  (`derivations/flapping_power_bound.md:192-198`).

### 2.2 Rotors [C, M]
- BEMT with Prandtl tip loss and Glauert forward-flight inflow (`physics/rotor.py:9-33`),
  tabulated per rotor and medium; validated on the APC 10x4.7: CP rms 0.003, CT rms 0.012
  over J 0.1-0.6, static CT 19% low; MATH_AUDIT R-01, risk low (`docs/MATH_AUDIT.md:60`;
  `runs/physics_1020.log:74-75` CT0 0.0938 vs 0.1158).
- **Under water**: the same tables in water density/viscosity, blended by submerged
  fraction — "a propeller swims too" (`rotor.py:262-284`). No wet penalty beyond the physics;
  drag torque ~1000x is integrated by backward Euler at the end of the step and an implicit
  `2|Q|/|Ω|` damping split (`rotor.py:538-555, 598-629`), added after arch43 (rotor plunge to
  |qvel| 787,158; `docs/ROADMAP.md:2864-2884`); guarded by `test_a_propeller_can_go_under_water`
  (`tests/test_physics.py:4600-4621`). No cavitation/ventilation (`docs/model_validity.md:75-76`).
- Velocity servo `kv = τ/(0.1·Ω_max)`, `ctrlrange 0..Ω_max`, forward only (`mjcf.py:795-805`);
  the CPG channel is a steady throttle the policy moves through the offset
  (`triphibian.py:731-739`).

### 2.3 Jets [C, M]
`thrust = ρQ²/A` from the joint rate (`docs/MATH_AUDIT.md:481-487`); pumping work now charged
(J-01 closed, `docs/MATH_AUDIT.md:47, 114-124`); **J-02 open, high: no limiter, 1,649 N on an
8.6 kg body** (`docs/MATH_AUDIT.md:65, 481-487`). Works in water, ~nothing in air
(`genome.py:47-49`).

### 2.4 Battery and mission energy [C]
- Cells: liion 200 Wh/kg, 3C; lipo 130 Wh/kg, 25C; lis 350 Wh/kg, 2C (`materials.py:158-180`);
  mass `Wh/(Wh/kg)`, continuous limit `Wh·C` (`materials.py:150-155`; `energy.py:110-115`).
- Overdraw is allowed with a 25% penalty per unit over the limit (`energy.py:126-140`).
  Avionics 6 W always (`energy.py:152`).
- Tier-1 energy: `E_req = Σ_media P_seg·300 s·3 cycles + transitions`, available
  `0.85·battery_wh`, enters `mission_fraction` as `energy_fraction` (`envs/evaluate.py:420-427,
  177-179, 217-223`; `MissionSpec` 3 cycles × 300 s, `triphibian.py:79-111`).
- Tier-2: 9 legs of `300/12 = 25 s`, a leg fails below competence 0.15, energy charged for
  300 s per leg and the mission stops past `0.85·battery_wh` (`evaluate.py:442, 585, 597-607`).

### 2.5 The back-driven servo booking [C, M]
- Electrical power is `|τ·ω| + copper + iron + viscous`, never negative, no regeneration
  (`energy.py:68-89`; batched `energy.py:192-216, 282`). A joint driven backwards by the
  fluid books positive consumption at the full `|τ·ω|`.
- Measured: an arch51 eel elite (37 actuators, 9.6 kg, 158 Wh) hits the water at 3.3 s, a
  joint runs to 865 rad/s at stall torque (48 N·m), booked ~40 kW against a 474 W pack
  limit, pack flat at 6.66 s; other seeds of the same body stay at 97-100% soc
  (`docs/ROADMAP.md:113`). arch54 rotor gannets: 8-15 kW mean from t = 0, then joints run to
  1,200-2,300 rad/s (`experiments/air_turn_bar/README.md:309-318`).
- "Battery exhausted" share on 8-draw screens, **still arm** / elite: arch54 9.4%/52.0%,
  arch56 0.0%/19.6%, arch58 40.4%/46.2%, arch59 21.9%/50.6%
  (`experiments/air_turn_bar/README.md:478-495`). A still machine exhausting its pack is a
  booking artefact; the fix is open and is the user's decision (`README.md:492-495`;
  `docs/ROADMAP.md:34-39`).

### 2.6 What limits flapping frequency [C, M]
- Genome draw 1.5-12 Hz (`genome.py:642`); Tier-0 spar inertial check closes the gannet/teal
  band at **7.80 Hz** (`derivations/flapping_power_bound.md:145-153, 194, 288`), checked at
  `genome.flap_frequency`, not at the CPG's commanded frequency (up to 20 Hz, unstored) — 14 of
  224 arch49 flapping elites sit above their own band (`docs/ROADMAP.md:4063`).
- Torque binds a full stroke at `f_bind` 2.70 Hz (gannet/teal), 2.73 (beetle), 1.70-8.10 (bat)
  (`derivations/flapping_power_bound.md:136-141, 192-198`); the stroke servo's closed-loop corner
  is 2.88 Hz (feathering teal; 80° lag at 11.33 Hz) (`flapping_power_bound.md:169-176`).
- The series spring is tuned on the phenotype inertia (0.180 kg·m²) against MuJoCo's
  0.242-0.372, cancelling 48-74% of it (`flapping_power_bound.md:177-185`).

---

## 3. Per medium: requirement vs measured capability (Q3)

### 3.0 What each task asks [C]
- **Air** (8 s, launched at (−40, 0, 30 m) at the body's trim speed, `triphibian.py:569-573,
  774-780`): cruise on the launch heading, then turn 45-90° to a drawn side
  (`envs/tasks.py:86-89, 212-219`), scored `turn·(0.5 + 0.5·height)` on a mirrored pair
  (`triphibian.py:3042-3072`). Height pays "flight" only if `level_margin ≥ 0.7`
  (`triphibian.py:327, 1099-1105, 2034-2040`), else glide credit at most 0.25/0.80 of it
  (`triphibian.py:2041-2043`). A glider can score; a non-turner cannot.
- **Water** (8 s, released 4 m deep, `triphibian.py:571`): hold at a drawn depth 5.0-6.5 m,
  still (drift < 0.30 m/s, net vertical < 0.05 m/s), and cruise on a drawn heading at
  0.30 m/s (`tasks.py:73, 84, 112, 128, 201-211`); antipodal pair cancels unchosen motion
  (`tasks.py:265-284`); gated on submerged and upright (`triphibian.py:2781-2783`).
- **Land** (8 s on the beach at x = 15, slope 0.12 rad, `triphibian.py:572`;
  `mjcf.py:116-123`): walk 0.04 m/s on a drawn heading, and stop (`tasks.py:73, 220-228`);
  gated `posture = 0.5·contact-while-upright + 0.5·upright` (`triphibian.py:2988-2998`), so a
  hovering body keeps half the posture gate [C].
- **Crossings** (6 s; 1.5 s commanded hold, then go; `envs/transitions.py:250-256, 325-499`):
  - `air_to_water`: from 4 m, level at trim (`transitions.py:240-241, 515-526`); the hold
    counts `max(height lost, energy height z + v²/2g lost)` and needs ≤ 1.0 m lost in 1.5 s
    to pass `HOLD_PASS` 0.5 (`transitions.py:256-262, 447-459`). Measured gliders fail it
    (they lose 3.4-5.7 m energy height, `experiments/still_leak/README.md:21-29`; seeds
    2.6-9.5 m, `transitions.py:257-260`). [I] The bar is dE/dt ≥ −0.67 m/s, i.e.
    `T ≥ D − 0.67·m·g/U`: at L/D 10 and U 15 m/s that is T ≈ 6% of weight, or L/D ≳ 23 with
    no thrust — far below VTOL level, which is consistent with two 0-rotor arch56 gannets
    (#72, #77) having crossed (§5).
  - `water_to_air`: from 2 m deep (`transitions.py:527-528`); terminal clearance > 0.5 m at
    6 s (`transitions.py:483-486`) → leave the water and stay up: **VTOL from the surface**.
  - `water_to_land`: root 0.15 m under at x ≤ 6 (`transitions.py:243-248, 529-555`); must end
    `ashore` (touching ground with the ground under the root above the water,
    `transitions.py:287-305, 487-490`) → climb ≥ 2 m of submerged ramp in 4.5 s.
  - `land_to_air`: terminal clearance > 0.5 m (`transitions.py:483-486`) → **VTOL from
    land** (a leap that has landed does not count; the teal docstring, `bodyplans.py:568-578`).
  The search scores these four (`evaluate.py:410-417`; `envs/batchroll.py:1238-1239`).

### 3.1 AIR

**Sustained level flight.**
- Rotor control: the quad, flown by a textbook cascade through the real segment and score,
  competence 0.864 / 0.858 / 0.922, sink 0.004-0.008 m/s, airborne 1.0, turn 0.99-1.0,
  0.636-0.660 Wh per 8 s (≈ 286-297 W) (`experiments/rotor/results/fly.json`, written
  2026-09-26 18:52, keys `competence, measured_sink_rate, turn_response, energy_wh`;
  `docs/ROADMAP.md:2886-2889`) [M, P1, pre-X, pre-AP; the cascade bypasses the policy,
  `experiments/rotor/fly.py:42-96, 104`].
  Rest-throttle `level_margin` 0.85 (`docs/ROADMAP.md:2826`).
- Flapping: `level_margin` of every seed < 1 (beetle 0.939, gannet 0.913, teal 0.826, medusa
  0.005, eel −0.07, ray −0.08, bat −1.03; `docs/ROADMAP.md:2817-2826`) [M, P1]. Level flight
  exists only with feathering and ideal kinematics (teal 1.26 at 11.3 Hz, share ≥ 1 of
  0.003), never heave-only (`docs/ROADMAP.md:6266-6273`); with free joints the same gait gives
  0.42, motors torque-limited 60% (`docs/ROADMAP.md:6277-6286`); 2x/3x motors 0.543/0.494,
  springs 0.143-0.153 (`docs/ROADMAP.md:2830-2844`) [M, P1]. Best open-loop flapper sinks 9-14
  m/s (`docs/ROADMAP.md:6275-6276`).
- But power is not the wall: available stroke power / level-flight power is 6-28 (band
  maxima; ≥ 3.6 in every variant) (`derivations/flapping_power_bound.md:282-292, 326-327`);
  `St` at the band top 0.65-0.97, 0.25 reached at 2 Hz (`flapping_power_bound.md:412-425`).
  **N15 (efficiency of a heaving-pitching foil through `fluid.py`) was pre-registered as the
  decider (`docs/ROADMAP.md:4072-4090`) and has no result anywhere** (NOT FOUND: `rg N15` hits
  only `docs/ROADMAP.md:213, 4078`). Flapping flight is therefore OPEN.
- Gliders: gannet glide sink 1.48 m/s median after the fluid fixes (`docs/ROADMAP.md:6222-6225`);
  air `comp ≥ 0.012` is certified against the still machine on fresh seeds, pooled 34/946 vs
  0/946 over seven populations (`experiments/air_turn_bar/README.md:462-466`) [M, AP].
  [I] Glide ratio ≈ 15.27/1.48 ≈ 10 (trim speed `derivations/flapping_power_bound.md:416`),
  so ~20 s aloft from the 30 m launch: the 8 s segment never runs out of height for a gannet.
  For scale, the S-04 hull wall alone took a fixture glider (a different body) from L/D 5.31
  to 2.93 (`docs/MATH_AUDIT.md:228-231`).

**Commanded turn.** Quad 0.99-1.0 (fly.json). Policy-driven gliders: first commanded-turn
credit 0.111 (arch52 gen 27; `docs/ROADMAP.md:156-160`); one arch54 elite passed the 8-draw
sign test (7/8, p 0.035) (`docs/ROADMAP.md:82`); no replication: arch56 stage 2 0/16 at
Bonferroni, stage 3 refuted (`README.md:336-362`); arch58/59 R1 0 (`README.md:443-451`) [M, AP].

**Take-off from land.** Needs VTOL [C, §3.0].
- Seeds: none — `land_to_air` "never crossed the boundary" for all seven at gen 0 of arch51
  and arch60 (events `notes`) [M]; the teal's leap peaks 0.566 m and ends on the ground
  (`bodyplans.py:545-551, 568-578`) [M, P0 — the W/S 29.3 there is pre-correction; built teal
  W/S is 101.9, §1.4].
- **Measured lift-off (open loop)**: arch54's 17-rotor gannet lineage (7.8-9.6 kg, 41 dof,
  W/S 5,435-6,793, 210 Wh, ρ_rel 2.25-2.69) crossed `land_to_air` as elites #60, #55 and as
  base gaits #78, #111, #67 (`experiments/air_turn_bar/gate_arch54.json.rows.jsonl`, arm/index;
  physical meta `runs/arch54/archive_air.json` elites[60, 78, 111, 67, 55].meta); verdict
  `leak` against base (`gate_arch54_table.md`, `crossed:land_to_air` 2/149 vs base 3/149)
  [M, AP/X]. The same bodies end every gate air segment "battery exhausted" (rows
  `failures`); arch54's exhausting rotor-gannet family (126/149 elites share 210 Wh) replays at
  8-15 kW against a 5.25 kW pack limit (`experiments/air_turn_bar/README.md:309-316`) — that
  these five are among the replayed draws is [I].
- Also crossed, mechanism not traced: arch49 #216 eel (`results_arch49_airpair.json.rows.jsonl`),
  arch55 #36 gannet (1 rotor, W/S 49.7), arch59 base #28 teal (0 rotors) — a terminal
  snapshot of a leap would also pass `transitions.py:483-486` [I, unverified].
- N6: 153 of 191 arch49 rotor elites cannot lift their weight with upward rotors at top speed
  (upward T/W median 0.15); the hand hover law that flies the quad at 0.816 flew 0/24 of the
  lift-capable ones (best 0.446) (`runs/analysis_1010_failure_theory/N6_result.md:17-18, 31-57`;
  `docs/ROADMAP.md:3982-4001`) [M]. **Why the hand law fails lift-capable bodies: NOT FOUND**
  (N6 gives the lift split and the mixer notes only, `N6_result.md:11-13`).

**Take-off from water.** Needs VTOL from the surface [C].
- Seeds: no — "never crossed the boundary" for all seven at gen 0 of arch51 and arch60
  (`runs/arch51/events.jsonl`, `runs/arch60/events.jsonl`, `notes`) [M]; no seed has thrust. `water_to_air` was
  crossed by no arm of any body in the arch49-59 gates (my scan of every `*.rows.jsonl` under
  `experiments/no_model_gate/` and `experiments/air_turn_bar/`, `bars["crossed:water_to_air"]`).
- **Measured (open loop)**: arch46 amphibian #129 (6 rotors) rose from 2.0 m depth to the
  surface in 2.7 s on rotor thrust and was 1.0-2.3 m above it from 3.5 s; #137 (16 rotors)
  likewise (`experiments/still_leak/README.md:40-44`); crossed with hold 1.0 at back 0/0.5/1 m
  in the `rotors_on` arm (`experiments/transition_distance/results_gated_crossers.json`
  `transition_rows`; `README.md:217-219`) [M, X]. Not on command (rotors at throttle).

### 3.2 WATER
- **Propulsion speed**: after the entrainment-reaction fix, 6 s from the water spawn, peak
  speed beetle 0.43, eel 0.63, ray 0.59, gannet 0.36 m/s (were 4.70/3.78/0.63/0.36: "most of
  the beetle's and eel's swimming was this momentum") (`docs/ROADMAP.md:6374-6386`) [M, P1].
  The command is 0.30 m/s (`tasks.py:73`), so speed is not the binding limit [I].
- **Buoyancy**: fixed at build (§1.2). Seeds: medusa floats (0.757), all others sink
  (1.30-3.20) (§1.4). A dense eel sinks 13 m with its actuators off
  (`experiments/no_model_gate/README.md:134`). Depth is commandable only by hydrodynamic thrust
  [C]; rotors push one way only (`mjcf.py:804`).
- **Swim on command**: water ≥ 0.012 certified against still and base in arch49 (64/229 vs
  3 vs 13), arch51 (61/139 vs 0 vs 3), arch54/56/59 (14/149, 12/111, 15/146);
  water ≥ 0.15 certified only in arch51 (9/139 vs 0 vs 1)
  (`experiments/no_model_gate/results_arch49_table.md`, `results_arch51_table.md`;
  `experiments/air_turn_bar/gate_arch54_table.md`, `gate_arch56_table.md`, `gate_arch59_table.md`)
  [M, A]. The shared PPO learns a fixed water body: 0.0002-0.0004 → 0.040-0.069 held-out, above
  still and base (M4, `docs/ROADMAP.md:5682-5707`) [M, pre-A].
- **Surfacing**: in `water_to_air` 97% of arch46 elites pass the water hold and then never
  leave (approach 0.013) (`experiments/transition_distance/README.md:143-149`) [M, X].

### 3.3 LAND
- **Mechanism** [C]: hinged struts/feet/paddles in contact with a friction-0.9 beach
  (`mjcf.py:191-193`); no wheels. Land rungs `moves` (gated `land_speed ≥ 0.02`) and
  `climbs_slope` (≥ 0.05 m) are certified against still in arch48/49/51 (`judge.py:232-238`;
  `no_model_gate/README.md:80-81`; `results_arch49_table.md`; `results_arch51_table.md`)
  [M, A for 49/51].
- **Move on command**: land ≥ 0.012 certified arch49 (51/229 vs 0 vs 0), arch51 (27/139);
  land ≥ 0.15 certified arch51 (9/139 vs 0 vs 0); land ≥ 0.055 certified arch56 (7/111)
  (same tables) [M, A].
- **Fragility**: per-elite land score moves > 50% for 34.5% of land-competent arch49 elites
  under a ×1.01 change of `solref`; ×4 softer moves 69% by > 50%; only 31-33 of 58 reproduce
  their recorded land score (`docs/ROADMAP.md:4531-4566`) [M].
- **Exit water onto land** (`water_to_land`, needs `ashore`): arch46 #159 at back 2 m, "the one
  commanded crossing in the archive" (`experiments/transition_distance/README.md:214-216`);
  arch49 #146 and #110, still 0, base 0 (`results_arch49_airpair.json.rows.jsonl`); arch55 eels
  #117, #125, #136, #143 and base #132 (`gate_arch55.json.rows.jsonl`) [M, X/A].

---

## 4. Feasibility matrix (Q4)

Cells: yes / no / number, with source; "open-loop" = the base gait or rotors at throttle did
it too (not on command). Seeds' rows use the seed body itself, not descendants. Seeds'
"no" in the medium and crossing columns is the gen-0 evaluation under the shared policy:
arch60 (device path for air and water, pooled mean over 50 draws; batched for land:
`runs/arch60/events.jsonl` gen-0 `operators ['seed']` keys `scored_by`, `draw_stats.mean`)
— air 0 for all seven (teal 2.1e-5), water ≤ 0.0042, land ≤ 0.0028; and the crossing `notes`
of every seed in arch51 and arch60: `air_to_water` "did not hold before the command to
cross", the other three "never crossed the boundary" [M, AP; MJ15 for arch60].

| body | level flight | commanded turn | take-off land | take-off water | water entry w/o divergence | swim on command | exit water → land | move on land |
|---|---|---|---|---|---|---|---|---|
| beetle | no: `level_margin` 0.939 (ROADMAP:2819) | no (air 0) | no (gen-0 notes) | no (gen-0 notes) | UNMEASURED | no (water 0.00015); swims 0.43 m/s peak (ROADMAP:6385) | no (gen-0 notes) | no (land 0.0006) |
| medusa | no: 0.005 (ROADMAP:2822); no lifting surface | no | no | no | J-02 jet unbounded (MATH_AUDIT:65) | no (0.00005) | no | no (0) |
| bat | no: −1.03 (ROADMAP:2825) | no | no | no | UNMEASURED | no (0.0004) | no | no (0) |
| eel | no: −0.07 (ROADMAP:2823) | no | no | no | runaway after impact on a 37-actuator eel elite (ROADMAP:113) | no (0.0026); 0.63 m/s peak | no | no (0.0028) |
| ray | no: −0.08 (ROADMAP:2824) | no | no | no | yes after AK fixes, entry loads 93-450 kPa (ROADMAP:6396-6400) | no (0.0012); 0.59 m/s peak | no | no (0) |
| gannet | no: 0.913; glides, sink 1.48 m/s (ROADMAP:6224) | no (air 0) | no | no | yes: survives 20 m/s nose-first (ROADMAP:6315-6316) | no (0.0042); 0.36 m/s peak | no | no (0) |
| teal | no: 0.826; feathering ideal 1.26 / free 0.42 (ROADMAP:6272, 6282) | no (2.1e-5) | no: leap 0.566 m then lands (`bodyplans.py:545-551`, P0) | no | UNMEASURED | no (0.0035) | no | no (0) |
| quad (reference, hand cascade) | **yes**: 0.858-0.922, sink < 0.01 m/s (`rotor/results/fly.json`) | **yes**: turn 0.99-1.0 (fly.json) | UNMEASURED (air segment starts at 30 m, `fly.py:100-106`) | UNMEASURED | numerically stable plunge only, 1 s, 0 bad-qacc (`tests/test_physics.py:4600-4621`) | UNMEASURED | UNMEASURED | UNMEASURED |
| arch54 17-rotor gannets (#60, #55; base #78/111/67) | lifts off; "battery exhausted" in every gate air segment (gate_arch54 rows); family replays at 8-15 kW vs 5.25 kW (README:309-316) | no (air 0) | **yes, open-loop** (gate_arch54 rows) | UNMEASURED | no: water "diverged"/"unstable" (gate_arch54 rows `failures`) | no (water 0) | no | no (land 0) |
| arch46 #129 (6 rotors, amphibian) | UNMEASURED | UNMEASURED | UNMEASURED | **yes, open-loop** (still_leak README:40-44) | UNMEASURED | UNMEASURED (in that run's pre-A scoring) | yes, open-loop, back 0 (`results_gated_crossers.json`) | UNMEASURED |
| arch56 #11 gannet (5.46 kg, 1 rotor, W/S 320) | no level flight measured; aloft 23/24 fresh draws | air ≥ 0.012 on 6/8 fresh draws (`screen_arch56.rows.jsonl` idx 11), 6/24 in stage 2; positive share 7/23 (< 50%) (`stage2_arch56.rows.jsonl`) | UNMEASURED | UNMEASURED | no failure on 32 fresh air draws | water 0.009 at its draw (below 0.012) | UNMEASURED | land 0.096 at its draw (`gate_arch56.json.rows.jsonl`) |
| arch56 #21 gannet (4.90 kg, 0 rotors, W/S 501, ρ_rel 0.94) | glider | air 0.114 at its draw; 2/8 and 5/24 fresh draws ≥ 0.012 | UNMEASURED | UNMEASURED | no failure recorded | water 0.025 at its draw | UNMEASURED | land 0.016 at its draw |
| arch59 #137 teal (3.21 kg, 0 rotors, W/S 79.8, ρ_rel 2.33) | glider | air 0.076 at its draw; 5/8 fresh draws > 0 (mean 0.032) (`screen_arch59.rows.jsonl`) | UNMEASURED | UNMEASURED | no failure recorded | water 0.033 at its draw | UNMEASURED | land 0.024 at its draw |
| arch56 #72 gannet (2.84 kg, 0 rotors) | — | air 0.035 at its draw; 0/8 fresh | UNMEASURED | UNMEASURED | **crossed `air_to_water`** (passed the energy hold), still 0, base 0 | water 0 | UNMEASURED | land 0.013 |

Single-draw scores ("at its draw") have one-draw reliability 0.18-0.30 (C2,
`docs/ROADMAP.md:257-259`); water/land re-scores on fresh draws for the arch56/59 rows are
UNMEASURED (the screens re-score air only, `screen_*.rows.jsonl` keys).

---

## 5. Two media, or a crossing, on purpose? (Q5)

**Two media ≥ 0.012 on one body at its recorded draw** (counts of media ≥ 0.012 per elite,
my scan of each `*.rows.jsonl` `comp`, elite arm) [M]:

| population (regime) | 0 media | 1 | 2 | 3 |
|---|---|---|---|---|
| arch49, paired air (A, AP) `results_arch49_airpair` | 121 | 101 | 7 | 0 |
| arch51 (A, AP) `results_arch51` | 62 | 66 | 11 | 0 |
| arch53 `gate_arch53` | 112 | 33 | 0 | 0 |
| arch54 `gate_arch54` | 119 | 28 | 2 | 0 |
| arch55 `gate_arch55` | 112 | 32 | 1 | 0 |
| arch56 `gate_arch56` | 84 | 21 | 5 | **1** |
| arch57 / arch58 `gate_arch57/58` | 125 / 96 | 9 / 17 | 2 / 1 | 0 / 0 |
| arch59 `gate_arch59` | 118 | 27 | 0 | **1** |

- In arch49 and arch51 every two-media elite is water + land. Pairs that include air appear
  only in the air-island runs: arch54 #24 (air + water), arch56 #11, #26, #72 (air + land),
  arch57 #7 (air + water), arch58 #7 (air + land); the rest are water + land.
- **Three media**: arch56 #21 (0.114 / 0.025 / 0.016) and arch59 #137 (0.076 / 0.033 / 0.024);
  `weakest:c3 ≥ 0.012` elite 1/111 and 1/146, still 0, base 0, verdict `underpowered`
  (`gate_arch56_table.md`, `gate_arch59_table.md`) [M, AP]. Their `mission_fraction` is 1.6e-5
  and 1.7e-5 (rows `mission_fraction`): with no crossing and no take-off both factors sit at
  their floors, `max(transition_fraction, 0.05) · max(takeoff_fraction, TAKEOFF_FLOOR = 0.05)`
  (`evaluate.py:148, 217-223`).
- The arch48 triphibian-island elite #28 (0.036 / 0.095 / 0.651) is pre-A and is not evidence
  (`experiments/no_model_gate/README.md:148-153`; water leaked on that population, same file
  `:89-94`).

**Crossings made with still 0 and base 0** (`crossed:*` per arm, same files) [M]:
- arch49: `air_to_water` #116 (gannet, also land 0.282), `water_to_land` #146, #110,
  `land_to_air` #216 — all `underpowered` (`results_arch49_airpair_table.md`); under MJ15 #116
  and #110 still cross (`results_arch49_dm0_mj315_table.md`).
- arch56: `air_to_water` #72 (also air 0.035, land 0.013) and #77 — `underpowered`.
- arch46: `water_to_land` #159 at back 2 m (`experiments/transition_distance/README.md:214-216`).
- Crossings the base gait also makes (verdict `leak`): arch54 `land_to_air` and
  `air_to_water`, arch55 `water_to_land` (eels), arch59 base `land_to_air`
  (`gate_arch54_table.md`; `gate_arch55.json.rows.jsonl`; `gate_arch59_table.md`).

**One body, two media plus a crossing**: arch56 #72 (air + land + `air_to_water`) and arch49
#116 (land + `air_to_water`; air 0 under the pair) — one draw each [M].

**Continuous mission with both transitions**: 0 of 90 (arch41 top 30 × 3 leg orders,
2026-09-22, P0; `docs/ROADMAP.md:2783-2791`). NOT FOUND for any later continuous-mission
measurement with ≥ 1 commanded transition (checked: `envs/mission.py` users per its own note
`mission.py:146-151`; ROADMAP; experiments READMEs above).

**A reference body run through `CrossingTracker` or the water/land segments under a
controller**: NOT FOUND (the quad appears only in `experiments/rotor/fly.py`,
`experiments/rotor_authority/run.py:651-660` air check, device parity tests, and the 1 s
plunge; `rg quad experiments/ tests/`).

---

## 6. Model risks that block feasibility or would void a success (Q6)

1. **Joint runaway booked as consumption** (§2.5): no speed clamp, `|τω|` with no
   regeneration; exhausts still machines on up to 40% of draws (arch58). Blocks any reading
   that depends on the battery, including Tier-1 `energy_fraction` and Tier-2's 85% stop.
   Open, user decision (`README.md:492-495`).
2. **Divergence (QACC resets)**: diverged+unstable / rollouts = arch51 1.76%, arch56 0.08%,
   arch59 0.79%, arch60 0.17% (`runs/arch5{1,6,9}/generations.jsonl`, `runs/arch60/...`,
   keys `diverged_rollouts`, `rollouts`, my sums) [M]. Past instances at water contact: rotor
   plunge (fixed, `docs/ROADMAP.md:2864-2884`), stale implicit damping at first wet contact
   (fixed, `:6387-6395`), the 17-rotor gannets' water segments still end diverged/unstable
   (gate_arch54 rows).
3. **Water entry**: slam is a diagnostic, not fed to the dynamics
   (`docs/model_validity.md:56`); unresolved above ~20 m/s (`docs/ROADMAP.md:6430-6433`);
   hulls are walled for 12 m regardless of use (S-04, glide ratio 5.31 → 2.93,
   `docs/MATH_AUDIT.md:224-239`).
4. **Contact model**: land competence hypersensitive to the contact constants (§3.3);
   ground effect not modelled (`docs/model_validity.md:59`); device path changes the geom
   margin/CCD (`CLAUDE.md`, three evaluation paths).
5. **Rotor in water**: same BEMT tables, backward-Euler braking; no cavitation/ventilation
   (`docs/model_validity.md:75-76`); rotors cannot reverse (`mjcf.py:804`).
6. **Buoyancy**: static; bladder and ballast genes are Tier-0-only (§1.2) — a Tier-0/dynamics
   disagreement: `positive_buoyancy_available` and `can_submerge` count `gas_volume`
   (`phenotype.py:975, 989-1001`) that the simulator does not float.
7. **Added mass**: tensor + implicit damping (F-03 closed), entrainment reaction (closed
   09-30); translation-rotation coupling not representable (`docs/model_validity.md:85-87`);
   host subtree mass stale, harmless in float64 (`docs/MATH_AUDIT.md:995-997`).
8. **Open high-risk MATH_AUDIT items touching flight/crossings**: C-12 (thrust_margin
   lift-blind), A-02 (servo cannot reproduce a pitch-heave phase: +2.62 prescribed vs −1.64
   free), C-07/C-08 (identification underdetermined on 5 of 7 seeds; parameter directions do
   not survive a reseed), C-09 (damping 30x small), J-02 (jet unlimited)
   (`docs/MATH_AUDIT.md:58-67`).
9. **The terminal-clearance bar** (`transitions.py:483-486`) passes a body that is mid-leap at
   the last step [I]; arch59 base #28 (teal, 0 rotors) `land_to_air` may be that (§3.1).

---

## 7. INFERRED: the minimal physical spec, expressibility, closest plan (Q7)

Everything in this section is **[I]** unless a source is named.

**What the mission asks, physically** (from §3.0 and §2.4):
1. VTOL in air with T/W comfortably > 1 for `land_to_air` and `water_to_air` within 4.5 s of
   the go command (terminal clearance 0.5 m) — a ground run is not available in 4.5 s at
   15 m/s trim speeds. Flapping cannot be relied on (no measured level flight with real
   joints; N15 open). **Rotors are the only expressible propulsor with measured T/W > 1**
   (quad rotor lift ratio 4.90, `runs/physics_1020.log:76`).
2. Energy-height hold in air for 1.5 s (`air_to_water`): T ≥ D − 0.67·mg/U; trivially met by
   a multirotor, not by a passive glider (§3.0).
3. Turn authority in air: differential rotor thrust (quad turn 0.99-1.0).
4. Water: fixed buoyancy near neutral, ρ_rel ≈ 1.0-1.1 so that upward rotor thrust trims depth
   at 5-6.5 m (rotors cannot pull down; a floater cannot dive unless some rotor points down);
   horizontal 0.3 m/s on any heading by tilting thrust or by fins; rotor thrust in water is
   in the model (`rotor.py:262-284`) and lifted a body out (§3.1).
5. Exit to land: cover ≥ 2 m of submerged 0.12-rad ramp in 4.5 s and stay touching dry ground
   (≈ 0.45 m/s) — rotor thrust plus skids/legs.
6. Land: 0.04 m/s on any heading with ground contact upright; legs or thrust-assisted
   skidding (hovering keeps only half the posture gate, `triphibian.py:2988`).
7. Entry: hull and deadrise surviving a ~9 m/s vertical entry from the 4 m `air_to_water`
   start (√(2·9.81·4) ≈ 8.9 m/s plus sink); sealed rotor motors.
8. Energy: `0.85·Wh ≥ 900 s·(P_air + P_water + P_land) + transitions`
   (`evaluate.py:420-427`). The quad draws ≈ 290-297 W in air (fly.json `energy_wh` ×
   3600/8); 900 s of air alone is ≈ 74 Wh against 0.85 × 60 = 51 Wh, so **the reference quad
   as built cannot fund the mission's air time**; ≥ ~95 Wh lipo (~0.73 kg) would, and liion's 3C
   limit (~285 W at 95 Wh) is marginal, so the chemistry is lipo. At ~1.4 kg + 0.27 kg extra
   battery, T/W at top speed stays ≈ 4.1.

**Minimal body**: a quad-class multirotor (4 rotors R ≈ 0.12-0.15 m, 1.5-3 kg), sealed
near-neutral hull, two short legs or feet, ≥ 95 Wh lipo, rest throttle near hover.

**Can the genome express it today?** Structurally yes: the quad is HULL + STRUT with rotor
genes, `roll = π`, `reflect` (`bodyplans.py:674-688`); `mut_rotor` adds rotors to any part and
1 in 10 random genomes carries one (`genome.py:402-404, 1154-1181`); legs are `STRUT/FOOT`
parts; `battery_wh` and chemistry are free (`genome.py:1118-1120`). **Controllably, no**:
the rotor channel is reached only through an identified basis that moves rotor speed by
0.0003 of top speed per unit coefficient (`N6_result.md:22-26`), and the search's
population builds rotor bodies that cannot lift themselves (153/191) or that lift and
exhaust the pack in seconds (arch54). This is the boundary between R1 and R2 in the PREREG:
**the physics admits the body; the controller path and the search have not produced or
steered it.**

**Closest seed plan**: none carries thrust. For flight and VTOL the closest expressible body
is the **quad reference** (not a seed); for gliding, folding entry, legs and the measured
three-media single draws it is the **gannet/teal airframe** (arch56 #21, arch59 #137). A
quad hull with teal legs is the shortest genotypic path I see [I]; no measurement of such a
body exists.

**Measurements that would settle feasibility before any run** [I, ordered cheapest first]:
(a) the quad through the water segment, the land segment and the four `CrossingTracker`
probes with a hand cascade (the `fly.py` controller extended to depth and ground modes) —
none needs the search or PPO; (b) the same with `battery_wh` ~95 and legs; (c) N15, to decide
whether flapping is in or out of the design space.
