# A9 — Literature: real air/water/land machines and animals, and what made each transition work

Status: DONE 2026-11-10. Literature pass for `PREREG.md` rival R1 (physics) and, secondarily, R2/R3/R6.
Method: WebSearch + WebFetch; PDFs read page by page where the fetcher saved them (pdftotext on the fetcher's saved
copies, outside the repo). In the repo only read-only commands were used (ls, grep, git status; reading `PREREG.md` and
`docs/model_validity.md`, the latter to map literature effects onto the project's solver). No project code was executed
and no other file was edited.

Labels: **VERIFIED** = I opened the page (URL + year given) and the number/claim is on it.
**INFERRED** = my derivation from verified numbers, or a reading across sources; the arithmetic is shown.
**UNKNOWN** = searched and not found, or the page could not be opened (403/404/429). "UNVERIFIED" marks a
number seen only in a search snippet or press text whose primary source I could not open.
"Press" = a university press release or news article, opened, but not the paper.

## 0. Bottom line for the diagnosis

1. Every robot found that works in air, water AND on land is a 1.1-4.8 kg rotor platform with passive wheels or a
   rolling hull (TriphiBot 1.1 kg, CUHK triphibian 4.76 kg, Wukong-Omni 4.4 kg). Caltech M4 is air + land only.
   None reports transition success rates; all are finite-state machines with per-mode PID/MPC; none is learned. VERIFIED.
2. **No air-capable (triphibian) robot found crosses water -> land on the ground through a shoreline/beach.** CUHK rolls
   land -> water (passive self-righting flip); Wukong-Omni gets from water to the far shore by flying; TriphiBot rolls on
   the seabed in a tank. Land-water amphibious robots WITHOUT flight do cross shorelines (wheel-fin, propeller-leg,
   wheel-spoke-paddle designs surfaced in searches; not opened: UNVERIFIED). So the beach crossing is unshown for a
   machine that also flies, not unprecedented in robotics. UNKNOWN (not found) for triphibians.
3. Air <-> water with flapping wings works in hardware at 175 mg (Harvard RoboBee, 2017; exit needs a combustion
   impulse) and at ~250 g (MIT/EPFL FAAV, Science 2026; exit by flapping alone at ~70 deg pitch). FAAV has not closed the
   full cruise-plunge-swim-launch cycle, cannot turn, has no land mode, and runs timed open-loop sequences (press). VERIFIED/press.
4. No flapping-wing robot that flies, swims (submerged) and walks was found at any scale. The closest are insect-scale:
   RoboBee (fly + swim), RoboFly 74 mg (fly + ground + water SURFACE). UNKNOWN (not found).
5. Biology bounds the flapping option: wing-propelled pursuit divers "are flightless above ~1 kg and capable of flight
   below ~1 kg" (Elliott et al., PNAS 2013), and the 1 kg murre pays the highest flight cost measured in a vertebrate
   (146 W/kg, 31x BMR). Auks change wingbeat 3.1-3.6x between media (7.6-9.3 Hz air vs 2.5-2.9 Hz water). Heavier
   flying divers (shags 1.6-2.4 kg) swim with their feet, i.e. a separate water propulsor. VERIFIED.
6. Every working water exit uses one of four tricks: (i) an impulse (combustion/jet, peak T/W 27 to 23,000),
   (ii) a powered breach from depth reaching the surface at 0.5-2.7 m/s, with momentum carrying a 0.2-0.6 s thrust
   dead-window while the propulsor crosses the surface, (iii) take-off from a floating state with the props/lift surfaces
   clear of the water (stated air T/W 1.41-1.56), (iv) staggered propulsors (Naviator).
   Vehicles that swim submerged sit near neutral buoyancy (-8% to +3% where computable); strongly buoyant floats
   (AquaMAV, SUWAVE, jump-glider) leave by impulse or passive pivot; ballast designs pay 13-25 s per transition (Loon).
   VERIFIED (numbers per system), INFERRED (the grouping).
7. The project's own validity table says the surrogate extrapolates for k > 0.3 and at stroke reversal at low U; bird-scale
   flapping SWIMMING sits at k ~ 0.3-1.6 (INFERRED below), and every exit happens at low U. The surrogate is weakest exactly
   where a flapping triphibian must swim and leave the water. INFERRED.
8. Control in every hardware system: hand-designed mode machine + explicit medium sensing + per-medium gains or
   command->thrust maps, often pre-scripted or piloted. The only learned transition work is simulation-only and sits
   above a geometric controller (Grando et al. 2021), and the Kovac lab's Sep-2026 MSc brief lists learned air-water
   control on hardware as still to be done. VERIFIED.

Anchor recommendation (literature only, the decision is the user's): the evidence-weighted body family is a
near-neutrally buoyant multirotor (same props in air and water, medium-switched command map, props stopped on entry)
with passive wheels/rolling hull for land, air T/W >= ~1.5-2.5. A flapping arm is defensible only below ~0.5 kg, with a
geared rotary motor + crank (not positional servos), flexible wings and a ~70 deg exit, and it has no land precedent.

## 1. Triphibious / tri-modal robots (air + water + land)

| system | year | venue | mass | propulsion per medium | transition mechanism | key numbers | control approach | URL | label |
|---|---|---|---|---|---|---|---|---|---|
| TriphiBot (Li ... Gao, Cao; Zhejiang U.) | 2026 | arXiv 2602.01385, no venue | 1.1 kg | air: quadrotor (FOC drives); water: same 4 rotors, surface + submerged; seabed and land: 2 passive wheels pushed by rotor thrust; negatively buoyant | water->air from floating ("leaves the water surface in just 1.2 s"); air->water by descent; air<->land by pre-planned pitch rotation at +/-60 deg/s; no shoreline crossing | air 2 m/s, ground 3 m/s, 9 cm obstacle; underwater 0.898 N/W vs 0.265 N/W for TJ-FlyingFish; tank 5x3x1.6 m only; single demos, no success rates; source says both "designed with negative buoyancy" and that it floats on the surface before exit (unresolved) | FSM; hybrid NMPC air/ground (switch by height threshold, 200 Hz) + cascade PID for exit + PID underwater; no learning | https://arxiv.org/html/2602.01385 | VERIFIED |
| CUHK-Shenzhen "Novel Triphibian Robot" (Yang, Xue ... Qian) | 2022 (v2 2023) | arXiv 2211.17022 (RA-L preprint header) | 4.76 kg | 3 ducted fans (70 mm, 2.24 kgf each) for air and water surface (hovercraft); land: cylindrical body rolls | land->water: passive flip by off-centre CoG + foam; surface->air: servo pushes ducts out and rotates them 90 deg, take-off from the hovercraft state ("avoids the water-density problem"); air->land: none (future work) | static T/W ~1.41 (INFERRED 3x2.24/4.76); pool + Shenzhen lake; no speeds or endurance reported | ArduPilot tri-rotor model + manual RC; per-mode actuator allocation; automatic control future work | https://arxiv.org/html/2211.17022 | VERIFIED |
| Wukong-Omni (Liu ... Meng, Zhang; GDUT/HKUST) | 2026 | arXiv 2603.02602, no venue | 4.4 kg | one dual-output motor per arm with one-way bearings: one direction drives a 10-in air prop, the other a 180 mm "wheel-propeller" through 1:13 (wheel on land, screw underwater); tilting arms | underwater->air: props reversed, airborne 6.5 s, stable 9.3 s; ground->air ~6 s; water->land only by flying to the far shore | air 5 m/s (table 4.86), ~5 min, 942 W; water 1.2 m/s; land 1.0-1.1 m/s cruise (max 2.1 vs 3.3 m/s, text vs table); outdoor lake mission 4/4 (whole mission, not per transition); "positively buoyant" claim inconsistent with its own volume figure | FSM selecting per-mode cascaded PID (air, underwater), L1 guidance (land), manual surface; no learning | https://arxiv.org/html/2603.02602v1 | VERIFIED |
| Caltech M4 (Sihite, Kalantari, Nemovi, Ramezani, Gharib) | 2023 | Nat. Commun. 14:3323 | ~6.0 kg (Table 1: 5.6) | air: 4 shrouded props ~2.2 kgf each; land: shrouds are wheels, legs walk/crouch; **no water** | appendage repurposing (wheel <-> thruster <-> leg) | T/W ~1.5 (INFERRED 9/6.0); 8+ modes; 45 deg wing-assisted incline running | optimisation-based per-mode control + multi-modal PRM/A* planner with energy costs; autonomous switching only UAS<->UGV | https://pmc.ncbi.nlm.nih.gov/articles/PMC10300070/ | VERIFIED (air+land only) |
| Zhong, Cao, Chai, Bai tilting-rotor triphibian | 2021 | IEEE Access, doi 10.1109/ACCESS.2021.3050182 | ? | 2 tilting rotors on a quadrotor-like body + 4 legs with passive wheels | rotor tilt changes mode | performance analysis by "virtual simulation" (snippet) | ? | https://ieeexplore.ieee.org/document/9317864/ | UNVERIFIED (page empty/403) |
| Positive-buoyancy triphibian (called "BaiLong" in one search summary; name not confirmed) | 2026 | Mech. Based Des. Struct. Mach. 54(1), doi 10.1080/15397734.2026.2679620 | ? | 4 coaxial rotors + a propeller thruster; active-wheel undercarriage (snippet) | positive buoyancy for water-air transition (snippet) | ? | ? | https://www.tandfonline.com/doi/abs/10.1080/15397734.2026.2679620 | UNVERIFIED (403) |
| Rolling-gripper triphibian | 2026 | ASME J. Mech. Robot. 18(8):081001 | ? | ? | handover on water, perching (snippet) | ? | ? | asmedigitalcollection (403) | UNVERIFIED |
| "Mirs-X" | 2023 | = arXiv 2301.12344 v1 title of TJ-FlyingFish (air-water only, Section 2) | | | | | | https://arxiv.org/abs/2301.12344v1 | VERIFIED (same vehicle) |
| "Nezha" family (SJTU) | 2022-2025 | RA-L / JFR | | air + water; Nezha-SeaDart also takes off from ground standing on its tail, but has no ground locomotion (Section 2) | | | | https://nubot-uaav.github.io/ | VERIFIED (air-water, not triphibian) |

Findings from the table:
- All verified triphibians are multirotors whose land mode is a by-product of the airframe (passive wheels, rolling hull,
  shroud-wheels). None walks. INFERRED from the rows.
- Transitions are started from a prepared, quasi-static state (floating, standing, resting on the seabed) and are scripted
  by the mode machine; none is a dynamic crossing of a beach. VERIFIED per row.
- No transition success rates or repeat counts are reported for any triphibian. VERIFIED (absence in each paper).

## 2. Aerial-aquatic robots with dynamic transitions

| system | year | venue | mass | propulsion per medium | transition mechanism | key numbers | control approach | URL | label |
|---|---|---|---|---|---|---|---|---|---|
| Harvard hybrid RoboBee (Chen ... Kovac, Wood) | 2017 | Sci. Robot. 2:eaao5619 | 175 mg | same flapping wings (2 piezo actuators): air 220-300 Hz, water 9-13 Hz | entry: needs surfactant (Bond number 0.7 -> 2); exit: electrolysis fills a 33 mg chamber so buoyancy lifts wings out, then a spark detonates the oxyhydrogen | surface tension 6.6-18.8 mN vs max lift 3.1 mN; detonation 7.5-9 N for 1 ms = "thrust-to-weight ratio of 19,000 to 23,000"; take-off 1.8-3.4 m/s, 37 cm ballistic jump; lift/weight in air ~1.26 (INFERRED 220/175 mg) | Vicon + off-board adaptive control in air (tethered); swimming open loop; cannot fly after exit | https://smrl.mit.edu/wp-content/uploads/2019/10/science_robotics_fly_swim_robobee.pdf | VERIFIED |
| MIT/EPFL FAAV (Zufferey ... Ijspeert, Floreano) | 2026 | Science 393:207-211, doi 10.1126/science.aeb6744 | < 300 g (press: 250 g) | same flexible flapping wings, one motor + crankshaft, motorised tail; no feet, no wing folding | exit: from ~0.5 m depth, swim up and flap out at ~70 deg pitch so wingtips clear on the upstroke; < 1 s, ~8-10 strokes (press); plunge diving demonstrated | swim ~1 m/s, fly ~6-6.3 m/s; ~5 Hz both media (MIT News) / 5-11 Hz air (press); wings bend up to 90% underwater (press); 80 cm wings best of 60/80/100 cm; break-even ~15 m swim vs fly (press) | pre-programmed timed sequences (frequency, tail angle); no autopilot, no turning, full cycle not yet closed (press) | https://news.mit.edu/2026/new-flapping-robot-swims-and-flies-like-diving-bird-0709 ; https://dspace.mit.edu/handle/1721.1/169536 | VERIFIED (abstract + MIT News); detail numbers press-level (paper PDF hit HTTP 429 three times) |
| Imperial jump-glider (Zufferey ... Kovac) | 2019 | Sci. Robot. 4:eaax7330 | 160 g | no propeller: CaC2 + water -> acetylene, combustion water jet; delta-wing glide | floats passively at the 47 deg launch angle, fires when the IMU sees it | 6.5 bar in 25 ms -> 10 m/s, 26 m flight; max jet thrust 43 N (peak T/W ~27, INFERRED); jet ~200 ms; 22 flights; fuel for 10 launches | open loop, predetermined timing | https://nubot-uaav.github.io/static/pdf/squid-like%20aavs/Consecutive%20aquatic%20jump-gliding%20with%20water-reactive%20fuel.pdf | VERIFIED |
| Imperial AquaMAV (Siddall, Ortega Ancel, Kovac) | 2017 (concept 2014) | Interface Focus 7:20160085 | 201.3 g | fixed wing + propeller in air; CO2-powered water jet for take-off; wings sweep back 90 deg | plunge dive with wings retracted (< 0.5 s); 1 s jet burst to leave the water | retracted: drag -62%, lift -92%; field impact ~9.1 m/s; displaced water 265 g (floats, +32%, INFERRED); 10 m/s cruise, 14 min | not described in detail | https://pmc.ncbi.nlm.nih.gov/articles/PMC5206604 | VERIFIED |
| ETH Dipper (Rockenbauer, Jeger ... Siegwart) | 2021 | RSS XVII, paper 048 | UNKNOWN | fixed wing with actively swept wings; press: one motor, clutch selects tractor prop (air) or rear screw (water) | plunge dive with wings folded; repeated robust transitions (abstract) | press only: 150 m dive at 130 km/h, swims 10 min at 9 km/h | UNKNOWN | https://roboticsproceedings.org/rss17/p048.html | VERIFIED abstract; numbers UNVERIFIED (PDF > 10 MB) |
| SUWAVE (Peloquin, Thibault, Lussier Desbiens) | 2017 | IEEE RA-L 2(2):381-388 | 584 g | flying wing, one 11x6 folding prop; floats flat | passive take-off: SMA pin releases the centre body, which pivots under its weight until the prop points up; success iff it re-latches at 40-60 deg; lands by nose-first dive | T/W = 1.56 (stated); latch at ~0.4 s, 1.75 m/s; 0.7 g peak take-off accel; landing 15 g at 10 m/s, 0.6 m deep | passive take-off, RC flight | https://nubot-uaav.github.io/static/pdf/fixed-wing%20aavs/Design_of_a_Passive_Vertical_Takeoff_and_Landing_Aquatic_UAV.pdf | VERIFIED |
| Loon Copter (Alzu'bi, Mansour, Rawashdeh) | 2018 | J. Field Robot. 35(5):764-778 | 2.7 kg | same direct-drive props in air and water; ballast pump + 150 ml cylinder | flooding tilts it 90 deg and sinks it (props horizontal); emptying floats it upright; takes off from the surface | air->water ~25 s, water->air ~13 s; ballast 6 W vs rotors 450 W; flight 10.5-12 min; underwater 22 min at 0.5 m/s | closed loop in air/surface, open loop underwater | https://nubot-uaav.github.io/static/pdf/multirotor_based_aavs/Journal%20of%20Field%20Robotics%20-%202018%20-%20Alzu'bi%20-%20Loon%20Copter%20%20Implementation%20of%20a%20hybrid%20unmanned%20aquatic%20aerial%20quadcopter%20(1).pdf (doi 10.1002/rob.21777) | VERIFIED |
| Rutgers Naviator (Maia, Soni, Diez-Garias) | 2015 | arXiv 1507.01932 | < 2 kg | octo-quad: 4 arms x 2 stacked counter-rotating air props, all used in both media; quasi-neutral buoyancy | the prop at the interface slows/stops while its partner keeps thrusting, re-accelerates when clear: "under two seconds" | underwater 0.3-1 m/s, 0.5-10 N per prop, < 250 rpm, no cavitation; pitches 65-75 deg to swim | per-rotor throttle scheduling at the interface + PD pitch | https://arxiv.org/pdf/1507.01932 | VERIFIED |
| TJ-FlyingFish / Mirs-X (Liu ... Ben M. Chen) | 2023 | arXiv 2301.12344, ICRA 2023 (listing) | 1.63 kg | one prop per arm through a dual-speed planetary gearbox (forward direct for air, reverse 12.33:1 for water); tilting arms vector thrust underwater | (exit not detailed in the paper) | "aerial thrust to weight ratio: 3.75"; hover 6 min, ~40 min underwater; designed under-buoyant; 32 N per 122 g unit underwater, 0.265 N/W | stock PX4 cascade PID + RC; pilot switches propulsion mode | https://arxiv.org/abs/2301.12344 | VERIFIED |
| Beihang/Imperial remora hitchhiker (Li ... Kovac, Wen) | 2022 | Sci. Robot. 7(66):eabm6695 | 950 g | quadrotor with passively self-folding 5-in props (85 mm folded by water, 170 mm unfolded by centrifugal force) | rises from underwater and accelerates the props as they break the surface | exit 0.35 s at 100% throttle; minimum throttle for exit 35% (self-folding) vs 50% (stock props); 7 consecutive transitions in 20.3 s; exit 0.43 s, entry 0.13 s; one cycle = 1.9x hover power; field: mountain stream, ocean | remote piloted, standard flight controller | https://spiral.imperial.ac.uk/handle/10044/1/96997 | VERIFIED |
| Nezha-SeaDart (Jin, Zeng, Lian; SJTU) | 2025 (online 2024) | J. Field Robot., doi 10.1002/rob.22399 | 1.5 kg | tail-sitter delta wing (400 mm span) + four 5.1-in props used in air AND water; no buoyancy control (1550 g buoyancy vs 1500 g weight) | fast exit: dive to depth, pitch ~80-90 deg nose up, accelerate, break the surface on momentum; fast entry: nose-down dive with props stopped | 36 N static thrust (T/W ~2.45, INFERRED); at the surface 2.66 m/s (design estimate 4.77 m/s); thrust lost near the surface from t = 0.2 s, regained t = 0.6 s, clear at 0.8 s; underwater 3 m/s max; air 35 m/s; lake field test, 6 filmed exits | PX4 PID + Arduino mission computer; RC channel switches to underwater mode | https://nubot-uaav.github.io/static/pdf/hybrid_aavs/2025-Nezha-SeaDart%20A%20tail-sitting%20fixed-wing%20vertical%20takeoff%20and%20landing%20hybrid%20aerial%20underwater%20vehicle.pdf (doi 10.1002/rob.22399) | VERIFIED |
| JHU APL delta wing (Moore) | 2019 | arXiv 1906.01532 | 375 g | one 6x4 prop for both media; "buoyancy assist" (CoB aft, unstable underwater) | dynamic exit from underwater cruise "in less than 0.5s" | 8/20 hardware exits reached controlled flight (40%); 7 flipped back, 4 fell forward; suggests ~70 deg exit pitch | trajectory optimisation + mode-dependent TVLQR + guard LQR + hybrid EKF estimator (prop speed jump ~14x = medium sensor) | https://ar5iv.arxiv.org/html/1906.01532 | VERIFIED |
| Berkeley mini UAUV (Zha ... Mueller) | 2019 | HiPeRLab PDF (venue not on page) | 202 g | plain quadcopter, same props both media; ~8% negatively buoyant (INFERRED: 186e-6 m^3 vs 0.202 kg) | rise from 0.15 m at 1 m/s commanded; switch the command->thrust map 0.08 s after reaching the transition region (timer) | breaks the surface at ~0.5 m/s, then stalls on the surface ~0.2 s before climbing; air entrainment makes the region "a highly unsteady flow field"; model misses drag and added mass underwater | one controller, medium-switched thrust map, mocap | https://hiperlab.berkeley.edu/wp-content/uploads/2019/05/Towards-breaching-a-still-water-surface-with-a-miniature-unmanned-aerial-underwater-vehicle.pdf | VERIFIED |
| ZS-Puffin (Wang ... Wang; SYSU) | 2025 | arXiv 2508.08690 | 1.61 kg | 2 rotors fly; servo-tilted wings give lift in air and FLAP (~2.4 Hz) for thrust underwater; slightly positive buoyancy | belly-sitter exit ~3 s (fuselage out at 2.7 s, flight at 2.9 s) | underwater 0.63 m/s with rotors vs 0.21-0.29 m/s flapping | cascaded PID air, open-loop pilot mixing underwater, CPG for flapping | https://arxiv.org/html/2508.08690v1 | VERIFIED |
| Diving-beetle AALR (Hu ... Dong) | 2024 | arXiv 2409.09572 (IROS 2025 per listing) | 1.732 kg | quadrotor whose arms melt (low-melting-point alloy joints) into beetle-leg paddles | aquatic tests only | max swim 77 mm/s | not detailed | https://arxiv.org/pdf/2409.09572 | VERIFIED |

Water-exit strategies (INFERRED grouping of the verified rows):

| strategy | systems | what makes it work | numbers |
|---|---|---|---|
| impulse (combustion, gas jet) | RoboBee, jump-glider, AquaMAV | stored energy released in ms, insensitive to the submerged propulsor problem | RoboBee T/W 19,000-23,000 for 1 ms; jump-glider peak ~27 for ~200 ms, 10 m/s |
| powered breach from depth | Nezha-SeaDart, JHU delta, Berkeley, remora, FAAV | body carries kinetic energy through the dead window while the propulsor crosses the surface | surface speed 0.5-2.7 m/s; dead window 0.2-0.6 s; pitch 70-90 deg; 40% success (JHU), stall 0.2 s at 0.5 m/s (Berkeley); remora 0.35 s at full throttle |
| take-off from a floating state | Loon, CUHK triphibian, TriphiBot, Wukong-Omni, SUWAVE, ZS-Puffin, SeaDart slow mode | float the lift surfaces/props clear of the water first, then throttle up | air T/W 1.41 (CUHK, INFERRED) and 1.56 (SUWAVE, stated) where available; 1.2-13 s |
| staggered propulsors | Naviator | one prop of each pair is always clear of the interface | < 2 s |

## 3. Flapping-wing specifics

| system | year | venue | mass | media | flapping frequency | actuator | key numbers | control | URL | label |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid RoboBee | 2017 | Sci. Robot. | 175 mg | air + water (+ water exit) | air 220-300 Hz, water 9-13 Hz (ratio 17-33x, INFERRED) | piezo bimorphs | wing area cut 40% to raise the air frequency | Vicon, off-board | see Section 2 | VERIFIED |
| RoboFly (Chukewad ... Fuller) | 2020 | arXiv 2001.02320 | 74 mg | air + ground + water surface | not on abstract | piezo | the only flapping robot found that flies and moves on the ground and on water (surface) | — | https://arxiv.org/abs/2001.02320 | VERIFIED abstract |
| FAAV | 2026 | Science | < 300 g | air + water (+ plunge, exit) | ~5 Hz both (MIT News); 5-11 Hz air (press) | one motor + crankshaft | flexible wings essential | timed sequences | see Section 2 | VERIFIED/press |
| T3 triple-flapping-wing (Xu ... Lu) | 2025 | arXiv 2503.00805 (IROS submission) | not on abstract | air + ground | — | 3 independently actuated wing pairs | ground motion from wing vibration through passive elastic legs | SE(3) controller switches modes | https://arxiv.org/abs/2503.00805v1 | VERIFIED abstract |
| E-Flap (Zufferey ... Ollero) | 2021 | IEEE RA-L 6(2):3097 | 0.510 kg (+ up to 0.52 kg payload) | air | cruise ~4 Hz, max 5.5 Hz (5.6 +/- 0.2 at full payload) | 20 g 1130 Kv brushless (150 W rated) + 42:1 two-stage gearbox + 4-bar = 71 g | 1.5 m span; 2-6 m/s; peak power estimate 130 W (~255 W/kg of vehicle, INFERRED); 5-15 min | pitch PI on Optitrack | https://zenodo.org/record/4917204 | VERIFIED |
| RoBird Bald Eagle / Falcon; RoboRaven | — | cited in E-Flap | 2.1 kg (+1 kg payload); RoboRaven 0.2-0.7 kg | air | — | — | Bald Eagle 18 m/s, Falcon ~16 m/s, RoboRaven ~6 m/s | — | via E-Flap | VERIFIED as cited by E-Flap |
| Festo SmartBird | 2011 | Festo | 0.40-0.45 kg | air | — | brushless + gear/lever | ~2 m span, ~23-25 W | — | Festo pages (not opened) | UNVERIFIED |
| DelFly Nimble (TU Delft) | 2018 | Science 2018 per search (not opened) | 29 g | air | ~17 Hz hover | UNKNOWN (not on page) | 33 cm span, > 5 min, 7 m/s, T/W > 1.3 | onboard | https://delfly.nl/ | VERIFIED (numbers) |
| ZS-Puffin | 2025 | arXiv | 1.61 kg | rotor flight, flapping swim | water 2.4 Hz | servos (tilt) + rotors | flapping swim slower than rotor swim (0.21-0.29 vs 0.63 m/s) | CPG | see Section 2 | VERIFIED |
| MIT dual-media flapping foil (Izraelevitz, Triantafyllou) | 2015 | ICRA | bench | air + water | — | adds an in-line DOF | one wing spans both force ranges (bench only) | — | https://dspace.mit.edu/handle/1721.1/108693 | VERIFIED abstract |
| Bristol guillemot wing model (Lock ... Loveless) | 2010 | Bioinsp. Biomim. 5:046001 | model | air + water | — | — | retracting the wing underwater saves power (measured); press: bent wings cut profile drag up to 50% | — | https://research-information.bris.ac.uk/en/publications/development-of-a-biologically-inspired-multi-modal-wing-model-for/ | VERIFIED abstract; 50% UNVERIFIED |

Scaling and power (numbers VERIFIED where cited; arithmetic INFERRED):
- Wingbeat formula used to size E-Flap, from Pennycuick: f = zeta m^(3/8) g^(1/2) b^(-23/24) S^(-1/3) rho^(-3/8)
  (E-Flap RA-L 2021, citing Pennycuick; Pennycuick 1996 itself not opened). For geometric similarity (b ~ m^1/3,
  S ~ m^2/3) the exponents give f ~ m^(-1/6) (INFERRED: 3/8 - 23/72 - 2/9 = -1/6). Scaled from E-Flap's ~4-5 Hz at
  0.5 kg: ~6-7 Hz at 0.1 kg, ~3 Hz at 10 kg (INFERRED, E-Flap geometry). Birds with diving-compromised wings flap
  faster: guillemots 7.6-9.3 Hz at 0.95 kg in air (Sato et al. 2007).
- Power needed: murre in flight 146 W/kg body (metabolic; Elliott 2013, VERIFIED). Electrical power of working robots in
  forward flight or hover: E-Flap ~255 W/kg peak (estimate), Nezha-SeaDart 257 W/kg hover / 265 W/kg at 24.7 m/s,
  Wukong-Omni 214 W/kg, Loon Copter 167 W/kg (INFERRED: W / kg from verified numbers). Bird flight muscle delivers
  ~60-150 W/kg muscle cruising and ~400 W/kg (quail pectoralis) at take-off (UNVERIFIED: snippets of Askew & Marsh 2002 et al.).
- What servos deliver: Hitec HS-645MG 55.2 g, 0.94 N m, 0.20 s/60 deg; Dynamixel XM430-W350 82 g, 4.1 N m, 46 rpm
  (VERIFIED specs). Peak mechanical power ~tau_s*omega_0/4 = 1.2 W and 4.9 W, i.e. ~22 and ~60 W/kg; the largest
  +/-30 deg sinusoid they can follow even unloaded is ~1.6 and ~1.5 Hz (INFERRED). A flapping stroke at 4-10 Hz and
  +/-30-50 deg is out of reach of positional servos; every verified flapping FLYER above insect scale drives the stroke
  with a geared rotary motor and a crank (E-Flap: 20 g motor rated 150 W, ~7.5 kW/kg motor-only, INFERRED; FAAV: one
  motor + crankshaft). Servo-driven flapping appears only underwater and slowly (ZS-Puffin, ~2.4 Hz).
- Is "gliding plus rotor assist" the norm at kg scale? For aerial-aquatic machines at >= 0.375 kg, yes: every one found is
  a multirotor or a fixed wing/tail-sitter with propellers (Nezha-SeaDart, Dipper, SUWAVE, JHU, AquaMAV, ZS-Puffin).
  Flapping-only flight exists at 0.5-2.1 kg (E-Flap, RoBird) but none of those swims. Flapping air+water exists only at
  <= 0.3 kg (FAAV) and 175 mg (RoboBee). VERIFIED per system; the generalisation INFERRED.

## 4. Animals as existence proofs, and what they pay

| animal | mass | media | propulsion per medium | transitions | key numbers | compromise paid | URL | label |
|---|---|---|---|---|---|---|---|---|
| Diving beetles (Dytiscidae: Eretes griseus, Cybister japonicus 35-40 mm, Dytiscus) | ~grams (UNKNOWN exact) | air, water, land | air: hind wings (elytra open); water: drag-based hind-leg rowing; land: walking | Eretes "takes off directly from the water surface"; Cybister "pronounced flight capability when leaving the water"; hind wings unfold hydraulically in ~0.74 s (free) to 3-3.7 s (instrumented) | CFD of Eretes take-off: SLAPPING the water gives lift/weight max 18, mean 9.2; paddling 0.46/0.23; surface effect +25% lift below h/c 0.44 | sub-elytral air store makes them buoyant (Britannica snippet, UNVERIFIED); scale ~1e-3 of the project's bodies | https://www.fujipress.jp/jrm/rb/robot003600051122/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4979654/ | VERIFIED (2024 CFD abstract; 2016 paper); "must climb out to fly" UNKNOWN |
| Whirligig beetles | small | water surface -> air | — | water-to-air transition, not impulsive (Chen 2017 remark) | — | — | RoboBee paper | VERIFIED as stated there |
| Water-lily beetle | 2.2 mg | water surface + air | wings | interfacial flight | surface tension up to ~5x weight; needs lift/weight ~3.4 vs 1-2.5 airborne; 116 Hz | trapped by the meniscus | https://ar5iv.labs.arxiv.org/html/1405.2298 | VERIFIED |
| Auks: common/Bruennich's guillemot (murre), razorbill, auklet | 0.53-1.0 kg | air, water (surface, land at colonies) | the same wings in both media | murres run/patter across the water to take off (UNVERIFIED snippets); dive from the surface | flight 7.6-9.3 Hz vs swim 2.5-2.9 Hz; flight cost 141 W = 146 W/kg = 31x BMR, "highest flight costs recorded for vertebrates"; "flightless above ~1 kg"; great auk ~5 kg flightless | highest wing loading of any flying bird; dive cost still 30% above a same-size penguin | https://pmc.ncbi.nlm.nih.gov/articles/PMC3677478 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC1766382 | VERIFIED |
| Bruennich's guillemot underwater | 0.95 kg | water | wings | — | 3.2 Hz with both strokes against buoyancy near the surface, 2.4 Hz downstroke-only deeper, passive ascent from 18 m at 2.3 m/s | buoyancy is a cost going down, free propulsion coming up | https://pmc.ncbi.nlm.nih.gov/articles/PMC1691262 | VERIFIED |
| Northern gannet | ~3 kg (UNVERIFIED) | air, water (plunge) | flap/glide in air; momentum, then wings/feet underwater | plunge dive; returns using positive buoyancy, flapping, hydrophobic feathers (Zufferey 2019 intro) | impact up to 24 m/s; neck would buckle only at ~80 m/s; depth ~10-11 m by momentum (UNVERIFIED) | streamlined skull/neck, pre-contraction | https://pmc.ncbi.nlm.nih.gov/articles/PMC5087034 | VERIFIED (24/80 m/s) |
| American dipper | sparrow-sized (mass UNVERIFIED) | air, water, stream bed | wings underwater; walks the bed gripping with claws | dives 5-15 s | — | dense plumage, preen oil; heavy for its wing size (snippet) | https://www.nps.gov/articles/000/klmn_american-dipper.htm | VERIFIED (behaviour) |
| Shags / cormorants (foot-propelled) | 1.6-2.4 kg | air, water | feet underwater, wings in air | — | flight 5.7-5.8 Hz; swim 2.9-3.3 Hz (feet); pelagic cormorant 1.8 kg, 158 W in flight (87 W/kg), dive cost > 3x a same-size penguin | high cost in both media | PMC1766382, PMC3677478 | VERIFIED |
| Flying fish (Exocoetidae) | UNKNOWN (not on opened pages) | water, air (glide) | body/caudal fin underwater, taxi with tail lobe, glide on fins | accelerate underwater to ~10 m/s before leaving (Zufferey 2019 intro) | > 40 s, up to 400 m, ~70 km/h; L/D like petrels and wood ducks; best near the surface (ground effect) | no powered flight | https://sciencedaily.com/releases/2010/09/100910082308.htm | VERIFIED (press of Park & Choi JEB 2010) |
| Penguins (endpoint) | 1.1-24.5 kg | water, land | wings | — | swim 1.35-3.6 Hz (Sato) | flightless | PMC1766382 | VERIFIED |

What nature says (INFERRED from the rows): animals that fly, swim submerged and walk with the SAME wing as the
water propulsor are all <= ~1 kg (beetles use legs, not wings, in water; dippers and auks use wings); that solution
costs the highest flight power of any vertebrate at ~1 kg and is abandoned above it. Heavier flying divers keep a
SEPARATE water propulsor: foot-propelled shags/cormorants at 1.6-2.4 kg fly at 5.7-5.8 Hz and swim with their feet
(Sato 2007, VERIFIED); the FAAV press text reports the paper's reading that most diving birds still paddle with feet and
only light-bodied fliers skip it (press). The animals change frequency 3-4x between media, bend or partly fold the wing
underwater, use buoyancy as a free ascent, and leave the water by running/pattering or by impulsive slapping
(beetles), not by a gentle lift-off.

## 5. Modelling: when quasi-steady is acceptable and what it misses

Reduced frequency k = pi f c / U (assumed chords marked; INFERRED):

| case | f (Hz) | c (m) | U (m/s) | k | project validity table (`docs/model_validity.md` l. 48-61) |
|---|---|---|---|---|---|
| FAAV-like swimming | 5 (MIT News) | 0.10 (assumed) | 1.0 | ~1.6 | extrapolating (k > 0.3) |
| FAAV-like flight | 5-11 (press) | 0.10 (assumed) | 6.3 | 0.25-0.55 | usable +/-30% to extrapolating |
| guillemot swimming | 2.58 (Sato) | 0.07 (assumed, wing partly flexed) | 1.8 (assumed) | ~0.32 | extrapolating |
| guillemot flight | 9.33 (Sato) | 0.09 (assumed) | 18 (assumed) | ~0.15 | usable +/-30% |
| E-Flap slow flight | 4 | 0.25 (assumed mean; 0.36 m root on page) | 2-5 | 0.63-1.6 | extrapolating |
| any exit or take-off | — | — | 0.5-2.7 at the surface (Section 2) | U -> 0 at reversal | "stroke reversal at low U: extrapolating" |

Rule-of-thumb thresholds: quasi-steady for k <= ~0.05, unsteady above, "highly unsteady" k > 0.2 (Wikipedia, secondary;
UNVERIFIED as a primary threshold). A 2026 wind-tunnel study reports the quasi-steady approximation "is not valid for a
large range of frequencies where it is normally applied" (Malarczyk & Hultmark, arXiv 2608.13424, VERIFIED abstract).
Counterweights: Sane & Dickinson (JEB 205:1087, 2002) show a QS model with translational + rotational + added-mass terms
tracks the time course of flapping forces better than translation-only, with wake capture as the residual it cannot
model (VERIFIED abstract, https://authors.library.caltech.edu/13751); Whitney & Wood (JFM 660:197, 2010) report strong
agreement of a blade-element model with insect-scale measurements without fitted parameters (VERIFIED abstract);
the RoboBee team's dynamical swimming model matched measured ascent speed and pitch stability (Chen 2017, VERIFIED).

| effect | literature evidence | magnitude | in the project solver? | consequence for the search |
|---|---|---|---|---|
| unsteady circulation / LEV growth and shedding | Theodorsen-type attenuation; LEV heuristics | project doc: QS over-predicts circulatory lift 18-30% at k 0.1-0.3 | LEV as an instantaneous function of pitch rate | swimming flappers are scored with a fitted LEV term |
| wake capture, wing-wake interaction | Sane & Dickinson residual; CFD comparisons (snippets: errors up to ~20%) | — | not modelled | — |
| added mass in water | Berkeley: underwater tracking error from unmodelled added mass; water is ~800-1000x denser | dominates low-U acceleration | strip + bluff, diagonal only; jointless machines receive none (F-01) | rigid jointless bodies accelerate too easily underwater (check) |
| propulsor crossing the surface: ventilation, air entrainment, thrust collapse | Berkeley (0.2 s stall, "highly unsteady"), SeaDart (thrust lost 0.2-0.6 s), Naviator (slow the prop at the interface) | 0.2-0.6 s of ~zero thrust | ventilation "not modelled at all" | simulated exits may be easier than real ones (check) |
| water-surface ground effect | Eretes CFD +25% lift below h/c 0.44; flying-fish L/D up near the surface | +10-25% | ground effect not modelled | surface take-off by flapping may be harder in sim than in reality |
| slamming at entry | SUWAVE 15 g at 10 m/s, fitted m a = C_B rho g V_s + C_V A_f v^2 - m g; plunge force F ~ pi rho V^4 tan^3(beta/2) t^2 (Giordano et al. 2024) | ~v^2 to v^4 | "diagnostic only, not fed to the dynamics" | plunge entries cost nothing in sim (check) |
| surface tension | RoboBee: 6.6-18.8 mN vs 3.1 mN lift at 175 mg; Bond number 0.7 | dominant below ~1 g, negligible at 0.1-10 kg (INFERRED, Bond ~ L^2) | — | irrelevant at project scale |
| water carried out of the water | SUWAVE 0.14 kg in the centre body draining over 0.65 s; RoboBee 30-45 mg residue (~20% of mass) | 5-20% of mass for ~1 s | UNKNOWN (not in the doc) | — |
| wing flexibility | FAAV: flexible wings are one of the three enablers; wings bend up to 90% underwater (press) | first-order for flapping swim | rigid; optional 1-DOF chordwise pitch hinge only, no spanwise bending | the FAAV mechanism is only partly representable |
| buoyancy at partial immersion | Berkeley prism model f_B = rho g A (h + z_s - z_t) | — | rho g V f_sub (modelled) | — |
| jet propulsion energy | every real jet exit needed CO2 or combustion (AquaMAV, jump-glider, RoboBee) | — | jet pumping work not charged (J-01) | if jets are in the design space, a free jet is a candidate exploit (check) |
| MuJoCo's own fluid model | "Proper simulation of fluid dynamics is beyond the scope of MuJoCo"; one global medium, no free surface; inertia-box model has no added mass | — | the project uses its own solver instead | — |

## 6. Control practice in the working systems

| system | mode structure | medium sensing | per-medium control | learned? | label |
|---|---|---|---|---|---|
| TriphiBot | FSM | height threshold, rotor speed changes | hybrid NMPC air/ground, cascade PID exit, PID underwater | no | VERIFIED |
| Wukong-Omni | FSM with preset joint angles | — | cascaded PID, L1 guidance, manual surface | no | VERIFIED |
| CUHK triphibian | per-mode actuator allocation | pilot | ArduPilot + RC | no | VERIFIED |
| M4 | per-mode controllers + PRM/A* | — | optimisation-based | no | VERIFIED |
| Nezha-SeaDart | RC switch to underwater mode; scripted dive/pitch-up/breach | depth sensor | PX4 PID; Arduino depth/heading loops | no | VERIFIED |
| JHU delta | hybrid automaton, mode-dependent TVLQR, guard LQR | prop speed jump ~14x | trajectory optimisation | no | VERIFIED |
| Berkeley mini UAUV | one controller, switched command->thrust map, timer-triggered | pressure (fails at the interface), timer | same law both media | no | VERIFIED |
| Naviator | per-rotor throttle schedule at the interface | — | PD | no | VERIFIED |
| Loon Copter | ballast-driven transitions | — | closed loop air/surface, open loop underwater | no | VERIFIED |
| TJ-FlyingFish | pilot channel selects propulsion mode | — | PX4 cascade PID | no | VERIFIED |
| remora robot, CUHK, SUWAVE | piloted; SUWAVE take-off passive | — | standard FC | no | VERIFIED |
| RoboBee, jump-glider, FAAV | scripted open-loop sequences | IMU angle trigger (jump-glider) | — | no | VERIFIED / press |
| Grando et al. HUAUV (ICRA 2021) | DDPG/SAC policy -> velocity setpoints -> geometric tracking controller | — | Gazebo (RotorS + UUV simulator) only | yes, navigation level, SIMULATION ONLY; geometric baseline 100% vs 96-97% (DDPG) on the simple scene | VERIFIED |
| DoCRL, shared lidar/sonar policy (Grando group, 2023/2026) | navigation-level DRL | — | simulation | yes, sim | VERIFIED abstracts |
| Wang & Shang 2025 | three-zone hybrid model, twisting sliding-mode across the interface | — | cascade PID per medium | no | VERIFIED abstract |
| Empa/EPFL LSR MSc brief (Sep 2026) | plan: deep RL + sim-to-real vs gain-scheduled PID and MRAC | — | GPU sim "with added mass, buoyancy, interface effects" | planned, not done | VERIFIED |

Answer (INFERRED from the table, high confidence): every working transition is commanded by a hand-designed mode machine
with explicit medium detection and medium-specific gains or thrust maps, usually from a prepared state and often scripted
or piloted. No hardware system uses a single learned policy for a medium transition; learned work exists only in
simulation and only above a classical controller.

## 7. Synthesis

### 7.1 What works in reality, ranked by evidence

1. **Field-tested, repeated transitions (strongest):** Nezha-SeaDart (lake field test, 6 filmed momentum exits, air + water,
   ground VTOL); remora hitchhiker (7 consecutive crossings in 20.3 s; stream and ocean); Wukong-Omni (outdoor lake
   mission 4/4, air/water/land, water->land by flying); Loon Copter (outdoor, ballast); CUHK triphibian (lake). All rotor-propelled.
2. **Lab-repeatable:** Naviator (< 2 s), SUWAVE (repeated passive take-offs), jump-glider (22 flights), RoboBee (repeatable
   combustion exit), FAAV (exit and plunge shown separately), TriphiBot (tank, single demos), TJ-FlyingFish.
3. **Partial success:** JHU delta 8/20 exits; Berkeley breach with a 0.2 s surface stall; ZS-Puffin flapping swim slower
   than its rotors; AALR beetle-leg paddling 77 mm/s at 1.7 kg.
4. **Simulation or concept only:** Zhong 2021 tilting-rotor triphibian (snippet), Grando et al. RL, Lock 2010 model,
   Izraelevitz bench foil.

### 7.2 Thresholds a simulated triphibian must meet (envelopes from working systems, not single numbers)

- **Air T/W:** 1.41 (CUHK, from a floating hovercraft state, INFERRED from fan ratings), 1.5 (M4, air+land, INFERRED),
  1.56 (SUWAVE, stated), ~2.45 (Nezha-SeaDart, INFERRED), 3.75 (TJ-FlyingFish, stated). Among water-launched vehicles that
  state or allow T/W, the lowest is 1.41; RoboBee's air lift/weight of ~1.26 needs a combustion impulse to leave (INFERRED).
- **Water exit:** either an impulse (T/W 27 for ~0.2 s to 2e4 for 1 ms), or a powered breach that reaches the surface at
  0.5-2.7 m/s with pitch 70-90 deg and survives 0.2-0.6 s of near-zero propulsor thrust, or a floating start with the
  lift surfaces/props clear of the water (1.2-13 s). The remora robot needed >= 35% throttle with folding props, 50% with
  stock props. A flapping exit needs ~70 deg pitch so the wingtips clear on the upstroke (FAAV, press), and the tail close
  to the body.
- **Buoyancy:** near neutral: Berkeley -8% (INFERRED), TJ and TriphiBot negative, Naviator quasi-neutral, SeaDart +3%,
  ZS-Puffin slightly positive. Strongly positive floats (AquaMAV +32% INFERRED, SUWAVE, jump-glider) must use an impulse
  or a passive pivot to leave; ballast designs pay 13-25 s per transition (Loon).
- **Water entry:** stop the props (SeaDart), sweep or fold the wings (AquaMAV, Dipper), or dive nose-first; loads ~15 g at
  10 m/s for a 0.58 kg wing (SUWAVE); gannets enter at up to 24 m/s.
- **Flapping kinematics across media:** wingbeat must drop 3-4x from air to water (auks) or the wing must bend away most of
  its stroke underwater (FAAV); insect scale drops ~17-33x (RoboBee). Cruising animals sit at Strouhal 0.2-0.4 in both media
  (Taylor, Nudds & Thomas, Nature 425:707, 2003: UNVERIFIED, search snippet only).
- **Actuation:** the stroke needs a geared rotary motor + crank (E-Flap: 20 g motor rated 150 W for a 0.5 kg bird); vehicle
  electrical power in flight is ~170-265 W/kg across working robots (INFERRED); positional servos give ~20-60 W/kg and
  <= ~1.6 Hz at +/-30 deg (INFERRED from specs).
- **Land <-> water:** the only land->water crossing demonstrated by a flying machine is passive (roll in, self-right by CoG
  offset, CUHK); water->land by ground locomotion is not demonstrated by any air-capable robot found (UNKNOWN). Ground-only
  amphibious robots (wheel-fin, propeller-leg, wheel-spoke-paddle; titles surfaced in searches, not opened: UNVERIFIED) are
  the nearest hardware reference for calibrating a simulated beach crossing.
- **Land -> air:** Wukong-Omni ~6 s ground -> air with servo-tilted arms on a sinusoidal servo profile (avoids stall
  currents); TriphiBot pre-planned pitch rotation at +/-60 deg/s across a 90 deg attitude change; Nezha-SeaDart VTOL from
  standing on its empennage; M4 swings its legs so the wheels become thrusters; RAVEN's leg jump "contributes
  substantially to the initial flight take-off speed" (abstract); CUHK has no air -> land strategy. All VERIFIED. Ground
  take-off for a kg-scale flapper (E-Flap/RoBird class): UNKNOWN in the notes. Animals: Cybister unfolds its hind wings in
  ~0.74 s (free) before flight (VERIFIED); murres run on the water to take off (UNVERIFIED).

### 7.3 Is a flapping-only triphibian plausible at 0.1-10 kg?

- Flapping flight at 0.5-2.1 kg: proven (E-Flap 0.51 kg, RoBird 2.1 kg). VERIFIED.
- Flapping flight + swimming + water exit with the same wings: proven only at 175 mg and ~0.25 kg, the latter without a
  closed mission cycle, turning or autonomy. VERIFIED/press.
- Flapping + land locomotion: only at insect scale (RoboFly) or with passive vibration legs (T3); no bird-scale flapping
  robot walks. VERIFIED (absence: UNKNOWN beyond what was searched).
- Biology: the one-wing-for-both-media solution is used up to ~1 kg and abandoned above it; its price at 1 kg is the
  highest flight cost measured in a vertebrate. Above ~1 kg, flying divers swim with a separate propulsor (feet: shags
  1.6-2.4 kg). The diving beetle, the namesake triphibian, also rows with its legs, and the 1.7 kg robot copy of that
  stroke reached 77 mm/s. VERIFIED.
- Therefore (INFERRED): a same-wing flapping triphibian is physically plausible below ~1 kg, unprecedented in machines
  above ~0.3 kg, and unprecedented in nature above ~1 kg. Flapping flight plus a separate leg/foot water propulsor has a
  natural precedent to at least ~2.4 kg but no machine precedent. In the 1-10 kg part of the project's range every working
  three-medium machine uses rotors. Whether a flapping body is plausible *in the simulation* depends on what the simulated actuator is
  allowed to deliver (frequency, W/kg, bandwidth) and on the k > 0.3 regime the surrogate extrapolates in; that is a check
  for the diagnosis, not a verdict the literature can give.

### 7.4 Mapping to PREREG rivals (for the diagnosis lead)

- **R1 physics:** the literature supports rotor-based triphibians at 1-5 kg and flapping air-water machines <= 0.3 kg; it
  gives no precedent for a flapping triphibian, nor for an air-capable robot crossing water->land on the ground (ground-only
  amphibious robots do cross shorelines, UNVERIFIED). If the reachable design space is mostly flapping bodies above
  ~0.5 kg, R1 gains weight from the literature alone.
- **R2 control authority:** every hardware transition needs a medium-dependent command map or gain set switched at a
  precise moment (Berkeley's 0.08 s timer, JHU's prop-speed medium sensor) plus a prepared start state. A controller that
  cannot see or switch medium has no hardware precedent for crossing.
- **R3 learner starvation:** no learned transition exists on hardware; in simulation the learned policies sat on top of a
  geometric controller and did not beat it on the simple crossing.
- **R6 instrument:** the surrogate omits the four interface effects every working system documents (thrust collapse /
  ventilation, surface ground effect, slamming in the dynamics, passive wing bending) and charges no jet pumping work (J-01).
  Some of these make simulated crossings easier than reality, some harder. Worth a targeted check before trusting any
  crossing score.

## 8. Sources opened (year)

Robots: TriphiBot arXiv 2602.01385 (2026); CUHK triphibian arXiv 2211.17022 (2022/23); Wukong-Omni arXiv 2603.02602 (2026);
M4 Nat. Commun. 2023 (PMC10300070); TJ-FlyingFish arXiv 2301.12344 (2023); RoboBee Sci. Robot. 2017 (SMRL PDF);
FAAV Science 2026 (MIT News, DSpace 1721.1/169536, ZME Science); jump-glider Sci. Robot. 2019 (PDF); AquaMAV Interface Focus
2017 (PMC5206604); Dipper RSS 2021 (proceedings page); SUWAVE RA-L 2017 (PDF); Loon Copter JFR 2018 (PDF); Naviator
arXiv 1507.01932 (2015); remora Sci. Robot. 2022 (Spiral manuscript); Nezha-SeaDart JFR 2025 (PDF); JHU delta arXiv
1906.01532 (2019); Berkeley mini UAUV (2019 PDF); ZS-Puffin arXiv 2508.08690 (2025); AALR arXiv 2409.09572 (2024);
RoboFly arXiv 2001.02320 (2020); T3 arXiv 2503.00805 (2025); RAVEN Nature 2024 (RePEc abstract); E-Flap RA-L 2021 (Zenodo
PDF); DelFly site; Hitec HS-645MG and ROBOTIS XM430 spec pages; survey companion page nubot-uaav.github.io (JFR 2025);
Giordano et al. arXiv 2410.23722 (2024).
Animals: Elliott et al. PNAS 2013 (PMC3677478); Sato et al. Proc. R. Soc. B 2007 (PMC1766382); Watanuki et al. Proc. R. Soc. B
2003 (PMC1691262); Chang et al. PNAS 2016 (PMC5087034 In This Issue); Hashimoto et al. J. Robot. Mechatron. 2024; Sun et al.
Beilstein J. Nanotechnol. 2016 (PMC4979654); Mukundarajan et al. arXiv 1405.2298; Park & Choi 2010 (ScienceDaily); Princeton
BAM lab page; NPS dipper page.
Modelling/control: MuJoCo fluid docs; Sane & Dickinson 2002 (Caltech record); Whitney & Wood 2010 (Cambridge abstract);
Malarczyk & Hultmark arXiv 2608.13424 (2026); Grando et al. arXiv 2103.12883 (2021); Wang & Shang arXiv 2507.03869 (2025);
AAM-SEALS arXiv 2412.19744 (2024); Empa/EPFL LSR MSc brief (2026); project `docs/model_validity.md` (read-only).
Could not open (403/404/429/too large): FAAV paper PDF, Dipper PDF, IEEE Access 2021, ASME JMR 2026, MBDSM 2026,
Pennycuick 1996 PDF, Askew & Marsh 2002, Festo SmartBird pages, Taylor et al. 2003.
Surfaced in searches, not opened (ground-only land-water amphibious robots, the nearest shoreline-crossing references):
"Design and Experiment of a Wheel-Fin Hybrid Amphibious Robot" (IEEE conference, ieeexplore.ieee.org/abstract/document/10739822);
"Design and Optimization of a Multimode Amphibious Robot With Propeller-Leg"; "A transformable wheel-spoke-paddle hybrid
amphibious robot" (Robotica, Cambridge); Yu et al., "On a Bio-inspired Amphibious Robot Capable of Multimodal Motion"
(IEEE/ASME T-Mech). UNVERIFIED.

## Appendix A — extraction notes (raw, as found; the tables above are built from these)

- TriphiBot (Li ... Gao, Cao; Zhejiang Univ.; arXiv 2602.01385, 2026, no venue on page). VERIFIED https://arxiv.org/html/2602.01385
  1.1 kg quadcopter + 2 passive wheels, no extra actuator; FOC motor drive; negatively buoyant (CoB slightly above CoG, no magnitude);
  same props in air and water, 0.898 N/W in water (also 0.825 N/W peak elsewhere, unexplained) vs 0.265 N/W for TJ-FlyingFish;
  water exit from floating: "leaves the water surface in just 1.2 s"; no T/W given. Air max 2 m/s, ground max 3 m/s, 9 cm obstacle.
  Control: finite-state machine; hybrid NMPC (air/ground, switch variable from height threshold, 200 Hz), cascade PID for water exit, PID underwater. No learning.
- "Design and Verification of a Novel Triphibian Robot" (Yang, Xue ... Qian; CUHK-Shenzhen/AIRS; arXiv 2211.17022 v2 2023; RA-L preprint header on arXiv PDF). VERIFIED https://arxiv.org/html/2211.17022
  4.76 kg; 3 ducted fans 70 mm, each 2240 g max thrust (static T/W ~1.41, INFERRED 3x2.24/4.76); cylinder rolls as wheel on land;
  floats as hovercraft on surface and takes off FROM THE SURFACE (ducts above water, morphing servo rotates ducts 90 deg) — "avoids the water-density problem";
  land->water passive (off-centre CoG + foam flips it); no air->land strategy (future work); manual RC + ArduPilot tri-rotor; no speeds/endurance reported.
- Wukong-Omni (Liu ... Meng, Zhang; Guangdong Univ. of Tech + HKUST; arXiv 2603.02602, Mar 2026, 19 pp, no venue). VERIFIED https://arxiv.org/html/2603.02602v1
  4.4 kg; one dual-output motor per arm with one-way bearings: one direction = 10-in aerial prop, other = 180 mm wheel-propeller via 1:13 planetary (land wheel AND water screw);
  air 5 m/s (table 4.86), ~5 min, 942 W avg; water 1.2 m/s; land 1.0-1.1 m/s cruise (2.1 m/s text, 3.3 m/s table — internal conflict);
  water exit: props reversed, breaks surface, airborne in 6.5 s, stable flight 9.3 s; ground->air ~6 s. Claims "positively buoyant" (volume 3.21e-3 m^3 vs 4.4 kg — the summary flags this as net-negative by ~1.2 kg if the volume is the full hull: UNRESOLVED in source).
  Control: FSM selecting per-mode controllers (cascaded PID air/underwater, L1 guidance on land, manual surface). No learning.
- Caltech M4 (Sihite, Kalantari, Nemovi, Ramezani, Gharib; Nat. Commun. 14:3323, 2023). VERIFIED https://pmc.ncbi.nlm.nih.gov/articles/PMC10300070/
  ~6.0 kg text vs 5.6 kg Table 1 (internal conflict); 4 shrouded props ~2.2 kgf each, ~9 kgf total (T/W ~1.5 INFERRED); shrouds double as wheels; legs with 2 hip DOF each;
  modes: roll, fly, balance on 2 wheels (MIP), thruster-assisted MIP, walk, crouch, tumble, loco-manipulate, wing-assisted incline running (45 deg).
  AIR + LAND ONLY — no water operation in the paper. Control: optimisation-based nonlinear control per mode + multi-modal PRM/A* planner with energy costs; autonomous switching only UAS<->UGV; no learning.
- TJ-FlyingFish (Liu, Dou ... Ben M. Chen; Tongji/CUHK; arXiv 2301.12344 v2 Feb 2023, accepted ICRA 2023 per arXiv listing). VERIFIED (read the PDF pages) https://arxiv.org/abs/2301.12344
  1.63 kg, 380 mm wheelbase, "aerial thrust to weight ratio: 3.75", hover 6 min in air, ~40 min underwater; designed UNDER-buoyant ("to be suspended in water by the undeveloped control algorithm in PX4");
  one motor+prop per arm through a dual-speed planetary gearbox (forward = direct drive for air; reverse = 12.33:1 for water), unit 122 g; per-unit max aquatic thrust 32 N, aquatic specific thrust 0.265 N/W;
  tilting arms give thrust vectoring underwater; max dive depth 3 m. Its Table I comparators: AquaMAV 9.5 N/28 g, MAAQuad 4.8 N/19.3 g (sim), Loon-copter 12 N/130.4 g, Nezha-mini 15 N/83.4 g, TT100 21.6 N/290 g.
  Control: stock PX4 cascade PID + RC; pilot switches propulsion mode by a transmitter channel. No learning.
- Harvard hybrid aerial-aquatic RoboBee: Chen Y., Wang H., Helbling E.F., Jafferis N.T., Zufferey R., Ong A., Ma K., Gravish N., Chirarattananon P., Kovac M., Wood R.J.,
  "A biologically inspired, flapping-wing, hybrid aerial-aquatic microrobot", Sci. Robot. 2, eaao5619, 25 Oct 2017. VERIFIED (read the PDF pages 1-9)
  https://smrl.mit.edu/wp-content/uploads/2019/10/science_robotics_fly_swim_robobee.pdf
  175 mg; two piezo actuators; SAME WINGS in both media: air 220-300 Hz (prior 85 mg design: 140 Hz), water 9-13 Hz; passively stable swimming only at >= 9 Hz, at 7 Hz cannot rise; swims up at 20 mm/s; sinks passively when off;
  max lift raised 140 -> 220 mg (lift/weight ~1.26 INFERRED 220/175); wing area cut 40% to reach the air frequency;
  surface tension: "can exceed robot weight by over 10 times" at mm scale; measured surface-tension forces on beams+chamber 6.6 mN (soapy) and 18.8 mN (tap) vs max lift 3.1 mN => "direct liftoff from the water surface is infeasible";
  Bond number (weight/surface tension) 0.7 without surfactant (cannot enter), 2 with soap (enters);
  exit = two steps: electrolysis fills a 33 mg chamber so buoyancy lifts the wings out (frames to 129 s), then a 0.2 mJ spark detonates the oxyhydrogen: "average upward force ... in the first millisecond was between 7.5 and 9 N ... device thrust-to-weight ratio of 19,000 to 23,000";
  takeoff 1.8 m/s with micro-openings (3.4 m/s without, chamber cracked), typical 2.5 m/s, 37 cm ballistic jump, lands after 0.55 s; 30-45 mg water residue on the body afterwards;
  cannot fly after exit: no onboard sensing; aerial hover needs Vicon + off-board adaptive control through a tether; swimming open loop (IR tracking fails in water); landing success 60% on nylon, 10% on steel;
  paper's own remark: whirligig beetles do water-to-air transition, "none can do so impulsively".
- Imperial "jump-glider": Zufferey R., Ortega Ancel A., Farinha A., Siddall R., Armanini S.F., Nasr M., Brahmal R.V., Kennedy G., Kovac M., "Consecutive aquatic jump-gliding with water-reactive fuel", Sci. Robot. 4, eaax7330, 11 Sep 2019. VERIFIED (read PDF pp. 1-5)
  https://nubot-uaav.github.io/static/pdf/squid-like%20aavs/Consecutive%20aquatic%20jump-gliding%20with%20water-reactive%20fuel.pdf
  160 g delta-wing glider, no propeller; water drops + CaC2 -> acetylene, arc-ignited in a 491 ml chamber 40% water-filled, 6.8 mm nozzle; "average peak pressure of 6.5 bars in 25 ms, which accelerated the robot to 10 m/s, yielding a flight distance of 26 m";
  max jet thrust 43 N (=> peak T/W ~27, INFERRED 43/(0.16*9.81)); jet phase ~200 ms then glide; fuel for 10 launches; 22 flights; floats passively at the 47 deg launch angle (CoM behind CoB) and fires when the IMU sees that angle;
  control = predetermined timing on a microcontroller (open loop), no flight control during the jump.
  Paper's framing: "the transition from water to air is a power-intensive process"; "Takeoff from water typically requires that the aircraft position its lift and propulsion surfaces well clear of the water";
  animal notes there: flying fish accelerate underwater to ~10 m/s before leaving; gannets regain altitude using positive buoyancy, flapping and hydrophobic feathers; flying squid jet.
- Imperial AquaMAV: Siddall R., Ortega Ancel A., Kovac M., "Wind and water tunnel testing of a morphing aquatic micro air vehicle", Interface Focus 7:20160085, 2017. VERIFIED https://pmc.ncbi.nlm.nih.gov/articles/PMC5206604
  201.3 g fixed wing (earlier 100 g prototype took off from beneath the water with a water jet); span 592 mm, 0.056 m^2, AR 6.35, cruise 10 m/s, 14 min; wings sweep back 90 deg in < 0.5 s (6.5 g servo) before plunge;
  retracted wing: level-flight drag -62%, lift -92%; simulated terminal dive 36.6-39.9 m/s; field-test impact ~9.1 m/s; water-jet take-off = 1 s burst from a CO2 tank (pressure/thrust not stated in this paper);
  stops within ~1 m (1-2 body lengths) underwater. Siddall & Kovac 2014 (Bioinsp. Biomim. 9:031001): 2.6 g jet proof-of-concept reaches 4.8 m (search-result abstract, not opened: treat as UNVERIFIED-number).
- Loon Copter: Alzu'bi H., Mansour I., Rawashdeh O., J. Field Robotics 35(5):764-778, 2018, doi 10.1002/rob.21777. VERIFIED (read PDF pp. 1-5)
  https://nubot-uaav.github.io/static/pdf/multirotor_based_aavs/Journal%20of%20Field%20Robotics%20-%202018%20-%20Alzu'bi%20-%20Loon%20Copter%20%20Implementation%20of%20a%20hybrid%20unmanned%20aquatic%20aerial%20quadcopter%20(1).pdf
  2.7 kg with payload; ONE set of direct-drive props for air and water; ballast: pump + 150 ml cylinder (100 ml to submerge), 6 W vs 450 W for the rotors;
  flooding the cylinder shifts CoG so the airframe tilts ~90 deg and sinks -> props horizontal for underwater thrust; emptying it floats it back upright -> takes off from the SURFACE;
  air->water ~25 s, water->air ~13 s; flight 10.5 min outdoor/12 min indoor, underwater 22 min at 0.5 m/s, surface 20 min at 1.0 m/s; can hang 11 h underwater without props.
  Control: closed loop in air and on the surface, OPEN LOOP underwater; transitions are ballast-driven (quasi-static, not dynamic).
- Rutgers Naviator: Maia M.M., Soni P., Diez-Garias F.J., arXiv 1507.01932, 2015 ("Demonstration of an Aerial and Submersible Vehicle ... Seamless Air-Water Transition"). VERIFIED (read PDF) https://arxiv.org/pdf/1507.01932
  octo-quad (4 arms x 2 counter-rotating stacked props with a column gap), quasi-neutrally buoyant, < 2 kg, < 0.9 m span; SAME air props used underwater (no cavitation, < 250 rpm, 0.5-10 N per prop, cruise 0.3-1 m/s);
  water exit: while crossing, the prop at the interface slows/stops (avoids a drag spike) while the other of the pair keeps thrusting, then re-accelerates once clear: "performed in under two seconds";
  underwater it pitches to 65-75 deg (PD controller) to swim "horizontally". Control: hand-designed per-rotor throttle scheduling at the interface + PD.
- SUWAVE: Peloquin R.-A., Thibault D., Lussier Desbiens A., "Design of a Passive Vertical Takeoff and Landing Aquatic UAV", IEEE RA-L 2(2):381-388, 2017, doi 10.1109/LRA.2016.2633623. VERIFIED (read PDF)
  https://nubot-uaav.github.io/static/pdf/fixed-wing%20aavs/Design_of_a_Passive_Vertical_Takeoff_and_Landing_Aquatic_UAV.pdf
  584 g flying wing, 1 m span, one motor + 11x6 folding prop, T/W = 1.56 (stated), ~9 N static thrust; floats flat; takeoff is PASSIVE: an SMA pin releases the centre body, which pivots under its own weight (water seeps in, 0.14 kg) until the prop points up,
  full thrust pulls it out; success iff the body re-latches at 40-60 deg pitch (~0.4 s, 1.75 m/s at latch); peak vertical accel during takeoff 0.7 g; sensitive to wind gusts at takeoff;
  lands by nose-first dive "like a gannet": peak deceleration ~15 g at 10 m/s, 0.6 m max depth (fitted model: m*a = C_B*rho*g*V_s + C_V*A_f*v^2 - m*g, C_B=0.6, C_V=50).
- Beihang/Imperial remora hitchhiker: Li L., Wang S., Zhang Y., ... Kovac M., Wen L., "Aerial-aquatic robots capable of crossing the air-water boundary and hitchhiking on surfaces", Sci. Robot. 7(66), eabm6695, 2022. VERIFIED (read accepted manuscript pp. 1-10)
  https://spiral.imperial.ac.uk/handle/10044/1/96997
  950 g quadrotor, 40x40x14 cm; 5-inch blades on passively self-folding hubs (85 mm folded underwater by fluid force, 170 mm unfolded in air by centrifugal force);
  water->air in 0.35 +/- 0.01 s at 100% throttle; minimum throttle for water exit 35% with self-folding props (0.69 s) vs 50% with commercial props (1.39 s); 7 consecutive transitions in 20.3 s; exit 0.43 s, entry 0.13 s;
  a transition cycle costs 1.9x hover power; takes off from the bed of a flowing stream (~30 cm deep); remote controlled (pilot), standard flight controller.
  Paper's framing: "A rotor-based robot needs to expose its propellers above the water surface and rapidly accelerate them to a high rotational speed to generate enough force to lift the robot out of the water"; prior buoyancy-based exits took 13 s (Loon Copter) and 30 s (Lu et al.).
- **MIT/EPFL FAAV (flapping-wing aerial-aquatic vehicle)**: Zufferey R., Jeger S.L., Huesser M., Ruiz F., Lapsansky A., Ijspeert A., Floreano D., "Leaping out of the water: Aerial-aquatic locomotion with flapping wings", Science 393:207-211, 9 Jul 2026, doi 10.1126/science.aeb6744.
  VERIFIED (MIT News 2026-07-09 https://news.mit.edu/2026/new-flapping-robot-swims-and-flies-like-diving-bird-0709 ; MIT DSpace record with abstract https://dspace.mit.edu/handle/1721.1/169536 ; accepted manuscript PDF rate-limited at first attempt)
  < 300 g (ZME Science says 250 g); ONE waterproof motor + crankshaft flaps both wings, motorised tail; interchangeable wings 60/80/100 cm, 80 cm most reliable across swim/fly/transition; thin flexible membranes, hydrophobic coating;
  ~5 Hz in both media (MIT News; birds cited ~10 Hz air vs ~4 Hz water for small species); swims ~1 m/s, flies ~6 m/s;
  water exit: starts ~0.5 m deep, swims up and flaps out with NO wing folding and no feet; needs pitch ~70 deg at exit so the wingtips clear the water on the upstroke (steeper -> falls back);
  abstract: frequency modulation + flexible wings + strong actuation allow transitions without folding wings or legs; large wings help flight without much loss underwater; tail-body distance and exit angle govern exit.
  Control: pre-programmed wing frequency and tail angle per trial; no autopilot, no turning yet, not tested in chop/wind.
  More detail from press (ZME Science, 2026, VERIFIED as press text, primary PDF not opened): 250 g untethered; ONE small motor; air ~6.3 m/s at "about five to 11" beats/s; underwater frequency can drop to one beat per 10 s; wings bend passively "by up to 90%";
  silicone conformal waterproofing 13 g (~5% of mass); water exit "under one second", ~8-10 strokes; ~70 deg launch (shallower: submerged tail pulls it back; steeper: tumbles backward); tail must sit close to the body;
  estimated range 6 km flying / 2 km swimming; "Beyond roughly 15 meters, it's more energy-efficient to leap out and fly than to swim";
  "All operations are timed sequences with no user input"; diving and take-off tested SEPARATELY — the full cruise-plunge-swim-launch-cruise cycle not yet demonstrated; "Clearing the wings from the water surface just after transition was the most difficult part";
  bird claim reported: most diving birds still paddle with feet; only light-bodied fliers (kingfishers, dippers) skip it. https://www.zmescience.com/science/news-science/this-250-gram-robot-can-swim-underwater-then-fly-into-the-sky-using-its-flapping-wings-like-a-diving-bird/
- E-Flap (Univ. Seville GRVC): Zufferey R., ..., Ollero A., "Design of the High-Payload Flapping Wing Robot E-Flap", IEEE RA-L 6(2):3097-3104, 2021. VERIFIED (text of accepted PDF) https://zenodo.org/record/4917204
  0.510 kg (incl. battery, no payload), span 1.5 m, wings 82 g, flapping engine (1130 Kv brushless 20 g + 42:1 two-stage gearbox + 4-bar) 71 g = 14% of mass; max flapping 5.5 Hz, cruise ~4 Hz;
  sized with the Pennycuick-derived formula f = zeta * m^(3/8) g^(1/2) b^(-23/24) S^(-1/3) rho^(-3/8), zeta = 1.48 -> 3.687 ~ 4 Hz;
  peak wing-root torque 6.5 N m (model), "A 150W motor is selected, capable of supplying the peak power estimation of 130W" (=> ~255 W/kg of vehicle peak, INFERRED);
  speeds 2-6 m/s; carried up to 520 g payload (100%) at 5.6 +/- 0.2 Hz full throttle; flight 5 min, 15 min with +93 g battery; indoor pitch PI on Optitrack.
  Its related-work numbers: RoboRaven 200-700 g, ~6 m/s; RoBird Falcon ~16 m/s; RoBird Bald Eagle 2.1 kg (+1 kg payload), 18 m/s; Festo SmartBird 450 g, ~1.5-2 m span; Dove ~10 m/s.
- Festo SmartBird: ~2 m span, 400-450 g, ~23-25 W, electromechanical efficiency ~45% (search snippets of Festo/press pages; not opened: UNVERIFIED numbers).
- RAVEN: Shin W.D., Phan H.-V., Daley M.A., Ijspeert A.J., Floreano D., "Fast ground-to-air transition with avian-inspired multifunctional legs", Nature 636:86-91, Dec 2024. VERIFIED abstract (RePEc page) — fixed-wing + bird-like legs: walk, hop, jump into flight; jump "contributes substantially to the initial flight take-off speed" and is more energy efficient than standing take-off. Numbers not opened.
- T3 tailless triple-flapping-wing robot (Xu ... Lu; arXiv 2503.00805, 2025, submitted IROS 2025). VERIFIED abstract: flapping flight + ground locomotion by elastic passive legs driven by wing vibration; SE(3) controller switches modes. (Air + land only.)
- Auks: Elliott K.H., Ricklefs R.E., Gaston A.J., Hatch S.A., Speakman J.R., Davoren G.K., "High flight costs, but low dive costs, in auks support the biomechanical hypothesis for flightlessness in penguins", PNAS 110:9380-9384, 2013. VERIFIED https://pmc.ncbi.nlm.nih.gov/articles/PMC3677478
  thick-billed murre ~1 kg: flight 141 +/- 18 W = 146 W/kg = 31x BMR (cost of transport 0.83 J/(N m)) — "highest flight costs recorded for vertebrates" with pelagic cormorant (1.8 kg, 158 W, 87 W/kg);
  murre flight cost 33% above biomechanical-model prediction; dive cost 27% below a same-size foot-propelled diver but 30% above a same-size penguin;
  quote: wing-propelled pursuit divers "are flightless above ~1 kg and capable of flight below ~1 kg"; great auk ~5 kg flightless; "as wings become more efficient for swimming they become less efficient for flying";
  wing areas printed as "46 cm2" (murres, 1.0 kg), "42 cm2" (razorbill 0.72 kg), "~38 cm2" (great auk) — physically these must be ~10x larger per bird (a 1 kg murre has ~0.045 m^2 total wing area, INFERRED); treat the printed unit as a slip.
- Gannet plunge dive: Chang B., Croson M., Straker L., Gart S., Dove C., Gerwin J., Jung S., "How seabirds plunge-dive without injuries", PNAS 113:12006-12011, 2016. VERIFIED via the PNAS In-This-Issue page https://pmc.ncbi.nlm.nih.gov/articles/PMC5087034 :
  impact up to 24 m/s; neck would buckle only at ~80 m/s; pre-impact neck-muscle contraction. Other numbers (search snippets, not opened): plunge depth limited to ~10-11 m by momentum (Garthe et al. 2014); dives from up to ~30 m height (Ropert-Coudert et al. 2004).
- Diving beetles (the project's namesake): Hashimoto N., Suzuki R., Kikuchi K., "CFD Analysis of Takeoff from a Water Surface for an Insect-Scale Aerial/Aquatic Robot", J. Robot. Mechatron. 36(5):1122-1133, 2024. VERIFIED abstract https://www.fujipress.jp/jrm/rb/robot003600051122/
  the diving beetle Eretes griseus "takes off directly from the water surface"; CFD (VOF free surface) of its takeoff: SLAPPING the water gives lift per body weight max 18 / mean 9.2, PADDLING only 0.46 / 0.23; water-surface (ground) effect +25% lift below normalised height 0.44, gone above 0.7.
  Sun J. et al., Beilstein J. Nanotechnol. 7:904-913, 2016 (Cybister japonicus, 35-40 mm): "pronounced flight capability when leaving the water", fly to light "directly from the water"; hind wings unfold hydraulically in ~0.74 s (free) to 3-3.7 s (instrumented). VERIFIED https://pmc.ncbi.nlm.nih.gov/articles/PMC4979654/
  Dytiscus marginalis "climb out of the water and sit in a rather upright position" before take-off (hobby page microcosmos.nl, search snippet only: UNVERIFIED). Whether large dytiscids MUST leave the water to fly: UNKNOWN (no peer-reviewed source found); at least Eretes and Cybister take off from the surface.
- Surface tension at insect scale: Mukundarajan H., Bardon T.C., Kim D.H., Prakash M., "Surface tension dominates insect flight on fluid interfaces", arXiv 1405.2298 (2014; J. Exp. Biol. 219:752, 2016 per the JEB URL in search). VERIFIED https://ar5iv.labs.arxiv.org/html/1405.2298
  water-lily beetle 2.2 mg, 116 Hz; surface-tension restoring force up to ~5 mg; required lift/weight ~3.4 vs 1-2.5 airborne.
- Learned control across the air-water boundary:
  Grando R.B., de Jesus J.C., Kich V.A., Kolling A.H., Bortoluzzi N.P., Pinheiro P.M., Neto A.A., Drews-Jr P.L.J., "Deep Reinforcement Learning for Mapless Navigation of a Hybrid Aerial Underwater Vehicle with Medium Transition", arXiv 2103.12883 (ICRA 2021 per listings). VERIFIED https://ar5iv.labs.arxiv.org/html/2103.12883
   SIMULATION ONLY (Gazebo, RotorS + UUV simulator); DDPG/SAC policy outputs velocity setpoints (0-0.25 m/s, +/-0.25 m/s vertical, +/-0.25 rad yaw) to a geometric tracking controller (Lee) that does the actual flying;
   success over 100 trials, scenario 1: air->water 96% (DDPG) / 72% (SAC) vs 100% for the geometric baseline; water->air 97/75 vs 100; scenario 2 (risers): 54/37 vs 0 and 57/71 vs 0 (baseline cannot avoid obstacles). No real robot.
   Follow-ups DoCRL (arXiv 2308.09811) and 2602.21259 (shared lidar/sonar policy): navigation-level, simulation (abstract-level, not opened in detail).
  Empa/EPFL LSR (Kovac) MSc task agreement "Learning-Based Control of a Hybrid Aerial-Aquatic Multirotor", HS 2026-FS 2027. VERIFIED https://www.epfl.ch/labs/lsr/wp-content/uploads/2026/09/Aerial_Aquatic_Multirotor_Project.pdf
   "typical off-the-shelf drone control stacks are unable to track position waypoints both outside and inside the water"; plan: GPU-parallel sim with "added mass, buoyancy, interface effects, and the decrease of rotor efficiency when submerged", deep RL + sim-to-real, compared against gain-scheduled cascaded PID and MRAC — i.e., learned transition control on hardware is still a planned project in the leading lab as of Sep 2026.
  Wang R., Shang Z., arXiv 2507.03869 (2025): hybrid model (air / water / transition zone with variable buoyancy, added mass, drag; CFD for submerged rotor thrust), twisting sliding-mode attitude control across the interface + cascade PID; < 0.1 m height error, < 5 deg attitude during repeated crossings. VERIFIED abstract https://arxiv.org/abs/2507.03869
  AAM-SEALS (Yang ... Zha, UMD, arXiv 2412.19744, 2024): Isaac Sim PBD-particle water; says many simulators "neglect aerial parts and air-water transitions"; no RL result for the crossing itself; no sim-to-real. VERIFIED https://arxiv.org/html/2412.19744v1
- Modelling references found so far: MuJoCo fluid docs (VERIFIED https://mujoco.readthedocs.io/en/stable/computation/fluid.html): "Proper simulation of fluid dynamics is beyond the scope of MuJoCo"; stateless; one global density/viscosity (no free surface or partial immersion described); inertia-box model has no added mass; ellipsoid model has added mass (not scalable), Kutta and Magnus lift; viscous terms "may overestimate dissipation".
  Reduced-frequency rule of thumb k = pi f c / U: quasi-steady k <= 0.05, unsteady above, highly unsteady k > 0.2 (Wikipedia "Reduced frequency" — secondary; arXiv 2608.13424 reports unsteadiness even below 0.05 in dynamic stall). Search snippets only so far.
- Note: "Mirs-X" is the arXiv v1 title of TJ-FlyingFish (same CUHK/Tongji vehicle, v2 renamed). VERIFIED by the v1 listing (search result https://arxiv.org/abs/2301.12344v1).
- Nezha-SeaDart: Jin Y., Zeng Z., Lian L. (SJTU), "Nezha-SeaDart: A tail-sitting fixed-wing vertical takeoff and landing hybrid aerial underwater vehicle", J. Field Robotics 42(1), 2025 (online 2024), doi 10.1002/rob.22399. VERIFIED (text of VoR PDF)
  https://nubot-uaav.github.io/static/pdf/hybrid_aavs/2025-Nezha-SeaDart%20A%20tail-sitting%20fixed-wing%20vertical%20takeoff%20and%20landing%20hybrid%20aerial%20underwater%20vehicle.pdf
  1500 g, length 525 mm, delta wing span 400 mm (AR 1.26, 68 deg sweep), W/S 115 N/m^2; four 5.1-in props "can produce 36 N thrust in the air" (static T/W ~2.45, INFERRED); NO active buoyancy control: 1550 g buoyancy vs 1500 g weight (+3%);
  same aerial props used underwater (max 3 m/s; 21.5 min at 1 m/s); aerial cruise up to 35 m/s (4.15 min at 35.2 m/s, 705 W), hover 385 W, VTOL climb 18 m/s;
  FAST WATER EXIT: dive to depth, pitch nose-up (~80-90 deg), accelerate, break the surface on momentum: nose cone at surface at 2.66 m/s; thrust drops as props near the surface (decelerates from t = 0.2 s); props regain thrust at t = 0.6 s; clear of water at t = 0.8 s; design estimate had been 4.77 m/s needed; 6 repeats filmed;
  fast entry: nose-down dive with props STOPPED (avoid current spikes), elevons stabilise. Also slow mode: float, stabilise, throttle up off the surface.
  Control: PX4 PID on Pixhawk + Arduino mission computer; an RC channel switches to underwater mode (hand-designed mode logic). Paper's own claim: no dedicated water-exit propulsion needed.
- JHU APL delta wing: Moore J., "Closed-Loop Control of a Delta-Wing Unmanned Aerial-Aquatic Vehicle", arXiv 1906.01532, 2019 (design paper ICRA 2018). VERIFIED https://ar5iv.arxiv.org/html/1906.01532
  375 g, span 0.61 m, ONE 6x4 propeller for both media, "buoyancy assist" (CoB aft -> passively unstable underwater); exits from underwater cruise "in less than 0.5s";
  control = trajectory optimisation (direct transcription, SNOPT) + mode-dependent time-varying LQR + an LQR near the water-air guard + hybrid estimator (3 EKFs; propeller speed jump ~14x is the medium sensor);
  hardware: 8 of 20 exits reached controlled flight (40%); 7 flipped backward, 4 fell forward; suggests ~70 deg pitch at exit would work better.
- Berkeley mini UAUV: Zha J., Thacher E., Kroeger J., Maekiharju S.A., Mueller M.W., "Towards breaching a still water surface with a miniature unmanned aerial underwater vehicle", 2019 (HiPeRLab PDF; venue not on page — ICUAS 2019 per lab listing is UNVERIFIED). VERIFIED (PDF text)
  https://hiperlab.berkeley.edu/wp-content/uploads/2019/05/Towards-breaching-a-still-water-surface-with-a-miniature-unmanned-aerial-underwater-vehicle.pdf
  202 g plain quadcopter (7500 KV motors), V = 186e-6 m^3 => slightly NEGATIVELY buoyant (net ~ -16 g, INFERRED); same controller in both media, only the command->thrust map switches (water map vs air map);
  breach: rise from 0.15 m depth at 1 m/s commanded, breaks surface at ~0.5 m/s, then "loses its initial momentum and stays on top of the free surface for about 0.2 seconds" (0.08 s programmed switch delay + 0.12 s unexplained: prop spin-up, air entrainment) before climbing;
  the transition region is "a highly unsteady flow field" from "propeller-induced air entrainment"; pressure depth estimate fails there; underwater tracking error attributed to underestimated drag and "not considering the added mass effect".
- Wingbeat in air vs water for the same wing: Sato K., Watanuki Y., Takahashi A., Miller P.J.O., Tanaka H. et al., "Stroke frequency, but not swimming speed, is related to body size in free-ranging seabirds, pinnipeds and cetaceans", Proc. R. Soc. B 274:471-477, 2007. VERIFIED Table 1 https://pmc.ncbi.nlm.nih.gov/articles/PMC1766382
  common guillemot 0.95 kg: swim 2.58 Hz / flight 9.33 Hz; Bruennich's guillemot 0.95 kg: 2.46 / 7.59; razorbill 0.57 kg: 2.92 / 9.33; rhinoceros auklet 0.53 kg: 2.71 / 8.96 (air/water ratio 3.1-3.6, INFERRED);
  foot-propelled shags 1.6-2.4 kg: swim 2.9-3.3 Hz (feet) / flight 5.7-5.8 Hz; swimmers' stroke f = 3.56 m^-0.29 (R^2 0.99), swim speed ~independent of mass (1.88 m^-0.05 m/s); "Seabirds that use their wings for both swimming and flying stroked at a lower frequency".
  Watanuki Y. et al., Proc. R. Soc. B 270:483-488, 2003 (Bruennich's guillemots): surges at 3.2 Hz on BOTH strokes against buoyancy early in descent, 2.4 Hz downstroke-only deeper; stop stroking at 18 m on ascent and rise passively to 2.3 m/s on expanding air. VERIFIED https://pmc.ncbi.nlm.nih.gov/articles/PMC1691262
- Flight-muscle power (search-result level, primary pages 403): blue-breasted quail pectoralis ~390-400 W/kg MUSCLE at take-off (Askew & Marsh, J. Exp. Biol. 205:2153, 2002); pigeon in vivo ~70 W/kg muscle at 5-6 m/s, 8.7 Hz (Biewener lab); mallard 131 W/kg muscle (17 W/kg body) slow level flight; cruising range 60-150 W/kg muscle. UNVERIFIED (snippets), consistent with each other.
- ZS-Puffin (Wang Z., Jiang Y., Zhen Z., Jiang Y., Tan Y., Wang W.; Sun Yat-sen Univ.; arXiv 2508.08690, Aug 2025). VERIFIED https://arxiv.org/html/2508.08690v1
  1.61 kg, span 690 mm; two 2216 rotors (31.6 N max thrust, per-rotor vs total unclear) for flight; "amphibious wings": servo-tilted wings give lift in air and FLAP for thrust underwater (~2.4 Hz), amphibious tail also flaps;
  slightly positive buoyancy (CoB above CoG, no magnitude); underwater 0.63 m/s with vectored rotors vs 0.29 / 0.21 m/s flapping; water exit ~3 s (fuselage leaving at 1.1 s, out at 2.7 s, vertical flight at 2.9 s), belly-sitter;
  control: cascaded PID in air, open-loop pilot mixing underwater, CPG (3 coupled phase oscillators) for flapping. No learning. (Rotor flight + flapping swim: the flapping is the WATER mode only.)
- Diving-beetle-inspired AALR: Hu J., Chen P., Feng T., Wen Y., Wu K., Dong J., "A Novel Aerial-Aquatic Locomotion Robot with Variable Stiffness Propulsion Module", arXiv 2409.09572, Sep 2024 (IROS 2025 per search listing). VERIFIED (PDF text)
  1.732 kg quadrotor whose arms become beetle-leg paddles (power/recovery strokes) by melting low-melting-point-alloy joints (rigid for flight, flexible for rowing); max swim 77 mm/s; aquatic tests only;
  states "conventional propellers typically achieve only 5% propulsive efficiency underwater" (citing its ref [1]; UNVERIFIED primary).
- Flapping propulsors for both media (bench/model work, no free-flying vehicle):
  Izraelevitz J.S., Triantafyllou M.S., "A novel degree of freedom in flapping wings shows promise for a dual aerial/aquatic vehicle propulsor", ICRA 2015, pp. 5830-5837. VERIFIED abstract https://dspace.mit.edu/handle/1721.1/108693 — adds an active in-line (fore-aft) DOF so one wing spans the force range of both fluids; bench prototype only.
  Lock R.J., Vaidyanathan R., Burgess S.C., Loveless J.H., "Development of a biologically inspired multi-modal wing model for aerial-aquatic robotic vehicles through empirical and numerical modelling of the common guillemot, Uria aalge", Bioinsp. Biomim. 5(4):046001, 2010. VERIFIED abstract https://research-information.bris.ac.uk/en/publications/development-of-a-biologically-inspired-multi-modal-wing-model-for/ — measured the saving from wing retraction underwater, numerical morphing-wing model + evolutionary optimisation; press: guillemot swims with bent wings, profile drag down up to 50% (phys.org 2010, snippet).
  Xiong X. et al., "Bionic Collapsible Wings in Aquatic-aerial Robot", arXiv 2304.04302, 2023 — flying-fish robot with soft-hydraulic collapsible wings, simulation-centred; abstract gives no numbers. VERIFIED abstract.
- Servo/actuator reality check (manufacturer specs VERIFIED; power and bandwidth INFERRED from them):
  Hitec HS-645MG: 55.2 g, 9.6 kgf cm (0.94 N m) at 6.0 V, 0.20 s/60 deg (5.24 rad/s no-load). https://hitecrcd.com/products/servos/analog/sport-2/hs-645mg/product
  ROBOTIS Dynamixel XM430-W350: 82 g, 353.5:1, stall 4.1 N m at 12 V, no-load 46 rpm (4.82 rad/s). https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/
  INFERRED: linear-motor peak mechanical power ~ tau_stall*omega_0/4 = 1.2 W (HS-645MG, ~22 W/kg) and 4.9 W (XM430, ~60 W/kg); max sinusoid frequency at +/-30 deg amplitude even at no load = omega_0/(2 pi 0.524) = 1.6 Hz and 1.5 Hz.
  Contrast (VERIFIED above): E-Flap's flapping engine is a 20 g brushless motor rated 150 W with a 42:1 gearbox + crank (71 g total) for ~4-5.5 Hz at 0.5 kg; FAAV uses one motor + crankshaft. Every working flapper found drives the stroke with a geared brushless/DC motor + crank (or piezo at insect scale), not a positional servo.
- Project's own solver scope (read-only, docs/model_validity.md lines 21-100): quasi-steady strip model with Kramer rotational force, strip and bluff added mass, buoyancy rho g V f_sub; validity table: k 0.1-0.3 usable +/-30%, k > 0.3 extrapolating, stroke reversal at low U extrapolating; water entry diagnostic only (slam not fed to dynamics); ventilation, ground (surface) effect, wake capture, spanwise flow, flexibility (except a 1-DOF pitch hinge) NOT modelled; jet pumping work not charged (J-01); jointless machines get no added mass (F-01).
- RoboFly (UW): Chukewad Y.M., James J., Singh A., Fuller S., "RoboFly: An insect-sized robot with simplified fabrication that is capable of flight, ground, and water surface locomotion", arXiv 2001.02320 (2020, submitted to T-RO). VERIFIED abstract https://arxiv.org/abs/2001.02320
  74 mg flapping robot; wings drive flight, ground locomotion and air-water INTERFACIAL (surface) locomotion; not underwater. Closest flapping "triphibian" found, but at insect scale and surface-only in water.
- Not opened (403/empty): Zhong G., Cao J., Chai X., Bai Y., "Design and Performance Analysis of A Triphibious Robot with Tilting-Rotor Structure", IEEE Access 2021, doi 10.1109/ACCESS.2021.3050182 (two tilting rotors on a quadrotor-like body + 4 legs with passive wheels; search snippet says performance analysis by virtual simulation): UNVERIFIED.
  "A Triphibious Robot With a Novel Rolling-Gripper for Direct-to-Hand Delivery in Multidomain Rescue", ASME J. Mechanisms Robotics 18(8):081001, 2026: UNVERIFIED (403).
  "Design of a triphibious robot with positive buoyancy for stable water-air cross-domain transition", Mech. Based Des. Struct. Mach. 54(1), 2026, doi 10.1080/15397734.2026.2679620 (search snippet: four coaxial rotors + propeller thruster, active-wheel undercarriage, positive buoyancy): UNVERIFIED (403).
- Flying fish: Park H., Choi H., "Aerodynamic characteristics of flying fish in gliding flight", J. Exp. Biol. 213:3269-3279, 2010 (via ScienceDaily 2010-09, VERIFIED https://sciencedaily.com/releases/2010/09/100910082308.htm): airborne > 40 s, up to 400 m, ~70 km/h; glide L/D comparable to petrels and wood ducks; best L/D with body parallel to the surface; ground effect over water raises L/D. Princeton BAM lab page (VERIFIED https://www.bamlab.princeton.edu/flying-fish): glides up to 400 m, taxiing (tail sculling with lower lobe in water) between glides.
- American dipper (Cinclus mexicanus): "swimming, diving, or walking underwater along the stream bottom", grips the bed with claws, dives 5-15 s, short strong wings used underwater; dense plumage, preen oil. VERIFIED https://www.nps.gov/articles/000/klmn_american-dipper.htm (sparrow-sized, ~50-60 g: mass not on page, UNVERIFIED).
- Auk take-off from water: common murre "has to run across the surface of the water to take off" (Georgia Aquarium page, search snippet); razorbill feet patter along the surface; ancient murrelet takes off directly (Audubon, snippet). UNVERIFIED (snippets only).
- Survey with a transition-strategy taxonomy: Liu K., Xiao J., Peng H. et al., "Transition Strategy for Unmanned Aerial-Aquatic Vehicles (UAAVs): A Survey", J. Field Robotics 2025, doi 10.1002/rob.70138 (NUDT). Companion page VERIFIED https://nubot-uaav.github.io/ (lists 17 hybrid AAVs: SailMAV 2019, Nezha-Mini 2022, Nezha-F 2023, Nezha IV 2024, Nezha-SeaDart 2025, WuKong RA-L 2025, SurfAAV 2025, ...).

