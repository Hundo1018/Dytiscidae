# Review 1 of `experiments/flapper/PLAN.md` (draft 1) — 2026-11-12

Reviewer: fresh context, no code run, no simulation, nothing edited but this file.
Read: PLAN.md, A12, A13, A11 in full; ROADMAP top paragraphs (2026-11-10 to 11-12);
DIAGNOSIS 7b, 7c; CLAUDE.md "Designing a measurement"; `docs/model_validity.md` (grep only).
A9 was not re-read. Two web searches (radio under water, Taiwan registration); results at the end.

Labels: **SERIOUS** = the plan fails him without it; **SHOULD** = costs time; **MINOR**.
"[arith]" = my arithmetic on A11/A13 numbers, shown. "[spec]" = my speculation, not checked.

## Verdict in five lines

1. The plan is a waterfall. Its only loop is five words in Phase 5 ("Measurements correct the models") and "or a repeat" in Phase 3.
2. Flight for 15 minutes closes on paper, but at about 0.6-0.85 kg, not the plan's 0.4-0.6 kg. Mass and energy are not what kills it.
3. What most likely cannot coexist with the rest: **one hobby drive that serves air, water and land**, and **leaving the ground or the water under its own power at this mass**. Neither is in the plan's gates or its stop condition.
4. No model in the table covers the things real ornithopters and the one flying-swimming flapper (FAAV) depend on: a compliant wing, in both fluids. The only "higher model" that can check that is a bench, which the plan puts last.
5. The four "right direction" checks test the optimiser, not the direction.

---

## 1. Iteration

**1.1 SERIOUS — there is no loop in the plan.** Phases 0-5 each end in a gate and hand forward. The plan's three "counted design runs" each pose a different problem ("1 wing and stroke ... 2 the full co-design ... 3 crossings and control, or a repeat"), so no run is ever a second attempt at the first with what was learned. The three-run count was inherited from the RL budget (ROADMAP 2026-11-12, "the budget of three full runs stands"); it is a cap on expensive learning runs, and applying it to design passes forbids exactly what the owner asked for.

Where a real process goes round, and what the plan assumes instead:

| loop | why it must repeat | what the plan does |
|---|---|---|
| requirements against feasibility | 15 min, same wings in water, wing-walking, self take-off, hobby parts: nobody knows these are jointly satisfiable (A9: no such robot at any scale) | Phase 0 writes them once; the stop condition tests one of them (flight time) after Phase 3 |
| model against hardware | "hobby parts have no torque curves", spar stiffness 3.9x apart, passive hinge "over-damped" (the plan's own table) | measured once, in Phase 5 |
| shape against motion against actuator | the F-series already showed it: the air stroke in water asks 27x the hover torque (F3); take-off asks 4-5x flight torque (T1) | Phase 3 run 1 fixes wing and stroke before run 2 adds actuators |
| design against build | print tolerances, glue mass ("glue added significant weight", A13 C.3), sealing | one build package at the end |

**1.2 SERIOUS — the cheapest full loop exists today and is not the spine.** Three nested loops, each "a design -> evidence about that design":

- **L0, hours, paper.** A sizing sheet: mass closure, pack power and energy, drive torque by the span-cubed rule, hover/take-off margin, water torque at the swim frequency, buoyancy. Section 3 below is a first pass of it; it already moves the envelope and exposes two contradictions. This is Phase 0, but as a loop that is re-run every time a number changes, with the failing row named.
- **L1, a day per pass, simulation.** The F-series scripts already are an end-to-end evaluator (`fly.py`, `swim.py`, `landwalk.py`, `takeoff.py`, do-nothing arms). Re-run them on ONE crude 0.5-0.7 kg design with a **torque-speed-limited single crank drive** in place of ideal servos. Every F0-T1 verdict was earned with ideal servos at 108 g and 12 Hz on four independently feathered wings; none transfers. A first L1 pass will fail somewhere, and where it fails is the next week's work.
- **L2, a week per pass, bench.** One motor, one gearbox, a printed crank, one wing pair, a kitchen scale or a load cell, a wattmeter; then the same wing in a tub. It gives the three calibrations no model in Phase 1 has: the drive's torque-speed curve, a real wing's cycle-mean thrust per watt, and what the wing does in water.

The plan should be: L0 until it closes, L1 on the L0 design, L2 on the L1 design, corrections back into L0/L1, repeat; the evolutionary search enters as the thing that proposes the next design inside L1, once L1 has been corrected by L2 at least once. Each pass ends with a one-page "what changed and why" (the project already writes these).

**1.3 SHOULD — no budget on anything.** No phase has a duration, an evaluation count, or a money figure. Without them "iterate" has no meaning and a pass cannot be called slow.

---

## 2. Order and dependency

**2.1 SERIOUS — bench measurement is last and should be first.** Plan: "Tests ordered by model risk ...: drive on a bench; one wing on a stand; tethered in a tank; flight; crossings" sits in Phase 5. But Phase 1's gate is "each model inside an error bar written first", and for the actuator row the plan itself says "hobby parts have no torque curves" (A13 "Not found" confirms: no servo or drive curve on any fetched page). That gate cannot be closed on paper. The "bought spar, measured" row already concedes that hardware starts in Phase 1; extend it to the drive and one wing.

**2.2 SERIOUS — the crossings are scheduled last and they size the machine.** Phase 2's gates are "15 minutes on the pack, torque and speed inside the drive, moves in each medium"; crossings appear only as run "3 crossings and control, or a repeat". The project's own record: flight asks a shoulder for 2.0 N m, depth change 4.0-5.5, take-off 7.9-9.4 (ROADMAP 2026-11-12). Runs 1 and 2 would select wings and drives that cannot leave the ground, and run 3 would find out. Take-off and water exit must be feasibility gates from the first pass, or be explicitly dropped by the owner (see 3.4).

**2.3 SERIOUS — Phase 3 needs Phase 4.** A candidate on a free body cannot be scored in flight without a stabiliser: F4 found "the body does not glide", nose from +10 to -46 deg in a second, and flew only with an attitude loop "every step" on ideal per-wing feather. On a held rig the score is not evidence (F3: "the held rig cannot say the machine swims"). So either the controller with real sensor and actuator limits exists before Phase 3, or Phase 3 scores are held-rig numbers. The plan has control after the design is chosen.

**2.4 SHOULD — the high-fidelity row's own precondition may never be met.** "reproduce a published case first": A12 found no open tabulated force data at Re 3e4-2e5, no cost figure for any 3-D flapping run at that Re, and uniform-grid GPU solvers under-resolved there by their authors' statement. A gate that cannot close either stalls the plan or gets waved through.

**2.5 SHOULD — the first crude end-to-end design is missing.** Nothing before Phase 3 produces a whole machine. A hand-sized "reference machine" (it is a fixture, not the design; the owner's objection is to hand-assembled *results*) is what L0-L2 run on.

**2.6 MINOR — Phase 0's gate is "the user confirms".** Confirm against what? Give him the L0 closure table with the failing rows, not a specification.

---

## 3. Feasibility against A13 and A11 (arithmetic)

**3.1 Flight for 15 minutes: closes, but heavier than the plan says. SHOULD.**
One-pair machine, crank drive, tail, [arith] from A13:

| item | g | source |
|---|---|---|
| 3 x P28A, welded (+4%) | 144 | A.1, A.3; 27.6 Wh in 15 min, 111 W sustained |
| protection board | 4-12 | A.3 |
| drive (motor, gears, linkage) | 71 | B.1 E-Flap |
| wing set, 0.25-0.30 m2 at 187 g/m2 | 47-56 | C.4 |
| ESC, flight controller, depth sensor | 8 + 10 + 14 | D |
| two tail servos, IP67 | 57 | B.2 HS-5086WP 28.5 g each |
| receiver, wiring | ~20 | not in A13 [spec] |
| **sum before any structure** | **375-392** | |

At 0.5 kg that leaves 108-125 g for fuselage, tail, a sealed hull, feet, and any fold or feather actuator. E-Flap's "everything else" is about 300 g of its 510 g [arith: 510 - 71 - 82 - battery, battery mass my estimate ~50 g]. So 0.5 kg does not close; about 0.6-0.7 kg does. At 0.6 kg: 73-105 W (122-175 W/kg) against 111 W sustained, 18-26 Wh against 27.6: margin 1.5 at Phoenix's figure, 1.05 at E-Flap's. With 3 x P45B (218 g, 41 Wh, 165 W) at ~0.68 kg: margin 1.4-2.0. Two pairs (the owner's choice of 2026-11-12, made on a 108 g ideal-servo result) add a second drive, wing set and ESC, about +130 g: ~0.8-0.85 kg, 98-149 W, margin 1.1-1.7 on 3 x P45B.
The plan's "0.4-0.6 kg" comes from A13 limit 7, which is a battery-fraction bound, not a closure. Fix: carry this table as L0.

**3.2 MINOR/SHOULD — "15 minutes in each" is ambiguous.** One charge for 45 minutes, or 15 per charge? If one pack: flight 18-26 Wh + swim 0.8-4.2 Wh (A13 limit 8 at 0.6 kg) + walking (no measured figure anywhere) against 30.9 Wh low-rate on 3 x P28A fails at the upper end. Ask him.

**3.3 SERIOUS — one drive, three regimes. This is the requirement most likely to be impossible together with the others.** The plan's limits paragraph gives the answer for air only: "Geared brushless motor and crank: 71 g for about 130 W ... Servos: 11-83 times less power per kg; waterproof ones 1-2 W". So the stroke is a crank: one degree of freedom, fixed amplitude, both wings together, continuous rotation. Then:

- *Water.* Same wing, same amplitude: torque ratio water/air = (rho_w/rho_a)(f_w/f_a)^2 = 833 (f_w/f_a)^2 [arith]. Air at 5 Hz, water at 1 Hz: 33x. At 0.5 Hz against 4 Hz: 13x. Equal torque only at f_air/28.9 = 0.14-0.19 Hz (A11 D gives the 28.9). Against 2.8-4.8 N m in air (plan) and 6.5 N m estimated at E-Flap's hinge, 1 Hz in water asks 90-160 N m. Shedding it needs a wing that bends away (FAAV: "flexible enough to minimize flapping amplitude in water"), folds (auklet: half excursion), or a variable-amplitude mechanism. A crank cannot change amplitude; a servo cannot supply the power. The repo found the same at 108 g: "the air stroke carried into water asks the shoulder for 27 times the hover torque" (F3).
- *Reaction.* Added mass of a 0.25 m2, 1.1 m wing heaved broadside is of order rho pi (c/2)^2 b = 1000 x pi x 0.113^2 x 1.1 = 44 kg [arith; a flapping wing sees roughly a third of that at the tip-equivalent] against a 0.5 kg body. The repo's figure at small scale was 3 kg on 0.105 kg (F3e). With rigid air-sized wings the body flaps and the wings stand still. RoboBee met the same wall (unstable at the density-scaled frequency).
- *Slow running.* 0.2-1 Hz at the wing through 42:1 is 500-2500 rpm at a 1130 rpm/V motor, 4-20% of no-load speed at 11 V, at high torque. Hobby sensorless ESCs are poor there [spec; general knowledge, not in A13]. Needs a sensored or field-oriented drive: different part, not surveyed.
- *Land.* Walking on the wings needs the wing planted and held while the body is levered over it (A11 C: duty factor 0.62-0.72, "planted flipper does not slip"), and steering needs left and right apart (A11 item 7: recipes that keep left and right independent). One coupled crank gives a symmetric crutch-hop with no steering and no hold. A11 item 10 says so: "a pure up-down crank stroke gives only (iii)", the 5 cm/s shuffle. The fore-aft sweep both A11 and F3 call for is one more powered degree of freedom per wing, and the plan's parts admit no actuator for it above 1-10 W.
- *Immersion.* A13 D: brushless motors run flooded with no lifetime given; potting "not a reliable solution"; sealing a non-waterproof servo "extremely challenging"; nothing supports more than 1 m for a bought servo. The sim's water tasks command 1.5-6.5 m (ROADMAP F3d/F3e). Also heat: at ~100 W electrical and Festo's ~45% electromechanical efficiency (A13, SNIPPET) about 50 W is heat; in a sealed hull in air for 15 minutes that is a thermal problem nobody has sized [spec].

**3.4 SERIOUS — leaving the ground and the water under its own power at 0.5-0.8 kg.** DIAGNOSIS 7c lists "take-off under its own power, entering and leaving the water" as part of done. Evidence against at this scale:
- No bird-type flapper of 0.3 kg or more in A13 has static thrust above weight. Robo Raven II: thrust 1.72-2.32 N on 3.0-3.7 N of weight. Hover machines stop at 127 g (A11 A). The one flapping water exit (FAAV) is under 300 g and worked on one of three wing sets.
- Hover frequency by A11's own identity, m 0.5 kg, S 0.25 m2, R 0.55 m, stroke 80 deg, CL 1.2, r2^2 0.30: f = (1/(pi x 1.40 x 0.55)) sqrt(19.6/0.108) = 5.6 Hz; for thrust/weight 1.3 (A11 item 3), 6.4 Hz [arith]. The drive's ceiling is 5.5 Hz. And CL 1.2 at 80 deg of stroke assumes a stable leading-edge vortex at chord Re ~1e5; Hubel and Tropea (A12 C) report it "not stabilised" at Re 28,000-113,000.
- Power: hover machines 210-220 W/kg at disc loading 5.8-9.5 N/m2; this one is 4.9 N / (1.40 x 0.55^2) = 11.6 N/m2, so roughly 300 W/kg, 150 W to hover and ~220 W at 1.3 [arith, same figure of merit assumed] against "close to 200 W" at the drive's maximum.
- Torque: if T1's ratio of push-up to flight torque (7.9-9.4 against 2.0) carries over, 11-22 N m against 6.5 [spec scaling].
- This collides with 3.1: 15 minutes on cylindrical cells pushes mass up to 0.6-0.85 kg; self take-off wants under 0.3 kg.

So the likely outcome: a 0.6-0.8 kg machine that flies 15 minutes after a hand launch, cannot take off or exit, and whose water and land modes depend on a wing-compliance or fold mechanism nobody has modelled. Or a ~0.25 kg FAAV-class machine that exits the water and flies 5 minutes on a LiPo. The plan should force that choice in week one.

**3.5 SERIOUS — the stop condition tests the wrong thing.** "If no design inside the part limits passes the 15 minute flight gate: stop". Flight time is the requirement that closes. Replace with a week-one closure table, one row per requirement (fly 15 min; swim with the same wings; walk on them; take off; exit water; drive survives immersion; one drive type for all), each with its margin and the row it conflicts with, and stop conditions on each.

**3.6 SHOULD — spar strength, not only stiffness, differs between suppliers.** The plan quotes "stiffness differs 3.9x". A13 C.2 also gives flexural strength 250-400 MPa (Easy Composites) against 1.37 GPa (Goodwinds tubes): 3.4-5.5x. A 6/4 mm tube (I = 51 mm4, Z = 17 mm3) carries 4.3-6.8 N m at the low figure [arith] against 2.8-4.8 N m shoulder torque in flight: a safety factor near 1 before any take-off, water or landing load. T1 already recorded a 23 N slap at 108 g.

---

## 4. The models

| row | is the check enough? |
|---|---|
| UVLM (Ptera) | **No, SERIOUS for one check.** "efficiency peak at Strouhal 0.2-0.5" cannot come out of an inviscid code: A12 C quotes "Inviscid theory predicts efficiency close to 100% under light thrust conditions and for low frequency". A peak appears only after a drag term is added by hand, and its position is then set by the number typed in. Elliptic loading and Wagner are sound but steady or 2-D. The "one published ornithopter case" is one case, pressure-derived lift only, digitised from a figure, with a phase error (A12 A). No thrust or power comparison exists, and thrust and power are what the design is ranked on. Per-candidate cost on this laptop is unknown (A12 item 3: under a minute to 22.5 min). SHOULD: time it before deciding where it sits. |
| Quasi-steady + MuJoCo | "the vortex lattice where both apply" is the attached-flow overlap only. Where this model makes its decisions (take-off, hover, rowing: high incidence) the UVLM is invalid, and `docs/model_validity.md` rates its own deep-stall branch "usable for drag, not for lift" with `alpha_stall` "calibrated, no source". "published flapping force data" at this Re: A12 found none tabulated. So the take-off and rowing decisions rest on a fitted lift curve with no check. **SERIOUS** given 3.4. |
| High-fidelity spot checks | Cannot be trusted at Re 3e4-2e5 (2.4). Cannot do the free surface (plan admits). A few under-resolved points a month bound nothing. See 9. |
| Actuator and battery | "measured power of two published ornithopters" checks the product drive x wing x airframe, not the drive. E-Flap's 175 W/kg is an upper bound assuming the pack was emptied (A13 E.1). Needs a bench curve (2.1). The cell model is the one row that is sound: datasheet rate tables exist. |
| Structure and compliance | "the bought spar, measured" checks a beam. The decision that matters is the wing's twist and bend under load in air and in water, and the plan's own blind-spot cell says the only compliance element is "over-damped" (F1a: "a passive outer wing in this simulator is close to rigid"). **SERIOUS**: see 4.1. |
| Sensors | Datasheets are fine for noise and range. See 7. |

**4.1 SERIOUS — regimes no row covers and the plan does not admit.**
- *The compliant wing.* Every flyer in A13 B.1/C.4 is a membrane on a spar whose twist under load sets the angle of attack; F1a measured twist worth 35-58% of cruise power; FAAV's water mode *is* wing bending. The UVLM takes prescribed shapes (Ptera's aeroelastic module is "beta", one torsion spring per section), the strip model's hinge is critically damped. The plan's "five wing shape fields" (chord, twist, camber, thickness, dihedral) describe a rigid shape; stiffness distribution, the property a builder actually tunes, is not in the representation.
- *Hydroelastic behaviour of an air wing under water* (3.3). No tool, no data.
- *Flapping-averaged stability of the free body* with a tail in the wing's pulsed wake. E-Flap and FAAV both steer with a tail; "tail-to-body distance" decided FAAV's exit (A11 B).
- *Ground contact of a compliant wing* (walking, the push-up, landing): MuJoCo contact on rigid plates.
- *Wet wing in air after exit*: water carried in fabric and trapped in the frame; FAAV used hydrophobic coating.
- *Drive dynamics*: the crank's non-sinusoidal stroke, gear backlash, torque ripple into the body at the flapping frequency.

**4.2 SERIOUS as written, SHOULD if rewritten — "tip and wake effects must be imported".** Sound form: sectional forces from the strip model, induced velocity per strip per time step from a vortex wake (A12 item 4 says exactly this; it is how nonlinear lifting-line and FLOWUnsteady work). Unsound form: a correction factor or table fitted in the UVLM and multiplied onto strip forces. The UVLM's wake strength comes from attached-flow circulation; the strip model is consulted where the flow is separated; a factor fitted in one and applied in the other is a number nothing can check, and it would be the single largest untested coefficient in the design. The repo already has a version of this (`wakefb.py`, "times a gain", gain 1 "a quarter to a third" of Wagner, gain 3 "about its size"): a gain with a threefold range is the warning. If imported at all: coupled, restricted by a guard to incidences below stall, and the guard's trip rate reported per elite.

---

## 5. "Guarantee it evolves in the right direction"

The four checks (known answers; repeatability and heritability; random search and hill climbing at matched evaluations; elites re-scored in a higher model) establish that **the search optimises the model**. None tests whether the model's uphill is reality's uphill. **SERIOUS.**

What passes all four and is still an artefact:
- An elite that sits where both models share a blind spot. Low frequency, large amplitude, high aspect ratio, sharp taper: the strip model has no tip penalty (ROADMAP: "a rectangular wing pays nothing for its tips"), the UVLM has no stall. Re-scoring in the UVLM agrees, because it is wrong the same way.
- An elite on the rigid-wing optimum, which a membrane wing cannot hold (4.1).
- An elite exactly on the gates. "feasibility gates first ... then energy" drives every survivor to zero margin on torque, speed and pack, where a 30% error in the span-cubed rule (A13 limit 10) or the 122-175 W/kg spread removes it. (Also: DIAGNOSIS 7c says "energy efficiency is not counted"; if energy is back as the objective, he should say so.)
- The two known answers themselves. Elliptic loading is recovered in the steady inviscid model only (A12 item 10: the strip model "cannot pass (a) for a planform change, by construction"), a regime no design decision is made in. The Strouhal peak is typed in (row 1 of the table above). The quarter-beat feather lead is near-geometric in a quasi-steady model.

What a sceptical aerodynamicist would ask for:
1. **Reproduce a real machine end to end before designing one.** Put E-Flap (0.51 kg, 1.5 m, 0.438 m2, 3.5-4 Hz cruise, 2-6 m/s, 89 W, hinge 6.5 N m) and Robo Raven II (lift 2.59-3.47 N, thrust 1.72-2.32 N, 13.6-17.6 W, 4 Hz) through the whole pipeline with predictions written first. This is the project's own rule ("re-measure on the current population, not a fixture") turned on the model. One day, no hardware. If the pipeline says E-Flap cruises on 20 W, stop there.
2. **A rank test on hardware.** Three to five printed wings chosen by the model to span good to bad, including one it predicts is poor; cycle-mean thrust and electrical power on the bench; the model's rank order written first. Koos et al. (A12 E) needed about ten transfers. This is the only "higher model" that does not share the blind spots.
3. **A design-space null**, the analogue of the still machine: a flat rectangular plate, and the known-bad wings. If the search's elites do not beat them on the bench by more than repeat noise, the shape search is decoration.
4. **Sensitivity of the optimum to each uncalibrated coefficient** (stall angle, post-stall lift, wake gain, spar EI 1.7-6.8 N m2). If the elite changes identity inside a coefficient's band, the direction is set by the coefficient.
5. **A validity guard**: per elite, the fraction of the stroke spent outside each model's stated envelope (incidence past stall in the UVLM, reduced frequency, Re). Elites that live outside are flagged, not ranked.
6. **Margins as the objective**: maximise the worst gate margin under the uncertainty bands, not energy after zero-margin gates.

Repeatability, heritability, matched random search and hill climbing: sound, keep (the ROADMAP's own history is why).

---

## 6. Motion fitted to each wing

Plan: "a low-dimensional stroke optimised per candidate inside the actuator limits, starting from the parent's" and, in Phase 4, "model-based first, learning only where it measurably helps". The owner's sentence as given to me ("the generated wing is used for learning so that the motion fits the evolved wing, not a clumsy controller") I could not find in the repo, so I take it as quoted. On that reading the plan substitutes a different thing and does not tell him. **SERIOUS as a decision made silently; put it to him.**

What an open-loop periodic stroke loses against learning:
- Feedback. It cannot hold a body that "does not glide"; so it is scored on a rig or behind a separate stabiliser (2.3), and the wing is then fitted to the stabiliser, not to its own motion.
- Anything aperiodic: push-up, take-off, water exit, landing, turning. These are where the wing's shape matters most (3.4).
- Use of the wing's own dynamics (compliance, phase of passive twist), which is what "fits the wing" means in animals. Moot until 4.1 is fixed: with a rigid wing there is little for learning to find.
- It also loses less than it seems on a one-degree-of-freedom crank: frequency, a tail, and perhaps a fold are the whole action space. Most of "motion" is then set by the linkage geometry, which is a design variable, not a control. The plan does not notice this.

What "properly" costs: per-candidate RL inside the search is out (A12 item 11: DERL 2e10 interactions; the ROADMAP's own 1e3-1e4x). The shared body-conditioned policy is the published cheap route and failed four times here (memory, arch33). Affordable middle:
- **Closed-loop parameters in the same optimiser.** Stroke parameters plus a handful of feedback gains and a tail law, read through the sensor model, 10-20 numbers, CMA-ES warm-started from the parent. Same cost class as the plan's stroke fit, and the motion is fitted on the free body.
- **Learning on finalists only, as a check of the proxy.** After each design pass, train a policy on the top 3-5 and two mid-ranked candidates and ask whether the ranking survives. Collection throughput measured here: 580 decisions/s on 8 processes, 1e7 decisions in 4.8 h (ROADMAP P0.12, on a different body at 25 Hz; the flapper's rate is not measured). USD 31 buys ~20 h of a 64-vCPU spot instance (A12 B: 1.571 USD/h), of order 2e8 decisions a month if it scales [spec]. So 5-10 single-body learning runs a month are affordable; thousands are not. This counts against the three-run rule only if he says so: ask.

---

## 7. Sensors and control

**SHOULD — listed carefully, used nowhere.** Phase 0's sensor line is good ("every quantity a controller reads must map to a real sensor"). But sensors then vanish until Phase 4: no sensor is in the Phase 2 representation, none is in a gate, and Phase 3 scores designs with simulator state. Position "part of the design" (Purpose) is not something the evolution or anyone else decides in the plan.

Concrete gaps:
- The current law's observation includes "signed distance to the waterline and its bearing", "clearance", "each rotor's wetness", "the air task's height error" (ROADMAP E0s). No hobby sensor gives distance and bearing to a shoreline. Barometer height is good to metres (Bar30: +/-2 m absolute, A13 D); a barometer port also has to survive immersion and reads wing pressure pulses [spec].
- Airspeed at 2-6 m/s: dynamic pressure at 4 m/s is 10 Pa; hobby pitot sensors are marginal there, in a flow pulsed by the wings, and the port floods [spec].
- No position under water (no GPS), heading from a magnetometer beside steel-can cells and motor currents [spec].
- Radio: 2.4 GHz does not usefully penetrate water (web check below: shallow depth only; no clean dB/m figure found). The plan's "link loss under water handled" is right but understated: under water the machine is autonomous, always. That makes "flown by a person" false for one of three media and should be said.

**SERIOUS — the hardest control problem is not in the plan: attitude with flap-rate authority that the hardware does not have.** The body heaves and pitches at the flapping frequency (3-6 Hz), which is inside the band the attitude loop needs; the accelerometer's gravity reference is buried under that periodic acceleration, so the estimator needs flap-phase-aware filtering. The actuators are a tail on servos (0.05-0.18 s per 60 deg, A13 B.2) and whatever the crank averages over a beat, so usable bandwidth is about 1 Hz. F4's law used per-wing feather "every step" on an airframe that diverges in one second. Either the airframe is passively stable when flapping (tail volume, centre of mass, dihedral: design variables that must be in Phase 2 with a stability gate measured hands-off), or it does not fly. "Vibration needs hardware" in the table undersells this: it is a design requirement, and an IMU log from the L2 bench rig answers the estimator half early.

Second hardest: estimating state through the crossings (which medium, how deep, which way is up when the accelerometer saturates at entry).

---

## 8. Missing entirely

SERIOUS
- **Wing stiffness distribution as a design variable**, and how it is built and measured (4.1).
- **Landing.** Air to land and air to water with an airframe that does not glide. Not mentioned.
- **Buoyancy, trim and ballast**: centre of mass against centre of buoyancy against centre of lift in three media; 0.6 kg needs ~600 cm3 displaced to be neutral, the cells displace ~75 cm3 [arith]; trapped air in wings; a hull that size is flight drag. F3e: the neutral body rolls over in 6 s when left alone.
- **Thermal** in a sealed hull (3.3), and the opposite in water.

SHOULD
- Mechanism: linkage geometry, bearing loads, gear wear, backlash; fatigue of printed parts (4 Hz x 15 min = 3,600 cycles a flight); PLA softens at 57 C (A13 C.1) beside a 50 W heat source; print orientation (strength across layers 30-47 MPa).
- Cells: immersion, short-circuit and venting inside a sealed hull, cold-water capacity (M50: 18.2 -> 14.6 Wh at 0 C), low-voltage cut-off in flight, a pack that must be removable to charge through a seal.
- Fresh or salt water. The sim has a beach. Salt changes corrosion, conductivity and radio.
- Crash tolerance and repair time; trimming an ornithopter means many crashes. A12/A13 give E-Flap's wing as rebuilt "in less than 3 hours": a useful design target.
- Adjustability: the build needs trim knobs (centre of mass, tail incidence, stroke plane, wing tension) and a system-identification plan saying which test fits which parameter. "Measurements correct the models" has no procedure.
- Firmware and logging: which autopilot stack flaps (none of the common ones does out of the box [spec]), log rates fast enough to see inside a beat.
- Where it may be flown and tested: a 0.6 kg, 1.2 m machine outdoors; a tank deep and wide enough for a 1.2 m span. Taiwan requires registration from 250 g (web check below; licensing between 250 g and 2 kg conflicted between sources: check the CAA portal).
- Parts budget and lead times; one supplier per critical part, measured on arrival.
- What F0-T1 still mean. "keeps its methods and instruments and is not the design" is right; say also that its verdicts (walks, swims, takes off) are void at the new scale and drive.

MINOR
- "18650 or 21700" in Purpose, P45B in the limits; the 0.41-0.60 kg figure in A13 limit 7 is for 3 x 18650.
- Principle 4 (a matched-budget control for every component) is not applied to any named component; say which control goes with the UVLM import, the CFD checks, the tandem wake.

---

## 9. Complexity

Most likely to be found inert or harmful, by this project's own history (an inert critic, a shared policy never scored, a passive twin costing 38%, structural operators that "dilute parameters"):

1. **High-fidelity cloud CFD.** Cannot be validated at this Re, cannot do the free surface, a few points a month. Cut; the bench is the higher model. Spend the USD 31 on finalist learning or on parts.
2. **The UVLM-to-strip import as a correction factor** (4.2). Either coupled induced velocity with a guard, or nothing.
3. **Tandem wake modelling**, which exists only because of two wing pairs, which were chosen at 108 g with ideal servos. Every machine with a precedent at this scale (E-Flap, Kestrel, Robo Raven, FAAV) is one pair and a tail. Reopen the count before modelling its wake.
4. **Topology in the genome**: "body, wing count and placement ... drive type". The parts table admits one drive type that can fly this mass; wing count is one or two. These are two or three discrete options to be compared by L0/L1 passes, not dimensions for evolution to wander in.
5. **The search machinery itself** if MAP-Elites, islands and tiers are carried over. For ~15 continuous parameters, CMA-ES with restarts, a matched random arm, and three seeds.
6. **Energy as the objective** after the gates.

**Smallest plan that could still reach the purpose** (and still gives the shape to evolution, which is his requirement):
- Week 1: L0 closure table with a stop condition per row; the three questions below to the owner; E-Flap and Robo Raven II reproduced in the present pipeline (5.1); order one drive, spar stock and skin.
- Weeks 2-4: L1 on one crude reference machine with a torque-limited crank, free body, sensor-limited stabiliser; L2 bench for the drive curve and three wings in air and in a tub; correct the strip model from it. Fix the compliance element first: without it nothing below is about a real wing.
- Then, repeatedly: evolution over wing planform, stiffness distribution, linkage geometry, tail and mass placement (~15 numbers), closed-loop motion parameters fitted in the same loop, margins as the objective, the guard and the nulls of section 5; each pass ends by printing the top wing and one predicted-poor wing and measuring both; finalist learning as a check of the motion proxy.
- Build the whole machine early and badly (pass 2 or 3), tethered, hand-launched; crossings added as they are earned.

Questions that are the owner's and should be asked before draft 2:
1. One charge for 45 minutes, or 15 per charge?
2. If self take-off and 15 minutes cannot both be had (3.4): which gives, or two machines?
3. Is motion to be *learned* per wing (finalists only is what the budget allows), or is a fitted closed-loop stroke acceptable?
4. Are two wing pairs still his choice once the drive is a crank?
5. Fresh water or sea?

---

## What I checked, and what I could not

Checked by arithmetic on A11/A13 rows: everything marked [arith]. Checked by reading: every quote from PLAN, A11, A12, A13, ROADMAP, DIAGNOSIS. Web: (a) 2.4 GHz in fresh water: sources agree it works at shallow depth only; I found no trustworthy dB/m figure (Sendra et al., IEEE Commun. Lett. 2013 / Sensors 2016 not opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC4934316 , https://dissem.in/p/17963051/underwater-wireless-communications-in-freshwater-at-24-ghz). (b) Taiwan: registration from 250 g is consistent across secondary sources; the licence tier for 250 g-2 kg is not (https://drone-laws.com/drone-laws-in-taiwan , https://www.elitelaw.com/new-drone-regulations-to-come-into-force-in-2020/); primary CAA page not opened.

Not checked:
- No simulation or timing was run; the per-candidate cost of anything is unknown to me.
- The scaling of T1's take-off torque ratio and of hover power to 0.5 kg is extrapolation ([spec]/[arith] as marked), not a measurement; an L1 pass would replace it.
- E-Flap's battery mass, receiver and wiring masses are my estimates.
- Sensorless ESC behaviour at low speed, pitot behaviour at 4 m/s, thermal load in a sealed hull, autopilot support for flapping: general knowledge, no source opened.
- The owner's sentence about learning was not found in the repo.
- A9 not re-read; FAAV's paper is press-only in A11, so every use of FAAV here inherits that.
- What `physics/fluid.py` actually does at high incidence beyond the `model_validity.md` lines grepped.
