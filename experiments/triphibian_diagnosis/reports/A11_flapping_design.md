# A11 — Literature: numbers for sizing one flapping wing that flies, swims and carries the body on land

Survey date 2026-11-12. Web search and page fetches only; no code, no simulation. Follows
`A9_lit_hardware.md` (its sections 2 and 3 are not repeated; rows that A9 already holds are referred to as "A9").

Labels, per number:

- **VERIFIED** — the number was seen today on the fetched page or in the fetched PDF. "press" = the page is a
  press article, not the paper. "as cited in X" = seen in X, which attributes it to the named source.
  "fig." = read off a figure in the fetched PDF (reading error of a few percent).
- **SNIPPET** — only in a search-result summary or a secondary page that was not opened.
- **INFERRED** — arithmetic on labelled numbers; the arithmetic is shown.
- **UNKNOWN** — not found; what was tried is in the last section.

Local text extraction (`pdftotext`) was used on PDFs that the fetch tool saved but could not read; the URL given is
the one fetched.

---

## A. Hover-capable flapping robots, 3 g to 130 g (plus two membrane-wing forward flyers)

| system | mass | key numbers | mechanism in one line | URL | label |
|---|---|---|---|---|---|
| KUBeetle-S (Phan, Aurecianus, Au, Kang, Park; RA-L 2020) | 15.8 g (18.1 g with payload) | span 20 cm; wing length 9 cm, area 25.5 cm² per wing (23.7 cm² after the inboard cut-out), aspect ratio 3.2; stroke ≈ 190°; hover ≈ 18 Hz; peak lift/weight 1.34 at 3.7 V; hover 8.8 min on 160 mAh 1S (4.0 g); take-off 1.1 m/s, 2.5 m/s translation. Mass budget: motor 3.5 g, mechanism 2.6 g, 3 servos 1.8 g, wings 0.4 g, battery 4.0 g. Wing loading "about 37.5 N/m²" as stated. | Two wings; one coreless DC motor (Chaoli CL720) through 28:1 gears, four-bar linkage plus pulley-string to amplify the stroke; pitch and roll by tilting the stroke plane (which also changes twist through the constrained root spar), yaw by moving the two root spars in opposite directions; three 0.6 g servos. | https://arxiv.org/pdf/2005.06715 | VERIFIED |
| KUBeetle-S, derived | | Stated 37.5 N/m² reproduces with 18.1 g on 2 × 23.7 cm² (0.1776 N / 0.00474 m² = 37.5); at 15.8 g the same area gives 0.1550 / 0.00474 = 32.7 N/m². Disc loading 0.1550 N / (Φ R² = 3.316 × 0.0081 m²) = 5.8 N/m². Mean wingtip speed 2ΦfR = 2 × 3.316 × 18 × 0.09 = 10.7 m/s. Input power ≈ 15.8–16 gf ÷ 4.6–4.8 gf/W (Fig. 5A, 8A) = 3.3–3.5 W = 209–222 W/kg. Re = 2c̄ΦfR/ν = 2 × 0.0263 × 3.316 × 18 × 0.09 / 1.5e-5 ≈ 1.9e4. | | same | INFERRED (fig. for power) |
| KUBeetle-S wing | wings 0.4 g (pair) | Membrane 10 µm Mylar, veins 0.3 mm carbon rod, frames 0.8 mm carbon/epoxy; membrane forms a loose sleeve on the leading edge and the root spar so it rotates freely (passive twist). Geometric angle of attack 60–90° from root to 25% span, falling to ≈ 20° at the tip (Fig. 6B). Cutting the membrane from root to 25% span: lift −2%, lift/power +4.8 ± 1.2%. A 190° stroke gives 28.9 ± 6.8% more lift/power than 120°; +27% wing area gives +10.9 ± 4.1% lift/power. | Passive twist set by a slack membrane on a free sleeve; the root spar angle is the control input. | same | VERIFIED (tip angle: fig.) |
| DelFly Nimble (Karásek et al., Science 2018) | 29 g ("just under 29 g") | span 33 cm; wing length 14 cm; ≈ 17 Hz at hover (16 Hz peak in flight spectra); thrust/weight > 1.3; payload 4 g; hover > 5 min; 3 m/s best range (> 1 km); 7 m/s top. | Four wings in two clap-and-fling pairs, each pair with its own motor and gear-crank mechanism; roll = left/right flapping-frequency difference; pitch = one servo changes the dihedral of both mechanisms (moves the mean thrust fore/aft of the centre of mass); yaw = one servo deflects the wing roots of the two pairs in opposite directions. | https://delfly.nl/ ; https://pure.tudelft.nl/ws/portalfiles/portal/84439908/6.2020_1964.pdf | VERIFIED |
| DelFly Nimble stroke | | Stroke angle 44° per wing on a DelFly II wing pair of 140 mm half-span (a DelFly test rig in an IMAV paper); the Nimble inherits the DelFly II wings and mechanism. If 44° holds: swept area 4 × ½ × 0.768 rad × 0.14² = 0.0301 m², disc loading 0.2845 N / 0.0301 = 9.5 N/m². Wing area, membrane thickness, gear ratio: UNKNOWN. | | search summary of imavs.org papers | SNIPPET; disc loading INFERRED on it |
| Flapper Nimble+ (commercial descendant of the Nimble) | 102 g minimum, 114 g with shells and gear, 127 g maximum | span 49 cm; 12 Hz at hover, 20 Hz at full throttle; payload 25 g; 8 min at 3 m/s (102 g), 5 min hover at maximum payload; LiPo 2S 300 mAh; 2 brushless motors + 2 coreless servos. | Same four-wing, two-mechanism layout as the Nimble at 3.5 times the mass. | https://flapper-drones.eu/nimbleplus/ | VERIFIED |
| Flapper Nimble+, derived | | Upper bound on thrust/weight if thrust ∝ f² at fixed amplitude: (20/12)² = 2.8. Upper bound on hover power: 7.4 V × 0.3 Ah = 2.22 Wh in 5 min = 26.6 W = 210 W/kg at 127 g. Scale check against the Nimble with f ∝ √m / span²: 17 Hz × √(102/29) / (49/33)² = 17 × 1.876 / 2.205 = 14.5 Hz predicted, 12 Hz stated. | | same | INFERRED |
| Quad-thopter (De Wagter, Karásek, de Croon; IMAV 2017) | 33 g | 28 cm tip to tip; four wing pairs; fastest pair 15 Hz; 40° roll step in < 266 ms (< 4 beats); 3.5 m/s and 80° roll in a 3 m side-step; flight ≥ 9 min. | Four independent flapping pairs arranged like a quadrotor: roll and pitch by thrust difference, yaw by a fixed tilt of diagonal pairs; no servos. | https://www.imavs.org/papers/2017/36.pdf | VERIFIED |
| Tailless triple-flapping-wing robot ("T3"; arXiv 2503.00805) | 37.4 g | wingspan 140 mm, aspect ratio 1.75; up to 25.1 Hz; maximum stroke 130°; 18.1 g lift per module, 54.2 g total, lift/weight 1.45 (54.2/37.4 = 1.449); 5.5 m/s; ≈ 6.5 min; 380 mAh 1S. Wing: 5 µm polyester film, 0.3 mm carbon rods, 1 mm leading edge. | Three modules, each a coreless motor through 1:25.4 two-stage gears and a double crank-rocker; lift, pitch and roll by the three throttles; no active yaw. | https://arxiv.org/html/2503.00805 | VERIFIED |
| Purdue hummingbird robot (Fei, Tu, Yang, Zhang, Deng; ICRA 2019) | 12 g | span 168 mm; two motors, one per wing, with a torsion spring for resonance; thrust 0.11 N at ≈ 9.5 V to 0.225 N at 16 V (Fig. 5); stroke ≈ ±60°, passive rotation limited to ≈ ±45° (Fig. 6); wingbeat period ≈ 0.029 s in Fig. 6. Control torque maps (Fig. 5): roll ≈ ±3 N·mm for ±3 V amplitude difference; pitch ≈ ±0.5–0.8 N·mm for ±2 V stroke bias; yaw ≈ ±0.6 N·mm for split-cycle ±0.3. Wing: "bi-stable design with the majority of area constructed as a rigid plate". | Direct drive, no crank: each motor oscillates its wing; thrust = amplitude, roll = amplitude difference, pitch = stroke-centre bias, yaw = split cycle (faster one half-stroke). | https://arxiv.org/pdf/1902.09628 | VERIFIED (curves: fig.) |
| Purdue, derived | | Thrust/weight on the bench 0.225 / (0.012 × 9.81) = 1.91; frequency 1/0.029 s ≈ 34 Hz (press pages give 30–40 Hz, SNIPPET). | | same | INFERRED |
| KU Leuven stroke-cam flapper (bench; Leys et al. 2016) | 3.39 g set-up (motor 1.64 g, wing 0.048 g) | wing 50 mm long, mean chord 18 mm; 175° peak-to-peak at 40 Hz; pitch setpoint 45° on both half-strokes; lifts 5.1 g (5.1/3.39 = 1.5); thrust 17.1–51.1 mN over 25–40 Hz; 6 mm coreless motor, total gear ratio 14.75; membrane 15 µm polyester; 0.1 mm nylon cables. | A cam prescribes the stroke; pitch is set, not left to the membrane. | https://pmc.ncbi.nlm.nih.gov/articles/PMC5004601/ | VERIFIED (ratio INFERRED) |
| NUS wing-root-deflection rig (Chan, Nguyen, Debiasi; IMAV 2014) | wing 0.6 g each | half-span 90 mm, chord 50 mm; stroke ±64°; 13–18 Hz; brushless OverSky HP03S 16600 kV through 16:1 gears and a four-bar crank-rocker; membrane 40 g/m² PE-laminated non-woven fabric; leading edge 0.8 mm carbon rod, root 0.5 mm piano wire, stiffener 0.3 mm carbon rod; 10° slack at the root. | Tailless pitch and yaw from deflecting the wing-root spar fore/aft (range from −6.7°), which slackens the membrane on one side of the stroke. | https://www.imavs.org/papers/2014/paper09.pdf | VERIFIED |
| AeroVironment Nano Hummingbird | 19 g (17.5 g "Saturn" version) | span 160 mm; 11 mph (18 km/h); endurance 4 min with payload, 11 min without; wing loading ≈ 45.7 N/m². Flapping frequency, stroke, power: UNKNOWN. | Two wings; wing-twist modulation through the root spars for roll, pitch and yaw; string-based mechanism on a continuously rotating crankshaft (SNIPPET). | https://en.wikipedia.org/wiki/AeroVironment_Nano_Hummingbird ; https://arxiv.org/pdf/2005.06715 | VERIFIED (mass, span on Wikipedia; endurance and wing loading as cited in Phan 2020); mechanism SNIPPET |
| Nano Hummingbird, derived | | Total wing area 0.019 × 9.81 / 45.7 = 40.8 cm². | | | INFERRED |
| COLIBRI (ULB; Roshanbin et al. 2017) | 22 g | span 21 cm; 22 Hz; hover 15–20 s; later thesis 23 g, 21 Hz. | Two wings; pitch and roll by wing-twist modulation (camber change), yaw uncontrolled in 2017; height by frequency. | https://research.tudelft.nl/en/publications/colibri-a-hovering-flapping-twin-wing-robot | SNIPPET |
| Bat Bot B2 (Ramezani, Chung, Hutchinson 2017) — forward flight, not hover | 93 g | 56 µm silicone membrane; 7–10 Hz; 9 joints (5 active, 4 passive); span ≈ 47 cm (SNIPPET). | Articulated carbon skeleton under a stretchable membrane; the wing folds and the legs move the trailing membrane. | https://ece.illinois.edu/news/3716 | VERIFIED (press); span SNIPPET |
| Aerobat (Northeastern) — forward flight, not hover | UNKNOWN | maximum span 30 cm; brushless motor with a 75:1 gearbox; the two wings coupled by 7 spur gears; membrane is a flexible PCB; tail-less and pitch-unstable; no hover or untethered flight reported on the page. | One motor drives gears, cranks and four-bar "computational structure" that folds the wing on the upstroke. | https://ar5iv.labs.arxiv.org/html/2103.15948 | VERIFIED |
| Festo BionicFlyingFox — forward flight | UNKNOWN | span 228 cm. | Membrane-wing semi-autonomous flyer; human starts and lands it. | Festo brochure via search | SNIPPET |

Reference animals quoted in the same sources: rufous hummingbird 3.4 g, ≈ 44 Hz, stroke 112°, stroke plane ≈ 14° from
horizontal, 56 Hz at maximum load (VERIFIED, PMC5004601); rhinoceros beetle wing loading 38.9 ± 3.7 N/m², stroke
"more than 180°" (VERIFIED as cited in Phan 2020).

Rows found (INFERRED from the table): every hover machine here except the Purdue robot (one oscillating motor per
wing, no crank) turns a continuously rotating motor through 14.75:1 to 28:1 gears and a crank or linkage; the
frequency at weight support falls from 40 Hz (3.4 g bench set-up) to 12 Hz (102 g); thrust/weight is 1.3–1.45 in
free flyers and 1.5–1.9 on a bench; two-wing machines use 112–190° of stroke, four-wing clap-and-fling machines
about 44° per wing (SNIPPET).

---

## B. One wing in air and in water

| system | mass | key numbers | mechanism in one line | URL | label |
|---|---|---|---|---|---|
| FAAV (Zufferey et al., Science 393:207, 2026) | < 300 g | Three wing sets 60, 80, 100 cm; only the 80 cm set did all of flying, swimming and transitioning reliably. ≈ 5 Hz in both media; swims ≈ 1 m/s, flies ≈ 6 m/s. Exit: starts ≈ 0.5 m deep, body pitched 70°; steeper tips it back into the water, shallower lets the wingtips hit the surface. Wings: membranes coated with hydrophobic nanoparticles, "flexible enough to minimize flapping amplitude in water" and stiff enough for flight. Tests in a tank and in Lake Geneva. No turning. | One waterproof motor and a crankshaft flap both wings up and down; a motorised tail sets pitch for climb, dive and exit; frequency and tail angle are pre-programmed sequences. No feet, no wing folding. | https://techxplore.com/news/2026-07-birdlike-robot-underwater-flight.html ; https://actu.epfl.ch/news/a-flapping-robot-swims-and-flies-like-a-diving-b-2/ | VERIFIED (press) |
| FAAV, abstract | | The paper's own summary: frequency adjustment, flexible wings and strong actuation give the transitions; large wings improve flight without greatly reducing underwater efficiency; tail-to-body distance and egress angle affect the exit. | | https://dspace.mit.edu/handle/1721.1/169536 | VERIFIED (abstract) |
| FAAV, derived | | Advance per stroke: water 1 / 5 = 0.20 m, air 6 / 5 = 1.2 m. For St = fA/U in 0.2–0.4 the tip excursion would be A = 0.04–0.08 m in water (10–20% of a 0.4 m semi-span) and 0.24–0.48 m in air (60–120% of it). Same frequency, so the amplitude must fall about 6-fold in water; the press says the wing's bending does that. | | | INFERRED |
| Hybrid RoboBee (Chen et al., Sci. Robot. 2017) | 175 mg | Earlier 85 mg design: 140 Hz in air; the density scaling relation gave 5 Hz in water, at which the body pitched 14.8° and plunged; stable at ≥ 9 Hz (9.4° at 11 Hz; simulated ascent 80 mm/s mean). Final design: 9–13 Hz in water, so air frequency raised to 220–300 Hz by cutting wing area 40%. Actuators failed above 15 Hz in water. | Same wings and passive hinges in both media; only the drive frequency changes. Low frequency couples wing and body, which sets a lower bound on the water frequency. | https://smrl.mit.edu/wp-content/uploads/2019/10/science_robotics_fly_swim_robobee.pdf (copy fetched for A9, text read today) | VERIFIED |
| Dual aerial/aquatic foil with in-line motion (Izraelevitz, Triantafyllou; ICRA 2015) | bench | Chord 152 mm, semi-span 3 chords; towed at 0.2 m/s in water (Re 30,000, equal to 2.5 m/s in air); St = 0.4, heave/chord 0.75, 0.35 Hz; maximum angle of attack 45°. Symmetric stroke (β = 90°): Cx = 0.20, Cy = −0.029. Stroke swept backward on the downstroke (β = 135°, turtle-like): Cx = 0.24, Cy = 0.38. Stroke swept forward on the downstroke (β = 45°, bird-like): Cy = 1.1, "over triple" the symmetric case at the same angle of attack and frequency. Wing: NACA0013 ribs, 0.5 mm mylar skin, interior flooded. Valid range claimed Re 3,000–300,000. | A third degree of freedom moves the wing fore/aft during the stroke; the sign of that sweep selects weight support (air) or thrust (water) from one wing. Driven by three Dynamixel EX-106+ through cables. | https://arxiv.org/pdf/1412.3843 | VERIFIED |
| Guillemot-inspired multi-modal wing (Lock, Vaidyanathan, Burgess; 2010, 2014) | model / bench | Bent wings in swimming cut profile drag "by as much as 50%"; design target 20 m/s over 2 km in air and 1.5 m/s over 500 m in water. The 2014 paper tests extended and retracted foils on a 2-DOF flapping rig for efficiency, thrust coefficient and torque. Optimal retraction: UNKNOWN. | Span and area are reduced for water; the stroke stays symmetric. | https://phys.org/news/2010-11-seabird-morphing-wings-robots.html ; https://research-information.bris.ac.uk/en/publications/impact-of-marine-locomotion-constraints-on-a-bio-inspired-aerial-/ | VERIFIED (press, abstract) |
| Common murre | species set spans 450 g–1 kg | Air 7.0 ± 0.5 Hz, 70 ± 4°; water 2.7 ± 0.6 Hz, 94 ± 13°. St: water horizontal 0.18 ± 0.02, descending 0.24 ± 0.01; air 0.12–0.25 (airspeed-based). Swim 1.63 m/s at 2.4 Hz (wild birds 1.61 m/s, 2.6 Hz). | | https://par.nsf.gov/servlets/purl/10201075 (eLife 9:e55774) | VERIFIED |
| Horned puffin | | Air 8.9 ± 0.6 Hz, 63 ± 7°; water 2.3 ± 0.3 Hz, 97 ± 8°. St water 0.13 ± 0.01 / 0.21 ± 0.04; air 0.13–0.27. | | same | VERIFIED |
| Pigeon guillemot | | Air 11 ± 1.3 Hz, 81 ± 19°; water 2.8 ± 0.3 Hz, 96 ± 10°. St water 0.15 ± 0.01 / 0.29 ± 0.06; air 0.18–0.27. | | same | VERIFIED |
| Tufted puffin | | Air 9.0 ± 0.3 Hz, 63 ± 4°; water 2.2 ± 0.3 Hz, 82 ± 8°. St water 0.15 ± 0.02 / 0.29 ± 0.03; air 0.16–0.25. | | same | VERIFIED |
| Four alcids, pooled | | Stroke-plane angle to the direction of travel 79 ± 7° in air, 92 ± 7° in horizontal swimming, 93 ± 12° descending. Stroke duration 2–3 times longer in water. Stroke angular velocity: downstroke ≈ 1000–1600 deg/s in air against ≈ 480–620 deg/s in water; upstroke ≈ 300–480 deg/s in water (Fig. 3). Chord angle to the body in horizontal swimming 10–30° on the upstroke, 5–40° on the downstroke (Fig. 6); the upstroke chord angle rises with descent angle. Amplitude in water measured at the wrist, in air at the wingtip. | In water the wing is flexed at the wrist and the stroke plane rotates forward by ≈ 13°; in horizontal swimming the upstroke is a quick, flat stroke that pushes down against buoyancy. | same | VERIFIED (velocities, chord angles: fig.) |
| Alcids, derived | | Air/water frequency ratio 7.0/2.7 = 2.6, 8.9/2.3 = 3.9, 11/2.8 = 3.9, 9.0/2.2 = 4.1. | | | INFERRED |
| Rhinoceros auklet (Kikuchi et al. 2015) | | Flying 8.9 Hz, 15.3 m/s, tip excursion 0.39 m; swimming 2.6 Hz, 1.3 m/s, 0.18 m; the smaller excursion in water comes from partly folding the wing at the same dorso-ventral angle (87° water, 88° air: VERIFIED as cited in eLife 55774). St = 8.9 × 0.39 / 15.3 = 0.23 in air, 2.6 × 0.18 / 1.3 = 0.36 in water. | Excursion in water is 0.18/0.39 = 46% of that in air at an unchanged stroke angle, i.e. the working span is roughly halved. | search summary of J. Avian Biol. 46:406 | SNIPPET; St and 46% INFERRED |
| Atlantic puffin under water (Johansson & Aldrin 2002) | | Upstroke produced forward acceleration in 2 of 24 upstrokes; wing pronated at the start of the downstroke, supinated at its end; partly folded wings act as aft-swept tips. | Downstroke carries most thrust; the upstroke is sometimes active. | search summary of JEB 205:371 | SNIPPET |
| Brünnich's guillemot under water | 0.95 kg | 3.2 Hz with thrust on both half-strokes near the surface, 2.4 Hz downstroke-only deeper (A9, section 4). | Thrust on both half-strokes only while buoyancy must be fought. | A9 | VERIFIED in A9 |
| Penguins | 1.1–24.5 kg | 1.35–3.6 Hz (A9). Efficient band quoted for penguin swimming St 0.2–0.4; forward thrust on the downstroke, turning on the upstroke (gentoo). Amplitude and angle of attack: UNKNOWN. | Rigid wing, lift-based, bending improves thrust (title of Harada et al. 2021). | search summaries | SNIPPET |
| Green sea turtle | adult | Limb beat ≈ 0.23 Hz at ≈ 0.6 m/s; 0.18–0.21 Hz between nestings; 60–80 beats/min at the start of a descent falling to 25–40 after 30 s. Loggerhead hatchling swims 5 body lengths/s (VERIFIED, PMC2880067). Amplitude and angle of attack: UNKNOWN. | Lift-based dorso-ventral stroke with fore-aft excursion. | search summaries | SNIPPET |
| Dippers, shearwaters | | Dipper: "partly opened wings" under water, legs assist at the start of a dive. Frequencies: UNKNOWN. | | search summary | SNIPPET |

---

## C. Moving on the ground with wings or flippers

| system | mass | key numbers | mechanism in one line (stance / swing / body) | URL | label |
|---|---|---|---|---|---|
| Common vampire bat, walk and bound (Riskin & Hermanson 2005; numbers as restated in Riskin et al. 2006) | 23.1 ± 2.0 g | 61 treadmill trials, 0.12–1.14 m/s. Walk (lateral sequence) to 0.56 m/s: duty factor forelimb 0.72 ± 0.07, hindlimb 0.62 ± 0.06, body at constant height. Bound 0.28–1.14 m/s (Froude 0.3–4.9): duty factor forelimb 0.62 ± 0.08, hindlimb 0.40 ± 0.10; "the aerial phase is initiated by push-off with the forelimbs". | Folded wing: the wrist/thumb is the foot. In the bound the two forelimbs are the propulsive pair and stay on the ground longer than the hindlimbs; the body leaves the ground each stride. | https://journals.biologists.com/jeb/article/209/9/1725/16963/Terrestrial-locomotion-of-the-New-Zealand-short | VERIFIED (as cited in Riskin 2006) |
| Vampire bat, derived | | Froude length from Fr = v²/(g h): 1.14² / (9.81 × 4.9) = 0.027 m, and 0.28² / (9.81 × 0.3) = 0.027 m: support height ≈ 2.7 cm. Speed in body lengths: UNKNOWN (body length not on the page). | | | INFERRED |
| Vampire bat, jump take-off (Schutt et al. 1997) | | Vertical velocity 2.4 m/s "in less than 30 ms" (VERIFIED as cited in Riskin 2006). Peak force 6.51 body weights, take-off 2.38 m/s (SNIPPET, from a review). The two do not agree: 2.4 m/s in 30 ms is 80 m/s² = 8.2 g mean, i.e. ≥ 9.2 body weights mean ground force. Primary paper not opened. | Body is set over the pectoral limbs, which generate the jump force; hindlimbs do not. | same; search summary | VERIFIED as cited / SNIPPET; conflict INFERRED |
| New Zealand short-tailed bat (Riskin et al. 2006) | 13.9 ± 0.9 g | Treadmill 0.20–0.59 m/s (Froude 2.1); force plate 0.13–0.95 m/s; one gait, a lateral-sequence walk with no aerial phase; kinematically a walk, kinetically run-like; energy recovery 26.0 ± 18.1%. Does not bound. | Folded wings as forelimbs in a four-limb walk. | same | VERIFIED |
| Pterosaur quadrupedal launch (Habib 2008) | Anhanguera ≈ 5 m span, Zhejiangopterus ≈ 3 m; Quetzalcoatlus 10.4 m span, mass estimates 70–250 kg | Evidence is bone strength, not kinematics: in 155 birds of 20 species the femur is stronger than the humerus above ≈ 500 g; the three pterosaurs have humerus/femur strength ratios above every sampled bird. Vampire-bat humeri are stronger in bending for their weight than other bats'. Launch "under a second" (SNIPPET, press). | Leap dominated by the forelimbs and pectoral muscles (the flight muscles), as in vampire bats. | https://epub.ub.uni-muenchen.de/12011/1/zitteliana_2008_b28_09.pdf | VERIFIED |
| DALER (Daler, Mintchev, Stefanini, Floreano 2015) | UNKNOWN | 20 m/s in the air; 6 cm/s on the ground; maximum step 6 cm (so ≥ 1 step/s, INFERRED 6 / 6). | Wingtips ("wingerons") rotate about a spanwise hinge as whegs; the wing folds so the robot walks on its haunches; wingeron axis near its centre of pressure. | https://actu.epfl.ch/news/a-flying-robot-that-can-walk-3 ; https://robohub.org/daler-a-bio-inspired-robot-that-can-both-fly-and-walk/ | VERIFIED (press) |
| Triple-flapping-wing robot, ground mode (arXiv 2503.00805) | 37.4 g | 5.4 cm/s top speed (5.4 / 14 = 0.39 wingspans/s, INFERRED); 28 min at 5 cm/s; forward only; maximum vibration throttle is half of the hover throttle; self-rights in 0.5 s with the two modules nearest the ground. | Wings do not touch the ground: two modules flap, the vibration passes through asymmetric elastic legs that slip one way; turning by left/right frequency. | https://arxiv.org/html/2503.00805 | VERIFIED |
| RoboFly, ground and water surface (Chukewad et al. 2020) | 74 mg | Wings 13 mm. Ground: 0.5–2.7 cm/s at 40–80 Hz and 200–250 V (2.7 / 1.3 = 2.1 wing lengths/s, INFERRED); lift-off ≈ 140 Hz. Rods of equal mass and inertia in place of the wings: no motion, so the push is aerodynamic drag. Second-harmonic drive, ratio up to 0.3, makes one half-stroke faster. Water surface: ≈ 0.5 cm/s at 35 Hz. Flying costs ≈ 0.02 mJ/mm, about 25 times less than the cheapest ambulation; hover ≈ 50 mW (50 mW / 74 mg = 676 W/kg, INFERRED). | Wings do not touch the ground: a fast backward and slow forward stroke gives a net drag impulse that slides the robot on four passive legs; steering by left/right amplitude. | https://arxiv.org/pdf/2001.02320 | VERIFIED |
| Loggerhead hatchling on sand (Mazouchova et al. 2010) | 19.5 ± 2.2 g | Body length 6.9 ± 1.6 cm; flipper 3.5 × 1.3 cm; diagonal gait; stance duty factor 0.66 ± 0.05; body lifted 2.2 ± 0.9 mm each cycle and set down at its end; stride 4.0 ± 1.9 cm on sand, 4.7 ± 2.9 cm on hard ground (4.0 / 6.9 = 0.58 body lengths, INFERRED); mean speed on sand 28% below hard ground, maxima equal; peak speed 2.88 times the mean; shoulder excursion 111 ± 17°; flipper inserted 0.76 ± 0.13 cm, perpendicular to the surface at mid-stance (99 ± 17°); slip in 5.6% of steps on sand, 2.6% on sandpaper. Up to 3 body lengths/s (SNIPPET), which with a 4.0 cm stride is 20.7 / 4.0 = 5.2 strides/s (INFERRED). | Stance: flipper edge is planted and stays put while the body is levered up and over it; on hard ground a claw at the wrist hooks and the wrist stays straight. Hind feet lift, not push. Body clears the ground during thrust. | https://pmc.ncbi.nlm.nih.gov/articles/PMC2880067 | VERIFIED |
| FlipperBot (Mazouchova, Umbanhowar, Goldman 2013) | 790 g (press); 970 g on another page (SNIPPET) | 19 cm long, limb span 40 cm; two servo-driven front limbs with flat flippers; tested on 122 cm of poppy seeds. A free (passive) wrist beat a fixed wrist: the flipper stays locked in undisturbed material. With shallow penetration or a long stroke the step shortens every cycle and the robot stalls within a few steps. Speed: UNKNOWN. | Both flippers together; body lift and base drag were measured, values not on the page. | https://www.eurekalert.org/news-releases/472530 | VERIFIED (press) |
| Mudskipper and MuddyBot (McInroe et al., Science 2016) | UNKNOWN | Both climbed 20° sand only when the tail pushed in step with the fins; with fins alone they slid back. | Crutching: the two pectoral limbs plant together and lift the body to cut belly drag; the tail adds a push on slopes. Robot: two limbs and a tail. | https://techxplore.com/news/2016-07-robot-animals-million-years.html | VERIFIED (press) |

Share of body weight carried by the wing or flipper: not reported in any row (UNKNOWN). What is reported instead:
duty factors (vampire forelimb 0.62–0.72, hatchling 0.66) and whether the body clears the ground (vampire bound: yes,
aerial phase; hatchling: 2.2 mm; mudskipper and MuddyBot: lifted; T3 and RoboFly: body rides on passive legs).

---

## D. Sizing relations

| relation | formula and numbers | source | URL | label |
|---|---|---|---|---|
| Wingbeat or fin-beat frequency against mass and wing area | Weight support F ∝ ρ A² f² = m g, hence f ∝ √m / A (correlation 0.95 over 414 animals: 176 insects, 212 birds, 25 bats, plus an ornithopter). For swimmers: replace ρ_air (0.0012 g/cm³) by ρ_water (1.0025 g/cm³) and m by the buoyancy-corrected mass m(ρ_water − ρ_animal)/ρ_animal. Fails at very low Reynolds number. The value and exact form of the constant: UNKNOWN (equations are images on the page). | Jensen, Dyre, Hecksher, PLoS ONE 2024 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303834 | VERIFIED |
| Hummingbird hover allometry | f ∝ m^−0.597 in hummingbirds against m^−0.269 in other birds; f ∝ (wing length)^−1.25 against −1.03 in small passerines (Greenewalt 1975). Data point: rufous hummingbird 3.4 g, 44 Hz, 112° (VERIFIED, PMC5004601). | SICB abstract | search summary | SNIPPET |
| Bird cruise frequency | Pennycuick: f = ζ m^(3/8) g^(1/2) b^(−23/24) S^(−1/3) ρ^(−3/8) | A9 section 3 | A9 | VERIFIED in A9 (via E-Flap) |
| Hover frequency from blade elements | For a sinusoidal stroke of full amplitude Φ, mean φ̇² = (πfΦ)²/2, so L = ½ρ C̄L S r̂₂² R² (πfΦ)²/2 = m g and **f = (1/(πΦR)) √(4 m g / (ρ C̄L S r̂₂²))**, S = total wing area. Check on KUBeetle-S (0.155 N, S = 47.4 cm², R = 0.09 m, Φ = 3.316 rad, 18 Hz): C̄L = 0.98 with r̂₂² = 0.39, 1.28 with r̂₂² = 0.30. | this report | | INFERRED |
| Wing loading and disc loading in hover | Wing loading: KUBeetle-S 32.7–37.5 N/m², Nano Hummingbird ≈ 45.7 N/m², rhinoceros beetle 38.9 ± 3.7 N/m². Disc loading (weight over stroke-swept area ΦR² for two wings): KUBeetle-S 5.8 N/m²; DelFly Nimble 9.5 N/m² if its stroke is 44° per wing. Hummingbird disc loading: UNKNOWN. | section A | | VERIFIED / INFERRED as marked in A |
| Translational coefficients, Reynolds-dependent fit | C_L = (1.966 − 3.94 Re^−0.429) sin 2α; C_D = 0.031 + 10.48 Re^−0.764 + (1.873 − 3.14 Re^−0.369)(1 − cos 2α); Re = 2 c̄ Φ f R / ν. | Lee, Lua, Lim, Yeo 2016, as printed in Phan et al. 2020 | https://arxiv.org/pdf/2005.06715 | VERIFIED (as cited) |
| Translational coefficients, revolving wing (robotic fly) | C_L = [(k₀₂ + 2k₀₁µ + k₀₀µ²)/(r̂₂² + 2r̂₁µ + µ²)] sin α cos α; C_D = same form with k₁ on sin²α plus the k₂ offset. Fitted: k₀₂ = 1.38, k₀₁ = 1.65, k₀₀ = 2.01; k₁₂ = 1.38, k₁₁ = 1.44, k₁₀ = 1.38; k₂₂ = 0.15, k₂₁ = 0.24, k₂₀ = 0.32; r̂₁ = 0.59, r̂₂² = 0.39; µ = tip velocity ratio (forward speed over tip speed). | Dickson & Dickinson, JEB 207:4269, 2004 | https://homes.cs.washington.edu/~diorio/MURI2003/Publications/2003-2004pubs/DicksonDickinson2004.pdf | VERIFIED |
| The same, hover limit µ = 0 | C_L = (1.38/0.39) sin α cos α = 1.77 sin 2α; C_D = 3.54 sin²α + 0.385. So C_L,max = 1.77 at 45°, C_D(45°) = 2.15, C_D(90°) = 3.92, L/D at 45° = 0.82. Non-revolving limit µ → ∞: C_L = 1.005 sin 2α, C_D = 1.38 sin²α + 0.32. | | | INFERRED |
| Rotational and added-mass terms | dF_rot = ρ V_T c_rot α̇ c² dr with c_rot = π(0.75 − x̂₀), x̂₀ = pitch-axis position from the leading edge as a fraction of chord (Sane & Dickinson 2002); added mass dF_A = (π/4) ρ c² a_w dr. Zero rotational force for an axis at 75% chord; 2.36 for an axis on the leading edge (π × 0.75, INFERRED). | as printed in Phan et al. 2020 | https://arxiv.org/pdf/2005.06715 | VERIFIED (as cited) |
| Passive pitch at mid-stroke | Hinge angle ψ, angle of attack α = π/2 − ψ. Torque balance about the hinge: Kψ − (F_L cos α + F_D sin α) R_cop = I_xx α̈ + (I_yy − I_zz) φ̇² cos ψ sin ψ. Because C_L ≈ sin 2α, lift peaks when mid-stroke α ≈ 45°. Hinge stiffness of a flexure K = E t³ w / (12 l). Built hinges (Kapton, E = 2.5 GPa, w = 1.25 mm): 0.73, 1.4, 2.4, 3.4 µN·m/rad for a 54 mm² wing of 4 mm mean chord at 85–145 Hz. With K = 1.4: hinge angle is one curve against wingtip speed, ≈ 45° at ≈ 4–4.5 m/s, 70° at ≈ 9 m/s (Fig. 5b). Centre of pressure R_cop/c̄ rises from ≈ 0.15 at α = 25° to ≈ 0.6 at 75° (Fig. 6a), not the quarter-chord. The model fails when the pitch phase lag δ is large (soft hinge, high frequency). Kapton loss coefficient ≈ 0.02. | Chen, Gravish, Desbiens, Malka, Wood, JFM 791, 2016, which extends Whitney & Wood 2010 | https://smrl.mit.edu/wp-content/uploads/2019/10/jfm2016_experimentalandcomputational.pdf | VERIFIED (curves: fig.) |
| Passive pitch, static estimate | Dropping the inertial terms: K ≈ F_N R_cop / ψ. Check on the row above at 45° and 4.25 m/s tip speed with C_N = √(1.77² + 2.15²) = 2.78, r̂₂² = 0.30, R_cop = 0.35 c̄: F_N = ½ × 1.2 × 2.78 × 54e-6 × 0.30 × 4.25² = 4.9e-4 N; K = 4.9e-4 × 0.0014 / 0.785 = 0.87 µN·m/rad against 1.4 built. The static estimate is 0.6 of the built value. | this report | | INFERRED |
| Strouhal number | Peak efficiency of a heaving and pitching foil at 0.2 < St < 0.4 (Triantafyllou 1991, 1993; Anderson 1998); cruising birds 0.12 < St < 0.47 (Taylor 2003); a shift from 0.25 to 0.35 costs cetaceans 5–10% efficiency (Rohr & Fish 2004). Alcids in water 0.13–0.18 horizontal, 0.21–0.29 descending. Izraelevitz used St = 0.4. | eLife 55774; arXiv 1412.3843 | section B | VERIFIED (as cited) |
| Air/water frequency ratio from density alone | From F ∝ ρ A² f² with the same wing, amplitude and force: f_air / f_water = √(ρ_water/ρ_air) = √(1.0025/0.0012) = 28.9. RoboBee's first design: 140 Hz → 5 Hz predicted (ratio 28), unstable there. | Jensen 2024; RoboBee | sections B, D | INFERRED (RoboBee numbers VERIFIED) |
| Ratios in use | RoboBee 17–33 (220–300 Hz / 9–13 Hz), same amplitude, robot denser than water. Alcids 2.6–4.1, with 82–97° of stroke in water against 63–81° in air and the wing flexed. FAAV ≈ 1 (5 Hz both), amplitude cut by wing bending. Brünnich's guillemot 7.6–9.3 Hz against 2.4–3.2 Hz (A9). | sections A9, B | | VERIFIED per row; ratios INFERRED |

---

## What a designer can take from this (inference of this report, not a finding of any source)

1. **Drive.** One geared rotary motor per independently driven wing or wing pair: gear ratios 14.75:1 (3 g), 16:1,
   25.4:1, 28:1 (16 g); weight support at 40 Hz (3.4 g, bench), 18–25 Hz (16–37 g), 17 Hz (29 g), 12 Hz (102 g). Rests on:
   KU Leuven, NUS, T3, KUBeetle-S, DelFly Nimble, Flapper Nimble+. Hover input power where it can be computed is
   ≈ 210–220 W/kg (KUBeetle-S; Nimble+ upper bound).
2. **First-cut wing and frequency.** Wing loading 33–46 N/m², mean C̄L 1.0–1.3, and the blade-element identity in D.
   Worked point: 100 g, two wings of R = 0.20 m and aspect ratio 3.2 (S = 0.025 m², 39 N/m²), Φ = 120°, C̄L = 1.2,
   r̂₂² = 0.30 → f = (1/(π × 2.094 × 0.2)) √(3.924 / 0.0108) = 14.5 Hz; with Φ = 180°, 9.7 Hz. The Nimble+ hovers
   102 g at 12 Hz on four wings of 49 cm span. Rests on: KUBeetle-S, Nano Hummingbird wing loading, Nimble+.
3. **Thrust margin.** Free-flying hover machines carry lift/weight 1.3–1.45 (Nimble > 1.3, KUBeetle-S 1.34, T3 1.45).
   A design below 1.3 has no flying precedent in this table.
4. **Stroke amplitude decides the layout.** Two wings need 120–190° of stroke (T3 130°, NUS 128°, Leuven 175°,
   KUBeetle-S 190°), which takes a linkage plus string or cam amplification; raising the stroke from 120° to 190°
   bought 29% lift per watt. Four wings in clap-and-fling pairs fly on a crank alone with about 44° per wing
   (SNIPPET) at the cost of a second mechanism. Rests on: KUBeetle-S, T3, NUS, Leuven, DelFly.
5. **Wing build.** 5–15 µm polyester membrane, 0.3 mm carbon veins, 0.8–1.0 mm carbon leading edge, membrane on a
   free sleeve along the leading edge and the root spar; resulting geometric angle of attack ≈ 60–90° inboard and
   ≈ 20° at the tip; the inner 25% of span can be left open (−2% lift, +4.8% lift per watt). Heavier skins that
   have flapped: 40 g/m² PE-laminated non-woven (NUS), 0.5 mm mylar over ribs with the interior flooded
   (Izraelevitz, in water). Rests on: KUBeetle-S, T3, NUS, Izraelevitz.
6. **Pitch by compliance, and what water does to it.** Size the root hinge or membrane slack for mid-stroke
   α ≈ 45° in air with K ≈ F_N R_cop / ψ (the static estimate reads 0.6 of a built value) or K = E t³ w / 12 l for a
   flexure. Hinge deflection follows dynamic pressure; equal dynamic pressure in water needs 1/28.9 of the air
   stroke speed, and alcids run at about half, not 1/29, so a hinge sized for air will be driven far past 45° in
   water unless frequency, amplitude or exposed span is cut. FAAV's flexible wing and the auklet's half-folded wing
   are the two observed ways of shedding that load. Rests on: Chen 2016, eLife 55774, FAAV press, Kikuchi (SNIPPET).
7. **Tailless attitude control, four working recipes.** (a) left/right frequency for roll, dihedral servo for
   pitch, root-twist servo for yaw (Nimble: 2 motors + 2 servos); (b) stroke-plane tilt for pitch and roll, opposite
   root-spar offsets for yaw (KUBeetle-S: 1 motor + three 0.6 g servos); (c) one motor per wing with amplitude
   difference, stroke bias and split cycle (Purdue: ≈ ±3, ±0.5–0.8, ±0.6 N·mm at 12 g); (d) three or four
   independent wing modules and no servos (T3, Quad-thopter). Recipes (a), (c), (d) leave left and right wings
   independently driven, which is also what steering by rowing or crutching needs.
8. **Water stroke.** Diving birds of 0.45–1 kg: 2.2–2.8 Hz, 82–97° at the wrist, St 0.13–0.18 level and 0.21–0.29
   descending, stroke plane ≈ 13° further forward than in air, wing flexed to about half the tip excursion
   (SNIPPET). A < 300 g robot kept 5 Hz in both media and reached ≈ 1 m/s. A lower bound exists: RoboBee was
   unstable at the density-scaled frequency and needed about twice that, because slow strokes rock the body.
   A fore-aft sweep added to the stroke moves the same wing from thrust (Cx 0.24) to weight support (Cy 1.1).
   Rests on: eLife 55774, FAAV press, RoboBee, Izraelevitz.
9. **Water exit.** The one flapping exit on record starts ≈ 0.5 m down, holds the body at 70° with the tail, and
   depends on wing size (of 60, 80, 100 cm only 80 cm worked) and on tail-to-body distance. Rests on: FAAV press
   and abstract; A9 for the non-flapping exits.
10. **Land.** Three mechanisms have numbers. (i) Folded wing as a forelimb: vampire bat bounds at 0.28–1.14 m/s
    at 23 g with forelimb duty factor 0.62 and a support height of ≈ 2.7 cm, and jumps to 2.4 m/s from the
    forelimbs. (ii) Flipper crutch: hatchling turtle, duty factor 0.66, stride 0.58 body lengths, body lifted
    2.2 mm, planted flipper does not slip, passive wrist; a tail is needed on 20° sand. (iii) No wing contact:
    flapping below take-off frequency shuffles the body on passive legs at 5.4 cm/s (T3, half hover throttle) or
    0.5–2.7 cm/s (RoboFly, 60–80 Hz against 140 Hz lift-off). Robot wing-as-leg speed on record is 6 cm/s (DALER).
    (i) and (ii) need the wing to sweep fore-aft while loaded, the same extra degree of freedom as item 8;
    a pure up-down crank stroke gives only (iii). Rests on: Riskin 2006, Mazouchova 2010, techxplore 2016, T3,
    RoboFly, DALER press.

---

## Not found

- **FAAV paper and supplement** (wing area and stiffness, amplitude in each medium, tail geometry, motor, gear
  ratio, thrust, strokes and time to exit): `dspace.mit.edu/bitstreams/f687c5d6-…/download` returned HTTP 429 with
  Retry-After 900 on three attempts today (and three in A9). Only press pages and the abstract were read.
- **DelFly Nimble** stroke amplitude, wing area, membrane, gear ratio: `repository.tudelft.nl` refused the
  connection on three attempts; the Science paper was not reachable; `mavlab.tudelft.nl` carries no numbers.
  **DelFly II** beyond 16 g and 28 cm (Wikipedia): not found.
- **Nano Hummingbird** flapping frequency, stroke angle, power: the AIAA 2012 paper is paywalled, the AeroVironment
  PDF is 404, and no search result gave them.
- **COLIBRI** wing, mechanism, control moments: SAGE 403, the SciSpace PDF came back empty, TU Delft refused.
- **Bat Bot B2** paper (Georgia Tech copy: certificate rejected; CORE: 403); **Festo BionicFlyingFox** data sheet
  (403 and 404; mass not confirmed anywhere); **Aerobat** mass and flapping frequency (not on the arXiv page).
- **Lock, Vaidyanathan, Burgess**: the optimal span retraction and the efficiency figures (Bristol PDF 403,
  CORE 403).
- **Johansson & Aldrin 2002** kinematic numbers; **Kikuchi 2015** primary text; penguin and sea-turtle stroke
  amplitude and angle of attack; any number for dippers or shearwaters under water.
- **Riskin & Hermanson 2005** and **Schutt et al. 1997** primary texts; the vampire bat's body length, hence its
  speed in body lengths; the two secondary statements about the jump (6.51 body weights; 2.4 m/s in < 30 ms) do
  not agree.
- **Habib 2008** contains no launch time, speed or kinematics (stated in its abstract).
- **DALER** mass and wingspan (EPFL Infoscience 429, NCCR PDF 403); **MuddyBot** mass and displacement per cycle
  (crablab PDF: certificate rejected); **FlipperBot** speed and step length (press page only; PubMed behind a
  cookie wall).
- **Share of body weight carried by a wing or flipper on land**: reported by no source opened.
- **Dickinson, Lehmann & Sane 1999** sinusoidal fits for C_L(α), C_D(α) and **Sane & Dickinson 2002** primary
  text (Silverchair 403; the JEB landing URL tried was a different article): the coefficient forms above come from
  Dickson & Dickinson 2004 and from Lee et al. 2016 as reprinted, not from the 1999 paper.
- **Whitney & Wood 2010** primary text (Harvard URL 404, Cambridge paywall): the model is given as restated by
  Chen et al. 2016.
- **Greenewalt** primary data; hummingbird disc loading; the constant in the f ∝ √m / A relation.
