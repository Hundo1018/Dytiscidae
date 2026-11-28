# A16 — Morphing-wing survey: other fold families, inflation, and shape change within the beat

Survey date 2026-11-15. Web search and page fetches only; no project code, no simulation. Written for the last
section of `experiments/flapper/L0/MECHANISM.md` ("Three more proposals from the user (2026-11-15) and two
checks"), items 1-3. It follows `A15_folding_wing.md` and does not repeat it: Miura kinematics and stiffness,
the SNU ladybird glider, the earwig origami wing (Ishiguro 2023), the Konkuk beetle robots, Bat Bot B2,
RoboFalcon, Flapperoo, AquaMAV, the alcid frequencies and the murre flexed-wing test, line and winch sizing, and
the polypropylene living-hinge figure (about 1e6 flexes) are all in A15 and are cited as "A15 §x".

Machine in view: about 0.6 kg, one wing pair, each wing 0.66 m × 0.26 m (0.17 m²), 3-4.5 Hz in air, 0.5-2.5 Hz in
fresh water, root bending moment about 5.5 N·m a wing at take-off. Hobby builder with a 3D printer.

Labels (A14/A15's): **VERIFIED** — read today in the fetched page or the PDF's own text (PDFs the fetch tool
could not read were read with `pdftotext`); **SNIPPET** — only in a search-result summary or a secondary page;
**INFERRED** — this report's arithmetic, inputs shown; **UNSOURCED** — standard physics or general knowledge no
page fetched today confirms; **UNKNOWN** — not found. Source class after the label: *paper*, *preprint*,
*maker*, *shop*, *press*. "Demonstrated" means a machine did it and its mass is given; otherwise "possible".

One search result (a Wikipedia mirror on Yoshimura buckling) carried an embedded instruction unrelated to the
task; it was ignored and nothing from that mirror is used.

---

## 1. Fold patterns other than Miura, judged by the properties a wing fold needs

The five properties (MECHANISM.md item 1): (P1) rigid-foldable with thick panels; (P2) one or few DOF;
(P3) contracts in span **and** chord; (P4) deployed state locked and stiff in bending and twist at about
5.5 N·m; (P5) folded stack dimensions.

### 1.1 Tessellations and tubes

| pattern | P1 rigid / thick | P2 DOF | P3 span and chord | P4 deployed lock | P5 stack | source | label |
|---|---|---|---|---|---|---|---|
| **Waterbomb tube** (Feng, Ma, Chen, You, Sci. Rep. 2018) | rigid-foldable; facets rigid, creases as revolute joints; thickness not treated. A 3D-printed version had 1 mm walls, stiff Verowhite facets and soft Tangoblack creases | **1** ("rigidly foldable with one degree of freedom") | contracts **radially and axially at once**; radius of each vertex ring r = a·sin(φ/2)/sin(π/n); axial strain ε = (L0 − Lt)/L0. After full radial squeeze the tube **twists**, up to 72° (n = 5, m = 3) | no lock described; compressive force "approximately doubled" in the twist stage; no stiffness in N/m | not given | https://pmc.ncbi.nlm.nih.gov/articles/PMC6015070 | VERIFIED paper |
| Waterbomb tessellation, general | generally several DOF; reduces to 1 when folded symmetrically; negative Poisson ratio (shrinks in both directions); a thick-panel version is claimed single-DOF with negative Poisson ratio "at any spatial size" | 1 if symmetric | yes, both directions (no ratio formula found) | — | — | search summary of https://ori.ox.ac.uk/media/11004/2020_open-access_cl-pm-yy-zy_idetc.pdf (page then 404) and https://core.ac.uk/download/327951942.pdf | SNIPPET paper |
| **Kresling tube** (truss model, arXiv 2003.10411) | **not rigid-foldable**: the triangles "must bend or stretch" between the two stable states; that is what makes it bistable | 1 coupled (axial + twist) | **axial only**: the end polygons keep their shape (INFERRED from the n-gon geometry), so a Kresling spar shortens the span and rotates the tip; chord unchanged | **bistable**: monostable for δ0 < 24°, asymmetric bistable 24-49.5° (folded state stable but strained), symmetric bistable 49-67.5° (both states stress-free); deployed state "significantly stiffer and more robust to off-axis perturbation"; zero axial stiffness near δ0 = 67° | compaction η = h_c/h_d; "a range for which the theoretical compaction ratio is 0", limited in practice by material thickness | https://arxiv.org/pdf/2003.10411 | VERIFIED preprint |
| **Yoshimura** (diamond) | closed cylinder "impossible to fold as rigid origami" (forum); bellows theorem rules out rigid folding of a cylinder whose ends keep their shape | — | axial (as a tube) | high axial stiffness when folded | flat-foldable only in the traditional proportions | https://discourse.mcneel.com/t/cylindrical-origami-folding/153099 ; https://www.arxiv-vanity.com/papers/1507.08472/ | SNIPPET forum / preprint |
| **Flasher** (Zirbel, Lang, Thomson et al., J. Mech. Des. 2013) | thick panels by an elastic membrane on one side with gaps at valley folds, or by letting panels fold on their diagonals: **the panels are not rigid** in the second method | not found | **yes, radially in both directions**: 1/20-scale prototype 1.25 m deployed, 0.136 m stowed diameter, **ratio 9.2** | not found | wraps into a cylinder; height not found | https://langorigami.com/publication/accommodating-thickness-in-origami-based-deployable-arrays/ (search summary); NASA follow-up https://ntrs.nasa.gov/api/citations/20230004438/downloads/IDETC_2023_Flasher_Thickness_Accommodation__Non_Copyright_Version.pdf | SNIPPET paper |
| **Curved-crease** (waterbomb base with curved creases; Liu, Tachi, Paulino multistable shells) | panels must bend by construction (curvature changes) | 1 per crease, coupled | not a planform-contraction pattern in any source read | **bistable**, energy barrier tunable; no wing built | — | https://collaborate.princeton.edu/en/publications/effect-of-crease-curvature-on-the-bistability-of-the-origami-wate/ ; https://paulino.scholar.princeton.edu/document/4221 | SNIPPET paper |
| **Fan / pleat** (accordion) | rigid-foldable with thick panels trivially (parallel or radial hinges); used in the earwig fan stage and in Ishiguro's wing (A15 §A.1) | 1 | **one direction only** (INFERRED: pleats parallel to the chord shorten the span; a radial fan about a pivot closes the sector angle, area ratio ≈ closed angle / open angle) | none of its own; needs a separate stop or spar | pleat count × panel thickness (INFERRED) | geometry | INFERRED |

Read (INFERRED): of the tessellations, only the **waterbomb** family contracts in both directions with one DOF
and rigid facets; nobody has published a contraction formula for it in the pages reached, a deployed lock, or a
wing. **Kresling** gives the lock (bistability) but only along its axis, with an unwanted twist coupled to the
span change, and it is not rigid-foldable. The **flasher** has the best area ratio found (diameter 9.2, area
about 1/85 by (1/9.2)², INFERRED, planform only) but its thick-panel version bends or hinges its panels on
diagonals and it is a deployable disc, not a lifting surface.

### 1.2 Insect hindwings beyond A15

| item | numbers | URL | label |
|---|---|---|---|
| Earwig wing | opens to **10 times** its folded area; a version obeying classical rigid origami (straight creases, 360° at each vertex) folds only to **one third**; the mid-wing joint's crease angles are "incompatible with rigid origami"; resilin joints act as springs and lock the wing open and closed (bistable) | https://www.sciencedaily.com/releases/2018/03/180322140854.htm (Faber, Arrieta, Studart, Science 2018) | SNIPPET press |
| Ladybird hindwing (Saito et al., PNAS 2017) | a tape-spring-like vein frame does most of the transformation; folding is done by the elytron and abdomen, not the wing's own muscle; the wing combines "thick veins for rigidity" with complex creases. Area ratio not reached | https://www.iis.u-tokyo.ac.jp/en/news/2697 | SNIPPET press |

Read: both insects buy their fold ratio with **non-rigid** behaviour (stretching joints, snapping veins), and both
fold with an external tool (the elytron), once per flight, not with a wing muscle.

### 1.3 Bird and bat skeletons as linkages

| item | numbers | URL | label |
|---|---|---|---|
| "Drawing parallels" (Garrod 1875; Vazquez 1994): radius and ulna as two parallel bars coupling elbow and wrist, so one input flexes both | the 2017 pigeon study found every optimal **planar four-bar** either locks, cannot follow the 3D bone paths, or needs radius and ulna to cross; a **six-bar** with a sliding ulnare fits | https://pubmed.ncbi.nlm.nih.gov/28794161/ ; https://sicb.org/?p=12414 | SNIPPET paper |
| Bennett-linkage wing (arXiv 2604.07677, 2026) | two coupled spatial four-bars; one driven (stroke), one **passive** (fold, moved by the stroke and the air); wing **87 g**; stops limit the fold joint to ±75°; **no lock**, no aerodynamic test, no span ratio | https://arxiv.org/html/2604.07677 | VERIFIED preprint |
| **USGull** (Kim et al., Chungnam, IMAV 2018) — the machine nearest this project found | **460 g, span 1.8 m, mean chord 0.25 m, AR 6.8, 2-3 Hz**, inner wing 30 cm, outer 60 cm (1:2), elbow fold driven from upper/lower spars (SmartBird scheme), EPP skin, carbon plate. Load cell: swept wing about 20% more thrust than rectangular at equal area 0.1352 m². **"Due to the structural problems on the elbow joint, the continuous flapping movement was not performed"**; larger wings loaded the elbow more and the authors propose a double elbow joint | https://www.imavs.org/papers/2018/14.pdf | VERIFIED paper |
| Bat-like chained four-bars (ICAS 2020) | inner, middle, outer sections; folded on the upstroke "to reduce drag"; no span number in the excerpt | https://www.icas.org/icas_archive/ICAS2020/data/papers/ICAS2020_0333_paper.pdf | SNIPPET paper |

Read (INFERRED): linkage folds are rigid-panel and one-DOF by construction (P1, P2) and are the only family flown
at this size (A15 §A.2, plus USGull). They change **span**, and chord only through sweep. Their weak point in the
one case at this mass and span is the **elbow under cyclic load**, which is the P4 question for every hinge in
this project.

### 1.4 Rigid-foldable patterns on real deployable structures, 0.1-2 m

| structure | numbers | URL | label |
|---|---|---|---|
| Flasher solar array, 1/20 scale | 1.25 m deployed, 0.136 m stowed; full-scale design 25 m / 2.7 m (press) | above; https://www.thekurzweillibrary.com/?p=214658 | SNIPPET |
| Space Flyer Unit 2D array (1995) | Miura-folded array deployed in orbit by one input; dimensions not found | https://en.wikipedia.org/wiki/Miura_fold | SNIPPET |
| **Oru folding kayak** (marine, 0.1-2 m panels) | 5 mm double-layer extruded polypropylene; **"rated for 20,000 fold cycles"** (maker's claim, not independently tested); Lake model 7.7 kg; one buyer reports pinholes at fold corners after "a handful" of uses | https://paddlingmag.com/boats/boat-review-oru-kayak-the-coast/ ; https://www.publiclands.com/p/oru-lake-9-inflatable-kayak-23anzulk98kykxxxxpsk/23anzulk98kykxxxxpsk | SNIPPET shop/press |
| Origami fin actuator (Vu et al. 2024) | polypropylene film, laser cut, in a **135 mm** cuttlefish robot | https://arxiv.org/html/2407.16821v1 | SNIPPET preprint |

Not found: any **rigid thick-panel** origami pattern operated underwater at 0.1-2 m, and any origami lifting
surface above the 660 mm SNU glider (A15 §A.1).

---

## 2. Inflation and pneumatics

### 2.1 Inflatable wings that flew

| aircraft | numbers | URL | label |
|---|---|---|---|
| **I-2000** (NASA Dryden, wings from Vertigo/ILC Dover) | **6.8 kg (15 lb)**; span **1.63 m (64 in)**, chord **0.184 m (7.25 in)**, NACA 0021 (so about 39 mm thick, INFERRED 0.21 × 184 mm); **five spanwise cylindrical spars of spirally braided Vectran over a urethane bladder**, fabric spar caps, ripstop skin. Ground tests at 150, 225, 300 psig; **180 psig (12.4 bar) chosen for a 3.5 g envelope** (wrinkling onset). Gas: **35 in³ (0.57 L) tank at about 1,800 psig (124 bar)**, blowdown through a regulator used as an orifice. In flight: inflation commanded at 27.9 s, final shape at about 28.6 s, so **about 0.7 s** (INFERRED from the stated times); "inflation was rapid… dynamics benign" | https://ntrs.nasa.gov/api/citations/20020039162/downloads/20020039162.pdf | VERIFIED paper |
| Apteron (ILC Dover) | 3.18 kg, span 1.55 m, 373 W | https://www.sjsu.edu/people/nikos.mourtos/docs/micro%20UAV.pdf (search summary) | SNIPPET |
| Goodyear Inflatoplane GA-468 | span 6.7 m, 110 ft²; empty 102 kg, gross 249 kg; **7-9 psi (0.48-0.62 bar)**; inflation about 5 min | https://en.wikipedia.org/wiki/Goodyear_Inflatoplane | SNIPPET |
| NASA Prandtl-M | **not inflatable**: a folding composite flying wing; negative finding for the brief's "Prandtl-style inflatable" | https://www.compositestoday.com/2015/07/nasas-new-composite-aircraft-could-be-the-first-to-fly-on-mars/ | SNIPPET press |

### 2.2 Spar stiffness against pressure

- Comer and Levy: **collapse moment M_c = π p D³/8 = π p r³**, independent of the fabric. Measured collapse
  moments differ from it by up to 42-45%. (https://www.yorku.ca/gzhu/Paper_copy/Marked%20Proofs.pdf, search
  summary; SNIPPET paper.) Wrinkling onset is taken as **half** the collapse moment, **M_w = π p r³/2**
  (https://www.fzt.haw-hamburg.de/pers/Scholz/ewade/2007/CEAS2007/papers2007/ceas-2007-160.pdf, SNIPPET paper).
- So MECHANISM.md's M = π p r³/2 is the wrinkling moment, and its radii stand (INFERRED):
  r = (2M/(π p))^(1/3); at 2 bar gauge, (11/(π·2e5))^(1/3) = (1.75e-5)^(1/3) = **26 mm**; at 1 bar, **33 mm**.
  At the collapse moment instead, 2 bar gives (5.5/(π·2e5))^(1/3) = **20.6 mm**.
- Turned round (INFERRED): a spar the size of a thin wing, r = 13 mm (26 mm thick, 10% of the 0.26 m chord),
  needs p = 2M/(π r³) = 11/(π·2.20e-6) = **1.6 MPa, 16 bar** to reach wrinkling at 5.5 N·m. The I-2000 ran at
  12.4 bar with five spars and a 21% thick section: **this is the pressure regime of a working inflatable wing,
  not of a hobby pump** (2.3 below).
- The Kentucky inflatable-wing work: stiffness and twist "rely exclusively on inflation pressure"; wings show
  **wash-in** (angle of attack rises as they bend). https://uknowledge.uky.edu/gradschool_diss/589 — SNIPPET.
- Inflated stiffening of a swimming foil (Micklem, Weymouth, Thornton, IROS 2022): rubber tubes in a silicone
  foil; relative stiffness ≈ 2p̂ + p̂²; **stiffness doubled from 0 to 0.8 bar for 2 J**; resonant gain 5-7×.
  https://arxiv.org/pdf/2211.13627 — SNIPPET preprint (abstract-level).

### 2.3 Pneumatic actuators for folds

| actuator | numbers | URL | label |
|---|---|---|---|
| Pouch motor (Niiyama et al.) | linear: about **28% stroke, up to 100 N**; angular: up to 80° | https://core.ac.uk/download/83232394.pdf (search summary) | SNIPPET paper |
| Fluid-driven origami muscle (Li, Vogt, Rus, Wood, PNAS 2017) | **> 90% contraction, about 600 kPa stress, > 2 kW/kg**, runs on vacuum, in air or water | https://dspace.mit.edu/handle/1721.1/117016 | SNIPPET paper |
| Earwig wing pouch (A15 §A.1) | opens at 0.2 MPa; closes by thread | A15 | VERIFIED (A15) |
| Printed TPU bellows (fatigue) | **6,410 cycles at 3 bar** mean to failure, against 3,439 at 1 bar for silicone | https://flore.unifi.it/bitstream/2158/1379015/1/Characterization%20of%20fatigue%20behavior%20of%203D%20printed%20pneumatic%20fluidic.pdf | SNIPPET paper |

### 2.4 Hobby pneumatics: pumps, valves, CO2, reservoir, depth

| part | numbers | URL | label |
|---|---|---|---|
| TOPSFLO TM30A-B 12 V | 4.5 L/min free flow; **95 kPa max** | https://evelta.com/tm30a-b-12v-micro-diaphragm-air-vacuum-pump-60kpa-max-flow-4-5l-min/ | SNIPPET shop |
| TOPSFLO TM30A-A 12 V | 6 L/min; 120 kPa max | https://evelta.com/tm30a-a-12v-micro-diaphragm-air-vacuum-pump-70kpa-max-flow-6l-min/ | SNIPPET shop |
| 559-BD-04 (4 series) | 11 L/min at 0 kPa; **2.5 bar max**; 12 V, ≤ 0.6 A no load | https://evelta.com/content/datasheets/559-BD-04A11L.pdf (image PDF; numbers from the search summary) | SNIPPET shop |
| 559-BD-02 (2 series) | **71 g**, 53 × 40 × 24.3 mm | https://evelta.com/content/datasheets/559-BD-02A1L.pdf | SNIPPET shop |
| KNF NMP830 | 2.5 L/min at atmospheric; 1.4 bar max | https://www.im-tek.com/docs/knf-nmp830kndc-b-customized-micro-diaphragm-gas-pump-86227.pdf | SNIPPET maker |
| Cheap FA0520 mini valves (6/12 V) | **0-350 mmHg** (= 46.7 kPa, INFERRED ×0.1333), < 2.5 W; 3-way version 20 g | https://electropeak.com/fa0520b-micro-gas-air-electric-solenoid-valve ; https://zbotic.in/product/fa0520f-12v-dc-2-position-3-way-small-mini-electric-solenoid-valve-for-air-gas/ | SNIPPET shop |
| Parker X-Valve (5 V) | **4.5 g** | https://ph.parker.com/us/en/912-000004-006 | SNIPPET maker |
| Festo 197034 (10 mm, 3/2) | 10 g; 10 l/min nominal | https://www.sp-spareparts.de/en/p/197034-festo | SNIPPET shop |
| Bürkert 6606 / 6011 | 6606: 65 l/min, about 63 g, to 28 psi; 6011: 1.2 mm orifice, 48 l/min, 0-21 bar | https://www.smcpneumatics.com/otherPDFs/burkert/Miniature_Valves.pdf ; https://static.chipdip.ru/lib/993/DOC028993798.pdf | SNIPPET maker |
| IMI MICROSOL (15 mm) | 6-120 l/min at 2 bar, 10-15 ms response | search summary (PDF then 404) | SNIPPET maker |
| 16 g CO2 cartridge | about **7.5 L at room temperature**; 56-58 g full | https://knowledge.restube.com/how-much-content-does-a-cartridge-have ; shop listings | SNIPPET maker/shop |
| CO2 inflator heads with flow knob | 12 g (Prestaflator Micro), 22.5 g (Topeak), 35 g (PRO regulator) | https://pro-bikegear.com/global/accessories/pumps/team-co2-regulator ; https://www.bikebug.com/topeak-airbooster-16g-inflator-p-105202.html | SNIPPET shop |

Arithmetic (INFERRED; isothermal ideal gas, UNSOURCED standard physics; a deflated bag holds about no gas):

- Free air to fill V to gauge pressure p_g: V·(1 + p_g/1 bar). 0.5 L at 1 bar: 1 L; 1 L at 1 bar: 2 L;
  3 L at 2 bar: 9 L.
- Fill time, **lower bound** (free-flow rating; flow falls toward zero at the pump's maximum, curves not read):
  0.5 L at 1 bar on the 4.5 L/min pump: 1/4.5 min = 13 s, and that pump stops at 0.95 bar, so not at all at 1 bar
  in practice. 1 L at 1 bar on the 11 L/min pump: 2/11 min ≥ 11 s. 3 L at 2 bar on the same pump: 9/11 min ≥ 49 s.
  Against MECHANISM.md's "a water-to-air crossing gives seconds", **no hobby pump found inflates during the
  crossing**; it can pre-inflate at the surface.
- Check on the CO2 figure: 16 g / 44.0 g/mol = 0.364 mol × 24.5 L/mol (25 °C) = 8.9 L ideal, so 7.5 L is in
  range. One cartridge fills 7.5/2 = **3.7 L to 1 bar** or 7.5/3 = **2.5 L to 2 bar**, once.
- Stored-air reservoir (blowdown, final pressure p_f absolute, reservoir V_r, wing V_w):
  p_r = p_f·(V_r + V_w)/V_r. Wing 0.5 L to 1 bar gauge (2 bar abs) from 0.5 L: p_r = 2·1.0/0.5 = 4 bar abs.
  Wing 3 L to 2 bar gauge (3 bar abs) from 0.5 L: p_r = 3·3.5/0.5 = 21 bar abs. The I-2000 did it with
  0.57 L at 124 bar for a five-spar 1.63 m wing in 0.7 s.
- Deflation: venting 1 L from 1 bar gauge releases 1 L of free air; through a 48 l/min valve that is of order
  1-2 s (1/48 min = 1.25 s, but QNn is rated at 6 bar inlet and 1 bar drop, so slower at 1 bar). Order only.
- **Depth**, fresh water: Δp = ρgh = 1,000 × 9.81 × 2 = **19.6 kPa at 2 m** (9.8 kPa per metre). A gas volume
  shrinks by 101.3/(101.3 + 19.6) = **0.84** at 2 m. MECHANISM.md's 0.68 L inflated wing would lose
  0.68 × 0.16 = 0.11 L, **1.1 N of buoyancy, 18% of the machine's 5.9 N weight**: a gas swim bladder is unstable
  in depth (sinks faster as it sinks) unless it is regulated.

### 2.5 Buoyancy control on swimming robots

| machine | method | numbers | URL | label |
|---|---|---|---|---|
| Biomimetic UUV (Malec 2021) | rigid cylinder, piston on a lead screw | vehicle **63.9 kg**; ballast **0.6 L** | https://reference-global.com/pdf/10.37705/TechTrans/e2021024 | VERIFIED paper |
| Hybrid UUV (same paper) | elastic **water** bladder in a steel frame, bidirectional gear pump | vehicle **18.58 kg**; bladder **0.8 L** max; pump **0.6 L/min** each way, so a full stroke takes about 80 s (INFERRED 0.8/0.6 min) | same | VERIFIED paper |
| Pearl arowana robot (Nguyen et al. 2011, cited) | piston ballast in the bow | **12 ml = 9.6% of the vehicle's volume** | same (citing) | VERIFIED paper (as cited) |
| Tuna robot (arXiv 2310.12715) | syringe, piston, DC gearmotor, PID depth hold | robot **2.305 kg**; syringe volume not found | https://arxiv.org/pdf/2310.12715 | VERIFIED preprint |
| SoFi (Katzschmann 2018, cited) | two piston cylinders fore and aft | volumes not read | Malec (citing) | VERIFIED paper (as cited) |

Read (INFERRED): every buoyancy device found on a swimming robot moves **water** with a piston or pump, not gas;
none uses inflation of a lifting surface, and the fastest found takes tens of seconds. A gas bladder sized for
this machine would have to be regulated against the 16% volume loss at 2 m.

---

## 3. Shape change within the beat: spread on the power stroke, fold on the recovery

### 3.1 Biology

| animal | numbers | URL | label |
|---|---|---|---|
| **Diving beetle** *Cybister bengalensis* (Dytiscidae), Sci. Rep. 2021 | body **36.3 mm**; forward cycle **272 ms (3.7 Hz)**; power stroke **47%** of the cycle (38% turning, 30% reversing); speed max **12.9 cm/s** at 60% of the power stroke, mean **8.74 cm/s**. Power stroke: legs extend, **swimming hairs spread** and rotate with the tarsus. Recovery: tarsus rotates back, hairs **fold up passively**, tarsal subsection angles collapse "under the resistance of water". In a turn the outside leg reverses the pattern (hairs folded on its forward swing) | https://www.nature.com/articles/s41598-021-96158-1.pdf | VERIFIED paper |
| Same paper, cited figures | "recovery drag is only **1/40** of the power-stroke thrust" — cited to Nachtigall 1981 (a review); "efficiency **84%**" — cited to Nachtigall 1961, whose title is on the **whirligig beetle *Gyrinus***, not a dytiscid. Neither original was read | same, refs 3 and 19 | VERIFIED (as cited); the 84% does not belong to Dytiscidae |
| Water boatman *Cenocorixa bifida* | setae splay on the power stroke, collapse on recovery; hydromechanical efficiency **0.52**, 0.42 with inertial and added-mass power, **about 0.40** over the whole cycle including recovery; Corixidae swim about 15 cm/s with a 40 cm/s stroke | https://scilit.com/publications/89a4139267541ca5cb9eeff66ef0fb7a ; https://cob.silverchair.com/jeb/article/217/15/2740/12229/The-hydrodynamics-of-swimming-at-intermediate | SNIPPET paper |
| Aquatic insects generally (APS DFD 2025) | beetles, boatmen and backswimmers also **feather** (rotate) the limb between strokes | https://archive.aps.org/dfd/2025/z09/4 | SNIPPET |
| Ducks | foot plantarflexed with digits spread on the power phase, dorsiflexed with **web collapsed** on recovery; power phase **66-70%** of the cycle; drag gives thrust for the first 80% of the power phase, lift at its end | https://pmc.ncbi.nlm.nih.gov/articles/PMC2935360 ; https://www.nature.com/articles/nature01695 | SNIPPET paper |
| Frogs (*Xenopus*) | thrust mostly from foot **rotation**; translation 1.1-21% of thrust impulse (69% in *Rana pipiens*); web shrinks with limb flexion | https://cob.silverchair.com/jeb/article-pdf/211/19/3181/1263167/3181.pdf ; https://sicb.org/?p=27951 | SNIPPET paper |
| Human front crawl | bent-elbow, low-hand recovery recommended to cut drag: **coaching pages only, no measurement found** | https://www.220triathlon.com/news/front-crawl-how-to-improve-your-arm-recovery-technique/ | SNIPPET (coaching) |
| **Birds, upstroke span ratio** (upstroke span / downstroke span) | house martin (Rosén et al. 2007): **0.39 at 4 m/s, 0.41 at 6, 0.33 at 8, 0.31 at 10**; "R < 1 in all previously studied species" | https://pmc.ncbi.nlm.nih.gov/articles/PMC2373391 | VERIFIED paper |
| | zebra finch **< 0.20**, hummingbird **> 0.95** at all speeds (Tobalske et al.) | https://sicb.org/?p=31522 (summary) | SNIPPET paper |
| | chukar and duck wings: upstroke posture gave **49%** of downstroke lift, L/D 2.5 | https://sicb.org/?p=25378 (summary) | SNIPPET paper |
| **Bats** (six pteropodids, 33.6 g to 1,078 g; Riskin et al. 2012) | span ratio **0.3-0.4** on average, down to **0.16**; folding makes total inertial power **about 65%** of an extended wing's (saving 29-41% by species); **folding and unfolding are about 44%** of the remaining inertial cost (37-59%; 58% on the upstroke) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3385481 | VERIFIED paper |
| Auks, puffins (A15 §E.1) | wrist flexed in water, no area fraction found; extended wings 1.5-6× the force of flexed | A15 | — |
| Gentoo penguin | a bent (real-shaped) wing: propulsive efficiency **1.8×** that of a flattened one, mainly on the upstroke; emperor penguins 0.92 Hz at most | https://www.titech.ac.jp/english/news/2023/065906.html ; https://openpolar.no/Record/fthighwire:oai:open-archive.highwire.org:jexbio:205%2F24%2F3769 | SNIPPET paper/press |

### 3.2 Engineering

| device | numbers | demonstrated on | URL | label |
|---|---|---|---|---|
| **Tendon-coupled duck foot** (arXiv 2409.14707) | 1.5 Hz; foot 75 mm, nylon cloth and Kevlar rope; net thrust **1.72 ± 0.06 N vs 0.86 ± 0.09 N**; propulsive efficiency **2.86% vs 1.45%** (no extensor tendon) **vs 0.84%** (passive paddle): 2.0× and 2.4×. Recovery drag not separated; extensor versions showed a larger negative peak in recovery | robot leg in a tank; mass not stated; 100 cycles averaged | https://arxiv.org/html/2409.14707v1 | VERIFIED preprint |
| **FLHex** flapped paddle (Burzynski et al. 2021) | overlapping louver flaps open one way, shut the other (bird-feather idea); swam 0.5-1.33 Hz; best **39 cm per cycle** at 160° amplitude, 0.5 Hz; 4.8 m in 10 s at 1.33 Hz (0.48 m/s, INFERRED). **No thrust figure against a conventional flipper** in the tables; robot not waterproof, swam on a float | RHex-type hexapod; mass not read | https://journals.pan.pl//Content/120816/PDF/Z_21_02278_Bpast.No.69(6)_OK.pdf | VERIFIED paper |
| Flapped paddle-fin (Simha et al., RA-L 2020) | soft cascaded flaps; "high degree of improvement in net thrust and propulsive efficiency"; numbers not reached | fin rig | https://ws.lib.ttu.ee/digibase/en/Publ/Item/091b2a78-311a-47a9-87ca-ac7f346de05f | SNIPPET paper |
| Stiffness-adjustable articulated paddle (Kwak 2023) | 8 joints, 20 passive flaps | paddle rig | https://scholarworks.unist.ac.kr/bitstream/201301/82316/2/Advanced%20Intelligent%20Systems%20_%202023%20_%20Kwak%20_%20Development%20of%20a%20Stiffness%e2%80%90Adjustable%20Articulated%20Paddl.pdf | SNIPPET paper |
| Pressurised beetle foot (cited in MDPI 2025) | projected area **+66%** under pressure, mean maximum thrust **+34.5%** | robot foot | https://www.mdpi.com/2313-7673/10/3/182 (403; search summary) | SNIPPET paper |
| **Spring-loaded area reduction on the upstroke, ornithopter** (Texas A&M Morpheus lab) | best at about 7 Hz; average **static lift "approximately 400%"** of the rigid wing, but the morphing wings **"produced far less thrust"** | ornithopters, mass not stated | https://morpheuslab.engr.tamu.edu/passive-wing-morphing | VERIFIED (lab page) |
| One-way valve / louvre flapping wing | CFD thesis (valve count, inlets, opening angles) and patents; **no measured gain found** | none | https://cendekia.unisza.edu.my/neuaxis-e/Record/um-12114/Details ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5899408 | SNIPPET |
| Per-beat fold, flapping (A15 §A.2) | Flapperoo cycle lift near or below zero; RoboFalcon 2.0 flies at 800 g; USGull elbow failed (1.3) | — | A15; IMAV 2018 | — |
| Feather-tucking drone, per flight not per beat (Di Luca et al. 2017) | area **−41%** gives lift **−32%**, drag **−40%**, top speed 6.3 → 7.6 m/s, turn radius 3.9 → 6.6 m | propeller drone, mass not found | https://spectrum.ieee.org/artificial-feathers-let-drones-morph-their-wings-like-birds | SNIPPET press |

### 3.3 What a rigid symmetric stroke gives up (INFERRED, quasi-steady drag rowing, UNSOURCED physics)

- Net thrust of a drag-based rower ∝ C_D·(A_p·u_p² − A_r·u_r²). With the same area and speed both ways the net is
  zero; this is the FLHex paper's stated starting point. Halving the area on recovery at equal speed leaves half
  the power-stroke impulse; the beetle's cited 1/40 leaves 39/40.
- This is the drag-rowing case. The project's water stroke is a lift-based flapping foil; for that case the only
  measured analogues are the bird and bat numbers (span ratio 0.16-0.41) and the TAMU ornithopter (more lift,
  less thrust). **No measured thrust or efficiency gain of an upstroke area cut on a lift-based flapping wing in
  water was found.**

### 3.4 Fatigue of parts that move every beat

Cycle counts (INFERRED): air 4 Hz × 15 min = 3,600 (A15 §0); water 2 Hz × 10 min = 1,200.

| part | measured life | URL | label |
|---|---|---|---|
| Moulded PP living hinge | about 1e6 (A15 §D) → 280 flights of air beats | A15 | VERIFIED (A15) |
| Printed TPU living hinge | **> 2 million folding cycles**; other printed polymers failed under 1,000 | https://link.springer.com/article/10.1007/s00170-024-14216-0 | SNIPPET paper |
| Printed TPU bellows | 6,410 cycles at 3 bar | 2.3 | SNIPPET paper |
| PP kayak panel fold | 20,000 (maker's rating, folds not beats) | 1.4 | SNIPPET shop |
| Kapton laminate crease | 15 cycles tested (A15) | A15 | VERIFIED (A15) |
| Articulated ornithopter elbow, 460 g | failed in continuous flapping | 1.3 | VERIFIED paper |
| Fringes, flaps or bristles flexed every beat, wet | **not found** | — | UNKNOWN |

INFERRED: 2e6 / 3,600 = 556 flights for a TPU web, if a printed web in this geometry and strain matches the test
strip, which the source does not establish.

---

## (i) Options

"Works wet": operated in or after water by any machine found. "Hobby" judges parts and fabrication only.

| option | changes | actuation | cycle | demonstrated on (machine, mass) | works wet | hobby-buildable |
|---|---|---|---|---|---|---|
| Waterbomb tessellation / tube | span **and** chord (radial + axial), twists after full squeeze | 1 DOF, actuated (line or pressure) | per flight | none (card and 3D-printed tubes) | unknown | printable (rigid facets, soft creases, as Feng 2018) |
| Kresling tube as a spar | span only, couples a tip twist | 1 coupled DOF; bistable snap | per flight | none (wine-tote and truss models) | unknown | yes; not rigid-foldable |
| Yoshimura | axial | — | — | none as a wing | unknown | yes |
| Flasher | span and chord, radially (9.2 diameter) | 1 input, panels hinge on diagonals | per flight | 1.25 m solar-array model (mass not found) | no | hard (membrane, gaps) |
| Curved crease | bistable shape, not planform | snap | per flight | none as a wing | unknown | plausible |
| Fan / pleat | one direction | 1 DOF | per flight or per beat | earwig wing, 11.4 g (A15) | unknown | yes |
| Linkage fold (elbow/wrist, drawing parallels, Bennett) | span; chord via sweep | 1 DOF, actuated or passive from the stroke | per beat or per flight | RoboFalcon 2.0, 800 g (A15); USGull 460 g (elbow failed); Bennett wing 87 g (bench) | unknown | yes (printed links, carbon) |
| Inflatable spar or wing | stiffness, shape | actuated: stored gas, CO2, pump | per flight | I-2000, 6.8 kg, at 12.4 bar, 0.7 s; Apteron 3.18 kg | no (and floats: MECHANISM.md) | the bladder yes; 12-16 bar gas supply is not hobby kit |
| Inflated tubes stiffening a foil | stiffness ×2 at 0.8 bar | actuated, pump | per manoeuvre or 2× per beat (thesis) | silicone foil rig (mass not stated) | **yes** | yes |
| Pneumatic fold actuator (pouch, vacuum origami muscle) | drives a fold | actuated, 0.2-0.6 MPa or vacuum | per flight | earwig wing 11.4 g (A15); muscles on rigs | air and water (vacuum muscle) | yes |
| Swim bladder (water ballast piston/bladder) | buoyancy | actuated, pump or piston | per dive | 18.58 kg and 63.9 kg UUVs; 2.305 kg tuna robot | **yes** | yes |
| Swim bladder (gas) | buoyancy, unstable with depth (−16% at 2 m) | actuated, pump/CO2 + valve | per dive | none found | — | parts yes |
| Passive collapsing hairs / fringes | area on the recovery | passive, by the flow | per beat | diving beetle 36 mm, water boatman (animals) | **yes** (animals) | plausible (film or bristle strips) |
| One-way louvre flaps on a paddle | area on the return | passive | per beat | FLHex hexapod (mass not read), paddle-fin rig | **yes** | yes |
| Tendon-coupled spreading foot | area on the power stroke | passive tendon, from the leg motion | per beat | duck-leg robot (mass not stated), efficiency 2-2.4× | **yes** | yes |
| Spring area cut on the upstroke (air) | area on the upstroke | passive springs | per beat | TAMU ornithopters (mass not stated): lift ×4, thrust down | no | yes |
| Valve / louvre wing (air) | upstroke porosity | passive | per beat | none measured | — | yes |

## (ii) What this survey could not establish

- **Contraction ratios or a closed formula for a waterbomb tessellation in both in-plane directions**, and its
  deployed bending and twist stiffness; the Oxford thick-panel waterbomb paper returned 404.
- **Any fold pattern that meets all five properties.** Waterbomb: both directions and one DOF, no lock and no
  wing. Kresling: lock, but axial only and not rigid. Linkages: rigid and flown, span only, and the one machine at
  this mass and span (USGull, 460 g, 1.8 m) broke its elbow in continuous flapping.
- Flasher stack height and DOF; Space Flyer Unit array size; Saito's ladybird area ratio.
- **A pressure-stiffness curve measured on a small inflatable spar**, and a wrinkling test at 1-2 bar; the
  Comer-Levy figures are summaries, and tests differ from them by up to 45%.
- Pump flow at 1 and 2 bar back-pressure (only free-flow and maximum pressure were read), so every fill time
  above is a lower bound; full valve mass-and-flow pairs at 1-2 bar are thin (a 4.5 g Parker part at 5 V and a
  10 g Festo part from a reseller). CO2 temperature drop on release and its effect on a fill: not looked up.
- **Any gas-filled buoyancy device on a swimming robot**, with numbers; all found move water.
- **Nachtigall's original *Dytiscus* drag ratio.** The 1/40 is a citation of a review; the 84% belongs to
  *Gyrinus*. Leg area spread vs folded, hair count and stiffness for any dytiscid: not found.
- Duck-foot drag folded vs spread; human recovery drag (coaching only); pigeon and magpie span ratios (Tobalske &
  Dial 1996 PDF not readable); alcid swimming span or area reduction (Lapsansky dissertation 403).
- **Measured thrust or efficiency gain of a per-beat area change on a lift-based flapping wing, in air or water.**
  The TAMU ornithopter raised static lift and lost thrust; no valve-wing measurement exists in what was reached.
  FLHex and the Simha paddle-fin report no number against a symmetric fin in the parts read.
- **Wet fatigue of any per-beat flap, fringe or hinge.** The TPU 2-million figure is a dry test strip.
