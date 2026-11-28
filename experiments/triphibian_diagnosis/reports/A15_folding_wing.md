# A15 — Folding-wing survey: origami, tendons, creases, and the other ways a wing sheds area in water

Survey date 2026-11-15. Web search and page fetches only; no project code, no simulation. Follows
`A11_flapping_design.md` (flapping machines, one wing in two media) and `A13`/`A14` (hardware, power). Written
for the last section of `experiments/flapper/L0/MECHANISM.md` ("The fold itself: origami and a tendon"), which
asks for exactly this before anything is designed.

The machine in view: about 0.6 kg, span about 1.4 m, each wing about 0.66 m by 0.26 m, 3–4.5 Hz in air, the same
wings in fresh water and on land. L1 pass 1b found the full wing too large for water; it must shed span and
chord, then redeploy, many times, under its own power.

Labels, per number (A14's):

- **VERIFIED** — seen today in the fetched page or in the PDF's text (PDFs the fetch tool could not read were
  read with `pdftotext`). Followed by the source class: *paper*, *preprint*, *maker*, *press*, *shop*, *poster*.
- **SNIPPET** — only in a search-result summary or a secondary page; same class tags.
- **INFERRED** — arithmetic of this report on labelled numbers; formula and inputs are shown.
- **UNSOURCED** — general knowledge no page fetched today confirms. Marked each time.
- **UNKNOWN** — not found; what was tried is in the last section.

"Demonstrated" means a machine did it and its mass is given. "Possible" means nothing more than that the
geometry or the numbers allow it.

---

## 0. The wing as arithmetic (INFERRED)

- Area of one wing, as a rectangle: 0.66 × 0.26 = 0.172 m² (upper bound; the planform is not in the brief).
- Chord Reynolds number in air, ν = 1.5e-5 m²/s (UNSOURCED, standard air): 6 m/s × 0.26 / 1.5e-5 = 1.0e5;
  9 m/s gives 1.6e5. Inside the range the brief asks about (3e4–2e5).
- Two different fatigue counts, which the open question in MECHANISM.md runs together: (a) **fold cycles**, a
  few per flight (each water entry and exit); (b) **flap cycles under air load on a deployed crease**, about
  3,600 per 15-minute flight at 4 Hz. A crease that is locked when deployed sees (a) as bending and (b) only as
  load across a closed joint. A crease held "just short of flat" as a corrugation sees (b) as bending.
- Pass 1c's water cases keep the inner 0.6, 0.4 and 0.25 of the span. If chord shrinks by the same factor the
  area ratios are 0.36, 0.16 and 0.06. These are the targets the fold options below are read against.

---

## A. Folding wings that have flown or swum

### A.1 Origami patterns on machines

| machine | pattern, fold | numbers | flaps? fold in flight? | URL | label |
|---|---|---|---|---|---|
| SNU ladybird glider (Baek et al., Sci. Robot. 2020) | "compliant origami": curved (tape-spring) facets that flatten when folded, store energy and self-lock when deployed | Glider module **35 g without electronics, span 660 mm, folds to 1/8 of deployed area**. One joint deploys in 116 ms and bears 210 g, 150 times its weight (so the joint is 1.4 g, INFERRED 210/150). The whole glider deploys in **466 ms** on stored elastic energy. Folding is by "a winding motor with a wire-driven mechanism with a crown gear clutch"; a second "trigger" motor releases it. Frame facet 200 µm thick, radius 12 mm, arch 20 mm | Glides (thrown, and after a jump). Deploys in the air; refolds on the ground. A "deployable flapping mechanism" is shown in a supplementary movie only; no flapping flight is reported | https://www.biorobotics.snu.ac.kr/_files/ugd/f10924_3e90c64e4a714ac7933ead4b726531a3.pdf | VERIFIED paper |
| Earwig-inspired origami wing (Ishiguro et al., Front. Robot. AI 2023) | earwig fan fold, first folding stage only | One wing **11.4 g**, span 125 mm, chord 86–93 mm, open area 8,450 mm² (FW8) / 7,590 mm² (FW10), folded 1,690 / 1,110 mm²: **ratio 4.99 and 6.87**. Thickness 0.16 mm open, 1.3–1.7 mm closed. Skin: three layers of 25 µm Kapton with 50 µm polyamide hot-melt. Opens by a pneumatic pouch motor (0.2 MPa), **closes by a thread reeled by a small motor through holes in each segment**. Wind tunnel 1–4 m/s, Re 5.6e3–2.2e4, angle of attack 0°: FW8 at 4 m/s CL 1.19, CD 0.700, L/D 1.7; FW10 CL 0.542, CD 0.711, L/D 0.76 | Fixed, on a force sensor. Opened and closed in a 4 m/s stream. No free flight, no flapping | https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1255666/full | VERIFIED paper (L/D for FW10 is the fetch tool's division of the table) |
| ETH/Purdue "spring origami" (Faber, Arrieta, Studart, Science 2018) | earwig wing; creases act as extensional and rotational springs, bistable | Earwig wing unfolds to about 10 times its folded area. Prototypes are 4D-printed bistable patterns that self-fold; "only prototypes" | Not a flying machine | https://www.sciencedaily.com/releases/2018/03/180322140854.htm | SNIPPET press |
| Konkuk beetle robot (Phan & Park, Science 2020) | rhinoceros-beetle hindwing: an origami-like fold near the tip that collapses on impact and springs back within one stroke | Robot mass given as 18 g by one secondary page; wing and fold dimensions not read | Flaps (tailless hover). The fold acts in flight, passively, as a collision fuse; it is not an area control | https://techxplore.com/news/2020-12-beetle-wings-mavs-recover-midair.html | SNIPPET press |
| Konkuk/EPFL beetle robot (Phan et al., preprint 2024) | whole wing folds at the base against the body; no crease in the membrane | **18.2 g** robot; tip-to-tip span **20 cm deployed, 3 cm folded**. Wings rise to the stroke plane within two flapping cycles by centrifugal force alone and retract "within 100 ms of motor deactivation" by an elastic tendon at the armpit. Needs > 14 Hz to hold the wing up; tested at 20 Hz. Membrane 10 µm Mylar, 1 mm carbon tube spar. Untethered take-off, hover, landing. With the stop at 90° the wing sagged 12.1° and lift fell; moving the stop to 100° cut the sag to 6.3° and restored lift | Flaps. Deploys and retracts with no fold actuator, but only as all-or-nothing with the motor on or off | https://arxiv.org/pdf/2407.18180 | VERIFIED preprint |
| UPenn "passive folding wing" | Miura origami **tube** as the wing's structure, for a hybrid vehicle | Poster: airfoils thick enough to hold the collapsed tube give enough lift; camber raises lift and "greatly increases drag". No numbers reached (page 404) | Not flown as far as the summary says | https://presentations.curf.upenn.edu/poster/design-and-characterization-passive-folding-wing | SNIPPET poster |

Not found: a **Miura-ori sheet used as a flapping wing surface** on any machine; a **waterbomb or Kresling**
wing that flew; any origami wing above 35 g of wing module or on a machine above about 0.1 kg; any origami wing
operated wet.

### A.2 Bat-type wings (jointed skeleton, elastic membrane)

| machine | numbers | fold | URL | label |
|---|---|---|---|---|
| Bat Bot B2 (Ramezani, Chung, Hutchinson, Sci. Robot. 2017) | **93 g**, self-contained; flaps at about 10 Hz; flew at 4–6 m/s (5.6 m/s average in one test); membrane custom silicone **56 µm**, modulus 11.33 MPa along the fibre direction. Five actuated degrees of freedom. Sized for bats of span 30–40 cm, mass 50–150 g; B2's own span is in table S1, not read ("roughly one-foot" per a Caltech summary, SNIPPET) | Elbow flexion-extension is driven by a "spindle mechanism hosted in the shoulder" (a push-pull screw moving the radial link against the humeral link); the wrist follows the elbow. Used asymmetrically in flight to steer. The paper's own caution: a flexed wing has less area, "thereby reducing the lift force" | https://faculty.cc.gatech.edu/~seth/ResPages/pdfs/ScienceRobotics2017.pdf | VERIFIED paper |
| RoboFalcon (Chen et al., arXiv 2021, IEEE T-RO 2023) | Mass, span and fold ratio not read (the PDF exceeds the fetch limit). Bat-style membrane wing | One motor drives flap and fold together through a "conical rocker mechanism": extended on the downstroke, tucked on the upstroke, every beat. A decoupler lets a servo change the fold on one side for roll. Reported defect: the membrane wrinkles when tucked | https://arxiv.org/abs/2111.02118 | SNIPPET preprint |
| RoboFalcon 2.0 (Science Advances) | **800 g, span 1.2 m**, up to 7.5 Hz, stroke 85°, wing loading 3.64 kg/m², sweep variable by 25°. A14 has its power (400 W peak) | Flap, sweep and fold coupled in each beat by the same mechanism family. Took off under its own power; weak pitch stability at speed and limited yaw control | https://www.techjuice.pk/?p=193821 ; https://www.notebookcheck.net/Engineers-build-first-robotic-bird-to-take-off-like-a-real-bird-with-FSF-motion.1118996.0.html | SNIPPET press |
| Flapperoo (arXiv 2408.15577, 2024) | Span about 70 cm, mean chord 15 cm, 3 Hz, tested at 2 and 4 m/s. Mass not stated | Fold to 70° every beat by a second motor through a **timing belt**; twist by a third motor through **a Bowden cable with antagonistic lines**. Tethered in a wind tunnel on a force sensor. Cycle-averaged lift "close to or even below zero" for the kinematics tried | https://arxiv.org/html/2408.15577v1 | VERIFIED preprint |
| Virginia Tech senior design, bat drone (2024) | Target span 50 cm, 7–10 Hz | Two fold designs compared. Lead screw: active and adjustable, but "large mass" and two motors to synchronise. **Tendon folding: "reduced mass", "single motor", but "unreliable folding", "string tensioning", "string friction"**. The final design dropped both for bevel gears with a cam and carriage, folding on the upstroke only | https://me.vt.edu/content/dam/me_vt_edu/documents/seniordesign/2024/flapping.pdf | VERIFIED poster |
| Festo BionicFlyingFox | **580 g, span 228 cm**, body 87 cm; elastic membrane over "foldable wing kinematics" | The wing folds; whether it folds in flight and by how much was not read (Festo's page returned 403) | https://www.festo.com/us/en/e/about-festo/research-and-development/bionic-learning-network/highlights-from-2015-to-2017/bionicflyingfox-id_32755/ | SNIPPET maker |
| Festo SmartBird | 450 g, span 1.96–2.0 m | No fold found in the sources read | https://en.wikipedia.org/wiki/Festo_SmartBird | SNIPPET |

Read (INFERRED): every flapping machine above 90 g that changes span does it with a **jointed skeleton and a
stretch membrane**, folding once per beat, driven by gears, screws, belts or cams. None of them stops flapping,
holds a small wing, and later redeploys. The one team that tried a tendon on a flapping fold reported it
unreliable for three stated reasons, all about the line.

### A.3 Feathered wings (overlapping stiff elements)

| machine | numbers | fold | URL | label |
|---|---|---|---|---|
| LisHawk (Ajanic et al., Sci. Robot. 2020) | **284 g** ready to fly, maximum span **1.05 m**, wing chord about 30 cm, tail 24 cm; propeller-driven | Artificial feathers (plates) "attached to rods and cables" spread or tuck on command, in flight. Tucked and extended areas not read | https://www.popsci.com/story/technology/robotic-hawk-drone ; https://robohub.org/a-raptor-inspired-drone-with-morphing-wing-and-tail | SNIPPET press (propeller: VERIFIED press) |
| PigeonBot (Chang et al., Sci. Robot. 2020) | 40 real pigeon feathers on elastic bands, four servo-driven wrist and finger joints; mass and span not found | Morphs in flight to turn. **Propeller thrust; the wings do not flap** (flapping and folding are named as future additions) | https://spectrum.ieee.org/automaton/robotics/drones/pigeonbot-uses-real-feathers-to-explore-how-birds-fly | VERIFIED press (feather count SNIPPET) |

Neither flaps. No feathered machine was found swimming.

### A.4 Aerial-aquatic machines

| machine | numbers | fold | URL | label |
|---|---|---|---|---|
| AquaMAV (Siddall, Ortega, Kovac, Interface Focus 2017) | **201.3 g** fixed wing; open span 592 mm, open area 0.056 m², root chord 120 mm, aspect ratio 6.35 | Two outer wing panels pivot on sealed bearings at the quarter chord, geared together, **swept back up to 90° by one 6.5 g servo (MKS DS65K, 0.22 N·m) in under 0.5 s**. Plunge-dived at about 9.1 m/s in a field test. Redeploy after a dive is described, not shown in the test | https://pmc.ncbi.nlm.nih.gov/articles/PMC5206604 | VERIFIED paper |
| Dipper (ETH, RSS 2021) | "light-weight fixed-wing UAV with actively swept wings"; a review gives 3 m/s under water. Mass and span not found | Folds the wings for the water, surfaces, spreads them and takes off | https://roboticsconference.org/2021/program/papers/048/index.html | VERIFIED paper (abstract only); speed SNIPPET |
| Beihang foldable-wing vehicle (Yang et al.), NC State EagleRay (Stewart et al.) | Named in a review; no numbers reached | Beihang: foldable wings; EagleRay: fixed wing | https://www.mdpi.com/2313-7673/9/10/581 | SNIPPET paper |
| MIT/EPFL flapping aerial-aquatic vehicle (Science 2026; see A11) | About 250 g to under 300 g; swims near 1 m/s at about 5 Hz, leaves the water and flies at about 6 m/s; wing sets of 60–100 cm span tried, about 80 cm best | **No fold.** Flexible wings "reduce their effective stroke while underwater, lowering hydrodynamic loads without requiring the complex wing-folding mechanisms" | https://news.mit.edu/2026/new-flapping-robot-swims-and-flies-like-diving-bird-0709 | SNIPPET press |

Read (INFERRED): the only folds demonstrated wet are **rigid sweeps about one pivot** on propeller-driven fixed
wings of about 0.2 kg. The only flapping machine that both flies and swims with one wing pair does not fold at
all; it is also less than half this project's mass and about 0.6 of its span.

---

## B. Miura-ori as a load-bearing or aerodynamic surface

### B.1 Kinematics (Wei, Guo, Dudte, Liang, Mahadevan, PRL 2013; arXiv 1211.6396) — VERIFIED preprint

Unit cell with ridge lengths l1, l2, acute parallelogram angle α, dihedral fold angle θ (θ = π flat, θ = 0
fully folded):

```
ξ = sin α · sin(θ/2)          ζ = cos α · (1 − ξ²)^(−1/2)
w = 2·l2·ξ                    width  (the direction that closes to zero)
l = 2·l1·ζ                    length (closes only to 2·l1·cos α)
h = l1·ζ·tan α·cos(θ/2)       corrugation height
in-plane Poisson ratio  ν_wl = −(dw/w)/(dl/l) = 1 − ξ^(−2)   (always negative)
```

Ranges the paper states: l ∈ 2·l1·[cos α, 1], w ∈ 2·l2·[0, sin α], h ∈ l1·[sin α, 0]. One degree of freedom
when the facets are rigid: the proposal's "one line furls the whole wing" is correct for the ideal sheet.

What that gives (INFERRED, α = 60°, ratios to the flat sheet: w/w_flat = ξ/sin α, l/l_flat = ζ):

| θ | w ratio | l ratio | area ratio | h / l1 |
|---|---|---|---|---|
| 170° | 0.996 | 0.989 | 0.985 | 0.15 |
| 150° | 0.966 | 0.912 | 0.881 | 0.41 |
| 90° | 0.707 | 0.632 | 0.447 | 0.77 |
| 60° | 0.500 | 0.555 | 0.277 | 0.83 |
| 0° | 0 | 0.500 | 0 | 0.87 |

Three consequences for this wing (INFERRED):

- **The two contractions are unequal and one has a floor.** The l direction never goes below cos α. To take
  both span and chord to 0.4 the pattern needs cos α ≤ 0.4, α ≥ 66°; for 0.25, α ≥ 75.5°. Steeper α means a
  stack that is taller when folded (h → l1·sin α).
- **The folded sheet is thick, and normal to the surface.** At area ratio 0.28 the corrugation stands 0.83·l1
  high. With l1 = 30 mm that is a 25 mm deep, 0.13 m long zig-zag facing the water: a furled Miura wing is a
  bluff corrugated block, not a small wing. Whether that block can serve as the water foil at all is not
  answered by any source found.
- **Near flat, a small shortfall costs little area but real depth.** Held at θ = 170° the wing keeps 98.5% of
  its area with corrugations 0.15·l1 deep (4.5 mm for l1 = 30 mm, 1.7% of a 0.26 m chord).

### B.2 Stiffness, and what happens when flat

| finding | source | label |
|---|---|---|
| With rigid facets and elastic creases the in-plane stretching rigidity Kx grows as (π − θ)^(−1) toward the flat state and as θ^(−1) toward fully folded: the sheet resists further in-plane stretching at both ends of its travel | arXiv 1211.6396 | VERIFIED preprint |
| Rigid-facet Miura cannot bend out of plane at all; bending and twisting require the **facets themselves to bend** (modelled as one extra diagonal fold per facet). Bending stiffness per width Bx is then set by the crease and facet-diagonal stiffnesses and the two angles; Bx/By = ν_b² | arXiv 1211.6396 | VERIFIED preprint |
| The saddle (bending) and twist modes "are the most flexible" over a wide range of geometry and material; the in-plane mechanism is the softest only when the facets are much stiffer in bending than the folds. The out-of-plane coupling is equal and opposite to the in-plane Poisson ratio | Schenk & Guest, PNAS 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC3587190 | VERIFIED paper |
| A measured bending stiffness (N·m² or N/m) of a partly unfolded Miura sheet at 0.1–1 m scale | — | UNKNOWN |

Read (INFERRED): the corrugation gives depth, and depth gives bending stiffness across the ridges, but the
pattern's own softest modes are exactly the two a wing is loaded in: spanwise bending (saddle) and twist.
A flat Miura sheet has only the stiffness of its skin. Nothing found shows a Miura sheet carrying a flapping
wing's root bending moment without separate spars.

How deployed folds are locked, as found on machines:

| lock | where | label |
|---|---|---|
| Curved facet that snaps to its curved shape (tape-spring): locks and stores the deploy energy; 1.4 g joint holds 210 g | SNU ladybird glider | VERIFIED paper |
| Bistable crease from a pre-stretched membrane | ETH spring origami | SNIPPET press |
| Hard stop past the plane (100° instead of 90°) plus centrifugal load from flapping | Konkuk beetle robot | VERIFIED preprint |
| Hard stop at full sweep; the servo holds the other end | AquaMAV | VERIFIED paper |
| Line tension alone as the lock of a load-bearing wing | — | not found on any machine |

### B.3 Lift and drag of corrugated or origami surfaces

| finding | Re | source | label |
|---|---|---|---|
| Corrugated "dragonfly" section against a flat plate and a GA(W)-1 airfoil, 4 mm wooden models, PIV: the corrugated section shows no large-scale separation up to about 12.5°; the smooth airfoil shows large-scale separation at 5° and the flat plate at 8°. The text read gives flow fields, **no lift or drag coefficients** | 34,000 | Tamai, Wang, Rajagopalan, Hu, AIAA 2007-0483, https://cfd.ku.edu/papers/aiaa-2007-0483.pdf | VERIFIED paper |
| Corrugated section gives higher lift than a smooth airfoil and a flat plate (Murphy & Hu 2010) | 58,000–125,000 | search summary | SNIPPET paper, no numbers |
| Flat corrugated wings give positive lift at zero angle of attack | 3.0e4–6.0e4 | APS DFD 2025 abstract, https://meetings-archive.aps.org/dfd/2025/t29/1 | SNIPPET |
| Negative: a corrugated fruit-fly wing gave no more lift than a flat plate (Luo & Sun 2005); at Re 1000 corrugation adds pressure drag and no benefit; flat plates have the best L/D at Re 500–10,000 (Vargas et al.) | ≤ 1e4 | search summaries | SNIPPET paper |
| **An actual origami wing: CL 1.19, CD 0.70, L/D 1.7 at best; the higher-ratio fold gave L/D 0.76.** The authors blame hinge flexibility letting the trailing edge deflect | 2.2e4 | Ishiguro et al. 2023 (A.1) | VERIFIED paper |
| Lift and drag of a **Miura-ori** surface, any Reynolds number | — | — | UNKNOWN |

Read (INFERRED): the dragonfly literature says chordwise pleats of a few percent of chord need not cost lift at
Re of a few 10⁴; none of it is a Miura pattern (whose ridges zig-zag and run in both directions), none is above
1.25e5 with numbers read, and the one foldable origami wing with force data has a lift-to-drag ratio under 2.

### B.4 Panels with real thickness

Techniques named in the BYU review ("A Review of Thickness-Accommodation Techniques in Origami-Inspired
Engineering", Applied Mechanics Reviews 70) — SNIPPET
paper, https://www.designreview.byu.edu/collections/beyond-paper-how-to-make-origami-out-of-thick-materials ;
https://ms.copernicus.org/articles/7/69/2016/ms-7-69-2016.pdf :

- **Hinge (axis) shift** (Tachi): each hinge moves to the valley face of the panel.
- **Tapered panels / volume trimming**: material removed at panel edges so they clear.
- **Offset panel**: hinges stay on the zero-thickness surface, panels stand off on posts; keeps the full range
  and the kinematics; "robust standoffs" are the difficulty.
- **Offset crease / membrane hinge**: stiff panels bonded to a continuous flexible layer with a gap at each
  crease as wide as the stack needs.

The last is what the SNU and earwig wings and the film-laminate methods of D use. No Miura wing built by any of
these was found.

---

## C. Tendon and pulley deployment

### C.1 Where a line furls or deploys a wing

| example | line's job | outcome | label |
|---|---|---|---|
| SNU ladybird glider, 35 g, 660 mm | winding motor reels a wire to **fold**; a clutch releases it; springs in the facets **deploy** | works; deploy 466 ms | VERIFIED paper |
| Earwig origami wing, 11.4 g | thread through holes in every segment, reeled by a small motor, **closes**; a pneumatic pouch opens | hinge "intact even after 15 openings and closings"; only the first fold stage achieved | VERIFIED paper |
| Konkuk beetle robot, 18.2 g | elastic tendon (no motor) **retracts**; flapping deploys | works above 14 Hz | VERIFIED preprint |
| LisHawk, 284 g | feathers on "rods and cables" | flies, propeller-driven | SNIPPET press |
| Flapperoo, 70 cm span | Bowden cable, antagonistic pair, drives **twist** through the flapping joint at 3 Hz | works tethered | VERIFIED preprint |
| Virginia Tech bat drone | tendon **fold** on a flapping wing | rejected: unreliable folding, tensioning, friction | VERIFIED poster |
| Tendon-sheath warping of a 160 g membrane wing (Tjahjowidodo & Lee 2017) | sheath-guided tendon | works; "strongly nonlinear" friction made control hard | SNIPPET paper |
| Capstan-driven variable-span small aircraft (Penn State, AIAA SciTech 2026) | capstan and cable extend and retract a full-chord wing section | "successful and repeatable in-flight morphing"; mass and span not on the abstract page | VERIFIED paper (abstract only) |

The layout that recurs (INFERRED): **the line folds, a spring deploys and locks.** No machine found deploys a
wing against air load with a line, and none uses two opposed lines on a fold.

### C.2 Line

| property | UHMWPE (Dyneema/Spectra; braided fishing line and kite line are this fibre, UNSOURCED) | para-aramid (Kevlar) | URL | label |
|---|---|---|---|---|
| density | 0.97 | not read | https://www.fiber-line.com/fibers/uhmwpe | SNIPPET maker |
| fibre strength | 3.1 GPa tensile (one database) | not read | https://matmatch.com/materials/mama00005-dyneema- | SNIPPET |
| creep | 1.7–5.0% at 40–58% of breaking load (Fiber-Line); "almost non-existent" below 20% of break and below 40 °C for the DM20 grade (maker); SK99 only incrementally better than SK78 | "very low or near-zero" | https://www.fiber-line.com/fibers/uhmwpe ; https://www.upffront.com/blog/guides-4/creep-and-long-term-elongation-in-sailing-ropes-strength-stretch-and-stability-explained-361 | SNIPPET maker/shop |
| water | floats; "strength is not affected by water" | moisture regain 5.0% (another sheet: 3.5–7.0%); wet yarn strength unchanged, wet fabric 3–13% weaker (1980 parachute-fabric study) | https://www.premiumropes.com/rope-advice/rope-materials ; https://www.fiber-line.com/fibers/kevlar | SNIPPET shop/maker/paper |
| bending | — | "sensitive to UV and flex fatigue" | upffront guide above | SNIPPET shop |
| pulley size | Cordage Institute CI 1401: tread diameter at least **8 times the line diameter for braided fibre rope**, 10 times for three-strand; another rigging source says 6 times for double braid | same rule | https://www.hhilifting.com/en/news/post/types-of-sheaves-and-their-applications | SNIPPET |

Arithmetic (INFERRED): a 0.5 mm braid asks for a 4 mm pulley tread by the 8× rule. A solid 0.5 mm fibre section
at 3.1 GPa would hold 0.196 mm² × 3,100 N/mm² = 609 N; a braid holds less by its packing and knot losses, which
no page read today quantifies. Kept under 20% of break for creep, that is still tens of newtons from a line
weighing well under a gram per metre (0.196 mm² × 0.97 g/cm³ = 0.19 g/m). The line is not the mass.

Read: for a wing that swims, UHMWPE is the line (unaffected by water, flex-tolerant); its fault is creep, which
matters only if **tension is the lock** — one more reason to lock with geometry and use the line to move.

### C.3 Winches

| actuator | numbers | URL | label |
|---|---|---|---|
| MKS DS65K servo (AquaMAV's wing sweep) | 6.5 g, 0.22 N·m | PMC5206604 | VERIFIED paper |
| GWS S125 1T sail-winch servo, one turn | 49.9 g; 6.62 kg·cm at 4.8 V, 7.63 kg·cm at 6 V | https://servodatabase.com/servo/gws/s125-1t | SNIPPET shop |
| Graupner Regatta Eco sail winch | 70 g; 4.59 kg·cm | https://servodatabase.com/servo/graupner/regatta-eco | SNIPPET shop |
| Hitec HS-785HB sail winch, 3.5 turns | 110 g; 11.0 / 13.2 kg·cm at 4.8 / 6 V; 1.38 s per turn at 6 V; "splash proof" | https://hobbycity.nz/products/hitec-hs-785hb-sail-winch-servo-splash-proof-3-5t-4-8v-11kg-cm-1-68sec-360-6-0-13-2kg-1-38sec-360-59x29x50mm-110g | SNIPPET shop |
| Pololu 298:1 micro metal gearmotor (N20 size) | 9.5 g (low-power version); stall 1.8 kg·cm standard, 6.5 kg·cm for the 6 V high-power version; needs a driver and a position sensor | https://www.pololu.com/product/994/specs | SNIPPET maker |

Arithmetic (INFERRED): force = torque / drum radius; stroke = 2π·radius·turns.

- GWS S125 on a 10 mm radius drum: 7.63 kg·cm / 1 cm = 7.63 kgf = 74.8 N; stroke 62.8 mm in its one turn.
- HS-785HB on a 10 mm drum: 13.2 kgf = 129 N; stroke 3.5 × 62.8 = 220 mm; a full stroke takes
  3.5 × 1.38 = 4.8 s unloaded.
- Pololu high-power on a 5 mm drum: 6.5 / 0.5 = 13 kgf = 127 N at stall, unlimited stroke; at a quarter of stall
  (UNSOURCED rule for continuous use of a brushed gearmotor) about 32 N.

The line travel a Miura wing needs is set by how the line is laced, which no source gives; if it runs the full
span through the stack, travel is of the order of the span lost (0.66 × (1 − 0.4) = 0.40 m for the 0.4 case),
which is beyond every servo winch above on a 10 mm drum and is a gearmotor-and-spool job. Mass budget from
parts actually found: 10–50 g for the winch, shared by both wings if it sits in the hull.

### C.4 Antagonistic or spring return; routing through the shoulder

- Spring (or stored-strain) return with one line is what every working example in C.1 uses. Antagonistic pairs
  appear only on Flapperoo's twist, where both lines run in Bowden housings. VERIFIED preprint.
- Constant length through a rotating joint: Shah, Parmiggiani and Kim (IEEE/ASME AIM 2020, "Constant Length
  Tendon Routing Mechanism through Axial Joint") route four tendons through one rotary joint with full
  decoupling over ±180°, and state that such routing "was missing" from earlier work.
  https://ieeexplore.ieee.org/document/9158889 — SNIPPET paper (abstract and slides only).
- The general condition (Inouye et al., https://valerolab.org/usc-bbdl.github.io/Papers/InouyeRSS2.pdf, SNIPPET
  paper): a tendon does not move with a joint when its moment arm about that joint is zero, that is, when it
  crosses **on the joint's axis**. A flapping shoulder is one axis, so a line led along the flapping axis
  (through a hollow pivot) keeps its length through the stroke; it twists by the stroke angle instead
  (INFERRED from the stated condition, not demonstrated on a flapping wing in any source read).
- Bowden housing is the demonstrated alternative at 3 Hz (Flapperoo). Its cost is friction, the fault named by
  both the Virginia Tech team and the tendon-sheath paper.
- A fold line surviving 3–4.5 Hz flapping for 3,600 cycles: not found.

---

## D. Crease materials and fatigue

| crease | measured life | conditions and limits | URL | label |
|---|---|---|---|---|
| Polypropylene living hinge, injection-moulded homopolymer | "no test has ever worn out a properly designed and molded hinge"; tests "simply terminated after approximately a million flexes"; one unit flexed at 23 °C and one at −20 °F (figure of 900,000 in the search summary) | Hinge 0.20–0.25 mm thick to start, 0.38 mm maximum; land about 1.5 mm; must be flexed right after moulding to orient the polymer; **fillers and glass fibre harm it**; stamped hinges (0.50 mm) "less durable in flexing but stronger in tear" | https://stuff.mit.edu/afs/athena/course/2/2.75/resources/random/Living%20Hinge%20Design.pdf | VERIFIED maker (design guide) |
| Same, industry statements | "more than a million cycles" | thicker hinges exceed the elastic limit and crack at the surface | https://www.arburg.com/media/daten/publications/brochures/Arburg_freeformer_application_note_living_hinges_en_us.pdf ; https://www.plasticstoday.com/materials/design-polypropylene-part-design-part-2-living-hinges | SNIPPET maker/press |
| Kapton film crease (25 µm, three layers, hot-melt bonded) | "intact even after 15 openings and closings" | the only number that paper gives | Ishiguro et al. 2023 | VERIFIED paper |
| Heat-laminated compliant facet, 200 µm (SNU) | not stated in the text read; the paper says creases that concentrate strain "in repeated folding and unfolding" are what the curved facet avoids | — | SNU PDF above | VERIFIED paper (no count) |
| Printed hinges for origami: polyamide, multimaterial photopolymer, aramid-fibre composite (Wagner, Huang, Okle, Paik, Spolenak, Materials & Design 2020) | fatigue is characterised; plain flexure hinges have "limited loading and fatigue resistance"; cycle counts not read (403) | — | https://infoscience.epfl.ch/record/276486 | SNIPPET paper |
| Printed TPU hinge | no cycle data found | a print guide suggests 0.4–0.8 mm thick at 0.2 mm layers | https://print3dbuddy.pages.dev/posts/how-to-print-tpu-hinges-living-hinges/ | SNIPPET forum |
| Smart-composite-microstructure flexures (carbon fibre with polyimide film), tape or fabric hinges | — | — | UNKNOWN |

Water (INFERRED from the material notes above, not from any soak test): polypropylene and UHMWPE do not take up
water; polyimide film and aramid do (aramid regain 3.5–7%, C.2). No source read gives a hinge's cycle life wet.

Read (INFERRED): a million cycles is about 280 flights of flap-rate bending (1e6 / 3,600), or effectively
unlimited fold cycles. But that figure belongs to an oriented, moulded, unfilled polypropylene web 0.2–0.4 mm
thick; a printed polypropylene or TPU web is not that material state, and nothing found says what it lasts.

---

## E. Shedding area in water without origami

### E.1 Birds that fly in both media

| fact | numbers | URL | label |
|---|---|---|---|
| Alcids' wingbeat frequency | **about 2–4 Hz in water against 7–11 Hz in air**; about 50% slower stroke velocity in water; stroke durations 2–3 times longer in water; wingbeat amplitude **greater** in water in all four species; Strouhal number 0.10–0.40 in both media. Species 450 g to 1 kg | https://elifesciences.org/articles/55774 (text read from https://par.nsf.gov/servlets/purl/10201075) | VERIFIED paper |
| Common murre | swimming 1.63 m/s at 2.4 Hz (this study), 1.61 m/s at 2.6 Hz in the wild (Watanuki 2006); flying at 13.3–24.7 m/s | same | VERIFIED paper |
| Guillemots (Uria) | about 8.7 Hz in air, 1.9–2.8 Hz in water | search summary of a J. Exp. Biol. paper | SNIPPET paper |
| Brünnich's guillemot, loggers | 3.2 Hz at the start of descent falling to 2.4 Hz; 1.2–1.8 m/s | https://pmc.ncbi.nlm.nih.gov/articles/PMC1691262 | SNIPPET paper |
| Rhinoceros auklet | stroke amplitude 87° in water, 88° in air (Kikuchi 2015, as cited) | eLife text above | VERIFIED paper (citing) |
| Wing posture | In water these birds "flex the wrist and elbow", reducing effective span and area. **No fraction is given.** On ten murre wing pairs spun as propellers, **extended wings made 1.5–6 times more vertical force than flexed ones** in both media; flexed wings did better only in a gliding test. The authors could not say why the birds flap flexed, and suggest a structural or muscular constraint | https://sicb.org/?p=39618 (Lapsansky & Tobalske, SICB abstract) | VERIFIED paper (abstract) |
| Atlantic puffin | wings "partly folded, with the handwings directed backwards"; an active upstroke in 2 of 24 strokes; the folded hand may act as an aft-swept tip | https://lup.lub.lu.se/record/951365 (Johansson & Aldrin, J. Exp. Biol. 2002) | SNIPPET paper |
| Dippers (Cinclus) | swim with wings partly folded, as auks do (Goodge 1959, cited in The Auk 1988) | https://sora.unm.edu/sites/default/files/journals/auk/v105n03/p0446-p0451.pdf | SNIPPET paper |
| Gannet (about 3 kg) | sweeps the wings partly back at the start of a dive and fully back at impact; dives at up to 60 mph; reaches 10 m | PMC5206604 | VERIFIED paper (as cited there) |
| One wing for both media, tank experiments | a dual-use wing is "of intermediate planform size", between flying-only and swimming-only wings; the wing compensates by its motion | https://link.aps.org/accepted/10.1103/PhysRevFluids.3.073102 (Izraelevitz, Kotidis, Triantafyllou 2018) | SNIPPET paper |
| Span and area reduction fractions for any auk, puffin, dipper or gannet | — | — | UNKNOWN |

Read (INFERRED): the birds' recipe is three things at once: a wing already small for the air (murres have the
highest wing loading of flying birds, SNIPPET), a frequency cut to about a third (2–4 against 7–11 Hz), and a
wrist flex that sweeps the hand back. The fold is a **sweep of the outer wing**, not a pleating of the surface,
and the one force measurement on it found the folded wing worse at making force. The frequency ratio is the
piece this project already has (the variable throw and low water frequency of pass 1b).

### E.2 Engineered alternatives

| option | numbers | demonstrated on | URL | label |
|---|---|---|---|---|
| Sweep-back about one pivot | 90° in < 0.5 s; one 6.5 g, 0.22 N·m servo for both wings; sealed bearings | AquaMAV, 0.201 kg, propeller, wet | PMC5206604 | VERIFIED paper |
| Wrist-type fold of the outer panel, once per beat | 70° fold at 3 Hz (Flapperoo); flap-sweep-fold at up to 7.5 Hz (RoboFalcon 2.0, 0.8 kg) | flapping machines, dry | A.2 | VERIFIED / SNIPPET |
| Telescoping spar | pneumatic telescopic spars at 50–70 psi; aspect ratio changed by 114% (one spar) and 230% (two); carries more than 15 lb/ft²; small drag penalty against a fixed wing. MAK-10 aircraft: 62% span extension | wind-tunnel wings (Blondeau, Maryland); a 2026 capstan-driven small aircraft in flight | https://drum.lib.umd.edu/items/a3391d5a-72a2-444d-b675-6d27c246bcb8/full ; https://pure.psu.edu/en/publications/design-manufacturing-and-flight-test-of-a-capstan-driven-variable/ | SNIPPET paper |
| Roll-up or stowable membrane | patents only; one notes that full-scale membrane wings met "complex vibration problems due to membrane flutter" | none found | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4667898 | SNIPPET |
| Elastic membrane on a folding skeleton | 56 µm silicone, 93 g machine | Bat Bot B2, dry | A.2 | VERIFIED paper |
| Passive bending (no fold) | flexible wings cut their own effective stroke in water | MIT/EPFL vehicle, about 0.25–0.3 kg, flies and swims | A.4 | SNIPPET press |

Telescoping and roll-up change span only; a sweep changes span and the chord presented to the stroke; none of
them reduces chord as a Miura sheet does. Nothing found puts a telescoping or roll-up wing on a flapping
machine.

---

## F. What a hobbyist with a 3D printer could plausibly build

| item | what it gives | URL | label |
|---|---|---|---|
| Earwig origami wing recipe | 25 µm polyimide film and 50 µm hot-melt adhesive film laminated in three-plus-two layers; a thread and a small motor to close. Film, hot-melt web and a laminator are hobby-buyable (availability not checked) | Frontiers paper, A.1 (open access) | VERIFIED paper; buyability INFERRED |
| SNU compliant origami recipe | curved facets formed by "predesigned molds in the heat laminating step"; 200 µm facets; winding motor, wire, crown-gear clutch. The moulds are printable in principle | SNU PDF, A.1 | VERIFIED paper; printability INFERRED |
| Konkuk passive wing | 1 mm carbon tube, 10 µm Mylar, 0.3 mm carbon rod, carbon/epoxy hinge plates 0.5–1.0 mm, an elastic thread | arXiv 2407.18180 | VERIFIED preprint |
| Virginia Tech bat drone | N20 gearmotor, bevel gears, cam and carriage, assembly jigs; a student build with a public poster, no files found | A.2 | VERIFIED poster |
| MakerWorld / Thingiverse folding-wing models | printed fixed-wing aircraft with a wing fold "with only 1 additional servo", TPU inserts; not flapping, not origami | https://makerworld.com/models/1221812 ; https://3dgo.app/models/thingiverse/6984316 | SNIPPET shop |
| Printed TPU hinges | 0.4–0.8 mm webs, print-in-place | D | SNIPPET forum |
| An open design, kit or build guide for an origami or tendon-furled wing | — | — | not found |

Read (INFERRED): the parts are all within reach (film, hot-melt, carbon rod, UHMWPE line, an N20 gearmotor or a
sail-winch servo, printed pulleys of 4 mm tread or more). What is not available is a design to copy at this
size: the largest origami wing with numbers is a 35 g, 660 mm glider module, and the largest tendon-folded
flapping wing found was abandoned.

---

## Summary table

Area ratio is folded/deployed. "Added mass" is the fold hardware where the source separates it. "Wet" means the
fold itself was operated in or after water.

| fold option | area ratio | deploy time | added mass | demonstrated on (machine, mass) | works wet | flaps while deployed | hobby-buildable |
|---|---|---|---|---|---|---|---|
| Miura sheet furled by a line (the proposal) | any, by geometry; l-direction floor cos α (INFERRED from VERIFIED formulas) | unknown | unknown; winch parts 10–50 g, line < 1 g | **none found** | unknown | no instance | parts yes; no design to copy |
| Compliant (tape-spring facet) origami, wire-folded, spring-deployed | 1/8 | 466 ms | whole glider module 35 g at 660 mm span | SNU jump-glider (robot mass not read) | unknown | no (glide); flapping only as a supplementary demo | plausible (laminated film, printed moulds) |
| Earwig fan origami, thread-closed, pouch-opened | 1/5 to 1/6.9 | not reported | wing 11.4 g for 0.0085 m² (1.35 kg/m²; scaled by area to 0.172 m²: 232 g, INFERRED) | bench wing in a 4 m/s tunnel | unknown | no; L/D ≤ 1.7 | plausible; needs compressed air to open |
| Whole wing folds at its base, elastic-tendon retract, flapping deploys | span 20 → 3 cm (0.15) | within 2 beats; retract < 100 ms | one elastic thread | Konkuk robot, 18.2 g, hover | unknown | yes, at > 14 Hz only | yes at that scale; the centrifugal lock does not scale to 3–4.5 Hz (INFERRED: force ∝ frequency²) |
| Jointed skeleton + elastic membrane, fold each beat | not read | each beat | not separated | Bat Bot B2, 93 g; RoboFalcon 2.0, 800 g (press) | unknown | yes | hard (gears, cams, screws; thin silicone) |
| Same, held folded then redeployed | — | — | — | none found | — | — | — |
| Overlapping feather plates on rods and cables | not read | not read | not separated | LisHawk, 284 g; PigeonBot (mass not found) | unknown | no (propeller) | plausible |
| Rigid sweep-back about one pivot | span to chord-wise stowage; area unchanged, frontal area "very small" | < 0.5 s | 6.5 g servo | AquaMAV, 201 g; Dipper (mass not found) | **yes** | no (propeller) | yes |
| Telescoping spar | span only; up to 62% extension (MAK-10), AR +114–230% | not read | not quantified | tunnel wings; one small aircraft in flight (2026) | unknown | no | hard (pressure, or capstan and sliding fit) |
| Roll-up membrane | — | — | — | none found | — | — | — |
| No fold: flexible wing, lower frequency | 1 | 0 | 0 | MIT/EPFL vehicle, 0.25–0.3 kg (press) | **yes** | yes | yes |
| Wrist flex as auks do it (biology) | not found | within a stroke | — | murres, puffins, 0.45–1 kg; flexed wing made 1/1.5 to 1/6 the force | yes | yes | no machine found |

---

## What this survey could not establish

- **Any Miura-ori sheet used as a flying or swimming wing surface**, and any measured lift, drag or bending
  stiffness of one. The Miura rows above are kinematics from a paper and nothing else. The UPenn poster on a
  Miura-tube wing returned 404.
- **Any origami or tendon-furled wing on a machine near 0.6 kg**, any that flapped while carrying the machine's
  weight, and any that was operated wet. The only wet folds are rigid sweeps on propeller aircraft of 0.2 kg.
- **A wing that flaps, stops, furls to a small wing that still works as a foil, and redeploys.** Every flapping
  fold found is once per beat or all-or-nothing.
- **Lift and drag of corrugated sections with numbers at Re 1e5–2e5.** The Hu group's Re 34,000 paper was read
  and gives flow fields only; Murphy & Hu (58,000–125,000) and Hu & Tamai (J. Aircraft 2008) were reached only
  as summaries.
- **Span and area reduction fractions for auks, puffins, dippers and gannets under water.** The eLife paper and
  the SICB abstract state the reduction without a number; Lapsansky's 2021 dissertation and Johansson & Aldrin
  (2002) were not read in full. Murre and puffin wing areas in cm² were not found.
- **Mass, span and fold ratio of RoboFalcon (2021)** (PDF over the fetch limit); **Bat Bot B2's span and wing
  area** (table S1); **PigeonBot's mass and span; LisHawk's tucked and extended areas** (the NCCR copy returned
  403); **Dipper's mass, span and sweep actuator** (PDF over the limit); **Festo flying fox's fold range,
  membrane mass and whether it folds in flight** (403); numbers for the Beihang and EagleRay vehicles; the
  Konkuk 2020 robot's fold dimensions; the SNU robot's total mass, glide ratio and the glider's wing area.
- **Cycle life of any crease other than a moulded polypropylene living hinge**: smart-composite flexures,
  taped film or fabric hinges, printed TPU or printed polypropylene, the SNU facets (count not in the text
  read), and the EPFL printed-hinge study (403). **No wet cycle life for any hinge.**
- **Rated strength and stretch of hobby braided line by diameter**, a measured wet-against-dry strength for
  aramid cord (the one study is on parachute fabric), and a sheave rule from a line maker (the 8× figure is a
  lifting-equipment page citing the Cordage Institute). Dyneema's SK75 sheet returned 403.
- **Line travel and line force to furl a Miura sheet of this size**, in air or against water. No source gives a
  lacing pattern; the winch arithmetic in C.3 therefore sizes parts against nothing measured.
- **A line run through a flapping shoulder for thousands of cycles.** The axis-routing statement in C.4 is the
  general zero-moment-arm condition applied by this report; the one paper on constant-length routing through a
  rotary joint was reached as an abstract, and it concerns a robot forearm.
- **Mass of a fold system at this scale.** MECHANISM.md's typed guess of 10–20 g a side is neither supported nor
  refuted: the nearest data are 35 g for a complete 660 mm glider module and 1.35 kg/m² for a small Kapton
  origami wing, which scale to opposite conclusions.
- **Open designs or kits** for an origami or tendon-furled wing: none found.
