# Separated / leading-edge-vortex law: what measurements exist at Re 1e4-2e5

Compiled 2026-11-17 by web research. Scope: check the model's "separated / LEV"
strip law (cl = A sin 2a held past stall, weighted by Rossby number and chords
travelled) against measurements at chord Re 1e4-2e5 on cambered or bird-like
wings, and collect static post-stall data.

Numeric tables are in `separated/*.csv`. Downloaded source PDFs and their text
extractions are in `separated/raw/` (about 46 MB, untrusted data, not executed;
delete or gitignore it).

## How to read the provenance labels

- **TEXT**: the number is printed in the text or a table of the primary source,
  which I opened and read (PDF text or PubMed Central full text).
- **EYE**: read by eye from a figure of the primary source that I rendered and
  looked at. No digitiser was used. Treat as +-0.03 to +-0.1.
- **FIT**: evaluated from an equation printed in the primary source.
- **SECONDARY**: stated by another paper about the source; I did not open the
  source itself.
- **NOT OBTAINED**: paywalled or blocked; only what the abstract states is given.

Nothing below is filled from memory.

## Summary of what the evidence says

1. **High lift on a revolving wing does persist to Re 1e5.** A dried pigeon
   wing and its flat card replica on a propeller rig at Re 54,000-108,000 gave
   CL,max 1.64 and 1.44 at 38-43 deg, against about 1.2 at 8-25 deg for bird
   wings in wind tunnels (Usherwood 2009, TEXT). The author states the flat
   card wing at Re 54,000-108,000 matches flat hawkmoth models at Re 8,000 and
   that "a critical Reynolds number above which the high-lift mechanisms ...
   fail" is hard to support.
2. **A stable LEV was not shown at Re 1e5.** The same paper found no
   low-pressure band along the leading edge in 16-point pressure maps, and says
   a tight LEV is "unnecessary" to explain the forces. What is measured is
   delayed stall and high force, not the vortex. Pressure variability jumps at
   33-38 deg, near maximum lift.
3. **Camber or real-wing shape raises the maximum by about 0.2** at Re 1e5
   (1.64 real against 1.44 flat card, TEXT). At Re 9,000-25,000, 6% cambered
   plates reach CL,max 1.4-1.5 at 35-45 deg for every aspect ratio from 2 to 10
   (Kruyt 2015, EYE), and the authors contrast this with an earlier Ellington
   spinning flat-plate study at Re 1e4-5e4 that did not produce high lift,
   suggesting camber helps beyond the insect regime. Lentink & Dickinson 2009
   report the same conflict and attribute it, without proof, to thin sharp
   sections being sub-critical at 1e4 < Re < 5e4.
4. **The radial limit is about 4 chords from the axis of rotation**, not a
   whole-wing property: on revolving wings of aspect ratio 2-10 at Re
   5,000-25,000 the LEV stays attached for r/c < 4 and separates outboard
   (Kruyt 2015, TEXT); vortex lift drops for r/c > about 3 on a real
   hummingbird wing. For propellers and rotors at Re 250,000 and above the same
   paper cites stall delay to about 3 chords (SECONDARY). Lentink & Dickinson
   put the limit at local Ro = r/c < 3. No measurement at Re 1e5 of the limit
   itself was found.
5. **A translating flat plate of aspect ratio 2-5 at Re 4e4-1e5 holds CL
   0.65-0.75 from 12 to 30 deg**, with no lift drop at stall (Ananda et al.
   2012/2015, EYE and TEXT: CL,max 0.55-0.70 by their moment-break definition).
   This is the static branch the model should fall back to outside the
   Rossby-number limit.
6. **Not found with usable numbers**: dynamic-stall force data (item 4), the
   Usherwood & Ellington 2002 tables, Pelletier & Mueller cambered-plate
   tables, and most ornithopter wind-tunnel data. See "Not obtained".

## 1. Revolving / propeller-rig wings at Re >= 1e4

All coefficients below are blade-element coefficients: force divided by
0.5 rho Omega^2 S2 (second moment of area), torque by 0.5 rho Omega^2 S3. That
is the speed at the radius of gyration, not the tip speed. A coefficient
referenced to tip speed would be smaller by (Rg/R)^2.

### Usherwood 2009, pigeon wing, Re 54,000-108,000

J. R. Usherwood, "The aerodynamic forces and pressure distribution of a
revolving pigeon wing", Experiments in Fluids 46(5):991-1003, 2009.
doi:10.1007/s00348-008-0596-z. Full text read at
https://pmc.ncbi.nlm.nih.gov/articles/PMC3380271/

| item | value | where | provenance |
|---|---|---|---|
| Re | about 108,000 at 8 Hz (10 angles near zero); about 54,000 at 4 Hz; 49,000 at 3.6 Hz for the extreme angles | Sec. 2.4, Fig. 3 | TEXT |
| dried wing | R 0.326 m, S 3.22e-2 m2, S2 9.53e-4 m4, S3 2.09e-4 m5 | Table 1 | TEXT |
| card replica (3 mm corrugated card) | R 0.343 m, S 3.66e-2 m2, S2 1.13e-3 m4, S3 2.61e-4 m5 | Table 1 | TEXT |
| aspect ratio R^2/S | 3.3 (wing), 3.2 (card) | computed here from Table 1 | derived |
| definitions | Cv = 2 Fv / (rho Omega^2 S2); Ch = 2 Q / (rho Omega^2 S3); CL, CD,pro from Cv, Ch with a Rankine-Froude induced-velocity correction | Eq. 2, 4 | TEXT |
| CL,max dried wing | 1.64 | Abstract, Sec. 3.2 | TEXT |
| CL,max card replica | 1.44 | Abstract, Sec. 3.2 | TEXT |
| Cv at maximum, dried wing | 1.49, at wingtip angle of attack 38-43 deg | Sec. 3.2 | TEXT |
| Cv, Ch at 40 deg | Cv 1.48-1.49, Ch 0.86-1.02 | Sec. 3.5 | TEXT |
| CL / CD,pro at maximum lift | 5.4 (8 Hz), 2.0 (4 Hz) | Sec. 3.2 | TEXT |
| card replica Cv,max | about 1.3 at Ch about 1.1-1.2 | Fig. 3A | EYE |
| Ch near 90 deg | about 2.9-3.0 | Fig. 3A | EYE |
| two further pigeon wings, Cv,max | about 1.7 and 1.9 | Fig. 3C, D | EYE |
| wind-tunnel bird wings | CL,max 1.2 at 8-25 deg (Withers 1981; Nachtigall 1979) | Sec. 3.2 | SECONDARY |
| pressure variability | rises suddenly between 33 and 38 deg | Abstract, Sec. 3.4 | TEXT |
| LEV | low-pressure band near the leading edge "is not observed"; unclear whether absent or unresolved by 16 sensors | Sec. 3.4 | TEXT |
| tip deflection | about 100 mm at the maximum-force angle | Sec. 3.1 | TEXT |

Per-angle values are only plotted (Fig. 3), not tabulated. The polar is Cv
against Ch, so angle labels cannot be read from it.

### Usherwood & Ellington 2002a, 2002b

"The aerodynamics of revolving wings I. Model hawkmoth wings", J. Exp. Biol.
205:1547-1564, doi:10.1242/jeb.205.11.1547; "II. Propeller force coefficients
from mayfly to quail", 205:1565-1576, doi:10.1242/jeb.205.11.1565.

**NOT OBTAINED.** The publisher site returned a bot check to every fetch.
Abstract only (TEXT, abstract): part II covers Re about 1,100 to 26,000, model
mayfly, bumblebee and quail wings and real quail wings; hawkmoth planforms of
aspect ratio 4.53 to 15.84; the effect of aspect ratio is "relatively minor,
especially at angles of incidence below 50 deg". Part I: lift and profile drag
coefficients derived from the propeller coefficients are "relatively
successful" up to 50 deg.

SECONDARY (Usherwood 2009, Introduction): the revolving quail wing at
Re about 26,000 indicates "lift coefficients above 1.5 ... at angles of attack
around 45 deg". SECONDARY (Usherwood 2009, Sec. 3.2): adding camber and twist
to the flat hawkmoth model wings "largely failed" to improve on flat-wing
performance at Re about 8,000.

### Kruyt et al. 2015, cambered plates, aspect ratio 2-10, Re 5,000-25,000

J. W. Kruyt, G. F. van Heijst, D. L. Altshuler, D. Lentink, "Power reduction
and the radial limit of stall delay in revolving wings of different aspect
ratio", J. R. Soc. Interface 12:20150051, 2015. doi:10.1098/rsif.2015.0051.
Full text read at https://pmc.ncbi.nlm.nih.gov/articles/PMC4387534/

| item | value | where | provenance |
|---|---|---|---|
| wings | rectangular, chord 15 mm, thickness 0.4 mm, camber 6%; aspect ratio (R/c) 2, 3, 4, 5, 6.5, 8, 10; root offset 9.5 mm | Sec. 2.2-2.3 | TEXT |
| Re | 5,000-25,000 in steps of 4,000; averages over 9,000-25,000 | Sec. 2.5, Fig. 2 | TEXT |
| angles | 0-60 deg (1 deg steps below 20, 3 deg above) | Sec. 2.5 | TEXT |
| CL,max | "similar for all aspect ratios independent of Re", "two times higher than ... translating wings" | Sec. 3 | TEXT |
| CL,max | about 1.4-1.5 at about 35-45 deg, all aspect ratios | Fig. 2a | EYE |
| CL at 20 deg | about 1.0 (aspect ratio 2) to about 1.35 (aspect ratio 10) | Fig. 2a | EYE |
| CL at 10 deg | about 0.45 (aspect ratio 2) to about 0.9 (aspect ratio 10) | Fig. 2a | EYE |
| CD at 35 deg | about 0.6-0.85 | Fig. 2b | EYE |
| radial limit of stall delay | r/c < 4; vortex lift "steeply decreases for r/c > 4"; on the real hummingbird wing it drops beyond r/c about 3 | Abstract, Sec. 3, Fig. 1b, Fig. 4 | TEXT |
| Re dependence of the limit | vortex-lift distribution "Re insensitive" at 5,000, 13,000, 25,000 | Fig. 4c caption | TEXT |
| power | aspect ratio 10 needs up to 48% less power than 2 below 20 deg; aspect ratio 4 needs up to 21% less than 10 above 20 deg | Sec. 3 | TEXT |
| larger scale | stall delay to about three chord lengths on propellers and rotors at Re 250,000 to millions | Discussion | SECONDARY (their refs 37-39) |
| flat against cambered | "contrasts an earlier study by Ellington using spinning flat plates, the cambered plates tested here perform much better, suggesting that camber might help delay stall at Reynolds numbers beyond the laminar flow regime of insects" | Discussion | TEXT |

The aspect ratio here is R/c of one wing measured from the centre of rotation,
which is the Rossby number in Lentink & Dickinson's definition.

### Kruyt et al. 2014, real hummingbird wings

J. W. Kruyt, E. M. Quicazan-Rubio, G. F. van Heijst, D. L. Altshuler,
D. Lentink, "Hummingbird wing efficacy depends on aspect ratio and compares
with helicopter rotors", J. R. Soc. Interface 11:20140585, 2014.
doi:10.1098/rsif.2014.0585. https://pmc.ncbi.nlm.nih.gov/articles/PMC4233735/

- 26 wings of 12 species, aspect ratio 3.0-4.5 (TEXT).
- Fit for *Calypte anna*, alpha in degrees, 0 to 90 (Fig. 6a caption, TEXT):
  CL = 0.0031 + 1.5842 cos(0.0301 a + 4.7124),
  CD = 8.3171 + 8.1909 cos(0.0073 a + 3.1416).
  Evaluated in `separated/kruyt2014_canna_revolving_fit.csv` (FIT): CL 0.90 at
  20 deg, 1.25 at 30, 1.48 at 40, 1.55 at 45, maximum 1.59 at 52 deg.
- The fit is the authors' smoothing, and they state torque did not reach static
  equilibrium beyond 35 deg (TEXT), so CD above 35 deg is less certain.
- Minimum hover power at about 16 deg; no prominent LEV at 16 deg; stable LEVs
  at 30 and 45 deg for eight species, attached near the root and separating at
  75% span (TEXT).
- The C. anna wing Re is given as 14,000 in Kruyt 2015, Fig. 1c caption (TEXT).

### Lentink & Dickinson 2009, Re 110 to 14,000

D. Lentink, M. H. Dickinson, "Rotational accelerations stabilize leading edge
vortices on revolving fly wings", J. Exp. Biol. 212:2705-2719, 2009.
doi:10.1242/jeb.022269. PDF read from the Caltech
repository: https://authors.library.caltech.edu/15114

| item | value | where | provenance |
|---|---|---|---|
| definitions | CL = 2L / (rho S Vg^2), Vg at the radius of gyration; Ro = R/c of one wing | Methods, Eq. 6 | TEXT |
| Re 14,000 | reached in water; force augmentation at low Ro "from Re=110 to 14,000" | Fig. 6 caption | TEXT |
| Ro effect | augmentation decreases from Ro 2.9 to 3.6 to 4.4 (Re 1,400); no LEV shedding seen in force or video at those Ro | Results, Fig. 6A, B | TEXT |
| bursting | LEV bursts above Re about 1,000; "bursting does not result in a loss of force augmentation" | Results | TEXT |
| Re dependence | polars at Re 1,400 and 14,000 overlap below 45 deg | Fig. 7 caption | TEXT |
| revolving, Re 14,000 | CL,max about 1.85-1.9 at CD about 1.9; CD at 90 deg about 4.0 | Fig. 7B | EYE |
| translating, Re 1,400-14,000 | CL,max about 1.0 at CD about 0.9 | Fig. 7A | EYE |
| ratio | revolving wings "generate easily up to twice as much lift and drag" as translating, 110 < Re < 14,000 | Discussion | TEXT |
| local limit | a slender wing supports a stable LEV near the base where local r/c < 3 | Discussion | TEXT |
| higher Re | "Ellington and Usherwood (2001) found ... revolving model hawkmoth wings operating at 10,000<Re<50,000 failed to produce high lift. The reason for this is unclear"; contrasted with their own Re 14,000 result and the quail wing at 26,000; airfoil shape (thin, sharp, sub-critical) proposed as the cause | Discussion, "Lift augmentation at Reynolds numbers higher than 14,000" | TEXT |
| propellers | section lift coefficients up to 3 near the hub of Himmelskamp's propeller, Re 280,000 | Fig. 11 caption, Discussion | SECONDARY |

The model wing is a flat fruit-fly planform. Their revolving CL,max of about
1.9 is higher than the bird-wing and cambered-plate values above (1.4-1.7).

### Heers, Tobalske & Dial 2011, chukar wings

A. M. Heers, B. W. Tobalske, K. P. Dial, "Ontogeny of lift and drag production
in ground birds", J. Exp. Biol. 214:717-725, 2011. doi:10.1242/jeb.051177.
https://pmc.ncbi.nlm.nih.gov/articles/PMC3036546/

- Maximum L:D rises from 0.88 at 8 days to 4.00 in adults, at 31, 38, 28, 24
  and 15 deg for the five age classes (TEXT).
- Aspect ratio 2.5-3.6, camber 0.43-0.55 in their measure, neither correlated
  with peak CL (|r| <= 0.20); feather stiffness and asymmetry did (r > 0.95)
  (TEXT).
- Adult wing forces "very similar to previously published results for pigeons
  (Usherwood, 2009)" (TEXT).
- Adult (100 d.p.h.) CL peaks at about 1.7 near 40-50 deg; CL about 1.15 at
  20 deg; CD about 1.0 at 45 deg; 49 d.p.h. wing peaks at about 1.5 (Fig. 1A,
  B, EYE).
- A Re of 12,893 is given for the 10 d.p.h.-equivalent condition (TEXT). The
  adult Re was not extracted.

### Crandell & Tobalske 2011, pigeon wing

"Aerodynamics of tip-reversal upstroke in a revolving pigeon wing", J. Exp.
Biol. 214:1867-1873. doi:10.1242/jeb.051342. **NOT OBTAINED** (publisher bot
check). Abstract as returned by a search listing (SECONDARY): peak L:D 2.5 in
upstroke posture and 6.3 in downstroke posture.

### Jardin & Colonius 2018 (CFD, Re 577), for the Rossby-number rule only

"On the lift-optimal aspect ratio of a revolving wing at low Reynolds number",
J. R. Soc. Interface 15:20170933, 2018, doi:10.1098/rsif.2017.0933 (volume and
year inferred from the DOI).
https://pmc.ncbi.nlm.nih.gov/articles/PMC6030621/ Read through a summarising
fetch, not line by line; treat as SECONDARY quality.

- Flat rectangular plates, 45 deg, Re 577 at the radius of gyration. Not in the
  target Re range.
- Ro defined as Rg/c. CL rises roughly linearly as Ro falls from 6.03 to 0.58
  at fixed aspect ratio; at fixed Ro, CL rises with aspect ratio.
- Inboard stable LEV to r/c about 3, unsteady outboard, for aspect ratio > 4.
- Mean lift over the first 2 chords of travel is nearly independent of Ro above
  Ro about 3: the early-stroke lift does not need rotation.

### Garmann & Visbal, Harbig et al., Jardin & David

**NOT OBTAINED.** Not searched to full text; all are computations or
experiments at Re of order 1e2-1e4 (from memory, unverified) and would not
extend the Re range.

## 2. Does the LEV stay attached at Re 1e5, and where does it end

- Force evidence: yes for high lift, to Re 108,000, on a wing of R/c about 3.3
  (Usherwood 2009).
- Flow evidence: none at Re 1e5 in what I could open. The pigeon pressure maps
  did not show the vortex.
- Radial limit: r/c about 4 at Re 5,000-25,000 measured (Kruyt 2015); about 3
  quoted for rotors at Re 250,000 and above (SECONDARY); Ro = R/c of 2.9 to 4.4
  already reduces augmentation at Re 1,400 (Lentink & Dickinson 2009).
- Forward flight: Lentink & Dickinson note, citing Hubel 2006, a stable LEV
  near the base of a flapping goose model in forward flight, where local r/c is
  small (SECONDARY). Hubel & Tropea's SICB abstract, as returned by a search
  listing, says the LEV "is not stabilized over the wingbeat cycle" at bird
  scale (SECONDARY).

For the 0.26 m chord, 0.66 m half-span wing: r/c at the tip is 2.5 measured
from the root hinge, so in pure flapping at zero advance the whole span is
inside the r/c < 3 to 4 limit. The limit that matters in forward flight is the
advance ratio (U against Omega r), which none of the hover-rig papers varied:
all are at zero advance (Usherwood 2009 notes only a geometric offset term of
0.09 from mounting the wing 34 mm off the axis, TEXT).

## 3. Static, translating plates and wings past stall

### Ananda, Sukumar & Selig, flat-plate wings, aspect ratio 2-5, Re 4e4-1.6e5

G. K. Ananda, P. P. Sukumar, M. S. Selig, "Measured aerodynamic
characteristics of wings at low Reynolds numbers", Aerospace Science and
Technology 42:392-406, 2015.
https://m-selig.web.engr.illinois.edu/pubs/AnandaSukumarSelig-2015-AeroSciTech-LowReWings.pdf
Conference version with every wing's curves: AIAA 2012-3026,
https://m-selig.web.engr.illinois.edu/pubs/AnandaSukumarSelig-2012-AIAA-2012-3026.pdf

- Ten flat-plate wings, 4.3% thick, 5:1 elliptic leading edge, 10:1 trailing
  edge, mean chord 88.9 mm, aspect ratio 2, 3, 4, 5, taper 0.5, 0.75, 1;
  alpha -15 to 30 deg; Re 60,000-160,000 (2015), 40,000-100,000 curves in 2012
  (TEXT).
- CL,max "in the range of 0.55 to 0.7" for all wings, defined as CL where the
  moment-curve slope is largest; "relatively constant with variation in aspect
  ratio"; stall at about 10 deg (TEXT, 2015 Sec. 3.1-3.2, Fig. 14-15).
- "In the post-stall region, both lift and moment coefficients are ...
  relatively constant over the range of angles of attack tested (up to
  30 deg)"; no hysteresis (TEXT).
- **No downloadable data files exist for these wings.** The UIUC low-speed
  airfoil data site covers 2D airfoils only. The full set is in Ananda's 2012
  MS thesis (https://hdl.handle.net/2142/42319), which I did not open.
- `separated/ananda2012_flatplate_rect_wings.csv`: rectangular wings, aspect
  ratio 2, 3, 4, 5, alpha 5 to 30 deg (EYE, from AIAA 2012-3026 Figs 13, 16,
  19, 20). Between 12 and 30 deg: aspect ratio 2, CL 0.64-0.75; 3, 0.67-0.70;
  4, 0.65-0.75; 5, 0.67-0.81. CD at 30 deg about 0.39-0.40, close to
  CL tan(alpha).

### Ostowari & Naik, NACA 44xx, aspect ratio 6 to infinity, to 110 deg

C. Ostowari, D. Naik, "Post-stall wind tunnel data for NACA 44XX series
airfoil sections", SERI/STR-217-2559, 1985.
https://www.osti.gov/servlets/purl/5791328

- Non-rotating rectangular blades, NACA 4418, 4415, 4412, 4409; aspect ratio
  6, 9, 12, infinity; Re 0.25e6 to 1.0e6; alpha -10 to 110 deg (TEXT). Above
  the target Re range, and thicker than the wing here.
- Post-stall lift and drag "relatively insensitive to Reynolds number"; lift
  falls with aspect ratio; CD,max at 90 deg is 2.06 at infinite aspect ratio
  and 1.32 at aspect ratio 6 (NACA 4418, Re 0.25e6, TEXT, Sec. 4.6).
- The report has figures only. `separated/ostowari_naik_naca4409_AR6.csv` is
  NACA 4409, aspect ratio 6, Re 0.25e6 (EYE, Fig. A.16): CL,max about 1.07-1.12
  at 18-20 deg, a plateau of 0.80-0.84 from 25 to 50 deg, 0.73 at 60, 0.06 at
  90; CD 0.46 at 30, 0.80 at 45, 1.43 at 90.

### Sheldahl & Klimas, NACA 00xx, 0-180 deg

R. E. Sheldahl, P. C. Klimas, "Aerodynamic characteristics of seven
symmetrical airfoil sections through 180-degree angle of attack ...",
SAND80-2114, 1981. https://www.osti.gov/servlets/purl/6548367

- Two-dimensional sections. Measurements at Re 3.6e5-7e5; the low-Re blocks
  (1e4-1.6e5) are computed below stall and reuse the measured post-stall rows,
  so the low-Re pre-stall and near-stall values are not data (TEXT).
- `separated/sheldahl_klimas_naca0012_poststall.csv`: NACA 0012, 30-90 deg
  (TEXT, Table 2, cross-checked across Re blocks): cl 0.915 at 30, 1.075 at 40,
  1.085 at 45 (the post-stall maximum), 0.875 at 60, 0.07 at 90; cd 0.57 at 30,
  1.075 at 45, 1.80 at 90.
- Use as the infinite-aspect-ratio bound. Finite wings read lower: compare
  0.84 at 45 deg for aspect ratio 6 above.

### Not obtained in this group

- Pelletier & Mueller 2000, J. Aircraft 37(5):825-832 (flat and 4% cambered
  plates, Re 6e4-2e5). Abstract only (search listing, SECONDARY): "cambered
  plates offer better aerodynamic characteristics and performance"; no
  hysteresis. Ananda et al. 2015 (TEXT) say their aspect-ratio-3 flat plate
  agrees with Pelletier & Mueller's lift and drag at Re 80,000 and note that
  model was 2.6% thick with a 5:1 trailing edge and semi-span.
- Mueller 1999 (RTO/VKI lecture), Torres & Mueller 2004, Okamoto & Azuma 2011,
  Hoerner. Not opened. Ananda et al. 2015 (TEXT) note Torres & Mueller's rise
  of CL,max at aspect ratio <= 2 was not seen at aspect ratio 2-5.
- **No static cambered thin-plate table for aspect ratio 2-6 past stall was
  obtained.** The only cambered post-stall finite-wing data here is the 9%
  thick NACA 4409 at Re 2.5e5.

## 4. Dynamic stall, pitching and plunging at Re 1e4-1e5

**NOT OBTAINED: no force numbers.** Springer, DTIC, the Michigan repository
and Cambridge all refused automated access.

From the abstract of Ol, Bernal, Kang & Shyy, "Shallow and deep dynamic stall
for flapping low Reynolds number airfoils", Experiments in Fluids 46:883-901,
2009, doi:10.1007/s00348-009-0660-3, as returned by a search listing
(SECONDARY): SD7003, Re 10,000-60,000; Re affects the laminar separation
"where instantaneous angle of attack was well below stall, but ha[s] limited
effect on post-stall flowfield". From the abstract of the companion flat-plate
work (SECONDARY): a sharp leading edge gives a closed separation in the
shallow-stall case and an LEV in the deep-stall case, with strong Re effects
for the SD7003 and not for the flat plate.

Baik et al. 2012, Granlund & Ol, Rival et al.: not opened.

## 5. Complete bird-scale flapping wings

### Send et al. 2012, SmartBird

W. Send, M. Fischer, K. Jebens, R. Mugrauer, A. Nagarathinam, F. Scharstein,
"Artificial hinged-wing bird with active torsion and partially linear
kinematics", ICAS 2012, paper 053.
https://icas.org/ICAS_ARCHIVE/ICAS2012/PAPERS/053.PDF (all TEXT)

| item | value | where |
|---|---|---|
| span, area, mean chord | 2 m, 0.5 m2, 0.25 m | Fig. 1 caption |
| mass | 0.48 kg with battery | Fig. 1 caption |
| design point | 5 m/s, 2 Hz, 23 W average | Fig. 1 caption |
| free flight | u0 = 4.7 m/s; electrical power 18 W, of which servos 5 W | Sec. 3.4 |
| total efficiency | about 0.25 (thrust power over net electrical power, tethered) | Sec. 3.4, Fig. 5 |
| aerodynamic efficiency | above 0.6 and up to 0.8 (tethered, whirling-arm rig) | Sec. 3.2 |
| thrust | 0.7 N, computed as 0.25 x 13 W / 4.7 m/s; cT = 0.1 | Sec. 3.4, Eq. 4 |
| lift coefficient | 0.7 at 0.48 kg; cL/cD about 7 | Sec. 3.4, Eq. 5 |
| Re, reduced frequency | about 80,000; omega c / (2 u0) = 0.3 to 0.45 | Sec. 3.4 |

The thrust is inferred from power and efficiency, not measured in free flight.
Chord and Re match the wing here (0.25 m against 0.26 m; Re 8e4); the span is
1.5 times larger and the mass 0.8 times.

### Not obtained in this group

- Hubel & Tropea 2010, J. Exp. Biol. 213:1930-1939, doi:10.1242/jeb.040857.
  Search listing only (SECONDARY): goose-sized model, reduced frequency
  0.05-0.3, Re 28,000-141,000 in an earlier study; unsteady effects "present
  and non-negligible".
- Mazaheri & Ebrahimi 2011, J. Fluids Struct. 27:586-595,
  doi:10.1016/j.jfluidstructs.2011.04.001. Abstract only (SECONDARY): thrust
  rises with flapping frequency; higher tunnel speed and angle of attack lower
  thrust and raise lift. No numbers.
- Harmon 2008, MS thesis, University of Maryland,
  https://drum.lib.umd.edu/handle/1903/8968. The PDF host refused the
  download. Abstract only.
- DeLaurier 1993, Gerdes et al. (Robo Raven): not searched.

## Consequences for the model, as far as these sources reach

- Keeping high lift past static stall on the flapping strip is supported at
  Re 5e4-1e5 for r/c up to about 3-4 and low advance ratio. The measured
  ceiling for bird-like and cambered wings is CL 1.4-1.7 at 35-50 deg
  (radius-of-gyration reference), below the 1.8-1.9 of flat insect-scale
  models. A sin 2a law peaks at 45 deg, which matches; an amplitude A near
  1.5-1.6 fits the cambered and real wings, 1.44 the flat card at Re 1e5.
- Drag on that branch: Ch 0.86-1.02 at 40 deg (pigeon, Re 54,000), CD about
  1.0 at 45 deg (chukar), about 0.56 at 45 deg (hummingbird fit), about 3.0 at
  90 deg (pigeon). The spread at 45 deg is a factor of two between sources.
- Outside the rotational limit the measured static values are CL 0.65-0.75
  from 12 to 30 deg for flat plates of aspect ratio 2-5 at Re 4e4-1e5, and
  0.80-0.84 from 25 to 50 deg for a cambered aspect-ratio-6 wing at Re 2.5e5.
  A model reading 0.1-0.4 below static data just past stall should be checked
  against these two files first.
- The "chords travelled since reversal" weight is not tested by any source
  opened here at Re >= 1e4. The only relevant result is the Re 577 computation
  (lift over the first 2 chords nearly independent of Ro above Ro 3).
- The dependence on advance ratio is not measured in any source opened here.
