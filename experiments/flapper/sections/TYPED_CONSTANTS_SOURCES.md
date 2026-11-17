# Typed constants of the strip solver: published data and formulas to replace them

Compiled 2026-11-17 by web research. Companion to `SEPARATED_SOURCES.md`
(not repeated here: Ananda flat-plate wings, Ostowari & Naik, Sheldahl &
Klimas, Usherwood, Kruyt). Nothing in the repository was edited; no data files
were added. Source PDFs were only read (fetch cache, not kept in the repo).

## Provenance labels

- **TEXT**: printed in the text or a table of the source, which I opened and
  read as extracted text (pdftotext, or OCR for Polhamus).
- **EYE**: read by eye from a figure I rendered and looked at. No digitiser.
  +-0.03 in CL and +-1 deg unless stated.
- **EYE-2ND**: read by eye from a figure in which a *later author replotted*
  the original data (McArthur 2008 replots Schmitz, Mueller, Laitone, Selig).
  Second-hand; same +-0.03, +-1 deg, plus whatever the replot lost.
- **FETCH**: returned by a summarising fetch of the source (a small model read
  the page or OCR text and answered); I did not see the raw text. Treat as
  second-hand quality even when the source is primary.
- **COMPUTED**: computed here from a stated formula (script described inline).
- **SECONDARY via X**: stated by X about the source.
- **MEMORY**: from memory, unverified. Used only where marked.
- **NOT OBTAINED**: could not open.

---

## A. Stall of thin sections at Re 1e4-3e5

### A.0 What the data say, in one paragraph

A thin flat plate (1.3-4.3 % thick) does not "stall" at 11 deg and CL 1.1. Its
lift curve bends at 6-8 deg at CL 0.60-0.76 and then stays on a plateau of
0.6-0.8 out to at least 20-30 deg with **no lift drop**, almost independent of
Re from 1e4 to 4e5 and of aspect ratio from 2 to infinity. A 4-6 % cambered
thin plate reaches CL,max 1.0-1.1 (two-dimensional) at 8-12 deg at Re 2e4-1.7e5,
then **drops** by 0.15-0.25 to a plateau of 0.85-0.90. Above Re about 4e5 the
cambered plate's maximum rises to about 1.3. The mechanism is the thin-airfoil
(long-bubble) stall: separation at the sharp edge from the first degree, the
reattachment point walks aft, lift rounds over when it reaches the trailing
edge at about 7.5 deg (Polhamus 1996, TEXT).

A second error in the default law, not asked for but visible in the same
figures: at Re 1e4-6e4 a 5 % circular-arc plate carries CL about 0.2 at zero
incidence, not the thin-airfoil 4*pi*f = 0.63; its zero-lift angle is about
-2 deg, not -5.7 deg. The full camber lift is only approached at Re >= 1.7e5.

### A.1 Data table

2D = with end plates or wall to wall. "plateau" = CL after the bend/drop.
Camber f and thickness t in % chord.

| Re | f % | t % | AR | alpha at CL,max (deg) | CL,max | just past stall | lift slope | source | provenance |
|---|---|---|---|---|---|---|---|---|---|
| 1e4 | 0 | 3.6 | 2D | 7-8 | 0.62 | plateau 0.60-0.64 to 19 deg, no drop | -- | McArthur 2008, Fig. 5.1a | EYE |
| 1e4 | 0 | 3.6 | 6 | 8 | 0.60 | plateau 0.57-0.62 to 19 deg | -- | McArthur 2008, Fig. 5.1b | EYE |
| 2e4 | 0 | 3.6 | 2D | 7 | 0.60 | plateau 0.59-0.66 to 19 deg, slowly rising | about 0.086/deg mean over 0-7 deg (0.78 x 2 pi), not straight | McArthur 2008, Fig. 5.2a | EYE |
| 2e4 | 0 | 3.6 | 6 | 7 | 0.56 | plateau 0.56-0.64 to 19 deg | -- | McArthur 2008, Fig. 5.2b | EYE |
| 2.1e4 | 0 (thin wedge) | -- | 6 | 9-17 | 0.70-0.75 | smooth rise to a plateau | -- | Laitone 1997, replotted in McArthur Fig. 5.7 | EYE-2ND |
| 4.2e4 | 0 | 2.9 (MEMORY) | 2D | 9-15 | 0.62-0.69 | plateau | -- | Schmitz 1942, replotted in McArthur Fig. 5.7 | EYE-2ND |
| 8e4 | 0 | 1.93 | 2D | 10 | 0.76 | gentle decline to 0.71 at 20 deg | about 0.09/deg over 0-7 deg | Mueller / Pelletier & Mueller, replotted in McArthur Fig. 5.7 | EYE-2ND |
| 8e4 | 0 | 1.93 | 6 | 13-15 | 0.65 | plateau | -- | same | EYE-2ND |
| 8e4 | 0 | 1.93 | 2 | 16 | 0.68 | flat to 25 deg | -- | same | EYE-2ND |
| 1.68e5 | 0 | 2.9 (MEMORY) | 2D | 7, then 16-20 | 0.71, rising to 0.80 | plateau, rising | -- | Schmitz, replotted in McArthur Fig. 5.7 | EYE-2ND |
| 4.2e5 | 0 | 2.9 (MEMORY) | 2D | about 10-20 | 0.6-0.7 | plateau | -- | same | EYE-2ND |
| 4e4-1.6e5 | 0 | 4.3 | 2-5 | about 10 | 0.55-0.70 | constant to 30 deg | -- | Ananda et al. 2015 (in SEPARATED_SOURCES.md) | TEXT there |
| 1e4 | 0 | 1, 3, 5 | 2D (RANS) | slope constant to about 7 deg, then shallower | not given | -- | -- | Winslow et al. 2018, Fig. 9 | FETCH (third-party copy) |
| 1e4 | 5 (circular arc) | 3.6 | 2D | 7-8 | 1.00 | falls to 0.84 by 11-12 deg, plateau 0.84-0.86 | CL(0) about 0.2; zero lift at about -2 deg | McArthur 2008, Fig. 5.1a | EYE |
| 1e4 | 5 | 3.6 | 6 | 9-11 | 0.86-0.89 | plateau 0.78 | -- | McArthur 2008, Fig. 5.1b | EYE |
| 2e4 | 5 | 3.6 | 2D | 9 | 1.09 | drops to 0.87 at 13 deg, plateau 0.87-0.90 to 18 deg | CL(0) about 0.19; slope break at 5-7 deg | McArthur 2008, Fig. 5.2a | EYE |
| 2e4 | 5 | 3.6 | 6 | 10-11 | 0.99 | drops to 0.82 at 14 deg, plateau 0.82 | -- | McArthur 2008, Fig. 5.2b | EYE |
| 2e4 | 5 (circular arc) | thin | 6 | 10-11 | 0.97 | about level | -- | Laitone 1997, replotted in McArthur Fig. 5.10 | EYE-2ND |
| 4.2e4 | 5.8 at 40 % c (Goe 417a) | 2.9 (MEMORY) | 2D | 8 | 1.07 | declines to about 0.90 at 15-20 deg | CL(0) about 0.45 | Schmitz, replotted in McArthur Fig. 5.10; camber stated in McArthur text | EYE-2ND |
| 6e4 | 4 (circular arc) | 1.93 | 2D | 12 | 1.07 | declines to 0.90 at 20 deg | CL(0) about 0.25-0.3 | Pelletier & Mueller, replotted in McArthur Fig. 5.10 | EYE-2ND |
| 6e4 | 4 | 1.93 | 6 | 11-12 | about 1.0 | declines to 0.9 | -- | same | EYE-2ND |
| 6e4 | 4 | 1.93 | 3 | 15 | 0.97 | about level | -- | same | EYE-2ND |
| 1.68e5 | 5.8 (417a) | 2.9 (MEMORY) | 2D | 7.5 | 1.11 | about 0.95 | CL(0) about 0.5 | Schmitz, replotted in McArthur Fig. 5.10 | EYE-2ND |
| 4.2e5 | 5.8 (417a) | 2.9 (MEMORY) | 2D | >= 9 (curve ends) | >= 1.30 | not shown | CL(0) about 0.6 | same | EYE-2ND |
| 2e4 to 1e6 | 6 | 2 | 2D (RANS) | -- | 1.15 at 2e4, 1.32 at 1e6 | -- | -- | Winslow et al. 2018, Fig. 16 / Sec. H | FETCH |
| 1e4 | 3 to 9 | 1 | 2D (RANS) | -- | +47 % from 3 % to 9 % camber | -- | best L/D 15 at 6 % | Winslow et al. 2018, Fig. 11 | FETCH |
| 4e4 to 1.2e5 | cambered plate | -- | -- | -- | rises 4 % over this Re range (N60 section: over 180 %) | -- | -- | Schmitz/Hoerner, SECONDARY via Winslow 2018 Sec. III | FETCH |
| high (not stated in the passage) | 0 (double wedge) | 4.26 | 2D | reattachment reaches the trailing edge at about 7.5 deg | about 0.83 | gentle | -- | Polhamus 1996, p. 9-10, Fig. 1.3 (data of Rose & Altman 1950) | TEXT (OCR) |
| high | 0 (NACA 64A006) | 6 | 2D | jog at 5 deg, then thin-airfoil stall | about 0.90 | gentle | -- | Polhamus 1996, p. 10 (data of McCullough & Gault) | TEXT (OCR) |
| high | 0 (NACA 63-009) | 9 | 2D | reseparation at about 8.5 deg | about 1.15 | abrupt (leading-edge stall), some recovery | -- | Polhamus 1996, p. 11 | TEXT (OCR) |
| high | 0 (NACA 63(1)-012) | 12 | 2D | about 13 deg | 1.5 | large sudden loss | -- | Polhamus 1996, p. 11 | TEXT (OCR) |
| 0.7e6-3e7 | NACA 6-series in the thin-airfoil-stall region | thin | 2D | -- | 0.8 to 1.0, no appreciable rise with Re | gradual | -- | Polhamus 1996, Sec. 1.5, Fig. 1.6 | TEXT (OCR) |
| 1e4, 2e4 | 4 (Eppler 387, 9 % thick) | 9 | 2D | -- | Cl,max below 0.8 at Re 1e4 and 2e4; jumps 0.7 to 1.2 at higher Re when the flow reattaches | lost again above 14 deg | Cl_alpha about 2 pi at 6e4, falling as Re^0.19 below | Spedding & McArthur 2010, Fig. 2, 6, 7 | TEXT |

Supporting statements:

- "Thin plates and airfoil sections up to 3% thickness continue to have
  essentially constant maximum lift coefficients" with Reynolds number
  (Hoerner, Fluid-Dynamic Lift, p. 4-14; FETCH of archive.org OCR).
- Sharp leading edges give higher CL,max than round-nosed sections below
  Re about 1e5; the maximum lift of "all but very thin airfoils" rises sharply
  near Re 1e5 (Hoerner FDL p. 4-11, 4-12, Figs. 11, 12, 16; FETCH). Fig. 12
  plots CL,max of thin plates (A = 5) against camber ratio: **figure not read,
  NOT OBTAINED as numbers**.
- Flow follows a sharp leading edge up to CL about 0.7 (wedge at 6 deg), then
  the lift rounds over, long-bubble type (Hoerner FDL p. 4-2, Figs. 2-3; FETCH).
- Fully stalled sections of A = 5-6: CL 0.6-0.7 near 40 deg, independent of
  Re below 1e5 (Hoerner FDL p. 4-23, Fig. 34; FETCH).
- Zero-lift angle of a circular-arc section moves with Re: about -3 deg at
  1e5, -6 deg at 3e5, -5 deg at 5e5, with nearly the same lift slope (Hoerner
  FDL p. 4-5 to 4-6, Fig. 6; FETCH, OCR noisy).
- No hysteresis and no laminar separation bubble on thin flat and 4 % cambered
  plates at Re 6e4-2e5: laminar separation without reattachment at all angles
  (Pelletier & Mueller, SECONDARY via McArthur 2008 Sec. 1.3-1.4, TEXT).
- Laitone: at Re 2e4-7e4 the 5 % circular-arc plate has the highest lift at
  every angle; thin plates and the 5 % plate are insensitive to Re and
  turbulence, the NACA 0012 is not below Re 5e4; a NACA 0012 flown backwards
  (sharp edge first) lifts more (abstract, FETCH; and McArthur 2008, TEXT).
  Laitone's measured slopes were about 20 % above lifting-line values
  (SECONDARY via McArthur); Pelletier & Mueller found lifting-line slopes
  adequate (SECONDARY via McArthur). The two disagree.
- NACA 0012: peak lift falls 23 % from Re 1e6 to 3e5 and about 46 % from 1e5
  to 1e4; flat plates roughly Re-independent and better when thinner (Winslow
  2018, RANS, Figs. 6, 8, 9; FETCH).

### A.2 A rule that fits the table (my fit, not a published law)

For a thin plate (t <= 4 %), sharp or 5:1 elliptic leading edge,
two-dimensional, 1e4 <= Re <= 2e5:

1. Attached branch: CL = a (alpha - alpha0), a about 0.09/deg (5.2/rad) at
   Re 6e4-8e4, about 0.085/deg at 2e4. alpha0 is **not** -2f at these Re: the
   5 % plate reads alpha0 about -2 deg at 1e4-2e4 (thin-airfoil: -5.7 deg), and
   CL(0) grows with Re (0.2 at 2e4; 0.25-0.3 for 4 % at 6e4; 0.45-0.5 for 5.8 %
   at 4.2e4-1.7e5; 0.6 at 4.2e5). A usable form: effective camber lift
   = k(Re) * 4 pi f with k about 0.3 at 2e4 rising to about 0.8-1.0 at
   2e5-4e5. EYE / EYE-2ND; three models, three facilities; +-0.1 in k.
2. Limit: CL,max = 0.70 + 8 f, capped at 1.10 (f as a fraction; 0.70 flat,
   1.02 at 4 %, 1.10 at >= 5 %). Scatter of the table about this: +-0.07.
   Above Re 2e5 let the cambered cap rise to about 1.3 at 4e5-1e6 (Schmitz
   4.2e5 EYE-2ND; Winslow RANS 1.32 at 1e6). The flat value does not rise
   (0.6-0.8 at 4.2e5; 0.83 for the double wedge at high Re).
   The angle follows from 1: flat 7-10 deg; 4-6 % camber 8-12 deg.
3. Past the limit: **flat plate, hold** CL at 0.60-0.75 (no drop) until the
   separated-plate law overtakes it (CN 1.98 sin a cos a reaches 0.64 at
   20 deg and 0.86 at 30 deg; Ananda's measured wings stay at 0.65-0.75 to
   30 deg, so a finite wing stays on the plateau longer). **Cambered plate,
   drop** over about 3-4 deg to 0.85-0.90 and hold; that is about 0.15-0.2
   above the flat plateau, still present at 18-20 deg.
4. Sections 6 % thick and more with round noses are a different regime
   (leading-edge stall, CL,max 0.9-1.5, abrupt, strongly Re-dependent near
   1e5); the plate rule must not be used for them.

Against this, the default law (2 pi (alpha + 2f) to 11 deg x Re factor, then a
6 deg blend to CN 1.98) gives a flat plate CL about 1.0-1.1 at 10-11 deg where
every source above reads 0.60-0.76.

### A.3 Not obtained (A)

- Pelletier & Mueller 2000 and Mueller 1999 (RTO/VKI) primary text and
  tables: AIAA paywall; DTIC ADP010760 returned 403; SciSpace copies returned
  empty. Only McArthur's replots and descriptions.
- Schmitz, NASA TM X-60976 (NTRS 19700029685): PDF over the fetch size limit.
  Thickness of Schmitz's plates is MEMORY.
- Laitone 1997 full text (Springer paywall). Okamoto & Azuma 2011 (not found;
  it is low-aspect-ratio work at Re 1e4, SECONDARY via Chen & Qin 2013).
- McCullough & Gault TN 2502 itself (mirror unreachable); values are via
  Polhamus. The test Reynolds number of the double wedge / 64A006 data is not
  in the passages read (5.8e6 from MEMORY).
- Selig UIUC low-speed airfoil data volumes: not opened. McArthur's Fig. 5.7
  includes Selig flat-plate points at Re 1e5-3e5, small angles only.
- Hoerner FDL Figs. 11, 12 (CL,max of plates against camber): text only.

---

## B. Span efficiency (the constant 0.75)

### B.1 Two different numbers under one name

Spedding & McArthur 2010, J. Aircraft 47(1):120-128, doi:10.2514/1.44247
(TEXT throughout this subsection):

- Inviscid span efficiency: CD,i = CL^2 (1 + delta) / (pi AR), e_i = 1/(1 + delta)
  (their Eq. 2-3). Measures only the departure from elliptic loading.
- Oswald efficiency: CD = CD,0 + CL^2/(pi AR e), e = 1/(1 + delta + k pi AR)
  (Eq. 6-7), where k is the section's own d(Cd)/d(Cl^2) (Eq. 4). It absorbs
  profile-drag growth with lift.
- Glauert's delta for taper ratios 1 to 0.25: 0.049 >= delta >= 0.016, i.e.
  0.95 <= e_i <= 0.98. For a rectangular unswept wing of AR 6: delta = 0.046,
  e_i = 0.96.
- Whole aircraft, Oswald sense: 0.6 low-wing, 0.8 high-wing; 0.73 (Cessna
  C-310) to 0.93 (Gulfstream G-II). Bird-flight induced-power factors 1.1-1.2.
- Measured, Eppler 387, AR 6 and 2D, Re 1e4-6e4, turbulence 0.025 %:

| Re | k = a2 (2D), quadratic fit, Table 1 | a2 (AR 6), Table 1 | k (2D), low-alpha linear fit, Table 2 | AR 6, Table 2 |
|---|---|---|---|---|
| 1e4 | 0.24 | 0.34 | 0.14 | 0.21 |
| 2e4 | 0.24 | 0.30 | 0.13 | 0.17 |
| 3e4 | 0.22 | 0.14 | 0.08 | 0.09 |
| 6e4 | 0.09 | 0.05 | 0.05 | 0.03 |

  Oswald e at AR 6 from k = 0.24: 1/(pi 6 0.24) = 0.22; from the low-alpha
  fit k = 0.14: 0.27. Inviscid-sense e from (a2,3D - a2,2D): 0.53 (quadratic
  fits) or 0.76 (linear fits, difference 0.07); the authors call this
  subtraction unreliable ("can be negative" at higher Re) and show (Fig. 5)
  that 2D drag plus CL^2/(pi AR e) matches the AR 6 data for any e from 0.5 to
  1. (The column-to-Re assignment of the tables is my reading of the extracted
  text; the worked examples in the text check it for the first two rows.)
- Lift slope: Cl_alpha about 2 pi at Re 6e4 and falls with Re as Re^0.19
  (2D) and Re^0.18 (AR 6).

### B.2 Inviscid delta and tau from lifting line (COMPUTED)

Glauert's monoplane equation, untwisted wing, section slope 2 pi, 199 odd
Fourier terms, collocation (script run in the scratchpad with the project
venv's numpy). CL_alpha = a0 / (1 + (a0/(pi AR))(1 + tau)),
CD,i = CL^2 (1 + delta)/(pi AR).

Rectangular:

| AR | CL_alpha (/rad) | delta | e_i = 1/(1+delta) | tau |
|---|---|---|---|---|
| 1 | 2.04 | 0.003 | 0.997 | 0.04 |
| 2 | 3.04 | 0.010 | 0.990 | 0.07 |
| 3 | 3.63 | 0.019 | 0.981 | 0.10 |
| 4 | 4.03 | 0.029 | 0.972 | 0.12 |
| 5 | 4.31 | 0.038 | 0.963 | 0.14 |
| 6 | 4.53 | 0.048 | 0.954 | 0.16 |
| 8 | 4.84 | 0.068 | 0.937 | 0.20 |
| 10 | 5.05 | 0.086 | 0.921 | 0.23 |
| 12 | 5.20 | 0.103 | 0.907 | 0.25 |

Fit, AR 1-12: delta = 0.0094 AR - 0.008 (max error 0.002).
Check against print: 0.048 here against 0.046 quoted by Spedding & McArthur
for AR 6; at AR 2 pi, taper 1 / 0.25: 0.051 / 0.014 here against Glauert's
0.049 / 0.016 as quoted there.

Taper at AR 6 (delta): 1.0 0.048; 0.8 0.031; 0.6 0.017; 0.5 0.012; 0.4 0.009;
0.3 0.010; 0.2 0.019; 0 (pointed) 0.129.

Limits: lifting line is not valid below AR about 3-4 (Spedding & McArthur,
TEXT: Helmbold's slope 2 pi AR / (2 + sqrt(4 + AR^2)) fits better below AR 4);
the low-AR rows are the formula's output, not a claim about real wings.

### B.3 Consequence

A strip solver that already charges each strip its own profile drag as a
function of its angle of attack must use the **inviscid** value, 0.95-0.99 for
rectangular AR 2-6, or it charges the lift-dependent profile drag twice. 0.75
is an Oswald-type number. If instead the section drag is a constant CD,0, then
e = 1/(1 + delta + k pi AR) with k from the section polar: k 0.05-0.24 at
Re 6e4-1e4 for the E387 gives e 0.22-0.5 at AR 6, far below 0.75.

### B.4 Not obtained (B)

Glauert 1926, Anderson Fig. 5.20, Abbott & von Doenhoff, McCormick: not
opened; the table above is computed from the same equation, checked against
the two values quoted in Spedding & McArthur. Pelletier & Mueller's and
Mueller's own efficiency factors: not obtained.

---

## C. Axial drag of non-lifting parts (frontal-area coefficients)

Hoerner, Fluid-Dynamic Drag, 1965. Read as FETCH of the archive.org OCR text
(identifier FluidDynamicDragHoerner1965); the OCR garbles exponents, so every
formula below is the fetch's reading **and** agrees with my memory of the
printed form. Check against the page before quoting in a paper.

### C.1 Streamline bodies of revolution (Ch. VI, p. 6-16 to 6-19)

- Wetted-area form (Eq. 28): CD,wet / Cf = 1 + 1.5 (d/l)^(3/2) + 7 (d/l)^3.
  The 1.5 term is the supervelocity (Eq. 26), the 7 term separation (Eq. 27).
- Wetted to frontal area (Eq. 30): S_wet / S_front about 4 l/d (Eq. 29: wetted
  area about 0.7-0.8 x length x perimeter).
- Frontal-area form (Eq. 31): CD / Cf = 3 (l/d) + 4.5 (d/l)^(1/2) + 21 (d/l)^2.
  (The fetch read the last two exponents as 3/2 and 3; multiplying Eq. 28 by
  3 l/d gives 1/2 and 2, which is the form I remember. Use 1/2 and 2.)
- Optimum on frontal area: l/d about 2.7 (d/l 37 %); on volume: l/d about 4.
- Cf: laminar 1.328/sqrt(Re_l); turbulent, e.g. 0.455/(log10 Re_l)^2.58 or
  0.074/Re_l^(1/5) (MEMORY; Hoerner Ch. II equation numbers not read).
- Streamline sections (struts), Eq. 6 (p. 6-6): CD (on chord) / (2 Cf)
  = 1 + 2 (t/c) + 60 (t/c)^4 (MEMORY for the turbulent form; the fetch read
  only the laminar-type Eq. 7 clearly: 1 + 1.2 (t/c) + 70 (t/c)^4). Optimum
  strut thickness on frontal area c/t about 3.7 (FETCH).

Evaluated (COMPUTED from Eq. 31), frontal-area CD:

| l/d | CD/Cf | CD at Cf 0.0042 (laminar, Re_l 1e5) | CD at Cf 0.0074 (turbulent 0.074 Re^-0.2, Re_l 1e5) | CD at Cf 0.0047 (turbulent, Re_l 1e6) |
|---|---|---|---|---|
| 2 | 14.4 | 0.060 | 0.107 | 0.068 |
| 3 | 13.9 | 0.058 | 0.103 | 0.065 |
| 4 | 15.6 | 0.065 | 0.115 | 0.073 |
| 6 | 20.4 | 0.086 | 0.151 | 0.096 |
| 8 | 25.9 | 0.109 | 0.192 | 0.122 |

At the low Reynolds numbers of these machines the laminar boundary layer
separates on the afterbody; Hoerner gives a separate subcritical pressure-drag
term (Eq. 21, about 0.33 (d/l) on frontal area, exponent unreadable in the
OCR), which dominates below the body's critical Re. So a "hull" at Re_l 1e4-1e5
is closer to 0.1-0.3 than to 0.06. The typed 0.20 is inside that range; the
formula says which end.

### C.2 Bluff shapes (Ch. III), frontal area

| shape | CD | condition | where | provenance |
|---|---|---|---|---|
| sphere | 0.47 | subcritical, Re 1e3 to about 2e5-4e5 | p. 3-7 | FETCH |
| sphere | about 0.1 | supercritical, around Re 1e6; critical Re defined at CD 0.3, near 4e5 or below (lower with turbulence, roughness) | p. 3-7 to 3-8 | FETCH |
| sphere, Re < 1 | 24/Re | Stokes | Eq. 1, p. 3-1 | FETCH |
| sphere, 1 < Re < 1e3 | 24/Re (1 + 0.15 Re^0.687) | Schiller-Naumann | not in Hoerner | MEMORY |
| disk normal to flow | 1.17 | Re > 1e3 (peak near Re 300; 20.37/Re at low Re, Eq. 2) | p. 3-14, 3-16 | FETCH |
| plate normal, two-dimensional | 1.98 | -- | p. 3-16 | FETCH |
| circular cylinder, cross flow | 1.2 (1.18 two-dimensional), finite length lower | subcritical | p. 3-5, 3-8 | FETCH |
| circular cylinder, cross flow | 0.3-0.4 | supercritical | p. 3-5 | FETCH |
| inclined cylinder | normal-force coefficient x sin^2 (drag x sin^3) of the angle to the axis | subcritical only | Eq. 23, p. 3-11 | FETCH |
| cylinder, axis along the flow, flat face | 1.17 at l/d = 0, falling once l/d passes about 1, about 0.81 for l/d > 2 | -- | Fig. 21, p. 3-12; values SECONDARY via aerospaceweb.org q0231 reading Hoerner's graph | FETCH |
| same, White's table | l/d 0.5: 1.15; 1: 0.90; 2: 0.85; 4: 0.87; 8: 0.99 | Re >= 1e4 | F. M. White, Fluid Mechanics, Table 7.3 | MEMORY (search did not confirm) |
| cylinder with rounded nose | forebody drag falls 50 to over 90 %; nose radius r/d = 0.1 is enough in three dimensions | -- | Fig. 23, p. 3-13 | FETCH |
| blunt base behind a long forebody | base drag about 0.09 for full-diameter base; under 0.01 at d_base/d = 0.5 | -- | Eq. 35, p. 3-21 | FETCH |
| cones, wedges | about linear in half-vertex angle from 0 to the disk value at 90 deg; 60 deg cone 0.5 | Re 1e4-1e6 | Fig. 34, p. 3-18; values SECONDARY via aerospaceweb | FETCH |

A law by shape, from the above: flat-faced stub (strut end-on, foot)
1.17 for l/d < 0.5, 0.8-0.9 for l/d >= 1, plus Cf x 4 l/d of skin friction
for long ones; the same with a rounded nose (r/d >= 0.1): 0.1-0.2 plus base
drag 0.09 if the tail is cut square; streamlined: Eq. 31; sphere-like ballast:
0.47 below the critical Re. A strut or wire **across** the flow is the
cross-flow cylinder, 1.1-1.2 on its projected area.

### C.3 Not obtained (C)

Hoerner's Fig. 21 as numbers (only the two end values, second-hand); White's
table (MEMORY); Leishman's online text and the MIT hull-drag notes (403 /
unreachable). NASA Glenn's shape page gives only: flat plate 1.28, sphere
0.07-0.5, bullet 0.295 (FETCH).

---

## D. Pulsed-jet cavity: what the refill costs

No source opened gives a refill thrust coefficient, and none supports +0.25
(thrust during intake through the jet orifice). What was found:

- Suction draws fluid "from all directions at relatively low peak velocities,
  so its negative thrust is small compared with the blowing phase" (review,
  arXiv 1801.09714, "Bio-inspired underwater propulsors"; FETCH,
  qualitative). Jet momentum force there: F = mdot v_j (their Eq. 4); impulse
  per pulse and thrust coefficient C_T = F / (0.5 rho U_j^2 pi d_o^2/4)
  (Eq. 18); stroke ratio l/d_o about 4 is where a trailing jet forms and
  thrust per pulse stops improving; below about 1 the ring can be re-ingested.
- Squid whole-cycle efficiency, which is the published form of the intake
  cost: eta = 2 V V_j / (2 V_R V + 3 V^2 + V_j^2), V swimming speed, V_j jet
  speed relative to the body, V_R refill inflow speed; against the jet-period
  ("rocket") efficiency 2 V V_j/(V^2 + V_j^2) and Froude 2V/(V + V_j)
  (Anderson & DeMont 2000, SECONDARY via Zhu & Xiao 2022, Bioinspir. Biomim.,
  accepted manuscript, Sec. 1; TEXT of that review).
- "the refill period of squid plays a greater role, and the jet period a
  lesser role" in the low whole-cycle efficiency (Anderson & DeMont 2000
  abstract, via a search listing; SECONDARY). Squid whole-cycle efficiency
  about 49 % recalculated, against 70-93 % jet-phase figures (search listing
  of a later paper; SECONDARY, context not checked).
- With one opening for both directions "the refilling flow is in the opposite
  direction of the jet so that its kinetic energy does not contribute";
  during inflation "little thrust is generated"; slowing the refill cuts the
  waste but lowers speed; real squid refill through a different opening (the
  mantle aperture) (Zhu & Xiao 2022, Sec. 5, TEXT).
- In a free-swimming deformable body the added-mass term changes sign:
  positive in deflation, negative in inflation, net positive because the body
  is faster during deflation (Zhu & Xiao 2022, Sec. 2.2, TEXT).
- Thruster measurements: forces during both expulsion and refilling are
  "accurately modelled" by a pressure analysis; circulation is lost to viscous
  dissipation at the end of refill (Krieg & Mohseni 2015, J. Fluid Mech.
  769:337-368, abstract only, FETCH). The numbers are in the paywalled text.
- Jet-phase augmentation: nozzle-exit over-pressure during ring formation adds
  impulse beyond mdot u_j at small stroke ratio (Krueger & Gharib 2003, Phys.
  Fluids 15(5), search listing; SECONDARY; the size of the effect was not
  confirmed, and a later numerical study (arXiv 2101.00075) disputes it for
  tube nozzles).

Momentum balance (derived here, not from a source; state it as a derivation):

- At rest, ideal sink intake: the far field of a sink carries no net momentum
  flux, so the impulse the body receives over a complete refill equals minus
  the momentum left in the cavity fluid when the refill ends, which is zero
  once that fluid is at rest in the body. **Net refill impulse at rest = 0.**
  Within the refill the force is first against the thrust direction (inflow
  accelerating) and then with it (inflow stopping against the cavity wall).
  Coefficient at rest: 0, not +0.25.
- At speed U: the ingested fluid, at rest in the water, leaves the refill
  moving with the body, so the body pays an impulse m_in U per refill:
  **intake (ram) drag = mdot_in U**, the same term as for any air-breathing
  or water-jet intake. Per cycle, net impulse = m (u_j - U) with u_j relative
  to the body (the jet phase already counts u_j relative to the body), i.e. a
  fraction U/u_j of the jet-phase impulse is lost to refill. At U/u_j = 0.3
  that is 30 %; at rest nothing. This is the Froude-efficiency structure
  2U/(U + u_j) quoted above.
- A rearward-facing intake that takes its inflow from behind the body does
  not change the ram term; viscous loss of the inflow kinetic energy
  (0.5 mdot V_R^2) is an energy cost, not a thrust.

Not obtained (D): Weihs 1977 (Fortschr. Zool. 24:171-175), Siekmann 1963,
Anderson & DeMont 2000 and Anderson & Grosenbaugh 2005 full texts (publisher
403), Krieg & Mohseni 2008 / 2013 / 2015 full texts, Dabiri's papers.

---

## E. Force cap (60 x weight)

Numerical guard. Nothing to source.

---

## F. Open solvers for a cross-check

Target: span 1.4 m, chord 0.26 m, 2-3.5 Hz, 6-8 m/s (k about 0.25, mostly
attached) and at rest (separated, LEV, Re about 1e5); section polars at
Re 5e4-2e5.

| code | method | validated against | licence, language | install | last release seen | flapping + twist, forces, power | limits |
|---|---|---|---|---|---|---|---|
| **PteraSoftware** | steady horseshoe / ring VLM; unsteady ring UVLM, fixed or free wake; Kutta-Joukowski or pressure loads | Yeo et al. 2011 flapping-robot wing pressures ("trends and magnitudes of the net lift over time", Urban & Agarwal 2020/2021, WashU reports; no error figure obtained); report in the repo's `validation/` | MIT, Python (numba) | `pip install pterasoftware`; wheel 1.4 MB plus numpy, scipy, numba, pyvista, matplotlib, mujoco >= 3.2.6 | 5.1.0, 2026-07-15 (PyPI) | wing and each wing cross-section move by arbitrary time functions of sweep, pitch, heave, so flap plus a spanwise twist schedule can be prescribed; forces and moments; **power not listed** (compute from panel loads x velocity yourself) | potential flow: no separation, no LEV, no viscous drag; needs Python >= 3.11 and pulls its own mujoco |
| AeroSandbox | steady VLM, lifting line, AVL/XFOIL wrappers, optimisation | not checked | MIT, Python | `pip install aerosandbox` (casadi, numpy >= 2.2) | 4.2.10 (date not read) | **no unsteady flapping solver** | steady only |
| NeuralFoil | neural surrogate of XFOIL (trained on about 8 M XFOIL runs); test MAE CL 0.012, ln CD 0.020 | XFOIL itself | MIT, Python | `pip install neuralfoil`, wheel 7.6 MB | 0.3.3, 2026-07-11 | section polars only | inherits XFOIL's physics; post-stall only through AeroSandbox's empirical extension; not a measurement at Re 5e4-1e5 for thin plates |
| XFOIL | panel + integral boundary layer, e^n transition | widely (not re-checked here) | GPL, Fortran | source/binaries from MIT; python wrappers exist (not checked) | 6.996, 2026-01-01 (MIT page) | section polars | "limited trailing edge separation", just past CL,max; thin-plate long-bubble stall is outside it |
| AVL | steady extended VLM + slender bodies | -- | GPL, Fortran | source/binaries from MIT | 3.52, 2025-09-03 | steady / quasi-steady only | no unsteady wake |
| XFLR5 | LLT, VLM, 3D panels, XFOIL | -- | GPL, C++ (MEMORY) | GUI binaries | not obtained (page 404) | steady only (MEMORY) | -- |
| OpenVSP / VSPAERO | VLM / panel, unsteady rotor mode (MEMORY) | -- | NOSA (MEMORY), C++ | binaries | OpenVSP 3.56.0, 2026-11-05 | rotating blades; arbitrary flapping not confirmed | potential flow |
| DUST (Politecnico di Milano) | panels + vortex lattice + lifting lines, vortex-particle wake (MEMORY for the element list) | rotorcraft / eVTOL cases (2021 paper cited on the site; not read) | MIT, Fortran (MEMORY) | build from GitLab source (repository unreachable today) | not obtained | prescribed rigid motions and hinges (MEMORY); flapping with twist would need hinge chains | separation only through lifting-line section tables |
| FLOWUnsteady (BYU) | reformulated vortex-particle method (meshless LES) + VLM / lifting line / blade elements | propeller, rotor hover, rotor-wing interaction (Alvarez & Ning 2022-2023) | MIT, Julia | Julia + several unregistered packages + ExaFMM build; heavy | not shown | prescribed kinematics for tilting/maneuvering vehicles; flapping not a documented case | README: "Viscous drag and separation is only captured through airfoil lookup tables"; single node |
| VortexLattice.jl (BYU) | vortex-ring VLM, steady and time-domain with free wake | AVL (steady), Katz & Plotkin (unsteady) | MIT, Julia | `pkg> add VortexLattice` | not shown | unsteady with prescribed velocities; flapping surfaces need per-step geometry updates | potential flow; wake not FMM-accelerated |
| SHARPy (Imperial) | UVLM + geometrically-exact beam; source panels for bodies | JOSS paper (del Carre et al. 2019); validation list not read | BSD-3, Python + C++ | conda environment or Docker; compiled libraries | not shown | prescribed nodal motions exist; built for flexible aircraft, not flapping | potential flow; set-up heavy |
| UnsteadyFlowSolvers.jl (UNSflow, K. Ramesh) | discrete-vortex methods; LESP-modulated leading-edge shedding (LDVM of Ramesh et al. 2014, J. Fluid Mech.) -- LDVM attribution is MEMORY, the README says only "discrete-vortex method" | pitching/plunging plate experiments and CFD at Re 1e3-1e5 (MEMORY) | MIT, Julia | `pkg> add UnsteadyFlowSolvers`; tutorials in UNSflowTutorials | 340 commits, last push late 2024 (listing) | **2D section**, arbitrary pitch-plunge-surge history; lift, drag, moment; LEV shedding switched by a critical LESP that must be calibrated per section and Re | two-dimensional; no spanwise flow, so no LEV stabilisation by rotation |
| OpenFOAM overset / IBAMR | Navier-Stokes, moving overset mesh / immersed boundary with AMR | many flapping-wing studies (not collected here) | GPL / BSD-3 (MEMORY), C++ | OpenFOAM packages; IBAMR via its `autoibamr` script (builds PETSc, SAMRAI, libMesh) | not checked | anything | Re 1e5 with a LEV needs LES or a transition-capable model: order 1e7 cells and days on a workstation per case (MEMORY, order of magnitude only); a 2D laminar/RANS section run is hours |

### Picks

- **(i) Attached forward flight, k about 0.25: PteraSoftware.** It is the only
  candidate that is pip-installable, written for prescribed flapping, lets the
  cross-sections pitch on their own schedule (the twist), and has a flapping
  validation in the repository. Cost: an isolated venv (it wants Python >= 3.11
  and numpy >= 2.1, and would pull its own mujoco, so not into the project
  venv), about half a day to transcribe the wing and kinematics, minutes per
  run. It gives inviscid lift, induced drag and thrust; add the strip profile
  drag by hand before comparing power. Independent of that, the same case has
  closed forms (Theodorsen / Garrick for the section, lifting line for the
  mean) that cost nothing and should be the first check.
- **(ii) Take-off at rest, separated with LEV: nothing cheap is validated for
  it.** No vortex-lattice or particle code above models leading-edge
  separation except through section tables. The affordable option is
  UnsteadyFlowSolvers.jl (LDVM) on two or three span stations, which tests the
  section-level claim (lift held past static stall while the LEV forms) but
  not the rotational stabilisation; about a day including installing Julia,
  seconds to minutes per run, and its critical-LESP input is itself a
  calibration. A real answer is 3D CFD (OpenFOAM overset or IBAMR) at days per
  case, or the measurements already listed in `SEPARATED_SOURCES.md`, which
  remain the better reference for this case.
- **Section polars, Re 5e4-2e5:** for thin plates use the measurements in A
  (and the UIUC volumes, not opened); XFOIL / NeuralFoil are a cross-check for
  the attached slope and drag of a thick section, not for plate stall.

### Not obtained (F)

Release dates for AeroSandbox, FLOWUnsteady, SHARPy, VortexLattice.jl, DUST,
XFLR5; PteraSoftware's quantitative validation error; DUST's repository; any
python XFOIL wrapper's status; measured cost of a CFD run.
