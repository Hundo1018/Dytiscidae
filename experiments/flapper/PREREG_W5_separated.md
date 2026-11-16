# Pass W5, pre-registration: the separated-flow branch against measurements at Re 1e4-2.5e5

2026-11-17, branch `triphibian-diagnosis`, on `18613a3`. Frozen before
`w5_separated.py` is run.

## 0. Question and the decision that hangs on it

The solver sends a wing strip to a "separated" law, `CN sin a cos a` lift and
`CN sin^2 a` drag, with `CN` 1.98 for a plate and 3.4 under a leading-edge
vortex (`lev` from the strip's Rossby number and its travel since reversal).
The constants come from Re 1e2 flat plates (Dickinson 1999). Pass W3 found
99% of the take-off load and about half of pass 1i's cruise load on that
branch, and the plate branch 0.12-0.41 low in lift within 6 deg past a
section's stall.

Does the branch hold at Re 1e4-2.5e5, flat and cambered? The decision: whether
the take-off and wing-shape numbers (W4, and W6 after this) can be read as
they are, need a correction, or carry a stated band.

## 1. Data (all in `sections/separated/`, sources and provenance per number in `sections/SEPARATED_SOURCES.md`)

Gathered by a subagent on 2026-11-17; I checked the Kruyt 2014 fit's peak by
evaluating the printed equation (1.587 at 52 deg) and read the five CSVs.

| set | what | Re | kind of number |
|---|---|---|---|
| Usherwood 2009 | revolving pigeon wing, R/c 3.3; flat card replica | 54k-108k | text: real wing Cv 1.48-1.49 and Ch 0.86-1.02 at 40 deg, CL,max 1.64; card CL,max 1.44, Cv,max 1.3 near Ch 1.1-1.2 (by eye) |
| Kruyt 2014 | revolving hummingbird wing, AR about 4 | 14k | the paper's printed fit, 0-90 deg |
| Kruyt 2015 | revolving 6% cambered plates, AR 2-10 | 9k-25k | by eye: CL,max 1.4-1.5 at 35-45 deg, CD 0.6-0.85 at 35 deg |
| Lentink & Dickinson 2009 | revolving flat fly wing, Ro 2.9 | 14k | by eye: CL,max 1.85-1.9, CD about 1.9 near 45 deg |
| Ananda 2012 | static flat-plate wings AR 2-5 | 40k-100k | by eye, 5-30 deg, +-0.03 |
| Ostowari & Naik 1985 | static NACA 4409 wing AR 6 | 250k | by eye, 10-110 deg, +-0.03/0.04 |
| Sheldahl & Klimas 1981 | 2D NACA 0012 | 360k-700k | table, 30-90 deg |

Not obtained: Usherwood & Ellington 2002 (abstract only), any static cambered
thin plate of AR 2-6, any dynamic-stall or reciprocating force data. So
**nothing here tests the travel-since-reversal weight**, and nothing tests a
wing that both revolves and advances. Definitions differ between sources:
Usherwood's Cv, Ch are the vertical and horizontal force (induced tilt
included), Kruyt's CL, CD are a blade-element reduction. The harness returns
vertical and horizontal force on the second moment of area, Usherwood's
definition; against Kruyt and Lentink that is a comparison with an unknown
induced-tilt difference, and I read those as a band, not a target.

Rivals for any disagreement: the rule is wrong at this Re; the definitions
differ (above); the by-eye readings are off by their stated +-0.05-0.1; the
harness geometry (rectangle, root at the axis) differs from the wing tested.

## 2. What was seen before freezing

- The data, all of it (section 1).
- Three harness runs to see that it works: static flat AR 4 at 5 deg CL 0.364
  (thin-wing value 0.365); Goe 417a AR 6 at 5 deg CL 0.765 and with the
  camber flipped 0.284.
- No run of either rig past stall, none revolving.
- Hand values of the coefficient laws alone (no inflow), which is where the
  predictions below come from: `3.4 sin a cos a` = 1.09, 1.47, 1.67, 1.70 at
  20, 30, 40, 45 deg; `3.4 sin^2 a` = 1.12, 1.40 at 35, 40 deg;
  `1.98 sin a cos a` = 0.55, 0.64, 0.86, 0.99 at 17, 20, 30, 45 deg.

## 3. Predictions (the model as it is)

P1. Revolving flat wing, Re 54k, R/c 3.3, 40 deg: Cv 1.3-1.6 and Ch 1.1-1.4.
    Against the flat card (Cv,max 1.3, Ch 1.1-1.2): within 0.25 on both.
P2. Same rig, Goe 417a section: Cv and Ch within 0.05 of the flat wing's at
    every angle from 20 deg up (the section is not consulted under the
    vortex). So against the real wing at 40 deg (Cv 1.49, Ch 0.86-1.02) the
    model's Ch is high by 0.2-0.5, and it shows none of the 0.2 in lift that
    the real wing has over its flat replica.
P3. Revolving at Re 14k: the flat wing's peak Cv is 1.4-1.7 near 45 deg,
    below Lentink's 1.85-1.9; against the hummingbird fit the model's
    horizontal force at 35-45 deg is 1.5-3 times the fit's CD.
P4. Static flat wings AR 2-5, 5-30 deg, against Ananda: rms error in CL
    under 0.12; the model is low by 0.05-0.20 somewhere in 15-20 deg and high
    by 0.05-0.20 at 30 deg; CD at 30 deg 0.42-0.55 against 0.39-0.40.
P5. Static AR 6 from 25 to 50 deg against NACA 4409's plateau of 0.84 (the
    separated branch has no section in it, so the flat law is the model's
    statement here): within 0.15 at every angle; CD within 0.15 at 30-50 deg.
P6. 2D plate against NACA 0012 at 30-90 deg: CL within 0.10, CD within 0.20.

Falsified: P1 if either coefficient is out by more than 0.25; P2 if the
section moves the revolving wing by more than 0.05 at 20 deg or above; P4-P6
by their stated numbers.

## 4. Decisions, fixed now

- **The vortex constant and the Rossby rule.** If P1 holds, `CN_LEV` and
  `LEV_ROSSBY` stay as they are and are recorded as checked against one flat
  revolving wing at Re 54k. If the flat card is missed by more than 0.25, the
  constant is not retuned on one by-eye point: the miss is reported as a band
  on take-off lift and power.
- **Camber under the vortex.** If P2 holds, the model overstates a cambered
  wing's take-off drag (so power and torque) by the measured gap. The sources
  disagree on that gap by a factor of two (Ch 0.94 against CD 0.47 at 40 deg)
  and by definition, so no coefficient is fitted. The W4/W6 take-off power is
  reported with the band this gives, and the direction (the model is the
  pessimistic end) is stated.
- **Lift just past stall (W3's gap).** Whatever P4-P6 read, `SectionPolar`
  gets an opt-in second table: the section's own measured lift run, which
  goes 6-12 deg past the drag run. Inside it the lift is the measurement.
  Drag there was not measured; it is taken as the force normal to the chord
  that the measured lift implies (`cl tan a`, plus skin friction), entered
  across `SEPARATION_COMPLETE` from the end of the drag run. The handover to
  the plate then runs from the end of the lift run to 30 deg if P5 and P6
  hold at 30 deg (the plate within 0.15 of both wings there), else to the
  lowest angle at which they do. The default law is not changed (the main
  search's scores stay comparable); its dip is recorded.
- **Then the machine.** W4's two reference wings (Goe 417a rectangle, twist 0
  and -8 deg) are re-read on W4's grids with the extended section: cheapest
  cruise at 8 m/s, best take-off lift, gate counts. Prediction P7: cruise
  power moves by under 10% (those points fly attached); take-off lift moves
  by under 5% (all vortex).

## 5. Controls, read points, stopping

One run of `w5_separated.py` (seconds). Deterministic; no seeds. The
inert control is the harness itself at 0 deg, flat: Cv and CL within 0.02 of
zero. Read once, after the run, against section 3 verbatim.

## 6. The drawing

The rigs are a single flat or cambered rectangle on a hinge or a slide, not
the machine; the machine's figure is drawn for the re-read in section 4
(`draw.py`, preset W4, already checked in pass W4) and reopened then.
