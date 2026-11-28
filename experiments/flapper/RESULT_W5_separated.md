# Pass W5, result: the separated-flow branch against measurements at Re 1e4-2.5e5

2026-11-17. Frozen text: `PREREG_W5_separated.md` (`93f0dce`). Run:
`w5_separated.py` -> `results/W5_separated.jsonl`, read by `w5_read.py`; the
machine re-read `w5_wing.py` -> `results/W5_wing.jsonl`, read by `w4_read.py`.
Data and provenance: `sections/SEPARATED_SOURCES.md`, `sections/separated/`.

## Outcome in one paragraph

A revolving wing's lift under the vortex rule is supported at Re 14k-108k: the
model is within 0.07 of a hummingbird wing's published fit from 5 to 50 deg
and within 0.04 of a pigeon wing's vertical force at 40 deg. Its drag there is
not: 1.3-1.5 times a pigeon wing's and 2.2-2.7 times the hummingbird fit's,
because the model loads every separated wing normal to its chord and real
cambered wings are not loaded that way. Static wings past 30 deg agree with
the plate law within 0.10 (lift) and 0.08-0.19 (drag). Between stall and 30
deg the model was wrong both ways: the default law stalls a flat wing 3-5 deg
late and 0.3-0.5 too high, then drops 0.14-0.17 too low; a section lost
0.2-0.5 of lift just past stall. The second is fixed for a measured section
(opt-in); the first is in the default law and is recorded, not changed.

## Against the frozen predictions

| | frozen | read | |
|---|---|---|---|
| P1 | revolving flat, Re 54k, R/c 3.3, 40 deg: Cv 1.3-1.6, Ch 1.1-1.4; within 0.25 of the flat card (Cv,max 1.3 near Ch 1.1-1.2) | Cv 1.53, Ch 1.30 at 40 deg. At the card's Ch (1.1-1.2, about 37 deg in the model) the model's Cv is 1.47-1.50: 0.17-0.20 above the card | holds, at the edge |
| P2 | Goe 417a within 0.05 of flat from 20 deg; Ch high by 0.2-0.5 against the real wing | identical to three decimals from 0 to 90 deg; Ch 1.30 against 0.86-1.02: high by 0.28-0.44. Cv 1.53 against 1.49 | holds |
| P3 | Re 14k flat peak Cv 1.4-1.7, below Lentink's 1.85-1.9; horizontal force 1.5-3x the hummingbird fit's CD at 35-45 deg | peak 1.56 at 45 deg; 2.6, 2.7, 2.8 times at 35, 40, 45 deg | holds |
| P4 | static flat AR 2-5 against Ananda: rms under 0.12; low 0.05-0.20 in 15-20 deg, high 0.05-0.20 at 30 deg; CD at 30 deg 0.42-0.55 | rms 0.165; **high** by 0.22-0.48 at 12-15 deg (AR 4-5), 0.27 at 15 deg (AR 3) and 0.30 at 20 deg (AR 2), low by 0.14-0.17 at 20-25 deg, within 0.08 at 30 deg; CD 0.36-0.42 | **falsified** |
| P5 | static AR 6, 25-50 deg, within 0.15 of NACA 4409's lift; CD within 0.15 at 30-50 deg | 30-50 deg: lift within 0.08, drag within 0.06. 25 deg: 0.67 against 0.90 | **falsified at 25 deg**, holds from 30 |
| P6 | 2D plate against NACA 0012, 30-90 deg: CL within 0.10, CD within 0.20 | CL -0.10..-0.01, CD -0.04..+0.19 | holds |
| P7 | the machine: cruise power moves under 10%, take-off lift under 5% | below | holds |

Inert control: flat at 0 deg reads Cv -0.000, Ch 0.011 revolving.

Where I was wrong (P4): I expected the default law only to dip after stall.
It also stalls late. With the solver's inflow the strips of an AR 3-5 wing do
not reach the law's 11 deg until the wing is at 15-17 deg, and the law's 2D
slope takes them to 1.0-1.16 there; the flat-plate wings Ananda measured stop
at 0.64-0.70 from 10-12 deg and stay there. A flat plate's own stall is nearer
7-8 deg and its maximum near 0.7. **The default law overstates a flat wing's
maximum static lift by 0.3-0.5.** This is the law the main search scores with.
It does not touch a wing with a measured section (its attached lift is the
table's).

## The revolving wing in numbers

Re 54k, R/c 3.3 (flat and Goe 417a are the same row):

| deg | 10 | 20 | 30 | 40 | 45 | 60 | 90 |
|---|---|---|---|---|---|---|---|
| model Cv | 0.51 | 0.97 | 1.32 | 1.53 | 1.57 | 1.40 | 0.00 |
| model Ch | 0.10 | 0.37 | 0.78 | 1.30 | 1.58 | 2.43 | 3.40 |

Measured (Usherwood 2009, text): pigeon wing Cv 1.48-1.49 and Ch 0.86-1.02 at
40 deg; Ch near 90 deg 2.9-3.0 (by eye). Flat card Cv,max 1.3 near Ch 1.1-1.2
(by eye).

Re 14k, R/c 4, against the hummingbird wing's published fit (Kruyt 2014):

| deg | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| model Cv | 0.26 | 0.52 | 0.96 | 1.31 | 1.52 | 1.54 | 1.38 | 0.00 |
| fit CL | 0.24 | 0.47 | 0.90 | 1.25 | 1.48 | 1.58 | 1.54 | 0.67 |
| model Ch | 0.04 | 0.11 | 0.37 | 0.78 | 1.30 | 1.86 | 2.42 | 3.37 |
| fit CD | 0.13 | 0.15 | 0.21 | 0.32 | 0.47 | 0.67 | 0.90 | 1.83 |

The fit's lift not returning to zero at 90 deg is the fit's form, not a
measurement I can check. The two drags are different quantities (section 1 of
the frozen text): the model's is the horizontal force with the induced tilt
in it, the fit's is a blade-element reduction. Lentink's flat wing at Re 14k
(by eye) has CL 1.85-1.9 with CD 1.9 near 45 deg, a force normal to the chord,
as the model's is, and 0.3 above the model in both.

So: flat revolving wings are loaded normal to the chord and the model is
within 0.2-0.3 of two of them in both coefficients, on either side. Real
cambered wings make the same lift for 0.3-0.4 less horizontal force (pigeon)
or less than half of it (hummingbird fit). The model cannot show that.

## Decisions (as fixed before the run)

- `CN_LEV` and `LEV_ROSSBY` stay. Checked against one flat revolving wing at
  Re 54k (within 0.20) and one at Re 14k (0.3 under).
- **Camber under the vortex: no coefficient fitted.** The take-off numbers of
  W4 and W6 carry this band: for a cambered wing the stroke's power and
  torque at the same lift are 0.66-0.78 of the model's by the pigeon wing
  (Ch 0.86-1.02 over 1.30), and possibly under half by the hummingbird fit.
  The model is the pessimistic end. The same data say a flat wing gets no such
  relief, so this is a reason to build the wing cambered, not flat.
- **Lift past stall.** `SectionPolar(lift_tables=...)`, loaded as
  `"goe417a+lift"` (`sections.load`): the lift run's own values past the drag
  run, drag `cl tan a` + skin friction, the plate by 30 deg (`PLATE_BY`; P5 and
  P6 hold at 30 deg and P5 fails at 25, so 30 is the lowest angle the frozen
  rule allows). Opt-in; the default law and plain `"goe417a"` are unchanged.
  Goe 417a at Re 1e5 now reads 1.37, 1.21, 1.20, 0.97 at 12, 16, 20, 25 deg
  where it read 1.36, 0.68, 0.64, 0.76.
- **The default law's late, high flat-wing stall is not changed here.** It is
  in the search's scoring path and its Mojo mirror; changing it is a
  comparability boundary and the user's decision.

## The machine (P7)

W4's grids, Goe 417a rectangle, held rig, seated fixture:

| twist | section | 8 m/s: sustaining, cheapest | 6 m/s | take-off gate |
|---|---|---|---|---|
| 0 | goe417a (W4) | 4 of 80, 20.9 W | 1 of 80, 30.1 W | 6 of 100, 1.88 weights |
| 0 | goe417a+lift | 6 of 80, 21.5 W | 1 of 80, 30.4 W | 6 of 100, 1.88 weights |
| -8 deg | goe417a (W4) | 8 of 80, 9.7 W | 0 of 80 | 0 of 100, 1.45 weights |
| -8 deg | goe417a+lift | 15 of 80, 10.0 W | 0 of 80 (short by 0.02 N at 30.7 W) | 0 of 100, 1.45 weights |

Cruise power moves 3%; more points sustain (strips that pass stall for part
of the beat keep their lift); take-off is unchanged to two decimals because
every strip is on the vortex law there. W4's conclusions stand.

## What is still not checked

- **The travel-since-reversal weight** (`LEV_TRAVEL`): no reciprocating or
  dynamic-stall force data at Re >= 1e4 was obtained. A flapping wing at rest
  is on the vortex law by its Rossby number alone (r/c <= 2.5 on this
  fixture), so take-off does not lean on it; a wing in forward flight does,
  for the first chords after each reversal of the chordwise flow.
- **A wing that revolves and advances.** No source varied the advance ratio.
  The model takes the Rossby number as U over the angular rate times chord,
  so forward speed turns the vortex off; that is an assumption.
- **The vortex itself at Re 1e5.** The pigeon wing's pressure maps show no
  leading-edge suction band; what is measured is the force.
- Usherwood & Ellington 2002 report flat hawkmoth wings at Re 1e4-5e4 that
  did not make high lift (second-hand here; the papers were not obtained).
  If that holds for the wing as built, take-off lift is overstated for a flat
  wing; the cambered sources do not show it.
- Sections: no static cambered thin-plate wing of AR 2-6 was found; the
  finite-wing check is against a 9% thick NACA 4409 and flat plates.
- A harness defect found on the way: the first 2D case (a 100 m span of 1 cm
  chord) read zero force and was replaced by an 0.8 m wing with the inflow
  off; P6 is read from the second.

## The drawing against the design

The two rigs are a bare rectangle on a hinge and on a slide (no machine). The
machine rows above are W4's Goe 417a rectangle, twist 0 and -8 deg, whose
figures (`figures/W4_taper1.0_twist*.png`) were opened and checked in pass
W4; the body is unchanged (only the section's table past stall differs), so
they were not redrawn.

## Tests

`tests/test_physics.py::test_a_measured_section_replaces_the_default_law_where_it_was_measured`
gained six checks for the lift run; mutations `section-lift-run-ignored`,
`section-stalled-drag-is-skin-friction`, `section-plate-handover-late`.
