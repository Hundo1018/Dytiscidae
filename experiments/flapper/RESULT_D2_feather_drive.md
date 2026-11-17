# Pass D2, result: what turns the root feather at take-off

2026-11-17. Frozen text: `PREREG_D2_feather_drive.md` (`7562773`) and its
addendum D2a (`38fd8bf`). Code: `drive/drive.py` (feather modes `motor`,
`crank`, `crank2`, `passive`), `drive/d2_run.py`, `drive/d2_read.py`,
`drive/draw_d2.py`. Rows: `results/D2_grid.jsonl` (20) and
`results/D2_fx2.jsonl` (6, FX2 re-run after a bug fix, below). Nothing is
chosen here.

## Outcome in one paragraph

At take-off (4.5 Hz, stroke 85 deg by D1's four-bar at 60 mm, feather 50 deg
a quarter beat ahead), **the root feather needs about 27-29 W of shaft
power, and how it is delivered sets the bill.** The ideal servo (F0) costs
28.8 W on top of the stroke motor's 149.4 W: 178.2 W for the machine. A
second witness motor on the feather (FM) tracks it to 0.6 deg, but its
copper loss on a slow, high-torque, reversing load makes the machine 295.5 W
at its best gear (16). A feather crank driven off the stroke's crank can
be built only on a shaft of its own (FX, the arm on the stroke crankshaft
as frozen, closes at no rod length). That geared version (FX2) runs, but
the engine does not hold its loop: 2 mm gaps, 3.4 kN in the rod, 3% energy
error in a vacuum. Its rows are labelled and not read as a candidate's
performance. No passive spring makes the feather: either it swings 52-65
deg half a beat late and the wings lift downward, or it is stiff and moves
17-29 deg; the best lift is 47% of F0's.

## The candidates

Machine = every motor's electrical power plus the servo's net power. Lead:
the feather's fundamental against the stroke's; -90 deg is a quarter beat
ahead in this sign convention (F0, which follows the servo's reference,
reads -86).

| candidate | lift (weights) | vs F0 | machine W | stroke motor W | feather motor W | feather amp / mean / lead (deg) | beat | stroke motor on limit | peak loop force | max gap |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 servo | 1.927 | | 178.2 | 149.4 | servo 28.8 | 50.1 / -0.1 / -86 | 1.000 | 0 | 296 N | 0.32 mm |
| FM gear 12 | 1.847 | -4.2% | 314.0 | 146.2 | 167.8 | 50.6 / +0.3 / -88 | 1.000 | 0 | 308 N | 0.32 |
| FM gear 16 | 1.862 | -3.4% | **295.5** | 147.3 | 148.2 | 50.5 / +0.3 / -88 | 1.000 | 0 | 311 N | 0.32 |
| FM gear 24 | 1.878 | -2.6% | 307.3 | 148.6 | 158.7 | 50.3 / +0.3 / -87 | 1.000 | 0 | 314 N | 0.33 |
| FM gear 32 | 1.856 | -3.7% | 345.9 | 148.7 | 197.2 (on limit 19%) | 50.2 / +0.7 / -88 | 1.000 | 0 | 317 N | 0.33 |
| FX (as frozen) | does not assemble | | | | | | | | | |
| FX2, stroke gear 16 † | 1.425 | -26.0% | 437.1 | 437.1 | - | 62.1 / -11.9 / -72 | 0.827 | 0.19 | 3186 N | 2.00 |
| FX2, stroke gear 24 † | 1.640 | -14.9% | 302.4 | 302.4 | - | 61.7 / -8.1 / -72 | 0.899 | 0.23 | 3430 N | 2.15 |
| FX2, stroke gear 32 † | 1.674 | -13.1% | 212.3 | 212.3 | - | 61.5 / -7.8 / -78 | 0.870 | 0.89 | 3386 N | 2.13 |
| FP k 0.5 | -2.521 | | 160.7 | 160.7 | - | 64.6 / +3.1 / +170 | 0.999 | 0 | 412 N | 0.43 |
| FP k 1 | -2.001 | | 169.1 | 169.1 | - | 63.9 / +1.8 / -170 | 0.988 | 0.06 | 427 N | 0.44 |
| FP k 2 | -0.429 | | 168.9 | 168.9 | - | 51.8 / -0.5 / -138 | 0.985 | 0.08 | 396 N | 0.42 |
| FP k 4 | 0.911 | -52.7% | 217.5 | 217.5 | - | 29.2 / -0.1 / -100 | 0.977 | 0.13 | 466 N | 0.44 |
| FP k 8 | 0.679 | -64.8% | 250.2 | 250.2 | - | 16.7 / -0.4 / -82 | 0.958 | 0.20 | 445 N | 0.42 |

† fails G1: the loop is not modelled well enough to read (below).

Where FM's power goes: its shaft delivers 26.5-27.5 W (the servo's 28.8 W
less the bit the stroke motor now carries), and in a vacuum it already
draws 27-113 W rising with the gear: copper loss on the reflected rotor
inertia reversing 9 times a second, as for direct drive in D1. The witness
motor was not sized for this load (2.86 N m peak at the hinge, 24.7 rad/s
peak speed); a motor matched to it would cost less, and
nothing here measures how much.

## Against the frozen checks

| | frozen | read | |
|---|---|---|---|
| G1 | in a vacuum, input per beat within 1% of the largest motor's peak shaft power, F0, FM, FX | F0 +0.02%; FM -0.05 to -0.25% (the feather motor regenerates what the stroke motor puts in, 7.0 W each way). FX2 (in FX's place, D2a) **+3.2%** at gear 24, with a 2.1 mm gap and 3.35 kN in the loop | holds for F0, FM; **fails for FX2** |
| G2 | FM within 3 deg; at least 23 W + copper; machine 175-200 W | amplitude 50.2-50.6; feather motor 148-197 W; **machine 295.5 W** at its best gear | first two hold; **the total is wrong** by half again: I under-counted the copper loss of a reversing load |
| G3 | FX: amplitude 40-60, lead 70-110, lift within 10% of F0, power within 15% of FM, limit under 10% | FX as frozen does not close at any rod length (0.5-1.5 times the rest distance). FX2: amplitude **61.5-62.1**, lead 72-78, lift **-13 to -26%**, power +2.3% at gear 24 (-28% at 32, +48% at 16), limit **19-89%**, beat 0.83-0.90 | **FX: no candidate. FX2: unreadable (G1)**; as read, amplitude, lift and limit miss |
| G4 | no spring gives 40 deg with a lead within 30 deg of a quarter beat; best lift under 80% of F0 | k 2: 52 deg at -138; k 4: 29 deg at -100; best lift 47% (k 4) | **holds** |
| G5 | FX's drawing at four crank angles, checked | drawn for FX2 (`figures/D2_fx2.png`): arm 23.0 mm, rod 82.8, horn 30.0 mm from the hinge at every angle, the loop closed to 0.002-0.082 mm (vacuum, held body, before the mass is set: see below) | holds, with one difference from the text: **the horn is 30 mm aft of the hinge, not ahead**; the hinge is 0.1 chord (26 mm) from the leading edge, so 30 mm ahead would be off the wing |

## What happened on the way (disclosed)

- FX2 was added after FX would not assemble (addendum D2a, frozen before any
  grid row). Its first runs locked because the feather shaft turned the
  wrong way (axis sign), seen in smoke runs and fixed before the freeze.
- The grid's first FX2 rows (in `D2_grid.jsonl`) are void: freerig calls
  `mj_setConst` after setting the mass, and MuJoCo then recomputes each
  connect's body2 anchor from the built pose, which put the rod's far end
  back where it was built instead of on the horn (10-line probe:
  `eq_data[3:6]` is overwritten). `init_state` now writes the horn back;
  FX2 was re-run (`D2_fx2.jsonl`) and the reader takes those rows. No other
  candidate has a rod anchored off its built pose.
- The feather crank's design swings the feather 55.6 deg each way, and the
  hinge's own stop is at 57 deg. Run, it goes to 61-62 deg, so the stop is
  in the loop. A disclosed variant with the stop moved to 85 deg locks at
  -125 deg (the loop passes its toggle and changes branch). The geometry
  sits too close to its toggle at the feather's extremes for the frozen
  rule's arm and rod. That is a fact about this geometry, not about feather
  cranks: a longer horn or a smaller arm-to-horn ratio moves the toggle away,
  and nothing here has swept it.

## What this says, as ranges, not a choice

- The feather at take-off is a 27-29 W shaft load with a 2.9 N m peak per
  wing. Every option measured that delivers it pays more than that. The
  servo is the ideal reference. A second witness motor pays 150-200 W of
  copper on top, a sizing question this motor cannot answer. A crank off the
  stroke could not be read in this engine at this geometry.
- A passive feather does not do take-off at any spring tried. At cruise
  (D2 §0) the root feather does no work and holding it is a lock, so this
  verdict is for take-off only.
- Open, not run: a feather crank with its toggle moved away (horn length,
  arm ratio swept), and a motor matched to the feather load. Both are
  needed before FX2 or FM can stand beside the servo as candidates.

## What these numbers rest on

As D1: a witness motor, no gearbox or bearing loss, ideal regeneration,
links without fluid load, the body held, take-off only. The second wing's
feather is coupled to the first's by a joint equality (FM) or by its own
mirrored arm (FX2).
