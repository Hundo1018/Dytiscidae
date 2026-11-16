# R0 result: two machines that flew, through the present pipeline (2026-11-15)

**Erratum (2026-11-16, found on the drawings, `REVIEW_figures_past.md`): the
wings were not built at the reported spans.** The roots sat on the hull's
centre line, so the spans were 1.42 m (E-Flap, 1.50 reported) and 1.25 m
(Robo Raven II, 1.33). Area and mass were as reported. Every figure below is
for those shorter spans until the pass is re-run on the seated fixture.

Pre-registration: `PREREG_R0_reproduce.md` (commit a6407c4, before any
rollout). Data: `results/R0_reproduce.jsonl`, 240 points, 191 trimmed, one
pass, nothing tuned after reading. Command:
`MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/reproduce.py experiments/flapper/results/R0_reproduce.jsonl`.

## Against the frozen text

| | frozen prediction | measured | |
|---|---|---|---|
| P1 | E-Flap sustains at 4 m/s, 3.5-4 Hz, for some Phi of 60-100 deg | yes, at every one of them | held |
| P2 | lowest aerodynamic power among sustaining points at 3.5-4 Hz, 3-5 m/s: predicted 15-35 W; **bar 31-67 W** | **23.2 W** (4 Hz, Phi 40 deg, 5 m/s); 41.7 W is the lowest at 3.5 Hz (Phi 60) | prediction held; **bar failed on the optimistic side**, as predicted |
| P3 | null: E-Flap at 2 Hz, no sustaining point at any Phi up to 100 deg | **3 sustaining points, all at Phi 100 deg**, 29-31 W | **failed** |
| P4 | Robo Raven II sustains at 6.7 m/s, 4 Hz, Phi 57-67 deg, at 6-14 W | yes, at 14.1 W (Phi 57) and 25.3 W (Phi 67) | sustains; the power prediction missed high |
| P5 | null: Robo Raven II at 2 Hz, none | none | held |
| f = 0 | a wing that does not flap never sustains | 0 of 64 | held |

**Verdict by the frozen rule: the pipeline does not reproduce E-Flap** (P2
outside its bar, P3 failed). It is not cleared to rank designs as it is.

## What was read after the run, and what it changes

The pre-registration left one number unread: whether E-Flap's "30-50 deg" is
half or full stroke. The paper was then read (Zufferey et al., RA-L 2021,
text extracted from the PDF): the amplitude "corresponds to the sum of
upstroke and downstroke angles", so **30-50 deg is the full stroke**; the
wing has a flexible trailing edge that pitches passively; and the 520 g
payload flights were at 100% throttle, 5.6 +- 0.2 Hz, "at the limit of
manoeuvrability and climb rate".

With that, the E-Flap rows that describe the real machine are Phi = 40 deg
only:

| frequency | measured in the model, Phi 40 deg, 3-5 m/s | the machine |
|---|---|---|
| 2 Hz | no point trims | not flown there |
| 3.5 Hz | trims, horizontal force -0.07 to -0.67 N: **does not sustain, by a little** | cruises "about 4 Hz" |
| 4 Hz | sustains at all three speeds, 23-24 W, 0.6-0.9 N to spare | cruises |

So on whether it flies, the model puts the boundary between 3.5 and 4 Hz at
the machine's own stroke, which is where the machine cruises. The three
points that broke the null are at a stroke of 100 deg, twice what the linkage
allows; the null as frozen swept strokes the machine does not have. It stays
recorded as failed, and the reason is in the design of the null, not
established to be in the model.

On power the miss stands. 23 W aerodynamic against 89 W electrical (itself an
upper bound) is 0.26. Either the machine's chain from battery to air is that
poor (the paper's own estimate excludes "wing elasticity, camber, power
transmission losses" and its drive must also swing the wing's inertia, which
the rig does not charge), or the model's wing is too efficient: it is rigid
and feathered ideally at every station, where the real wing twists passively.
This run cannot separate the two. A bench power measurement can.

Also seen, not predicted: at a fixed frequency the model's power rises about
as the cube of the stroke (4 Hz: 24, 64, 144, 273 W at 40, 60, 80, 100 deg)
and its thrust far exceeds the need at large strokes (up to 11 N on a 5 N
machine). A design search on this model would be drawn to large strokes only
if thrust were scarce; at small strokes it is close to the real machine.

Robo Raven II, with the wing area typed at 0.25 m2: sustains at 4 Hz and not
at 2 Hz; 14.1-16.5 W at the stroke read from its plunge velocity (57 deg)
against a reported 13.6-17.6 W of unestablished kind. Agreement, on a typed
input.

## Belief before and after

Before: the model flies things too easily and too cheaply. After: on whether
a real machine flies at its real stroke and frequency it agreed on the two
tried (one of them with a typed wing area); on power it reads a quarter of
the one electrical figure there is, cause unseparated. It would move back to
"too easily" if the payload check below, or a bench wing, shows thrust the
real machine does not have.

## Not covered

A held body (no heave, no pitch), no torque limit, rectangular rigid wings at
the reported span and area, no hull or tail drag beyond the fuselage shape,
one pass on a coarse grid (the lowest-power point sits on a grid edge in
feather amplitude: every minimum is at Psi = 30 deg, the largest tried).

## R0b, post hoc (not pre-registered; exploratory)

Run after reading the paper, because it gives a limit: 1.03 kg all-up at
5.6 Hz was the most E-Flap carried. `reproduce_payload.py`,
`results/R0b_payload_posthoc.jsonl`, 128 points at 5.6 Hz, 4-7 m/s.

| all-up mass | Phi 40 deg | Phi 50 deg |
|---|---|---|
| 1.03 kg (the machine's limit) | sustains, 62 W, 1.65 N to spare | sustains, 107 W, 4.0 N to spare |
| 1.3 kg | sustains at one point of 16, 0.05 N to spare | sustains |
| 1.6 kg | none | one point of 16, 0.33 N to spare |
| 2.0 kg | none | none |

The model carries what the machine carried, and its own limit is 1.3 kg at a
40 deg stroke and 1.6 kg at 50 deg: **1.3 to 1.6 times the machine's**. Which
stroke the payload flights used was not read, and they were flown indoors,
where speed may have been the limit (the model's best points are at 7 m/s,
the edge of the grid). So: the model's lift and thrust capacity is at most
1.3-1.6 times the real machine's, in the optimistic direction. Its power at
the machine's limit (62-107 W aerodynamic) sits under the estimated 130 W
mechanical peak and 200 W electrical.

## For L1

The model is used with two corrections carried as bands, not as fits: a
capacity factor of 1/1.3 to 1/1.6 on what a design can lift, and electrical
power at 2.6-4 times the model's aerodynamic power until a bench wing says
otherwise. Both come from one machine.
