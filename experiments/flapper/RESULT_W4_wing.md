# Pass W4 result: the wing's planform, twist and section (2026-11-17)

Pre-registration: `PREREG_W4_wing.md`. Data: `results/W4_wing.jsonl`, 2,360
rows, all finite, one pass. Reader: `w4_read.py`. Held rig, seated fixture,
hinge at 0.1 chord, cam offset trimmed to the weight at every point.

## The drawings against the design

`figures/W4_taper1.0.png`, `W4_taper0.6.png`, `W4_taper0.4.png`, each with the
section's profile. Checked by eye and by a numerical read of the built model:
span 1.40 m and area 0.340 m2 for every taper, a straight leading edge, root
and tip chords 0.368 and 0.147 m at taper 0.4, 8 deg of washout at both tips
the same way, the section's upper side up on both wings (equal lift left and
right), thickness 3.4% and height 7.3% of chord for Goe 417a. The pre-registration says all three were opened
before freezing; one was, the other two just after the sweep started. They
show nothing wrong.

## Results

Cheapest sustaining cruise within the crank cap (number of sustaining points
of 80), and take-off gate points of 100:

| section | tip/root chord | twist | 6 m/s | 8 m/s | take-off |
|---|---|---|---|---|---|
| default law, flat | 1.0 | 0 | none (0.05 N short at 30 W) | 28.3 W (2) | not run |
| Goe 417a | 1.0 | 0 | 30.1 W (1) | 20.9 W (4) | 6; 1.88 weights |
| **Goe 417a** | **1.0** | **-8 deg** | none (0.11 N short) | **9.7 W (8)** | **0; 1.45 weights** |
| Goe 417a | 0.6 | 0 | none | 28.9 W (1) | 4; 1.73 |
| Goe 417a | 0.6 | -8 | none | 29.5 W (1) | 1; 1.68 |
| Goe 417a | 0.4 | 0 | none | none (0.12 N short) | 3; 1.66 |
| Goe 417a | 0.4 | -8 | none | none (0.06 N short at 7.6 W) | 0; 1.58 |
| E387 | 1.0 | 0 | none | 27.8 W (2) | not run |
| E387 | 1.0 | -8 | none | 22.3 W (3) | not run |
| E387 | 0.6 | 0 / -8 | none | none / 29.0 W (1) | not run |

The washout rectangle's sustaining points at 8 m/s: 9.7 W (2 Hz, stroke 50,
feather 5, crank 2.7 N m), 10.2 W, 11.2 W, then 21.5 W and up. **The three
cheapest sustain by 0.00-0.03 N**; the first with 0.1 N in hand costs 21.5 W.

## Against the frozen text

- **P1 refuted.** The untwisted Goe 417a rectangle needs 20.9 W (15
  predicted).
- **P2 confirmed for the rectangle** (20.9 to 9.7 W, -54%), not for taper
  0.6 (28.9 to 29.5 W).
- **P3: one half refuted narrowly.** Taper 0.4 meets the take-off gate at 3
  points against the rectangle's 6: half, not fewer than half. Taper 0.6
  meets it at 4.
- **P4 refuted.** The only shape that cruises for 12 W or less meets the
  take-off gate nowhere: its most lift under the caps is 1.45 weights.
- **P5 refuted.** At 6 m/s one point of 880 sustains, at 30.1 W.
- **P6 refuted.** E387 costs 22.3 W where Goe 417a costs 9.7 (washout
  rectangle) and 27.8 against 20.9 (no twist): far outside what three models
  of one section differ by.
- Control: the default law on the flat rectangle, same grid: 28.3 W. So the
  finer grid alone buys nothing; the section and the twist do.

## Reading

- Rival (b) holds: twist opens cheap cruise, and only with a section that
  has the lift-to-drag to use it. A curved thin plate with 8 deg of washout
  cruises at 8 m/s for about a third of the flat rectangle's power in this
  model, on a slow stroke (2-2.5 Hz, crank under 3 N m of 12).
- **The same fixed washout loses the take-off** (1.88 to 1.45 weights under
  the caps), and rival (c) holds for taper as well: it costs take-off and, in
  this model, buys no cruise.
- So one rigid wing shape does not serve both. Cruise wants the tip washed
  out; take-off wants it not. With pass 1i's finding that the feather
  amplitude also has to differ between the two, the wing needs a twist along
  its span that changes with the flight regime: a tip that can be turned
  against the root (a wrist, as pass F1a built, or a wing that twists under
  load as real flapping wings do).
- **Decision: the reference wing becomes a rectangle of Goe 417a section
  with a spanwise twist that is a control, not a constant.** Taper is
  dropped on this evidence. Next: the twist as a second feather input at the
  tip, with take-off and cruise each given their own setting.

## What this does not establish

- The 9.7 W has no margin (0.02 N), and no tail, trim drag or released body
  is in it. With 0.1 N in hand the same wing needs 21.5 W.
- 6 m/s: nothing sustains cheaply. The grid stops at stroke 60 deg and
  3.5 Hz and at 8 deg of twist; more twist or a lower wing loading is not
  tried.
- Everything pass W3 lists: steady tables used under flapping, the
  leading-edge-vortex rule taken from insect-scale plates, lift read low
  just past stall, no finite-wing check. The ranking of sections rests on
  the attached part of the stroke being attached.
- Taper's loss may be the model's: strips carry no tip loss beyond the one
  inflow for the whole span, so a rectangle's loaded tip costs nothing extra
  here. A real rectangle pays for it.
- Structure: a 3.4% thick curved plate 0.66 m long has to carry the flapping
  loads; its spar and mass are not designed.
