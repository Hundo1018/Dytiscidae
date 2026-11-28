# Pass W3 result: measured wing sections in the fluid model (2026-11-17)

Pre-registration: `PREREG_W3_section.md` (commit 40b767c). Data:
`sections/` (the measured polars), `results/W3_section.jsonl` (2,328 rows,
all finite, one pass). Readers: `w3_section.py check`, `w3_read.py`.
Mechanism: pass 1n's seated fixture, still a rectangle, with a section
(`figures/W4_taper1.0.png` shows it with the section's profile).

## The check against the separate lift runs (no simulation)

- **P1 refuted at 1 of 26.** Inside the drag runs' range the model's lift is
  within an rms of 0.003-0.045 of the lift runs, except SD7037 at Re 61k:
  0.061 (bar 0.06).
- **P2 confirmed.** In the 6 deg past the top of each table the model reads
  low by 0.12-0.41 (five of six sections by more than 0.2), and by 0.33-0.65
  beyond that. A wing just past stall keeps much more lift than the solver's
  separated flat plate gives it. The default law has the same branch, so it
  has the same fault. Not repaired in this pass.
- **P3 refuted narrowly.** The three E387 models differ in best L/D by 9% at
  Re 1e5 (51.5, 53.4, 56.1) and 18% at 6e4. That is the floor under any
  ranking of sections here.

## The sweep (pass 1i's air grid, seated rectangle)

| | sustaining points | meet the cruise gate | cheapest sustaining |
|---|---|---|---|
| default law (flat) | 128 | 28 | 26.9 W |
| Goe 417a | 142 | 32 | 27.1 W |
| SD7003 | 114 | 16 | 27.8 W |
| E387 | 124 | 24 | 26.6 W |

- **P4 refuted.** I predicted 32-48 W and 2-18 gate points for Goe 417a; it
  is 27.1 W and 32. **P5 refuted**: E387 is 0.5 W cheaper, inside the 9%
  floor. **P6 confirmed trivially**: take-off is identical to four digits.
- Where a point sustains under both laws its power is the same (median
  ratio 1.00, range 0.98-1.00).

## Why a section changes so little on this grid (`probe_lev.py`, probes)

- At take-off 99% of the wing's dynamic pressure is on strips the solver has
  handed to its leading-edge-vortex plate (93% of it above 25 deg of
  incidence). No section is in play. That is the regime the law was fitted
  for.
- At the cheapest cruise point, half is on the vortex plate and 35% is above
  12 deg. The grid's sustaining points flap hard: about 27 W of aerodynamic
  power to fly 0.6 kg at 8 m/s, where the same wing gliding would need a few
  watts of thrust power.
- At the grid's gentlest strokes (2.5 Hz, stroke 40 deg) the section does
  matter and nothing sustains: at 8 m/s every law is 0.10-0.21 N short at
  5-7 W; at 6 m/s Goe 417a is 0.72 N short, the default 0.96, E387 1.01,
  SD7003 1.24.

## Reading

- The model can now tell sections apart, and on pass 1i's grid it barely
  matters, because that grid found flight only by flapping in the separated
  regime. The gates of passes 1i-1n are the gates of that regime.
- The cheap flight a designed wing should give lies between the grid's
  gentle points (a tenth of a newton short at 5-7 W) and its sustaining ones
  (27 W). Nothing has searched there. That is pass W4, with the planform and
  twist.
- By the decision rule Goe 417a (the curved plate) goes forward: it has the
  most sustaining points and the smallest shortfall at 6 m/s, and it is the
  one a hobbyist can make from sheet. E387 is carried as the comparison.

## What this does not establish

- Any section under flapping: the tables are steady. The rule that a
  leading-edge vortex takes a strip to the plate law at any incidence comes
  from insect-scale plates (Re 1e2-1e4); whether a cambered section at Re
  1e5 flapping at these rates is attached or not is not known here, and half
  the cruise load turns on it.
- A flat plate's measured drag (no table found), so the flat rectangle of
  every earlier pass has no measured counterpart.
- A finite wing: no tabulated wing data found to check the strip-plus-inflow
  chain against.
