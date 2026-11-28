# Repair R1, result: the default law's stall and camber

2026-11-17. Frozen text: `PREREG_R1_stall.md` (`daeb24b`). Checks:
`r1_check.py`.

## Outcome

The frozen replacement **is not the default**. It cut the error against
Ananda's flat-plate wings from 0.165 to 0.068 rms and the worst point from
0.48 to 0.15, but it fails five of its own checks, and one of them
contradicts it with primary data: the UIUC table of Goe 417a already in this
repository reaches CL 1.37 at 12 deg at Re 6e4-2e5, where the fit caps a
5.8% thin plate at 1.10. By the rule frozen in section 3 it is not shipped.
It is in the solver as an opt-in (`FluidSolver.stall_law = "plate"`, numpy
path only); `"legacy"` remains the default, bit for bit, and the two GPU
mirrors are untouched. **The default law is still wrong for a flat wing.**

## Against the frozen checks

| | frozen | read | |
|---|---|---|---|
| V1 | Ananda's wings: rms under 0.08, every point within 0.15 | rms 0.068; worst 0.152 (AR 2 at 20 deg: 0.60 against 0.75) | rms holds, **the worst point misses by 0.002** |
| V2 | 2D flat plate at 10, 15, 19 deg within 0.10 of 0.60-0.76; no drop over 0.06 after the maximum | 0.65, 0.66, 0.66; a bump to 0.73 at 7 deg and a dip to 0.62 | level holds, **the bump fails** (0.11) |
| V3 | 2D 5% arc, Re 2e4: CL(0) 0.15-0.25; maximum 0.95-1.10 at 8-12 deg; 0.82-0.92 at 14-18 | 0.19; 1.106 at 9 deg; 0.885 | **maximum 0.006 over** |
| V4 | AR 6, 5% arc: maximum 0.90-1.05 | 1.107 at 12 deg | **fails**: the law gives a finite wing the 2D maximum; the data's is 0.1 lower |
| V5 | against Goe 417a's own table at Re 1e5, within 0.15 | off by up to 0.51: the table reads 1.29-1.37 at 10-14 deg, the law 0.86 | **fails** |
| V6 | the battery unchanged | not re-run: the default is the legacy law to the bit (the full physics suite passes unchanged) | holds by construction |

What I had wrong: I took the gathered table's cambered rows as settled.
They are second-hand readings of replotted figures, and they disagree with
the one cambered thin section whose measured table is in the repository.
The flat-plate half of the fit is supported by every source, including
Ananda's wings, which were not used to make it. The cambered half is not.

## What is in the code

- `fluid.plate_lift_limit`, `camber_lift_factor`, `_stall(..., law=)`,
  `FluidSolver.stall_law` ("legacy" default). The rotor model and every
  stored score are on "legacy".
- `tests/test_physics.py::test_the_plate_stall_law_is_opt_in_and_reads_the_thin_plate_table`
  (8 checks) and three mutations (`plate-stall-*`), all caught.
- Found on the way and fixed before commit: writing the separated drag as
  lift times tan(alpha) lost a broadside plate's drag at exactly 90 deg (15
  physics checks failed); and two existing checks ("the leak is there")
  flip when the legacy law's arithmetic changes in the last bit, so the
  legacy branch keeps its original expressions. Those two checks are
  trajectory-sensitive at the 1e-16 level and should not be.
- A mutation showed my own camber-moment test could not tell a reversed
  flow's lift direction; a check was added.

## What a second attempt needs

1. The flat plate as here, without the bump (a monotone handover).
2. The cambered limit from primary tables: the UIUC thin sections already in
   `flapper/sections/` (Goe 417a, BW-3, CR-001) give CLmax and the lift past
   it at Re 6e4-2e5 directly.
3. A finite wing's maximum below the 2D one (V4, and Ananda's AR 4-5 at 10
   deg read 0.06-0.08 high).
4. Then the two mirrors and the comparability boundary.

## Addendum, same day: the primary tables, read before a second attempt

UIUC two-dimensional lift runs already in `flapper/sections/` (read by
`sections.lift_runs`; geometry from the stored coordinates):

| section | camber, thickness | Re | zero-lift | slope, per deg (-2..4 deg) | CL at 6, 8, 10, 12, 14, 16, 18 deg | maximum |
|---|---|---|---|---|---|---|
| flat plate | 0, 3.1% | 6e4 | 0.1 | 0.124 | 0.73 0.80 0.79 0.82 0.84 0.89 0.99 | still rising: 1.18 at 23 |
| flat plate | | 1e5 | 0.3 | 0.124 | 0.69 0.78 0.79 0.77 0.80 0.84 0.87 | 1.21 at 24 |
| flat plate | | 1.2e5 | 0.3 | 0.118 | 0.71 0.81 0.81 0.80 0.80 0.87 0.94 | 1.24 at 24 |
| Goe 417a | 5.8%, 3.4% | 6e4 | -0.3 | 0.157 | 0.98 1.15 1.29 1.35 1.36 1.24 1.24 | 1.40 at 13.6 |
| Goe 417a | | 1e5 | -1.1 | 0.171 | 1.00 1.15 1.27 1.34 1.32 1.18 1.19 | 1.35 at 13.5 |
| Goe 417a | | 2e5 | -2.4 | 0.152 | 1.07 1.21 1.31 1.37 1.29 1.22 1.18 | 1.37 at 12.4 |
| BW-3 | 5.6%, 5.0% | 6e4-2e5 | -1.4 to -3.0 | 0.136-0.168 | 1.00-1.07 at 6; 1.30-1.41 at 12; 1.16-1.20 at 16-18 | 1.32-1.41 at 11.8-12.9 |
| CR-001 | 4.0%, 7.3% | 6e4-2e5 | -2.7 to -4.2 | 0.111-0.137 | 1.01-1.05 at 6; 1.19-1.28 at 12; 1.08-1.14 at 16-18 | 1.27-1.30 at 11.3 |

Against the second-hand table R1 was fitted to:

- **Flat plate**: bends at 7-8 deg at 0.78-0.82 and holds 0.77-0.84 to 14
  deg (R1: 0.66-0.70), then *rises* to 1.2 by 24 deg, where the separated
  plate law reads 0.74 and Ananda's finite flat wings 0.70-0.81. McArthur's
  two-dimensional plates at Re 1e4-2e4 hold 0.60-0.66. So the plateau grows
  with Reynolds number from about 0.62 to about 0.80, a finite wing sits
  about 0.1 under the two-dimensional value, and the two-dimensional data
  past 15 deg disagree with the finite-wing data by more than either's
  error.
- **Cambered, 4-6%**: maximum 1.27-1.41 at 11-14 deg and 1.1-1.25 after it
  (R1: 1.02-1.10 and 0.84-0.89). Pelletier & Mueller's 4% plate at Re 6e4
  reads 1.07 (second-hand). The UIUC models are 3.4-7.3% thick with rounded
  leading edges; the plates of the other set are 1.9-3.6% thick. The
  maximum lift of a thin cambered wing at these Reynolds numbers depends on
  how its leading edge is made by about 0.3.
- **Camber lift** is not a shift of a 2 pi line: the zero-lift angle is -0.3
  to -2.4 deg for a 5.8% section (theory -6.6) and the slope near zero is
  0.15-0.17 per deg (2 pi is 0.110); by 6 deg the lift is 1.0, which a 2 pi
  line reaches with 0.47 of theory's shift.

What follows: there is no one default law to fit. A strip's lift past 6-8
deg depends on its section's construction, which the genome does not carry.
The honest default is a *named* measured section (a table, as the flapper
rigs already use), chosen by how the wing will be built; the choice and the
comparability boundary it brings are the user's. `stall_law="plate"` stays
as R1 left it, labelled as failing its checks.
