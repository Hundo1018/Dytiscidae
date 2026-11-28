# Pass X2, result: an outside solver (OpenFOAM) on the wing's section, static and in motion

2026-11-18. Frozen text: `PREREG_X2_cfd.md`. Run by a worker agent; every
number below was re-printed by me from its stored rows with its own
readers (`x2/compare.py`, `x2/compare_b.py`, `x2/summary.py`), and the
measured values come from the repo's own section files
(`sections.lift_runs("goe417a")`, `sections/goe417a_Re100k.csv`,
`flatplate_uiuc`). Code and rows: `x2/` (`mesh.py`, `case.py`, `run_a.py`,
`run_b.py`, `kinematics.py`, `strip2d.py`, the readers, `results.jsonl`,
`NOTES.md` with the run log). Case directories (2.6 GB) are outside the
repo in `../.cfd/cases/X2`. OpenFOAM v2412, two-dimensional, Re about 1e5.

## Outcome in one paragraph

**The outside solver does not qualify by the frozen rule (A1-A3 with one
closure), so every stage-B number is labelled unqualified.** Its
transition closure (kOmegaSST-LM) matches the measured Goe 417a lift from
2 to 10 deg within 0.05, but not its drag (+36% to +104% from 6 deg), and
it stalls early: maximum lift 1.30 near 9.7 deg against the measured 1.355
at 13.5. With that closure: **at the cruise motion the strip solver's
normal force is within 7.4% of OpenFOAM's, but it puts 0.03 more thrust
and 28% more power on the section (B1 fails, narrowly on both).** At the
6 m/s high-amplitude motion OpenFOAM reads a lift peak of 4.17 on the
downstroke (static maximum 1.355), 47% more mean normal force, and thrust
where the strip solver reads drag. That is the dynamic stall the strip
solver does not model, as was pre-registered (B2, B3 hold).

## Stage A: the static section (mesh L0, 38,041 cells)

| alpha | measured CL | measured CD | SST CL | SST CD | SST-LM CL | SST-LM CD |
|---|---|---|---|---|---|---|
| -2 | -0.160 | .0588 | 0.132 | .0404 | -0.059 | .0504 |
| 0 | 0.195 | .0413 | 0.398 | .0292 | 0.232 | .0381 |
| 2 | 0.539 | .0314 | 0.631 | .0235 | 0.546 | .0315 |
| 4 | 0.813 | .0243 | 0.835 | .0221 | 0.843 | .0253 |
| 6 | 1.003 | .0183 | 1.017 | .0246 | 1.052 | .0248 |
| 8 | 1.154 | .0188 | 1.172 | .0302 | 1.169* | .0320* |
| 10 | 1.267 | .0302 | 1.283 | .0405 | 1.300* | .0615* |
| 12 | 1.341 | - | 1.293* | .0640* | 1.069* | .1176* |
| 14 | 1.321 | - | 0.980* | .1660* | 0.980* | .1698* |
| 16 | 1.177 | - | 1.225* | .3002* | 1.111* | .2731* |
| 20 | 1.196 | - | 1.634* | .6145* | 1.290* | .4254* |

\* no steady solution: the mean of a time-accurate run over a window cut
short by the 90-minute cap (the planned window was t 20-40; the windows
used start at 4-19). The range of lift inside each window is in
`summary.py`'s print: wide at 10, 12, 16 and 20 deg (e.g. SST-LM 10 deg:
1.16-1.44).

| | frozen | read | |
|---|---|---|---|
| A1 | mesh: lift at 4 and 12 deg within 3%, drag within 8% on the finer mesh | 4 deg holds (lift 0.2%, drag 5.5%); 12 deg: SST lift 3.5%, drag 14%; SST-LM lift 26%, drag 31% | **fails for both**. The 12 deg windows are short and inside the coarse mesh's own oscillation, so this is not a clean read of mesh dependence; the fall-back (whole grid on the fine mesh, 8-10 h) was not run |
| A2 | SST-LM, 2-10 deg: lift within 0.10, drag within 40% | lift worst 0.050; drag +36% (6), +70% (8), +104% (10) | lift holds, **drag fails** |
| A3 | max lift within 0.15 of 1.355, angle within 2 deg of 13.5 | SST-LM 1.30 at 9.7 deg; SST 1.63 at 20 deg | **fails for both** |
| A4 | 0 deg either closure off by up to 0.2 (gates nothing) | SST-LM +0.036; SST +0.202 | as read |
| A5 | SST misses drag (too high below 6 deg), holds lift | lift worst 0.092; drag **too low** below 6 deg (-31% to -9%) | **wrong in direction** |

Flat plate (3.1%, semicircular edges, an assumed edge shape; gates
nothing): at 5 deg SST-LM 0.558, SST 0.574, 2 pi alpha 0.548, the tunnel
0.602, as predicted (at or near 2 pi alpha, not the tunnel's). From 15 deg
the mean lift is 1.07-1.20, inside the predicted 0.9-1.3.

## Stage B: the section in motion (SST-LM, L0; all unqualified)

The section at 0.462 m span, chord 0.2576 m; plunge `h0 sin wt` and pitch
`theta0 + A cos wt` about the quarter chord, derived from the cruise and
6 m/s motions; five beats, the last three averaged (beat means settle to
within 0.5% by beats 3-5). Moving-mesh check: added mass in still fluid
0.8131 against pi c^2/4 = 0.7854 (1.035).

| station | quantity | strip solver | OpenFOAM |
|---|---|---|---|
| B8 (8 m/s, 2.5 Hz, h0/c 0.63, k 0.25, pitch 2.96 + 10.5 cos) | mean normal force coeff. | 0.644 | 0.692 |
| | mean streamwise (+ thrust) | 0.072 | 0.042 |
| | mean power coeff. | 0.108 | 0.084 |
| | peak normal force, downstroke | 1.411 | 1.339 |
| B6 (6 m/s, 3.5 Hz, h0/c 0.94, k 0.47, pitch 11.2 + 28.0 cos) | mean normal force coeff. | 1.252 | 1.841 |
| | mean streamwise (+ thrust) | -0.149 | +0.184 |
| | mean power coeff. | 0.241 | 0.852 |
| | peak normal force, downstroke | 3.000 | 4.166 |

OpenFOAM's mean pitching moment is -0.094 (B8) and -0.269 (B6); the strip
solver's is about 0.

| | frozen | read | |
|---|---|---|---|
| B1 | B8: normal force within 15%; streamwise within 0.03; power within 20% | +7.4%; 0.0305; -22% | **fails, narrowly on two of three** |
| B2 | B6: OpenFOAM's normal force higher by over 15%, power higher | +47%; 3.5x | holds |
| B3 | B6: a lift peak above the static 1.355 on the downstroke | 4.17 at phase 0.55 (4.97 with the plunge's added mass taken out: not added mass) | holds |

## What this says, as ranges

- Where the section's flow stays attached (2-10 deg static), the measured
  table the strip solver reads (repair R3), and OpenFOAM's transition
  closure agree on lift to 0.05. Neither OpenFOAM closure here can judge
  stall or drag on this section at this Reynolds number.
- At cruise motion, OpenFOAM and the strip solver agree on the normal
  force (lift) within 7.4% and disagree on what pays for it: the strip
  solver reads more thrust (0.072 against 0.042) and more power (+28%) on
  this section. That is one 2-D section, unqualified, against a strip of
  a 3-D wing; it is a direction to look, not a correction.
- At the 6 m/s, 28 deg pitch motion the flow separates and reattaches each
  beat in OpenFOAM, and the forces it reads there are 1.5-3.5 times the
  strip solver's. Anything the project has read at low speed and large
  feather (take-off, the 6 m/s point) rests on a model without dynamic
  stall, as `docs/model_validity.md` says; this run measures how far that
  can be off for one section.

## Changes after launch, disclosed (each made before any result it could affect)

- Boundary-layer thickness 0.03 to 0.02 c (Goe) and 0.025 c (plate): gmsh
  left a degenerate cell otherwise. Trailing edge opened to a blunt base
  0.25% c thick.
- Convergence mechanics: steady runs stop when the rule holds; a run stopped
  early by residualControl is judged on its last half. A bug had sent SST
  6/8/10 deg to the time-accurate path; read steady now, and they agree
  with those runs to 0.5%.
- Stage B Courant limit 4 to 8, and its time cap 90 min to 4 h (over the
  frozen 90-min rule), for cost.
- The packaged OpenFOAM loaded a dummy parallel library, so parallel runs
  failed (the two fine-mesh 12 deg cases, re-run); the MPICH library is now
  preloaded for the solver only.
- The strip solver's 2-D aspect ratio was forced for the comparison (its
  default 3.9 gave CL 0.565 at 4 deg; forced, it matches the table to
  0.005). The section station is at 0.462 m from the shoulder (the code's
  wing is 0.66 m from a shoulder on the hull side, not the brief's 0.70 m).

Wall time: stage A about 9.4 h (predicted 2-4 h), stage B about 2.4 h.
`results.jsonl` keeps superseded rows; the readers take the last row per
case, as NOTES says.
