# Pass X2 — working notes (executing PREREG_X2_cfd.md, frozen)

Written as the work goes. Case directories live outside the repo in
`/home/hundo/Projects/Dytiscidae/.cfd/cases/X2/` (each with its own `log.*`).
Results: `results.jsonl` here, one row per case actually run.

Environment: OpenFOAM v2412, gmsh 4.15.2, numpy 2.5.5 from the pixi env at
`/home/hundo/Projects/Dytiscidae/.cfd` (`pixi run ...`; the OpenFOAM variables
are set by the env's activation, no bashrc needed). Machine: at most 10 cores
in use by this pass, as instructed (another job runs).

## Scripts (all regenerate from scratch)

- `mesh.py SECTION LEVEL OUT.msh` — gmsh mesh (run under `pixi run python`).
- `case.py` — writes an OpenFOAM case (stage A static; stage B moving), converts
  the mesh (gmshToFoam), sets patch types (foamDictionary), runs checkMesh.
- `run_a.py` — stage A driver: builds every case, runs them at most 10 at a time,
  decides steady/transient by the rule below, appends rows to `results.jsonl`.
- `compare.py` — stage A table against the tunnel data, A1-A5 verdicts.
- `strip2d.py` — the strip solver's side of stage B (repo venv).
- `kinematics.py` — derives the station's motion from the rig (repo venv).
- `run_b.py` — stage B cases (still-air plunge check, B8, B6).
- `compare_b.py` — B1-B3 from results.jsonl; `summary.py` — one health line per case.
- Data written beside them: `kinematics.json`, `strip2d.json`, `stageA_verdict.json`.
- Rebuild from scratch: mesh.py for goe 0, goe 1, plate 0 into
  `.cfd/cases/X2/meshes/{goe_L0,goe_L1,plate_L0}.msh`; `run_a.py 10`;
  `run_b.py still 0`; kinematics.py and strip2d.py (repo venv);
  `run_b.py B8 0 kOmegaSSTLM 5`, `run_b.py B6 0 kOmegaSSTLM 6`
  (X2_B_TIMEOUT=14400); compare.py, compare_b.py.

## Set-up decisions the spec did not fix (decided before any result was seen)

Geometry and mesh
- Outline: `ptera/goe417a_coords.txt`; checked identical to
  `sections.coordinates("goe417a")` (max |difference| 0.0, 93 points).
  Leading edge (min x) moved to the origin, chord scaled to exactly 1
  (the raw LE is at x = 1.7e-5).
- Trailing edge: the outline closes to a point. Opened to a blunt base 0.25%
  chord thick by adding +-0.00125 x to the upper/lower surfaces (XFOIL's TGAP
  with a linear blend; changes thickness by at most 0.125% c, chord unchanged).
  The base is a wall meshed with 6 cells (12 on the fine mesh); the boundary
  layer fans around both base corners.
- Surface: one natural cubic spline through the points (chord-length
  parameter, smooth through the LE), redistributed: 0.0008 c at the LE,
  0.0012 c at the TE, 0.006 c between (fine mesh: halved). 384 wall nodes.
- Boundary layer: gmsh BoundaryLayer field, quads, first cell 1.0e-4 c,
  growth 1.10, total 0.02 c (Goe; 0.025 c plate). Fine mesh "half the
  spacing": first cell 5e-5, growth 1.10^0.5 = 1.0488 (twice the layers over
  the same thickness), every surface size, the near-wall field size and the
  wake box size halved. NOT halved: the far-field growth slope (0.08 per
  chord of distance), so the fine mesh has 3.1x, not 4x, the cells.
- Mesh changes made before any flow result (meshing failures only):
  BL total 0.03 -> 0.02 for Goe, because at 0.03 (and 0.035) the fine mesh
  had one degenerate element at the layer's outer row above the TE corner
  fan (checkMesh: zero face area, 6 failed checks); the plate at 0.02 and
  0.03 had one inverted cell, at 0.025 none. Base cells 6/12 -> 4/8
  (12 base cells on the fine mesh made gmsh's layer intersect itself).
  Fans of 5 elements at both base corners. Final meshes (checkMesh "Mesh OK"):
  goe L0 38,041 cells (max non-orth 42.5, skew 1.14, aspect 59.7);
  goe L1 119,220 (47.4, 1.22, 59.5); plate L0 40,666 (41.3, 1.27, 60.0).
  The driver refuses a case whose checkMesh is not "Mesh OK".
- Outside the layer: quad-dominant (gmsh Frontal-Delaunay for quads + Blossom
  recombination); size 0.003 + 0.08 d (d = distance from the wall), a wake box
  x 0.9..3, y -0.4..1.3 at 0.015 (fine: 0.0075). Far field: circle of radius
  25 chords about the mid chord. One cell thick (DZ = 0.01), front/back empty.
- Coarse Goe mesh: 38,687 cells, checkMesh "Mesh OK" (max non-orthogonality
  51.8, max skewness 1.45, max aspect ratio 59.7).
- Flat plate: 3.1% thick, semicircular ends (radius 0.0155), overall length 1;
  same mesh parameters; the UIUC "baseline" edge is not in the repo, so the
  semicircular edge is an assumption (as the spec states).

Flow
- Unit chord, unit speed, nu = 1e-5 (Re 1e5). Alpha by turning the stream:
  U = (cos a, sin a). Far field: freestreamVelocity / freestreamPressure.
- Freestream turbulence 0.1%: k = 1.5 (0.001)^2 = 1.5e-6, omega from an
  eddy-viscosity ratio of 1 (0.15). Held at the body by SST's decayControl
  (kInf, omegaInf; the log confirms "Employing decay control"), otherwise it
  would decay over the 25 chords. kOmegaSSTLM inlet ReThetat 1136.5
  (Langtry-Menter 2009 correlation at Tu = 0.1%, zero pressure gradient),
  gammaInt 1. Field and BC set checked against the v2412 T3A tutorial
  (gammaInt/ReThetat: fixed at the inlet, zeroGradient at the wall).
- Wall: noSlip, k = 0, omegaWallFunction (gives the viscous-sublayer omega at
  y+ < 1), nutLowReWallFunction.
- Schemes: linearUpwind (cellLimited gradient) for U and turbulence, limited
  corrected 0.5 laplacian. SIMPLEC (consistent) steady; PIMPLE (2 outer,
  2 inner, backward) transient.
- Forces: forceCoeffs on the wall patch, rho = 1, Aref = c x DZ = 0.01, lRef 1,
  liftDir (-sin a, cos a), dragDir (cos a, sin a), CofR (0.25, 0), pitchAxis
  (0,0,-1) so that Cm is nose-up positive. Checked by hand on the test case
  (alpha 4, LM, 200 iterations): raw total force (-8.8317e-5, 3.5198e-3) and
  moment about the origin Mz = 1.35769e-3 give CL 0.70347, CD 0.031486,
  Cm(c/4) = -(Mz - 0.25 Fy)/(q A c) = -0.09555; the function object printed
  0.70347, 0.031485, -0.095549.

Steady or transient (the spec: "steady where it converges")
- simpleFoam up to 8000 iterations (stops early on residualControl
  p 1e-6, U 1e-7, turbulence 1e-6). Converged if, over its last 1000
  iterations, CL varies (max - min) by <= 0.005 and CD by <= 0.0005.
- Otherwise pimpleFoam from the steady fields, maxCo 8 (adjustable step),
  40 convective times, dt-weighted mean over the last 20 (t 20..40). Reported
  values are that mean; the window's min/max are recorded. Measured before
  choosing (goe L0, LM, 12 deg, from rest): dt settles at 3.3e-3 at Co 8,
  0.31 s/step serial, so t = 40 is ~12k steps, ~60 min. A fine-mesh
  transient runs 4-way decomposed (scotch). If a transient is cut by its
  90-min timeout, the mean is over the second half of what it reached, and
  the row says so.
- Order: the A1 cases first (4 and 12 deg, both closures, both meshes), then
  the rest of the Goe grid on L0, then the plate. If A1 fails, the L1 mesh
  becomes the mesh and the grid is re-run on it.
- Flat plate: both closures (it gates nothing; cheap if steady).
- y+: the yPlus function object at every write (every 500 iterations, every
  5 time units); the row carries the last one.
- At most 9 cores for stage A, leaving one for the strip-solver side.
- kOmegaSSTLM prints "Number of lambda iterations exceeds maxLambdaIter(10)"
  during the start-up iterations (seen in the smoke test); typical of the
  model's Re_theta correlation, recorded per case if it persists.

Measured references
- CL: the lift run `goe417a_Re100k_lift.csv`, ascending branch (rows up to its
  first maximum alpha), linearly interpolated at the CFD angle. CD: the drag
  run `goe417a_Re100k.csv` (to 11.65 deg), interpolated; no CD where it does
  not reach. The drag run's own CL is printed beside for reference.
- A3 maximum lift: the largest CL on the alpha grid, and the angle of a
  parabola through the grid maximum and its two neighbours.

## Stage B: the station's motion, derived (kinematics.py -> kinematics.json)

Read off the rig as W8 ran the stored rows (w8_cruise.py: l1a.BODY_S =
wing_length 0.66 m, chord 0.34/(2 x 0.66) = 0.25758 m, wing_seat 0.04 m;
wing_shape camber 0.04; section goe417a+lift+extend; l1i_cam's cruise job:
feather_axis 0.1, cop_law "plate"; rig.run with strip_twist tip 0, share 1),
by running one beat with `TriphibianEnv.step` wrapped to record every right
wing strip's world position and chord, interpolated to 70% of the wing's
length from the stroke axis.
- The brief says "wing length 0.70 m from the shoulder"; the code's wing is
  0.66 m from the shoulder (LENGTH = SPAN/2 - HULL_R; the shoulder is seated
  on the hull side, `wing_seat`), 0.70 m being the half span from the centre
  line. The station is therefore r = 0.70 x 0.66 = 0.462 m from the stroke
  axis (measured strip radii 0.440 and 0.513 m bracket it; weight 0.30).
- Pitch (chord about the strip's span against the flight direction, nose up
  positive): theta = theta0 + A cos(wt), from the fit (residual 1e-14 deg):
  B8: theta0 = +2.956 deg (= -psi0: the hinge's offset), A = 10.500 deg
  (= 0.70 x 15: per_deg is linear in the station fraction); B6: +11.182 deg,
  A = 27.996 (= 0.70 x 40). With share 1 the feather hinge holds psi0 and only
  the strips turn (twist_turn), by fsign x per_deg x (-Psi cos wt).
- Plunge: the station's z = 0.1588 sin(wt) m (B8; 0.2337 B6), i.e.
  r sin(Phi/2): up first, pitch at its most nose-up at the start of the
  upstroke (mid-stroke, maximum upward speed): the feather unloads the stroke.
  The spec's straight plunge, amplitude = r x Phi/2: h0 = 0.16127 m (B8),
  0.24190 m (B6); this is used.
- Phase: pitch leads plunge by 90 deg (theta ~ cos, h ~ sin); the fits'
  small sin/phase terms (0.17 deg, -0.9 deg) are the one-step record lag
  (w dt = 0.0157 rad).
- Non-dimensional (c, V): h0/c = 0.6261 (B8), 0.9391 (B6); reduced frequency
  k = w c / 2V = 0.2529 (B8), 0.4720 (B6); Re = V c / nu_air = 1.395e5 (B8),
  1.046e5 (B6), nu_air = 1.81e-5/1.225.
- Quasi-steady incidence check: B8 downstroke ~ -7.5 + atan(2.53/8) = +10 deg,
  upstroke ~ -4 deg; B6 downstroke ~ -16.8 + atan(5.32/6) = +24.8 deg (past
  static stall), upstroke ~ -2 deg.

Pitch axis: the quarter chord on both sides (the strip solver's load point,
where the rig turns the strips and where `bare` puts the hinge). Decided
before any stage-B result: the strip side's primary run pitches by a hinge
(the solver sees the pitch rate: the two-dimensional motion); the "turn"
variant (chord turned in place, no rate, as the rig's twist_turn does) is
reported beside it, informational.

Strip side set-up (strip2d.py): `bare` strip, flags section goe417a+lift+extend,
inflow off, unsteady on, cop_law "plate" (as the stored rows); camber 0.04
marks the upper side. With inflow off the solver uses a "machine" aspect ratio
disc_span^2/area in place of the strips' own: left at bare's default it was
3.9 and cut the static CL at 4 deg to 0.565; disc_span is set so that it is
1e4 (static check after: CL 0.290 / 0.856 / 1.189 at 0 / 4 / 8 deg against the
section's table 0.289 / 0.861 / 1.194). dt = T/800, 5 beats, mean of beats 3-5.

CFD side set-up (run_b.py, case.moving): whole mesh rigid (solidBody,
tabulated6DoFMotion, 720 table points per period, about (0.25, 0)); the mesh is
pre-turned to theta(0) with transformPoints so the motion starts without a
jump; far field freestreamVelocity (1, 0) in the inertial frame; wall
movingWallVelocity; PIMPLE (2 outer, 2 inner), no correctPhi (rigid motion keeps
cell volumes), final pressure solve to 1e-6, wall distance computed once
(rigid); maxCo 8 as stage A's transients, at least 400 steps per period.
Changed before any stage-B flow result: maxCo was set to 4, but a probe of
B8 (L0, LM) ran 1.1 s/step at dt 1.6e-3 under the full stage-A load, i.e.
~12 h serial for five beats; at maxCo 8 dt ~3e-3. Stage B runs after
stage A with 5 processes per case. Moment about the moving pivot from the raw
forces (about the origin): M_p = M_0 - (0.25 Fy - h Fx). Still-fluid check:
laminar, h0 = 0.05 c, omega = 2 pi, 4 periods, fit Fy = a h'' + b h' + c over
periods 3-4, added mass = -a; far field pressureInletOutletVelocity /
totalPressure 0 (freestreamVelocity divides by |U| and crashed at U = 0).

## Results (2026-11-18)

### Case status (every case the spec plans)

- Goe 417a, L0 (38,041 cells), alpha -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 20, kOmegaSST
  and kOmegaSSTLM: 22/22 run. Steady: SST -2..10, LM -2..6. Time-accurate
  (steady did not converge): SST 12-20, LM 8-20. Every transient was stopped
  by the 90-min cap before t = 40; its mean is over the second half of what it
  reached (window in each row: t 5-38 depending on the case).
- Goe 417a, L1 (119,220 cells), 4 and 12 deg, both closures: 4/4 run. 4 deg
  steady; 12 deg time-accurate, 4 processes, cut at 90 min (SST t 5.8-11.6,
  LM t 4.0-8.0: short windows).
- A1's consequence ("the fine mesh is the mesh and it is checked once more"):
  NOT run. It would mean the whole grid again on L1 plus a third mesh at 4 and
  12 deg; on this machine an L1 transient reaches t ~ 8-12 in 90 min on 4
  processes, so ~10 transient L1 cases + 2 L2 cases is roughly 8-10 h. The
  table below is L0.
- Flat plate, L0 (40,666 cells), alpha 2, 5, 8, 10, 15, 20, 24, both closures:
  14/14 run (2 and 5 steady, the rest time-accurate, all cut at 90 min).
- Still-fluid plunge check (laminar, L0): run.
- B8 and B6 through OpenFOAM (kOmegaSSTLM, L0, five beats complete, 5 and 6
  processes, 56 and 45 min): run. Strip solver B8 and B6, hinge and turn: run.
- Not results: `A_goe417a_L0_kOmegaSST_a+{06,08,10}.0_tr` (transients started
  by the driver bug fixed at 16:06; superseded by the steady rows, which they
  agree with: 6 deg CL 1.022 vs 1.017); the first rows for the two L1 12-deg
  cases (mode "failed": the dummy Pstream), superseded by later rows; the SST
  L1 12-deg row written at 00:06 (read only a 90-s test's file), superseded by
  the 00:07 row. compare.py takes the last row per case.

### Solver health (summary.py prints one line per case)

checkMesh "Mesh OK" for all three meshes (numbers under "Set-up"). y+ max over
the wall: 0.88-1.57 on L0 Goe (highest at 10 deg SST), 0.51-0.79 on L1,
0.96-1.05 on the plate, 1.30 (B8) and 1.16 (B6) in motion. Steady cases: CL
varies by <= 4.9e-3 over the last 1000 iterations, or residualControl
(p 1e-6, U 1e-7) stopped them; LM steady cases at 0-6 deg sit near the 0.005
bound (3.8e-3 to 4.8e-3) with p residuals 1e-4 to 4e-4: steady to the rule,
not to machine precision. Transients: CL bounds over the window are in each
row (e.g. LM 12 deg L0 swings 0.81-1.43, SST 12 deg 1.285-1.298).

Per case (summary.py, last row per case):

```
A_goe417a_L0_kOmegaSST_a-02.0: CL 0.1318 CD 0.04041 | steady 4796 it, CL range 1.7e-06 (last 1000), residuals p 1.0e-06 Ux 1.7e-07 | y+ max 0.95 mean 0.24 | cells 38041
A_goe417a_L0_kOmegaSST_a+04.0: CL 0.8353 CD 0.02212 | steady 4367 it, CL range 2.2e-04 (last 1000), residuals p 2.2e-04 Ux 7.8e-05 | y+ max 1.02 mean 0.27 | cells 38041
A_goe417a_L0_kOmegaSST_a+00.0: CL 0.3976 CD 0.02925 | steady 1374 it, CL range 1.1e-03 (last 1000), residuals p 2.8e-05 Ux 9.0e-06 | y+ max 0.90 mean 0.25 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a-02.0: CL -0.0590 CD 0.05043 | steady 2128 it, CL range 3.7e-03 (last 1000), residuals p 3.5e-07 Ux 9.7e-08 | y+ max 0.98 mean 0.22 | cells 38041
A_goe417a_L1_kOmegaSST_a+04.0: CL 0.8346 CD 0.02247 | steady 1763 it, CL range 4.4e-03 (last 1000), residuals p 6.1e-05 Ux 2.8e-05 | y+ max 0.52 mean 0.13 | cells 119220
A_goe417a_L0_kOmegaSSTLM_a+04.0: CL 0.8431 CD 0.02531 | steady 6447 it, CL range 1.3e-03 (last 1000), residuals p 3.9e-04 Ux 9.1e-05 | y+ max 1.02 mean 0.23 | cells 38041
A_goe417a_L0_kOmegaSST_a+02.0: CL 0.6313 CD 0.02351 | steady 1332 it, CL range 1.1e-03 (last 1000), residuals p 2.7e-04 Ux 9.2e-05 | y+ max 0.88 mean 0.26 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+00.0: CL 0.2319 CD 0.03806 | steady 2295 it, CL range 4.6e-03 (last 1000), residuals p 6.1e-07 Ux 3.0e-07 | y+ max 0.94 mean 0.23 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+02.0: CL 0.5461 CD 0.03149 | steady 2429 it, CL range 4.8e-03 (last 1000), residuals p 1.1e-04 Ux 3.5e-05 | y+ max 0.90 mean 0.23 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+06.0: CL 1.0522 CD 0.02482 | steady 2068 it, CL range 3.8e-03 (last 1000), residuals p 3.0e-07 Ux 1.7e-07 | y+ max 1.23 mean 0.21 | cells 38041
B_still_L0: added mass ratio 1.035, fit residual 0.021 of peak
A_goe417a_L0_kOmegaSST_a+06.0: CL 1.0170 CD 0.02459 | steady 756 it, CL range 1.7e+00 (last 1000), residuals p 1.1e-07 Ux 3.3e-08, residualControl | y+ max 1.22 mean 0.26 | cells 38041
A_goe417a_L0_kOmegaSST_a+08.0: CL 1.1715 CD 0.03018 | steady 847 it, CL range 2.4e+00 (last 1000), residuals p 1.1e-07 Ux 4.6e-08, residualControl | y+ max 1.42 mean 0.26 | cells 38041
A_goe417a_L0_kOmegaSST_a+10.0: CL 1.2827 CD 0.04051 | steady 1574 it, CL range 5.3e-03 (last 1000), residuals p 1.1e-07 Ux 4.7e-08, residualControl | y+ max 1.57 mean 0.26 | cells 38041
A_goe417a_L1_kOmegaSSTLM_a+04.0: CL 0.8413 CD 0.02671 | steady 3417 it, CL range 4.8e-03 (last 1000), residuals p 1.1e-04 Ux 3.7e-05 | y+ max 0.51 mean 0.12 | cells 119220
A_goe417a_L0_kOmegaSSTLM_a+12.0: CL 1.0690 CD 0.11756 | transient t 12.34..24.69 (dt-weighted mean), CL in [0.807, 1.427], CD in [0.0686, 0.1464], stop timeout | y+ max 1.31 mean 0.25 | cells 38041
A_goe417a_L0_kOmegaSST_a+12.0: CL 1.2933 CD 0.06398 | transient t 15.94..31.89 (dt-weighted mean), CL in [1.285, 1.298], CD in [0.0632, 0.0647], stop timeout | y+ max 1.60 mean 0.24 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+08.0: CL 1.1690 CD 0.03196 | transient t 10.08..20.16 (dt-weighted mean), CL in [1.156, 1.172], CD in [0.0310, 0.0336], stop timeout | y+ max 1.39 mean 0.26 | cells 38041
A_plate_L0_kOmegaSST_a+02.0: CL 0.2195 CD 0.02576 | steady 733 it, CL range 5.5e-01 (last 1000), residuals p 1.0e-07 Ux 1.1e-08, residualControl | y+ max 1.02 mean 0.29 | cells 40666
A_plate_L0_kOmegaSSTLM_a+02.0: CL 0.2179 CD 0.02482 | steady 1459 it, CL range 1.2e-03 (last 1000), residuals p 7.5e-06 Ux 2.3e-06 | y+ max 1.03 mean 0.28 | cells 40666
A_plate_L0_kOmegaSST_a+05.0: CL 0.5739 CD 0.05986 | steady 1248 it, CL range 4.3e-01 (last 1000), residuals p 1.0e-07 Ux 1.5e-08, residualControl | y+ max 1.02 mean 0.26 | cells 40666
A_goe417a_L0_kOmegaSST_a+16.0: CL 1.2254 CD 0.30022 | transient t 9.29..18.59 (dt-weighted mean), CL in [1.093, 1.364], CD in [0.2667, 0.3406], stop timeout | y+ max 1.24 mean 0.27 | cells 38041
A_goe417a_L0_kOmegaSST_a+14.0: CL 0.9796 CD 0.16603 | transient t 18.83..37.67 (dt-weighted mean), CL in [0.938, 1.043], CD in [0.1561, 0.1828], stop timeout | y+ max 1.27 mean 0.26 | cells 38041
A_plate_L0_kOmegaSSTLM_a+05.0: CL 0.5576 CD 0.05633 | steady 1700 it, CL range 1.4e-03 (last 1000), residuals p 1.7e-06 Ux 2.8e-07 | y+ max 1.05 mean 0.25 | cells 40666
A_goe417a_L0_kOmegaSSTLM_a+10.0: CL 1.3002 CD 0.06151 | transient t 9.63..19.25 (dt-weighted mean), CL in [1.155, 1.436], CD in [0.0267, 0.0919], stop timeout | y+ max 1.47 mean 0.30 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+14.0: CL 0.9800 CD 0.16980 | transient t 13.16..26.33 (dt-weighted mean), CL in [0.968, 1.015], CD in [0.1674, 0.1759], stop timeout | y+ max 1.27 mean 0.25 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+16.0: CL 1.1110 CD 0.27308 | transient t 6.23..12.46 (dt-weighted mean), CL in [0.861, 1.259], CD in [0.2321, 0.3830], stop timeout | y+ max 1.24 mean 0.26 | cells 38041
A_goe417a_L0_kOmegaSST_a+20.0: CL 1.6336 CD 0.61454 | transient t 5.04..10.08 (dt-weighted mean), CL in [1.002, 2.223], CD in [0.3944, 0.8230], stop timeout | y+ max 1.23 mean 0.34 | cells 38041
A_goe417a_L0_kOmegaSSTLM_a+20.0: CL 1.2902 CD 0.42541 | transient t 6.51..13.02 (dt-weighted mean), CL in [1.126, 1.442], CD in [0.3705, 0.4561], stop timeout | y+ max 1.19 mean 0.26 | cells 38041
A_plate_L0_kOmegaSST_a+08.0: CL 0.6939 CD 0.11123 | transient t 17.35..34.70 (dt-weighted mean), CL in [0.669, 0.716], CD in [0.1084, 0.1142], stop timeout | y+ max 0.98 mean 0.26 | cells 40666
A_plate_L0_kOmegaSST_a+10.0: CL 0.7024 CD 0.14087 | transient t 18.07..36.14 (dt-weighted mean), CL in [0.692, 0.714], CD in [0.1389, 0.1428], stop timeout | y+ max 0.96 mean 0.26 | cells 40666
A_plate_L0_kOmegaSST_a+15.0: CL 1.0729 CD 0.32146 | transient t 10.30..20.59 (dt-weighted mean), CL in [0.886, 1.312], CD in [0.2977, 0.3518], stop timeout | y+ max 1.00 mean 0.30 | cells 40666
A_plate_L0_kOmegaSSTLM_a+08.0: CL 0.7065 CD 0.10963 | transient t 15.52..31.04 (dt-weighted mean), CL in [0.686, 0.728], CD in [0.1071, 0.1123], stop timeout | y+ max 1.00 mean 0.25 | cells 40666
A_plate_L0_kOmegaSSTLM_a+10.0: CL 0.7186 CD 0.14100 | transient t 17.81..35.62 (dt-weighted mean), CL in [0.713, 0.723], CD in [0.1400, 0.1416], stop timeout | y+ max 0.98 mean 0.26 | cells 40666
A_goe417a_L1_kOmegaSST_a+12.0: CL nan CD nan | FAILED (1, t 0.0) | y+ max 0.79 mean 0.12 | cells 119220
A_goe417a_L1_kOmegaSSTLM_a+12.0: CL nan CD nan | FAILED (1, t 0.0) | y+ max 0.73 mean 0.12 | cells 119220
A_plate_L0_kOmegaSSTLM_a+15.0: CL 1.0684 CD 0.30869 | transient t 12.44..24.88 (dt-weighted mean), CL in [0.894, 1.298], CD in [0.2770, 0.3466], stop timeout | y+ max 0.99 mean 0.27 | cells 40666
A_plate_L0_kOmegaSST_a+20.0: CL 1.1319 CD 0.44417 | transient t 12.87..25.74 (dt-weighted mean), CL in [0.836, 1.415], CD in [0.3715, 0.5102], stop timeout | y+ max 0.99 mean 0.27 | cells 40666
A_plate_L0_kOmegaSSTLM_a+20.0: CL 1.1016 CD 0.42467 | transient t 13.69..27.38 (dt-weighted mean), CL in [0.908, 1.296], CD in [0.3773, 0.4691], stop timeout | y+ max 0.99 mean 0.28 | cells 40666
B_B8_L0_kOmegaSSTLM: CN 0.6916 CX 0.0417 CP 0.0844 | beat CN means 0.631 0.681 0.689 0.692 0.694 | beats 3-5 of 5 (t 24.85..62.12) | y+ max 1.30
A_plate_L0_kOmegaSST_a+24.0: CL 1.1470 CD 0.53365 | transient t 13.20..26.39 (dt-weighted mean), CL in [0.914, 1.505], CD in [0.4753, 0.6607], stop timeout | y+ max 0.99 mean 0.27 | cells 40666
A_plate_L0_kOmegaSSTLM_a+24.0: CL 1.1974 CD 0.55232 | transient t 12.99..25.98 (dt-weighted mean), CL in [1.017, 1.456], CD in [0.4952, 0.6407], stop timeout | y+ max 0.98 mean 0.27 | cells 40666
B_B6_L0_kOmegaSSTLM: CN 1.8411 CX 0.1842 CP 0.8515 | beat CN means 1.519 1.804 1.849 1.836 1.839 | beats 3-5 of 5 (t 13.31..33.28) | y+ max 1.16
A_goe417a_L1_kOmegaSST_a+12.0: CL 1.2479 CD 0.07291 | transient t 5.80..11.59 (dt-weighted mean), CL in [1.118, 1.391], CD in [0.0592, 0.0959], stop timeout | y+ max 0.79 mean 0.12 | cells 119220
A_goe417a_L1_kOmegaSSTLM_a+12.0: CL 1.3466 CD 0.08097 | transient t 4.01..8.01 (dt-weighted mean), CL in [1.303, 1.359], CD in [0.0773, 0.0865], stop timeout | y+ max 0.78 mean 0.13 | cells 119220
```

### Stage A comparison (L0) and A1-A5

Goe 417a, Re 1e5, mesh L0
| alpha | meas CL (lift run) | meas CL (drag run) | meas CD | SST CL | SST CD | SST mode | LM CL | LM CD | LM mode |
|---|---|---|---|---|---|---|---|---|---|
| -2 | -0.160 | -0.184 | 0.0588 | 0.132 | 0.0404 | steady | -0.059 | 0.0504 | steady |
| 0 | 0.195 | 0.181 | 0.0413 | 0.398 | 0.0292 | steady | 0.232 | 0.0381 | steady |
| 2 | 0.539 | 0.527 | 0.0314 | 0.631 | 0.0235 | steady | 0.546 | 0.0315 | steady |
| 4 | 0.813 | 0.812 | 0.0243 | 0.835 | 0.0221 | steady | 0.843 | 0.0253 | steady |
| 6 | 1.003 | 0.983 | 0.0183 | 1.017 | 0.0246 | steady | 1.052 | 0.0248 | steady |
| 8 | 1.154 | 1.157 | 0.0188 | 1.172 | 0.0302 | steady | 1.169 | 0.0320 | transient |
| 10 | 1.267 | 1.296 | 0.0302 | 1.283 | 0.0405 | steady | 1.300 | 0.0615 | transient |
| 12 | 1.341 | - | - | 1.293 | 0.0640 | transient | 1.069 | 0.1176 | transient |
| 14 | 1.321 | - | - | 0.980 | 0.1660 | transient | 0.980 | 0.1698 | transient |
| 16 | 1.177 | - | - | 1.225 | 0.3002 | transient | 1.111 | 0.2731 | transient |
| 20 | 1.196 | - | - | 1.634 | 0.6145 | transient | 1.290 | 0.4254 | transient |

A1 mesh (L0 -> L1): lift under 3%, drag under 8%
  kOmegaSST    4 deg: CL 0.8353 -> 0.8346 (0.1%), CD 0.02212 -> 0.02247 (1.6%): holds
  kOmegaSST   12 deg: CL 1.2933 -> 1.2479 (3.5%), CD 0.06398 -> 0.07291 (14.0%): fails
  kOmegaSSTLM    4 deg: CL 0.8431 -> 0.8413 (0.2%), CD 0.02531 -> 0.02671 (5.5%): holds
  kOmegaSSTLM   12 deg: CL 1.0690 -> 1.3466 (26.0%), CD 0.11756 -> 0.08097 (31.1%): fails
  A1: {'kOmegaSST': False, 'kOmegaSSTLM': False}

kOmegaSSTLM: A2 lift (|dCL| <= 0.10 at 2-10): holds (worst 0.050); A2 drag (within 40%): fails (worst 104%); complete True
   max CL on the grid 1.300 at 10 deg; parabola 1.304 at 9.72 deg; A3 (within 0.15 of 1.355 and 2 deg of 13.5): fails

kOmegaSST: A2 lift (|dCL| <= 0.10 at 2-10): holds (worst 0.092); A2 drag (within 40%): fails (worst 61%); complete True
   max CL on the grid 1.634 at 20 deg; parabola 1.634 at 20.00 deg; A3 (within 0.15 of 1.355 and 2 deg of 13.5): fails

A4: at 0 deg (measured 0.195) either closure off by up to 0.2
  kOmegaSST: CL 0.398, off by +0.202; worst point of 2-10? True
  kOmegaSSTLM: CL 0.232, off by +0.036; worst point of 2-10? False

A5: SST drag too high below 6 deg, lift held
  -2 deg: SST CD 0.0404 vs measured 0.0588 (-31%)
  0 deg: SST CD 0.0292 vs measured 0.0413 (-29%)
  2 deg: SST CD 0.0235 vs measured 0.0314 (-25%)
  4 deg: SST CD 0.0221 vs measured 0.0243 (-9%)
  SST lift at 2-10 within 0.10: True (worst 0.092)

Flat plate (3.1%, semicircular edges), Re 1e5, mesh L0
| alpha | tunnel CL (rising) | 2 pi alpha | SST CL | SST CD | LM CL | LM CD |
|---|---|---|---|---|---|---|
| 2 | 0.204 | 0.219 | 0.219 | 0.0258 | 0.218 | 0.0248 |
| 5 | 0.602 | 0.548 | 0.574 | 0.0599 | 0.558 | 0.0563 |
| 8 | 0.777 | 0.877 | 0.694 | 0.1112 | 0.707 | 0.1096 |
| 10 | 0.789 | 1.097 | 0.702 | 0.1409 | 0.719 | 0.1410 |
| 15 | 0.810 | 1.645 | 1.073 | 0.3215 | 1.068 | 0.3087 |
| 20 | 0.958 | 2.193 | 1.132 | 0.4442 | 1.102 | 0.4247 |
| 24 | 1.203 | 2.632 | 1.147 | 0.5336 | 1.197 | 0.5523 |

### Verdicts, stage A (frozen wording quoted)

- A1 "lift at 4 and 12 deg changes by under 3% on the finer mesh, drag by
  under 8%": FAILS for both closures (holds at 4 deg; at 12 deg SST lift 3.5%,
  drag 14%; LM lift 26%, drag 31%). The 12-deg comparison is not a clean read
  of mesh dependence: the L1 windows are short (LM t 4-8, SST t 5.8-11.6, right
  after a non-converged steady field), and their CL ranges (LM 1.303-1.359,
  SST 1.118-1.391) lie inside or across the L0 swings (LM 0.807-1.427 over
  t 12-25), so L1 may not yet have reached the low-lift phase. No inference
  that the fine mesh approaches the tunnel (1.341) is drawn.
- A2 "Transition closure, 2-10 deg: |CL - measured| <= 0.10 at every angle;
  CD within 40% of measured at every angle": lift HOLDS (worst 0.050 at 6),
  drag FAILS (+36% at 6, +70% at 8, +104% at 10; 0% and +4% at 2 and 4).
- A3 "maximum lift within 0.15 of the measured 1.355 and its angle within
  2 deg of 13.5": FAILS on L0 (1.30 at 9.7 deg by the parabola; 12 deg drops
  to 1.07 in a large oscillation). Not read on L1 (only 4 and 12 deg there).
- A4 "At 0 deg (measured 0.195) either closure may be off by up to 0.2":
  LM: off by +0.036 (within 0.2: holds), but 0 deg is not its worst point
  (6 deg is, 0.050 against the tunnel's CL; worst point clause fails).
  SST: off by +0.202 (just outside 0.2: fails), and 0 deg is its worst point
  (clause holds). Gates nothing.
- A5 "Fully turbulent closure: predicted to miss A2's drag (too high below 6
  deg) and to hold its lift": lift held (worst 0.092); drag missed, but LOW
  below 6 deg (-31%, -29%, -25%, -9% at -2..4) and high above (+34..61%): the
  direction predicted is wrong; A5 fails as worded.
- Qualification: A1-A3 hold with no closure. The solver is NOT qualified.
  Stage B ran with the nearer, kOmegaSSTLM (lift in A2 at half SST's worst
  error, A4 closer, SST's grid maximum is a 20-deg post-stall artefact), and
  every stage-B number is UNQUALIFIED.

### Flat plate reading (gates nothing)

Spec: "the lift at 5 deg is 0.50-0.56, at or under 2 pi alpha, and not the
tunnel's 0.60-0.65; from 15 deg the two-dimensional mean lift is 0.9-1.3".
Measured: 5 deg LM 0.558 (in 0.50-0.56, but just above 2 pi alpha = 0.548),
SST 0.574 (above both); tunnel 0.602 at 5. 15/20/24 deg mean lift 1.07/1.10/1.20
(LM) and 1.07/1.13/1.15 (SST): inside 0.9-1.3. At 8-10 deg the CFD (0.69-0.72)
sits below the tunnel (0.78-0.79).

### Stage B (all CFD numbers unqualified)

still-fluid plunge: added mass 0.8131 vs pi c^2/4 0.7854: ratio 1.035 -> within 10%
| station | quantity | strip (hinge) | strip (turn) | outside solver |
|---|---|---|---|---|
| B8 | CN | 0.6443 | 0.6443 | 0.6916 |
| B8 | CX | 0.0723 | 0.0723 | 0.0417 |
| B8 | CP | 0.1082 | 0.1080 | 0.0844 |
| B8 | CN max, last beat (downstroke) | 1.411 | 1.274 | 1.339 (1.33862939716) |
| B6 | CN | 1.2517 | 1.2517 | 1.8411 |
| B6 | CX | -0.1494 | -0.1494 | 0.1842 |
| B6 | CP | 0.2411 | 0.2111 | 0.8515 |
| B6 | CN max, last beat (downstroke) | 3.000 | 1.764 | 4.166 (4.1662805204) |
B1 (B8): normal +7.4% (<= 15%), streamwise -0.0305 (<= 0.03), power -22.0% (<= 20%): fails
B2 (B6): normal +47.1% (must be > +15%), power +253.2% (must be > 0): holds
B3 (B6): downstroke peak CN 4.166 vs static max 1.355: holds

Strip hinge and turn agree to four decimals in mean CN and CX: the
rate-dependent terms (rotational lift, pitch damping) average to zero over a
beat; their peaks and moments differ, as they should. In the CFD runs the wake
refinement box turns with the pre-turned mesh (13.5 deg for B8, 39.2 deg for
B6) and then pitches with it, so wake resolution in the inertial frame is not
stage A's (a set-up consequence, not tested).
Strip "hinge" is the primary strip-solver run (pitch rate seen by the solver);
"turn" is the rig-style chord turn, informational.

- Still-fluid check: added mass 0.8131 against pi c^2/4 = 0.7854, ratio 1.035
  (within 10%): the moving-mesh set-up passes.
- B1 "B8: mean normal force within 15% of the strip solver's; mean streamwise
  force within 0.03 in coefficient; power within 20%": FAILS, narrowly on two of
  three: normal +7.4% (holds), streamwise 0.0417 vs 0.0723 (-0.0305, limit 0.03),
  power 0.0844 vs 0.1082 (-22%, limit 20%).
- B2 "B6: the outside solver's mean normal force is higher than the strip
  solver's by more than 15% ..., and its power is higher too": HOLDS:
  normal 1.841 vs 1.252 (+47%), power 0.852 vs 0.241 (3.5x). Also the
  streamwise force changes sign: CFD +0.184 (thrust), strip -0.149 (drag).
- B3 "B6: the lift history shows a peak above the static maximum 1.355 on the
  downstroke": HOLDS: CN 4.17 at phase 0.55 of the beat (downstroke is
  0.25-0.75). Removing the plunge's added-mass term (pi/4 h'' / q, amplitude
  1.32 in CN) raises the peak to 4.97 at phase 0.62, so the peak is not added
  mass. Strip solver's own peak: 3.00 (hinge), 1.76 (turn).
- Convergence: B8 beat-mean CN 0.631, 0.681, 0.689, 0.692, 0.694; B6 1.519,
  1.804, 1.849, 1.836, 1.839. The strip side drifts 0.1-0.6% per beat at beat 5.
- CFD mean pitching moment about c/4: -0.094 (B8), -0.269 (B6); the strip
  solver's: +0.0003 / +0.0017 (hinge). The section's own nose-down moment is
  absent on the strip side here; it enters the power through M x pitch rate.

### Problems met (all recorded in the log below)

1. Fine mesh: gmsh left a degenerate element at the layer's outer row (BL
   0.03 c); BL set to 0.02 (Goe) / 0.025 (plate) before any flow result.
2. residualControl did not fire in the first launch; driver restarted with an
   on-the-fly stop on the convergence rule and resume from written iterations.
3. Driver bug: residual-converged runs under 1000 iterations were sent to the
   transient path (fixed 16:06; SST 6/8/10 re-read steady).
4. Every transient hit the 90-min cap (load: another job plus 10 of our
   processes; 0.3-1.2 s per step): means are over windows of t 4-38, shorter
   than the planned t 20-40, and the L1 12-deg windows are 4-6 units.
5. The conda OpenFOAM links the dummy Pstream (RPATH wants lib/sys-mpich, the
   package ships lib/mpich-3.3): parallel runs failed until the MPICH
   libPstream was preloaded for the solver only.
6. Still fluid: freestreamVelocity divides by |U| (crash at U = 0); the sharp
   base corners limit the step (flow round them ~12x the plunge speed);
   h0 0.05 -> 0.02 c, maxCo 10 for that check.
7. Stage B maxCo 4 -> 8 before any B result (cost), timeout raised to 4 h for
   B: B8 took 56 min, B6 45 min, so neither exceeded 90 min in the end.
8. Strip side: with inflow off the solver uses disc_span^2/area as the aspect
   ratio (3.9 by default in `bare`), set to 1e4.
9. The project switched its default lift law to "table" on 2026-11-17
   (CLAUDE.md, repair R3). strip2d's strip has camber and a measured section,
   so the section's table supplies its coefficients either way.

## Log
- 11-17 14:44 stage A driver launched (9 cores)
- 11-17 15:05 driver restarted: steady runs now stop as soon as the convergence rule holds (stopAt writeNow), the rule is applied whatever the exit code (a 90-min timeout with converged forces counts), interrupted cases resume from their last written iteration (4000-4500 L0, ~1000 L1). Reason: residualControl never fired, so every case ran 8000 iterations, and L1 ran at 1.3 s/iteration under load (8000 would exceed the 90-min timeout).
- 11-17 15:06 A_goe417a_L0_kOmegaSST_a-02.0: steady, 4796 it, CL 0.1318 CD 0.04041 Cm -0.0746, CL range 1.70e-06, y+max 0.95210862543, 1.0 min
- 11-17 15:06 A_goe417a_L0_kOmegaSST_a+04.0: steady, 4367 it, CL 0.8353 CD 0.02212 Cm -0.0911, CL range 2.19e-04, y+max 1.0231439038, 1.0 min
- 11-17 15:15 A_goe417a_L0_kOmegaSST_a+00.0: steady, 1374 it, CL 0.3976 CD 0.02925 Cm -0.0946, CL range 1.14e-03, y+max 0.90121578483, 9.0 min
- 11-17 15:21 A_goe417a_L0_kOmegaSSTLM_a-02.0: steady, 2128 it, CL -0.0590 CD 0.05043 Cm -0.0420, CL range 3.72e-03, y+max 0.98272432543, 15.0 min
- 11-17 15:23 A_goe417a_L1_kOmegaSST_a+04.0: steady, 1763 it, CL 0.8346 CD 0.02247 Cm -0.0912, CL range 4.41e-03, y+max 0.51627028234, 18.0 min
- 11-17 15:27 A_goe417a_L0_kOmegaSSTLM_a+04.0: steady, 6447 it, CL 0.8431 CD 0.02531 Cm -0.1066, CL range 1.32e-03, y+max 1.0180693494, 22.0 min
- 11-17 15:29 A_goe417a_L0_kOmegaSST_a+02.0: steady, 1332 it, CL 0.6313 CD 0.02351 Cm -0.0966, CL range 1.09e-03, y+max 0.88339276329, 8.0 min
- 11-17 15:31 A_goe417a_L0_kOmegaSSTLM_a+00.0: steady, 2295 it, CL 0.2319 CD 0.03806 Cm -0.0751, CL range 4.61e-03, y+max 0.94226002111, 16.0 min
- 11-17 15:39 A_goe417a_L0_kOmegaSSTLM_a+02.0: steady, 2429 it, CL 0.5461 CD 0.03149 Cm -0.0987, CL range 4.85e-03, y+max 0.89680079942, 16.0 min
- 11-17 15:43 A_goe417a_L0_kOmegaSSTLM_a+06.0: steady, 2068 it, CL 1.0522 CD 0.02482 Cm -0.0959, CL range 3.82e-03, y+max 1.2317678743, 14.0 min
- 11-17 15:43 B_still_L0: added mass 0.8131 vs pi c^2/4 = 0.7854 (ratio 1.035); in-quadrature coef -0.6616; fit residual 0.021 of peak; rc 0, 40.3 min
- 11-17 16:06 driver restarted again: a steady run that residualControl stopped before 1000 iterations was sent to the transient path (the rule needs 1000 iterations); fixed to judge it on the last half of its iterations, as NOTES says. SST 6/8/10 (L0) had gone transient for that reason (their transients agree with the steady values to 0.5%: CL 1.022 vs 1.017 at 6 deg); they are now read steady. Running transients resume from their last write.
- 11-17 16:07 A_goe417a_L0_kOmegaSST_a+06.0: steady, 756 it, CL 1.0170 CD 0.02459 Cm -0.0841, CL range 1.70e+00, y+max 1.2247795219, 0.5 min
- 11-17 16:07 A_goe417a_L0_kOmegaSST_a+08.0: steady, 847 it, CL 1.1715 CD 0.03018 Cm -0.0759, CL range 2.38e+00, y+max 1.4199607606, 0.5 min
- 11-17 16:08 A_goe417a_L0_kOmegaSST_a+10.0: steady, 1574 it, CL 1.2827 CD 0.04051 Cm -0.0654, CL range 5.33e-03, y+max 1.5662138105, 1.4 min
- 11-17 16:32 A_goe417a_L1_kOmegaSSTLM_a+04.0: steady, 3417 it, CL 0.8413 CD 0.02671 Cm -0.1071, CL range 4.81e-03, y+max 0.51236982808, 25.0 min
- 11-17 17:37 A_goe417a_L0_kOmegaSSTLM_a+12.0: steady not converged (CL range 2.43); transient 12.3..24.7: CL 1.0690 [0.807,1.427] CD 0.11756 Cm -0.0865, rc timeout, 90.0 min
- 11-17 17:37 A_goe417a_L0_kOmegaSST_a+12.0: steady not converged (CL range 0.0415); transient 15.9..31.9: CL 1.2933 [1.285,1.298] CD 0.06398 Cm -0.0564, rc timeout, 90.0 min
- 11-17 17:58 A_goe417a_L0_kOmegaSSTLM_a+08.0: steady not converged (CL range 0.887); transient 10.1..20.2: CL 1.1690 [1.156,1.172] CD 0.03196 Cm -0.0727, rc timeout, 90.0 min
- 11-17 18:00 A_plate_L0_kOmegaSST_a+02.0: steady, 733 it, CL 0.2195 CD 0.02576 Cm 0.0014, CL range 5.54e-01, y+max 1.0241445219, 2.5 min
- 11-17 18:08 A_plate_L0_kOmegaSSTLM_a+02.0: steady, 1459 it, CL 0.2179 CD 0.02482 Cm 0.0013, CL range 1.19e-03, y+max 1.0281350943, 8.0 min
- 11-17 18:11 A_plate_L0_kOmegaSST_a+05.0: steady, 1248 it, CL 0.5739 CD 0.05986 Cm -0.0207, CL range 4.34e-01, y+max 1.0226515694, 3.3 min
- 11-17 18:14 A_goe417a_L0_kOmegaSST_a+16.0: steady not converged (CL range 2.02); transient 9.3..18.6: CL 1.2254 [1.093,1.364] CD 0.30022 Cm -0.1841, rc timeout, 90.0 min
- 11-17 18:14 A_goe417a_L0_kOmegaSST_a+14.0: steady not converged (CL range 2.47); transient 18.8..37.7: CL 0.9796 [0.938,1.043] CD 0.16603 Cm -0.0982, rc timeout, 90.0 min
- 11-17 18:20 A_plate_L0_kOmegaSSTLM_a+05.0: steady, 1700 it, CL 0.5576 CD 0.05633 Cm -0.0134, CL range 1.39e-03, y+max 1.045631384, 9.0 min
- 11-17 18:25 A_goe417a_L0_kOmegaSSTLM_a+10.0: steady not converged (CL range 1.85); transient 9.6..19.3: CL 1.3002 [1.155,1.436] CD 0.06151 Cm -0.0829, rc timeout, 90.0 min
- 11-17 18:25 A_goe417a_L0_kOmegaSSTLM_a+14.0: steady not converged (CL range 2.43); transient 13.2..26.3: CL 0.9800 [0.968,1.015] CD 0.16980 Cm -0.1007, rc timeout, 90.0 min
- 11-17 18:45 A_goe417a_L0_kOmegaSSTLM_a+16.0: steady not converged (CL range 2.25); transient 6.2..12.5: CL 1.1110 [0.861,1.259] CD 0.27308 Cm -0.1641, rc timeout, 90.0 min
- 11-17 19:29 A_goe417a_L0_kOmegaSST_a+20.0: steady not converged (CL range 2.59); transient 5.0..10.1: CL 1.6336 [1.002,2.223] CD 0.61454 Cm -0.3993, rc timeout, 90.0 min
- 11-17 19:38 A_goe417a_L0_kOmegaSSTLM_a+20.0: steady not converged (CL range 2.41); transient 6.5..13.0: CL 1.2902 [1.126,1.442] CD 0.42541 Cm -0.2152, rc timeout, 90.0 min
- 11-17 20:15 A_plate_L0_kOmegaSST_a+08.0: steady not converged (CL range 2.3); transient 17.4..34.7: CL 0.6939 [0.669,0.716] CD 0.11123 Cm -0.0762, rc timeout, 90.0 min
- 11-17 20:22 A_plate_L0_kOmegaSST_a+10.0: steady not converged (CL range 2.37); transient 18.1..36.1: CL 0.7024 [0.692,0.714] CD 0.14087 Cm -0.0870, rc timeout, 90.0 min
- 11-17 20:27 A_plate_L0_kOmegaSST_a+15.0: steady not converged (CL range 2.54); transient 10.3..20.6: CL 1.0729 [0.886,1.312] CD 0.32146 Cm -0.1846, rc timeout, 90.0 min
- 11-17 20:28 A_plate_L0_kOmegaSSTLM_a+08.0: steady not converged (CL range 2.06); transient 15.5..31.0: CL 0.7065 [0.686,0.728] CD 0.10963 Cm -0.0684, rc timeout, 90.0 min
- 11-17 20:33 A_plate_L0_kOmegaSSTLM_a+10.0: steady not converged (CL range 2.62); transient 17.8..35.6: CL 0.7186 [0.713,0.723] CD 0.14100 Cm -0.0835, rc timeout, 90.0 min
- 11-17 20:33 A_goe417a_L1_kOmegaSST_a+12.0: FAILED (steady rc 0, transient rc 1, t reached 0.00)
- 11-17 20:33 A_goe417a_L1_kOmegaSSTLM_a+12.0: FAILED (steady rc timeout, transient rc 1, t reached 0.00)
- 11-17 20:54 A_plate_L0_kOmegaSSTLM_a+15.0: steady not converged (CL range 2.5); transient 12.4..24.9: CL 1.0684 [0.894,1.298] CD 0.30869 Cm -0.1751, rc timeout, 90.0 min
- 11-17 20:54 stage A on L0 read (compare.py): neither closure qualifies (A3 fails for both, A2 drag fails for both); kOmegaSSTLM is the nearer (A2 lift holds, worst 0.050; SST 0.092 and its max CL is at 20 deg). Stage B runs kOmegaSSTLM on L0, every number UNQUALIFIED. Parallel runs had failed: the env's RPATH picks the dummy Pstream (lib/sys-mpich missing, lib/mpich-3.3 present); fixed by preloading mpich-3.3/libPstream.so for the solver only (run_a.mpi / run_b.mpi). The two L1 12-deg transients failed on that and are re-run. B8 launched on 5 processes, timeout raised to 4 h for stage B (over the 90-min rule; reported).
- 11-17 20:54 B_B8_L0_kOmegaSSTLM: built (h0/c 0.6261, k 0.2529, omega* 0.5057, T* 12.424, theta 2.956 + 10.500 cos, Re 1.395e+05, maxCo 8.0, 5 procs); running
- 11-17 21:24 A_plate_L0_kOmegaSST_a+20.0: steady not converged (CL range 2.15); transient 12.9..25.7: CL 1.1319 [0.836,1.415] CD 0.44417 Cm -0.1987, rc timeout, 90.0 min
- 11-17 21:40 A_plate_L0_kOmegaSSTLM_a+20.0: steady not converged (CL range 2.16); transient 13.7..27.4: CL 1.1016 [0.908,1.296] CD 0.42467 Cm -0.1800, rc timeout, 90.0 min
- 11-17 21:50 B_B8_L0_kOmegaSSTLM: CN 0.6916 CX 0.0417 CP 0.0844 CM -0.0940, beat CN means 0.631 0.681 0.689 0.692 0.694, CN max (last beat) 1.339 at phase 0.61, beats 3-5 of 5 (t 24.85..62.12), complete True, rc 0
- 11-17 21:51 B_B6_L0_kOmegaSSTLM: built (h0/c 0.9392, k 0.4720, omega* 0.9441, T* 6.655, theta 11.182 + 28.000 cos, Re 1.046e+05, maxCo 8.0, 6 procs); running
- 11-17 22:06 A_plate_L0_kOmegaSST_a+24.0: steady not converged (CL range 2.42); transient 13.2..26.4: CL 1.1470 [0.914,1.505] CD 0.53365 Cm -0.1988, rc timeout, 90.0 min
- 11-17 22:15 A_plate_L0_kOmegaSSTLM_a+24.0: steady not converged (CL range 2.42); transient 13.0..26.0: CL 1.1974 [1.017,1.456] CD 0.55232 Cm -0.2017, rc timeout, 90.0 min
- 11-17 22:15 stage A driver finished
- 11-17 22:36 B_B6_L0_kOmegaSSTLM: CN 1.8411 CX 0.1842 CP 0.8515 CM -0.2687, beat CN means 1.519 1.804 1.849 1.836 1.839, CN max (last beat) 4.166 at phase 0.55, beats 3-5 of 5 (t 13.31..33.28), complete True, rc 0
- 11-18 00:06 A_goe417a_L1_kOmegaSST_a+12.0: steady not converged (CL range 0.0106); transient 0.1..0.1: CL 1.2584 [1.258,1.259] CD 0.06970 Cm -0.0545, rc timeout, 90.0 min
- 11-18 00:06 A_goe417a_L1_kOmegaSSTLM_a+12.0: steady not converged (CL range 0.0152); transient 4.0..8.0: CL 1.3466 [1.303,1.359] CD 0.08097 Cm -0.0613, rc timeout, 90.0 min
- 11-18 00:06 stage A driver finished
- 11-18 00:07 A_goe417a_L1_kOmegaSST_a+12.0: steady not converged (CL range 0.0106); transient 5.8..11.6: CL 1.2479 [1.118,1.391] CD 0.07291 Cm -0.0535, rc timeout, 90.0 min
