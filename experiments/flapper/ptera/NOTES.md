# PteraSoftware cross-check of the flapper's cruise numbers (2026-11-17)

An independent unsteady vortex-lattice reading of the cruise points of the
home-made strip solver. Nothing here was tuned toward the strip solver's numbers.

## Install

- `python3.12 -m venv /home/hundo/Projects/Dytiscidae/.venv-ptera` (outside the repo),
  `pip install pterasoftware` -> **PteraSoftware 5.1.0**, 52 packages in the venv
  (numba, scipy, pyvista, vtk, ...), **1.3 GB** on disk. The project's `.venv` was
  not touched; nothing was installed system-wide.
- The 5.x API is not the 2.x/3.x one (`Wing(wing_cross_sections, Ler_Gs_Cgs,
  angles_Gs_to_Wn_ixyz, ...)`, `WingCrossSection(Lp_Wcsp_Lpp, angles_Wcsp_to_Wcs_ixyz)`,
  movements as amp/period/phase per component). It was read from the installed
  source; the wheel ships no examples.

## Files

| file | what |
|---|---|
| `run_ptera.py` | the script: `probe`, `all [workers]`, `read` |
| `results.jsonl` | 159 rows, one per run (all `ok`), every run's settings and means |
| `read.txt` | output of `run_ptera.py read`: trims, every sweep, controls |
| `goe417a_coords.txt` | the section outline dumped from `sections.coordinates('goe417a')` with the project's `.venv` |

Re-run: `/home/hundo/Projects/Dytiscidae/.venv-ptera/bin/python run_ptera.py all 10`
(skips rows already present), then `... run_ptera.py read`. The whole set took
about 25 min on 10 processes (numba at one thread each).

## Probe

Steady ring-vortex solver, flat rectangular AR 5.4 wing, 8 x 24 panels, 5 deg:
CL = 0.365, CDi = 0.00762 (CL^2/(pi AR) = 0.00785). Lifting line
2 pi a AR/(AR+2) = 0.400, Helmbold 0.382. Sane.

## Set-up and conventions

- Geometry axes (PteraSoftware): x aft, y to the right, z up. `alpha = 0`,
  `beta = 0`, so the free stream is +x; incidence is put on the wing, not on the
  operating point. Lift = sum of panel F_z, thrust = - sum of panel F_x.
- One `Wing` with `Ler_Gs_Cgs = (0, 0.04, 0)`, `symmetric=True` about y = 0
  ("type 5" symmetry: PteraSoftware makes a second, reflected wing; both get the
  same movement, the side force is 1e-16 N). Chord 0.2576, tip at y = 0.70.
  **The 8 cm between the roots is open air**: no hull, no image, nothing carries
  lift across it. See "The root gap" below, this is the largest effect in the set-up.
- Mesh: 8 chordwise (uniform, as the docs recommend for unsteady) x 12 spanwise
  per wing (6 segments of 2, uniform) = 192 panels.
- Section: `Airfoil(name="goe417a", outline_A_lp=<the 93 points>)`. PteraSoftware
  resamples the outline (400 per side) and uses only its **mean camber line**:
  maximum camber 5.83% at x/c = 0.381, thickness (3.4%) is dropped. Thin-aerofoil
  zero-lift angle of that camber line: -5.33 deg; the finite wing's zero-lift
  angle in control 4a is -4.97 deg (open gap), -4.79 deg (sealed).
- Flapping: `ampAngles_Gs_to_Wn_ixyz[0] = Phi/2`, period 1/f, phase 0, i.e.
  phi = (Phi/2) sin(2 pi f t) about a fore-aft axis through the root; positive
  is tip up for both wings (checked: tip z runs +-0.226 m at Phi = 40, = 0.66 sin 20).
- Incidence: `angles_Gs_to_Wn_ixyz[1] = inc`, positive = leading edge up
  (checked: the built root chord sits at +2.00 deg for inc = 2). The rotation is
  intrinsic x then y', so incidence is about the wing's own spanwise axis.
- Pivot: `rotationPointOffset_Gs_Ler = (0.1 c, 0, 0)`: flap, incidence and rigid
  feathering all turn about the point 0.1 chord behind the leading edge at the root.
- Tip feathering (share 1): seven cross-sections, each twisted A/6 relative to
  its parent (`ampAngles_Wcsp_to_Wcs_ixyz[1]`), so the twist is piecewise linear
  in the angle, zero at the root and A at the tip. A cross-section rotates about
  its own leading point; to keep the 0.1-chord line straight each section's
  leading point is also moved by 0.1 c sin(A/6) in phase (`ampLp_Wcsp_Lpp[2]`).
- Rigid feathering (share 0): no twist, `ampAngles_Gs_to_Wn_ixyz[1] = A`.
- Time: step = (1/f) / N with N = round((1/f) / (c / 8 / V)) so a wake panel is
  about as long as a bound panel (N = 99 at 8 m/s and 2.5 Hz; 53-93 at 6 m/s).
  3 cycles, **prescribed wake**, means over the last N steps. Cycle 2 to cycle 3
  the means move by at most 0.025 N lift, 0.001 N thrust, 0.010 W.

### Sign check (from the geometry the script builds, not from the inputs)

At every step the script reads the built mesh of the y > 0 wing: tip height, and
the pitch of the tip and root chords (leading edge up positive). For run 1 at inc = 2:

| | tip chord pitch | root chord pitch | tip twist relative to root |
|---|---|---|---|
| mid-downstroke (tip dz/dt most negative) | -12.93 deg | +2.00 | **-14.92 (leading edge DOWN)** |
| mid-upstroke | +16.97 deg | +2.00 | **+14.97 (leading edge UP)** |

So in a leading-edge-up-positive angle the script's twist is +A cos(2 pi f t)
(phase +90 deg on PteraSoftware's sine); this is the brief's
"theta_tip = -A cos" with theta counted leading edge down, the physical sense the
brief asks for. Rigid feathering, (3.5, 50, 15) at inc = 2: whole wing at -12.85
at mid-downstroke, +16.93 at mid-upstroke. The opposite-sign control reads
+14.92 / -14.87.

### Power

PteraSoftware has no power output. The script subclasses the solver and, in
`_load_calculation_moment_processing_hook` (which receives the force on each
panel's four bound legs and the unsteady `-rho dGamma/dt A n` force), computes

    P = - sum_legs F_leg . v_leg  -  sum_panels F_unsteady . v_collocation

with v the velocity of the leg centre / collocation point relative to the body:
the solver's own backward difference of the point's position (the same velocity
it uses in the boundary condition and in the Kutta-Joukowski force). The body is
held, so this is the rate of work the wings do on the air through the flap and
feather motion; it does not include thrust x V. Sanity: 0 for the static wing,
positive on the downstroke, negative for part of the upstroke.

## Results

Weight 5.884 N. `inc_trim` by linear interpolation between the two 2-deg sweep
points that bracket lift = weight (lift is linear in inc to the third digit).
Thrust is **inviscid**: induced drag is in it, profile drag and hull drag are not.

### 1-3. Trimmed points (open 8 cm root gap, as the brief specifies)

| V | f | Phi | A | feathering | inc_trim (deg) | thrust (N) | power (W) | strip solver (net thrust incl. viscous + hull, W) |
|---|---|---|---|---|---|---|---|---|
| 8 | 2.5 | 40 | 15 | tip | +2.42 | +0.177 | 4.37 | +0.045, 7.24 |
| 8 | 2.0 | 50 | 15 | tip | +2.55 | +0.177 | 4.36 | +0.060, 7.34 |
| 8 | 3.0 | 40 | 20 | tip | +2.36 | +0.280 | 5.55 | +0.084, 7.50 |
| 8 | 3.5 | 50 | 15 | rigid | +2.20 | +0.925 | 14.46 | +0.127, 21.5 |
| 8 | 2.5 | 40 | 5 | rigid | +2.42 | +0.330 | 6.76 | (not given) |
| 6 | 2.0 | 60 | 15 | tip | +8.41 | +0.363 | 7.04 | no sustaining point |
| 6 | 2.5 | 50 | 15 | tip | +8.05 | +0.443 | 7.78 | " |
| 6 | 3.0 | 40 | 15 | tip | +7.86 | +0.352 | 7.01 | " |
| 6 | 3.5 | 60 | 20 | tip | +6.95 | +2.226 | 26.06 | " |
| 6 | 2.5 | 60 | 20 | tip | +8.00 | +0.786 | 10.74 | " |

Controls on the same footing:

| case | inc_trim | thrust | power |
|---|---|---|---|
| run 1, **opposite twist sign** | +2.56 | +0.493 | 16.38 |
| run 1, **root gap sealed** (roots 2 mm apart) | +0.92 | +0.416 | 5.80 |
| 6 m/s (2.5, 50, 15), root gap sealed | +5.02 | +0.816 | 9.99 |
| 6 m/s (2.5, 50, 15), free wake, inc 6 and 8 only | lift -0.07/-0.08 N, thrust -0.005 N, power -0.13/-0.16 W against the prescribed wake |

The full sweeps (inc -8..+8, to +12 at 6 m/s) are in `read.txt`.

### 4a. Static wing, V = 8, Phi = 0, A = 0 (unsteady solver, 30 chords of wake)

| | dCL/dalpha (/rad, on panel area) | zero-lift | line residual (CL) | vs Helmbold 4.383 (AR 5.43) | CDi pi AR / CL^2 |
|---|---|---|---|---|---|
| goe417a, open gap | 3.438 | -4.97 deg | 3.8e-3 | 0.784 | 1.54-1.67 |
| flat, open gap (2 points) | 3.474 | 0.00 | - | 0.793 | 1.65 |
| goe417a, sealed | 4.191 | -4.79 deg | 5.0e-3 | 0.956 | 0.96-1.03 |
| flat, sealed (2 points) | 4.231 | 0.00 | - | 0.965 | 1.01 |

Lift against inc is a straight line (residual under 0.005 in CL over +-8 deg; the
cambered wing's residual is the small-angle curvature of a rotated camber line,
the flat one's is zero).

### The root gap

With the gap open the two panels behave as two wings of aspect ratio 2.56 that
each shed a root vortex: 22% less lift slope than Helmbold and 1.5-1.7 times the
elliptic induced drag. Closing the gap recovers 0.96 of Helmbold and the elliptic
induced drag. The real hull lies between the two; UVLM without a body cannot say
where. This moves the trimmed answer more than any numerical setting: run 1
trims at +0.177 N / 4.37 W open and +0.416 N / 5.80 W sealed.

### 4b. Sensitivity, run 1 (V 8, 2.5 Hz, 40, 15, tip) at inc = 0 and 2

| change | lift | thrust | power |
|---|---|---|---|
| half the time step (200 steps/cycle, 8 chordwise) | +0.001 N (0.0%) | +0.011 N | +0.070 W (+1.6%) |
| 16 chordwise panels (step halves with it) | +0.06..0.08 N (+1.4..1.6%) | +0.002..0.005 N | +0.06 W (+1.4%) |
| 24 spanwise panels per wing | -0.06..-0.09 N (-1.5%) | -0.005 N | -0.06 W (-1.4%) |
| 5 cycles instead of 3 | +0.002 N | 0.0000 | 0.000 |
| free wake | -0.010..-0.014 N (-0.3%) | -0.001 N | -0.02 W (-0.5%) |

Numerical uncertainty of a trimmed point: about 2% in lift and power, about
0.01 N in thrust.

## What PteraSoftware does not represent here

- **No viscous drag at all** (no profile drag, no skin friction, no hull drag).
  The thrust column is thrust minus induced drag only.
- **No stall, no separation.** The 6 m/s trims sit at root incidence 7-8.4 deg on
  a 5.8%-camber section (about 12-13 deg above zero lift, mean CL 0.79, more on
  the downstroke); the UVLM keeps lift linear there. Those rows say what attached
  flow would give, not that the wing sustains.
- **Leading-edge suction is included in full, implicitly.** There is no suction
  term or suction-efficiency parameter in the source (no occurrence of the word).
  The force is Kutta-Joukowski, rho Gamma (v x l), on all four bound legs of every
  ring with the local velocity (free stream + induced + motion), plus the unsteady
  term along the panel normal. A force perpendicular to the local velocity on the
  leading bound vortex is the lattice equivalent of full leading-edge suction
  (the static sealed wing returns CDi = CL^2/(pi AR), which needs it). A thin
  plate at this Reynolds number (about 1.4e5) will not realise all of it, so the
  thrust here is an upper bound on the inviscid thrust of a thin section.
  This is my reading of `_calculate_loads`, not a statement in the docs.
- **Thin surface**: the mean camber line only; 3.4% thickness dropped.
- **No hull**: the root gap is open (bracketed by the sealed runs above).
- Wake: prescribed (convected with the free stream) except the labelled free-wake
  runs; vortex-core growth uses nu = 15.06e-6 (the default).

## Deviations from the brief

1. Sealed-gap runs (4a, run 1, one 6 m/s point) were added; not asked for. They
   are there because the open gap alone explains control 4a missing Helmbold by 22%.
2. The reference area for CL is the panel area (0.3400 m2 open gap); the brief's
   AR 5.43 is b^2 / (b c) with the gap counted. Both normalisations are printed.
3. Twist is piecewise linear over 6 segments (exactly linear at the 7 sections,
   straight-line interpolated mesh between them), not a continuous linear twist.
4. Averages are the plain mean over the last N steps of a 3N+1-step run, not
   PteraSoftware's own trapezoid `finalMeanForces_W`.
5. 4a uses the unsteady solver with static geometry (same mesh as the flapping
   runs), not the steady solver.
6. At 6 m/s the sweeps run to inc = +12 because the trims are at 7-8.4 deg.
7. Section 2/3 trims use the full 9-point (11 at 6 m/s) sweep, interpolated
   between the bracketing pair only.
8. Free-wake checks: run 1 at inc 0 and 2, and (6 m/s; 2.5, 50, 15) at inc 6 and 8.
9. `results.jsonl` was rewritten once to drop duplicate rows: a bug in the resume
   key (the output field `steps_per_cycle` overwrote the requested one) made the
   second `all` re-run 130 finished cases. The repeats were bit-identical
   (max difference 0). The field is now `steps_per_cycle_used`.

## Not verified

- The power formula is mine (above); it was not checked against an independent
  energy balance (e.g. wake kinetic energy flux).
- PteraSoftware's unsteady force for a twisting, flapping wing was not checked
  against a closed form (Theodorsen) here; only the static limit (4a) and
  mesh/step/wake sensitivity (4b) were.
- That the reflected wing's mesh mirrors the y > 0 wing exactly was inferred from
  the side force being 1e-16 N, not read from its vertices.
- The 0.1-chord line staying straight under twist was built in (the leading-point
  correction) and the tip 0.1-chord height peaks at 0.2255 m against 0.66 sin 20 = 0.2257 in the probe, but
  intermediate sections were not read back.
- With the gap "sealed" the roots are 2 mm apart and the flap hinges are at
  y = +-0.001, so the sealed flapping runs are a different machine (longer
  panels, 0.699 m), a bracket and not the brief's geometry.
