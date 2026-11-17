# Pass X2, pre-registration: an outside viscous solver as a witness for the section

2026-11-17. Frozen before any case is built or run. The user, 2026-11-17:
agreed to install a viscous CFD solver to check the low-speed and take-off
regime, where the strip solver has static tables and no dynamic stall, and
where pass X1's inviscid lattice cannot go.

Installed (the user's approval, conda-forge, in its own environment outside
the repository, `/home/hundo/Projects/Dytiscidae/.cfd`, 1.5 GB): OpenFOAM
v2412, CalculiX 2.23, gmsh 4.15. Nothing has been run with them but
`simpleFoam -help`.

## 0. What this can and cannot be

Two-dimensional, incompressible, Reynolds-averaged. It can test a section:
static lift and drag, and a section that plunges and pitches as one station
of the wing does. It cannot test the wing: no span, no tip, no spanwise flow
(which is what holds a leading-edge vortex on a revolving wing), no hull.
Two-dimensional unsteady RANS is known to over-read lift once the flow is
massively separated, and transition at Re 1e5 is modelled, not resolved. So
the solver has to qualify first, against the tunnel data the repo holds,
and what it is trusted for afterwards is only what it qualified on.

## 1. Stage A: qualification, static

Goe 417a (the tunnel model's outline, `ptera/goe417a_coords.txt`; check it
against `sections.coordinates("goe417a")`), Re 1e5, alpha = -2, 0, 2, 4, 6, 8,
10, 12, 14, 16, 20 deg. Two closures, both reported: k-omega SST fully
turbulent, and its transition form (kOmegaSSTLM), freestream turbulence
0.1%. Steady where it converges, time-accurate and averaged where it does
not. Mesh: wall-resolved (y+ about 1), far field at 20 chords or more; at 4
and 12 deg a second mesh with half the spacing; stated cell counts.

Against `sections/goe417a_Re100k.csv` (lift and drag to 11.65 deg) and
`goe417a_Re100k_lift.csv` (lift to 19.7 deg).

- A1. Mesh: lift at 4 and 12 deg changes by under 3% on the finer mesh, drag
  by under 8%. If not, the fine mesh is the mesh and it is checked once more.
- A2. Transition closure, 2-10 deg: |CL - measured| <= 0.10 at every angle;
  CD within 40% of measured at every angle.
- A3. Transition closure: maximum lift within 0.15 of the measured 1.355 and
  its angle within 2 deg of 13.5.
- A4. At 0 deg (measured 0.195, a lower-surface separation) either closure
  may be off by up to 0.2: predicted worst point, gates nothing.
- A5. Fully turbulent closure: predicted to miss A2's drag (too high below 6
  deg) and to hold its lift.

**The solver is qualified for a cambered section up to stall if A1-A3 hold
with one closure; that closure is used in stage B.** If none holds, stage B
still runs with the nearer one, and every number from it is labelled
"unqualified".

Also read, gating nothing: a flat plate 3.1% thick with semicircular edges
(the UIUC plate's "baseline" edge shape is not in the repo: an assumption,
stated) at Re 1e5, alpha = 2, 5, 8, 10, 15, 20, 24 deg. Predicted: the lift
at 5 deg is 0.50-0.56, at or under 2 pi alpha, and not the tunnel's
0.60-0.65; from 15 deg the two-dimensional mean lift is 0.9-1.3 and is not
evidence either way for a wing (see 0).

## 2. Stage B: the section in motion

One station, 70% of the wing's length from the shoulder, of two stored
cruise points (`results/W8_cruise.jsonl`, `W8b_cruise.jsonl`, treatment
"extend"):

- B8: 8 m/s, 2.5 Hz, stroke 40 deg, tip feather 15 deg, offset -2.96 deg
  (7.24 W, the attached case the strip solver is trusted on);
- B6: 6 m/s, 3.5 Hz, stroke 60 deg, tip feather 40 deg, offset -11.18 deg
  (16.4 W, where the outer wing passes static stall every beat).

The station's plunge is its arc's vertical travel taken as straight
(amplitude = radius x half the stroke), its pitch is the feather the rig
gives that station (read off `rig.twist_turn`, not assumed here), the
stream is the flight speed, the chord is the wing's there. The same
two-dimensional motion is run through the strip solver on a bare strip
(`physics_extremes/run.py`'s heaving rig with pitch added; section
`goe417a+lift+extend`; inflow off, unsteady on) and through the outside
solver (whole mesh moved rigidly, five beats, the last three averaged).

Read: beat-mean force normal to the stream and along it, as coefficients on
the station's chord and the stream's dynamic pressure; beat-mean power
(force times plunge rate plus moment times pitch rate) as a coefficient;
the lift's history over a beat.

- B1. B8: mean normal force within 15% of the strip solver's; mean
  streamwise force within 0.03 in coefficient; power within 20%.
- B2. B6: the outside solver's mean normal force is **higher** than the
  strip solver's by more than 15% (dynamic stall holds lift past the static
  peak), and its power is higher too. Falsified by a difference under 15%
  or of the other sign.
- B3. B6: the lift history shows a peak above the static maximum 1.355 on
  the downstroke.

What each outcome would mean: B1 failing puts the strip solver's unsteady
terms in doubt even where the flow is attached (or the outside solver's
moving-mesh set-up: a still-air plunge check, added mass against
pi rho c^2/4 within 10%, is run first to tell them apart). B2 holding says
the 16 W at 6 m/s is pessimistic or optimistic by the sign and size read;
it does not move the number, because one station of a wing is not the wing.

## 3. Not done here

Three dimensions, the hull, the root gap, a wing that flaps about a
shoulder, take-off at rest (no stream: a different case), water. Any of
them is a later pass with its own frozen text.
