# Pass S2, pre-registration: what holds the wing's twist, against a structural solver

2026-11-17. Frozen before any model is built. The user, 2026-11-17: agreed
to check the wing's elasticity against a beam or plate solver. Tool:
CalculiX 2.23 (`/home/hundo/Projects/Dytiscidae/.cfd`), meshed with gmsh.

## 0. The question

Pass W7 found one stiffness that both cruises and takes off: 10 N m/rad at
each of the hinges between six panels of a 0.70 m wing, which is a
continuous torsional stiffness GJ of about 10 x 0.70/6 = 1.17 N m^2, and a
worst torque of about 1.0 N m held at take-off (0.55-0.69 in cruise). The
model's hinges and springs are not a structure. The user will build the
wing as a thin cambered plate. Does such a plate have that stiffness, at a
mass the 0.60 kg machine can carry (both wings 0.34 m^2), and if not, what
does?

## 1. Cases

One wing, cantilevered at the root, 0.70 m long, chord 0.258 m (constant
here), arc of 5.8% camber. Loads: a tip torque of 1.0 N m about the span,
read as tip twist and as GJ = T L / twist; and separately a uniform
pressure of 8.6 N/m^2 x 3 (the 0.60 kg weight on 0.34 m^2 is 17.3 N/m^2
over both wings; three times it as a manoeuvre) read as tip deflection.
Linear static. Materials, typical hobby stock (values are assumptions,
stated; the result is reported per unit of G and E as well so another
value can be put in): pultruded carbon rod and tube, E 120 GPa along the
fibre, G 4.5 GPa, 1600 kg/m^3; carbon sheet 0/90, E 60 GPa, G 4.5 GPa,
1550 kg/m^3; depron foam 6 mm, E 15 MPa, G 6 MPa, 40 kg/m^3.

- P: the plate alone, carbon sheet, thickness 0.3, 0.5, 1.0 mm (shell
  elements), root clamped.
- T: a round pultruded tube alone along the span, 6x4, 8x6, 10x8 mm.
- TP: tube 8x6 at 25% chord carrying the 0.3 mm plate (shell and beam
  joined).
- F: 6 mm depron in the same arc with the 8x6 tube.

Checks of the tool itself, before any case is believed: a round tube's GJ
against G pi (ro^4 - ri^4)/2 within 2%; a flat strip's against G c t^3/3
within 5% with the root free to warp; mesh halved, twist changes under 2%.

## 2. Predictions

- S1. The plate alone is far too soft in twist: 0.5 mm gives GJ about
  G c t^3/3 = 0.048 N m^2, a twentieth of what W7 asks, and 1 N m would
  twist it by more than a full turn in linear theory; the clamped root's
  warping restraint and the camber raise it by less than a factor 3. At
  1.0 mm (0.39 N m^2 open-section) it is still under 1.17 and the pair of
  wings would weigh 0.53 kg.
- S2. The 8x6 tube alone gives 1.24 N m^2 (within 10%), at 25 g a side;
  6x4 gives 0.46, 10x8 gives 2.6. Shear stress at 1 N m in the 8x6 tube is
  about 15 MPa.
- S3. With the tube, the 0.3 mm plate adds under 15% to GJ; the pair of
  wings (two tubes, two plates) weighs 0.21 kg, a third of the machine.
- S4. Depron on the tube adds under 5% to GJ and weighs 0.05 kg a side
  with the tube.
- S5. Under three times the weight's pressure the 8x6 tube alone (the
  plate carried by it, giving no bending stiffness of its own counted)
  deflects at the tip by 5-15% of the wing's length.

Falsified by the stated numbers. What follows from each: if S1 holds, "a
thin cambered plate" cannot be the whole wing and the twist has to be
carried by a spar, which also means the twist the cruise needs (15 deg at
the tip, 40 at 6 m/s) has to be driven through that spar or the plate has
to be free to turn on it; that design question is put to the user, not
settled here.

## 3. Not done here

Flutter, fatigue, the joint between the wing and the shoulder, buckling of
the thin plate under the chordwise load, the film or skin's own tension,
and the nonlinear twist of a soft plate.
