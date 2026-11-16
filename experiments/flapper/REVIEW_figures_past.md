# Earlier passes looked at again through their drawings (2026-11-16)

Asked for by the user: every earlier mechanism test checked by eye for
anything unreasonable, without necessarily re-running it. Each drawing is made
by `draw.py past` from the pass's own body definition, through the same
builder the pass used, with a table of what the body asked for against what
was built. All ten were opened and read. Nothing here was re-measured; what a
finding does to a result is stated only where a later pass measures it.

| drawing | passes | checked by eye | finding |
|---|---|---|---|
| `figures/past_F0_F3_W_small.png` | F0, F0b/c, F3, W1, W2 | wing 0.25 x 0.083 m, feather axis on the leading edge, stroke axis along the hull | **span 0.50 m against 0.536 m**: roots on the centre line (finding 1) |
| `figures/past_F1a_hand.png` | F1a, F1a-2 | outer half on a wrist at mid-length, wrist axis along the span on the leading edge | as intended; span as above |
| `figures/past_F2_layoutA.png` | F2, F3b | thumbs at 0.25 of the wing, half above and half below it; tail at the hull's rear | span as above; the tail is a 9 x 1.2 cm blade (finding 3) |
| `figures/past_F2_layoutB.png` | F2-F4, T1 | two wing pairs at 0.2 and 0.75 of the hull, a thumb through each wing | **span 0.42 m against 0.462 m**; the hind pair sits in the fore pair's wake (finding 4) |
| `figures/past_R0_eflap.png`, `past_R0_raven2.png` | R0 | rectangular wings, area and mass as reported | **span 1.42 and 1.25 m against the reported 1.50 and 1.33 m** (finding 2) |
| `figures/past_L1a_reference.png` | 1a, 1b, 1e-1i, 1k | area 0.340 m2, chord 0.258 m, mass 0.60 kg | **span 1.32 m against 1.40 m** (finding 1) |
| `figures/past_L1g_axis0.3.png` | 1g-1i | feather axis 0.3 chord behind the leading edge; the wing turns about it at the stroke ends | as intended |
| `figures/past_L1d_fold0.6x0.6.png` | 1c, 1d | shorter, narrower wing; the leading edge stays at the shoulder, the rear is what goes | as intended; span 0.79 m against 0.87 m |
| `figures/past_L1j_flap.png` | 1j | front 40% on the feather axis, rear 60% on a hinge along the span at 0.4 chord | as intended; span 0.53 m against 0.61 m |

## Findings

1. **Every pass so far built its wings with their roots, and the stroke
   axis, on the hull's centre line.** The genome attaches a part on its
   parent's axis. Each body's area and chord are what was asked for; its span
   is short by the hull's diameter: 7% on the small flapper, 9% on layout B,
   6% on the reference machine. The inner part of each wing is inside the
   hull, and the two wings of a pair overlap at the centre line. Each strip
   is as far from its stroke axis as designed, so the flapping loads should
   be close; the span the induced flow is spread over is not. Pass 1n
   measures this on the reference machine. F0-F4, W1, W2 and T1 are not
   re-run.
2. **Pass R0 did not reproduce the two published machines at their reported
   spans.** Its text says "rectangular wings at the reported span and area";
   the built spans are 1.42 m (E-Flap, 1.50 reported) and 1.25 m (Robo Raven
   II, 1.33). R0 is the calibration the L1 gates lean on (its band on lift).
   To be re-run on the seated fixture. The rectangular planform and the
   missing tail were declared in R0 and are as drawn.
3. Layout A's tail (F2, F3b) is a flat blade 9 cm long and 1.2 cm wide lying
   along the hull's axis, 0.0011 m2, 2.6% of the wing area, with its chord
   wholly to one side of the centre line (6 mm off). It is the ground prop F2
   meant, not an aerodynamic tail. The 6 mm offset is not intended; its
   effect is not measured and is expected to be small.
4. Layout B's hind wings (F2-F4, T1, the body that flew and took off) begin
   1.7 cm behind the fore wings' trailing edges, at the same height. Strips
   do not see each other, so the hind pair flies in undisturbed air in the
   model. F4's flight and T1's take-off rest on that. Declared nowhere in
   those results as far as searched; recorded here.
5. **Several mechanisms the results name are not parts of any model.** There
   is no cam, no crank and rocker, no fold, no return spring housing and no
   wrist motor in the drawings because none is built: "cam-driven" is a
   feather angle prescribed as a function of the stroke, the fold is a
   shorter wing, the crank is a formula applied to the torque. Their mass,
   friction, backlash and the space they take are not in any result. The
   results say so pass by pass; the drawings make it visible in one place.
6. Thumbs pass through the wing, half above it. This is how F2 built them
   (one strut per wing serves both the upright and the mirrored side).

## Not drawable

W1 and W2 (the wake model: no mechanism), 1e (a chord scale on the fluid
strips of 1c's body), 1h (a centre-of-pressure law), 1l and S1 (drawn in S1).
