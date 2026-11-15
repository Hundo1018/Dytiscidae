# L1 pass 1j: the one-way flap as a built panel. Frozen 2026-11-15

The user: after the power stroke the wing should return in its least-drag
shape. Pass 1e tested that by scaling the fluid strips' chord in time and
found 3.1-3.6 times the fixed wing's force, but its artefact check failed
(negative mean power at 10.2% of points), so the gain was reported unproven.
It named the test that would settle it: build the area change into the
wing. This is that test. Written before the grid below was run.

## What is built (`flapwing.py`, `rig.run(..., one_way_flap=...)`)

- Each wing is cut along the span at 0.4 of its chord. The front 40% stays on
  the wing body. The rear 60% is a body of its own on a hinge along the cut.
- The flap can swing one way, up to 80 deg, and is stopped flat against the
  front part the other way (a joint limit). A weak spring (0.005 N m/rad)
  and damping (0.002 N m s/rad) return it.
- Its fluid strips ride on the flap body. Area, incidence and added mass
  follow its real motion; no strip changes size in time.
- Mass is split by chord. The flap feels the prescribed stroke and feather
  through (M a) taken from the current pose every step.
- Arms: welded (the same split wing with no hinge: the control), flap free to
  swing up, flap free to swing down ("up" is the level machine's +Z on both
  wings).

## Disclosed before freezing

- **A direction bug, fixed.** The mirrored wing's normal points the other way
  in the world, so in the first probe the two wings' flaps opened on opposite
  half-strokes and "up" and "down" read identically (-2.02 N both). The
  opening direction is now set per wing.
- **`probe_flap.py`, after the fix.** Pass 1e's probe point: wing 0.4,
  0.3 m/s, stroke 85 deg, 0.5 Hz, body level.

  | feather | undivided wing | welded | flap up | flap down |
  |---|---|---|---|---|
  | 30 deg | 5.92 N, 11.3 N m | 6.71 N, 10.3 N m | -1.79 N, 5.3 N m, 0.65 W | -2.24 N, 12.9 N m |
  | 15 deg | 0.69 | 2.34 | -2.47 | -2.86 |
  | 0 deg | -4.18 | -2.34 | -4.71 | -4.99 |

  The flaps opened 36-53 deg. Pass 1e's chord law read 3.86 N at the
  feather-30 point.
- **`probe_row.py`.** Hull vertical, so the wings sweep fore and aft facing
  the 0.3 m/s stream; wing 0.4, stroke 85 deg, no feather.

  | frequency | welded | flap up | flap down |
  |---|---|---|---|
  | 0 | -16.06 N | -11.64 N (the stream holds it 50 deg open) | -15.86 N |
  | 0.5 Hz | -17.58 N, 11.0 N m | -10.35 N, 6.2 N m | -17.58 N |
  | 1.0 Hz | -31.18 N, 31.7 N m | -26.84 N, 27.4 N m | -37.40 N |

  A spring ten times stiffer read -12.79 N at 0.5 Hz, flap up.
- No rowing point in still water was seen: P4 below is a prediction made
  blind.
- The per-step coupling was checked on pass 1h's cheapest passive point: it
  reads the same as the old coupling (1.631 weights, 69.0 W, 13.13 N m).

## 0 Question and decision

Does a flap that opens by itself on one half of the beat give more net
forward force within the torque caps than the same wing with the flap welded?
Decision: whether the water wing carries a per-beat flap, or a fold set once
for water (passes 1c-1d) is the whole of the shape change.

## 1 Rivals

(a) The up-down stroke with a feather is lift-based: both halves make
thrust, and a flap that lets go on one half throws that half away. Pass 1e's
gain was the solver's, not the wing's.
(b) The flap cuts the torque of the half that pays least, so more force fits
under the cap, as pass 1e read.
(c) The flap is a one-way valve, and a valve needs a drag-based (rowing)
stroke to matter. It gives thrust where a welded paddle gives none, but not
in the posture and stroke of passes 1b-1e.
(d) The model cannot say: strips do not see each other, so the open flap is
not in the front part's wake and there is no flow through the gap.

## 2-3 Blocks, gates and predictions

**Block A, pass 1e's grid.** Body level, 0.3 m/s, wings 0.4 and 0.6 of the
length, stroke 40 and 85 deg, 0.25, 0.5, 1.0, 1.5 Hz, feather 0, 10, 20,
30 deg, three arms; and each arm at rest. Caps as 1e: shoulder torque
<= 15.8 N m and crank-equivalent <= 12 N m.
- P1 (rival a against b): for each wing, the better flap arm's best force
  under the caps is **less** than the welded arm's best. Pass 1e's gain is
  not reproduced. Falsified if the ratio exceeds 1.2 for either wing.
- P2 (control on the split): the welded arm's best under the caps is within
  25% of pass 1e's fixed wing (5.92 N and 3.28 N).
- P3 (energy): mean power is negative at fewer than 2% of the flap points.

**Block B, a rowing stroke.** Hull vertical, no feather, wings 0.4 and 0.6,
stroke 40 and 85 deg, 0.25, 0.5, 0.75, 1.0 Hz, three arms, in still water and
at 0.3 m/s; each arm at rest at 0.3 m/s.
- P4 (rival c, blind): in still water at wing 0.4, stroke 85 deg, 0.5 Hz:
  - welded: |Fx| < 0.5 N;
  - flap up: +2 to +10 N;
  - flap down: -2 to -10 N, within 20% of flap up in size.
- P5: at 0.3 m/s no arm has Fx >= 0 at any point within the caps, and flap
  up reads a higher Fx than welded at every point.

**Block C, the flap itself.** The P4 point with the cut at 0.2, 0.4, 0.7 of
the chord and the spring at 0.005 and 0.05 N m/rad. Descriptive; no
prediction.

## 4-5 Measurement and design

`l1j_flap.py`, read by `l1j_read.py`. 198 + 102 + 6 = 306 points, 3 beats
each with the first dropped, one pass, no tuning after reading. Predicted 25
minutes on 6 processes. A point that is not finite is excluded.

Not covered:
- rival (d): no wake or gap flow between the parts;
- the flap as anything but a rigid plate; impact at the stop;
- power lost in the flap's hinge and stop is not charged to the motor
  separately (the stroke axis's fluid torque times its rate is);
- a rowing stroke with the hull in a streamlined posture, which needs a
  stroke plane the reference body does not have;
- the centre-of-pressure law (off here, as in passes 1b-1e, so the
  comparison with 1e is like for like);
- free body; air.
