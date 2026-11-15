# L1 pass 1i: the feather driven from the stroke by a cam. Frozen 2026-11-15

Passes 1f-1h found that a spring-hinged feather alone does not meet the
gates. Here the feather is tied to the crank by a cam:
- it moves as pass 1a prescribed it, psi0 - Psi cos(wt), a quarter beat
  ahead of the stroke;
- the same motor pays for it;
- the offset psi0 is a slow trim, set by a small servo that moves the cam
  follower (TYPED, not modelled: about 10 g, not drawing beat power).

The pitching-moment model of pass 1h is on, under both laws, because the
feather's torque about its hinge is what the cam has to supply. Written
before the grid below was run.

## A new torque metric, and why

Passes 1a-1h judged torque at the shoulder: the fluid stroke torque peak
combined with the wing's peak stroke inertia (`l1a.finish`). Those two peaks
fall at different times, and the inertia peak falls at reversal, where a
crank's mechanical advantage is unbounded.

For a motor turning a crank at a steady rate w, the exact torque at the crank
is the instantaneous power of every driven axis, fluid and inertia, over w.
It is reported here as the shoulder torque a K1 rocker at 85 deg would show,
T / sin(42.5 deg), so the 12 N m cap keeps its meaning. It assumes:
- an ideal, rigid, lossless linkage;
- a steady crank, which needs enough motor or flywheel inertia;
- the sinusoidal stroke and feather of the rig, which a real crank-rocker
  only approximates.

Because this changes the gate, both metrics are reported for every point.
Pass 1h's eight cheapest passive take-off points and four cheapest
sustaining cruise points are re-measured under it, in the same run.

## Disclosed before freezing (`probe_cam.py`)

| point | law | hinge h | lift | power | shoulder metric | crank, stroke only | crank, with cam | feather torque peak |
|---|---|---|---|---|---|---|---|---|
| take-off, 4 Hz, stroke 85, Psi 60, body 90 | none | 0 | 1.70 weights | 72 W | 11.0 N m | 6.2 | 9.8 | |
| same | plate | 0 | 1.70 | 94 W | 11.9 | 6.6 | 12.3 | 2.7 N m |
| same | plate | 0.1 | 1.46 | 83 W | | 5.5 | 9.2 | |
| same | plate | 0.2 | 1.24 | 76 W | | 5.1 | 9.4 | 3.0 N m |
| same | insect | 0, 0.1, 0.2 | within 0.7 N m of plate | | | | | |
| cruise, 8 m/s, 2.5 Hz, stroke 40, Psi 20, psi0 -8 | | | -1.2 N net | about 0 W | | | crank 1.6-1.8 | |

- The first row is pass 1a's own point, reproduced exactly with no
  centre-of-pressure law (1a: 1.70 weights, 72 W, 11.0 N m).
- Moving the hinge axis changes lift even with the feather prescribed. The
  quarter chord's plunge from pitching about the hinge changes with the
  hinge position.

## 0 Question and decision

Does one motor that drives the stroke and, through a cam, the feather meet
the take-off and cruise gates? Decision: whether K1 becomes K1 with a cam,
the reference drive for L1's next passes (free body, then water and land).

## 1 Rivals

(a) A cam gives what 1a's prescribed feather gave, at a modest extra
torque. (b) The feather's hinge moment through the same motor pushes the
crank over its cap. (c) The gates were never about the feather: under the
crank metric the passive hinge of 1h already meets take-off, and the
earlier take-off failures were the metric's.

## 2-3 Gates and predictions

Gates: as 1a, with the crank metric (cam included) <= 12 N m in place of
the shoulder metric.
- Take-off: >= 1.6 weights, <= 12 N m, aero power <= 150 W.
- Cruise: sustains, <= 41 W, <= 12 N m.

Predictions:
- P1 (take-off with cam): met under both laws, narrowly, at h = 0.1 or
  0.2. Falsified if no point meets it.
- P2 (cruise with cam): met under both laws. Falsified if no point meets it.
- P3 (cam cost, rival b): at each law's cheapest gate-meeting take-off
  point, the crank metric with the cam exceeds the stroke-only crank metric
  by 30-90%.
- P4 (rival c):
  - all 8 re-measured passive take-off points read <= 12 N m under the crank
    metric, so the passive hinge meets the take-off gate under it;
  - the 4 passive cruise points still cost more than 41 W.
  - Falsified if any of the 8 take-off points reads above 12.
- Control: each re-measured passive point reproduces its pass-1h lift and
  shoulder metric to 0.01 weights and 0.1 N m (same code path).

## 4-5 Measurement and design

`l1i_cam.py`:
- Take-off: body 45, 60, 75, 90 deg; 3.0-5.0 Hz in steps of 0.5; Psi 30-70
  in steps of 10; h 0, 0.1, 0.2; law plate and insect. 600 points.
- Cruise: 6 and 8 m/s; 2.5, 3.0, 3.5, 4.0 Hz; stroke 40, 60, 85; Psi 0,
  10, 20, 30; the same h and laws; psi0 trimmed to the weight. 576 points.
- Comparison arm: 12 passive points.

One pass, no tuning after reading. Predicted 60 minutes on 6 processes.
A point that is not finite is excluded from every gate.

Not covered:
- free body;
- a real crank-rocker's non-sinusoidal motion;
- cam friction and the follower's mass;
- motor torque-speed limits beyond the cap;
- the negative-torque part of the beat (reported as the crank minimum, not
  gated).
