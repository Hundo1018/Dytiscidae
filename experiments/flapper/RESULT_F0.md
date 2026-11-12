# F0 result — 2026-11-12

`PREREG_F0.md` was frozen and committed first. Reproduce:
`PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/hover_map.py experiments/flapper/results/hover_map.jsonl --jobs 8`
(280 rig runs, under two minutes; no row non-finite).

## The map

Mean vertical force of one wing pair, grams-force, stroke 130 deg peak to
peak, feather amplitude 45 deg, aspect ratio 3 per wing, body upright and
held:

| wing length | 5 Hz | 8 Hz | 12 Hz | 16 Hz | 20 Hz | 25 Hz | all-up mass of this build |
|---|---|---|---|---|---|---|---|
| 0.10 m | 1.3 | 3.4 | 7.6 | 13.5 | 21.1 | 33.0 | 68.6 g |
| 0.15 m | 6.7 | 17.1 | 38.4 | 68.5 | 107 | 167 | 70.8 g |
| 0.20 m | 21.1 | 54.1 | 122 | 217 | 339 | 529 | 73.7 g |
| 0.25 m | 51.5 | 132 | 297 | 529 | 827 | 1293 | 77.1 g |
| 0.30 m | 107 | 274 | 616 | 1098 | 1716 | 2681 | 81.1 g |

The all-up mass is what the pipeline builds for this hull with the avionics
set to 5 g in this experiment (the package's 0.30 kg is sized for the 15 kg
reference and by itself is four times this machine). Lift over that weight
passes 1.5 at 0.20 m and 12 Hz (1.65), at 0.25 m and 8 Hz (1.71), and at
0.15 m and 20 Hz (1.51). A 160 deg stroke adds about 47%.

## The seven predictions

1. *"One feather sign lifts and the other pushes down by a like amount
   (within 25%)."* **Held**: +1.192 N and -1.017 N.
2. *"... log-log slopes 1.7-2.1 and 3.5-4.2."* **Held**: 2.0 against
   frequency and 4.0 against wing length at every line checked.
3. *"Solver over hand estimate: between 0.5 and 1.3 everywhere on the map."*
   **Held**: 0.77 to 0.95, median 0.88, over 270 points.
4. *"There is a point with wing length <= 0.25 m and frequency <= 16 Hz that
   lifts 100 gf, and one with wing length <= 0.15 m and frequency <= 25 Hz
   that lifts 30 gf."* **Held**: 132 gf at 0.25 m and 8 Hz; 38 gf at 0.15 m
   and 12 Hz.
5. *"Of the three feather amplitudes, 45 deg lifts most at most map
   points."* **Held**: 59 of 90 (55 deg at the other 31, 35 deg at none).
6. *"Inert arms: wings held still read |force| under 1e-6 N; flapping with
   feather 0 ... under 5% of the lifting case."* First half **not held as
   written**: 1.2e-5 N, which is 1e-5 of the lifting force at that point and
   was not traced. Second half **held**: 5e-7 N.
7. *"Halving the time step moves the force by under 5% at 25 Hz."*
   **Held**: -0.1% at three wing lengths.

## What it settles, and what it does not

- The simulator's fluid model gives a flapping wing in hover about what
  quasi-steady blade-element theory gives, and it is not sensitive to the
  time step at these frequencies. So the envelope can be read off the map:
  **wing length 0.20-0.25 m (span 0.42-0.52 m), 8-12 Hz, all-up mass
  70-150 g**, with a lift margin of 1.6 or more and the stroke not at its
  limit. Lower frequency is kept on purpose: it is what a real drive has to
  reach in stage 2.
- It says nothing about flight: the body was held. Stability, control and
  the wing's inertial reaction on a 75 g body come with the free hover.
- The model is quasi-steady: no leading-edge vortex dynamics, no wake
  capture, no clap and fling. The agreement with the hand estimate is an
  agreement between two quasi-steady reckonings, not with an experiment.
- Rigid wings. As built, the membrane wing fails two of the package's
  structural checks (root bending -0.54, sweep load in water -0.96): the
  spar is to be sized before the water work.
- The 0.30 kg avionics constant has to become scale-aware in the package
  before anything at this scale goes through the normal pipeline.
