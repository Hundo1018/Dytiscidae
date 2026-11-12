# F2 result — 2026-11-12

`PREREG_F2.md` was frozen first (commit 65b0f0a) and is not edited. MuJoCo 3.15,
CPU, time step 1 ms. No learning, no optimiser, no fitting: gaits, bodies and
servo gains were set by hand on seeds 100-103 (`trials_F2.jsonl`, 257 rollouts)
and scored once on seeds 0-7 (`results/landwalk.jsonl`, 58 rows, none dropped,
no `bad_qacc`, no failure). Code: `landwalk.py` (sha256 `3f85a218d7c299fb…`,
frozen before the scored rows), `body.py`, `rig.py`.

**Outcome.** Both layouts walk on their wings on 8 of 8 seeds. By the frozen
rule, A walking decides it: **the machine has one pair of wings and a tail.**
A does it as a mudskipper does, on both thumbs together with the tail trailing,
and its margin on wing share is thin (0.76-0.78 against the bar of 0.7). B does
it with nothing but thumbs on the ground. Predictions: 1 held, 2 refuted, 3
held, 4 held, 5 held, 6 refuted.

## 1 The parts added, and the built-geometry check

`body.py::flapper` gained keyword options that are off by default: `thumb_length`
(with `thumb_radius`, `thumb_u`), `tail_length` (with `tail_chord`, `tail_radius`,
`tail_motor_mass`, `tail_range`, `tail_el`, `tail_roll`) and `hind_u`. The
default body is unchanged: `rig.run` on it at (V 8, f 6, stroke 100, feather 20,
psi0 -8.0) gives 4.1008 W with the committed `body.py` and with mine. The
3.9878 W in `results/cruise_trim.jsonl` is `flight_modes.rig`'s number (it builds
`wing_chord = R/3`, `stroke_range = 1.45`), and that path still gives 3.9878.

| layout | body options | all-up | added parts (mass each, from the built model) |
|---|---|---|---|
| bare A wing | defaults | 77.25 g | — |
| **A** | `shoulder_u 0.6, thumb_length 0.045, thumb_u 0.25, tail_length 0.09, tail_chord 0.012` | 82.49 g | 4 thumb struts 0.32 g (two per wing, see below); tail paddle 0.09 x 0.012 m on a pitch hinge, 3.81 g with its 3 g motor |
| bare B wing, one pair | `wing_length 0.21, wing_chord 0.07` | 74.30 g | — |
| **B** | `wing_length 0.21, wing_chord 0.07, shoulder_u 0.75, hind_u 0.2, thumb_length 0.045, thumb_u 0.30` | 101.28 g | second wing pair 2 x 11.82 g (with motors); 8 thumb struts 0.32 g |

Checked on the built model (`landwalk.py geometry`, hull frame: x to the nose, z up,
origin at the tail end of the 0.16 m hull):

- **Thumbs.** The mirrored wing is the first turned half a turn about the body's
  long axis, so one edge that points down on the left wing points up on the
  right (found by printing: centres at z -0.02 and +0.02), and `reflect=True` on
  the edge only duplicates the strut in place. Two edges at azimuth +-pi/2 are
  used: every wing carries one strut below it and an idle one above. Thumb tips:
  A (0.096, +-0.0625, -0.045); B fore (0.120, +-0.063, -0.045), hind (0.032,
  +-0.063, -0.045).
- **Tail.** Elevation -pi/2 puts it along -x from the hull's tail end; its hinge
  axis is (0, 1, 0), so it pitches. The paddle's chord sits a quarter chord to one
  side of the hinge line (centre at y -0.003 with the 0.012 m chord; a 0.06 m
  chord put it 0.015 m off centre and the body yawed). It is a ground strut here,
  not a tail sized for flight or swimming.
- **Second pair.** `hind_u 0.2` and `shoulder_u 0.75` put the shoulders at x 0.032
  and 0.120 so the two 0.07 m chords do not overlap (at 0.65 and 0.2 they do, by
  0.013 m).
- Signs are read off the model, not assumed: for every wing a +0.2 rad step on
  each shoulder axis says which way the thumb tip moves (stroke sign -1, feather
  sign -1 on all wings; tail sign -1).
- Centre of mass: A x 0.0725 (2.4 cm behind the thumbs), B x 0.0695 (between the
  pairs).

Stage 1 settings on the experiment's own model: position servos with stiffness
30 / 20 / 5 N m/rad and damping 0.10 / 0.03 / 0.01 (stroke / feather / tail), no
force limit. The built joints carry 0.0226 N m s/rad of drive damping and 1e-4
kg m2 of armature, which the first gains (6 / 1.5 / 1) could not follow. The pack
is made unlimited (energy is not counted; an ideal servo's draw ended the first
rollouts as "battery exhausted"); the mass is the 0.6 Wh pack's. Targets are
rate-limited to 60 rad/s.

## 2 The probes of wing share and hull contact

`landwalk.py probes`, last 0.5 s of each hold:

| layout | probe | ground_support | wing share | tail share | hull contact | bodies the ground pushes on (N) |
|---|---|---|---|---|---|---|
| A | resting on the hull (wings and tail raised 0.6 rad) | 0.9989 | 0.0 | 0.0 | 1.0 | hull 0.767 |
| A | propped (stance pose) | 0.9999 | 0.842 | 0.158 | 0.0 | thumbs 0.326, 0.322; tail 0.12 |
| A | airborne | 0.0 | missing | missing | 0.0 | none |
| B | resting on the hull (wings raised 0.6 rad) | 0.9993 | 0.2304 | 0.0 | 1.0 | hull 0.732; hind wings 0.089, 0.129 |
| B | propped (stance pose) | 1.0001 | 1.0 | 0.0 | 0.0 | thumbs 0.17, 0.17, 0.306, 0.306 |
| B | airborne | 0.0 | missing | missing | 0.0 | none |

A propped reads its true share: the tail carries 16%. B at rest does not read 0:
facing up the 0.12 slope, the hind wings' trailing edges lie on the beach and
carry 23%, which is their true share (0.218 of 0.950 N). Hull contact reads 1
there, as it should.

Wing share is the vertical ground force entering through wing and thumb bodies
over all of it, summed over the window; hull contact is the share of steps with
a hull geom in contact. The window is the late half of the walk phase, and the
worse of the pair's two halves is reported (E0b's convention); the whole walk
phase is given in brackets.

## 3 Rows

### Walk, seeds 0-7 (unmodified `evaluate.run_segment`, antipodal pair)

| layout | seed | land score | ground_support | wing share | tail share | hull contact | bad_qacc | walks |
|---|---|---|---|---|---|---|---|---|
| A | 0 | 0.589 | 1.011 [1.006] | 0.771 [0.778] | 0.124 | 0.038 [0.024] | 0 | yes |
| A | 1 | 0.643 | 1.010 [1.006] | 0.768 [0.763] | 0.132 | 0.036 [0.029] | 0 | yes |
| A | 2 | 0.762 | 1.000 [1.006] | 0.762 [0.759] | 0.137 | 0.033 [0.030] | 0 | yes |
| A | 3 | 0.763 | 0.994 [1.006] | 0.760 [0.757] | 0.138 | 0.030 [0.033] | 0 | yes |
| A | 4 | 0.844 | 0.999 [1.003] | 0.771 [0.752] | 0.137 | 0.027 [0.041] | 0 | yes |
| A | 5 | 0.449 | 1.012 [1.007] | 0.775 [0.780] | 0.116 | 0.052 [0.030] | 0 | yes |
| A | 6 | 0.792 | 0.997 [1.005] | 0.763 [0.760] | 0.138 | 0.025 [0.029] | 0 | yes |
| A | 7 | 0.536 | 0.999 [1.005] | 0.769 [0.786] | 0.135 | 0.051 [0.028] | 0 | yes |
| B | 0 | 0.377 | 1.002 [0.999] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 1 | 0.539 | 0.999 [0.998] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 2 | 0.637 | 0.995 [0.998] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 3 | 0.758 | 0.995 [0.997] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 4 | 0.929 | 0.999 [0.998] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 5 | 0.600 | 1.004 [0.997] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 6 | 0.831 | 0.999 [0.997] | 1.000 | 0 | 0.000 | 0 | yes |
| B | 7 | 0.636 | 1.003 [0.999] | 1.000 | 0 | 0.000 | 0 | yes |

A: 8 of 8 with either window. B: 8 of 8. In A the wings carry 76-78%, the tail
12-14% and the hull the remaining tenth: A's gait sets the hull down for a moment
in every stride (3-5% of the steps). 18-38% of the steps in these windows carry
no ground force at all (A 0.20-0.32, B 0.18-0.38): both gaits leave the ground
between contacts.

The gaits (`GAITS` in `landwalk.py`; stroke lowers the thumb, the feather sweeps
it):

- **A crutches.** Both thumbs together at 1.5 Hz, stance 85% of the cycle, wings
  held raised 0.27 rad so the hull rides about 5 mm up; in the swing the thumbs
  lift (to 0.45 rad), the hull sets down, the thumbs return and press again
  (lift, then return, then lower). The tail is held 0.35 rad down. It turns on
  the spot by sweeping the two thumbs opposite ways: 85 to 118 degrees in 4 s.
- **B crawls.** One thumb in the air at a time (phases 0, 0.5, 0.75, 0.25), 1.5
  Hz, stance 80%, stroke 0.10 down and 0.25 up.
- Steering is E0b's with the fade made a switch: within `e_go` of the heading (or
  of its opposite, walking backward) full stride with a proportional turn; beyond
  it, turn on the spot. The land score is `clip(along) * exp(-across^2)` in units
  of 0.04 m/s, so walking fast off the heading is paid almost nothing and the
  turn rate is what the score buys.
- No step in the first 0.15 s of a rollout (the segment drops the body 7 cm).

### Null arms, 8 seeds each

| layout | propped (every actuator at the stance pose) | heading frozen at the first one seen |
|---|---|---|
| A | 0.0000 on 8 of 8 | 0.0000 on 8 of 8 |
| B | 0.0000 on 8 of 8 | 0.0000 on 8 of 8 |

The propped arm reads support 1.000, wing share 0.842 (A) and 1.000 (B), hull
contact 0: a body standing on its wings scores nothing. The frozen arm walks
(same support and shares as the walk arm) and cancels exactly.

### Top speed (each layout's fastest hand-tuned gait, straight along the shore, 8 s)

| layout | seed | m/s | hull lengths/s | sideways in 8 s | wing share | hull contact | steps with no ground force |
|---|---|---|---|---|---|---|---|
| A | 0 | 0.1101 | 0.69 | -0.069 m | 0.723 | 0.026 | 0.41 |
| A | 1 | 0.1103 | 0.69 | -0.067 m | 0.722 | 0.026 | 0.41 |
| A | 2 | 0.1093 | 0.68 | -0.057 m | 0.722 | 0.026 | 0.41 |
| B | 0 | 0.2570 | 1.61 | -0.378 m | 1.000 | 0.000 | 0.73 |
| B | 1 | 0.3011 | 1.88 | -0.400 m | 1.000 | 0.000 | 0.71 |
| B | 2 | 0.3233 | 2.02 | -0.305 m | 1.000 | 0.000 | 0.73 |

A: the crutch at 2.5 Hz, sweep 0.6 rad, stance 80%. B: a trot (diagonal thumbs
together) at 4 Hz, sweep 0.8 rad, stance 60%; it is off the ground 71-73% of the
time and veers 0.3-0.4 m sideways in 8 s, so it is a bound that goes fast, not a
gait that holds a line. The body is turned to face along the shore before the
run so the line is level.

### What the walking hardware costs in the air (`rig.run`, body held)

| build | all-up | hover 12 Hz, stroke 130, feather 45: lift | aero power | cruise 8 m/s, 8 Hz, stroke 80, feather 20: trim offset | lift/weight | aero power |
|---|---|---|---|---|---|---|
| bare A wing | 77.25 g | 2.985 N | 53.21 W | -7.22 deg | 0.991 | 5.26 W |
| A | 82.49 g | 2.979 N | 53.26 W | -8.00 deg | 1.007 | 5.27 W |
| bare B wing, one pair | 74.30 g | 1.454 N | 21.81 W | -10.96 deg | 1.009 | 1.84 W |
| B | 101.28 g | 2.719 N | 41.65 W | -8.00 deg | 0.972 | 4.00 W |

All four cruise points trimmed (within 3%). The tail is held at 0 on the rig
(`rig.py` now gives a hinge that is not on a wing the role "held"; before, it
would have been flapped as a stroke). B's four wings beat in phase and the hind
pair does not feel the fore pair's wake (PREREG section 4).

### The largest torque each joint asked for (N m)

| | stroke axis | feather axis | tail |
|---|---|---|---|
| A, late walk window, 8 seeds | 0.56-0.70 | 0.56-3.27 | 0.04-0.22 |
| B, late walk window, 8 seeds | 0.54-0.63 | 0.62-3.34 | — |
| A, whole rollout | 4.28-4.32 | 2.69-3.28 | |
| B, whole rollout | 2.32-2.33 | 2.77-3.34 | |
| A top-speed gait | 0.65 | 1.27 | 0.08-0.09 |
| B top-speed gait | 0.96-1.03 | 1.23-1.79 | — |
| propped, late window | 0.01 | 0.00 | 0.01 |

Steady walking asks 0.54-0.70 N m of the stroke axis and about 0.56-0.75 of the
feather. The values near 3 N m on the feather are the moments the steering
switches between turning and walking (a step in the target, rate-limited to 60
rad/s), and the 4.3 and 2.3 on the stroke are the first move from the spawn pose
to the stance. These are what a stiff servo asks through a joint with 0.0226
N m s/rad of built-in drive damping; they are not the load. Holding the body up
takes 0.01 N m.

## 4 The six predictions

1. "B walks on its wings by the rule above. I hold this at 0.5." **Held**: 8 of
   8; score 0.377-0.929, support 0.995-1.004, wing share 1.000, hull contact 0.
2. "A reaches the score and the support (>= 0.15 and >= 0.9 on >= 7 of 8) but
   not the wing share or the hull contact: its hull and tail carry 30% or more,
   or its hull touches more than 20% of the time. I hold this at 0.6."
   **Refuted**: A meets all four on 8 of 8. Hull and tail carry 22-24%; the hull
   touches 2.5-5.2% of the time.
3. "Null arms: propped and still under 0.012 on 8 of 8 for both layouts; heading
   frozen under 0.012 on 8 of 8 for both." **Held**: 0.0000 on all 32 rows.
4. "Top speed: B 0.10 m/s or more; A 0.05 m/s or more." **Held**: B 0.257-0.323,
   A 0.109-0.110.
5. "Air cost: thumbs and tail change A's hover lift by under 5% and its mass by
   under 10%; B's four wings give hover lift within 15% of A's pair at the same
   frequency and weigh 15-40% more all-up." **Held**: A's lift -0.2%, mass
   +6.8%; B's lift 2.719 N against A's 2.979 (-8.7%; -8.9% against the bare
   pair), mass +22.8% over A (+31.1% over the bare A wing).
6. "The largest torque asked of a shoulder on land is under the 0.42 N m the
   hover stroke asks for by its inertia alone." **Refuted**: 0.54-0.70 N m on the
   stroke axis in steady walking on every seed of both layouts, and more at the
   switches (table above).

## 5 The decision the frozen rule gives

A walks on its wings (8 of 8 against the 7 required). "If A walks on its wings,
the machine has one pair (fewer parts) whatever B does." **One pair and a tail.**

What the rule does not see, for whoever reads the decision: A passes the wing
share by 0.06-0.08, B by 0.30. A's hull touches the ground every stride; B's
never does. A's top speed is 0.11 m/s against B's 0.26-0.32. A is 18.8 g lighter
than B and has 5 actuators against 8.

## 6 Reproducers

Prefix each with `cd /home/hundo/Projects/Dytiscidae/dytiscidae && MUJOCO_GL=disable PYTHONPATH=.`
(`F=$PWD/experiments/flapper`). All together under five minutes on 6 processes.

```bash
.venv/bin/python $F/landwalk.py geometry
.venv/bin/python $F/landwalk.py probes
.venv/bin/python $F/landwalk.py run --layout A,B --arm walk,propped,frozen --seeds 0-7 --jobs 6 --out $F/results/landwalk.jsonl
.venv/bin/python $F/landwalk.py speed --layout A,B --seeds 0-2 --fast --jobs 6 --out $F/results/landwalk.jsonl
.venv/bin/python $F/landwalk.py air --out $F/results/landwalk.jsonl
.venv/bin/python $F/landwalk_table.py            # the tables above from the rows
```

`--out` appends; without it a command appends to `trials_F2.jsonl`.

## 7 What was not checked, and what the lead should know

**A harness fault found and fixed before these rows.** My token controller kept
its last action across rollouts, and the evaluator asks for a pose before its
first decision, so the second half of every antipodal pair was posed by the
first half's last action: the two halves did not start from one state (they
differed from the second step). It showed on the frozen-heading arm, which
scored 0.420 (A, seed 7) and 0.355 (B, seed 2) where it must cancel. After the
fix the two halves of that arm are identical to the last step and it reads
0.0000. The first full set, scored with the fault, is kept as
`results/landwalk_before_fix.jsonl`: A 8 of 8, B 7 of 8 (B was then on a trot),
frozen 7 of 8 for each. All tuning before trial 16 was done with the fault; both
layouts were re-checked on seeds 100-103 after it, and B's gait was changed
there (trot to crawl) before the scored rows.

Not checked:

- One scoring run per seed. The gaits hop, and a hopping gait is sensitive: on
  tuning seeds the same gait moved by 0.1-0.3 in score with small changes
  elsewhere. I did not repeat seeds 0-7 under perturbation.
- Friction is the repo's (0.9), contact is MuJoCo's default on a rigid beach, and
  17-38% of walking steps carry no ground force. I did not test a softer contact
  or a lower friction; the crutch's turn depends on the thumbs out-gripping the
  tail.
- Rigid wings. A's thumbs are at a quarter of the span on a wing that is 32% of
  the machine's mass; lifting a thumb by the stroke throws the body the other
  way. The first A gaits (stroke lift of 0.35 rad) could not make progress for
  that reason and for one other: the body fell onto thumbs that were still
  moving forward and was pushed back.
- The wing strikes the ground before the thumb if the stroke is lowered past
  about 12 degrees with these thumbs (the wing tip, and with the feather swept,
  the trailing edge). A's stance therefore holds the wings raised.
- The torque numbers are servo and damping, not load (section 3); stage 2 needs
  them re-read with the real drive.
- The tuning record is incomplete. About 60 early diagnostic rollouts (traces of
  one gait cycle, turn and reverse probes on seed 100) were run from scratch
  scripts and printed, not appended to `trials_F2.jsonl`; the `turn` command
  that logs them was added afterwards. The straight-line rows of "A sweep 1" to
  "A sweep 5" in the trials file started the body at right angles to its
  commanded heading, so their speeds are a turn, not a walk.
- Top speed is one hand search of about ten settings per layout on seed 100.
- `rig.run` and `flight_modes.rig` disagree on the default wing (4.1008 W against
  3.9878 W at the same point) because they build different chords and stroke
  ranges; that predates this work.

For the water and free-flight work:

- `control.outer.SplitController` cannot drive this body: `outer.observe` writes
  four rotor wetness channels and fails with none. `landwalk.TokenController`
  is the same interface without them, and it resets its held action when the
  clock goes back; anything installed on `env.cpg.command` needs that reset.
- Every built joint has 0.0226 N m s/rad of damping and 1e-4 kg m2 of armature.
  At flapping rates that is the torque: 12 Hz with a 130 degree stroke peaks at
  86 rad/s, 1.9 N m of damping alone.
- The thumbs cost nothing measurable in the air on the held rig (hover lift
  -0.2%, cruise power +0.2%); each wing carries an idle strut above it that a
  real build would not.
- A's tail here is a 9 cm rod with a 1.2 cm chord, chosen so it would not yaw
  the body on the ground. The package puts a paddle's chord a quarter chord to
  one side of its hinge; a tail wide enough to steer in water will need two
  mirrored parts or a different part kind.
- A's centre of mass is 2.4 cm behind the thumbs. Moving the shoulder from 0.65
  to 0.6 of the hull bought two points of wing share; anything that moves mass
  aft (a bigger tail, a tail motor) spends it.
- The land spawn always faces up the beach and the walk phase is 4 s, so the land
  score is mostly a turning test: A turns 21-30 degrees/s on the spot, B about
  15.
