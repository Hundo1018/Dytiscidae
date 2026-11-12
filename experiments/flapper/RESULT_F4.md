# F4 result — 2026-11-12

`PREREG_F4.md` was frozen first (commit 84da7ad) and is not edited. MuJoCo 3.15,
CPU, 1 ms step (83 steps a beat), body B with the neutral hull
(`FLAPPER_BODY_B='{"hull_radius": 0.0209}'`, 104.9 g simulated, 1.029 N). No
learning, no optimiser: the stroke and the law were set by hand on seeds 100-103
(`trials_F4.jsonl`, 119 rows) and scored once (`results/fly.jsonl`, 113 rows,
none dropped; no `bad_qacc`, no failed rollout). Code: `fly.py` (sha256
`e03893661df8b70f…`, frozen before the scored rows). No part was added.

**Outcome.** Both bars are met. By the frozen rule **the three ways of moving are
established for stage 1 on one body, and the work moves to the transitions.**
Predictions 1-5 held; 6 held on frequency (12 Hz, the top of its range) and is
refuted on torque (2.04-2.08 N m against 1.5).

Two things beside the bars. The machine flies only while it flaps and steers: the
same launches with every actuator held do not glide, they nose over and reach
the sea in under 4 s. And the flight rests on a model in which the hind wings do
not feel the fore wings' wake (section 6 gives the geometry for a sensitivity
test).

## 1 The launch, and the held rig for this body

`fly.py facts`: the package launches this body at **7.2 m/s, 10 degrees nose-up**,
30 m up. No air gate; `level_margin` 0.743 (so `flies_level` is true and the
height term of the score is live); `lift_margin` 30.1.

Is that a state it can fly from? Not by itself: with its wings held at neutral
the free body's nose falls from +10 to 0 degrees in 0.1 s and to -46 in 1 s
(3.1 m lost in the first second; `trials_F4.jsonl`, "open loop, wings still").
Level on four wings at one incidence is not a trim for this body: the fore pair
has to carry more than the hind. It is a state the law flies from: the launch
was not changed.

Rig (`rig.run`, `rig.trim`; body held, all four wings in phase, every wing at one
incidence):

| | |
|---|---|
| hover, 12 Hz, stroke 130, feather 45 | lift 2.719 N = 2.64 weights, aerodynamic power 41.7 W |
| cruise 6 m/s: strokes that carry the weight and hold the speed | 2 of 16: 12 Hz, stroke 80, feather 20 or 30 (incidence 12.6 / 12.0 deg, 15.2 / 11.5 W) |
| cruise 8 m/s | 2 of 16: 12 Hz, stroke 80, feather 20 or 30 (incidence 8.0, 15.3 / 10.2 W) |
| cruise 7.2 m/s (the launch speed) | 2 of 16: 12 Hz, stroke 80, feather 20 or 30 (incidence 9.5 / 9.4 deg, net force along +0.113 / +0.127 N, 15.3 / 10.8 W) |
| best net force along at 8 Hz and at 10 Hz, 7.2 m/s | -0.217 N and -0.069 N: neither holds the speed |

The grid was 6, 8, 10, 12 Hz by stroke 60, 80 deg by feather 20, 30 deg. On it
this body needs 12 Hz; that chose the stroke.

## 2 The stroke and the law

**Stroke flown**: 12 Hz, stroke +-0.7 rad (80 deg peak to peak), feather +-0.5 rad
(29 deg), leading edge down on the downstroke, on F3b's diagonal (fore-left with
hind-right against fore-right with hind-left). The diagonal is not a choice: with
all four in phase, as on the rig, the free body lost its speed (0.5 m/s after 4
s, 7.9 m lost; the wings are 45% of the machine's mass and their reaction heaves
the hull).

**Two loops.**

- The inner loop (`Wings.act`, every step) reads the body's attitude and rates
  and holds a pitch and a bank by feather offsets. This is new against F2 and
  F3, whose stroke generators read only the clock: the pitch and roll modes here
  are too fast for 25 Hz (0.1 rad of offset turns the nose at about 49 rad/s^2).
- The outer law (`Pilot`, 25 Hz, from `TriphibianEnv.observation` alone): bank
  = heading error (1 rad per rad, up to 0.7 rad, left wing down to turn left);
  pitch = 0.17 rad + 0.05 rad per m of height lost + 0.13 rad per rad of bank;
  the stroke at full amplitude always.

**How each axis is controlled, each sign found by measurement** (launched as the
segment launches it, no scatter; `trials_F4.jsonl`):

| axis | what makes it | measured |
|---|---|---|
| pitch | feather offset, fore pair against hind pair | fore leading edges up 0.1 rad, hind down: nose at +14 deg after 0.1 s against 0 with no offset; the other way -29. About 49 rad/s^2 per 0.1 rad |
| roll | feather offset, left against right | left leading edges up 0.1 rad: left wing up, 19 deg after 0.1 s and 47 after 0.2; mirrored the other way. About 66 rad/s^2 per 0.1 rad |
| yaw | stroke amplitude, right against left | right larger by 0.2 of the amplitude and left smaller: nose left, 21 deg after 0.2 s and 40 after 0.4 (mirrored the other way). About 18 rad/s^2. **Measured and not used** |
| turn | bank | left wing down 18 deg: nose and path turn left at 20 deg/s; 35 deg: 45 deg/s. The nose follows the path with no yaw loop |
| height | pitch | bank 35 deg at 0.17 rad of pitch lost 1.3 m in 3 s; at 0.25 rad it gained 0.1 m |
| speed | nothing | the flight settles at 7.8-8.6 m/s by itself |

Two loops were written and are left at zero because they lost the flight on the
tuning seeds: a yaw loop on slip and yaw rate through the stroke amplitude (it
shook the stroke at every step; shoulder torque 28 N m, speed down to 1.9 m/s),
and a speed loop through the stroke amplitude (10-13 m lost in 8 s).

Servos: 60 / 30 N m/rad on stroke / feather, no force limit; the stroke follows
to within 0.06 rad.

## 3 The two bars

### Level bar, seeds 0-7 (the segment's launch and its own scatter draw, told to fly straight, 8 s)

| seed | net height change | lowest / highest | speed at launch -> end (against 7.2) | lowest up-z | met | every actuator held: height change |
|---|---|---|---|---|---|---|
| 0 | -1.48 m | -1.48 / +0.00 | 8.28 -> 7.41 (+3%) | 0.87 | yes | -30.23 m |
| 1 | +0.52 | -0.08 / +0.52 | 8.09 -> 8.01 (+11%) | 0.95 | yes | -30.11 |
| 2 | +0.80 | +0.00 / +0.80 | 7.70 -> 8.18 (+14%) | 0.96 | yes | -30.16 |
| 3 | +0.74 | -0.32 / +0.74 | 6.20 -> 8.18 (+14%) | 0.96 | yes | -30.11 |
| 4 | +0.10 | -0.22 / +0.10 | 8.14 -> 7.83 (+9%) | 0.93 | yes | -30.40 |
| 5 | +0.77 | -0.09 / +0.77 | 6.16 -> 8.25 (+15%) | 0.97 | yes | -30.06 |
| 6 | +0.60 | +0.00 / +0.60 | 8.25 -> 8.02 (+11%) | 0.95 | yes | -29.59 |
| 7 | +0.37 | -0.18 / +0.37 | 6.84 -> 8.01 (+11%) | 0.95 | yes | -29.96 |

**8 of 8** (7 required). Seed 0 passes by 2 cm of height. The held body is not a
glide to compare with: it is in the sea within 3.5-4 s on every seed (seed 0: 26.6
m up after 1 s at -40 deg of pitch, 17.9 m after 2 s, down at 4 s), and it rolls
75 degrees or more on all eight.

### Turn bar, seeds 0-15 (unmodified `evaluate.run_segment`, the mirrored pair)

| clause | required | measured |
|---|---|---|
| score 0.15 or more | 14 of 16 | **16 of 16**, 0.907-0.983 |
| `bad_qacc` | 0 | 0 on all 48 rows |
| every actuator held, under 0.012 | 16 of 16 | 16 of 16, all exactly 0 |
| command frozen at the first seen, under 0.012 | 16 of 16 | 16 of 16, all exactly 0 |

The turn response is 1.000 on every seed (pair mean 1.12-1.18 before the clip)
and the height term 0.81-0.97. The frozen arm flies straight in both halves
(height term 0.62-0.76, turn made 0 degrees) and scores 0; the held body falls.

Turn made against turn asked, each half (the path's direction over 6-8 s against
2-4 s):

| seed | first half: told / made | second half: told / made | score |
|---|---|---|---|
| 0 | +82 / +78.1 | -82 / -77.9 | 0.951 |
| 1 | +82 / +78.0 | -82 / -78.1 | 0.971 |
| 2 | -53 / -50.6 | +53 / +51.1 | 0.915 |
| 3 | +69 / +65.7 | -69 / -65.6 | 0.907 |
| 4 | -58 / -55.6 | +58 / +54.8 | 0.914 |
| 5 | -55 / -52.1 | +55 / +52.0 | 0.908 |
| 6 | +78 / +74.6 | -78 / -73.9 | 0.959 |
| 7 | +82 / +77.4 | -82 / -78.0 | 0.971 |
| 8 | -67 / -64.5 | +67 / +64.3 | 0.942 |
| 9 | -67 / -64.4 | +67 / +64.4 | 0.940 |
| 10 | +89 / +84.0 | -89 / -84.0 | 0.978 |
| 11 | +76 / +72.6 | -76 / -72.9 | 0.956 |
| 12 | -72 / -68.6 | +72 / +69.3 | 0.945 |
| 13 | -88 / -83.2 | +88 / +82.9 | 0.983 |
| 14 | -58 / -55.0 | +58 / +55.6 | 0.923 |
| 15 | -74 / -71.1 | +74 / +70.8 | 0.950 |

Every turn is made to within 2.1-4.7 degrees short of what was asked, to the
side asked, in both halves. Over the 32 halves the height changes by -0.57 to
+0.51 m, the speed ends at 8.12-8.64 m/s, the largest bank is 42 degrees and the
lowest up-z 0.73.

## 4 The other reads

- **Stroke used**: 12 Hz; stroke 80 deg peak to peak; feather 29 deg either way;
  body pitch +8 to +17 degrees in level flight.
- **Largest shoulder torque** (after the first 0.25 s): stroke axis 2.04-2.08 N m
  on every flying row; feather axis 1.89-4.39 N m in the level flights and
  4.83-10.30 in the turns, where the bank command arrives as a step. Stiff ideal
  servos through joints with 0.0226 N m s/rad of built-in damping, as before:
  at 12 Hz and +-0.7 rad the damping alone is 1.2 N m.

## 5 The six predictions

1. "The level bar is met. I hold this at 0.7." **Held**: 8 of 8.
2. "The held glide loses 2.5 m or more over the same 8 s on at least 6 of 8."
   **Held**: 8 of 8, and by far more than the prediction meant: 29.6-30.4 m, the
   whole height.
3. "The turn bar is met. I hold this at 0.45: steering a tandem flapper by a
   hand law is what I trust least." **Held**: 16 of 16, 0.907-0.983.
4. "Null arms under 0.012 on 16 of 16 each (a glider that does not turn scores
   0 under the pair)." **Held**: exactly 0 on all 32.
5. "No part is added: the law flies the body as F2 and F3 left it." **Held**.
6. "The flight stroke is between 6 and 12 Hz and asks a shoulder for under 1.5
   N m." Frequency **held** at the edge (12 Hz); torque **refuted** (2.04-2.08
   N m on the stroke axis in steady flight).

## 6 The decision, and the wake the model leaves out

"Both bars below met: the three ways of moving are established for stage 1 on
one body, and the work moves to the transitions." Both are met.

For sizing a wake sensitivity test (nothing here models the wake):

| | |
|---|---|
| chord | 0.07 m; wing 0.21 m; both pairs on the hull's axis, in one plane |
| fore pair to hind pair | 0.088 m spar to spar = **1.26 chords**; the gap from the fore trailing edge to the hind leading edge is 0.017 m = 0.24 chord |
| flight speed | 7.4-8.6 m/s over the scored rows (8.2 typical) |
| travel per beat | 8.2 / 12 = 0.68 m = **9.8 chords** (0.62-0.72 m over the speed range) |
| the pair spacing in beats | 0.088 / 0.68 = **0.13 of a beat** (46 deg of phase; 0.12-0.14 over the speed range): what the fore wing sheds reaches the hind wing 11 ms later |
| stroke phase between the two | the hind wing on each side beats half a beat from the fore wing ahead of it (the diagonal), so it meets that wake 0.63 of a beat (or -0.37) into its own stroke |
| reduced frequency | pi f c / V = pi x 12 x 0.07 / 8.2 = **0.32** (0.31-0.36) |
| tip travel | 0.27 m peak to peak vertically (stroke +-0.7 rad on 0.21 m): Strouhal f a / V = 0.40 |
| height of the fore wake at the hind wing | the body flies 8-17 degrees nose-up, so the hind roots sit 1.2-2.6 cm (0.2-0.4 chord) below the line the fore roots trace |

At 0.24 chord of gap and 0.13 of a beat the hind pair is inside the fore pair's
near wake for the whole stroke. Two things in this result lean on its absence:
the hind pair's lift (the pitch trim is a fore-against-hind offset) and the
stroke's net thrust (0.11-0.13 N of margin on the rig at 7.2 m/s).

## 7 Reproducers

Prefix each with

    cd /home/hundo/Projects/Dytiscidae/dytiscidae && FLAPPER_BODY_B='{"hull_radius": 0.0209}' MUJOCO_GL=disable PYTHONPATH=.

About four minutes on 6 processes; `--out` appends and is an absolute path.

```bash
.venv/bin/python experiments/flapper/fly.py facts
.venv/bin/python experiments/flapper/fly.py rig --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/fly.jsonl
.venv/bin/python experiments/flapper/fly.py level --arm fly,glide --seeds 0-7 --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/fly.jsonl
.venv/bin/python experiments/flapper/fly.py run --arm fly,still,frozen --seeds 0-15 --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/fly.jsonl
```

The 49 rig rows in `results/fly.jsonl` are the ones measured first and written
to `trials_F4.jsonl`, copied, not run twice.

Stored rows still match, re-run at the end: F2's B walk seed 3 scores
0.758264810841501 and F3e's `S2d` swim seed 51 (with `DYTISCIDAE_DEPTH_PAIR=1`
and the neutral hull) scores 0.4581414836861409 (0.4581), every stored field
equal but wall time. `landwalk.py`, `swim.py`, `body.py` and `rig.py` are not
touched by this work.

## 8 What was not checked

- **One run per seed.** Level seed 0 is inside the height bar by 2 cm.
- **The wake** (section 6), still air, rigid wings: the frozen file's limits.
- **The rig and the flight are not the same stroke.** The rig flaps four wings
  in phase on a held body at one incidence; the flight is on the diagonal with a
  fore-against-hind offset and a feather of 29 degrees where the rig's grid
  stopped at 30. No rig row was run for the diagonal.
- **Nothing below 12 Hz was flown free**; on the rig nothing below it holds the
  speed. Stroke amplitudes above 80 degrees were tried once on a tuning seed
  (100 degrees at 12 Hz: the body spun in yaw and fell) and not pursued.
- **Yaw is uncontrolled.** The nose follows the path in every scored row, and
  the one probe that spun says it need not. The authority is there (table in
  section 2); a loop that uses it without shaking the stroke was not found in
  the time.
- **Speed is not controlled**: the flight sits 9-15% above the launch speed.
- The servo gains were raised for 12 Hz (60 / 30 against F2's 30 / 20) and not
  varied; 1 ms was not halved.
- Torque is servo and joint damping, not load.

## 9 For the transitions

- **The body does not glide and does not sit in trim.** Wings held, it noses
  over within a tenth of a second. With the attitude loop closed and the wings
  not beating it sinks at 3.3 m/s at 6-6.8 m/s, nose 10 degrees down (a glide
  ratio near 2, against the 10 of the one-pair rig): arriving anywhere will be
  under power.
- **It must be flapping and controlled from the first instant of flight**; the
  inner loop cannot be the 25 Hz law.
- **Flight speed is 7-8.6 m/s and the stroke is at the top of its range.** The
  only margin seen is the feather: at 20 degrees instead of 29 the same stroke
  climbed 0.8 m in 4 s and reached 9.8 m/s.
- **Hover on the rig is 2.64 weights at 12 Hz, in phase, body held (41.7 W).**
  In phase is what the free body cannot do: its wings' reaction took the speed
  away in cruise. Whether the diagonal lifts the weight at zero speed was not
  measured, and is the first thing a take-off needs.
- The wings carry 4.5 cm thumbs below them and the body lands at 7 m/s or
  hovers; nothing here says which.
- From the water the body is neutral and level flight needs 7 m/s; F3e's
  fastest stroke swims at about 0.4 m/s.
