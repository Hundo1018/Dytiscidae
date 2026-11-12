# F3b result — 2026-11-12

`PREREG_F3b.md` was frozen first (commit 13324a9) and is not edited. MuJoCo 3.15,
CPU, time step 1 ms. No learning, no optimiser: strokes and the law were set by
hand on seeds 100-103 (`trials_F3b.jsonl`, 185 rollouts) and scored once on seeds
0-7 (`results/swim.jsonl`, 87 rows, none dropped; no `bad_qacc`, no failure on
any row). Code: `swim.py` (sha256 `dbdab4e14617d716…`, frozen before the scored
rows). The harness is F2's, imported: body B, ideal servos (30 / 20 N m/rad),
unlimited pack, the token controller that forgets its held action between the
two halves of a pair.

**Outcome.** None of the three strokes of the frozen file swims: 0 of 8 seeds
reach 0.15 for S1, S2 and S3 alike (largest scores 0.033, 0.025, 0.013). By the
frozen rule a fore-aft wrist is to be built and a rowing stroke measured under a
new frozen file.

**What the rule does not cover, and the lead should decide on.** A fourth
phasing of the same shoulders does swim: fore-left with hind-right against
fore-right with hind-left ("S2x" below). It scores 0.350-0.519 on 8 of 8 seeds
and swims straight at 0.33-0.34 m/s with 3 mm of heave. It is not one of S1-S3, I
did not put it in their place, and one clause of the bar fails for it as written
(the heading-frozen null, section 3).

Predictions: 1 refuted, 2 refuted, 3 held, 4 refuted for the three strokes, 5
and 6 not measured (no stroke of the three passes).

## 1 The body in the water

`swim.py facts`:

| | |
|---|---|
| dry mass | 97.68 g (the pipeline's all-up figure is 101.28 g) |
| weight | 0.958 N |
| buoyancy | 0.801 N |
| **net** | **-0.157 N: B sinks, by 16% of its weight** |

The one-pair build floated by 0.027 N; the second pair and eight thumbs added
weight without volume. Left alone with its joints at 0, B sinks 0.02, 0.13 and
0.43 m in 1, 3 and 6 s and pitches nose-up as it goes (5, 31 and 52 degrees).

## 2 The strokes and the law

    stroke_i  = A sin(th_i)                          th_i = 2 pi (f t + phase_i)
    feather_i = -g_i (Psi cos(th_i) + lift and pitch offsets)       g_i = drive -+ turn

| stroke | phases (fore-left, fore-right, hind-left, hind-right), beats | f | A | Psi |
|---|---|---|---|---|
| S1, together | 0, 0, 0, 0 | 1 Hz | 1.0 rad | 0.6 rad |
| S2, against | 0, 0, 0.5, 0.5 | 1 Hz | 1.0 | 0.6 |
| S3, scull | 0, 0, 0.5, 0.5 (feather only) | 6 Hz | 0 | 0.4 |
| S2x, diagonal (not in the frozen file) | 0, 0.5, 0.5, 0 | 1 Hz | 0.5 | 0.6 |
| S2x_slow (the same at half the rate) | 0, 0.5, 0.5, 0 | 0.5 Hz | 0.5 | 0.6 |

The law (`Swim`, 25 Hz, from `TriphibianEnv.observation` alone) is the same for
every stroke: heading from the left-right difference of the drive, with F2's
switch (turn on the spot beyond 0.4 rad, swim backward when the heading is
behind); depth from a pitch reference under way (0.2 rad per unit of lift
command) held by the fore-hind feather difference; in a hold, beat flat and
brake. Targets are rate-limited to 5 rad/s (F2's 60 rad/s asked the feather for
10-20 N m whenever the drive switched). The law was tuned on S2x, the only stroke
that moved; the three strokes were then each given their best of three hand
settings under it (`trials_F3b.jsonl`, "the three strokes under the law").

Three signs in it are measured, not reasoned, and each was wrong the first time:
the feather offset that climbs is leading edge down; the pitch loop's stabilising
sign; and both flip with the direction of the drive (a loop that levels a forward
swim tumbles a backward one).

## 3 Rows

### Water score, seeds 0-7 (unmodified `evaluate.run_segment`, antipodal pair)

| stroke | scores, seeds 0-7 | >= 0.15 | progress | hold | gate | lowest up-z | bad_qacc |
|---|---|---|---|---|---|---|---|
| S1 | 0.027 0.024 0.030 0.030 0.027 0.028 0.033 0.025 | 0 of 8 | 0.01-0.06 | 0.00-0.03 | 0.98-0.99 | 0.07 | 0 |
| S2 | 0.000 0.013 0.010 0.005 0.024 0.000 0.025 0.000 | 0 of 8 | 0.00-0.08 | 0.00-0.05 | 0.54-0.63 | -1.00 | 0 |
| S3 | 0.009 0.000 0.013 0.000 0.005 0.011 0.007 0.007 | 0 of 8 | 0.00-0.04 | 0.00-0.02 | 0.50-0.96 | -1.00 | 0 |
| S2x | 0.387 0.452 0.500 0.504 0.472 0.350 0.519 0.427 | 8 of 8 | 0.69-0.94 | 0.01-0.10 | 0.99-1.00 | 0.93 | 0 |
| S2x_slow | 0.150 0.019 0.241 0.162 0.159 0.099 0.295 0.207 | 6 of 8 | 0.00-0.55 | 0.01-0.05 | 0.97-1.00 | 0.88 | 0 |

S2 and S3 turn the body over (up-z -1.00). S2x's score is almost all progress
along the commanded heading; its hold earns 0.01-0.10 because the commanded depth
is 1.0-2.5 m below the release and the body gets a few tenths of a metre of the way.

### Null arms (run for S2x and S2x_slow; no stroke of the three passes)

| stroke | still (every actuator held) | heading and depth command frozen at the first seen |
|---|---|---|
| S2x | 0.000 on 8 of 8 | 0.004 0.046 0.063 0.043 0.031 0.005 0.058 0.004: under 0.012 on 3 of 8 |
| S2x_slow | 0.000 on 8 of 8 | 0.005 0.019 0.029 0.016 0.012 0.007 0.029 0.005: under 0.012 on 4 of 8 |

The frozen arm's progress is 0.00 on every seed (the two halves cancel exactly).
What it keeps is the hold term, 0.01-0.13: a water task commands one depth for
both phases and both halves, so "frozen at the first seen" is the command itself
and this arm follows it as the swim arm does. As the bar is written ("both null
arms under 0.012 on 8 of 8") S2x does not meet it. The still arm, which sinks at
0.05-0.16 m/s, scores 0.

### Straight level swim, 8 s, seeds 0-2 (the law holding the starting heading and depth)

| stroke | speed | hull lengths/s | heave over a beat | pitch swing over a beat | largest torque, stroke / feather (last third) | lowest up-z |
|---|---|---|---|---|---|---|
| S1 (1 Hz, A 1.0) | 0.022 m/s | 0.14 | 230.5 mm | 24.9 deg | 1.36 / 0.70 N m | 0.96 |
| S2 (1 Hz, A 1.0) | 0.026-0.038 | 0.16-0.24 | 98.6-107.1 mm | 113.8-158.1 deg | 3.69-4.04 / 1.98-3.03 | -0.32 |
| S3 (6 Hz) | -0.040 to 0.050 | -0.25 to 0.31 | 2.0-4.9 mm | 5.6-10.0 deg | 0.82-1.22 / 2.96-4.11 | -1.00 |
| S2x (1 Hz, A 0.5) | 0.330-0.344 | 2.06-2.15 | 2.7-3.7 mm | 1.2-2.8 deg | 0.93-0.97 / 1.12-1.23 | 1.00 |
| S2x_slow (0.5 Hz) | 0.171-0.183 | 1.07-1.14 | 5.7-5.9 mm | 3.4-3.6 deg | 0.30-0.52 / 0.31-0.46 | 1.00 |

This is F3's forecast on a free body. With all four wings together the hull goes
up and down 23 cm a beat and the machine makes 2 cm/s. With fore against hind the
heave halves and the reaction comes back as pitch: the hull swings through
114-158 degrees a beat and rolls over. Only when the reactions cancel in force,
in pitch and in roll at once does the hull stay put and the wings move the water.

In the scored rollouts the largest torques after the first second are higher,
because the law switches the drive: S2x 1.47-2.28 N m on the stroke and 2.61-4.45
on the feather; S2x_slow 0.68-1.45 and 1.75-4.05.

## 4 The six predictions

1. "S2 passes the bar. I hold this at 0.5." **Refuted**: 0 of 8 at 0.15 or more;
   largest 0.025.
2. "S1's hull heaves more than S2's by a factor of three or more at the same
   frequency and amplitude." **Refuted**: 230.5 mm against 98.6-107.1 mm at 1 Hz
   and 1.0 rad, a factor of 2.2-2.3. S2 pays for it in pitch (114-158 degrees a
   beat against 25).
3. "S3 does not reach 0.15 on 7 of 8." **Held**: 0 of 8; largest 0.013.
4. "The straight swim under the best stroke is 0.15 m/s or faster at 1 Hz or
   less." **Refuted** for the three strokes: the fastest is S3's 0.050 m/s on one
   seed (at 6 Hz), S2 0.026-0.038, S1 0.022. Outside the file: S2x 0.330-0.344
   m/s at 1 Hz, S2x_slow 0.171-0.183 at 0.5 Hz.
5. "Null arms: still and frozen under 0.012 on 8 of 8." **Not measured**: the
   arms are per passing stroke and none of the three passes. On S2x: still 0.000
   on 8 of 8; frozen under 0.012 on 3 of 8, largest 0.063 (section 3).
6. "The largest shoulder torque in the passing stroke is under 0.7 N m (what
   F2's walk asked)." **Not measured**: no passing stroke among the three. On
   S2x it would be refuted (0.93-0.97 N m on the stroke axis swimming straight);
   S2x_slow is under it swimming straight (0.30-0.52) and does not reach the bar
   (6 of 8).

## 5 The decision the frozen rule gives

"If none passes, a fore-aft wrist on the outer wing is built and a rowing stroke
is measured under a new frozen file." None of S1, S2, S3 passes. **The rule
sends the design to a wrist.**

Beside that, for the decision the rule was written to make: the shoulders the
machine has do swim it, at 2 hull lengths a second with the hull still, when the
four wings beat on the diagonals. Whether that counts as "S2" in spirit, and
whether the frozen-heading clause should stand as written for a task whose depth
command never changes, are the lead's to say.

## 6 Reproducers

Prefix each with `cd /home/hundo/Projects/Dytiscidae/dytiscidae && MUJOCO_GL=disable PYTHONPATH=.`
(`F=$PWD/experiments/flapper`). About four minutes on 6 processes.

```bash
.venv/bin/python $F/swim.py facts
.venv/bin/python $F/swim.py run --stroke S1,S2,S3,S2x,S2x_slow --arm swim --seeds 0-7 --jobs 6 --out $F/results/swim.jsonl
.venv/bin/python $F/swim.py run --stroke S2x,S2x_slow --arm still,frozen --seeds 0-7 --jobs 6 --out $F/results/swim.jsonl
.venv/bin/python $F/swim.py straight --stroke S1,S2,S3,S2x,S2x_slow --seeds 0-2 --jobs 6 --out $F/results/swim.jsonl
```

`--out` appends; without it a command appends to `trials_F3b.jsonl`.

Stored F2 rows still reproduce: `landwalk.row_land("B", "walk", 3)` gives score
0.758264810841501 and `("A", "walk", 5)` gives 0.4494066500421542, every stored
field equal but wall time; `landwalk.py` (3f85a218d7c299fb) and `body.py`
(94d4fe37dcf77743) are unchanged.

## 7 What was not checked, and what the lead should know

- **The three strokes got less tuning than S2x.** Each had an open-loop sweep of
  six settings and three settings under the law; S2x had the law built around
  it. I do not think more tuning changes their answer (open loop, with nothing
  to steer, S1 made at most 0.04 m/s and S2 0.12 m/s while tumbling), but it was
  not shown.
- **Depth is barely controlled.** Under way, a 1.5 m dive command moved the body
  0.5-0.7 m in 8 s at 0.21-0.24 m/s. In a hold the body neither sinks to the
  commanded depth in time (it sinks at 0.12 m/s left alone) nor can it be driven
  there: rowing for lift (the wing edge-on for half the beat) and folding the
  wings both turned the body over, and are left out. S2x's hold score is 0.01-0.10.
  The machine passes on swimming where it is told, not on being at depth.
- **Beating flat holds the sinking body up** (1 cm/s of sink against 12 cm/s
  left alone). That is the quasi-steady model's drag on a beating plate; I did not
  look for where the lift comes from.
- Speed did not rise with beat rate in the open-loop sweep (0.18-0.22 m/s from
  0.5 to 2 Hz at like amplitudes) and rose with feather amplitude. Not
  explained.
- The model's limits are the frozen file's: quasi-steady with the package's
  added mass, no wake between the pairs, rigid wings. S2x depends on four
  reactions cancelling; a real pair of wings in the other pair's wake will not
  cancel exactly.
- One scoring run per seed. S1's three straight rows are identical to the last
  digit (the reset moves only the position and the stroke is symmetric).
- Torques are what stiff ideal servos ask through joints with 0.0226 N m s/rad of
  built-in damping, as in F2.
- B is 16% heavy. The frozen file fixes one buoyancy; a hull that floats would
  change the hold and not the stroke.
