# F3e result — 2026-11-12

`PREREG_F3e.md` was frozen first (commit 0964274) and is not edited. MuJoCo 3.15,
CPU, 1 ms step, body B with the neutral hull, `DYTISCIDAE_DEPTH_PAIR=1`
throughout. No learning, no optimiser: the law and the stroke were set by hand on
seeds 100-103 (`trials_F3e.jsonl`, 184 rollouts) and scored once on seeds 40-55
(`results/swim_F3e.jsonl`, 54 rows, none dropped; no `bad_qacc`, no failed
rollout). Code: `swim.py` (sha256 `92428940c032155e…`, frozen before the scored
rows).

**Outcome.** Both bars are met. By the frozen rule **the water is done for stage
1: the machine swims where it is told, at the depth it is told.** All six
predictions held.

It gets there by standing nearly on end (76-78 degrees of pitch), on a stroke
that asks the shoulders for 4.0-5.5 N m, and it does not hold still at the depth
inside the segment (the pair hold score is 0 on 14 of 16 seeds). Those are the
reads of section 3, not clauses of the bars.

## 1 The law and the stroke

New stroke name `S2d`: F3b's diagonal phasing at 1.5 Hz, stroke 1.0 rad, feather
0.6 rad, under the new law `Dive`. `S2x` is untouched and keeps F3b's law
(`Swim`): the stroke's entry names its law, and only `S2d` names `dive`.

`Dive` reads what `Swim` read (`TriphibianEnv.observation`), plus the body rates
that were already in it:

- a depth error becomes a pitch reference (1.2 rad per m, up to 1.2 rad in a
  hold and 0.3 rad under way to a heading) and a surge to make (2 per second,
  up to 1 m/s);
- in a hold the body drives forward along its nose while the pitch loop turns
  the nose toward the depth, so the path is an arc; within 0.4 m the stroke
  drops to half amplitude; it keeps 0.2 m/s of way on until the nose is back
  within 0.25 rad of level, and only then brakes (inside 0.2 m of the depth);
- under way to a heading the drive and the turn are F3b's, and the pitch
  reference is added;
- when the drive is small the fore pair is driven against the hind pair, so the
  pitch offsets have something to act through, and the pitch loop is stiffer
  (4 against 1.5); a left-against-right feather offset holds the roll.

Signs, re-measured on the neutral body before anything was built on them
(`trials_F3e.jsonl`, "signs on the neutral body", "bank sign"; open loop, 3 s):

| command | result | sign |
|---|---|---|
| drive forward, pitch command +0.5 / -0.5 | nose to -35 / +29 deg | + pitch command puts the nose down |
| drive backward, the same | nose to -78 / +82 deg | the same sign, both directions (the offsets ride on the drive, as F3b left them) |
| fore against hind, no way on, +0.5 / -0.5 | nose to -9 / +35 deg (baseline +22) | the same sign, about a third of the authority |
| drive forward, bank command +0.3 / -0.3 | roll to -59 / +61 deg | + bank command rolls negative |
| drive backward, bank +0.3 | roll to -40 deg | the same |
| drive forward, lift command +0.5 / -0.5 | climbs 0.35 / sinks 0.29 m/s, nose to +80 / -74 deg | as coded in F3b; it acts by pitching the body, and is not used |

The neutral body is not the heavy one in another way too: left alone with its
joints at 0 it pitches nose-up to 71 degrees in 3 s and has rolled over by 6 s
(up-z -0.68). Beating flat holds it level.

## 2 The two bars

### Segment bar, seeds 40-55 (unmodified `evaluate.run_segment`)

| clause | required | measured |
|---|---|---|
| it still swims: score 0.15 or more | 14 of 16 | **15 of 16** (0.171-0.560; seed 50 scores 0.005) |
| `pair_depth_response` 0.5 or more | 12 of 16 | **16 of 16** (0.537-0.840) |
| `bad_qacc` | 0 | 0 on all 48 rows |
| every actuator held, under 0.012 | 16 of 16 | 16 of 16, largest 2.1e-18 |
| the command frozen, under 0.012 | 16 of 16 | 16 of 16, all exactly 0 |

| seed | score | progress | response | pair hold | told / was at (second half), m | hold first |
|---|---|---|---|---|---|---|
| 40 | 0.171 | 0.39 | 0.718 | 0.000 | 2.84 / 3.30 | no |
| 41 | 0.455 | 1.00 | 0.813 | 0.000 | 2.56 / 2.82 | no |
| 42 | 0.319 | 0.70 | 0.715 | 0.000 | 2.41 / 2.80 | yes |
| 43 | 0.560 | 1.00 | 0.839 | 0.324 | 2.96 / 3.10 | no |
| 44 | 0.466 | 0.99 | 0.715 | 0.000 | 2.84 / 3.14 | yes |
| 45 | 0.276 | 0.63 | 0.671 | 0.000 | 2.03 / 2.64 | yes |
| 46 | 0.453 | 1.00 | 0.774 | 0.000 | 1.98 / 2.40 | no |
| 47 | 0.444 | 0.95 | 0.820 | 0.000 | 2.62 / 2.93 | no |
| 48 | 0.440 | 0.99 | 0.684 | 0.000 | 2.22 / 2.77 | yes |
| 49 | 0.527 | 1.00 | 0.840 | 0.211 | 2.67 / 2.84 | no |
| 50 | 0.005 | 0.01 | 0.707 | 0.000 | 2.75 / 3.09 | yes |
| 51 | 0.458 | 0.99 | 0.698 | 0.000 | 2.76 / 3.14 | yes |
| 52 | 0.440 | 1.00 | 0.607 | 0.000 | 2.03 / 2.79 | yes |
| 53 | 0.471 | 0.99 | 0.814 | 0.000 | 2.39 / 2.68 | no |
| 54 | 0.432 | 1.00 | 0.537 | 0.000 | 1.61 / 2.41 | yes |
| 55 | 0.432 | 1.00 | 0.702 | 0.000 | 1.67 / 2.56 | no |

On seed 50 one half of the pair went the wrong way along its heading (the
half's own `along` is -0.196); the depth response is there (0.707). Not traced.

The still arm turns over on every seed (up-z down to -0.70) and its response is
0; the frozen arm swims and dives to the first command in both halves (it was at
4.8-5.8 m when told 1.6-3.0) and its response and progress are both 0.

### Step bar (released level and still at 4 m, 12 s, seeds 40-42)

| told | seed | at the end | last 2 s: vertical rate | last 2 s: drift | within 0.3 m from | largest pitch | pitch at the end | met |
|---|---|---|---|---|---|---|---|---|
| 2.5 m | 40 | 2.590 m (+0.090) | 0.0092 m/s | 0.188 m | 3.45 s | 78 deg | +13 deg | yes |
| 2.5 m | 41 | 2.619 (+0.119) | 0.0052 | 0.031 | 3.45 s | 78 | +4 | yes |
| 2.5 m | 42 | 2.604 (+0.104) | 0.0055 | 0.034 | 3.45 s | 78 | -1 | yes |
| 5.5 m | 40 | 5.369 (-0.131) | 0.0037 | 0.263 | 3.78 s | 76 | -1 | yes |
| 5.5 m | 41 | 5.389 (-0.111) | 0.0028 | 0.051 | 3.78 s | 76 | +8 | yes |
| 5.5 m | 42 | 5.425 (-0.075) | 0.0102 | 0.062 | 3.78 s | 76 | +1 | yes |

**Six of six.** Once inside 0.3 m none leaves again, and none overshoots the
target. The drift is the largest distance from where the body was 2 s before the
end. The three seeds differ only in where the reset puts the body, and the runs
part after the arrival.

## 3 The reads beside the bars

- **Time to come within 0.3 m of a depth 1.5 m away**: 3.45 s going up, 3.78 s
  going down. The path, seed 40 going up (time: depth / pitch): 0.5 s 3.94 m
  / +41, 1 s 3.77 / +74, 2 s 3.23 / +59, 3 s 2.89 / +31, 4 s 2.74 / +18, 6 s
  2.63 / +10. It covers 1.4-1.9 m over the ground on the way.
- **Pitch used**: 76-78 degrees at most in the step; lowest up-z 0.15-0.16. In
  the segments the lowest up-z is 0.04-0.75 and the gate 0.87-0.98.
- **Largest shoulder torque**: step 4.02-4.58 N m on the stroke axis and
  3.02-3.64 on the feather; segments 4.51-5.51 and 2.67-3.83. F3b's S2x asked
  0.93-0.97 swimming straight: the faster stroke (1.5 Hz at twice the amplitude)
  is what the depth response costs.
- **Pair hold score**: 0.000 on 14 of 16, 0.211 and 0.324 on the other two. The
  hold wants the body at the depth and still over the last two seconds of its
  phase; the body arrives in 3.5-3.8 s when the depth is 1.5 m away and the
  commands are 1.0-2.5 m away.

## 4 The six predictions

1. "The step bar is met, six of six. I hold this at 0.6." **Held**: six of six.
2. "The segment bar's response clause is met (0.5 or more on 12 of 16). I hold
   this at 0.5." **Held**: 16 of 16, 0.537-0.840.
3. "It still swims: 0.15 or more on at least 14 of 16." **Held**: 15 of 16.
4. "The pair hold score stays under 0.3 on most seeds even where the response
   is met, because the hold wants the depth reached within about two seconds
   and then stillness." **Held**: under 0.3 on 15 of 16; 0 on 14.
5. "Going up and going down take the same time within 30% (the body is
   neutral)." **Held**: 3.45 s and 3.78 s, 10% apart.
6. "Null arms under 0.012 on 16 of 16 each." **Held**: still at most 2.1e-18,
   frozen 0.

## 5 The decision the frozen rule gives

"Both bars below met: the water is done for stage 1 (it swims where it is told,
at the depth it is told)." Both are met.

## 6 Reproducers

Prefix each with

    cd /home/hundo/Projects/Dytiscidae/dytiscidae && DYTISCIDAE_DEPTH_PAIR=1 FLAPPER_BODY_B='{"hull_radius": 0.0209}' MUJOCO_GL=disable PYTHONPATH=.

About two minutes on 6 processes; `--out` appends and must be an absolute path
(without it a command appends to `trials_F3b.jsonl`).

```bash
.venv/bin/python experiments/flapper/swim.py run --stroke S2d --arm swim,still,frozen --seeds 40-55 --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/swim_F3e.jsonl
.venv/bin/python experiments/flapper/swim.py step --stroke S2d --seeds 40-42 --targets '[2.5, 5.5]' --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/swim_F3e.jsonl
```

Stored rows still reproduce, checked before the law path was changed and again
after: `S2x` swim seed 24 with both variables set scores 0.45440525711380964
(0.454; `results/swim_F3d.jsonl`); `S2x` swim seed 3 with neither scores
0.5043534687718976 (`results/swim.jsonl`); F2's B walk seed 3 scores
0.758264810841501 with every field equal. In the two swim rows every stored
field is equal but `law` (the gain table gained one key, `off_bank`, which F3b's
law never commands) and the new field `pair`. The old law stays reachable as it
was: every stroke but `S2d` runs `Swim`, and `Stroke.act` is F3b's whenever the
three new commands (split, bank, a beat below 1) are 0 or 1, which is all `Swim`
ever sends.

`swim.py` rows now carry `pair` (`pair_depth_response`, `pair_second`,
`pair_hold_cmd`, `pair_hold_depth`) as a field of its own, and `step` is a
subcommand.

## 7 What was not checked

- **One run per seed, and the arrival is sensitive.** On the tuning seeds the
  step test met 5 of 8 before the pitch loop was stiffened at rest and 8 of 8
  after; several nearby settings met 7 of 8 (`trials_F3e.jsonl`, "step trials 9,
  10"). Six of six here is what the frozen bar asks, not a margin.
- **Seed 50's lost progress** was not traced.
- **The body is not level while it travels and barely level when it stops.** It
  ends within 13 degrees of level in the step test; it does so by swimming
  forward until the nose comes down, because at rest the offsets cannot beat the
  hull's own pitching moment (the first versions of the law stopped at the depth
  standing on end). A hull trimmed so that it does not pitch up by itself would
  make this easier; that is a body change and was not tried.
- **Torque.** 4-5.5 N m at the shoulder is five times F3b's stroke and comes
  from stiff ideal servos through joints with 0.0226 N m s/rad of built-in
  damping. A gentler stroke (1 Hz, 0.5 rad) also met the step bar on a tuning
  seed, arriving in 7.1 s with 2.2 N m; it was not scored, and it would not have
  met the segment's response clause (0.40 m of mean travel in the scored window
  against the 0.5-1.25 m needed).
- **Frequency above 1.5 Hz did not help** (2 Hz was slower on the tuning seed),
  as in F3b; not explained.
- The step test's three seeds are close to one run repeated until the arrival.
- The model's limits are the frozen file's: quasi-steady, no wake between the
  pairs, rigid wings, one buoyancy.
