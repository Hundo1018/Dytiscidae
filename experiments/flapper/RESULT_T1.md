# T1 result — 2026-11-12

`PREREG_T1.md` was frozen first (commit a4b6513) and is not edited. MuJoCo 3.15,
CPU, 1 ms step, body B with the neutral hull
(`FLAPPER_BODY_B='{"hull_radius": 0.0209}'`, 1.029 N), `FLAPPER_WAKE_GAIN`
unset. No learning, no optimiser, no part added: the stroke, the loop and the
take-off were set by hand on seeds and entry states 100-103
(`trials_T1.jsonl`, 67 rows) and scored once (`results/takeoff.jsonl`, 54 rows,
none dropped; no `bad_qacc`, no failed rollout). Code: `takeoff.py` (sha256
`2c48bc78e4c84e69…`, frozen before the scored rows).

**Outcome.** All three parts are met. By the frozen rule **take-off is
established and the water crossings are next.** Predictions 1-4 and 6 held (6
with one word changed: it pushes off its fore wings, not its thumbs); 5 is
refuted (the hover asks a shoulder for less than F4's flight, not more).

## 1 How it leaves the ground

It lies level on its four thumbs and is told to stay; it does not move (0.00-0.03
m in the 1.5 s). Told the air, it does a push-up. The two fore wings swing down
at 3 rad/s; they meet the beach, the nose is thrown up about the tail, and in under
a third of a second the body is 43 degrees nose-up, still within a centimetre of
the ground. There the hover loop takes over. All four wings begin to beat on the
diagonal, about the body's long axis, and the thrust that points along the nose
carries it off the beach 0.37-0.43 s after the command while the loop finishes
standing it up: it is 0.4 m clear at 0.5 s, a metre at 0.7 s and at its
height of 1.6 m by 1.1 s, 72-82 degrees nose-up. Once a metre up it leans its
nose 10 degrees toward its belly, so that it ends the right way up by the
probe's reading (the root's up axis at +0.18), and it drifts belly-first at 0.7
m/s for the rest of the 6 s, 2.4-2.5 m over the ground in all. There is no run
along the beach.

What stands it up is the fore wings themselves, not their thumbs: past about 17
degrees of swing the wing tip is lower than the thumb, and the contacts read at
the push are on the wing bodies (1.6 N at 41 degrees of pitch, a 23 N slap at
57). The hind thumbs carry the body until then.

## 2 The three parts

### T1a, lift at rest (held rig, body upright and still, 12 Hz)

| phasing | stroke 80, feather 30 | 80, 45 | 130, 30 | 130, 45 | largest sideways force in a beat |
|---|---|---|---|---|---|
| diagonal | 0.945 weights | 1.151 | **2.312** | **2.639** | 0.00 N |
| in phase | 0.902 | 1.148 | 2.225 | 2.642 | 2.83-9.44 N |

Bar: 1.3 weights at some setting. **Met**: 2.31 and 2.64 weights at 130 degrees
of stroke. The diagonals lift what the in-phase stroke lifts and throw nothing
sideways; the in-phase stroke swings 3-9 N across the body's axis within each
beat, which is what took the speed away in F4.

### T1b, hover (free body, released upright and still at 30 m, told to stay, 6 s)

| arm, seeds 0-7 | height | drift | nose from vertical | met |
|---|---|---|---|---|
| hover | -0.00 to +0.12 m (ends +0.09) | 0.00 m | 0 deg | **8 of 8** |
| every actuator held | -30.05 m (into the sea) | 8.12 m | turns over | 0 of 8 |

The eight hover rows are one run: a release that is exactly upright and exactly
still has nothing in it for the seed to change (the reset moves only where the
body starts), and the stroke is symmetric. So the bar is met and says little by
itself. What says more is the disturbed releases of the tuning seed
(`trials_T1.jsonl`, "hover trials 2" and "4"): tilted 17 degrees about either
axis, or both, or 29 and 11 degrees with 0.9 m/s of way on, or released at 1.4
m/s: back upright in every case, height within 0.2 m. Drift 0.16-2.28 m over the six
cases before the drift gain was raised (one outside the 2 m bar), and 0.14-0.64
m on the four of them re-run with the final gain.

Stroke: 12 Hz, +-0.653 rad (75 deg peak to peak), feather +-45 deg, on the
diagonal. The loop reads the body's attitude, its rates, the velocity of its
centre of mass and its height over what is underneath, **each as a mean over
the last beat**. Fed back raw at every step it was unstable within 0.05 s: the
wings are 45% of the machine, and a stroke bias moves the hull the other way at
once.

**How each axis is controlled, each sign found by measurement** (released
upright, no loop, the rates read over the first 0.17 s):

| axis (body frame: x the nose, y the left wing, z the back) | what makes it | measured |
|---|---|---|
| about z | stroke amplitude, left against right | left larger by 0.2 of the amplitude: -5.8 rad/s. About -170 rad/s^2 per unit |
| about y | a common stroke bias (every wing swept toward the belly) | 0.2 rad: -8.6 rad/s. About -250 rad/s^2 per rad |
| about x (spin) | feather offset, left against right | 0.2 rad: +6.2 rad/s, with no tilt. About +180 rad/s^2 per rad. Used as damping only |
| climb | the common amplitude | +0.2 of it: 2.36 m/s after 0.5 s against 0.97 |
| drift | leaning the nose against it | 0.35 rad per m/s, up to 0.3 rad |
| not used | fore against hind feather offset (about y, -5.6 rad/s per 0.2 rad); a common feather offset (about y, +6.0); fore against hind amplitude and left against right bias (nothing) | |

### T1c, take-off (`run_transition(env, "land_to_air", ctrl)`, `DYTISCIDAE_CROSSING_GATES=1`, entry states 0-7)

| arm | crossed | hold (drift in the 1.5 s) | clearance, peak / at 6 s | up axis at 6 s | lowest up axis | failure |
|---|---|---|---|---|---|---|
| take-off law | **8 of 8** | 1.00 (0.002-0.026 m) | 1.69-1.79 / 1.59-1.60 m | +0.18 | -0.12 to +0.14 | none |
| every actuator held | 0 of 8 | 1.00 | 0.00 / 0.00 | +0.99 | | never crossed the boundary |
| command frozen at the first seen | 0 of 8 | 1.00 | 0.00 / 0.00 | +0.99 | | never crossed the boundary |

Hover loop on 0.28-0.30 s after the command, at 43 degrees of pitch on every
entry; off the ground for good 0.37-0.43 s after it; 0.67 m/s and
2.35-2.53 m of travel at the end. Entry 0 (time: clearance, pitch): 1.5 s 0.00 m
/ +7, 1.7 s 0.00 / +25, 1.8 s 0.01 / +44, 2.0 s 0.42 / +52, 2.2 s 1.00 / +72,
2.5 s 1.52 / +80, 3 s 1.74 / +79, 6 s 1.59 / +80.

The entry states differ little for a body that starts by standing still on four
thumbs, and the eight take-offs are alike to the second decimal.

### Is F4's flight reachable from the hover? (PREREG section 2)

T1b's hover for 6 s, then F4's own law (`fly.Pilot` over `fly.Wings`, untouched,
given a straight air task) for 4 s, seeds 0-2:

| handed over | at hand-off | lowest | height change over the 4 s | last 2 s | end speed | up axis, lowest / over the last 2 s | F4's level bar |
|---|---|---|---|---|---|---|---|
| straight from the hover | 0.0 m/s, nose vertical, up axis 0.00 | -0.96 m | -0.56 m | +0.39 m | 8.18 m/s (1.14 of launch) | -0.03 / 0.97 | not met |
| after 1 s of the hover leaned 0.5 rad | 0.67 m/s, 10 deg from vertical, up axis +0.18 | -0.91 m | -0.80 m | +0.12 m | 8.01 m/s (1.11) | +0.18 / 0.95 | not met |

**By the letter it does not meet F4's bar, and it could not:** the bar asks the
root's up axis above 0.5 throughout, and a hover starts at 0. In substance the
flight is reached: F4's law lays the body level in a quarter of a second (up
axis 0.87 at 0.25 s, 0.97 from 0.5 s on), it gathers 7 m/s in 2 s, bottoms out
0.96 m below the hover and is climbing at 8 m/s by the end, inside F4's height
and speed clauses. The three seeds are again one run.

## 3 Torques (largest asked of a shoulder, N m)

| | stroke axis | feather axis |
|---|---|---|
| hover (T1b), after the first 0.25 s | 1.85 | 1.45 |
| hover from the disturbed releases | 1.94-2.29 | 1.46-1.47 |
| take-off (T1c), from the command on | 7.90-9.39 | 2.71-3.59 |
| F4's law taking over a hover | 8.73 | 4.11 |
| for reference, F4's level flight | 2.04-2.08 | 1.89-4.39 |

The take-off's 8-9 N m is the moment the stroke takes over from the push-up's
pose (blended over 0.08 s; handed over as a step it was 68 N m) and the fore
wings slapping the beach. As before these are stiff ideal servos through joints
with 0.0226 N m s/rad of built-in damping.

## 4 The six predictions

1. "T1a is met at 130 deg of stroke. I hold this at 0.8." **Held**: 2.31 and
   2.64 weights.
2. "T1b is met. I hold this at 0.5." **Held**: 8 of 8, with the caution of
   section 2 that the eight are one run.
3. "T1c is met. I hold this at 0.35: nothing here has stood up from the ground,
   and the wings' tips pass through the ground plane when the body lies level."
   **Held**: 8 of 8. The tips meeting the ground is how it stands up.
4. "Null arms: 0 of 8 each in T1c; the held arm of T1b loses more than 20 m."
   **Held**: 0 and 0 of 8; 30.05 m.
5. "The hover asks a shoulder for more than F4's 2.0 N m." **Refuted**: 1.85
   N m (1.94-2.29 while recovering from a disturbed release). The hover stroke
   is 75 degrees against the flight's 80, at the same 12 Hz.
6. "If T1c is met, the machine leaves the ground by pushing off its thumbs and
   pitching up, not by a run along the beach." **Held**, with one word changed:
   it pushes off its fore wings, which reach the ground before their thumbs do,
   and pitches up; 2.4-2.5 m over the ground in 4.5 s, all of it after it is in
   the air.

## 5 The decision the frozen rule gives

"All three: take-off is established and the water crossings are next." All three
are met.

## 6 Reproducers

Prefix each with

    cd /home/hundo/Projects/Dytiscidae/dytiscidae && FLAPPER_BODY_B='{"hull_radius": 0.0209}' MUJOCO_GL=disable PYTHONPATH=.

and, for the take-off only, `DYTISCIDAE_CROSSING_GATES=1` as written. About two
minutes on 6 processes; `--out` appends and is an absolute path.

```bash
.venv/bin/python experiments/flapper/takeoff.py rig --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/takeoff.jsonl
.venv/bin/python experiments/flapper/takeoff.py hover --arm hover,held --seeds 0-7 --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/takeoff.jsonl
DYTISCIDAE_CROSSING_GATES=1 .venv/bin/python experiments/flapper/takeoff.py takeoff --arm go,still,frozen --entries 0-7 --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/takeoff.jsonl
.venv/bin/python experiments/flapper/takeoff.py handoff --seeds 0-2 --lean-s '[0.0, 1.0]' --jobs 6 --out /home/hundo/Projects/Dytiscidae/dytiscidae/experiments/flapper/results/takeoff.jsonl
```

The entry states are the repo's draw plus 7919 times the entry number, as
`nulls.set_entry_state` made them for E0; entry 0 is the repo's own.

Stored rows still match, re-run at the end: F4's turn on seed 7 scores
0.9706859952759738, every stored field equal but wall time. `fly.py` is imported
for the hand-off and not changed; `landwalk.py`, `swim.py`, `body.py` and
`rig.py` are not touched.

## 7 What was not checked

- **The scored hover and hand-off rows carry no disturbance** (section 2). The
  disturbed releases are tuning rows on one seed, six cases.
- **The take-off is the same on every entry**, and one run each. I did not start
  it tilted, on a different slope, or facing down the beach; the push-up leans
  on the fore wings striking the ground cleanly.
- **The fore wings are slapped against the beach** (23 N at one instant, 22 body
  weights) and the wings are rigid. A real membrane wing is not a leg; whether
  this is a take-off the user would accept is not a question the probe asks.
- **No ground effect** on the wings (the frozen file's limit), and the hind pair
  begins to beat within a few centimetres of the beach.
- **The wake feedback is off**, as the frozen file says; the bars were not read
  at gains 1 and 3.
- The hover loop's memory is one beat of its own signals. Nothing else was tried
  in its place (a slower loop on raw signals, a filter).
- Yaw in the hover is damped, not held: the body keeps whatever heading it has.
- About ten traces were run from scratch scripts and printed, not appended to
  `trials_T1.jsonl` (the first closed-loop hovers, and the contact read of
  section 1).
- Torque is servo and joint damping, not load.

## 8 What leaving the water into the air will need from this

- **The hover is the way out.** Thrust is along the nose and 2.3-2.6 weights at
  130 degrees of stroke; the body leaves a surface nose-first. From the water
  that means arriving at the surface nose-up, which F3e's law already does to
  change depth (76-78 degrees of pitch), with the air stroke taking over as the
  wings clear.
- **The air stroke must not be run under water.** F3 measured 11 N m at the
  shoulder for the air stroke carried into water; the swim stroke is 1-1.5 Hz
  and the hover 12 Hz. The exit needs a rule for which wings are wet (the fore
  pair clears first when the body stands nose-up: its shoulders are 9 cm above
  the hind pair's).
- **The hover loop needs a beat of history** before it is trusted, and its gains
  are for air. Half in the water, the forces on the wet pair are some 850 times
  larger at the same stroke (the ratio of the densities).
- **The body is neutral in water and weighs 1.03 N in air**: thrust has to rise
  from nothing to more than a weight as the hull comes out, and the hind pair
  alone at 130 degrees lifts about 1.2-1.3 weights on the rig's figures, if the
  fore pair is already dry.
- **Once clear, the rest exists**: the hover holds and climbs, and F4's law
  takes a hover to level flight at 8 m/s in two seconds for less than a metre of
  height.
- The probe's "right way up" reads the root's up axis, which is zero for a body
  hovering on its tail: every crossing that ends in a hover has to end leaned,
  as this one does (10 degrees, +0.18).
