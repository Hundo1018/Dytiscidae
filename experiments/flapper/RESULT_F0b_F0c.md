# F0b and F0c results — 2026-11-12

Both pre-registrations were frozen and committed before their rollouts.
Reproduce (about 9 and 8 minutes on 8 processes; no row non-finite):

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/flight_modes.py experiments/flapper/results/flight_modes.jsonl --jobs 8
    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/flapper/cruise_trim.py experiments/flapper/results/cruise_trim.jsonl --jobs 8

Weight here is the simulated dry mass times g (0.687 N and 0.721 N for the
0.20 m and 0.25 m wings; F0 quoted the phenotype's mass, 5% more).

## F0b: the frozen grid (2,075 rig runs)

Least aerodynamic power among points that carry the weight and hold the speed:

| airspeed | wing 0.20 m | wing 0.25 m |
|---|---|---|
| 0 m/s | 16.4 W (12 Hz, body at 70 deg) | 13.3 W (12 Hz, 70 deg) |
| 3 m/s | 14.1 W (12 Hz, 70 deg) | 10.2 W (12 Hz, 45 deg) |
| 6 m/s | 10.9 W (12 Hz, 45 deg) | 8.0 W (12 Hz, 45 deg) |
| 9 m/s | 8.3 W (12 Hz, 45 deg) | 18.7 W (12 Hz, 20 deg; the only cheap points over-lift 2.2x) |

Glide, stroke stopped: best lift over drag 8.9 and 10.6; least sink 0.80 m/s
at 6.6 m/s and **0.54 m/s at 5.4 m/s**, both at 12 deg of wing incidence;
stalled at 20 deg (lift over drag 2.1).

Predictions: 1 *"Level flight is found at 6 and at 9 m/s on both wings at
8 Hz or less"* — **refuted** for the 0.20 m wing (12 Hz only). 2 *"at most
half the least hover power"* — **refuted on this grid**: 0.51 and 0.60.
3 *"At 0 m/s only theta = 70 or 90 deg flies; at 3 m/s the least-power point
has theta of 45 deg or more"* — first half **refuted** (20 and 45 deg also
meet the test, by lifting while accelerating forward, which the test
allows), second half **held**. 4 *"best lift over drag between 3 and 7;
least sink under 2 m/s on the 0.25 m wing"* — first half **refuted upward**
(8.9, 10.6), second **held** (0.54). 5 *"reads the same vertical force here
within 2% ... power within a factor 0.5-2 of [momentum power over 0.5]"* —
force **held** (1.1964 against 1.1922 N), power **refuted**: 17.1 W is 3.1
times it. At 45 deg of incidence a flat plate's lift and drag are about
equal, so the power is near thrust times wing speed.

The grid's feather offsets and amplitudes hold no bird-like cruise; F0c was
written for that before this was read as a verdict.

## F0c: trimmed level flight, body level (800 trims)

| airspeed | wing 0.25 m: least power, and the stroke | points that fly level |
|---|---|---|
| 5 m/s | 3.96 W: 8 Hz, 80 deg, feather 30, incidence 16.7 deg | 3 of 80 |
| 6 m/s | 5.33 W: 8 Hz, 80 deg, feather 20, incidence 12.6 | 3 of 80 |
| 7 m/s | 5.27 W: 8 Hz, 80 deg, feather 20, incidence 9.6 | 4 of 80 |
| 8 m/s | 3.99 W: 6 Hz, 100 deg, feather 20, incidence 8.0 | 5 of 80 |
| 9 m/s | 4.86 W: 8 Hz, 80 deg, feather 20, incidence 5.9 | 6 of 80 |

The 0.20 m wing flies level at no point of this map (frequency up to 8 Hz).

Predictions: 1 *"every speed ... on the 0.25 m wing and at three or more of
the five on the 0.20 m wing"* — first half **held**, second **refuted** (0).
2 *"at most 2.0 W on each wing"* — **refuted**: 3.96 W, and nothing on the
smaller wing. 3 *"frequency <= 6 Hz and stroke <= 80 deg"* — **refuted** (8
Hz). 4 *"a feather amplitude of 10 deg or more"* — **held** (30; a wing that
only heaves never held the speed). 5 *"peak inertial torque at most one
fifth of the hover point's"* — **refuted**: 0.19 against 0.42 N m.

## By the frozen rule, and what else it shows

- On the 0.25 m wing level flight at 6 and 9 m/s takes 5.3 and 4.9 W against
  13.3 W in hover (0.40 and 0.37), and the stopped wing glides at 10.6. The
  rule's two conditions hold: **the machine is flown like a bird, forward
  flight and glide; static thrust above weight is kept as the manoeuvre that
  leaves the ground and the water.** The design wing is 0.25 m or longer; the
  0.20 m wing is dropped.
- Flapping to hold the speed is dear in this model: the glide says the drag
  power is about 0.4-0.5 W and holding the speed costs 4-5 W. The wing is
  rigid and its feather angle is the same from root to tip, while the
  stroke's speed grows with radius, so the inner wing is over-feathered and
  the outer wing under-feathered. Spanwise twist, by a second panel or by
  compliance, is the first design variable this points to, and it is where
  "membrane or feather, stiff or soft" gets decided by a number.
- Hover, where it is needed, costs 170-220 W per kg of aerodynamic power on
  these wings; the literature's hover machines run at about 210-220 W/kg of
  input power (report A11, KUBeetle-S, inferred there).
- The hand estimate in F0 used Dickinson's 1999 lift fit from memory; A11
  could not open that paper and derives `CL = 1.77 sin 2a` from Dickson and
  Dickinson (2004). At 45 deg the two give 1.80 and 1.77.

Limits: held body, quasi-steady model, rigid wing, coarse grids (each "least
power" is the least on its grid, an upper bound), aerodynamic power only.
