# Pass M1b, result: V+, P+ and Vs+ on fresh entries

2026-11-18. Frozen text: `PREREG_M1b_takeoff.md` (`78f470e`). Runner:
`m1b_run.py`, reader `m1b_read.py`. Rows:
- `results/M1b_takeoff.jsonl`: the freeze row first, then 136 scored rows,
  none with a QACC reset and none excluded;
- `results/M1b_tuning.jsonl`: 5 Vs+ tuning rows.

The freeze row records these hashes:

| file | sha256 |
|---|---|
| `m1_takeoff.py` | `ba0b7f69180a50ee…` (M1's, checked by the runner) |
| `m1_lib.py` | `37f3452a…` |
| `mission.py` | `bee12386…` |
| `m1b_run.py` | `cf1354be…` |
| `M1b_vsplus_gains.json` | `ca3d163b…` |

The controller reads the true state.

## Outcome

**V+ (vault, then fly) stands as a take-off verdict, and so does Vs+, the gentler strike.**
- **V+:** 13 of 14 fresh entries and 12 of 12 contact-jitter rows; 8 of 8 at
  each lighter rotor.
- **Vs+:** 14 of 14 and 12 of 12.
- **P+ (push-up, then fly) does not stand:** 8 of 14 and 4 of 12. It passes
  every up/down entry and fails every across-slope and oblique one.
- **Nulls:** N0, N2 and VO fail all 14 entries.

**V+'s one failure** is the oblique entry (28, 135). It hovered at 3.9 m for
5.5 s, then the lateral mode M1 found won. The wing went down, the body fell
onto the beach at 6.6 s (35 weights on contact) and climbed again. This is
the same instability the added lever holds everywhere else: under a slower
strike the lever holds at that entry.

**Vs+ strikes the beach at 14.6-25.4 weights** on its worst wing, against
V+'s 22.3-32.9. On the same fresh entry Vs+ is 0.63-0.74 of V+. It leaves
the ground for good at 0.10 s and hovers at 4.0 m, nose 87 deg. Its body
rate over the last 3 s is 0.08-1.44 rad/s; V+ across the slope runs
1.13-1.47.

**Peak electrical power for all motors together:**

| | V+ | Vs+ |
|---|---|---|
| instantaneous peak | 3.5-3.7 kW | 3.47-3.56 kW |
| largest 0.1 s mean | 1.78-1.83 kW | 1.57-1.69 kW |
| hover, last 3 s | 343-350 W | 343-351 W |

All at J_r 2e-5; for comparison, three cells give 462 W continuous
(reported, not a gate).

**Model caveats, same as M1:**
- The climb and hover are where the strip solver has no dynamic stall
  (X2: low by 1.5-3.5 times there, unqualified).
- The strike is a contact event between rigid wing boxes and a beach with a
  4 ms soft contact and no give. The structure that would carry 15-33
  weights at a wing tip is not modelled.

## Table (J_r 2e-5, gear 24 unless stated)

| arm | rows | class | pass | G1 G2 G3 G4 | t_off s | t H>=2 m s | H end m | span max deg | rate last 3 s | drift m | peak wing contact, weights | P peak W | P 0.1 s W | P last 3 s W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V+** | fresh | up/down | **8/8** | 8 8 8 8 | 0.12 | 1.33-1.52 | 3.99 | 0.1-0.4 | 0.07-0.09 | 0.5-1.0 | 25.5 | 3590-3657 | 1824-1826 | 343 |
| V+ | fresh | across | **4/4** | 4 4 4 4 | 0.12 | 1.69 | 3.98 | 7.4-7.9 | 1.19-1.34 | 5.7-5.8 | 25.3-25.7 | 3662-3679 | 1823-1825 | 349 |
| V+ | fresh | oblique | **1/2** | 1 1 1 1 | 0.12 / 6.58 | 1.59-1.64 | 3.44-4.00 | 5.4 / 68.2 | 0.40 / 6.43 | 5.3-6.1 | 25.4 / 202.7 | 3619-3657 | 1825 | 345 / 422 |
| V+ | jitter | all 6 entries x 2 seeds | **12/12** | 12 12 12 12 | 0.11-0.12 | 1.24-1.68 | 3.98-3.99 | 0.0-7.6 | 0.07-1.47 | 0.5-5.8 | 22.3-32.9 | 3511-3655 | 1778-1814 | 343-350 |
| V+ | 5e-6 (32), fresh | up/down | **8/8** | all | 0.10-0.11 | 1.12-1.43 | 4.00 | 0.1-0.5 | 0.06-0.10 | 0.5-1.2 | 24.0 | 2653-2658 | 1589-1594 | 166 |
| V+ | 1e-6 (32), fresh | up/down | **8/8** | all | 0.10-0.11 | 1.12-1.46 | 4.00 | 0.1-0.4 | 0.07 | 0.4-1.4 | 21.2 | 2370-2372 | 1552-1556 | 140 |
| **Vs+** | fresh | up/down | **8/8** | all | 0.10 | 1.18-1.39 | 4.00 | 0.4-1.2 | 0.10-0.26 | 0.6-1.6 | 16.1-16.2 | 3471-3528 | 1689-1691 | 343 |
| Vs+ | fresh | across | **4/4** | all | 0.10-0.11 | 1.44-1.45 | 4.00 | 7.7-8.0 | 0.18-0.24 | 5.9 | 16.4-17.3 | 3486-3487 | 1688-1691 | 344 |
| Vs+ | fresh | oblique | **2/2** | all | 0.10 | 1.31-1.47 | 3.99-4.00 | 5.8-6.2 | 0.45-1.12 | 4.7-5.5 | 16.4-18.7 | 3480-3518 | 1687-1689 | 345-348 |
| Vs+ | jitter | all 6 x 2 | **12/12** | all | 0.10 | 1.16-1.44 | 3.98-4.00 | 0.2-8.0 | 0.08-1.44 | 0.4-6.0 | 14.6-25.4 | 3465-3555 | 1566-1656 | 343-351 |
| **P+** | fresh | up/down | **8/8** | all | 1.67-3.24 | 2.31-3.92 | 3.96-4.00 | 0.6-6.0 | 0.13-1.16 | 0.5-2.3 | 48.5-61.5 | 3604-3756 | 1982-2163 | 338-344 |
| P+ | fresh | across / oblique | **0/4**, **0/2** | 0 0 0 0 | 5.7-9.0 | 3.9 | -1.4 to 4.4 | 8.7-70.2 | 4.95-8.21 | 6.4-17.7 | 145-188 | 3759-3773 | 1975-2160 | 354-828 |
| P+ | jitter | up/down / across / oblique | **4/4**, **0/4**, **0/4** | | 1.6-8.9 | | | | | | 47-215 | 3625-3911 | 1977-3501 | 338-618 |
| N0 | fresh | all | **0/14** | 0 0 0 14 | (on the beach) | - | 0.00 | - | 0 | 0 | 0 | 0 | 0 | 0 |
| N2 | fresh | all | **0/14** | 0 0 0 14 | (back down) | 0.58-0.61 | -0.09 to -0.04 | - | 0 | 0.7-1.5 | 25.3-70.9 | 2902-2911 | 1823-1826 | 0-1 |
| VO | fresh | up/down / across / oblique | **0/8**, **0/4**, **0/2** | 12 5 5 0 over the 14 | 0.12-9.0 | 0.66-0.80 | -2.8 to 8.5 | 45-85 | 1.2-7.2 | 10-27 | 25.5-165 | 3124-4685 | 1824-3926 | 213-3805 |

Jitter seeds: seed 1 is time constant ×1.169 with friction 0.509; seed 2 is
×0.848 with friction 0.714. Both are applied to every geom.

Notes on the table:
- N0 and N2 "pass" G4 only because they never leave the beach; G4 reads
  attitude from t_off on.
- N2 is thrown to H 2.02-2.05 m for an instant, then falls. G1 and G3
  catch it.
- VO open loop climbs to 23-25 m on the up/down entries and pitches over
  (G4).

## Why P+ fails across the slope (from the drawing)

`figures/M1b_Pplus_19_90.png`:
- **The stand:** the push-up stands the body on its wing tips at 0.55 m, as
  on the up/down entries.
- **No rock-back:** across the slope it never rocks back onto its tail. The
  slope is under the wings, not along the body, so the nose stays at 0-4
  deg.
- **Flapping from level:** FLY starts on P's 3 s timeout from that level
  stand. The first beats strike the beach (11 weights), and the body hops to
  2.5 m nose-up at 60-70 deg. It turns over and falls below its start
  height, since it slid down the slope. It launches again and falls again.
- **The cause is P's switch rule** (nose >= 35 deg or 3 s, frozen in M1),
  not the flying law. Up and down the beach the body rocks back by itself
  and the switch is on the nose.

## The drawings (built model, re-run through the frozen code, opened and checked)

- **`M1b_Vplus_24_270.png`** (V+, across the slope): the same sequence as
  M1's V+ at (20, up). Flat on the beach with the slope under one wing, the
  strike, clear at 0.15 s, nose vertical by 1 s, hover at 4 m. The span axis
  goes to 7-8 deg and holds, the lean against the drift. G 1111, wing
  contact 25.6 weights. Matches the row.
- **`M1b_Vsplus_24_270.png`** (Vs+, same entry): the slower strike shows at
  +0.06 s as a wide inverted V with load 12.7 weights, and at +0.15 s as the
  wings at about 40 deg below the body, not straight down. Hover at 4.0 m,
  wings level in the front view. Peak wing contact 16.4 weights. Matches the
  row.
- **`M1b_Vplus_28_135.png`** (V+, the failing oblique row): hover 3.8-3.9 m,
  nose 85-88 deg until 5.5 s. Then the nose falls to -38 deg and the span
  axis to 68 deg. It hits the beach at 6.6 s (a 35-weight spike) and climbs
  to 4.4 m again. Matches G 0000, t_off 6.58 s.
- **`M1b_Pplus_19_90.png`:** described in the section above.
- **Not in the design, shown in every drawing:** the wing used as a leg.

## The frozen predictions against what was read

| | frozen | read | |
|---|---|---|---|
| D1 | V+ stands (0.7); up/down 8/8; across+oblique 5/6; jitter 11-12/12; contact 25.5 up/down, 24-27 across; peak 3.5-3.7 kW; hover 340-350 W | stands; 8/8; 5/6; 12/12; 25.5, 25.3-25.7; 3.59-3.68 kW; 343-349 W | **holds** in every part |
| D2 | P+ up/down 8/8; across+oblique 3/6, so short of standing (0.6); contact 30-65 weights | 8/8; **0/6**; short (8/14, 4/12); 48.5-61.5 where it passes, 145-215 where it falls | the stand and up/down hold; **across+oblique wrong** (0, not 3); the contact range holds only where it passes |
| D3 | Vs+ >= 12/14 (0.5); contact 15-17 weights and >= 30% under V+'s at every entry; peak 3.4-3.6 kW; 0.1 s 1.6-1.8 kW | 14/14; **16.1-18.7** fresh; under V+'s by 26-37% (**26% at (23, 45)**; at (28, 135) V+'s row is a fall); 3.47-3.53 kW; 1.69 kW | the pass, power and 0.1 s hold; the contact range and "every entry" are **wrong** narrowly |
| D4 | V+ 8/8 at each lighter rotor; hover 160-170 W / 135-145 W | 8/8, 8/8; 166 W, 140 W | holds |
| D5 | N0 0/14; N2 0/14, peak H 1.9-2.1 m; VO 0/14 | 0/14; 0/14, 2.02-2.05 m; 0/14 | holds |
| D6 | no QACC reset | none in 136 rows | holds |

## What this establishes, as ranges

- **A standing take-off from the beach, from rest on the belly, free body,
  every hinge on its witness motor, under the frozen gates.** It is held
  across 24 headings and positions for V+ (M1's 8 scoring and 2 stress,
  M1b's 14) and two
  contact jitters (time constant ×0.85-1.17, friction 0.51-0.71).
  - V+: 13 of 14 fresh, 12 of 12 jittered.
  - Vs+ (strike ramped over 0.2 s): 14 of 14 and 12 of 12, with a third less
    wing load.
- **The procedure is a wing strike: the wings used as legs.** The push-up
  alone (P+) works only facing up or down the beach, where the body rocks
  back onto its tail.
- **The flying law needs the per-side feather-amplitude lever.** That lever
  was found after M1's data and is now confirmed on fresh entries.
  - The hover's lateral mode is held, not removed. At one oblique entry
    under the harder strike it won after 5.5 s.
  - Across the slope V+'s body rates in the last 3 s sit at 1.13-1.47
    rad/s, close to the 1.5 bar; Vs+'s are 0.15-0.26 there.
- **Lift and torque are not the limit:** hover is 140-350 W over the
  rotors. What is outside the model or over budget:
  - the strike load: 15-33 weights at a wing tip, against no modelled
    structure;
  - the strike power: 3.5-3.7 kW instantaneous and 1.6-1.8 kW over 0.1 s
    for all motors together, against 462 W of cells;
  - dynamic stall in the hover regime.

## Not checked

- Modelled sensors (M5).
- Ground effect; wakes; flexible wings; wing structure; sand that gives.
- Jitter is global (one scale on every contact), because the harness has
  none per contact.
- Vs+ at the lighter rotors (not pre-registered).
- Whether a real feather drive can change its amplitude per side beat to
  beat: a motor on each feather can, and D2's geared crank cannot.

## Reproducers

Prefix each with
`cd /home/hundo/Projects/Dytiscidae/dytiscidae && MUJOCO_GL=disable PYTHONPATH=.`

```bash
.venv/bin/python experiments/flapper/m1b_run.py score     # 136 rows, 4 processes
.venv/bin/python experiments/flapper/m1b_read.py
.venv/bin/python experiments/flapper/m1_draw.py Vs+ 24 270   # writes figures/M1_Vsplus_24_270.png (renamed M1b_ here)
```
