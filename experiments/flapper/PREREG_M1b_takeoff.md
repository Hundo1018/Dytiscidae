# Pass M1b, pre-registration: V+ and P+ on fresh entries, and a gentler strike

2026-11-18, on `c52a237`. Frozen before any M1b rollout. M1
(`RESULT_M1_takeoff.md`, `331e737`) found that no pre-registered arm takes
off. Two arms built after its data took off at 8 of 8: **V+** (vault, then
the flying law with a left-right feather-amplitude lever) and **P+**
(push-up, then the same law). Their lever was chosen after seeing the frozen
arms fail. This pass decides whether they stand as a verdict, on entries
neither arm has seen, and adds one arm with a gentler strike.

The controller reads the true state (modelled sensors are M5's).

## 0. Seen before freezing

- Everything in M1: the tuning rows, the 186 scored rows and the figures.
- The coordinator re-ran V+ at (20, up), J_r 2e-5, gear 24, and it
  reproduced every field.
- **Nothing at the M1b entries below, and no Vs+ rollout.**

## 1. What is frozen

- **Code, unchanged, by hash:**
  - `m1_takeoff.py` sha256 `ba0b7f69180a50eecb8b9fad6edc7d0aa2cd27c5ca9f616f2e80e9983a5344de`
    (V+ and P+ exactly as scored in M1: `GAINS`, `PLUS = dict(kpP=1.5,
    kdP=0.2)`, the arms, the reads, the gates);
  - `m1_lib.py` `37f3452a…`;
  - `mission.py` `bee12386…`, the same as M1's freeze row.
  The runner `m1b_run.py` (new) only calls `m1_takeoff.run` and writes rows.
  Its own hash, and those of the three files above, are written as the
  first row of `results/M1b_takeoff.jsonl`. The run refuses to start if
  `m1_takeoff.py` does not match the hash here.
- **Machine, world, motors, start** (lying on the belly, 1 s held, command at
  t_c = 1 s, 9 s after it), reads and gates G1-G4: as M1, sections 1 and 3,
  word for word. **A gate is never loosened.**
- **Rotor and gear:** J_r 2e-5, gear 24 for every arm. V+ is also run at 5e-6
  and 1e-6 with gear 32, on the 8 up/down entries.

## 2. Fresh entries (none used in M1's tuning, scoring or stress sets)

M1 used:
- x 15, 20, 25, 30 m, facing up and down the beach;
- 17 and 27 m up, 18 and 27 m down;
- 22 and 25 m across the slope.

M1b uses 14 entries:

- **Up/down (8):** x 16, 21, 26, 31 m, nose up the beach (0 deg) and down it
  (180).
- **Across the slope (4):** (19, 90), (24, 270), (29, 90), (33, 270).
- **Oblique (2):** (23, 45), (28, 135).

**Contact jitter.** The harness has none (`mission.Machine` and the scene
fix one contact). Two seeds stand in for it. Seed s draws, from
`numpy.random.default_rng(1000 + s)`:
- a contact time-constant scale `2**U(-1, 1)`;
- a friction `U(0.5, 1.1)`.

Both are applied to every geom, through the `contact` argument
`m1_takeoff.run` already has. Seed 1 is ×1.169 with friction 0.509; seed 2
is ×0.848 with friction 0.714. They are global, not per contact. Jitter rows
use 6 entries: (16, 0), (21, 180), (19, 90), (24, 270), (23, 45) and
(28, 135).

## 3. Arms

- **V+** and **P+**: as M1.
- **Vs+, the gentler strike.** M1's Vs procedure (stroke target ramped to
  -1.35 rad over 0.2 s) followed by the V+ law: `m1_takeoff.run("Vs+", …)`,
  which the frozen code already accepts.
  - It starts from V+'s gains unchanged.
  - **Tuning budget: 12 rollouts, on M1's 5 tuning entries only**, logged to
    `results/M1b_tuning.jsonl`.
  - Only the gains dict may change, in the keys kpP, kdP, kpy, kdy, a0,
    switch_s and switch_q. The ramp, the gates and the code may not.
  - If Vs+ passes all 5 tuning entries with V+'s gains it is frozen as is.
  - The gains used are written to `results/M1b_vsplus_gains.json`. That file
    is hashed into the freeze row before any scored row.
- **Nulls alongside:** N0 (do nothing), N2 (the strike, wings then held),
  VO (vault, open loop), as M1.

Rows:
- V+, P+, Vs+, N0, N2, VO on all 14 entries (84 rows);
- V+ at 5e-6 (32) and 1e-6 (32) on the 8 up/down entries (16 rows);
- V+, P+, Vs+ with contact jitter seeds 1 and 2 on the 6 jitter entries
  (36 rows).

136 rows, at most 4 processes.

## 4. What stands

- **An arm stands as a take-off verdict** when it passes:
  - at >= 12 of the 14 fresh entries at 2e-5;
  - and at >= 10 of its 12 jitter rows;
  - with no QACC reset among the rows it passes (a reset counts as not
    passing, as M1).
- V+ at the lighter rotors is reported against >= 7 of 8.
- Nulls must fail: N0 and N2 at 0 of 14. VO is reported (M1: 0 of 8).

## 5. Predictions

- **D1, V+:** stands, held at 0.7. Up/down 8 of 8. Across and oblique 5 of
  6: the across-slope entries were V+'s weakest in M1 (body rate 1.30-1.35
  rad/s against the 1.5 bar). Jitter 11 or 12 of 12. Peak wing contact
  25.5 weights up/down and 24-27 across. Peak electrical power 3.5-3.7 kW.
  Hover in the last 3 s 340-350 W.
- **D2, P+:** up/down 8 of 8. Across and oblique 3 of 6: its tuning
  across-slope entry failed in M1. So P+ falls short of standing (fewer
  than 12 of 14), held at 0.6. Peak wing contact 30-65 weights.
- **D3, Vs+:** passes at >= 12 of 14, held at 0.5. Peak wing contact
  15-17 weights, at least 30% under V+'s at every entry. Peak power
  3.4-3.6 kW (the strike's power is the motors at stall, not how hard the
  wing lands). Largest 0.1 s mean 1.6-1.8 kW.
- **D4, lighter rotors:** V+ 8 of 8 at each; hover 160-170 W at 5e-6 and
  135-145 W at 1e-6.
- **D5, nulls:** N0 0 of 14. N2 0 of 14, with peak centre-of-mass rise
  1.9-2.1 m. VO 0 of 14.
- **D6:** no QACC reset in any row.

Each prediction is falsified by its stated numbers. A failed D5 means a gate
reads something other than a take-off, and the arm passes are labelled until
it is understood.

## 6. Output

`m1b_run.py`, `m1b_read.py`; rows `results/M1b_takeoff.jsonl`; figures
`figures/M1b_*.png` drawn from the built model with `m1_draw.py`: V+ and
Vs+ at one across-slope entry, and any failing V+ or P+ row. Then
`RESULT_M1b_takeoff.md`, outcome first. The outcome states the X2 caveat
(the climb and hover are where the strip solver has no dynamic stall) and
that the strike is a contact event on rigid wings.
