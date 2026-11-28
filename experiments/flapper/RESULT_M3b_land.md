# Pass M3b, result: a turn-then-hop law, and a shoulder sweep hinge

2026-11-19. Frozen text: `PREREG_M3b_land.md` (commit `33896f7`, not edited).

- Code:
  - `m3b_sweep.py`: the sweep machine, built in a copy of the compile step;
    `mission.py` is not edited.
  - `m3b_land.py`: arms and runner.
  - `m3b_specs.py`: the scored list.
  - `m3b_read.py`: M3's gates on the 12/14 s reads, plus the 24 s secondary.
  - `m3b_freeze.py`: M3's freeze rule.
  - `m3b_draw.py`: a copy of `m3_draw.py` with a top view added.
  - `m3b_probe.py`: the phase-1 probes.
  - All hashes in `results/M3b_code_sha.txt` match after scoring.
- Rows:
  - `results/M3b_tuning.jsonl`: 52 tuning rollouts, none dropped.
  - `results/M3b_land.jsonl`: 112 scored rows.
  - Traces are in `results/M3b_traces/`.
- Read and tables:
  - Frozen sets: `results/M3b_frozen.json` (sha `696fe0da…`).
  - The gate read: `results/M3b_read.txt`.
  - The table: `results/M3b_table.md`.
- No QACC reset in any of the 164 runs.
- Every controller reads the true state.

## Outcome

- **Arm K, the sweep machine, does all three things M3 asked for.**
  - It walks a commanded heading both ways: **6 of 6 pairs** at M3's 12 s
    gate.
  - It turns to a commanded heading both ways: **2 of 3 pairs** at M3's 14 s
    gate, and 3 of 3 at 24 s.
  - It stands on four of four orientations: KS, 13.3-13.7 W.
  - The machine is 0.66 kg with one fore-aft sweep hinge per shoulder (+-35
    deg) on its own witness motor. Of the 0.06 kg added, two 30 g motors are
    typed, and the hinge structure is not modelled.
- **How K walks.** Up/down pair means are 0.341-0.362 m/s.
  - Each half covers 3.7-4.8 m in 12 s.
  - The uphill half is the slower on every up/down pair: 3.7-4.0 m against
    4.5-4.8 m.
  - The downhill and -y halves walk backward by the cycle run in reverse.
  - Cross-track is at most 0.88 m against an along-track of 4.1-4.5 m.
- **How K turns.** It turns in place, sweeping the two sides in opposite
  senses.
  - It is within 15 deg of the command 3.4-4.0 s after t_c.
  - The failed pair, R1, starts facing up the ramp. It reaches the heading,
    then creeps 1.8-2.2 m on the slope while re-correcting. Over [12, 14] s
    its error is 16-23 deg mean and 31-34 deg max, against bars of 15 and 20.
- **K is a stepping gait, but not one on its wings alone.**
  - Each stride is a planted tip swept back about 0.4 m relative to the hull.
  - The swing sets the hull down: body contact 19-31% of steps, wings and tail
    0.59-0.70 of the ground force.
  - F2's "on its wings" label holds on 1 of 12 halves.
  - Each touch-down slams. Peak electrical power over all motors is 3.4-4.3 kW
    for a mean of 36.5-43.9 W. Peak contact is 21-49 N on one wing (3.2-7.6
    weights) and up to 645 N on the hull. The sweep and stroke hinges see 27 N m
    and 18-25 N m.
  - **Read these peaks as an artefact of the contact model: rigid boxes on a
    rigid beach at solref 0.004, with no structure, so impact forces are not
    trustworthy.** The mean power and the speeds are what this pass can stand
    on.
- **Arm T, the 0.6 kg machine with the turn-then-hop law: no.** 0 of 6 walk
  pairs and 0 of 3 turn pairs, at both 12/14 s and 24 s.
  - Holding one wing planted while the other hops turns the body 3-38 deg in
    14 s, and at most 50 deg in 24 s.
  - The antipode half never turns around. It hops backward down the heading's
    opposite, 0.3-1.6 m the wrong way. In W1 its yaw is -15 deg at 6 s and -34 deg at 12 s.
  - The forward halves hop 1.25-2.6 m in 12 s, airborne 16-64% of the window.
  - On this machine the land question is closed for the laws tried: M3 found
    no reversing law, and M3b finds no turning one.
- **The model's known blind spots.**
  - T's hops are in the strip solver's blind regime: large feather at near-zero
    airspeed, no dynamic stall (X2, forces off by 1.5-3.5 times). Its hop
    distances should not be trusted.
  - K's walk barely uses the fluid; it rests on contact. The same contact
    caveat applies: what K's speeds and turns show is kinematic feasibility of
    a sweep-hinge stepping gait in this contact model, not its loads.
- **The nulls fail every gate.** N0 and N1 on both machines and U-K fail all
  walk and turn pairs. U-T passes one walk pair (W1, 0.031 m/s, by hopping);
  T passes none, so no T verdict rests on it.

## 1. Table (scored, J_r 2e-5; full columns in `results/M3b_table.md`)

| machine / arm | drive | walk pairs (12 s) | turn pairs (14 s) | 24 s read | along per half, 12 s (m) | P window (W) | peak P, all motors (W) | peak wing N (weights) | peak body N | stroke / sweep tau (N m) | airborne | body contact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep K | 8 motors (sweep, stroke and feather per side; 2 tail); no D-series drive has a sweep | **6/6** | **2/3** | 6/6, 3/3 | 3.73-4.81 | 36.5-43.9 | 3357-4254 | 21-49 (3.2-7.6) | 210-645 | 17.9-24.7 / 27.0 | 0.19-0.28 | 0.19-0.31 |
| sweep KS stand | 8 motors | stand 4/4 | - | - | drift 5.4-6.3 mm | 13.3-13.7 | 17-18 window | 4 (0.6-0.7) | 0 | 2.2-2.3 / 0.3-0.8 | 0 | 0 |
| sweep U | 8 | 0/6 | 0/3 | 0/6, 0/3 | -5.58 to +1.34 | 112-499 | 3640-4853 | 71-205 | 416-627 | 25-33 / 27-37 | 0.63-0.71 | 0.08-0.14 |
| sweep N1 | 8 | 0/6 | 0/3 | 0/6, 0/3 | 0.012 | 7.5-8.1 | 107-129 | 3-5 | 0 | 1.4-1.6 / 5.4-5.9 | 0 | 0 |
| sweep N0 | none | 0/6 | 0/3 | 0/6, 0/3 | 0.000 | 0.1 | 0 window | 0 | 6 | 0.1 / 0.0 | 0 | 1.00 |
| ref T | 6 motors; one side held while the other beats needs the sides apart (no D-series) | 0/6 | 0/3 | 0/6, 0/3 | -1.57 to +2.61 | 83.5-192.8 | 2300-4597 | 125-251 (21-43) | 0-263 | 31.0-31.2 / - | 0.16-0.64 | 0.00-0.06 |
| ref U | 6 | 1/6 (W1) | 0/3 | 0/6, 0/3 | -0.51 to +0.91 | 143-361 | 2755-4884 | 165-350 | 0-270 | 37-39 / - | 0.25-0.61 | 0 |
| ref N1 | 6 | 0/6 | 0/3 | 0/6, 0/3 | 0.004 | 8.8-10.4 | 21-23 | 5-6 | 0 | 2.3-2.4 / - | 0 | 0 |
| ref N0 | none | 0/6 | 0/3 | 0/6, 0/3 | 0.001 | 0.1 | 0 window | 0 | 6 | 0.1 / - | 0 | 1.00 |

**K's walk pairs (12 s):**

| pair | pair mean (m/s) | half along/across (m) | half along/across (m) |
|---|---|---|---|
| W1 up/down, yaw 0 | 0.362 | up 3.886/0.002 | down 4.808/0.002 |
| W2 up/down, yaw +15 | 0.350 | up 3.962/0.223 | down 4.447/0.146 |
| W3 up/down, yaw -15 | 0.341 | up 3.731/-0.213 | down 4.453/-0.157 |
| W4 across, yaw 90 | 0.362 | +y 4.318/0.355 | -y 4.379/-0.602 |
| W5 across, yaw 105 | 0.358 | +y 4.462/0.818 | -y 4.132/-0.755 |
| W6 across, yaw 75 | 0.345 | +y 4.161/0.156 | -y 4.120/-0.879 |

**K's turns:**

| pair | at 14 s: yaw change, error mean/max (deg) | at 24 s: error mean (deg) |
|---|---|---|
| R1, facing up | +84.6, 16.0/33.8; -61.0, 22.6/31.0 (fail) | 5.5, 4.0 (pass) |
| R2, across | +80.6, 7.8/7.8; -92.9, 1.4/1.4 | as at 14 s |
| R3, facing down | +76.8, 13.2/13.2; -77.2, 12.8/12.8 | as at 14 s |

**Frozen sets** (`results/M3b_frozen.json`; the rest are the defaults in
`m3b_land.py`):

- K: T 1.5 s, D 0.6, stance stroke -12, lift +5, feather 0, sweep +-30 about 0,
  tail -20, roll loop 1, both sides in phase, e_go 30, k 1, A_turn 1, e_stop 6.
- T: T 1.0 s, stance stroke -10, drop 10, lift +15, feather -20 to +20, D 0.85,
  tail -20, pitch loop 1, inner wing fully held (h_turn 1), tsign -1, e_go 30,
  k_h 1, e_stop 6.

## 2. Each frozen check against what was read

- **P1, pairing. Held.** No pair failed on hashes. The N0 and N1 pair means are
  exactly 0 on both machines.
- **P2, T turns** (p 0.25): **no**, 0 of 3. Its rate is under 3 deg/s, against
  the 6.5-15 predicted.
- **P3, T walks** (p 0.10; secondary p 0.25): **no**, at 12 s and at 24 s. The
  uphill halves go 1.25-2.6 m in 12 s, inside the predicted 0.6-3, airborne
  0.16-0.64 (predicted 0.4-0.7). Peak wing force is 21-43 weights, inside the
  predicted 20-50.
- **P4, K walks** (p 0.55): **yes**, 6 of 6.
  - Falsified:
    - Pair means read 0.341-0.362 m/s against the predicted 0.05-0.25.
    - Power reads 36.5-43.9 W against 10-40; the top is out.
    - Peak wing force reads 3.2-7.6 weights against 1-6.
    - Peak stroke torque reads 17.9-24.7 N m against 2-10.
    - Peak sweep torque reads 27.0 N m against 1-10.
  - Held:
    - The uphill half is the slower on every up/down pair.
    - Stroke and sweep spend 0 and 0.5% of steps on their motor curves (under
      2%).
    - Airborne is 0.19-0.28 (under 0.30).
  - "On its wings" (p 0.4): 1 of 12 halves.
- **P5, K backward. Held.** Every down and -y half has positive along:
  4.12-4.81 m in 12 s.
- **P6, K turns** (p 0.5): **yes**, 2 of 3. Rate: within 15 deg after 3.4-4.0
  s, about 19-22 deg/s, faster than the predicted 5-15. That part is falsified.
- **P7, U. Held.** U-K is below the bar on 6 of 6 pairs; its pair means are
  -0.18 to -0.05 m/s. U-T is below it on 5 of 6; W1 reads 0.031 and passes the
  full pair gate.
- **P8, KS. Held.** Stands 4 of 4 at 13.3-13.7 W, against 10-15 predicted.
  Drift 5.4-6.3 mm, under the 20 mm bar.
- **P9, QACC. Held.** No reset in 112 scored or 52 tuning runs.
- **P10, nulls.** They fail every pair: held. Falsified: "along under 0.01 m".
  The sweep machine's N1 creeps 0.012 m in 12 s and 0.018 m in 24 s.

## 3. Figures (built model at recorded states; side, front and top views; opened and checked)

- `figures/M3b_K_stride.png` (K, W1 up the ramp):
  - Plant on both tips with the tips swept forward. In the top view the wings
    point ahead of the shoulders.
  - Mid-stance with the sweep near 0, then late stance swept back.
  - In the swing the hull carries 6 N while the sweep returns in the air, then
    a re-plant at 7.1 / 7.4 N.
  - Checked: the planted tips stay at the same world x while the hull advances.
    The motor spheres are at the shoulders.
- `figures/M3b_K_turn.png` (K, R2 +90):
  - The top view shows the two sweeps opposite at 1 s.
  - It holds at yaw 172 deg on four tip corners at the end. It started at yaw
    92, so that is an 80 deg turn.
- `figures/M3b_T_turn.png` (T, R1 +90): the right wing held flat, the left one
  hopping (105.8 N on it at 7 s), and 39 deg of yaw at 14 s.
- `figures/M3b_T_antipode.png` (T, W1 commanded down the ramp): the body still
  faces up the ramp at 6 s and 12 s (yaw -15, -34). It hops in place and never
  turns around.

## 4. Disclosed: changed or added after data

1. **Tuning budgets.**
   - K froze after 22 of its 48 rollouts, by the rule: one set, A 30 and T 1.5,
     passed every tuning gate (margin +0.418).
   - **T stopped at 34 of 48: a deviation.** None of the six turn
     variations turned more than 35 deg in 14 s. Those were the held side, the
     hold fraction, the stride period 0.8-1.5 s, the feather pair and the stance
     share. The eligible T set's margin was -20.4. The 14 unused rollouts were
     not run.
2. **`m3b_land.py` was edited twice after the prereg commit, before any tuning
   row.**
   - The KS stand arm and its drift and yaw reads were added. They were named
     in the prereg but not built at commit time.
   - A `KeyError` in the print line for windows shorter than 12 s was fixed.
   - The code hashes in `results/M3b_code_sha.txt` were taken at scoring and
     still match.
3. **The probe specs and rows** (four turn probes) are in the scratchpad, not
   in `results/`. They are described in prereg section 0.
4. **`m3b_draw.py` is a copy of `m3_draw.py`** with a top view added, because
   the sweep is not visible in side or front view.
5. **Wall time.** The scored set took about 3.7 h at machine load 6-15, against
   the 3.4 h predicted; four workers were used throughout.

## 5. Not checked

- **Loads.** K's impacts (3-8 weights on a wing, up to 645 N on the hull, 27
  N m at the sweep) come from rigid contact. A structure, a compliant tip or a
  softer beach would change them; no structural check was made.
- **The sweep hinge as hardware.** Its mass (2 x 30 g) is typed. Its bracket,
  bearing and the extra moment it puts on the stroke hinge are not modelled. It
  costs nothing measured in the air here, and its effect on flight (M1, M4) is
  not read.
- **Sensors.** Every verdict reads the true state (sensors come in M5).
- **J_r sensitivity** was not pre-registered for M3b and was not run.
