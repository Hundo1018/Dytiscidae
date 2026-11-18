# Pass S0, pre-registration: from a machine that passes every scenario, back to the limits

**FROZEN, 2026-11-19**, after M2b's result (1875d71): rung A5, the swim item A5w,
the posture item A5p and the entrainment line are filled in from it. No scenario ran before this commit;
probe P1 is disclosed in §0b. Any change after it is a new pre-registration.

## 0. Question and what hangs on it

The user, 2026-11-18: "先不考慮任何結構限制或電力限制，幾乎只考慮環境物理
用一個必定能成功三棲運動的結構 從該結構作為出發點逐步符合各項限制". On
2026-11-19 they added: wing count and form are free ("不限定翅膀數量或型態"),
and buoyancy may be variable or fixed ("浮力可變可不變").

There are two questions:

- **Q-up.** With the hardware limits lifted, which freedoms does a machine
  need to pass all six scenarios of `TRIPHIBIAN_SPEC.md` under the
  environment's physics alone?
- **Q-down.** Starting from that machine and putting the limits back one at
  a time, which limit is the first to break a scenario, and by how much
  margin?

What hangs on the answer: which limit L0/L1 must spend on first, or that the
concept does not close at this mass (TRIPHIBIAN_SPEC, "Not found, also a
result").

The end point still has to meet TRIPHIBIAN_SPEC items 1-6 unchanged. The
relaxed start is a route to a working point, not a loosened gate.

## 1. What is known, and the rivals

The start is C1 (BODY_S, W6 wing, 0.06 m2 tail, 0.6 kg on `mission.Machine`).
It is the only body with measured passes:

| capability | result | source |
|---|---|---|
| standing take-off and hover | V+ 13/14 and 12/12; Vs+ 14/14 and 12/12 | M1b, 8822764 |
| six-axis cruise and ±90° turns | H 6/6, T 6/6 at P35 and P25 | M4, d44adbc |
| stand on the ramp | 4/4 | M3 |
| walk both ways, turn in place (with the sweep hinge, 0.66 kg) | walk 6/6, turn 2/3 at 14 s | M3b, 07fafb5 |
| water exit | XV 5/6 under "wetting", 0/6 under the default entrainment | M2, cadf9ab |
| exit then 30 s aloft | XS 1/6 under "wetting", 0/6 under the default (GX30 needs 4/6) | M2b, 1875d71 |
| swim under water, surface turn | WT 0/6, WT-VB 0/6, ST 0/6 in both modes | M2b, 1875d71 |

Measured failures, and the cause each result names:

- **Dive.** Buoyancy is 1.85 times the weight (M2).
- **Swim.** No pre-registered arm passed: surface 0/6, under water 0/6, VB
  0/6 (M2).
- **Entries.** Placing failed: the water entry lands 8.9-14.3 m past the
  commanded place, and the land arrival is too hard (M4, 0/4 each). The
  glide path is anchored late, and C1's height loop trails it by 1-3 m.
- **Flight after exit.** It falls back into the water in 4-10 s (M2). In
  M2b, every XS fall under "wetting" (seeds 10, 12, 14, 15, at 2.7-8.2 s) had
  the stroke and feather motors on their limit for 76-98% of steps at
  3.0-3.7 kW. The one seed that stayed up (11, 39.6 s at 4.0 m, 351 W) had
  no motor on its limit. Under the default the vault never clears the water
  (CoM peak 0.34-0.46 m, motors on the limit 93-95%).
- **Swim.** M2b: the flow-angle feather law makes no commanded progress in
  either mode (|P| <= 0.018 m/s under the default; -0.22 to +0.11 m/s
  under "wetting", with no depth hold), as built or at 1.08x buoyancy. The
  rowing turn turns, but stands the hull on its nose or rolls it inverted
  on 22 of 24 halves. There is no law with a measured swim anywhere.

Rivals for "S0 passes":

- **R1, it passes by the freedoms.** The freedoms below are what the
  failures lacked.
- **R2, it passes by the model's blind regions.** The model is wrong where
  the machine relies on it:
  - X2: low speed with large feather, which is where take-off, exit and the
    flapping hop happen;
  - rigid contact with a 4 ms time constant;
  - the entrainment mode.
  This is the boring winner when a relaxed machine "succeeds".
- **R3, nothing generalises.** A pass on the tuning starts fails on the
  scored starts.

The guards in section 4 are aimed at R2 and R3.

## 2. Step 0: the starting machine and its freedom rungs (Q-up)

The rungs are taken **in this order, cumulatively**. After each rung, every
scenario that has not passed yet is run. **Stop at the first rung where all
six pass**, and freeze that machine as S0*.

A rung is taken only if a scenario still fails, and only the rung named for
that failure. Every rung taken is reported with the scenario that needed it.

- **A0, the base.** C1 with:
  - **Ideal actuators.** Every motor at `tau_s = 1e4`, `omega_0 = 1e7`. This
    is probe P1 (section 0b): tracking error 0.017 rad, never on the band,
    0 QACC. Torque, speed, power and contact load are recorded every step,
    never gated.
  - **Per-side feather amplitude.** The strip-twist stand-in, as V+.
  - **Laws read the true state.**
  - **The frozen laws.** V+ or Vs+ for take-off and hover, M4's L for
    cruise, M3b's K for walking and turning.
- **A1, shoulder fore-aft sweep.** M3b arm K's hinge, ±35°, now ideal. Land
  needs it: M3b's T is 0/6 without it, K is 6/6 with it. It is taken at once
  for the land, land<->air and three-media scenarios.
- **A2, fixed density.** Hull ballast set so that the buoyancy ratio is
  b = 1.00. If dive or exit fails at b = 1.00, b goes to 0.97 and then 1.03,
  and the first b that passes both is kept. All three values are fixed here.
- **A3, variable buoyancy.** Taken only if no single b from A2 passes both
  dive and exit.
  - Mass is moved between the hull and the sea as ballast water: ±8% of the
    displacement, at most 2% per second.
  - It is applied as a change of `body_mass` on the hull, followed by
    `mj_setConst` on a scratch `MjData`.
  - Buoyancy is still the solver's displaced-volume force. No force is
    typed.
- **A4, hover descent for entries.**
  - At the commanded place, the machine changes from cruise to V+ hover, then
    descends at a commanded -0.5 m/s until first contact.
  - This is a law, not structure. It answers M4's failure to place the
    entry.
- **A5, exit then flight.** M2b's XS did not pass (1/6 and 0/6). A5 is
  therefore: M2b's XS law as frozen (the vault, then V+ verbatim from the
  first moment the hull is clear), now on ideal actuators (A0), and tuned
  under §5's budget only if it still fails. The measured cause of the
  "wetting" falls was motors on their limit, which A0 removes. The cause of
  the default's failure (the vault never clears) is not a motor limit by
  that reading. If the exit passes only under "wetting", the water<->air
  verdict is "not established, the modes disagree", never a pass.
- **A5w, a swim law.** No pre-registered swim law has passed (M2b). If the
  water scenario's swim or surface turn still fails after A2 (or A3), the
  WT and ST laws are tuned under §5's budget on ideal actuators. The free
  parameters are the feather's offset and amplitude relative to the flow
  angle, the stroke amplitude and frequency (inside §4's band), and for ST
  the side-to-side amplitude split. Nothing else is free. A swim that fails
  here goes on to A5p.
- **A5p, posture in water.** M2b's ST failed on posture (nose-down or
  inverted on 22 of 24 halves), and WT made no commanded progress. Neither
  cause is reached by A5w's knobs alone. Taken only if the swim or surface
  turn still fails after A5w, in this order:
  1. **Attitude feedback through the A1 sweep.** The two shoulder sweeps,
     already in the machine from A1, take an offset ``kp * error + kd *
     rate``: pitch through the common offset, roll through the difference.
     The gains are tuned under §5's budget.
  2. **Ballast low.** A2's (or A3's) ballast is placed at the hull's keel,
     40% of the hull's half-height below the centre of buoyancy, which
     gives a righting moment in water. The ballast mass is the same; only
     where it sits changes. This is a body change through `body.built`.
  A swim that fails after A5p goes on to A6.
- **A6, wing count and form.** Taken only for a scenario that still fails
  after A0-A5. The candidates are fixed now, in order:
  1. tip chord ×1.5 on the same span;
  2. span ×1.25;
  3. a second wing pair in tandem, 0.3 m behind the first, on the same laws
     (the hind pair at the fore pair's phase plus 180°).
  
  Each candidate is a fresh build through `body.built`. The scenario that
  failed is re-run with that candidate.

If the scenarios still fail after A6, the outcome is "no step-0 machine on
this harness". It is reported with the scenario and the gate that blocked
it.

### 0b. Seen before freezing (disclosed)

- **P1** (2026-11-19, lead): `Machine("air")`, 4 Hz stroke ±0.6 rad, 0.5 s.
  - Witness motor: tracking error ≤ 0.097 rad, on the band 5.4% of steps,
    peak torque 24 N m.
  - Ideal (`tau_s` 1e4, `omega_0` 1e7): tracking error ≤ 0.017 rad, never on
    the band, peak torque 162 N m (start-up).
  - 0 QACC in both.
  - No scenario was run.

## 3. The scenarios and their gates

Every gate is an existing frozen gate. **None is re-written.**

| scenario | legs, in order | gates (frozen in) |
|---|---|---|
| air | assisted launch (decision 2), cruise hold 60 s, ±90° turn pair | M4 H and T |
| land | stand, walk the commanded heading both ways, turn in place | M3 stand, M3b walk/turn |
| water | float, dive and hold depth, swim both ways, surface | M2b GW pairs; dive and surface: M2's depth clauses |
| land<->air | standing take-off, cruise 60 s, hover-descent landing at a commanded place, take off again | M1 G1-G4, M4 H, M4 land E1-E5 |
| water<->air | exit from afloat, flight 30 s, water entry at a commanded place, exit again | M2b GX30, M4 water E1-E5 |
| land<->air<->water | the five legs above, chained, 15 min in total, energy read from the run | each leg's gates, at each crossing |

Scoring follows the frozen gates:

- **Entries and the per-medium pair gates.** Every gate uses ≥2 scored
  starts. Antipodal or commanded pairs are used where the frozen gate
  defines them. Starts are disjoint from every M-series tuning start.
- **Water.** Every water gate is run under **both** entrainment modes, and
  passing needs both. M2b showed the modes can agree on a verdict while
  disagreeing on the behaviour (the exit: up to 4.0 m under "wetting",
  never clear under the default). A gate that passes in one mode and fails
  in the other is "not established", and S0* cannot be declared on it.
- **Do-nothing arms.** One per gate: actuators held still, the machine
  dropped, floated or set down as that gate's frozen null. A null that
  passes makes its gate unable to tell, and no verdict rests on that gate.
- **The 15-minute three-media run** is run once at S0* and once at the
  final step of §6, on 2 seeds. The legs are scored as they come.
  - Cost estimate: M4's 60 s cruise took 1,500 s of wall time, so 15 min is
    about 6 h per run.

## 4. Guards against a relaxed machine that only fools the model (R2, R3)

The hardware is relaxed. **The model's valid range is not.**

- **Stroke frequency** stays inside the band the strip solver was checked
  in: the robofly validation and M1-M4, 2.5-6 Hz.
- **A gate passed only in the X2 regime** (airspeed < 6 m/s with feather
  > 30° over more than half of the gate's window) is reported **"passes
  in a regime the model is known wrong in"**. It is counted, and named in
  the outcome paragraph.
- **Contact.** Every row reports peak contact as a multiple of the weight.
  Above 10 weights, the row is flagged as a "rigid-contact figure". For the
  gate that leans on it, a re-run at time constant ×2 must also pass. This
  is the contact-jitter idea of M1b.
- **Time step.** One row per scenario is re-run at dt/2. A gate that flips
  is reported as not resolved.

## 5. Tuning, budget, stopping

- **Tuning starts** are separate from scored starts. The budget is 24
  rollouts per rung per scenario that it is taken for. A rung whose budget
  is spent with no pass on the tuning starts is "rung fails".
- **Freeze rule** (as M3b): the first parameter set that passes every
  tuning gate is frozen by hash. The scored rows follow.
- **Processes:** at most 4. Nothing in `runs/`.
- **Output** per row:
  - recorded peaks of torque, speed and power per hinge;
  - energy per leg;
  - peak contact.
  These feed §6.
- **Figures:** for every rung taken, a figure from the built model with the
  added part visible, opened and checked against this file.

## 6. Q-down: putting the limits back

The order is set after S0*, by a rule fixed now: **the limit S0* exceeds by
the largest ratio goes first.** The ratio is recorded peak over the limit's
real value. Ties go in the order listed.

The limits:

1. **Torque-speed motor**, witness of D1, per hinge. The real value is the
   witness curve. The sweep and any added hinge get the witness at the gear
   that M3b typed.
2. **Power:** 462 W continuous for three cells, and the burst figure of
   A14.
3. **Mass:** 0.6 kg all up, including the added hinges, the ballast system
   and any second pair.
4. **DOF count:** each freedom taken in §2 is removed again, newest first.
5. **Hull size:** 40 mm, and every mechanism inside it, as TRIPHIBIAN_SPEC
   lists.
6. **Sensors:** `sensors.py` in place of the true state (M5's content), with
   the do-nothing arms.

Each limit is tightened in four geometric steps, from the recorded peak to
the real value (ratio^(1 - k/4), k = 1..4). After each step, the scenarios
that limit can touch are re-run:

- A limit that passes at k = 4 is kept, and the next limit is taken.
- The **first failing step** is the finding. It is reported with:
  - the step k;
  - the margin (the last passing value over the real value);
  - the gate that broke;
  - the figure.

Two notes on the failing case:

- After a failure, the pass goes on with the next limit, but with the broken
  limit relaxed to its last passing step. The report names every limit that
  failed, not only the first.
- **No gate is loosened at any point.** A limit that cannot close is "not
  found", as TRIPHIBIAN_SPEC says.

## 7. Predictions

- **S1, air and land<->air.** Pass at A0 plus A4 (land entry), with A1 for
  the land legs. (0.7)
- **S2, land.** Passes at A1, as M3b's K did. (0.8)
- **S3, the exit under "wetting".** Passes GX30 (>= 4/6) at A0 with XS
  untuned, because the falls were motor-limited. (0.6)
- **S4, the exit under the default.** Still fails at A0: the vault does not
  clear, and ideal motors do not change what the default entrainment
  holds. It fails through A5's tuning too, so water<->air ends "not
  established, the modes disagree". (0.55)
- **S5, swim.** Fails at A0-A2 with the WT law. A5w's tuning gives a pair
  progress > 0 in both modes, but the depth hold fails in at least one mode,
  so the water scenario reaches A5p. A5p's low ballast keeps the hull
  upright (no inverted half in either mode), but the swim still fails the
  depth hold in one mode, so the scenario reaches A6. (0.5)
- **S6, buoyancy.** Dive and exit close at one fixed b, so A3 is not taken.
  (0.6)
- **S7, the outcome of Q-up.** At least one water scenario is not
  established or fails after A6, so the outcome is "no step-0 machine on
  this harness" with the blocking gate named, more likely than an S0*.
  (0.55)
- **S8, if S0* exists: the first limit to break is the torque-speed
  motor.** It breaks at the strike of a standing take-off or of a water
  exit. S0*'s recorded peak torque will be ≥ 4 times the witness's 24 N m at
  the stroke, and the failing step will be k = 1 or 2. (0.6)
- **Rival R2** predicts instead: S0*'s passes rest on X2-regime windows or
  on contact peaks above 10 weights, and the contact ×2 re-run fails. If
  that happens, S0* is not a working point, and the outcome says so.

If S7 holds, Q-down is not run on a machine that does not exist. The pass
instead reports which gate blocked and the run order of §2 stops there.
Each scenario's verdict is still reported.
