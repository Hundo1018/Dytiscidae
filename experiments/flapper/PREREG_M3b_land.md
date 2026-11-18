# Pass M3b, pre-registration: answering both halves of the pair by turning, and a shoulder sweep

2026-11-18, on `c52a237` (branch `triphibian-diagnosis`), MuJoCo 3.15. Written
before any arm below is tuned or scored. It follows M3 (`RESULT_M3_land.md`):

- The 0.6 kg machine stands on its wing tips.
- It does not walk a commanded heading both ways or turn to one at any M3 arm.
- What moved it was a flapping hop whose direction is set by the feather angles,
  not by running the cycle backward. M3's steering law walked backward by
  running the cycle in reverse, so it could not answer the antipodal half.

The lead asked for two things:

- **Arm T** keeps the machine and changes the law: turn in place toward the
  heading, then hop forward. It has no backward mode.
- **Arm K** changes the morphology: a third hinge per shoulder, a fore-aft sweep
  on its own witness motor, with a true stepping gait. The morphology is ours to
  change. K is a design candidate beside T, not a loosening of M3.

**The gates are M3's, at M3's tolerances and windows, unchanged.** A longer
window is read beside them, labelled secondary, and decides nothing.

## 0. Seen before freezing (probes, disclosed)

All runs are in `m3b_probe.py`, plus `m3b_land.py` on four short probe specs:
six runs in all, every one listed here. The probe specs are kept in the
scratchpad and their rows are not in any results file.

- **`build`: the sweep machine compiles** (`m3b_sweep.py`, the env's compile
  step swapped for a copy while the machine is built; `mission.py` is untouched).
  - Mass is 0.660 kg: two 30 g motors at the shoulders. The CoM moves from
    x 0.226 to 0.232 m in the hull frame.
  - Each sweep hinge has range +-35 deg, the stroke's witness motor (gear 24),
    and armature 0.01185 kg m2 as the stroke's.
  - Signs: +0.3 rad on the left sweep moves the wing back 0.094 m; on the right,
    forward 0.100 m. The controller's +sweep means "tip toward the nose" on both
    sides.
- **`step`: three strides of a keyframed sweep step**, both wings together:
  - The keyframes: stance at stroke -12, feather 0, tail -20; sweep +20 to -20
    in 1.2 s; lift to +5; sweep back; plant; 2.1 s per stride.
  - The CoM went **+1.25 m up the ramp in 6 s** (0.21 m/s, about 0.42 m a
    stride). No QACC reset; 13 W stroke, 5.5 W sweep. Pitch read +15.3 deg at
    every 0.5 s sample, which I have not explained.
- **Turns, 6 s window, yaw 7, commanded +-90:**
  - T with the right wing held planted while the left hops (tsign +1, h_turn 1,
    M3's H hop): +1.2 and -5.1 deg. Weak, so the side and the hold are tuned.
  - K with the two sweeps in opposite senses (A 20, T 2, A_turn 1): +22.0 and
    -30.3 deg, the commanded sense. That is 3.7-5.1 deg/s, against the 6.4 deg/s
    a 90 deg turn needs to finish in 14 s.

## 1. What is the same for every arm

- **Machines:**
  - **ref**, `mission.Machine("land")` as in M3.
  - **sweep**, `m3b_sweep.SweepMachine("land")`: ref plus, per shoulder, a hinge
    about the hull's vertical axis. It is placed first in the wing body, so it
    acts before stroke and feather. Range +-35 deg; the controller keeps it
    within +-32. The stroke's witness motor: tau_s 1.0 N m, omega_0 1000 rad/s,
    J_r 2e-5, gear 24, the rotor reflected as armature, the same PD and torque
    band as every hinge. A 30 g contact-free point mass at the shoulder (TYPED:
    a motor and gearhead of the witness's class).
  - **Added: 0.060 kg (all-up 0.660 kg) and 0.01185 kg m2 of armature on each
    sweep hinge.** Hull pitch inertia rises by the two point masses at 0.09 m
    from the CoM: about 2 x 0.03 x 0.09^2 = 5e-4 kg m2. The real hinge's
    structure is not modelled.
- Everything else is as M3 section 1:
  - dt 1e-4, wn 300, contact and fluid as built, the true state read by the
    controller.
  - The start: x, yaw, `drop_to_ground`, 1 s held at 0, a 1 s rise to the arm's
    stance, the command at t_c = 2 s.
  - A fresh machine and controller per half, with a hash of both halves at t_c.
  - The readings, QACC exclusion, and peak electrical power of all motors
    together per run (window and run).
- **Run length: every walk and turn run has a 24 s window.** The gates read the
  first 12 s (walk) and 14 s (turn), which are M3's windows, from the 25 Hz trace.
  - CoM displacement at t_c + 12 s; heading error over [t_c + 12, t_c + 14] s.
  - These are the primary gates and the only ones that give verdicts.
  - The 24 s read is secondary: the same per-second bars and the same angles over
    the full window (walk: pair mean 0.020 m/s, 0.005 m/s per half, cross-track
    0.5; turn: error over its last 2 s). It is reported and labelled, and decides
    nothing.

## 2. The arms

- **T** (ref machine, six motors). The hop is M3's cycle: plant, power, lift,
  return. The box is M3's G1 box with the plant feather fa limited to -54..-20.
  - The turn mechanism: one wing, the inner one, is held toward its stance pose
    (stroke th_s, feather fa) by a fraction h of its cycle. The other wing runs
    the hop. h_turn in 0..1; tsign +-1 picks which side is inner for a CCW turn.
  - Walk task: beyond e_go of the heading (10..60 deg, with 10 deg of
    hysteresis), turn in place with h = h_turn. Within it, hop forward, steering
    by h = min(h_turn, k_h |e|) on the inner side (k_h 0..3 per rad).
  - **There is no backward mode:** the antipode half turns about 180 deg first.
  - Turn task: turn until |e| <= e_stop (3..12 deg). Then blend over 0.3 s to
    M3's frozen stance S (stroke -12, feather 0, tail -20, roll loop 1) and hold.
    Turning resumes if |e| > e_stop + 10.
  - Pitch loop kp on the feather (0..2).
  - D-series: not reachable. Holding one wing while the other beats needs the
    two sides apart.
- **K** (sweep machine, eight motors). The stepping cycle per side:
  - Stance, a share D of the period: the planted tip swept from sw0 + A to
    sw0 - A, so the body goes forward. Then lift (stroke to th_up), sweep forward
    in the air, plant.
  - The feather is held at f_s plus a pitch loop. A roll loop runs on the
    differential stroke. The two sides are in phase (off 0) or alternate
    (off 0.5).
  - Box: T 0.8..3 s, D 0.4..0.85, th_s -25..-5, th_up -5..15, f_s -15..15,
    A 5..30, sw0 -10..10, tail -31..31, kr 0..2, kp 0..2, off in {0, 0.5},
    e_go 10..60, k 0..2, A_turn 0..1, e_stop 3..12.
  - **Steering is M3's law:** walk forward or backward (the cycle run in
    reverse), each side's sweep amplitude scaled by d +- k e. Beyond e_go of
    both the heading and its opposite, turn on the spot, the two sides' cycles
    in opposite senses at A_turn.
  - Turn task: as T's, holding the stance with the sweep at sw0.
  - D-series: no D-series drive has a sweep; K says what a land drive must add.
- **U** (each machine): the tuned gait with its lift (stroke) clock at 1.618
  times its sweep or feather clock. T's U is M3's U; for K the lift no longer
  keeps time with the sweep. Steering and turn laws are unchanged.
- **N1** (each machine): the rise to the frozen gait's stance pose, then every
  hinge held there.
- **N0** (each machine): every hinge held at 0 from t = 0.
- **KS** (sweep machine, a stand check, since K adds mass): M3's stand gate
  (section 3 of PREREG_M3) on its four orientations, with the stance and roll
  loop K freezes.

## 3. Gates (frozen; M3's, unchanged)

Starts are M3's scored starts:

- Walk pairs W1-W6: up/down at yaw 0, +15, -15 and across at yaw 90, 105, 75;
  phases 0, 1/3, 2/3; x 18, 24, 30.
- Turn pairs R1-R3: yaw 0, 90, 180; x 20; phase 0; commanded +-90.
- Stand: yaw 0, 45, 90, 180 at x 18, 22, 26, 30.

The bars:

- **Walk pair passes** if:
  - pair mean >= 0.020 m/s over the 12 s window;
  - each half's along >= 0.06 m;
  - each half's |across| <= 0.5 x its along;
  - the gait's pair mean is at least twice U's at that start where U's is
    positive;
  - both halves are valid.
  - "Walks" means at least 5 of 6 pairs, with at least 2 of 3 up/down and 2 of 3
    across.
- **Turn half passes** if the heading error over [12, 14] s has mean <= 15 deg
  and max <= 20 deg, and the yaw change has the commanded sign. A pair passes if
  both halves pass. "Turns" means at least 2 of 3 pairs.
- **Stand (KS):** M3's gate, 3 of 4.
- **Nulls must fail** every gate the arm passes; if one passes, that gate is
  void for that machine.

## 4. Checks and predictions (written now)

- **P1, pairing.** Every pair's hashes are equal at t_c. N0 and N1 pair means are
  exactly 0.
- **P2, T turns** (2 of 3 pairs at 14 s): p = 0.25.
  - If it turns, the rate is 6.5-15 deg/s.
  - The probe gave under 1 deg/s with one held side; the tuned side and hold are
    unknown.
- **P3, T walks** (primary, 12 s): p = 0.10. The antipode half needs about
  180 deg of turn and 0.06 m of headway in 12 s.
  - Secondary, 24 s: p = 0.25.
  - If the uphill-facing halves pass, their along is 0.6-3 m in 12 s (M3's H:
    1.3-2.3 m), airborne 0.4-0.7, peak wing force 20-50 weights.
- **P4, K walks** (primary): p = 0.55.
  - If it walks: pair means 0.05-0.25 m/s; the uphill half slower than the
    downhill half on every passing up/down pair.
  - Power 10-40 W total; peak normal force on one wing 1-6 weights; peak stroke
    torque 2-10 N m; peak sweep torque 1-10 N m; stroke and sweep on their
    curves under 2% of steps; airborne under 0.30.
  - "On its wings" (wings and tail at least 0.7 of the vertical ground force,
    body contact on at most 20% of steps): p = 0.4.
- **P5, K backward.** The cycle run in reverse walks the other way, because the
  sweep moves the planted tip and does not rely on thrust: every K down half and
  -y half has along > 0. p = 0.7.
- **P6, K turns** (primary): p = 0.5; rate 5-15 deg/s. The probe gave 3.7-5.1.
- **P7, U.** U-K is below the 0.020 m/s bar on at least 5 of 6 pairs: p = 0.7.
  U-T is below it on at least 5 of 6: p = 0.8.
- **P8, KS stands** 4 of 4: p = 0.8, at 10-15 W (M3's S: 11 W).
- **P9, QACC.** No reset in at least 95% of runs.
- **P10, nulls.** N0 and N1 of both machines fail every walk and turn pair
  (certain by construction for the pair mean; their along is under 0.01 m).

Falsified by the stated numbers.

## 5. Tuning, budget, stopping, output

- **Tuning starts:** M3's (yaw 7, -25, 97; x 21; phase 0.5), never scored. For
  each candidate set: walk pairs at the three yaws (headings 0/180, 0/180,
  90/270) and turns of +-90 at yaw 7. Windows of 14 s (walk) and 14 s (turn) are
  read at 12 and 14 s as the gates are.
- **Budget:** at most 48 tuning rollouts each for T and K, all appended to
  `results/M3b_tuning.jsonl`, none dropped.
- **Freeze rule:** M3's, `m3_freeze.py`'s normalisation (each gate quantity over
  its bar, minus 1; a set's margin is the smallest over its rollouts) and its
  eligibility (read on all three walk pairs and both turns). The frozen set is
  the eligible one with the largest margin.
  - A family freezes when its budget is spent, or when one set passes every
    tuning gate, whichever is first.
  - KS takes K's frozen stance (th_s, f_s, tail, kr, sw0); it is not tuned.
  - No parameter outside a box. A box widened after data would be a new arm,
    disclosed.
- **Scored runs:**
  - Per machine: gait 12 + 6, U 12 + 6, N1 6 + 3, N0 6 + 3 = 54; 108 in all.
  - KS: 4.
  - That is 112 runs at about 26 s simulated each.
  - At the present load (about 17 s of wall per simulated second, 4 workers):
    **about 3.4 h**, plus about 1.5 h of tuning.
- **Code and rows:** `m3b_sweep.py` (the machine), `m3b_land.py` (arms and
  runner), `m3b_read.py` (the gates, reusing `m3_read`'s pair logic on the 12/14
  s reads). Rows in `results/M3b_land.jsonl`, traces in `results/M3b_traces/`.
- **Figures** from the built model at recorded states, opened and checked:
  - K: plant, mid-stance, lift, swing of one stride, and one instant of a turn.
  - T: one instant of its turn, and the hop after it in an antipode half.
  - Files: `figures/M3b_*.png`.
- **The result** (`RESULT_M3b_land.md`): the outcome first, with the X2 blind
  spot (no dynamic stall at large feather and low speed, where every hop lives)
  and the contact caveat (rigid boxes on a rigid beach, solref 0.004, no
  structure) in the outcome paragraph. Then a table, each check against what was
  read, and everything changed after data.
