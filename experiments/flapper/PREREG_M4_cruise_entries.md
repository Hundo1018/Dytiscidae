# Pass M4, pre-registration: six-axis cruise with heading, and entries into water and onto land

2026-11-18, on `32e3e16` (branch `triphibian-diagnosis`), MuJoCo 3.15.
Written before any scored rollout. `TRIPHIBIAN_SPEC.md` names this pass M4
(L0 row 6: "air to land, air to water: unknown, none computed"). Harness:
`mission.py` (`Machine`, M0 and its addendum), **not edited**; `sensors.py`,
`freerig.py`, `drive/` not edited; `m4_probe.py` is the lead's and is not
edited. New code only in `experiments/flapper/m4_*.py`; rows in
`experiments/flapper/results/M4_*.jsonl`.

Two questions. (a) With all six axes free, does the 0.6 kg machine hold level
flight at its cruise speed on a held heading, and turn to a commanded heading
both ways (an antipodal pair, `CLAUDE.md` "Designing a measurement" rule 8),
for 60 s, with attitude, height and speed inside stated bands? (b) From that
cruise, does it (i) enter the water and end afloat or submerged, upright, not
tumbling, and (ii) touch down on the beach and come to rest upright on its
hull or wings, each at a commanded place, with the loads at entry recorded?

The controllers read the **true state** (position, velocity, attitude, rates,
which strips are wet, ground contact). Modelled sensors come in M5; one
reported-only arm here reads `sensors.py` (LS, section 2). Every verdict below
is a verdict on a machine with a perfect state estimate.

## 0. Seen before freezing (probes, disclosed)

Every run below is disclosed; none is a scored row. Harness at its defaults
(dt 1e-4, PD wn 300, witness motors, gears 24/16/50, J_r 2e-5), the machine of
C1 as `m4_probe.py` builds it (hull transverse inertia set to 0.012 kg m2, as
C1 and pass 1m), the tip's feather by the strips turned per side
(`Machine.enable_strip_twist`, D3's stand-in for its tube and rod).

1. **The lead's probe** (`m4_probe.py`, `results/M4_probe_6dof.jsonl`, seven
   runs, 40 beats from 8 m/s level with 5 deg of roll), read again from its
   rows: with no roll loop, 3.5 Hz / 30 deg / tip 10 deg ("P35") held roll
   within 1 deg after 2 s, yaw within 2.8 deg and height within 0.2 m over
   11.4 s, speed rising 7.9 to 8.6 m/s; 2.5 / 40 / 15 ("P25", the cheapest
   C1 point) rolled 5 to 13 deg and flew a coordinated turn of 141 deg in
   16 s (turn rate within 10% of g tan(roll)/V) at constant height. The
   roll loop on the tips' feather offset at kr = +-1 deg/deg, krd = +-0.1:
   with kr = -1 both points rolled over within one beat; with kr = +1, P35
   held roll within 1.3 deg for 4 beats while yaw ran -1.4, -4, -10, -21,
   -45 deg beat by beat, then it crashed; P25 the same, slower. Electrical
   19-37 W.
2. **What each lateral input does** (`m4_derivs.py`, `results/M4_derivs.jsonl`
   13 runs and `M4_derivs2.jsonl` 4 runs). Launch level, C1's longitudinal
   pilot, no lateral loop, 8 beats; one differential input stepped and held
   3 beats; read against the unstepped baseline from the same launch
   (deterministic: same code, same state). Inputs are differential, + on the
   +y side (seg1, seg4; `set_twist`'s "right" is the +y side). Change in roll
   / yaw / track (velocity heading), deg, after 1, 2, 3 beats:

   | input, step | P35 roll | P35 yaw | P35 track (3 beats) | P25 roll | P25 yaw |
   |---|---|---|---|---|---|
   | root feather offset, 2 deg | -13.3, -21.1, -22.4 | -4.1, -14.0, -27.5 | +6.8 | -18.2, -28.0, -24.9 | -12.2, -29.0, -61.0 |
   | tip feather offset (strips), 3 deg | -11.9, -18.7, -19.9 | -3.7, -13.9, -27.5 | +6.0 | -17.2, -27.6, -24.8 | -11.1, -27.7, -56.9 |
   | tip feather amplitude (strips), 3 deg | -3.0, -0.9, +4.8 | -2.9, -12.7, -27.0 | - | -4.3, -2.6, +3.7 | -6.2, -17.5, -30.7 |
   | stroke offset, 4 deg | -3.5, -2.4, -0.9 | -0.7, -2.5, -4.4 | -1.2 | -3.3, -1.7, -0.1 | -1.5, -3.1, -4.3 |
   | stroke amplitude, 4 deg | -0.4, -0.7, -0.6 | -0.1, -0.7, -1.1 | +0.2 | -1.9, -1.9, -2.1 | -0.9, -1.5, -1.5 |
   | tail halves, 5 deg (second batch) | -3.1, -3.7, -3.0 | -1.0, -4.8, -9.2 | - | -3.9, -3.6, -2.9 | -4.3, -8.7, -10.0 |

   Read: the root feather offset and the tip feather offset are one input
   (same signature, same pitch-up of 5-12 deg and height loss of 0.3-0.9 m
   in 3 beats); each rolls the machine hard and **yaws its nose the other
   way from the turn its bank makes** (track +6.8 deg, nose -27.5 deg: 34
   deg of sideslip in 0.86 s; the bank, -22 deg, has the +y side down). There is no fin, and nothing stops it. The
   tip feather *amplitude* is nearly a pure yaw input (roll small and
   changing sign). Stroke offset and amplitude are weak. This is why the
   lead's kr = +-1 loop failed: its roll gain was about ten times what the
   step response allows (2 deg gives 22 deg of bank), and nothing held the
   nose to the track. In the first batch the "tail" rows drove both halves
   together (my side mapping read only the wings; fixed before batch two):
   a common 5 deg step, pitch -2.6 deg, no roll. Not used.
   Copy check: the lead's no-loop P35 run re-flown through `m4_common` agrees
   over 11 beats within 0.004 deg in pitch, 0.003 deg in roll and yaw, 0.2 mm
   in height, 0.6 mm/s in speed (not bit-identical; the launch is set by a
   different code path).
3. **The law's structure, once, with hand-picked gains** (`m4_smoke.py`,
   `results/M4_smoke.jsonl`, six runs; these gains are the frozen law L of
   section 2, not tuned on anything else). Bank by the root feathers' offset,
   track by bank, sideslip by the tips' amplitude:
   - P35, launched rolled 5 deg, heading held 15 s: after 5 s roll within
     1.0 deg, track within 0.4 deg, sideslip within 0.7, height +0.10 to
     +0.23 m, speed 8.35-8.68 m/s, inputs within 0.5 deg (feather) and 1.2
     (tip amplitude); mean electrical 27 W.
   - P25, the same: roll within 2.4, track within 1.1 deg, height within
     0.10 m, speed 7.58-7.88, 18 W. (Open-loop it turned 141 deg in 16 s.)
   - P35 told +90 and -90 deg at 3 s (two runs from one state): exact mirror
     images to the printed digit; peak bank 27.1 deg (asked 15: the roll
     loop overshoots), track overshoot to 97.6, inside 10 deg of the command
     3.25 s after it and within 1 deg from about 7 s, height -0.14 to +0.22
     m, speed 8.18-8.67, sideslip within 6.2 deg, peak 50 ms electrical 113 W.
   - **Do-nothing over the sea** (P35 at 3 m, cruise law to 2 s, then every
     hinge held: strokes level, feathers at the offset, tips untwisted, tail
     where it was): it glided, nose falling to -15 deg, and met the water at
     8.2 m/s, -2.0 m/s vertical, 30 m on, at x = 6.2 m, where the beach
     already lies 0.2 m under the surface; it stopped there in 0.45 s and
     ended afloat, level in roll, 4 deg nose-up. **The water arms below start
     far enough out that nothing reaches the shore.**
   - **Do-nothing toward the beach** (P35 at 4.5 m, held at 1.5 s, x = 7.4):
     it pitched up to 31 deg, climbed 1.5 m, stalled (2.4 m/s), dived to -53
     deg, recovered and met the ramp at x = 24.5 at 3.2 m/s; it was at rest
     0.5 s later, upright (roll -1.5, 10.8 deg nose-up on the 6.8 deg ramp).
     One 10 Hz sample read 76 weights of contact force (the impact).
   - Hold transient: holding the strokes at 0 from mid-beat drew 786 W for a
     50 ms bin. The do-nothing arms below start at a beat boundary (stroke at
     0, rising), which removes the position snap but not the braking: the
     stroke is at its fastest there (about 5.8 rad/s at P35). The braking
     transient is reported with each null's power, not gated.
   - Cost: 8.1 min of wall for 76 s of simulated flight on 4 processes beside
     the other passes: **about 20 s of wall per simulated second per process**
     (M3's 12 s was on a quieter machine).

   What the two do-nothing runs say for the gates: a still machine let go
   from cruise **can end afloat and upright, and at rest and upright on the
   beach**. "Ends upright at rest" does not tell a controlled entry from a
   fall here; the commanded place and the state at first contact must
   (section 3).

## 1. What is the same for every arm

- **Machine.** `m4_common.build`: `Machine("air", body=c1_pilot.body())`,
  0.6 kg, BODY_S, W6 wing (Goe 417a, camber 0.04, feather hinge 0.1 chord
  behind the LE), the 0.06 m2 all-moving tail, hull transverse inertia 0.012
  kg m2, witness motors (stroke and feather tau_s 1.0 N m, omega_0 1000 rad/s,
  gears 24 and 16; tail 0.1 N m, gear 50), dt 1e-4, wn 300. Built in air at
  its launch state, so the PD gains are the air ones (M2 section 0 item 4);
  in the water phases that loop is soft, and the hinge tracking error is
  reported. **Entrainment mode: the lead's decision at review, as in M2.**
  Written here as the solver's default (`solver.entrainment` as built), with
  "wetting" (M2a's primary: it reacts only water arriving at a strip, which
  is what an entry is) as the sensitivity on EW's best pair and NW at S1. If
  the lead makes "wetting" primary, as for M2, the two swap and nothing else
  changes. Strip solver in
  every medium; forces only from shape, fluid and contact, nothing added.
- **Tip twist.** The tip's feather (its amplitude and its per-side offset) is
  the fluid strips turned (`Machine.enable_strip_twist`), D3's stand-in for
  its tube and rod. Every verdict says so in its outcome paragraph. The
  roll input is the root feather hinge, on its own motor (not a stand-in).
- **Longitudinal law, every arm while it flies:** C1's tail PI (kp 1.0, kd
  0.1, ki 2.0, limit 25 deg, rate 8 rad/s, one-beat filter) and pilot (kh 6,
  kvz 6, pitch ask within 12 deg, frequency within +-20% at kf 0.08) as
  `m4_probe.fly`, held height = the launch height, speed asked 8 m/s.
- **Points.** P35 = 3.5 Hz, stroke 30 deg, tip 10 deg, root feather -3.02
  deg (W6/C1's point with thrust in hand), **the primary point**: every entry
  arm flies it. P25 = 2.5 Hz, 40 deg, tip 15, -2.96 (the cheapest; C1: cannot
  climb 2 m). Cruise gates run at both; a verdict names its point.
- **Launch and command time.** Launched level at V0 along heading psi0 with
  roll0, as each start says (section 3); law L on from t = 0, told psi0.
  t_c = the first beat boundary (phase a multiple of 2 pi) after 5.0 s. Up to
  t_c every arm and both halves of a pair run identical code on an identical
  state; at t_c the runner hashes `qpos`, `qvel`, `act`, `ctrl` and the
  controller's filter state of each half, and a pair whose hashes differ is
  not scored (a harness fault, reported).
- **Arms that ignore the command** (L0, N, NW, NL) are run once per start;
  their two halves are that one run (the engine is deterministic: M3 section
  0; checked again here by C1). Every other half is a fresh `Machine` and a
  fresh controller.
- **Readings, every run, 10 Hz plus event peaks.** Position, velocity, track
  (atan2 of the hull's horizontal velocity), body yaw, roll, pitch, sideslip
  (yaw - track), rates, the inputs; per motor `E_elec` and `E_shaft`
  differenced over the window; peak electrical power of all motors summed,
  over a 50 ms window and in a single step; share of steps on each motor's
  curve; peak |hinge torque| per role (`actuator_force`); steps at a joint's
  range limit and the peak limit constraint force (`efc_force`, joint
  limits). Around an entry (from 0.5 s before first contact to 2 s after,
  every step): the shoulder reaction of each wing (`cfrc_int` after
  `mj_rnePostConstraint`: force in weights and moment in N m, all loads,
  fluid, inertia and contact together), the fluid force on each wing
  (`xfrc_applied` after the step, less the added mass's weight cancellation
  `m_add g` the solver writes), the ground contact normal force per body
  group (wings, tail, hull with boom; `mj_contactForce`), in weights; wet
  strip count. Structure is not modelled (S2 gives ranges only): **loads are
  reported, never gated.** Energy and peak power are reported beside L0's
  three-cell continuous maximum of 462 W, not gated.
- **QACC.** A run with `bad_qacc > 0` is excluded from every verdict, counted
  as not passing, listed with its gate, never read as a zero.

## 2. The arms

Inputs (`m4_common`): all differential, + on the +y side.

**L, the lateral law (frozen now, `m4_common.LAW_L`, not tuned):**

    phi_c   = clip(-0.5 * wrap(psi_c - track), +-15 deg)            bank asked
    u_f     = -0.09 phi_c + 0.1 (roll - phi_c) + 0.01 p              root feather offset, deg, within +-6
    u_y     =  0.4 beta + 0.15 d(beta)/dt                            tip feather amplitude, deg, within +-8

roll, p (deg/s, about the hull's x), beta = wrap(yaw - track) and
d(beta)/dt = world yaw rate less the track's rate, all low-passed over one
beat as the pitch loop's inputs are. The sign of each term is from section 0
item 2 (feather + puts the +y side down, roll negative, and turns the track
toward +y; tip amplitude + turns the nose toward -y; `Machine.roll_deg` is
positive for a rotation that lifts the +y side). With this law the root feather does the roll; the
tips' twist carries the sideslip (strips: the stand-in).

- **L0, no lateral loop.** At t_c the lateral inputs go to 0 and stay there;
  the longitudinal law goes on. "C1's pilot alone".
- **N, do nothing.** At t_c (a beat boundary) every hinge is held: strokes at
  0, root feathers at the point's offset, tips untwisted, tail where it is.
  No law of any kind. The run ends at the sea (CoM under 0.5 m).
- **F, frozen command** (one start per point, the turn): both halves told
  psi0 + 90. They must be the same run to the last bit: a check of the
  pairing.
- **LS, sensors (reported, not gated).** L and C1's pitch loop reading
  `sensors.Sensors` (defaults, `init_true=True`: armed before launch, 30 s
  filter) for attitude and rates, and its pressure height; speeds and track
  still true (no air-data or GPS model exists). P35, start K1, hold and the
  +90 half.
- **dt (reported).** L at P35, start K1, the +90 half at dt 5e-5.

**Entries.** P35, L on (told psi0 = 0, along +x) until first contact. After
t_c the entry law E of the medium, one family:

    approach: height asked = cruise until the glide path is met; then along a
              straight path at angle gamma to the flare point, height h_f over
              the surface under the CoM (sea: 0; beach: the ramp), at x_f = x_e - d_f;
              on the path the height loop's vertical-speed term reads (vz - vz_path).
    flare:    from x_f, pitch asked theta_f (height loop off), stroke amplitude
              times a_f, frequency times r_f, speed loop off.
    contact:  from first contact on, every hinge held (strokes at b_c, root
              feathers at the offset, tips untwisted, tail at e_c). No law.

x_e is the commanded place of first contact (section 3). Box, tuned per
medium in phase 2 on the tuning starts only: gamma 2-15 deg; d_f 0-12 m; h_f
0.2-2.0 m; theta_f -5 to +25 deg; a_f 0-2.8 (2.8: an 84 deg stroke, M0's
take-off stroke is 85); r_f 0.8-1.3 (1.3: 4.55 Hz, M0's is 4.5); b_c -20 to
+60 deg; e_c -30 to +30 deg.

- **EW, water entry** with E tuned over the sea; **EL, land entry** with E
  tuned onto the beach.
- **NW, NL, do nothing.** From t_c of the same start, N's hold. NW is
  launched over the sea, NL toward the beach, as their E arm.
- **EF, frozen command** (one start per medium): both halves told the far
  x_e. Must be bit-identical.

D-series: the lateral law moves each root feather by its own motor and the
tips' twist amplitude per side; a single crank for both wings' stroke (D1's
four-bar) is compatible with L (L leaves the strokes common), but D3's
crank-geared tube makes one twist amplitude for both tips and cannot give
u_y. E's flare changes amplitude and frequency in flight, which only DD (D1)
does. Stated, not judged.

## 3. Gates (frozen)

Samples at 10 Hz unless stated; "1 s mean" is a moving mean over 1 s.
Dorsal axis z_b (hull z in world), z_b.z its vertical component.

**Cruise starts** (scored; x0 -40 m, z0 30 m): K1 (roll0 +5, psi0 0, V0 8.0),
K2 (-5, 60, 8.0), K3 (+3, 135, 8.4), K4 (-3, -45, 7.6), K5 (0, -120, 8.2),
K6 (+8, 180, 7.8). The same six at both points. No tuning starts: L is
frozen.

**H, hold** (one half per start): told psi0 from t_c; window t_c to t_c + 60
s. Passes if at every sample of the window: |wrap(track 1 s mean - psi0)| <=
5 deg; |roll| <= 10 deg; |pitch| <= 10 deg; |z - z0| <= 1.0 m; 1 s mean
horizontal speed in 7.0-9.5 m/s; and the run is valid. (The height band is
twice C1's because a 2 deg roll input alone moved height 0.3 m in 3 beats,
section 0 item 2. Both bands are checked against C1's own late window under
this pilot, 100 beats, x-z-pitch free (`results/C1_pilot.jsonl`, arm A2):
P35 +0.19 to +0.25 m and 8.58-8.78 m/s, P25 -0.07 to +0.01 m and 7.66-7.90
m/s, so a hold that fails these bands fails for something other than the
standing offset C1 measured. The speed band admits that offset and stops
above 7.0, short of the 6 m/s regime where X2 reads the strip
solver off by 1.5-3.5 times.)

**T, turn pair** (per start): half A told psi0 + 90 at t_c, half B psi0 - 90;
each 60 s. A half passes if: during t_c to t_c + 15 s, |roll| <= 35 deg,
|pitch| <= 15, |z - z0| <= 2.0 m, speed in 6.5-10.0; from t_c + 15 s to t_c
+ 60 s every hold clause above, about the new heading; and the unwrapped
track change over the half has the commanded sign. The pair passes if both
halves pass and are valid, and the pair mean P_t = (dpsi_A - dpsi_B) / 2 >=
60 deg (dpsi: unwrapped track change from t_c to the mean over the last 10
s). For any arm that ignores the command P_t = 0 exactly.

**Cruise verdict at a point:** "holds heading and turns" if L passes H on at
least 5 of 6 starts and T on at least 5 of 6, N passes neither on any start,
and L0 passes T on none. **If L0 passes H on any start at a point, H at that
point is reported as unable to tell the loop from no loop** (the lead's P35
run drifted 2.8 deg in 11 s; it could stay inside 5 deg for 60 s), and that
point's verdict rests on T alone, which holds the new heading for 45 s.

**Entry starts** (heading 0, along +x): water from x0 -110 m, z0 3.0 m;
land from x0 -50 m, z0 5.0 m. Scored: S1 (roll0 +4, V0 8.0, dz0 0, psi0 0),
S2 (-4, 8.3, +0.3, +3), S3 (+2, 7.7, -0.3, -3), S4 (-2, 8.0, 0, +6) (dz0
added to z0, deg and m/s as above). Tuning: U1 (+6, 8.1, +0.2, -5), U2 (-6,
7.9, -0.2, +5). **Commanded pair**: half A told x_e near, half B far; water
-45 and -33 m, land 16 and 28 m (ramp height there 0.98 and 2.42 m), 12 m
apart. "First contact": water, the first step any strip's submerged fraction
is above 0; land, the first step with a contact between a machine geom and
the ground carrying normal force.

A half passes if:

- E1, place: the CoM's x at the step of first contact within 3 m of x_e (3 m
  is typed: about two spans). (A law that ignores
  x_e meets the water or the ramp at one place in both halves, and 12 m > 2
  x 3 m: it can pass one half at most.)
- E2, arrival, at first contact: CoM velocity along the surface's normal
  (water: vz; land: the ramp's normal) >= -1.0 m/s; |roll| <= 15 deg; pitch
  -5 to +30 deg (nose-up positive; land: relative to the ramp's 6.8 deg).
- E3, rest: water, within 10 s of first contact, over the last 2 s of those
  10: CoM at most 0.3 m above the surface (in the water: afloat or under),
  1 s mean CoM speed <= 0.3 m/s, mean |omega| <= 0.5 rad/s, z_b.z >= 0.5 at
  every sample (upright: dorsal axis within 60 deg of up). Land, within 6 s
  of first contact, over the last 1 s of those 6: CoM speed <= 0.05 m/s,
  |omega| <= 0.2 rad/s, z_b.z >= 0.5, CoM over the ramp (x 8-39.8 m), and
  hull and wings carry at least 0.8 of the vertical ground force (on its hull
  or wings, not its tail).
- E4, not tumbling: from first contact to the end of the E3 window, z_b.z >=
  -0.25 at every sample (M2's clause).
- E5, valid: no QACC reset; hashes equal at t_c.

A pair passes when both halves pass. Reported per pair: P_e = (x_A - x_B) /
(x_eA - x_eB), with x the first-contact places (0 exactly for an arm that
ignores the command, 1 for perfect placing). **Entry verdict per medium:** E
"enters" if its pairs pass on at least 3 of 4 scored starts, with its N arm
on 0 of 4. E2 is what a controlled entry adds beyond the place (the smoke's
do-nothing met the water at -2.0 m/s, 15 deg nose-down). E2 has no bound on
the horizontal speed, on purpose: the land smoke's do-nothing met the ramp at
3.2 m/s after a stall and a dive, slower than this machine can fly at P35, so
a speed bound would pass that null and fail a flown approach. That null had
recovered to 10.6 deg nose-up by contact, so E2's attitude clauses may pass it
too; on land the place (E1) is the clause no null can pass on both halves.
E3 and E4 are required but are not evidence of control (both do-nothing smoke
runs ended upright and still).

**Nulls must fail.** N fails H and T on every start; L0 fails T on every
start; NW and NL fail every pair. If a null passes a gate, that gate is
reported as unable to tell a still machine from a working one and no verdict
of that gate is claimed.

## 4. Checks and predictions

Probabilities and numbers are mine, written now, after section 0.

- C1. Pairing: every scored pair has equal hashes at t_c; F and EF halves
  bit-identical (state difference 0.0); L0, N, NW, NL pair statistics exactly
  0; a second run of L0 at K1 bit-identical to the first.
- C2. L at P35: H on 6 of 6 (p 0.85); on passing runs, largest track error
  under 2 deg, |roll| under 5 deg, height within 0.5 m, speed 8.0-9.0 m/s;
  mean electrical 22-40 W.
- C3. L at P35: T on at least 5 of 6 (p 0.75); peak bank 20-32 deg;
  overshoot 3-15 deg; inside 10 deg of the command 3-8 s after it; height
  excursion during the turn 0.1-1.5 m; P_t 85-95 deg; the two halves of each
  pair within 10% of each other on peak bank and time to arrive (the machine
  is symmetric). Peak 50 ms electrical in a turn 60-200 W.
- C4. L at P25: H on at least 5 of 6 (p 0.7); T on at least 4 of 6 (p 0.5).
  If T fails there, the clause that fails is height or speed during the turn
  (C1: no climb margin), not heading.
- C5. L0: at P25 fails H on at least 5 of 6 (track drift over 5 deg in 60 s;
  open loop it turned 141 deg in 16 s). At P35 no prediction on the count;
  p 0.5 that it passes H on at least one start (then H is not claimed at
  P35). Fails T on 12 of 12 by construction.
- C6. N: in the sea within 40 s on 12 of 12; fails H and T everywhere.
- C7. Power: L's mean electrical at P35 22-40 W, at P25 15-30 W; L0 within
  10% of L's at the same start; every peak 50 ms window in cruise under 250
  W, under L0's 462 W.
- C8. Water, EW: enters on at least 3 of 4 (p 0.45). On passing halves:
  vertical speed at first contact -0.2 to -1.0 m/s, horizontal speed 3-8
  m/s, upright and still within 10 s, ending afloat (CoM within 0.15 m of
  the surface; buoyancy 1.85 weights, M2). Peak shoulder reaction in the 2 s
  after contact 1-10 weights per wing; peak stroke hinge torque 1-15 N m.
- C9. NW: fails E1 on 4 of 4 (certain: one place for both halves); E2 fails
  on at least 3 of 4 (vertical speed under -1.0 m/s or nose down past -5:
  smoke -2.0 and -15); E3 and E4 pass on at least 2 of 4.
- C10. Land, EL: enters on at least 3 of 4 (p 0.3). The risk: it cannot fly
  slower than about 6 m/s on P35's stroke, and a flare with a take-off-sized
  stroke stalls near the ground (C1's climb failure) or costs over 300 W. On
  passing halves: horizontal speed at contact 2-6 m/s, slide 0.5-4 m.
- C11. NL: fails E1 on 4 of 4; comes to rest upright (E3) on at least 2 of 4
  (smoke did).
- C12. Loads on land: peak ground normal force on one wing 2-20 weights and
  on the hull 2-30 weights at 1e-4 s contact (impulsive: a 10 ms mean is
  reported beside it); peak stroke torque 2-24 N m (24 is the motor's stall
  through gear 24; a joint at its range limit is reported separately with its
  constraint force).
- C13. Entry power: peak 50 ms electrical in EW's flare under 300 W; in EL's,
  150-700 W (above 462 W if a_f is above 2).
- C14. LS (reported): holds the heading within 5 deg over 60 s (p 0.6); the
  +90 half arrives within 10 deg (p 0.6).
- C15. dt 5e-5: final heading within 1 deg, peak bank within 2 deg, mean
  electrical within 5% of the dt 1e-4 half.
- C16. QACC: resets in under 1 in 10 runs.

Falsified by the stated numbers. A failed C1 means the pairing is not trusted
and no pair verdict is read until it is fixed (disclosed).

## 5. Tuning, budget, stopping, output

- **Cruise: no tuning.** L is the law of section 0 item 3, frozen. If it
  fails, that is the result; a re-tuned L would be a new arm, disclosed as
  added after data.
- **Entries.** Hand tuning, no optimiser, on U1 and U2 only, both halves of
  their commanded pairs. Every tuning rollout (one half, at most 25 s
  simulated) is appended to `results/M4_tuning.jsonl` with its full parameter
  set and its gate quantities; none is dropped. Budget per medium: 30
  rollouts and 750 simulated s. The parameter set is frozen into
  `results/M4_frozen.json` (sha256 quoted in the result) when the budget is
  spent or when one set passes every clause on both tuning pairs, whichever
  comes first; the frozen set is the one with the largest smallest margin
  over the tuning pairs' gate quantities. No parameter outside the box.
- **Scored runs.** Cruise, per point: L 18 halves (6 H, 12 T), L0 6, N 6, F
  2; LS 2, dt 1; about 4,000 simulated s (N ends early). Entries: per medium
  8 E halves, 4 N, 2 EF, about 350 simulated s. Tuning up to 1,500 simulated
  s. At about 20 s of wall per simulated second per process: about 34 h of
  CPU, **about 8.5 h of wall at 4 workers**. Detached with a log
  (`results/M4_run.log`), output filtered.
- **Stopping.** When the scored table is filled; or when an arm is shown
  blocked with its trace (then its remaining rows are still run as frozen,
  and a failing table is the result).
- **Code.** `m4_common.py` (machine, laws, recorder; at this freeze it holds
  the build, C1's pilot, L and the do-nothing hold, as the probes ran them), `m4_run.py` (rows),
  `m4_read.py` (gates and checks), `m4_draw.py` (figures from the built
  model at recorded states). Rows: `results/M4_cruise.jsonl`,
  `results/M4_entries.jsonl`, traces in `results/M4_traces/`.
- **Figures** (`figures/M4_*.png`), from the built model at recorded states,
  opened and checked against section 1 (span 1.400 m, chord 0.258 m, tail
  0.500 x 0.120 m, hull, CoM): cruise at mid-turn (bank, tip twist
  differential visible), water at first contact and at rest, land at first
  contact and at rest, each in side and front view, contacts and their
  forces marked.
- **Result** `RESULT_M4_cruise_entries.md`: outcome first (cruise per point;
  water; land; each yes or no with the arm), a table, each check above
  against what was read, everything added after data disclosed. Every
  outcome paragraph states that the tip twist is the strips' stand-in for
  D3's mechanism, and every entry verdict that the model has no splash,
  wave, spray or free-surface deformation and no dynamic stall (X2: forces
  off by 1.5-3.5 times at large feather and low speed, which is where a
  flare lives).

## 6. Harness change needed

None. Everything read above is available from outside `mission.py`:
`xfrc_applied` still holds the step's fluid wrench after `Machine.step`
(it is zeroed at the top of the next call), `cfrc_int` after
`mj_rnePostConstraint`, contact forces by `mj_contactForce`, limit forces by
`efc_force`, and the lateral inputs through `Machine.step`'s targets and
`set_twist`.

## Addendum M4a (frozen 2026-11-18 by the lead, before any phase-2 row)

Entrainment: "wetting" is primary and the solver's default the sensitivity,
the same as M2 (addendum M2a), so the water-entry rows and M2's swim rows
read the same water. Nothing else changes; the budget stands as written
(P25 at six starts kept).
