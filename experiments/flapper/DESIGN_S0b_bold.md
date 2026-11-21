# S0b, design note: the first machine starts with every freedom, not the fewest

2026-11-19, the lead. Not a pre-registration. It states the machine and the
probes that come before one. The user, 2026-11-19: "設計可以大膽一點
盡力找到第一個機構設計".

## Why S0's ladder was the wrong side to start from

`PREREG_S0_continuation.md` (dd89844) starts from C1 and adds one freedom per
measured failure, with 24 tuning rollouts each. Its own prediction S7 (0.55)
is that it stops at a water gate. The user's method is the other direction:
a machine that surely does all three media first, the limits afterwards.
S0's ladder is held after its A0 read; its A0 numbers are kept as the
reference for what C1 does on ideal actuators.

## The idea: one thrust-vector mode in every medium

The only machine in this repository that has done every part of the mission
is the hand-flown quad B1 (`experiments/triphibian_diagnosis/existence`). It
does one thing everywhere: a thrust vector it can point, plus attitude
control on three axes. It hovers, hover-taxis on land, swims by pointing the
thrust, enters by descending, and leaves the water level and staged.

The flapping machine that has come closest is the same thing in one medium:
V+ is an axial thrust with attitude control, nose up (M1b: 13/14, 12/12).
Every measured failure is a place where the flapper is not that machine:

| failure | what the thrust-vector machine does instead |
|---|---|
| swim: no commanded progress (M2b WT) | points the body, thrust along the axis, at the stroke frequency water asks for |
| surface turn: nose-stood, inverted (M2b ST) | attitude is controlled on three axes under water, as in hover; ballast low |
| dive: 1.85x buoyant (M2) | b = 1.00, and swims down |
| exit: the vault never clears under the default mode (M2b) | swims up to speed and breaches, then the air stroke |
| after the exit: falls in 3-8 s on saturated motors (M2b XS) | ideal actuators, and attitude authority on more than one pair |
| entries: 9-14 m long, hard on land (M4) | stops in hover over the place and descends |

## The machine, T4

Flapping only. No typed force. No material, construction or drive is chosen.

- **Four wings in two pairs** (fore and hind, `body.flapper(hind_u=...)`),
  the W6 wing on each. Two pairs give pitch authority by fore-aft thrust
  difference and roll by left-right, in every medium, without leaning on
  the tail.
- **Three axes per shoulder**: stroke, fore-aft sweep (M3b's hinge), feather,
  plus the per-wing strip-twist stand-in. With sweep as a stroke the wing
  can beat in any plane, so the thrust vector can be turned without turning
  the hull.
- **Ideal actuators** on every hinge (`tau_s` 1e4, `omega_0` 1e7), every
  torque, speed and power recorded.
- **Hull**: ballast to b = 1.00 placed low (keel), so the hull rights itself
  in water. Variable ballast (±8% of displacement, 2%/s, by `body_mass` and
  `mj_setConst`) is allowed from the start.
- **Tail**: kept as C1's, so the frozen cruise law still applies.
- **Mass is whatever this comes to.** It is recorded; 0.6 kg is a limit for
  later.

## Laws, and a bolder budget

One law family, "axial thrust plus three-axis attitude", written once and
used in hover, under water (frequency scaled by the medium), and for the
breach. The frozen laws (V+, M4 L, M3b K) are tried first where they still
apply to T4.

Free parameters are tuned by an optimiser (CMA-ES or a plain random search
with restarts), not by hand: **up to 400 rollouts per manoeuvre**, on tuning
starts only. The guards are unchanged from S0 §4, and the verdicts are
unchanged: frozen gates, scored starts disjoint from tuning, do-nothing
arms, water in both entrainment modes with a split counted "not
established", the X2 flag, contact above 10 weights re-run at time constant
x2, one dt/2 re-run per scenario. An optimiser finds a model's blind
regions faster than a hand does, so the X2 flag and the both-modes rule
carry more weight here, not less.

## Probes before the pre-registration (to be disclosed in it)

Each is a few rollouts. They decide what the pre-registration freezes.

- **B0, build.** T4 builds on `mission.Machine` with twelve shoulder hinges
  on ideal motors; a figure from the built model; mass, inertia, buoyancy
  ratio before and after ballast.
- **B1, hover.** V+ on the fore pair with the hind pair in phase opposition,
  untuned: does it leave the ground, and what are the attitude moments
  available per axis (a held-body force map, as `hover_map.py`)?
- **B2, water thrust.** Held at 3 m depth, both modes: axial force against
  stroke frequency 0.5-6 Hz for the hover kinematics. Is there a frequency
  with thrust above the drag at 0.3 m/s and moments on three axes?
- **B3, breach.** Free, b = 1.00, from 3 m, nose up, B2's best frequency:
  the speed at the surface and the height the centre of mass reaches with
  the wings stopped at the surface, both modes. Does it clear by more than
  one wing length?
- **B4, still machine.** B2 and B3 with the actuators held: thrust 0, no
  breach.

If B2 or B3 fails in the default mode with every freedom above, that is the
first finding about the mechanism, and it goes to the user before anything
else is built.

## Addendum, 2026-11-19: what probes B0-B4 said, and the level hover

Probes: `results/S0b_probes.md`, `results/S0b_probes.jsonl` (106 rows, 0
QACC; the solver's fluid-force limiter engaged on 56, which are not
evidence). T4 as built: 0.918 kg, 1.159 kg with 0.241 kg of low ballast at
b = 1.0000, 14 hinges (`figures/S0b_T4_built.png`).

- **Hover, nose up, held:** 1.84 weights of thrust, moments on three axes
  (1.6-2.2 N m), peak torque under the witness stall. Free and untuned it
  held height 8 s and then diverged sideways.
- **Water, held at 3 m:** the one limiter-free point (30 deg, 0.5 Hz) gives
  61-63 N of thrust and 3-4 N m on three axes in both modes.
- **Water, free:** the hull recoils unless the wings' reactions cancel;
  only the diagonal phasing (FL+HR against FR+HL) does that, and it needs
  both pairs.
- **Breach: fails in both modes.** Best case ("wetting") reaches the surface
  at 1.55 m/s and the centre of mass peaks 0.17 m above it; clearing needs
  about 3.6 m/s. The default mode climbs 3.6x slower.
- **"Ideal actuator" (from S0's A0, 57bc449):** an unlimited hinge PD turns
  stepped targets into full-force strikes. From here the primary actuator
  is the witness curve x4 in torque and speed with blended targets; the
  unlimited motor is a diagnostic twin; a row with limiter steps or torque
  over 10x the witness stall is not evidence.

**The exit must not need the wings to work at the surface.** Both exits
tried so far (the vault, the breach) ask a wing to push on water and then on
air within one manoeuvre, which is where the two entrainment modes disagree
and where the limiter engages. A spar-buoy float (nose up, fore pair above
the waterline) does not fit this body: the fore wing spans hull x 0.13-0.39
of 0.50, so 74% of the hull would have to stand out of the water and the
most buoyant state is b = 1.26.

**Level hover instead (the sweep as the stroke).** The hull stays level,
ballast low. The wings are raised to a dihedral above the hull's plane and
beat fore and aft about the sweep hinge, feathering at each reversal, so the
stroke plane is near horizontal and the thrust is up: the hover of a
dragonfly, and the kinematics this solver was checked against (robofly).
Four wings beating like that are the quad B1 with wings for rotors: pitch
from the fore-hind difference, roll from left-right, yaw from the feather
offset.

- **Exit:** ballast out (b up to 1.26), the hull floats level, the raised
  wings are dry. Lift off as from land. No wing touches water, so the two
  entrainment modes should not differ; if they do, the float is wetter than
  drawn and the figure will show it.
- **Take-off from land:** the same, with no wing strike on the ground, which
  removes the rigid-contact peaks V+ rests on.
- **Entries, both:** stop in level hover over the place and descend onto it.
- **Under water:** ballast in to b = 1.00, diagonal phasing, as probed.
- **Cruise:** the frozen M4 law on the fore pair if it still holds on T4,
  or forward flight by tilting the hover.

What this asks of the body beyond T4 as built: the sweep range from ±35 to
about ±70 deg, a dihedral setting on the stroke hinge held during hover,
and the two pairs' stroke planes kept apart (different dihedral) so the
wings do not meet. The solver has no wing-wing interference; that is a
stated blind spot of any tandem result.

**Probes B5-B8** (before the pre-registration):

- **B5, level hover held.** Force and moment map in air for the sweep
  stroke at dihedral 20-50 deg, amplitude 40-70 deg, 2.5-6 Hz: thrust in
  weights, moments per axis, peak torque against the witness x4.
- **B6, the float.** At b = 1.26 and 1.00, level, wings raised: how many
  wing strips are wet, at rest and through one stroke, both modes. A figure
  with the waterline.
- **B7, lift-off from the float and from the beach**, free, both modes,
  with a plain attitude and height loop; then a descent back onto each.
- **B8, still machine** for B5-B7.

## Addendum 2, 2026-11-19: what probes B5-B8 said, and T5 (wings above, floats below)

Probes: `results/S0b_probes.md` (B5-B8 part), `results/S0b_probes2.jsonl`
(223 rows, 0 QACC; limiter on 22, every one with a wing or the tail in
water, not evidence), `results/S0b_clearance.json`,
`figures/S0b_T4_float.png`, `figures/S0b_T4_liftoff.png`. Every flying row
is in the X2 regime (share 0.97-1.00).

- **Level hover in air: yes.** Held, 4.5 Hz, dihedral 35, amplitude 55 deg:
  4.4 weights out of ground effect at 14.8 N m of sweep torque (witness
  stall 24); the x4 curve is never reached and the unlimited twin is
  identical. Roll, pitch and yaw each have their own input (left-right
  amplitude, common mean sweep, left-right feather offset). Free: 10 s hold,
  drift 0.08 m.
- **From the beach: yes, through wing strikes.** 2.01 m, 10 s hold, drift
  0.07 m, down again and left upright; wings hit the sand at 38 weights on
  the way up and 21 on the way down (rigid-contact figures). The hull pitches
  +-10 deg at stroke frequency.
- **From the float: no.** T4 as built is inverted in 0.6-1.6 s. The float
  itself fails first: the hull floats awash, the feather flip puts the
  trailing edge 0.19 m under the spar, so the roots work in water; with the
  wings raised the float is tail-heavy (nose to 49-76 deg) and, trimmed,
  top-heavy in roll. With the inner 0.11 m of each wing absent (T4c) the
  wings are dry, and then the tail and stern are what the modes split on.
- **Descent onto water: no** (inverted, or left nose-up 28-79 deg).
- **Wings cross**: the inner 0.10 m of each wing passes through the opposite
  root and the hull at dihedral 35.

Every failure is one geometric fact: the wings, the tail and the hull share
one height, and that height is the waterline and the ground.

**T5.** T4's wings, hinges, laws and actuator definition, with the heights
separated:

- **Floats below.** Two pontoons, parallel to the hull, apart in y and
  below it on struts. They are the only thing in the water at the float and
  the only thing on the ground. Sized from the model, not typed: at the
  float (ballast out) the lowest point of any wing through the whole hover
  stroke, and of the tail, is at least 0.03 m above the water, and the same
  above the sand on the beach; a 20 deg push in roll or pitch returns.
- **Wings above.** Shoulders stay on the hull (now the top of the machine),
  inner 0.11 m of each wing absent as T4c, or shoulders moved outboard if
  that clears with the whole wing.
- **Ballast is floodable volume**, not +-8%: under water b = 1.00 needs the
  pontoons' spare buoyancy taken up. Mass and fill rate are recorded as a
  limit for later; no rate is assumed faster than the probe shows is needed.
- **Under water** the stroke and phasing probed on T4 (diagonal); the
  pontoons' drag is measured, not assumed.

**Probes B9-B13** (both entrainment modes wherever water is involved,
witness x4 primary, unlimited twin on the rows that decide):

- **B9, build and float.** Figure from the built model with the waterline;
  mass and b in both ballast states; wet strips and limiter steps through
  the hover stroke held over the float (target 0 and 0); push tests.
- **B10, lift-off and return**, float and beach: hold 10 s at 2 m, descend,
  left upright; wing contact and limiter steps (target 0).
- **B11, under and back.** From the float: flood, level at 3 m, hold; held
  thrust against measured drag at 0.3 m/s; free swim on a heading and its
  antipode; surface; settle; lift off.
- **B12, still machine** for B9-B11.
- **B13, the x2 checks** on the rows that pass: contact time constant x2 for
  land, dt/2 once per scenario.

## Addendum 3, 2026-11-21: what B14-B23 said about T5b, and T5d (columns through the waterline, a keel fin)

T5b (T5 plus a dorsal float, pontoons 0.07 m lower; built and probed from the
approved plan of 2026-11-20, never written here) is in
`results/S0b_probes.md` parts "D1-D2 and B14-B17" and "B19-B23". What stands:
float, lift-off, landing, flood, still rows. What does not, with the measured
cause of each:

| failure | measured cause (row or table) |
|---|---|
| surfacing arrives at 6-54 deg, one start in six ends heeled at 40 deg | the empty machine has 26 N of surplus buoyancy on 2.1 kg and **no waterplane over 0.11-0.18 m of draft** (fill table: hull +0.204 m at fill 0.2, +0.023 m at 0.3, K -1.4 / -1.65 at 0.3), so it breaches at 0.3-0.5 m/s; a fixed float that closes the gap is 3.27-4.30 kg dry against 3.19 kg of lift |
| pitch divergence in a fast rise | sink-pose wings are a fin (K +2.46 at rest, -14.5 at 0.25 m/s; safe below 0.08 m/s); flat wings carry 125 kg of water centred 0.12 m aft |
| turn reaches 0-150 of 180 deg | turning one side's wings over pitches the machine 12-18 deg nose-down and gives a reverse paddle that the two entrainment modes disagree on by 3x; the held couple (17 N m) agrees between modes but does not carry the free machine |
| swim 0.299 / 0.214 m/s | thrust is not the limit (balance above 0.45 m/s at 0.7 Hz); the wetting mode loses the heading (-50 deg), the lateral mode D1 found undamped without the added-mass reaction |

### T5d: two additions, both freedoms, neither a typed force

1. **Floodable columns through the waterline.** Four vertical columns, one at
   each end of each pontoon, from the pontoon top to at least the height of the
   dorsal float's axis, open to the same ballast as the pontoons (flooded they
   weigh their shell only, so neutrality costs shell mass, not a kilogram per
   litre). Emptied, they are the waterplane the gap lacks: draft becomes a
   continuous function of fill, and their spacing in x and y is the waterplane
   inertia for pitch and roll. Size from the static table (the `kf` method of
   B19 step 3), not by rule: radius, height and x positions are free; the
   dorsal float may shrink or go if the columns replace it. The tank model
   (where the water of a partly filled column sits) is stated as built; if the
   harness can only place fill mass at a fixed point, say so and bound the
   error with the two extreme placements.
2. **A keel fin in the hull's xz plane, aft of the CoM, near its height.** A
   plate in that plane is edge-on to a vertical climb and to pitch (it cannot
   become the fin of B19 step 1) and gives side force in sideslip: yaw
   stiffness and yaw and roll damping in water through the solver's own
   strips. Size it from the held yaw derivative so that the wetting mode's
   yaw stiffness at 0.3 m/s is positive with margin; check that it stays dry
   or harmless on the float and in the hover (held hover derivatives re-read).

Laws that change with them (all hinge kinematics and ballast):
- **Surfacing is a controlled ballast rise**, not a blow: the fill is driven by
  a heave-rate loop (rise at or below 0.08 m/s, the measured safe speed of the
  raised pose), slowing to rest in the columns' waterplane before the wings
  cross, then to the float. Wing pose during the rise is free (flat with the
  sweep pitch loop, or raised from the start, or a symmetric dihedral ramp,
  the lever B19 left unprobed).
- **The turn is a forward U-turn on differential thrust** (both sides forward,
  amplitudes unequal, tail and sweep holding pitch, ballast holding depth); no
  wing is turned over. The antipode of the swim is this turn plus the same
  forward swim.

### Frozen before any T5d rollout (targets are those of B14, B17, B19-B21; none is loosened)

| probe | passes when |
|---|---|
| S (static, no rollout) | at every fill 0, 0.05 ... 1.0 with the wings in each pose the surfacing uses: draft changes by less than 0.05 m between neighbouring fills, K pitch and K roll at least +1 N m/rad; flooded b within 1.000 +- 0.002 at some fill; dry weight at most lift / 1.3 at the hover law's amplitude limit (lift re-measured held on T5d) |
| B24 float, lift-off, landing | as B14 and B17: every wing and tail strip dry afloat, returns from 20 deg pushes in roll and pitch, lift-off to 2 m, 10 s hold, landing upright, kicked hover recovers; both modes; 0 wing contact |
| B25 surfacing | 3 of 3 fresh starts S7-S9 (2-3% tank mismatch, 3-4 deg initial roll or pitch, 2-3.5 m deep; not used by any tuning row), both modes: largest roll and pitch from the first dry wing strip to 2 s after afloat under 10 deg, wings dry on the float, lift-off to 2 m held 5 s |
| B26 swim | 60 s along the heading and 60 s along its antipode after the U-turn: mean signed speed at least 0.3 m/s over the swim phases in both modes, modes within 25%, yaw error at the end under 15 deg |
| B27 turn | 180 deg (within 15 deg) in at most 80 s with a depth swing under 1 m, both modes |
| B28 still | every B24-B27 row repeated with hinges held (ballast loops running where the law has them): no lift-off, no turn, speed 0 along both commands; a ballast-only rise may reach the surface but must not count as surfacing |
| B29 chain | float, lift-off, hold, landing, flood to 3 m, swim, U-turn, swim back, surfacing, lift-off, hold: complete in both modes with 0 QACC and 0 limiter steps; dt/2 on every passing B25-B27 row and on the chain, pass/fail unchanged |

Rules as before: both entrainment modes on every water row and a disagreement
is "not established"; a row with QACC resets, limiter steps or torque above
10x the witness stall is not evidence; main actuator the witness curve x4, the
unlimited motor a twin only; X2 flagged on flying rows; rigid-contact figures
flagged. Tuning: at most 150 free rollouts over B24-B27, every one a row in
`results/S0b_probes6.jsonl`, counted; reported rows on starts no tuning row
used. Stop rules: if S cannot be met within the weight bar, stop and report
the best static table (that is the finding); if a probe is not met when the
budget is gone, report it not met.

Predictions (mine, written before the build): S is met with columns of
0.02-0.03 m radius and under 0.25 kg of added shell; B25 passes in the default
mode and its exit peak in wetting stays over 10 deg in at least one start; B27
passes in the default mode by differential thrust; B26's 25% mode agreement is
the gate most likely to stay open.

## Addendum 4, 2026-11-21: T5d stopped on wing clearance (my layout error), and T5e

Probe S on T5d (`results/S0b_probes.md`, part "T5d: S and B24-B29"): the
static table is met at column radius 0.030 m, but I placed the columns at the
pontoon ends, inside the wings' swept volume: the wings pass 0.03-0.10 m
through them in every stroke, and the model has no self-collision, so it would
have flown. Addendum 3 had no clearance line; that was the omission. Columns
cut short enough to clear the wings fail the table in the same band T5b did.
The envelope map (`s0b_t5d_envelope.py`) says where a vertical part may stand:
at y 0.20 only ahead of hull x 0.97 and behind -0.60 for the whole stroke
family.

### T5e: the waterplane goes where the wings are not

Two candidate layouts, both judged statically and kinematically (no rollout)
before either flies; fly the one that passes, A if both do (it leaves the
thrust line where it is):

- **A, columns fore and aft of the wing discs.** Each pontoon is carried
  forward and aft (as a longer thin pontoon or as a slender boom, sized by the
  table and the weight bar) to columns standing outside the envelope; their
  lateral station is free there (wider than 0.20 m is allowed and buys roll
  stiffness), the aft pair must also clear the tail surfaces over their whole
  travel. Long pitch arm, so a smaller column radius may do.
- **B, shoulders raised** by more than the trailing edges' swing (0.15 m plus
  clearance), columns at the pontoon ends as in T5d. Costs a higher thrust line
  and CoM; the float push test and the held swim pitch moment are re-read.

The keel fin of Addendum 3 stays (measured: +1.58 N m/rad at 0.3 m/s, both
modes, dry on the float). Laws, probes B24-B29, pass conditions, rules, the
150-rollout budget and the stop rules of Addendum 3 are unchanged. Probe S
gains one line, frozen now:

| S, added | passes when |
|---|---|
| clearance | at least 0.03 m between every wing and tail surface and every fixed part (hull, pontoons, columns, booms, fin, dorsal float) over the full command ranges of the hover, swim, U-turn and surfacing laws including their control-input limits, by the envelope method; reported as the smallest distance and where it occurs |

If neither A nor B passes S with the clearance line and the weight bar
(lift / dry weight at least 1.3), stop and report both tables: that is the
finding. Prediction: A passes S with under 0.15 kg added over T5d's 2.347 kg
is not assured (the bar leaves 0.108 kg); if A fails on weight, B passes S but
its swim pitch moment grows by more than half.
