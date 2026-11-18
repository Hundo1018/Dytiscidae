# Pass M2, result: free swim with the flying wings, and water to air

2026-11-18. Frozen text: `PREREG_M2_water.md` (e70721f) with addendum M2a
(32e3e16). Code: `m2_common.py`, `m2_laws.py`, `m2_fly.py`, `m2_run.py`
(sha256 4bd6e821784b2e35, 1cab9aef650c01b4, 6e71350755375f63,
869593034bcf6389, written into the first row of each rows file), readers
`m2_read.py`, `m2_draw.py`, tuning `m2_tune.py`, added checks `m2_checks.py`,
`m2_extra.py`. Rows: `results/M2_trials.jsonl` (61 tuning rollouts),
`results/M2_rows.jsonl` (127 scored rows), `results/M2_checks.jsonl` (C0,
C8, C9 and the added checks). Logs, the driver script, the section-0 probe
scripts and working notes: `results/m2_logs/`. `mission.py` and every other
existing file unedited. Every row: dt 1e-4, gear 24, entrainment "wetting"
unless stated, controllers reading the true state (modelled sensors are M5).
No row had a QACC reset (0 of 127 scored, 0 of the checks).

## Outcome

**(a) Swim: no.** The machine does not swim where it is told, at the surface
or under it, on any pre-registered arm. At the surface (GS) the paddle moves
the body tail first at 0.00-0.02 m/s (pair mean; the gate asks 0.05), with
yaw authority of a few degrees per ten seconds, so it cannot turn to an
antipodal heading: 0 of 6. Under water (GW) the 1.85-times buoyancy wins:
every as-built arm (WV, WK and the still W0) ends at or near the surface
whatever depth it is told (told 4.1-4.4 m, the do-something arms end at
0.0-2.5 m): 0 of 6 for WV and WK. The labelled buoyancy variant VB (1.08
times weight) stayed near its 3 m start but could neither go down when told
nor make way: 0 of 6.

**(b) Water to air: not by the pre-registered strategies, yes by one added
after data, and only under one of the two added-mass bookkeeping modes.**
The two strategies frozen before data both fail on 6 of 6, as predicted
(C6): the stroke biased up afloat (XB) and the take-off stroke on surfacing
from depth (XD) put the wings in the water, the motors sit on their limit
70-99% of steps, and the body never rises 0.35 m. A strategy found in
tuning and **added after data** (trials T19, T20: a down-stroke into the
water threw the body clear), the **vault** (XV: the stroke stepped down to
-1.0 rad against the water, then M1's flying law the moment no main-wing
strip is wet), meets the frozen exit gate on **5 of 6 scored seeds** from
afloat, and on 3 of 6 from depth (XDV: rise passively, vault on surfacing):
clear of the water 0.4 s after the strike, climbing 1.0-1.5 m/s to 3.5-6.8
m, the do-nothing and swim-only arms at 0 of 6. **Every one of those
flights ends back in the water after 4-10 s** (attitude lost, large yaw and
lateral drift); the gate asks 3 s. **Under the solver's default entrainment
reaction the vault leaves the water on 0 of 6** (CoM at most 0.33-0.46 m
up; check C9 and its addendum): the exit rests on which of two unvalidated
added-mass bookkeepings (S1) is right. And, as the frozen text requires
here: the strip solver has no dynamic stall (X2: forces off by 1.5-3.5
times at large feather and low speed, which is where the strike and the
first beats of the climb happen), and the free surface is a density ramp
per strip over its half-height, with buoyancy on the submerged fraction
and no wave, splash, spray, surface deformation, wave drag, planing lift or
exit suction. The vault throws a 0.6 kg body off 26 kg of wing added mass
in 0.4 s; nothing in this model says how much of that a real wing gets
from real water. The strike costs 690-850 J in 0.4-0.5 s, 4.1-4.3 kW
peak (50 ms window) against L0's 462 W three-cell pack; the climbs that
met the gate drew 300-330 W.

## The table (scored seeds 0-5; frozen gates)

| gate | arm | met | what was read |
|---|---|---|---|
| GS (surface) | WS paddle | **0 / 6** | pair-mean progress -0.002 to +0.024 m/s; one half of each pair goes the wrong way on 5 of 6; stays at the surface and upright on 6 of 6; mean electrical 5-204 W, peak 0.42-3.96 kW |
| | S0 still | 0 / 6 | progress 0 exactly; drift under 0.001 m/s |
| | SF frozen | 0 / 6 | progress 0 exactly |
| GW (under water) | WV pitched sweep | **0 / 6** | half told deeper (4.1-4.4 m) ends at 0.0-1.8 m on 6 of 6; told shallower (1.7-1.9 m) ends at 0.0-1.2 m; P +0.009 to +0.060 m/s (>= 0.05 on 1 of 6); upright on 4 of 6 |
| | WK scull | 0 / 6 | P -0.108 to -0.007 m/s (negative on 6 of 6); ends at 0.0-2.5 m; dorsal axis past the bound on 5 of 6; 158-646 W |
| | W0 still | 0 / 6 | at the surface (0.0 m) by 40 s on 6 of 6 (0.5-1.0 m after 10 s); P = R = 0 exactly |
| | WF frozen | 0 / 6 | P = R = 0 exactly |
| | VB (WV on 1.08 x buoyancy; labelled variant) | 0 / 6 | half told deeper ends at 3.03-3.37 m (target 4.09-4.35); told shallower 2.01-3.17 m (target 1.65-1.91); R -0.05 to +0.46; P -0.019 to +0.019 m/s |
| | VB-W0 still on VB | 0 / 6 | rises to 0.15-0.17 m by 40 s |
| GX (water to air) | XB biased stroke afloat | **0 / 6** | never clear; CoM at most 0.05-0.17 m up; stroke motors on limit 98-99% of steps, 3.75-3.85 kW mean |
| | XD take-off stroke on surfacing | **0 / 6** | the shoulder reaches the surface at 10.7-11.6 s on 6 of 6; clear at most 0.1 s; CoM at most 0.13-0.33 m up |
| | **XV vault afloat (added after data)** | **5 / 6** | clear from 0.4-0.5 s for 4.1-10.3 s; height 3.5-6.5 m; seed 4 never clear (its strike did not empty the wings in 2 s) |
| | **XDV rise, vault on surfacing (added after data)** | **3 / 6** | seeds 0, 2, 3 clear 4.3-6.7 s, 3.9-6.8 m; seeds 1, 5 clear 0.7-0.9 s; seed 4 never |
| | X0f still afloat | 0 / 6 | never clear, never above the surface |
| | X0d still from depth | 0 / 6 | surfaces; CoM at most 0.01-0.03 m up |
| | X1f paddle afloat, no exit | 0 / 6 | never clear; at most 0.09 m up |
| | X1d sweep at depth, no exit | 0 / 6 | never clear; it rose from 3 m to 1.0-1.7 m though told to hold 3 m |

The exit flights, read from the 10 Hz traces (`m2_read.py flights`):

| arm, seed | first clear | clear until | max height | mean climb, first 2 s clear | flight power | strike energy to first clear |
|---|---|---|---|---|---|---|
| XV 0 | 0.5 s | 4.6 s | 3.48 m | 1.34 m/s | 330 W | 852 J |
| XV 1 | 0.4 | 6.5 | 6.04 | 1.01 | 311 | 757 |
| XV 2 | 0.4 | 9.6 | 3.98 | 1.20 | 331 | 765 |
| XV 3 | 0.4 | 10.7 | 6.44 | 1.50 | 308 | 769 |
| XV 5 | 0.4 | 7.2 | 6.52 | 1.47 | 301 | 749 |
| XDV 0 | 12.2 | 18.9 | 6.75 | 1.17 | 298 | 688 |
| XDV 2 | 12.2 | 16.5 | 3.85 | 1.30 | 318 | 716 |
| XDV 3 | 12.5 | 17.2 | 4.78 | 1.03 | 322 | 814 |

Times from the arm's start (after the 3 s float settle for XV). Every
flight in the table ends wet before the row's end.

## Against the frozen checks

| | frozen | read | |
|---|---|---|---|
| C0 | exit-arm machine held at 90 deg, M0's take-off stroke 6 beats: lift within 2% of 1.840 | 1.797 weights (-2.3%); its stroke kp 1613 (M0's 1798) | **fails by 0.3 points**: by the frozen rule the exit rows are on a different loop than M0's validated one. Why the stroke inertia differs (0.0179 against 0.020 kg m2 for a machine built in air) was not traced |
| C1 | W0 rises 0.2 m or more on 6/6 both halves; GW 0/6; S0 drift under 0.02 m/s, GS 0/6; P = R = 0 exactly; X0f, X0d GX 0/6, X0d max height under 0.15 m | W0 rises 3.0 m (to the surface) on 6/6; GW 0/6; S0 drift at most 0.0004 m/s; GS 0/6; P, R exactly 0 (W0 seed 0 run for both halves: traces identical); X0f, X0d 0/6, X0d max 0.03 m | holds |
| C2 | WS meets GS on 5/6, progress 0.05-0.25 m/s, 10-80 W | 0/6, P -0.002 to +0.024 m/s, 5-204 W | **falsified** |
| C3 | no as-built arm meets GW; failing clause the deeper half's depth on 4/6 for the better arm; P >= 0.05 on 3/6 for the better arm | 0/6 for WV and WK; WV's deeper half short by 2.5-4.3 m on 6/6; WV P >= 0.05 on 1/6 | holds but for the progress clause (**falsified**: 1 of 6) |
| C4 | VB's arm meets GW on more seeds than as built; depth-held on 4/6 | 0 against 0; held on 0/6 (deeper half 0.7-1.3 m short) | **falsified** |
| C5 | SF, WF: abs(P) < 0.02, abs(R) < 0.2 on 6/6 | 0 and 0 on 6/6 | holds |
| C6 | no arm meets GX on any seed; XB max height under 0.5 m on 6/6, motors on limit > 30% while wet; XD shoulder clears on 3/6, no strip clear 1 s on any seed | XB and XD: 0/6; XB max 0.05-0.17 m, stroke on limit 98-99%; XD shoulder at the surface on 6/6, clear at most 0.1 s. **XV 5/6 and XDV 3/6** | holds for every pre-registered arm; **falsified by the arms added after data** |
| C7 | swim arms mean under 100 W and peak under 1 kW; exit arms peak over 400 W | WS mean 5-204 W, peak 0.42-3.96 kW; WV mean 36-251 W, peak 1.6-4.2 kW; exit peaks 4.1-4.3 kW | **falsified** for the swims (most rows over 1 kW peak); holds for the exits |
| C8 | best swim arm (WS, seed 0, named before data) at dt 5e-5: P within 10%; J_r 5e-6, 1e-6: power falls | P +0.0094 against +0.0080 (+18% of a number 0.04 of the bar); half A power 171 against 160 W. J_r: WS half A 160, 167, 167 W (does not fall); XV seed 0: 2925, 2371, 2377 W, GX met at every J_r (clear 4.2, 6.9, 7.1 s) | WS dt: over 10% (on a progress of 0.008 m/s); WS J_r: **falsified**; XV J_r: falls from 2e-5, then flat |
| C9 | the solver's default entrainment on the same rows: P within 20%, GX count unchanged | WS seed 0: P -0.0071 against +0.0080; XV seed 0: not met (max 0.43 m) against met | **falsified on both** |
| QACC | under 1 in 10 rows | 0 of 127 (and 0 in every check) | holds |

Reported, not gating (lead's request): the peak electrical power of all
motors together over 50 ms exceeds L0's 462 W three-cell figure in every
powered exit row (4.1-4.3 kW) and in the swims' peaks (WS 0.42-3.96 kW,
WV 1.6-4.2 kW, WK 2.0-2.7 kW); the still arms stay under 10 W.

## What this says, and what limits it

- **Buoyancy decides the water row.** Fully submerged the machine is 1.85
  times its weight (the hull alone 1023 cm3). Nothing the wings make under
  water at these motors outweighs the 5 N excess: told 1.1-1.4 m deeper,
  every as-built do-something arm ended shallower than it started. With the excess
  cut to 0.47 N (VB) the body holds about its depth but the depth response
  is under half of what was asked and the progress is 0 +- 0.02 m/s. So
  the as-built failure is buoyancy first, and with buoyancy removed it is
  thrust: at a body-to-wing added-mass ratio of about 1:44 a flapping stroke
  moves the body, not the water (section 0, p6), and the only propulsion
  found was the feather oscillating (sculling), which in this model drives
  the body tail first at 0.04-0.09 m/s for 80-1600 W (T3, T7, T9, T12). A pitching plate at
  almost no forward speed is outside what the strip solver was built for
  (real pitching foils thrust toward the leading edge); that direction is
  a model reading, not a fact about the wing.
- **Steering is the second wall at the surface.** The waterplane holds the
  heave that sinks the stroke under water, and the paddle makes way, but
  differential amplitude and feather offset turn the body 4-17 deg per 10 s
  in open loop (T13-T15) and 15-56 deg in 20 s closed loop (T54, T55, T61):
  a heading behind the machine is out of reach in 40 s.
- **The exit is an impulse off the water, then flight.** Pressing the wings
  down 1 rad against the water (26 kg of added mass) throws the 0.6 kg body
  up at about 3 m/s and pitches it nose-up, so that the wings are clear
  0.4 s later; M1's flying law then climbs at 1.0-1.5 m/s. The flying law is
  M1's, with its own post-hoc gains and two of mine; it was not built for a
  thrown, spinning start, and every flight is lost within 4-10 s. A machine
  that is airborne for 3 s and then falls back meets this gate; it has not
  shown it can fly away.
- **The exit stands on one added-mass bookkeeping.** With `entrainment =
  "wetting"` (S1's corrected mode, the lead's primary) the strike throws the
  body clear; with the solver's default (every growth of a strip's
  projected added mass reacted as `-dm/dt v`, which S1 found to be a
  one-sided drag on a flapping wing) the same strike lifts the CoM at most
  0.33-0.46 m on 6 of 6 seeds and the wings never leave the water. S1 could
  not say which is right; only a flapping plate in a tank, with force
  measured through the beat, can.
- The solver's force limiter (60 weights per body) engaged in every powered
  exit row, and timed on XV seeds 0 and 3 (added check) only after the
  flight, when the body was back in the water with the flying stroke still
  beating: not during the strike or the climb.

## Figures (from the built model, opened and checked)

`figures/M2_XV_s3.png`, `M2_XDV_s0X.png`, `M2_XB_s0X.png`,
`M2_XD_s0X.png`, `M2_WS_s0A.png` and `M2_WS_s0B.png` (one scored swim
pair), `M2_WV_s0A.png`, `M2_VB_s0A.png` (half A only): side
(x, z) and top (x, y) views at several instants, every geom of the compiled
model, each strip's centre coloured wet or dry, the surface line. The
model was **posed from the stored 10 Hz trace** (hull attitude from its
nose and dorsal axes, joint angles, CoM; `m2_draw.py trace`), not re-run
(the script's re-run mode, `capture`, was not used);
the wet count of the posed model equals the row's recorded count at every
drawn instant (printed beside each). Checked against section 1:

- Hull 0.5 m long from 0.23 m behind the CoM, wings from 0.04 to 0.70 m
  off the axis with chord 0.26 m, tail 0.56 m behind the CoM, as built.
- XV seed 3: at 0.3 s both wings at -41 deg, pressed 0.15 m under the
  surface, CoM +0.25 m, 8 strips wet; at 0.6 s none wet, CoM +0.65 m, nose
  20 deg; 1.5-4.0 s nose 68-81 deg, wings sweeping, 2.2-3.4 m; 8.0 s nose
  47 deg and 4.9 m (losing attitude); 11.0 s back in the water, 20 wet.
- XDV seed 0: surfaces nose-up 73 deg at 11.4 s with the wings in the
  surface, thrown clear by 12.2 s, flies 13 m along -x and to 6.1 m, back
  in the water at 19.5 s.
- XB seed 0: wings stroked up and partly dry at 0.5 s, the stroke then
  beats into the water and the body is pitched nose-down 31-52 deg: the
  wings beat the water, as L0 row 5 warned.
- WS seed 0: the paddle with the wings 0.25-0.4 m under the surface and the
  body pitched 17-39 deg nose-up at the waterline; WV and VB seed 0: the
  body near level to 59 deg nose-up, all 47 strips wet, rising (WV) or near
  3 m (VB).
- WS seed 0 half B (told the heading behind it): the steering law holds the
  +y wing near the surface (amplitude share near 0) while the -y wing
  paddles 0.4-0.6 m under; the body turns little (the half's progress is
  -0.023 m/s).
- In the top views at t = 0 the two wings are not mirror images about the
  drawing's x axis: the whole machine is yawed by the seed's attitude
  scatter (the tail is off the axis by the same angle). Not a fault.

## Disclosed: changed or added after data

1. **The arms XV and XDV** (the vault, from afloat and on surfacing) were
   added after tuning trials T19 and T20 showed a down-stroke throw the body
   clear and T33 showed it fly. They are judged by the frozen GX unchanged
   and reported apart from the frozen verdict. Their nulls are the frozen
   X0f, X0d, X1f, X1d.
2. **The flying law is M1's `Fly`, copied** into `m2_fly.py` from
   `m1_takeoff.py` (sha256 9ef9178465606daa, uncommitted, 2026-11-18) and
   `m1_lib.py` (37f3452a286f64d5) so that it could not change under the
   frozen rows. It includes M1's own post-hoc gains (`PLUS`, kpP 1.0, kdP
   0.3). **My changes to it:** pitch damping kdy 0.25 (M1 0.10), strike
   -1.0 rad (M1's vault -1.35), and the switch to flight when no main-wing
   strip is wet (M1: when the stroke passes -1.2 rad or after 0.25 s), tuned
   on tuning seed 102 (T38-T45) and checked on seeds 100, 101, 103 (T46-T48),
   all four meeting GX before freezing.
3. **The force-limiter count and slam maximum** in every row's summary
   (`clamped_steps`, `slam_max_N`), added before the scored rows.
4. **C8 at dt 5e-5 on XV seed 0** (the strike is the fastest event in the
   pass): met, clear 4.2 s, 3.48 m, the same as at 1e-4.
5. **C9 on XV seeds 1-5 and the limiter's timing** (`m2_extra.py`), after
   reading C9 on seed 0: default entrainment 0 of 6; limiter only after the
   flights.
6. The rows of the arms that cannot read the command (W0, S0, VB-W0) and
   the frozen arms (WF, SF, whose half-B command is half A's) were run once
   and stand for both halves; W0 seed 0 was run for both and the two traces
   are identical.
7. Before any scored row, a review found the runner would start WS under
   water with a depth command; fixed before launch (no row was run that
   way).
8. The best swim arm for C8 and C9 was named WS seed 0 before the rows,
   because none had a best by any gate.
9. Tuning: 63 rollouts run, 61 recorded, about 1080 simulated s, of the
   150 / 4000 cap. Two rollouts (the first T31, T32) were lost to a fault in
   the exit law, not recorded, and re-run; T10's per-wing feather offsets were ignored by the law as then
   written (it ran as a plain scull). The tuning printer's `roll_end` is a
   constant 0 (a typo), read nowhere.
10. X1d's command: the frozen text gives X1 a heading (-x); for the
    depth start I added the depth to hold (3 m, its start), since a swim
    arm under water needs one. X1f has the heading only.
11. Wall cost: 20-35 s per simulated second per process under the load of
    M1, M3 and M4 beside this pass; the scored set took about 10 h on 4
    processes.

## What this does not establish

- That the machine can fly away from the water: every flight here is lost
  within 10 s, by a law not built for this start.
- Anything the bench has not checked: the strike, the added mass it pushes
  against, the surface, the motors' 4 kW peaks (the pack is not modelled).
- A swim with a different body: the fold (L1c, L1d), a smaller hull, or a
  ballast that cuts the buoyancy; VB shows that cutting buoyancy alone does
  not make the wings swim.
- Modelled sensors (M5).
