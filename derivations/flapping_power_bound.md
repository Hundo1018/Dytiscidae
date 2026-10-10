# Flapping power bound `R(f) = P_avail(f) / P_req`

ROADMAP N7, §"2026-10-10 — why the search does not progress", theory T4. Every
number below is printed by

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python derivations/flapping_power_bound.py

(about 2 s, no GPU, no rollout). The script also checks that each source line
this document cites still says what it is cited for (39 lines), and fails if
one has moved.

## Definition

For a flapping machine of weight `W = m g`, lifting area `S` and span `b`:

* **`P_req`** is the smallest mechanical power that level flight can cost,
  over all trim speeds `U`: induced plus profile power, as ROADMAP N7 words it,
  with parasite (body) power added as a second figure.
* **`P_avail(f)`** is the largest mean mechanical power the machine's stroke
  motors can put into the air when they flap at frequency `f`. It is limited by
  each motor's torque limit, the stroke amplitude the joint allows, the inertia
  each motor has to swing back and forth, and the joint's damping.
* **`R(f) = P_avail(f) / P_req`**, evaluated over the band of frequencies the
  genome can draw and the Tier-0 spar check accepts.

`R < 1` means no gait at that frequency can fly, whatever the controller does,
because the motors cannot deliver the power. `R >= 1` does *not* mean the
machine can fly. It means power is not what stops it. The bound is a necessary
condition only.

The **binding frequency** `f_bind` of a joint is the frequency above which the
torque needed just to accelerate and decelerate a full-amplitude stroke exceeds
the motor's limit.

## Assumptions

1. **Steady level flight, actuator-disc induced power.** The wake is a disc of
   diameter `b`. Induced power is `k W v_i` with Glauert's forward-flight
   inflow, and `k = 1/e` uses the solver's own Oswald factor `e = 0.75`
   (`physics/fluid.py:345-346`). Pennycuick's usual `k = 1.2` is not used,
   because the project's drag polar implies `1/e = 1.33`.
2. **Profile power is the drag of a gliding wing at `U`.** The two-sided
   flat-plate skin friction `skin_friction_cd(Re)` (`fluid.py:242`), at
   `Re = U c_bar / nu` with `c_bar = S/b`. The stroke's own speed is ignored
   (Failure conditions, item 3).
3. **Parasite power** uses the hull's bluff `C_D = 0.20`
   (`core/phenotype.py:1128`) on the phenotype's frontal area
   (`phenotype.py:774`).
4. **The stroke is sinusoidal**, `theta = A sin(wt)`, as the CPG commands it
   (`control/cpg.py:171`). The amplitude is at most half the joint's travel:
   `stroke_amplitude` is clipped to [0, 1] of half-travel
   (`core/genome.py:563`), and `CPGParams.clipped` caps the amplitude at the
   half-span (`cpg.py:89`).
5. **The motor's only limit is its torque.** The MuJoCo position servo has
   `forcerange = +/- stall_torque` (`core/mjcf.py:717`). Stall torque is
   `km sqrt(p_cont) G eta_G` (`physics/energy.py:61`), with
   `p_cont = specific_power * motor_mass` (`energy.py:46`),
   `km = 0.05 (m/0.1)^0.75 eta_peak/0.88` (`energy.py:53`) and
   `eta_G = 0.97^(log_5 G)` (`energy.py:49`). Nothing in the dynamics enforces
   the rated speed `p_cont / stall_torque` (`energy.py:66`). That is the model as
   built, and a variant below enforces the rated speed anyway.
6. **The load is the most favourable a motor could face.** The air takes any
   power that reaches it, entirely in phase with the joint's velocity. It adds
   no inertia (added mass) and no stiffness.
7. **Inertia is MuJoCo's own.** `I` is the diagonal of the joint-space mass
   matrix at the build pose, with every other joint locked and the hull held
   (as on the 09-26 rig). Joint damping `d = 0.02 + 0.5 m_seg`
   (`mjcf.py:567`) is a loss.
8. **Which joints count.** Only stroke motors (`_a` actuators) whose subtree
   carries a surface that `S` counts (WING, MEMBRANE, FIN:
   `phenotype.py:564`). Feathering motors (`_f`), rotors, legs and paddles are
   excluded. On the bound's own terms one feathering motor of the feathering
   teal (60.8 N m on 0.006 kg m^2) could deliver 1705 W at 7.80 Hz and 2200 W at
   11.33 Hz. That says more about the bound than about pitching, so they are
   left out.

## Derivation

### Power required

Momentum theory for a disc of area `S_d = pi b^2 / 4` moving at `U`, with thrust
equal to the weight. Glauert's inflow satisfies
`v_i^2 (U^2 + v_i^2) = T^2`, where `T = W / (2 rho S_d)`, so

    v_i^2 = ( -U^2 + sqrt(U^4 + 4 T^2) ) / 2,        P_ind = W v_i / e.

At `U >> v_i` this gives `v_i -> T/U` and `P_ind -> W^2 / (2 e rho S_d U)`. That
is exactly the power of the solver's drag polar, `D_i U` with
`C_Di = C_L^2 / (pi e AR)` and `AR = b^2/S` (`phenotype.py:312`):

    D_i U = (1/2 rho U^2 S) (W / (1/2 rho U^2 S))^2 / (pi e b^2 / S) U
          = 2 W^2 / (pi e rho U b^2) = W^2 / (2 e rho S_d U).        ok

The other two terms are

    P_pro = 1/2 rho U^3 S C_Df(U c_bar / nu),     P_par = 1/2 rho U^3 C_D,hull A_f,

and `P_req = min_U (P_ind + P_pro)`. The figure with parasite is
`min_U (P_ind + P_pro + P_par)`, taken at its own speed. Momentum theory needs
no lift-coefficient cap, because a flapper can exceed a fixed wing's `C_L`. The
script also reports the minimum under `C_L <= CN_LEV/2 = 1.7`, the largest
`C_L` the model has (`fluid.py:274`). The induced + profile minima already sit
at `C_L <= 0.77`. The cap moves the with-parasite minimum by at most 4.4%
(teal 52.2 -> 54.5 W).

### Power available, one joint

The joint obeys `I theta'' + d theta' + tau_L = tau_m` with `|tau_m| <= tau`.
Averaged over a period, the inertial term does no work, so the power that
reaches the load is

    P = < tau_m theta' > - d < theta'^2 >.

`theta'` is a pure sinusoid, so only the fundamental of `tau_m` does work
against it. Write that fundamental as `a1 cos(wt) + b1 sin(wt)`. The in-phase
part does work, `< tau_m theta' > = a1 w A / 2`, and the quadrature part must
supply the inertia, `b1 = -I w^2 A` (Assumption 6). The fundamental of any
function bounded by `tau` has amplitude at most `(4/pi) tau` (a square wave),
so `a1^2 + b1^2 <= (4 tau / pi)^2`, and

    P_j(f, A) = (w A / 2) [ sqrt( (4 tau/pi)^2 - (I w^2 A)^2 ) - d w A ],   0 < A <= A_max.

`P_avail(f) = sum_j max_A P_j(f, A)`.

Without damping the interior optimum is at `I w^2 A* = (4 tau/pi)/sqrt 2`, which
gives

    P*_j = (4 tau / pi)^2 / (4 I w) = 4 tau^2 / (pi^2 I w),

falling as `1/f`. Below the frequency at which `A*` reaches `A_max`, `P_j`
grows with `f`. So `R(f)` rises to a maximum near `f_bind` and falls after it.
At the top of the band the minimum is therefore `R(f_hi)`.

### Where torque binds

The full stroke's inertial torque `I w^2 A_max` equals the limit at

    f_bind = (1 / 2 pi) sqrt( tau / (I A_max) ).

Above it, no sinusoidal torque can produce the commanded stroke. With a square
wave the cut-off moves up by `sqrt(4/pi) = 1.13`.

### The allowed band

`flapping_inertial_check` (`physics/structure.py:179-182`) loads the spar with
`I_root A_half (2 pi f)^2`, at the genome's own frequency and with half the
joint travel (`phenotype.py:899-900`). The stress goes as `f^2`, so the
frequency at which the worst wing reaches its allowable stress is

    f_spar = f_genome sqrt( sigma_allow / sigma(f_genome) ).

The band is `[1.5, min(12, f_spar)]`, where 1.5-12 Hz is the draw in both
`random_genome` (`genome.py:384`) and `mut_gait` (`genome.py:620`).

## Dimensional analysis

    [P_j]  = (1/s)(rad)(N m)                          = W                    ok
    [P*]   = (N m)^2 / (kg m^2 * 1/s)  = kg m^2 s^-3  = W                    ok
    [T]    = N / (kg m^-3 * m^2)       = m^2 s^-2,  so  [v_i] = m/s          ok
    [P_ind]= N * m/s                                  = W                    ok
    [f_bind] = sqrt( N m / (kg m^2 * rad) )           = 1/s                  ok

## Numerical implementation

`derivations/flapping_power_bound.py` builds each plan (`phenotype.build`) and
compiles it (`mjcf.compile_phenotype`). It runs one `mj_forward` to read the mass
matrix. Nothing is stepped. Where the code differs from the derivation:

* **The servo is not the optimal controller the bound assumes.** It is a PD
  law. `kp = 2 tau g` and `kv = 0.15 tau g` (`mjcf.py:715-716`, with `g` =
  `drive_compliance`), and the reference rate is fed forward
  (`ctrl = q_ref + (kv/kp) q_ref'`, `envs/triphibian.py:659, 1746`). There is
  no acceleration feed-forward. For the feathering teal's stroke motor, the
  closed-loop corner `sqrt(kp/I)/2 pi` is **2.88 Hz**. At 11.33 Hz the
  unsaturated response is |q/q_ref| **0.351**, lagging **80 deg**. The servo
  delivers less than the bound at every frequency above its corner.
* **The phenotype's joint inertia, which tunes the series spring
  (`mjcf.py:585`), is not the inertia MuJoCo integrates.** The phenotype puts
  each segment's mass at its centroid (`phenotype.py:766`). On the plain
  teal/gannet stroke joint that gives 0.180 kg m^2 against MuJoCo's 0.242. On
  the feathering teal it is 0.180 against 0.372, because the feathering motor
  rides on the wing and the phenotype leaves its mass out. A
  "`series_stiffness = 1`" spring on the feathering teal therefore cancels
  48% of the real inertia, not 100%. Tail joints read 0 in the phenotype and
  0.0005 in MuJoCo.
* **Gear ratio is a free torque multiplier.** `mut_actuator` moves it over
  1-200 (`genome.py:475`) at no mass, with only 3% loss per stage. The
  forcerange is clipped at 400 N m (`mjcf.py:686`), and no speed limit applies.

Inputs read from the built machines:

| plan | m (kg) | S (m^2) | b (m) | stroke motor tau (N m) | I (kg m^2) | A_max (rad) | f_bind (Hz) | spar f_max (Hz) |
|---|---|---|---|---|---|---|---|---|
| gannet | 5.309 | 0.514 | 2.400 | 60.8 (x2), tail 6.0 (x2) | 0.242 | 0.875 | 2.70 | 7.80 |
| teal | 5.345 | 0.514 | 2.400 | 60.8 (x2), tail 6.0 (x2) | 0.242 | 0.875 | 2.70 | 7.80 |
| beetle | 5.129 | 0.261 | 1.816 | 24.3 (x2) | 0.133 | 0.620 | 2.73 | 12.81 |
| bat | 3.960 | 0.411 | 1.649 | 11.2 / 8.2 / 6.0 (x2 each) | 0.136 / 0.029 / 0.003 | 0.725 | 1.70 / 3.16 / 8.10 | 70.07 |
| teal, feathering | 6.327 | 0.514 | 2.400 | 60.8 (x2), tail 6.0 (x2) | 0.372 | 0.875 | 2.17 | 7.80 |

`P_req` (W), minimum over `U`:

| plan | induced + profile | at U (m/s), C_L | + parasite | at U (m/s), C_L |
|---|---|---|---|---|
| gannet | 29.2 | 15.1, 0.73 | 41.3 | 10.6, 1.48 |
| teal | 29.5 | 15.1, 0.73 | 52.2 | 8.4, 2.34 |
| beetle | 35.7 | 20.2, 0.77 | 60.1 | 11.8, 2.26 |
| bat | 31.1 | 16.5, 0.57 | 40.4 | 12.7, 0.96 |
| teal, feathering | 37.9 | 16.4, 0.73 | 67.2 | 9.2, 2.34 |

## Validation

* **Cited constants**: the script asserts all 39 source lines (`sources: 39
  cited lines checked`).
* **Spar band, against the record.** The gannet's band top comes out at
  **7.80 Hz**. ROADMAP §"Why nothing flies, measured a third time" §5 recorded
  the check at 1.99x allowable at 10.95 Hz, which puts it at **7.76 Hz**: 0.5%
  apart.
* **Torque-limited, against the record.** At the 09-26 teal level gait (11.33 Hz,
  amplitudes from `runs/_logs/level_best_teal_feathering.pkl`), the left stroke
  joint needs **22.1x** its torque limit just for inertia, and the right needs
  **3.0x**. The best reach either joint can have is **0.058** and **0.422** of
  its commanded stroke. The rig recorded the motors torque-limited 60% of the
  time and "several joints" reaching 0.19-0.39 (`runs/_logs/plan_0923b.md:36`).
  Saturation is what the bound predicts. The rig did not record reach per
  joint, so the two reach figures cannot be checked one to one. The servo's
  unsaturated lag of 80 deg falls inside the recorded 30-141 deg.
* **The 11.33 Hz gait is outside the allowed band.** For the teal airframe the
  spar check closes the band at 7.80 Hz. The genome passed Tier 0 because the
  check runs at the genome's `flap_frequency` (5 Hz), and the rig commanded
  11.33 Hz through the CPG.
* **The 0.42 margin is not reproduced** (last section). Post hoc, and not the
  prediction's metric: the same wing held still at the rig's 6.83 m/s, at
  `C_L = 1.7`, lifts **0.402** of the phenotype weight. The 0.42 free-joint
  margin is about what the wings give as a fixed wing at that speed. This was
  computed after seeing the result and is one number.

## Failure conditions

Each omission, and which way it moves `R`:

1. **Unsteady effects** (Wagner lag, LEV history, added mass). Added mass is a
   quadrature load: it raises the torque each stroke needs and lowers
   `P_avail`. Wagner lag lowers the lift a stroke makes. Neither can lower the
   momentum-theory minimum `P_req`. **`R` is too high.**
2. **Feathering phase and propulsive efficiency.** The bound counts every watt
   at the joint as useful flight work. A real stroke turns a fraction `eta` of
   its power into thrust times speed, and the feathering phase sets `eta`. The
   true ratio is `eta R`. At each plan's best frequency, `R` falls below 1 only
   if `eta` is below **0.036** (gannet, teal), **0.057** (feathering teal),
   **0.131** (bat) or **0.160** (beetle). With parasite power the thresholds
   are 0.050 / 0.064 / 0.101 / 0.170 / 0.269. **`R` is too high, by `1/eta`.**
3. **Profile power at `U`, not at the stroke's speed.** At the 09-26 gait the
   commanded tip stroke speed of the 1.2 m wing is 60.9 m/s, against
   `U = 6.8 m/s`. **`R` is too high**, increasingly with
   `f`.
4. **The `(4/pi) tau` fundamental.** This assumes a square-wave torque. With a
   sinusoidal torque, the band maximum falls by 31% and the band-top value
   by 32-38% ("sine tau" columns below). **`R` is too high** for a servo that
   does not saturate.
5. **A load that takes any power.** A cap from the model's largest
   normal-force coefficient (`CN_LEV`, every strip at its largest speed) lowers
   the band maximum by at most 4.2%. It lowers the band-top value by up to 36%,
   mostly by removing the tail motors' share ("load-capped"). The cap is
   loose. **`R` is too high.**
6. **Inertia with the hull held.** With the root free, the effective inertia
   is lower. **`R` is too low.**
7. **Genes the headline holds at the seed's values.** Each of these raises
   `R`, so **the headline is too low** for the reachable genome:
   - a tuned series spring (`mut_drivetrain`, `genome.py:441`), which cancels
     the inertial cap;
   - gear ratio up to 200;
   - joint travel up to +/-2.6 rad (`genome.py:461`).
8. **Rated speed not enforced.** If it were, the band maxima would fall to
   6.26-15.06 (teal 27.84 -> 14.91) and the band minima to 1.33-11.70.

So the large errors (1-3) all push `R` up. `R < 1` would have been decisive.
`R > 1` is decisive only for "power is the wall", and says nothing about
whether flight is possible.

## N7 against the prediction

### Numbers (ROADMAP N7's table: ratio across the band, per plan)

`R(f)` with `P_req` = induced + profile, over the band `[1.5, min(12, f_spar)]`:

| plan | band (Hz) | f_bind (Hz) | R at 1.5 Hz | R max (at f) | R at band top | R min | + parasite: max / min |
|---|---|---|---|---|---|---|---|
| gannet | 1.5-7.80 | 2.70 | 21.77 | **28.13** (2.3) | 12.68 | 12.68 | 19.88 / 8.96 |
| teal | 1.5-7.80 | 2.70 | 21.55 | **27.84** (2.3) | 12.55 | 12.55 | 15.72 / 7.09 |
| beetle | 1.5-12.00 | 2.73 | 4.84 | **6.26** (2.3) | 1.33 | 1.33 | 3.72 / 0.79 |
| bat | 1.5-12.00 | 1.70 | 6.26 | **7.66** (6.0) | 4.63 | 4.63 | 5.89 / 3.56 |
| teal, feathering | 1.5-7.80 | 2.17 | 16.05 | **17.55** (1.9) | 7.47 | 7.47 | 9.89 / 4.21 |

Robustness over the same band, `R` max / min. "main + load-capped, + parasite"
divides the main-wing, load-capped `P_avail` by the with-parasite `P_req`:

| plan | main wings only | load-capped | main + load-capped | main + load-capped, + parasite | rated speed enforced | sine torque |
|---|---|---|---|---|---|---|
| gannet | 26.83 / 8.57 | 26.96 / 9.06 | 26.83 / 8.57 | 18.96 / 6.06 | 15.06 / 11.70 | 19.51 / 8.42 |
| teal | 26.56 / 8.49 | 26.69 / 8.97 | 26.56 / 8.49 | 14.99 / 4.79 | 14.91 / 11.58 | 19.31 / 8.34 |
| beetle | 6.26 / 1.33 | 6.26 / 1.33 | 6.26 / 1.33 | 3.72 / 0.79 | 6.26 / 1.33 | 4.34 / 0.82 |
| bat | 4.71 / 1.11 | 7.66 / 4.63 | 4.71 / 1.11 | 3.62 / 0.86 | 7.54 / 4.58 | 5.27 / 2.85 |
| teal, feathering | 16.74 / 4.31 | 16.84 / 4.75 | 16.74 / 4.31 | 9.44 / 2.43 | 11.20 / 6.71 | 12.18 / 5.07 |

At the band top, `R` under each variant:

| plan | headline | sine tau | spring (as mjcf tunes it) | spring (ideal) | gear 200 | rated power cap |
|---|---|---|---|---|---|---|
| gannet | 12.68 | 8.42 | 36.39 | 109.52 | 237.43 | 12.68 |
| teal | 12.55 | 8.34 | 36.02 | 108.42 | 235.04 | 12.55 |
| beetle | 1.33 | 0.82 | 5.02 | 35.29 | 330.90 | 1.33 |
| bat | 4.63 | 2.85 | 21.77 | 46.98 | 384.54 | 4.63 |
| teal, feathering | 7.47 | 5.07 | 11.42 | 84.38 | 128.84 | 7.47 |

The 09-26 teal level gait (feathering teal, 11.33 Hz, its own amplitudes):

* `P_avail` is 266.3 W, or 112.6 W from the main wings alone, load-capped.
* `R` is **7.03** against the minimum `P_req` (37.9 W), **3.90** against
  `P_req` at the rig's 6.83 m/s (68.3 W), and **2.97** for the main wings,
  load-capped.
* As force-equivalents `R^(2/3)`: 3.67 and 2.48.

spread: n/a. The computation is deterministic, with no rng and no sample. The
robustness and sensitivity tables above are the range across the bound's own
modelling choices. The smallest band maximum in any variant is 3.62 (bat,
main + load-capped + parasite).

### Against the frozen prediction

prediction: "ratio < 1 for every flapper plan at every frequency the spar
allows; the 0.42 margin of 2026-09-26 is reproduced within 30%. *Falsified* if
any plan's ratio exceeds 1 inside the allowed frequency band."

outcome: **REFUTED**

by:

* Every plan's ratio exceeds 1 inside its band: band maxima gannet 28.13,
  teal 27.84, beetle 6.26, bat 7.66, feathering teal 17.55.
* It stays above 1 in every robustness variant. The smallest maximum is 3.62.
* The second clause fails too. At the 09-26 gait the ratio is 2.97-7.03
  against 0.42. That comparison sets a power ratio against a force margin. The
  prediction text does not convert between the two, and no conversion tried
  here comes within 30%.

What the refutation leaves standing:

* The power bound does not show flapping flight infeasible under the actuator
  limits. At the frequencies the spar allows, the stroke motors have 6-28x the
  minimum power of level flight.
* At the frequency of the largest ratio, the gannet/teal stroke runs at
  Strouhal 0.28, inside the efficient band.
* The 09-26 failure happened at 11.33 Hz. That frequency is outside the teal's
  spar band (7.80 Hz) and four times the servo's corner (2.88 Hz). There the
  stroke needs 22x its torque limit.
* T4's mechanism therefore has to be something other than power:
  - the model's conversion efficiency at in-band frequencies, which must fall
    below the 3.6-16% break-even (item 2 above) for power to matter;
  - or the PD servo's bandwidth;
  - or the fact that the quasi-steady model makes its thrust only above the
    spar's band (ROADMAP §5: every best gait at 10.8-11.8 Hz).

## N13 — the Strouhal gap

ROADMAP N13, §"2026-10-10 — why the search does not progress". Reproduced by

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python derivations/flapping_power_bound.py --strouhal

(about 30 s, CPU only, no rollout; it writes `derivations/flapping_power_bound_strouhal.json`,
reads `runs/arch49/archive_*.pkl` and `experiments/spar_band_read/results_arch49.json`, and
checks 11 more cited source lines first). Running the script without the flag prints
exactly what it printed before this section existed.

### What the N7 script's `St` was

The `St` in the N7 output (gannet and teal 0.28, beetle 0.12, bat 0.03, feathering teal 0.21)
is `f 2 r sin(A) / U` with

* `f` the frequency of the largest power ratio (2.3 Hz for gannet and teal, 6.0 Hz bat),
  **not the band top**;
* `A` the power-optimal amplitude at that frequency (`joint_power`'s argmax). It equals
  the half-travel 0.875 rad for gannet and teal, but is 0.053 rad for the bat;
* `r` the largest strip-centre distance from the stroke axis (1.20 m), not the tip;
* `U` the minimum-power speed `U2` of the N7 table (15.1 m/s), not a measured trim speed.

It answers "what is the Strouhal number where the motors are strongest". It is not
N13's quantity.

### Definition

`St = 2 f A_tip / U`, with

* `f = min(12 Hz, f_spar)`, the band top of the table above (`genome.py:620`; the spar
  scaling is the derivation's "allowed band"): 7.80 Hz gannet, teal and feathering teal;
  12 Hz beetle and bat.
* `A_tip = R_tip sin(A)`, the half excursion of the wing tip normal to the stroke plane.
  `R_tip` is the outer edge of the outermost lifting strip of the main stroke joint
  (`panels.pos_local` + `dr/2`: 1.233 m for the 1.2 m semi-span wing). A variant uses the
  arc `R_tip A`.
* `A = Part.stroke_amplitude x half-travel` at the gene's maximum. The gene is clipped to
  [0, 1] (`genome.py:563`) and the CPG commands `clip(gene, 0, 1) * half` (`triphibian.py:719-722`,
  `cpg.py:89`), so the maximum is the joint's own half-travel (0.875 rad gannet, teal;
  0.620 beetle; 0.725 bat). The gannet's wing carries gene 0.0 (`bodyplans.py:499-502`, "a trim surface,
  not an oscillator") and the teal inherits it (`bodyplans.py:606`, "Wing, tail and fuselage carry over from the
  gannet"; its 5 Hz drives the legs); N13 asks for the maximum, so the gene is set to 1.0 and the "own
  gene" column shows what the plan commands.
* `U` is the measured trim speed: `TriphibianEnv._trim()` (`triphibian.py:1181-1186`,
  `_measure_trim_speed`): 1.2 x the stall speed found by bisection on the solver's own
  lift, clipped to 6-30 m/s (`triphibian.py:243, 1316`). A machine that cannot lift its
  weight at 30 m/s gets 6.0 exactly.

### Numbers

| plan | f top (Hz) | A (rad) | R_tip (m) | A_tip (m) | U trim (m/s) | lift margin | **St at band top** | arc | U = min-power | f = 12 Hz | plan's own gene |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gannet | 7.80 | 0.875 | 1.233 | 0.947 | 15.27 | 5.58 | **0.967** | 1.103 | 0.982 | 1.488 | 0.000 |
| teal | 7.80 | 0.875 | 1.233 | 0.947 | 15.41 | 5.47 | **0.959** | 1.093 | 0.978 | 1.474 | 0.000 |
| beetle | 12.00 | 0.620 | 0.946 | 0.549 | 20.19 | 3.18 | **0.653** | 0.697 | 0.654 | 0.653 | 0.309 |
| bat | 12.00 | 0.725 | 0.819 | 0.543 | 16.93 | 4.52 | **0.770** | 0.842 | 0.790 | 0.770 | 0.372 |
| teal, feathering | 7.80 | 0.875 | 1.233 | 0.947 | 16.76 | 4.62 | **0.881** | 1.005 | 0.900 | 1.355 | 0.000 |

What St 0.25 would take, the other two held: gannet and teal need `f = 2.0 Hz`, or
`U = 59 m/s`, or a tip excursion of 0.245 m (a quarter of the maximum). The band starts
at 1.5 Hz, so for these two wings the band spans St 0.19 to 0.97. With the joint range at
the genome's largest, 2.6 rad (`genome.py:460-461`), St at the band top is 0.64-0.65.

arch49's flapping elites: the 157 of the 229 merged-archive elites with a lift-carrying
stroke joint (the derivation's `machine` definition) whose stroke gene is above 0. Each
at its own joint travel (gene 1.0), measured trim speed and band top
`min(12, as-built spar f_max)` (92 of the 157 have no spar check at all, so 12 Hz):

| subset | n | median St | IQR | < 0.15 | 0.15-0.25 | >= 0.25 |
|---|---|---|---|---|---|---|
| flapping elites | 157 | 0.500 | 0.173-0.906 | 35 | 23 | 99 |
| ... n_rotors == 0 | 111 | 0.441 | 0.182-0.646 | 21 | 19 | 71 |
| ... a real trim (lift margin >= 1) | 80 | 0.422 | 0.095-0.585 | 23 | 6 | 51 |
| ... real trim and n_rotors == 0 | 58 | 0.279 | 0.050-0.531 | 18 | 5 | 35 |

The median trim speed of the 157 is 6.0 m/s, the floor of the launch band that a
machine which cannot lift its weight is given, so the first row is inflated by small
`U`; the "real trim" rows exclude those. At the elite's own gene instead of 1.0 the first row has
median 0.229 (68 / 15 / 74). Forcing the spar check onto designs that skip it changes
the first-row median from 0.500 to 0.483.

### Against the frozen prediction

prediction: "gannet and teal sit below 0.15 at 7.80 Hz, so no in-band gait can reach the thrust
regime at their trim speeds, and the reachable levers are stroke amplitude (joint range),
trim speed (wing loading) and the band (spar), not frequency. *Falsified* if either plan's
`St` at the band top is >= 0.25."

outcome: **FALSIFIED**. Gannet 0.967 and teal 0.959 at 7.80 Hz, four times the threshold, and
>= 0.25 in every variant above except the plan's own gene (0 for both: their wings are held;
smallest of the others for any plan at the band top is 0.65; 0.25 itself is crossed at 2.0 Hz). In-band, these wings are not too slow to reach the thrust regime; at the
top of the band they stroke at roughly four times the frequency the thrust regime needs, so
the Strouhal argument gives no support for "the band is too low". The opposite problem
shows: St passes 0.4 at 3.2 Hz (gannet), so most of the allowed band is above the efficient range,
and the 2.3 Hz figure the N7 script printed (0.28) was the one point of the band
inside it. This says nothing about whether the model's wing makes thrust at St ~ 0.3; it
says only that the premise "frequency is out of reach of the thrust regime" is false for the seed
wings.
