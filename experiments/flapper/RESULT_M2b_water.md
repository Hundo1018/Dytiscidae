# Pass M2b, result: stay up after the exit, swim by flapping, turn on the surface

2026-11-19. Pre-registration `PREREG_M2b_water.md` (commit 57d766e, frozen).
Rows `results/M2b_rows.jsonl` (168 scored rows plus 3 hash rows, 6 seeds
10-15 x 2 entrainment modes), read by `m2b_read.py` (`results/M2b_read.txt`)
and `m2b_extra.py` (`results/M2b_extra.txt`). Harness `mission.py` unedited.

## Outcome

**No gate is met by any arm, in either entrainment mode.** Under the
co-primary rule every verdict below is "not met in both modes". No verdict is
"not established" anywhere, but in one place the modes disagree on the
behaviour behind the verdict: the exit stays up only under "wetting". The
verdicts rest on a model that has no dynamic stall (X2) and no wave, splash
or free-surface deformation. Its surface is a linear density ramp per strip.

| gate | arm | wetting | default | verdict |
|---|---|---|---|---|
| GX30 (>= 4/6) | XS (vault, then V+ verbatim) | **1/6** | 0/6 | not met in both modes |
| GX30 | X0f (still) | 0/6 | 0/6 | not met in both modes |
| GW (>= 5/6) | WT (flow-angle tracking, as built) | 0/6 | 0/6 | not met in both modes |
| GW | WT-VB (the same, 1.08x buoyancy) | 0/6 | 0/6 | not met in both modes |
| GW | W0, VB-W0 (still) | 0/6, 0/6 | 0/6, 0/6 | not met in both modes |
| GW | WTF, WTF-VB (frozen) | 0/6, 0/6 | 0/6, 0/6 | not met in both modes |
| GS (>= 5/6) | ST (rowing turn) | 0/6 | 0/6 | not met in both modes |
| GS | S0 (still), SF (frozen) | 0/6, 0/6 | 0/6, 0/6 | not met in both modes |

M2's 3 s exit gate GX, read on the XS rows: 4/6 under "wetting" (seeds 10,
11, 14, 15) and 0/6 under the default. Under the default the exit never
leaves the water, so on the exit the two modes disagree on the behaviour.
They agree only on the 4-of-6 verdict. QACC resets: 0 of 168 rows.

What limited each arm:

- **XS.** Under "wetting", seed 11 met GX30. It was clear of the water
  39.6 s after t* 0.4 s and held 4.0 m at a mean of 351 W, with no motor on
  its limit (`figures/M2b_XS_s11_wetting.png`). It hovered nose-up (pitch
  85-89 deg), which is M1's V+ posture, and drifted up to 18 m in x. Seeds 10,
  14 and 15 climbed to 3.7-4.0 m and fell back in at 6.0, 8.2 and 4.9 s. Seed
  12 fell back at 2.7 s and seed 13 never cleared. Every fall had the stroke
  and feather motors on their limit for 76-98% of steps, the force limiter
  active (104-1564 steps) and a mean of 3.0-3.7 kW. That is the same
  lateral-divergence loss that the tuning seed 102 showed at about 2.7 s
  (`figures/M2b_XS_series.png`). Under the default, all six seeds peaked at
  0.34-0.46 m CoM height with 4-6 strips still wet, then settled back
  (`figures/M2b_XS_s11_default.png`). The run averaged about 3.8 kW with the
  motors on their limit 93-95% of the time.
- **WT, as built (1.85x buoyancy).** Under "wetting", 4 of 6 half-A rows
  (told deeper) ended at the surface. Pair progress was -0.22 to +0.11 m/s,
  and R exceeded 0.5 only on seed 12, which sank to 5.6 m and came back. Under
  the default, the same law kept the buoyant body at 2.3-3.6 m on every half
  without following either command (R -0.25 to +0.34, progress |P| <= 0.018
  m/s), while the still body W0 was at the surface by 40 s on 6/6
  (`figures/M2b_WT_series.png`).
- **WT-VB.** Under "wetting", both halves sank on most seeds (half A ended
  4.4-8.7 m deep against targets of 4.0-4.3 m, `results/M2b_extra.txt`). R >= 0.5 on 5/6 came from the
  two halves sinking at different rates, not from depth held: the hold clause
  failed on 6/6. Seed 14 was the only one with progress and R both passing,
  and it failed the hold (`figures/M2b_WT-VB_series.png`,
  `M2b_WT-VB_s14A_wetting.png`). Under the default, the body stayed at
  2.4-3.7 m with |P| <= 0.016 m/s (`M2b_WT-VB_s14A_default.png`). In both
  modes the mean electrical power was 1.4-2.1 kW.
- **ST.** The rowing law turned the body, but it also stood it on its nose
  or tail and rolled it inverted. The nose was steeper than 44 deg on 0-73%
  of samples, and the minimum dorsal z was -0.5 to -1.0 on 22 of 24 halves (default seed 12 stayed upright, +0.66, and never reached its heading).
  Under "wetting" it also dove to 0.5-1.9 m, which fails GS's surface clause
  (`figures/M2b_ST_series.png`, `M2b_ST_s13A_wetting.png`). Pair progress was
  -0.036 to +0.080 m/s under "wetting" and at most 0.027 m/s under the default.

## C0, the stroke-kp trace (before scoring)

Section 0 item 2 traced the mismatch. `Machine("air")` without `vx` keeps
`TriphibianEnv.reset(AIR)`'s 7.2 m/s launch speed. The PD gains are read from
the mass matrix with the wings' air added mass projected on that flow, which
gave M2 a stroke kp of 1613. Built at rest (`m2b_laws.build2`), kp is 1798,
matching M0. This is a harness property, reported and not edited.

The held take-off lift on `build2` (`results/M2b_checks.jsonl`):

| mode | lift, weights | vs 1.840 | within 0.5% |
|---|---|---|---|
| default (`True`) | 1.8404 | +0.02% | yes |
| "wetting" | 1.7951 | -2.44% | **no** |

With the gains matched, the remaining gap is the entrainment mode itself.
M0's 1.840 was a default-mode figure. By the pre-registered rule, the XS rows
under "wetting" are therefore labelled: their lift at take-off is 2.4% below
M0's. The one GX30 pass is a "wetting" row. C0 first ran on `m2b_laws.py` sha
1990304d3e44c026, before the later Track/Row edits. After scoring, it was
re-run in both modes on the frozen sha 5cee216e29c5ecaf that the scored rows
used (the last two lines of `M2b_checks.jsonl`). It reproduced 1.7950904654628161
and 1.8404495277165682 exactly, with the same kp.

## Predictions (PREREG section 4)

| | prediction | measured | holds |
|---|---|---|---|
| C0 | lift within 0.5% of 1.840 | default 1.8404; "wetting" 1.7951 (-2.4%) | default yes, "wetting" no (rows labelled) |
| C1 | nulls 0/6 in both modes | W0, VB-W0, S0, X0f, WTF, WTF-VB, SF: 0/6 in both | yes |
| C1 | W0 at the surface by 40 s on 6/6 | 6/6 both modes (depth 0.00-0.03 m) | yes |
| C1 | P = R = 0 exactly for still and frozen | 0.000 to three decimals on every row; frozen halves are identical, so P and R are 0 by construction | yes |
| C2 | XS GX30 0-2/6 under "wetting", 0 under default; not met (0.6) | 1/6, 0/6; not met | yes |
| C2 | XSt GX30 2-5/6 under "wetting" | XSt = XS (no tuned setting was better), so 1/6 | no |
| C2 | M2's 3 s GX on XS under "wetting" >= 4/6 | 4/6 | yes |
| C3 | WT GW 0/6 in both modes (0.7) | 0/6, 0/6 | yes |
| C3 | half A short of its target by > 0.5 m on >= 4/6 | "wetting" 5/6, default 6/6 | yes |
| C4 | WT-VB GW 1-3/6 under "wetting" | 0/6 | no |
| C4 | R >= 0.5 on >= 3/6 | "wetting" 5/6 (from the halves sinking at different rates, hold 0/6); default 0/6 | yes under "wetting", read with that caveat |
| C4 | mean electrical 300-1500 W | 1.4-1.8 kW "wetting", 1.4-2.1 kW default | no (above) |
| C5 | ST GS 0-2/6 in both modes | 0, 0 | yes |
| C5 | turn rate under 5 deg/s on every half | 3 of 22 measurable halves under 5; up to 30.6 deg/s | no (it turns faster, and tumbles) |
| C5 | a half within 45 deg of heading-or-reverse by 15 s on >= 3/6 | "wetting" 6/6, default 3/6 (SF the same, S0 1/6) | yes |
| C6 | the two modes' counts differ by >= 1 on some gate for every do-something arm | XS yes (GX30 1 vs 0, GX 4 vs 0); WT, WT-VB, ST no (0 vs 0 on every gate) | no |
| C7 | peak over 462 W on every WT, WT-VB, XS row | 2.5-4.3 kW on every one | yes |
| C7 | ST mean under 462 W | 5 of 24 halves | no |
| QACC | under 1 in 10 rows | 0/168 | yes |

On C6, the counts agree for WT, WT-VB and ST, but the behaviour differs
strongly between the modes. Under "wetting" the flapping as-built body rises
to the surface, and under the default it stays at 2.3-3.6 m. ST dives under
"wetting" and stays at the surface under the default. A gate failed in both
ways is not the same failure.

On C5's heading read, the turn toward heading-or-reverse is the same in both
halves until the law locks on. SF, which runs half A's command in both halves,
therefore scores the same as ST, and the read does not show that the command
was read. S0 reaches 1/6 because seed 10 starts 9 deg from its heading.

## Disclosures (anything set or added after data)

1. **Probes Q1-Q8** (seed 100, before freezing; section 0 item 4). Q1-Q4
   ran with the feather-gain fault (the gain read at rest was 0). This was
   fixed before tuning.
2. **Tuning:** 44 rollouts and 1166 simulated s (`results/M2b_trials.jsonl`),
   within the 120 / 3000 cap, on tuning seeds 100-103 only. The XSt search
   (strike depth, kdx, kpP/kdP, k_v/lean, kpz/kdz) never beat XS on the tuning
   seeds, so XSt = XS as the prereg rule says, and it was not run as a separate
   scored arm.
3. **Options added to the tracking law during tuning:** `dorsal` (depth error
   projected on the dorsal axis), frozen in WT and WT-VB; `course` (steer on
   the velocity heading), rejected and not frozen. The frozen specs are in the
   hash row: WT and WT-VB `sgn 1, gain 0.2, tailfirst, kz -15 deg, kh -1.5,
   dorsal`; ST `f 0.6 Hz, A 80 deg, b -40 deg, down 0.25, Psi_r 55 deg,
   fwd_sign -1, turn_sign -1`. In the probes and tuning, the depth lever's
   sign was found to be inconsistent on VB.
4. **GX30's rate clause in the reader** was first coded as the mean of |omega|,
   which the within-beat wobble fails on every row. It was changed during
   tuning, before any scored row, to the norm of the 0.5 s mean body-rate
   vector (M1's G4 reads beat means). The gate's 1.5 rad/s bar is unchanged.
5. **C5's turn read** (`m2b_extra.py`) was written after the ST rows existed.
   Because ST stands on its nose or tail, the nose heading is undefined there,
   so the read uses only samples with |nose z| <= 0.7. The 5 s rate is taken
   over windows that are valid throughout, and two halves have no such window.
   Without the mask the rate read 13-273 deg/s, which was unwrap noise from a
   vertical nose.
6. **Null and frozen arms** (W0, VB-W0, S0, X0f, WTF, WTF-VB, SF) ran once per
   seed and mode, and the reader mirrors the command for half B. A frozen
   arm's two halves are one rollout.
7. **Figures** are drawn by a new `m2b_draw.py` (reusing `m2_draw.figure` and
   `snap`) instead of `m2_draw.py` as the prereg says, because `m2_draw`
   reads `M2_rows` and M2's `build`. The poses come from the stored 10 Hz
   traces. In each `pose` figure the strip wet count matched the trace's.
8. The scored run was one parent process with 4 workers. Brief
   single-process reads and draws (no stepping), and after scoring the two
   C0 re-checks, ran beside it or after it. A leftover shell waiter from the
   earlier session only polled files.
9. The rows file has 171 lines: 168 scored rows, no duplicate key, and 3
   hash rows (one per `--only` invocation), all with the same shas.

## Figures (each opened and checked)

- `figures/M2b_XS_series.png`: CoM height and nose heading for all XS seeds,
  both modes. The heading is meaningless in the nose-up hover.
- `figures/M2b_XS_s11_wetting.png`: the seed that stays up, posed from 0.5 to
  39.5 s.
- `figures/M2b_XS_s11_default.png`: the same seed under the default, rising
  to 0.43 m and settling back.
- `figures/M2b_WT_series.png`, `figures/M2b_WT-VB_series.png`: depth and
  heading against time, both modes, both halves.
- `figures/M2b_WT-VB_s14A_wetting.png`, `figures/M2b_WT-VB_s14A_default.png`:
  the best WT-VB half posed.
- `figures/M2b_WT_s12A_wetting.png`, `M2b_WT_s12B_wetting.png`,
  `M2b_WT_s12A_default.png`: a WT pair. Half A sinks to 5.6 m and comes back,
  half B surfaces by 28 s, and under the default half A stays at 2.7-3.3 m.
- `figures/M2b_ST_series.png`, `figures/M2b_ST_s13A_wetting.png`: the rowing
  turn, showing the nose-down stand and the dive under "wetting".

## What this means

The V+ law can hold the machine up after the vault, at 351 W, but only
sometimes (1 seed in 6) and only under "wetting". Under the default the vault
does not leave the water, which repeats M2's finding. The take-off transient
puts the motors on their torque-speed limit, and from there the lateral loss
is the failure. Flapping with the flow angle held does not swim to a commanded
heading or depth in either mode. In this model and with this law, its depth
behaviour depends on the entrainment mode more than on the command. A rowing
stroke at 0.6 Hz and 80 deg turns the hull, but it pitches and rolls it over.
Reading "turns" from the nose heading on this hull needs an attitude gate
first.
