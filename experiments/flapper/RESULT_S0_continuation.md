# Pass S0, result (A0 only): the base machine on ideal actuators

2026-11-19. Frozen text: `PREREG_S0_continuation.md` (dd89844, not edited).
Plan written before any scored row: `results/S0_plan.md`. Code (new):
`s0_machine.py`, `s0_laws.py`, `s0_run.py`, `s0_specs.py`, `s0_read.py`,
`s0_draw.py`. `mission.py` and every M-series file are unedited. Rows:
`results/S0_rows.jsonl` (101 A0 rows, 0 QACC resets, 0 errors), the gate
read `results/S0_read_A0.txt`, traces `results/S0_traces/`.

**Scope.** On the lead's instruction the ladder stops at A0: no A1-or-later
rung is reported, the §4 guards (dt/2, contact x2) are not done, and no
figure was drawn. A0 here is the base machine with A1's sweep machine for the
land, land<->air scenarios ("taken at once", §2).

## Outcome

**At A0 two scenarios pass their gates and four do not.** Air passes. Land
(on the A1 sweep machine) passes, but every land row's peak contact is over
10 weights, so its verdict needs the contact-x2 re-run, which was not done.
Water, land<->air and water<->air fail. The common cause is the ideal
actuator itself: any law that steps a hinge target (the V+ strike, M4's
contact hold) or that the witness motor used to saturate (every water law)
now gets the full force of the hinge PD at wn 300. Probe P1 (§0b) tested only
a 4 Hz sinusoid in air and did not show this.

| scenario | leg | gate (frozen in) | result at A0 (2 scored starts) | do-nothing | verdict |
|---|---|---|---|---|---|
| air | cruise hold, ±90 pair | M4 H, T | H 2/2, T 2/2 (P_t 90.0, 90.1 deg; track error ≤ 0.26 deg; roll ≤ 0.67 deg) | N fails H and T 2/2 | **passes** (dt/2 not run) |
| land (A0+A1) | stand, walk, turn | M3 stand, M3b walk 12 s / turn 14 s | stand 2/2 (13.4-13.6 W); walk pairs 2/2 (0.344, 0.342 m/s; U below the bar 2/2); turn pairs 2/2 (errors 3-11 deg) | N0 fails all | **passes on rigid contact** (peaks 51-111 weights; the x2 re-run is not done) |
| water | dive D | M2 held clause | 0/2 both modes | D0 fails | fails |
| water | swim GW | M2b GW | 0/2 both modes | W0 fails | fails |
| water | surface U | GS surface clause | wet 1/2 (seed 21 was thrown 89 m into the air), default 0/2 | **U0 passes 2/2: cannot tell** | no verdict |
| water | surface turn GS | M2b GS | 0/2 both modes | S0 fails | fails |
| land<->air (A0+A1) | take-off | M1 G1-G4 | **0/2** (G 0000; 1110 with G4 failing) | N0 fails | **fails: blocking leg** |
| land<->air | cruise 60 s | M4 H | 2/2 | N fails | passes |
| land<->air | landing | M4 land E1-E5 | 0/2 pairs (E1 near +12.5 to +13.2 m; E2 vn -1.08 to -1.33 m/s, pitch_rel -8.6 to -9.5 deg) | NL fails | fails |
| land<->air | take off again | M1 G1-G4 | 0/4 halves | - | fails |
| water<->air | exit, 30 s | M2b GX30 | wet **0/2** (never clear), default **1/2** (seed 20 clear 35.3 s, X2 flag 0.63) | X0f fails 4/4 | fails in both modes (behaviour differs) |
| water<->air | entry | M4 water E1-E5, both modes | 0/2 pairs in each mode (E1: +8.8 to +12.3 m late) | NW fails | fails |
| water<->air | exit again | GX30 | default 2/4 halves, wet 0/4 | - | fails |
| three media | - | - | not run (no S0*) | - | - |

S0* does not exist at A0. Under §2 the next rungs would be A2 (water dive,
water<->air exit), A4 (both entries) and A5/A5w/A5p; **no rung in §2 is
named for the take-off failure**, so land<->air could only reach A6, and
none of A6's wing candidates removes a hinge-force cause.

## Why each failing scenario fails (from the rows)

- **Take-off.** V+'s strike is a step to -1.35 rad. On the witness motor it
  peaks at 35 N m at the hinge; on the ideal one it is **2,427 N m** at each
  stroke hinge, the wing hits the beach at 432-1,781 weights, and the
  instantaneous electrical power is 20.5 MW. Seed (26.5, 210) still climbs to
  3.9 m and fails only G4 (attitude); (19.5, 30) is thrown and lands again
  (t_off 4.3 s). Diagnostic, outside the spec (`results/S0_diag.jsonl`, on
  the tuning entry (23.5, 330)): the same V+ row passes G1-G4 on witness
  motors without the sweep or the I0 override (wing contact 27.8 weights),
  fails only G4 on witness motors with the sweep machine (24.7 weights), and
  fails on ideal motors with or without the sweep (436-477 weights). The
  ideal actuator is the cause; the sweep machine costs G4 on its own.
- **Water.** Every water law (WT, ST, frozen at 0.25 and 0.3-0.6 Hz, both
  outside §4's 2.5-6 Hz band) becomes violent: mean body rate 15-44 rad/s,
  the hull inverted (dorsal z -0.6 to -1.0), hinge torques of 10^4-10^5 N m,
  peak contact 10^4-10^5 weights (the machine reaches the sea floor at 16-17
  m in "wetting"), and electrical energy of 10^9-10^10 J per 40 s. No QACC
  reset is flagged on any of them; these rows are outside any regime the
  model was checked in. Under the default mode the same laws stay near the
  surface (GS max depth 0.01-0.41 m) but tumble.
- **Exit.** The modes swap relative to M2b: under "wetting" XS never clears
  (the vault into the water reaches 20,000-31,000 weights of contact), and
  under the default the vault clears on both seeds (t* 4.7 and 4.0 s), one
  of which holds 30 s. Seed 20's pass has 63% of its window in the X2 regime.
- **Entries.** M4's frozen law E still lands 8.8-13.2 m past the commanded
  place (E1), as in M4: the ideal motor does not change where the glide path
  is anchored. On land E2 fails on arrival speed and pitch. In water the
  contact hold (stroke stepped to +30 deg in water) gives 10,800-129,000
  weights in "wetting" rows.

## Recorded peaks per hinge (A0, do-something rows; hinge-side torque)

Witness stall at the hinge, for scale: stroke 24 N m, feather 16 N m, tail
5 N m, sweep 24 N m. "Shaft" is |torque x hinge speed|, peak single step.
Electrical energy uses the harness's copper term (`omega_0/tau_s` = 1e3, as
the witness), so at ideal torques it is a witness-motor figure, not a
prediction.

| scenario, leg | stroke N m / rad/s / W | feather | tail | sweep | peak contact, weights | max E_elec per row, J |
|---|---|---|---|---|---|---|
| air, cruise | 62 / 6 / 118 | 27 / 6 / 54 | 0 / 2 / 0 | - | 0 | 1,780 |
| land, stand | 10 / 1 / 6 | 1 / 0 / 0 | 1 / 0 / 0 | 2 / 0 / 0 | 68 | 155 |
| land, walk | 534 / 32 / 6,245 | 13 / 3 / 29 | 5 / 1 / 7 | 689 / 41 / 9,412 | 111 | 4,723 |
| land, turn | 20 / 2 / 24 | 5 / 1 / 4 | 2 / 0 / 1 | 66 / 7 / 322 | 63 | 407 |
| land<->air, take-off | 2,427 / 108 / 118,135 | 592 / 74 / 37,568 | 148 / 25 / 3,611 | 661 / 30 / 12,989 | 1,769 | 4.1e6 |
| land<->air, cruise | 62 / 6 / 118 | 27 / 6 / 54 | 0 / 2 / 0 | 1 / 0 / 0 | 0 | 1,819 |
| land<->air, landing (+ again) | 2,169 / 90 / 103,654 | 525 / 75 / 19,834 | 64 / 20 / 1,254 | 913 / 46 / 22,653 | 2,563 | 5.6e5 |
| water<->air, exit | 3,628 / 129 / 174,388 | 2,075 / 195 / 233,283 | 858 / 166 / 108,725 | - | 31,157 | 1.0e8 |
| water<->air, entry (+ exit again) | 4,838 / 185 / 523,635 | 2,273 / 240 / 343,846 | 1,264 / 308 / 377,674 | - | 129,332 | 1.1e8 |
| water, dive | 87,976 / 128 / 6.0e6 | 58,891 / 287 / 7.5e6 | 1,792 / 316 / 5.3e5 | - | 141,882 | 1.7e10 |
| water, swim | 113,892 / 81 / 6.9e6 | 85,028 / 312 / 2.0e7 | 6,280 / 710 / 4.4e6 | - | 333,199 | 6.7e10 |
| water, surface | 98,055 / 101 / 7.1e6 | 101,978 / 247 / 1.1e7 | 2,244 / 275 / 6.1e5 | - | 332,761 | 6.5e10 |
| water, surface turn | 92,011 / 180 / 6.8e6 | 102,383 / 315 / 2.3e7 | 3,733 / 618 / 2.3e6 | - | 368,157 | 5.9e9 |

Peak contact is over the whole row, including the drop before t_c (the land
N0 rows read 49-63 weights from that drop alone). Windowed bins were added to
the instrument after A0 and are not in these rows.

## Predictions, where A0 reaches them

| | frozen text | read at A0 | |
|---|---|---|---|
| S1 | "air and land<->air. Pass at A0 plus A4 (land entry), with A1 for the land legs. (0.7)" | air passes at A0; land<->air fails at its take-off on ideal actuators, a leg A4 does not touch | **air held; land<->air failing** (A4 not run) |
| S2 | "land. Passes at A1, as M3b's K did. (0.8)" | passes at A0+A1, 2/2 on each gate; contact x2 not run | held pending the contact guard |
| S3 | "the exit under "wetting". Passes GX30 (>= 4/6) at A0 with XS untuned, because the falls were motor-limited. (0.6)" | 0/2, never clear | **failed** |
| S4 | "the exit under the default. Still fails at A0: the vault does not clear ..." | the vault clears on 2/2 seeds; GX30 met 1/2 | **failed** (it clears) |
| S5 | "swim. Fails at A0-A2 with the WT law. ..." | fails at A0 (only A0 run) | consistent so far; not decided |
| S6, S7, S8 | A2/A3, the Q-up outcome, Q-down | not reached | not decided |
| R2 | "S0*'s passes rest on X2-regime windows or on contact peaks above 10 weights" | land's pass rests on rigid-contact peaks (51-111 weights); the one exit pass is an X2 row | consistent; not tested (no S0*) |

## Disclosures (anything run or changed outside the spec)

1. **The hull's transverse inertia 0.012 kg m2** (M4's C1) is applied on every
   machine. M1b, M2b and M3b ran on the default 0.0076.
2. **Ideal actuators** through `Machine`'s `motors=` argument (J_r and gears
   frozen); the sweep hinge inherits the stroke's dict. `s0_machine.configure`
   rebinds `mission.Machine` and `m3b_sweep.SweepMachine` in the worker only.
3. **Assembly, not chaining**, as the plan says: cruise legs start from M4's
   assisted launch; the water<->air entry starts from M4's cruise launch, so
   its "exit again" runs XS on M4's machine (PD gains read at the 8 m/s
   launch, not at rest as in M2b's `build2`). Re-take-off and exit-again
   legs run on the same rollout as the landing or entry.
4. **Diagnostics outside the spec:** `results/S0_diag.jsonl`, 4 take-off rows
   on the tuning entry (23.5, 330) with `witness` / `no_I0` switches added to
   `s0_machine` for that purpose. Smoke rows (1 s windows) in the scratchpad;
   their traces `results/S0_traces/SMK_*.npz` are in the traces folder.
5. **Code changed while A0 ran:** a pose-capture hook (inert unless
   `CAPTURE` is set) was added to `s0_machine.py` mid-batch, so later A0 rows
   ran on a different sha than the hash row records; `s0_read.py` was an
   empty file when the hash row was written. After A0: the 50 ms contact
   bins (inert for A0's rows).
6. **Started after A0 finished, before the scope change arrived, then
   stopped by PID:** the A2 tuning batch (`results/S0_specs_A2tune.json`; 2
   rows in `results/S0_tuning.jsonl`, b = 1.00 dive on seed 200, mass 1.110
   kg) and an A0 guard batch (`results/S0_specs_A0guard.json`; 4 contact-x2
   land rows in `S0_rows.jsonl`, rung `A0c2`). Neither is read here.
7. **A process that was not mine was hit.** While stopping those batches I
   killed PIDs 3479003 and 3479004, which were pool workers of
   `s0b_probe.py b1map 2` (parent 3478980), not of my runs. Its pool
   respawned two workers, but the tasks those two held are lost and the
   probe may hang or miss rows.
8. The surface leg's clause passed on "wetting" seed 21 because the machine
   was thrown 89 m above the sea; its null passes anyway, so no verdict rests
   on it.
9. Not done: dt/2 and contact-x2 for any scenario, the figures, any rung
   after A0.

## One row to re-run bit-exactly

`s0_run.run_row(spec)` with `spec` = the entry `id == "A0_air_Q1_L_H"` of
`results/S0_specs_A0.json` (kind cruise; `half = dict(kind="cruise",
arm="L", point="P35", start=(3.0, 30.0, 8.2), start_name="Q1", cmd=30.0,
id="A0_air_Q1_L_H", half="H")`, cfg {}). Its recorded state hash at t_c is
in the row (`result.hash`). It was not re-run after the batch.
