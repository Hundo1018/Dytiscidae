# E0n and E0m — results

Spec: `PREREG_nulls_mutants.md` (frozen 2026-11-11, not edited). Run 2026-11-11,
main @ a7770c2, MuJoCo 3.15, numpy path, one process at a time, no training, no
optimiser, no retry. `triquad.py` sha256 before, during and after:
`1cc5f4bd199505f412146f468e3a14f71b20ec55f7409491d50103ba99ecee67`. `Hand` and its
gains are untouched; the arms are wrappers in `nulls.py`, the mutants are run
through `triquad`'s own functions from `mutants.py`.

Rows: `results_nulls.jsonl` (236), `results_mutants.jsonl` (528), sample
`mutants.json`. A row with a `bad_qacc`, a divergence, a flat pack or an
exception is a failed row, never a 0. Both runs finished; nothing was stopped
early (E0n 547 s of rollouts, E0m 1,567 s of the 3,600 allowed).

## 1 Identity tests (run first; `logs/nulls_ident.log`, pasted)

```
IDENT 1 PASS: hand B1 seed 5 air from nulls.run_row: score 0.9243465325555722 vs results_final.jsonl 0.9243465325555722; 12 fields (15 measurements) compared, 0 differ []
IDENT 2 PASS: shuffled shift 0 vs hand, B1 seed 0, first halves: air: first-half competence 0.910937852897209 / 0.910937852897209, 32 measurements identical (pair score hand 0.9118, shift-0 0.0000); water: first-half competence 0.2470816700460142 / 0.2470816700460142, 26 measurements identical (pair score hand 0.2227, shift-0 0.0000); land: first-half competence 0.7108105893368514 / 0.7108105893368514, 24 measurements identical (pair score hand 0.7130, shift-0 0.0000)
IDENT 3 PASS: blind B1 water_to_air: environment commanded [(0.0, 'water'), (1.52, 'air')]; the law was handed ['water']; mode trace [(0.0, 'no-command'), (0.04, 'water-hold')] (no-command is the first 0.04 s, as in the hand arm); crossed False
IDENT 4 PASS: base B1 air seed 0: commanded rotor speed min 655.2466061649545 max 655.2466061649545 rad/s over 2017 calls; rest throttle 0.5547765694394671 x top speed 1181.1 = 655.2466061649545
```

Two things the lines show that the spec's wording did not: the environment's
command changes at step 375 (1.5 s) and reaches any controller at the next 25 Hz
decision, 1.52 s; and in a crossing or a mission the hand law has no command
for its first 0.04 s (it clears it at the first step), in every arm including
E0's.

## 2 E0n — the table

`logs/nulls_summary.log` is the full table. n is rows; "failed" rows are not
measurements and are excluded from the counts to their right.

| arm | item | B1 | B0 |
|---|---|---|---|
| base | air | 1 of 8 measured: 0.000. Failed 7 (bad_qacc 6, battery exhausted 1) | 3 of 8 measured: 0.000. Failed 5 (bad_qacc 3, diverged 2) |
| base | water | 5 of 8 measured: max 0.000. Failed 3 (bad_qacc) | 5 of 8 measured: max 6.4e-23. Failed 3 (bad_qacc, battery exhausted, diverged) |
| base | land | 8 of 8: max 1.0e-20 | 8 of 8: max 4.0e-21 |
| base | air_to_water | 1 measured: not crossed, hold 0.00. Failed 7 | 1 measured: not crossed, hold 0.00. Failed 7 |
| base | water_to_air | 5 measured: 0 crossed, hold 1.00, graded approach 0.14-0.56. Failed 3, **one of them credited** (entry 7, bad_qacc) | 4 measured: **1 crossed** (entry 1, hold 1.00), others approach 0.00. Failed 4, **one of them credited** (entry 2, bad_qacc) |
| base | water_to_land | 7 measured: **1 crossed** (entry 2, hold 1.00), one approach 0.60, others 0.00. Failed 1 | 1 measured: not crossed. Failed 7 |
| base | land_to_air | 8 measured: 0 crossed, hold 0.79-1.00, approach 0.01-0.16 | 8 measured: 0 crossed, hold 0.80-1.00, approach 0.00-0.07 |
| base | mission from air / water / land | failed (1 of 1, then bad_qacc; 0.251) / 0 of 2, 0.138 / 0 of 2, 0.113 | 1 of 2, 0.486 / failed (bad_qacc; 0.129) / 0 of 2, 0.113 |
| shuffled | air | 8 of 8: 0.000 | 8 of 8: 0.000 |
| shuffled | water | 0.000 0.065 0.289 0.318 0.000 0.000 0.000 0.000 (median 2e-17) | 0.000 0.099 0.291 0.383 0.037 0.000 0.000 0.000 (median 0.019) |
| shuffled | land | 8 of 8: max 3.6e-18 | 8 of 8: max 1.2e-17 |
| blind | each of the 4 crossings | 0 of 8 crossed, approach 0.000 in all 32 rows, hold 0.87-1.00 | 0 of 8 crossed, approach 0.000 in all 32 rows, hold 0.96-1.00 |
| blind | mission from air / water / land | 0 of 2 each; 0.150 / 0.230 / 0.148 | 0 of 2 each; 0.150 / 0.230 / 0.148 |

Failed rows by arm: base 49 of 118 (bad_qacc 34, diverged 13, battery exhausted
2); shuffled 0 of 48; blind 0 of 70. The open-loop machine is what breaks the
simulation: rotors commanded at 655 (B1) or 531 (B0) rad/s that meet water.
The two "battery exhausted" rows are the same event billed to the pack (a
200 Wh pack flat inside 8 s).

The two base crossings credited on rows that are measurements: B0
`water_to_air`, entry state 1 (hold 1.00, ends 1.12 m clear, lowest up-vector
-0.96: it tumbles out, 1.60 Wh in 6 s); B1 `water_to_land`, entry state 2 (hold
1.00, ends ashore at x = 8.34, lowest up-vector -0.98). `run_transition` does
not read the bad-qacc counter, so as the instrument stands it credits two more
(`water_to_air`, B1 entry 7 and B0 entry 2): **4 of 64 by the instrument, 2 of
the 35 rows that are measurements.**

Shuffled water, the read the spec asked for. Progress cancels exactly in every
row (`pair_along_mean` is 0.0 to the sign bit), so what a command-deaf competent
machine keeps is the hold: it is paid when the depth it was fed is close to the
one commanded and the phase order matches. Seeds 2 and 3 (fed 5.65 for 5.85 m,
and 5.74 for 5.65 m, cruise first in both): hold 0.67-0.88, score 0.29-0.38.
Pooled over the 16 rows: 9 are 0, 7 are >= 0.012, **4 are >= 0.15**, maximum
0.383, median 0. The hand arm on the same seeds: 0.21-0.72 (B1), 0.40-0.85 (B0).

### Verdicts

**N1.** *"base, media (3 media x seeds 0-7 x 2 bodies): air 0.0 on 16 of 16;
water and land < 0.012 on 16 of 16 each. Falsified by any value >= 0.012."*
**Not falsified; confirmed for land, not measurable on most of air.** Land 16
of 16 (max 1.0e-20). Air 0.0 on the 4 of 16 rows that are measurements; water
at most 6.4e-23 on the 10 of 16 that are. The other 18 rows failed (bad_qacc
13, diverged 3, battery exhausted 2).

**N2.** *"base, crossings (4 kinds x the 8 entry states of probe X1 x 2
bodies): **at least one crossing is credited**, in `land_to_air` or
`water_to_air` (my confidence 0.6). 0 of 64 is rival (a) for this arm."*
**Confirmed.** 2 credited among the 35 measured rows (one `water_to_air` on B0,
one `water_to_land` on B1), 4 of 64 as `run_transition` counts. `land_to_air`:
0 of 16, graded approach up to 0.16. 29 of the 64 rows failed (bad_qacc 19,
diverged 10). Rival (b) holds for this arm at a low rate; the uncrossed B1
`water_to_air` rows are still paid a graded approach of 0.14-0.56.

**N3.** *"shuffled, media: air 0.0 on 16 of 16 (by construction of the pair; a
violation is an instrument defect). Land < 0.012 on at least 14 of 16. Water:
no zero is predicted, because the hold pays depth keeping and the commanded
depth may coincide across seeds; predicted median < 0.15. The water
distribution is the read: it is the floor a learned policy's water score has to
be referenced to."* **Confirmed.** Air 0.0 on 16 of 16 (first halves alone score
0.89-0.93 on the seeds whose fed turn has the commanded sign; the pair takes it
to 0). Land 16 of 16 (max 1.2e-17). Water medians 2e-17 (B1) and 0.019 (B0);
distribution above.

**N4.** *"blind, crossings: 0 of 64 crossed, graded approach <= 0.05 in every
row. Falsified by any crossing."* **Confirmed.** 0 of 64, approach 0.000 in
every row, no failed row.

**N5.** *"missions (3 starts x 2 bodies): base at most 1 of 2 transitions,
blind 0 of 2; `continuous_score` pays each of them more than 0.1 from at least
one start (the leak E0 found with the still arm)."* **Confirmed.** Base: 1 of 2
from air on B0, 0 of 2 on three rows, two rows failed (B1 from air had 1 of 1
when it hit a bad_qacc). Blind: 0 of 2 on 6 of 6. `continuous_score`: base
0.113-0.486 on its four measured rows; blind 0.148-0.230 from every start.

What the three arms say together, for the lead's gate decision: a competent
machine that ignores the command (blind, shuffled) is paid nothing by the
crossings, by air or by land, and up to 0.38 by water (0.15 cleared on 4 of
16). An open-loop thrust machine (base) is paid nothing by the media where it
can be measured, is credited 2-4 crossings of 64, and destroys 42% of its own
rollouts. `continuous_score` pays every arm.

## 3 E0m — the sample

`mutants.json`; `logs/mutants_draw.log`. `np.random.default_rng(20261111)`. Each
mutant is one call of `mutate` (`dytiscidae/core/genome.py:1234`) as the search
calls it at `dytiscidae/evolution/loop.py:2128-2129`,
`mutate(parent.genome, rng, operators=operators, n_ops=regime.n_mutations)`,
with the plan drawn as `Curator.choose_operators` draws it
(`dytiscidae/evolution/curator.py:251-258`). 38 draws; 6 left the genome
unchanged (in three nothing applied: `cppn_weights`, `actuator`, `stroke`,
`cppn_structure` and `jet` had nothing to act on; in three `edge_topology` or
`radial_symmetry` reports itself applied and no gene differs); the first 32
that changed it are M00-M31, none dropped. All 32 build.

| mutant | operators applied | Tier-0 | body | mass kg | rotors | parts | density ratio | crossings a2w w2a w2l l2a | missions air water land | media air; water; land (seeds 0,1) | still crossings | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M00 | radial_symmetry+add_part | pass | changed | 3.077 | 4 | 5 | 1.164 | - - - X | 1/2 1/2 1/2 | 0.00,0.00; 0.00,0.08; 0.00,0.00 | 0 | level margin 0.61 |
| M01 | part_kind+duplicate_part | pass | changed | 2.125 | 4 | 5 | 1.095 | X X - X | 1/2 0/2 2/2 | 0.93,0.96; 0.19,0.16; 0.71,0.73 | 0 | |
| M02 | edge_topology | pass | changed | 2.036 | 4 | 4 | 1.080 | X X X X | 2/2 2/2 2/2 | 0.90,0.90; 0.22,0.56; 0.69,0.00 | 0 | |
| M03 | part_kind | FAIL `has_propulsor` | changed | 2.045 | 4 | 4 | 1.082 | not run | not run | not run | not run | the fin is no longer a surface |
| M04 | material+scale | pass | changed | 2.195 | 4 | 4 | 0.992 | X X - X | 2/2 1/2 2/2 | 0.91,0.90; 0.24,0.32; 0.71,0.74 | 0 | |
| M05 | add_part | pass | changed | 4.507 | 4 | 5 | 1.159 | N/A | N/A | N/A | 0 | law not applicable: `AssertionError` in `Hand.__init__` (8 actuators that are not rotors) |
| M06 | part_kind+scale | pass | changed | 1.985 | 4 | 4 | 1.127 | X X X X | 2/2 2/2 2/2 | 0.91,0.91; 0.23,0.74; 0.72,0.74 | 0 | |
| M07 | rotor | pass | changed | 1.757 | 0 | 4 | 0.929 | N/A | N/A | N/A | 0 (2 rows diverged) | law not applicable: `IndexError` in `Hand.__init__` (no rotor) |
| M08 | edge_placement | pass | changed | 2.060 | 4 | 4 | 1.083 | - - - - | 1/2 0/1 1/2 | 0.00,F; 0.00,0.23; 0.63,0.00 | 0 | diverged in `water_to_land` and two missions; bad_qacc |
| M09 | global_energy+radial_symmetry | pass | changed | 3.349 | 8 | 4 | 1.737 | - - - - | 0/0 0/2 1/2 | F,F; 0.00,0.00; 0.00,0.00 | 0 | 8 rotors, Li-Po; diverged, bad_qacc |
| M10 | material | pass | changed | 2.046 | 4 | 4 | 1.082 | X X X X | 2/2 2/2 2/2 | 0.91,0.90; 0.23,0.69; 0.71,0.74 | 0 | |
| M11 | scale+remove_part | pass | changed | 2.063 | 4 | 3 | 1.039 | X X X X | 2/2 2/2 2/2 | 0.92,0.90; 0.24,0.45; 0.74,0.73 | 0 | |
| M12 | scale | pass | changed | 2.426 | 4 | 4 | 0.896 | X X - X | 2/2 0/1 2/2 | 0.91,0.90; 0.21,0.25; 0.71,0.74 | 0 | bad_qacc in the mission from water |
| M13 | global_buoyancy+edge_placement | pass | changed | 2.037 | 4 | 4 | 1.080 | X X X X | 2/2 2/2 2/2 | 0.89,0.90; 0.25,0.67; 0.71,0.68 | 0 | |
| M14 | radial_symmetry+part_dimensions | pass | changed | 2.064 | 4 | 4 | 1.084 | X X X X | 1/2 2/2 2/2 | 0.87,0.89; 0.24,0.54; 0.71,0.74 | 0 | |
| M15 | global_buoyancy | pass | same | 2.045 | 4 | 4 | 1.081 | X X X X | 2/2 2/2 2/2 | 0.91,0.90; 0.22,0.72; 0.71,0.74 | 0 | |
| M16 | global_buoyancy | pass | same | 2.045 | 4 | 4 | 1.081 | X X X X | 2/2 2/2 2/2 | 0.91,0.90; 0.22,0.72; 0.71,0.74 | 0 | |
| M17 | duplicate_part | pass | changed | 2.046 | 4 | 5 | 1.082 | X X X X | 2/2 2/2 2/2 | 0.89,0.89; 0.25,0.73; 0.71,0.74 | 0 | |
| M18 | edge_topology+material | pass | changed | 1.961 | 3 | 4 | 1.042 | - - - - | 0/0 0/1 1/2 | F,F; 0.15,0.35; 0.01,0.00 | 0 | 3 rotors; diverged, bad_qacc |
| M19 | global_buoyancy | pass | changed | 2.045 | 4 | 4 | 1.081 | X X X X | 2/2 2/2 2/2 | 0.91,0.90; 0.22,0.66; 0.71,0.74 | 0 | |
| M20 | global_buoyancy+body_field | pass | changed | 2.578 | 4 | 4 | 0.632 | X X - X | 2/2 2/2 2/2 | 0.91,0.91; 0.23,0.21; 0.73,0.74 | 0 | `water_to_land` diverged; Tier-0 passes with `can_submerge` -0.12 |
| M21 | part_dimensions+add_part | pass | changed | 2.765 | 4 | 5 | 0.767 | X X - X | 2/2 1/2 2/2 | 0.58,0.58; 0.21,0.20; 0.74,0.74 | 0 | level margin 0.69: the air flight term is gated off |
| M22 | radial_symmetry+global_buoyancy | pass | changed | 2.045 | 4 | 4 | 1.081 | X X - X | 2/2 2/2 2/2 | 0.91,0.90; 0.22,0.37; 0.69,0.00 | 0 | `water_to_land` diverged |
| M23 | edge_placement+phase_gradient | pass | changed | 2.039 | 4 | 4 | 1.080 | X X X X | 2/2 2/2 2/2 | 0.91,0.91; 0.23,0.72; 0.66,0.71 | 0 | |
| M24 | global_energy | pass | changed | 1.881 | 4 | 4 | 0.995 | X X X X | 2/2 2/2 2/2 | 0.91,0.90; 0.21,0.36; 0.72,0.74 | 0 | 143 Wh |
| M25 | remove_part+radial_symmetry | pass | changed | 2.095 | 5 | 3 | 1.115 | - - - - | 0/0 0/2 1/2 | F,0.00; 0.00,0.04; 0.00,0.00 | 0 | 5 rotors; diverged, bad_qacc |
| M26 | material+scale | pass | changed | 2.429 | 4 | 4 | 0.889 | X X - X | 2/2 2/2 2/2 | 0.91,0.90; 0.25,0.25; 0.71,0.74 | 0 | |
| M27 | scale+edge_topology | pass | changed | 2.052 | 4 | 4 | 1.077 | X X X X | 1/2 2/2 2/2 | 0.91,0.90; 0.22,0.69; 0.71,0.74 | 0 | |
| M28 | radial_symmetry+part_dimensions | pass | changed | 2.058 | 4 | 4 | 1.071 | X X - X | 2/2 2/2 2/2 | 0.91,0.90; 0.22,0.66; 0.71,0.74 | 0 | |
| M29 | remove_part+scale | pass | changed | 2.243 | 4 | 3 | 0.941 | X X X X | 1/2 2/2 2/2 | 0.91,0.90; 0.20,0.28; 0.73,0.73 | 0 | |
| M30 | remove_part+scale | pass | changed | 1.593 | 4 | 3 | 1.678 | X - X X | 2/2 2/2 2/2 | 0.92,0.91; 0.37,0.81; 0.72,0.73 | 0 | |
| M31 | part_kind+phase_gradient | pass | changed | 2.052 | 4 | 4 | 1.083 | X X - X | 2/2 2/2 2/2 | 0.90,0.90; 0.23,0.70; 0.71,0.74 | 0 | `water_to_land` not crossed, with a bad_qacc |

X crossed with no bad_qacc; - not; F a failed row; N/A law not applicable. B1
itself on the same wiring: X X X X, 2/2 2/2 2/2 (`results_final.jsonl`).

Counts (`logs/mutants_summary.log`):

| | n | mission from all three starts | all four crossings | both |
|---|---|---|---|---|
| Tier-0-passing, body changed | 29 | **15 (0.517)** | 12 (0.414) | 9 |
| of those, density ratio inside 1.02-1.10 | 15 | 10 | 9 | |
| of those, outside | 14 | 5 | 3 | |
| Tier-0-passing, body unchanged (M15, M16) | 2 | 2 | 2 | 2 |

Crossings lost among the 29, by kind: `air_to_water` 7, `water_to_air` 8,
**`water_to_land` 16**, `land_to_air` 6. Missions lost by start: air 11, water
11, land 7. Hand rows across the 31 mutants: 403, of which 26 are "law not
applicable" (M05, M07), 12 carry a bad_qacc and 11 more diverged.

### Verdicts

**M1.** *"At least 16 of 32 one-step mutants pass Tier-0. Below that, the anchor
sits on a Tier-0 edge and the search would discard most of its children before
any rollout."* **Confirmed: 31 of 32.** The one rejection is the fin losing its
surface kind (`has_propulsor` -1.00). Two pass the gate with a failing check
(`can_submerge` -0.12 and -0.05; the gate is at -0.85).

**M2.** *"Among Tier-0-passing mutants whose body changed: at least half
complete the mission (both transitions) from all three starts, and at least
half make all four crossings. Losses concentrate in `water_to_air` and in
mutants whose density ratio left 1.02-1.10."* **Mixed.** Mission from all three
starts: 15 of 29, confirmed by one mutant. All four crossings: 12 of 29,
**refuted**. Concentration by kind: **refuted**, the crossing that goes first
is `water_to_land` (16 lost against 8 for `water_to_air`): four of the sixteen
diverged and two more carry a bad_qacc. Concentration by density ratio:
confirmed, 11 of the 14 outside the band lose a crossing against 6 of the 15
inside, and 9 of 14 against 5 of 15 lose the mission.

**M3.** *"Still arm: 0 crossings on every mutant."* **Confirmed: 0 of 124** (31
mutants x 4). 122 rows are measurements (largest graded approach 0.021); the 2
that are not are the rotorless M07 held still, which diverged (x > 1,000 m) in
`air_to_water` and `water_to_land`.

**Decision rule** (frozen): share of body-changing Tier-0-passing mutants that
complete the mission from all three starts = **15 of 29 = 0.517, the 0.3-0.7
band**: "run 2 needs the law's model inputs in the policy and a per-child check
before any fine-tune". A 95% Wilson interval on 15 of 29 is 0.34-0.69, inside
the band at both ends. The count does not depend on how the two "law not
applicable" children are treated enough to move it: without them 15 of 27 =
0.556; with the two unchanged bodies included, 17 of 31 = 0.548.

Which children lost the mission, for whoever restricts the operators (read
off the table after the fact, not predicted). One start of three: M04, M12,
M14, M21, M27, M29. Two or three: M00 (3.1 kg, `radial_symmetry+add_part`), M01
(`part_kind+duplicate_part`), M08 (`edge_placement`), and all three whose rotor
count changed, M09 (eight), M18 (three), M25 (five). Law not applicable: M05
(new joint motors), M07 (no rotors).

## 4 What I decided that the spec did not fix

E0n:

1. **base** is `Controller(params=env.cpg.base, policy=None, bases=None)` on the
   CPG's own `command`: rotors at the genome's `rotor_throttle`, 655.25 rad/s
   on B1 and 531.50 on B0, nothing else actuated.
2. **blind** freezes at the first medium the evaluator shows after the law's
   reset at step 0, i.e. at the second control decision (0.04 s). It is the
   start medium in every row.
3. **shuffled** is a view of the environment in which only `_phase_now` and
   `_active_task` are replaced, by the first-half schedule of seed (s+1)%8 with
   the air speed filled in as `_arm_task` fills it, the same in both halves.
   The real environment scores its own task.
4. One fresh environment per body, seed and arm; media first, in
   air-water-land order; a crossing row's `seed` is its X1 entry state;
   missions once per start, body and arm (they do not read the seed).
5. A crossing row with a bad_qacc is a failed row although `run_transition`
   credits it; N2 is therefore reported both ways (2 and 4).
6. The table was not stopped at failed rows: section 5 stops on an item that
   raises, and none raised.

E0m:

1. The plan comes from a fresh `OperatorBandit()` through the same `select`
   calls as `Curator.choose_operators`, and is never credited between draws, so
   it is two distinct operators drawn uniformly from the 23. `n_mutations` 2 and
   `structural_bias` 1.6 are what `update_regime` sets while an archive holds
   fewer than 20 elites (`curator.py:472-476`). So "one step" is one `mutate`
   call applying up to two operators (20 of the 32 applied two, 12 one). One
   generator serves the plan and the mutation. No crossover (`loop.py:2132`
   needs a second parent).
2. "Genome differs": sha256 over every gene, bookkeeping excluded (lineage,
   ids, generation, body plan).
3. "Body changed": a hash of the compiled model (bodies, geoms, joints,
   actuators, meshes), the fluid panels, the rotors, the motors, the pack, and
   the mass, buoyant volume and gas volume the law reads. Gait genes alone do
   not change it. M15 and M16 (`global_buoyancy`) are the two unchanged bodies.
4. Tier-0 is `triquad.tier0_gate`, the search's predicate.
5. Per mutant: one environment at seed 0 for air, water, land, the four
   crossings and three missions (hand), then the four crossings (still); a
   second at seed 1 for the three media. A mission "completes" with 2 of 2
   transitions, no failure string and no bad_qacc; a crossing counts with
   `crossed` and no bad_qacc.
6. "Law not applicable" children stay in the denominator as not completing.
7. The 60-minute budget was counted per mutant including environment builds.

## 5 Reproduce

```bash
cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
E=experiments/triphibian_diagnosis/existence
.venv/bin/python $E/nulls.py ident                                            # 36 s
.venv/bin/python $E/nulls.py row --body B0 --arm base --item water_to_air --seed 1   # one E0n row
rm -f $E/results_nulls.jsonl; .venv/bin/python $E/nulls.py table              # 236 rows, about 11 min
.venv/bin/python $E/nulls.py summary
.venv/bin/python $E/mutants.py draw                                           # the 32, no rollouts
.venv/bin/python $E/mutants.py run --only 14 --out $E/logs/mutants_M14_rows.log  # one E0m mutant, 60 s
rm -f $E/results_mutants.jsonl; .venv/bin/python $E/mutants.py run            # 528 rows, 26 min
.venv/bin/python $E/mutants.py summary
```

The two single-item commands were re-run after the tables and returned the
stored rows with 0 differences (`logs/nulls_row_check.log`,
`logs/mutants_M14_rows.log`).
