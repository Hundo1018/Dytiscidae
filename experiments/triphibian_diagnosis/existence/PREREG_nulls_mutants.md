# E0n and E0m — command-ignoring arms on the anchor, and the hand law on mutated anchors

Frozen 2026-11-11 before any rollout, commit a7770c2, by the lead. Two
no-learning measurements asked for by the review (`reports/R1_review.md` P2,
P5). Same rules as `PREREG.md` in this directory: the repo's own scores,
unmodified; only the command source is substituted; no optimiser; nothing
under `dytiscidae/` is edited. Nothing in this file is edited after the first
rollout; outcomes go in `RESULT_nulls_mutants.md`.

## E0n — does a thrust-capable machine that ignores the command get paid?

### 0 Question and decision

E0's only null was rotors stopped. On B1 and B0, do three machines that have
thrust and ignore the command earn media scores, crossings or mission
transitions from the unmodified instruments?

Decision: which null arms the run gates must carry, and whether
`CrossingTracker` and the paired segment scores can define "done" for this
body as they stand, or the station gate (`docs/ARCH46_SPEC.md:406-414`, "Not
built") comes first.

### 1 What is known, and the rivals

- Still arm (E0): media <= 1.4e-17, crossings 0 of 8, `continuous_score`
  0.15-0.58.
- Open-loop rotors crossed `water_to_air` and `water_to_land` on arch46
  elites by outlasting the hold (`ARCH46_SPEC.md:406-414`).
- arch47-60, recounted today from `results/events_arch*.json`: all 46
  `land_to_air` and all 6 `water_to_air` crossings were made by bodies whose
  air score is under 0.012; 13 of the 52 have no rotor.
- B1's base gait is its rotors at 1.05 x hover (`RESULT.md:239`).

Rivals: (a) tight: every command-ignoring arm reads 0; (b) open-loop thrust
is paid as an into-air crossing; (c) the pair does not cancel a competent
machine that is deaf to the draw; (d) the arm is not the null it claims to be
(an implementation error), checked by the identity tests in section 4.

### 2 Mechanism

An into-air crossing is "0.5 m clear at the last step, after a hold of at
least 0.5 in the start medium". A body with thrust just over its weight and a
rotor that takes time to spin up can sit through most of the hold and then
rise. The paired scores cancel motion the command did not choose: mean signed
speed over a heading and its antipode, and a turn pair whose mean is <= 0 for
a machine that turns the same way twice.

### 3 Arms and predictions

- **base**: the genome's own gait, no feedback (what `evaluate_tier1` runs
  when no policy and no hand law is installed).
- **blind**: the hand law, with the commanded medium it sees frozen at the
  medium the rollout starts in. Crossings and missions only.
- **shuffled**: the hand law, fed in both halves of a pair the first-half
  task of seed `(s + 1) % 8` for the same medium instead of its own. Media
  segments only.

N1. base, media (3 media x seeds 0-7 x 2 bodies): air 0.0 on 16 of 16; water
    and land < 0.012 on 16 of 16 each. Falsified by any value >= 0.012.
N2. base, crossings (4 kinds x the 8 entry states of probe X1 x 2 bodies):
    **at least one crossing is credited**, in `land_to_air` or `water_to_air`
    (my confidence 0.6). 0 of 64 is rival (a) for this arm.
N3. shuffled, media: air 0.0 on 16 of 16 (by construction of the pair; a
    violation is an instrument defect). Land < 0.012 on at least 14 of 16.
    Water: no zero is predicted, because the hold pays depth keeping and the
    commanded depth may coincide across seeds; predicted median < 0.15. The
    water distribution is the read: it is the floor a learned policy's water
    score has to be referenced to.
N4. blind, crossings: 0 of 64 crossed, graded approach <= 0.05 in every row.
    Falsified by any crossing.
N5. missions (3 starts x 2 bodies): base at most 1 of 2 transitions, blind 0
    of 2; `continuous_score` pays each of them more than 0.1 from at least
    one start (the leak E0 found with the still arm).

Not tested here: a timed script that follows the fixed 8 s schedule without
reading the command (review P2). It needs a script written for the purpose.

### 4 Measurement

- `run_segment`, `run_transition`, `run_continuous`, `continuous_score`,
  unmodified, wired exactly as `triquad.py` wires them. Entry states: the
  eight of `probes/x1_entry_states.py`. Seeds 0-7.
- Missing is not zero: a rollout ending in `bad_qacc`, divergence, a flat
  pack or an exception is reported as failed with its reason.
- Floors: 0.012 (existence) and 0.15 (certified water and land).
- Identity tests, run first and pasted in the result:
  1. `triquad.py` is not modified (sha256 before and after), and the hand arm
     on B1, seed 5, air, reproduces its `results_final.jsonl` row to the last
     digit from the new script's code path.
  2. shuffled with shift 0 (its own seed's first-half task in the first half)
     reproduces the hand arm's first-half measurements to the last digit on
     one seed per medium.
  3. blind: the controller's mode trace is constant through one crossing
     while the environment's command changes at 1.5 s (one line of log).
  4. base: the commanded rotor speed equals the rest throttle throughout one
     rollout (one line of log).
- The rollouts are deterministic (the lead's re-run of E0 seed 5 differed in
  0 digits of 20 rows), so there is no sampling noise inside a row.

### 5 Design

Bodies B1 and B0. `Hand` and its gains are not changed: the arms are
subclasses or wrappers in a new file. No retry with different settings; every
rollout is a row. Stop when the table is filled or an item raises (reported
with the trace).

## E0m — does the same hand law still work on the anchor's children?

### 0 Question and decision

Of one-step mutants of B1 drawn by the search's own operator
(`dytiscidae/core/genome.py:1234`, `mutate`, called as the search calls it),
what share pass Tier-0, and what share of those does the **unchanged** hand
law still take through each crossing and the mission?

Decision: whether run 2 ("bodies around the anchor; a child inherits the
controller") has a premise, and what a child's budget has to cover.

### 1 What is known, and the rivals

- B1 passes Tier-0 by +0.117; B0 fails it (`RESULT.md` section 4).
- One law on two near-identical quads: 32 of 32 against 29 of 32 crossings.
- 153 of 191 arch49 rotor elites cannot lift their weight (ROADMAP:3984-3999).

Rivals: (a) robust: most children keep the mission; (b) brittle: most lose
it; (c) trivially robust: the mutation did not change the body (recorded per
mutant: mass, part count, rotor count, density ratio, and whether the built
phenotype differs); (d) a child "crosses" passively (still arm on every
mutant's crossings).

### 2 Mechanism

The law reads the model (mass, inertia, rotor positions, buoyant volume), so
it should follow parametric changes. Its way-points, gear geometry and staged
water exit were set on B1, so a change of topology or a density ratio outside
1.02-1.10 can break it.

### 3 Predictions

M1. At least 16 of 32 one-step mutants pass Tier-0. Below that, the anchor
    sits on a Tier-0 edge and the search would discard most of its children
    before any rollout.
M2. Among Tier-0-passing mutants whose body changed: at least half complete
    the mission (both transitions) from all three starts, and at least half
    make all four crossings. Losses concentrate in `water_to_air` and in
    mutants whose density ratio left 1.02-1.10.
M3. Still arm: 0 crossings on every mutant.

Decision rule, fixed now, on the share of body-changing Tier-0-passing
mutants that complete the mission from all three starts: under 0.3, run 2 as
written has no premise (restrict it to the operators that kept the mission,
or re-plan); 0.3-0.7, run 2 needs the law's model inputs in the policy and a
per-child check before any fine-tune; 0.7 or more, the premise holds as an
upper bound (the law reads the model; a policy would have to be given it).

### 4 Measurement

- Sample: `np.random.default_rng(20261111)`; each mutant is one call of
  `mutate` on a fresh copy of B1's genome with the arguments the search uses
  (cite the call site); the first 32 whose genome differs from B1's, in
  order, none dropped. A mutant that cannot be built counts in the
  denominator as "not buildable".
- Per Tier-0-passing mutant: hand arm on the four crossings (standard entry
  state), the mission from three starts, and each medium at seeds 0 and 1;
  still arm on the four crossings.
- An exception in `Hand.__init__` or in a rollout is the outcome "law not
  applicable", recorded with its text. It is not something to fix.
- n = 32 puts a share near 0.5 within about +-0.17 (95%).

### 5 Design

`Hand` and its gains unchanged. No selection among mutants. Stop at 32
mutants or 60 minutes of rollouts, whichever comes first; if stopped early,
the first k in order are reported.
