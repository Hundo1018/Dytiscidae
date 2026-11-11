# E0b — does a hand-driven mechanism carry the anchor's weight over the ground?

Frozen 2026-11-11 before any rollout, commit a7770c2, MuJoCo 3.15. No
learning, no search, no optimiser: a hand-written law, bodies the genome
expresses today, the repo's own physics and scores. (The user's rule of
2026-11-10 forbids new training; this is a measurement.) Written by the lead;
the rollouts are run by an agent that may not edit this file.

## 0 Question and decision

E0's land rows (0.711-0.761 on 8 of 8 seeds) were a rotor-borne taxi 3 cm up.
On 2026-11-11 the user decided that land locomotion is weight on the ground,
by jointed legs (i) or passive wheels (ii), that the rotor-borne taxi does not
count, and that the direction is crawling on the flapping wings.

Can B1, with jointed limbs made of part kinds the genome has today, move the
commanded way on land with its weight on the ground and its rotors stopped,
and still fly, swim and cross?

Decision, fixed now. Arm L passes: land in the plan is jointed legs, and the
legged body becomes the anchor candidate for P0.9. Arm L fails: the
passive-wheel part kind is built (a source change with its own test and
mutation) and this file's rows are repeated on it under a new frozen file.
Arm W is exploratory and decides nothing in the plan; if it passes it is
reported to the user as the first evidence for their stated direction.

## 1 Rivals

- **Legs carry it** (the mechanism exists; nobody had asked for it).
- **The body cannot**: servo torque, limb length or the hull's ground
  clearance leave no gait that moves 2 kg; or legs that walk cost the air,
  water or crossing rows (mass, drag, buoyancy, rotor wash).
- **The score refuses or is fooled**: the land score's posture or domain test
  reads 0 for a walker, or pays a gait that ignores the command; or the
  support quantity below is wrong (reads 1 for a hover, or 0 for a walker).
- **Nothing there**: the gait moves the body by less than the scatter does.

## 2 Mechanism under test

Arm L: B1's four fixed FOOT legs replaced by jointed, actuated limbs of
existing part kinds, driven through the existing CPG channels by a hand law
that sets gait amplitude, offset and phase from the commanded heading and the
body's own heading error. Rotors stopped during the land segment.

Arm W (exploratory): limbs that are flapping surfaces (the genome's wing or
paddle kinds on their own joints) used as the ground limbs, rotors stopped.
The same surfaces need not fly the body; the quad's rotors still do that.

## 3 Quantity added, and its probes

**Support share** `s`: over the scored part of a land half, the time mean of
the vertical component of the ground's contact force on the body, divided by
`m g`. Read from MuJoCo's contact forces each step. It is not a score here;
it is reported beside the unmodified land score.

Probes that have to pass before any arm L number is read:

- a body resting still on the beach reads `s` in 0.95-1.05;
- E0's hover taxi (the stored E0 land law on B1) reads `s` <= 0.05;
- a skid with the rotors carrying about 94% of the weight reads `s` <= 0.15;
- a body in free fall or airborne reads exactly 0, and a rollout with no
  land steps reports `s` as missing, not 0.

## 4 Predictions (written before running)

1. Arm L, unmodified land score through `evaluate.run_segment` (antipodal
   pair): >= 0.15 on >= 7 of 8 seeds, with `s` >= 0.9 and every rotor
   stopped on each of them. I hold this at about 0.5. Falsified by fewer than
   7 seeds, or by any passing seed with `s` < 0.9.
2. Arm L's body through E0's law, rows and bars unchanged
   (`existence/PREREG.md` §3): air >= 0.5 on >= 7 of 8, water >= 0.012 on
   >= 7 of 8, all four crossings, both mission transitions from each start
   with the land leg walked, not taxied. I expect water or `water_to_land`
   to be what breaks if anything does (E0m: `water_to_land` was lost by 16
   of 29 mutants).
3. Nulls on arm L's body, land segment, 8 seeds each: still (every actuator,
   `held_still_params`) < 0.012 on 8 of 8; the gait with the command frozen
   (heading fixed at the first one seen) < 0.012 on 8 of 8, because the
   antipodal pair cancels motion the command did not choose; the gait fed
   another seed's task < 0.15 on 8 of 8.
4. The land score, as it stands, pays the walker less than the taxi (the
   taxi's 0.71-0.76 was earned at a speed legs will not reach). I expect
   arm L between 0.15 and 0.5.
5. Arm W: `s` >= 0.9 and land score >= 0.012 on at least 4 of 8 seeds. I
   hold this at about 0.3. Nothing in the plan depends on it.

## 5 Design

- Scores unmodified; only the command source is substituted, as in E0.
  Nothing under `dytiscidae/` is edited. No gate is weakened or bypassed.
- Seeds 0..7 per item, fixed, no selection. Gait and gains are tuned by hand
  on seeds 100..103 only; every tuning rollout is appended to
  `trials_land.jsonl`. No automated optimisation of any kind.
- A rollout that ends in `bad_qacc`, divergence or a flat pack is failed,
  never 0. `bad_qacc` is read on every rollout.
- Every body edit is listed with its Tier-0 result; the body has to pass
  Tier-0 as the search runs it.
- Controls: the still arm, the command-frozen arm, the shuffled arm; B1 as
  built (hover taxi) beside arm L on every row.
- Stopping rule: stop when the table is filled for arm L, or when a row is
  shown blocked by physics or by a score gate (with the trace), or after
  about three hours of work on arm L and one on arm W. A clean negative is a
  result.
- Not measured here: whether a learned policy can drive the gait; wheels.
