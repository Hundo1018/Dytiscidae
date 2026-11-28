# N0 result — 2026-11-12

`PREREG_neighbourhood.md` was frozen and committed first (0a… see `git log`).
Reproduce: `PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python
experiments/triphibian_diagnosis/scripts/neighbourhood.py OUT.jsonl --jobs 8`
(rows: `results/neighbourhood.jsonl`). One departure, made before the run:
a draw with no operator applied is redrawn; the pre-registration said "equal
to the parent's", which the genome has no test for.

## Children of the anchor, 24 per depth

| steps | pass Tier-0 | controller cannot be built | flown | keep the mission (6 of 6) | of Tier-0 passers |
|---|---|---|---|---|---|
| 1 | 24 | 1 | 23 | 20 | 0.83 |
| 2 | 23 | 4 | 19 | 17 | 0.74 |
| 4 | 23 | 3 | 20 | 15 | 0.65 |
| 8 | 23 | 13 | 10 | 4 | 0.17 |

Split by whether the child has the parent's four rotors and four limb
actuators:

| steps | same counts: keep | changed counts: keep | changed counts: cannot be built |
|---|---|---|---|
| 1 | 20 of 22 | 0 of 2 | 1 |
| 2 | 15 of 16 | 2 of 7 | 4 |
| 4 | 11 of 14 | 4 of 9 | 3 |
| 8 | 2 of 6 | 2 of 17 | 13 |

All 21 "cannot be built" are array-shape errors in the controller (a rotor or
limb count the code fixes), none a physical reason. Missions lost by children
that flew: 10 failed rollouts; of the 38 judged, 14 the land leg after the
water, 8 the water leg after the air, 6 and 4 and 4 the first leg in water,
air and on land, 2 the air leg after the land.

## The five predictions

1. *"One step: between 0.3 and 0.7 of the Tier-0-passing children keep the
   mission."* **Refuted, upward**: 0.83.
2. *"... at 4 steps under 0.25, at 8 steps under 0.1."* **Refuted, upward**:
   0.65 and 0.17.
3. *"Every child whose rotor count or limb-actuator count differs ... either
   cannot be built by the controller or loses the mission."* **Refuted**: 8
   of 35 such children keep it.
4. *"Among children with the parent's rotor and limb counts, at least 0.6
   keep the mission at one step."* **Held**: 20 of 22.
5. *"The lost legs are mostly the land leg after the water."* **Not held**:
   it is the largest group (14 of 38), not most.

## What it says, and what it does not

- There is a neighbourhood, and it is wider than E0m suggested: out to four
  steps about two thirds of children are still flown through the mission by
  a controller built from their own model, with the teacher unchanged.
- Out to four steps the main loss is not physics but the code: with the
  parent's counts 46 of 52 keep the mission; children with other counts
  mostly cannot be built. At eight steps 13 of 23 cannot be built.
- So a search around the anchor can use the fixed teacher as its controller,
  and the first thing that would bound it is the controller's assumption of
  four rotors and four limbs.
- Inflated by construction: several operators (gait, phase gradient) change
  only the body's own pattern generator, which the split controller
  overrides; such a child is the parent to this framework. At one step 16 of
  24 children have the parent's density ratio to two digits. The share that
  matters is among children whose body changed, and that was not separated.
- Two seeds per start, one parent, 24 children per depth; "keeps" is six of
  six, with no interval on it.
