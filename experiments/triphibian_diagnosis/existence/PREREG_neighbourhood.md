# N0 — how far from the anchor does the framework still evaluate a body?

Frozen 2026-11-12 before any rollout, commit bedc800. No learning. Written
after the user restated the goal (DIAGNOSIS §7b): the body has to come out
of the search; one machine doing the mission only checks the framework.

## 0 Question and decision

The search varies bodies. The framework that reads a body (inner loop built
from the body's model, the teacher, the mission verdict) was written on one.
For children of the anchor at 1, 2, 4 and 8 mutation steps, what share can
the framework fly through the mission unchanged?

Decision. The share at one step sets whether a search around the anchor can
use the fixed teacher as its controller at all (under 0.3: no; the
controller has to be made generic over bodies before any search run). The
depth at which the share falls under 0.1 is how far such a search can reach
before it is selecting on the controller's limits instead of the body's.
What the losers have in common (rotor count, limb count, density) is the
list of what the inner loop and the teacher must stop assuming.

## 1 Rivals

- **A neighbourhood exists** (E0m on B1, the stateful law: 15 of 29 at one
  step).
- **A point**: the split law is tuned to this body's numbers and almost any
  change loses it.
- **Structural only**: children lose it when the count of rotors or limbs
  changes and not otherwise, i.e. the limit is the code's assumptions (four
  rotors in `observe`, limb actuators by name), not the physics.
- **Instrument**: Tier-0 or the model reader rejects children the mission
  never gets to judge; reported as its own column, never as a miss.

## 2 Method

- Parent `ANCHOR_PLANS["legged_quad"]`. Children by `core.genome.mutate`
  with `n_ops = d`, d in 1, 2, 4, 8; 24 draws per depth from
  `np.random.default_rng([20261112, d])`; a draw that leaves the genome
  equal to the parent's is redrawn (counted).
- Each child: built; Tier-0 as the search applies it; then
  `teacher_controller` (model constants read from the child) on
  `mission.evaluate_mission` from each start, seeds 200 and 201.
- A child **keeps the mission** if it completes all six. Columns, kept
  apart: rejected by Tier-0; the controller cannot be built (an exception,
  with its text); flown and kept; flown and lost (with which legs).
- Per child: rotor count, limb-actuator count, mass, density ratio, the
  operators applied.

## 3 Predictions (written before running)

1. One step: between 0.3 and 0.7 of the Tier-0-passing children keep the
   mission (E0m: 0.52 under another law on another body).
2. The share falls with depth: at 4 steps under 0.25, at 8 steps under 0.1.
3. Every child whose rotor count or limb-actuator count differs from the
   parent's either cannot be built by the controller or loses the mission.
4. Among children with the parent's rotor and limb counts, at least 0.6 keep
   the mission at one step: the limit is mostly structural.
5. The lost legs are mostly the land leg after the water, as for the student.

## 4 Size

96 children, 6 missions each, about 12 minutes on 8 processes. One parent,
one operator set, two seeds per start: a first map, not a census.
