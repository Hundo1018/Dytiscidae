# T1 — does the two-pair machine leave the ground under its own power?

Frozen 2026-11-12 before any rollout, commit 1025048. No learning, no
optimiser. Written by the lead; the agent that runs it may not edit this
file. First of the transitions, which the user set as the second part of
"done" (DIAGNOSIS §7c): take-off, entering and leaving the water, coming
ashore.

## 0 Where this stands

Each way of moving is established on the body with its do-nothing arms at
zero (F2, F3c-F3e, F4), and holds under a fore-to-hind wake fed back at
gains 1 and 3 (W1, W2). The body does not glide: with its wings stopped it
is in the sea in 4 s. Its flight was measured from a launch at 7.2 m/s. So
everything that leaves the ground or the water has to make its own lift at
no airspeed, and has to be controlled there. On the held rig the four wings
in phase lift 2.64 weights at 12 Hz with the body upright (F4's rig rows);
in free flight only the diagonal phasing kept the speed, and whether the
diagonals lift the weight at rest was not measured.

## 1 Three questions, in order, and the decision

- **T1a, lift at rest (held rig).** Body held upright and still, the
  diagonal stroke at 12 Hz, stroke 80 and 130 deg, feather 30 and 45 deg:
  mean force along the body's axis over the weight. Bar: 1.3 weights or more
  at some setting (the least any free-flying hover machine in A11 carries).
- **T1b, hover (free body, no ground).** Released upright and still at 30 m
  and told to stay: for 6 s, height within 1 m, horizontal drift within 2 m,
  the body's axis within 45 deg of vertical throughout, on at least 7 of 8
  seeds. Beside it, every actuator held (it falls).
- **T1c, take-off.** The repo's own crossing probe, `land_to_air` through
  `transitions.run_transition` with `DYTISCIDAE_CROSSING_GATES=1`: 1.5 s on
  the ground told to stay, then told to go, and at the end of 6 s clear of
  the ground by more than 0.5 m and the right way up; from eight entry
  states (the repo's draw and seven more, as E0 made them): crossed on at
  least 6 of 8. Every actuator held, and the command frozen: 0 of 8 each.

Decision. All three: take-off is established and the water crossings are
next. T1a not met: the wing or its stroke is re-sized before anything else.
T1a met and T1b not: there is lift and no control at rest; what axis lacks
authority is reported and a body change (an active tail, the pairs further
apart) is measured under a new file. T1a and T1b met and T1c not: the
machine hovers and cannot get from lying on its thumbs to hovering; the
report says where it stops.

## 2 Method

Body B, neutral hull, ideal servos and unlimited pack, 1 ms step or less,
`bad_qacc` on every rollout (a failed rollout is failed, never 0). Laws
tuned by hand on seeds and entry states 100-103 only, every tuning rollout
appended to `trials_T1.jsonl`. No part added. The stroke may differ from
F4's; whatever it is, F4's flight must still be reachable from it (T1b's
end state handed to F4's law for 4 s on three seeds: it flies level by F4's
bar, or the report says it does not).

## 3 Predictions (written before running)

1. T1a is met at 130 deg of stroke. I hold this at 0.8.
2. T1b is met. I hold this at 0.5.
3. T1c is met. I hold this at 0.35: nothing here has stood up from the
   ground, and the wings' tips pass through the ground plane when the body
   lies level.
4. Null arms: 0 of 8 each in T1c; the held arm of T1b loses more than 20 m.
5. The hover asks a shoulder for more than F4's 2.0 N m.
6. If T1c is met, the machine leaves the ground by pushing off its thumbs
   and pitching up, not by a run along the beach.

## 4 Limits fixed now

Quasi-steady model without ground effect on the wings; rigid wings; still
air; the wake feedback is not on here (read afterwards at gains 1 and 3 if
the bars are met). Stopping rule: rows filled, or a row shown blocked with
its trace, or about four hours. A clean negative is a result; no rotors, no
part added without a new frozen file, no bar loosened.
