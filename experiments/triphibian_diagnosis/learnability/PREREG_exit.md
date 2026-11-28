# L1 — where does the student lose the exit from the water? A diagnosis, not a run

Frozen 2026-11-12 before any rollout, commit 482d8d1. Follows `RESULT.md`
(L0): the round-3 student completes 22, 24, 32 of 32 and every judged miss
is the land leg after the water. No new data are collected for fitting
except in arm W; the round-3 weights and data of L0 (`data/weights3.npz`,
`data/round3.npz`) are used as they are. About 7 x 96 missions and one refit.

## 0 Question and decision

Which part of the action loses the exit: the memory flag, the other flags,
or the continuous demand; and is the loss confined to the seconds of the
exit? The answer picks one of three designs for run 1's first phase: more
data at the exit (if a takeover there is enough and weighting helps), a
different output form for the flags (if a flag swap is what restores it), or
a change to the body or the exit itself (if even the teacher's continuous
demand through the student's flags does not restore it).

## 1 Arms, all on validation seeds 200-231, three starts, round-3 student

"Exit window": the commanded medium is not water and the body is wet (any
rotor's wetness above 0.01, or the depth channel positive).

- S: the student (L0's round-3 row, re-run with traces).
- M: the student, its memory flag replaced by the teacher's at every decision.
- F: the student, all four flags replaced by the teacher's.
- C: the student's flags, the six continuous channels replaced by the teacher's.
- X: the teacher's whole action inside the exit window, the student's outside.
- W: a refit of the same network on `round3.npz` with every exit-window row
  weighted 10x in the loss (nothing else changed, fit seed 0), then flown as S.

Trace, arm S: for every decision in an exit window, which flags differ from
the teacher's and the largest continuous difference in units of the training
standard deviation; summarised over missions that completed and that did not.

## 2 Predictions (written before running)

1. X completes at least 30 of 32 from every start: the loss is in the exit
   seconds. I hold this at 0.7.
2. C restores more missions than F does (the continuous demand, not the
   flags, is what loses it). I hold this at 0.55.
3. M alone restores fewer than half of what X restores.
4. W reaches 25 of 32 from every start. I hold this at 0.4.
5. In the trace, missions that miss have a flag mismatch in the exit window
   no more often than missions that complete (the flags are not the cause),
   and a larger continuous error. This is prediction 2 seen another way and
   can fail independently.

## 3 Limits fixed now

One student (L0 round 3, one fit seed). The validation seeds have now been
read by L0 and are read again here: nothing from L1 is a held-out result,
and run 1's seeds will be fresh. W is one pre-chosen weighting, not a sweep.
