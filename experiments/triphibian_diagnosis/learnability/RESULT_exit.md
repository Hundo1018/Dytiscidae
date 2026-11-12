# L1 result — 2026-11-12

`PREREG_exit.md` (sha256 d523057c...) was frozen and committed first.
Reproduce: `PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python
experiments/triphibian_diagnosis/learnability/diagnose.py --jobs 8` (needs
L0's `data/`; rows in `results_exit.jsonl`, `summary_exit.json`).
`probe.fit` gained an optional `weight`; without it it is L0's fit.

Round-3 data: 274,247 rows, of which the exit window is a few percent and the
memory flag is on in under one percent (`summary_exit.json`, `info`).

## Missions completed of 32 per start (failed rollouts in brackets)

| arm | from air | from water | from land |
|---|---|---|---|
| S, the student | 22 (6) | 24 (3) | 32 (0) |
| M, memory flag from the teacher | 19 (6) | 24 (3) | 32 (0) |
| F, all four flags from the teacher | 19 (5) | 25 (3) | 32 (0) |
| C, continuous channels from the teacher | 30 (1) | 25 (3) | 32 (0) |
| X, teacher inside the exit window only | 32 (0) | 32 (0) | 32 (0) |
| W, refit with exit rows weighted 10x | 13 (5) | 26 (4) | 32 (0) |

S reproduces L0's round-3 row exactly (22, 24, 32).

Trace of S inside the exit window: the only flag that ever differs from the
teacher's is the memory flag (34 decisions of 3,918; the other three, 0).
Missions that missed had a flag mismatch in 9 of 18, missions that completed
in 16 of 46. The largest continuous error, in training standard deviations:
median 3.57 in missions that missed, 1.95 in those that completed.

## The five predictions

1. *"X completes at least 30 of 32 from every start: the loss is in the exit
   seconds."* **Held**: 32, 32, 32, and no failed rollout.
2. *"C restores more missions than F does."* **Held**: C +8 and +1, F -3 and
   +1.
3. *"M alone restores fewer than half of what X restores."* **Held**: M
   restores none (-3, 0).
4. *"W reaches 25 of 32 from every start."* **Refuted**: 13 from the air,
   nine fewer than unweighted.
5. *"... missions that miss have a flag mismatch in the exit window no more
   often than missions that complete ..., and a larger continuous error."*
   First half **not held as written** (0.50 against 0.35; with 18 and 46
   missions these are not separable). Second half **held**.

## What it says

- The whole loss is inside the exit window. Outside it the round-3 student
  is the teacher, as far as the mission can tell: X is 96 of 96.
- The flags are not the cause. Giving the student the teacher's flags does
  not help and from the air it hurts a little; three of the four flags never
  differ at all.
- The continuous demand is most of it from the air (22 -> 30) and not from
  the water (24 -> 25), where 3 rollouts still fail and neither swap alone
  repairs what the two together do. So it is the demand's precision during
  the exit, with an interaction this design does not resolve.
- Weighting the exit rows 10x is not a remedy as chosen: better from the
  water by 2, worse from the air by 9, and more memory-flag mismatches.

## Against the frozen decision table

It does not map onto one row. "More data at the exit" needed W to help, and
it did not. "A different output form for the flags" needed F to restore it,
and it did not. "The body or the exit itself" needed C to fail, and C
restores the air start. What is left standing is narrower: the exit asks for
a continuous demand more exact than this fit gives (the teacher waits for
3 degrees of tilt and 0.3 rad/s before it climbs), and a takeover confined
to those seconds is sufficient.

Limits: one student, one fit seed; validation seeds read a second time;
W is one weighting; the trace has no trajectories, only counts.
