# L0 — can a feed-forward network learn the teacher? A probe, not one of the three runs

Frozen 2026-11-12 before any fit, commit e01e90e, MuJoCo 3.15. The user
lifted the no-training rule on 2026-11-12. This is a probe of about 3% of
run 1's size (about 3e5 decisions against 1e7), declared to the user as not
one of the three runs. It selects nothing, launches no search, and its
network is thrown away.

## 0 Question and decision

Everything measured so far shows the teacher works. Nothing shows a network
can stand in for it. Does a feed-forward network fitted to the teacher's
`(o, u)` pairs complete the mission in closed loop, and if plain imitation
does not, do a few rounds of relabelling the student's own states
(DAgger) close it?

Decision. If the student reaches the bar within three DAgger rounds: run 1's
design stands (DAgger, then PPO with a teacher pull), and the round at which
it got there sizes run 1's first phase. If it does not: where it fails (which
leg, which action channel, the memory bit or not) is the next design
question, and run 1 is not launched on the present design.

## 1 Rivals

- **Learnable** as it stands.
- **The memory bit**: a regressed flag that the student itself feeds back
  drifts, and the water exit never leaves its staged phase or leaves it early.
- **Discontinuity**: the law switches mode on thresholds; a smooth network
  blurs them and the closed loop chatters at every boundary.
- **Compounding error**: the fit is good on the teacher's states and useless
  one step off them (what DAgger is for).
- **Instrument**: the student passes because the inner loop carries it, i.e.
  a network that ignores its input would pass too. Checked by arm Z.

## 2 Method

- Body `ANCHOR_PLANS["legged_quad"]`, `SplitController`, the mission verdict
  `mission.evaluate_mission` (legs 7-13 s), a fresh environment per mission.
- Data: every decision's `(o, teach(o))`. Round 0: the teacher flies 96
  missions (training seeds 1000-1031, three starts). Round r >= 1: the
  round r-1 student flies the same 96 seeds shifted by 1000 r, the teacher
  labels every state it visits, and the data are aggregated.
- Student: inputs standardised on round 0's data (channels with zero spread
  left as they are), two hidden layers of 128, tanh. Outputs: the six
  continuous channels (acceleration 3, tilt limit, limb drive, limb turn)
  by squared error on standardised targets; the four flags (thrust below
  the horizon, rotors on, limb pose, memory) by cross-entropy, cut at 0.5
  when it acts. Adam, 60 epochs, batch 512, seed 0, refitted from scratch
  each round on all data so far. No tuning on the validation seeds: these
  settings are fixed here and a change is a new file.
- Read: validation seeds 200-231, three starts, the student of each round.
  Seeds 0-31 are not touched (run 1's test seeds will be fresh ones too).
- Arm Z: the round-0 student with its input replaced by the training mean
  (a network that ignores what it sees), same 96 validation missions.
- Controls: the teacher on the same validation seeds.

## 3 Predictions (written before running)

1. The teacher completes at least 30 of 32 from each start on seeds 200-231.
2. Round 0 (plain imitation) is under the bar (25 of 32) from at least one
   start. I hold this at 0.6.
3. Some round r <= 3 reaches 25 of 32 from every start. I hold this at 0.5.
4. Where round 0 fails it fails mostly on the leg that leaves the water (the
   memory bit and the staged exit), not in the air.
5. Arm Z completes 0 of 96 and makes at most 2 transitions.
6. Held-out fit error does not predict closed-loop success: round 0's flag
   accuracy on held-out teacher states is above 0.98 whatever its missions do.

## 4 Stopping rule and size

Three DAgger rounds at most, 96 missions each, then stop whatever the
result. About 4 x 96 x 800 = 3e5 decisions of collection and 5 x 96
validation missions. Failed missions (`bad_qacc`, divergence, flat pack) are
failures, counted against the student, and their decisions up to the failure
are kept as data.
