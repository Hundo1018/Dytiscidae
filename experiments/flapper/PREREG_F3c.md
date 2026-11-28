# F3c — the diagonal stroke on seeds nobody has seen

Frozen 2026-11-12 before any rollout, commit ce249df. No learning. Run by
the lead with `swim.py` as the agent committed it (sha dbdab4e1...), no
setting changed.

F3b's three frozen strokes do not swim the free two-pair body (0 of 8 each),
and its rule sends the design to a fore-aft wrist. The agent also reported a
stroke outside the file, **S2x**: the fore-left and hind-right wings beat
against the other diagonal, at 1 Hz, with the shoulders the machine has. It
scored 0.35-0.52 on seeds 0-7. That was found after looking, on the seeds it
was tuned beside, so it is a hypothesis. This file tests it.

## Rule

S2x, unchanged, on seeds 8-23 (16 seeds, never used): water score 0.15 or
more on at least 14, `bad_qacc` 0. Null arms on the same seeds: every
actuator held, under 0.012 on 16 of 16; command frozen, **progress** part
under 0.012 on 16 of 16. The frozen arm's whole score is reported but not
gated: the water task carries one depth command for the whole segment, so
freezing it changes nothing and what remains is hold credit (the open item
of DIAGNOSIS P0.2, the depth command answerable one way only).

If it holds: the machine swims on its diagonals with the shoulders it has,
F3b's rule is superseded by this measurement and no wrist is added for the
water. If it does not: F3b's rule stands and the wrist is built.

## Predictions

1. It holds: 14 or more of 16. I hold this at 0.75.
2. The still arm reads under 1e-6 on every seed.
3. The frozen arm's progress part is 0.00 on every seed; its whole score is
   above 0.012 on at least 4 of 16.

Not settled here whatever happens: the body sinks (0.958 N against 0.801 N
of buoyancy) and depth is barely held (0.01-0.10 of the hold credit), and
the four reactions cancel in a model with no wake between the pairs.
