# W2 — does the flight verdict survive a fore-to-hind wake?

Frozen 2026-11-12 before any rollout, while W1 (water) was still running and
unread. No learning. Same instrument and same reason as `PREREG_W1.md`.

F4 (commit 93527fb, re-run by the lead on five rows, every field equal): the
free two-pair body holds height and speed on 8 of 8 launches and turns to
the side it is told with an air score of 0.907-0.983 on 16 of 16, at 12 Hz
on the diagonal. In that flight the pairs are 1.26 chords apart, the fore
wake reaches the hind pair 0.13 of a beat later, and the reduced frequency
is 0.32. Tandem wings in the air are where a missing wake matters most.

## Method

F4's rows again, `fly.py` unchanged (sha e0389366...), neutral hull, with
`FLAPPER_WAKE_GAIN` 1 and 3 (gain 1 is a quarter to a third of Wagner's
effect, gain 3 about its size; `wake_check.py`).

- Level bar, seeds 0-7, arm fly.
- Turn bar, seeds 0-15, arm fly; arm frozen on seeds 0-3 at gain 3.

## Verdict rule

As W1: the flight verdict is robust to the wake if both of F4's bars are
still met at gain 1 and at gain 3 with the law untouched. A bar met at 1 and
lost at 3 holds up to a third of a Wagner-sized interaction and not a full
one; the law is then re-tuned under the wake before the air is called done.

## Predictions (written before any rollout)

1. Gain 1: level bar met (7 of 8 or more). I hold this at 0.7.
2. Gain 1: turn bar met (0.15 or more on 14 of 16). I hold this at 0.7.
3. Gain 3: level bar met. I hold this at 0.4: F4's level margin is thin (one
   seed inside the height bar by 2 cm, only 12 Hz holds the speed).
4. Gain 3: turn bar met. I hold this at 0.5.
5. No `bad_qacc` at either gain.
6. The frozen arm at gain 3 is under 0.012 on 4 of 4.
