# F3d — the water read again, on a neutral body with a paired depth command

Frozen 2026-11-12 before any rollout. No learning. Run by the lead.

F3c's rule failed on its null: the two-pair body held still was paid on 2 of
16 seeds by sinking through the commanded depth. Two things were changed
since, neither of them the stroke or its law:

- **Buoyancy.** Hull radius 0.0209 m (was 0.018): weight 1.03 N, buoyancy
  within 0.5% of it (bisected on the built model; the body was 16% heavy).
  Passed by `FLAPPER_BODY_B`, so the stored layouts are untouched.
- **The instrument.** `DYTISCIDAE_DEPTH_PAIR=1` (package, off by default;
  `tasks.depth_pair`): the second half of the water pair is told the depth on
  the other side of the release depth, and the hold is scaled by the measured
  share of that commanded difference. Test and three mutations in the package.

`swim.py` is as committed at ce249df: stroke S2x and its law are not
re-tuned for the new body or the new command.

## Rule

Seeds 24-39 (16 seeds, never used). The machine **swims** if S2x scores 0.15
or more on at least 14 with `bad_qacc` 0, every actuator held scores under
0.012 on 16 of 16, and the command-frozen arm scores under 0.012 on 16 of 16
(whole score now: a frozen command answers neither the heading nor the
depth).

## Predictions

1. All three clauses hold. I hold this at 0.65; the untuned law on a heavier,
   neutral body is what I trust least.
2. Still arm: under 1e-9 on every seed.
3. The swimmer's depth response (share of the commanded difference in depth
   it makes) is under 0.3 on most seeds: it was barely holding depth before,
   and it has never been asked to rise.
