# How much of a Tier-1 score is the draw? (C2)

ROADMAP M2 (2026-10-08), second half. The first half is built: each candidate
is now scored on its own task draw (`SearchConfig.draw_per_candidate`), so 16
candidates no longer compete on one heading. That removes the best-of-16-per-
draw selection. It does not make one draw a good estimate of a design: the
archive still keeps whichever score a cell's best challenger happened to get on
its own draw. This experiment measures how large the draw's share of a score
is, which decides whether a design needs more than one draw to be placed
(sequential evaluation, 2 → 4 → 8 draws, a lower confidence bound to place).

M1 (`experiments/refine_criterion`) gave a first, two-draw read on the same
elites: draw variance / design variance 2.16 (air), 1.01 (water), 1.04 (land),
and elites lost 22-70% of their medium scores from their recorded draw to a
fresh one.

## Method

arch48's merged archive (200 elites), current code (antipodal pair). Each
elite is scored through the search's batched path
(`batchroll.evaluate_tier1_batch`) with the shared network that scored it, its
own stored policy and mobility bases, no identification, and no refinement, at
`1 + K` draws, `K = 6`:

- `s`: its recorded `eval_seed` (the draw it was selected on);
- `f1 … f6`: fresh seeds derived from `(s, elite index, j)`, used only here.

Each machine in a batch carries its own seed (the M2 change), so one call
scores an elite at all seven draws.

Read for each medium's Tier-1 competence (air, water, land) and for the two
scores placement reads (`island`, `stage`, as in `refine_criterion`):

- `draw_var` = mean over elites of the sample variance across the six fresh
  draws;
- `design_var` = variance of the elites' fresh means minus `draw_var / K`
  (floored at 0);
- `R = draw_var / design_var`, with a 95% bootstrap interval over elites;
- `icc1 = design_var / (design_var + draw_var)`, the reliability of one draw;
  `k80 = 4 R`, the draws a mean needs to reach reliability 0.8;
- `rank`: Spearman between fresh draw `f1` and the mean of `f2 … f6`;
- `curse`: `1 - mean(fresh mean) / mean(score at s)`.

A medium in which fewer than 20 of the 200 elites score above zero on any fresh
draw is reported as underpowered and gets no verdict.

This population is the archive, i.e. designs already selected, which narrows
the between-design spread relative to the candidates selection actually ranks.
So `R` here is biased upward, and a verdict of "draw dominates" is weaker
evidence than one of "design dominates".

## Pre-registered predictions (written before any data)

- **P1.** In water and land, `R >= 1` (one draw's reliability at most 0.5). M1
  read 1.01 and 1.04 from two draws.
- **P2.** In water and land, `curse >= 0.20`. M1 read 0.29 and 0.22.
- **P3.** In water and land, `rank < 0.7`.
- **P4.** For `stage`, the score the curriculum half of placement reads,
  `R >= 1`.

## Decision rule (also written before the data)

Per medium and per placement score:

- `R`'s interval entirely above 1: the draw dominates. Sequential evaluation
  earns a build (M3 territory): place on a mean of `ceil(k80)` draws or an LCB
  over them, with incumbents re-scored when challenged.
- `R`'s interval entirely below 1: the design dominates. One draw per
  candidate (built) is enough; no sequential evaluation.
- Otherwise: undecided; report `k80` and build nothing.

## Run

    systemd-run --user --unit draw-variance -p MemoryMax=3500M --same-dir \
        env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/draw_variance/run.py \
        --run runs/arch48 --out experiments/draw_variance/results_arch48.json

    python experiments/draw_variance/run.py --selftest     # no simulator
