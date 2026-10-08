# Can identification spend fewer probes? (ROADMAP G, after M6)

M6 (2026-10-08) measured mobility identification at 57% of a batched Tier-1
evaluation's wall on arch48's 16 latest elites (130 s with, 56 s without).
`identify_batch` runs `n_probes = 24` probe pairs per domain. The cost is
linear in the probe count; the question is what fewer probes do to the
score a design gets with the bases they produce.

## Method

The same 16 elites, the network that scored the first, each elite's own
`eval_seed`. For `n_probes` in {24, 12, 8}: identify (air and water, as
`evaluate_tier1_batch` does, same seed) with `identify_batch(n_probes=...)`,
time it, then score every elite once, deterministically, with those bases and
its own stored policy (`identify_axes=False`). Read per elite: the summed
medium competence against the 24-probe score, and the identification wall.

## Pre-registered prediction and decision (written before any data)

- **P1.** At 12 probes the median |score − score at 24| over the 16 elites is
  below 0.005. The scale: C2's per-draw standard deviation of the summed
  medium competence on arch48's elites is 0.044, so 0.005 is about a ninth of
  one draw's noise.
- **Decision.** If P1 holds and no elite's score moves by more than 0.044 (one
  draw's SD), `n_probes` becomes 12 by default, saving about half of
  identification's wall. If it fails, 24 stays and G is closed for probes.

    PYTHONPATH=. .venv/bin/python experiments/identification_probes/run.py \
        --run runs/arch48 --out experiments/identification_probes/results_arch48.json
