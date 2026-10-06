# Tier gap: how often Tier-1 credit is not reproduced by Tier-2, per medium

Measured 2026-10-06 from `runs/arch45..arch48/events.jsonl` (no simulation; read-only).
Diagnostic only, after the "overconfident-failure rate" of arXiv 2610.02740
(`docs/papers/2610.02740.md`). Nothing here gates anything.

```bash
PYTHONPATH=. .venv/bin/python experiments/tier_gap/run.py    # ~6 s, writes results.json
```

`run.py`'s docstring defines everything below. In short, per medium, among promoted
designs whose Tier-1 competence could be recovered, the 2x2 `a/b/c/d` is
(Tier-1 >= bar, Tier-2 leg >= 0.15) as pass/pass, pass/fail, fail/pass, fail/fail.
**OFR** = b/(a+b) = P(Tier-2 fail | Tier-1 pass). **US** = c/(c+d) = P(Tier-2 pass |
Tier-1 fail). Brackets are one-sided 95% Clopper-Pearson (lower, upper)
(`dytiscidae.domain.evidence.clopper_pearson`).

## What the data allow

* **Tier-2 values** (`values`) exist only from arch47 (`tier2_media` on the promote
  event). With `tier2_label_all_media` every promotion has all three media (legs
  the chain never reached are probed from a fresh reset), so nothing "never ran"
  and the correlation is computable.
* **arch45 and arch46 have no `tier2_media`.** For all four runs a `chain` view is
  recovered from the promote `notes`: Tier-2 stops at the first failed leg, so the
  chain ran the failing leg and the cyclic predecessors (which passed), and
  nothing after. Pass/fail only; no correlation. Consistency check on arch47/48:
  a chain-failed medium reads >= 0.15 in `tier2_media` in 0 of 120 / 0 of 191.
* **Tier-1 per medium is not on the promote event** (this change adds
  `tier1_media`; arch45-48 predate it). It is joined from the evaluate event that
  placed the elite, key `(island, cell)`, map cleared at every `descriptor_refit`
  (cells are re-binned), kept only when the evaluate's `mission_fraction` equals
  the promote's `tier1_fraction`. The join is partial:

| run | promotions | joined | join rate | no candidate in cell | mismatch | refits | Tier-2 fraction > 0, joined | ... unjoined |
|---|---|---|---|---|---|---|---|---|
| arch45 | 147 | 61 | 0.415 | 79 | 7 | 34 | 3/61 [0.01, 0.12] | 1/86 [0.00, 0.05] |
| arch46 | 84 | 33 | 0.3929 | 49 | 2 | 19 | 2/33 [0.01, 0.18] | 6/51 [0.05, 0.22] |
| arch47 | 120 | 82 | 0.6833 | 38 | 0 | 9 | 3/82 [0.01, 0.09] | 2/38 [0.01, 0.16] |
| arch48 | 192 | 121 | 0.6302 | 71 | 0 | 7 | 5/121 [0.02, 0.08] | 4/71 [0.02, 0.12] |

  Joined and unjoined promotions reach a positive Tier-2 mission at similar rates,
  but the joined ones are by construction placed since the last refit.
* **Promotions are not a sample of the archive.** The curator picks them
  (`should_promote`), so OFR is conditional on being promoted. A Tier-1 pass that was
  never promoted is not in these tables.
* Tier-2 "pass" is competence >= 0.15 (`envs/evaluate.py:403`). The chain also
  fails a leg that did not survive; the `values` view ignores that (0 disagreements).

## Bars (Tier-1 pass)

| bar | where | meaning |
|---|---|---|
| 0.15 | `envs/evaluate.py:403` `LEG_COMPETENCE_BAR` | Tier-2's own leg bar |
| 0.012 | `evolution/curriculum.py:106` `WEAKEST_BARS[1]` | triphibian island, "operates in all three media"; counted by `three_media` (`evolution/loop.py:1881`); set from arch46's data, so **applies from arch47**; shown for arch45/46 only for comparison |
| 0.25 | `evolution/curriculum.py:67` `STAGES[0]` | other islands' "operate in one medium" bar |

Only 0.15 and 0.012 are tabled; 0.25 is in `results.json`.

## Results: Tier-2 values (arch47, arch48)

| run | medium | bar | a/b/c/d | OFR | US |
|---|---|---|---|---|---|
| arch47 | air | 0.15 | 0/6/0/76 | 6/6 [0.61, 1.00] | 0/76 [0.00, 0.04] |
| arch47 | air | 0.012 | 0/10/0/72 | 10/10 [0.74, 1.00] | 0/72 [0.00, 0.04] |
| arch47 | water | 0.15 | 0/33/4/45 | 33/33 [0.91, 1.00] | 4/49 [0.03, 0.18] |
| arch47 | water | 0.012 | 2/54/2/24 | 54/56 [0.89, 0.99] | 2/26 [0.01, 0.22] |
| arch47 | land | 0.15 | 4/24/3/51 | 24/28 [0.70, 0.95] | 3/54 [0.02, 0.14] |
| arch47 | land | 0.012 | 5/44/2/31 | 44/49 [0.80, 0.96] | 2/33 [0.01, 0.18] |
| arch48 | air | 0.15 | 0/1/0/120 | 1/1 [0.05, 1.00] | 0/120 [0.00, 0.02] |
| arch48 | air | 0.012 | 0/7/0/114 | 7/7 [0.65, 1.00] | 0/114 [0.00, 0.03] |
| arch48 | water | 0.15 | 1/35/3/82 | 35/36 [0.87, 1.00] | 3/85 [0.01, 0.09] |
| arch48 | water | 0.012 | 2/73/2/44 | 73/75 [0.92, 1.00] | 2/46 [0.01, 0.13] |
| arch48 | land | 0.15 | 5/30/5/81 | 30/35 [0.72, 0.94] | 5/86 [0.02, 0.12] |
| arch48 | land | 0.012 | 6/52/4/59 | 52/58 [0.81, 0.95] | 4/63 [0.02, 0.14] |

## Results: chain legs (all four runs; pass/fail only)

| run | medium | bar | a/b/c/d | OFR | US |
|---|---|---|---|---|---|
| arch45 | air | 0.15 | 0/1/0/23 | 1/1 [0.05, 1.00] | 0/23 [0.00, 0.12] |
| arch45 | air | 0.012 | 0/2/0/22 | 2/2 [0.22, 1.00] | 0/22 [0.00, 0.13] |
| arch45 | water | 0.15 | 0/4/1/12 | 4/4 [0.47, 1.00] | 1/13 [0.00, 0.32] |
| arch45 | water | 0.012 | 0/7/1/9 | 7/7 [0.65, 1.00] | 1/10 [0.01, 0.39] |
| arch45 | land | 0.15 | 1/4/1/17 | 4/5 [0.34, 0.99] | 1/18 [0.00, 0.24] |
| arch45 | land | 0.012 | 1/9/1/12 | 9/10 [0.61, 0.99] | 1/13 [0.00, 0.32] |
| arch46 | air | 0.15 | 0/2/0/9 | 2/2 [0.22, 1.00] | 0/9 [0.00, 0.28] |
| arch46 | air | 0.012 | 0/4/0/7 | 4/4 [0.47, 1.00] | 0/7 [0.00, 0.35] |
| arch46 | water | 0.15 | 0/2/1/7 | 2/2 [0.22, 1.00] | 1/8 [0.01, 0.47] |
| arch46 | water | 0.012 | 0/5/1/4 | 5/5 [0.55, 1.00] | 1/5 [0.01, 0.66] |
| arch46 | land | 0.15 | 0/3/1/10 | 3/3 [0.37, 1.00] | 1/11 [0.00, 0.36] |
| arch46 | land | 0.012 | 1/8/0/5 | 8/9 [0.57, 0.99] | 0/5 [0.00, 0.45] |
| arch47 | air | 0.15 | 0/3/0/27 | 3/3 [0.37, 1.00] | 0/27 [0.00, 0.10] |
| arch47 | air | 0.012 | 0/4/0/26 | 4/4 [0.47, 1.00] | 0/26 [0.00, 0.11] |
| arch47 | water | 0.15 | 0/11/2/15 | 11/11 [0.76, 1.00] | 2/17 [0.02, 0.33] |
| arch47 | water | 0.012 | 1/19/1/7 | 19/20 [0.78, 1.00] | 1/8 [0.01, 0.47] |
| arch47 | land | 0.15 | 0/9/1/17 | 9/9 [0.72, 1.00] | 1/18 [0.00, 0.24] |
| arch47 | land | 0.012 | 0/17/1/9 | 17/17 [0.84, 1.00] | 1/10 [0.01, 0.39] |
| arch48 | air | 0.15 | 0/1/0/56 | 1/1 [0.05, 1.00] | 0/56 [0.00, 0.05] |
| arch48 | air | 0.012 | 0/3/0/54 | 3/3 [0.37, 1.00] | 0/54 [0.00, 0.05] |
| arch48 | water | 0.15 | 0/12/0/19 | 12/12 [0.78, 1.00] | 0/19 [0.00, 0.15] |
| arch48 | water | 0.012 | 0/18/0/13 | 18/18 [0.85, 1.00] | 0/13 [0.00, 0.21] |
| arch48 | land | 0.15 | 2/8/2/24 | 8/10 [0.49, 0.96] | 2/26 [0.01, 0.22] |
| arch48 | land | 0.012 | 3/14/1/18 | 14/17 [0.60, 0.95] | 1/19 [0.00, 0.23] |

## Correlation, share of promotions where a leg never ran, medians

`values`: Tier-1 vs Tier-2 value over joined promotions. `chain`: share of *all*
promotions whose chain never reached the medium (denominator is every promotion,
not only the joined ones); no values, so no correlation.

| run | medium | ran & joined | never ran (all promotions) | Spearman | Pearson | Tier-1 median | Tier-2 median | Tier-2 max |
|---|---|---|---|---|---|---|---|---|
| arch45 (chain) | air | 24 | 87/147 [0.52, 0.66] | - | - | 0.0 | - | - |
| arch45 (chain) | water | 17 | 104/147 [0.64, 0.77] | - | - | 0.0 | - | - |
| arch45 (chain) | land | 23 | 99/147 [0.60, 0.74] | - | - | 0.001 | - | - |
| arch46 (chain) | air | 11 | 53/84 [0.54, 0.72] | - | - | 0.0 | - | - |
| arch46 (chain) | water | 10 | 51/84 [0.51, 0.70] | - | - | 0.028 | - | - |
| arch46 (chain) | land | 14 | 56/84 [0.57, 0.75] | - | - | 0.0645 | - | - |
| arch47 (values) | air | 82 | 0/120 [0.00, 0.02] | 0.4789 | 0.1746 | 0.0 | 0.0 | 0.0114 |
| arch47 (values) | water | 82 | 0/120 [0.00, 0.02] | -0.0049 | -0.11 | 0.071 | 0.0 | 0.2865 |
| arch47 (values) | land | 82 | 0/120 [0.00, 0.02] | 0.1167 | 0.2577 | 0.0645 | 0.0002 | 0.5504 |
| arch47 (chain) | air | 30 | 77/120 [0.56, 0.71] | - | - | 0.0 | - | - |
| arch47 (chain) | water | 28 | 74/120 [0.54, 0.69] | - | - | 0.059 | - | - |
| arch47 (chain) | land | 27 | 84/120 [0.62, 0.77] | - | - | 0.063 | - | - |
| arch48 (values) | air | 121 | 0/192 [0.00, 0.02] | - | - | 0.0 | 0.0 | 0.0 |
| arch48 (values) | water | 121 | 0/192 [0.00, 0.02] | -0.0943 | -0.102 | 0.038 | 0.0 | 0.2529 |
| arch48 (values) | land | 121 | 0/192 [0.00, 0.02] | 0.019 | 0.1304 | 0.01 | 0.0 | 0.5251 |
| arch48 (chain) | air | 57 | 108/192 [0.50, 0.62] | - | - | 0.0 | - | - |
| arch48 (chain) | water | 31 | 139/192 [0.67, 0.78] | - | - | 0.041 | - | - |
| arch48 (chain) | land | 36 | 130/192 [0.62, 0.73] | - | - | 0.0085 | - | - |

(arch48 air: Tier-2 air is exactly 0 for all 121, so its correlation is undefined,
not zero.)

## Reading

* At the 0.15 bar Tier-1 passes are rare and almost none are reproduced: in water
  Tier-2 failed 33/33 (arch47) and 35/36 (arch48) of Tier-1 passes; land 24/28 and
  30/35; air 6/6 and 1/1 (air Tier-2 never exceeded 0.012). Lower bounds on OFR are
  0.91 and 0.87 (water), 0.70 and 0.72 (land).
* US is small (upper bounds 0.04-0.18) but not zero in water and land: Tier-2 does
  sometimes credit what Tier-1 did not.
* Tier-1 vs Tier-2 correlation is weak and not stable in sign (Spearman water
  -0.005 / -0.094, land 0.117 / 0.019; air 0.479 on 82 points that are almost all
  zero in Tier-2).
* The chain view reproduces the direction of the `values` view at every bar and in
  all four runs, but with n of 1-20 per cell its bounds are wide.
* These are Tier-1 and Tier-2 competences at the *same* scorer, but not the same
  episode: Tier-2's leg is a fresh reset, unplaced, with a different seed, and
  CLAUDE.md records that the two evaluation paths still differ in water. A large
  OFR here is a statement about the gap between the two, not proof that Tier-1 is
  the wrong one.
