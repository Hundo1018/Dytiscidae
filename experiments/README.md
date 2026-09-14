# experiments/

Every number this project quotes should be reproducible by running something.
A number that lives only in a source comment is a story about one run that
happened once.

```
PYTHONPATH=. python experiments/<name>/run.py
```

Each directory holds `config.json` (the inputs), `run.py` (the experiment),
`results/result.json` (what the last run wrote, committed so a claim has a
receipt) and `README.md` (the question, the design, and the answer).

`run.py` never invents a number: anything it prints it computed in the same
process. `results/result.json` carries the commit, the seed, the numpy version
and the wall time alongside the data.

## The experiments

| name | question | answer as measured |
|---|---|---|
| `dimensional_check` | does every formula produce the dimension its name claims? | 48 of 48 sound; 11 carry a literal with an undeclared unit |
| `analytic_vs_numerical` | where a quantity has a closed form, does the code match an independent computation of it? | 18 of 20 agree; hull buckling is 4.8x non-conservative, the lift slope is 11% below the value its docstring names |
| `added_mass` | does the added mass written into `body_mass` reach the dynamics, and does it know which way the body points? | a jointless machine gets none of it; a wing strip is isotropic, overstating edgewise entrained mass by `(c/t)^2` |
| `rank_threshold` | is `0.08 * sigma_0` a numerical rank or a guess? | a guess: the identification's own noise floor is `1.13 sigma_0`, and the rank the threshold reports is unanimous across four probe seeds on only 47.6% of bodies |
| `damping_lambda` | is `lam = 0.01 trace(G)/r` the right ridge? | no: out-of-sample error is minimised at 30x that value, and the incumbent is 18.3% worse |

## Shared modules

* `units.py` — a dimensional-analysis engine over (m, kg, s).
* `harness.py` — provenance, result writing, bootstrap CIs, principal angles.
* `mobility_data.py` — collects the CPG probes once (84 sets, ~22 min) and
  caches them, so the rank and damping sweeps analyse **identical** data and a
  difference between them is a difference in the analysis.

## Two rules that make these worth running

**A restatement must be tied to its source.** `dimensional_check` carries the
exact source fragment each of its checks is a restatement of, and fails the run
if that fragment is no longer in the file. A check cannot outlive the line it
describes.

**A finding is not fixed by being written down.** Findings that are still
present are re-measured on every run of `tests/test_math.py`, which prints them
under `[gap ]` with their `docs/MATH_AUDIT.md` identifier. The suite still
passes; the open findings cannot quietly stop being true.
