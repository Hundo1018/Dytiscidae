# The entropy bonus was wired to nothing, the KL could come out negative, and the ratio was fine

```
PYTHONPATH=. python experiments/ppo_estimators/run.py
```

Synthetic observations, no MuJoCo, no fluid solver, no GPU. Every number below
is a property of `learning/ppo.py` alone, so this reproduces on a machine with
none of the accelerators the search needs.

Three of the quantities `ppo_update` computes about itself were checked against
an independent construction of what they are named after. Two did not measure
it. Both are now fixed in `learning/ppo.py`; this file keeps the measurement
that says why, and the runner re-derives both sides on demand — the two probes
build the expressions themselves rather than calling the code under test, so
they keep working as the comparison after the fix.

## E. The entropy bonus

`ppo_update` weighted `ent = -logp.mean()` into the loss with `ent_coef`, where
`logp = policy.log_prob(obs[b], act[b])` and `act` came from the rollout — that
is, drawn from **π_old**. That expression is the cross entropy H(π_old, π_new),
not the entropy of π_new, and at the on-policy point where every PPO update
starts they are the same policy, so

    E_{a~π}[∇ log π(a)] = ∇ ∫ π(a) da = ∇ 1 = 0.

The expected gradient is not small. It is zero.

**d/d(log_std), 32 independently initialised policies, batch 4096:**

| | mean ± SE | t |
|---|---|---|
| the term as it was used | **−0.00013 ± 0.00188** | −0.07 |
| a reparameterised estimator of the true entropy | **+0.43114 ± 0.00106** | +408.53 |

**And the consequence.** Identical seeds, identical data, 30 updates, one
coefficient changed:

| `ent_coef` | learned mean `log_std`, before | after |
|---:|---|---|
| 0.0 | −0.5117 | −0.4943 |
| 0.01 *(the default)* | −0.5120 | −0.4830 |
| 0.1 | −0.5138 | −0.3895 |
| 1.0 | −0.5180 | −0.2817 |
| **1.0 minus 0.0** | **−0.0063** | **+0.2126** |

**Read down a column, not across one.** Within a column the seeds, the data and
the minibatch order are identical and only the coefficient moves, so the last
row is a controlled comparison. Between columns the whole random realisation
differs, for the reason below, so the absolute values are not.

A hundred times the default coefficient moved exploration by −0.006 on
`log_std` — and *downward*. It was a knob wired to nothing, and the `entropy`
field in the `ppo` telemetry line, which a reader would use to diagnose a
collapse, was a cross entropy under a name that made it look like a collapse
detector.

The fix is `SharedPolicy.entropy`: `-log π(a)` with `a` drawn from **this**
policy by `rsample`, so the gradient has a path down to `log_std` and the trunk.
`log_prob_and_entropy` returns both from one forward pass, because the actor
trunk is the expensive part and they are both functions of the same `latent`.

**This changes what `shared_ent_coef` does.** Runs before and after are not
comparable on exploration, and the coefficient is now worth re-tuning rather
than inheriting: at the default 0.01 the learned `log_std` now sits **+0.0113**
above the same run with the bonus switched off, and at 0.1 it sits **+0.1048**
above — where before those differences were −0.0003 and −0.0021.

The `ent_coef = 0` row moves too, even though the term is multiplied by zero
there. That is the estimator's `rsample` drawing from the same torch stream
every minibatch, so the whole random realisation shifts. It is a one-off at this
boundary, not a second effect — and it is why the paragraph above the table says
to read down a column.

## K. The KL the epoch loop stops on

`target_kl` ended the epoch loop on `(logp_old - logp).mean()`. That is unbiased
for KL(old‖new) and has enough variance to come out negative, which a divergence
cannot. Measured over 320 minibatches of real updates:

| estimator | mean | min | fraction negative |
|---|---|---|---|
| `log π_old − log π_new` | +0.007473 | **−0.007173** | **20.0%** |
| k3, `(r − 1) − log r` | +0.007861 | +0.000000 | 0.0% |

On **6.9%** of minibatches the naive estimate put the update inside the 0.015
bound while k3 put it at or over. `ppo_update` now reports and stops on k3.

The roadmap quotes this number as evidence about the learning rate — "arch33's
policy hit the KL ceiling on 88% of updates by generation 450". That reading is
of the old estimator and is ~5% low on average, so it is if anything an
understatement; no conclusion in the roadmap reverses.

## R. The importance ratio before the first gradient step

PPO's derivation requires `π_new/π_old = 1` on the first minibatch of the first
epoch. The stored log-probability is computed from the pre-squash sample `u`;
`log_prob` recovers it with `atanh` of the clamped action, which has no accuracy
left where `|a|` approaches 1.

| `log_std` | max \|ratio − 1\| | actions past \|a\| = 0.999 |
|---|---|---|
| −0.5 (the initial value) | **2.861e−06** | 0.0% |
| 2.0 | 1.071e+02 | 60.4% |

**Not a defect at operating conditions**, and the boundary is what is worth
recording: nothing clamps `log_std`, and the entropy fix above now pushes it
*up*. `tests/test_ppo.py::test_the_importance_ratio_starts_at_one` asserts both
halves, so a future change that lets the squash saturate is caught here rather
than in a 21-hour run.
