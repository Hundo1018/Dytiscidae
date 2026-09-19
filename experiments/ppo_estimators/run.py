#!/usr/bin/env python3
"""Three scalars `ppo_update` computes about itself, and whether they measure it.

0. The problem
--------------
`learning/ppo.py` reports three quantities that a reader uses to decide whether
the shared policy is healthy, and one of them is used by the code itself:

  * `entropy`, with `ent_coef` weighting it into the loss -- the only mechanism
    that is supposed to stop the policy's exploration collapsing;
  * `kl`, which `target_kl` compares against to end the epoch loop early, and
    which the roadmap quotes ("arch33's policy hit the KL ceiling on 88% of
    updates by generation 450") as evidence about the learning rate;
  * the importance ratio, which PPO's whole derivation requires to be exactly
    1 on the first minibatch of the first epoch.

All three are computed from `policy.log_prob(obs, act)`, where `act` was drawn
from the *old* policy and is recovered through `atanh`.  That is the right
quantity for the surrogate and it is not obviously the right quantity for the
other three uses.

1. Existing knowledge and competing explanations
------------------------------------------------
  E-H1  the three are correct as written and this is a reading error;
  E-H2  `ent = -logp.mean()` is the entropy of the *current* policy, as its
        docstring says ("the one-sample estimator -- unbiased");
  E-H3  it is the cross entropy H(pi_old, pi_new), whose gradient at the
        on-policy point is zero by the score-function identity, so `ent_coef`
        buys nothing at any value.

  K-H1  `logp_old - logp` is a usable KL estimate at these step sizes;
  K-H2  it is the high-variance estimator that goes negative -- which a
        divergence cannot -- often enough to let an update through the bound.

  R-H1  `atanh(tanh(u))` recovers `u` well enough that the ratio is 1 to
        floating-point accuracy;
  R-H2  it does not, for actions near the edge of the squash.

2. Predictions that separate them
----------------------------------
  E-H2 predicts d(ent)/d(log_std) is positive and of the same size as the
  gradient of a real entropy estimator.  E-H3 predicts it is zero within
  sampling noise, and predicts further that sweeping `ent_coef` over two orders
  of magnitude leaves the learned `log_std` where it was.

  K-H2 predicts a measurable fraction of minibatches with a negative estimate,
  and a set of minibatches where the naive estimate is under `target_kl` while
  the standard k3 estimator is not.

  R-H2 predicts |ratio - 1| grows with how often `|a|` approaches 1, so it
  should be invisible at the initial `log_std = -0.5` and large at a `log_std`
  that saturates the squash.

3. Measurement
--------------
Synthetic observations, no environment: every claim above is a property of the
learner and mixing in a rollout would only add variance.  The entropy gradient
is compared against a reparameterised estimator of the true entropy, which is
the same quantity with the sample drawn from the current policy and a path for
the gradient to travel down.

Run:  PYTHONPATH=. .venv/bin/python experiments/ppo_estimators/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.harness import ExperimentResult, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Shared rollout: a policy acting on synthetic observations, banked as PPO
# trajectories.  Deliberately the same call path `run_search` uses.
# --------------------------------------------------------------------------


def rollout(policy, rng, n_steps: int, reward_fn, n_obs: int):
    from dytiscidae.learning.ppo import Trajectory

    t = Trajectory()
    for _ in range(n_steps):
        o = rng.normal(size=n_obs)
        a, lp, v = policy.act(o)
        t.obs.append(o)
        t.act.append(a)
        t.logp.append(lp)
        t.val.append(v)
        t.phi.append(0.0)
    t.terminal_reward = float(reward_fn(np.asarray(t.act)))
    return t


def fill(policy, rng, cfg, reward_fn, n_traj: int, n_steps: int):
    from dytiscidae.learning.ppo import RolloutBuffer

    b = RolloutBuffer()
    for _ in range(n_traj):
        b.add(rollout(policy, rng, n_steps, reward_fn, cfg["n_obs"]))
    return b


def _mode_zero(acts) -> float:
    """Reward with a known optimum: drive coefficient 0 positive."""
    return float(acts[:, 0].mean())


# --------------------------------------------------------------------------
# E: the entropy bonus
# --------------------------------------------------------------------------


def entropy_gradient(cfg: dict) -> dict:
    """d/d(log_std) of the term in the loss, against that of a real estimator.

    The term in the loss is ``-mean log pi_new(a)`` with ``a`` drawn from
    ``pi_old``.  At the on-policy point the two policies are the same one, and
    ``E_{a~pi}[grad log pi(a)] = grad integral pi = 0``.  So the prediction is
    not "small": it is zero, and what is measured is sampling noise around it.
    """
    import torch

    from dytiscidae.learning.ppo import SharedPolicy

    n_obs, n_modes = cfg["n_obs"], cfg["n_modes"]
    batch = cfg["gradient_probe"]["batch"]
    used, true = [], []
    for r in range(cfg["gradient_probe"]["repeats"]):
        torch.manual_seed(cfg["seed"] + r)
        rng = np.random.default_rng(cfg["seed"] + r)
        p = SharedPolicy(n_obs, n_modes, hidden=cfg["hidden"])
        obs = torch.as_tensor(rng.normal(size=(batch, n_obs)).astype(np.float32))

        # The term exactly as ``ppo_update`` forms it: actions from the old
        # policy, scored by the new one.
        with torch.no_grad():
            base = p.latent(obs)
            act = torch.tanh(base.sample())
        ent_used = -p.log_prob(obs, act).mean()
        used.append(float(torch.autograd.grad(ent_used, p.log_std)[0].mean()))

        # A real entropy estimator: the sample is reparameterised from the
        # current policy, so the gradient has a path to travel down.
        base = p.latent(obs)
        u = base.rsample()
        a = torch.tanh(u)
        lp = base.log_prob(u).sum(-1) - torch.log1p(-a.pow(2) + 1e-6).sum(-1)
        ent_true = -lp.mean()
        true.append(float(torch.autograd.grad(ent_true, p.log_std)[0].mean()))

    used, true = np.asarray(used), np.asarray(true)
    return {
        "n_repeats": int(used.size),
        "d_used_mean": float(used.mean()),
        "d_used_se": float(used.std(ddof=1) / np.sqrt(used.size)),
        "d_true_mean": float(true.mean()),
        "d_true_se": float(true.std(ddof=1) / np.sqrt(true.size)),
        "t_used": float(used.mean() / (used.std(ddof=1) / np.sqrt(used.size))),
        "t_true": float(true.mean() / (true.std(ddof=1) / np.sqrt(true.size))),
    }


def entropy_sweep(cfg: dict) -> dict:
    """Train under several ``ent_coef`` and report where ``log_std`` ends up.

    The gradient probe is the mechanism; this is the consequence, and it is the
    one that decides whether the finding matters.  Identical seeds, identical
    data, one coefficient changed.
    """
    import torch

    from dytiscidae.learning.ppo import SharedPolicy, ppo_update

    s = cfg["entropy_sweep"]
    out = {}
    for coef in s["coefficients"]:
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        # The learner's own stream, explicitly.  The first version of this file
        # left ``rng`` unset, so ``ppo_update`` shuffled with a fresh
        # ``default_rng()`` and the sweep moved by 0.0025 on ``log_std`` between
        # two runs of the same command -- a tenth of the effect being measured.
        # ``ppo_update`` reports ``deterministic: False`` for exactly that case;
        # it is asserted below rather than trusted.
        shuffler = np.random.default_rng(cfg["seed"] ^ 0x5EED5EED)
        p = SharedPolicy(cfg["n_obs"], cfg["n_modes"], hidden=cfg["hidden"])
        opt = torch.optim.Adam(p.parameters(), lr=s["lr"], eps=1e-5)
        for _ in range(s["updates"]):
            b = fill(p, rng, cfg, _mode_zero,
                     s["trajectories_per_update"], s["steps_per_trajectory"])
            info = ppo_update(p, b, epochs=s["epochs"], minibatch=s["minibatch"],
                              optimiser=opt, ent_coef=coef, rng=shuffler)
            if not info.get("deterministic"):
                raise RuntimeError(
                    "the sweep is running a non-deterministic update; its "
                    "numbers would not reproduce")
        ls = p.log_std.detach().numpy()
        out[str(coef)] = {"log_std": ls.tolist(),
                          "log_std_mean": float(ls.mean()),
                          "sigma_mean": float(np.exp(ls).mean())}
    base = out[str(s["coefficients"][0])]["log_std_mean"]
    out["spread_over_sweep"] = float(
        max(v["log_std_mean"] for k, v in out.items() if k != "spread_over_sweep")
        - min(v["log_std_mean"] for k, v in out.items() if k != "spread_over_sweep"))
    out["largest_minus_zero"] = float(
        out[str(s["coefficients"][-1])]["log_std_mean"] - base)
    return out


# --------------------------------------------------------------------------
# K: the KL the epoch loop stops on
# --------------------------------------------------------------------------


def kl_estimators(cfg: dict) -> dict:
    """The naive estimator against k3, minibatch by minibatch, on real updates.

    k3 is ``(r - 1) - log r`` with ``r = pi_new/pi_old``: non-negative by
    construction, unbiased for KL(old||new), and the estimator the PPO
    implementations that reproduce published results use.  The naive
    ``log pi_old - log pi_new`` is unbiased too and has far more variance --
    enough to be negative, which a divergence cannot be.
    """
    import torch

    from dytiscidae.learning.ppo import SharedPolicy

    k = cfg["kl_probe"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    p = SharedPolicy(cfg["n_obs"], cfg["n_modes"], hidden=cfg["hidden"])
    opt = torch.optim.Adam(p.parameters(), lr=cfg["entropy_sweep"]["lr"], eps=1e-5)

    naive, k3 = [], []
    for _ in range(k["updates"]):
        b = fill(p, rng, cfg, _mode_zero, 8, 64)
        obs, act, lo, adv, _ret, _val = b.build()
        obs_t = torch.as_tensor(obs)
        act_t = torch.as_tensor(act)
        lo_t = torch.as_tensor(lo)
        a_t = torch.as_tensor(adv.astype(np.float32))
        a_t = (a_t - a_t.mean()) / (a_t.std() + 1e-8)
        n = len(lo)
        idx = np.arange(n)
        for _ep in range(k["epochs"]):
            np.random.shuffle(idx)
            for s in range(0, n, k["minibatch"]):
                bi = torch.as_tensor(idx[s:s + k["minibatch"]].copy())
                lp = p.log_prob(obs_t[bi], act_t[bi])
                ratio = (lp - lo_t[bi]).exp()
                loss = -torch.min(ratio * a_t[bi],
                                  ratio.clamp(0.8, 1.2) * a_t[bi]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    logr = lp.detach() - lo_t[bi]
                    naive.append(float((-logr).mean()))
                    k3.append(float((logr.exp() - 1.0 - logr).mean()))

    naive, k3 = np.asarray(naive), np.asarray(k3)
    bound = k["target_kl"]
    return {
        "n_minibatches": int(naive.size),
        "naive_mean": float(naive.mean()),
        "naive_min": float(naive.min()),
        "naive_fraction_negative": float((naive < 0).mean()),
        "k3_mean": float(k3.mean()),
        "k3_min": float(k3.min()),
        "k3_fraction_negative": float((k3 < 0).mean()),
        "k3_over_naive": float(k3.mean() / naive.mean()) if naive.mean() else float("nan"),
        "target_kl": bound,
        "fraction_naive_under_bound_k3_over": float(
            ((naive < bound) & (k3 >= bound)).mean()),
    }


# --------------------------------------------------------------------------
# R: the importance ratio before any gradient step
# --------------------------------------------------------------------------


def ratio_identity(cfg: dict) -> dict:
    """|ratio - 1| on freshly collected data, at the operating squash and past it.

    The stored log-probability is computed from the pre-squash sample ``u``;
    ``log_prob`` recovers ``u`` with ``atanh`` of the clamped action.  Where
    ``|a|`` is not near 1 the round trip is exact to floating point; where it is,
    ``atanh`` has no accuracy left.
    """
    import torch

    from dytiscidae.learning.ppo import SharedPolicy

    r = cfg["ratio_probe"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    p = SharedPolicy(cfg["n_obs"], cfg["n_modes"], hidden=cfg["hidden"])

    def measure(tag: str) -> dict:
        b = fill(p, rng, cfg, _mode_zero, r["trajectories"], r["steps"])
        obs, act, lo, *_ = b.build()
        with torch.no_grad():
            lp = p.log_prob(torch.as_tensor(obs), torch.as_tensor(act))
        d = np.abs((lp - torch.as_tensor(lo)).exp().numpy() - 1.0)
        return {"tag": tag, "n": int(d.size), "max_abs_ratio_error": float(d.max()),
                "fraction_over_1e-3": float((d > 1e-3).mean()),
                "fraction_over_1e-2": float((d > 1e-2).mean()),
                "fraction_action_over_0.999": float(
                    (np.abs(act) > 0.999).mean())}

    at_operating = measure("log_std=-0.5 (initial)")
    with torch.no_grad():
        p.log_std.fill_(float(r["saturating_log_std"]))
    saturated = measure(f"log_std={r['saturating_log_std']}")
    return {"operating": at_operating, "saturated": saturated}


# --------------------------------------------------------------------------


def main() -> int:
    cfg = load_config(HERE / "config.json")
    res = ExperimentResult(name="ppo_estimators", config=cfg)

    try:
        import torch  # noqa: F401
    except Exception as exc:                                 # pragma: no cover
        print(f"SKIP: torch unavailable ({type(exc).__name__}: {exc})")
        res.ok = False
        res.record("skipped", f"{type(exc).__name__}: {exc}")
        res.write(HERE / "results")
        return 0

    print(__doc__.split("0. The problem")[0].strip())
    print("\n--- E: does the entropy bonus have a gradient? ---")
    g = entropy_gradient(cfg)
    res.record("entropy_gradient", g)
    print(f"  as used in the loss : {g['d_used_mean']:+.5f} "
          f"+/- {g['d_used_se']:.5f}   t = {g['t_used']:+.2f}")
    print(f"  a real estimator    : {g['d_true_mean']:+.5f} "
          f"+/- {g['d_true_se']:.5f}   t = {g['t_true']:+.2f}")

    print("\n--- E: and what does sweeping ent_coef do to log_std? ---")
    s = entropy_sweep(cfg)
    res.record("entropy_sweep", s)
    for coef in cfg["entropy_sweep"]["coefficients"]:
        v = s[str(coef)]
        print(f"  ent_coef={coef:<5} -> log_std mean {v['log_std_mean']:+.4f}  "
              f"sigma {v['sigma_mean']:.4f}")
    print(f"  largest coefficient minus zero: {s['largest_minus_zero']:+.4f}")

    print("\n--- K: the estimator target_kl stops on ---")
    k = kl_estimators(cfg)
    res.record("kl_estimators", k)
    print(f"  naive: mean {k['naive_mean']:+.6f}  min {k['naive_min']:+.6f}  "
          f"negative on {k['naive_fraction_negative']:.1%} of minibatches")
    print(f"  k3   : mean {k['k3_mean']:+.6f}  min {k['k3_min']:+.6f}  "
          f"negative on {k['k3_fraction_negative']:.1%}")
    print(f"  under the {k['target_kl']} bound by the naive estimate and not by k3: "
          f"{k['fraction_naive_under_bound_k3_over']:.1%}")

    print("\n--- R: is the ratio 1 before the first step? ---")
    r = ratio_identity(cfg)
    res.record("ratio_identity", r)
    for key in ("operating", "saturated"):
        v = r[key]
        print(f"  {v['tag']:<24} max|ratio-1| = {v['max_abs_ratio_error']:.3e}  "
              f"|a|>0.999 on {v['fraction_action_over_0.999']:.1%}")

    res.write(HERE / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
