# Roadmap

Written 2026-09-03, after arch33. Every item names the measurement that
motivates it; nothing here is on the list because it seemed like a good idea.

Read [CPU_LEGACY.md](CPU_LEGACY.md) for the older backlog. This file supersedes
it wherever the two disagree.

---

## Where arch33 left things

arch33 (900 generations, 40.4 h, 13,353 evaluations) changed nine things and
eight of them worked.

| | arch31 | arch33 |
|---|---|---|
| corr(fitness, mission_fraction) | +0.159 | **+0.590** |
| corr(fitness, *min* competence) | +0.003 | **+0.404** |
| corr(fitness, stage) | −0.350 | −0.038 |
| lineage depth mean / max | 4.59 / 16 | **10.28 / 25** |
| curriculum typical / reached | 1 / 3 | **2 / 4** |
| population mean mission_fraction, first 100 → last 100 gens | 0.02304 → 0.01601 | 0.01856 → **0.02745** |
| that change | **−0.00703 ± 0.00063 (t = −11.1)** | **+0.00890 ± 0.00079 (t = +11.3)** |
| energy-infeasible share, first → last | 37.0% → 65.9% | 39.7% → **31.8%** |
| n_parts, first → last | 3.95 → 5.66 | 3.96 → **3.95** |

Restricted to designs that have a lifting surface at all, the mission gain is
**+0.00436 ± 0.00087 (t = +5.0)** — half the headline, still significant. The
other half is the wingless class being cheap on energy, which is what Phase 1
is about.

The ninth change failed. Giving the shared policy morphology in its observation,
a potential-based shaped reward and per-segment reward scaling did **not** make
it useful:

| bodies | baseline | with policy | delta ± SE | t | improved |
|---|---|---|---|---|---|
| 48 from arch33 (in-distribution) | 0.02614 | 0.02473 | −0.00140 ± 0.00155 | −0.90 | 13/48 |
| 48 from arch30+arch31 (held out) | 0.01383 | 0.01115 | −0.00268 ± 0.00193 | −1.39 | 17/48 |

Four architectures have now produced four indistinguishable-from-zero results.
Do not build a fifth.

---

## Phase 0 — measure, no code

- Re-measure the fluid solver's fixed/variable cost split on an idle machine.
  The standing figure is `0.86 us/panel + 233 us fixed per call`, which would
  make dispatch overhead the majority of the cost for a small machine.
- Re-run the 12 wingless designs recorded as *climbing* (`sink_rate < −0.5`) and
  confirm whether the spin is producing the lift.

## Phase 1 — fidelity. This is the binding constraint

**1.1 The air score does not measure flight.** Winged designs score 0.144 and
wingless 0.136; both fall at ~11 m/s median. Four separate causes, all fixable:

- **The launch is free and inversely earned.** `launch_speed` is measured per
  design and assigned by the environment; a machine with no trim speed anywhere
  is launched at the 30 m/s cap. The worse the design, the bigger the gift.
  Give every design the same launch, or require it to reach speed itself.
- **`air = airborne_fraction * (0.55*flight + 0.25 + 0.2*speed)`** pays 0.25 for
  merely being off the ground and 0.2 for the velocity it was given. Shrink or
  remove both.
- **A tumble is not distinguished from a commanded turn.** `turn_rate_held` is
  correctly gated on `sink < 0.5`, but the exploit champion logged 31.4 rad/s.
  Score the correlation between commanded twist and achieved attitude change.
- **Nothing measures station-keeping or hover.** `holds_height` is satisfied by
  a fast shallow glide.

**1.2 Tier-1.5: a 60 s single leg on promotion candidates only.** A Tier-1
segment is 8 s against a 300 s mission leg — 2.7% — so nothing that accumulates
is visible. Over 180 promotions, corr(tier1_fraction, tier2_fraction) = +0.077.
Cost scales with promotions, not population. Corroborating: `wa889_14254`, the
best legitimate design by mission_fraction, never submerges once on the
continuous mission despite being the water island's champion.

**1.3 Schedule aliasing.** The island rotates once per generation and there are
six, so any `gen % N == 0` with `gcd(N, 6) > 1` fires on a subset. Every audit
in arch30, arch31 and arch33 landed on `air`; Tier-2 reaches only `air` and
`amphibian`. Four islands have never had a design verified at full fidelity in
any run. Drive both off per-island counters.

**1.4 Tier-0 gates for the exploit family.** arch33's mission champion is 12 kg
with `wing_area 0.0000`, wing loading 1,178,337 N/m², zero actuated DOF, and is
recorded climbing at 2.08 m/s while spinning at 31.4 rad/s. Gate on
no-lifting-surface, extreme wing loading, and a spin-rate ceiling. Note the
exploit flag needs `mission_fraction > 0.35` to disqualify; this scored 0.18.

**1.5 Telemetry defect.** `meta["body_plan"]` mixes body-plan names with
mutation-operator names, so it cannot be used for diversity claims.

## Phase 2 — throughput

- **Multiprocess actors.** 16 machines are stepped in lockstep inside one
  process; 20 cores are idle. Near-linear, certain, and it makes every later
  experiment cheaper. Measured: 190.5 us of wall per machine-step, i.e. 1.31x
  real time per machine and 21x only because 16 run at once.
- Then re-cost longer segments. At present 8 s → 24 s would take a generation
  from 160 s to 481 s (120 h for 900 generations).

## Phase 3 — the learner

- **No fifth shared-policy variant.**
- **Distillation test first.** The per-candidate (1+1)-ES refines ~16 policies
  per generation in each body's own basis and demonstrably improves that body's
  mission fraction; they are free specialist teachers that have never been used
  as anything but individual controllers. Train one morphology-conditioned
  student to match them by supervised regression (Rusu et al. 2015; and for this
  exact problem arXiv 2402.06570 and 2211.14296). It answers **representability**
  with data already on disk: if a conditioned network cannot even *fit* the
  per-body optima, then no shared controller is reachable and the capacity
  question is answered with a number.
- **Then PGA-MAP-Elites** (Nilsson & Cully, GECCO 2021). Put the gradient into
  the *variation operator* — a replay buffer over the population's transitions,
  a TD3 critic, half the offspring produced by gradient ascent on archive
  solutions and half by the genetic operator. This is the architecture the
  project already has: it is MAP-Elites, it discards every transition, and
  `evolution/cmaes.py` still holds an `Emitter` class nothing consumes. Each
  offspring keeps its own weights, which is what the four null results say is
  necessary.
- If a shared learner is revisited later: GePPO (arXiv 2111.00072) for
  principled sample reuse, APPO/IMPALA with V-trace for the 99.8%/0.1%
  simulation-to-learning cost ratio, dual-clip for the large negative advantages
  a near-episodic reward produces. RUDDER (arXiv 1806.07857) would replace the
  hand-written potential with a learned return decomposition.

## Phase 4 — PPO implementation hygiene, as one arm

Present: advantage normalisation, gradient clipping, tanh activations. Absent:
**state normalisation, entropy bonus, learning-rate decay, orthogonal
initialisation, Adam epsilon (default 1e-8 against the usual 1e-5), value
clipping**, and the policy is not a squashed Gaussian (`tanh` is applied to the
mean only, so σ = 0.607 samples leave [−1, 1] and `INTENT_AUTHORITY = 0.5`
rescales afterwards).

These are coupled — annealing changes the KL curve, orthogonal init changes
saturation, an entropy bonus changes exploration — so change them together and
compare against arch33, rather than spending five runs to learn less.

Note the diagnostic this resolves: arch33's policy hit the KL ceiling on 88% of
updates by generation 450, and the trajectory over 46 snapshots is significantly
anti-correlated (consecutive-step cosine −0.061 ± 0.019 overall, −0.128 ± 0.016
in the last third, displacement ~ n^0.388). **A fixed learning rate alone
produces that signature**, so it cannot be read as evidence that the gradient
direction rotates.

## Phase 5 — structural evolution

Graph-level recombination (`crossover` deliberately takes one parent's graph
wholesale, so structure is effectively asexual), a gene duplication-and-
divergence operator, and removing the 8-part cap.

Deliberately last: while corr(tier1, tier2) ≈ 0, stronger variation only climbs
the wrong hill faster.
