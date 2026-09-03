# Roadmap

Written 2026-09-03 after arch33, and revised the same day once every phase was
built. Every item names the measurement that motivates it; nothing here is on
the list because it seemed like a good idea, and nothing is marked done without
the number it produced.

All six phases are implemented. **None of it has been through a training run** —
every number below is from offline measurement: the seed plans, the stored
arch33 archives, re-runs of arch33 elites, and timing probes. The next run is
the test, and it is one arm rather than six, which is a deliberate choice and
not an oversight: Phase 4's items are mutually coupled, Phase 1 changes what the
score means, and arch33 already spent a run per item to learn less.

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
Do not build a fifth. Phase 3 below now says *why* with a number.

---

## Phase 0 — measure, no code — **done**

**The fluid solver's cost split is not what the standing figure says.** Measured
on this machine over 25 designs (7 archetypes and 18 perturbations), panels
26–189, bodies 3–22, best-of-seven over 300 calls each:

| model | fit | R² |
|---|---|---|
| panels only | 2.67 µs/panel + 368 µs fixed | 0.216 |
| bodies only | 12.4 µs/body + 430 µs fixed | 0.080 |
| both | 3.19 µs/panel − 5.5 µs/body + 391 µs fixed | 0.223 |

The standing figure was `0.86 µs/panel + 233 µs fixed`. Both terms are larger
here, and more importantly **panel count explains 22% of the variance and
nothing else explains the rest**: calls range 367–1072 µs at panel counts that
overlap almost completely. Decomposing one call (`beetle`, `medusa`, `gannet`):
the per-body `mj_objectVelocity` loop is 1.8–5.1% of it and the medium query
0.8–2.1%; the remainder is roughly fifty numpy calls inside `FluidSolver.apply`
whose cost is *per call*, not per element.

So the roadmap's original reading is right and stronger than it was stated: at
the sizes this search actually evolves, the fluid call is dominated by fixed
dispatch overhead. That is what batching machines together amortises, and it is
the argument for Phase 2 and for the Mojo port. Reproduce with the probe in
`docs/` history or re-derive in ten lines; the numbers above are from an
otherwise idle machine.

**The wingless "climbers" were not climbing; they were being thrown.** Of
arch33's 13,353 evaluations, 60 cleared the *entire* air ladder and **nine of
those had `wing_area` exactly 0.0000**, at wing loadings of 268,000 to 1,178,337
N/m². Thirteen such designs were still sitting in the archives, scoring 0.427 to
0.853 in air.

Re-run under the new code, all thirteen score 0.0000 to 0.0017:

| island | wing loading N/m² | actuated DOF | air, old | air, new |
|---|---|---|---|---|
| air | 1,178,337 | 0 | 0.853 | **0.0000** |
| land | 344,664 | 2 | 0.845 | 0.0000 |
| generalist | 364,058 | 4 | 0.837 | 0.0016 |
| aerial_diver | 267,726 | 0 | 0.812 | 0.0016 |
| … nine more, 0.427–0.585 | 164,530–509,150 | 0–21 | | 0.0000–0.0017 |

And the decomposition says which change did it. Their old score of 0.853 under
`frac*(0.55*flight + 0.25 + 0.2*speed)` needs an airborne fraction of at least
0.85, because the bracket cannot exceed 1. Released at the *floor* of the launch
band instead of the cap, the same machines are airborne for 0.32 of the segment
and never long enough for a sink rate to be measurable at all. The gate then
takes 0.032 to 0.0016.

So the answer to the original question is no: the spin was not producing the
lift, and neither was anything else. **The launch was the whole score.** Thrown
at 30 m/s these objects stayed above the 0.3 m clearance threshold long enough
for the airborne fraction to carry the flat 0.25, and the rising beach supplied
the rest. That is also the strongest single argument for Phase 1.1: the term
that had to go was not the one anybody would have picked.

## Phase 1 — fidelity. This was the binding constraint — **done**

### 1.1 The air score now measures flight

All four causes fixed together, because each one alone leaves the others paying.

* **The launch is no longer inversely earned.** A design with no trim speed
  anywhere in the band was released at the 30 m/s cap — the worse the design,
  the bigger the gift. It is now released at the *floor*: the correct answer for
  a machine that cannot fly is to drop it, not to throw it.
* **Nothing is paid for being off the ground.** `0.55*flight + 0.25 +
  0.2*speed` is now `0.55*flight + 0.25*glide + 0.20*station`. The flat 0.25
  became a graded glide term that has to be earned by the descent rate (full
  marks at zero sink, nothing by 6 m/s, where the flight path at the bottom of
  the launch band is steeper than 45° and no part of the trajectory is being
  produced by lift). The 0.2 for the launch velocity the environment handed the
  machine is simply gone.
* **A tumble is not a commanded turn.** `turn_rate_held` was the mean rate of
  change of the up-vector, gated only on not sinking; arch33's champion logged
  31.4 rad/s of it with **zero actuated degrees of freedom**. It is now gated on
  the correlation between the commanded body twist (`MobilityBasis.twist_of`)
  and the achieved angular rate one control interval later. No controller, no
  commands, no manoeuvring credit.
* **Station keeping is measured.** `station_keeping` is the fraction of the late
  airborne window spent within a quarter-span of the height the machine settled
  at, and it is a new ladder rung (`holds_station`, between `holds_height` and
  `climbs`). A 0.4 m/s creep clears the 0.5 m/s sink bar for a whole segment and
  scores 0.38 here.

Measured on the seven seed plans, old formula against new on the *same*
rollouts:

| plan | old | new | airborne | sink m/s | launch m/s |
|---|---|---|---|---|---|
| gannet — the one that flies | 0.4500 | **0.1832** | 1.00 | 1.71 | 13.3 |
| beetle | 0.1619 | 0.0229 | 0.56 | 5.12 | 17.8 |
| eel | 0.1362 | **0.0001** | 0.46 | 8.83 | 28.4 |
| medusa | 0.1214 | 0.0024 | 0.41 | 11.67 | 6.0 |
| ray | 0.1158 | 0.0032 | 0.41 | 13.85 | 11.8 |
| bat | 0.1093 | 0.0022 | 0.37 | 14.74 | 15.8 |
| teal | 0.0000 | 0.0000 | 0.00 | — | 13.5 |

The six non-flyers used to occupy a band 0.109–0.162 wide while the flyer scored
0.45 — a best-to-next ratio of 2.78. It is now 7.99. The eel is the clearest
case: descending at 8.8 m/s it scored 84% of what the beetle scored, entirely
because it was launched at 28.4 m/s and kept the speed it was given.

**Air scores from arch33 and earlier are not comparable to arch34's.** That is
deliberate — the whole finding was that the old number did not measure flight —
and the raw measurements travel with every design, so old runs stay analysable.

### 1.2 Tier-1.5: a 60 s single leg on promotion candidates

`envs.evaluate.evaluate_tier1_5`, called from `_verify_and_label` before the
Tier-2 mission. One leg, not three, and it is the **weakest** one — the mission
fraction is `min(competence)**0.5 * mean(competence) * …`, so the minimum is the
binding term twice over. Sixty seconds is 7.5 Tier-1 segments and a fifth of a
mission leg; the cost scales with promotions (at most three per verification
round) rather than with the sixteen candidates per generation.

Reported, not enforced. What the retention distribution looks like is the
measurement this exists to produce, and gating Tier-2 on it before that
distribution is known would be choosing a threshold from nothing — and would
also remove the ground-truth pair the critic learns from. `tier1_5_retention`
now appears on every promote event and on the elite in the archive.

### 1.3 Schedule aliasing

Every periodic job fired on `gen % N` while the island rotates once per
generation, so with six islands any `N` sharing a factor with six could only
ever reach a subset: `tier2_every = 15` reaches islands 0 and 3, `audit_every =
30` reaches island 0 alone. Both now count **per-island visits**, which removes
the aliasing at exactly the same total cost — 60 verifications and 30 audits
over 900 generations either way, but spread across all six islands instead of
two and one.

The defect is visible in arch33's own archives: of 550 stored controllers, the
28 that a promotion actually fitted to their body belong to **two islands**,
`air` and `amphibian`. Four islands never had a controller refined, because a
controller is only refined at a promotion and a promotion only happens at a
verification.

### 1.4 Tier-0 gates for the exploit family

`envs.triphibian.airworthiness` returns the flight gates a design fails, from
geometry and mass alone:

* **no lifting surface** — `wing_area < 1e-3 m²`, the same figure
  `Phenotype.is_plausible_flyer` already used;
* **wing loading past any usable speed** — above `0.5 ρ V² CL_max` = 992 N/m² at
  the top of the launch band, no attitude at any speed this mission will fly
  produces the machine's weight in lift. Derived, not chosen. arch33's champion
  was at 1,178,337 N/m².
* and measured during the segment, **spinning faster than one revolution per
  second** — at the reference's 2.2 Hz stroke that is 160° of body rotation
  within a single stroke, so nothing averaged over the window is an attitude
  that was held.

Tier 0 already knew all of this (`p_air` is infinite, the note reads "no lifting
surface"); Tier 1 then simulated an air segment for the machine anyway and
scored it. The gates carry the Tier-0 finding into the air score.

A gate does not reject the design and does not raise the exploit flag — a
wingless submarine is a legitimate water specialist and the islands exist so it
can be one. It scores a twentieth of what it would otherwise (`GATED_AIR_CREDIT
= 0.05`), which separates the two classes decisively (0.144 against 0.007 rather
than 0.144 against 0.136) without the hard zero that would flatten
`mission_fraction` for every wingless machine at once and remove the gradient
back toward growing a wing. Its ladder-visible `sink_rate`, `station_keeping`
and `turn_rate_held` are neutralised so it cannot climb the ladder either, while
`measured_sink_rate` keeps the honest number.

### 1.5 Telemetry defect

`Genome.body_plan` is now its own field, set once at birth by the constructor
that builds the archetype and preserved through `copy`, `mutate` and
`crossover`. `meta["body_plan"]` used to read `lineage[0]`, and `lineage` is a
rolling window of the last 24 *mutation operator* names — so the field reported
a plan for a design's first 24 mutations and an operator name (`jitter_cppn`,
`body_field`) for the rest of its life. Every diversity claim made from it in
arch30, arch31 and arch33 was reading a mixture of the two, and none of them can
be repaired from the stored runs.

## Phase 2 — throughput — **done**

**Multiprocess actors.** `envs/actors.py`, `--workers N`. A persistent pool of
worker processes, each holding its own batched evaluator and its own GPU
pipeline. Also where PPO's data collection becomes parallel: each worker fills
its own `RolloutBuffer` and the parent concatenates the trajectories, so the
batch the optimiser sees — including `_terminal_scale`, which is computed over
the merged set — is exactly the batch it saw before.

**What a batched step is made of**, measured at batch 16, best of five over 200
steps:

    step 1179 us = fluid 580 (49%) + mj_step 199 (17%)
                   + power budget 207 (18%) + Python 194 (16%)

Barely half of it is the part that already runs on the GPU — and that half is
host-side marshalling rather than kernel time, which is why it parallelises too.

**Raw stepping throughput**, W processes of 8 machines each:

| W | per-process step | machine-steps/s | vs W=1 |
|---|---|---|---|
| 1 | 853 µs | 9.4k | 1.00x |
| 2 | 914 µs | 17.5k | 1.86x |
| 4 | 910 µs | 35.2k | 3.72x |
| 8 | 1005 µs | 63.7k | 6.7x |
| 16 | 1074 µs | 119.1k | **12.7x** |

Sixteen processes cost 26% more per process and return 12.7 times the
throughput. Growing the batch inside one process instead saturates: k=32 gives
78 µs/machine against k=16's 89, and 12.8k machine-steps/s against 119k.

**End to end**, a full evaluation (identification, three 8 s segments, three
transitions) of 32 designs on a warm pool:

| workers | wall | speedup |
|---|---|---|
| 1 | 141.8 s | 1.00x |
| 2 | 87.2 s | 1.63x |
| 4 | 65.1 s | **2.18x** |
| 8 | 64.4 s | 2.20x |

The gap between 12.7x and 2.2x is the reason `min_shard` exists and is the
useful finding here. At 32 designs, eight workers means shards of four, and a
shard of four costs 149 µs per machine-step against 105 in eight and 89 in
sixteen — so the extra parallelism is spent entirely on smaller batches. Shard
size and worker count trade against each other; the floor is 8, which is where
that curve flattens, and **using more cores means raising `--batch`**, which is
a search-design decision and is therefore left to the caller rather than done
automatically.

**Sharding cannot change a score, and that is asserted rather than assumed.**
Nothing per machine depends on which other machines share its batch: the scatter
draw and the identification deltas are each a `default_rng` built fresh per
machine, the force limiter is per machine, and the fluid kernel has no
cross-machine reduction. Measured, the same designs score bit-identically at 1,
2, 4 and 8 workers, and `tests/test_search.py` pins it. The one deliberate
exception is a *learning* shared policy, which samples its actions: a run with
one is reproducible per worker count rather than across worker counts.

Longer segments can now be re-costed against this rather than against the
single-process figure.

## Phase 3 — the learner

- **No fifth shared-policy variant.**
- **The distillation test is done, and it answers the capacity question.**
  `python -m dytiscidae.ops.run distill --run runs/arch33`
  (`learning/distill.py`) trains one morphology-conditioned student to match the
  per-body controllers stored in the archives, holding out **bodies** it never
  sees. Held-out R² against the teachers' own variance, 550 teachers, 165 held
  out, 256 sampled observations each:

  | | train R² | held-out R² |
  |---|---|---|
  | constant (the population mean command) | 0.014 | 0.011 |
  | shuffled — morphology paired to the wrong teacher | 0.405 | −0.473 |
  | unconditioned — the same net, morphology channels zeroed | 0.030 | 0.024 |
  | **conditioned** | **0.664** | **−0.047** |

  Training harder makes it worse, which is the whole finding: at hidden 128 /
  600 steps the conditioned student reaches +0.124 held-out; at 128 / 4000,
  −0.047; at 256 / 8000, train 0.862 and held-out −0.295. **Capacity is not the
  constraint — generalisation across bodies is.** The student can memorise the
  teachers it was shown (0.992 train R² on the 28 genuinely refined ones) and
  cannot predict an unseen body's controller better than commanding the
  population average.

  The shuffled control says the eight morphology channels are not *nothing*:
  honest pairing sits +0.426 above the noise floor on held-out bodies. It also
  says they are nowhere near enough — the unconditioned student, which cannot
  see morphology at all, scores +0.024 against the conditioned student's −0.047.

  Two caveats, stated because they bound the claim. The observation states are
  drawn from the observation's own envelope rather than from recorded traces,
  and that envelope is wider than the manifold a rollout visits — so a *high*
  score would be strong evidence and a low one is a reason to repeat the
  measurement on traces before concluding. And a (1+1)-ES given six steps on 168
  weights is a weak optimiser, so some of what the teachers encode is optimiser
  noise; the shuffled baseline is in the study precisely to bound that, and it
  is what makes the +0.426 gap meaningful rather than assumed.

  Read together with the four null results, this says the failures were not
  about reward shaping or sample budget. There is no shared controller of this
  class to find.

- **So: PGA-MAP-Elites** (Nilsson & Cully, GECCO 2021) is now the indicated
  direction rather than one option among several. Put the gradient into the
  *variation operator* — a replay buffer over the population's transitions, a
  TD3 critic, half the offspring produced by gradient ascent on archive
  solutions and half by the genetic operator. Each offspring keeps its own
  weights, which is exactly what the distillation result says is necessary.
  The project is already MAP-Elites, it already discards every transition, and
  `evolution/cmaes.py` still holds an `Emitter` class nothing consumes.
- If a shared learner is revisited later: GePPO (arXiv 2111.00072) for
  principled sample reuse, APPO/IMPALA with V-trace for the 99.8%/0.1%
  simulation-to-learning cost ratio, dual-clip for the large negative advantages
  a near-episodic reward produces. RUDDER (arXiv 1806.07857) would replace the
  hand-written potential with a learned return decomposition.

## Phase 4 — PPO implementation hygiene, as one arm — **built, unmeasured**

All seven changed together, in `learning/ppo.py`:

| | before | now |
|---|---|---|
| observation normalisation | none | running mean/var, carried in `state_dict`, updated after each epoch loop so no ratio is computed across two normalisations |
| entropy bonus | `ent_coef = 0.0` | 0.01, sample-estimated (a squashed Gaussian has no closed-form entropy) |
| learning-rate schedule | fixed | linear anneal to zero over the run |
| initialisation | torch default | orthogonal, gain √2 in the trunk, 0.01 on the policy head, 1.0 on the value head |
| Adam epsilon | 1e-8 | 1e-5 |
| value loss | unclipped | clipped against the previous estimate |
| the policy | `tanh` on the *mean*, then sample | genuine squashed Gaussian with the tanh Jacobian in the log-density |

The squashed Gaussian is the one that was a defect rather than a refinement.
With `log_std = −0.5` the standard deviation is 0.607, so a large share of every
sample fell outside [−1, 1] — and `MobilityBasis.coeffs_for_twist` clips its
input to that interval. The policy was scored on actions it had not taken.

They are coupled — annealing changes the KL curve, orthogonal initialisation
changes where the tanh units start, an entropy bonus changes exploration — so
this is one arm against arch33, not five runs.

Note what this does **not** resolve: arch33's policy hit the KL ceiling on 88%
of updates by generation 450, and its trajectory over 46 snapshots was
significantly anti-correlated step to step (cosine −0.061 ± 0.019 overall,
−0.128 ± 0.016 in the last third, displacement ~ n^0.388). **A fixed learning
rate alone produces that signature**, so it was never evidence that the gradient
direction rotates, and the anneal is in this arm for that reason rather than
because a diagnosis was made.

Given Phase 3, this arm is worth running as *hygiene on the learner the project
keeps for the PGA variation operator*, not as a fifth attempt at a shared
controller.

## Phase 5 — structural evolution — **done, and it was gated on Phase 1.2**

Three changes, all in `core/genome.py`.

**Graph-level recombination.** `crossover` took one parent's graph wholesale and
imported only the other's global genes and CPPN weights, so structure was
effectively asexual: every graph in an archive descended from one seed graph by
mutation alone, and two islands that independently found a good wing and a good
fin could never produce a design carrying both — which is most of the argument
for having islands at all. `graft_subtree` transplants a whole coherent limb,
with its own shape genes, joint and phase, onto a site in the receiving graph,
which is otherwise untouched. The original reasoning against graph crossover is
sound and is about *cut-and-splice*, which severs both graphs and rejoins the
halves; subtree grafting is what genetic programming actually uses. Measured
over 60 archetype pairs it fires every time, adds 1.33 parts on average, and
everything it produces builds. The lineage records `graft` rather than
`crossover` when structure moved, so the two are separable in the record.

**Duplication and divergence.** `mut_duplicate_part` copies a module, attaches
the copy where its twin is attached and then displaces it, and diverges the
copy's dimensions and phase. The copy gets *its own* CPPN entry rather than the
original's index — sharing the index would be duplication without divergence,
two parts locked to one shape gene that no later mutation can separate. This is
the move that builds a graded series (a wing, a smaller wing, a fin); before it,
the only additive operator drew a part from the prior, so every member of such a
series had to be stumbled on independently.

**The 8-part cap is gone.** It was a compute-era number and it was never binding
on quality — arch33's population mean sat at 3.95 parts from the first
generation to the last — but at eight parts a duplication is usually refused,
so it had to go with the operator. `MAX_PARTS` is 24 and the guard that actually
binds is on *bodies*: one part with `radial=6` and `reflect` becomes twelve
rigid bodies per level of recursion. `estimated_bodies` walks the graph for
that, and it is a lower bound and known to be one — measured against the built
phenotype over 207 designs the real-to-estimated ratio is 1.00 median, 2.44 at
the 95th percentile and 11.0 at worst, because a surface part expands into
spanwise segments the graph cannot see. So the ceiling sits an order of
magnitude below the resource it protects, and `MIN_CAP_BODIES` was raised from
2,048 to 8,192 so the backstop is far away.

**Why this was last, and what unlocks it.** The stated condition was that while
corr(tier1, tier2) ≈ 0, stronger variation only climbs the wrong hill faster.
That condition has not been *measured* away — it needs a run — but the two
things it depends on were both fixed first: the air score now measures flight
(Phase 1.1) and `tier1_5_retention` (Phase 1.2) is the number that will say
whether an 8 s score survives a 60 s leg. If it does not, these operators should
be turned down, not tuned: `STRUCTURAL_OPERATORS` is what the curator throttles
and `duplicate_part` is registered there with the rest.
