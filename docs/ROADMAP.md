# Roadmap

Written 2026-09-03 after arch33, and revised the same day once every phase was
built. Every item names the measurement that motivates it; nothing here is on
the list because it seemed like a good idea, and nothing is marked done without
the number it produced.

Revised 2026-09-05, after arch34 ran. The six phases below are history now and
keep the measurements that motivated them; **the current work list is
[arch35](#arch35--the-work-list)**, immediately after the results.

Read [CPU_LEGACY.md](CPU_LEGACY.md) for the older backlog. This file supersedes
it wherever the two disagree.

---

## What arch34 measured

900 generations, 21.7 h, 14,006 evaluations, seed 20260901, batch 16,
`--workers 4 --min-shard 4`. Run directory `runs/arch34`, report at
`runs/arch34/report.html`.

| | arch33 | arch34 |
|---|---|---|
| wall for 900 generations | 40.4 h | **21.7 h** |
| evaluations | 13,353 | 14,006 |
| islands receiving promotions | **2** | **6** (of 180) |
| lineage depth, last 20 gens / max | 10.15 / 25 | 9.54 / **26** |
| curriculum typical / reached | 2 / 4 | 2 / 4 |
| n_parts, first 100 → last 100 | 3.96 → 3.95 | 4.59 → **6.98** |
| energy-infeasible, first → last | 39.7% → **31.8%** | 41.4% → **56.3%** |
| corr(fitness, mission), final gen | +0.784 | +0.697 |

**`mission_fraction` levels do not compare across the arch33/arch34 boundary and
were not compared.** Phase 1.1 redefined the air term and 1.4 gates it, and
`mission_fraction` is built from air. Both within-run changes are positive and
significant (arch33 t=+11.3, arch34 t=+5.1); neither magnitude may be read
against the other.

**What worked.** Phase 1.3 exactly: promotions 6/6/6/6/6/6 by island against
arch33's two. Phase 2: the same generation count in 54% of the wall time.
Phase 1.4: airworthiness gates fired on 19.2% of evaluations, 1,311 of them for
"no lifting surface", and **zero exploit events were raised in the whole run**
against arch33's wingless-thrown-glider family. Best design of the run:
`mission_fraction` 0.3474 at gen 447, amphibian island, gannet plan, 8 parts,
6 DOF, 7.26 kg, 2.60 m span, wing loading 137 N/m², energy margin +0.581, no
gates — a real flyer.

**What Phase 5 cost.** Four measured consequences of the structural operators,
all tracking `n_parts` 3.95 → 6.98: per-generation time +7%, rollout divergence
0.25% → ~2% (eightfold), archive pressure, and the largest — **energy
infeasibility 41% → 56%**, reversing arch33's 40% → 32% gain. Within arch34 the
link is direct: infeasible share runs 20.6% at 2 parts, 52.8% at 4, 57.9% at
5–6. **arch33's 8-part cap was doing energy work nobody had credited it with.**

**Phase 4 (PPO hygiene) remains unattributable**, as designed — it was one arm
with Phase 1.

### The three findings that set arch35's list

**1. Tier-1 does not predict Tier-2, and never did.**
`corr(tier1_fraction, tier2_fraction)` = **+0.105** over 180 promotions
(arch33: **+0.077** over its own 180 — the disconnect is not new, arch33 simply
had no instrument for it). Split by era it wanders between −0.01 and +0.28 with
n≈36 per band: no trend, just noise around zero.

Tier-1.5 says why. Over 179 promotions the 60 s retention is **bimodal**: 66.5%
in [0, 0.25), 24.0% at or above 1.0, only 9.5% in the whole middle. Median
0.124. And the 8 s score predicts which mode a design lands in — *inversely*:

| 8 s Tier-1 score | n | median retention over 60 s |
|---|---|---|
| above the median | 101 | **0.037** |
| below the median | 78 | **0.390** |

`air` was the binding domain at 60 s in 53 of the first 61 cases.

**2. Every air score this project has produced was earned by a machine that was
thrown.** `TriphibianEnv.SPAWN[Domain.AIR]` is 30 m altitude and `reset` sets
`qvel[0] = launch_speed`. `envs/mission.run_continuous` has *one* placement —
its docstring: "Start in the first commanded domain, the only placement in the
run" — so in a continuous mission the air and water legs begin with the machine
wherever the land leg left it. `DOMAIN_CYCLE` is [AIR, WATER, LAND] and the
three scored transitions are `air_to_water`, `water_to_air`, `water_to_land`:
**`land_to_air` is not among them.**

Filmed, every elite gives the same result — the fitness-best, the mission-best,
at 8 s legs and at 60 s legs: land 98–100% on-task, **air 0%, water 0%,
transitions 0/2, max depth 0.0 m**. The beach needs 11.4 m of travel to reach
water deep enough to submerge; the best elite covers 3.18 m in 30 s.

**3. The population is not uniformly incapable of leaving the ground — a probe
said so and the probe was wrong.** Driving elites from `cpg.base` rather than
through the evaluation path reported that none rise. Measured properly over 64
elites: 18.8% gain height, 12.5% more than 0.20 m, 7.8% more than 0.45 m, best
1.04 m. Hopping and sustaining flight are different capabilities and only the
first exists.

---

## arch35 — the work list

**Status: A and B ran. See "What arch35 measured" below for what they did.**
Items C–I below were not attempted in arch35 and carry forward; C's precondition
is now satisfied and D–I are re-ordered in the arch36 list.

### Done and unrun — the next run tests these

**A. Take-off is measured, gated, laddered, and enters the mission score.**
`takeoff_height` is the projectile estimate `clearance + max(0, vz)²/2g` as a
gain over resting clearance, gated on airborne-and-upright, with
`measured_takeoff_height` beside it. A `takeoff` ladder runs 0.02 / 0.10 / 0.30
/ 0.80 m. `mission_fraction` multiplies by `max(takeoff_fraction, 0.05)` with
full credit at 0.30 m — the transition term's shape, deliberately **not** a
fourth competence, where `min(competences)` would multiply the population by its
own zero.

Thresholds come from the measured distribution. A threshold anywhere above the
population reads zero and carries no gradient, which is the failure Wang et al.
name in *Towards Quadrupedal Jumping and Walking for Dynamic Locomotion using
Reinforcement Learning* (arXiv 2510.24584) and fix by densifying with the
projectile equations.

**`mission_fraction` is not comparable across this boundary either.** The trade
is deliberate: what is given up is comparison with a score at which arch34 ran
0/2 transitions and 0% on-task in air and water.

**B. `land` gains `stirs` before `moves`.** On `land_peak_speed` — the best
one-second displacement rate over windows held upright throughout — at 0.08 m/s.
61.6% of arch34 sat at the rung below `moves`, upright and self-supporting and
under its 0.1 m/s *mean*; measured peak is 0.086 m/s median against a 0.026 m/s
mean. A machine that covers half a metre in one second and falls averages an
eighth of what it produced.

Both A and B were caught scoring falls before they were gated — ungated, the
apex estimate read p90 0.378 m and peak speed 3.23 m/s, a body rebounding off
the beach and a body sliding down it on its side. That is the **third** time
this project has found a score paying for uncontrolled motion, after the thrown
glider and the tumble-as-turn, and the third time the fix was a gate rather than
a coefficient.

### Not done, in order

**C. Score `land_to_air`, once the take-off ladder has produced a distribution.**
It is already in `TRANSITION_ENDPOINTS`; only the scoring loops in
`batchroll.py:689` and `evaluate.py:260` exclude it. It was excluded because
nothing could do it — `transitions.py` records "none: nothing gets off the
ground" for all six seed plans — and that reason is now partly obsolete. Wait
for one run with A in place: if the take-off rungs stay empty, adding the
transition adds a constant zero.

**D. `--segment-seconds 24`.** Cost measured, and it is not what the README
said: 8 s costs 57.2 s per 8 designs, 16 s costs 65.1 s (1.14x), 24 s costs
77.0 s (1.35x), because identification is 67% of an evaluation and does not
scale with the window. **Tripling the window costs a third more, not three
times more.** Demoted below A–C: a longer window on a thrown machine measures a
longer throw, so take-off has to be scored first for window length to mean
anything.

**E. Charge for parts, or restore a cap.** Energy infeasibility went 41% → 56%
as `n_parts` went 3.95 → 6.98, and within arch34 the link is monotone in part
count. The search is buying structural complexity with energy feasibility and
is not being charged for it. Not a return to 8 parts — the cap was a compute-era
number — but the energy cost of parts has to reach the selection pressure.

**F. `descriptor_refit_every`.** The archive is losing ground to its own
refits: the archive size *at* each refit fell 105 → 116 → 105 → 108 → 107 → 95
→ 88 → 77 over the run, and the four learned axes do not converge — lead-term
agreement between successive refits ran 4/4, 1/4, 2/4, 4/4, 2/4. Every refit
merges 16–31 cells (331 total, worst 107 → 76) and the search stops refilling
to its previous peak. `qd_score` fell ~60 → 36 and coverage ~16% → 10% over the
same stretch. Candidate fixes: refit less often, freeze the axes once they
settle, or require a refit to improve a criterion before it is accepted.

**G. Identification is 67% of an evaluation.** `identify_axes_every` is 1, so
every candidate is re-identified every generation, and `batchroll.py` states it
cannot be batched — each machine runs a different perturbation experiment.
`Genome.body_plan` is now its own field, so whether structure changed is known:
an unchanged morphology could reuse its axes. This is the largest single
throughput lever left and it is worth more than any further pool-shape work.

**H. `np.cross` is 16% of an evaluation** — 308k calls of a three-element cross
product plus numpy's dispatch overhead, inside `FluidSolver.apply`. Explicit
component arithmetic is the standard 5–10x. Mechanical, low risk.

**I. Nothing rewards going *toward* the water.** `SegmentResult.distance` is
`norm(end - start)`, direction-free. A design that walks well has no gradient
pulling it down the beach, and the mission's water leg is 11.4 m away.

---

## What arch35 measured

496 generations of a planned 900, 7806 evaluations, 12.1 h, 74 s/gen, seed
20260901. **The run did not finish**: the machine ran out of memory at 11:04 on
2026-09-06 and the kernel OOM killer fired at 11:05:17. Not a search failure and
not a crash — uptime was unbroken. Archives and `search_state.pkl` are intact.
Working notes: `runs/arch35_notes.md`.

### 1. B worked exactly as designed, and is finished

Land ladder by rung name, whole run against arch34:

| rung | arch34 | arch35 |
|---|---|---|
| (below) | 28.7% | 26.7% |
| stays_upright | 1.6% | 1.8% |
| supports_itself | **61.6%** | **16.2%** |
| stirs | — | **46.4%** |
| moves | 7.3% | 8.5% |
| walks | 0.4% | 0.2% |
| climbs_slope | 0.5% | 0.2% |

The 61.6% that arch34 stranded moved onto the new rung, and `moves` did not move
(7.3 → 8.5%) — so `stirs` inserted *below* `moves` rather than cannibalising it.
That is the whole of what B was for. Nothing further is owed here.

### 2. A moved the distribution and did not move the mission

Take-off occupancy climbed band over band, n≈1550 per band:

| gens | unweights | hops | clears | climbs_out |
|---|---|---|---|---|
| 0–99 | 37.9% | 15.9% | 4.2% | 1.2% |
| 100–199 | 38.2% | 20.2% | 7.9% | 2.8% |
| 200–299 | 39.3% | 17.4% | 6.8% | 2.6% |
| 300–399 | 45.2% | 24.1% | 10.9% | 4.5% |
| 400–449 | **50.2%** | **26.7%** | **11.8%** | **5.3%** |

This is the first time anything in this project has pushed designs off the
ground. **And the filmed mission is arch34's result unchanged**: the
mission-best elite of 174 runs `on-task 33%, transitions 0/2, max depth 0.0 m`,
with the air and water legs at 0%.

Both facts are true at once and the second is the one that matters. A is a
working gradient on an 8 s land segment; it has not yet produced a machine that
leaves the ground in a 900 s continuous mission. `mission_fraction` is not
comparable across the arch34→arch35 boundary, but `transitions 0/2` is, and it
did not move.

### 3. A is being bought with land posture

Monotone over five bands:

| gens | below `stays_upright` | stirs | hops |
|---|---|---|---|
| 0–99 | 21.4% | 48.5% | 15.9% |
| 100–199 | 27.4% | 47.2% | 20.2% |
| 200–299 | 28.1% | 46.4% | 17.4% |
| 300–399 | 30.2% | 43.1% | 24.1% |
| 400–499 | **37.7%** | **37.0%** | **27.0%** |

`below` gained 16 points while `hops` gained 11 and `stirs` lost 11.5. `moves`
never moved. The two groups are disjoint — the take-off gate is per-sample
`free & level`, so nothing in the `below` group can be feeding the take-off
numbers — so this may be the population splitting rather than a trade. It is
unresolved, and it is the most important open question A leaves behind: a third
of the population can no longer stand up, and the mission's land leg needs that.
Same shape as arch31's "competence +32% bought with energy −36%".

### 4. The take-off tail is the fourth instance of the recurring lesson

~0.2% of evaluations read 4–10 m of take-off height. Of the 55 at or above
`climbs_out` by gen400, **six have `wing_area == 0.0` and an explicit
`no lifting surface` air gate**, one of them reaching 4.15 m *measured*. The
contamination is episodic (all of it in the 50–99 and 150–199 bands; none in
200–249) and it does **not** grow with the distribution, so it is not being
amplified by the reward.

The gate is not broken. `free & level` is applied per sample
(`triphibian.py:1424`), and the segment-mean `upright` in the measurements is an
aggregate, so a low mean is consistent with "upright at departure, tumbling
after". The hole is the one the code's own comment names — the gates cannot
separate a push-off from a bounce — and `measured_takeoff_height` was the
fallback meant to catch it. **It reads 6.82 m on the worst case, so it does
not.**

### 5. F reproduced, harder and earlier than arch34

Archive size *at* each refit: 21, 53, 72, 85, 95, 107, 118, 128, 138, 151, 133,
119, 116, 115, 107. It refilled past its previous peak for nine refits and then
stopped. Merges grew with it: 2, 6, 14, 14, 9, 23, 11, 27, 13, **37, 38**, 22,
29, 21, 26.

Mechanism, measured: between refit 12 (after=97) and refit 13 (before=116) the
search refilled **+19 cells in 27 generations**, and the refits themselves
removed 21–38. **The merge outruns the refill by 10–20 cells per cycle.**
`descriptor_refit_every` is 400 evaluations ≈ 26 generations, so the archive
gets 26 generations to recover from a cut it cannot cover.

Every island peaked between gen196 and gen252 and fell 28–38% (aerial_diver 154
→ 95, air 149 → 98, generalist 151 → 107, land 128 → 82, amphibian 135 → 94,
water 135 → 97). 66 of the last 100 generations reported `stagnant`. `best` kept
climbing throughout — the search was still finding good designs and losing the
shelf it stores them on.

The axes still do not settle: lead-term agreement between successive refits ran
4/4, 0/4, 3/4.

### 6. The pool-shape paragraph in CLAUDE.md overstates its case

arch35 ran the documented optimum `--workers 4 --min-shard 4` and reached a
steady **72–74 s/gen** (n=172, p10 65, p90 86). arch34 ran `--workers 2` with the
default `--min-shard 8` and its median was **74 s** (n=890, p10 67, p90 85). The
sweep's 2.3× (4×4 = 31.7 s against 1×16 = 72.2 s) is real on the evaluation path
and **does not appear at generation level against arch34's actual shape**. Keep
the flags — they are not worse — but the paragraph should say what it buys
against the configuration that was actually used before, which is ~2 s/gen,
inside the spread.

---

## arch36 — the work list

Ordered by what arch35 measured, cheapest decisive thing first.

**A. A lifting surface is a precondition for `climbs_out`.** A design with
`wing_area == 0` cannot depart, and six of them scored the top take-off rung.
Gate the upper two rungs (`clears`, `climbs_out`) on having a lifting surface
and on the machine still being airborne at the end of the window rather than at
its apex. Mechanical, and it is the fourth time the fix for a score paying for
uncontrolled motion has been a gate rather than a coefficient.

**B. Fix F, as its own arm.** The mechanism is measured, so the candidates are
now testable rather than speculative: (i) raise `descriptor_refit_every` until
the refill covers the merge — the refill rate is ~0.7 cells/generation and
merges cost 21–38, so the cadence has to be ≥ ~45 generations to break even;
(ii) require a refit to improve a criterion before it is accepted; (iii) freeze
the axes once lead-term agreement holds across two consecutive refits. **This
must not ride along with anything else** — arch34's Phase 4 came back "ran,
unattributable" for exactly that reason.

**C. Resolve the posture trade before adding more take-off pressure.** Split the
population by whether it holds `stays_upright` and read `mission_fraction` for
each half. If the `below` group's mission is not worse, A is producing
specialisation and should be left alone; if it is, take-off credit has to be
conditioned on retaining posture. This decides whether item E is safe.

**D. Score `land_to_air`.** Its precondition — "wait for one run with A in
place; if the take-off rungs stay empty, adding the transition adds a constant
zero" — is now satisfied: 11.8% clear 0.30 m. Do it **after** A, or it scores
the same falls A is there to exclude. `batchroll.py:689` and `evaluate.py:260`.

**E. Make a 900-generation run survive the machine.** arch35 died at 55% because
the desktop and the search together exhausted 15 GB. `--resume` exists and was
never exercised after an unplanned stop; a run that costs 19 h needs a tested
resume path and a memory ceiling, not the hope that nothing else runs.

**F. Identification is 67% of an evaluation.** Unchanged from arch35's list and
still the largest throughput lever. `Genome.body_plan` is its own field, so an
unchanged morphology could reuse its axes.

**G. `np.cross` is 16% of an evaluation.** Unchanged. Mechanical, low risk.

**H. Charge for parts, or restore a cap.** Unchanged. arch35's mean parts was
5.41 against arch34's 6.01, so the pressure moved slightly on its own; the link
between part count and energy infeasibility was not re-measured.

**I. Nothing rewards going *toward* the water.** Unchanged, and now more
pointed: the filmed machine spent 99% of its land leg in the land medium and
covered no ground, and the water leg is 11.4 m away.

---

## arch34's phases, kept for the measurements behind them

Everything below is history as of 2026-09-05. It is kept because each
item carries the number that motivated it, and those numbers are still
the reason the code looks the way it does.

### Where arch33 left things

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

### Phase 0 — measure, no code — **done**

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

### Phase 1 — fidelity. This was the binding constraint — **done**

#### 1.1 The air score now measures flight

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

#### 1.2 Tier-1.5: a 60 s single leg on promotion candidates

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

#### 1.3 Schedule aliasing

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

#### 1.4 Tier-0 gates for the exploit family

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

#### 1.5 Telemetry defect

`Genome.body_plan` is now its own field, set once at birth by the constructor
that builds the archetype and preserved through `copy`, `mutate` and
`crossover`. `meta["body_plan"]` used to read `lineage[0]`, and `lineage` is a
rolling window of the last 24 *mutation operator* names — so the field reported
a plan for a design's first 24 mutations and an operator name (`jitter_cppn`,
`body_field`) for the rest of its life. Every diversity claim made from it in
arch30, arch31 and arch33 was reading a mixture of the two, and none of them can
be repaired from the stored runs.

### Phase 2 — throughput — **done**

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

### Phase 3 — the learner

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

### Phase 4 — PPO implementation hygiene, as one arm — **ran, unattributable**

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

### Phase 5 — structural evolution — **ran; see "What Phase 5 cost" above**

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
