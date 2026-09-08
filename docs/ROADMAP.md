# Roadmap

Written 2026-09-03 after arch33, and revised the same day once every phase was
built. Every item names the measurement that motivates it; nothing here is on
the list because it seemed like a good idea, and nothing is marked done without
the number it produced.

Revised 2026-09-08, after arch37 ran. Everything below is kept for the
measurement that motivated it; **the current work list is
[arch38](#arch38--the-work-list)**, at the end, after the four results sections.

arch37 is the run that made flight visible to selection — the share of the
archive that can hold its own weight went 27.4% -> 58.0% and gliding capability
rose 9.7x — and its own data refutes the explanation that was about to be given
for the rung above. Read [What arch37 measured](#what-arch37-measured) §2 before
proposing anything about flight.

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

**Corrected at arch36 gen300: about two thirds of that climb was the ungated
tail.** Applying arch36's two gates to arch35's own record retroactively — same
data, same bands — the genuine capability grew far less:

| gens | unweights | hops | clears | climbs_out |
|---|---|---|---|---|
| 0–99 | 26.4% | 8.2% | 0.9% | 0.2% |
| 100–199 | 22.0% | 9.5% | 3.1% | 1.0% |
| 200–299 | 23.8% | 7.6% | 2.1% | 0.5% |
| 300–399 | 26.5% | 10.6% | 2.9% | 1.1% |
| 400–499 | 25.5% | 10.7% | 3.3% | 0.7% |

`unweights` is flat across the whole run and `climbs_out` is noise. The real
growth is `clears` 0.9% → 3.3% and `hops` 8.2% → 10.7%. **A did move genuine
take-off capability, by roughly a third of what the ungated series showed** —
the rest was wingless designs and bodies that were not holding posture.

**And the filmed mission is arch34's result unchanged**: the mission-best elite
of 174 runs `on-task 33%, transitions 0/2, max depth 0.0 m`, with the air and
water legs at 0%.

A is a working gradient on an 8 s land segment; it has not yet produced a
machine that leaves the ground in a 900 s continuous mission.
`mission_fraction` is not comparable across the arch34→arch35 boundary, but
`transitions 0/2` is, and it did not move.

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

**A. A lifting surface, and posture, are preconditions for a scored take-off
— done, unrun.** Two gates on the *scored* `takeoff_height`:

- A design under `WING_AREA_FLOOR` is capped at `TAKEOFF_WINGLESS_CAP` = 0.29 m,
  just below `clears`. It keeps `unweights` and `hops` — being thrown does leave
  the ground, which is all those two rungs ask — and cannot claim `clears` ("a
  crossing's worth of height") or `climbs_out` ("a departure, not a hop"), both
  of which are flight claims `airworthiness` already refuses it.
- Segment-mean posture must reach `TAKEOFF_POSTURE_BAR` = 0.7, the bar
  `stays_upright` already uses, or the take-off scores zero. The existing
  per-sample `free & level` catches only "upright at the instant it left", which
  a body flung by a contact impulse passes on its way through.

`measured_takeoff_height` is deliberately left ungated by both: it is the
diagnostic that found this, and gating it would hide the next one. A test now
asserts that a body upright only at its apex scores 0.0 while its measured
height still reads 0.85 m.

Thresholds set from arch35's 7806 land segments rather than from what a take-off
ought to look like. What the two gates together leave standing:

| | as scored in arch35 | with both gates |
|---|---|---|
| unweights | 41.6% | 24.9% |
| hops | 20.5% | 9.3% |
| clears | 8.1% | 2.4% |
| climbs_out | 3.1% | 0.7% |

Thinner and not empty — a rung nobody stands on carries no gradient.

**B. Fix F, as its own arm.** The mechanism is measured, so the candidates are
now testable rather than speculative: (i) raise `descriptor_refit_every` until
the refill covers the merge — the refill rate is ~0.7 cells/generation and
merges cost 21–38, so the cadence has to be ≥ ~45 generations to break even;
(ii) require a refit to improve a criterion before it is accepted; (iii) freeze
the axes once lead-term agreement holds across two consecutive refits. **This
must not ride along with anything else** — arch34's Phase 4 came back "ran,
unattributable" for exactly that reason.

**C. The posture trade — answered, and it drove A above.** Splitting arch35's
7806 evaluations on the same 0.7 bar:

|  | holds `stays_upright` | below it |
|---|---|---|
| share | 71.5% | 28.5% |
| mission **as scored** | mean 0.00127 | mean **0.00250** |
| mission, take-off multiplier divided out | mean **0.00720** | mean 0.00625 |
| take-off median | 0.0079 m | 0.0426 m |

As scored, the designs that cannot stand up look *twice as good*. That is
entirely A's own multiplier: `mission_fraction` is multiplied by
`max(takeoff_fraction, 0.05)` and the `below` group's take-off is five times the
other's, so the comparison was circular. Dividing the multiplier back out —
`base = mission_fraction / max(takeoff_fraction, 0.05)` — reverses it, in every
band: ratios 0.81, 0.93, 0.91, 0.95, 0.79.

**A was paying designs that cannot stand up, and they are worse at the mission
once you stop paying them for it.** Hence the posture gate in A.

One thing this turned up and did not chase: `land` competence is *higher* for
the below-bar group (median 0.072 against 0.049), which is the wrong direction
for a score that is supposed to reward locomotion on land. Worth a look before
anything else is built on the land score.

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

## What arch36 measured

900 generations, 22.0 h, 14,048 evaluations, seed 20260901, `--workers 4
--min-shard 4`. **It finished** — the first complete 900-generation run since
arch34; arch35 died at 495 to a machine OOM. Identical configuration to arch35,
so the only difference is the two take-off gates. One arm.

### 1. The gates did exactly what they were specified to do

Over 14,048 land segments: **zero** wingless segments above the 0.29 m cap,
**zero** segments below the 0.7 posture bar scoring any take-off, and the
uncontrolled tail is closed — maximum scored take-off 4.16 m against arch35's
10.17 m, with only one segment in 3,247 above 1.0 m at the point that was
checked.

The diagnostic survived, which was the point of leaving
`measured_takeoff_height` ungated: 18 segments in the first hundred generations
scored 0.0 while their measured height read above 0.30 m, the largest at 1.55 m.
Those are the bodies that used to be paid.

### 2. Gated capability improved, and it is a modest effect

Whole run, against arch35 with the same gates applied retroactively:

| | arch35 (496 gens) | arch36 (900 gens) |
|---|---|---|
| hops | 9.3% | **11.7%** |
| clears | 2.4% | **4.0%** |
| climbs_out | 0.7% | 0.9% |
| depth ≥ 10 m | 15.3% | 16.9% |

arch36 is ahead in every band at matched generations, by roughly 10-40%. It is
real and it is not large. **And it corrected the arch35 write-up**: that run's
apparent climb from 4.2% to 11.8% at `clears` was about two thirds ungated tail.

### 3. The posture gate delays the drift and does not stop it

Share of the population below `stays_upright`, by band:

| gens | arch35 | arch36 |
|---|---|---|
| 0–99 | 21.4% | 16.9% |
| 100–199 | 27.4% | 16.1% |
| 200–299 | 28.1% | 16.1% |
| 300–399 | 30.2% | 21.0% |
| 400–499 | 36.6% | 22.1% |
| 800–899 | — | **42.5%** |

At every matched band arch36 is 5–14 points lower. Its own final band is 42.5%,
higher than anything arch35 reached before it died, and arch35 has no band past
499 to compare against.

The run's **best take-off band and worst posture band are the same band**
(700–799: `climbs_out` 2.5%, posture 34.8%). The gate makes those consistent —
everything scored is holding posture — and what it shows is that the search is
now pushing hard toward leaving the ground and most of what it tries falls over.
**Removing the payment did not remove the fact that the nearest reachable thing
to flight, from this population, is a body that topples.** That is a
variation-operator problem, not a scoring one.

### 4. F is unfixed, in both, as expected

Archive size at each refit: arch35 peaked at 151 over 19 refits and ended at 94
(−38%); arch36 peaked at 121 over 34 refits and ended at 79 (−35%). arch36
oscillated where arch35 declined monotonically, but both end well below peak.
Nothing was done to `descriptor_refit_every` and nothing changed.

### 5. The mission did not move. Three runs running.

`showcase --design runs/arch36 --by mission` over 179 elites: fitness 0.8523,
mission fraction 0.0495, aerial_diver island, 4.41 kg, 0.18 m span, aspect ratio
0.3, density ratio 4.65, 2 DOF.

    on-task 33%   transitions 0/2   max depth 0.0 m
      leg 1 land   93%
      leg 2 air     5%
      leg 3 water   0%

arch34, arch35 and arch36 all report **0/2 transitions and 0.0 m of depth**. The
air leg moved 0% → 5%. The segment ladders have improved in every run and the
continuous mission has not moved at all.

**The arch37 list below says why, and neither reason is in the scoring.** No
island's objective contains `land_to_air`, so nothing has ever asked a machine
to leave the ground; and the water segment releases the machine four metres
under the surface, so no water score has ever been evidence a machine can get
wet. Those are structural, and they were found by asking what the islands cover
and where the segments start — not by looking at a score.

---

## Why nothing has ever flown

Asked after thirty-odd generations of footage showing free fall and limp
non-motion and never flight. The observation was right and the telemetry says
why. All figures from arch36's 14,048 evaluations and its 179 final elites.

### 1. Three quarters of the population cannot fly, at any speed

`_measure_trim_speed` sweeps pitch at a series of airspeeds looking for one
where lift reaches the machine's weight. When none exists in `LAUNCH_SPEED_RANGE`
= (6, 30) m/s it returns the **bottom** of the band — the design is dropped, not
launched, which is the correct answer for something that cannot fly.

**130 of 179 elites (72.6%) come back at exactly 6.0 m/s.** A real trim is
`clip(v_stall * 1.2, 6, 30)` and `v_stall >= 6`, so 6.0 exactly can only be the
cannot-fly branch. Three quarters of the archive is being dropped from 30 m.

### 2. The air ladder's first two rungs are paid for falling

`leaves_surface` and `stays_up` read `airborne_fraction` at 0.10 and 0.60. A
body released at 30 m is airborne for the 2.5 s it takes to arrive, so in an 8 s
window it scores 0.31 without doing anything. Measured: `airborne_fraction`
median **0.445**, minimum **0.350** — which is exactly "fall, then lie in the
water". 53% of the population reached one of those two rungs.

Population median `sink_rate` is **9.900 m/s**. That is free fall.

`holds_station`, `climbs` and `manoeuvres` were reached **zero times in 14,048
evaluations**. `holds_height` 1.3%.

### 3. The airworthiness gate certifies designs that cannot fly

`MAX_WING_LOADING` is `0.5 * rho * LAUNCH_SPEED_RANGE[1]^2 * CL_MAX` — it asks
"could this fly at 30 m/s". The gate flags **35.8%** while the measured trim says
**72.6%** cannot fly at any speed. So roughly **37% of the archive is nominally
airworthy and physically incapable**, and the gate cannot see it because it is a
Tier-0 geometric check at a speed these machines never reach.

Consistent with that, the designs scoring best on sink rate have *higher* wing
loading than the population (289 against 158 N/m²). The score was not reading
aerodynamics.

### 4. There was no gradient between "makes no lift" and "flies"

This is the one that matters. A design at lift ratio 0.1 and one at 0.9 scored
identically — both fell — and the only rungs either could reach were the two
that being dropped satisfies. Three quarters of the population had no direction
to move in, which is why thirty generations produced no flight.

### 5. Parameters are diluted by the structural operators

`flap_hz` sits at exactly 2.20 for 50.4% of arch36 and 0.60 for 24.9% — the
beetle and gannet seed defaults. The operator that jitters it, `global_energy`,
fires normally (767 selections, 5.5%, 72.9% hit rate). The cause is the mix:
structural operators run at 16-18% each against ~5.5% for each parameter
operator, so over a lineage of depth 10 a given parameter has a ~57% chance of
never being touched. Not a bug in one operator — `structural_bias` = 1.6 doing
what it says. **Its own arm, not arch37's.**

---

## arch37 — the work list

**Two changes, both flight, bundled because their measurements do not overlap:**
the island work is read from `land_to_air` transition scores and `land_air`
promotions, the air work from the `lift_margin` distribution and the air ladder.
That is what makes it unlike arch34's Phase 4, where two changes shared one
metric.

**A. Four `lift_margin` rungs at the bottom of the air ladder — done, unrun.**
`lift_margin` is `best_lift(30 m/s) / weight`, already computed by the trim
sweep and thrown away until now, so it costs no simulation. Thresholds from the
measured distribution over arch36's 179 elites (min −1.105, median 0.333,
p90 5.591):

| rung | at | leaves standing |
|---|---|---|
| `makes_lift` | 0.10 | 72.1% |
| `carries_a_third` | 0.30 | 51.4% |
| `nearly_flies` | 0.60 | 36.9% |
| `carries_itself` | 1.00 | **27.4%** |

`lift_margin >= 1.0` and "has a real trim speed" select the same designs —
27.4% both ways, 0 disagreements over 179 — which is the check that this is the
right quantity.

**The ordering is the gate.** `rung_reached` stops at the first unmet rung, so a
body that makes no lift cannot reach the `airborne_fraction` rungs at all.
Verified: a machine airborne for a whole segment at zero sink now scores air
rung **0** if its lift margin is zero, and rung 8 if it is 1.5. Before this it
scored four rungs for the same episode.

**What it does not reward.** `lift_at` evaluates a fixed attitude at `t=0` with
no flapping, so `lift_margin` is the *airframe's static lift* — a glider's wing,
not flapping thrust. That is the right first question and it is not the whole
one; propulsion is still measured only by the sink-rate rungs above.

**B. The islands — done, unrun.** `land_air` island, `land_to_air` in the `land`
and `generalist` transitions, and the hybrid routing fix.

**Window length stays at 8 s.** The offline probe built to answer it failed
three times, each time because it was not the evaluation path — the last one
drove elites with their own stored `Policy` while the run had driven them with
the shared PPO network. Its timing is the one usable output: with identification
on, 17.52 / 19.68 / 21.82 s per design at 8 / 16 / 24 s, so **tripling the window
costs 25%**, cheaper than the 1.35x ROADMAP measured without identification. A
third change does not go in this arm.

### The rest, set by four questions asked during arch36, each answered by reading the code and
the telemetry rather than by opinion. Ordered by how much is broken.

### A. The islands do not cover the pairing the whole take-off effort is about

Six islands: `air`, `water`, `land` (singles), `amphibian` (water+land),
`aerial_diver` (air+water), `generalist` (all three). Three singles, **two of the
three pairs**, and the whole. The missing pair is **land+air**.

Worse, `islands.py` gives each island a transition tuple, and **`land_to_air`
appears in none of them**. The `land` island's only transition is
`water_to_land` — arriving on land, never leaving it. So the search has been
given a take-off ladder, a take-off factor in `mission_fraction`, and two gates
on it, while **no island's objective asks a machine to depart from the ground**.

And hybridisation cannot fill the gap. `Archipelago.migrate` crosses the three
specialists pairwise — air×water, air×land, water×land — then:

    for dst in ("generalist", "amphibian", "aerial_diver"):
        if dst in self.archives:
            out.append({...}); self.hybrids += 1
            break

The `break` fires on the first destination that exists, and `generalist` always
exists, so **every hybrid ever made has gone to `generalist`**; `amphibian` and
`aerial_diver` have received none. Confirmed by count: arch36 logged 156
migrations and 39 hybrids over 13 migration events — 3 per event, exactly the
three pairs, all to one island.

So the air×land cross — a walking flyer, the thing this whole line of work
wants — is built every migration and is only ever scored on an island that also
demands water competence.

**Do:** add a `land_air` island (`domains: ("land", "air")`, transitions
including `land_to_air`); put `land_to_air` into the `land` island's transitions;
and route each hybrid to the island that matches its parents rather than to the
first one in a tuple. These are three small changes to `islands.py` and they are
prerequisites for D below meaning anything.

### B. Depth: the water score has the same defect the air score had

**`max_depth >= 0.5 m` is true of 100% of evaluations, in every band of both
runs.** Not "insufficient" — saturated. The cause is one line:

    SPAWN[Domain.WATER] = (-8.0, 0.0, -4.0)

The water segment **releases the machine four metres under the surface**. This
is the same defect as `SPAWN[Domain.AIR]` being a 30 m launch — the finding that
set arch35's entire work list — and nobody applied it to water.

At a threshold inside the distribution, the capability is real and growing:
depth ≥ 10 m ran **7.8% → 21.9%** over arch36 and median depth 4.89 → 6.48 m.
Water competence 0.596 → 0.641.

And the continuous mission reports **max depth 0.0 m**, because
`run_continuous` has one placement. A water competence of 0.64 and a mission
depth of 0.0 m are both true; they measure different things, and only the second
is the mission.

**Do:** the same treatment take-off got. Either measure depth as a *gain* over
the spawn depth, or score a water segment that begins at the surface. Until
then, no water number on any chart is evidence a machine can get wet. The report
now says this in section 7e and its comparison chart uses 10 m.

### C. Mass is free, energy is the only thing pushing back

Measured over arch36's 12,338 evaluations:

| | p10 | median | p90 | max |
|---|---|---|---|---|
| mass (kg) | 3.35 | 5.38 | 8.31 | **66.76** |
| wing_area (m²) | **0.000** | 0.261 | 0.868 | 4.39 |
| wing_loading (N/m²) | 70.6 | 218.6 | **300,314** | 1,604,419 |
| battery (Wh) | 175.7 | 260.0 | 270.8 | 574.2 |
| energy_margin | −0.95 | **−0.165** | +3.95 | +56.6 |

Three things fall out:

- **There is no mass cap anywhere in the codebase.** 66.76 kg was reached.
- **The median design cannot power its own mission** (`energy_margin` −0.165).
- **The search is not buying battery.** Mutation clamps `battery_wh` to
  [10, 2000] Wh and the p90 is 271 Wh — it is nowhere near the ceiling. Either
  the mass cost of a battery is over-modelled relative to its benefit, or
  nothing in the score rewards carrying one.

So the asymmetry the question proposed is real, but not as "energy too tight":
energy is the *only* term that pushes back on structural growth, while mass and
part count are free. The wing-loading p90 of 300,000 N/m² says at least a tenth
of the population is carrying a token surface it could never fly on.

**Do:** measure before changing anything. (i) Sweep `battery_wh` against
`energy_margin` and `mission_fraction` on fixed morphologies — if a bigger
battery is strictly better and the search is not taking it, the operator is at
fault, not the model. (ii) Then decide between charging for parts (the standing
item) and capping mass. Do not do both in one arm.

### D. Score `land_to_air`, after A

Unchanged in substance from the last two lists, and now explicitly blocked on A:
scoring the transition is pointless while no island's objective contains it.

### E. Everything else carries forward

`descriptor_refit_every` (F, still unfixed and still needing its own arm), a
tested resume path and a memory ceiling, identification at 67% of an evaluation,
`np.cross` at 16%, charging for parts, and nothing rewarding travel toward the
water. See the arch36 list above.

### F. A contact-fraction gate on take-off

arch36 left one segment in 3,247 scoring 4.16 m with a lifting surface, holding
posture, and **touching the ground for 77% of the segment**. A departure means
leaving, not visiting. `contact_fraction` is already measured. Not added
mid-arch36 because a third gate would have destroyed that arm.

---

## What arch37 measured

900 generations, 24.0 h, 14,092 evaluations, 82 s/gen, seed 20260901, same pool
shape as arch35 and arch36. Clean exit — no OOM, no `memory_stop`, no traceback.
Two changes, A and B, measured by quantities that do not overlap.

**The first launch was void and is kept in `runs/arch37_void_missing_liftmargin/`.**
Two of the three exits from the air branch never published `lift_margin`, and a
missing metric stops `rung_reached` where it stands, so 50 of 180 air segments
scored rung 0 whatever their airframe could do. The short-hop exit is the path
every falling design takes — free fall from the 30 m spawn is 2.5 s, 31% of an
8 s segment, under that branch's 35% bar. Caught fifteen minutes after launch by
checking the published measurement rather than the code that was edited.

### 1. A worked, and it is the clearest result this project has had

| | arch36 archive (gen900) | arch37 archive (gen900) |
|---|---|---|
| `lift_margin` median | 0.333 | **1.377** |
| share >= 1.0 (can hold itself up somewhere) | 27.4% | **58.0%** |

Endpoint to endpoint, elite to elite. Capability over *evaluations*, which no
archive bookkeeping can move:

| gens | glides+ | holds_height+ | holds_station+ |
|---|---|---|---|
| 0–99 | 1.3% | 0.4% | 0.1% |
| 400–499 | 4.8% | 2.2% | 0 |
| 800–899 | **12.6%** | **6.8%** | 0 |

`glides` and above rose **9.7x**, `holds_height` and above **17x**, and both
accelerated in the last third. A plateau called at gen600 off one flat band was
wrong.

The mechanism is visible in the split between the two populations: over the same
bands the archive share at `lift_margin >= 1.0` rose 55.1% -> 62.0% while the
*candidate stream* fell 64.9% -> 60.8%. Mutation degrades lift and always did;
what changed is that selection can now see it and keep the flyers.

### 2. The wall is `holds_station`, and the reason is not what the ladder assumes

Reached once, in band 0–99, and never again in the remaining 12,400 evaluations.

The first explanation — that the population lacks thrust, because among gliders
the height-holders differ from the sinkers in `flap_hz` (2.690 against 2.200,
the beetle seed default) and not in `lift_margin` (2.336 against 2.355) — is
**refuted by the run's own data**. Over 740 glider segments (`lift_margin >= 1`,
`|sink| < 3`):

    corr(station_keeping, flap_hz)  -0.201     corr(., span) +0.015
    corr(station_keeping, mass)     -0.143     corr(., dof)  -0.189

Higher flap frequency scores *lower*, and the best band is 0–1 Hz — the gannet
seed value. `station_keeping` is uncorrelated with everything the archive
records, sitting at a median of 0.05–0.08 across every band of every body
variable.

**What it is instead: the flight paths are not straight.** Take the 173 segments
that hold height on the ladder's own measure (`lift_margin >= 1`, `|sink| < 0.5`)
and ask what a straight-line descent at that same sink rate would have scored,
from the definitions in `triphibian.py` — band `max(0.25 * span, 0.5)` m, late
window at most 4 s:

| | median `station_keeping` | share >= 0.6 |
|---|---|---|
| straight descent at the measured sink | 0.704 | 59.5% |
| **actually measured** | **0.062** | **0.58%** |

A tenth of what the geometry allows. These machines return near their starting
height by the end of the window while leaving it by more than a quarter of their
own span in between — and `sink_rate`, which is the difference between the two
endpoints of the late half, cannot see any of it.

So `holds_height` is being cleared, in part, by a trajectory that goes down and
comes back. **This is the fourth time a score has paid for uncontrolled motion**
— after the 30 m/s wingless launch, the tumble scored as a turn and the rebound
scored as a take-off — and it is the first time the gate was already in place:
`station_keeping` refuses it correctly. The defect is in the rung *below* it.

**What is missing is one altitude trace.** Every statement above is an inference
from summary telemetry, because the run stores no trajectory and `showcase` does
not load the shared PPO snapshot the run flew with (it trains a fresh controller
or loads a pickle), so the flown path cannot be reconstructed offline. That is
the arch38 item: publish the excursion, do not infer it.

### 3. B is live, contributes, and is never crossed

`land_air` finished with 73 elites, second of seven, and **produced the
mission-best design of the run**. All seven islands took 18+ promotions evenly,
so the hybrid routing fix works and the island's objective is reachable from the
seeds. And `transition:land_to_air` carries a mean PPO reward of **+0.0025**
against `water_to_land`'s +0.7130 and `water_to_air`'s +0.1847.

Scored, wanted, essentially never achieved. **Making the search want something it
cannot do does not make it happen** — which is the answer to the arch36 list's
item D, and it is not the answer that list expected.

### 4. The mission is unchanged for a fourth run — but the machine is not

`showcase --by mission` over 188 elites: fitness 0.9945, `mission_fraction`
0.2892, `land_air` island, 8.53 kg, 2.64 m span, **aspect ratio 16.7**, density
ratio 0.74, 20 dof.

    on-task 32%   transitions 0/2   max depth 0.0 m
      leg 1 land   96%     leg 2 air   0%     leg 3 water   0%

arch34, arch35, arch36 and arch37 all film **0/2 transitions, 0.0 m depth**.

What changed is the body. arch36 filmed a dense lump — aspect ratio 0.3, 0.18 m
span, density ratio 4.65. arch37 films a glider. **The airframe problem is being
solved and the mission problem is not**, because the mission's air leg begins
wherever the land leg left it and nothing can take off.

`mission_fraction` **is not comparable across the arch36 -> arch37 boundary**:
the air ladder went from 7 rungs to 11, so air competence sits at a different
fraction of its ladder and the product that uses it moved with it.

### 5. F, unchanged as designed

34 refits, peak 100, final 65 (−35%). arch36: peak 121, final 79 (−35%). The
same relative decline for the third run running. Nothing was done to
`descriptor_refit_every` and nothing changed.

---

## arch38 — the work list

Ordered by what the evidence supports, not by what would be satisfying. The
headline is that **A solved the problem it was aimed at and revealed the next
one**, and the next one is a measurement defect, not a hardware limit.

Two items that looked like the obvious arm are withdrawn here rather than
deferred — B on three measurements and H on one — because a list that only ever
grows stops being a record of what the evidence supports.

**The arm this list recommends is no arm.** A costs nothing and changes no
selection, so a run carrying only A is simultaneously the excursion measurement
and a *replication of arch37 at a second seed* — and arch37's headline (lift
rungs doubling the flyable share, glides up 9.7x) is currently n=1. Every other
item here is either blocked on what A returns or, in C's case, not yet designed
to a state where it can be run without destroying the gradient arch37 built.

### How long to run it — 900, read at 500, extend if it is still climbing

Asked as "900 -> 1000 or 900 -> 500?" and answered from what the last two runs
did per generation rather than from what a longer run ought to buy.

| | glides+ gain already present at gen 500 | slope, gens 300–599 | slope, gens 600–899 |
|---|---|---|---|
| arch36 | 38.4% | +0.67 pts/100gen | +1.00 pts/100gen |
| arch37 | **31.3%** | +0.67 pts/100gen | **+2.15 pts/100gen** |

arch37's `glides+` ran 1.31% -> 4.83% at gen 500 -> **12.57%** at gen 900. **The
most productive third of the run is the last third, in both runs, and arch37's
was still accelerating when it stopped.** Cutting to 500 generations discards
roughly two thirds of the result and specifically the part that compounds.

But that is only true of *capability* measurements. The two kinds separate
cleanly:

| what the arm measures | decidable by | evidence |
|---|---|---|
| a distribution over candidates | **gen 100–200** | arch37's `lift_margin >= 1` share ran 64.9% in band 0–99 and 63.4% in band 800–899 — flat all run, settled in the first band |
| a capability | **not by 900** | `glides+` still accelerating in the final band |

So neither 500 nor 1000 is a decision that has to be made in advance.
`--resume` extends a finished run — `tests/test_search.py` runs 3 generations,
resumes with `generations=6`, and checks that exactly 3 new evaluations happen —
and the checkpoint bug that would have told a resumed run there was nothing left
to do is fixed. **Run 900, read the report at 500, and extend to 1000+ only if
the capability bands are still climbing.** At arch37's late slope the 100
generations from 900 to 1000 cost 2.7 h (11%) and would be the single most
productive hundred of the run.

### A. Publish the altitude excursion. Costs nothing, decides everything below

`_score_segment` already holds `clearances` and the `late` index window. It
publishes `sink_rate` (the two endpoints) and `station_keeping` (occupancy of a
band) and throws away the shape between them. Add, ungated, next to
`measured_sink_rate`:

- `altitude_excursion` — `max|clearance − ref|` over the late window, in metres
- `excursion_ratio` — the same divided by the station band, so it is comparable
  across a 0.5 m machine and a 3 m one

This is the `lift_margin` pattern exactly: a quantity the code already computes
and discards, published as a diagnostic with no rung on it, so the question is
answered by the next run for free. **Do not put a rung on it in arch38** — a
threshold set before the distribution is known is the mistake this project keeps
writing down.

It settles, in one run, whether the population is porpoising (excursion ≫ band
with sink ≈ 0), stalling and recovering, or whether the inference in §2 above is
wrong and something else produces those numbers.

### B. Make `flap_frequency` searchable — **withdrawn, on three measurements**

Kept here with its evidence because it was the obvious next arm and it is not
one. Three things were checked before proposing it and each weakened it further.

**1. The premise is half gone. The search already moved the parameter.** Share
of evaluations sitting on exactly a seed default (2.20 or 0.60 Hz):

| | arch36 | arch37 |
|---|---|---|
| exactly a seed default | 75.4% | **47.7%** |

and arch37's most common single value is **2.690 Hz at 24.3%** — a mutated value
that took a quarter of the population. **The operator mix was identical in both
runs.** What changed was the ladder. The parameter was never unsearchable; it
was unselectable, because nothing scored what a better frequency buys.

**2. The bandit's credit says the suppressed operators are better, and a second
instrument says they are not.** Over arch37's 31,139 selections:

| | tries | bandit lifetime | bandit windowed | accepted into archive | mean `mission_fraction` |
|---|---|---|---|---|---|
| 13 parameter operators | 9,388 | 0.0873 | 0.0724 | **30.26%** | 0.00241 |
| 8 structural operators | 20,463 | 0.0485 | 0.0386 | **30.34%** | 0.00175 |

The bandit's own credit separates the two kinds cleanly and with no overlap —
every parameter operator outranks every structural one, `global_energy` included
at 0.0900 against a best structural 0.0531. And the archive **accepts them at
the same rate to within 0.08 points over 31,000 trials.**

So `structural_bias` is not costing the search acceptance, and the thing that
needs explaining is why the bandit's reward and the archive's decision disagree
this completely. That is a finding about the credit signal, not a licence to
retune the tilt. **Do not change `structural_bias` on the strength of the bandit
numbers alone** — they are the instrument under suspicion.

**3. It was never the explanation of the wall anyway.** §2 above: station keeping
correlates −0.201 with `flap_hz`.

**What survives as work:** find out why bandit credit and archive acceptance
disagree. It is offline, it costs nothing, and until it is answered every
operator-mix decision is being made on a number that predicts nothing.

### C. The air spawn — a real distortion with no safe fix yet designed

**Not an arm until someone designs it, and the naive version is known to fail.**

The distortion is real: **no air number in this project is evidence a machine got
airborne on its own**, and it is why `land_to_air` can be scored, wanted and
never crossed — the air segment does not need the transition, so nothing ever
practises it.

But "move the spawn to the ground" is the fix that was already tried and
reverted, and the measurement is in `triphibian.py` beside `SPAWN`: the old
6 m release gave 1.1 s of free fall inside an 8 s segment, capping
`airborne_fraction` — which multiplies every air term — at 0.15 no matter how
well the machine flew. Measured 0.136 to 0.173 across all five body plans, with
air scores of 0.035 to 0.044. **The ceiling was the height of the drop, not
aerodynamics.** Grounding the spawn now would put that ceiling back and take the
gradient arch37 just built with it.

So the machinery for a ground-up path already exists and is not the air segment:
the `takeoff` ladder, its two arch36 gates, and the `land_to_air` transition.
They are all in place, all scored, and the answer arch37 gave is that **making
the search want something it cannot do does not make it happen.** Which makes
this a capability question, not a scoring one — and A is the measurement that
says which capability is missing.

**Do:** wait for A. Then design a segment that starts on the ground *without*
making airborne fraction the binding term — most likely by scoring the take-off
attempt on its own ladder, as take-off already is, rather than by moving where
the air segment begins.

### D. The water spawn — carried forward unchanged, and now the oldest item here

`SPAWN[Domain.WATER] = (-8.0, 0.0, -4.0)` releases the machine four metres under.
`max_depth >= 0.5 m` is therefore true of 100% of evaluations in every band of
every run. Measure depth as a *gain* over the spawn depth, the way
`takeoff_height` is a gain over resting clearance. Until then no water number is
evidence a machine can get wet, and the report's comparison chart uses 10 m to
say something at all.

### E. Item F, `descriptor_refit_every` — three runs, three identical declines

−35%, −35%, −38%. It needs its own arm and has been deferred three times.

### F. Mass is free; the battery sweep is written and unrun

`runs/probe_battery.py` exists and has never been run. It answers whether a
bigger battery is strictly better on fixed morphologies — if it is and the search
is not taking it, the operator is at fault and not the energy model. Only then
decide between charging for parts and capping mass, and not both in one arm.

### G. Retired: the contact-fraction gate on take-off

Listed in the arch37 work list and **withdrawn without being implemented.** In an
8 s window `contact_fraction` measures how *quickly* a machine leaves, not
whether it left: a machine that departs at t=6 s scores 0.75 and one that departs
at t=2 s scores 0.25, and both departed. The gate would have punished the slower
departure and called it a failure to depart. Revisit only against a window long
enough for the distinction to exist — the offline probe that was to decide the
window length failed three times and was abandoned, its one usable output being
that tripling the window costs 25%, not the 1.35x measured without identification.

### H. On changing the actuator model

Raised externally: stage the work as flapping actuator -> fixed-rig lift test ->
autonomous take-off -> water -> land, and stop trying to be triphibian first.

The staging argument is the same one C and D make and it is right. The actuator
change is **not yet supported by evidence**. The case for it was that the ladder
had reached the limit of static airframe lift and the next rung needs thrust —
but §2 shows `holds_station` is not gated by thrust in any way the data can see,
and A above is the cheap measurement that would tell us. Changing the actuator
model now would rebuild the physics on the strength of an inference that the
run's own correlations refute.

Order: publish the excursion (A), read it, then decide. If the excursion says
the machines are oscillating at a frequency nothing in the body predicts, that
is a control problem. If it says they stall and recover, that is the actuator
model, and *then* the change is supported.

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
