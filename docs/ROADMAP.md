# Roadmap

Written 2026-09-03 after arch33, and revised the same day once every phase was
built. Every item names the measurement that motivates it; nothing here is on
the list because it seemed like a good idea, and nothing is marked done without
the number it produced.

Revised 2026-09-19, after arch38 ran. Everything below is kept for the
measurement that motivated it; **the current work list is
[arch40](#arch40--the-work-list)**; arch39's list is kept below it, and
[What 2026-09-20 fixed](#what-2026-09-20-fixed-and-what-is-not-comparable-across-it)
is the boundary across which nothing is comparable.

Two things to read before proposing anything.

[What arch38 measured](#what-arch38-measured) §3 and §6: rewarding thrust moved
the thrust distribution **down**, because the rung at the floor was cleared by
flapping that does nothing — the fifth time a score has paid for absent motion;
and arch38's replication failure showed that "the changes must be measured by
quantities that do not overlap" protects attribution but **not selection
pressure**, which is shared and finite.

[Thrust needs a joint move](#thrust-needs-a-joint-move-and-every-gait-operator-is-axis-aligned),
measured after arch38: thrust is reachable in the model and unreachable by this
search, because every variation operator moves one gait coordinate and the target
needs four moved together. That is a search defect, not evidence about the
physics, and it un-promotes item H.

arch37 is still the run that made flight visible to selection — the share of the
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

## Why nothing flies, measured a second time — and the answer is thrust

Six questions asked on 2026-09-09, each answered from arch37's 14,092
evaluations and its 182 final elites rather than from opinion. The last one
changes the diagnosis in "What arch37 measured" §2 and supersedes it.

### 1. Nothing in this project has ever measured thrust

The air ladder reads `lift_margin` (the airframe's **static** lift, fixed
attitude, no flapping), `airborne_fraction` (where the machine was), `sink_rate`
(how fast it came down) and `station_keeping` (whether it stayed). Sustained
flight is thrust >= drag. **In four runs and 14,092 evaluations no measurement
anywhere read whether the flapping produces net forward force.**

`thrust_margin` is the same quantity for propulsion that `lift_margin` is for
lift — `(<Fx>_cycle − Fx_static) / drag` at the machine's own trim speed and
pitch, so 1.0 is "the gait produces the airframe's entire drag" and the machine
holds speed instead of trading height for it. Quasi-static, the same class of
estimate `lift_at` already is, 0.12 s per phenotype, cached.

Measured over arch37's 182 final elites at their own gaits:

| | |
|---|---|
| median | **−0.0030** |
| p90 | +0.0354 |
| maximum | **+0.2390** |
| share >= 0.10 | 3.3% |

**Every hand-built seed plan is in the same place.** The gannet reads −0.0001;
the teal, a 5 Hz flapper, reads **−0.3796** — its flapping produces 38% more
drag than holding the wings still. Correlation with `lift_margin` is +0.242, so
this is not the question the ladder already asks: 58.0% of the archive can hold
its own weight up and 3.3% can push itself along.

**And positive thrust is reachable**, which is what makes this a rung rather
than a wall. A random search over 400 gaits per body:

| plan | at its own gait | best of 400 | at |
|---|---|---|---|
| gannet | −0.0001 | **+0.8466** | 10.95 Hz |
| teal | −0.3796 | +0.4633 | 7.78 Hz |
| beetle | +0.0231 | +0.1731 | 4.50 Hz |
| bat | −0.0074 | +0.1171 | 0.78 Hz |

The gaits that make thrust run at **4.5–11 Hz with the joints spread across the
cycle** — a travelling wave. The population sits at 2.2 Hz, the beetle seed
default, and `corr(flap_hz, thrust_margin)` over the archive is **+0.009**
because the whole archive is inside the dead zone.

**So the chain is complete.** The population found gliders: 58% hold their
weight statically, 12.6% glide, 6.8% hold height. A glider converts height into
speed and comes down. `holds_station` and `climbs` are exactly the rungs that
need thrust, and thrust was never in the score — which is precisely where lift
was before arch37, when 72.6% of the archive could not lift its own weight and
nothing said so. **This supersedes the "the paths are not straight" reading of
the `holds_station` wall.** That inference may still be true and arch38's
excursion diagnostic still tests it, but it is downstream of this: a machine
with no thrust cannot hold station however straight its path.

### 2. GRPO — the right algorithm, and not yet the binding constraint

The observation already carries eight body-identity channels
(`MORPHOLOGY_DIM = 8`: mass, density ratio, wing area, span, aspect ratio, wing
loading, actuated count, battery), so "the critic cannot see the body" — the
usual argument for a group baseline — does not apply here.

What is true is that the reward is **one terminal scalar spread over ~2,000
steps** by GAE, and the shared policy has measured as contributing nothing four
times running (−0.0014 ± 0.0016 in-distribution). A group-relative advantage
over several rollouts of the *same body* would cancel the body's contribution
exactly and leave only "did this sampled action sequence beat this body's other
attempts", which is the only signal that can teach the policy anything.

**But it is not the binding constraint, and §1 says why: no controller can be
rewarded for producing thrust when nothing measures thrust.** Put the signal in
first; then GRPO has something worth approximating. Its cost is G rollouts per
body, so a partial form — a handful of bodies per generation given G=4
learning-only rollouts — is the affordable version when it is time.

### 3. Topology growth is half productive

Medians, first band to last, with the within-band correlation against reaching
`glides`:

| | gen 0–99 | gen 800–899 | corr (within band) |
|---|---|---|---|
| `n_parts` | 4 | 9 | **+0.191** (last band +0.29) |
| `dof` | 8 | 15 | +0.119 |
| `cppn_complexity` | 18 | **73** | +0.060, flat all run |

Parts are the strongest predictor of flight in the whole body vector and growing
them is reasonable. **CPPN complexity quadrupled and buys almost nothing** — and
`dof` drives identification, which is 67% of an evaluation, so the drift is paid
for in wall time every generation.

### 4. Wing morphology is being evolved in the direction that does not matter

**`aspect_ratio` correlates +0.004 with reaching `glides`** — and that is not a
whole-run artefact: within every one of the nine generation bands it sits
between −0.02 and +0.05. Over the same run it climbed **8.47 → 14.76, +74%**.

What does correlate is `wing_area`, at +0.102, and **its correlation rises with
the population** (+0.06 in the first band to +0.21 in the last). `wing_area` went
0.215 → 0.230 — **+7%, essentially flat.**

**The search grew the dimension that does nothing and left flat the one that
buys.** No scoring defect has been identified behind this, so nothing is changed
for it directly; the prediction is that `thrust_margin` fixes it as a side
effect, because thrust scales with the area being flapped and nothing before now
rewarded area for its own sake. **That is a prediction arch38 tests, not a
result.**

### 5. Sparsity is in the mission and the transitions, not in the ladders

Share of arch37's evaluations at each ladder's rung 0, and where the rest sit:

| ladder | rung 0 | shape |
|---|---|---|
| air | 9.8% | spread across rungs 1–6 — healthy, this is what arch37 bought |
| water | 0.2% | **79.3% piled on rung 2** — the spawn handed out the first two |
| land | 27.3% | 42.9% on rung 3 |
| transition | 29.1% | **70.6% on rung 1, 0.3% above it** |

And `mission_fraction` has a median of **0.000300** against a maximum of 0.2992
— three orders of magnitude, a needle. The segment ladders are not the sparse
part and have not been since arch37.

### 6. The rewards conflict, specifically, and the conflict deepens

Within-band correlations, so the shared time trend is removed:

| | air competence | water competence |
|---|---|---|
| `wing_area` | **+0.076** | **−0.195** |
| `density_ratio` | −0.086 | **+0.518** |
| `mass` | +0.137 | −0.085 |

**A wing costs two and a half times more in water than it buys in air.** And the
competences themselves diverge as the run proceeds — `corr(air, water)` runs
+0.04, +0.05, +0.04, −0.05, −0.06, −0.05, −0.10, −0.16, **−0.17** across the nine
bands, monotonically. **The more the population specialises, the more the two
objectives fight.** That is the triphibian premise showing its cost, and the
`land_air` island — added in arch37, second of seven, source of the run's
mission-best — is the shape of the answer: pairs, not the whole.

Separately, `corr(mission, structure) = +0.703` over the three objectives, so
two of the three are largely one.

---

## arch38 — the work list

Ordered by what the evidence supports, not by what would be satisfying. The
headline is that **A solved the problem it was aimed at and revealed the next
one**, and the next one is a measurement defect, not a hardware limit.

Two items that looked like the obvious arm are withdrawn here rather than
deferred — B on three measurements and H on one — because a list that only ever
grows stops being a record of what the evidence supports.

**Revised after the one-arm rule was challenged, and the challenge was right.**
The rule exists so an effect can be attributed. Across arch34–arch37 exactly one
run produced a large effect, so most of the time there is nothing to attribute
and the rule buys a day for nothing. What actually has to hold is the weaker and
more useful condition arch37 already used: **the changes must be measured by
quantities that do not overlap.** That scales past two.

So arch38 carries four changes with four separate measurements, and the ones
that could not clear that bar were **measured and dropped rather than bundled**
— a mass cap (§F), a longer segment window (§I), flap-frequency searchability
(§B) and the actuator model (§H) are all refuted below, three of them by
measurements taken while assembling this list.

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

### J. `thrust_margin`, and three rungs on it — **done, unrun.** The largest item on this list

The measurement and its distribution are in "Why nothing flies, measured a
second time" §1 above. Published on **all three** exits of the air branch, like
`lift_margin` and for the same reason — a rung reads it, and a metric published
on one exit of three is what voided arch37's first launch.

Three rungs, inserted between `holds_height` and `holds_station`:

| rung | at | leaves standing |
|---|---|---|
| `flaps_forward` | 0.00 | ~49% — flapping is not a net cost |
| `makes_thrust` | 0.05 | 7.7% |
| `pushes_itself` | 0.15 | ~2% |

**Above `holds_height`, not below it, and that placement is the whole design.**
Everything below can be earned by a glider, so putting thrust at the bottom
would cap 58% of the archive and destroy the gradient arch37 built. Everything
above — holding a station, climbing — genuinely requires thrust, and those rungs
have been reached once and never in 14,092 evaluations. So the change costs
nothing that exists and adds a direction where the population is stuck.

No rung is placed above +0.239, the largest value ever measured, even though the
gait sweep reached +0.8466. The headroom is real; naming a rung in it before
anything is near would be the `moves`-at-0.1-m/s mistake.

**The reach is 6.8% of the population initially** — only designs that already
hold height see these rungs. That is the honest cost of the ordering, and
arch37 grew `holds_height+` seventeen-fold in one run, so the audience grows.

### K. The transition ladder, rebuilt from its own distribution — **done, unrun**

70.6% of arch37 sat on rung 1 and 0.3% above it, for three reasons all visible
in the measurements:

1. `crossed_fraction` counts how many of four transitions were made, so it takes
   five values. A rung at 0.34 is "two of four" (70.9%) and the next at 0.99 is
   "all four" (0.3%). **Three-of-four — 17.1% of the population — had no rung.**
2. **Completeness was demanded before quality.** Every quality rung sat above
   `crosses_all`, so a machine that crossed two boundaries beautifully scored
   the same rung as one that crossed two badly.
3. **Two quality rungs were set at or above the population's maximum.**
   `arrives_usable` asked for `exit_state >= 0.7` and the largest value ever
   measured is **0.689** — a rung nobody can stand on. `stays_controlled` asked
   for `control >= 0.6` against a p90 of 0.296.

Eight rungs now, every threshold inside the measured distribution, shares
decreasing monotonically: `crosses` 70.9%, `stays_controlled` (0.20) 56.4%,
`arrives_usable` (0.30) 48.2%, `arrives_well` (0.37) 27.0%, `crosses_three`
(0.60) 17.1%, `crosses_efficiently` (0.75) 17.1%, `survives_entry` (0.50) 16.4%,
`crosses_all` 0.3%.

**Transition rungs are not comparable across arch37 → arch38.**

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

### D. Depth as a gain — **done, unrun.** The water ladder had the air ladder's defect

`SPAWN[Domain.WATER] = (-8.0, 0.0, -4.0)` releases the machine four metres under,
and the **minimum `max_depth` over arch37's 14,058 water segments is 3.34 m**.
So the water ladder's first two rungs — `submerges` at 0.5 m and `dives` at
3.0 m absolute — were cleared by **100% of every evaluation this project has
ever run, by being dropped.** That is precisely the defect the air ladder had,
where `leaves_surface` and `stays_up` were paid for falling, and it **outlived
the fix to that one by three runs** because nobody looked at the water column
after fixing the air one.

`depth_gain = max_depth − depth_at_release` is now published, and the ladder and
the water headline bar both read it. Thresholds from the measured gain
distribution (p25 +0.16 m, median +2.23 m, p75 +5.45 m, p90 +8.11 m):

| rung | at | leaves standing |
|---|---|---|
| `submerges` | 0.25 m | 70.8% |
| `dives` | 2.0 m | 51.9% |
| `reaches_depth` | 5.0 m | 28.8% |
| `goes_deep` | 8.0 m | 10.4% |

The gain is floored at zero by construction — a maximum cannot fall below the
first sample — so a machine that only rises sits at rung 0 instead of scoring
the same rung as one that dove nine metres. `max_depth` is still published
unchanged, so the record stays honest and every old number stays re-derivable.

**One correction, caught by the test rather than by review.** Those quantiles are
`max_depth − 4.0` against a release randomised by 0.2 m, which produced apparent
negative gains that the real definition cannot have. The claim "15.9% float up"
was that artefact; the lowest rung is approximate to about the jitter, and
arch38 measures the gain directly.

**Water rungs and water competence are not comparable across arch37 → arch38.**
The ladder went from 5 rungs to 6 and three of them changed metric.

### E. Item F — **the telemetry was measuring one island. Fixed, unrun.**

Deferred three times on a number that could not answer it. In `loop.py` the
`descriptor_refit` event fired **outside** the per-island loop, so `stats` held
whichever island happened to be iterated last. Every "the archive shrinks 35%
across a run" claim in this file — arch35, arch36, arch37 — is one island out of
six or seven, and the archipelago totals were never recorded at all.

What the final archives actually hold: **arch36 428 cells, arch37 480 cells**
across all islands, against the 79 and 65 those runs reported.

And the loss is not what it looked like in a second way: **parents are drawn
from `archive.fronts`, not `archive.cells`** (`curator.py`: "Drawn from the
cells' full Pareto fronts"), and `rebin` re-populates fronts while cells
collapse. arch37 ended at 1.36 front members per cell, so the buffer is thin —
but a merged elite is not necessarily gone, and the old event could not tell the
difference.

The event now carries archipelago totals, a per-island breakdown, and the front
population. **This is a measurement fix, not a change to the refit** — the
interval stays at 400. arch38 is the first run that can say what item F costs.

### F. Mass and battery — **both halves answered offline. Nothing to implement.**

**The battery half.** `runs/probe_battery.py` was finally run: four body plans ×
nine pack sizes from 60 to 2000 Wh. `energy_margin` rises monotonically with
pack size for three of four plans and peaks at 1400–2000 Wh for all four — so a
bigger pack *is* strictly better on the constraint. But the best
`mission_fraction` sits at 260, 60, 900 and 900 Wh, nowhere near it. **The
mission is not energy-limited, so battery mass is bought for nothing and the
search refusing it is correct.** (The mission column is 0.00004–0.00056 and
cannot be ranked; the monotonicity result rests on the margin column alone,
which is clean.)

**The mass half — a cap is refuted.** Set from arch37's distribution rather than
from what a flying machine ought to weigh:

| mass | n | `lift_margin >= 1` | `glides+` |
|---|---|---|---|
| 3–5 kg | 5,704 | 65.6% | 2.21% |
| 5–8 kg | 4,357 | 64.2% | 5.53% |
| **8–12 kg** | 2,279 | 58.0% | **15.93%** |
| 12–20 kg | 476 | 41.2% | 8.82% |
| > 20 kg | 34 | 17.6% | 0% |

**The gliders are the heavy machines.** A 10 kg cap removes 14.9% of every
glider in the run. The only cap that costs no gliders is 20 kg, which touches
0.24% of the population — a change that does nothing. Mass being uncapped is not
what is stopping flight.

### G. Retired: the contact-fraction gate on take-off

Listed in the arch37 work list and **withdrawn without being implemented.** In an
8 s window `contact_fraction` measures how *quickly* a machine leaves, not
whether it left: a machine that departs at t=6 s scores 0.75 and one that departs
at t=2 s scores 0.25, and both departed. The gate would have punished the slower
departure and called it a failure to depart. Revisit only against a window long
enough for the distinction to exist — the offline probe that was to decide the
window length failed three times and was abandoned, its one usable output being
that tripling the window costs 25%, not the 1.35x measured without identification.

### I. A longer segment window — **done, and its first launch was void**

**arch38's first launch ran 4.3 h and 1,601 air segments before it was killed,
and it is kept in `runs/arch38_void_window_gate/`.** The window change was
correct and the thing that read it was not.

`_score_segment` gated episode measurements on
`np.sum(airborne) * timestep < 0.35 * res.duration` — **a fraction of a
configurable window**, exactly the defect this item is about, one level below
where the item was looking. At 8 s it meant 2.8 s; at 24 s it silently meant
8.4 s. Result: **1,571 of the first 1,601 air segments took the short-hop exit
and 24 took the full one**, so `glides`, `holds_height`, the three new thrust
rungs, `holds_station`, `climbs` and `manoeuvres` — nine of fourteen — were
unreachable for 99.6% of the population. And the share was *falling*, 2.4% to
0.4% over the first hundred generations, because selection could not see the
rungs above the gate and stopped paying for staying airborne.

**The rule that failed here is worth writing down.** "The changes must be
measured by quantities that do not overlap" protects *attribution*. It does not
protect against *interference* — one change disabling another's measurement.
After bundling, the second question is: for each change, what else reads the
thing it moved?

Two gates are now absolute durations, `MEASURABLE_AIR_SECONDS = 2.8` and the
0.4 s never-airborne branch, both being what the old fractions meant at an 8 s
segment so nothing measured before moves. `submerged > 0.5` in water and
`contact` on land were checked and left: those genuinely ask "was it there for
most of the episode", which is a fraction by intent.

Whether a measurement is *taken* no longer moves with the window. How long the
machine had to survive still does, via `airborne_fraction` and the two rungs
that read it, which is the point of the longer segment.

### I, part two — the second launch was void too, and the item is **withdrawn**

`runs/arch38_void_ladder_inverted/`, 6.2 h and 2,220 evaluations. The gate fix
held — the full measurement path ran 65–75%, `thrust_margin` was published with
zero omissions — and the ladder came apart one rung higher up.

`stays_up` reads `airborne_fraction >= 0.60`. At a 24 s segment from a 30 m
launch that is 14.4 s aloft, **a sink rate under about 2.08 m/s** — and the rung
directly above it, `glides`, asks for `sink_rate < 3.0`. Measured over the run's
1,588 full-path segments:

| | share |
|---|---|
| `airborne_fraction >= 0.60` (rung 5) | **0.06%** |
| `sink_rate < 3.0` (rung 6, above it) | **9.8%** |

**The lower rung is 160x harder than the one above it.** `rung_reached` stops at
the first unmet rung, so 1,418 of 2,220 evaluations recorded rung 5 and one
recorded rung 6 — and the three thrust rungs at 8, 9 and 10 had an audience of
zero. The run could not measure its own headline item.

The ladder's contract is that it is a progression. **A fractional threshold
inside a ladder is a threshold whose difficulty moves with a command-line flag,
and at 3x the window it moved past the rung above it.** That is the third form
of this defect found in one day, after `station_keeping` and the measurability
gates, and it is the one that survives being careful about the first two:
`airborne_fraction` *should* be a fraction, because it is the survival question
— the mistake was raising the window while leaving a rung ordered against a
different one.

**Decision: the segment goes back to 8 s and this item is withdrawn.** The
alternative — making the airborne rungs absolute at 0.8 s and 4.8 s, their
meaning at an 8 s segment — restores the order, but then the longer window
affects nothing in the ladder at all and buys only a longer diagnostic tail for
**+74% wall time** (measured: 143 s/gen against arch37's 82, where the estimate
had been 102). That is not worth a day.

**What survives, and it is the durable part:** every gate that decides whether a
measurement is *taken* is now an absolute duration — `MEASURABLE_AIR_SECONDS`,
the 0.4 s never-airborne branch, `STATION_WINDOW` — so the segment length can be
raised in a later run without any of this recurring. Revisit the window when
there is a population that stays airborne, and re-derive the `airborne_fraction`
rungs from that population's distribution at the same time.

### I (original). A longer segment window — after the first answer here was wrong

This item was first written as "refuted": lengthening the window makes
`station_keeping` harder, because a drifting machine has more time to leave the
band, so it raises the wall it was meant to lower.

**That is an argument for not measuring the problem.** If a longer window
produces more drift, the drift was already there and the short window was
hiding it; a design that holds height for four seconds and not for twenty is not
holding height. The correct target is a machine that works over an unbounded
duration, and every previous fix on this list has the same shape — the 6 m air
drop capped `airborne_fraction` at 0.15, the water spawn handed out 4 m,
`sink_rate` reads two endpoints. **Each time the ceiling was the measurement,
not the machine.**

What the objection was really detecting is a genuine defect, and it is worse
than "harder": **`station_keeping` is not a capability threshold at all, it is a
sink-rate threshold that depends on the window length.** For a steady descent at
v it evaluates to `band / (v · T)`, so the rung's 0.6 asks for

| measurement window | sink rate `holds_station` actually demands |
|---|---|
| 4 s (today's 8 s segment) | ≤ 0.21 m/s |
| 12 s (a 24 s segment) | ≤ 0.069 m/s |

One rung name, two physical requirements, decided by a command-line flag. That
breaks this ladder's own contract that a rung means the same thing on day one
and day five, and the 24 s version would sit a factor of three beyond anything
arch37 reached — the "`moves` at 0.1 m/s left 61.6% of the population with
nowhere to stand" mistake, again.

**The fix separates the segment from the measurement window.** A long segment
asks two questions and they now have two answers:

- *did it survive* — `airborne_fraction`, which gets harder as the segment
  grows, which is the entire point of growing it;
- *did it hold a height* — `sink_rate` and `station_keeping`, measured over a
  fixed `TriphibianEnv.STATION_WINDOW` of 4.0 s regardless of segment length.

4.0 s is what those statistics implicitly used at an 8 s segment, so **nothing
measured before this moves and every arch37 number stays comparable.** Verified:
a 0.4 m/s descent scores station 0.376 at 8 s and 0.375 at 24 s.

And `altitude_wobble` is added beside `altitude_excursion` — the maximum
deviation from the *fitted trend* over the whole airborne stretch, in metres and
in band units. `sink_rate` carries the trend and the wobble carries what is left,
so the two are orthogonal and neither grows with the window: a 2.0 m oscillation
reads 2.24 m at both 8 s and 24 s, while a straight 0.4 m/s descent reads
0.000 m of wobble against 1.60 m of excursion.

**arch38 runs at `--segment-seconds 24`.** Identification does not scale with
segment length, so the measured cost is +24.5% per design — about 102 s per
generation against arch37's 82, or 25.5 h for 900 generations. `energy_required`
is computed from `spec.seconds_per_domain`, not from the segment, so the energy
term does not move with it.

It would help the excursion diagnostic see an oscillation period, and that is
not worth 8% on its own. Revisit once A says what the paths look like.

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

## What arch38 measured

900 generations in 24.3 h, 83 s/gen, 14,066 evaluations, Tier-0 rejection 3.2%,
seed 20260908. Report at `runs/arch38/report.html`, full record in
`runs/arch38_notes.md`, which also holds the two void launches that preceded it.
Every comparison below is computed for arch38 **and arch37 by the same code** —
`runs/verify_arch38_predictions.py` and `runs/probe_thrust.py` — because the two
runs' ladders differ and only identical computation is comparable.

| | prediction, written before the run | verdict |
|---|---|---|
| **A** | excursion published, `wobble_ratio` above 1.0 | **confirmed, and not close** |
| **D** | water rung occupancy spreads | **confirmed — the run's real result** |
| **J** | thrust share moves off 7.7% | **falsified; it moved down** |
| **K** | transition occupancy spreads off 70.6% | **confirmed** |
| **E** | archipelago near 480 cells | works; the predicted number was wrong |
| **repl.** | arch37's headline reproduces | **failed, and it cannot be attributed** |

### 1. A — nothing has ever held an altitude, and now that is measured

53.6% of air segments carry the metrics. `wobble_ratio` median **9.59** over all
7,390 published, 22.1 for gliders and **25.4 for height-holders**, against a
prediction of 8–10 from the arch37 arithmetic. `excursion_ratio` median 36.0.
**100.0% of published air segments are above 1.0, in every subset**, and the
report's `steady_path` and `steady_flight` series read **0.0% in all 18 bands**:
across 900 generations not one air segment kept its altitude inside the station
band it is judged against.

So the arch37 inference was right and nothing below A is blocked. It also
settles §H: what the paths do is now measured rather than inferred.

### 2. D — a gradient was the missing thing in water, exactly as it was for lift

Marginal shares at the four gain rungs went **39.9 / 15.0 / 4.0 / 0.6%** early to
**70.2 / 58.2 / 30.7 / 12.1%** in generations 800–899. The share gaining 5 m went
**4.0% → 30.7%**. arch37 had **99.8% clearing the first two rungs by being
dropped** 4 m under; occupancy is now spread across all six.

The early shares also refute the prediction that produced them — 71 / 52 / 29 /
10% was computed from `max_depth − 4.0` on arch37 and overstated the true gain.
That is the reason arch38 measures the gain directly.

### 3. J — falsified in the opposite direction, and the mechanism is visible

Elite to elite, the same probe on both archives: the share at
`thrust_margin >= 0.05` went **7.7% → 1.8%**, and the archive maximum **fell from
+0.2390 to +0.0905**. Over evaluations the share rose to 5.6% by generation 300
and **decayed back to 2.0%** by 899.

What moved is the rung at zero. `>= 0.00` went **28.0% → 45.4%**, and the share
within ±0.001 of *exactly* zero went 10.7% → 16.6% of evaluations and 9.3% →
18.3% of elites. `flap_hz` median rose 1.80 → 3.50 Hz, but only 9.4% sits in the
4.5–11 Hz band where the gait sweep found thrust, and
**`corr(flap_hz, thrust_margin)` is −0.129 over evaluations and −0.149 over
elites** — the prediction required it to be positive.

`thrust = <Fx>_cycle − Fx_static`, so **a wing that barely moves reads exactly
0.0 and clears `flaps_forward`.** The population converged on the cheapest
clearance and the two rungs above it lost their audience. **This is the fifth
time a score has paid for absent or uncontrolled motion**, and the rule this file
keeps re-deriving says the answer is a gate, not a coefficient: ask what the
measurement reads for a machine that is not flapping.

### 4. K — occupancy spread, and `land_to_air` moved the way nothing predicted

Occupancy 24.2 / 38.5 / 7.2 / 11.7 / 13.3 / 0.0 / 0.2 / 4.9 / 0.06% against
arch37's 70.6% on rung 1 and 0.3% above it. Nobody is stuck at
`crosses_efficiently`, which is worth a look before the next rebuild.

`transition:land_to_air` PPO reward, measured the same way for both runs:
**+0.0066 → +0.0143** weighted mean, nonzero on **88/900 → 181/900** updates,
last hundred updates +0.0185 → +0.0363 — roughly double on both measures against
a prediction of "probably still no". It is still 5% of `water_to_land`'s +0.70.

### 5. E — the telemetry works and retires a claim this file repeated

Archipelago totals are **660 cells median before a refit, 891 at most**, not the
~480 predicted. Merge loss per refit is **19.0% median, 11.1–31.2%** for the
archipelago. **Every "the archive shrinks 35%" claim in the earlier sections of
this file is one island out of six or seven** — read them with that correction.
The interval is unchanged at 400; this run only made the item measurable.

### 6. Replication failed, and the bundle is why it cannot be attributed

`glides+` in generations 800–899: **arch37 12.57% → arch38 2.69%.** The
measurement-based form agrees (`sink_rate <= 3.0` among published, 24.55% →
12.86%), so this is not a rung-index artefact. `lift_margin >= 1.0` among
published: **63.4% → 31.7%.** PPO air reward is flat across arch38
(+0.0185 → +0.0194) where arch37's quadrupled (+0.0110 → +0.0452).

Seed variance or a cost of the bundle **cannot be separated from this run.** D
gave water a real gradient for the first time and water PPO reward rose
+0.4123 → +0.5843 while air went flat: a fixed evaluation budget moved to the
domain that had just been given somewhere to climb.

**So the non-overlap rule needs its second amendment.** The four changes were
checked for non-overlapping *measurements* and they were non-overlapping. Nothing
checked that they did not compete for **selection pressure**, which is shared and
finite. That is the same class of error as the first void launch — "what else
reads the thing it moved" — one level up, at selection rather than at
measurement. Answering it needs an arm that changes water alone, or arch37's seed
re-run under arch38's code.

### 7. What arch38 does not license

- **Not comparable across arch37 → arch38:** water rungs and water competence,
  transition rungs, air rungs above `holds_height`, and therefore
  `mission_fraction`. `airborne_fraction` and its two rungs are not comparable
  either, because the third launch ran at 8 s after two void launches at 24 s.
- **H, the actuator model, is not promoted by this run.** arch38's own notes read
  its pre-registered criterion as promoting H, and the gait measurement of
  2026-09-13 below withdraws that: thrust is unreachable *by this search* because
  the variation operators are axis-aligned, which is a search defect and not
  evidence about the physics. H stays unsupported until a joint gait move has
  been tried.

---

## What is set by measurement, and what is typed

Asked on 2026-09-13: which weights in this project are fabricated — a number
with no measurement and no physics behind it. The answer is a short list, and it
is worth separating from the much longer list of numbers that *are* derived,
because lumping them together is how a real finding gets ignored.

**Recorded, not changed.** Twelve coefficients and a handful of normalisers
cannot be derived from anything; they decide what the search optimises. Changing
them all at once would redefine selection pressure with no way to attribute the
result, so they are written down here and left alone. If they are ever to be
touched, the cheap first step is a sensitivity pass: perturb each one ±50%
against a stored archive offline and report how far the elite ordering moves.
What nothing depends on can be deleted; what matters has earned a derivation.

### The blends

| where | the coefficients |
|---|---|
| air competence, `triphibian.py` | `0.55·flight + 0.25·glide + 0.20·station` |
| water competence | `0.35·reached + 0.25·hold + 0.2·submerged + 0.2·upright` |
| land competence | `posture = 0.5·contact + 0.5·upright`, then `0.65·progress + 0.35·climb` |
| `mission_fraction`, `evaluate.py` | the exponent in `min(competences) ** 0.5` |

Each carries a comment explaining which terms were removed and why — real
reasoning about the *shape* — and nothing at all about the numbers.

### The normalisers, which set what "full marks" means

- `flight = 1 − sink/1.5` — 1.5 m/s
- `glide = 1 − sink/SINK_BALLISTIC`, and `SINK_BALLISTIC` is
  `LAUNCH_SPEED_RANGE[0]` = 6.0 m/s: **a launch speed reused as a sink-rate
  normaliser**, which is the clearest case on this page of a number doing a job
  it was not measured for
- `progress = mean_speed/0.6` — 0.6 m/s
- `climb = climbed/0.5` — 0.5 m
- `GATED_AIR_CREDIT = 0.05`, the `0.10` brief-hop credit, `TAKEOFF_FULL = 0.30`,
  `TAKEOFF_FLOOR = 0.05`
- `mission_weight = 0.3`, `reward_shaping = 0.2`, `descriptor_bins = 5`
- the curator's regime table: `structural_bias` 1.6 / 1.5 / 0.7 / 2.2 / 2.6 with
  `n_mutations` 2 / 2 / 1 / 3, and the four triggers that select between them —
  `len(cells) < 20`, `cov_growth > 0.004`, `qd_growth > 0.01`,
  `feasible_frac < 0.15`
- the UCB exploration constant `c = 0.6`

### What is *not* on this list, and should not be added to it

Every ladder rung, each of which carries the share of the population it leaves
standing — the lift rungs (72.1/51.4/36.9/27.4%), the depth-gain rungs
(70.8/51.9/28.8/10.4%), the thrust rungs (36.2/21.2/7.5%), the re-derived land
rungs (48.8/35.0/31.2/15.0%) and the rebuilt transition rungs. `CL_MAX = 1.8`
(a post-stall flat plate), `MAX_WING_LOADING` (derived from `CL_MAX` and the
launch band), `MAX_SPIN_RATE = 2π` (one revolution a second),
`TAKEOFF_WINGLESS_CAP = 0.29` and `TAKEOFF_POSTURE_BAR = 0.7` (both from
measurement), `MEASURABLE_AIR_SECONDS = 2.8` and `STATION_WINDOW = 4.0` (each
the value its predecessor fraction had at an 8 s segment, chosen so nothing
measured before moved). And `PLATEAU_ALPHA`, whose own comment records that it
replaced two typed numbers precisely to stop this.

---

## The batched and single-machine evaluators disagree in water

Found while making the cross-path agreement permanent (`tests/test_search.py`,
"the batched and single-machine paths agree"). With `identify_axes=False` the two
paths agree to about 2e-6 — floating-point summation order — in all three
domains. With identification on, **land and air still agree to 2e-6 and water
does not**: `depth_error` differs by 0.024 m and `water_speed` by 0.029 m/s on
the two plans tested.

`batchroll` asserts in a comment that it reuses "`basis_from_probes` the
unbatched path uses, so the fitting is untouched". That is true of air and not of
water, where the solver carries slam and wake history and the two paths clear it
differently — `bf.reset_slam()` against `solver.reset()`.

The residual is millimetres of depth error and centimetres per second of speed,
well under the first water rung at 0.25 m of gain, so it is bounded in the test
rather than treated as urgent. But it means **the water basis a score was earned
against is not quite the one a verification reproduces**, and that is the class
of thing that hid a factor of sixty on land until this week.

**Do:** compare `identify_batch` and `identify_mobility` probe-for-probe on one
body in water, and find which history the batched path is not clearing.

---

## The `--min-shard` default is still the documented trap

`CLAUDE.md` records that `--min-shard 8` at batch 16 makes a single Tier-0
rejection collapse the pool to one shard running in the parent, and that two
generations in three ran that way before it was found. The measured optimum is 4.
**The default in `SearchConfig` is still 8**, so the documented trap is what an
invocation without the flag gets.

## Thrust needs a joint move, and every gait operator is axis-aligned

Measured 2026-09-13, before arch39, and it supersedes the reading that arch38's
thrust failure was about flap frequency.

**1. The frequency band where thrust exists is already populated.** The gait
sweep found positive thrust at 4.5–11 Hz and the population's median sits near
2.2, which looked like the explanation. It is not: **8.91% of arch38's
evaluations are already at `flap_hz >= 4.5`**, p99 is 6.89 Hz and the maximum is
12.61 Hz. `random_genome` draws `uniform(1.5, 8.0)`, so the random seeds land in
the band without any drift being needed.

Drift could not have got there, which is worth recording separately.
`flap_frequency` moves only through `mut_global_energy`, which touches it on one
roll in four, by `x * exp(normal(0, 0.22))`. Measured over arch38: that operator
fires on 1.94% of applications, so **0.0113 frequency touches per child**. Walked
with no selection, a lineage of depth 10 starting at 2.2 Hz reaches 4.5 Hz in
**0.00%** of 20,000 trials; depth 100 reaches it in 0.65%. Analytically 2.2 →
10.95 Hz is 7.3 sigma of log step, about 53 touches, which is 4,689 children in
one lineage.

**2. So frequency is not the binding constraint, and one variable at a time is.**
Sweeping each gait coordinate alone from the seed default, 250 draws each, best
`thrust_margin` reached:

| | gannet | beetle | teal |
|---|---|---|---|
| default gait | −0.0001 | +0.0228 | −0.3765 |
| frequency alone | −0.0001 | +0.1210 | −0.1490 |
| amplitude alone | −0.0001 | +0.1447 | +0.0030 |
| phase alone | +0.0003 | +0.0230 | −0.0725 |
| offset alone | +0.0000 | +0.0296 | −0.0931 |
| **all four together** | **+0.7693** | +0.1658 | **+0.6916** |

On two of the three plans **no single coordinate produces any thrust at all and
all four together produce two thirds of the airframe's drag.** The earlier sweep
that found +0.8466 varied all four at once and the result was attributed to
frequency because frequency is what got printed.

**3. Every gait operator moves one coordinate, and most move it on one part.**
`mut_global_energy` changes the global `flap_frequency`, and only on one roll in
four. `mut_stroke` changes `stroke_amplitude` on **one movable part**.
`mut_phase_gradient` shifts phases. Nothing in `MUTATION_OPERATORS` moves
frequency, amplitude, phase and offset together across the actuated set.

So thrust is reachable in the model and unreachable by this search, and the
mechanism is now named: **the variation operators are axis-aligned and the target
requires a joint move.** No ladder can fix that, which is why arch38's rungs
failed, and it is not evidence about the actuator model either — ROADMAP item H
should be un-promoted until this is tried.

**The precedent is in this repository.** `mut_drivetrain`'s docstring reads
"Together, because separately neither is worth anything" — spring stiffness and
gear ratio, bundled for exactly this reason, and the principle was applied there
and nowhere else.

### What arch39 should carry

**A `mut_gait` operator that resamples the whole gait at once** — frequency,
per-part stroke amplitude, phase and rest offset across the actuated set, drawn
the way `random_genome` draws them for a fresh design rather than jittered from
the parent. Its measurement is the `thrust_margin` distribution, which does not
overlap anything else on the arch39 list.

**And it is pre-registerable, which arch38's thrust arm was not.** Monte Carlo
the new operator offline against `thrust_margin` on the seed plans and record,
before the run, what fraction of draws reach positive thrust. If that fraction
is near zero the operator is wrong and no run is needed to find out; if it is
appreciable, the run has a stated number to be judged against.

## arch39 — the work list

Revised 2026-09-19, after arch38 ran and after the gait measurement above.
Status against the source was audited the same day; every "not implemented"
below was checked in the code, not read off this file.

**The headline is S, and it is the only item here that has a pre-registered
prediction.** Everything else is either cheap and open, or waiting on a
distribution S is about to move.

### S. `mut_gait` — a joint gait move. **Not implemented; the arch39 arm**

The measurement is §"Thrust needs a joint move" above: thrust is reachable in the
model and unreachable by this search, because every variation operator is
axis-aligned and the target needs all four gait coordinates moved together. Audit
of `MUTATION_OPERATORS` (`dytiscidae/core/genome.py:1017`) confirms it — three
operators touch gait and each moves one coordinate:

| operator | file:line | what it moves |
|---|---|---|
| `mut_global_energy` | `core/genome.py:984` | global `flap_frequency`, on one roll in four |
| `mut_stroke` | `core/genome.py:521` | **one** movable part: amplitude *or* rest offset, one of four branches |
| `mut_phase_gradient` | `core/genome.py:499` | `phase_offset` only |

`mut_gait` resamples frequency, per-part stroke amplitude, phase and rest offset
across the actuated set **at once**, drawn the way `random_genome` draws them for
a fresh design rather than jittered from the parent. The precedent is in this
repository: `mut_drivetrain`'s docstring reads "Together, because separately
neither is worth anything."

**Pre-register before the run, not after.** Monte Carlo the operator offline
against `thrust_margin` on the seed plans and record the fraction of draws
reaching positive thrust. **If that fraction is near zero the operator is wrong
and no run is needed to find out**; if it is appreciable, the run has a stated
number to be judged against. The single-coordinate sweep already gives the
baseline it must beat — gannet −0.0001, beetle +0.0228, teal −0.3765 from the
default gait, against +0.7693 / +0.1658 / +0.6916 when all four move together.

Its measurement is the `thrust_margin` distribution, which overlaps nothing else
on this list.

### T. The thrust rungs are gated by a coefficient, and the gate is sitting unread

Half done. After arch38, `thrust_margin()` returns `None` rather than `0.0` when
it cannot be measured (`envs/triphibian.py:806`), `rung_reached` treats `None` as
failing (`evolution/judge.py:273`), and the three rungs were re-derived from
arch38's own 80 re-scored elites to 0.002 / 0.010 / 0.020
(`evolution/judge.py:140`). That removes the "unmeasurable scored as measured
zero" defect, which was the worst of it.

**What is still a coefficient is the bar itself.** 0.002 is a number chosen so
that a wing which barely moves falls below it. This file's own rule — restated
six times, most recently by `land_speed` and `slope_climbed` moving one percent
with the actuators switched off — is that the answer is a **gate**, not a
coefficient. And the gate already exists: `flap_travel()`
(`envs/triphibian.py:778`) is computed and written into `res.measurements` at
four call sites and **read by nothing** — `grep flap_travel evolution/judge.py`
is empty. The land rungs have their held-still control (`judge.py:187`); the
thrust rungs have none.

Cheap, and it belongs with S because S is the thing that will finally push
designs at those rungs.

### U. `min_shard` still defaults to 8 in all three places

`evolution/loop.py:90`, `envs/actors.py:166` and the CLI at `ops/run.py:681` all
default to 8. CLAUDE.md and §"The `--min-shard` default is still the documented
trap" above both record that at batch 16 a single Tier-0 rejection makes
`15 // 8 = 1`, one shard, and the pool silently runs in the parent with the
workers idle — two generations in three before it was found — and that the swept
optimum is 4. `adapters/trainers/search.py:51` documents the trap in a comment
without changing the number. Change the default; the flag stays.

### R. `shared_ent_coef` — the estimator is fixed, the coefficient is not swept

Half done, and the half that is done is the hard half. The entropy term is now a
real estimator (`learning/ppo.py:203`, sampling from π_new) rather than the cross
entropy H(π_old, π_new) whose expected gradient at the on-policy point is exactly
zero — measured **−0.00013 ± 0.00188 (t = −0.07)** against **+0.43114 ± 0.00106
(t = +409)**. `experiments/ppo_estimators/run.py` re-derives both sides and needs
no GPU.

**The default is still 0.01, which is the value that was measured to do nothing.**
`experiments/ppo_estimators/` reproduces the diagnostic, not a tuning sweep. At
the fix, 0.01 buys +0.0113 on the learned `log_std` against the bonus off and 0.1
buys +0.1048. Pick it from a sweep before it rides along in a run, and note that
exploration is **not comparable** across the boundary.

Two findings travel with it and are closed: the KL `target_kl` stops on was
negative on 20.0% of minibatches (now k3), and the shared policy's initial
weights were not reproducible from `--seed` at all (‖dW‖ = 7.4). Full audit in
`docs/LEARNER_AUDIT.md`.

### V. The replication failure needs its own arm, and it is not S

arch38's `glides+` came in at 2.69% against arch37's 12.57% at matched
generations, and §6 of "What arch38 measured" says why it cannot be attributed:
the bundle's four changes had non-overlapping *measurements* but competed for
**selection pressure**, which is shared and finite. D gave water a gradient and
water PPO reward rose while air went flat.

The arm that answers it is **arch37's seed (20260901) re-run under arch38's
code**, changing nothing else. It needs no new code, which makes it the cheapest
thing on this list and the only one that can tell a seed effect from a bundle
cost. It does not overlap S: S is judged on `thrust_margin`, this on `glides+`
and `lift_margin`.

## What 2026-09-20 fixed, and what is not comparable across it

arch39 was stopped at generation 200 by the user ("39 does not need to
continue; fix everything that should be fixed first"), and the day's work is
below. **Nothing measured before today is comparable with anything measured
after**, for two independent reasons: the GPU physics changed (it had been
frozen five weeks behind its source) and every medium's competence was
redefined.

### 1. The GPU kernel was five weeks older than its source — the largest of them

`mojo/build/full_pipeline.so`, which `evaluate_tier1_batch` scores with, was
built 2026-08-04. `mojo/src` was changed 2026-09-15 by F-05, which replaced the
stall blend: a logistic `1/(1+exp(-(|a|-stall)/6deg))` in the binary against a
compactly supported smoothstep `t^2(3-2t)` over `stall+16deg` in the source and
in `fluid.py`. So the search scored with one physics while Tier-2 verification,
every probe and the showcase used another.

`test_the_two_evaluation_paths_score_the_same_machine_the_same` exists to catch
this and could not run: the driver mismatch fixed the same morning made it one
of seventeen crashes. On the stale binary it fails; rebuilt:

    air + land agree              worst 1.040857, 2.001011  ->  0.000000, 0.000002
    all three, no identification  worst 0.171108, 0.573743  ->  0.000000
    water with identification     the known residual, 0.05023, still open

`tests/test_gpu_mirror.py` passed throughout, because it compares the two
*sources*, which agreed. Nothing compared the running binary to its source.
`pixi run build-all` — named by every "build it with" message in the Python half
and absent from `pixi.toml` — now builds both extensions and writes a manifest
of source hashes; `batchroll.usable()` refuses a stale kernel; the checkpoint
records which kernel scored the run.

**arch39's scores were produced by the 08-04 kernel and cannot be reproduced
with today's code.** Its own record is internally consistent.

### 2. Competence is what the machine did beyond doing nothing

The inert probe, through the scored path, on the seed plans:

| medium | own gait | actuators held still | still / own |
|---|---|---|---|
| water | 0.510 | **0.533** | **104%** |
| air | 0.029 | 0.032 | 112% |
| land | 0.048 | 0.010 | 20% |

A still machine sinks — the eel gains +4.45 m held still against +4.52 m
flapping — and depth gained, depth held, being submerged and being upright all
paid for it. Item D had replaced absolute depth with depth *gained*, removing
the free 4 m of the release and not the free metres of gravity: the air
ladder's "paid for falling", in water, and the **seventh** instance of the
recurring lesson.

Every segment now runs a twin from the same state with every joint held at the
angle it starts at, and scores the difference. **Motion is netted** — forward
distance and the medium's motion terms. **State is a gate** — airborne,
submerged, upright, posture, served multiply, because netting a state would
score a stable hull 0 and a tumbling-but-corrected one 1, i.e. reward being
passively unstable. A twin that does not survive leaves no baseline and the
segment scores 0, so blowing up when held still cannot erase your own control.

**Forward distance carries 0.6 of every medium** (the user's decision: "forward
distance matters more"), where air and water had no distance term at all. Full
marks is one rule across the three: land keeps 0.6 m/s, which is the 0.993
quantile of its own net-speed distribution over 140 arch39 elites driven as they
were scored; water and air take the same quantile of theirs — **0.45 and
6.2 m/s**. Measured net speeds:

    land   p50 +0.024  p90 +0.168  p99 +0.416  max +7.283 m/s
    water  p50 +0.007  p90 +0.191  p99 +0.437  max +0.575
    air    p50 -0.028  p90 +1.739  p99 +5.611  max +9.348

The twin leaves no trace: the scored run is bit-identical with and without it on
both evaluators in all three media. Held still, every plan scores exactly
0.000000 where the eel's gross water score is 0.6301.

### 3. No operator can go dormant

`OperatorBandit.select` is a deterministic argmax over `(mean + UCB) x tilt`,
and every regime but `refining` multiplies the eight structural operators by
1.5–2.6. Measured over arch39 generations 60–200, 2,060 mutated children:
structural operators took **68.9%** of operator slots against a uniform 36%, and
`gait` made **0.7%** after making none from generation 58 to 141. An operator
outside the top slots can hold a good windowed mean and never be tried again,
so its window never updates. A 0.2 exploration floor gives every operator at
least ~2% of children and leaves the structural share near 62%.

### 4. Smaller, with their measurements

- **A MuJoCo auto-reset was a teleport the continuous mission never saw.** On a
  bad acceleration MuJoCo resets the state and keeps stepping; `run_continuous`
  checked only for non-finite or >400 m, which a reset never trips. Injected at
  step 400 it ran all 1,500 steps reporting `survived=True`. Scored segments
  were already gated on `bad_qacc`; the thrust and lift rigs never step, so
  `thrust_margin` could not be hit. 0.15% of arch39's scored rollouts diverged.
- **A checkpoint named whatever HEAD was at write time**, not the code running:
  arch39 launched from 09bab3b and its generation-200 checkpoint says ca05812.
- **`showcase --island` chose among the elites that survived the cross-island
  merge** (34 of the air island's 78), and could not rank by the island's own
  domains at all — so "the air island's best" was a land machine.
- **`min_shard` defaulted to 8**, the documented trap, in all three places.
- **`batchroll.AVAILABLE` was an import-time answer**, so a driver mismatch read
  as available and turned three suites' skips into seventeen failures.

### 6. The ladders stay on the gross measurements, and that is a decision

The curriculum's stage 1 -- which is *selection* -- read `sink_rate`,
`depth_error` and `land_speed` gross, so it paid for sinking and for gliding
from the 30 m launch. It now reads `sink_reduction`, `depth_error_reduction`
and `land_speed_net`, scaled by the p90 of each over the same 140 elites:
6.2 m/s, 0.54 m and land's own 0.4 m/s (whose p90 is 0.214, left where it was so
land does not move under this change).

**The judge's ladders were left on the gross quantities**, deliberately.
Measured over those elites:

    water.depth_gain       p50 +1.421  p90 +7.318  max +11.163 m
    water.depth_gain_net   p50 -0.128  p90 +0.618  max  +3.306 m

    rung           gross occupancy   net occupancy
    >= 0.25 m          67.1%            15.0%
    >= 2.00 m          45.0%             1.4%
    >= 5.00 m          18.6%             0.0%
    >= 8.00 m           7.1%             0.0%

**The median machine dives less than gravity alone would.** Moving the rungs
onto the net quantity at these thresholds would empty the ladder, and choosing
thresholds that keep the old occupancy would put the first rung at a *negative*
net gain -- paying for sinking again, under a new name. Setting a bar that
leaves most of the population with nowhere to stand is the error this file
records at `moves` (0.1 m/s left 61.6% of arch34 below it).

So the ladders remain the cross-run comparability anchor and `rung_reached` in
water is still clearable by sinking. What that costs: rung telemetry, the
judge's within-rung bar and the scout's features are inflated in water. What it
buys: arch34–arch39 rung numbers stay readable. **The right fix is not a
threshold change, it is a population that can dive under power** -- which is
what forward-distance-dominant, net-of-passive competence is now selecting for.
Re-derive the water ladder from arch40's distribution, not from arch39's.

### 5. T — the thrust rungs need no motion gate. Closed by measurement

`flap_travel` is computed and read by nothing, and the plan was to gate the
thrust rungs on it. Measured over 3,224 arch39 air evaluations: **every**
evaluation clearing `flaps_forward` has `flap_travel >= 0.104 rad`, and the 127
designs under 0.05 rad clear nothing, with a maximum thrust of exactly +0.0000.
Thrust needs stroke velocity, so the post-arch38 threshold at 0.002 — strictly
above the floor — already does the gate's work, and a travel gate would penalise
small-amplitude high-frequency flapping, which is real thrust. **Do not add it.**

---

## What 2026-09-21 changed: the passive twin is gone, and water is gated

The first arch40 launch was stopped by the user at generation 54 (2.31 h), and
its data is in `runs/arch40_stopped_passive_twin`. Two reasons, both the
user's:

1. **Cost.** Every segment ran twice. Measured: an 8 s water segment costs
   2.084 s with the twin against 1.047 s without, x1.99, and a whole generation
   ran at 117 s against arch39's 85 and arch38's 90 at the same generations
   (gen 7-60 median). `85 + 32 = 117` puts segment rollout at ~38% of a
   generation.
2. **It scored the wrong thing.** Netting forward distance against a twin says
   passive forward motion is worth nothing. The user's decision: a glide, or a
   hull that carries itself through water, *is* a capability and the learner
   should be free to find it. **Sinking is the thing that must be chosen** —
   "in water a machine must at least go forward or hold a position".

### What the run measured before it was stopped

Over its 985 evaluations, scored through the real path with the policy driving
and a twin of the same design held still beside it:

| medium | passive competence, median | what it means |
|---|---|---|
| water | **0.422** | the defect |
| air | 0.002 | already gated |
| land | 0.006 | already gated |

**Water was the only medium paying for doing nothing**, and two fifths of that
0.422 was the constant `0.2 * submerged + 0.2 * upright` in the additive water
formula — paid for being *released* four metres under and for being a stable
hull. That is the §W leak seen from the other side: water was the cheapest
medium in the run, so 96.4% of the air island's elites were better in water
than in air.

Also measured, and it refuted the first fix that was tried: over 56 elites and
seed plans, driven against actuators-still, **depth station keeping alone does
not separate them** (0.506 against 0.414), nor does horizontal speed (0.160
against 0.153 m/s), nor depth error (5.171 against 5.349 m). The twin is the
only instrument that ever separated driven from still in water. What a gate can
do instead is stop the *state* from paying and require the motion to be chosen.

### The change

- **The passive twin is removed** from both evaluators, with `passive_control`,
  `_net_of_passive`, `competence_gross` and every `*_net` measurement.
- **Water competence is `gate * active * motion`**, where `gate` is
  `served * submerged * (0.5 + 0.5 * upright)` — state multiplies now, it does
  not add — `motion` is `0.6 * headway + 0.4 * (0.35 reached + 0.25 hold)/0.6`,
  and **`active = max(headway, depth_station_keeping)`** is the user's rule: a
  machine that neither makes headway nor holds a depth scores zero however deep
  it gets.
- **`depth_station_keeping` and `depth_excursion_ratio`** are published, the
  same measurement the air branch makes on height, for the same reason: a
  machine sinking straight through the target scores well on `depth_error` and
  a brick earns `depth_gain` by being dense.
- **`FORWARD_REF_SPEED["water"]` is 0.79 m/s**, the 0.993 quantile of the gross
  water speed distribution over the stopped run's 985 segments (median 0.112,
  p90 0.291, p99 0.700). The old 0.45 was that quantile of the *net*
  distribution and would have paid the median 0.25 for drifting.
- **Curriculum stage 1** reads `station_keeping / 0.17` (air), `water_headway`
  (water) and `land_speed / 0.4` (land) — gross quantities that are each gated
  or shaped, rather than the net forms that went with the twin. The scales are
  each quantity's own 0.999 quantile over the stopped run, which is where
  land's 0.4 m/s already sat.

### What it is worth, measured

50 designs from the stopped run's archives plus 8 seed plans, 8 s segments,
base gait against actuators held still:

| medium | driven | actuators still | still ÷ driven |
|---|---|---|---|
| air | 0.0032 | 0.0031 | 97% |
| water | **0.0944** | **0.0628** | **66%** |
| land | 0.0366 | 0.0049 | 13% |

Water's still score falls **7.2x** (0.451 -> 0.063 on the 56-design set) and the
water-to-land gap closes from 11.6x to 2.6x. It does **not** fall to zero, and
nothing without a twin makes it: 66% is what a gate can do, and the honest
statement is that in this population a still machine in water is still worth
two thirds of a driven one. Air is untouched by this and remains ~0 for nearly
everything, which is the older "nothing flies" problem and not this one.

Four mutations hold it: `water-scores-being-underwater`,
`sinking-opens-the-water-gate`, `water-station-keeping-cannot-fail` and
`stage-one-reads-gross-measurements`. All four are caught; the last one survived
its first version because the test fixture happened to make the mutant produce
the same number, which is the mutation harness earning its keep.

**Nothing before 2026-09-21 is comparable with anything after it**, including
the 54 generations of the stopped launch.

## What arch40 measured

600 generations in 16.44 h (9,433 evaluations, 85 s/generation), seed
20260921, on the gated water score of 2026-09-21 morning. Report
`runs/arch40/report.html`; notes `runs/arch40_notes.md`.

1. **The water leak closed, most of the way.** Median competence water/land was
   22.8x in arch39 at matched evaluations and 2.7x here. The §W share of the
   land island's elites better in another medium went 96.4% → 50.0%; the air
   island's 96.4% → 82.9%, which is mostly scale — air competence is 0.003 at
   the median for the whole population, so "better elsewhere" is nearly
   automatic.
2. **The operator floor works, and continuity was the thing to measure.**
   `gait` was 5.6% of children against arch39's 4.2% — indistinguishable — but
   arch39 ran eight consecutive 20-generation blocks with none (gens 60–199)
   and arch40 ran none in 600 generations.
3. **The population flattened in the second half.** Competence medians over
   gens 150–249 and 500–599: air 0.003/0.003, water 0.090/0.100, land
   0.035/0.037. The curator called 47% of gens 300–399 `stagnant`. Archive
   elites kept improving; the population did not.
4. **The mission did not move.** Filmed by mission, driven as evaluated: 0/2
   transitions, 0% of the water leg on task. Water `holds_depth` cleared by 0%
   of the last 1,500 evaluations.

What it licenses: the gate fixed the cross-medium leak it was built for, and
nothing about the reward's structure below that — the controller was still told
where it was and never what to do. Item X is the next change, and arch41 its
baseline.

## 2026-09-22 — every open item, started

The user: "Roadmap事項全部開始." Every item below is started today. What keeps
that from repeating arch38 — a bundle whose result nobody could attribute — is
that **anything that changes selection lands behind a flag, default off**, so
the next run (arch42) is a clean baseline for the code as it now stands, and
each flag is an arm of its own afterwards. Status is verified against the code,
not taken from the headings below (two of which were stale).

| item | what "started" means | status |
|---|---|---|
| Z film = evaluation | pool fix, trim fix, clearance fix, promotion record, `viz/film.py`, automatic `postrun` | **done** |
| U `min_shard` defaults | already 4 in all three places (`loop.py:95`, `actors.py:171`, CLI) | **done** — heading was stale |
| S `mut_gait` | implemented (`genome.py:549`) with the operator floor; needs only its measurement, which arch42 carries | in arch42 |
| X task reward | hold depth 5.0–6.5 m and land command 0.04 m/s, from arch41's stop; the stillness band **stays 0.30 m/s** — at arch41's p75 (0.64) it paid a held-still beetle 0.093 in water, because that population's drift *is* passive sinking | **done** |
| Y chatter | `command_rate`, `command_reversal` published on every segment; `action_rate_penalty` (default 0) charges the rate | **done**, flag off |
| Y / O continuity | measured first (below): a continuous start would be a wall today | **designed**, not built |
| M complexity | `runs/analysis_M_complexity.md`: do not charge — it is the air/water conflict (+0.139 air, −0.280 water), not bloat | **closed** |
| L aspect ratio | `runs/analysis_L_aspect_ratio.md`: no cost — AR buys static lift (+0.153) that never becomes flight | **closed** |
| P refit | `runs/analysis_P_refit.md`: each refit erases more than islands grow between refits; subspace overlap now recorded, `descriptor_keep_if_overlap` (default 0) keeps unchanged axes | **done**, flag off |
| R `shared_ent_coef` | a short sweep, after arch42 (it needs the machine) | queued |
| N GRPO | the partial form, flag | after X/Y |
| W, T, V | superseded by X / closed / subsumed | closed |
| Q triphibian conflict | a written decision for the user: what "a chain of pairs" would change | memo |
| TEST_AUDIT 7 | re-run every mutation whose only catching suite is `test_search`, on the fixed harness | queued |

### Y / O — what a continuous start would do, measured before building it

`runs/_logs/probe_continuity.py`: arch41's top 30 elites, each in three leg
orders, 8 s legs, driven as scored, never re-placed. **Of 90 continuous
missions none completed both transitions and 25 (28%) completed one — nearly all
of them air → water, the 30 m launch falling into the sea.** No machine got from
land into water or into air under its own power. Making the evaluation
continuous today would zero every leg after the first for almost everyone: the
"moves at 0.1 m/s" wall again, at the scale of the whole mission.

The design that has a gradient: **move the start of each transition probe back
from the interface as the population learns to cross it.** The four transition
probes already start at the interface and score the crossing. Give each a start
distance — 0 at the waterline, the shore or the launch height — that steps
further back (up the beach, deeper, lower) whenever a set share of evaluations
cross from the current distance, and publish the distance. At its limit the
probe *is* the continuous mission; at its start it is today's probe; in between
it is a curriculum on the thing no score has ever read. The air spawn (O) is the
same curriculum run the other way: launch height steps down from 30 m.
Not built: it changes what every transition score means, and the step rule
needs its own measurement of crossing rates by distance first.

### Q — the decision memo

Measured three ways now: `corr(air, water)` −0.17 by arch37's end; a wing
costing 2.5x in water what it buys in air; and today, CPPN complexity buying
air (+0.139) and costing water (−0.280) with part count held fixed. The
`land_air` island produced arch37's mission-best, and every transition any
machine makes in a continuous mission is one gravity makes for it. The option
the project has not tried is to make the mission **a chain of pairs** — separate
lineages for land↔air and water↔air, each selected on its pair and its
crossing, with the triphibian as a hand-off between them rather than one body
asked to be good at three media at once. It changes what the project is for,
so it stays the user's decision; nothing here acts on it.

## arch40 — the work list

Written 2026-09-20, after the day's fixes and before any run on them.

**The first arm is a baseline, not a hypothesis.** Every competence in every
medium was redefined, the curriculum's two lowest stages changed, the operator
bandit gained a floor, and the GPU physics moved five weeks forward. Nothing
from arch34–arch39 can be compared with what comes out. So arch40 is **one run
that changes nothing further**, and its only job is to say what the new
distributions are: forward speed per medium, the water gate's two terms, the
share of children each operator makes, and how far the archives get. Read at
generation 200, which is where distributions settle.

*Amended 2026-09-21.* The first launch was stopped at generation 54 and the
scoring changed again — the passive twin is gone and water is gated instead;
see the section above. The baseline is re-run from scratch on the new code, at
600 generations rather than 900 at the user's instruction. The quantities to
read at generation 200 are `water_headway` and `depth_station_keeping` (the two
the water gate is built from), `station_keeping` in air, `land_speed`, the
operator shares, and the §W table — the share of each island's elites that are
better in another medium than in their own, which is the number this change was
aimed at.

What it costs to skip this: every number in the next list would be read against
arch39's, and arch38 already showed what that produces -- a replication failure
nobody could attribute.

Then, in this order:

1. **Re-derive the water ladder** from arch40's own `depth_gain_net`
   distribution (§6 above). Not from arch39's, where the median machine dives
   less than gravity.
2. **The air ladder, the same way.** `stays_up` and `glides` are cleared by
   falling from 30 m with the actuators off; `sink_reduction` is published now
   and its distribution over arch40 is what the rungs should sit in.
3. **S, `mut_gait`, re-run.** arch39 never tested it: the bandit spent 0.7% of
   children on it and none at all between generations 58 and 141. With the
   exploration floor it gets ~2%, and the offline Monte Carlo (80.4% / 94.0% /
   69.5% of draws clearing the first thrust rung against 0–2.4% for every
   single-axis operator) is the number it is judged against.
4. **V, the replication arm** -- arch37's seed under current code -- is now
   subsumed by arch40 itself, which is a fresh baseline for everything.
5. **O, the air spawn**, unchanged and still the largest structural distortion:
   every air score is earned from a 30 m launch, and the filmed machine that
   scores air 0.768 spends 0% of its air leg airborne because it starts on the
   beach. With forward distance netted against a passive twin, the launch's free
   glide no longer *scores*, which removes the reason the spawn was tolerable.
6. **R, `shared_ent_coef`**, still the value measured to do nothing, still
   un-swept.
7. **Q, the triphibian conflict**, unchanged and still written down rather than
   acted on.

---

### X. The reward has no task — raised by the user 2026-09-21, **implemented the same day, unrun**

The user's objection: "if the operation's purpose is to go forward, forward should
score; if it is to hover in a medium, it should hover then, and movement should be
penalised — not one reward applied everywhere." Read against the code, it is
right, and the cause is one level below the reward:

**The policy is never told what to do, only where it is.** Its command channel is
a three-way one-hot of the medium (`TriphibianEnv.observation`), plus the depth
error to a fixed 10 m. So each medium's score has to blend every goal that medium
might have, and the policy — rewarded once per segment with that blend
(`collector.finish(buffer, [s.competence ...])`) — can only learn one compromise:

| medium | told | scored on | the conflict |
|---|---|---|---|
| air | "air" | `0.55 flight + 0.25 glide + 0.20 station`, all vertical | no forward term at all, yet a flapping wing holds height only by flying forward: it is asked to hover and not given the means |
| water | "water", depth error to 10 m | `gate * max(headway, station) * (0.6 headway + 0.4 depth)` | move, hold and dive in the same 8 s. **corr(headway, depth_station_keeping) = −0.264** over arch40's 6,288 water segments; the top quartile of both holds 151 segments against 393 if independent. The formula pays a perfect hover 0.20 and a perfect mover 0.80 — "or hold" is a gate, and the goal it actually states is *move* |
| land | "land" | `posture * (0.65 progress + 0.35 climb)` | the most coherent of the three |

And "forward" is `norm((end - start)[:2])` everywhere: displacement in *any*
horizontal direction. Drifting sideways is progress.

**The proposed fix is a task, not a better blend.** The standard in legged
locomotion RL is command tracking: the policy observes a commanded velocity and
is rewarded for matching it, and a command of zero *is* standing still, with
motion penalised. Here:

1. Each segment becomes a short script of commanded phases at the same total
   length and cost — e.g. water: hold at 6 m, then cruise at `v` on a heading at
   that depth; land: walk at `v`, then stop.
2. The policy observes the command: task one-hot, commanded speed, heading error,
   depth/altitude error to the *commanded* target.
3. Each phase is scored on its own objective only: tracking error on the heading
   for cruise, with lateral drift and depth/altitude excursion penalised;
   position error including vertical for hold; reaching the commanded depth for
   dive — where sinking is the chosen operation and may score.

It also retires the passive twin's job for free: **a passive machine does the
same thing whatever it is commanded**, so it cannot score on a hold phase and a
cruise phase in the same segment. The command switch is the "are the actuators
doing anything" test, at no extra rollout.

Cost: `OBS_DIM` changes, so the shared policy, every checkpoint and every
comparison start over — a new baseline. Open, for the user: what "hold" means in
air for machines that cannot hover; how many phases per segment; whether the
ladder rungs are re-cut as task success.

**Decided by the user ("照建議做"), and built** (`dytiscidae/envs/tasks.py`,
`TriphibianEnv._task_scores`, `TriphibianEnv.task_channels`):

| medium | phase A | phase B | combined |
|---|---|---|---|
| water | hold at a commanded depth 5.5–8 m (released at 4) | cruise at 0.30 m/s on a drawn heading at that depth | mean; order drawn |
| air | cruise on the launch heading at the machine's own trim speed, holding height | the same after a commanded 45–90° turn | mean |
| land | walk at 0.08 m/s on a drawn heading | stop and stay stopped | `walk * (0.5 + 0.5 stop)` — a rock stops perfectly, so stopping qualifies the walk instead of adding to it; order drawn |

Each phase is measured over its late half. Cruise is `1 - |v - v_cmd| / |v_cmd|`
on the mean horizontal velocity, times holding the vertical (water: the
commanded depth, against the descent it asked for; air: graded from level
through a glide to ballistic, as `flight` and `glide` have been since arch37).
Hold is *at the commanded depth* times *not moving*, both required. The task is
drawn per generation from the evaluation seed, the same for every candidate, in
both evaluators through one helper; the controller observes it in six new
channels (purpose, commanded speed, heading error, height lost), so `OBS_DIM` is
33. The command speeds are ~p80 of arch40's gross speeds (water p75 0.271 / p90
0.446, land p75 0.060 / p90 0.116 m/s over 6,383 segments each). The ladder's
task rungs read the phases — `holds_depth` = `hold_error` ≤ 1 m (distance from
the commanded depth plus distance moved), water `manoeuvres` = `cruise_score` ≥
0.5, air `manoeuvres` = `turn_tracking` ≥ 0.5, `walks` = `walk_tracking` ≥ 0.5 —
and curriculum stage 1 reads the cruise phase's tracking in every medium. The
three 0.5 bars are set on their physical meaning (error under half the command)
because nothing has yet been trained to follow a command; **re-derive them from
arch41's own distribution.**

Measured before the run, on the real rollout path:

- **A machine with its actuators held still**, averaged over eight task draws:
  water 0.011 (beetle) and 0.008 (eel), land 0.003 and 0.001. The additive
  water formula paid it 0.422; the gate that replaced it that morning, ~0.063.
  No passive twin runs.
- The first version was a wall: with a flat 1 m depth band, **97% of 64 designs
  scored exactly zero in water**, driven or still, and air lost the glide
  gradient arch37 added for exactly this reason. Errors are now measured
  against the displacement the command asked for, so sitting where you were put
  is exactly 0 and halfway is a half; water non-zero rose to 29%.
- The default heading was 0, the spawn's own nose direction, and a still beetle
  gliding forward as it sank scored 0.45 on tracking it. It is now 90°: a
  default should not line a passive glide up with the command by construction.
- Command-blind machines — open-loop gait or held still — score near zero in
  every medium (water p90 0.014 against 0.007, land 0.106 against 0.003, air
  ~0.03 for both). **Only a controller that follows the command can score**,
  which is the point, and also the risk: if water and air task scores have not
  risen by generation 50 there is too little gradient, and the command speeds
  and bands are the first things to re-derive.

Six mutations hold it: `cruise-pays-for-speed-in-any-direction`,
`hold-ignores-motion`, `hold-scale-is-absolute`, `land-stop-adds-to-the-walk`,
`controller-is-not-told-the-task`, `evaluators-ask-different-tasks`.

**arch41 ran it for 52 generations and was stopped** (`runs/arch41_stopped_gen50_walls`).
The pre-registered gen-50 bar failed: task-score medians 0 in every medium.
Tracking had gradient (water `cruise_tracking` p90 0.34 → 0.50); the
multipliers were walls — water machines mostly never descend (max depth p50
4.71 m against a 4 m release) and drift 0.41 m while told to hold (band 0.30),
86% of air segments fall out of the sky in 2.65 s, land barely moves against a
0.08 m/s command. Re-derived values (hold depth 5.0–6.5 m, stillness band
0.64 m/s, land command 0.04 m/s) are in the run's notes, **measured and not
applied**: the user decided not to relaunch.

### Z. No film was of the scored experiment — and the search's own scores were not what it meant them to be — **fixed 2026-09-22**

The user, after item Y: film each medium from where it was evaluated, beside
the continuous mission, "and make it happen every session, automatically, or I
keep judging from the wrong video." Building that meant checking that a film
reproduces its score, and on the first run it did not: arch41's mission-best
elite, filmed as the showcase drove it, scored land 0.002 against a recorded
0.283. Four defects, found in that order, each by a measurement that ruled out
the one before it:

1. **The actor pool dropped the shared policy from every re-score.** It shipped
   the shared network to its workers only when a rollout buffer came with it —
   the generation's learning rollout — so the noise-free re-score and every
   refinement trial, whose numbers are what the archive keeps, ran without it on
   any generation split into more than one shard. On `--workers 4 --min-shard 4`
   that is nearly every generation of every run since the pool was written
   (2026-09-03, arch35–arch41). The elite reproduces 0.034 / 0.090 / 0.283
   exactly with the shared policy removed; batch composition and the shared
   network's generation were each measured and ruled out first. **The shared
   policy trained on rollouts it took no part in scoring, and refinement tuned
   each design's own policy against a sum whose other half was zero** — the
   failure `_refine_controllers`' docstring names. It may be part of why the
   shared policy "failed" four times. Fixed in `envs/actors.py`; the sharding
   test now drives a shared network through three shards (it passed with none
   before, so it could not see this).
2. **Measuring trim reset the simulation being run.** `launch_speed` is computed
   the first time it is read — inside `reset`, for the air, after the spawn and
   its random offset are written — and it probed on the live `MjData`. So the
   first air reset of every phenotype object started at the bare spawn with the
   last probe's velocities, and later ones did not: process history in the
   score. Fixed with a scratch `MjData`.
3. **`clearance` counted the camera's wing strips.** `detail=True` models (for
   filming) put render-only geoms on the wings; the lowest-point search included
   them, so the air sink rate and the controller's height channel read
   differently on a filmed model — diverging at the eleventh control decision.
   Now physical (colliding) geometry only; plain models are unchanged (0 of 19).
4. **Promotion overwrote the scored policy.** Tier-2 stored its refined weights
   over `meta["policy"]`, leaving Tier-1 scores paired with weights never scored
   at Tier-1. Now `policy_promoted`, beside it; inheritance reads it first, so
   the search behaves as before.

And the films themselves: `showcase` filmed only the continuous mission;
`render --top` drove elites open-loop, at seed 0, without the evaluation's
scatter, for 10 s, printing no score.

**What exists now.** `viz/film.py` re-runs `evaluate_tier1` — the scoring
function — with a camera on `on_step`; stamps recorded and reproduced
competence on every clip; puts each continuous-mission leg beside its medium;
writes `film_manifest.json`. Each scored elite records
`scored_with_shared_policy`, and `scoring_networks/gen*.npz` keeps the exact
network that scored each generation, so a film can drive it exactly; a legacy
record is filmed with whichever candidate control law reproduces it, and says
which. `ops.run search` and the job trainer run `postrun` (report + films) on
completion. Tests: `test_a_film_is_the_evaluation`,
`test_the_first_air_reset_is_like_every_other`,
`test_a_film_reproduces_the_scored_experiment` (a real sharded search with a
shared policy, every raw measurement compared); mutations
`pool-drops-the-shared-policy-from-rescores`, `trim-probes-the-live-state`,
`clearance-reads-render-geometry`, `film-ignores-the-evaluation-seed`.

**Not comparable across this commit:** re-scores now include the shared policy,
and the first air reset of each body carries its spawn offset.

### Y. The filmed arch40 machine never leaves the beach — raised by the user 2026-09-21, **measured, not acted on**

The user, watching `runs/showcase_arch40/mission.mp4`: "it is only trembling in
place." It is. Reproduced on arch40's own code (`2bb8d32`), the continuous
mission gives the film's numbers exactly — on-task 35%, 0/2 transitions, 0.0 m
depth — and the body's net displacement over the three 8 s legs is 0.10, 0.16
and 0.02 m. Two causes, one under the other.

**It has nowhere it can go from the beach.** The filmed elite (aerial_diver
island, beetle plan, 3 parts, **2 DOF — the two wings**, 4.71 kg, 1.92x the
density of water) is worth something in exactly one medium, and its Tier-1
record reproduces: air 0.033 (from the 30 m launch it stays up 2.7 s), land
0.028, **water 0.767**. From its evaluated water initial condition:

| 8 s in water | horizontal | depth gained | competence |
|---|---|---|---|
| its own controller | 6.96 m | 2.39 m | 0.763 |
| open-loop gait | 1.98 m | 9.15 m | 0.090 |
| **actuators held still** | **5.24 m** | 1.51 m | **0.519** |

A dense hull whose wings, held at their starting angle, turn sinking into
forward glide: two thirds of its water score needs no actuation. On land the
wings give no traction — 8 s moves it 0.21 m driven, 0.16 m held still. It
cannot take off. Every medium in evaluation starts from a placed spawn (air
launched at 30 m and trim speed, water released 4 m under, land on the beach),
so nothing in selection asked it to get from one medium to another under its own
power; the continuous mission starts it on the beach, and the one thing it can
do is in the water it cannot reach. This is item O's distortion reaching the
film, and the fifth run whose film shows 0/2 transitions.

**Why it trembles rather than flaps.** In the air and water legs the controller
drives the wings through bases identified in fluid, while the body sits on sand.
The wings strike the ground, the ground-contact channel flips on a third of
control steps, the yaw-rate channel jumps by 0.62 of its range per step, and the
controller answers the jolt: its command reverses direction on **72%** of 40 ms
control steps (smooth control is well under 50%). The loop runs through the
body rates — freezing those six channels halves the step size (0.165 → 0.074)
and reversals fall to 57%; freezing the stroke channels changes nothing (70%).
The policy is the deterministic mean, so this is the closed loop, not
exploration noise. It has essentially never seen ground contact in the legs
whose bases it is using.

What arch41 changes and does not: its task reward removes the passive glide's
pay (a still machine scores ~0.01 in water), and does nothing about the spawns
or about getting between media, so its film can show the same beach. Candidate
fixes, none attempted: evaluate a leg from where the previous one ended (item O
generalised); an action-rate penalty or command low-pass, standard in legged RL,
for the chatter; and film each medium from its evaluated spawn beside the
continuous mission — **done, item Z** (`render --top` turned out to film a
different experiment, open-loop at seed 0).

### W. The specialist islands were selecting for water — curriculum **fixed, unrun**; water score **measured, not fixed**

Raised 2026-09-20 during arch39: a pure-habitat island's elites should not be
better in another medium than in the island's own. Measured on the archives,
elites better elsewhere than in their own island's medium:

| | air island | land island | water island |
|---|---|---|---|
| arch39, gen 180 | **96.4%** (73/83 best in water) | **96.4%** (103/110 in water) | 7.9% |
| arch38, gen 899 | **97.1%** | **75.9%** | 20.3% |

Median own competence 0.032 on the air island and 0.055 on the land island.
**This is two failures in series, and they need separate fixes.**

**1. The channel — the curriculum was island-blind. Fixed (`b2db2e6`), unrun.**
`stage_score` read all three media at every stage and `run_search` built each
island's curriculum without telling it which island it served; stage 0 scored
"the best medium anywhere" with the island's own score at the 0.25 handover
floor. Now every stage reads only the island's own media and crossings
(`islands.curriculum_for`). The generalist is unchanged at every stage; a
curriculum unpickled from an earlier run keeps the old behaviour, so a resumed
run does not change selection mid-run. On arch39's air archive the stage-0 top
ten go from ten water machines (air 0.00–0.03) to air 0.71 / 0.67 / 0.60 at the
top.

**2. The source — water competence is paid for sinking. Fixed 2026-09-20**
(`6a21af6`; see §"What 2026-09-20 fixed" §2 for the design and the numbers).
The measurement that motivated it:
With every actuator held still, the seed plans score:

| medium | own gait | **actuators still** | still ÷ own |
|---|---|---|---|
| water | 0.510 | **0.533** | **104%** |
| air | 0.029 | 0.032 | 112% |
| land | 0.048 | 0.010 | 20% |

A still machine sinks — the eel gains +4.45 m with its actuators held against
+4.52 m flapping, the bat +3.02 m — and every water term pays for it: `reached`
(depth gained toward 10 m), `hold` (nearness to 10 m), `submerged` (released
underwater), `upright` (a hull is passively stable). Item D replaced absolute
depth with depth *gained*, which removed the free 4 m of the release and not the
free metres of sinking: the air ladder's "paid for falling" defect, in water.
**Seventh instance of the recurring lesson.** Land is the one medium where the
actuators matter, because it was gated after the same probe caught it.

It was fixed by the first of those candidates, for all three media rather than
water alone: every segment runs an actuators-held-still twin from the same
state and scores the difference. It redefines `mission_fraction`, the islands'
own scores and PPO's reward in every medium at once, which is why **nothing
before 2026-09-20 is comparable with anything after** — and why arch40 gets a
replication arm before anything is concluded from it.

**Still open:** the ladder rungs read the *gross* measurements, so the water
rungs are cleared by sinking (a still eel gains 4.45 m, clearing the 2 m rung)
and the curriculum's stage 1 reads `sink_rate`, `depth_error` and `land_speed`
gross. The net forms are published now (`depth_gain_net`,
`depth_error_reduction`, `sink_reduction`, `land_speed_net`); moving the rungs
onto them needs thresholds re-derived from their own distribution.

**Order matters.** 1 alone moves the air and land islands' curriculum onto their
own media, but the water, amphibian and aerial-diver islands still select for
sinking. 2 alone changes which medium is cheapest and the island-blind leak
would find the next one. With 1 in, 2 is contained to the islands that own
water.

### L. Aspect ratio grows and buys nothing — and thrust did not fix it

Carried from before arch38 with its conditional now answered. The instruction was
"first check whether `thrust_margin` fixes it as a side effect, since thrust
scales with the area being flapped." **arch38 rewarded thrust and the thrust
distribution moved down**, so the side effect was never available and L is
unanswered rather than resolved. No cost on span or aspect ratio exists anywhere
in `fitness()` (`envs/evaluate.py:618`); `aspect_ratio` is an aerodynamic
coefficient and a descriptor feature and nothing else. **Re-ask it after S**,
which is the first change that would make flapping area worth paying for.

### M. CPPN complexity drift — measure the confound before charging for it

18 → 73 over a run, correlation +0.060 and flat, while `n_parts` at +0.191 is the
strongest predictor in the body vector. `Genome.complexity`
(`core/genome.py:281`) is recorded as metadata and charged nowhere. Charging for
it is the obvious move and the obvious risk: complexity and part count are not
independent and part count is the thing that works. **Measure their correlation
and the conditional effect of each before touching either.**

### N. GRPO for the shared policy

Not implemented — `learning/` holds `ppo.py` and `distill.py` and no GRPO
anywhere. Right algorithm, still not the binding constraint, and it waits on
thrust being in the score in a form the search can climb. Partial form when it
comes: G=4 learning-only rollouts for a few bodies per generation.

### O. The air spawn — still the largest structural distortion, still no safe fix

`SPAWN[Domain.AIR] = (-40.0, 0.0, 30.0)` at `envs/triphibian.py:483`, read
unchanged at three sites, varied by nothing but 0.2 m of reset noise. Every air
score in this project has been earned from a 30 m launch. No design yet exists
that lowers it without making the air ladder unreachable for everything.

### P. `descriptor_refit_every`

Still 400 (`evolution/loop.py:181`, CLI `ops/run.py:733`). arch38 made it
measurable for the first time and measured it: **19.0% median merge loss per
refit, 11.1–31.2%, over 660 median archipelago cells.** Unfixed, and now with a
number to judge a change against.

### Q. The triphibian conflict itself

`corr(air, water)` runs +0.04 to −0.17 across arch37's nine bands and a wing
costs 2.5x more in water than it buys in air. arch38 added a second instance of
the same shape: a fixed evaluation budget moved to water the moment water had
somewhere to climb. The `land_air` island — added in arch37, finished second of
seven, produced that run's mission-best — is evidence that pairs work where the
whole does not.

**Whether the mission should be a chain of pairs rather than one machine asked to
be good at three things at once is a change to what this project is for, so it is
written down and not acted on.** It is now supported by two runs rather than one.
reproducible from `--seed` at all (‖dW‖ = 7.4 between runs). Full audit in
`docs/LEARNER_AUDIT.md`.

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
