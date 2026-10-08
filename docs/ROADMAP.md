# Roadmap

**2026-10-08, later: the work list was swept; see
[the work-list sweep](#2026-10-08-later--the-work-list-sweep).** arch49 (ran
10-07) had never been read: its held-out retention is below 0.35 in every
medium and Tier-1→Tier-2 Spearman is 0.17/0.18. C2 measured a single draw's
reliability at 0.18–0.30 (draw variance 2.5–4.6× design variance). Built: one
draw per candidate (M2), a still machine that is still, loud post-run, config
export, the hyperparameter table, placement on several draws (off). arch50
(M2, refine 0) is running. Seven questions for the user close the section.

**2026-10-08: a MuscleMimic comparison was checked against the code and arch48's
telemetry; see [the MuscleMimic comparison](#2026-10-08--the-musclemimic-comparison-reconciled-with-the-repo).**
Its thesis (the evaluation signal, not PPO, is the bottleneck) agrees with
PAPERS_2610. Two findings are new: the (1+1)-ES refinement accepts on
`mission_fraction`, which was above zero in 7 of arch48's 4,792 evaluations
(29 acceptances in 300 generations), and it keeps trials scored on the draw it
reports. PPO's 10 epochs never reached the KL bound (0 of 300 updates), so the
proposed epoch sweep comes last. **M1 is read (same section, §D):** a
one-step refinement keeps 8-23% of the gain it reports at a fresh draw and is
not separable from a random perturbation, so the next arm runs
`--refine-steps 0`; elites still lose 22-70% of their medium scores at a fresh
draw under the antipodal scoring, so one draw per candidate (M2) stays first.

**2026-10-06, later: an outside review was checked claim by claim against the
repo; the correction plan is
[external review reconciled](#2026-10-06--external-review-reconciled-with-the-repo-and-the-correction-plan).**
Two of its headline claims do not hold for the runs that exist (the controller is
refined: `--refine-steps 2` since arch34, 6 steps at promotion, PPO inside the
search; the GPU kernel scores the search). It holds on four: the
`controller_refine_steps` default is a trap (0), no pass/fail bar defines "done",
one seed per arm, and Tier-1/Tier-2 agreement is unmeasured since the fix.

**2026-10-06: read [PAPERS_2610.md](PAPERS_2610.md) first.** Three outside papers
(ReCo, Prospective Hindsight, NeutronGym) were applied to arch48, and two of
their checks found what this list had not. **Tier-1 water and land credit belongs
to the generation's task draw:** a fresh seed alone takes Tier-1 passes from 15/16
to 3/16 (water) and 13/16 to 0/16 (land), because one heading is shared by 16
candidates and the archive keeps the best of each draw (NeutronGym's winner's
curse). That is why Tier-1 does not predict Tier-2 (Spearman -0.09 / 0.02) and why
the critic has no skill. **Water competence does not separate elites from still
machines** (45 vs 35 of 200 at 0.15; land does, 45 vs 12). The auditor's held-out
check, which should have caught the first, never ran with the mission at zero; it
now re-measures each medium (a note, not an invalidation). The ranked list is at
the end of that file. **Fixed the same day:** water and land are scored on an
antipodal pair of headings from one initial state, on the mean signed progress,
so motion the command did not choose cancels exactly; re-gated on arch48, every
competence bar is certified or underpowered against still machines and
unsearched gaits (water >= 0.012: elite 41 / still 2 / base 2, was 121 / 103 /
112). Water and land competence are not comparable across it.

**2026-10-06: arch48 finished; the current work list is
[2026-10-06 — arch48](#2026-10-06--arch48-finished-and-the-work-list), ranked by
cost, then speed, then learning.** Mission and every crossing stayed at 0 for
300 generations with the crossing curriculum on. It could not have moved: its
start rate was already 0 at back 0. The first item is a measurement, not a
run.

**2026-10-03: arch45 finished; read [ARCH46_SPEC.md](ARCH46_SPEC.md) first.**
arch45's critic never fitted: a `Tier-1 mission > 1e-4` gate dropped 128 of
147 labels, and every kept label was 0. The critic now learns per-medium
residuals (built, unrun). The spec ranks the rest of the outside proposal
(SAIL, Deb, T-DominO, PGA, RUDDER) against arch45's measurements. **Its §8 is
the largest finding:** `water_to_land` credits a still machine for settling
*into* the water (any wetness flip counts), `air_to_water` is crossed by
gravity, and these are the learner's two largest rewards. **Fixed the same day,
all three steps** (directional, commanded hold-then-go, economy gated); every
transition score is not comparable across it.

Written 2026-09-03 after arch33, and revised the same day once every phase was
built. Every item names the measurement that motivates it; nothing here is on
the list because it seemed like a good idea, and nothing is marked done without
the number it produced.

**Revised 2026-09-30: the current work list is
[arch45](#arch45--the-work-list), ranked by build cost first, then loop speed,
then what it does for the search and the learner.** Everything older is kept for its
measurements.

Revised 2026-09-19, after arch38 ran. Everything below is kept for the
measurement that motivated it; **the current work list is
[arch40](#arch40--the-work-list)**; arch39's list is kept below it, and
[What 2026-09-20 fixed](#what-2026-09-20-fixed-and-what-is-not-comparable-across-it)
is the boundary across which nothing is comparable.

**Revised 2026-09-26, later: AG and AI are done, AH was probed and built
nothing, AD waits on arch43's gen-200 read — see
[AG, AH and AI, and arch43](#2026-09-26-later--ag-ah-and-ai-and-arch43).**

**Revised 2026-09-26. Read
[AB-AF executed](#2026-09-2326--ab-af-executed-every-open-fluid-item-closed-and-a-rotor-control)
first:** every open fluid item is closed, a quadrotor control flies the real air
segment (0.963-0.987; 0.86-0.93 under the later AE height term), and level flapping flight exists only with feathering
and is lost to actuator torque at 7-12 Hz. Its work list, AG-AK, comes first.

Revised 2026-09-23. Read
[Why nothing flies, measured a third time](#why-nothing-flies-measured-a-third-time--2026-09-23)
first.** Five model defects were fixed and the air score's fall credit was
gated. With the physics corrected, no seed plan's heave-only flapping reaches
the thrust level flight needs, at any of 300 random gaits (best: gannet +0.65).
Its work list, AB-AF, comes before arch40's. The paragraph below on thrust
being reachable was measured on the bugged physics and is superseded.

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
| AA gait gain | the policy could not stop a machine (74% of amplitude left at best); a gain channel on both policies, `--gait-gain` | **done**, in arch42 |
| Z2 audit = evaluation | the audit re-runs the scored experiment (own + shared policy, own seed), perturbed and nothing else | **done**, in arch42 |
| U `min_shard` defaults | already 4 in all three places (`loop.py:95`, `actors.py:171`, CLI) | **done** — heading was stale |
| S `mut_gait` | implemented (`genome.py:549`) with the operator floor; needs only its measurement, which arch42 carries | **in arch42**, running |
| X task reward | **reshaped as S3 after arch42's first launch ran flat to gen 102 (see X)**; hold depth 5.0–6.5 m and land command 0.04 m/s, from arch41's stop; the stillness band **stays 0.30 m/s** — at arch41's p75 (0.64) it paid a held-still beetle 0.093 in water, because that population's drift *is* passive sinking | **done** |
| Y chatter | `command_rate`, `command_reversal` published on every segment; `action_rate_penalty` (default 0) charges the rate | **done**, flag off |
| Y / O continuity | measured first (below): a continuous start would be a wall today | **designed**, not built |
| M complexity | `runs/analysis_M_complexity.md`: do not charge — it is the air/water conflict (+0.139 air, −0.280 water), not bloat | **closed** |
| L aspect ratio | `runs/analysis_L_aspect_ratio.md`: no cost — AR buys static lift (+0.153) that never becomes flight | **closed** |
| P refit | `runs/analysis_P_refit.md`: each refit erases more than islands grow between refits; subspace overlap now recorded, `descriptor_keep_if_overlap` (default 0) keeps unchanged axes | **done**, flag off |
| R `shared_ent_coef` | a short sweep, after arch42 (it needs the machine) | queued |
| N GRPO | held on purpose: arch42 is the first run in which the shared policy is scored at all (item Z), and whether it helps decides whether a better estimator for it is worth building | after arch42's read. **2026-10-03: built, off by default** (`--shared-learner ppo\|grpo\|ppo+grpo`, default `ppo`); see item N below |
| W, T, V | superseded by X / closed / subsumed | closed |
| TEST_AUDIT 7 | 15 mutations are caught only by `test_search`. The 5 for this commit's code are caught by named checks — two of them against their own test function after the full-suite run timed out under four-way contention. The 10 older ones wait for the machine, after arch42 | 5 of 15 |

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

## 2026-09-26, later — AG, AH and AI, and arch43

The user asked for the recommended next steps (AG, then AH), for the search to
be allowed to evolve rotorcraft (AI), and then for a new training run.

- **AG — done.** `TriphibianEnv.level_margin` is `min(<Fz>/W, 1 + <Fx>/W)` on a
  rig. The airframe is held at its trim speed and attitude, its own gait runs on
  its real servos and motors, the fluid keeps its full history, and the force is
  averaged over 0.7 s after a 0.5 s settle. It costs about 0.3 s a design,
  cached per phenotype. It is published beside `thrust_margin` on every air
  exit, and the rig does not spend the battery.

  | plan | level_margin |
  |---|---|
  | beetle | 0.939 |
  | gannet | 0.913 |
  | teal | 0.826 |
  | medusa | 0.005 |
  | eel | -0.07 |
  | ray | -0.08 |
  | bat | -1.03 |
  | quad | 0.85 |

  For a glider it reads about `1 - 1/(L/D)`, and it reaches 1 only with thrust,
  so it is a gradient toward flight rather than a step.
- **AH — probed; nothing built, as pre-registered.** The teal's level-flight gait
  on the rig with free joints (`experiments/flight_audit/probe_actuation.py`):

  | condition | margin | torque-limited |
  |---|---|---|
  | own motors | 0.365 | 61% |
  | motors 2x | 0.543 | 67% |
  | motors 3x | 0.494 | 65% |
  | spring at the gait's own offset | 0.143 | — |
  | spring, compliant drive | 0.153 | 14% |

  Nothing clears 1. Heavier motors stay just as saturated, so this is not simply
  a torque-size problem. The search already has the genes in question (motor
  mass, series spring, compliance, feathering), and with AG it has a measure of
  what they deliver.
- **AD — deferred to arch43's gen-200 read.** Thresholds are set from a measured
  distribution, and there is none yet under the corrected physics.
  `level_margin` is published from gen 0.
- **AI — done.**
  - `mut_rotor` adds, removes and retunes a propeller on any part (radius,
    pitch/diameter, rest throttle), and one fresh random genome in ten carries
    one. The rotor spins about the part's own +Z, so the placement genes decide
    where it points.
  - A rotor's CPG channel is a steady throttle the policy moves through the
    offset.
  - Rotor forces come from a nondimensional CT/CQ table per rotor and medium,
    with Reynolds number as an axis. It is within 4% of static thrust and 11% of
    static torque of direct BEMT, and 190x faster (3.5 ms -> 19 us a call); a
    quadrotor's segment had cost 28 s.
  - Actuator order was checked over 40 random designs with rotors and feathering
    joints: 281 actuators, none misordered.

### arch43's first launch — stopped at gen 3, and why

At gen 2, 5.1% of rollouts diverged, against the 1% pre-registered. Four of the
five diverging designs came from `mut_rotor`, and all five were "unstable" on
air -> water.

**The cause.** Reproduced offline, a rotor spinning at 550 rad/s put under water
reached |qvel| 787,158. Two faults compounded. The drag torque (Omega^2, about
1000x in water) was explicit on a tiny inertia. And thrust was taken at the
start-of-step spin while the rotor braked in about 1 ms. That gave about
40 N s of impulse per rotor, where momentum theory with the rotor's 13 J allows
about 3, so the machine was fired out of the water.

**The fixes:**

- Backward Euler on the drag for the end-of-step spin, with thrust and torque
  taken there. The motor's torque is left to MuJoCo's servo; predicting it too
  put the controller a step behind.
- An implicit split of 2|Q|/|Omega| on the rotor's hinge. It uses |Q|, because a
  windmilling rotor has Q < 0 and negative damping oscillated.

The plunge now has 0 bad-qacc (`test_a_propeller_can_go_under_water`; mutation
`rotor-thrust-at-start-of-step`).

**A correction to §3 above.** The reference quad scores **0.86-0.93**, not
0.963-0.987, under AE's worst-second height term: its controller sags about a
metre in turns. The committed tree before these fixes gives the same, so the
rotor fix moves air flight by at most 0.006.

**Cost.** The launch read 620-700 s/gen in the burst, about 2x. Profiled on four
designs, one batched evaluation:

| tree | seconds |
|---|---|
| before the fluid rewrite (`54d2f45`) | 37.3 |
| after it, as launched | 73 |
| after the fixes below | **51** (1.37x) |

The extra time was host-side Python per step and per machine, and three changes
remove most of it:

- The damping sum per body instead of per strip, the same to 2e-10.
- The damping bound refreshed every 4 steps. The explicit compensation still
  uses this step's velocity, so the force is exact.
- The inflow updated every 4 steps. Its time constant is 100 ms or more, and the
  lag is integrated exactly.

### arch43 — what it is

The first run under the corrected physics, and the first in which rotorcraft
are reachable. Its configuration and pre-registered reads are in
`runs/arch43_notes.md`. **Nothing before it is comparable** on air or water
forces, actuated motion, the air score's short exit and height term, or
curriculum stage 1.

---

## 2026-09-27 — where a worker's time went, and what moved it

Asked: find why the training load sits almost entirely on the CPU, and
parallelise it, move it to the GPU, or use an acceleration library.
Measurements and method: `experiments/perf/NOTES.md`; harnesses beside it.

**What the time was.** Four workers at 83-96% of a core each, the parent
idle, the GPU 5% busy (60 s of `/proc` jiffies on arch43). Inside a worker
(py-spy, sampled, not cProfile -- cProfile inflated small Python calls by ~2x):
the per-step GPU round trip was the largest single item, ~30% of real time --
382 us per step for a 4-machine shard of only 190 panels, of which 5 uploads, 4
fills and 17 downloads of pageable memory were 210 us and eleven kernel
launches 90 us. `mj_step` was 7-8%. The rest was host-side numpy on arrays of a
handful of elements, ~25 calls per machine per step. **CPU time equals wall
time** in every run: the CUDA wait is a busy spin, so GPU latency was CPU time.

**What was done, all bit-identical** (700 segment and transition numbers of a
4-design benchmark compared to the last bit after every step):

- one packed upload and one download a step, the descriptor built once, the
  scalars read by pointer (`FullPipeline.step`);
- the ten per-panel kernels fused into one (`panel_kernel`; each stage is an
  `*_at` body that its own kernel also calls, so there is one copy of the
  physics);
- `launch`/`wait`: `step_batch` launches the next step's fluid as soon as
  `mj_step` is done, and the caller's recording and policy run while the device
  works; the download is in `wait`. Measured in place: `launch` 18 us, `wait`
  16 us, 8993 of 9000 steps found their fluid already launched (`PRELAUNCH`;
  bit-identical on and off, held by
  `test_the_early_fluid_launch_changes_nothing`). **Not with pinned memory**:
  Mojo's first host buffer reserves a ~1.35 GB pinned pool per process, which
  tripped arch44's first launch on its memory ceiling at gen 0 (8793 MB) and
  got an 8x2 pool OOM-killed; the blocks are ordinary numpy arrays;
- `BatchedPower`: the energy model for the whole batch in one pass
  (`test_the_batched_power_budget_is_the_power_budget`);
- `np.cross` written out, the CPG's clip as min/max and its clipped params
  cached between control decisions.

Result, bit-identical to `df95498` on all 700 numbers
(`experiments/perf/reference_df95498.json`): benchmark CPU on the idle machine
13.27 s -> 8.25 s (1.61x); a generation's main evaluation (16 designs with
identification, 4x4 pool) 55.9 s -> 42.6 s (1.31x). GPU cost per step 382 us
-> 34 us of host time. arch44's first generation reproduced arch43's line
exactly in 486 s against 892 s.

**What did not work, and why.** `DeviceContext(api="cpu")` exists but kernel
launch on it is `Unimplemented`, so the kernels cannot simply run on the host.
A CPU port would scale with panel count and compete with the workers for the
same cores, where the GPU's cost is latency the early launch now hides. And
**on this laptop the pool is power-bound, not latency-bound**: at 96 C and 3.2
of 4.7 GHz, the sweep beside the live run moved only from 77.4 s (HEAD, 4x4)
to 74.6 s (new, 4x4) and 64.3 s (new, 8x2). Throughput under full load is set
by CPU work, which is why the later steps went after instructions, not waits.
And the worker count is bounded by memory before cores: 4x4 stays.

**Hazards found on the way.**
- `mojo build -o` overwrites its output in place (same inode), and a live run
  has `mojo/build/*.so` mapped. `build-all` now builds to a temporary name and
  renames. `DYTISCIDAE_KERNEL_DIR` points a process at a staged build.
- `mojo/tests/test_full_gpu.py` had been stale since 2026-09-23 (a hand-copied
  descriptor); it goes through `BatchedFluid` now -- 7 plans, xfrc 4.0e-16 and
  added mass 3.5e-16 relative to `FluidSolver`.
- arch43 died at gen 39 in a global OOM (`runs/arch43_notes.md`): runs launched
  from a Claude session live in its cgroup scope, and `setsid` does not leave
  it. Launch with `systemd-run --user`.

**Not done, measured as the next levers.** The remaining host time is spread
thin (the damping projection 8-9%, `mj_step` 8%, clearance and twist recording
~8%, the inflow ~4%, the CPG ~3%). Vectorising the per-machine sums across the
batch would take most of it, but `reduceat` sums are not bit-identical to slice
sums, so that step needs the path-agreement noise floors instead of exact
comparison. Identification is 53% of the main evaluation and waits on the
device more than the rollouts do (7.8%): its steps have little host work to
hide the launch behind.

## 2026-09-30 — arch44 finished; its pre-registered reads, written up late

arch44 completed 600/600 generations and its postrun (60.6 h, 9343
evaluations); the reads pre-registered in `runs/arch44_notes.md` were not
written up when the run finished. Full numbers and method are there; summary:

- **gen 10 cost:** 130 s/gen steady state, in line with arch43's 166 on a
  busier machine.
- **gen 50 health:** nothing broken — divergence 0.60%, `level_margin`
  published on 99.4% of evaluations, water/land scoring normally, air still
  near-zero (expected this early).
- **gen 200, AD's threshold:** `level_margin` p50 0.006, p90 0.705, max 1.354,
  1.39% >= 0.95. Correlation to air competence is weak (+0.11) — clearing the
  rig check is necessary, not sufficient. **The user set the gate at 0.7
  (2026-09-30); it is item 4 of the arch45 list.**
- **gen 200, rotorcraft vs flapping:** no separation yet at gen 200. By the
  run's end, rotor-carrying archive elites do lead on air (best 0.038 of 228,
  against 0.002 of 188 non-rotor) — but 0.038 is nowhere near the 0.3 bar for
  "first evolved machine to fly."
- **gen 600, mission:** `mission_fraction` 0.001, **transitions 0/2** — the
  sixth run running at 0/2. Water leg on-task 0.0%. Item Y's diagnosis
  (nothing in selection asks a design to move between media under its own
  power) still stands; item O (the air spawn) is unchanged.

Two follow-ups this surfaces: AD's gate (set at 0.7 by the user, 2026-09-30;
item 1 below), and the rotor-vs-flapping split has no gen-resolved
data because the run kept no periodic archive snapshots — only the final
state and per-evaluation events, which undercounts rotor share (inherited
rotors don't show as a fresh `mut_rotor` operator).

## arch45 — the work list

arch44 ran 600 gens from 2026-09-27 and finished 2026-09-30 (reads in the section above).

**Decisions, 2026-09-30 (the user).** AD's gate is `level_margin >= 0.7`.
The search keeps building rotorcraft (AI). The mission stays one triphibian
machine, not a chain of pairs.

**Ranked, 2026-09-30 (the user), by three keys in strict order**: first what it
costs to build (cheapest first); among equal cost, how much it shortens the
loop (a generation, and so every future experiment); only then how much it
helps the search find better mechanisms or the learner learn. An item that only
measures ranks by what its measurement unlocks. Dependencies are kept: an item
that needs another's read says so.

| # | item | build cost | loop speed | mechanisms / learning | why here |
|---|---|---|---|---|---|
| 1 | **AN** refuse `identify_axes_every > 1` | trivial | none | none | a knob that silently lies; one config check and one mutation |
| 2 | **Telemetry bundle**: AK stage walls + acceptance, AL stage walls, AJ per-shard wall, periodic archive snapshots | low, no behaviour change | decides 3, 7, 8, 9 | indirect: gen-resolved rotor/flapping split and refinement's real effect | cheapest item that unlocks every speed lever below |
| 3 | **AK** cut or target `--refine-steps 2` | low once 2 has read (a flag) | **high**: re-score + 2 steps ~ half a shard's generation (estimate) | risk: may cost selection, hence an arm | largest speed lever on the list, and nearly free to try |
| 4 | **AD** level-margin gate at 0.7 | low: one gate on the flight rungs, one mutation | none | **high**: first selection that reads flight through real actuation | the cheapest direct push toward flight; can share a launch with 1-2 |
| 5 | **R** `shared_ent_coef` sweep | low | none | learner: the shared policy's value is still unmeasured | a short sweep; decides whether 11 is worth anything |
| 6 | ray entry under corrected added mass; TEST_AUDIT 7's remaining 10 mutations | low-medium | none | hygiene | when the machine is otherwise idle |
| 7 | **AM** host-side per-machine loops | medium; needs its own noise floor | ~1/5 of a worker | none | the larger of the two medium-cost speed levers; cannot use the bit-identity gate |
| 8 | **AL** promotions through the pool, the scored basis reused | medium | <= ~5% | exactness gain: Tier-2 gets the basis Tier-1 was earned with | close if 2 reads < 3% on an idle machine |
| 9 | **AJ** work stealing | medium-high, and a reproducibility decision | unknown until 2 reads; close under ~10% idle | none | cost is certain, gain is not |
| 10 | **Y/O** transition-distance curriculum (and the air launch stepped down from 30 m) | high: changes what every transition score means; needs crossing rates by distance first | none | **highest**: six runs at 0/2 transitions, nothing in selection asks for a crossing | the mission's actual wall; after 4 so the two arms do not share their reads |
| 11 | **N** GRPO for the shared policy | high | negative (G rollouts) | learner, conditional on 5 | only if 5 shows the shared policy carries weight. **Built 2026-10-03, off by default; switching it on is still conditional on 5** |

AH is not listed: probed on the rig, nothing cleared 1, and the search already
owns the genes (motor mass, spring, compliance, feathering) with AD to select on
them.

**Status after the 2026-09-30 build pass** (the user: build as much as
possible, measure only where it cannot be avoided, do not start arch45). Each
item's own section has the detail and the mutations that hold it.

| # | item | state | what decides the rest |
|---|---|---|---|
| 1 | AN | built: identification per candidate | nothing |
| 2 | telemetry | built; an 8-gen run is identical to `5221fe1` | nothing |
| 3 | AK | both levers built; the funnel is off | `stages.placement_changed` against `refined`, from the next run |
| 4 | AD | built and **on**: air scores are not comparable across `5221fe1` | its pre-registered read (share clearing 0.7 by gen 100) |
| 5 | R | not run: it is a sweep | the sweep; it gates 11 |
| 6 | ray entry | built: entrainment reacted on both paths, damping refreshed at the surface | none; the mutation re-runs are in their own section |
| 7 | AM | not built: the gain only exists as a timing | a timing against the path-agreement noise floor |
| 8 | AL | built: promotions batched from the scored basis, Tier-1.5 and Tier-2 on workers; the audit stays in the parent | the audit's `stage_wall` |
| 9 | AJ | built: per-machine exploration noise (on), a shard queue and cost balance (off) | `stages.idle` above ~10%, then a sweep |
| 10 | Y/O | built and off; its step numbers are placeholders | crossing rate by start distance on arch44's elites |
| 11 | N | not built | 5 |

Found and fixed on the way: until this pass, only the batched path scattered a
crossing's entry state, so every crossing that Tier-2 or a film measured
started somewhere the scored one had not (up to 7.3 m/s of entry speed apart
at `5221fe1`). See the ray-entry notes under AK.

### 1. AN. `identify=any(...)` breaks `identify_axes_every` -- **fixed 2026-09-30, not refused**

**Done.** Refusing `> 1` was not possible: nine sites in `tests/test_search.py`
pass `identify_axes_every=999` to keep identification off. So the knob now does
what its name says. `evaluate_tier1_batch(identify_axes=...)` takes one bool or
one per phenotype and identifies only the machines asked for, and
`ActorPool.evaluate_tier1` slices the list per shard. A child that is not
identified drives with the bases its controller arrived with, and a fresh
controller arrives with none, which is what `identify_axes=False` always meant.
At the value every run uses (1), the list is all-True and the batch is the same
as before. Held by the sharding test (bases only where asked, identical in 1 or
3 shards, no batch re-run in the parent) and by two mutations,
`identify-one-means-identify-all` and `identify-list-not-sliced-per-shard`.
Both caught.

Raised 2026-09-30, and true as code: each candidate gets its own
`identify = (counter % identify_axes_every) == 0`, and the generation then
passes `identify=any(b[2] for b in built)` to `evaluate_candidates`, so one
identifying candidate makes the whole batch identify. The knob cannot do what
its name says for any value above 1.

It has changed nothing so far: `identify_axes_every` is 1 in every run's
`run_start` config (arch40-arch44, all launches and void launches), where every
candidate identifies and `any` is the same answer.

**Fix, when anyone wants a value above 1** -- and not before, because that is
itself a search-design change: identification on the subset (`identify_batch`
already takes an arbitrary group of environments), and a decision about what a
non-identified child drives through. Today it would keep its parent's inherited
bases, which is the exact thing the overwrite in `evaluate_tier1_batch` says
identification exists to prevent. **Now:** refuse `identify_axes_every > 1` at
config load, with a mutation that sets it to 2 and expects the refusal, so the
knob cannot silently lie.

### 2. Telemetry bundle -- **built 2026-09-30**

**Done, no behaviour change, and checked against a baseline run.** Two 8-gen
searches at seed 7 (batch 8, 2 workers, 2 s segments, refine 2, shared policy)
were run, one at `5221fe1` and one with this change. Every `generations.jsonl`
line and every event outside the new fields was identical, across all 8
generations (241 events). What each generation now writes:
- a `stages` event: walls of `main`, `rescore` and each refinement step;
  `accepted` trials per step; `placed`, `refined` (designs a step changed) and
  `placement_changed`; `shards`, each sharded call's per-shard walls inside the
  workers; and `idle`, the share of worker time spent waiting on the slowest
  shard. `placement_changed` comes from a dry run: `_score_candidate(commit=False)`
  plus `Archive.would_add`. That is the same code `_place` and `Archive.add`
  run, split out rather than copied (`_verdict`). The only difference is that a
  dry run's population standings do not yet count the candidate itself.
- `stage_wall` events for `verify` and `audit`, and `refine_wall`,
  `tier1_5_wall` and `tier2_wall` on every `promote`.
- `snapshots/gen{NNNN}_{island}.json`, every island's archive every
  `snapshot_every` (50) gens, with `n_rotors` on every elite's meta.

What the 8-gen check already showed, as a size and not a result: at 2 s
segments, refinement accepted **0 of 8 trials in every step of every
generation**, and two of the six generations ran as one shard (one Tier-0
rejection at batch 8 gives `7 // 4 = 1`).

The measurement steps of AK (per-stage wall, trials accepted per refinement
step, placement status before against after refinement via a dry-run `_place`),
AL (wall of promotion, Tier-1.5, Tier-2 and audit), AJ (per-shard wall in
`ActorPool`), and an archive snapshot every 50 generations (arch44 could not
split rotor from flapping by generation without one). The 700-number benchmark
must not change.

### 3. AK. What `--refine-steps 2` buys has never been measured -- **both levers built 2026-09-30; the read decides which**

**Built.** Cutting is a flag, as it always was (`--refine-steps`). Targeting is
now one too: `--refine-funnel m` (`controller_refine_funnel`) refines only a
candidate whose noise-free score, filed dry against the archive, is at least
`(1 - m)` of its cell's incumbent. An empty cell always qualifies. The other
candidates leave every step's batch, and that is where the saving comes from.
Off by default. Held by `test_the_refinement_funnel_refines_only_what_it_selects`
and the mutation `funnel-refines-everyone` (caught). The read below still
decides between 0, 1 and the funnel: its numbers are in the `stages` event
(item 2).

Raised 2026-09-30 by a review that assumed the opposite: that
refinement runs only at promotion (`controller_refine_steps` defaults to 0,
`promotion_refine_steps` is 6) and that this was the right design to keep. The
default is 0, but **every run since arch34 has passed `--refine-steps 2`**
(`runs/arch34_notes.md` through `runs/arch44`; checked in each `run_start`
config), so every candidate of every generation gets a noise-free re-score and
two (1+1)-ES steps after its main evaluation: four batched evaluations of the
whole batch per generation (`_refine_controllers`, `evolution/loop.py`).

**Its cost, estimated from `experiments/perf/NOTES.md`** (one 4-design shard,
idle machine): main evaluation with identification ~33 s, a batched evaluation
without it ~12 s. So a shard's generation is ~33 + 12 + 2 x 12 = ~69 s, of
which the two refinement steps are ~35% and the re-score ~17%. Estimate, not
measurement -- the stage walls are not logged.

**Its worth has one measurement**, `docs/CPU_LEGACY.md` §1: three seeds at
`segment_seconds=0.4`, `steps=4` against `steps=0`, two of three moved. Nothing
since, and `_refine_controllers` logs neither how many trials were accepted nor
whether acceptance changed a candidate's placement.

**Order.**
1. Telemetry, no behaviour change: per generation, the wall of each stage
   (main, re-score, each refine step), trials accepted per step, and how many
   candidates' placement status (`new` / `improved` / `rejected`) differs
   between the pre-refinement and the final score. The second needs a dry-run
   `_place` against the pre-refinement result; it must not touch the archive.
2. Read over >= 100 generations of a run. If acceptance at step 2 is small and
   placement changes are rare, cut to 1 step or 0; if acceptance is
   concentrated in candidates already close to their cell's incumbent, refine
   only those (a funnel, placed where the cost is).
   Either is a search-design change and needs an arm, not a read.

**Pre-registered.** If refinement changes fewer than 5% of placements, it is
not buying selection and at most one step stays.

### 4. AD. Gate the flight rungs on `level_margin >= 0.7` -- **built 2026-09-30, unrun**

**Built, in both places selection and the record read.** In the air task
(`_task_scores`), the height term is `(0.55 * flight + 0.25 * glide) / 0.80`.
Its `flight` part is now paid only when `TriphibianEnv.flies_level()` is true,
meaning `level_margin >= LEVEL_GATE = 0.7`. A margin the rig could not measure
fails the gate. The glide part is paid to everyone, so a level trajectory under
the gate scores 0.312 where it scored 1.000, and a 2 m/s glide scores 0.208
either way. On the ladder, `flies_level` (`level_margin`, 0.7) sits after
`pushes_itself`, so it gates `holds_station`, `climbs` and `manoeuvres`.
Selection reads the task, and the ladder goes into `meta` and the scout's
`distance_to_next_rung`. **Air competence is not comparable across this
change**, and the air ladder now has 15 rungs. Held by
`test_holding_height_is_flight_only_if_the_actuators_can` and two mutations
(`level-gate-removed-from-the-task`, `-from-the-ladder`). Both caught.

The threshold is the user's (2026-09-30), set from arch44's gen-200
distribution: p50 0.006, p90 0.705, max 1.354. **At gen 200 about one design
in ten clears it**, so it must gate the rungs that mean flight, not the whole
air score -- gating every air rung would be the `moves at 0.1 m/s` wall again,
with 90% of the population on the floor. A missing `level_margin` (the rig
could not measure) stops `rung_reached` where it stands; it is not a zero.

**Pre-registered.** The share of evaluations clearing 0.7 rises between gen 50
and gen 300; the best air competence beats arch44's 0.038 by gen 600. If fewer
than 2% clear by gen 100 the gate is a wall and is lowered, not waited out.
Add the mutation that removes the gate and expect a check to fail.

### 5. R. `shared_ent_coef` sweep -- **queued since 2026-09-22**

See R in the arch39 list. Short, needs the machine idle.

### 6. Ray entry under corrected added mass; TEST_AUDIT 7's remaining 10 mutations -- **ray entry built; the mutations re-run 2026-09-30**

The ray half is under AK in the AB-AF list. **The mutation half is closed.**
The list said 10 older mutations waited for the machine; counted in
`tools/mutate.py` it was 17 whose `suites` still named the whole `test_search`
suite (25-40 minutes each). Each now names the one function that catches it,
and each was run alone, on `main` `1fd0a24` with the kernel linked in: 17 of 17
print `caught 1/1` with a named `[FAIL]` check, in 0.2 s to 297 s (12 of them
under 4 s; the three that drive the batched evaluator take 95, 180 and 297 s).
None survived, none needed the whole suite, none timed out; each function
passes on the unmutated tree. The table is in `docs/TEST_AUDIT.md` §7.

Found on the way: `gait-touches-one-part` had stopped applying (`mut_gait`'s
draw moved to 1.5-12 Hz on 2026-09-23) and the harness called it `MISAPPLIED`;
its text is updated. And one function picked by name did not catch its mutation
(`stage-one-reads-gross-measurements` against the gradient test), a hole in that
function, not in the suite; the island-curriculum test holds it.

### 7. AM. Host-side per-machine loops -- **the lever already named above, as an item**

"Not done, measured as the next levers" in "where a worker's time went" is the
same finding as a review's "a batch step is still a Python loop per
candidate" (2026-09-30), and it is true: `step_batch` still loops over
machines for `mj_step`, the observation, the CPG command, clearance and the
twist recording. Two parts of that review are out of date or impossible. The
energy model is already one pass for the batch (`BatchedPower`). And `mj_step`
cannot become a batch operation: the batch is heterogeneous MuJoCo models,
MJX needs one model, and `mujoco.rollout` takes its applied forces as a
sequence fixed in advance, where the fluid force depends on each step's state.

**What is left, measured** (py-spy, 2026-09-27): the damping projection 8-9%,
clearance and twist recording ~8%, inflow ~4%, CPG ~3% of worker time. Worth
roughly a fifth of a worker, not a multiple. **Constraint:** `reduceat` sums
are not bit-exact to slice sums, so these steps cannot use the 700-number
bit-identity benchmark; they need the path-agreement noise floor, and the
first such step must say what that floor is before it claims anything.

### 8. AL. Promotion, Tier-1.5, Tier-2 and the audit run in the parent with the pool idle -- **promotion and verification built 2026-09-30; the audit stays**

**Built: the round's promotions are refined as one batch through the pool,
from the basis each elite was scored with** (`_refined_controllers_for`). The
elite's `mobility_basis` is rebuilt by `MobilityBasis.bases_from_record`, which
the film's reader (`ops/run.py`) now shares. Only an elite without a recorded
basis is identified, in the same batch (per machine, AN). With a shared policy,
the baseline evaluation is skipped: `_refine_controllers` re-scores at the mean
anyway. `should_promote` is asked again before each Tier-2, so a round that
spends the budget stops where the one-at-a-time loop did. Held by
`test_promotion_spends_refinement_and_keeps_what_it_buys` (stage walls present,
and the refined controller drives the recorded basis) and the mutation
`promotion-re-identifies` (caught).

**Also built: each elite's Tier-1.5 leg and Tier-2 mission run on a worker**,
one elite per worker (`_verification_job`, `ActorPool.map`), still on the numpy
path. Seeds are drawn in the parent in the order the sequential loop drew them
(Tier-1.5, then Tier-2, elite by elite). The only difference is that an elite
later skipped by the budget recheck has had its seeds drawn, and its legs run,
for nothing. The budget, the critic label and the elite's meta stay in the
parent, in the old order. Each `promote` event gains `verify_wall`, the wall
of the whole round's legs. Held by the pool's `map` check (order kept, from
real processes, nothing re-run in the parent) and the mutation
`pool-map-out-of-order` (caught). **Not moved: the audit.** It re-evaluates
through a closure the auditor calls several times per elite and mutates the
archive as it goes. Its wall is its own `stage_wall` event, so whether it is
worth moving is read, not guessed.

The concrete cost behind "the islands are not parallel" (2026-09-30): the
islands themselves are not the lever, the idle pool during the parent stages is. `_verify_and_label` and `_audit` (`evolution/loop.py`) run
one elite at a time in the parent: `_refined_controller_for` calls
`evaluate_tier1_batch([pheno], ...)` with a batch of one and **no pool**, then
six refinement steps on that batch of one, then a 60 s Tier-1.5 leg and the
Tier-2 mission; the audit re-evaluates each elite several times on the numpy
path. The four workers do nothing meanwhile.

**Measured on arch44 (gens >= 7):** the 35 generations that carry a `promote`
or `audit` event (the audit gens all coincide with promote gens) took a median
**405 s** longer than the median plain generation (313 s), 7.5% of the run's
wall. arch44's timings are confounded by the laptop's other load (see AJ), so
that is a size, not a number to beat.

**Candidates.**
- The three promotions of a round refined together through the pool, as one
  batch of three, instead of three batches of one.
- **Do not re-identify the promoted elite.** The stored elite "carries weights
  but not the basis" -- but the basis the elite was *scored* with is already
  written to its `evaluate` event (`mobility_basis`, both media). Re-measuring
  it at a fresh seed costs a full identification per promotion and hands
  Tier-2 a basis the Tier-1 score was not earned with. Keyed by
  `(genome_id, eval_seed)` this is exact, not a cache approximation.
- Tier-2 and the audit stay on the numpy path on purpose (verification runs on
  the half that is verified), so they can go to worker processes but not onto
  the GPU path.

**Ceiling.** Three promotions spread over four workers recovers at most about
two thirds of that 7.5%, roughly 5% of a run. Log the stage walls first (same
telemetry as AK), and close this if the idle-machine share is under 3%.

### 9. AJ. Work stealing across the pool -- **built 2026-09-30, off by default; `idle` decides**

**Built, with the reproducibility decision taken first.** The learning
rollout's exploration noise now comes from one numpy stream per machine,
`default_rng([seed, place in the generation, segment])`
(`evaluate_tier1_batch(streams=...)`, `SharedPolicy.act_many(rngs=...)`), and
the pool passes every machine its place. So a machine explores identically in
any shard. Before this, the noise was torch's stream seeded per shard, so pool
shape changed what the learner saw. That contradicted `_RESOURCE_FIELDS`'
claim that pool shape "changes the result by nothing". It no longer does, to
the 1e-7 of the shared network's batch width. `plan_shards` (`envs/actors.py`)
adds `--pool-per-worker` (more shards than workers; the executor is the queue)
and `--pool-balance` (machines assigned by predicted cost,
`35 + 0.9 * DOF` s, longest first). Composition is fixed before anything runs,
so a run stays reproducible from its seed. Both are off by default, which is
exactly the old split. Held by the sharding test (a sampled rollout banks the
same 42 trajectories in one shard and in a balanced queue of three; max
distance difference 8.6e-8) and two mutations (`exploration-noise-per-shard`,
0.33; `queue-results-in-shard-order`, 19.6), both caught. **The sweep of
per-worker and balance against the real path is still to do**, after `idle`
has read above ~10%.

Raised 2026-09-29 while reading arch44 at gen 402. The pool is CPU-bound in
the workers, and it wastes part of that on the tail of every generation.

**What was seen.** 130 s of `/proc` jiffies on arch44: while evaluating, each
busy worker sits at 100% of a core, the parent at 0%, the GPU at 9%, and the
machine has 20 cores of which the run uses 4. At several instants only 2-3 of 4
workers were busy: the rest had finished their shard and were waiting on the
slowest one. Per-design wall (`evaluate` events, `wall`) varies widely inside a
generation and grows with DOF (about 0.9 s per DOF above a ~35 s floor), so the
four shards of four are not equal work, and `split()` fixes them before anyone
knows which is slow.

**What is not known, and must be measured first (no code before this):**
- the idle fraction: sum over a generation of `(slowest shard - each shard)`
  as a share of `workers x slowest shard`. Every `evaluate` event has `wall`;
  the shard's end time is in the parent's `_batched` call. Log per-shard wall
  in `ActorPool` first, for a few hundred generations of an existing run.
  If it is under ~10%, close this item.
- the ceiling. 16x1 took 93.3 s against 4x4's 31.7 s (CLAUDE.md, pool sweep):
  the lockstep batch amortises the fluid launch, so **the unit stolen cannot be
  one machine**. The candidates are more shards than workers (say 6 shards of
  2-3 pulled from a queue by 4 workers) or splitting by predicted cost (DOF)
  instead of by count. Sweep both against the real evaluation path; do not model
  it -- a wall-time model was wrong by 3x once already.
- memory. Worker count is bounded by memory, not cores (8x2 was OOM-killed);
  a queue keeps 4 workers, so it should not change that, but check RSS.

**Constraints already established.**
- Scores do not depend on sharding (`envs/actors.py`; `tests/test_search.py`
  asserts it), except that the *shared policy's sampled exploration* depends on
  how work was split. A dynamic queue makes the split depend on timing, so a run
  stops being reproducible from its seed unless each machine's actions are
  drawn from a per-machine stream. Decide that before building; it is what
  arch44's "gens 0-39 reproduce arch43" check relies on.
- `split()` and the `--min-shard` trap (one Tier-0 rejection at batch 16 gives
  `15 // 8 = 1`) are the same code; a queue removes the trap if done right.
- Bit-identity gate: the 700-number benchmark
  (`experiments/perf/reference_df95498.json`) must not change.

**Not the same as the other perf levers** in "where a worker's time went":
those cut instructions per machine; this recovers wall time the workers spend
idle. They compose.

**Pre-registered read.** Generation wall at fixed seed, before against after,
on an idle machine, over at least 30 generations (arch44's per-generation
timing is confounded by whatever else the laptop is running: 284-384 s/gen at
gens 250-400 against 86-130 s at gens 7-32). A gain under the run-to-run spread
is not a gain.

**Two more framings of the same item, raised 2026-09-30.** "Workers are capped
by `min_shard`" (`k = max(1, min(workers, n // min_shard))`, so at batch 16 any
`--workers` above 4 adds nothing) and "`batch` is both the evolutionary batch
and the hardware batch" are both this item. The hardware batch is already the
shard, not `batch`; what is still coupled is that the shard *count* cannot
exceed `batch // min_shard`, and a queue of small shards is what removes that.
One option this adds to the candidates above: pull the next generation's first
shards into the tail of this one. The next generation visits a different
island, so its parents do not depend on this generation's placements -- but
its candidates would be scored with the shared policy *before* this
generation's PPO update, and the curator's operator credit would lag by one
generation. Both change the algorithm, not just the schedule; that needs its
own arm.

### 10. Y/O. Transition-distance curriculum -- **built 2026-09-30; ON by default since 2026-10-05; its step rule is unmeasured**

**2026-10-05: turned on by the user's decision, numbers unchanged.** On arch46's elites the crossing share at back 0 is 0% against `advance_share` 0.5, and 4.1% of air segments reach 0.1 competence against the same share for the launch step, so it is expected to hold at back 0 and 30 m: it opens the door and measures, it does not yet apply pressure. `--no-distance-curriculum` restores the old probes. arch47 (curriculum off) is the baseline for the first run with it on.

**Built as a mechanism, with the step rule's numbers as parameters.**
- **The probe start.** `MissionSpec.transition_back` sets, per transition kind,
  how far back from its interface the probe starts. `air_to_water` starts
  higher, `water_to_air` deeper, `water_to_land` further seaward, and
  `land_to_water` further up the beach, where it is set down on the ground
  rather than inside it. `land_to_air` has no interface and ignores it.
- **The air launch (item O).** `MissionSpec.air_launch_height` is the air
  segment's launch height. `None` is the 30 m spawn.
- **Both paths read the spec.** `evaluate_tier1_batch` and `evaluate_tier1`
  honour both fields. Every `TransitionResult` records its `start_back`.
- **The step rule.** `curriculum.DistanceCurriculum` steps a start back by
  `step` (0.5 m) once `advance_share` (0.5) of the last `window` (200)
  evaluations crossed from the current distance. Evidence from any other
  distance is ignored, and the window clears on a step. The launch steps down
  by 2 m, to a 4 m floor, when the same share of air segments reach 0.1
  competence.
- **In the search.** `--distance-curriculum` turns it on. The curriculum is
  checkpointed and restored into the spec on resume, every step is published as
  a `distance_step` event, and the generation report carries its state.

Off by default, so every probe starts where it always has. **The four numbers
above are placeholders**: the first measurement named below (crossing rate by
start distance on arch44's elites) is what sets them, before this is turned on
for a run. Held by `test_a_transition_can_start_back_from_its_interface`
(physics) and `test_the_distance_curriculum_steps_back_only_on_evidence`
(search, both paths), and by two mutations, both caught:
`batched-transition-ignores-its-start` and `distance-counts-other-distances`.

The design is in "Y / O -- what a continuous start would do" (2026-09-22):
each transition probe's start steps back from the interface as the population
learns to cross it, and the air launch steps down from 30 m the same way.
First measurement: crossing rate by start distance on arch44's elites; the
step rule is set from it. It changes what every transition score means, so
nothing after it is comparable across it on transitions.

### 11. N. GRPO for the shared policy -- **conditional on 5**

As described under N (arch40 list). Only if R (item 5) shows the shared
policy carries weight.

**2026-10-03: built, off by default.** The mechanism is done and tested; the
decision to run it is not. See item N (arch40 list) for what it is, what it
costs, and the gates that hold it to "changes nothing when off".

---

## 2026-10-08, later — the work-list sweep

The open items of the 10-06 lists (arch48 items 1–5, PAPERS_2610, the external
review's B–E) and the MuscleMimic list (M1–M6), taken in cost-then-speed order.
Each entry carries its number; items that need a decision from the user are
collected at the end of this section, not decided here.

### arch49 — the run nobody read (read 2026-10-08)

arch49 ran 2026-10-07 (300 gens, seed 20261007, arch48's flags, at PR #32's
antipodal pair; `runs/arch49_notes.md` registered PAPERS_2610 §5's reads). It
finished with `post-run exited 0` and a `report.html`, which closes the second
half of arch48 item 2. Its draw was still shared per generation (pre-M2). One
seed: a difference from arch48 is not an effect, and water and land are not
comparable with arch48 across the antipodal pair.

| read | arch49 | arch48 |
|---|---|---|
| `mission_best`, crossings | 0, 0 in every kind | 0, 0 |
| gens with `three_media` | **0** | 46 (from 254) |
| `domain_best` max air / water / land | 0.422 / 0.294 / 0.430 | 0.553 / 0.482 / 0.676 (not comparable on water, land) |
| `weakest_best` max | 0.001 | 0.036 |
| median wall/gen (8–99 / 100–199 / 200–299) | 133 / 135 / 143 s | 116 / 122 / 129 s (+18% physics from the pair, as predicted) |
| divergence | 0.19% | 0.38% |

**Pre-registered reads (PAPERS_2610 §"Pre-registered reads for the next run"):**

1. *Auditor `mean_held_out` per medium ≥ 0.5.* **Failed:** land 0.151 (6
   audits), air 0.328 (5), water 0.091 (1). Below 0.35 in every medium, which
   is PAPERS_2610 item 5's confirmation of item 1: Tier-1 credit belongs to the
   draw, still, after the pair. M2 (one draw per candidate) is the fix that
   arch49 did not have.
2. *Tier-1 vs Tier-2 Spearman > 0.3 in water and land.* **Failed:** water
   0.174, land 0.182 (n = 192 promotions, every leg labelled;
   `experiments/tier_gap/results_arch49.json`); air 0.403. arch48, same
   protocol: -0.09 / 0.02. Tier-1 pass, Tier-2 fail at 0.15: water 6/7, land
   21/23. Better than arch48, short of the bar.
3. *The no-model gate certifies every bar the elites clear in ≥ 10% of
   cases.* **Held** (`experiments/no_model_gate/results_arch49.json`, 229
   elites, the fixed still arm, 5,736 s): the two competence bars cleared by
   ≥ 10% of elites are certified against both controls: water ≥ 0.012 (elite
   64 / still 3 / base 13), land ≥ 0.012 (51 / 0 / 0); also water and land ≥
   0.055 (14 / 0 / 0, 20 / 0 / 0).

**What the gate found besides.** *Air competence leaks*, against the elites:
a still machine clears air ≥ 0.012 on 11 bodies, the elites on 6; ≥ 0.055, 9
against 3. All 11 are gannets gliding from the launch; 7 of them clear it held
still and not under their own gait. On those bodies flapping destroys a glide
that holding still keeps, and the controller cannot hold still: the gait gain
(AA, `--gait-gain`) that gives a policy that authority has been off in every
run since arch42, whose read stopped at gen 50 on the old physics. Curriculum
stage 1 leaks (elite 15 / still 8 / base 8) and stage 4's `mission > 0` leaks
(8 / 2 / 1); crossings are now non-zero for elites and zero for still machines
(air→water 1, water→land 2, land→air 1 of 229: underpowered). The ladder's
water and land state rungs still leak, as PAPERS_2610 said they would
(telemetry, not fitness).

**The critic on arch49** (`experiments/critic_skill/results_arch49.json`):
corr(Tier-1, Tier-2) rose to 0.26 (water) and 0.25 (land), from 0.04 / 0.17;
the critic's out-of-fold prediction stays at 0.04 / 0.04, so its skill over
the cheap score is still 0. Air: 6 non-zero labels in 194.

### arch48 item 1 — where crossings fail (read)

From every evaluation's failure notes (arch48: 4,792; arch49: 4,749). The event
kept only the first three notes, so `land_to_air`, the fourth kind, was mostly
cut off; it keeps eight from 2026-10-08 (`loop.py`, the `evaluate` event).

| kind | did not hold before the command | never crossed the boundary | battery / diverged / unstable |
|---|---|---|---|
| `air_to_water` | 4,295 / 4,103 (90% / 86%) | 459 / 486 | 38 / 147 |
| `water_to_air` | 0 / 0 | 4,736 / 4,688 (99%) | 56 / 61 |
| `water_to_land` | 0 / 0 | 4,771 / 4,679 (99%) | 21 / 68 |

(arch48 / arch49.) **The wall is not the hold gate except from the air**, and
there it is the air wall again: a body that cannot stay up (C3: 73 of 200 stop
at `stays_up`) cannot hold altitude through the commanded hold. From the water,
machines hold and then never reach the boundary: neither the surface (to air)
nor the beach (to land) is reached in 6 s. **Decision:** no crossing build
before the air and water-cruise items; the crossing curriculum stays off
(item 5), because its start rate is still 0.

### M2 — one draw per candidate (built)

`evaluate_candidates` drew a seed per candidate and then passed only the first
one to the batched evaluator, so every candidate in a generation faced one
scatter and one heading (arch48: one distinct `eval_seed` in 299 of 300
generations). Now `batchroll.evaluate_tier1_batch` takes one seed per machine
(environment stream, identification probes, scatter, task), the actor pool
splits the list with the shards, refinement and the re-score keep each
candidate's draw, and GRPO's groups use the draw their body was scored on.
`SearchConfig.draw_per_candidate` (default on; `--no-draw-per-candidate`
restores the shared draw). The stage event records `draws`, the number of
distinct draws a generation faced. Held by
`test_each_candidate_faces_its_own_draw` (a machine in a mixed-seed batch
scores exactly what it scores alone at its seed, in one process or across
shards) and four mutations, all caught. **Comparability boundary:** water and
land competence, and everything that reads them, are not comparable across it.

### C2 / M2 second half — how much of a score is the draw (read)

`experiments/draw_variance` (predictions committed in 1a45e0b before the
data): arch48's 200 elites, current scoring, each at its recorded draw and six
fresh draws of its own, 1,400 evaluations, 5,466 s.

| score | R = draw var / design var | 95% CI | one-draw reliability | draws for 0.8 | rank(1 draw, mean of 5) | curse at `s` | verdict |
|---|---|---|---|---|---|---|---|
| air | 4.61 | 1.85–13.8 | 0.18 | 18 | 0.41 | 7% | draw dominates |
| water | 3.01 | 1.61–5.76 | 0.25 | 12 | 0.33 | 19% | draw dominates |
| land | 2.49 | 1.45–5.15 | 0.29 | 10 | 0.30 | 21% | draw dominates |
| `stage` | 2.36 | 1.32–4.77 | 0.30 | 9 | 0.48 | 32% | draw dominates |
| `island` | 0.83 | 0.50–1.56 | 0.55 | 3 | 0.55 | 47% | undecided |

P1 held (R ≥ 1 in water and land), P2 held for land and failed for water
(curse 0.19 < 0.20), P3 held (rank 0.33 / 0.30 < 0.7), P4 held (`stage` R ≥ 1).
Air has 29 elites non-zero on some draw, above the 20 floor. This population
is selected, which narrows design variance, so R is biased upward; but every
interval's lower end is above 1.3.

**Decision (pre-registered rule): the draw dominates in every medium and in
`stage`; sequential evaluation earns a build (M3).** One draw ranks a design
at Spearman 0.30–0.41 against its own five-draw mean. Reaching reliability 0.8
by plain averaging costs 9–18 draws, so the build is the sequential form:
score cheaply, re-score only candidates that would displace an incumbent, and
place on a lower bound. M2 alone (built) removes the best-of-16 on one draw,
but not this.

**Replicated on arch49** (`results_arch49.json`, 229 elites selected under the
current antipodal scoring, 8,654 s): R air 5.76 [2.22, 38.0], water 3.42
[1.77, 7.87], land 1.83 [1.19, 2.96], `stage` 2.33 [1.52, 4.06], `island`
1.09 [0.72, 1.63]; one-draw reliability 0.15 / 0.23 / 0.35 / 0.30; curse at
the recorded draw 52% / 25% / 18% / 31%. Same verdict in every score: the draw
dominates, on the population the current scoring selected as well as on the
old one.

### M6 — identification's share of the wall (read)

`experiments/identification_share/run.py`: arch48's 16 latest elites (rotors
0–13 each), one single-process batch, the scoring network, each elite's own
draw, timed three times each way on an otherwise idle machine: **130.3 s with
identification, 55.6 s without; identification is 57% of the batched
evaluation's wall** (74.7 s). Above the third the rule set, so G stays open as
a throughput lever. The review's cache is still not valid (bases depend on the
gait's base as well as the body, and every child carries an operator), so the
lever is the probe count: `experiments/identification_probes` (pre-registered
in b80c5a2) asks whether 12 or 8 probes per domain give bases that score the
same as 24.

**Read** (`experiments/identification_probes/results_arch48.json`, the same
16 elites, each scored once at its own draw with the bases each probe count
produced):

| probes per domain | identify wall | mean summed competence | median / max abs change vs 24 |
|---|---|---|---|
| 24 | 129.8 s | 0.0239 | — |
| 12 | 67.8 s | 0.0144 | 0.0049 / 0.046 |
| 8 | 47.5 s | 0.0264 | 0.0104 / 0.248 |

P1 held only just (median 0.0049 < 0.005), and the decision's second
condition failed: one elite moved 0.046, past one draw's SD (0.044), and the
mean fell 40%. **24 probes stay; G is closed for the probe count.** The wall
identification costs buys bases the scores depend on.

### M3 — placement on more than one draw (built, off; the form is open)

`SearchConfig.placement_draws` / `--placement-draws` (default 1): the
generation's noise-free re-score scores each candidate at its own draw and
`n - 1` more (`placement_draw_seed`), and places the median one by summed
medium competence (`median_draw`). The placed result is a real rollout: its
`eval_seed` is the draw it came from, so a film reproduces it. Refused with
refinement on (its trials are scored on one draw); promotions are untouched.
The stage event records `placement_draws` and the median spread. Held by
`test_a_candidate_is_placed_on_its_median_draw` and two mutations
(`placement-keeps-the-best-draw`, `placement-redraws-the-same-seed`), both
caught. Cost: `n - 1` more re-scores per generation (the re-score was ~41 s of
a 129 s arch48 generation, so about +80 s at n = 3).

**What the median buys, from C2's rows** (`experiments/draw_variance/median3_arch48.json`):
Spearman against an independent mean of three other draws.

| score | one draw | median of 3 (by summed competence) | mean of 3 | best of 3, inflation |
|---|---|---|---|---|
| air | 0.41 | 0.39 | 0.53 | +319% |
| water | 0.36 | 0.48 | 0.57 | +113% |
| land | 0.30 | 0.36 | 0.57 | +55% |
| `stage` | 0.48 | 0.54 | 0.65 | +61% |
| `island` | 0.59 | 0.58 | 0.72 | +71% |

The median of three by summed competence is a small gain in water, land and
`stage` and none in air or `island`, and because it picks the draw by the sum
it biases single media down (land −57%, air −74% against the reference mean).
A per-medium mean of three is clearly better (0.53–0.72) but is a result no
single rollout produced, which collides with the rule that a film must
reproduce its score. **Not turned on.** The choice (median, mean with a film
that replays every draw, or common random numbers per cell) is in the
questions below.

### PAPERS_2610 item 4 — the still machine is still (built)

Two leaks in `held_still_params`, both measured on arch48's elites before the
fix:

* **The "windmilling" rotor was commanded.** `CPGParams.clipped` keeps every
  offset 5% of half-travel inside the joint's range. A rotor's channel is a
  speed from 0 to top speed, so "stop" became 2.5% of top speed: 27.2 rad/s
  commanded on elite 1, 14.6 rad/s reached in water after 2 s. The 2026-10-04
  check passed because it read the *commanded* offset, not the speed. Now
  `CPG.margin` is per channel and 0 for rotor channels (a speed has no end
  stop).
* **The servo snap.** `scatter` posed the joints at the body's gait at a random
  phase; a still machine's gait has no stroke, so the servos snapped the joints
  to the held offset. `scatter(pose=)` now poses them at the gait the machine
  will be driven with (`ctrl.params`), in all four call sites (both segment
  paths, both transition paths). Peak joint rate in the first 0.5 s on elites
  1 / 0 / 5: 4.06 / 0.80 / 18.96 rad/s before, 1.07 / 0.28 / 0.08 after.

Held by `test_a_still_machine_is_still_from_the_first_step` (reads rotor speed
and joint rates, not commands) and the mutations
`rotor-keeps-the-travel-margin` and `scatter-poses-the-body-gait`, both caught.
**Comparability:** a candidate's own pose is unchanged (its params are its
gait), but rotors can now be commanded below 2.5% and above 97.5% of top speed
in the search: rotor bodies are not comparable across this on anything rotor
thrust touches. Every still-machine read before 2026-10-08 used the leaky arm.

### arch48 item 2 — post-run is checked (built)

`launch_postrun` now treats a non-zero exit, or an exit of 0 without
`report.html`, as a failure: it prints a `** POST-RUN FAILED **` banner with the
last lines of `postrun.log` and the command to re-run it, and writes
`postrun_status.json`. `job status` reads it for a finished search job and
prints the failure (`ops/postrun_status.py`, stdlib only so the CLI does not
import numpy). Held by `test_a_failed_postrun_is_loud` and the mutation
`postrun-failure-is-quiet`. The second half of the item, the next run's report
appearing unaided, is read when that run finishes.

### arch48 item 3 — why the critic has no skill (read)

`experiments/critic_skill/run.py` on arch48's 192 critic labels
(`search_state.pkl`):

| medium | Tier-2 non-zero | corr(Tier-1, Tier-2) | corr(critic out-of-fold, Tier-2) | best single feature |
|---|---|---|---|---|
| air | 1 / 192 | 0.126 | 0.001 | — |
| water | 92 / 192 | 0.041 | -0.067 | `log_mass` -0.138 |
| land | 94 / 192 | 0.165 | 0.034 | `land` 0.165 |

The critic is not broken, it is uninformed. Air has one non-zero label in 192,
so nothing can be learned there. In water and land the labels vary, but no
cheap feature correlates with Tier-2 beyond |0.17|, and a ridge fitted on all
sixteen to Tier-2 directly does no better out of fold (-0.08, 0.13). Every
feature is a Tier-1 quantity measured on the generation's shared draw, and
PAPERS_2610 §3 showed that draw carried the Tier-1 credit, so the critic's
inputs held no information about a design's Tier-2. **Decision:** keep it on;
it discounts nothing at skill 0. Its read moves to the first run with
per-candidate draws (M2), beside C1: a skill still at 0 there means the
features themselves are wrong, not the draw.

### C3 / arch48 item 4 — which air gate is closed (read)

arch48's 200 elites, from the archive's recorded air rungs (air scoring did not
change across the antipodal pair, so these are current). `air_gates` is empty
for 154: the airframe is not the stopper for most of the population.

| air rungs cleared | elites | stopped at |
|---|---|---|
| 0–3 | 82 | `lift_margin` 0.1 / 0.3 / 0.6 / 1.0 (12 / 14 / 40 / 16) |
| 4 | 33 | `leaves_surface`, airborne ≥ 0.1 |
| 5 | 73 | `stays_up`, airborne ≥ 0.6 |
| 6 | 10 | `glides`, sink < 3 m/s |
| 7 | 1 | `holds_height`, sink < 0.5 m/s |
| 8 | 1 | `flaps_forward`, `thrust_margin` ≥ 0.002 |

No elite clears a thrust rung. The largest group (73) carries its weight
somewhere and leaves the launch, then is down before 60% of the 8 s segment: a
fall, not a flight. Rungs 1–4 are cleared by still machines as well
(PAPERS_2610 §4: airframe, not behaviour), so the first rung that measures
behaviour, `stays_up`, is the binding one, and thrust (item 4's
`level_margin` 0.29 vs 0.7) sits three rungs above where the population is.
Air gates with a reason: 35 "no lifting surface", 11 wing loading over 937
N/m². **Decision:** item 4 (air thrust) is not the next build; nothing reaches
the rung where thrust is scored. Whether `stays_up` is closed by the body or by
the model (C3's question) needs D2's reduced-frequency read first; it is below.

### D2 — how much of the population flaps where the model extrapolates (read)

`experiments/reduced_frequency_share/run.py` on arch48's 200 elites: nominal
`k = pi f c / U` with `c = wing_area / span` and `U` the measured launch (trim)
speed. 39 have no lifting surface; of the 161 flappers, **55 (34%) are above
`k = 0.3`**, where `model_validity.md` labels the solver extrapolating (median
k 0.11, p90 1.50).

| air rungs cleared | k > 0.3 | k ≤ 0.3 | no surface |
|---|---|---|---|
| 0–3 (lift) | 25 | 19 | 38 |
| 4 (leaves) | 9 | 24 | 0 |
| 5 (stopped at `stays_up`) | 18 | 54 | 1 |
| 6+ | 3 | 9 | 0 |

34% is not small, so D2 does not stay documented-only by its own rule. But it
does not explain the air wall: 54 of the 73 elites stopped at `stays_up` flap
at `k ≤ 0.3`, where the quasi-steady model is in its validated regime. **C3's
answer:** the binding air gate is closed by the bodies and their control in
the regime the model handles, not by the model. **Decision:** D2 (wake
feedback, dynamic LEV) enters the list behind the air items that act in the
valid regime, as the fidelity item for the third of flappers above k = 0.3; no
build now.

### E2 — a pass/fail definition of "done" (proposal, for the user)

Proposed, not adopted: it defines what the project is for, so the bars are the
user's call. Built from what is already certified against still machines and
unsearched gaits (PAPERS_2610, 2026-10-06 re-gate), and from rule 8 (a score
must survive a fresh seed).

| level | passes when | today (arch48 elites, current scoring) |
|---|---|---|
| L1 medium | each of air, water, land has ≥ 5 distinct elites over a bar certified against both controls, retaining ≥ 50% of it on the auditor's held-out seeds | water ≥ 0.012: 41; land ≥ 0.15: 7; air: no certified behaviour bar (C3: nothing past `stays_up`) |
| L2 crossing | each of the four crossings is crossed by ≥ 5 elites while still machines cross 0 (`CrossingTracker` gate) | 0 in every kind (arch47, arch48) |
| L3 mission | one design completes the continuous mission (`mission_fraction > 0` with both transitions, film reproduced) on ≥ 3 of 5 held-out seeds | 0 |
| L4 target | L3 at 15 kg, 10 m depth, 45 min (README §"On the 15 kg / 10 m / 45 min target") | — |

Each level is read on the auditor's held-out seeds, never on the draw that
selected the design.

### Items closed or deferred by the reads above

| item | status | why |
|---|---|---|
| arch48 item 4, air thrust | deferred | C3: nothing reaches a thrust rung; the binding air gate is `stays_up`, and in gliders holding still beats the gait |
| D1, flexibility | deferred | its measurement is thrust lost to deflection, and thrust is not where any elite stops (C3) |
| D3, off-diagonal added mass | deferred | its own condition: only if D1/D2/C3 leave air short of thrust; they leave it short of staying up |
| G, identification | closed for probe count | M6 found it 57% of the wall; 12 probes moved one elite past one draw's noise and the mean by 40% |
| B1, `SearchConfig` and job-path defaults | open, for the user | the CLI now matches practice (refine 0 since M1); the dataclass still defaults to batch 4, 1 worker, no shared policy, and changing it changes every plan digest |
| E1 / E3, multi-seed ablations | open, for the user | ≥ 5 seeds per arm at ~12 h per 300 gens is 60 h per arm; M1 already answered the refine-steps arm offline |
| PAPERS item 2, water cruise chosen | open, for the user | the 2026-09-21 rule; the antipodal pair cancels motion the command did not choose, so what remains is the rule's wording |
| PAPERS item 3, leaking bars | partly read | water and land competence certified on arch49; stage 1 and air leak through the same glide (question 1 below); stage 4's `mission > 0` decides nothing measurable (elite missions ≤ 1e-5) |
| Skill pretraining, learnability score | waiting on M4 | M4 (running) says whether the learner learns a fixed body at all |
| M5, PPO epochs | waiting on M4 | the harness takes `--epochs`/`--minibatch` (a fixed gradient-step budget is `minibatch = 2048·E/10`) |

### Questions for the user (collected 2026-10-08)

1. **Is a passive glide a capability?** With the still machine fixed, still
   gannets clear air ≥ 0.012 on 11 bodies against the elites' 6, and pass
   curriculum stage 1 (8 vs 15). The 2026-09-21 rule says a glide is a
   capability, and the water/land half of the same question (PAPERS item 2)
   was answered by the antipodal pair. Keep paying a still glider in air, or
   score air on a commanded difference too?
2. **Turn on the gait gain (`--gait-gain`) in arch51?** It gives a policy the
   authority to stop flapping, which is what those gliders lack; it has been
   off since arch42, whose read stopped at gen 50 on the old physics.
3. **M3's form.** Placement on more than one draw is built and off. The median
   of 3 (built) buys little; a per-medium mean of 3 buys more (Spearman
   0.53–0.72 against 0.30–0.48 for one draw) but needs the film to replay
   every draw to stay "a film reproduces its score". Cost either way ≈ +80 s
   per generation at 3 draws. Which, if any?
4. **E2's definition of done** (proposal above): adopt, change the bars, or not
   yet?
5. **B1:** make `SearchConfig`'s defaults the CLI's (batch 16, 4 workers,
   shared policy), accepting new plan digests for the job path?
6. **E1/E3 budget:** run ≥ 5 seeds for the arms that matter (60 h per arm)?
7. **PRs:** #37 (refine-steps default 0) and #38 (this sweep, stacked on #37)
   are open; arch50 runs from #38's commit 2b7bb53.

## 2026-10-08 — the MuscleMimic comparison, reconciled with the repo

A second outside review compared this project with MuscleMimic (JAX + MuJoCo
Warp, thousands of parallel environments, PPO on a fixed human body imitating
reference motion). Its thesis: the bottleneck is not PPO but an evaluation
signal that morphology search, task sampling and controller learning
contaminate together, so the evaluation protocol comes before any learner
change. The thesis agrees with PAPERS_2610 §1–3. Each concrete claim was checked
against the code and against arch48's own telemetry (`runs/arch48/events.jsonl`,
`generations.jsonl`); claims about MuscleMimic itself (8192 environments, one
PPO epoch being better, network depth) were not checked.

### A. Claims checked

| review's claim | repo / arch48 | verdict |
|---|---|---|
| 10 PPO epochs over a non-stationary batch go stale; MuscleMimic finds 1 epoch better, so sweep 1/2/4/6 | `shared_epochs=10`, minibatch 2048, `target_kl=0.015` stops the epoch loop (`ppo_update`). arch48, 300 updates: KL median 0.0062 (p90 0.0078, max 0.0091), clipfrac median 0.076, **`stopped_early` 0 of 300**, 90–100 gradient steps each. The `ppo_update` docstring records the opposite failure in arch30: KL 0.0008, the policy "barely being asked to" learn | **no evidence the epochs over-reuse data**: every update ended under half the KL bound. What starves the learner is its reward: over arch48's last 50 updates `reward_by_tag` reads air 0, all four crossings 0, land 0.049, water 0.037. M5 |
| The shared policy is obs → 64 → 64 → action, ~1,200 parameters | 13,133: actor 6,726, critic 6,401; obs 33 (25 + 8 morphology), action 6 (`TWIST_DIM`) | **figure wrong, conclusion right**: no measurement says width is the limit |
| Potential-based shaping is in | `ppo.potential_of`, shaping 0.2 in every arch48 update | **true** |
| Condition the policy as π(a \| s, m, d) | the observation already carries 8 morphology features, the commanded domain (3) and 6 task channels (`TriphibianEnv.OBS_DIM`) | **already built** (task-conditioned since arch41) |
| Train tasks and selection tasks are the same, so the policy is promoted on the task it just learned | the PPO update runs after the generation is scored, so θ_t is never scored on the draw it trained on. **But the (1+1)-ES refinement does exactly this**: `_refine_controllers` (`evolution/loop.py:675`) scores each trial at the generation's seed, keeps it if `mission_fraction` rose, and the kept result is what the archive receives. That is best-of-(1 + steps) on one draw, the shape of the 63% best-of-five bias the re-score at `loop.py:715` was written to remove | **true, in a different place than claimed.** M1 |
| One task draw per generation makes Tier-1 a measurement of the draw | PAPERS_2610 §3. The antipodal pair (PR #32) cancels motion the command did not choose; it does **not** give each candidate its own draw: `evaluate_candidates` still passes `seed=seeds[passed[0]]` for the whole batch (`loop.py:641`); arch48 had one distinct `eval_seed` in 299 of 300 generations | **true and still open** (PAPERS_2610 item 1). M2 |
| Rank elites by a lower confidence bound over repeated draws; evaluate sequentially (2 → 4 → 8) | nothing computes a per-candidate spread. C2 (≥2 draws on a held subset, variance ratio) is the measurement that says whether it is needed | **not present; C2 first.** M2, M3 |
| Identification is 67% of an evaluation; skip it when the morphology has not changed | item G (arch34). arch48: identification is 460,800 of the main path's 657,584 steps (70%), 44% of all physics steps in a generation. It is batched on the GPU since `identify_batch`, so its share of *seconds* is unmeasured. It perturbs around the gait's base (`e.cpg.base`), so it depends on the gait as well as the body, and in arch48 every child carried at least one operator: reusing a parent's axes is the inherited-bases case AN says identification exists to prevent | **cost confirmed; the cache as proposed is not valid.** M6 |
| Benchmark the controller on one fixed body before coupling it to evolution | the 09-26 flight audit fixed bodies for a hand-written rotor control; no learning curve of the shared PPO on a fixed body exists | **missing.** M4 |
| Pretrain primitive skills, then search morphology | not built; no measurement says the shared policy can learn a skill on a body that does not change | **deferred to M4's read** |
| Score learnability (improvement per 1,000 transitions) | not built. The refinement log has `pre` and post results per candidate, but its criterion reads 0 (M1), so it would measure 0 | **deferred to M1** |
| The critic has no skill; factorise it after the data is clean | the critic already learns per-medium residuals (ARCH46_SPEC §1); skill 0.0 in all three media in arch48 | **agrees with arch48 item 3**, which reads it first |

### B. What arch48 shows about refinement (found on this check)

The (1+1)-ES accepts a trial only if `mission_fraction` strictly rises
(`loop.py:829`). In arch48, `mission_fraction` was above zero in **7 of 4,792**
evaluations (max 0.0001). Across 300 generations refinement accepted **29**
trials and changed **13** placements (`stages` events). So it climbs a criterion
that is flat at zero, and on the rare occasion it moves, it selects on the draw
it is scored on. Its first step rides in the re-score batch
(`merged_step_one`), so its own wall is small; the re-score itself (median 40.8 s
of a 129 s generation) is the de-noising AK kept and is not in question.

### C. The work list this adds (cost, then speed, then learning)

These slot beside the 10-06 lists, not ahead of arch48 items 1–3.

| # | item | build cost | measurement that decides it |
|---|---|---|---|
| M1 | **Refinement's criterion and its draw.** Replace `mission_fraction` with the score selection pays for, and score accepted trials on a draw other than the one reported (or report the pre-refinement score). A comparability boundary | low | offline on arch48's 200 elites: acceptances under each criterion, and an accepted trial's gain re-measured at a fresh seed. A gain that vanishes at the fresh seed is the same-draw bias, measured |
| M2 | **One draw per candidate, then C2's variance ratio.** PAPERS_2610 item 1's cheapest form, and the σ an LCB would need | trivial for the draw, low for C2 | seed variance vs between-design variance per medium. Above 1, sequential evaluation (2 → 4 → 8 draws, LCB to place) earns a build; below 1, it does not |
| M3 | **A fixed selection set (common random numbers), rotated.** Incumbent and challenger face the same draws; at rotation, incumbents are re-scored | medium | only if M2 says seed variance dominates. Overfitting to the fixed set is read by Tier-2's fresh seed, which already exists |
| M4 | **The shared policy on one fixed body.** One arch48 elite per medium, the shared PPO from scratch, 5 seeds, learning curve against a still machine and the base gait (`experiments/no_model_gate`'s arms) | medium | does the learner learn at all when the body cannot move? No: the learner is the wall and skill pretraining is premature. Yes: the coupling is |
| M5 | **PPO epochs 1/2/4/10**, at a fixed transition and gradient-step budget | low, but an arm per value | not before M4 shows a learning curve to compare; arch48 says the KL bound never bound |
| M6 | **Identification seconds**, now that it is batched: its share of `evaluate.main` wall at arch48's late bodies. A cache, if any, keys on body plan *and* base gait | trivial read | under a third of main's wall: G is closed as a throughput lever |

### D. M1 read (2026-10-08, `experiments/refine_criterion/`)

arch48's 200 elites, current code (antipodal pair), shared network, no
identification; 3 one-step trials per elite (arch48 ran `--refine-steps 1`),
every base and trial scored at the elite's own draw `s` and a fresh draw `s'`.
6,266 s wall. Predictions were committed before the data (bba2efd).

| criterion | accepted | gain at `s` (95% CI) | gain at `s'` | all trials at `s'` (control) | retained |
|---|---|---|---|---|---|
| `mission` (today) | 2 / 600 | 2e-7 | 0.0 | 1e-8 | 0 |
| `island` | 107 / 600 | 0.0018 [0.0011, 0.0027] | 0.0004 [0.00005, 0.0008] | 0.00009 | 23% |
| `stage` | 155 / 600 | 0.045 [0.034, 0.057] | 0.0038 [-0.0020, 0.0094] | -0.0026 | 8.5% |

* **P1 held** (0.3% < 2%): the criterion in use is flat at zero.
* **P2 failed** for `island` (17.8% < 20%); `stage` 25.8%.
* **P3 held**: an accepted trial keeps 23% (`island`) and 8.5% (`stage`) of
  the gain it reports. Refinement as run reports a draw-selected gain 4-12x
  its real one.
* **P4, as operationalised in `run.py`** (accepted CI lower bound > control
  mean): `island` not above, `stage` above, so the pre-registered rule splits:
  `island` → `--refine-steps 0`; `stage` → switch criterion and report a draw
  other than the one that chose. **A direct test of the difference**
  (bootstrap, accepted minus all trials at `s'`, 5,000 resamples, added after
  the data) includes zero for both: `island` +0.0003 [-0.00005, 0.0007],
  `stage` +0.0063 [-0.0004, 0.0133]. One-step acceptance is at best marginally
  better than drawing a perturbation at random.

**Decision.** Refinement's gain, honestly measured, is under 0.01 of
`stage_score` (elite mean 0.021) and not separable from noise; its cost is
about 20 s of a 129 s generation, and more at the CLI default (`--refine-steps
2`, best of 3 on one draw, where the same-draw inflation can only grow). By
cost then speed, `--refine-steps` defaults to 0 from this date; a criterion switch is
not worth building until M4 shows the policy weights can be improved at all.
Promotion's 6 steps (`--promotion-refine-steps`) are not measured here.

**By-product: C2's first read (M2).** Each elite's `base` at `s` against `s'`:

| medium | mean at `s` | mean at `s'` | nonzero s / s' | draw var / design var |
|---|---|---|---|---|
| air | 0.00095 | 0.00029 | 12 / 8 | 2.16 |
| water | 0.0119 | 0.0084 | 87 / 86 | 1.01 |
| land | 0.0188 | 0.0146 | 89 / 72 | 1.04 |

The winner's curse survives the antipodal pair: elites selected on `s` score
22% (land), 29% (water) and 70% (air) lower at a fresh draw, under the current
scoring. The ratio is at or above M2's threshold of 1 in every medium, but this
is two draws on a selected, range-restricted population (which shrinks design
variance), so it is a first read, not M2's measurement. M2 (one draw per
candidate) stays first.

### E. Not adopted

* **A larger policy, or MuscleMimic's scale of parallel environments.** No
  measurement here says either is the limit; the review says the same.
* **Skill pretraining and a learnability score, now.** Each waits on a read
  above (M4, M1).
* **Fixing evaluation seeds without rotation.** Over 900 generations of ~16
  candidates, a fixed set is itself selected against; M3 rotates it.

## 2026-10-06 — arch48 finished, and the work list

arch48: 300 gens at commit 98306a9, seed 20261006, `distance_curriculum=1`, all
else as arch47, 11.9 h. Notes and method: `runs/arch48_notes.md`. No
pre-registered reads were written before the launch, so everything here is
description, not a test. One seed per arm: a difference from arch47 is not an
effect.

| read | arch48 (gens 0–299) | arch47 (gens 0–299) |
|---|---|---|
| `mission_best`, transitions crossed | 0, 0 (all 4 kinds, PPO and ladder) | 0, 0 |
| crossing curriculum | on; `back` never advanced, `window_share` ~0 | off |
| `weakest_best` max | 0.036 | 0.015 |
| gens with `three_media` = 1 | 46 (from 254 to the end) | 6 |
| `domain_best` max air / water / land | 0.553 / 0.482 / 0.676 | 0.510 / 0.487 / 0.887 |
| qd summed over the last 8 gens (one per island) | 472 | 573 |
| critic | fitted at gen 83, skill 0.0 in all three media | fitted at gen 243 |
| divergence | 123 / 32,592 rollouts (0.38%) | 61 / 65,128 over 600 gens |
| median wall/gen (8–100 / 100–200 / 200–299) | 116 / 123 / 129 s | 108 / 121 s (8–100 / 100–300) |

**The three_media elite is real but thin.** It is one gannet (4.7 kg): air
0.036, water 0.095, land 0.651, against a bar of 0.012. Its film reproduces all
three. Held still (`held_still_params`) it scores air 0.0, water 0.068, land
0.0, so air and land are earned. Water is 72% passive, from cruise *progress*,
which `_task_scores` allows by design. The still machine scores cruise 0.009 and
hold 0. In the continuous mission it makes 0/2 transitions: land on-task 98%,
then never airborne.

**Why the crossing curriculum did nothing.** Y/O was measured on 2026-10-04 at
0 of 218 elites crossing at back 0, against `advance_share` 0.5. A curriculum
that advances on success cannot start from a success rate of zero. arch48
confirms it: the curriculum opened the door and nothing came through. This was
predictable from the 10-04 measurement and is not new evidence about crossings.

**Air is the binding medium.** In the film elite, `thrust_margin` is 0.0004,
`level_margin` 0.29 (AD's gate is 0.7), and the air task score is 0.0007. In the
triphibian elite, air is the weakest of the three. Every crossing except
`water_to_land` needs a body that can leave a surface under its own power.

**Found on the way (fixed, PR #29).** Automatic post-run had failed since at
least arch47. Importing the Mojo kernel C-`setenv`s `PYTHONEXECUTABLE` (a pyenv
shim), so children with `env=None` ran as the system Python. `ops.run.child_env`
fixes it; test plus mutation `child-inherits-c-environment`. Two runs'
`post-run exited 1` lines went unread, and nothing checks for that line.

### The work list (cost, then speed, then learning)

| # | item | build cost | loop speed | mechanisms / learning | why here |
|---|---|---|---|---|---|
| 1 | **Where crossings fail**: `experiments/transition_distance` on arch48's elites, broken down by failure note (`never crossed the boundary` vs `did not hold before the command`) per kind | none: an existing experiment | none | **highest**: says whether the wall is the hold gate, reaching the interface, or the medium change itself | crossed 0 in arch47 and arch48, the two runs since the 10-03/10-04 crossing fixes; no build should be chosen before this read |
| 2 | **Post-run is checked**: `job status` and the run log fail loudly on a non-zero `post-run exited`, and the next run's `report.html` is confirmed to appear unaided | trivial | none | hygiene | two runs' reports silently missing |
| 3 | **Critic skill 0**: why does a fitted critic with 192 labels have no skill? Label distribution per medium, out-of-fold predictions against the cheap score | low: a read on the checkpoint | none | learner: the critic is either uninformative by construction or starved of labels | arch45 never fitted it and arch48 fitted it to nothing; decides whether it stays on |
| 4 | **Air thrust**: from `level_margin` 0.29 toward AD's 0.7 on the search's own bodies. Re-read the AH probe (torque, 7–12 Hz) against arch48's air elites before building anything | medium; set by 1 and AH | none | **high** for the mission: no air-to-anything crossing exists without it | the binding medium in both elites read |
| 5 | **The next run**: notes with pre-registered reads before launch, 900 gens read at 500 (`how long to run`), crossing curriculum off unless 1 finds a nonzero start rate | none | — | — | arch48 had neither notes nor the length for `three_media` (born at gen 254) to show anything |

## 2026-10-06 — external review reconciled with the repo, and the correction plan

An outside review (a README-level reading of the project; no run logs, no code
execution) concluded the project misses its 15 kg / 10 m / 45 min goal because of
five causes. Each claim was checked against the repo before anything entered the
plan. Dates and the review's Gantt chart are dropped on purpose; items are ranked
by dependency, and the existing arch48 list above (items 1–5) still comes first.

### A. Claims checked

| review's claim | repo | verdict |
|---|---|---|
| The search does not train the controller (`controller_refine_steps=0`); fitness is an untrained controller | the default is 0 (`evolution/loop.py:154`), but every run since arch34 passes `--refine-steps 2` (AK, line ~2620), promotion refines 6 steps (`_refined_controllers_for`), and the shared PPO policy trains inside the search (`--shared-policy`) | **wrong for the runs that exist.** True of the *default*, which is a trap: a run launched without the flag gets 0. B1 |
| Tier-1 vs Tier-2 Spearman ≈ +0.077 | arch48 measured -0.09 (water) / 0.02 (land), air undefined (Tier-2 air is 0 for all 121); cause found: one task draw shared by 16 candidates (PAPERS_2610 §1–3) and fixed the same day | **outdated figure, right symptom.** The cause is identified; the post-fix correlation is unmeasured. C1 |
| Fluid model omits vortices and added-mass tensor | `docs/model_validity.md`: no wake feedback, LEV is an instantaneous fit, 6x6 added mass is reduced to a scalar mass + diagonal inertia; strip and bluff added mass exist | **true, already documented.** Not new |
| No structural dynamics | `model_validity.md`: "Bodies are rigid in the dynamics. Structural compliance is checked statically" | **true, already documented.** No measurement yet says it limits the search. D1 |
| No GPU use (README wording) | Mojo GPU kernel scores the search (`mojo/build/*.so`, CLAUDE.md) | **README is stale, if it says so.** B3 |
| Shared policy does not generalise across bodies | not tested in this review; `docs/LEARNER_AUDIT.md` and arch48's critic (skill 0.0 in all three media) are the repo's evidence | **unverified.** Covered by arch48 item 3 |
| No hyperparameter documentation | exposed in `SearchConfig` and recorded in each `run_start`; no single table, no sensitivity sweep | **partly true.** B2, E1 |
| Goal unmet; no quantified success criterion | README §"On the 15 kg target" argues the limit; `mission_best` is 0 in arch47 and arch48; no pass/fail bar for "done" | **true.** A rung ladder exists but no end condition. E2 |
| One seed per arm, no statistical test | ROADMAP says "One seed per arm: a difference from arch47 is not an effect" | **true, and already acknowledged.** E3 |
| No issue tracker / PR review record | out of scope for the roadmap | **dropped** |

### B. Configuration and reproducibility

| # | item | measurement that decides it |
|---|---|---|
| B1 | **Make the default match the practice** | **partly done 2026-10-06**: `ops.run search` now defaults to batch 16, workers min(4, cores), `--refine-steps 2` (0 since 2026-10-08, M1), shared policy on when torch imports (`test_the_search_cli_defaults_are_the_stored_run_configuration`, mutation `search-cli-refine-default-two`, was `-zero`). These are the stored-run values, **not measured optima**: only pool shape, batch and workers were swept; refine 2 and the shared policy have no read (AK, E1). **Open:** `SearchConfig` still defaults to 0/off, and so does the job path (`experiment new --set`) when a plan omits them; changing `SearchConfig` touches every test that builds one, so it needs its own change and an explicit decision on plan digests |
| B2 | **One hyperparameter table**: every `SearchConfig` field, its default, the value arch48 used, and whether a measurement or a typed number set it (the existing §"What is set by measurement, and what is typed" is the template) | `test_index.py`-style check that the table names every field of `SearchConfig` |
| B3 | **README matches CLAUDE.md** on compute: GPU kernel, `--workers`, queue pool, the per-generation cost (110 s early, 225 s late) | **done 2026-10-06**: the README said "4 CPU cores with no accelerator", "controller not trained at all", `--min-shard (8)` and quoted rho +0.077 as current; each was false against the code and is rewritten with the date of the change that made it so. The external review repeated all four, so a stale README is a measured source of wrong conclusions, not a style issue |
| B4 | **A run is a checked-in config**: `experiment new` already stores one; add a `runs/<run>_config.json` export to the notes template so a second machine can reproduce a launch without the shell history | launch the same config on a second seed set from the export alone |

### C. Evaluation validity

| # | item | measurement that decides it |
|---|---|---|
| C1 | **Re-measure Tier-1 → Tier-2 after the antipodal pair**: Spearman and the Tier-1-pass/Tier-2-fail table (PAPERS_2610 §3) on the next run's verified elites, per medium, with the same 121-elite protocol | the pre-fix numbers are the baseline; a post-fix rho still near 0 means the gap is not only the draw. Pre-register the read in the next run's notes (arch48 item 5) |
| C2 | **Tier-1 ≈ Tier-2 draw protocol**: Tier-1 scores averaged over ≥2 task draws on a held subset, to put a number on seed variance against between-design variance | variance ratio per medium; if seed variance exceeds the design signal, selection is noise and no later item matters |
| C3 | **Air is zero in Tier-2 for all 121 elites**: split the air score into its gates (`thrust_margin`, `level_margin`, height) and report which gate is closed for each, before any fluid change | this is arch48 item 4's read; the review's "improve the fluid model" is only justified if a gate is closed *by the model* rather than by the body |

### D. Physics fidelity (build only if a measurement says it limits the search)

| # | item | measurement that decides it |
|---|---|---|
| D1 | **Flexibility**: before building a compliant-wing model, measure how much of the rigid-wing thrust a static-deflection correction would remove. `physics` already checks compliance statically; apply that deflection as a pitch lag in the strip model on 20 elites and compare thrust | thrust change under 10% means rigidity is not the wall; over 30% means D1 enters the work list ahead of fluid changes |
| D2 | **Wake feedback and dynamic LEV**: the two entries in `model_validity.md` "cannot represent". Rank by a reduced-frequency sweep (`derivations/reduced_frequency.md`) on the elites: what fraction of elites has `k > 0.3`, where the model is labelled extrapolating | share of elites in the extrapolating regime; if small, D2 stays documented-only |
| D3 | **Off-diagonal added mass**: build only if D1/D2 and C3 leave air short; the F-01 and J-01 errors in `model_validity.md` come first because they are measured model *errors*, not omissions | the existing `experiments/added_mass` and `experiments/jet_energy` reproductions |

### E. Experimental design

| # | item | what it adds |
|---|---|---|
| E1 | **Ablation arms, one change each**: refine-steps 0 vs 2 vs promotion-only (AK's two levers are built and unread), shared policy on vs off, critic on vs off | AK has no read; one arm per lever on the same seed set |
| E2 | **A pass/fail definition for "done"**: for each medium and each of the four crossings, the bar, the elite count required, and the held-out seed count. Take the bars from the measured still-machine and unsearched-gait distributions, as the water bar (0.012) was | closes the review's "no quantified success criterion"; reuse the certified/underpowered table from 2026-10-06 |
| E3 | **Seeds and a test**: ≥5 seeds per arm for the arms that matter (E1), a rank-based test on `weakest_best`, `domain_best` per medium and crossing count, effect sizes reported with the interval. At 12 h per 300 generations a 5-seed arm is 60 h, so run E1 on `segment_seconds`-reduced configs first and confirm one pair at full length | a difference under one seed's spread is not reported as an effect |
| E4 | **Comparability**: every new arm states which of the boundaries in CLAUDE.md it crosses; across the antipodal-pair boundary, water and land competence are "not comparable" | already a rule; checked at each notes file |

### F. What the review proposed that this plan does not adopt

* **Raising `controller_refine_steps` to 50–100.** AK estimates two steps at
  ~35% of a shard's generation; 50 steps would cost an order of magnitude more
  with no measurement that refinement is the limit. E1 reads the existing 0/2/6
  first.
* **Optuna/Ray Tune over CMA-ES and PPO.** The binding failures are scores that
  rewarded the wrong thing (rules 1–8 in CLAUDE.md), not tuned constants.
  Sweeping a hyperparameter against a score that fails the still-machine check
  tunes the leak. Revisit after E2.
* **Domain randomisation / conditional policy for the shared controller.** The
  critic has skill 0.0; the shared policy's generalisation is unmeasured. Arch48
  item 3 reads it first.
* **A larger GPU budget (4–8 GPU-months, >100 GB).** Not supported by any
  measurement here; the pool is bound by memory before cores (CLAUDE.md).

### G. Documents are updated in the same change as the fact

The review's two false headline claims came from README sentences that were true
once (4 CPU cores, controller default 0, `--min-shard 8`, rho +0.077) and were
not edited when the code moved. Rule, also in CLAUDE.md: a change that makes a
sentence in README, ROADMAP, CLAUDE.md or `docs/` false edits that sentence in
the same commit. A number quoted as current carries its run and date; a number
that was superseded says so where it is quoted.

| # | item | measurement that decides it |
|---|---|---|
| G1 | **Doc-drift check**: a script (`tools/doc_drift.py`) that extracts defaults quoted in README/CLAUDE.md (`--min-shard`, `--refine-steps`, `controller_refine_steps`, `--workers`) and compares them with `SearchConfig` and `ops.run` argparse defaults; exits 1 on mismatch | run it on the pre-fix README: it must report `--min-shard 8` and the "defaults to 0" sentence (mutation: reintroduce each) |

### Order

1. arch48 items 1–5 (unchanged).
2. B1–B3 (cheap, remove ways to run the wrong config).
3. C1 and C2 inside the next run's pre-registered reads; C3 with arch48 item 4.
4. E2, then E1 and E3.
5. D1, D2, D3 only on the measurements above.

## 2026-10-03 — the eight-item pass (the user: do not start arch47 yet)

The user's list, in their words: R sweep; AM noise floor and build; AJ on by
default; Y/O measured; N if it can be done; item 6 tested; one new island that
covers all three media at once; the queue pool now, and keep looking for the
real fix to rotor cost. State, updated as each lands:

| item | state |
|---|---|
| AJ | **on by default**: `--min-shard 2 --pool-per-worker 2` (8 shards of 2 on 4 workers, the budget sweep's 0.875x). `shard_cost` now reads rotors: on 21 728 evaluations of arch45/46, rotors R^2 0.42/0.25, DOF 0.19/0.07, and DOF's coefficient is -0.03/+0.03 once rotors are in (`experiments/budget_sweetspot/cost_model.py`). Balance stays off until a re-sweep with the rotor cost. Test `test_the_pool_queues_and_balances_by_rotors`, mutation `balance-reads-dof-not-rotors` (caught). |
| rotor cost | **vectorised, bit-identical on the development machine** (2026-10-03; on a GitHub runner the rotor check differs by up to 2.27e-13, so CI compares at 1e-12): `RotorBatch` steps every rotor of a shard as one array computation and `bemt_many` builds a table in one pass. Was 198 us per rotor-step, 82% of a rotor-heavy evaluation's wall, plus ~2.5 s per table; the +1.6 s/rotor slope is the re-score's 10,500 steps x 198 us. Independently re-checked 2026-10-04: `profile_shard` against its base `be3dbe0`, 780 numbers, 0 differ (the checked-in `reference_df95498.json` is stale: 408 of 700 numbers moved since by the ray-entry and crossing changes, so it is replaced by `reference_be3dbe0.json`). Interleaved A/B: per rotor-step 0.120-0.134x, rotor-heavy evaluation 0.180-0.211x (4 arch46 elites, 52 rotors); 747 + 750 + 780 dumped numbers and 32 fixture cases bit-identical (`experiments/perf/NOTES.md`). Mutations `rotor-batch-one-table`, `rotor-table-one-sum` caught. |
| AM | **noise floor measured, one part built** (2026-10-03): floor on rotor-bearing elites 1.2e-9 (bar 1e-5), unused because every change is bit-identical. `clearance` (18% of a shard's profile, read up to 4x a step on an unmoved state) is memoised per state and filled per batch in one pass (`clearance_many`): profile 4.36 -> 1.77 s, wall on/off 0.87-1.08 (mean 0.93), inside load noise on the rotor batch. Not done: damping projection (7%, batching breaks bit-identity), inflow (~7%), CPG (1.6%), entrainment (3.4%). Mutations `clearance-batch-first-slice`, `clearance-memo-keyed-on-time`. |
| island | **built 2026-10-03, unrun: `triphibian`, the eighth island.** Why: in arch46's 7,804 Tier-1 evaluations the weakest medium is 0 in 98.2%, c2 (the second-best) in 77.6%, and no evaluation clears 0.15 in all three; `generalist` scores `mission_fraction` (zero for 98.8%) and its curriculum stages 0-3 pay the best one or two media. Objective: `triphibian_score` = harmonic mean of `c_i + e` minus `e`, `e` = 0.006: any machine missing a medium scores < 0.012, any machine with all three >= 0.012 scores >= 0.012, the score is strictly increasing in every medium (so it still ranks c2 where c3 = 0), and it carries no energy, transition or take-off factor. The geometric mean was rejected: it puts air 0.9 + water 0.9 + land 0 (0.164) above 0.1 in all three. Ladder (`Curriculum(weakest=True)`): stage 0 reads c2 (bar 0.055 = p95 of c2; p90 is 0.020 and a still machine reaches 0.0199), stage 1 reads c3 (bar 0.012 = p95 of c3 among the 396 evaluations past stage 0), stages 2-3 read the crossings gated on c3, and each stage holds at 0.4 of its own bar. Still machine (`experiments/triphibian_still`, 7 plans x 2 seeds, real Tier-1 path): max island score 0.0081, max c2 0.0199, max c3 0.0000, 0 promotions. A one-medium machine (arch45's fitness-1.0 design, 0/0.337/0.003) scores 0.0047; two perfect media and no third 0.0118; 0.1 in all three 0.100. Routing: specialist pairs stay with their pair islands, and each pair island's champion is crossed with the specialist of its missing medium and sent to `triphibian` (`TRIPLE_HOME`, 3 crosses per migration). On resume an island the checkpoint lacks starts empty and is offered the archipelago's best elites by its own objective as immigrants (`Archipelago.colonists`). The generation report gains `weakest_best` and `three_media` from this island's archive. Tests: `test_the_triphibian_island_pays_the_weakest_medium`, `test_a_still_machine_climbs_nothing_on_the_triphibian_island`, `test_a_pair_cross_reaches_the_triphibian_island`, `test_the_triphibian_island_joins_a_resumed_run`; mutations `triphibian-island-pays-the-best-medium`, `triphibian-ladder-reads-the-best-medium`, `triphibian-curriculum-is-the-shared-ladder`. Not comparable: a run with eight islands visits each 1/8 of generations, not 1/7. |
| R | **read 1 (2026-10-03), the shared policy's weight**: arch46's 218 elites re-scored paired, with minus without the network that scored them (`experiments/shared_policy_value`): land +0.0247 (t 3.50, 69 better / 46 worse), water +0.0126 (t 2.59), air +0.0061 (t 1.69), mission 0; median delta 0 in every medium; the land island holds ~62% of the land delta. An upper bound (elites were selected with the network). **Read 2, the sweep**: `runs/r_{ent0.01,ent0.1,ent0,off}`, 60 gens each at seed 20261003, batch 16, 6 s segments, refine 0; arm 1 at `be3dbe0`, the rest at `6989c7b` (= `be3dbe0` + the bit-identical rotor change, same results, 0.73x the wall). **Read 2026-10-04**: the coefficient does what it says to the width (`log_std` over 60 gens: ent0 -0.50->-0.76, ent0.01 -0.50->-0.56, ent0.1 -0.47->-0.17), but nothing it buys shows in competence. Paired with-minus-without per arm: ent0 water +0.0135 (t 2.24), others |t| < 2 (ent0.01 land +0.0144, t 1.98); top-10 competence (air/water/land) ent0 0.19/0.39/0.52, ent0.01 0.29/0.37/0.47, ent0.1 0.21/0.31/0.52, policy off 0.12/0.38/0.55 -- inside one seed's swing (air ±0.1). **Decision: keep 0.01; GRPO (N) stays off**; the shared policy's weight at 60 gens is not distinguishable from zero, and arch46's +0.025 land is the upper bound of a selected population. `--shared-ent-coef` exists; the `ppo` event carries `log_std` |
| Y/O | **measured 2026-10-04, the curriculum cannot start** (`experiments/transition_distance`, arch46's 218 elites, current crossings): elites cross 0-2.3% at back 0 (air_to_water 1.4% flat over 0-8 m, water_to_air 0, water_to_land 2.3 / 0.9 / 1.4 / 2.3 / 0.5 / 0% at 0/0.5/1/2/4/8 m, land_to_water 0), against `advance_share` 0.5; a 200 window holds 3-5 crossings. **Still machines cross at or above the elite rate** (air_to_water 1.8-2.3%, water_to_land up to 5.0%, 5 of 8 still crossers at 0 m are aerial_diver bodies): the 10-03 fix cut the leak from 0.83-0.87 to ~0.02-0.05, not to 0, so these cells are not evidence of skill. Air competence falls as the launch is lowered (elite mean 0.0123 at 30 m, 0.0017 at 4 m; share >= 0.1 4.1% -> 0.5%), so the downward launch step would remove the only air signal. Next: close the still-machine crossings, then re-measure; the typed numbers stay, unreachable rather than wrong. **2026-10-04, still crossings closed** (the ninth instance; section below): with the air hold on energy height, land arrivals required `ashore`, shore progress counted from the command, aborted probes holding nothing, and rotors stopped in the still machine, still machines cross **0 of 218** in every kind at 0 m and 0 of 29 former crossers at 0-8 m (were 5 / 1 / 8 of 218 at 0 m). Elites at 0 m: 0 of 218 (were 3 `air_to_water`, 5 `water_to_land`), all of them still-machine crossings. One real crossing remains: elite 159, `water_to_land` from 2 m, not made by its still or rotors-on twin. The curriculum still cannot start (0 at back 0 against `advance_share` 0.5), and now for the right reason. Transition scores are not comparable across this change |
| N | **built, off by default, 2026-10-03**: `--shared-learner ppo\|grpo\|ppo+grpo` (default `ppo`), `--grpo-bodies 4`, `--grpo-group 4`; switched on only if R says the shared policy carries weight. Off is bit-identical to the old update (weights and report equal against `main`'s `ppo.py`); generation 0's archive is equal with it on and off (`test_grpo_rollouts_never_reach_the_archive`); mutations `grpo-*` caught 7/7 (re-run independently) |
| 6 | ray entry: whole `test_physics.py` at `be3dbe0` (kernel linked), 0 `[fail]`, `all physics checks passed` with no skip, `test_entry_shock_is_hydrodynamic_not_a_speed_limit` included. Mutations: all 117 in `tools/mutate.py` re-run at `be3dbe0` (52 name a `test_search` function, not 18): **caught 116/117**; the survivor `audit-perturbs-another-seed` is a fixture hole -- the test sets the audited elite's `eval_seed` to 0, so re-running at seed 0 is the same experiment. **Fixed 2026-10-04**: the fixture audits at seed 5 (the gannet's mission base is nonzero at seeds 0 and 5, zero at 1, 2, 3, 7); the test passes and the mutation now reads `retained 2.8731`, caught 1/1, so **117/117** |

### 2026-10-04 — still machines that still crossed (the ninth instance)

`experiments/transition_distance` found arch46's elites held still crossing at or
above the elite rate under the 10-03 `CrossingTracker`. Traced singly
(`experiments/still_leak/`), the mechanisms were:

- `air_to_water`: gliders coast through the 1.5 s height hold on their launch
  speed. They lost 0.09-0.32 m of height and 3.4-5.7 m of energy height, then
  glided in.
- `water_to_land`: a float with its root 0.09 m above the water and its hull on
  the submerged ramp is LAND by `medium_of`, 8 m out.
- `water_to_air`: the still arm left rotors at throttle (a rotor's channel is a
  speed held at its offset).

The gates, in `transitions.CrossingTracker`, so both paths:

- the air hold reads `max(height, energy height)` lost;
- a land arrival must be `ashore` (dry ramp under the root);
- shore progress counts from the go command (a capsize during the hold paid
  0.136);
- an aborted probe holds nothing (a blow-up paid 0.543).

`TriphibianEnv.held_still_params` stops rotors. Tests:
`test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing` (24 bodies, a
fixture of their genomes) and two new checks in
`test_a_crossing_is_commanded_and_a_still_machine_makes_none`. Mutations
`crossing-air-hold-reads-height-only`, `crossing-land-is-any-ground-contact`,
`crossing-shore-progress-counts-the-hold`, `crossing-aborted-probe-keeps-its-hold`
and `still-machine-leaves-rotors-spinning` are all caught.

Re-measured: still machines cross 0 everywhere, elites 0 at 0 m, and one real
crossing remains (elite 159, `water_to_land` at 2 m).

Left open, both in ARCH46_SPEC §8:

- **Open-loop rotors still cross.** Rotors at throttle with amplitude zero cross
  `water_to_air` (elites 129, 137) and `water_to_land` (129, 146). They ignore
  the command and reach the interface after the hold. Closing it needs a station
  gate on the hold.
- **Graded `water_to_land` pays a still glide.** A sinking glide that drifts
  shoreward earns up to 0.145 for a still body.

**Transition scores are not comparable across 2026-10-04.**

## 2026-09-23..26 — AB-AF executed, every open fluid item closed, and a rotor control

The user asked for three things: take the AB-AF list through to verification and
fixes, add a propeller model as a comparison, and close every open item in the
fluid model. Every number below came from a script in `experiments/robofly`,
`experiments/rotor` or `experiments/flight_audit`, and the running log is
`runs/_logs/plan_0923b.md`.

**The short answer.**

- **Physics and score can see flight.** A quadrotor built through the same
  pipeline scores **0.963-0.987** on the real air segment (0.86-0.93 once AE's
  worst-second height term is in -- see "arch43's first launch"). The best flapper ever
  scored 0.378.
- **Level flapping flight exists only with feathering.** It is rare: one teal
  gait in about 300, at margin 1.26, with ideal kinematics. No heave-only gait
  of any plan reaches it.
- **Real actuation cannot deliver that stroke.** On a fixed rig the same gait
  flies level with its kinematics prescribed (margin +1.28). With its joints
  free the margin is 0.42, the motors are torque-limited 60% of the time, and
  some joints reach only 19% of their stroke. **The wall has moved from the
  fluid model to actuation at 7-12 Hz.**

### 1. The fluid model — every open item closed (MATH_AUDIT)

| id | what was wrong | fix | measured |
|---|---|---|---|
| F-02 | LEV keyed on pitch rate, zero on a revolving wing | LEV strength = max(Rossby, travel since reversal); separated branch `CN sin a cos a`, CN 1.98 -> 3.4 | robofly (Re 136): CL rms 0.473 -> **0.159**, CD 0.678 -> **0.095** |
| F-03 | isotropic wing added mass, 2.8x; its surplus was holding explicit lift/drag stable | tensor on, plus `ImplicitAeroDamping` on **articulated joints only** | dt 0.004, 5 s: teal in water 1718 rad explicit, 2.29 implicit; every plan bounded in both media |
| F-04 | limiter per strip | per machine, one uniform scale | test |
| F-08 | no stall: CL rose monotonically to 45 deg | handover starts at stall and completes 6 deg later (a modelling choice) | CL peaks in the first third of it and loses >10% by its end |
| F-12 | AR per part (induced drag ~2x); pressure drag in attached flow | inflow over the tip-to-tip span; the pressure term only on the separated branch | camber no longer costs L/D |
| F-13 | no inflow, no Wagner | Glauert inflow with a one-radius lag; Jones's Wagner lag | hover and forward-flight limits to 1e-3 |
| F-14 | entrained weight cancelled at the strips (a couple) | at the centre of mass | test |
| S-02 | slam peak 4x | derived: `(pi / (2 tan b))^2` | test |
| N-02 | medusa 79,045 rad | closed by F-09 (the lever arm) | 1.06 rad |

Four defects found on the way, each fixed and tested:

- **Jets and the damping split.** Jets overwrote `dof_damping` with a dry
  snapshot, which erased the split's damping and left explicit anti-damping;
  medusa ran away. Now the solver owns the array and jets add to it.
- **The slam diagnostic.** It differenced the direction-dependent tensor, which
  read flapping as slamming. It now uses the wing's normal entrained mass.
- **The quasi-static probes.** Trim, lift and thrust would have restarted the
  new history at every pose. `FluidSolver.steady()` evaluates them without it.
- **The split on the free root.** It added about dt*B of pitch inertia and broke
  the gannet's glide under the launch scatter (sink 1.5 -> 10.5 m/s). It now
  applies to articulated joints only, and the gannet glides at 1.48 m/s median
  (air score 0.162).

The Mojo kernels mirror all of it. Their unsteady history sits in persistent
buffers, and the inflow and the limiter run in numpy that both paths share.
Gannet and beetle agree step by step to 1e-12 over 400 steps; eel drifts to
1e-9 by step 164.

### 2. AF — the robofly fixture

`experiments/robofly/run.py`: a revolving wing (R/c 2.9, Re 136, mineral oil)
compared with the fits of Dickinson, Lehmann & Sane (1999). **CN_LEV = 3.4 is
read from the same fit**, so CD at 90 deg is not an independent check. The
shape across incidence, and CL through the inflow, are. It is gated in
`test_physics`: CL rms < 0.25, CD rms < 0.20.

### 3. The rotor control

- **The model.** `physics/rotor.py` is blade-element momentum theory with
  Prandtl tip loss and Glauert's forward-flight term. Its section coefficients
  are the wing functions.
- **Checked against the UIUC APC 10x4.7 SF.** One number is fitted: camber, to
  static CT. Out of sample, CP rms is 0.003 and CT rms 0.012 over J 0.1-0.6.
  Static CT is 19% low. MATH_AUDIT R-01.
- **The machine.** A rotor is a gene on a part: a spinning body on a velocity
  servo, with its own motor and mass. A mirrored rotor is the opposite hand, so
  pairs cancel in yaw. A machine its rotors can lift is airworthy and trims
  level.
- **`REFERENCE_PLANS["quad"]`, never seeded.** A textbook cascade controller
  flies it through the real segment (`experiments/rotor/fly.py`): **0.963,
  0.974, 0.987**, sink about 0, height 0.975, turn 0.95-1.0. Under AE's worst-second height term it is 0.86-0.93
  ("arch43's first launch").

### 4. AB — the feathering wing: built, probed, and where it stops

- **Built.** `"universal"` is now a stroke hinge plus a feathering hinge about
  the span, each with its own motor. The feathering lead is a gene,
  `feather_lead`.
- **Quasi-static.** The best `thrust_margin` over 300 gaits is gannet +2.50 and
  teal +4.51 (heave-only gannet: +0.65). **But `thrust_margin` is blind to
  lift:** the gannet's best gait pulls the machine *down* with 0.39 of its
  weight (MATH_AUDIT C-12).
- **Level flight exists, rarely.** `level_flight.py` searches gait, speed and
  pitch, with margin `min(<Fz>/W, 1 + <Fx>/W)`:

  | plan | heave-only | feathering |
  |---|---|---|
  | gannet | 0.84 | 0.80 |
  | teal | 0.73 | **1.26** (11.3 Hz, 6.8 m/s, 43 deg; share >= 1: 0.003) |
  | beetle | 0.92 | 0.57 |

- **Free flight: none.** The best quasi-static gaits flown open loop through the
  scored segment (`feather_fly.py`) sink at 9-14 m/s, competence 0.
- **The fixed rig separates why** (`fixed_rig.py`, `rig_level.py`).

  | gait | prescribed kinematics | free joints |
  |---|---|---|
  | gannet +2.50 (thrust) | +2.62 / +2.86 with the unsteady terms | -1.64 |
  | teal level gait (margin) | **+1.28** | **0.42** |

  On the teal gait the motors are torque-limited 60% of the time, and the lags
  are 30-141 deg (MATH_AUDIT A-02). So the fluid model is consistent, and the
  new unsteady terms help flapping slightly.
- **Springs as built do not rescue it.** A series spring tuned to the gait
  frequency gives margin 0.115; a compliant drive cuts saturation to 15% but
  not the margin.

### 5. AC, AD, AE

- **AC — done.** `flap_frequency` is drawn from 1.5-12 Hz in both
  `random_genome` and `mut_gait`, and the Tier-0 spar check closes the band.
- **AD — deliberately not done.** Of the candidate signals, `thrust_margin` is
  lift-blind and the quasi-static level margin does not survive the servo, and
  selecting on a signal nothing can reach is arch38 again. The prerequisite is
  item AG below.
- **AE — done.** The air height term takes the worst one-second drop inside the
  airborne stretch: a 5 m drop-and-recover scores 0.052 against 1.000.
  Curriculum stage 1 counts air progress only while airborne.

### Test changes, disclosed

- The reversed-flow and robofly baselines subtract the entrained weight
  directly, not the still-air force; the tensor made those two differ.
- `test_stall_blend.py` is rewritten for F-08 and the normal-force branch. The
  F-05 properties are kept.
- The series-spring checks. The compliant drive measures 0.49x the rigid cost
  per unit of motion, so the original < 0.6x check stands. A spring on the stiff
  drive saves 11%, so that check now says "buys little" (> 0.8x) instead of
  "buys nothing" (> 0.9x).
- Two of the ray's entry orderings are printed, not asserted (AK): nose-first
  against flat at the same speed, and nose-first at twice the speed. Flat
  getting worse with speed, the hull limit, and the gannet surviving a 20 m/s
  nose-first entry are all still asserted.
  (Closed 2026-09-30 by AK: see AK below for what is asserted now.)
- The audit fixture audits a gannet (its mission base is nonzero).
- Four tests read new model constants (CL_MAX = CN_LEV/2, the LEV strength
  argument).

### Comparability

**Not comparable across 2026-09-23..26:** every fluid force, every actuated
motion, the air score's short exit and height term, and curriculum stage 1.
This is the second boundary in a week. Nothing has been run since the first.

### The work list this sets

- **AG. A level-flight margin through real actuation, published.** The fixed-rig
  measurement with free joints, at the machine's own gait, speed and attitude.
  Then AD selects on it, as a gate.
- **AH. Actuation at 7-12 Hz.** Torque is the wall (A-02). Three candidate
  directions, each to probe on the rig before building:
  - resonance co-design: a spring reference at the gait's offset, and tuning
    that counts the added mass
  - sizing the feathering motor to the pitch load rather than to the stroke
    motor
  - a torque-feasibility term in the gait operators
- **AI. Should the search build rotorcraft?** Yes (the user, 2026-09-26, and
  again 2026-09-30); `mut_rotor` built for arch43.
- **AJ. Path agreement against the noise floor — done, and the first
  explanation was wrong.** Two checks failed after the fluid work: eel,
  no-identification, 1.5e-5 against a 1e-5 bar; record against film, 2.7%
  against a 2% bar.
  - **One kick is the wrong probe.** A 1e-16 m/s wind kick is below float
    resolution. At 1e-12 the eel's floor came out at 2e-8, which does not
    explain 1.5e-5.
  - **What the paths actually differ by.** Tracing step by step, the first
    difference is 2.5e-15 relative at step 2 (rounding). It grows smoothly to
    1e-9 by step 131, about e^27 per second, on a contact-rich gait.
  - **No leak between machines.** One machine alone and the same one in a batch
    of four are identical.
  - **Eel.** The single path with its fluid forces dithered at 1e-15 each step
    moves 1.45e-5, the same as the path difference.
  - **Film.** The shared policy is evaluated shard-wide in the record and row by
    row in the film, and the test documents that at about 1e-7. A 1e-7 dither on
    the *commands* moves the film 2.8%, the same as the record/film difference.
    The implicit damping split makes this gait more sensitive to it: with the
    split off, record and film agree to 0.05%.
  - **Both bars** are now max(old bar, 2x the path's own floor under the
    matching noise) (`_nudged` in `test_search.py`). A real path defect shows as
    tens of percent.
  - **The eel's floor is the largest of three dithered runs.** One draw is a
    weak estimate of a chaotic spread: six seeds gave 1.0e-4 to 7.2e-4, one
    draw read 2.6e-4, and the paths differed by 7.8e-4 once the damping and
    inflow ran on their 4-step cadence.
  - **A latent mismatch, fixed along the way.** The GPU ran the unsteady history
    even with `FluidSolver.unsteady` off. It now honours the flag.
- **AK. The ray's entry under the corrected added mass -- done 2026-09-30.**
  Its membranes oscillated through the surface after a nose-first entry, with
  the joints driven or held. Two defects, both in `physics/fluid.py`, and
  neither was the added mass itself. Probes in `experiments/ray_entry/`.
  - **The mass matrix created momentum.** Added mass is folded into the mass
    matrix and MuJoCo keeps `qvel` when it changes, which supplies
    `-m_a dv/dt` and never `-dm/dt v`: every step the entrained mass grew, the
    water came along at the body's speed with no reaction. Nose-first at
    8 m/s the hull *accelerated* from 9.2 to 21.9 m/s downward in water while
    the net fluid force on it pointed up, and the slam peak read 1292 kPa
    against 760 at 20 m/s. The ray spawned in air flapped itself 4.5 m under.
    `entrainment_reaction` adds `-max(dm, 0)/dt v` (and the rotational term)
    at each body's CoM, which conserves momentum exactly and cannot speed a
    body up. Shed mass is not reacted: the two-sided form ran the eel to
    1.1e6 m/s. From the water spawn over 6 s, peak speed with/without:
    beetle 0.43/4.70 m/s, eel 0.63/3.78, ray 0.59/0.63, gannet 0.36/0.36.
    So most of the beetle's and eel's swimming was this momentum.
  - **The implicit damping was stale at first contact.** B is refreshed every
    4 steps and scales with density, so contact could meet up to three steps
    of water loads with a B formed in air (~1/800). One strut joint took
    up to 530 N m and gained 100 rad/s in two steps. The entry peak then depended
    on which step contact fell on: 164-1732 kPa nose-first at 8 m/s as the
    start height moved 0-6 cm. `ImplicitAeroDamping.due` now also refreshes
    when a strip is wet, or will be next step, and was dry when B was formed
    (or the reverse). Held, over one refresh cycle: 279-1307 kPa before,
    372-418 after. `REFRESH_EVERY = 1` gives the same peaks within 1.5 kPa.
  - **The two printed orderings are not the ray's.** With both fixed, peak kPa
    over six start heights (flat 4 / flat 8 / nose 4 / nose 8 / nose 20):
    driven 157-198 / 257-354 / 202-286 / 174-257 / 244-293, held 130-156 /
    245-367 / 93-120 / 300-450 / 183-285. At 4 m/s flat beats nose-first in
    all six driven entries and loses all six held. Nose-first at 8 m/s loses
    to flat at 4 in 11 of 12. The ray's membranes and struts set its entry
    load, not its attitude. `test_entry_shock_is_hydrodynamic_not_a_speed_limit`
    now asserts what holds in every driven draw instead: at 8 m/s nose-first
    beats flat; flat's load grows 1.48-2.01x from 4 to 8 m/s and nose-first's
    0.83-1.02x; no entry leaves the machine sinking faster than it hit the
    water (worst 0.97x, 2.38x without the reaction); and a held entry's peak
    is within 2x across a refresh cycle. Mutations `entrainment-not-reacted`
    and `damping-stale-at-the-surface` are each caught 1/1.
  - **Side effect.** `test_resonance_needs_a_soft_drive`'s ray had been
    measured 4.5 m under water. At the surface the stiff spring reads 148
    against 171 W/rad rigid (0.87x, the > 0.8x check holds), and the compliant
    drive 75.
  - **Closed on merge, 2026-09-30: the batched path has the reaction too.**
    `BatchedFluid.finish` calls `entrainment_reaction` after the limiter, as
    `FluidSolver.apply` does. It uses the machine solver's `_prev_mbody` (so
    the first step after a reset is skipped on both paths) and the per-body CoM
    velocities the batch already holds (`vel6`). Mutation
    `batched-entrainment-unreacted` is caught by the path-agreement test (air
    `spin_rate` 0.16 apart without it).
  - **Found while checking that, and fixed: the numpy path never scattered a
    crossing's entry state.** `run_transition_batch` drew one per kind;
    `run_transition` started from the bare placement. So every crossing that
    Tier-2, a film or an offline probe measured started somewhere the scored
    one had not. The gap predates this session: at `5221fe1` peak entry speed
    differed by up to 7.3 m/s between the paths on the seed plans. Both paths
    now draw from `transition_scatter_seed(kind)`, and agree to 4e-12 to 1e-5
    with the same crossed/not-crossed outcome on every kind. The
    path-agreement test now asserts it, and mutation
    `single-path-crossing-unscattered` is caught (11.6 m/s).
  - **Open. The slam diagnostic stops resolving entry above ~20 m/s.** One
    step there is 8-12 cm of travel, and a thin strip's density blend crosses
    in a single step. Nose-first at 23.8 m/s reads 446-1539 kPa as the start
    height moves 0-8 cm. The survival check stays at 20 m/s for that reason.

---

## Why nothing flies, measured a third time — 2026-09-23

Asked by the user: why can nothing fly, is the reward reasonable, and how
rigorous is the fluid model. Three read-only audits were run and every claim
below was then re-checked with its own probe before anything was changed. The
probes are in `experiments/flight_audit/`; notes in `runs/_logs/flight_audit_0923.md`.
Every figure in a table here was re-measured. The audit-only figures, not
re-measured, are marked (audit): some correlations in §4, the joint shares and
the spar check in §5, and the tumbler, curriculum and robofly figures in §6-7.

**The short answer.** The model had five defects between the airframe and the
force, in both evaluation paths. Some made flapping look easier and some made it
harder. The air score paid a fall more than a slower descent. **With all of them
fixed, no seed plan can make the thrust level flight needs, at any of 300 random
gaits.** The earlier finding that thrust is reachable came from the bugs. What
stops flight is the flapping kinematics this genome can express, not the reward
and not the search.

### 1. Three fluid bugs, one of them in both paths — all fixed

| id | defect | direction | fix, measured |
|---|---|---|---|
| F-09 | strip lever arm from `xpos` while `mj_objectVelocity(mjOBJ_BODY)` is at `xipos` (probe: hinge body, CoM at 1 m, reported lin vel 2.0 at 2 rad/s) | flapping **too easy**: strip speed 1.8-3.6x in `U^2 S`; in `fluid.py` *and* `fluid_gpu.mojo` | arm from the CoM; 1 m from a 4 rad/s hinge now 4.000 m/s, was 6.0 |
| F-10 | reversed flow folded by a mirror, `atan2(sin, \|cos\|)` | **too easy**: a plate swept back and forth at one pitch lifted the same way both strokes (-3.770 N both) | shift by pi, camber negated; +8.050 / -8.050 N |
| F-11 | Kramer rotational force sign: `d(alpha)/dt = -omega_s`, term used `+omega_s` | **too hard**: it opposed lift during pitch-up | nose-up 12.38 N, nose-down 4.68 N; difference 7.70 N = the analytic term |

`tests/test_gpu_mirror.py` could not see F-09, because it compares the two
*sources* and both carried the bug. The kernel was rebuilt (`pixi run build-all`).

### 2. The actuator could not flap — fixed (item H's evidence, supplied)

Item H said to change the actuator model only on evidence. Here it is. Each
position servo had `kp = 2 tau g`, `kv = 0.15 tau g`, and `kv` damps the
joint's **absolute** velocity. That is a fixed 75 ms lag with a 2.1 Hz corner,
whatever the motor. Hull held, joints driven at `0.5 sin(2 pi f t)`
(`experiments/flight_audit/servo_probe.py`):

| | 2.2 Hz | 4 Hz | 7.3 Hz | 11 Hz |
|---|---|---|---|---|
| gannet, as it was | 0.69 (sat 0%) | 0.47 (0%) | **0.27 (0%)** | 0.19 (0%) |
| gannet, feed-forward | 0.97 (0%) | 0.98 (6%) | 0.98 (21%) | 0.98 (32%) |
| beetle, as it was | 0.72 | 0.48 | 0.27 | 0.17 (3%) |
| beetle, feed-forward | 1.04 (33%) | 0.98 (23%) | 0.99 (33%) | **0.65 (58%)** |

The first row is the finding. At 7.3 Hz the stroke reached 27% while the torque
**never saturated**. The motor had the authority and the control law threw it
away. A trajectory-tracking servo feeds the reference rate forward, and
`ctrl = q_ref + (kv/kp) dq_ref/dt` is exactly `kp(q_ref - q) + kv(dq_ref - dq)`.
Now the torque limit binds where it should: the beetle at 11 Hz reaches 65%
because its motor saturates 58% of the time, not because of a lag. Implemented
as `TriphibianEnv.servo_command`, used by `step` and `batchroll.step_batch`.
The rate comes from the CPG (`CPG.pop_rate`), so a command that is not a CPG's
gets no lead.

It costs the gannet's glide at some initial conditions. That glider's gait flaps
at 0.6 Hz, and it now flaps as commanded. Six scatter seeds through
`evaluate_tier1`:

- **unchanged:** seeds 1-4, competence within 0.012 (seed 3 falls either way)
- **worse:** seed 0 from 0.213 to 0.022, and seed 5 from 0.012 to 0.001
- **better:** none

Seed 0 was the only draw `test_the_seeds_include_something_that_flies` used. It
now takes medians over seeds 0-2 (gannet: airborne 1.00, sink 2.50 m/s, 0.074
against a next best of 0.000). A glider that wants to glide needs its gait held
still, which the gait gain (AA) can do and an open-loop gait cannot.

### 3. The controller jumped the stroke whenever it touched frequency — fixed

The CPG phase was `2 pi f t` on the absolute clock, and every identified basis
spans frequency (arch42 air bases: median frequency weight 0.43, audit). A 0.1 Hz
change at t = 4 s moved the stroke by 2.5 rad in one 40 ms control step. So any
policy that used the frequency mode was scrambling the stroke. A continuity term
now moves only when `f` changes, and constant-frequency commands are
bit-identical to the closed form.

### 4. The air score paid a fall more than staying up — gated

Air-segment exits in arch42, by time spent airborne:

| | n | median air competence |
|---|---|---|
| in the sea in < 2.6 s | 393 | 0.030 |
| 2.6-2.8 s | 676 | 0.033 |
| longer (the scored branch) | 1,786 | **0.000** |

The short branch paid `credit * frac * 0.10`. Every body in it reached the sea
from the 30 m launch in under 2.8 s, and free fall takes 2.47 s. A body that
stayed up longer but sank faster than `SINK_BALLISTIC` = 6 m/s scored 0. Air
competence correlated **-0.44** with time aloft and **-0.20** with
`lift_margin` (audit). The famine regime starved air in 113 of 178
generations and selected on exactly this. The short branch now scores 0, which
is the flight it showed. It is gated, not re-weighted.

### 5. With all of it fixed, thrust for level flight is out of reach

Best `thrust_margin` over 300 random gaits per seed plan: amplitude, phase,
offset, and frequency 1.5-12 Hz, quasi-static at trim
(`experiments/flight_audit/gait_search.py`, same rng on both trees):

| plan | old tree: best (p90) | fixed tree: best (p90) | share > 0, fixed |
|---|---|---|---|
| gannet | +1.289 (+0.569) | **+0.651** (+0.287) at 11.8 Hz | 0.98 |
| teal | +0.819 (+0.361) | **+0.287** (+0.108) at 10.8 Hz | 0.56 |
| beetle | +0.369 (+0.243) | **+0.208** (+0.138) at 11.4 Hz | 1.00 |
| bat | +0.097 (+0.049) | **+0.057** (+0.016) at 3.0 Hz | 0.24 |

Three things follow:

- **"Thrust is reachable in the model" was the bugs.** The old tree put the
  gannet over 1.0. The fixed one halves every figure, and nothing reaches the
  1.0 that level flight at trim needs.
- **Every best gait sits at 10.8-11.8 Hz.** `mut_gait` redraws frequency from
  1.5-8 Hz (`genome.py`), so the search cannot even visit them. The spar check
  fails the gannet at 10.95 Hz (1.99x allowable, audit).
- **Open loop, nothing flies on either tree**
  (`experiments/flight_audit/fly_probe.py`). The gannet glides at 1.7 m/s sink
  and scores 0.072; every other plan falls at 6-14 m/s.

The binding constraint is the flapping stroke itself. Every joint is one hinge
(`mjcf.py`). The `"universal"` joint kind in `genome.py` is built as a plain
hinge. 54% of arch42's powered wing joints only pitch, 33% only flap (audit), and no
wing can flap and feather with a controlled phase between the two, which is how
every animal flapper makes thrust. A heave-only wing gets its thrust from lift
tilt alone, and the table above is what that buys.

### 6. The reward, reflected on — what it still gets wrong

§4 was the defect that inverted the gradient. The rest are real and not fixed
here, because each is a reshaping that needs its own validation:

- **Thrust is in no term of fitness.** The ladder (`judge.py`) is metadata and
  scout features only (`loop.py`), so `flaps_forward`, `makes_thrust` and
  `pushes_itself` apply no selection pressure. And they sit above
  `holds_height`, which 0.1% of evaluations reach.
- **No gradient between 6 and 14 m/s of sink.** Median measured sink is 9.9 and
  p90 14.1, and all of it scores the same 0.
- **The height term reads endpoints.** Designs spinning at 4.4-5.9 rad/s (just
  under `MAX_SPIN_RATE`), with a wobble ratio of 27-72, scored 0.16-0.25: the
  best air scores after gen 120 in arch42. This is `sink_rate` against
  `station_keeping` again (arch37).
- **Curriculum stage 1 reads air `cruise_progress`**, which is not gated on
  being airborne. Bodies in the sea by 3.5 s read 0.23 at the median.
- **The PPO reward for air is terminal.** The shaping potential
  `-tanh(d/5)` saturates above ~10 m, which is the whole air segment.
- `thrust_margin` is cached per phenotype at the first gait it sees, so a gait
  the policy modulates is never re-measured. It also reads *commanded*
  kinematics. After §2 those match what is reached, except under saturation.

### 7. The fluid model, reflected on — what it is and is not

Strip theory with a quasi-steady CL/CD, an LEV term, Kramer rotation and scalar
wing added mass. It is carefully labelled and **has never been checked against
flapping data**. `tests/test_physics.py` checks signs and orders of magnitude.
Open, and recorded in `docs/MATH_AUDIT.md`:

- F-12: the aspect ratio is taken per part, which roughly doubles induced drag,
  and there is no leading-edge suction.
- F-13: no inflow, Wagner or Theodorsen model. This is fine in cruise
  (`k ~ 0.15`) and outside the model's range in hover and swimming (`k ~ 1`,
  induced velocity about equal to tip speed).
- F-14: the weight cancellation for added mass is applied at the strips.
- F-02: the LEV is keyed on pitch rate, so a heave-only wing never gets it
  (robofly at 45 deg: CL 1.8, model 1.1).
- F-08: there is no post-stall lift loss.

**None of them is the reason nothing flies.** §5 is a kinematic ceiling, and F-12
and F-02 each lower thrust further for the wings this genome builds. The
validation that would settle the model's standing is Dickinson's robofly
(Dickinson, Lehmann & Sane 1999) and Sane & Dickinson (2001) as a fixed-rig
fixture: prescribed stroke and pitch, force time histories, compared.

### Comparability

**Nothing air- or water-force related, and nothing actuated, is comparable
across 2026-09-23.** Flapping strip speeds, reversed-flow forces, rotational
lift, the reach of every stroke above ~2 Hz, and the air score's short branch
all changed. arch42 (stopped at gen 177) stands as a record under the old model.
Two test fixtures moved with it, each with its measurement in the test:
`test_the_seeds_include_something_that_flies` takes medians over three scatter
seeds (§2). The audit fixture audits a gannet, because the drawn beetle's
mission of 0.0009 is 0.0 under the corrected fluid model.

### The work list this sets, in order

- **AB. A flapping wing that can feather.** Build the `"universal"` joint the
  genome already names, or a stroke hinge with a pitch hinge in series, and give
  the CPG a pitch-heave phase. **Probe first.** Put a pitch joint on the gannet
  wing and re-run `gait_search.py` with the phase lead free. If best
  `thrust_margin` does not clear 1.0, the ceiling is elsewhere and nothing
  should be built.
- **AC. `mut_gait`'s frequency range** runs 1.5-8 Hz, and every best gait above
  is 10.8-11.8 Hz. Widen it only together with the spar check that forbids
  those frequencies, so the search is not handed designs that break.
- **AD. Thrust into selection,** as a gate on the air task and not as another
  rung, once AB says there is thrust to select for. Doing it first would repeat
  arch38 (§3 there): a term with nothing reachable above its floor moves the
  distribution down.
- **AE. The endpoint height term, and stage-1 coast** (§6), each validated on
  the population held still before it is used.
- **AF. A robofly fixture** for the fluid model (§7). Until it exists, "the
  model says" means "the model says".

---

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

---

### X. The reward has no task — raised by the user 2026-09-21; **reshaped twice, S3 running in arch42**

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

**arch42 applied them and was stopped flat at generation 102**
(`runs/arch42_stopped_gen102_flat`; the user: "停掉，改任務計分的形狀"). Task
score over the archive's 163 elites: water median 0.000 (29% non-zero), land
0.001 (52%), air 0.000. Tracking was the wall: `1 - |v - v_cmd| / |v_cmd|` is
zero for standing still, for drifting across the heading *and* for going twice
as fast, and it was multiplied by vertical terms most machines failed.

#### S3 — the shape arch42 relaunched on (2026-09-22)

Relaunch bar, set before the work: on arch42's own top 40 elites, driven as
scored, the task-score median must be above zero, and the same bodies with
every actuator held still must score ~0 on what has to be chosen. Progress may
be passive — the user's rule allows a glide, and forward motion in water
through the mechanism — so it is reported, not barred.

| medium | score |
|---|---|
| water | `0.5 progress + 0.5 hold`; hold = at the depth × not moving × **not sinking** |
| land | `progress * (0.5 + 0.5 stop)` — a rock stops perfectly, so stopping qualifies the walk |
| air | `0.5 turn + 0.5 height`; turn = the velocity change across the launch heading toward the new one, over `speed * sin(turn)` |

Progress is `clip(v·h / v_cmd, 0, 1) * exp(-(v_across / v_cmd)^2)`: forward is
forward, and overshooting the command is not a penalty.

**Four leaks found only on the real path** — the fabricated traces and the two
seed plans held still passed every one of them. Each now has a check and a
mutation that the check catches:

| leak | who, held still | paid | fix | mutation |
|---|---|---|---|---|
| air height while floating | seed flappers, fallen into the sea | 0.141 (gannet 0.187) | height counts only time in the air | `air-height-counts-floating` |
| hold = passing through the target | an eel at 0.094 m/s, a gannet at an oscillation's turning point | 0.63, 0.51 | net vertical rate against `HOLD_SINK_SPEED` 0.05 m/s | `hold-ignores-sinking` |
| turn = stopping after a slide | two gannets, slid away from the commanded side on the water | 0.28, 0.34 | the second phase's own velocity must point to the new side | `turn-counts-stopping` |
| turn on the water | the same | — | the turn is gated on being airborne | `turn-ignores-airborne` |

`HOLD_SINK_SPEED` is set *below* the passive distribution on purpose: the held
holders crossed their target at 0.048–0.233 m/s, and the 7 *driven* holders at
0.026–0.172. **No controller in arch42 held depth better than a body doing
nothing**, so hold was paying coincidence. Two checks were added for mutants
the new gates made redundant: sideways drift at depth (`hold-ignores-motion`)
and reversing along the launch line (`turn-counts-slowing-down`).

Air height is measured from the start of each phase, for as long as the
machine stays up, weighted by the share of the phase it does — not over the
late half, which is time to turn or descend. 86% of air segments are in the sea
by 2.65 s, and a late-half window read zero for every driven design.

Validation, `runs/_logs/validate_shape_real_path.py`, 288 segments through
`rollout()` (driven / held still):

| | median | mean | on what must be chosen |
|---|---|---|---|
| water | **0.133** / 0.018 | 0.161 / 0.110 | hold 0.008 / **0.001** |
| land | **0.058** / 0.000 | 0.166 / 0.009 | beyond progress 0 / **0** |
| air | **0.000** / 0.000 | 0.015 / 0.009 | turn 0.010 / **0.000** |

Water and land pass. **Air does not, and not because of the shape.** With the
leaks closed, 84% of driven air segments fall at a ballistic average over their
whole flight. Grading below that would pay drag and tumbling — the defect the
air ladder's first two rungs had ("Why nothing has ever flown" §2). Air's honest
gradient is a glide, which 16% of driven segments have (mean 0.015). The rest
has to come from the design side: the lift rungs, not the controller. arch42's
gen-50 bar was always water and land for this reason.

And hold is now a capability nobody has: 2% of driven water segments score on
it, one real holder at 0.333 against a held-still maximum of 0.037.

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

### Z, part two. The audit was not the scored experiment either — **fixed 2026-09-22**

Found when S3 made a one-generation test fixture lose its only elite. The
auditor's `reevaluate` ran the design **with no controller, at seed 0**, and
divided that by the `mission_fraction` the design had earned with its own policy
plus the shared one, on the batched path, at its own seed. Since the task is
drawn from the seed, the audit also changed the task. So the "perturbation"
ratio compared two different experiments, and an audit that removes a design
from the archive, labels it 0 for the critic and can veto the judge was acting
on that comparison.

- arch40 invalidated **25 of its 42 audits**, with held-out ratios of 2.1 and
  4.5 beside collapses to 0–3%: noise in the base, read as model dependence.
- arch42's first launch invalidated none, only because every audited base was
  exactly zero and the check skips a zero base. S3's graded scores would have
  switched it on.
- Measured on the fixture's elite, a **×1.0** perturbation read **2.2152**
  against the recorded score. After the fix it reads 1.0000.

Now the audit rebuilds the elite's control law (its own policy plus the shared
one), re-measures the base under that law at the elite's eval seed, and
perturbs that same experiment. The held-out seeds still change the seed, and
with it the task, which is what a held-out seed is for. Held by
`test_an_audit_perturbs_the_scored_experiment_and_nothing_else` and two
mutations (`audit-perturbs-another-seed`, `audit-base-is-the-record`).

Not done: a floor on the base. A design at 0.0004 of the mission has nothing to
collapse, and whether a same-experiment collapse at that level still means "the
design depends on the model" is a question for arch42's audit events.

### AA. The policy could not stop a machine — **gait gain, built 2026-09-22, in arch42**

arch42 on S3 (`runs/arch42_s3_stopped_gen208_no_stop_authority`) climbed on
progress and never learned a command. At gen 200 the archive's task medians
were water 0.092 and land 0.033, and rising. Hold was 0% above 0.1. Land machines that
walked did not stop when told: 2.0% of evaluations did both, against 5.5% if
the two were independent (corr −0.20), unchanged since gens 6–50. Stopped at
gen 208 by the user ("現在停，立刻做步態增益").

**The cause was the action space, not the reward.** The policy commands
`base + modes^T c` with `c` bounded to [−1, 1] by tanh, and the modes come from
small probes around the design's own gait. Per actuator the modes could reach
zero amplitude, but a stop needs *one* `c` that zeroes every actuator at once.
Measured on the top 8 elites (`runs/_logs/probe_stop_authority.py`), the best
bounded `c` left a median **74%** of the base amplitude on land and **50%** in
water, and moved phase and offset by up to 1.5 rad doing it. For most designs
stopping and hovering were not in reach, so no task shape could teach them.

**The headroom, before building** (`runs/_logs/probe_gain_headroom.py`, 24
elites, same law, seed and task): an oracle that sets the amplitude to zero on
stop and hold phases lifts the land task score from **0.362 to 0.501** and stop
from 0.356 to 0.902, at a cost in progress of 0.567 → 0.535. Water does not
move (0.231 → 0.225, hold 0.000 → 0.000): zeroing the gait lets a body sink or
float, and holding depth needs depth feedback. So the gain is necessary for
hold and not sufficient.

**Built** (`--gait-gain`, `SearchConfig.gait_gain`, off by default):

- The per-candidate policy gets one more output and the shared policy one more
  action. The two intents are summed, like the mode coefficients, into
  `g = clip(1 + intent, 0, 2)`, which multiplies the commanded amplitude
  (`control.cpg.GAIN_RANGE`, `gait_gain`, `split_command`).
- An intent of zero leaves the gait bit-identical. A policy without the
  channel, which covers every stored run, drives at gain 1, so old films still
  reproduce.
- The single-machine paths go through one method, `MobilityBasis.command_policy`.
  The batched pool sums the two halves itself, and a test holds the two paths
  to the same distances.
- Published per segment: `gain_cruise`, `gain_hold`, `gain_stop`, the mean
  commanded gain over each kind of phase. Absent without the channel.

Held by `test_the_gait_gain_can_stop_a_machine`, where the walker's stop goes
from 0.000 to 0.994 when its gain output saturates low, and by
`test_the_gait_gain_drives_the_same_on_every_path`. Five mutations are caught.

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

**Built 2026-10-03, off by default** (`learning/grpo.py`;
`SearchConfig.shared_learner = "ppo"`). Right algorithm, still not the binding
constraint, and it waits on thrust being in the score in a form the search can
climb; whether to switch it on is item R's question (does the shared policy
carry weight at all).

What was built, the partial form: each generation `grpo_bodies` (default 4)
bodies drawn from those that passed Tier-0, `grpo_group` (default 4) learning-only
rollouts each, with different exploration streams, on the generation's own
evaluation seed (so the same scatter and task) and with the controllers the
bodies were archived with (no identification repeated). One batched call
through the actor pool. The results are dropped: nothing is scored, placed or
counted. Advantage = `(R - mean_g)/(std_g + 0.05)` per (body, segment kind),
broadcast over the trajectory's steps; replaces GAE for these rows. The rows
share minibatches with the ordinary PPO rows in one `ppo_update` (a second
update would see an observation normaliser the first had moved). No value loss
on group rows, entropy bonus kept, no potential shaping (it cancels in a group).
Unusable groups are dropped and counted in the `ppo` event. Modes:
`ppo+grpo` adds the rows, `grpo` trains on them alone (the ordinary rollouts
are still collected under both, because they are the generation's scores).

Cost, measured 2026-10-03 on a loaded machine, 1 worker, `batch=8`,
`segment_seconds=2.0`, 4 bodies x 4 = 16 rollouts, gens 0/1/2: GRPO stage 48.6 /
55.2 / 57.8 s against main 135.0 / 100.5 / 108.6 s and re-score 34.2 / 28.6 /
30.0 s, i.e. 0.29 / 0.43 / 0.42 of (main + re-score). The 16 GRPO machines are
fixed while main and re-score grow with `batch`, so at `batch=16` the ratio
should be about half of that (inference, not measured); the brief's estimate was
+15-25% per generation. Recorded per generation in the `stages` event (`grpo`:
wall, bodies, rollouts, per-shard walls) and in the `ppo` event (`grpo_*`:
groups, rows, mean within-group reward std per segment kind, zero-variance
groups, dropped counts). In that tiny run the within-group std was 0.00 for air
and every crossing and 0.02-0.07 for land and water: a group-relative advantage
is zero where every attempt earns the same, so **the signal exists only where
bodies already score**, which is the thing item R's reading should look at
before switching it on.

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
