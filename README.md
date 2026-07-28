# Dytiscidae

A generative design and control-learning loop for **triphibian flapping-wing
machines** — vehicles that fly, swim and walk, and can transition between all
three without stopping.

Named after the diving beetles, which are the only animals that solve this
problem well: sealed elytra over a flight wing, hair-fringed swimming legs, a
carried air bubble for buoyancy control, and the ability to switch domains in
seconds.

The system does not optimise toward a specification you hand it. It searches for
what specification is *achievable*, and returns a map.

---

## What this actually does

```
genome  ──►  phenotype  ──►  MJCF + panels  ──►  three-tier evaluation
  ▲          (mass, structure,   (MuJoCo rigid body        │
  │           feasibility)        + own fluid solver)      │
  │                                                        ▼
  └──────────  curator  ◄──────  MAP-Elites archive  ◄── fitness + descriptor
              (selects, credits,   (a map of what is
               verifies, culls)     achievable)
```

* **Morphology** comes from a recursive module graph (Sims-style) whose surfaces
  are shaped by evolved CPPNs — so wings, paddles and limbs are generated, not
  chosen from a menu.
* **Physics** is MuJoCo for rigid bodies and contacts, plus a quasi-steady
  blade-element fluid solver written here, because MuJoCo's own fluid model has
  no buoyancy and cannot represent flapping.
* **Control axes are discovered, not assumed** — see below.
* **Search** is MAP-Elites with CMA-ES emitters, run by a curator that manages
  the population actively instead of just keeping the winners.
* **Everything is observable**: JSONL telemetry, a self-contained HTML
  dashboard, and offscreen video of elites.

Nothing requires a GPU. Developed and tested on 4 CPU cores.

---

## Quick start

```bash
pip install -r requirements.txt

python -m dytiscidae.ops.run verify              # 30 physics checks
python -m dytiscidae.ops.run reference           # inspect the hand design
python -m dytiscidae.ops.run search --generations 200 --run runs/first
python -m dytiscidae.ops.run dashboard --run runs/first
python -m dytiscidae.ops.run skills              # train the actuator skills
```

`search` prints one line per generation and refreshes
`runs/first/dashboard.html` every five generations, so you can watch it live by
reloading that file. It checkpoints continuously; killing it loses at most one
generation.

The main cost dial is `--segment-seconds` (default 8). Halving it roughly halves
the run time and roughly doubles the variance of every Tier-1 score.

---

## Control axes are measured, not assumed

You asked that the operating theory not be constrained to pitch/yaw/roll. It
isn't — those axes are a convention from aircraft that have a fuselage, a nose,
and control surfaces arranged to give near-decoupled moments. A generated
triphibian has none of that guaranteed.

So each body's control axes are **identified empirically**: drive the pattern
generator with random parameter perturbations, measure the resulting body twist,
fit the Jacobian, take its SVD. The leading singular vectors are the things this
body can actually do, ranked by how well it can do them.

Real output from the reference design:

```
air:    mode0[sway+&roll-]   sigma=0.537     rank 3 of 19 parameters
        mode1[roll-&heave-]  sigma=0.269
        mode2[roll+&heave-]  sigma=0.126

water:  mode0[heave+&pitch+] sigma=0.193     rank 3 of 19 parameters
        mode1[surge-&heave-] sigma=0.049
```

The strongest thing this machine can do in air is a coupled sway-and-roll. There
is no name for it. In water it is a completely different set — because the same
surface that is a lifting wing in air is a paddle in water, so the machine's
controllable directions genuinely change when it crosses the surface. The
controller commands coefficients in whichever basis applies.

---

## The curator: selection, not just survival

You asked for more than survival of the fittest. `evolution/curator.py` makes
six decisions each generation, all from evidence the run produces:

| | |
|---|---|
| **Operator credit** | Every mutation operator is a bandit arm, rewarded by archive outcome, on a sliding window. `add_part` is worth a lot early and nothing later; `cppn_weights` is the reverse. A lifetime average would keep sampling the early winners forever. |
| **Parent selection** | Weighted by fitness, *curiosity* (how often this elite's offspring land anywhere), *frontier position* (how empty the neighbourhood is), and recency. Breeding from the crowded middle of a mapped region is the biggest waste in naive MAP-Elites. |
| **Fidelity promotion** | Tier-2 costs ~10x Tier-1, so it is a budget spent on elites that are near the top, brand new, long-unverified, or *improbably good* — the last being most informative, since an outlier is either a discovery or an exploit. |
| **Exploit quarantine** | Candidates that beat the simulator rather than the task are removed *and their archive cell is tainted*, so the search cannot rediscover the same trick from the same place. |
| **Regime control** | The run is classified each generation — bootstrapping, exploring, refining, stagnant — and structural mutation pressure, emitter mix and feasibility bias are set from that. Search settings are an output, not a config. |
| **Crowding control** | Dense regions are thinned, but only elites that are both crowded *and* weak relative to their own neighbourhood, so a region never loses its best representative. |

---

## Two environments

**`envs/triphibian.py`** — the mission. Random start domain, then cycle
air → water → land, five minutes each, three times, forty-five minutes total,
reaching and holding ten metres while submerged. The world is one continuous
scene (seabed, beach ramp through the waterline, open air), so the shoreline is
a *place*, not a mode.

Evaluated at three fidelities because simulating 45 minutes at 250 Hz is 675,000
steps per candidate:

| Tier | Cost | What it does |
|---|---|---|
| 0 | ~0.2 ms | Closed-form power and structural feasibility. Rejects ~90% of random genomes. |
| 1 | ~2–8 s | Short episodes per domain plus transitions. Measures controllability and steady power; extrapolates the 45-minute budget. |
| 2 | ~10–90 s | The real schedule with waves, current and wind. Run only on promoted elites — and it *checks the Tier-1 extrapolation*, flagging elites whose short window was not steady state. |

**`envs/skills.py`** — the actuator bench. Component-level control, learned to
convergence in seconds, transferable to any morphology carrying the component:

- `resonance_seek` — **the highest-value skill in the project.** A compliant
  flapping wing driven at resonance costs a fraction of the power of the same
  wing driven off it, because the spring returns the wing's kinetic energy at
  each reversal instead of the motor paying for it twice per cycle. The
  resonance depends on added fluid mass, so *it moves by about 5x when the wing
  enters water*. A controller that memorises one frequency fails.
- `depth_hold` — carried gas compresses with depth, so buoyancy falls as the
  machine sinks, so it sinks faster. Open-loop unstable; no passive fix.
- `thermal_thrust` — copper loss goes as torque², useful work as torque. The
  greedy policy always saturates and always overheats.
- `solenoid_latch` — pulse and latch rather than hold; holding current is pure loss.
- `vibrator_track` — piezo resonance for silt shedding, which also shifts when wet.
- `ballast_sequence` — vent, flood, arrest, without overshooting past depth.

---

## On the 15 kg / 10 m / 45 min target

You asked the system to find its own best specification, so these are reference
points rather than constraints. Two things are worth stating up front.

**The energy budget was never the problem.** At 15 kg, cruising at L/D ≈ 6 needs
about 500 W electrical; submerged cruise is about 20 W; the whole 45-minute
mission is roughly 150 Wh, well under a kilogram of cells.

**Flapping flight at 15 kg is at the biological limit.** The largest flying birds
— mute swan, kori bustard — are 12–15 kg with 2.4 m spans, and they need a
takeoff run. The binding constraints found by the structural pass are, in order:

1. **Inertial reversal**, not aerodynamic load. Root bending from the wing's own
   mass at stroke reversal scales as *f²*, which is what caps flap frequency at
   ~2 Hz for this span and is why large flapping animals are all slow.
2. **Hull buckling**, not hoop stress. At 10 m the gauge pressure is only 1 bar,
   but a thin shell under external pressure collapses long before it yields, and
   `p_cr ~ E(t/r)³` punishes large dry volumes hard.
3. **Buoyancy closure.** Sealed volume is buoyancy you then have to carry, pump
   out, and support against pressure. This is what pushes the search toward small
   pressure hulls with everything else free-flooding.

The hand-designed reference in `core/reference.py` closes at **5.8 kg**, not 15 —
carbon spars, 2.2 Hz, a small hull with a flooded fairing, and Li-Po chosen for
power rather than Li-ion for energy. It exists to prove the feasible set is
non-empty and to seed the archive; the search is free to disagree with it, and
the archive's mass axis spans 0.4–40 kg so it can.

---

## Physics: what is modelled and why

MuJoCo supplies rigid bodies, joints and contacts. **All fluid behaviour is
computed in `physics/fluid.py`**, because MuJoCo's own model has no buoyancy —
verified directly: a 500 kg/m³ body sinks in 1000 kg/m³ fluid — and cannot
represent the three effects that make flapping work:

- a **leading-edge vortex** that holds lift attached to ~40° of incidence, gated
  on reduced frequency so a gliding wing gets conventional static stall and a
  flapping one does not;
- **rotational (Kramer) circulation**, which is what generates useful force
  during stroke reversal when translational velocity is near zero;
- **added mass**, negligible in air and dominant in water — a single wing strip
  of the reference design carries ~32 kg of added mass submerged, against a
  5.8 kg vehicle.

The free surface is never a mode switch. Every element has a continuous
submerged fraction in [0,1], and density, viscosity, buoyancy and added mass all
blend through it — which keeps water entry and exit integrable and stops the
optimiser finding discontinuity exploits.

### Corrections made while verifying against hand calculations

These are recorded because each one produced plausible-looking output while
being wrong, which is the failure mode that matters in a generative pipeline.

| Bug | Symptom | Reality |
|---|---|---|
| Added mass applied as an explicit external force | NaN accelerations after 1.3 s in water | Classic added-mass instability: the feedback gain is m_added/m_body ≈ 5. Now folded into the mass matrix, which MuJoCo inverts implicitly — with gravity compensation, since added mass has inertia but no weight. |
| Added-mass backward difference not primed | 53 N spurious impulse on step 1 of every episode, against 7.7 N of real lift | No previous sample exists on the first call. |
| Wall thickness = 30% of radius | Every design ~3x too heavy (39 mm wall on a 130 mm hull) | Sized from buckling instead: t/r ≈ 0.038. |
| Buoyancy double-counted | Everything could sink | Flooding a tank removed displacement *and* added mass. It is one effect, not two. |
| Buoyant volume = outer envelope | Every design came out a balloon (ρ_rel ≈ 0.2) | A free-flooding fairing displaces water but is full of water. Buoyancy is material volume + sealed gas. |
| Water entry checked as plate bending on a curved hull | No design survived entry above ~1 m/s | Curved shells carry slam in hoop membrane, not bending — ~4x higher survivable speed. Flat panels *do* use bending, which correctly penalises flat bottoms. |
| Two buoyancy checks were binary | First generation scored zero with no gradient | Every check now returns a graded margin. |

Verify with `python -m dytiscidae.ops.run verify` (30 checks).

---

## Layout

```
dytiscidae/
  physics/     medium (free surface, waves), fluid (blade element),
               energy (motors, battery), structure (spars, hulls, slam)
  core/        cppn, genome, phenotype, mjcf, reference
  control/     cpg (pattern generator + mobility basis identification)
  envs/        triphibian (mission, 3 tiers), skills (actuator bench), evaluate
  evolution/   archive (MAP-Elites), cmaes, curator, loop
  viz/         dashboard (self-contained HTML), render (offscreen video)
  ops/         telemetry (JSONL), run (CLI)
tests/         test_physics.py — 30 checks pinning conventions and magnitudes
```

---

## Known limits

Worth knowing before trusting a result:

- **Quasi-steady aerodynamics.** No wake, no vortex shedding history, no
  wing–wing interaction. Good to maybe ±30% for a flapping wing, which is fine
  for ranking designs and not fine for predicting absolute performance.
- **Tier-1 extrapolation.** The 45-minute budget comes from a short window. Tier-2
  checks it, but only for promoted elites.
- **Isotropic added mass.** The full 6×6 added-mass tensor is directional; this
  uses a scalar per body, which overestimates in-plane inertia.
- **No structural dynamics.** Spars are checked against static and inertial
  loads but are rigid in the simulation, so flutter and aeroelastic divergence
  are invisible.
- **Land is the weakest domain.** Contact-rich locomotion needs finer timesteps
  than the fluid domains, and the mobility basis is identified from fluid probes,
  so walking gaits are less well served than swimming and flying.
