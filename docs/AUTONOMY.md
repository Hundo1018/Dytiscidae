# Removing the hand in the loop

Four corrections to the direction of this project, and what each one changes.

---

## 1. Observe, intervene rarely

The curator's famine rescue currently fires on a fixed rule (25 generations
below 0.35). That is still a hand-set policy, just one that runs automatically.

**What it should be:** intervention as a *budgeted, escalating* response, with
the run's own history deciding when it is warranted. The curator already tracks
everything needed — coverage growth, QD growth, per-domain bests, operator
payoffs. What it lacks is a notion of *cost of intervening*: every override
narrows the search, so it should be rare and it should be justified by evidence
that waiting is worse.

Concretely: intervention triggers when the posterior probability that a domain
is unreachable-under-the-current-space exceeds a threshold, estimated from the
plateau length and the variance of recent attempts, not from a fixed generation
count. Early in a run the prior is weak and intervention is cheap, which
reproduces the "more frequent early" behaviour without special-casing it.

## 2. No hand-written taxonomy; grow from the goal

This is the important one, and the current code fails it.

`PART_KINDS = [HULL, STRUT, WING, PADDLE, FOOT, BALLAST, MEMBRANE, BELL, FIN]`
is a taxonomy I wrote. Adding MEMBRANE, BELL and FIN to it looked like widening
the space but was only a longer list of categories I had chosen, with
hand-tuned parameters attached to each. A design that is 60% wing and 40% bell,
or something with no name at all, is still unreachable.

**What it should be:** parts have *continuous morphological traits*, and what
kind of thing a part is becomes an emergent region of that space rather than a
label. This is the representation used by voxel-based soft-robot evolution
(Hiller & Lipson 2012; Cheney et al. 2013, *Unshackling Evolution*), where
material properties vary continuously and morphology is a field rather than an
assembly of typed components.

The traits that matter here, each read directly by the physics:

| trait | 0.0 | 1.0 | what reads it |
|---|---|---|---|
| `flatness` | rod / tube | thin plate | lift slope, added mass, form drag |
| `enclosure` | open frame | sealed cavity | buoyancy, pressure loading |
| `compliance` | rigid spar | tension membrane | which structural check applies, load shedding |
| `cavity_actuation` | fixed volume | full jet stroke | momentum-flux thrust |
| `contact_affinity` | slick | high-friction foot | ground contact, walking |

A wing is then `flatness≈1, compliance≈0.2, enclosure≈0.3`. A bell is
`flatness≈0.2, enclosure≈0.9, cavity_actuation≈0.5`. Neither is declared; both
are points. And the space between them — which no taxonomy contains — is
reachable by interpolation, which is exactly where a triphibian compromise
would live.

**Prototypes become optional.** With a continuous space, the five body plans
stop being the only entry points and become *initial conditions*, one option
among: seed from prototypes, seed from noise, or seed by inverse design —
sampling traits that the Tier-0 analytic model says could meet the mission, and
growing structure from there. Inverse design is the third mode and the one that
most directly matches "evolve the structure backwards from the goal".

## 3. Learn the objective and the descriptors

Both the fitness weighting and the archive axes are currently mine:

```python
BD_AXES = [log_mass, density_ratio, air_competence, water_competence]
fitness = base + 0.10*margin + 0.10*energy + 0.10*land
```

Every constant there is a guess, and the air-competence metric being gameable by
floating (found and fixed by inspection, not by the system) shows what that
costs.

**Two changes, both standard and both CPU-affordable:**

*Descriptors:* replace the hand-picked axes with unsupervised ones learned from
recorded trajectories — AURORA (Cully 2019, *Autonomous skill discovery with
quality-diversity and unsupervised descriptors*). Record a fixed feature vector
per episode, fit a low-dimensional projection (PCA first; an autoencoder if it
earns its cost), and use the latent as the archive descriptor, refitting
periodically as the archive's behaviour distribution shifts. The system then
decides what "a different kind of machine" means.

*Objective:* delete the scalarisation rather than learn its weights. The mission
supplies measurable quantities — time in the commanded domain, depth reached,
energy remaining, structure intact, transitions completed — and a
Pareto-dominance test over those needs no weights at all. What remains to tune
is nothing. This is strictly better than learning the weights, because learned
weights are still a scalarisation and still hide trades.

This is also the change that most reduces the need for me to be in the loop,
which is the point: fewer hand-set numbers means fewer things only I can fix.

## 4. Fill in the missing physics

Gaps that currently *constrain the design direction* rather than merely
approximating it — a design cannot be found if the physics that would make it
work is absent:

- **Wake–body interaction.** The vortex wake is diagnostic only. Formation
  flight, wing–wing interference, and ground effect are all invisible, so any
  design whose advantage comes from them cannot be discovered.
- **Structural dynamics.** Spars are rigid in simulation. Resonant flapping is
  modelled in the skill bench but not in the vehicle, so the single largest
  efficiency mechanism available to a flapping machine does not exist where the
  designs are actually scored.
- **Free-surface dynamics.** The surface is a blended density field with a
  prescribed wave. It does not deform, so planing, wave-piercing, breaching and
  surface tension are all absent — and those are precisely the mechanisms a
  water-to-air transition would exploit.
- **Anisotropic added mass.** Currently one scalar per body; the real 6×6 tensor
  makes edgewise and broadside motion differ by an order of magnitude, which is
  the whole basis of rowing.
- **Cavitation and ventilation** at speed near the surface.
- **Actuator dynamics** beyond a torque limit: bandwidth, backlash, and series
  elasticity — the last being what makes resonant drive possible at all.

Ordered by how much each one currently *forbids* rather than merely
mis-estimates: anisotropic added mass, series elasticity, free-surface
deformation, then wake coupling.

---

## Order of work

1. Continuous traits replacing the kind taxonomy (unblocks 2 and most of 4).
2. Anisotropic added mass and series elasticity (physics that forbids designs).
3. Learned descriptors, then dominance-based selection (removes my constants).
4. Evidence-based intervention policy (makes 1 principled rather than fixed).
