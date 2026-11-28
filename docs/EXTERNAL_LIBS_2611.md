# External libraries: bajo and noeira, first pass (2026-11-20)

Question: does either repository replace or improve this project's physics
or RL? Method: GitHub pages only (README and directory listings read
2026-11-20); no code was cloned, built or run. Everything below is what the
pages say, not something measured here. Our side is from `CLAUDE.md`,
`docs/model_validity.md`, `docs/ARCHITECTURE.md`.

## bajo (github.com/jgsimard/bajo)

Pure-Mojo geometry and rendering engine: math, OBJ/MTL/PBRT loading, CPU and
GPU BVH, GPU sort, path tracing, a raylib viewer. Its README states that
particle simulation, fluids, rigid bodies and batched environments are
planned, not built. No license is stated on the page.

**Verdict: nothing to use.** It has no physics and no RL. Nothing in it
covers a need we have (we have no rendering bottleneck; films use MuJoCo's
renderer). Drop.

## noeira (github.com/noeira/noeira)

MIT, pure Mojo, beta, about 5,250 commits. Relevant parts:

| | noeira | this project |
|---|---|---|
| rigid-body engine | `noeira/physics3d`: MJCF loader, PGS / Newton / CG, elliptic cones, multi-CCD, tendons; "checked against MuJoCo 3.12" (tolerances not seen) | MuJoCo 3.15 reference; mujoco-warp on GPU (`envs/devroll`) |
| fluid / buoyancy | not mentioned | own quasi-steady solver (`physics/fluid.py`), Mojo kernel `mojo/src`, robofly-checked |
| batched GPU stepping | most functions have GPU versions; multi-world stepping not described; models defined at compile time | heterogeneous morphologies batched through the fluid panels; one MuJoCo model per machine |
| external force hook | not described | the fluid solver applies forces every step |
| RL | `agents/ppo.mojo`: PPO with clipped surrogate and GAE; no mention of value clipping, observation normalisation or shared policy over parallel envs | shared PPO across every evaluation (`tests/test_ppo.py`, 54 checks plus mutation audit) |
| quality-diversity | none found | MAP-Elites, eight islands |
| language / interface | Mojo only; no Python interface mentioned | Python with a Mojo kernel |

## Reading

- Physics is the only overlap that could matter, and it overlaps with the
  part we already have a GPU answer for (mujoco-warp). Our open physics
  problems are in the fluid side and the actuator and contact models, which
  noeira does not have. Swapping the rigid-body engine would reset every
  comparability boundary at once (compare the 3.11 to 3.15 upgrade, which
  moved water and land by more than 10 of 229 elites) and our contact-
  sensitive results (beach rows, 35-70 weight rigid-contact figures) depend
  on exact MuJoCo contact behaviour. Parity was checked against 3.12, not
  3.15.
- Per-machine models are compile-time in noeira as described; our search
  builds a different body per candidate. That is a structural mismatch, not a
  tuning gap.
- PPO and the QD loop are the cheapest things to own. Their PPO adds nothing
  we lack on the page; the shared-policy audit (`docs/LEARNER_AUDIT.md`) is
  specific to this project.

## Conclusion

**No need to reference or adopt either for the current work.** Revisit
noeira only if (a) its physics3d is shown to take per-world models at run
time and to match MuJoCo 3.15 on this project's contact cases, and (b) a
fluid-force hook exists. Cheap check when that day comes: read
`physics3d/README.md` Validation and `TEST.md`, then run its MuJoCo parity
tests on one of our MJCF exports (needs permission to clone and install).
Not done: reading source, checking tolerances, measuring speed.
