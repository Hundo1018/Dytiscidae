# A12 — Tool survey: unsteady aerodynamics, spot-validation CFD, validation references, optimisation and structure tools, co-design

Survey date: 2026-11-12. Method: web search and page fetches only; package metadata read from the
PyPI JSON API and the GitHub / GitLab / SourceForge APIs on the same day. Nothing was installed or run.

## How to read the labels

Every claim carries a letter and a source number, e.g. `[V3]`.

- **V** = VERIFIED: seen on the page, docs, paper or API response that was fetched (source number given).
- **S** = SNIPPET: search-result snippet or a secondary source describing the primary one.
- **U** = UNKNOWN: looked for and not found; what was tried is in section "Not found".
- **I** = INFERRED: not on any fetched page; the reasoning is stated next to it. Values quoted from
  memory are marked I and must be re-derived before use.

Source numbers are local to each topic and listed under its table.

Target regime restated for reference: 0.3-0.8 kg, span ~1 m, 3-8 Hz in air, 0.5-2 Hz in water, chord
Re 3e4-2e5 in air.

---

## A. Mid-fidelity unsteady aerodynamics (planform, tip vortices, wing-wake interaction)

Common limit, stated once: every tool in this table is a potential-flow method (vortex lattice, panel,
lifting line, vortex particles). None was found to model a free surface or cavitation `[U]`. Use in water
is a change of density and viscosity inputs only `[I: potential-flow loads scale with density; no page
fetched states water use]`.

| Tool | What it does | What it cannot do | Cost or run time | License and status | URL | Label |
|---|---|---|---|---|---|---|
| **Ptera Software** (Python) | Steady horseshoe VLM, steady ring VLM, unsteady ring UVLM; "Unsteady simulations support both fixed and free wakes"; one or more wings per aircraft; more than one airplane since v2.0.0; motion "defined using any time-dependent functions of sweep, pitch, and heave angles"; beta aeroelastic module (torsional spring per wing cross section); beta free flight coupling to MuJoCo, single airplane `[V1]`. Described on PyPI as "unsteady aerodynamics solver for analyzing flapping-wing flight" `[V2]`. | Inviscid: "As the UVLM is an inviscid solver, it cannot account for separation" `[V4]`. Leading-edge separation model is a to-do item ("We should implement a leading-edge model to account for flow separation") `[V3]`. No viscous drag `[I: follows from inviscid UVLM; no skin-friction feature on the README]`. Validation is one case: Yeo et al. 2011 ornithopter, case C, pressure-derived lift; MAE 0.0291 N over the flap cycle; a phase shift and an overestimate of lift magnitude in the first and last quarter of the cycle; the experimental data were digitised from a figure, not tabulated `[V4]`. No tandem-wing flapping validation found `[U]`. | README: "most unsteady simulations take under a minute" `[V1]`. 2021 validation report (older code): 22.5 min for 5 chordwise x 18 spanwise panels, 3 flap cycles, 224 time steps, prescribed wake, 4,104 ring vortices at the last step, laptop Xeon E3-1505M v5 `[V4]`. Converged at 3 flap cycles and 5x18 panels in that study `[V4]`. | MIT; v5.1.0 released 2026-07-15; last commit 2026-11-09; Python >=3.11,<3.14 `[V2][V5]`. | 1-5 | V |
| **SHARPy** (Imperial College) | UVLM coupled to a geometrically exact composite beam and multibody dynamics ("hinges, articulations and prescribed nodal motions"); wake as vortex rings or infinite horseshoes; linear source panels for fuselages `[V6]`. Wake shape can be free or prescribed helicoidal `[V7]`. | Beam model "limits analyses to slender structures" `[V6]`. No viscous or separation model found on the README or capabilities page `[U]`. No flapping-wing case on either page; listed verification cases are X-HALE, Sears gust, Goland flutter, wind-turbine rotor `[V7]`. | Not stated on the pages fetched `[U]`. | BSD-3-Clause; release 2.4 on 2025-03-21; last commit 2026-10-12 `[V8]`. | 6-8 | V |
| **DUST** (Politecnico di Milano) | "from single dimensional lifting line to fully three dimensional objects"; "Fast Multipole optimized particle wake"; components are assigned to a reference frame that moves; coupled version with MBDyn `[V9][V10]`. Element types named in the group's papers: surface panels, vortex lattice, lifting lines, vortex-particle wake; MBDyn coupling through preCICE `[S11]`. | Limits not stated on the pages reached; the Tugnoli et al. 2021 paper returned HTTP 403 `[U]`. Flapping-wing validation not found `[U]`. Python interface not mentioned `[U]`. | "mid-fidelity aerodynamics analysis at workstation level" `[V10]`; no timings found `[U]`. | MIT `[V9]`. Latest version and date not obtained (GitLab API gave no JSON; project page shows 32 tags, 8 releases) `[U]`. | 9-11 | V/S |
| **FLOWUnsteady** (BYU, Julia) | Reformulated vortex particle method ("meshless CFD method solving the LES-filtered incompressible Navier-Stokes equations in their vorticity form") with VLM, strip theory, blade elements and a 3-D panel method; wings enter as actuator line or actuator surface models; "Maneuvering vehicle with prescribed kinematics" `[V12]`. Has a two-wing example where the front wing oscillates at 25 Hz, 5 deg and its wake reaches a swept rear wing `[V13]`. | Verbatim limits: "Viscous drag and separation is only captured through airfoil lookup tables, without attempting to shed separation wakes"; "Incompressible flow only"; "CPU parallelization through OpenMP without support for distributed memory" `[V12]`. Listed validations are propeller, rotor in hover, blown wing, rotor noise; no flapping-wing validation found; the two-wing example has no comparison with experiment `[V12][V13]`. No GPU support stated `[V12]`. | Two-wing oscillating example: "~10 minutes on a Dell Precision 7760 laptop", 200 time steps `[V13]`. | MIT; v3.4.0 on 2025-08-25; last commit 2026-09-22 `[V14]`. FLOWVLM (its VLM): AGPL-3.0, v2.1.4 on 2025-06-16 `[V14]`. | 12-14 | V |
| **AeroSandbox** (Python) | Design optimisation by automatic differentiation; VLM; NeuralFoil airfoil model; interfaces to AVL, XFLR5, XFoil, ASWING, MSES `[V15]`. | No unsteady analysis is listed on the README `[V15]`; steady-only is therefore the working assumption `[I]`. | Not stated `[U]`. | MIT; v4.2.10 on 2026-07-05; last commit 2026-10-07 `[V16]`. | 15-16 | V |
| **OpenVSP / VSPAERO** | Parametric geometry plus VSPAERO; version 3.56.0 posted 2026-11-05 `[V17]`. VSPAERO runs in vortex-lattice (thin) or panel (thick) mode; unsteady modes are rotating blades, a 6-DOF "dynamic geometry" mode and sinusoidal p/q/r stability runs `[V18][S18]`. | Potential flow with no viscous modelling `[S18]`. Prescribed flapping, pitching or plunging of a wing is not among the unsteady modes described in the 2021 forum thread `[V18]`; whether a newer release adds it was not established `[U]`. | Not stated `[U]`. | NASA Open Source Agreement 1.3 `[V19]`; last commit 2026-11-04 `[V19]`. | 17-19 | V/S |
| **XFLR5** | Lifting line, VLM and 3-D panel analysis for model aircraft `[S20]`. | Inviscid; flat wake; uniform-strength panels; no boundary-layer coupling loop (author's own list of shortcomings, secondary copy) `[S20]`. No unsteady analysis found `[U]`. | Not stated. | GPL v3 since v6.30 (2016) `[S20]`; v6.62 file dated 2026-03-24 on SourceForge `[V21]`. Successor flow5: site returned empty pages, licence and capabilities not established `[U]`. | 20-21 | S/V |
| **VortexLattice.jl** (BYU, Julia) | Steady VLM with fixed wakes; "Free wakes through unsteady analysis"; multiple lifting surfaces; stability derivatives `[V22]`. | "does not yet incorporate the fast multipole method to speed up wake computations"; unsteady part verified against Katz and Plotkin results, not against flapping experiments `[V22]`. | Not stated. | MIT; v1.0.3 on 2026-10-23; last commit 2026-11-03 `[V14]`. | 22, 14 | V |
| **OpenAeroStruct** (Python, OpenMDAO) | Steady VLM plus beam finite elements with analytic derivatives, for aerostructural optimisation `[S23]`. | Steady only `[I: no unsteady capability appeared in the docs snippets]`. | Not stated. | v2.12.1 on 2026-11-10. GitHub reports Apache-2.0, PyPI metadata says BSD-3: the two disagree `[V2][V14]`. | 23 | S/V |
| **UnsteadyFlowSolvers.jl** (Glasgow) | "library of low-order solvers for unsteady aerodynamics and aeroelasticity", "based on the discrete-vortex method" `[V24]`. The method family (LDVM, Ramesh et al.) sheds a leading-edge vortex when a leading-edge suction parameter exceeds a critical value `[S25]`. | Scope (2-D or 3-D), validation and limits are not on the README `[U]`. The critical suction value is an input calibrated per airfoil and Re `[S25]`. | Not stated. | MIT; last push 2024-12-17 `[V14]`. | 24-25 | V/S |
| **MachUpX** (USU) | Numerical lifting line `[I: from memory, page not fetched]`. | Steady `[I]`. | Not stated. | MIT; last commit 2024-05-23 `[V14]`. | 14 | V (status only) |

Sources A:
1. https://raw.githubusercontent.com/camUrban/PteraSoftware/main/README.md
2. https://pypi.org/pypi/PteraSoftware/json (and the same API for the other PyPI packages named)
3. https://pypi.org/project/PteraSoftware/3.0.0/
4. https://raw.githubusercontent.com/camUrban/PteraSoftware/main/validation/validation_paper.pdf (Urban and Agarwal, 2021; text extracted locally)
5. https://api.github.com/repos/camUrban/PteraSoftware (release and commit dates)
6. https://github.com/ImperialCollegeLondon/sharpy
7. https://ic-sharpy.readthedocs.io/en/main/content/capabilities.html
8. https://api.github.com/repos/ImperialCollegeLondon/sharpy
9. https://www.dust.polimi.it/
10. https://public.gitlab.polimi.it/DAER/dust/-/raw/master/README.md
11. https://re.public.polimi.it/bitstream/11311/1173647/3/TUGNM_OA_01-21.pdf (403 on fetch; content from search snippets of this and related polimi papers)
12. https://github.com/byuflowlab/FLOWUnsteady and https://flow.byu.edu/FLOWUnsteady/
13. https://flow.byu.edu/FLOWUnsteady/examples/heavingwing/
14. https://api.github.com/repos/byuflowlab/FLOWUnsteady, /byuflowlab/FLOWVLM, /byuflowlab/VortexLattice.jl, /mdolab/OpenAeroStruct, /KiranUofG/UnsteadyFlowSolvers.jl, /usuaero/MachUpX
15. https://github.com/peterdsharpe/AeroSandbox
16. https://pypi.org/pypi/aerosandbox/json and https://api.github.com/repos/peterdsharpe/AeroSandbox
17. https://openvsp.org/
18. https://groups.google.com/g/openvsp/c/3iE6Dk69P4o (posts of 2021-09-30 and 2021-12-10; other OpenVSP group threads as snippets)
19. https://api.github.com/repos/OpenVSP/OpenVSP (licence file head and commit date)
20. https://www.scribd.com/document/494796986/Part-IV-Limitations and https://mac.softpedia.com/progChangelog/XFLR5-Changelog-76783.html
21. https://sourceforge.net/projects/xflr5/best_release.json
22. https://github.com/byuflowlab/VortexLattice.jl
23. https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/aero_walkthrough.html
24. https://github.com/KiranUofG/UnsteadyFlowSolvers.jl
25. https://arxiv.org/pdf/2206.11597

---

## B. High-fidelity tools for a handful of spot validations

| Tool | What it does | What it cannot do | Reported cost | License and status | URL | Label |
|---|---|---|---|---|---|---|
| **OpenFOAM, overset** (`overPimpleDyMFoam`, `overInterDyMFoam`) | 3-D flapping wing with overset mesh and dynamic-k LES has been published: Calado et al., OpenFOAM v2012, semi-elliptical rigid wing, span 50 mm, AR 3.25, Re 1e2 to 1e4, relative RMSE below 2 % on lift and drag against Lee et al. 2016 `[V1]`. A two-phase VOF overset solver (`overInterDyMFoam`) exists since v1706 with a floatingBody tutorial `[S2]`. | Published flapping cases found stop at Re 1e4 `[V1]`; none at 3e4-2e5 found `[U]`. "Profiling identifies the overset method as the main computational bottleneck" (Poletti et al. 2025) `[V3]`. No published water exit or entry of a flapping wing found `[U]`. No native GPU solver `[I: from memory, not fetched]`. | Calado et al.: component mesh 130e3 cells (grid study 41e3 to 430e3), 5 flapping periods per run, 165 runs; core-hours not reported `[V1]`. A VKI flapping case used about 3M cells `[S4]`. | GPL-3 `[I: from memory]`; OpenFOAM-dev last commit 2026-11-12 `[V5]`. | 1-5 | V/S |
| **WaterLily.jl** (Julia) | "solves the unsteady incompressible 2D or 3D Navier-Stokes equations on a Cartesian grid", Boundary Data Immersion Method, geometric multigrid; bodies given as signed-distance function plus a time-dependent map; CPU threads and CUDA / AMD / Apple GPU backends `[V6][V7]`. Moving-body validation: oscillating cylinder at Re 7620, 16.8M cells, mean power coefficient -4.39 against -4.52 measured (within 3 %) `[V7]`. Flapping whale-tail demo at Re 1e4, 96 cells per chord, compared qualitatively `[V7]`. Differentiable by automatic differentiation `[V7]`. | Uniform spacing: for a sphere at Re 3700 with 168 cells per diameter "only 3 grid points are used to represent the boundary layer", and "the current Cartesian-mesh method using constant spacing requires vast resources to fully resolve the boundary layer" `[V7]`. Multi-GPU and multi-CPU are a stated development goal, not a current feature `[V6]`. Free surface or two-phase flow: not mentioned `[U]`. Turbulence model: not mentioned on the README `[U]`. No flapping-wing force validation at Re 3e4-2e5 found `[U]`. | 1.57 ns per degree of freedom per time step on an NVIDIA H100 (Taylor-Green case, largest grid); 226M DOF uses 29 GB in FP32; sphere validation grids 43M to 299M DOF `[V7]`. README sphere example: about 1.3M DOF, 50 time units in about 40 s on a GPU `[V6]`. | MIT `[V8]`; v1.11.1 on 2026-11-09 `[V8]`. | 6-8 | V |
| **IBAMR** | "distributed-memory parallel implementation of the immersed boundary (IB) method" with Cartesian adaptive mesh refinement, MPI `[V9]`. Water entry and exit of a wedge and a cylinder simulated with its multiphase solver (Bhalla et al., arXiv 1904.04078) `[S10]`. | GPU support not mentioned `[V9]`. No flapping wing with free surface found `[U]`. | Not found `[U]`. | Custom BSD-style licence text (GitHub classifies it "Other") `[V11]`; v0.19.0 on 2026-03-11; last commit 2026-11-11 `[V11]`. | 9-11 | V/S |
| **FluidX3D** (LBM, OpenCL) | D3Q19 lattice Boltzmann, SRT or TRT, Smagorinsky-Lilly subgrid model, moving solid boundaries, voxelised moving and rotating geometry, force and torque on objects, volume-of-fluid free surface with PLIC, multi-GPU on one node `[V12]`. | "works only in the weakly compressible regime at Mach numbers < 0.3"; "Grid cell size is the same everywhere in the simulation box" (no refinement); "FluidX3D can model either water or air, but not both at the same time": the free-surface model "ignores the gas phase completely" `[V12]`. All arithmetic is FP32 `[V12]`. No accuracy figures for lift or drag on the README, no flapping example seen `[V12]`. Wall model not mentioned `[U]`. | 55 bytes per cell, 19M cells per GB VRAM; RTX 4090: 11,496 MLUPs/s; RTX 5090: 19,141 MLUPs/s (D3Q19 SRT, empty box, no extensions) `[V12]`. | **Licence terms** `[V13]`: use allowed for "public research, education or personal use"; "Commercial use is not allowed" without the owner's permission (no sale, no paid product or service deriving value from it); "Military use is not allowed"; "You may not train AI models on the source code"; if binaries of altered versions, or data or results from them, are published, "the altered source code must be published as well"; cite the listed papers; German law applies. v3.8 on 2026-09-05 `[V14]`. | 12-14 | V |
| **Palabos** (LBM, CPU) | Immersed walls, moving objects in exterior flow, free-surface and multiphase modules (release notes) `[S15]`. | GPU support not established `[U]`. Flapping-wing case not found `[U]`. | Not found. | AGPLv3 `[V16]`; latest tagged release v2.3.0 on 2021-06-05; project activity 2026-07-28 `[V16]`. | 15-16 | V/S |
| **waLBerla** (LBM) | "massively parallel framework for multiphysics"; "multiphase and free-surface flows"; "complex and moving geometries using the partially saturated cells method"; "vectorization, OpenMP, CUDA and HIP" `[V17]`. | Flapping-wing case not found `[U]`. | A showcase ran on 32 A100 GPUs `[V17]`; nothing at single-GPU scale found. | GPLv3; v7.3 tagged 2026-07-21 `[V17][V18]`. | 17-18 | V |
| **nekRS** (spectral element) | "Arbitrary-Lagrangian-Eulerian moving mesh", "Overlapping overset grids", "LES and RANS turbulence models", "MPI + OCCA supporting CUDA, HIP, DPC++" `[V19]`. | Needs a body-fitted hexahedral mesh that follows the wing `[I: spectral-element ALE]`. No flapping or pitching wing example found `[U]`. No free-surface capability claimed on the README `[V19]`. | Not found for a comparable case. | BSD 3-clause; v26.0 on 2026-01-27 `[V19][V14]`. | 19 | V |
| **PyFR** (flux reconstruction, GPU) | High-order compressible solver on GPUs `[I: from memory]`. | Moving boundaries: maintainer reply of 2025-04-22, "We don't currently support this." `[V20]` A flapping wing with a moving mesh is therefore out. | n/a | BSD; v3.1 on 2026-03-13; last commit 2026-11-12 `[V14]`. | 20 | V |
| **Reference for the Re range** | Zhang and Huang, J. Fluid Mech. 963 A16 (2023): wall-modelled LES with immersed boundary of a flapping batoid model at Re 1,480, 14,800 and 148,000 `[V21]`. Its introduction says most high-fidelity simulations were below Re 1e4 because of grid needs `[S21]`. | Grid size and CPU hours are behind the paywall `[U]`. | Not obtained. | Paper. | 21 | V/S |

### Cloud prices read on 2026-11-12 (USD per hour, before tax)

| Provider and instance | Hardware | On-demand | Spot / community | URL | Label |
|---|---|---|---|---|---|
| AWS `g6e.xlarge` | 1x NVIDIA L40S 48 GB, 4 vCPU, 32 GiB | 1.861 | 1.639 (spot) | https://instances.vantage.sh/aws/ec2/g6e.xlarge (data stamped 2026-11-12; region not stated on the page) | V |
| AWS `c7i.16xlarge` | 64 vCPU Sapphire Rapids, 128 GiB | 2.856 | 1.571 (spot) | https://instances.vantage.sh/aws/ec2/c7i.16xlarge (same stamp) | V |
| RunPod RTX 4090 | 24 GB VRAM, 6 vCPU, 41 GB | 0.89 (secure cloud) | 0.34 (community cloud) | https://www.runpod.io/pricing | V |
| RunPod RTX 5090 | 32 GB VRAM, 9 vCPU, 35 GB | 1.19 | 0.69 | same | V |
| RunPod L40S | 48 GB VRAM, 16 vCPU, 94 GB | 1.09 | 0.79 | same | V |
| RunPod H100 PCIe | 80 GB VRAM, 16 vCPU, 188 GB | 2.89 | 1.99 | same | V |
| Lambda A10 / A6000 / A100 40 GB / H100 PCIe | 24 / 48 / 40 / 80 GB | 1.29 / 1.09 / 1.99 / 3.79 | none listed | https://lambda.com/pricing (page states these prices take effect 2026-11-16 00:00 UTC) | V |

### Cost per validation run (inference, with the arithmetic)

- WaterLily on one H100 `[I]`: a grid of 50M cells is about 150M DOF (3 velocity components per cell: the
  README's 96x64x64 sphere grid is quoted as about 1.3M DOF). 150e6 DOF x 1e5 time steps x 1.57e-9 s =
  23,550 s = 6.5 h. At 1.99-3.79 USD/h that is 13-25 USD. The 1.57 ns figure is for a case with no body;
  in the paper's moving-cylinder profile the body-mapping routine took 10 % of a time step. The number of
  time steps (1e5) is an assumption, not a measurement.
- FluidX3D on one RTX 4090 `[I]`: 24 GB x 19M cells/GB = 456M cells; 456e6 x 1e5 steps / 11.496e9 per second
  = 3,970 s = 1.1 h, i.e. 0.4-1.0 USD at 0.34-0.89 USD/h. The MLUPs figure is for an empty box with no
  extensions; moving boundaries, the subgrid model and force summation are extensions, and their slowdown
  was not found.
- OpenFOAM on `c7i.16xlarge` `[I]`: no core-hour figure was found for a flapping overset case, so no
  estimate is given. Each 10 h of wall time costs 28.6 USD on demand or 15.7 USD spot.
- Resolution caveat that applies to both uniform-grid GPU solvers: at chord Re 3e4-2e5 a uniform grid does
  not resolve the boundary layer (WaterLily's own statement at Re 3700, source 7). Any such run is an
  under-resolved LES whose force accuracy at this Re is unverified until it reproduces a measured case
  from section C.

Sources B:
1. https://ar5iv.labs.arxiv.org/html/2212.09464
2. https://www.openfoam.com/node/297 and https://www.openfoam.com/node/480 (release notes, via search)
3. https://arxiv.org/abs/2510.24518
4. https://www.esi-group.com/sites/default/files/resource/other/2602/OpenFOAM_Conference_2021_Abstract_Poletti_Karman_Institute.pdf
5. https://api.github.com/repos/OpenFOAM/OpenFOAM-dev
6. https://github.com/WaterLily-jl/WaterLily.jl
7. https://arxiv.org/pdf/2407.16032 (Weymouth and Font, Comput. Phys. Commun. 315, 109748, 2025; pages 6-12 read)
8. https://api.github.com/repos/WaterLily-jl/WaterLily.jl (licence file and release)
9. https://ibamr.github.io/about
10. https://arxiv.org/pdf/1904.04078
11. https://api.github.com/repos/IBAMR/IBAMR
12. https://github.com/ProjectPhysX/FluidX3D
13. https://github.com/ProjectPhysX/FluidX3D/blob/master/LICENSE.md
14. https://api.github.com/repos/ProjectPhysX/FluidX3D (and /PyFR/PyFR, /Nek5000/nekRS)
15. https://palabos-forum.unige.ch/t/new-palabos-release-v1-5r0/1657
16. https://gitlab.com/unigespc/palabos and https://gitlab.com/api/v4/projects/unigespc%2Fpalabos/releases
17. https://www.walberla.net/
18. https://i10git.cs.fau.de/api/v4/projects/walberla%2Fwalberla/repository/tags
19. https://github.com/Nek5000/nekRS
20. https://pyfr.discourse.group/t/can-pyfr-be-used-to-solve-moving-boundary-problems/1113
21. https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/hydrodynamics-of-a-swimming-batoid-fish-at-reynolds-numbers-up-to-148-000/23BF7C45DC59579059A08F5D5505A88F

---

## C. Validation references and the numbers to compare

### Analytic

| Source | What it gives | Limits | Numbers | URL | Label |
|---|---|---|---|---|---|
| Prandtl lifting line, elliptic loading | Minimum induced drag at fixed span and lift | Does not cover "swept or low-aspect-ratio wings"; assumes steady flow `[V1]` | C_Di = C_L^2 / (pi AR), span efficiency e = 1; "any wing planform can achieve the same efficiency through twist" `[V1]` | 1 | V |
| Glauert form for other planforms | Departure from elliptic loading | Same as above | C_Di = C_L^2 / (pi AR) x (1 + delta), e = 1/(1 + delta) `[S2]`. A table of delta for untwisted rectangular wings was not found online; from memory delta is about 0.05 near AR 6 (e about 0.95) `[I: memory of Glauert's table; recompute from the monoplane equation before use]`. NASA's beginner page quotes e = 0.7 for a rectangular wing, which is a whole-aircraft rule of thumb and not the lifting-line value `[S2]`. | 2 | S / I |
| ADODG twist benchmark | Known-answer optimisation: recover elliptic loading by twist | Inviscid | Rectangular wing, NACA 0012 sections, Mach 0.5, target C_L = 0.375, design variables are section twists; "The purpose of this case is to produce a lift distribution that is close to elliptical and a span efficiency factor approximately equal to one" `[V3]`. Figure gives semi-span 3.06 c, so AR about 6.12 and the target C_Di = 0.375^2 / (pi x 6.12) = 0.00731 `[I: arithmetic from the figure dimensions]`. Yang et al. report both an Euler solver (SU2) and a VLM hybrid "recovering a nearly elliptical lift distribution", the hybrid two orders of magnitude faster `[V3]`. | 3, 4 | V |
| Theodorsen function C(k) = F + iG | Lift lag of a harmonically oscillating thin airfoil | 2-D, small amplitude, attached flow, flat wake | F(0) = 1, G(0) = 0; F -> 1/2 and G -> 0 as k -> infinity; closed form C(k) = K1(ik) / (K0(ik) + K1(ik)) `[S5]`. Tabulated values from memory: k = 0.1: 0.832 - 0.172i; 0.2: 0.728 - 0.189i; 0.5: 0.598 - 0.151i; 1.0: 0.539 - 0.100i `[I: memory; recompute from the closed form]`. SHARPy ships this function as an analytical utility `[S5]`. | 5 | S / I |
| Wagner function | Lift build-up after a step change of incidence | 2-D, small disturbance | Starts at 1/2 of the steady value and tends to 1; R. T. Jones fit phi(s) = 1 - 0.165 exp(-0.0455 s) - 0.335 exp(-0.3 s) `[I: memory; no page fetched states it]`. Garrick (1938) showed the equivalence with Theodorsen's function `[S5]`. | 5 | I / S |
| Garrick, NACA Report 567 (1936/37) | Thrust and efficiency of a plunging and pitching airfoil in inviscid flow | "assumes inviscid flow, small amplitudes ... and a non-deforming vortex wake" `[S6]`; no viscous drag, so net thrust at low frequency is overstated: "Inviscid theory predicts efficiency close to 100% under light thrust conditions and for low frequency ... fails to take into account the effect of drag" `[V7]` | Efficiency tends to the constant 1/2 as reduced frequency grows `[S6]`; positive thrust for every pure plunge `[S6]`. For pure plunge eta = (F^2 + G^2) / F, which gives 1 at k = 0 and 1/2 at k -> infinity `[I: from Garrick's thrust and power expressions with the Theodorsen limits above]`. NTRS record exists, PDF has no text preview `[V6]`. | 6, 7 | S / I |

### Experimental

| Source | Set-up | Limits for this project | Key numbers | URL | Label |
|---|---|---|---|---|---|
| Anderson, Streitlien, Barrett, Triantafyllou, J. Fluid Mech. 360 (1998) | Heaving and pitching foil; forces and power at Re 40,000, DPIV at Re 1,100 `[V8]` | 2-D foil between end plates `[I: memory]`; foil section and span not re-verified `[U]` | "High propulsive efficiency, as high as 87%" under optimal wake formation `[V8]`. In the review's wake map of these data (heave/chord = 1) the region with a reverse Karman street, 7 deg < alpha_max < 50 deg and 0.2 < St < 0.5, "contains the region of optimal efficiency"; a mild leading-edge vortex slightly raised efficiency and a strong one lowered it `[V7]`. | 7, 8 | V |
| Read, Hover, Triantafyllou, J. Fluids Struct. 17 (2003) 163-183 | Heaving and pitching foil, propulsion and manoeuvring | Abstract only reached through secondary pages | Efficiency plateau of roughly 50-60 %; best thrust at pitch-heave phase 90-100 deg; mean side-force coefficient up to 5.5; instantaneous lift coefficient up to 15 `[S9]`. The peak-efficiency value and its Strouhal number were not confirmed `[U]`. | 9 | S |
| 3-D flapping foils collected in Triantafyllou, Techet, Hover, IEEE J. Oceanic Eng. 29 (2004) | Review | Individual papers not fetched | Scherer: up to 70 % for rectangular moderate-AR wings. Lai et al.: up to about 75 % (with scatter), rectangular NACA 16-012, AR 4. DeLaurier and Harris: 18-50 %, rectangular NACA 0012, AR 4, heave amplitude 0.625 chord, Re 30,000. Lift-based fin propulsion about 60 % against 10 % for drag-based rowing. Tsutahara and Kimura (Weis-Fogh pair, AR 1.8, Re up to 300,000): up to 58 %. Chordwise flexibility can raise efficiency above 80 % (Prempraneerach et al.) `[V7]`. | 7 | V |
| Heathcote, Wang, Gursul, J. Fluids Struct. 24 (2008) 183-199 | Rectangular wing in pure heave, water tunnel, force balance and PIV, three spanwise stiffnesses `[V10]` | Heave only; Re not in the abstract `[U]` | For St > 0.2 a limited spanwise flexibility gave a small thrust increase and a small power decrease; a much more flexible wing lost thrust because tip and root moved out of phase; benefit range overlaps 0.2 <= St <= 0.4 `[V10]`. | 10 | V |
| Hubel and Tropea, Exp. Fluids 2009 and J. Exp. Biol. 213 (2010) 1930 | Goose-sized flapping model in a wind tunnel: span 1.13 m, mean chord 0.141 m, flapping up to 2.2 Hz, reduced frequency 0.05-0.3, Re 28,000-113,000; forces and PIV `[S11]` | Rigid simplified wing; flapping frequency below this project's 3-8 Hz; open data not found `[U]` | Flow "cannot be simplified as quasi-steady" under bird-scale conditions; leading-edge vortex present and not stabilised over the wing-beat cycle `[S11]`. Closest match found to this project's span and Re. | 11 | S |
| Yeo et al. 2011 (via the Ptera validation report) | Ornithopter on a wind-tunnel stand, differential pressure sensors on a rigid flapping wing, forward flight and hover `[V12]` | "did not provide the raw data in tabulated form"; the report notes an inconsistency in the stated free-stream speed (2.9 m/s) `[V12]` | Digitised case C ships in `validation/experimental_data` of the Ptera repository `[V12]`. | 12 | V |
| Usherwood and Lehmann, J. R. Soc. Interface 2008 | Hovering dragonfly robot in mineral oil, Re 105 (fore) and 125 (hind), wings 1.25 chords apart vertically `[V13]` | Re three orders of magnitude below this project; hover only | Best phase: hind wing leads by +25 % of the stroke (a quarter cycle): "22 per cent less power than hovering with only the fore wings" at equal mean lift; worst phase -25 %; +25 % needs 16 % less power than -25 %; interaction "largely detrimental in terms of lift" `[V13]`. | 13 | V |
| Boschitsch, Dewey, Smits, Phys. Fluids 26, 051901 (2014) | Two pitching hydrofoils in line, phase and spacing swept `[V14]` | Pitch only, 2-D; Re not captured `[U]` | Downstream foil thrust and efficiency between 0.5 and 1.5 times an isolated foil depending on spacing and phase; upstream foil changes only at close spacing; coherent wake mode enhances, branched mode reduces `[V14]`. | 14 | V |
| Warkentin and DeLaurier, J. Aircraft 2007 | Tandem flapping membrane wings in a wind tunnel, phase 0-360 deg, forward flight `[S15]` | Primary paper not fetched | At certain phase angles the tandem arrangement's propulsive efficiency was "almost doubled" relative to single wings; thrust increased only at certain phases `[S15]`. | 15 | S |
| Open data sets | Zenodo 4909338: lift and drag of flapping wings in water at Re about 3,600 with PIV, raw and processed `[S16]`. Calado et al.: 165 LES runs at Re 1e2-1e4 (data availability not checked) `[V17]`. | None found at Re 3e4-2e5 with tabulated forces `[U]` | - | 16, 17 | S |

Sources C:
1. https://en.wikipedia.org/wiki/Lifting-line_theory
2. https://archive.aoe.vt.edu/mason/Mason_f/C9EinAerodynamics.pdf, https://sites.usc.edu/drydenwt/files/2022/06/SMac_JA_10.pdf, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient
3. https://www.icas.org/icas_archive/ICAS2018/data/papers/ICAS2018_0440_paper.pdf (pages 1-2 read)
4. https://mdolab.engin.umich.edu/wiki/aerodynamic-design-optimization-workshop
5. https://ntrs.nasa.gov/api/citations/20190025361/downloads/20190025361.pdf, https://arxiv.org/pdf/2204.01204, https://ic-sharpy.readthedocs.io/en/rc-1.1.1/includes/utils/analytical/theo_fun.html
6. https://ntrs.nasa.gov/citations/19930091642 and https://riuma.uma.es/entities/publication/d6f719f0-f363-48c6-bb99-cc79fae468e7 (Fernandez-Feria 2016)
7. https://web.mit.edu/ahtechet/www/papers/Techet_IEEEreview2004.pdf (pages 586-591 read)
8. https://dspace.mit.edu/handle/1721.1/25614
9. https://independent.academia.edu/DouglasRead2 and https://scilit.com/publications/aed1fa833cc68e87cd438a175c2a7ebe
10. https://researchportal.bath.ac.uk/en/publications/effect-of-spanwise-flexibility-on-flapping-wing-propulsion/
11. https://cob.silverchair.com/jeb/article-pdf/1271015/2548.pdf and https://sicb.org/?p=29411
12. https://raw.githubusercontent.com/camUrban/PteraSoftware/main/validation/validation_paper.pdf
13. https://pmc.ncbi.nlm.nih.gov/articles/PMC2607445
14. https://collaborate.princeton.edu/en/publications/propulsive-performance-of-unsteady-tandem-hydrofoils-in-an-in-lin/
15. https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s10409-012-0210-8.pdf (Broering et al. 2012, citing the 2007 paper)
16. https://zenodo.org/records/4909338
17. https://ar5iv.labs.arxiv.org/html/2212.09464

---

## D. Shape and structure generation and optimisation tools

Version, date and licence are from the PyPI JSON API (`https://pypi.org/pypi/<name>/json`) and the GitHub
API (`https://api.github.com/repos/<owner>/<repo>`), read 2026-11-12, label V. "What it does" quotes the
package's own summary line unless marked.

| Tool | What it does | What it cannot do | Cost | License and status | URL | Label |
|---|---|---|---|---|---|---|
| **pycma** (`cma`) | "CMA-ES, Covariance Matrix Adaptation Evolution Strategy for non-linear numerical optimization in Python" | Single-objective continuous search; no archive of diverse solutions `[I]` | CPU, negligible beside the evaluation `[I]` | BSD-3-Clause; 4.5.0 on 2026-09-13; last commit 2026-10-16 | https://github.com/CMA-ES/pycma | V |
| **pymoo** | "Multi-Objective Optimization in Python" | - | - | Apache-2.0; 0.6.2 on 2026-06-28; last commit 2026-11-11 | https://github.com/anyoptimization/pymoo | V |
| **pyribs** (`ribs`) | "A bare-bones Python library for quality diversity optimization"; official implementation of CMA-ME, CMA-MAE, CMA-MEGA and Discount Model Search; GridArchive `[V: docs.pyribs.org]` | No statement on noisy evaluations or GPU on the docs front page | - | MIT; 0.12.1 on 2026-11-07 | https://docs.pyribs.org/en/stable/ | V |
| **QDax** | "A Python Library for Quality-Diversity and NeuroEvolution" (JAX-based `[I: memory]`) | Gains depend on the evaluation itself running in JAX on the accelerator `[I]` | - | MIT; 0.5.0 on 2025-05-27; last commit 2025-10-30 | https://github.com/adaptive-intelligent-robotics/QDax | V |
| **DEAP** | "Distributed Evolutionary Algorithms in Python" | - | - | LGPL-3.0; 1.4.4 on 2026-04-17 | https://github.com/DEAP/deap | V |
| **Nevergrad** | "A Python toolbox for performing gradient-free optimization" | - | - | MIT; 1.0.12 on 2025-04-23; last commit 2026-11-02 | https://github.com/facebookresearch/nevergrad | V |
| **EvoTorch**, **pygmo** | "evolutionary computation library built directly on top of PyTorch"; "Parallel optimisation for Python" | - | - | Apache-2.0, 0.6.1 on 2025-05-14; MPL-2.0, 2.20.0 on 2026-10-06 | PyPI | V |
| **BoTorch** | "Bayesian Optimization in PyTorch". Multi-fidelity models and acquisition functions (`SingleTaskMultiFidelityGP`, knowledge-gradient variants) `[I: memory; the tutorial URL tried returned 404]` | Gaussian-process cost grows with the number of evaluations; suited to hundreds, not tens of thousands, of points `[I]` | - | MIT; 0.18.1 on 2026-06-08; last commit 2026-11-10 | https://github.com/pytorch/botorch | V (status) / I (MF) |
| **Ax** | "Adaptive Experimentation" (front end over BoTorch `[I]`) | - | - | MIT; 1.3.1 on 2026-06-09 | https://github.com/facebook/Ax | V |
| **SMT** | "open-source Python package consisting of libraries of surrogate modeling methods": kriging variants (KRG, KPLS, sparse GP), multi-fidelity kriging and co-kriging (MFK, MFCK, MFKPLS and sparse versions), mixed-integer and hierarchical design spaces, EGO `[V: smt.readthedocs.io]` | - | - | BSD-3-Clause; 2.15.1 on 2026-11-04 | https://smt.readthedocs.io/en/latest/ | V |
| **OpenMDAO** | "OpenMDAO framework infrastructure" (multidisciplinary analysis and optimisation with analytic derivatives `[I: memory]`) | Pays off with gradient-based optimisation of differentiable models; adds little to a derivative-free evolutionary loop `[I]` | - | Apache-2.0; 3.45.2 on 2026-10-27 | https://github.com/OpenMDAO/OpenMDAO | V |
| **CadQuery** | "parametric scripting language for creating and traversing CAD models", on the OpenCASCADE kernel | - | - | Apache 2.0; 2.8.0 on 2026-06-21; last commit 2026-11-09 | https://github.com/CadQuery/cadquery | V |
| **build123d** | "A python CAD programming library" | - | - | Apache-2.0; 0.13.0 on 2026-09-21; last commit 2026-11-10 | https://github.com/gumyr/build123d | V |
| **cq-uav** | CadQuery library that generates a rectangular wing console with a box spar and an optional lattice body from an airfoil and wing size, with bending-stress and safety-factor checks | Rectangular console only; no rib or lightening-hole generation seen; repository not fetched | - | Not checked `[U]` | https://github.com/nomad-vagabond/cq-uav | S |
| **pytopo3d** | "3D SIMP Topology Optimization Framework for Python" | Laptop-scale limits not stated `[U]` | Not found | MIT; 0.3.0 on 2026-06-16 | https://pypi.org/project/pytopo3d/ | V |
| **FEniTop**, **DL4TO**, **JAX-FEM**, **ToPy** | "FEniCSx-based topology optimization supporting parallel computing"; "3D topology optimization ... based on PyTorch"; "Differentiable Finite Element Method with JAX"; "Topology Optimization using Python" | ToPy has had no push since 2022-08-31 | Not found | GPL-3.0 (last push 2024-11-01); Apache-2.0 (2024-11-19); GPL-3.0 (2026-10-09); other | https://github.com/missionlab/fenitop , /dl4to/dl4to , /deepmodeling/jax-fem , /williamhunter/topy | V |
| **fogleman/sdf**, **PicoGK** | "Simple SDF mesh generation in Python" (signed-distance modelling, usable for lattices and shells `[I]`); "compact and robust geometry kernel for Computational Engineering" | The PyPI package named `sdf` is a different project (Scientific Data Format); PicoGK is not a Python library `[I: memory]` | - | MIT, last push 2026-09-30; Apache-2.0, last push 2026-11-12 | https://github.com/fogleman/sdf , https://github.com/leap71/PicoGK | V |

---

## E. Co-design of body and motion on a small budget; checking that a search improves

| Source | What it demonstrates | What it does not show | Budget reported | Status | URL | Label |
|---|---|---|---|---|---|---|
| Gupta, Savarese, Ganguli, Fei-Fei, "Embodied intelligence via learning and evolution" (DERL), 2021 | Bilevel: tournament evolution of bodies outside, PPO learning inside each body | Nothing about small budgets; rigid-body locomotion only | Population 576; about 4,000 morphologies per run; 5 million interactions per morphology; tournament size 4; 288 workers x 4 cores = 1,152 CPUs; "less than 16 hours"; 3 runs per environment `[V]`. Product: 2e10 interactions per run `[I: 4,000 x 5e6]`. | paper | https://ar5iv.labs.arxiv.org/html/2102.02202 | V |
| Bhatia et al., "Evolution Gym", 2021 | Bilevel benchmark: genetic algorithm, Bayesian optimisation or CPPN-NEAT outside, PPO inside | "all tested algorithms fail to find robots that succeed in our hardest environments" | 6 seeds per curve; 80 Xeon processors; "several hours to twenty hours" per algorithm per task | paper, open benchmark | https://ar5iv.labs.arxiv.org/html/2201.09863 | V |
| Schaff, Yunis, Chakrabarti, Walter, "Jointly learning to construct and control agents", 2018 | One policy that receives the design parameters as input, trained over a distribution of designs that shifts toward better ones | Legged locomotion with fixed topology; budget not captured | 8 runs per experiment | paper | https://arxiv.org/abs/1801.01432 | S |
| Luck, Amor, Calandra, "Data-efficient co-adaptation of morphology and behaviour", 2019 | Reuses data from earlier designs (soft actor-critic) to predict the performance of new morphologies; "drastically less design prototypes" | Simulation only; no count in the abstract | not in abstract | paper | https://arxiv.org/abs/1911.06832 | V |
| Zheng, Hedrick, Mittal, J. Fluid Mech. 721 (2013) | Multi-fidelity pattern for flapping: a blade-element model searches the wing-stroke space, a Navier-Stokes solver re-scores the stroke it calls optimal (hawkmoth hover) | Kinematics only, body fixed; blade-element models "cannot accurately model time-history-related effects" | not captured | paper | https://engineering.jhu.edu/fsag/wp-content/uploads/2014/06/2013-Lingxiao-JFM.pdf | S |
| Ghommem et al., J. Fluids Struct. 33 (2012) | 3-D UVLM inside a deterministic global optimiser for flapping and morphing kinematics, maximising propulsive efficiency under lift and thrust constraints | Inviscid; reports efficiencies comparable to gradient-based studies "but at very different design points": the design space is multimodal; a hybrid local-plus-global scheme converged faster | not captured | paper | https://researchwith.stevens.edu/en/publications/global-optimization-of-actively-morphing-flapping-wings/ | S |
| Calado et al., arXiv 2212.09464 | Surrogate route: 165 overset LES runs train a data-driven force model, 25 runs held out as a test set | Re up to 1e4, one wing shape | 165 runs of 5 flapping periods | preprint | https://ar5iv.labs.arxiv.org/html/2212.09464 | V |
| Koos, Mouret, Doncieux, "The transferability approach", IEEE TEVC 17 (2013) | Second objective = a surrogate of the simulation-to-reality disparity, learned from a few transfers; works on a quadruped and an e-puck | The "reality" was a physical robot; using a higher-fidelity simulator in that role is this report's extension | "about ten experiments on the physical robot" | paper | https://arxiv.org/pdf/1307.1870 | S |
| Mania, Guy, Recht, "Simple random search ...", 2018 | A random-search method with linear policies matched state-of-the-art sample efficiency on MuJoCo locomotion; evaluation over "hundreds of random seeds" shows "high variability" | Says nothing about body search | at least 15x less computation than the fastest competing model-free method | paper | https://arxiv.org/abs/1803.07055 | V |
| Henderson et al., "Deep RL that matters", 2017; Agarwal et al., "... statistical precipice", 2021 | Without "significance metrics" it is hard to tell whether gains are real; with few runs, report interval estimates, interquartile mean and performance profiles (library `rliable`) | - | - | papers | https://arxiv.org/abs/1709.06560 , https://arxiv.org/abs/2108.13264 | V |
| Flageat and Cully, "Uncertain Quality-Diversity", 2023 | Treats fitness and descriptors as distributions; fixes a per-generation sampling budget; reports "a limitation of MAP-Elites in uncertain domains" and compares sampling-based variants | - | - | paper | https://arxiv.org/abs/2302.00463 | V |
| Justesen, Risi, Mouret, GECCO 2019; Hansen et al., IEEE TEVC 2009 | Adaptive sampling raises the number of evaluations per solution over time in MAP-Elites; UH-CMA-ES measures noise by rank changes on re-evaluation and reacts only when the ranking changes | - | - | papers | https://pure.itu.dk/da/publications/map-elites-for-noisy-domains-by-adaptive-sampling/ , https://researchportal.ip-paris.fr/en/publications/a-method-for-handling-uncertainty-in-evolutionary-optimization-wi/ | S |
| ADODG cases; Yang et al., ICAS 2018 | Known-answer shape optimisation: twist of a rectangular wing at C_L 0.375 must return near-elliptic loading and e about 1 (section C). ADODG Case 6 exists to "study the existence of multiple local minima" on the same wing | Steady, inviscid | VLM hybrid two orders of magnitude faster than Euler | benchmark | https://www.icas.org/icas_archive/ICAS2018/data/papers/ICAS2018_0440_paper.pdf , https://mdolab.engin.umich.edu/wiki/aerodynamic-design-optimization-workshop | V |

---

## What a planner can take from this (the surveyor's inference, not a finding)

Each statement names the rows it rests on. None of it was tested.

1. **Ptera Software is the only surveyed mid-fidelity tool with a published flapping-wing comparison**
   (one case, MAE 0.0291 N, digitised data), Python-defined motion, free wake and several wings in one
   run. It is inviscid with no leading-edge separation model. It can answer planform, tip-loss and
   attached-flow wake-interaction questions; it cannot answer stall or leading-edge-vortex questions.
   (A: Ptera rows, sources 1-4.)
2. **No surveyed tool has a published tandem flapping-wing validation.** Before a fore-hind wake result
   is used in design, the chosen tool should reproduce at least the sign and size of a measured phase
   effect: downstream foil 0.5-1.5x isolated (Boschitsch) or the 22 % / 16 % hover power figures
   (Usherwood and Lehmann, at Re about 100). (A, C.)
3. **The per-candidate cost of a UVLM is not known for this machine.** The published figures span
   "under a minute" (README, current code) to 22.5 min (2021 report, 5x18 panels, 3 cycles), and
   10 min for 200 steps in FLOWUnsteady. A timing on the project laptop at the panel count that
   converges is needed before it is placed inside or outside the evaluation loop. (A.)
4. **Viscous drag and separation stay with sectional data in every mid-fidelity tool** (FLOWUnsteady
   says so verbatim; UVLM codes have none). A realistic upgrade of the current strip model is "strip
   model for sectional forces, vortex wake for induced velocities", not a replacement. (A.)
5. **PyFR is out for a moving wing** (maintainer statement, 2025-04-22). nekRS and OpenFOAM can move
   meshes but need body-fitted or overset grids; WaterLily and FluidX3D avoid meshing and run on one
   GPU. (B.)
6. **Uniform-grid GPU solvers will be under-resolved at Re 3e4-2e5.** WaterLily's paper reports 3 grid
   points in the boundary layer at Re 3700 with 168 cells per diameter. A spot validation with these
   tools is only evidence after the same set-up reproduces a measured flapping case in this Re range,
   e.g. Anderson et al. (87 % at Re 40,000, optimum inside 0.2 < St < 0.5) or DeLaurier and Harris
   (18-50 %, AR 4, Re 30,000). (B source 7; C.)
7. **FluidX3D's licence allows personal and public-research use and forbids commercial use, military
   use and AI training on its source; results from a modified version oblige publication of the
   modified source.** If the built machine or its design could become a product, that needs the
   author's permission first. Its free-surface mode has no gas phase, so it cannot give air loads and
   water loads in one water-exit run. (B sources 12-13.)
8. **Two-phase water exit and entry with a moving body was found only in CPU codes** (OpenFOAM
   `overInterDyMFoam`, IBAMR; waLBerla claims free-surface and moving geometry). No flapping-wing
   water-exit simulation was found in any of them, and no cost figure. (B.)
9. **Order of magnitude of a GPU spot run: 1-25 USD and 1-7 h, under stated assumptions** (1e5 time
   steps; solver throughput measured without a moving body). The estimate is arithmetic on published
   throughput, not a measured run. (B cost section.)
10. **Known-answer checks available before any search is trusted:** (a) elliptic loading,
    C_Di = C_L^2/(pi AR), and the ADODG twist case (C_L 0.375, optimum e about 1, target C_Di about
    0.0073); (b) Theodorsen limits F: 1 -> 1/2; (c) Garrick's efficiency limit of 1/2 at high reduced
    frequency; (d) Anderson's 87 %. A strip model with one uniform inflow cannot pass (a) for a
    planform change, by construction. (C; the last sentence follows from the model description in
    the task.)
11. **Per-candidate reinforcement learning inside a body search is outside a laptop budget on the
    evidence found:** 2e10 interactions on 1,152 CPUs (DERL), or up to 20 h on 80 CPUs per run with
    failure on the hardest tasks (Evolution Gym). The options with published support for less compute
    are a shared body-conditioned controller (Schaff et al.), reuse of earlier candidates' data (Luck
    et al.), and cheap-model search with expensive-model re-scoring (Zheng et al.). (E.)
12. **Checks that a search is improving, each with a source:** a random-search arm with the same
    number of evaluations (Mania et al.); several seeds reported with intervals, not point estimates
    (Henderson; Agarwal); re-evaluation of archive members under fresh draws with a fixed sampling
    budget (Flageat and Cully; Justesen et al.); a held-out re-score of elites in the higher-fidelity
    model, with the disparity treated as an objective (Koos et al., about ten transfers). Maintained
    libraries for the search itself, with 2026 releases: pyribs 0.12.1, pycma 4.5.0, SMT 2.15.1 (multi-
    fidelity kriging), pymoo 0.6.2. (D, E.)

---

## Not found

What was tried is in brackets.

- **Tandem flapping-wing validation of any mid-fidelity code** (Ptera, SHARPy, DUST, FLOWUnsteady,
  VortexLattice.jl). [Searches on UVLM + tandem + flapping, 2024-2026; tool READMEs and docs.] One
  Spanish-language paper on a hybrid UVLM-VPM with tandem flapping wings appeared as a snippet only
  (https://amcaonline.org.ar/ojs/index.php/mc/article/download/6650/6712); no open code was linked.
- **DUST**: element list, limits, kinematics input, latest version and date from a primary page.
  [dust.polimi.it; GitLab project page; raw README; GitLab API returned no JSON; Tugnoli 2021 PDF
  returned 403.]
- **flow5** licence and capabilities; **xflr5** official page. [flow5.tech and xflr5.tech returned
  empty pages; one xflr5 URL returned 404.]
- **VSPAERO**: whether a release after 2021 can prescribe wing flapping. [Forum search; one thread read.]
- **SHARPy** flapping-wing use, run times, viscous corrections. [README, capabilities page.]
- **AeroSandbox** lifting-line and AeroBuildup details, and its optimiser back end, from a fetched
  page. [README only.]
- **Run time of a few flapping cycles** for SHARPy, DUST, VortexLattice.jl, VSPAERO.
- **Core-hours or wall time of any 3-D flapping-wing CFD run at Re 3e4-2e5**, in any solver.
  [Searches on overset, immersed-boundary and LES flapping at bird scale; Calado et al. and Poletti et
  al. report no cost; Zhang and Huang 2023 is paywalled.]
- **FluidX3D**: force accuracy figures, wall treatment, slowdown from moving boundaries, any flapping
  example with grid and time. [README read in two parts, licence file.]
- **WaterLily**: free-surface capability, turbulence model statement, a flapping-wing force
  validation. [README; paper pages 6-12.]
- **IBAMR, Palabos, waLBerla, nekRS**: any flapping-wing case with cost. Palabos GPU support.
- **Hetzner** price for a 32-48 dedicated-vCPU server (not on the page fetched).
- **Anderson et al. 1998**: foil section, aspect ratio, the exact parameter set at 87 %. [MIT DSpace
  PDF returned HTTP 429; abstract page and the 2004 review were read.]
- **Read et al. 2003**: peak efficiency and its Strouhal number, thrust-coefficient values. The
  figures often quoted from memory for this paper were not confirmed and are not used here.
  [Three searches; publisher page not reachable.]
- **Heathcote, Wang, Gursul 2008**: Reynolds number, aspect ratio, section, thrust-coefficient values.
  [Bath portal abstract only.]
- **Hubel and Tropea**: primary text. [JEB URL resolved to a different article; Springer redirected to
  a login.] Numbers above are snippets.
- **Warkentin and DeLaurier 2007**: primary text and the phase and spacing values.
- **Boschitsch et al. 2014**: Reynolds number, Strouhal range; a later paper quotes "up to 160 % thrust
  and 150 % efficiency", which does not match the abstract's 1.5x and was not resolved.
- **Glauert's delta table for rectangular wings**, Theodorsen table values, Wagner / R. T. Jones
  coefficients, and Garrick's efficiency expression from a fetched page. The values in section C marked
  I are from memory.
- **An open, tabulated force data set for a flapping wing at Re 3e4-2e5.** [Zenodo, 4TU, Dryad
  searches.]
- **Robotic flapper (Dickinson-type) measurements**: not re-fetched in this survey.
- **BoTorch multi-fidelity documentation** (tutorial URL returned 404); QDax, OpenMDAO, Ax feature
  descriptions beyond their one-line summaries.
- **A maintained open rib-and-spar generator for printable wings** other than cq-uav (seen as a snippet,
  repository not read). **Laptop run times for 3-D topology optimisation** in pytopo3d, FEniTop, DL4TO.
- **Budgets** for Schaff et al., Luck et al., Zheng et al., Ghommem et al. (abstracts or snippets only).
- **A published co-design of body shape and flapping gait under a laptop-scale budget.** Not found.
