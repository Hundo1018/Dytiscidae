# A10 — Literature: what learning and search methods have actually produced, and at what budget

Report for the triphibian diagnosis (`experiments/triphibian_diagnosis/PREREG.md`).
Written 2026-11-10 from web sources opened in this session (WebSearch / WebFetch).
No code was run and nothing else in the repo was touched.

## Tagging

- **[V]** VERIFIED: the number or sentence was read on the page at the URL given
  (arXiv abstract, ar5iv / arXiv HTML full text, publisher or project page) in this
  session. Quoted text is in quotation marks.
- **[I]** INFERRED: arithmetic or reading by this report from [V] numbers (the
  arithmetic is shown), or a judgement.
- **[U]** UNKNOWN: not found, not reported, or not opened. An empty cell says so.

Full-text pages were read through ar5iv (`ar5iv.labs.arxiv.org/html/<id>`, renders an
arXiv version that may be v1 rather than the camera-ready) or arXiv's native HTML.
Where only the abstract was reachable, budget cells read "not in abstract [U]".
Page reading was done by a fetch tool that summarises the page against a question;
quotes are what it returned as verbatim, and two pages rendered digits doubled
(flagged where it matters). Venues are given as listed on the arXiv record or as
commonly cited; the year is the arXiv submission year unless a venue year is
stated. Venue names were not individually verified and carry no tag.

Pages that could not be opened (so their numbers are [U]): Nature's own site for
Swift and M4 (read at PMC instead), the Won et al. 2017 dragon PDF, the Xu et al.
2019 hybrid-UAV PDF and the RoboGrammar PDF (over the fetch size limit), NerveNet on
OpenReview (browser check), the original PGA-MAP-Elites PDF on HAL (bot wall, not
bypassed), Justesen et al. 2019 (HTTP 403), Amorpheus full text (ar5iv conversion
failed).

The project's own budget, for comparison throughout (from the task brief, not
re-measured here): ~300-900 generations x 16 bodies = ~5,000-15,000 body
evaluations, ~1e6-1e7 control steps in total, spread over hundreds of changing
bodies, three media and six kinds of medium crossing, one shared small PPO policy.

---

## A. Sample budgets that worked (single body or small fixed body set)

Column "per (body, skill)" normalises the budget to environment steps per body and
per skill, the quantity the project spreads across hundreds of bodies x 3 media x 6
crossings.

| method | paper, year, URL | what it needs | budget reported | per (body, skill) | result | known failure modes |
|---|---|---|---|---|---|---|
| Massively parallel PPO, legged | Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL 2021, https://arxiv.org/abs/2109.11978 (read at ar5iv) | GPU sim (Isaac Gym), terrain curriculum, one fixed robot | "4096 robots and a batch size of 98304" (4096x24) [V]; "1500 policy updates in under 20 minutes" [V]; flat terrain "in under four minutes", uneven "in twenty minutes" [V]; RTX A6000 [V]. Total samples not stated; 98,304 x 1,500 = 1.47e8 [I] | ~1.5e8 steps for one body, one skill (rough-terrain walking) [I] | sim-to-real ANYmal locomotion [V] | "the algorithm struggles when we provide fewer than 25 consecutive steps" per env per update [V]; performance "drops sharply" when the number of robots is too high (batch too small per env) and decays slowly below a threshold; recommends "2048 to 4096 robots with a batch size of ≈100k or ≈200k" [V]; "it is essential to first train the policy on less challenging terrain" [V]; gait artifacts ("a dragging leg"), Cassie needed an extra single-foot reward "to achieve a walking gait" [V] |
| Teacher-student (privileged teacher, DAgger student) | Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning quadrupedal locomotion over challenging terrain", Science Robotics 2020, https://arxiv.org/abs/2010.11251 (ar5iv) | sim with privileged state (terrain heights, contacts, friction, disturbances: "noiseless information that we receive directly from a physics engine") [V]; adaptive terrain curriculum by particle filter keeping terrains at 0.5-0.9 traversability [V] | teacher "≈12 hrs", student "≈4 hrs" on one RTX 2080 desktop (Table S1) [V]; sample counts not found on the page [U] | one body, one gait (trot) [V] | blind ANYmal on rough natural terrain zero-shot [V] | "the presented controller only exhibits the trot gait"; "If the machine is commanded to walk off a cliff, it will" (blind) [V] |
| RMA: base policy + adaptation module | Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021, https://arxiv.org/abs/2107.04034 (ar5iv) | sim (RaiSim) with randomised physics; privileged env-factor encoder, then a history-based adaptation module trained by regression [V] | base: "PPO ... for 15,000 iterations each of which uses batch size of 80,000"; "Training takes roughly 24 hours on an ordinary desktop machine, with 1 GPU"; "it simulates 1.2 billion steps" [V]. Adaptation module: 1000 iterations; ar5iv renders the time and steps with doubled digits ("33 hours", "8080 million"), probably 3 h and 80 M [I] | 1.2e9 steps, one body (Unitree A1), one skill (walking) [V] | real A1 on many terrains [V] | stairs 70%, pebble piles 80%, oily patch 90% success; larger perturbations "sometimes lead to failures" [V] |
| Per-skill RL + soft-dynamics curriculum, then DAgger distillation into one vision policy | Zhuang et al., "Robot Parkour Learning", CoRL 2023, https://arxiv.org/abs/2309.05665 (ar5iv) | resettable GPU sim (Isaac Gym, 4096 envs) [V]; obstacle penetration allowed early ("soft dynamics constraints") then hard dynamics [V]; per-obstacle expert supervises the student [V] | per specialised skill: 12 h soft-dynamics + ~6 h hard-dynamics fine-tune on one RTX 3090 [V]; distillation on 4 machines (3 collect, 1 trains) [V]; total steps not stated [U] | 5 skills, 1 body; per-skill steps [U] | climb, leap, crawl, tilt, run on real A1/Go1 [V] | "Both RND and Oracles w/o Soft Dyn cannot make any learning progress on climbing and leaping" [V] (an exploration bonus did not replace the curriculum); environments must be "manually constructed" per skill [V] |
| Hierarchical: skill catalogue + RL navigation module selecting skills | Hoeller et al., "ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots", Science Robotics 2024, https://arxiv.org/abs/2306.14874 (ar5iv) | 5 RL skills (walk, jump, climb up, climb down, crouch) each trained on single obstacles with a curriculum; frozen skills; navigation policy at 5 Hz outputs a categorical skill choice + position/heading/time command; 4096 agents in Isaac Gym [V] | iterations and wall time not stated [U] | 5 skills, 1 body [V] | real ANYmal parkour course [V] | navigation "requires many iterations to converge"; "Without this step [the curriculum], the robot struggles to discover the correct behaviors and gets stuck in front of larger obstacles"; suggests pre-training navigation "using expert demonstrations"; eight networks to train [V] |
| Unified waypoint-heading reward, privileged phase then depth DAgger | Cheng, Shi, Agarwal, Pathak, "Extreme Parkour with Legged Robots", ICRA 2024, https://arxiv.org/abs/2309.14341 (ar5iv) | privileged scandots phase, DAgger to depth [V] | "trained on a single 3090 GPU in less than 20 hours" [V]; env counts, steps not stated [U] | 1 body [V] | high jumps, long jumps, handstand on a low-cost robot [V] | no limitations section [V] |
| PPO, quadrotor racing with residual sim models | Kaufmann et al., "Champion-level drone racing using deep reinforcement learning" (Swift), Nature 2023, https://www.nature.com/articles/s41586-023-06419-4 (read at PMC10468397) | sim + data-driven residual models of perception and dynamics fit on "three full rollouts in the real world, corresponding to approximately 50 s of flight time" [V] | "100 agents in parallel"; "1 × 10^8 environment interactions"; "50 min on a workstation (i9 12900K, RTX 3090)"; fine-tune "2 × 10^7"; policy and value 2x128 MLPs [V] (exponents lost in page text) | 1e8 steps, one body, one track [V] | beat human world champions [V] | "not trained to recover after a crash"; assumes the environment's appearance matches training [V] |
| Off-policy TD3, quadrotor RPM control | Eschmann, Albani, Loianno, "Learning to Fly in Seconds", RA-L 2024, https://arxiv.org/abs/2311.13081 (arXiv HTML) | fast simulator ("about 5 months of flight per second on a laptop GPU"); scheduled reward-penalty and exploration-noise curriculum every 100,000 steps [V] | "after about 300 000 steps (total number of interactions with the environment)"; "18 seconds of training on a consumer-grade laptop" [V] | 3e5 steps, one body, one skill (hover / position control) [V] | sim-to-real on a Crazyflie [V] | sim return showed only a "loose correlation" with real performance [V] |
| PPO, bird-inspired flapping wing in MuJoCo with stateless fluid model (closest analogue to this project's physics) | Cai, Sangli, Kim, Sreenath, "Learning-based Trajectory Tracking for Bird-inspired Flapping-Wing Robots", arXiv 2024, https://arxiv.org/abs/2411.15130 (arXiv HTML v1) | MuJoCo "state-less fluid force models" (ellipsoid wings/tail); PD-tracked joint targets at 50 Hz with a 7 Hz low-pass filter on flapping; staged curriculum [V] | "256 parallel environments"; ~8 M steps per curriculum stage; "a full controller trained in about 30 million steps"; "around 5 hours" on an Intel i7-11800H CPU [V] | 3e7 steps, one fixed body, one skill family (trajectory tracking incl. flap/glide mode switches) [V] | stable flight, spontaneous switching between flight modes, trajectory tracking in sim [V] | simulation only; hand-tuned fluid coefficients "may not match the physical model"; most sensitive to the Kutta lift coefficient; tracking degrades on dynamically infeasible targets [V] |
| Residual DDPG on top of a model-based flapping controller | Fei, Tu, Zhang, Deng, "Learning Extreme Hummingbird Maneuvers on Flapping Wing Robots", ICRA 2019, https://arxiv.org/abs/1902.09626 (ar5iv) | sim with "quasi-steady flapping-wing aerodynamics, actuator dynamics, contact dynamics, joint frictions", system-identified, 10 kHz sim / 500 Hz control [V]; the RL action is *added* to the model-based controller's inputs and is active only on a trigger [V] | epochs of 10,000 samples; optimised policy after roughly 500 episodes [V]; at >= 10 episodes per epoch that is <= ~5e5 samples [I]; wall time not reported [U] | ~5e5 samples, one body, one manoeuvre [I] | sim-to-real 12 g two-actuator hummingbird robot escape manoeuvre [V] | yaw authority weakest axis; robot takes ~20 wingbeats against the bird's ~10 [V] |
| Deep RL (DQL with CMA-inspired updates), example-guided flapping flight | Won, Park, Kim, Lee, "How to Train Your Dragon: Example-Guided Control of Flapping Flight", SIGGRAPH Asia 2017, https://mrl.snu.ac.kr/research/ProjectDragon/Dragon.htm | user keyframes guide learning; aerodynamically simulated creature [V] | not on the project page; PDF not retrievable (connection reset, 12.2 MB) [U] | [U] | flapping-flight skills for simulated winged creatures [V] | [U] |
| PPO for soft-bodied swimmers (FEM + simplified hydrodynamics) | Min, Won, Lee, Park, Lee, "SoftCon: Simulation and Control of Soft-Bodied Animals with Biomimetic Actuators", SIGGRAPH Asia 2019, https://mrl.snu.ac.kr/publications/ProjectSoftCon/SoftCon.pdf | muscle-excitation pattern generator as the action space (raw per-muscle control "produce[s] noisy, unstructured movements"); early termination "[Peng et al. 2018a]"; each episode's initial state "picked randomly from the pool of states in previous episodes" [V] | "approximately 4 to 48 hours depending on the number of muscles and tetrahedra"; stingray 4 h, octopus 48 h "until its learning curve plateaus on a single PC" (i7-7700, CPU only) [V]; step counts not stated [U] | one body per training run [V] | lamprey, starfish, octopus, stingray, cuttlefish swim and steer [V] | a CPG-parameterised octopus "is not robust even after two weeks of learning"; "small tweaking of the body shapes and muscle embedding substantially affect the level of motor ability" [V] |
| Recurrent deep RL in Navier-Stokes (2D/3D) | Verma, Novati, Koumoutsakos, "Efficient collective swimming by harnessing vortices through deep reinforcement learning", PNAS 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC6003313/ | high-fidelity flow solver (wavelet-adapted vortex methods), LSTM value function [V] | main text gives only "first 10,000 transitions" as an early-training marker; totals not in the main text [U] | one body, one skill [V] | follower fish harvests energy from the leader's wake [V] | results mostly with "a steadily swimming leader"; delayed action effects [V] |
| One PPO policy for land and water with a medium flag | Tian, Fu, Ning, Pey, Ijspeert, "Learning Whole-Body Control for a Salamander Robot", arXiv 2026, https://arxiv.org/abs/2603.16683 (arXiv HTML v1) | MuJoCo contact + per-segment buoyancy and linear/quadratic drag; binary medium indicator appended to the observation; the mode-dependent dynamics are not used for model-based control or explicit switching [V] | 8192 envs; 3×10^8 steps, about 1.5 hours on a single NVIDIA RTX 4070 (locomotion); transition task reuses the hyperparameters, no separate budget [V] | 3e8 steps, one body, two media [V] | real land walking; land-to-water transition shown in simulation only [V] | no transition success rate reported, one qualitative demo; water model simplified [V] |
| DDPG / SAC, hybrid aerial-underwater vehicle with medium transition | Grando et al., "Deep Reinforcement Learning for Mapless Navigation of a Hybrid Aerial Underwater Vehicle with Medium Transition", ICRA 2021, https://arxiv.org/abs/2103.12883 (ar5iv) | Gazebo + RotorS + UUV Simulator; the policy outputs *velocity* commands to a geometric tracking controller, not motor commands [V] | 1,000 episodes (scenario 1), 2,500 (scenario 2), <= 500 steps each, replay buffer 50,000 [V]; <= 1.25e6 steps in scenario 2 [I] | ~1e6 steps, one body, navigation over a stabilising controller [I] | 96-97% (DDPG) / 72-75% (SAC) success in the open tank; with obstacles 54% / 37% air-to-water [V] | simulation only; 100-trial evaluation, high timing variance for SAC [V] |
| RL controller for hybrid copter/fixed-wing UAVs | Xu, Du, Foshey, Li, Zhu, Schulz, Matusik, "Learning to Fly: Computational Controller Design for Hybrid UAVs with Reinforcement Learning", SIGGRAPH 2019, https://learn2fly.csail.mit.edu/ | the controller "does not distinguish among flying modes" [V] | PDF over the fetch size limit; budget not on the project page [U] | [U] | real hybrid UAVs fly "without any additional parameter tuning" [V] | [U] |

## B. Chains of rare sub-goals / hard exploration

| method | paper, year, URL | what it needs | budget reported | result | known failure modes |
|---|---|---|---|---|---|
| Go-Explore, 2019 version (restore emulator state; robustify by Backward Algorithm) | Ecoffet, Huizinga, Lehman, Stanley, Clune, "Go-Explore: a New Approach for Hard-Exploration Problems", arXiv 2019, https://arxiv.org/abs/1901.10995 (ar5iv) | phase 1 assumes a "deterministic and resettable" simulator; returns to cells by saving and restoring emulator state, which cut simulated steps by at least an order of magnitude; phase 2 imitates the found trajectories with Salimans & Chen's Backward Algorithm [V] | Montezuma with domain knowledge: level 1 solved in a mean 57.6 M frames, level 3 ~173 M; downscaled cells 1.2 B frames (level 1 at mean 640 M); robustification ~4.59 B frames (Montezuma), ~8.20 B (Pitfall) [V] | Montezuma mean 43,763 (no domain knowledge), 666,474 (domain knowledge); Pitfall 59,494 [V] | relies on determinism during exploration; applying it to stochastic training left open [V] |
| Go-Explore, 2021 Nature version (adds policy-based return) | Ecoffet, Huizinga, Lehman, Stanley, Clune, "First return, then explore", Nature 2021, https://arxiv.org/abs/2004.12919 (ar5iv) | return "by restoring simulator state or by running a goal-conditioned policy"; restoring is "not a requirement" [V]; a cell representation (downscaled frames or domain features; for robotics: gripper voxel, grasp, object-in-shelf, door/latch) [V] | Atari exploration 1-5 B frames, robustification 10-15 B frames [V]. Robotics (Fetch pick-and-place onto 4 shelves, sparse reward): exploration 20 M frames, robustification until >98.5% success for 150 iterations; 198/200 runs succeeded [V] | PPO found no reward in 1 B frames, count-based intrinsic motivation none in 2 B frames on the same robot task [V] | named failure modes of other explorers: "detachment" and "derailment" [V]. Conflict with the 2019 version, shown: 2019 required restore; 2021 makes it optional, but its headline Atari table is "with restoration of simulator state"; the policy-based variant (no restore) reached 97,728 on Montezuma after 12 B frames and 20,093 on Pitfall after 10 B [V] |
| Backward Algorithm from one demonstration | Salimans, Chen, "Learning Montezuma's Revenge from a Single Demonstration", arXiv 2018, https://arxiv.org/abs/1812.03381 (ar5iv) | one demonstration with full game states; an emulator that can be reset to any demonstration state; start point moves backward on success [V] | ~50 B frames, "128 GPUs over a period of 2 weeks" [V]; claims a sequence of N actions is learned "in a time that is quadratic in N, rather than exponential" [V] | 74,500 (demonstration 71,500) [V] | sticky actions drop the score to ~10,000; did not train the full curriculum on Gravitar and Pitfall; hyperparameter-sensitive [V] |
| Backplay | Resnick et al., "Backplay: 'Man muss immer umkehren'", arXiv 2018, https://arxiv.org/abs/1807.06919 (ar5iv) | one "good enough" demonstration of states (no actions), resettable environment; start window slides from the end toward the true start; schedule hand-tuned, too fast hurts [V] | mazes 3,500 epochs x 102,400 interactions for standard RL; Pommerman 72 h, ~50 M frames [V] | standard PPO "has immense trouble learning" sparse mazes; Backplay within five steps of optimal on 94-100% of mazes; beats the demonstrator in sparse Pommerman [V]; behavioural cloning imitated well but could not beat the expert [V] | "not a universal strategy"; one trajectory covers little of a randomised start distribution; suboptimal demos slow learning; needs arbitrary resets [V] |
| Reverse curriculum generation | Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017, https://arxiv.org/abs/1707.05300 (ar5iv) | "We can arbitrarily reset the agent into any start state"; one goal state; binary reward, no shaping, no demos; starts kept where success is in [0.1, 0.9]; new starts by 50-step Brownian motion from good starts; replay of old starts [V] | TRPO, 5 iterations x 50,000 timesteps per curriculum iteration; horizon 500 (2,000 ant maze); total iterations not stated [U] | uniform start sampling succeeds only near the goal (~10% ring-on-peg, ~2% key insertion); RCG solves from distant starts; asymmetric self-play fails on >40% of maze starts; adding a distance reward considerably hurt key insertion [V] | grows uniformly from the goal (needs a prior if the target start set is far); zero velocities only; Brownian approximation costs vs oracle [V] |
| Reference state initialisation (RSI) + early termination (ET) | Peng, Abbeel, Levine, van de Panne, "DeepMimic", SIGGRAPH 2018, https://arxiv.org/abs/1804.02717 (ar5iv) | reference motion clips; episodes start from states sampled along the reference (RSI); terminate on torso/head contact (ET) [V] | 41e6-191e6 samples per skill policy (Table 2), up to 466e6 for a task policy (Table 3); "several days per skill" [V] | normalised return RSI+ET / ET only / RSI only: backflip 0.791 / 0.730 / 0.379; walk 0.980 / 0.981 / 0.974; without RSI the backflip "never learns to perform a full mid-air flip" [V] | ablations mostly single runs; phase variable tied to time; multi-clip only for small clip sets; manual PD gains per morphology [V]. Composite of separately trained skills chosen by value functions works without extra training [V] |
| Transition policies with proximity reward | Lee, Sun, Somasundaram, Hu, Lim, "Composing Complex Skills by Learning Transition Policies", ICLR 2019, https://youngwoon.github.io/transition (full text read at https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2019/composing_complex_skills_by_learning_transition_policies__feb7d78e.txt) | pre-trained primitives (TRPO); a (rule-based) meta-policy; primitives that report success/failure; per-primitive proximity predictor trained from success/failure states gives a dense reward for reaching the next primitive's initiation set [V] | primitives up to 10,000 iterations x 10,000-step rollouts; 1,000 seed episodes per primitive; transitions capped at 100 steps; flat TRPO/PPO baselines given 5x longer training [V]; total steps not stated [U] | e.g. Serve 0.92 vs TRPO 0.32 / PPO 0.00; Patrol 3.33 vs 1.37 / 1.53; chaining without transitions much worse (catching 1.00 vs 4.97) [V]; flat TRPO with 8 engineered dense rewards beat it on Hurdle (4.13 vs 3.14) [V] | why chaining fails: a skill "may be executed in starting states not considered during training" [V]; assumes random exploration can find a successful transition [V] |
| Adversarial skill chaining (T-STAR) | Lee, Lim, Anandkumar, Zhu, "Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization", CoRL 2021, https://arxiv.org/abs/2111.07999 (ar5iv) | given sub-task decomposition, per-sub-task initialisation, success signal per sub-task, 200 demos per part [V] | 20-25 M steps per sub-task pre-training, +100 M chaining, 200 M total; baselines 200 M [V] | task progress 0.90 vs 0.63 (table_lack), 0.89 vs 0.77 (chair_ingolf) vs plain policy sequencing; BC, GAIL, SPiRL, PPO performed poorly [V] | "naively executing one policy after another would fail when the subtask policy encounters a starting state never seen during training" [V]; needs sub-task supervision; adversarial training unstable (best checkpoints reported) [V] |
| ALP-GMM teacher (absolute learning progress) | Portelas, Colas, Hofmann, Oudeyer, "Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments", CoRL 2019, https://arxiv.org/abs/1910.07224 (ar5iv) | bounded continuous task-parameter space; only episodic returns; no expert knowledge [V] | 20 M steps (Stump Tracks), 80 M (Hexagon Tracks) per run [V] | default walkers 39.6% vs random 18.2% tracks mastered; short walkers 14.9% vs ~0%; Hexagon ~80% vs hand-crafted oracle 68% [V] | quadrupedal walkers: random sampling already near 100% (curriculum adds nothing when the body is capable) [V]; degrades as the share of infeasible tasks grows [V] |
| POET (paired open-ended environment/agent evolution) | Wang, Lehman, Clune, Stanley, "Paired Open-Ended Trailblazer (POET)", arXiv 2019, https://arxiv.org/abs/1901.01753 (ar5iv) | parameterised environment generator; agent per environment (ES); transfer between pairs; minimal-criterion filter (parent scores 50-300) [V] | up to 25,200 iterations, ~20 active environments, 512 ES samples per step, ~10 days on 256 CPU cores per run [V] | direct ES on POET-made environments stalled at 13.6-39.6 (threshold 230); same-compute direct-path curriculum failed the hardest; without transfer no extremely challenging environment solved [V] | fixed body; one 2-D domain; "cannot guarantee reaching a particular preconceived target" [V] |
| Prioritized Level Replay | Jiang, Grefenstette, Rocktäschel, "Prioritized Level Replay", ICML 2021, https://arxiv.org/abs/2010.03934 (ar5iv) | levels resettable by identifier/seed; score = average magnitude of GAE (L1 value loss) mixed with staleness [V] | Procgen easy 25 M steps, hard 200 M [V] | significant test gains on 10 of 16 Procgen games; +76% mean normalised test return with UCB-DrAC [V] | score-only or staleness-only sampling worse than uniform; reduced sample efficiency on two games [V] |
| Model-based multi-modal robot (rolling, flying, balancing; planner picks modes) | Sihite, Kalantari, Nemovi, Ramezani, Gharib, "Multi-Modal Mobility Morphobot (M4) with appendage repurposing for locomotion plasticity enhancement", Nature Communications 2023, https://pmc.ncbi.nlm.nih.gov/articles/PMC10300070/ | Euler-Lagrange model; direct collocation for balancing / wheel-assisted incline; real-time optimisation-based feedback; mode choice by a multi-modal PRM + A* on energy cost [V] | no learning [V] | autonomous switching between ground (UGV) and flight (UAS) modes only; pond ground-to-air test "teleoperated (not autonomous)" [V] | mode-switch decision "needs further developments to be applicable in more complex scenarios"; no dynamic legged gaits [V] |
| Model-based walk + fly biped | Kim, Spieler, Lupu, Ramezani, Chung, "A bipedal walking robot that can fly, slackline, and skateboard" (LEONARDO), Science Robotics 2021, https://www.aphms.caltech.edu/news/leonardo-the-bipedal-robot-can-ride-a-skateboard-and-walk-a-slackline (paper DOI 10.1126/scirobotics.abf8136, not opened) | thrusters "controlled synchronously with leg joints" [V, news page] | no learning in the current controller (a learned landing controller is future work) [V, news page] | walks, flies, slacklines, skateboards; "can choose to use either walking or flying, or blend the two" [V, news page] | take-off/landing mechanism not described on the page [U] |
| RL walk-fly transition with adversarial motion priors | L'Erario et al., "Learning to Walk and Fly with Adversarial Motion Priors" (iRonCub), IROS 2024, https://arxiv.org/abs/2309.12784 (arXiv HTML v4) | Isaac Gym, 4096 agents, PPO at 60 Hz; walking prior from retargeted mocap, flying prior from trajectory optimisation; thrust penalty as an energy proxy [V] | about 2 hours on a Quadro RTX 6000; steps not reported [V/U] | walk-to-fly switching emerges without a planner (sim); ablations (paraphrased): with no motion prior no particular behaviour and more thrust; walking-only prior walked but did not fly; flying-only prior did not walk [V] | sim only; "Tuning the thrust penalization is not trivial" (too high: stays at the edge; too low: flies when it should walk); no numeric success rates [V] |
| Wheeled-legged: one learned locomotion policy that chooses walking vs driving, under a learned navigation policy | Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter, "Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots", Science Robotics 2024, https://arxiv.org/abs/2405.01792 (arXiv HTML v1) | privileged PPO teacher, DAgger student with an RNN encoder; low-level policy "first trained with known gait patterns like trot, pace, and static walk", then trained alternately with the high-level policy; adaptive terrain curriculum by a success-rate band [V] | teacher batch 500,000, episodes 10 s at dt 0.02; high-level batch 150,000; env counts, iterations, wall time not found [V/U] | "capable of making decisions regarding the gait and transitioning between walking and driving modes", learned "without handcrafted heuristics like CPG or predefined gait sequences"; 8.3 km autonomous mission [V] | high-level policy stuck when its memory filled; memoryless variant fell into local minima [V] |
| One continuous energy-aware policy over wheels, tilt servos, propellers | Li et al., "Learning Energy-Efficient Air-Ground Actuation for Hybrid Robots on Stair-Like Terrain", arXiv 2026, https://arxiv.org/abs/2603.26687 | hardware-calibrated power model in the reward; terrain curriculum; terminal reward for settled arrival [V] | not in abstract [U] | real prototype clears two 6 cm steps in 8/10 trials; ~27% less power than the best fixed thrust allocation (sim) [V] | heading control, contact robustness, recovery listed as open [V] |

## C. One controller for many bodies

Column "bodies fixed during training?" separates training on a fixed set of
morphologies from training while the bodies change underneath the policy.

| method | paper, year, URL | what it needs | budget reported | bodies fixed during training? | result | known failure modes |
|---|---|---|---|---|---|---|
| NerveNet (GNN policy over the body graph) | Wang, Liao, Ba, Fidler, "NerveNet: Learning Structured Policy with Graph Neural Networks", ICLR 2018, https://www.cs.toronto.edu/~tingwuwang/nervenet.html | body graph as the network structure [V] | not on the project page [U] | fixed bodies; transfer tested after training [V] | "the only model able to walk in the zero-shot evaluations of centipedes"; fine-tuning "significantly decreases the number of episodes" for size/disability transfer [V] | [U] (OpenReview blocked by a browser check) |
| SMP (shared modular policies, message passing) | Huang, Mordatch, Pathak, "One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control", ICML 2020, https://arxiv.org/abs/2007.04976 (ar5iv) | per-limb shared module, both-way message passing; TD3 [V] | total steps per agent not given; 1 M timesteps for the root-placement comparison [V] | fixed: shapes "static and not dynamic" [V] | 12 walkers, 8 humanoids, 15 cheetahs, 3 hoppers (planar); zero-shot to held-out 20% variants [V] | the plain multi-task baseline "fails to perform well in any environment"; one-way message passing fails on 2-3 limb walkers; cheetahs excluded from cross-category training [V] |
| Amorpheus (transformer, ignores the graph for message passing) | Kurin, Igl, Rocktäschel, Boehmer, Whiteson, "My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control", ICLR 2021, https://arxiv.org/abs/2010.01856 (abstract; ar5iv conversion failed) | transformer over limbs [V] | not in abstract [U] | fixed (SMP benchmark) [I] | "morphological information encoded in the graph does not improve their performance"; Amorpheus beats GNNs that use it [V] | [U] |
| MetaMorph (transformer over limb tokens) | Gupta, Fan, Ganguli, Fei-Fei, "MetaMorph: Learning Universal Controllers with Transformers", ICLR 2022, https://arxiv.org/abs/2203.11931 (ar5iv) | 100 training robots from DERL's UNIMAL space, chosen from robots already "optimized for the task of locomotion"; dynamic replay balancing by episode length [V] | 1×10^8 total timesteps, 16 workers x 32 envs, ~1.5 GPU days [V]; i.e. ~1e6 steps per robot [I]; claimed ~5x more sample-efficient than 100 independent MLPs [V] | fixed set of 100 [V] | matches per-robot MLPs on flat terrain and obstacles, slightly below on variable terrain; zero-shot to kinematic/dynamic variants; 2-3x faster fine-tuning on 100 new robots [V] | without dynamic balancing about 10-15% of robots fail to learn a good policy [V, paraphrased]; "performance drop on zero-shot generalization to new morphologies"; reversing node order drops ~75%; robot creation and controller pre-training are separate stages [V] |
| URMA (embodiment-aware attention encoder, multi-robot PPO) | Bohlinger et al., "One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion", CoRL 2024, https://arxiv.org/abs/2409.06366 (arXiv HTML v4) | joint descriptions per robot; randomised mass, joint limits, damping, control range [V] | 100M steps per robot (1.6B total), 48 MuJoCo envs on CPU, ~2.5 days, 1 GPU [V] | fixed set of 16 robots (4 more held out) [V] | zero-shot to the real A1, Silver Badger and unseen Honey Badger; a policy trained on A1 alone is "distinctly worse" at 100 M steps and catches up only at ~1.6 B [V] | no meaningful zero-shot on H1 and Talos humanoids; far out-of-distribution bodies remain problematic; shuffling joint descriptions makes A1 fall in one or two steps [V, paraphrased] |
| Embodiment scaling (per-body PPO experts, then distillation into URMA) | Ai et al., "Towards Embodiment Scaling Laws in Robot Locomotion", CoRL 2025, https://arxiv.org/abs/2505.05753 (arXiv HTML v2) | ~1,000 procedurally generated bodies (GENBOT-1K: 348 humanoids, 332 quadrupeds, 332 hexapods); one PPO expert per body, then behaviour cloning into one student [V] | experts: 4,096 envs each, batch 98,304, 17,500-42,500 iterations; all experts "approximately 5 days on 160 NVIDIA RTX 4090/3090 GPUs", over 2 trillion simulation steps; distillation ~2 B samples, one week on one H100 [V]; ~2e9 steps per body [I] | fixed subsets of a fixed generated set [V] | reward on unseen bodies roughly doubles from 5% to 100% of training bodies; quadrupeds/hexapods saturate near 100 bodies; zero-shot to real Go2 and H1 [V] | the authors chose two-stage (experts then distil) for "tractable memory usage and stable training dynamics" [V]; flat terrain only; H1 transfer weaker [V] |
| Body Transformer | Sferrazza, Huang, Liu, Lee, Abbeel, "Body Transformer: Leveraging Robot Embodiment for Policy Learning", arXiv 2024, https://arxiv.org/abs/2408.06316 (arXiv HTML) | masked attention over the body graph [V] | RL: 2048 envs, batch 8192; steps not reported [V/U] | single morphology: "our BoT is designed for tasks with a single morphology" [V] | beats MLP and vanilla transformer on MoCapAct imitation and 4 Isaac Gym RL tasks [V] | not a cross-embodiment method [V] |
| Design-conditioned shared policy over a *changing* design distribution | Schaff, Yunis, Chakrabarti, Walter, "Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning", ICRA 2019, https://arxiv.org/abs/1801.01432 (ar5iv) | policy receives the design parameters; Gaussian-mixture design distribution (8 components) updated by policy gradient; worst components pruned; same topology, parameters only [V] | Hopper/Walker "11 B environment timesteps" as rendered (probably 1 B) [V/I]; design updates start after 100 M (200 M for Ant); Ant 1.5 B; last 100 M with the design fixed to the mean while the policy fine-tunes [V] | **no**: designs change during training, but within one topology [V] | beats default Roboschool designs on level terrain for all three; on the incline for Hopper and Walker [V] | Ant on the incline: seven of eight seeds stuck balancing without moving forward until the upright reward was halved [V]; design variance narrows over training [V] |
| Population + individual SAC networks across designs, Q-function as design surrogate | Luck, Ben Amor, Calandra, "Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning", CoRL 2019, https://arxiv.org/abs/1911.06832 (ar5iv) | population actor/critic trained on all designs; individual networks per design initialised from it (~10% of batches from the population buffer); design chosen by maximising the critic [V] | 5 initial designs x 300 episodes; later designs 100 episodes (200 Daisy hexapod); ~50 designs vs 24,177 for the simulation-based baseline [V] | **no**: a new design every 100 episodes [V] | reaches and exceeds the simulation-based baseline on Half-Cheetah; beats OpenAI-ES and random design choice [V] | population networks alone (no per-design fine-tuning) reduced performance; early designs worse than baseline; surrogate quality depends on the start-state batch [V] |

## D. Morphology + control co-design

| method | paper, year, URL | what it needs | budget reported (inner per morphology x outer count) | result | known failure modes |
|---|---|---|---|---|---|
| Evolved bodies + evolved neural controllers, one behaviour per evolution | Sims, "Evolving Virtual Creatures", SIGGRAPH 1994, https://www.karlsims.com/papers/siggraph94.pdf | genetic graph language for body and brain; fitness after ~10 s virtual time; early stopping of hopeless simulations [V] | "population sizes were typically 300, and the survival ratio was 1/5"; "population size 300, run for 100 generations, might take around three hours to complete on a 32 processor CM-5"; evolutions ran 50-100 generations [V]; ~3e4 evaluations per evolution [I] | swimming, walking, jumping, light-following, each evolved separately [V]; "An evolved watersnake ... was placed on land and then evolved to crawl instead of swim" [V] (a second evolution, not one body doing both) | "many separate evolutions were performed" because a population "often converges toward homogeneity" [V] |
| DERL: per-body PPO inside tournament evolution (Darwinian) | Gupta, Savarese, Ganguli, Fei-Fei, "Embodied Intelligence via Learning and Evolution", Nature Communications 2021, https://arxiv.org/abs/2102.02202 (ar5iv) | fresh, randomly initialised controller per child: "only morphological information is inherited from the parent" [V] | inner: "5 million agent-environment interactions ... for each morphology" [V]; outer: population 576, tournaments of 4, ~10 generations, ~4,000 morphologies per run, 3 runs per environment [V]; 1,152 CPUs, "less than 16 hours" [V]; ~4,000 x 5e6 = 2e10 steps per run [I] | Baldwin effect: "average learning time is cut in half" within ~10 generations; bodies evolved in harder environments do better on test tasks [V] | single learning seed per body; the authors note single-seed fitness changes can be statistically unreliable [V] |
| Evolution Gym (GA / BO / CPPN-NEAT outer, PPO inner) | Bhatia et al., "Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots", NeurIPS 2021, https://arxiv.org/abs/2201.09863 (ar5iv incl. appendix) | voxel soft-robot sim; PPO per design [V] | inner: 1000 iterations x 128 steps x 4 processes = 512,000 RL steps per robot [V]; outer: population 25 (CPPN-NEAT 50), at most 250 / 500 / 750 unique robots per run [V]; "several hours to twenty hours" per algorithm and task on 80 CPU cores [V] | GA best overall; CPPN-NEAT fine on locomotion, poor on manipulation [V] | "all tested algorithms fail to find robots that succeed in our hardest environments" [V]; the inner loop "leads to a long iteration cycle"; noisy RL evaluations defeat BO's surrogate [V] |
| Neural Graph Evolution (GNN controller weights inherited from parent) | Wang, Zhou, Ba, "Neural Graph Evolution: Towards Efficient Automatic Robot Design", ICLR 2019, https://arxiv.org/abs/1906.05370 (ar5iv) | GNN policy so weights carry across topologies ("A different graph will only alter the paths of message propagation") [V] | example: 10 epochs x 2,000 timesteps per generation, population 64, 200 (max 400) generations; ~12 h on 64 cores per session [V] | beats random graph search (which failed to search effectively even after 12,800 graphs) and Sims-style ESS; fish evolves symmetric side fins and a tail fin [V] | transfer with unstructured (MLP) controllers "hardly successful" [V]; compute-performance trade-off on the harder task [V] |
| Transform2Act (one policy proposes the body, then controls it) | Yuan, Song, Luo, Kitani, "Transform2Act: Learning a Transform-and-Control Policy for Efficient Agent Design", ICLR 2022, https://arxiv.org/abs/2110.03659 (ar5iv) | GNN (+ per-joint MLP) policy with skeleton-transform, attribute-transform and execution stages, all trained by PPO from execution returns; a new design each episode [V] | 5e7 environment steps (batch 50,000 x 1,000 epochs), "more than 2000" designs per epoch, ~1 day on 20 CPU cores + 1 RTX 2080 Ti; ES baselines (ESS, NGE) given the same 5e7 [V] | "outperforms prior methods significantly in terms of convergence speed and final performance" on 2D/3D locomotion, swimmer, gap crosser [V] | removing the GNN causes a large drop; without per-joint MLPs designs go symmetric and Swimmer variance is very high [V]; sample efficiency named as future work [V] |
| Model-based inner controller (MPC) + learned graph heuristic over a grammar | Zhao et al., "RoboGrammar: Graph Grammar for Terrain-Optimized Robot Design", SIGGRAPH Asia 2020, https://people.csail.mit.edu/jiex/papers/robogrammar/ | grammar of fabricable assemblies; MPC as the per-design controller (from search snippets and the follow-up) [I] | RoboGrammar's own numbers: PDF over the fetch limit [U]. Follow-up MOGHS (Xu et al., ICRA 2021, https://arxiv.org/abs/2107.05858): "stochastic MPPI scheme based off the POLO algorithm"; 2,000 episodes in "approximately 20 hours on a 64-core Google Cloud machine", of which MPC evaluation ~11 h [V] | terrain-specific legged designs found among "hundreds of thousands of possible robot designs" [V] | MPC needs the simulator at run time (no deployable policy) [I]; sim only [V] |
| Morphological innovation protection (age-fitness Pareto selection) | Cheney, Bongard, SunSpiral, Lipson, "Scalable co-optimization of morphology and control in embodied machines", J. R. Soc. Interface 2018, https://arxiv.org/abs/1706.06133 (ar5iv) | "morphological age" reset to zero on a body mutation; selection on (age, fitness) [V] | (25+25) ES, 5,000 generations, 30 trials per condition, VoxCad [V] | 31.953 vs 21.717 voxels (p ≈ 6e-6); at 10x10x10 resolution 60.728 vs 32.575; protecting controller mutations instead did nothing (22.049, p = 0.240); the unprotected baseline stagnated from gen 1000 to 5000 [V] | "most evolutionary changes to morphology tend to adversely impact sensorimotor control" [V]; one robot class only [V] |
| Diagnosis of co-optimisation (exhaustive landscape) | Mertan, Cheney, "Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential", arXiv 2025, https://arxiv.org/abs/2508.17464; also "Investigating Premature Convergence ...", EuroGP 2024, https://arxiv.org/abs/2402.09231; "Modular Controllers Facilitate the Co-Optimization ...", GECCO 2023, https://arxiv.org/abs/2306.09358 | controllers trained "for each morphology in a design space of 1,305,840 voxel-based soft robots" [V] | per-body budget not in abstract [U] | co-optimisation "cannot consistently find near-optimal solutions"; it "gets stuck on morphologies one mutation away from better ones"; high-performing bodies missed by co-optimisation are easy to find when morphology is optimised alone (2024); modular controllers transfer to similar bodies and raise the rate of positive body mutations (2023) [V] | the failure is in valuing newly mutated bodies [V] |
| Learning inside evolution, budget accounting | Luo et al., "The Effects of Learning in Morphologically Evolving Robot Systems", Frontiers in Robotics and AI 2022, https://www.frontiersin.org/article/10.3389/frobt.2022.797393 | inherited controller as the start of lifetime learning (RevDE) [V] | 100 learning trials per body; 100 initial + 50 offspring x 30 generations; 4,650 evaluations (evolution only) vs ~465,000 (with learning) [V] | with learning, generation 1 already exceeded evolution-only's final fitness [V] | the authors: "this is not a fair comparison because the number of evaluations is much higher for Evolution + Learning" [V] |
| Adaptive control-learning budget in co-design | Song et al., "Shaping the Evolutionary Dynamics of Robot Morphology via Adaptive Control Learning" (AdaControl), arXiv 2026, https://arxiv.org/abs/2608.23100 | monitors "disproportionate selection for morphological intelligence"; gives each body "minimally sufficient control learning for unbiased fitness evaluation" [V] | cuts computation "by up to 80% versus exhaustive control" [V] | a simple GA "rivals state-of-the-art generative-model-based co-design methods" [V] | abstract only [U] |
| Co-design through trajectory optimisation (implicit function theorem) | Ha, Coros, Alspach, Kim, Yamane, "Joint Optimization of Robot Design and Motion Parameters using the Implicit Function Theorem", RSS 2017, https://m.roboticsproceedings.org/rss13/p03.html | motions as solutions of optimal control problems; design parameters (link lengths, actuator layout) continuous [V] | computation time not on the page [U] | manipulator and large quadruped (sim), small quadruped (hardware) [V] | continuous parameters only [I] |
| Bilevel co-design with a differentiable motion planner | Dinev, Mastalli, Ivan, Tonneau, Vijayakumar, "A Versatile Co-Design Approach For Dynamic Legged Robots", IROS 2022, https://arxiv.org/abs/2103.04660 | "any differentiable motion planner in the lower level" [V] | not in abstract [U] | "a more energy-efficient Solo robot" for jumping and trotting [V] | abstract only [U] |

## E. Quality-diversity under noise and with RL

| method | paper, year, URL | what it needs | budget reported | result | known failure modes |
|---|---|---|---|---|---|
| MAP-Elites map built offline in simulation, then Intelligent Trial and Error on hardware | Cully, Clune, Tarapore, Mouret, "Robots that can adapt like animals", Nature 2015, https://arxiv.org/abs/1407.3501 (ar5iv) | simulator of the robot; low-dimensional open-loop gait parameterisation; behaviour descriptor (5 values per dimension) [V] | "we stopped each MAP-Elites run after 40 million iterations" (hexapod), "roughly two weeks on one multi-core computer"; arm 20 M evaluations [V] | "approximately 13,000 different gaits"; damaged hexapod adapts in 66 s / 8 trials (median) [V] | map creation expensive, done once per design [V]. Note the order of magnitude: 4e7 evaluations to fill a map for ONE fixed body [I] |
| MAP-Elites at hardware-accelerated scale | Lim, Allard, Grillotti, Cully, "Accelerated Quality-Diversity through Massive Parallelism" (QDax), arXiv 2022, https://arxiv.org/abs/2202.01258 (ar5iv) | GPU simulator (Brax) [V] | "5 million evaluations for all QD-RL environments"; up to ~30,000 evaluations/s on an A100; 1 M evaluations of Ant-Omni in 69 s [V] | "N=131,072 which runs for only I=39 provides the same performance as when run with N=256 and I=19,532" (total evaluations, not iterations, set the result) [V] | [U] |
| Adaptive sampling + drifting elites | Justesen, Risi, Mouret, "MAP-Elites for Noisy Domains by Adaptive Sampling", GECCO 2019 companion (2 pages), https://pure.itu.dk/da/publications/map-elites-for-noisy-domains-by-adaptive-sampling/ | re-evaluate elites and challengers [I, from Flageat & Cully's description] | full text returned HTTP 403 [U] | gains on a noisy test function and a 2-D walker (from the publication record) [V] | sequential re-evaluation is slow in parallel settings; drifting elites can block their true cell (Flageat & Cully 2020, below) [V] |
| Deep-Grid MAP-Elites | Flageat, Cully, "Fast and stable MAP-Elites in noisy domains using deep grids", ALIFE 2020, https://arxiv.org/abs/2006.14253 (ar5iv) | cells of depth 50; random replacement inside a full cell; fitness-proportional parent choice within a cell [V] | 2e7 evaluations (Rastrigin, arm), 5e6 (hexapod, ~50 h on 32 CPUs per replication) [V] | **with one evaluation per solution "less than 20%" of elites sat in their correct cell** under fitness noise sd 0.05 and descriptor noise sd 0.01; Deep-Grid ~20% mis-evaluated; re-sampling N times shrinks sd only by sqrt(N) [V] | quality pulled down by averaging over a population; only invariant noise tested [V] |
| Uncertain-QD benchmark: Archive-Sampling, Parallel-Adaptive-Sampling, Deep-Grid-Sampling | Flageat, Cully, "Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains", arXiv 2023 (submitted to IEEE TEVC), https://arxiv.org/abs/2302.00463 (ar5iv) | fitness and descriptor as distributions; corrected archive from the median of M = 512 re-evaluations per cell [V] | per-generation sampling budget 256-16,384; 10 replications each, 1,120 runs; Brax arm, hexapod, walker, ant [V] | MAP-Elites has the highest QD-score loss (apparent minus corrected); Deep-Grid best on arm, hexapod, walker at any sampling size; Archive-Sampling / Parallel-Adaptive best on Ant at large budgets [V]; numeric loss values only in figures [U] | Archive-Sampling re-evaluates the whole archive every generation (undefined below archive size); Deep-Grid struggles on Ant and complex noise; gradient QD excluded for cost [V] |
| Extract-ME (re-evaluate 25% of each batch, depth-8 cells) | Flageat, Huber, Helenon, Doncieux, Cully, "Extract-QD Framework: A Generic Approach for Quality-Diversity in Noisy, Stochastic or Uncertain Domains", arXiv 2025, https://arxiv.org/abs/2502.06585 (arXiv HTML v1) | "dedicate a fixed proportion (here 25%) of the evaluations per generation to re-evaluating elites"; depth d = 8, only the top layer returned [V] | 1,024 evaluations per generation in the main comparisons [V] | vanilla MAP-Elites "performs among the worst across the board"; EME matches the best per task and beats all on Ant; its PGA variant beats PGA-ME on all tasks at equal budget [V] | addresses performance estimation only, not reproducibility maximisation [V]; no count of non-reproducing elites given [U] |
| PGA-MAP-Elites (TD3 critic as a variation operator) under uncertainty | Flageat, Chalumeau, Cully, "Empirical analysis of PGA-MAP-Elites for Neuroevolution in Uncertain Domains", arXiv 2022, https://arxiv.org/abs/2210.13156 (ar5iv); original Nilsson & Cully GECCO 2021 not reachable (bot wall) [U] | half the batch from a policy-gradient operator, half GA; uncertainty = Gaussian initial joint positions [V] | ~1e6 controller evaluations x 1,000 steps within 3 days on 32 cores [V]; ~1e9 steps [I] | PGA-ME max-fitness loss after 50 re-evaluations "less than 20% loss across all tasks" (<10% Walker, <5% HalfCheetah, Ant); MAP-Elites' loss "significantly higher" (p < 1e-7) [V] | no explicit uncertainty mechanism; the PG operator matters "only crucial for the first phase" [V] |
| DCG-MAP-Elites (descriptor-conditioned critic; archive distilled into one policy) | Faldor, Chalumeau, Flageat, Cully, "MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy", GECCO 2023, https://arxiv.org/abs/2303.03832 (ar5iv) | descriptor-conditioned critic and actor [V] | "over one million evaluations", 20 seeds [V] | QD-score +82% over PGA-ME on average; distilled single policy within 3% of the archive on Ant-Omni / AntTrap-Omni, but >40% worse on Hexapod-Omni and Walker-Uni [V] | PGA-ME's operator pushes omnidirectional robots toward standing still [V]; needs Markov/fully observable rewards [V] |
| PPGA (on-policy PPO inside differentiable QD) | Batra, Tjanaka, Fontaine, Petrenko, Nikolaidis, Sukhatme, "Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning", ICLR 2024, https://arxiv.org/abs/2305.13795 (arXiv HTML) | Brax, massively parallel envs [V] | per iteration 300 policies x 10 episodes x <= 1,000 steps ≈ 3e6 timesteps [V]; total iterations not read [U] | "4x improvement in best reward over baselines on the challenging humanoid domain" [V] | "less sample-efficient than other off-policy QD-RL methods"; PPO's collapsing std hurts exploration (Ant coverage) [V] |
| QD-PG (diversity policy gradient) | Pierrot et al., "Diversity Policy Gradient for Sample Efficient Quality-Diversity Optimization", arXiv 2020 (v5 2022), https://arxiv.org/abs/2006.08505 | time-step-level diversity gradient + quality gradient on MAP-Elites selections [V] | not in abstract [U] | "significantly more sample-efficient than its evolutionary competitors" [V] | [U] |

## F. Existence proofs without learning; demonstrations for learning

| method | paper, year, URL | what it needs | budget reported | result | known failure modes |
|---|---|---|---|---|---|
| Predictive Sampling / MuJoCo MPC | Howell et al., "Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo", arXiv 2022, https://arxiv.org/abs/2212.00541 (ar5iv) | the simulator as the model (MuJoCo runs the 27-DoF humanoid about 4000 times faster than real time on a single CPU thread); a residual cost; spline-parameterised nominal plan perturbed by Gaussian noise [V] | N = 10 rollouts per iteration; horizons 23 / 35 / 25 (humanoid / quadruped / hand); 1-20 ms per planning update [V] | humanoid stands and gets up after large pushes; A1 crosses uneven terrain and rolls off its back; in-hand cube reorientation; "surprisingly competitive with the more elaborate derivative-based algorithms" [V] | "Can only control what MuJoCo can simulate" (fluids named as out of reach); "cannot see past the fixed horizon"; local minima; needs a state estimator on hardware [V]. For this project the custom fluid forces must be inside the planner's rollouts (a MuJoCo callback or the built-in ellipsoid fluid model) [I] |
| MPPI (sampling MPC on a GPU) | Williams, Aldrich, Theodorou, "Model Predictive Path Integral Control using Covariance Variable Importance Sampling", arXiv 2015, https://arxiv.org/abs/1509.01149 | a dynamics model sampled in parallel on a GPU; a cost [V] | not in abstract [U] | compared with an MPC version of DDP in simulation [V] | [U] |
| Guided policy search (local trajectory-optimised teachers, supervised policy) | Levine, Finn, Darrell, Abbeel, "End-to-End Training of Deep Visuomotor Policies", JMLR 2016, https://arxiv.org/abs/1504.00702 (ar5iv) | "the full state of the system is observable at training time, but not at test time"; resets to known initial states [V] | ~20-25 samples per linear-Gaussian controller; 156-288 robot trials per visuomotor task; "Only about 15 minutes of the training time consisted of executing trials on the robot" [V] | real PR2 visuomotor skills (hanger, cube, hammer, bottle cap) [V] | local linearity can fail under stochastic dynamics; poor under drastic backdrop changes [V] |
| PLATO (MPC teacher adapted toward the learner; supervised student) | Kahn, Zhang, Levine, Abbeel, "PLATO: Policy Learning using Adaptive Trajectory Optimization", ICRA 2017, https://arxiv.org/abs/1603.00622 (ar5iv) | true state and an approximate model at training; no episodic resets needed (MPC-GPS needs them) [V] | 2 minutes of flight per training iteration, 20 evaluation flights per iteration, 10 seeds [V]; total not stated [U] | fewer crashes during training than DAgger variants; better than MPC-GPS in the forest task (sim quadrotor) [V] | needs a model and full state at training; KL weight to tune; sim only [V] |
| DAgger | Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011, https://arxiv.org/abs/1011.0686 | an expert that can label states visited by the learner [V] | not in abstract [U] | plain supervised imitation leads to "poor performance in theory and often in practice"; the iterative algorithm outperforms previous approaches on two imitation problems [V] | [U] |
| Teacher-student / privileged learning | Lee et al. 2020 and Kumar et al. 2021 (RMA), rows in section A | privileged simulator state for the teacher; student from proprioceptive history by DAgger or regression [V] | teacher 12 h + student 4 h (Lee); 1.2e9 steps + adaptation module (RMA) [V] | blind rough-terrain locomotion on hardware [V] | see section A |
| RL tracking trajectory-optimised references (DTC) | Jenelten et al., "DTC: Deep Tracking Control", Science Robotics 2024, https://arxiv.org/abs/2309.15462 (ar5iv) | a trajectory optimiser (TAMOLS) re-solved at each touch-down produces footholds, base trajectory, contact schedule; PPO policy tracks them [V] | 4,096 robots, 90,000 epochs (~two weeks; usable after ~one day / 6,000 epochs); ~23 years of optimised trajectories [V] | real ANYmal crossed a 0.1 m gap 10/10 and four 0.6 m gaps 4/4 where the RL baseline failed; outperformed RL "by a huge margin" on sparse terrain in sim [V] | not more sample-efficient than comparable RL; fixed trot, cannot change contact schedule; degraded on unseen foothold patterns; "by no means" a universal recipe [V] |
| GPU MuJoCo (MJX) training budgets, for scale | Zakka et al. [I], "MuJoCo Playground", arXiv 2025, https://arxiv.org/abs/2502.08844 (arXiv HTML) | MJX on GPU [V] | locomotion: "all policies are trained on flat terrain for 200 M timesteps" then 100 M rough; Go1 flat "within 5 minutes (2x RTX 4090)"; G1 / T1 "under 30 minutes"; DM Control PPO 60 M steps (some 100 M) [V] | sim-to-real joystick, handstand, footstand, fall recovery [V] | JIT compile 1-3 min; contact scaling limit [V] |

## G. Multi-task interference and mode-specific skills

| method | paper, year, URL | what it needs | budget reported | result | known failure modes |
|---|---|---|---|---|---|
| PCGrad (gradient surgery) | Yu, Kumar, Gupta, Levine, Hausman, Finn, "Gradient Surgery for Multi-Task Learning", NeurIPS 2020, https://arxiv.org/abs/2001.06782 (ar5iv) | per-task gradients; project each conflicting gradient "onto the normal plane of the other" [V] | total steps not in the read text [U]; independent SAC agents needed ~2 M (MT10) / ~15 M (MT50) more samples than SAC+PCGrad (garbled in the rendering) [V/I] | the interference "tragic triad": conflicting gradients (negative cosine), dominating magnitudes, high curvature [V]; a single multi-task SAC "is unable to acquire half of the skills in both MT10 and MT50"; with PCGrad all of MT10 and ~70% of MT50 [V] | can stall where gradients cancel exactly (argued unlikely) [V] |
| Policy distillation | Rusu et al., "Policy Distillation", ICLR 2016, https://arxiv.org/abs/1511.06295 (ar5iv) | frozen per-task teachers; supervised student (KL on sharpened Q) [V] | the largest multi-task student "used 34,000 hours of experience over 10 games" [V] | one student "achieves 89.3% of the generalization score of 10 single-task DQN teachers"; single-game students with "15 times fewer parameters perform on par" [V] | multi-task student was 4x a single DQN; no comparison with a jointly trained 10-game DQN [V] |
| Distral (shared distilled policy as a KL prior for per-task policies) | Teh et al., "Distral: Robust Multitask Reinforcement Learning", NeurIPS 2017, https://arxiv.org/abs/1707.04175 (ar5iv) | per-task policies + a distilled centroid policy; entropy term [V] | [U] | faster, better and more robust than multitask A3C in DeepMind Lab; not significantly better than separate A3C per task on laser-tag [V] | shared-parameter multitask RL: gradients interfere and a task with larger rewards dominates [V, paraphrased] |
| MELA: pre-trained experts fused by a gating network | Yang, Yuan, Zhu, Yu, Li, "Multi-expert learning of adaptive legged locomotion", Science Robotics 2020, https://arxiv.org/abs/2012.05810 (abstract) | "a distinct set of pre-trained experts, each in a separate deep neural network"; gating network blends them online [V] | not in abstract [U] | "more specialised experts and transitional skills across various locomotion modes"; real quadruped trots, steers, recovers from falls [V] | [U] (full text not read) |
| MCP: multiplicative composition of primitives | Peng, Chang, Zhang, Abbeel, Levine, "MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies", NeurIPS 2019, https://arxiv.org/abs/1905.09808 (ar5iv) | motion-imitation pre-training corpus; gating sees the goal, primitives see only the state [V] | total budget not stated [U] | transfer, normalised return: Carry 0.575 (scratch 0.027, one-at-a-time MOE 0.013); T-Rex Dribble 0.781 (scratch 0.065) [V] | Ant holdout directions: scratch (0.951) beat MCP (0.812); needs a chosen reference corpus [V] |
| Thousands of experts distilled into one latent policy (NPMP) | Merel et al., "Neural probabilistic motor primitives for humanoid control", ICLR 2019, https://arxiv.org/abs/1811.11711 (ar5iv) | 2,707 expert policies, one per mocap snippet, then offline distillation (noisy-rollout cloning or linear-feedback cloning) [V] | 100 trajectories per expert (cloning); 1 rollout per expert (LFPC) [V] | one latent policy reused by high-level RL for go-to-target, speed tracking with walk/jog/run transitions, gap-jumping obstacle course [V] | rarer behaviours (backward walking, jumping) poorly reproduced; full behaviour set needed in advance [V] |
| Task-conditioned single policies (pointer row) | salamander (medium flag, section A), DCG-MAP-Elites (descriptor-conditioned actor, E), DeepMimic skill selector (one-hot goal, B), PCGrad's Meta-World multi-task SAC (this section) | task or mode code in the observation [V] | see those rows | a medium flag sufficed for one body in two media at 3e8 steps (transition qualitative only); a descriptor-conditioned single policy stayed within 3% of the archive on two tasks but lost >40% on two others; a single multi-task SAC learned under half of MT10/MT50 without gradient surgery [V] | conditioning alone does not prevent interference; see PCGrad and Distral [V] |
| Skill-specific RL then distillation into one policy (for locomotion) | Zhuang et al. 2023 (parkour) and Ai et al. 2025 (embodiment scaling), rows in sections A and C | per-skill / per-body experts, DAgger or BC into one student [V] | see those rows [V] | the strongest multi-skill and multi-body single policies read in this survey were built this way [I] | see those rows |

---

## Synthesis

Every number below is copied from a table row above (its tag travels with it);
arithmetic done here is [I]. Each bullet names the PREREG rival it bears on.

### 1. Budget gap between this project and what worked (orders of magnitude)

The project's figures are the brief's (not re-measured here): 1e6-1e7 control steps
per run in total, one policy, "hundreds" of bodies, 3 media + 6 crossing kinds = 9
skill kinds.

- **Per body** [I]: 1e6-1e7 steps / 5e3-1.5e4 body evaluations ≈ 7e1-2e3 steps per
  body evaluated. That is consistent with a few 8-second segments per body.
  **Per (body, skill)** [I]: at most that, and only for the skills a body's draws
  cover; zero for the rest. Re-evaluated elites get more, but even 1e7 / (300
  bodies x 9) is only ~4e3. **Pooled per skill across all bodies** [I]: ~1e5-1e6.
- **The cheapest single-skill successes found** are 3e5 steps (quadrotor hover,
  off-policy TD3, Eschmann 2023 [V]), ~5e5 (a residual added to a model-based
  flapping controller, Fei 2019 [I from V]) and ~1e6 (hybrid aerial-underwater
  navigation on top of a stabilising velocity controller, Grando 2021 [I from V]).
  All three are one fixed body, one narrow skill, and two of them sit on a
  hand-built stabiliser. **Typical single-body locomotion skills**: 3e7 (flapping
  wing in MuJoCo with a stateless fluid model, the closest analogue to this
  project, Cai 2024 [V]); 4e7-1.9e8 per skill (DeepMimic [V]); 1e8 (Swift [V]);
  1.5e8 (Rudin 2021 [I from V]); 2e8 flat + 1e8 rough (MuJoCo Playground [V]);
  3e8 for one body in two media with one policy (salamander, 2026 [V]); 1.2e9 (RMA
  [V]).
- **Gap per (body, skill)** [I]: ~2-4 orders of magnitude below the cheapest
  narrow successes (3e5-1e6 against 7e1-2e3), and ~4-6 orders below typical
  locomotion skills (3e7-1.5e8).
  **Gap pooled per skill** [I]: 1.5-3 orders below what one body needs for one
  skill, before paying for the hundreds of bodies. (R3)
- **Multi-body controllers** spent: MetaMorph 1e8 for 100 fixed, pre-selected
  bodies, ~1e6 per body [V/I]; URMA 1e8 per robot, 1.6e9 for 16 [V]; embodiment
  scaling > 2e12 for ~1,000 bodies (~2e9 per body) plus 2e9 for distillation [V];
  Schaff 2019 ~1e9 for one parametric family of one topology [V/I]. The project's
  total is 1-6 orders below these, and it alone has bodies that change during
  training and three media. (R3)
- **Co-design inner loops**: DERL 5e6 steps per body x ~4,000 bodies (~2e10 per run
  [I]); Evolution Gym 5.12e5 per body x 250-750 bodies, and even that "fail[s] to
  find robots that succeed in our hardest environments" [V]. The project's outer
  count (5e3-1.5e4 bodies) is DERL-sized. Its inner budget per body (~7e1-2e3 steps
  [I]) is ~3.5-5 orders below DERL and ~2.5-4 below Evolution Gym. (R3, R4)
- **QD outer budgets for one fixed body**: 4e7 evaluations to fill the hexapod map
  (Cully 2015 [V]); 5e6 evaluations for QD-RL tasks (QDax [V]); ~1e6 evaluations
  x 1,000 steps for PGA-MAP-Elites [V]. (R4)
- **Noise**: with one evaluation per solution "less than 20%" of MAP-Elites elites
  sat in their correct cell (Flageat & Cully 2020 [V]); MAP-Elites has the largest
  apparent-minus-corrected QD-score loss of the methods benchmarked (2023 [V]).
  **The project's 2-6% retention does not match these figures** [I]. It points
  the same way: single-evaluation archives keep lucky elites. But 2-6% retention
  is a 94-98% loss, and nothing read here comes close:
  - "less than 20%" correct cells is a placement figure, not a fitness loss;
  - PGA-ME loses "less than 20%" of max fitness;
  - MAP-Elites' loss is only "significantly higher", with no number given.

  So either the project's draw-to-draw noise is far larger than in the standard
  uncertain-QD benchmarks, or its retention metric measures something these papers
  do not (the quantile fitness scalar, the gates). That distinction separates R4
  from R6 and belongs in DIAGNOSIS.md. (R4, R6)
  The fixes cost budget: Extract-ME spends 25% of every batch on re-evaluating elites [V]; the
  corrected archive in the benchmark is the median of 512 re-evaluations per cell
  [V]; N re-samples shrink the noise sd only by sqrt(N) [V]. (R4)
- **R7**: three more runs of the same design add at most 0.5 order of magnitude [I];
  the gap above is 2-6 orders. Nothing in the literature read here suggests that
  closing it by running longer is plausible.

### 2. The techniques with the strongest evidence for conjunctive sparse missions

Ranked by strength of evidence (independent papers with ablations first).

1. **Start the episode where the rare sub-goal is reachable, then move the start
   back** (R5). Independent, ablated evidence from five lines of work:
   - Backward Algorithm from one demonstration learns a sequence of N actions "in a
     time that is quadratic in N, rather than exponential" (Salimans & Chen [V]).
   - Backplay: standard PPO "has immense trouble learning" the sparse mazes;
     Backplay comes within five steps of optimal on 94-100% of them [V].
   - Reverse curriculum: uniform start sampling succeeds only near the goal (~10%
     ring-on-peg, ~2% key insertion); starts kept at 0.1-0.9 success solve from
     distant starts [V].
   - DeepMimic: without reference-state initialisation the backflip "never learns
     to perform a full mid-air flip" (0.379 vs 0.791) [V]. SoftCon starts each
     episode from a pool of previously visited states [V].
   - Go-Explore on a sparse-reward robot task: PPO found no reward in 1e9 frames
     and count-based intrinsic motivation none in 2e9; Go-Explore succeeded in 198
     of 200 runs after 2e7 exploration frames [V].

   What they need: a simulator that resets to arbitrary states (all five), plus one
   source of intermediate states: a demonstration, one goal state, or an archive of
   states already reached. MuJoCo provides the reset; whether this project's fluid
   solver carries state that must be restored too is not known here [U]. The
   project's own archive and its scored segments are a possible state source [I].

2. **Train skills separately, then handle the hand-off explicitly** (R5, R3).
   - Both skill-chaining papers measure the failure of naive sequencing: a skill
     "may be executed in starting states not considered during training" (Lee 2019
     [V]); "naively executing one policy after another would fail" (T-STAR [V]).
   - Transition policies with a proximity reward: Serve 0.92 against 0.32 and 0.00
     for flat TRPO and PPO [V]. T-STAR: 0.90 against 0.63 on table assembly [V].
   - ANYmal Parkour (5 frozen skills under a navigation selector) and the
     wheeled-legged controller (one learned policy that chooses walking or driving)
     both work on hardware. Both needed curricula or gait pre-training to
     converge [V].

   What they need: a given decomposition, a success signal per sub-task, and a reset
   into each sub-task.
3. **One policy with a mode or medium input can learn switches, once the body is
   already competent in each mode** (R2, R5). Examples: iRonCub walk-to-fly with
   motion priors (sim) [V]; wheeled-legged gait switching (hardware) [V]; the
   salamander's single policy in land and water (3e8 steps; the transition is a
   single qualitative demo in simulation) [V]; DoubleBee (8/10 real steps) [V]. In
   iRonCub, removing the priors removed the skill: with the walking prior only the
   robot learned to walk but not to fly, and with the flying prior only it did not
   learn to walk [V, paraphrased]. Mode choice by reward shaping is
   fragile: "Tuning the thrust penalization is not trivial" [V].
4. **Automatic curricula over task parameters** (ALP-GMM, POET, PLR) give gains of
   about 2x (39.6% vs 18.2% of tracks mastered) [V], add nothing when the body is
   already capable (quadrupedal walkers: random sampling near 100%) [V], and POET
   cost ~10 days on 256 cores per run [V]. They presuppose reward at the easy end.
   This is moderate evidence, useful for ramping crossing difficulty, not for
   finding the first crossing.
5. **Exploration bonuses alone have negative evidence** in exactly this kind of
   task: RND "cannot make any learning progress on climbing and leaping" (parkour
   [V]); count-based exploration found no reward in 2e9 frames (Go-Explore robot
   task [V]). (R5)
6. **Existence proofs on hardware are model-based, not learned**: M4 (direct
   collocation, optimisation-based feedback, an energy-cost roadmap planner;
   autonomous switching only between ground and flight, and the pond test
   "teleoperated (not autonomous)") [V]; LEONARDO (thrusters "controlled
   synchronously with leg joints", no learning) [V]. No source read here shows a
   learned controller crossing air, water and land on one body [I: absence in
   these searches, not proof]. (R1, R2)

### 3. What the literature says about one shared policy over evolving bodies

- **Absence**: none of the works read trains one small policy online while an outer
  evolutionary loop replaces bodies across media and reports success [I: from the
  searches above, not proof of absence]. (R3)
- **The nearest analogues** share a controller across changing designs only with
  one of three supports:
  - Design-conditioning plus ~1e9 steps, for parametric changes within one topology
    (Schaff 2019). Even there seven of eight seeds of the Ant on an incline stuck
    in a balance-without-moving optimum [V].
  - A population network plus a per-design network fine-tuned on each new design.
    The population network alone reduced performance (Luck 2019 [V]).
  - A graph network whose weights carry across topologies. Transfer with plain MLP
    controllers was "hardly successful" (NGE [V]); removing the GNN caused a large
    performance drop (Transform2Act [V]). Transform2Act, the one case where a
    single policy controls a stream of new bodies, used 5e7 steps in one medium,
    with the body chosen by the policy itself [V].
- **Universal controllers that work are trained on fixed body sets, with a
  body-aware architecture and >= ~1e6 steps per body**:
  - MetaMorph: 100 fixed bodies pre-selected for locomotion [V]. Without replay
    balancing about 10-15% of robots fail to learn a good policy, and there is a
    "performance drop on zero-shot generalization to new morphologies" [V].
  - URMA: 1e8 steps per robot [V]. Humanoids do not transfer zero-shot, and
    shuffled joint descriptions make the A1 fall within one or two steps [V].
  - Embodiment scaling: 1,000 bodies via ~2e9-step per-body experts, then
    distillation, chosen for "stable training dynamics" [V].
  - SMP's plain multi-task baseline "fails to perform well in any environment" [V].

  If this project's policy does not see a description of the body, the nearest
  measured case is that failing baseline [I]. (R3)
- **Selection undervalues new bodies under a shared or inherited controller**
  (R4):
  - "most evolutionary changes to morphology tend to adversely impact sensorimotor
    control"; protecting new bodies by age raised the result from 21.7 to 32.0, and
    the unprotected run stagnated from gen 1000 to 5000 (Cheney 2018 [V]).
  - Co-optimisation "cannot consistently find near-optimal solutions" and "gets
    stuck on morphologies one mutation away from better ones" (Mertan & Cheney 2025
    [V]). Bodies it misses are easy to find when morphology is optimised alone
    (2024 [V]).
  - DERL avoids the problem by giving every child a fresh controller and 5e6 steps
    [V].

  Under one small shared policy, a body's score therefore measures how well the
  current policy happens to fit it, not what the body could do. That fits the
  project's finding that random bodies score as well as the evolved archive [I].
- **What would make sharing defensible** [I, from the rows above]:
  - a fixed or slowly changing body set;
  - a body-aware architecture (graph or limb-token transformer);
  - per-body fine-tuning before a body is judged;
  - age protection for new body plans;
  - re-evaluated fitness.

### 4. Recommended decomposition for a fly-swim-walk-and-cross mission, and its risks

This plan is [I]: it is assembled from the rows above, not taken from any single
paper. Step 0 is offline. Steps 1-3 are the three runs. **Start no training run
until step 0 passes.** If no body can make a crossing under MPC, no run will fix
that.

- **Step 0: feasibility, no learning** (settles R1/R2 before any run).
  - Take 1-3 bodies: the rotor reference, the best archive multi-media body, and
    one hand-designed body.
  - Synthesise each medium skill and each crossing with sampling MPC (Predictive
    Sampling: N = 10 rollouts, horizon 23-35 steps, 1-20 ms per update in MuJoCo
    [V]; or MPPI), starting from boundary states.
  - The project's fluid forces must be inside the planner's rollouts [I].
  - Output: pass/fail per (body, skill), and a library of feasible trajectories and
    boundary states for steps 1-2.
  - If no body passes a crossing under MPC, learning will not fix it [I]. The
    redesign then becomes body-first with an MPC inner loop (RoboGrammar-style;
    its follow-up costs ~20 h on 64 cores for 2,000 episodes [V]).
- **Step 1 (run 1): per-skill experts on one fixed body** (R3, R5).
  - Massively parallel PPO, a privileged teacher allowed [V: Lee 2020, RMA, parkour].
  - Budget: 1e7-1e8 steps per skill (Cai 3e7 for flapping in MuJoCo [V]; salamander
    3e8 for two media [V]). For 9 skills that is ~1e8-1e9 steps [I], 1-3 orders
    above a current run [I]. Simulator throughput on the project's fluid path
    decides whether this fits in one run [U].
  - Every crossing gets reference-state starts from the step-0 library, early
    termination, and a reverse curriculum: start at the boundary, keep starts whose
    success is in 0.1-0.9, and move them back [V: DeepMimic, Florensa].
  - Gate each expert on fresh seeds against the still-machine null, as the project
    already does.
- **Step 2 (run 2): one controller.**
  - Distil the experts into one command- and medium-conditioned student by DAgger
    or BC (parkour, NPMP, embodiment scaling [V]). Keep a hierarchical fallback:
    frozen experts, a selector, and transition policies with a proximity reward
    (ANYmal Parkour, Lee 2019 [V]).
  - Then fine-tune on the whole mission with episodes started along the chain
    (Backward Algorithm / Go-Explore-style return to archived states [V]).
  - Accept on the continuous mission, fresh seeds, and a film that reproduces the
    score.
- **Step 3 (run 3): bodies, only after step 2 works** (R4).
  - Search a neighbourhood of the working body.
  - Fine-tune each candidate from the distilled policy (MetaMorph fine-tuning is
    2-3x faster than scratch [V]; Luck population + individual networks [V]).
  - Use a body-aware policy, age protection for new body plans (Cheney [V]), and a
    QD archive with re-evaluation (Extract-ME 25% [V] or Deep-Grid [V]).
  - Give each candidate >= 1e5-1e6 steps [I] before it is compared.
- **Risks**:
  - **Physics (R1)**: step 0 can fail. That answer is cheap and decisive, but it
    ends the "three runs" plan.
  - **Model exploitation**: MPC and RL both exploit simulator error. Cai's flapping
    controller was most sensitive to the Kutta lift coefficient [V]. This project's
    fluid model is quasi-steady, not CFD.
  - **Compute**: DTC took ~two weeks on 4,096 robots [V]; embodiment scaling took
    160 GPUs for 5 days [V]. Step 1's 1e8-1e9 steps may not fit one run on the
    project's fluid path [U].
  - **Distillation loss**: DCG-ME's single policy lost >40% on two tasks [V];
    policy distillation reached 89.3% of its teachers [V]; NPMP reproduced rarer
    behaviours poorly [V]; a single multi-task SAC learned under half of MT10/MT50
    without PCGrad [V]. The hierarchical fallback hedges this.
  - **Hand-offs**: they fail when terminal states fall outside the next skill's
    start set [V]. Step 2 must train the crossings from the experts' actual
    terminal states, not idealised ones [I].
  - **Mode choice by reward weights is fragile** (iRonCub thrust penalty [V]).
    Prefer commanded modes, which this project already uses.
  - **Reset realism**: the reverse curriculum set velocities to zero [V]. Boundary
    states for crossings need consistent velocities and fluid state [I].

### Confidence per section

- **A (budgets): high** for legged and quadrotor (full text, several papers);
  **medium** for flapping and swimming (Cai 2024 is a single arXiv preprint; SoftCon
  reports hours, not steps; Verma's totals not read); **low** for hybrid
  aerial-underwater (one group, sim only, velocity-level actions over a stabiliser).
- **B (sparse chains): high** that resettable starts beat plain RL on sparse
  chains (five independent lines with ablations); **medium** for multi-modal robot
  RL (sim-only transitions or two-mode hardware, few papers).
- **C (one controller, many bodies): high** that working universal controllers
  used fixed sets, body-aware networks and >= 1e6-1e8 steps per body; **medium** on
  "no one has made one small shared policy work over evolving bodies across media"
  (absence in these searches).
- **D (co-design): high** on per-body inner budgets (DERL, Evolution Gym);
  **medium** on inner-loop vs outer-loop advice (mostly voxel soft robots).
- **E (noisy QD): medium-high**. Single-evaluation archives misplace most elites
  ("less than 20%" correct) [V]; numeric QD-score losses are only in figures [U].
- **F (existence proofs): medium**. MPC in MuJoCo is shown for rigid bodies; MPC
  with a custom fluid model inside its rollouts is not shown in these sources [I].
- **G (interference): medium**. Interference evidence is from Meta-World and Atari;
  for locomotion the evidence is that per-skill-then-distil works, not a measured
  interference failure.
