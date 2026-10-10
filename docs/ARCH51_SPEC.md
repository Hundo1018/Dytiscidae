# arch51 — implementation spec for ROADMAP §"2026-10-10" N5, N11, the N10 telemetry lane, and the probes N1, N6, N8, N9

Written 2026-10-10 against `failure-theory-1010` at 5cc9db9. It is a plan, not a measurement: every number below cites where it was read. Line numbers are as of 5cc9db9; open the cited lines before editing. N2, N3, N4 and N10 are read (ROADMAP block "N2, N3, N4 and N10 were read the same day"); N7 is a derivation and is not planned here.

Reads that shape this plan (ROADMAP, 5cc9db9):
- N2: share of the window at exactly zero island score is air 0.965, water 0.586, land 0.605 (arch49). So `<` alone floors air only, and N5's floor is a **competence floor**.
- N4: Tier-2 elites sit in the bottom parent-weight decile in 24/24 islands. The Tier-2 lane stands.
- N10: not comparable, because runs log raw terminal rewards and not the shaping return. Lane L7 adds the missing fields.

Conventions for every lane:
- One git worktree per lane: `git worktree add ../a51-<lane> -b a51-<lane> failure-theory-1010` (or the merged predecessor for dependent lanes).
- One-function runner. Expected last line: `FAILURES []`.
  `PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('t','tests/<suite>.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.<fn>(); print('FAILURES', m.FAILURES)"`
- Mutation check: `PYTHONPATH=. .venv/bin/python tools/mutate.py --only <id>`. Expected: status `caught` by the named function, never `MISAPPLIED` or `SURVIVED`.
- Index after any source edit: `tools/index_gen.py write`, then `tools/index_gen.py check` (exit 0), then `tests/test_index.py`. On a merge conflict in `docs/index/MODULES.md` or `SYMBOLS.md`, take either side and re-run `write`.
- Cheap suites: `for s in index architecture domain application search_adapter adapters worker; do PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1; done`. Every line must be its suite's success string.
- New test functions go immediately after the named anchor function, and into `main()`'s `run_all([...])` list right after the anchor's entry. New mutations go into `MUTATIONS` right after the named anchor mutation. The anchors differ per lane, so parallel merges conflict only trivially.
- Mutation `find` strings below are exact lines the lane must write. Keep the code text byte-identical, or update the mutation in the same commit.

---

## §A. Builds

### A.0 What N5 is, exactly

The fitness scalar is built in `_score_candidate` (loop.py:1146-1172). After N5:
1. `Curriculum.standing` and `mission_standing` rank with strict `<`. A raw score `<= ZERO_SCORE` (1e-9) stands at 0 whatever the window size.
2. A design competent in none of the media its island reads (competence < 0.012 in each) stands at 0 on both halves (`isl_q = cur_q = mis_q = 0`) and is flagged `at_floor`.
3. `mission_weight` defaults to 0.0. The field stays: old checkpoints resume, and the job path accepts it.
4. A design at the floor fills only an empty cell (F1). A design above the floor entering a cell removes the members at the floor (F2).
5. Tier-2 no longer caps `fitness`. Its outcome is stored as `meta["tier2_fitness"]` and `meta["tier2_gen"]` beside `tier = 2`. The promotion pool skips tier-2 elites.

Pre-registered (ROADMAP N5, verbatim): "*Prediction for the first arm with it:* evaluated `fitness` median in gens 0-49 < 0.3 (was 0.68-0.78); share of elites with own-medium competence >= 0.012 rises from 9% to >= 40% by gen 150; Spearman(`fitness`, own competence) over the archive >= 0.7 (was 0.3). *Falsified* if any of the three misses."

Comparability boundary (L12a puts this sentence in ROADMAP and CLAUDE.md "Comparability boundaries"):
> From commit <N5 merge sha> (arch51 onward), stored `fitness`, `objectives[0]` and everything built on them (qd_score, parent weight, migrants, promotion and prune order, `best`) are not comparable with any earlier run. Three changes cause this:
> - a design competent in none of its island's media (< 0.012) stands at 0, where it used to stand at its window's zero share (0.46-0.99, N2);
> - `mission_weight` is 0.0, where it was 0.30;
> - Tier-2 no longer caps `fitness`.
>
> Per-medium competences, `mission_fraction` and coverage are not changed by the build.

### L1 — floor-rank (curriculum.py; tests/test_search.py; tools/mutate.py). No dependencies.

`dytiscidae/evolution/curriculum.py`:
- Add these module-level definitions before `class Curriculum`:
  ```python
  #: A raw score at or below this is zero (ROADMAP 2026-10-10 N5).
  ZERO_SCORE = 1e-9

  def _standing(window, x: float, min_n: int) -> float:
      """``x``'s standing in ``window``: the share strictly below it; 0 at zero."""
      x = float(x)
      if x <= ZERO_SCORE:
          return 0.0
      if len(window) < min_n:
          return 0.5
      return float(np.mean(np.asarray(window, float) < x))
  ```
- `standing` (472-478): body becomes
  `w = self._recent.get(int(stage), [])`, then
  `return (_standing([a for a, _, _ in w], island_score, self.min_rank_samples), _standing([b for _, b, _ in w], curriculum_score, self.min_rank_samples))`.
- `mission_standing` (499-503): `return _standing([c for _, _, c in w], mission, self.min_rank_samples)`.
- Behaviour: zero scores stand at 0 at every window size. Ties among equal non-zero scores receive the share strictly below them (the lower rank). The 12-sample fallback stays 0.5 for non-zero scores only. The candidate's own window entry no longer counts toward its standing. Update both docstrings in one sentence each, citing N2.

Test `test_a_score_of_zero_stands_at_zero` (CPU). Anchor: after `test_the_island_objective_takes_its_weight_back`.
- (a) Fill stage 0 with 230 × `observe_blend(0.0, 0.0, 0)` and 26 × `observe_blend(0.001*k, 0.002*k, 0)`. Then `standing(0.0, 0.0, 0) == (0.0, 0.0)` and `mission_standing(0.0, 0) == 0.0`.
- (b) A young window (5 samples at stage 3): `standing(0.0, 0.3, 3) == (0.0, 0.5)`.
- (c) Ties: stage 1 holds 32 × 0.0, 16 × 0.02 and 16 × 0.5. Then `standing(0.02, 0.02, 1)[0] == 0.5` exactly.

Mutations (anchor `curriculum-reads-every-medium`; suites `("test_search::test_a_score_of_zero_stands_at_zero",)`):
- `standing-ranks-ties-at-or-below`: find `    return float(np.mean(np.asarray(window, float) < x))` and replace `<` with `<=`. Breaks (c).
- `zero-takes-the-young-window-half`: find `    if x <= ZERO_SCORE:\n        return 0.0\n    if len(window) < min_n:` and replace it with `    if len(window) < min_n:`. Breaks (b).

Existing tests: `test_the_island_objective_takes_its_weight_back` (2546-2638) is unchanged. Its values were checked by hand: 0.021/0.059 are not tied with the window, and `mission_standing(0.001,1)` gives 1/64 < 0.2.

Acceptance: the one-function runner gives `FAILURES []`; both mutations `caught`; cheap suites pass.

### L2 — floor-competence (islands.py; test_search.py; mutate.py). No dependencies.

`dytiscidae/evolution/islands.py`, after `TRIPHIBIAN_FLOOR` (around line 135):
```python
#: Competence under which a medium counts as not done (ROADMAP 2026-10-10,
#: revised T1).  0.012 = curriculum.WEAKEST_BARS[1] = 2 x TRIPHIBIAN_FLOOR.
#: Certified against held-still machines on arch49 by experiments/no_model_gate
#: (results_arch49_table.md): water elites 64/229 vs still 3/229, land 51/229
#: vs 0/229.  NOT certified for air there (elites 6/229 vs still 11/229: a
#: leak); kept at 0.012 until the paired air score's gate says otherwise.
COMPETENCE_FLOOR = {"air": WEAKEST_BARS[1], "water": WEAKEST_BARS[1],
                    "land": WEAKEST_BARS[1]}

def island_media(island: str) -> tuple:
    """The media an island's score reads; every medium for an unknown name."""
    return tuple(ISLANDS.get(island, {}).get("domains", ("air", "water", "land")))

def below_competence_floor(island: str, result) -> bool:
    """True when the design clears the floor in none of its island's media."""
    segs = getattr(result, "segments", {}) or {}
    for d in island_media(island):
        if (d in segs
                and float(getattr(segs[d], "competence", 0.0)) >= COMPETENCE_FLOOR[d]):
            return False
    return True
```
Bar sourcing: a typed constant that reuses `WEAKEST_BARS[1]`, with its provenance in the comment. It is not read from `experiments/` at run time, because `runs/` and `experiments/` results are absent in a fresh container (docs/PORTING.md) and a run's selection must not depend on a file outside the package.

Test `test_a_design_competent_in_nothing_its_island_reads_is_below_the_floor` (CPU). Anchor: after `test_the_triphibian_island_pays_the_weakest_medium`. Build fake results as at test_search.py:2662-2681 (Seg with `competence`; `segments` keyed by str). Cases:
- water island: water 0.011 is True; water 0.012 is False; water 0.0 with air 0.5 is True.
- amphibian: water 0.0, land 0.02 is False.
- triphibian and generalist: 0.011 in all three is True; land 0.02 alone is False.
- water island with no water segment is True.
- `COMPETENCE_FLOOR == {m: WEAKEST_BARS[1] for m in ("air","water","land")}`.

Mutations (anchor `triphibian-curriculum-is-the-shared-ladder`; suite is the test above):
- `competence-floor-at-zero`: find `                and float(getattr(segs[d], "competence", 0.0)) >= COMPETENCE_FLOOR[d]):` and replace `COMPETENCE_FLOOR[d]` with `0.0`.
- `competence-floor-reads-every-medium`: find `    for d in island_media(island):\n        if (d in segs` and replace it with `    for d in ("air", "water", "land"):\n        if (d in segs`.

Existing tests: none changed. `island_score` is untouched.

### L3 — floor-tie (archive.py; test_search.py; mutate.py). No dependencies.

`dytiscidae/evolution/archive.py`:
- `Elite`: add `at_floor: bool = False` after `objectives` (line 79), with a one-line docstring. It must be a plain default and not `field(...)`: a plain default is a class attribute, so elites unpickled from older runs read False.
- `add(...)` and `would_add(...)`: add a keyword `at_floor: bool = False`, passed into `Elite(..., at_floor=bool(at_floor))`.
- `_verdict` (257-293):
  - right after `if not front: return "new", [cand]` insert
    ```python
            # F1 (ARCH51_SPEC L3): a design below the competence floor fills an
            # empty cell and nothing else.
            if cand.at_floor:
                return "rejected", None
    ```
  - replace lines 281-284 with
    ```python
            ranked = [e for e in front if not e.at_floor]
            if any(self._dominates(e.objectives, obj) for e in ranked):
                return "rejected", None
            # F2: a design above the floor displaces every member below it.
            kept = [e for e in ranked if not self._dominates(obj, e.objectives)]
    ```
- What "zero" means numerically is decided upstream (L5): `at_floor = below_competence_floor(...) or blend <= ZERO_SCORE`. The archive receives a bool and reads no number.

Test `test_margins_alone_cannot_fill_a_cell` (CPU). Anchor: after `test_cells_hold_a_pareto_front_not_a_weighted_sum`.
- An empty cell accepts a floor candidate as `"new"`.
- A floor challenger `[0.0, 3.0, 2.0]` against a ranked incumbent `[0.5, 1, 1]` returns `"rejected"`, and the front is unchanged.
- A floor challenger with higher margins against a floor incumbent returns `"rejected"`.
- A ranked `[0.3, 1, 1]` entering a cell held by a floor member `[0.0, 3, 2]`: the front is `[ranked]` and the representative is it.
- `would_add(..., at_floor=True)` equals `add`'s status.
- `at_floor` omitted reproduces the old verdicts (re-run one case from 840-845).

Mutations (anchor `island-archive-read-through-the-merge`):
- `floor-candidate-joins-the-front`: find `        if cand.at_floor:\n            return "rejected", None\n` and replace with empty.
- `floor-member-survives-a-ranked-entry`: find `        kept = [e for e in ranked if not self._dominates(obj, e.objectives)]` and replace `ranked` with `front`.

Existing tests: `test_cells_hold_a_pareto_front_not_a_weighted_sum` (40-design line including obj0 = 0.0) uses the default False, so it is unchanged.

### L4 — tier2-flag (curator.py; test_search.py; mutate.py). No dependencies.

`dytiscidae/evolution/curator.py`:
- `record_promotion` (400-406): delete `elite.fitness = min(elite.fitness, tier2_fitness)` and its comment. Add:
  ```python
          # The Tier-2 outcome is a flag beside the score, not a cap on it
          # (ROADMAP 2026-10-10 T1'/N4): a cap made the best design the worst parent.
          elite.meta["tier2_fitness"] = float(tier2_fitness)
          elite.meta["tier2_gen"] = int(self.archive.generation)
  ```
- New method after `should_promote`:
  ```python
      def promotion_candidates(self, k: int = 3) -> list[Elite]:
          """The top ``k`` representatives by ``fitness`` not yet verified (tier < 2)."""
          pool = [e for e in self.archive.cells.values() if e.tier < 2]
          return sorted(pool, key=lambda e: -e.fitness)[:k]
  ```
- Who reads the flag:
  - promotion: `should_promote` (`tier >= 2` returns False, as now) and `promotion_candidates`, which loop.py uses from L5;
  - cohort: `select_cohort(require_verified=True)`, as now.
- Who does not read it (state this in the `record_promotion` docstring): parent weight (`select_parent`), migration (`Archipelago.emigrants`, islands.py:315-320), prune (`Curator.prune`), and `Archive.best`. All of these keep the uncapped `fitness`.
- Consequence to record in ROADMAP (L12a): `should_promote` compares against `archive.best.fitness`. While every elite of an island is at the floor, `top = 0` and no elite promotes, so the critic gets no labels from that island.

Test `test_a_tier2_failure_does_not_make_the_best_design_the_worst_parent` (CPU). Anchor: after `test_curator_quarantines_repeat_exploits`.
- Five cells with fitness 0.9 / 0.8 / 0.7 / 0.6 / 0.5.
- `record_promotion(best, 0.1)`: then `best.fitness == 0.9`, `best.tier == 2`, `best.meta["tier2_fitness"] == 0.1`, `archive.best is best`, `should_promote(best) is False`.
- `Archipelago(n_migrants=2)` with this archive registered: `emigrants(name)[0] is best.genome`.
- After also promoting 0.8: `[e.fitness for e in promotion_candidates(3)] == [0.7, 0.6, 0.5]`.

Mutations (anchor `bandit-without-an-exploration-floor`):
- `tier2-caps-fitness-again`: find `        elite.meta["tier2_fitness"] = float(tier2_fitness)\n` and replace it with the same line plus `        elite.fitness = min(elite.fitness, tier2_fitness)\n`.
- `promotion-pool-keeps-verified-elites`: find `        pool = [e for e in self.archive.cells.values() if e.tier < 2]` and replace it with `        pool = list(self.archive.cells.values())`.

### L5 — blend-wiring (loop.py; test_search.py; mutate.py). Depends on L1-L4.

`dytiscidae/evolution/loop.py`:
- Imports (56, 58): add `ZERO_SCORE` from `.curriculum` and `below_competence_floor` from `.islands`.
- `SearchConfig.mission_weight` (169): the line must read exactly `    mission_weight: float = 0.0`. Docstring: "0.30 from 2026-09-01 through arch50; 0.0 from ARCH51_SPEC: `mission_fraction` is zero for 97-100% of every island's window (ROADMAP N2), so the term decided nothing and cost 0.3 of the scale. The field stays for resumes and the job path."
- `_score_candidate`:
  - after `w = state.curriculum.handover(sr.stage)` (1156):
    ```python
        floored = below_competence_floor(state.island, result)
        if floored:
            isl_q, cur_q, mis_q = 0.0, 0.0, 0.0
    ```
  - after `base = ...` (1165): `at_floor = bool(floored or base <= ZERO_SCORE)`.
  - add `"at_floor": at_floor` to the returned dict (1174).
- `_place` (1270): the line must read exactly
  `    status = state.archive.add(genome, fit, bd, meta, tier=result.tier, objectives=obj, at_floor=sc["at_floor"])`.
  Also add `"at_floor": bool(sc["at_floor"])` to `meta["score_parts"]` (1245-1249).
- `_dry_status` (1311): `    return state.archive.would_add(sc["fit"], sc["bd"], sc["obj"], at_floor=sc["at_floor"])`.
- `_verify_and_label` (2715): `    for elite in curator.promotion_candidates(3):`.
- Plan-digest consequence for the job path: hyperparameters not in the plan come from `SearchConfig` defaults. A plan that does not set `mission_weight` therefore hashes the same before and after this commit while scoring differently. Every job-path plan from arch51 on sets `--set mission_weight=0.0` explicitly, and the commit in provenance separates the rest. Same open issue as ROADMAP B1.

Test `test_the_scalar_stands_at_zero_below_the_competence_floor` (CPU). Anchor: after `test_promotion_spends_refinement_and_keeps_what_it_buys`.

Setup:
- Monkeypatch these `loop` module attributes and restore them in `finally`: `episode_features` (returns `np.zeros(16)`), `behaviour_descriptor` (returns `np.array([0.5, 0.5])`), `objectives` (returns `np.array([0.0, 3.0, 2.0])`), `_meta_light` (returns `{}`), `critic_features` (returns `np.zeros(4)`), `_meta` (returns `{"air": 0.0, "water": r.water, "land": 0.0}`).
- `state = SimpleNamespace(config=SearchConfig(), descriptors=None, judge=Judge(quantile=0.9, update_every=50), island="water", archive=Archive([("x",0,1,5),("y",0,1,5)]), curriculum=curriculum_for("water"), critic=None, curator=Curator(archive), telemetry=<object with .event(d) appending d>, scout=None, shared=None)`.
- Fake `Res(water)`: `segments={"water": Seg(water)}`, `mission_fraction=0.0`, `feasible=True`, `transitions=None`, `tier=1`, `exploit=""`, `wall_time=0.0`, `notes=[]`, `eval_seed=0`.
- Window at stage 0: 40 × `(0.0, 0.0)` plus 24 × `(0.001k, 0.002k)`.

Checks:
- (a) `_score_candidate(state, None, Res(0.005), None, commit=False)`: `at_floor` True, `isl_q == cur_q == 0.0`, `fit == 0.0`, `obj[0] == 0.0`.
- (b) `Res(0.2)`: `at_floor` False, `mw == 0.0`, `|base - (w*isl_q + (1-w)*cur_q)| < 1e-12`, `base > 0.5`.
- (c) Put a ranked incumbent `[0.3, 1, 1]` in cell `cell_of([0.5,0.5])`. Then `_place(state, "g", None, Res(0.005), None, None, ["scale"]) == "rejected"`, the front is still the incumbent alone, and `_dry_status(state, None, Res(0.005), None) == "rejected"`.

If `Judge`, `observe_domains` or the empty `TransitionSet` needs more fields, add them to the fakes. Do not change production code for the test.

Test `test_verification_offers_designs_not_yet_verified` (CPU). Same anchor, after the test above.
- Five cells with genomes "a".."e" and fitness 0.9 … 0.5. `cur.evaluations = 1000`. `cur.record_promotion` on "a" and "b".
- Monkeypatch `loop.build` to raise `RuntimeError(f"built {g}")`.
- Call `loop._verify_and_label(state, 0, None, np.random.default_rng(0))`.
- The genomes named in the stub telemetry's `kind == "error"` events must be exactly {"c", "d", "e"}. The old code tries only "c".

Mutations (anchor `promote-drops-tier1-media`):
- `score-ignores-the-competence-floor`: `    floored = below_competence_floor(state.island, result)\n` becomes `    floored = False\n`. Caught by (a).
- `mission-weight-back-in-the-blend`: `    mission_weight: float = 0.0\n` becomes `    mission_weight: float = 0.30\n`. Caught by (b).
- `place-forgets-the-floor`: `objectives=obj, at_floor=sc["at_floor"])` becomes `objectives=obj)`. Caught by (c).
- `dry-run-forgets-the-floor`: `sc["obj"], at_floor=sc["at_floor"])` becomes `sc["obj"])`. Caught by (c).
- `verify-offers-verified-designs`: `    for elite in curator.promotion_candidates(3):` becomes `    for elite in sorted(archive.cells.values(), key=lambda e: -e.fitness)[:3]:`. Caught by the second test.

Existing tests that may break:
- `test_promotion_spends_refinement_and_keeps_what_it_buys` (3093; GPU; generalist, `segment_seconds=0.5`) asserts "verification still promotes". If every seed is at the floor, `top = 0` and nothing promotes.
  - Do not weaken the assertion. Print the seeds' best own-medium competence, then raise that test's `segment_seconds`, or change its `islands`, to the smallest setting where at least one seed clears 0.012. Record the reason in the test.
- `test_the_loop_wires_every_layer_together` (4325-4333): the islands-differ check passes on genome identity (seeds are filed as `g.copy()`); verify.
- Run these single functions after the lane: `test_the_loop_wires_every_layer_together`, `test_promotion_spends_refinement_and_keeps_what_it_buys`, `test_a_film_reproduces_the_scored_experiment`, `test_grpo_rollouts_never_reach_the_archive` (GPU).

### L6 — cli-mission-weight (ops/run.py; tests/test_physics.py; mutate.py). No dependencies; merge together with L5.

- `ops/run.py:1127`: `    p.add_argument("--mission-weight", type=float, default=0.0,`. Help text: "0.30 through arch50; 0.0 from ARCH51_SPEC (mission_fraction is zero for 97-100% of every window, ROADMAP N2)."
- `test_physics.py::test_the_search_cli_defaults_are_the_stored_run_configuration` (around line 195): add `check("mission weight 0 (ARCH51_SPEC)", a.mission_weight == 0.0)` and `check("and the CLI agrees with SearchConfig", search_config_from_args(a, True).mission_weight == SearchConfig().mission_weight)`. Amend the docstring: the defaults are the stored-run configuration except refine steps (M1) and the mission weight (ARCH51).
- Mutation `cli-mission-weight-default-back` (anchor `search-cli-refine-default-two`): `default=0.0,` becomes `default=0.30,` on that line. Suite `("test_physics::test_the_search_cli_defaults_are_the_stored_run_configuration",)`.

### L7 — ppo-return-split, N10's fields (learning/ppo.py; tests/test_ppo.py; mutate.py). No dependencies.

The class is `RolloutBuffer` (ppo.py:414); there is no `PPOBuffer`.

`RolloutBuffer.__init__`: `self.decomposition: dict = {}`.

In `build()` (449-484): before the loop, `parts: dict = {}`. Inside the loop, after the shaping block (after 466), write exactly:
```python
            disc = self.gamma ** np.arange(n)
            term = np.zeros(n)
            term[-1] = t.terminal_reward / scale.get(t.tag, 1.0)
            shp = rew - term
            p = parts.setdefault(t.tag, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            p[0] += 1
            p[1] += float(disc @ shp)
            p[2] += float(disc @ term)
            p[3] += float(disc @ rew)
            p[4] += abs(float(disc @ shp))
            p[5] += abs(float(disc @ term))
```
After the loop:
```python
        self.decomposition = {
            tag: {"n": c, "shaping_return": s / c, "terminal_return": tm / c,
                  "return": r / c, "abs_shaping": a / c, "abs_terminal": b / c}
            for tag, (c, s, tm, r, a, b) in sorted(parts.items())}
```

`ppo_update`:
- add `"return_by_tag": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in getattr(buffer, "decomposition", {}).items()}` to the normal return (651);
- add `"return_by_tag": {}` to the skipped return (626).

The loop already spreads `**info` into the `ppo` event (loop.py:1801-1807), so no loop edit is needed. N10's share is `abs_shaping / (abs_shaping + abs_terminal)` per tag.

Test `test_the_return_splits_into_shaping_and_terminal` (test_ppo.py). Anchor: after `test_potential_shaping_telescopes_to_nothing`.
- Two tags with hand-made trajectories; build them as that test does. `gamma` 0.99, shaping 0.2.
- Per tag:
  - (1) `|shaping + terminal - return| < 1e-12`;
  - (2) `return` equals a from-scratch Σγ^i·rew_i, with rew rebuilt in the test (scale = max(std of the tag's terminals, 0.05));
  - (3) `shaping_return == mean(-0.2·phi[0])` within 1e-10 (telescoping, Φ(terminal) = 0);
  - (4) `terminal_return == mean(γ^(n-1)·terminal/scale)`.
- (5) `ppo_update(...)["return_by_tag"]` has both tags. Use the suite's torch-skip pattern.

Mutations (anchor `grpo-rows-renormalised-with-the-batch`; suite `("test_ppo::test_the_return_splits_into_shaping_and_terminal",)`):
- `return-split-drops-the-discount`: `            p[2] += float(disc @ term)\n` becomes `            p[2] += float(term.sum())\n`.
- `return-split-omits-shaping`: `            shp = rew - term\n` becomes `            shp = np.zeros(n)\n`.

Acceptance: `tests/test_ppo.py | tail -2` gives `shared PPO checks passed`. Check that no existing test asserts the exact key set of `ppo_update`'s dict; if one does, add the key there.

### N11 — re-evaluation of the archive (L8-L11)

Pre-registered (ROADMAP N11, verbatim): "*Prediction for the first arm with N5 + N11:* auditor held-out retention >= 0.5 in every medium (arch49: 0.09-0.33) at the same wall per generation; Tier-1→Tier-2 Spearman >= 0.3 (arch49: 0.17/0.18). *Falsified* if retention stays below 0.35 in any medium. Comparability: archive scores are not comparable across it (a median of draws, not a draw)."

Boundary sentence (L12a):
> From commit <N11 merge sha>, with `reeval_per_generation > 0`, an elite's recorded `fitness`, `objectives`, per-medium competences, `eval_seed` and `gen` are those of its lower-median draw over up to `reeval_depth` (8) draws, ordered by blend. Before, they were one draw. Archive scores are not comparable across it.

Semantics:
- Each generation, `k = reeval_per_generation` of the `batch` slots (arch51: 4 of 16) re-run uniformly chosen non-tainted **representatives** of the visited island.
- Each re-run uses a fresh seed from `state.rng`, never `meta["eval_seed"]`, the elite's own `meta["policy"]`, identification on, and the current shared network.
- The buffer lives on the elite (`meta["draws"]`). The score, the objective vector and the scored meta keys are those of the lower-median draw by `base` (blend). The lower median is an actual draw, so the film reproduces it: viz/film.py reads `eval_seed`, `gen`, `air`/`water`/`land`, `mobility_basis`, `policy` and `scored_with_shared_policy` from meta (film.py:218-283), and all of them are swapped together.

Interactions:
- **Dominance.** Challengers face members' median vectors. After each re-evaluation the cell's front is re-settled by dominance among its own members, then by F2 (members at the floor go if any member is above it). The representative is re-taken as max obj0. A re-evaluation never inserts, never trims by crowding, and never moves an elite to another cell (`meta["features"]` is never swapped).
- **Rebin.** `rebin` keeps representatives only (archive.py:345) and re-projects from `meta["features"]`. Re-evaluations target representatives only, so their buffers survive a rebin. Non-representative members are dropped as today, and N11 does not change that. Collisions rank by `_weakest_medium` (median draw's competences), then by `fitness` (median draw's).
- **Windows.** Re-evaluations are scored with `commit=False`: they feed no curriculum window, descriptor buffer, judge or stage seed. They do feed the PPO buffer, because they are on-policy rollouts.

### L8 — draw-buffer (archive.py; test_search.py; mutate.py). Depends on L3.

`Archive`:
- class attribute `draw_depth: int = 8` (comment: Extract-ME depth 8, arXiv 2502.06585; set from `SearchConfig.reeval_depth`).
- methods:
```python
    def record_draw(self, elite: Elite, draw: dict, depth: int | None = None) -> dict:
        """One more evaluation of ``elite``; its score becomes its lower-median draw.
        ``draw`` = {"base", "fit", "obj": [3], "at_floor", "meta": {scored keys}}."""
        draws = elite.meta.get("draws")
        if not draws:
            draws = [{"base": float((elite.meta.get("score_parts") or {}).get("blend", elite.fitness)),
                      "fit": float(elite.fitness),
                      "obj": [float(x) for x in elite.objectives],
                      "at_floor": bool(elite.at_floor), "meta": {}}]
            elite.meta["draws"] = draws
        draws.append(draw)
        d = int(depth if depth is not None else self.draw_depth)
        if len(draws) > d:
            del draws[: len(draws) - d]
        order = sorted(range(len(draws)), key=lambda i: draws[i]["base"])
        med = draws[order[(len(draws) - 1) // 2]]
        elite.fitness = float(med["fit"])
        elite.objectives = np.asarray(med["obj"], float)
        elite.at_floor = bool(med.get("at_floor", False))
        elite.meta.update(med.get("meta") or {})
        removed = self._settle(elite.cell)
        return {"n": len(draws), "median_base": float(med["base"]), "removed": removed}

    def _settle(self, cell) -> list:
        """Re-establish a front among its own members after a score changed."""
        front = self.fronts.get(cell) or []
        if not front:
            return []
        kept = [e for e in front
                if not any(self._dominates(o.objectives, e.objectives) for o in front if o is not e)]
        kept = [e for e in kept if not e.at_floor] or kept
        removed = [e for e in front if not any(e is k for k in kept)]
        self.fronts[cell] = kept
        self.cells[cell] = self._representative(kept)
        return removed
```
- `export_json` (452-465): write `meta["draws"]` as `[{"base","fit","eval_seed","gen","air","water","land"}]` per draw, not the full dicts (each holds policy weights and a basis).

Test `test_a_cells_score_is_the_median_of_its_draws` (CPU). Anchor: after L3's test. Helper `d(b, s) = {"base": b, "fit": b, "obj": [b, 1, 1], "at_floor": False, "meta": {"eval_seed": s, "water": b}}`.
- (a) Elite A is added with fit 0.9 and obj `[0.9, 1, 1]`. Record `d(.2, 11)` and `d(.5, 12)`. Then `A.fitness == 0.5`, `A.meta["eval_seed"] == 12`, `A.objectives[0] == 0.5`.
- (b) A fresh elite (0.9) plus `d(.2)`: `fitness == 0.2` (the lower median is an actual draw).
- (c) After 10 more draws, `len(meta["draws"]) == 8`.
- (d) A cell holds A2 `[.9, 1, 1]` and B `[.6, 2, 2]`. Record `d(.5)` and `d(.4)` on A2. Then `front == [B]` and `cells[cell] is B`.

Mutations (anchor: after L3's last; suite is the test above):
- `representative-is-the-max-draw`: `        med = draws[order[(len(draws) - 1) // 2]]` becomes `        med = draws[order[-1]]`.
- `draw-buffer-is-unbounded`: `        if len(draws) > d:\n            del draws[: len(draws) - d]\n` becomes empty.
- `rescore-leaves-the-front-alone`: `        removed = self._settle(elite.cell)` becomes `        removed = []`.

### L9 — reeval-loop (loop.py; test_search.py; mutate.py). Depends on L5 and L8.

`loop.py`:
- `SearchConfig`, after `draw_per_candidate`:
  - `reeval_per_generation: int = 0` ("0 = off, every run to arch50; arch51: 4 of batch 16, Extract-ME's 25%, ROADMAP N11");
  - `reeval_depth: int = 8`.
- Module constant `SCORED_KEYS`: the meta keys a draw carries.
  - Include: `"air","water","land","mission_fraction","energy_margin","tier","max_depth","air_gates","eval_seed","gen","scored_with_shared_policy","mobility_rank","mobility_cond","mobility_underdetermined","mobility_axes","mobility_basis","policy","objectives","score_parts","stage","stage_name","rungs","judged","ladder_measurements","critic_discount","critic_features"`, plus every other key `_meta` (978-1069) returns that reads `result` or `ctrl`. Read the function and list them.
  - Exclude: pheno-only keys (mass … feasible, margin, worst_check), `features`, `novelty`, `scout_features`, `island`, `policy_promoted`, `policy_refined`, `tier1_5`, `tier2_*`, `draws`.
- `run_search`: after archives exist or are restored, set `a.draw_depth = int(cfg.reeval_depth)` for every island archive.
- Build phase (1663-1695):
  - before the child loop:
    ```python
            picks = []
            if int(cfg.reeval_per_generation) > 0:
                pool_re = [e for c, e in archive.cells.items() if c not in archive.tainted]
                n_re = min(int(cfg.reeval_per_generation), len(pool_re), max(int(cfg.batch) - 1, 0))
                if n_re:
                    picks = [pool_re[i] for i in rng.choice(len(pool_re), size=n_re, replace=False)]
    ```
    Change the child loop to `for _ in range(cfg.batch - len(picks)):`. Then:
    ```python
            for elite in picks:
                built.append((elite.genome.copy(), elite.meta.get("policy"), True, ["reeval"], elite, int(rng.integers(1 << 30))))
    ```
  - With `reeval_per_generation == 0`, nothing is drawn from `rng`, so runs with the feature off reproduce bit for bit.
- Placement loop (1726-1748): after the evaluation counters (1734-1737), `if operators == ["reeval"]: _reevaluate(state, parent, pheno, result, ctrl); continue`.
- Guard every `curator.credit(operators, ...)` in the generation, including 1718-1719 and 1729, with `if operators != ["reeval"]`.
- `_score_candidate`: new keyword `at_cell=None`; `cell = state.archive.cell_of(bd) if at_cell is None else tuple(at_cell)`.
- New function after `_place`:
```python
def _reevaluate(state, elite, pheno, result, ctrl) -> str:
    """One more draw of an archived representative (ROADMAP N11).  Feeds no window."""
    gen = int(state.archive.generation)
    if result.tier == 0 or not any(m is elite for m in state.archive.fronts.get(elite.cell) or []):
        state.telemetry.event({"kind": "reevaluate", "gen": gen, "island": state.island,
                               "cell": list(elite.cell), "status": "orphan"})
        return "orphan"
    if result.exploit:
        state.curator.quarantine(elite.descriptor, result.exploit, elite.genome)
        state.telemetry.event({... "status": "exploit", "reason": result.exploit})
        return "exploit"
    sc = _score_candidate(state, pheno, result, elite, commit=False, at_cell=elite.cell)
    meta = _meta(pheno, result, ctrl)
    # same keys _place writes (1213-1249): gen, scored_with_shared_policy, objectives, island-free
    # stage/stage_name/rungs/judged/ladder_measurements/critic_discount/critic_features/score_parts
    ...
    draw = {"base": sc["base"], "fit": sc["fit"], "obj": [float(x) for x in sc["obj"]],
            "at_floor": bool(sc["at_floor"]), "meta": {k: meta[k] for k in SCORED_KEYS if k in meta}}
    first = (elite.meta.get("draws") or [{}])[0].get("meta", {}).get("eval_seed", elite.meta.get("eval_seed"))
    before = float(elite.fitness)
    out = state.archive.record_draw(elite, draw)
    state.telemetry.event({"kind": "reevaluate", "gen": gen, "island": state.island,
        "cell": list(elite.cell), "status": "recorded", "draws": out["n"],
        "fitness_before": round(before, 4), "fitness_after": round(float(elite.fitness), 4),
        "eval_seed": meta.get("eval_seed"), "first_eval_seed": first,
        "median_eval_seed": elite.meta.get("eval_seed"), "removed": len(out["removed"]),
        **{m: meta.get(m) for m in ("air", "water", "land")}})
    return "recorded"
```
- `_place`: when `cfg.reeval_per_generation > 0`, set `meta["draws"] = [{"base": base, "fit": fit, "obj": [float(x) for x in obj], "at_floor": bool(sc["at_floor"]), "meta": {k: meta[k] for k in SCORED_KEYS if k in meta}}]` right before `archive.add`.

Test `test_an_elite_is_re_measured_at_fresh_draws_and_kept_at_its_median` (GPU; guard with `needs_batched_evaluator`). Anchor: after L5's tests. Take the `run_search` configuration of `test_a_film_reproduces_the_scored_experiment`, then set `generations=4`, `batch=4`, `reeval_per_generation=2`. Asserts:
- (1) At least 2 `reevaluate` events with `status == "recorded"`.
- (2) Each has `eval_seed != first_eval_seed`.
- (3) Some elite has `len(meta["draws"]) >= 2`, and its `meta["eval_seed"]` equals the seed of the lower-median draw recomputed from `draws`.
- (4) Σ over islands of Σ `len(curriculum._recent[s])` equals the count of non-reeval Tier-1 evaluate events (while every window is < 256).
- (5) `viz.film.evaluate_on_film(elite_with_draws, tmp, film=False)` matches all three media within TOLERANCE.
- (6) Per generation, evaluate events plus reevaluate events plus tier0 rejects equal `batch`.

Mutations (anchor: after L5's last; suite is the test above):
- `reeval-reuses-the-stored-seed`: `["reeval"], elite, int(rng.integers(1 << 30))))` becomes `["reeval"], elite, int(elite.meta.get("eval_seed") or 0)))`. Caught by (2).
- `reeval-commits-to-the-window`: in `_reevaluate`, `commit=False, at_cell=elite.cell)` becomes `commit=True, at_cell=elite.cell)`. Caught by (4).
- `reeval-records-nothing`: `    out = state.archive.record_draw(elite, draw)` becomes `    out = {"n": 1, "median_base": 0.0, "removed": []}`. Caught by (3).

### L10 — reeval-cli (ops/run.py; tests/test_physics.py; mutate.py). Depends on L9.

- Add `--reeval-per-generation` (int, default 0) and `--reeval-depth` (int, default 8) next to `--placement-draws`.
- In `search_config_from_args`, add the lines `        reeval_per_generation=args.reeval_per_generation,` and `        reeval_depth=args.reeval_depth,`.
- In the CLI-defaults test, check defaults 0 and 8, and that `parse_args(["search","--reeval-per-generation","4"])` gives `search_config_from_args(...).reeval_per_generation == 4`.
- Mutation `cli-drops-reeval` (anchor: after `cli-mission-weight-default-back`): `        reeval_per_generation=args.reeval_per_generation,\n` becomes empty.
- Also run `test_physics::test_a_config_export_relaunches_the_same_search`.

### L11 — reeval-jobpath (adapters/trainers/search.py; tests/test_search_adapter.py; mutate.py). Depends on L9; parallel with L10.

- `_CONFIG_FIELDS`: the `"placement_draws",` line becomes `    "placement_draws", "reeval_per_generation", "reeval_depth",`.
- In `test_the_plan_translates_into_a_search_config` (test_search_adapter.py:214), a plan with `reeval_per_generation=4` gives `cfg.reeval_per_generation == 4`.
- Mutation `job-path-refuses-reeval` (anchor `child-inherits-c-environment`): `"placement_draws", "reeval_per_generation", "reeval_depth",` becomes `"placement_draws",`. With the names missing, the plan raises "unknown hyperparameter".

### L12 — docs, last (L12a: docs/ROADMAP.md, CLAUDE.md, docs/HYPERPARAMETERS.md; L12b: docs/index/FEATURES.yaml plus `index_gen write`)

L12a:
- ROADMAP: under N5 and N11, add "built <date>, <sha>", the two boundary sentences, and the Tier-2 label consequence (L4).
- CLAUDE.md "Comparability boundaries": add the ARCH51 boundaries.
- CLAUDE.md "Tests" table: no change.
- HYPERPARAMETERS.md rows:
  - `mission_weight` 0.0: typed; reason N2.
  - `COMPETENCE_FLOOR` 0.012: certified for water and land by no_model_gate arch49; air not certified.
  - `reeval_per_generation` 4 and `reeval_depth` 8: from Extract-ME, not swept.

L12b, FEATURES.yaml:
- Islands: add `evolution.islands.below_competence_floor`, `evolution.islands.island_media` and L2's test.
- QualityDiversity: add `evolution.archive.Archive.record_draw` and the L3, L8, L9 tests.
- Curation: add `evolution.curator.Curator.promotion_candidates` and the L4 test.
- Curriculum: add the L1 test.
- The PPO feature entry: add the L7 test.

---

## §B. Probes

Each probe is `experiments/<name>/run.py` with `results_arch49.json` beside it. It has a frozen README section (the ROADMAP prediction verbatim) before it runs, and a report at `runs/analysis_1010_failure_theory/N<k>_result.md` in REPORT_FORMAT.md's shape.
- Use `experiments/harness.py` for provenance (`git_commit`, `ExperimentResult`).
- One process, no pool.
- Before any GPU probe, run `systemctl --user list-units --type=service --state=running`. On 2026-10-10 arch50 was `inactive`.
- Launch with `systemd-run --user --unit <name> -p MemoryMax=3500M env MUJOCO_GL=disable PYTHONPATH=. DYTISCIDAE_KERNEL_DIR=<main>/mojo/build .venv/bin/python -u experiments/<name>/run.py …`, as no_model_gate does.

### N1 — still-twin fitness probe: `experiments/still_twin_fitness/run.py`

Frozen prediction (verbatim): "*Prediction (T1):* median twin `fit` >= 0.80 in every island; the twin is non-dominated or dominating in >= 60% of cells. *Falsified* if median twin `fit` < 0.5 in any specialist island, or non-dominated in < 30% of cells."
Also report against the ROADMAP's revised line: "What would move T1 back: N1 finding the still twin dominated in >= 70% of cells."

Tree:
- Run from a worktree at **5cc9db9**, i.e. before any N5 lane merges. L1 changes `standing`, which would make N1 measure the new floor.
- `git worktree add --detach ../n1-tree 5cc9db9`, with `PYTHONPATH=../n1-tree`.

Path: the batched path, `batchroll.evaluate_tier1_batch`. It is the path that wrote the archive and the windows. The single-machine path still differs in water with identification on (CLAUDE.md "Two evaluation paths"). Reuse `experiments/no_model_gate/run.py:measure` (257-343):
- `load_elites(run)`, the 229 elites of the merged archive;
- groups by `(gen, eval_seed)`, because arch49 shared its draw per generation;
- `plain_controller` plus `viz.film.control_laws` for the scoring network;
- arms: `elite` (own policy plus scoring network) and `still` (`Controller(params=env.held_still_params(), policy=None, bases=c.bases)`, `shared=None`);
- `identify_axes=False` (the stored bases), `seed=key[1]`, `segment_seconds` and `n_modes` from `run_provenance(run)["config"]`.

Restore:
- `D = pickle.load(open("runs/arch49/search_state.pkl","rb"))`: generation 299, 4820 evaluated.
- `cfg = SearchConfig(**{k: v for k, v in run_provenance(run)["config"].items() if k in SearchConfig.__dataclass_fields__})`. This gives `mission_weight` 0.30, as in arch49.
- Per elite:
  - `island = elite.meta["island"]`;
  - `arch = load_run_archive(run, island)[0]`, cached per island;
  - `ns = SimpleNamespace(config=cfg, descriptors=D["descriptors"], judge=D["judge"], critic=D["critic"], island=island, archive=arch, curriculum=D["curricula"][island])`;
  - `parent = SimpleNamespace(cell=arch.cell_of(elite.descriptor))`.
- Per arm, `sc = loop._score_candidate(ns, pheno, res, parent, commit=False)`. Nothing is written into any restored object.

Per elite, record:
- island, plan, `n_rotors`;
- per arm: `fit`, `base`, `isl_q`, `cur_q`, `mis_q`, `w`, stage, `obj`, and competences (air, water, land);
- twin vs the elite's front, `F = arch.fronts.get(arch.cell_of(elite.descriptor))`:
  - `nondom = not any(Archive._dominates(e.objectives, obj_twin) for e in F)`;
  - `dominates = any(Archive._dominates(obj_twin, e.objectives) for e in F)`;
- `would_add = arch.would_add(fit, bd, obj)` at the twin's own cell;
- reproduction of the elite arm: |competence − recorded| ≤ 0.005 per medium, as no_model_gate does.

Table per island:
- n;
- twin `fit` median [IQR];
- twin `isl_q` and `mis_q` medians;
- share of twins non-dominated or dominating;
- share dominated;
- `would_add != "rejected"` share;
- elite-arm `fit` median under the final windows;
- reproduction counts.

The outcome is decided by the frozen thresholds above. Specialist islands are air, water and land.

Wall: no_model_gate's 3 arms took 5736 s for these 229 elites (run_arch49.log), so 2 arms is about 3800 s, roughly 65 min. `_score_candidate` adds less than 1 s per elite.

### N6 — rotor authority through the basis: `experiments/rotor_authority/run.py`

Frozen prediction (verbatim): "*Prediction:* (i) scores air < 0.3 while (ii) scores >= 0.8 on the same body and seed, and the basis moves rotor speed by < 5% of top speed per unit coefficient. *Falsified* if (i) >= 0.6."

Bodies: arch49's eight island archives (`load_run_archive(run, island)`), deduplicated by `genome_id`, with `meta["n_rotors"] > 0`. There were 191 of 645 elites, counted 2026-10-10 from `runs/arch49/archive_*.json`. Any tree works: the probe reads stored meta and steps the env.

Part A, static, all rotor elites, seconds:
- `env = TriphibianEnv(build(genome), seed=eval_seed)`, used for `act_names`, `cpg.n` and `cpg.hi`.
- `B = MobilityBasis.bases_from_record(meta["mobility_basis"])` for "air", and for "water" if present.
- Assert `B.modes.shape[1] == 3*n + 1`, where the flat layout is `[amplitude(n), phase(n), offset(n), frequency]` (cpg.py:46-66).
- Rotor channels: `R = [i for i, a in enumerate(env.act_names) if a.endswith("_r")]`. Verify this naming; fly.py maps `a[:-2] + "_rot"`.
- Per elite:
  - `m_frac = max_k max_{i∈R} |modes[k, 2n+i]| / hi[i]`: fraction of top speed per unit coefficient;
  - `f_frac = max_{j<6} max_{i∈R} |(modes.T @ B.coeffs_for_twist(e_j))[2n+i]| / hi[i]`: at full intent, through `INTENT_AUTHORITY` 0.5;
  - `rotor_norm_share_k = Σ_{i∈R} modes[k,2n+i]²`.
- Report medians and p90 over elites.

Part B, dynamic, 24 bodies, stratified by `n_rotors` and island with `default_rng(0)`:
- First exclude bodies whose rotors cannot lift: `Σ_i bemt(spec_i, hi[i], 0, 0, AIR.rho, AIR.mu)[0] < 1.2·m·g`, as in fly.py:43-48. Report the excluded count.
- Setup, as in fly.py:99-105: `env.reset(Domain.AIR)`, then scatter, then task from `_scatter_seed(eval_seed, Domain.AIR)`, then `seg = env.rollout(8.0, domain=Domain.AIR)`. The score is `seg.competence`.
- This is the single-machine path. It is the only one that accepts a hand law in `env.cpg.command`. State that in the report, and note that air agrees across the two paths (`test_the_two_evaluation_paths_score_the_same_machine_the_same`).
- Arm (ii), direct on ctrl. `orig = env.cpg.command`. The new command is `orig(params, t)`, with rotor entries `R` overwritten by fly.py's cascade and mixer (fly.py:42-96), applied to the rotor subset only.
- Arm (i), through the basis. The command is `orig(B.command_params(params, B.coeffs_for_twist(intent), env.cpg.n), t)`, where `intent` (body frame, each clipped to [-1, 1]) is:
  - surge and sway: 0.5·(v_des − v);
  - heave: 0.5·(z0 − z) − 0.5·v_z;
  - roll and pitch: −1.0·angle − 0.3·rate;
  - yaw: 1.0·heading_err − 0.3·w_z.
  - Gains scaled by G ∈ {0.5, 1, 2}; report the best G per body, so bad gains are not the explanation.
- Seed: each body's own `eval_seed`.

Table:
- per body: (i)-best, (ii), `m_frac`, `f_frac`;
- shares of bodies with (i) < 0.3 and (ii) ≥ 0.8;
- the falsifier's count of bodies with (i) ≥ 0.6.

Wall: Part A under 1 min. Part B is 24 × 4 rollouts. Time one rollout first and cut bodies to fit 30 min, reporting the n that ran.

### N8 — random sampling against the archive: `experiments/random_sampling_control/run.py`

Frozen prediction (verbatim): "*Prediction:* coverage and summed competence within 15% of arch49's gen-37 island archives (the same evaluation count per island). *Falsified* if arch49 leads by > 30%."

Comparison base:
- arch49 islands reach about 300 evaluate events near gen 150, not gen 37. From events.jsonl: water 587, land 604 and air 611 evaluate events over gens 0-299; 90-94 per island by gen 37.
- Snapshots exist only at gens 0, 50, 100, 150, 200 and 250, and evaluate events carry no `features`, so a gen-37 archive cannot be rebuilt.
- Report the literal gen-37 comparison as NOT COMPARABLE. Run the parenthetical's intent against the snapshot nearest 300 evaluations per island: compute the generation from events.jsonl, which should be gen 150. Give gen 50 as context.

Prior:
- The run's own seeding distribution (loop.py:2516-2517): `seed_population(rng, k_ref)` plus `random_genome(rng)` in the run's 20:8 ratio, with `rng = default_rng(20261010)`.
- Draw until 300 bodies pass Tier-0 (k_ref = 20/28 of the draws). Report Tier-0 rejects. arch49's evaluate events also exclude Tier-0 rejects.

Scoring:
- `loop.evaluate_candidates(genomes, cfg, identify=True, spec=MissionSpec(), seeds=<rng.integers(1<<30) each>, shared=<arch49 scoring network at the comparison gen>, pool=None)` in batches of 16, with `cfg` from arch49's provenance.
- Load the network with `viz.film._load_network(path, template)`. The template is `SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM, hidden=64)` (loop.py:1596).
- Fitness is not used, so N8 does not depend on N5. The tree is free; record the commit.

Binning:
- `D = pickle.load(runs/arch49/search_state.pkl)`. `bd = D["descriptors"].project(episode_features(result, pheno))`.
- Per island, `axes = Archive.load(runs/arch49/archive_<island>.pkl).axes`, and `cell = Archive(axes).cell_of(bd)`.
- arch49 side: the elites of `snapshots/gen0150_<island>.json` re-projected from `meta["features"]` with the same `D["descriptors"]` into the same axes.

Per island, per cell, keep the max of the island's own competence:
- air, water, land: that medium;
- amphibian, aerial_diver, land_air: min of the two;
- triphibian: `triphibian_score([air, water, land])`;
- generalist: `mission_fraction`.

Coverage is filled cells / 625. Summed competence is the sum of kept values. Report each island and the pool of all eight; the ROADMAP does not say which decides, so give the per-island and pooled verdicts separately.

Wall: 1×16 measured 72.2 s per batch on the real path (CLAUDE.md pool section), so 19 batches is about 23 min, plus restore.

### N9 — MuJoCo Warp for the same-body work: `experiments/mjwarp_probe/run.py`

Frozen prediction (verbatim): "*Prediction:* the rigid-body part is >= 5x faster at 24 worlds; the fluid interop is the cost that decides it. *Falsified* if < 2x."

Step 0 is a download. The agent asks the user before installing (package name, source, size). Use a scratch venv outside the repo, e.g. `python3 -m venv ../mjw-venv && ../mjw-venv/bin/pip install mujoco-warp`. To verify:
- the PyPI name, or a git install from google-deepmind/mujoco_warp;
- compatibility with the repo's mujoco 3.11.0 (`.venv`);
- warp-lang on an RTX 3060 Laptop with 6 GB, shared with the Mojo kernel.

Step 1, feasibility, ≤ 10 lines (API names to verify):
```python
import sys, time, mujoco, warp as wp, mujoco_warp as mjw
mjm = mujoco.MjModel.from_xml_path(sys.argv[1]); mjd = mujoco.MjData(mjm)
m = mjw.put_model(mjm); d = mjw.put_data(mjm, mjd, nworld=24)
mjw.step(m, d); wp.synchronize()                      # compile
t = time.perf_counter()
for _ in range(300): mjw.step(m, d)
wp.synchronize(); print("24 worlds x 300 steps", time.perf_counter() - t, "s")
```
The input XML is `core.mjcf.build_model_xml(build(elite.genome))[0]`, written to the scratch dir. Choose the elite with median `dof` among arch49's 229.
- Pass: it compiles and steps without error.
- Fail: record the first unsupported feature. Candidates: rotor velocity servos (mjcf.py:682-698), universal `_f` joints, the beach terrain geoms in `scene_xml`.

Step 2, timing on the same body, 3 repeats, medians, compile time reported separately:
- (A) Current identification: `batchroll.identify_batch([env], Domain.AIR, seed=s)` (batchroll.py:650-724). That is 24 probes × 2 signs × 300 steps, sequential with a GPU fluid batch of 1. Also time a 16-copy batch, which is the per-generation shape.
- (B) CPU rigid only: 48 × 300 `mj_step` of the same model, no fluid.
- (C) MJWarp rigid only: `nworld` ∈ {24, 48}, 300 steps. Identification needs 48 rollouts (24 probes × 2 signs), all from one snapshot.
- (D) Fluid interop, estimated rather than measured: `BatchedFluid` over 48 envs of that body for 300 steps (launch plus finish), plus measured per-step `.numpy()` copies of `xpos`, `xmat`, `xipos`, `cvel` for 48 worlds, plus the upload of `xfrc_applied`.

Rigid speedup = B/C at 24 worlds; the prediction is read on it. Report (D)/(C) as the interop share.

What the current identification code assumes about CPU MuJoCo (file:line, 5cc9db9):
1. `TriphibianEnv.snapshot`/`restore` (triphibian.py:1340-1348): host copies of `qpos`, `qvel`, `time`, then `mj_forward`.
2. `step_batch` (batchroll.py:595-638): `e.data.ctrl[...] = e.servo_command(...)`, `e.data.xfrc_applied[:] = 0`, and one `e._mj.mj_step(e.model, e.data)` per env (625).
3. `BatchedFluid.launch`/`finish` (batchroll.py:220-575): reads `xpos`, `xmat`, `xipos`, `cvel` on the host. Writes `xfrc_applied`, and **writes the model arrays `body_mass` and `body_inertia` every step** (added mass, around 529-531). To verify: whether MJWarp allows per-world model fields.
4. Rotors: `RotorBatch.apply` with `(model, data)` per env (batchroll.py:566-571; rotor.py:384).
5. `BatchedPower.step` reads actuator state per env (batchroll.py:627-629).
6. `e.body_twist()` uses `mujoco.mj_objectVelocity` (triphibian.py:1352-1358).
7. The divergence guard `e.root_pos()` and the QACC reset semantics (CLAUDE.md hard rules). MJWarp's behaviour on NaN is to verify.
8. Mojo panel arrays are rebased per env by `body_id`. N worlds of one body need N panel copies (memory, to verify).

Wall: step 1 about 15 min after install; step 2 about 15 min.

---

## §C. Execution order and dispatch

Order:
- **Wave 1, parallel:** L1, L2, L3, L4, L6, L7, plus probes N1 (worktree at 5cc9db9), N6 and N8 (any tree), and N9 (after the user approves the download).
- **Wave 2:** L5 (needs L1-L4 merged; merge L6 with it) and L8 (needs L3).
- **Wave 3:** L9 (needs L5 and L8).
- **Wave 4, parallel:** L10 and L11.
- **Wave 5:** L12a and L12b.
- Merge each lane into `failure-theory-1010` in that order. After each merge, re-run that lane's gate on the merged tree.
- GPU probes and GPU tests share one 6 GB GPU. Run at most one GPU process at a time.

Context pack per lane (only this, nothing else):
- "Implement `docs/ARCH51_SPEC.md` §A <lane>, exactly as written. Files you may edit: <the lane's three>. Open the cited lines before editing. Follow CLAUDE.md 'Tests', 'Whether a test is worth anything' and 'Before adding anything'. Gate: <the lane's commands>. Return STATUS, FILES, and the gate output lines."
- Probe agents get "§B N<k>" plus REPORT_FORMAT.md, and return STATUS, FILES, OUTCOME and the one decisive number.

Gate per lane, scoped:
1. Each new test function through the one-function runner prints `FAILURES []`. GPU tests only on the GPU machine.
2. Each new mutation id passes `tools/mutate.py --only <id>` as `caught`.
3. The cheap suites pass.
4. `index_gen.py write`, then `check` (exit 0), then `test_index.py`.
5. Extra per lane:
   - L5: the four GPU functions listed under L5;
   - L7: `test_ppo.py | tail -2` gives `shared PPO checks passed`;
   - L6 and L10: `test_physics.py 2>&1 | grep -ciE '\[fail'` gives `0`.

Full canary before arch51, on the merged tree:
```bash
for s in index architecture domain application search_adapter adapters worker; do PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1; done
PYTHONPATH=. .venv/bin/python tests/test_ppo.py 2>&1 | tail -2          # shared PPO checks passed
PYTHONPATH=. .venv/bin/python tests/test_physics.py 2>&1 | grep -ciE '\[fail'   # 0
setsid nohup env PYTHONPATH=. .venv/bin/python tests/test_search.py > runs/canary_a51_search.log 2>&1 &   # tail -2: all search-machinery checks passed
for id in standing-ranks-ties-at-or-below zero-takes-the-young-window-half competence-floor-at-zero competence-floor-reads-every-medium floor-candidate-joins-the-front floor-member-survives-a-ranked-entry tier2-caps-fitness-again promotion-pool-keeps-verified-elites score-ignores-the-competence-floor mission-weight-back-in-the-blend place-forgets-the-floor dry-run-forgets-the-floor verify-offers-verified-designs cli-mission-weight-default-back return-split-drops-the-discount return-split-omits-shaping representative-is-the-max-draw draw-buffer-is-unbounded rescore-leaves-the-front-alone reeval-reuses-the-stored-seed reeval-commits-to-the-window reeval-records-nothing cli-drops-reeval job-path-refuses-reeval; do PYTHONPATH=. .venv/bin/python tools/mutate.py --only $id | tail -1; done   # every line: caught
PYTHONPATH=. .venv/bin/python tools/index_gen.py check                    # exit 0
```
Smoke: arch51's command line with `--generations 3 --run <scratch>`. Check that `events.jsonl` has `reevaluate` events with `status: recorded`, that `score_parts.at_floor` appears on evaluate events, that `mission_weight` is 0.0, and that `ppo` events carry `return_by_tag`.

arch51 command line: arch50's configuration (runs/arch50_notes.md), plus `--gait-gain` (the user's 10-09 answer 2), N5 and N11. Launch from a pinned worktree, `git worktree add --detach ../arch51-tree <merged sha>`:
```bash
systemd-run --user --unit arch51 --same-dir \
  --setenv=PYTHONPATH=<main>/../arch51-tree --setenv=DYTISCIDAE_KERNEL_DIR=<main>/mojo/build \
  bash -c '<main>/.venv/bin/python -u -m dytiscidae.ops.run search \
    --generations 900 --batch 16 --workers 4 --min-shard 2 --pool-per-worker 2 \
    --segment-seconds 8 --refine-steps 0 --shared-policy --no-distance-curriculum \
    --gait-gain --mission-weight 0.0 --reeval-per-generation 4 --reeval-depth 8 \
    --memory-ceiling-mb 6000 --seed <launch date YYYYMMDD> --run <main>/runs/arch51 \
    > <main>/runs/arch51.log 2>&1 < /dev/null'
```
Before launch, write `runs/arch51_notes.md` with:
- N5's three predictions and N11's two, verbatim, read at gens 150 and 300;
- the comparability sentences;
- the note that one arm carries both N5 and N11, so a read belongs to the pair.

Pending the user: whether the air-commanded-difference branch (not merged into main as of 5cc9db9) is in arch51.

## §D'. Decisions taken 2026-10-10 on the questions below (routine judgment; the user's items are marked)

1. Floor predicate: competent in *any* medium the island reads, as specified; the floor zeroes both halves. Air's bar stays 0.012 and ROADMAP records that it is uncertified in air until the paired air score's gate runs.
2. **Yes: above the floor, a design is ranked only among window entries that also passed the floor** (L5 amendment, below). With the whole window the competent designs sit at 0.9-1.0 and the scalar separates them by 0.1; the quantile should spend its range on the designs that did something.
   - `Curriculum.observe_blend(...)` gains a fourth tuple element `at_floor: bool` (default False; old pickles hold 3-tuples, read them as not-at-floor).
   - `standing` and `mission_standing` take `floor_only: bool = True` and rank against `[entry for entry in window if not entry.at_floor]`; with fewer than 12 such entries a non-zero score stands at 0.5, a zero score at 0.
   - This touches curriculum.py, so L5 edits four files (loop.py, curriculum.py, test_search.py, mutate.py); the blast radius is one function pair and accepted. Mutation `standing-ranks-against-the-floor` (the filter removed) is caught by a test with 200 floor entries and 12 ranked ones, in which the design holding the second-best ranked score stands at exactly 10/12 filtered and above 0.95 unfiltered; the test asserts the filtered value from its own counts.
3. F2 accepted as written: a design above the floor evicts members at the floor.
4. Accepted: Tier-2 and critic labels pause on an island whose best is at the floor; L12a records it.
5. N11: buffer per elite (representatives only), median by blend, cells never re-filed from re-evaluation features. As specified.
6. N8: match on evaluations per island, i.e. the arch49 snapshot nearest 300 evaluations per island (about gen 150), verdict per island with a pooled row as a secondary read. ROADMAP N8's "gen-37" is corrected.
7. **The user's:** N5 alone or N5 + N11 in arch51 (recommendation: together, they are read by different quantities: N5 by the fitness distribution, competence share and Spearman; N11 by held-out retention and Tier-1→Tier-2 Spearman); with or without the paired air score (recommendation: without, until its four failing checks are fixed and merged); and whether N9 may `pip install mujoco-warp` into a scratch venv.

## §D. Open questions (as raised by the plan; see §D' for the decisions)

1. Floor predicate: competent in *any* medium the island reads (specified), or in its *weakest*? Does the floor zero both halves of the blend (specified), or only the island quantile? Air's bar: 0.012 is a leak on arch49.
2. Should designs above the floor be ranked only among the window entries that also passed the floor? With the whole window they sit at about 0.9-1.0, and parent weight barely separates them.
3. F2: a design above the floor evicts members below it from the front. The ROADMAP states only the tie rule.
4. Tier-2 and critic labels pause on an island while its best elite is at the floor (`top = 0`). Accept, or give `should_promote` a floor-aware rule?
5. N11: buffer per elite or per cell? Median ordered by blend or by summed competence (M3's key)? Are cells ever re-filed from re-evaluation features?
6. N8: which evaluation-count match, and is the verdict per island or pooled?
7. arch51: N5 and N11 together (confounded) or N5 alone first? With or without the paired air score?
