## Reading the numbers above (not generated; each figure is from a table in this report or the rows)

- **The one crossing the station gate took** (arch49 air:80, `water_to_land`, gannet, 2 rotors,
  8.598 kg, arrives upright 0.979). Its own arm drifted 1.862 m through the hold; the same body on
  its base gait with no policy drifted 1.842 m through the same hold and made 0.461 of the shore
  distance without crossing (own: 0.522, crossed); held still it drifted 0.168 m. The policy does
  not change what this body does in the hold: it swims shoreward from step 0 on both arms. Rows:
  `population_gates.jsonl`, run arch49, island air, index 80.
- **The three crossings the upright gate took** arrive at `exit_upright` -0.100, -0.484 and -0.993,
  with `min_upright` -0.844, -1.000, -1.000. All three bodies have rotors (2, 16, 13).
- **Is 0.25 / 0.75 m a fair station for this population?** Own arm, rollouts read: 1,657 of 1,995
  water-start holds (83%) and 831 of 1,002 land-start holds (83%) stay within 0.50 m, which is
  where the hold passes; 1,323 (66%) and 704 (70%) stay within 0.25 m. The median own-arm drift
  is 0.124 m in water and 0.122 m on land. So most of the population passes the station as set.
  But the same bodies held still drift less: still-arm median 0.059 m in water and 0.034 m on
  land, and the own arm drifts more than its own still body in 201 of 341 water holds and 134 of
  175 land holds. The station, on these populations, is passed mostly by not moving, and a still
  body itself fails it (drift > 0.50 m) in 26 of 375 water holds and 13 of 189 land holds: 7% of
  passive holds exceed the bar with no actuation at all.
- **Crossed against not crossed** cannot be compared as distributions: the own arm has two
  water-start crossings (drift 0.348 and 1.862 m) and one land-start crossing (0.294 m) in the
  whole population.

## 5. What could not be done or reproduced

- **The search's control law is not reproduced exactly.** The search scored crossings on the
  batched path (`batchroll.run_transition_batch`) with the shared half sampled with exploration
  noise whenever a PPO collector was attached; this replay is the single path with the shared
  half at its mean (`SummedPolicy`). The network is the one stored for the elite's generation
  (`scoring_networks/genNNNNN.npz`, found for all 1,115 entries), the own policy and mobility
  bases are the stored ones. Of the 11 entries whose archive record has a completed crossing, 7
  complete any crossing here; which kind the archive credited is not stored per kind, so a
  per-kind reproduction cannot be checked.
- **Engine.** The replay runs on the main `.venv` (MuJoCo 3.15.0); arch49 and arch51 were scored
  under 3.11 and water and land are not comparable across the upgrade (CLAUDE.md, D-M0). The
  off-against-on comparison is within one engine and one trajectory, so it is unaffected; the
  absolute crossing counts are not the runs' own.
- **Base and still are a sample**, 191 entries (65 / 64 / 62), not the population; a base or
  still crossing rate below about 1 in 60 per run and kind is not resolved. With the gates on
  the count is 0 of 764 for each arm; with them off, base 1 of 764, still 0 of 764.
- **Own arm on stratum B was run without its base and still arms**, so "does the policy change
  the hold" (the comparison made for arch49 air:80 above) exists only for stratum A.
- **Item 4's counterfactual** is `transitions.py` at b652788 loaded from git beside the current
  module; it was run only on the 83 gates-off rows with `bad_qacc > 0`, not on the other 5,905
  rows (whose `run_transition` code path differs from the pre-commit one only in the read).
- **Entry state**: only the default (`back=0`, one fixed scatter draw per kind). The gates were
  set from eight entry states on hand-flown bodies; this is one.
- `mass` and `n_rotors` are the archive's `meta` fields, not re-measured from the built model.
- Nothing under `dytiscidae/`, `tests/`, `tools/` or `docs/` was edited; no cheap suite was run,
  because no source file changed.
