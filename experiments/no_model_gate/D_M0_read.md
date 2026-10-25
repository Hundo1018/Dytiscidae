# D-M0 read: mujoco 3.11 vs 3.15 on arch49's 229 elites (2026-10-25)

Same command, same elites, same session; the only change is the mujoco engine
(3.11 from `.venv`, 3.15 via `PYTHONPATH=mj315-overlay`). Reproduction = within 0.005
of the recorded per-medium competence (`reproduction` in each results JSON).

| run | air | water | land | wall |
|---|---|---|---|---|
| stored `results_arch49.json` (3.11, Oct 8) | 229 | 215 | 195 | 5736 s |
| D-M0 3.11 (`results_arch49_dm0_mj311.json`) | 222 | 229 | 229 | 4401 s |
| D-M0 3.15 (`results_arch49_dm0_mj315.json`) | 222 | 201 | 179 | 4331 s |

Frozen prediction (IMPL_1010_SPEC.md §3, D-M0 row): under 3.15, air >= 215, water >= 208,
land >= 185; falsified if any medium drops by more than 10 elites (3.11 reference 222/215/195).

- Against the spec reference: water 201 (< 208, drop 14) and land 179 (< 185, drop 16)
  miss their predictions; both drops exceed 10. Air 222 holds.
- Against the same-session 3.11 leg: water drops 28 and land 50. Also falsified.

Verdict: the falsifier fires under either baseline. Per the spec, the upgrade is a physics
boundary and needs the user's decision. Not decided here; `.venv` is still mujoco 3.11.0.

Baseline caveat: the same engine (3.11) gives water/land 229/229 this session and 215/195 on
Oct 8. The reference is not stable across runs, so the drop is measured against a moving
baseline. This needs explaining before the boundary is read as mujoco's doing alone.
