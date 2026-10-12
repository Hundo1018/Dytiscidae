# E1/E3: the shared policy on against off, on the CPU

Written 2026-10-12, before any run of this campaign. Decided by the user on
2026-10-12: one comparison first, 150 generations, on this container's CPU,
with the runs pushed to the branch `e3-cpu-runs`.

## Configuration

`run.py` holds the configuration; this section restates it.

- **Settings:** arch51's (`docs/ARCH51_SPEC.md`): batch 16, 4 workers, 8 s
  segments, `--refine-steps 0`, `--gait-gain`, `--mission-weight 0.0`,
  N11 re-evaluation 4 of depth 8, no distance curriculum, 150 generations.
- **Arms:** `shared` (`--shared-policy`) and `noshared`
  (`--no-shared-policy`). Nothing else differs between them.
- **Seeds:** 20261013-20261017, the same five for both arms. Runs go in pairs
  (shared, then noshared) for each seed.
- **Path:** the numpy fluid (`batchroll.HostFluid`). It matches the GPU path's
  scoring loop to 0 on the agreement test. Measured cost: 405-409 s per
  generation once the eight-island burst is over
  (`runs/cpu_arch51_timing`). Late generations are not measured.

**What `noshared` means here.** With `--refine-steps 0` and no shared
policy, a candidate's own policy weights stay zero, so it runs its open-loop
gait (`ops.run` prints this warning). The comparison is therefore "the shared
policy against open-loop gaits under the same search". It is not "the shared
policy against per-candidate refinement". The ROADMAP's E1 lists refinement
as a separate arm.

## Comparability

Nothing here is comparable with a GPU run's numbers until the two paths have
been compared on one design at the same seed. HostFluid matches
`TriphibianEnv.step` exactly, and `test_search`'s agreement test passes on
the CPU at 0. That agreement test does not run the Mojo kernel, so it does
not compare the GPU path. Within this campaign, both arms share the path,
the engine (mujoco 3.15.0 in this container's `.venv`) and the seeds.

## Pre-registered reads, at generation 150, on the final archives

Every read is per run, over the eight islands' final archives. The unit is
the run, so each read has n = 5 per arm.

1. **Competence share**, for air, water and land separately: the share of
   elites whose competence in their own medium is >= 0.012. This is N5's
   quantity.
2. **Best competence per medium**: the maximum across all archives.
3. **Crossings**: the number of elites crossing each of the four kinds at
   least once.
4. **E2 L1 counts** (ROADMAP E2, adopted 2026-10-12): distinct elites per
   medium above the certified bar (water 0.012, land 0.15; air has no
   certified bar yet), retaining >= 50% on the auditor's held-out seeds.

**Test.** Two-sided Mann-Whitney U, 5 against 5. The smallest attainable p
is 2/252 = 0.0079. A paired signed-rank test on 5 pairs cannot reach 0.05:
its smallest two-sided p is 0.0625. The effect is reported as the
Hodges-Lehmann shift with a bootstrap 95% interval over runs.

**Prediction.** The shared policy carries weight. `shared` exceeds
`noshared` on read 1 in water and land, with p < 0.05 and an interval that
excludes 0.

Evidence behind the prediction:
- M4 (2026-10-09): the shared PPO learned a fixed body in water.
- arch33 measured the network's in-distribution contribution at
  -0.0014 +/- 0.0016, which is no effect. arch33 ran an older scorer.

**Falsified** if no medium shows p < 0.05 with an interval excluding 0 on
read 1. If that happens, the conclusion to write is: "at 150 generations on
this configuration the shared policy's effect is not separable from the seed
spread". Do not write "no effect".

Reads 2-4 are secondary and reported with the same test. No threshold in
them was chosen after seeing data.
