"""Several generations' worth of machines, stepped in several processes.

Sixteen machines were stepped in lockstep inside one process while twenty cores
sat idle.  The batching is not the problem -- it is what makes the fluid solver
affordable -- but it is *within* one process, and a single Python interpreter
can only drive one MuJoCo step loop at a time.

What a batched step is actually made of
---------------------------------------

Measured on this machine, batch of 16, best of five over 200 steps:

    step 1179 us  =  fluid 580 (49%) + mj_step 199 (17%)
                     + power budget 207 (18%) + Python 194 (16%)

So barely half of it is the part that already runs on the GPU.  The other half
is per-machine CPU work that a second process would do at the same time.  And
the fluid half is not GPU-bound either at these sizes -- it is host-side
marshalling, which is exactly what `docs/ROADMAP.md` Phase 0 found for the numpy
solver and what remains true through the Mojo port.

Which is why this scales, and it does:

    workers x machines   per-process step   throughput
             1 x 8            853 us         9.4k machine-steps/s
             2 x 8            914 us        17.5k          (1.86x)
             4 x 8            910 us        35.2k          (3.72x)
             8 x 8           1005 us        63.7k          (6.7x)
            16 x 8           1074 us       119.1k          (12.7x)

Sixteen processes cost 26% more per process and return 12.7 times the
throughput.  Growing the batch inside one process instead saturates: k=32 gives
78 us/machine against k=16's 89, and 12.8k machine-steps/s against 119k here.

Each worker builds **its own** GPU pipeline.  That is required rather than
merely tolerated: constructing a second `FullPipeline` in one process hangs (see
`batchroll._get_pipeline`), and it is also the right shape -- Phase 0 says the
per-call cost is mostly fixed, so panel batching has to stay *inside* each
process rather than reverting to one solver call per machine.

This is also where PPO's data collection becomes parallel.  The learner itself
is about six thousand weights and is not worth a process; the 99.8% of the cost
that is simulation is what gets sharded here.  Each worker fills its own
`RolloutBuffer` and returns the trajectories; the parent concatenates them and
runs one update, so the batch the optimiser sees is exactly the batch it saw
before -- including `_terminal_scale`, which is computed over the merged set.

What sharding does and does not change
--------------------------------------

Nothing per-machine depends on which other machines share its batch.  The
scatter draw is `default_rng(scatter_seed)` constructed fresh per machine, the
identification deltas are `default_rng(seed)` constructed fresh per machine, the
force limiter is per-machine, and the fluid kernel has no cross-machine
reduction.  So a candidate scores the same alone, in a batch of sixteen, or in
any sharding of one -- and `tests/test_search.py` asserts exactly that.

The one exception is deliberate: when the shared policy is *learning* it samples
its actions, and the sequence of samples depends on how the work was split.  The
exploration noise is not part of the score, but it means a run with a shared
policy is reproducible per worker count rather than across worker counts.  Each
worker seeds torch from the evaluation seed and its own shard index, so a given
worker count is reproducible.
"""

from __future__ import annotations

import os
import sys

#: Per-worker state, so a policy and a GPU pipeline are built once and reused.
_WORKER: dict = {}


def _init_worker() -> None:
    """Make the repo importable in a spawned worker.

    ``spawn`` starts a bare interpreter, so neither the project nor the built
    Mojo extension is on the path unless the parent's environment happened to
    carry it -- and the CLI does not set PYTHONPATH.  ``batchroll`` already adds
    ``mojo/build`` itself; this adds the repo root, and imports the heavy
    modules now so the first real task is not also the first import.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from . import batchroll  # noqa: F401
    except Exception:
        pass


def _run_shard(payload):
    """One shard of a generation, inside a worker.  Returns what the parent needs.

    Sends back the controllers as well as the results: ``evaluate_tier1_batch``
    fills in each controller's rhythm and its measured mobility bases, and the
    caller stores both on the elite.  Losing them would leave every promoted
    design carrying its parent's axes, which is the failure the identification
    exists to prevent.
    """
    from . import batchroll

    (phenos, ctrls, kwargs, shared_spec, shard_index) = payload
    shared = buffer = None
    if shared_spec is not None:
        import torch

        from ..learning.ppo import RolloutBuffer, SharedPolicy

        n_obs, n_modes, hidden, state, shaping, seed = shared_spec
        policy = _WORKER.get("shared")
        if policy is None or (policy.n_obs, policy.n_modes) != (n_obs, n_modes):
            policy = SharedPolicy(n_obs, n_modes, hidden=hidden)
            _WORKER["shared"] = policy
        policy.load_state_dict(
            {k: torch.as_tensor(v) for k, v in state.items()}, strict=False)
        # Reproducible per worker count; see the module docstring.
        torch.manual_seed(int(seed) + 7919 * int(shard_index))
        shared, buffer = policy, RolloutBuffer(shaping=shaping)

    results = batchroll.evaluate_tier1_batch(
        phenos, controllers=ctrls, shared=shared, buffer=buffer, **kwargs)
    return results, ctrls, (list(buffer.trajectories) if buffer else [])


def split(n: int, workers: int, min_shard: int) -> list:
    """Contiguous shard boundaries, never finer than ``min_shard``.

    A small shard pays the fixed per-call cost of a batch for almost no work.
    Measured per machine-step: 238 us alone, 149 in a shard of four, 105 in
    eight, 89 in sixteen.  So shard size and worker count trade against each
    other, and the trade is real -- on a generation of 32, four workers of eight
    took 65.1 s and eight workers of four took 64.4 s, which is the extra
    parallelism being spent entirely on the smaller batches.

    Hence the floor, and hence eight: it is where that curve flattens.  The
    number of shards is capped by how many machines there are to spread, so
    using more cores is a matter of raising the generation size rather than the
    worker count -- which is a search-design decision and is left to the caller.
    """
    if n <= 0:
        return []
    k = max(1, min(int(workers), n // max(int(min_shard), 1) or 1))
    base, extra = divmod(n, k)
    out, at = [], 0
    for i in range(k):
        step = base + (1 if i < extra else 0)
        out.append((at, at + step))
        at += step
    return out


class ActorPool:
    """A persistent pool of worker processes, each with its own batched evaluator.

    Persistent because a worker pays for importing MuJoCo and torch and for
    allocating a GPU pipeline, and that is per worker rather than per
    generation.  Created once by the search and closed when it ends.

    ``workers <= 1`` makes every method a direct call with no processes at all,
    so the single-process path stays exactly what it was and is what the tests
    compare against.
    """

    def __init__(self, workers: int = 1, *, min_shard: int = 8) -> None:
        self.workers = max(int(workers), 1)
        self.min_shard = max(int(min_shard), 1)
        self._pool = None
        if self.workers > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # ``spawn``: forking a process that holds a CUDA context gives the
            # child an unusable one.  It costs a fresh interpreter per worker,
            # which is why the pool is persistent.
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker)

    # ------------------------------------------------------------------ tier 1

    def evaluate_tier1(self, phenos, *, controllers=None, shared=None,
                       buffer=None, **kwargs):
        """``batchroll.evaluate_tier1_batch``, spread over the pool.

        Returns the results in the caller's order.  The caller's controller
        objects are updated in place with the rhythm and mobility bases the
        workers measured, and ``buffer`` receives every worker's trajectories.
        """
        from . import batchroll

        n = len(phenos)
        ctrls = list(controllers) if controllers is not None else [None] * n
        shards = split(n, self.workers, self.min_shard)
        if self._pool is None or len(shards) <= 1:
            return batchroll.evaluate_tier1_batch(
                phenos, controllers=ctrls, shared=shared, buffer=buffer,
                **kwargs)

        spec = None
        if shared is not None and buffer is not None:
            spec = (
                shared.n_obs, shared.n_modes,
                int(shared.actor[0].out_features),
                {k: v.detach().cpu().numpy()
                 for k, v in shared.state_dict().items()},
                float(getattr(buffer, "shaping", 0.0)),
                int(kwargs.get("seed", 0)),
            )

        futures = []
        for j, (a, b) in enumerate(shards):
            futures.append(self._pool.submit(
                _run_shard, (phenos[a:b], ctrls[a:b], kwargs, spec, j)))

        # Collect everything before applying any of it.  A worker that dies
        # halfway would otherwise leave the buffer holding some shards'
        # trajectories and the controllers holding some shards' bases, and the
        # fallback below would then add them a second time.
        try:
            collected = [f.result() for f in futures]
        except Exception as exc:
            self._degrade(exc)
            return batchroll.evaluate_tier1_batch(
                phenos, controllers=ctrls, shared=shared, buffer=buffer,
                **kwargs)

        results = [None] * n
        for (a, b), (res, back, trajectories) in zip(shards, collected):
            results[a:b] = res
            for local, remote in zip(ctrls[a:b], back):
                if local is None or remote is None:
                    continue
                # In place: the caller holds these references and stores what
                # is on them after this returns.
                local.params = remote.params
                local.bases = remote.bases
                if local.policy is not None and remote.policy is not None:
                    local.policy.weights = remote.policy.weights
            if buffer is not None:
                for t in trajectories:
                    buffer.add(t)
        return results

    # ---------------------------------------------------------------- failure

    def _degrade(self, exc: Exception) -> None:
        """Fall back to one process, and say so.

        Loudly and once, because the alternative is a run that quietly loses
        its parallelism and still finishes -- the same failure mode as the
        silent CPU fallback in ``batchroll``, where two full timing probes were
        collected before ``nvidia-smi`` gave it away.

        The usual cause is a caller that starts a search at import time without
        an ``if __name__ == "__main__":`` guard: ``spawn`` re-imports the main
        module in each worker, which would start the search again.
        """
        import sys

        self.close()
        self.workers = 1
        print(
            f"\n*** actor pool failed; this run is single-process. ***\n"
            f"    reason: {type(exc).__name__}: {exc}\n"
            f"    If the cause is 'importing main', wrap the entry point in\n"
            f"    `if __name__ == \"__main__\":` -- spawn re-imports it.\n",
            file=sys.stderr, flush=True)

    # ----------------------------------------------------------------- cleanup

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
