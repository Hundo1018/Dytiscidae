"""Arm e / re-check: the promo_floor_trial.py configuration (run_search, 2 workers).
usage: arm_e.py <island> <nref> <segment_seconds>"""
import sys, tempfile, shutil, json
from pathlib import Path
from dytiscidae.evolution.loop import SearchConfig, run_search
from dytiscidae.envs.triphibian import MissionSpec
from dytiscidae.envs import batchroll
def main():
    island, nref, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    import os
    if os.environ.get("D1_FIX") == "unsetenv":
        os.unsetenv("PYTHONEXECUTABLE")   # C-level scrub after the Mojo import; os.environ never saw the key
    print("USABLE", batchroll.usable(), flush=True)
    tmp = tempfile.mkdtemp(prefix="dyt-pynewref-")
    try:
        state = run_search(SearchConfig(
            generations=2, batch=2, seed=3, segment_seconds=secs,
            n_reference_seeds=nref, n_random_seeds=0, islands=(island,),
            tier2_every=1, audit_every=999, migrate_every=999,
            checkpoint_every=999, run_dir=tmp, identify_axes_every=999,
            workers=2, min_shard=1, promotion_refine_steps=2), MissionSpec())
        for e in sorted(state.archive.cells.values(), key=lambda e: e.genome.genome_id):
            if str(e.genome.genome_id).startswith("seed"):
                print(f"  elite {e.genome.genome_id} fit={e.fitness:.4f} seed={e.meta.get("eval_seed")} air={e.meta.get("air")} water={e.meta.get('water')} land={e.meta.get('land')} swsp={e.meta.get('scored_with_shared_policy')}", flush=True)
        print("DONE n_elites", len(state.archive.cells), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
