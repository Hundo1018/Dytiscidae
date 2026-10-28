# GPU. Peak parent RSS of evaluate_tier1_device for N bodies at the search's K (50 mean + 10 sampled),
# the path arch60's gen-0 scoring runs. Before the chunking fix every body stayed resident (8 GB at ~10
# bodies); after it, streams-many. Usage: probe_device_rss_full.py N
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dytiscidae.core.bodyplans import BODY_PLANS, REFERENCE_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll.rollout_wp import evaluate_tier1_device
from tests.test_devroll_agreement import _a4_controller

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NAMES = ["teal", "quad", "eel", "medusa", "gannet", "ray", "bat", "beetle"]
phenos, ctrls = [], []
for i in range(N):
    n = NAMES[i % len(NAMES)]
    p = build((BODY_PLANS.get(n) or REFERENCE_PLANS[n])())
    c, shared = _a4_controller(p, 171)
    phenos.append(p)
    ctrls.append(c)
t0 = time.time()
out = evaluate_tier1_device(phenos, controllers=ctrls, seeds=[9200 + i for i in range(N)],
                            k_mean=50, k_sampled=10, segment_seconds=8.0, shared=shared, streams=4,
                            keep_records=False)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"bodies {N} wall {time.time() - t0:.1f} s; peak RSS {peak:.0f} MB; "
      f"draws {[len(d.draws) for d in out][:4]}", flush=True)
