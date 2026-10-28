# GPU. Parent RSS across the device path's stages for one body at the search's K (50 mean + 10 sampled),
# to find the step that makes arch60's gen-0 RSS jump to ~8 GB. Stages: body_worlds (host envs),
# DeviceSegments (device arrays), arm, run_bodies (graph capture + steps), download.
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R
from dytiscidae.envs.devroll.rollout_wp import _wp
from dytiscidae.evolution.loop import placement_draw_seed
from tests.test_devroll_agreement import _a4_controller

K_MEAN, K_SAMPLED, SECONDS = 50, 10, 8.0
NAME = sys.argv[1] if len(sys.argv) > 1 else "teal"


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return -1.0


def mark(stage: str, last: list) -> None:
    now = rss_mb()
    print(f"{time.strftime('%H:%M:%S')} {stage:<14} RSS {now:8.0f} MB  (+{now - last[0]:7.0f})", flush=True)
    last[0] = now


last = [rss_mb()]
mark("start", last)
p = build((BODY_PLANS.get(NAME) or BODY_PLANS[NAME])())
ctrl, shared = _a4_controller(p, 171)
mark("controller", last)
K = K_MEAN + K_SAMPLED
draw_seeds = [9200] + [placement_draw_seed(9200, d) for d in range(1, K)]
envs, doms, params, pols, bases, index = R.body_worlds(p, ctrl, draw_seeds, spec=None,
                                                       still=False, sea_state=None, perturb=None)
print(f"worlds {len(envs)}", flush=True)
mark("body_worlds", last)
seg = R.DeviceSegments(envs, seconds=SECONDS, dtype=_wp.float32, shared=shared,
                       policies=pols, bases=bases)
mark("DeviceSegments", last)
sampled = [int(k >= K_MEAN) for k in range(K) for _ in R.MEDIA for _h in (0, 1)]
noise = [placement_draw_seed(draw_seeds[k], 101 + j) for k in range(K)
         for j in range(2 * len(R.MEDIA))]
seg.arm(doms, params, sampled=sampled, seeds=noise)
mark("arm", last)
R.run_bodies([seg], streams=1, graph=True)
mark("run_bodies", last)
data = seg.download()
mark("download", last)
print(f"peak RSS {max(rss_mb(), 0):.0f} MB; ndec max {int(data['ndec'].max())}", flush=True)
