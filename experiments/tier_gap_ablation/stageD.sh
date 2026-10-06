#!/bin/bash
# Stage D: each difference ALONE from R0 (the generation's own seed kept), after stage C.
cd "$(dirname "$0")/../.."
while [ "$(systemctl --user is-active tga-arch48c)" != "inactive" ] && [ "$(systemctl --user is-active tga-arch48c)" != "failed" ]; do sleep 10; done
export MUJOCO_GL=disable PYTHONPATH=.
.venv/bin/python -u experiments/tier_gap_ablation/run.py --n-max 16 --sub 10 --rungs R0p,B_task,B_scatter,B_dist,B_ctrl,B_leg
