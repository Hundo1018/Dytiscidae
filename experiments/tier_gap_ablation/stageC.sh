#!/bin/bash
# Stage C: which component of a task draw carries the credit (after stage B finishes).
cd "$(dirname "$0")/../.."
while [ "$(systemctl --user is-active tga-arch48b)" = "active" ]; do sleep 10; done
export MUJOCO_GL=disable PYTHONPATH=.
.venv/bin/python -u experiments/tier_gap_ablation/run.py --n-max 16 --sub 10 --rungs T_heading,T_order,T_depth
