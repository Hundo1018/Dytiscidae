#!/bin/bash
# Stage E: a heading sweep (after stage D).
cd "$(dirname "$0")/../.."
while [ "$(systemctl --user is-active tga-arch48d)" != "inactive" ] && [ "$(systemctl --user is-active tga-arch48d)" != "failed" ]; do sleep 10; done
export MUJOCO_GL=disable PYTHONPATH=.
.venv/bin/python -u experiments/tier_gap_ablation/run.py --n-max 16 --sub 10 --rungs H0,H1,H2,H3,H4,H5,H6,H7
