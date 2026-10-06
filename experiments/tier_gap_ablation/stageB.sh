#!/bin/bash
# Stage B: everything after R0/R0s/R1 (stage A: --n-max 16 --rungs R0,R0s,R1 --still R0s).
# Runs on a fixed random 10 + 10 per medium (--sub 10).  One process at a time.
cd "$(dirname "$0")/../.."
export MUJOCO_GL=disable PYTHONPATH=.
PY=.venv/bin/python
common="--n-max 16 --sub 10"
$PY -u experiments/tier_gap_ablation/run.py $common --rungs S_task,S_scatter,S_env --still R0
$PY -u experiments/tier_gap_ablation/run.py $common --rungs R2b,R3b,R4b,R5b,R6b --still R6b
$PY -u experiments/tier_gap_ablation/run.py $common --rungs A_task,A_scatter,A_dist,A_ctrl,R0o --tier2 3
