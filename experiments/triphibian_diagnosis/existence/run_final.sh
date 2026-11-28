#!/usr/bin/env bash
# E0 final table: every item, seeds 0..7, both bodies, hand and held-still arms.
# One python process at a time.  Rows append to results_final.jsonl.
#   bash experiments/triphibian_diagnosis/existence/run_final.sh
set -u
cd "$(dirname "$0")/../../.."
E=experiments/triphibian_diagnosis/existence
ITEMS=air,water,land,air_to_water,water_to_air,water_to_land,land_to_air,mission_air,mission_water,mission_land
PY="nice -n 5 .venv/bin/python"
export MUJOCO_GL=disable PYTHONPATH=.
for B in B1 B0; do
  $PY $E/triquad.py facts --body $B > $E/logs/facts_${B}_final.log 2>&1
  timeout 1500 $PY $E/triquad.py run --body $B --arm hand --items $ITEMS --seeds 0 --out final --save-traj > $E/logs/final_${B}_hand_s0.log 2>&1
  timeout 2400 $PY $E/triquad.py run --body $B --arm hand --items $ITEMS --seeds 1-7 --out final > $E/logs/final_${B}_hand_s1-7.log 2>&1
  timeout 2400 $PY $E/triquad.py run --body $B --arm still --items $ITEMS --seeds 0-7 --out final > $E/logs/final_${B}_still.log 2>&1
done
echo done > $E/logs/final_DONE
