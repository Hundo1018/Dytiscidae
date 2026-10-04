#!/bin/bash
# Paired with/without re-score of each R arm's elites, on the arms' own code.
MAIN=/home/hundo/Projects/Dytiscidae/dytiscidae
export MUJOCO_GL=disable PYTHONPATH=/home/hundo/Projects/Dytiscidae/dyt-pinned-rotor DYTISCIDAE_KERNEL_DIR=$MAIN/mojo/build
cd /home/hundo/Projects/Dytiscidae/dyt-pinned-rotor
for arm in ent0.01 ent0.1 ent0; do
  out=/home/hundo/Projects/Dytiscidae/dyt-pass/experiments/shared_policy_value/results_r_$arm.json
  [ -f $out ] && continue
  $MAIN/.venv/bin/python /home/hundo/Projects/Dytiscidae/dyt-pass/experiments/shared_policy_value/rescore.py --run $MAIN/runs/r_$arm --out $out > $MAIN/runs/_pass1003/rescore_$arm.log 2>&1
  echo "$(date -Is) rescored $arm exit $?" >> $MAIN/runs/_pass1003/r_sweep.status
done
