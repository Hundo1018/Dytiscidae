#!/bin/bash
# ROADMAP R (2026-10-03 pass): shared_ent_coef sweep, plus a no-policy arm.
# Pinned at be3dbe0 (worktree ../dyt-pinned) so later merges cannot change it.
# Arms run one after another: the machine has 15 GB and other work on it.
# Read: experiments/shared_policy_value/rescore.py on each arm (paired with vs
# without the shared net), log_std from the ppo events, per-medium top-10.
set -u
TREE=/home/hundo/Projects/Dytiscidae/dyt-pinned
MAIN=/home/hundo/Projects/Dytiscidae/dytiscidae
PY=$MAIN/.venv/bin/python
export MUJOCO_GL=disable PYTHONPATH=$TREE DYTISCIDAE_KERNEL_DIR=$MAIN/mojo/build
cd $TREE
COMMON="--generations 60 --batch 16 --workers 4 --segment-seconds 6 --refine-steps 0 --seed 20261003 --no-postrun"
for arm in ent0.01 ent0.1 ent0 off; do
    out=$MAIN/runs/r_$arm
    [ -f $out/done ] && continue
    case $arm in
        off)  extra="" ;;
        ent*) extra="--shared-policy --shared-ent-coef ${arm#ent}" ;;
    esac
    echo "$(date -Is) start $arm" >> $MAIN/runs/_pass1003/r_sweep.status
    $PY -u -m dytiscidae.ops.run search $COMMON $extra --run $out > $out.log 2>&1 < /dev/null
    echo "$(date -Is) end $arm exit $?" >> $MAIN/runs/_pass1003/r_sweep.status
    touch $out/done
done
