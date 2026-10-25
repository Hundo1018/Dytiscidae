#!/bin/bash
# After arch54 ends: the no-model gate (R2) and the 8-draw screen (R1/R3) on its archive, one GPU lane.
M=/home/hundo/Projects/Dytiscidae/dytiscidae; cd $M
while pgrep -f '[o]ps.run search.*runs/arch54' >/dev/null; do sleep 300; done
sleep 60
echo "== gate start $(date -Is)"
env MUJOCO_GL=disable PYTHONPATH=$M DYTISCIDAE_KERNEL_DIR=$M/mojo/build $M/.venv/bin/python -u experiments/no_model_gate/run.py --run $M/runs/arch54 --out experiments/air_turn_bar/gate_arch54.json > experiments/air_turn_bar/gate_arch54.log 2>&1 || echo "gate failed $?"
echo "== screen start $(date -Is)"
env MUJOCO_GL=disable PYTHONPATH=$M DYTISCIDAE_KERNEL_DIR=$M/mojo/build $M/.venv/bin/python -u experiments/air_turn_bar/run.py screen --run runs/arch54 --out experiments/air_turn_bar/screen_arch54.json --k 8 --seed-base 2026102200 > experiments/air_turn_bar/screen_arch54.log 2>&1 || echo "screen failed $?"
echo "== reads done $(date -Is)"
