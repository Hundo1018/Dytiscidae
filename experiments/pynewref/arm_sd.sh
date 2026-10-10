#!/bin/bash
# arm f under the systemd user manager's default environment (what systemd-run --user gives a search)
cd /home/hundo/Projects/Dytiscidae/impl-d1
echo "systemd PATH head: $(systemctl --user show-environment | grep ^PATH= | cut -c1-120)"
echo "which python3 under that PATH: $(env -i PATH=$(systemctl --user show-environment | grep ^PATH= | cut -d= -f2-) bash -c 'command -v python3; python3 --version' 2>&1 | tr '\n' ' ')"
env PYTHONPATH=/home/hundo/Projects/Dytiscidae/impl-d1 MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=/home/hundo/Projects/Dytiscidae/dytiscidae/mojo/build timeout 200 /home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin/python -u experiments/pynewref/arm_f.py 1 2>&1 | grep -E "WORKER usable|PARENT usable"
