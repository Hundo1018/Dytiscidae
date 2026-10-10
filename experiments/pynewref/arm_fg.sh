#!/bin/bash
# arm f under four launch environments, then arm g (the 3.9 shim run directly)
cd /home/hundo/Projects/Dytiscidae/impl-d1
run() { echo "=== $1"; shift; env "$@" timeout 200 /home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin/python -u experiments/pynewref/arm_f.py 1 2>&1 | grep -E "WORKER (before|usable)|PARENT usable" | sed -E 's/"PATH": "[^"]*", //; s/"LD_LIBRARY_PATH[^}]*//' | cut -c1-330; }
run f1_pyenv_path
run f2_venv_path_first PATH=/home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin:$PATH
run f3_pyenv_path_MOJO_PYTHON_LIBRARY_preset MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0
run f4_pyenv_path_PYTHONEXECUTABLE_preset PYTHONEXECUTABLE=/home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin/python
run f5_both_preset MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 PYTHONEXECUTABLE=/home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin/python
echo "=== g_shim_python_runs_the_probe_directly"
timeout 100 ~/.pyenv/shims/python3 -c "import sys;print(sys.version.split()[0]);sys.path.insert(0,'/home/hundo/Projects/Dytiscidae/impl-d1/../dytiscidae/mojo/build');import full_pipeline as f;f.FullPipeline(64,64,4);print('CONSTRUCT_OK')" 2>&1 | tail -2 | cut -c1-200
