#!/bin/bash
cd /home/hundo/Projects/Dytiscidae/impl-d1
run() { echo "=== $1"; shift; env "$@" timeout 200 /home/hundo/Projects/Dytiscidae/dytiscidae/.venv/bin/python -u experiments/pynewref/arm_f.py 1 2>&1 | grep -E "WORKER (before|usable)|PARENT usable" | sed -E 's/"PATH": "[^"]*", //; s/"LD_LIBRARY_PATH[^}]*//' | cut -c1-330; }
run h1_pyenv_fix_unsetenv_PYTHONEXECUTABLE D1_FIX=unsetenv
run h2_pyenv_fix_putenv_PYTHONEXECUTABLE D1_FIX=putenv
