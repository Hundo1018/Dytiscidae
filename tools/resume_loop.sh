#!/usr/bin/env bash
# Run a search to its last generation through memory-ceiling stops, unattended.
#
# A search started with --memory-ceiling-mb stops cleanly with a checkpoint when its resident set
# passes the ceiling, and returns exit code 0 like a finished run. This loop tells the two apart from
# the run's own records and resumes the first kind, so a long arm does not wait for someone to press
# --resume (arch60 sat idle 30 h on 2026-10-29..31 for that reason).
#
#   resume_loop.sh RUN_DIR LOG GENERATIONS MAX_CYCLES -- <search flags, without --run/--resume/--generations>
#
# Environment: PY (python, default .venv/bin/python), WAIT_FOR (space-separated systemd units to wait
# out first, so two processes never write one run directory), plus whatever the search needs
# (PYTHONPATH, DYTISCIDAE_KERNEL_DIR, MUJOCO_GL).
#
# After each cycle, in this order:
#   done       the last generation in RUN_DIR/generations.jsonl is GENERATIONS - 1: run postrun once, exit 0
#   crashed    "Traceback" in the cycle's log tail: exit 2
#   ceiling    "stopping at generation N: ... ceiling" in the tail: resume (at most MAX_CYCLES)
#   otherwise  exit 3, so an unexplained stop is not resumed blindly
# Intermediate cycles pass --no-postrun: postrun on a partial run crashes its report and uses the GPU.
set -u

RUN_DIR=$1; LOG=$2; GENERATIONS=$3; MAX_CYCLES=$4; shift 4
[ "${1:-}" = "--" ] && shift
PY=${PY:-.venv/bin/python}

last_generation() {
    "$PY" - "$RUN_DIR/generations.jsonl" <<'EOF'
import json, sys
last = -1
try:
    for line in open(sys.argv[1]):
        d = json.loads(line)
        if "generation" in d:
            last = max(last, int(d["generation"]))
except OSError:
    pass
print(last)
EOF
}

for unit in ${WAIT_FOR:-}; do
    while systemctl --user is-active -q "$unit"; do sleep 30; done
done

cycle=0
while :; do
    if [ "$(last_generation)" -ge $((GENERATIONS - 1)) ]; then
        echo "resume_loop: done at generation $(last_generation); postrun once" >> "$LOG"
        "$PY" -m dytiscidae.ops.run postrun --run "$RUN_DIR" >> "$LOG" 2>&1
        exit 0
    fi
    if [ "$cycle" -ge "$MAX_CYCLES" ]; then
        echo "resume_loop: $MAX_CYCLES cycles used, stopping at generation $(last_generation)" >> "$LOG"
        exit 4
    fi
    cycle=$((cycle + 1))
    if [ -f "$LOG" ]; then offset=$(wc -c < "$LOG"); else offset=0; fi
    echo "resume_loop: cycle $cycle, resuming after generation $(last_generation)" >> "$LOG"
    "$PY" -u -m dytiscidae.ops.run search --generations "$GENERATIONS" --resume --no-postrun \
        --run "$RUN_DIR" "$@" >> "$LOG" 2>&1 < /dev/null
    tail -c +$((offset + 1)) "$LOG" > "$LOG.cycle"
    if [ "$(last_generation)" -ge $((GENERATIONS - 1)) ]; then continue; fi
    if grep -q "Traceback" "$LOG.cycle"; then
        echo "resume_loop: crashed in cycle $cycle (Traceback in the log), not resuming" >> "$LOG"
        exit 2
    fi
    if grep -qE "stopping at generation [0-9]+: .*ceiling" "$LOG.cycle"; then continue; fi
    echo "resume_loop: cycle $cycle ended without a ceiling stop or the last generation, not resuming" >> "$LOG"
    exit 3
done
