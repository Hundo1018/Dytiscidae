"""E1/E3 on a CPU container: the shared policy on against off, 5 seeds each.

The user, 2026-10-12: one comparison first, 150 generations, on this
container's CPU (`batchroll.HostFluid`), with the runs pushed to a dedicated
branch. Pre-registered reads: `NOTES.md` beside this file.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/e1_shared_policy/run.py
    PYTHONPATH=. .venv/bin/python experiments/e1_shared_policy/run.py --status

Safe to start again at any time. It first restores, from the branch, any run
directory missing here (the container is reclaimed when idle, and `runs/` is
gitignored). Then it works the queue in order. A finished run is skipped, a
partial one continues with `--resume`, and a new one starts. Every
`SYNC_EVERY` generations, and when a run ends, it replaces the single commit
on `BRANCH` with the current run directories (`events.jsonl` gzipped) and
force-pushes it. That branch holds data only and is written only by this
script.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "e1"
TREE = ROOT.parent / "e3-cpu-runs-tree"
BRANCH = "e3-cpu-runs"
GENERATIONS = 150
SYNC_EVERY = 20
SEEDS = (20261013, 20261014, 20261015, 20261016, 20261017)
#: arch51's configuration (docs/ARCH51_SPEC.md), less the GPU-only memory
#: assumptions; the arms differ in the shared policy only.
BASE = ["--generations", str(GENERATIONS), "--batch", "16", "--workers", "4",
        "--min-shard", "2", "--pool-per-worker", "2", "--segment-seconds", "8",
        "--refine-steps", "0", "--no-distance-curriculum", "--gait-gain",
        "--mission-weight", "0.0", "--reeval-per-generation", "4",
        "--reeval-depth", "8", "--memory-ceiling-mb", "12000", "--no-postrun"]
ARMS = {"shared": ["--shared-policy"], "noshared": ["--no-shared-policy"]}
#: Paired order, so a partial campaign still holds whole seed pairs.
QUEUE = [(arm, seed) for seed in SEEDS for arm in ARMS]
SKIP_FILES = {"dashboard.html"}


def run_dir(arm: str, seed: int) -> Path:
    return RUNS / f"{arm}_s{seed}"


def generations_done(d: Path) -> int:
    """Highest generation recorded, plus one; 0 when nothing is."""
    f = d / "generations.jsonl"
    if not f.exists():
        return 0
    top = -1
    for line in f.open():
        try:
            g = json.loads(line).get("generation")
        except json.JSONDecodeError:
            continue
        if isinstance(g, int):
            top = max(top, g)
    return top + 1


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          capture_output=True, text=True)


def push_with_retry(cwd: Path) -> None:
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        r = git("push", "-f", "origin", f"HEAD:refs/heads/{BRANCH}", cwd=cwd, check=False)
        if r.returncode == 0:
            return
        print(f"[sync] push failed ({r.stderr.strip()[:200]}), retrying", flush=True)
    raise RuntimeError("push to the data branch failed five times")


def status() -> dict:
    return {f"{arm}_s{seed}": generations_done(run_dir(arm, seed)) for arm, seed in QUEUE}


def sync() -> None:
    """Replace the data branch's single commit with the current run directories."""
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir(parents=True)
    git("init", "-q", str(TREE))
    url = git("remote", "get-url", "origin").stdout.strip()
    git("remote", "add", "origin", url, cwd=TREE)
    for d in sorted(RUNS.glob("*_s*")):
        for src in d.rglob("*"):
            if src.is_dir() or src.name in SKIP_FILES:
                continue
            rel = src.relative_to(RUNS)
            dst = TREE / "e1" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.name == "events.jsonl":
                with src.open("rb") as fi, gzip.open(str(dst) + ".gz", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                shutil.copy2(src, dst)
    (TREE / "e1").mkdir(exist_ok=True)
    (TREE / "e1" / "status.json").write_text(json.dumps(
        {"generations": status(), "target": GENERATIONS,
         "commit": git("rev-parse", "HEAD").stdout.strip(),
         "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    git("add", "-A", cwd=TREE)
    git("-c", "user.name=e1-runner", "-c", "user.email=noreply@anthropic.com",
        "commit", "-q", "-m", "data: e1 shared-policy runs", cwd=TREE)
    push_with_retry(TREE)
    print(f"[sync] pushed {status()}", flush=True)


def restore() -> None:
    """Bring back from the branch every run directory missing here."""
    if git("ls-remote", "--exit-code", "origin", BRANCH, check=False).returncode != 0:
        return
    if TREE.exists():
        shutil.rmtree(TREE)
    git("clone", "-q", "--depth", "1", "--branch", BRANCH,
        git("remote", "get-url", "origin").stdout.strip(), str(TREE))
    for d in sorted((TREE / "e1").glob("*_s*")):
        local = RUNS / d.name
        if generations_done(local) >= generations_done(d):
            continue
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(d, local)
        gz = local / "events.jsonl.gz"
        if gz.exists():
            with gzip.open(gz, "rb") as fi, (local / "events.jsonl").open("wb") as fo:
                shutil.copyfileobj(fi, fo)
            gz.unlink()
        print(f"[restore] {d.name} at generation {generations_done(local)}", flush=True)


def run_one(arm: str, seed: int) -> int:
    d = run_dir(arm, seed)
    d.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "dytiscidae.ops.run", "search", *BASE,
           *ARMS[arm], "--seed", str(seed), "--run", str(d), "--resume"]
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    print(f"[run] {d.name} from generation {generations_done(d)}", flush=True)
    with (RUNS / f"{d.name}.log").open("a") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        last = generations_done(d)
        while p.poll() is None:
            time.sleep(60)
            now = generations_done(d)
            if now - last >= SYNC_EVERY:
                sync()
                last = now
    sync()
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(status(), indent=1))
        return 0
    RUNS.mkdir(parents=True, exist_ok=True)
    restore()
    for arm, seed in QUEUE:
        if generations_done(run_dir(arm, seed)) >= GENERATIONS:
            continue
        rc = run_one(arm, seed)
        if rc != 0 or generations_done(run_dir(arm, seed)) < GENERATIONS:
            print(f"[stop] {arm}_s{seed} exited {rc} at generation "
                  f"{generations_done(run_dir(arm, seed))}; not starting the next",
                  flush=True)
            return 1
    print("[done] every run reached its generation budget", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
