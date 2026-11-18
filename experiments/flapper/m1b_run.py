"""Pass M1b (PREREG_M1b_takeoff.md, 78f470e): V+, P+ and Vs+ on fresh entries.

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1b_run.py tune '{"kpP": 1.5}'   # Vs+ on M1's tuning entries
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m1b_run.py score

Only calls the frozen `m1_takeoff.run`; adds no law and no read.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_takeoff as T  # noqa: E402

FROZEN_SHA = "ba0b7f69180a50eecb8b9fad6edc7d0aa2cd27c5ca9f616f2e80e9983a5344de"
OUT = HERE / "results" / "M1b_takeoff.jsonl"
TUNE = HERE / "results" / "M1b_tuning.jsonl"
VS_GAINS = HERE / "results" / "M1b_vsplus_gains.json"

UPDOWN = [(x, y) for x in (16.0, 21.0, 26.0, 31.0) for y in (0.0, 180.0)]
ACROSS = [(19.0, 90.0), (24.0, 270.0), (29.0, 90.0), (33.0, 270.0)]
OBLIQUE = [(23.0, 45.0), (28.0, 135.0)]
FRESH = UPDOWN + ACROSS + OBLIQUE
JITTER_ENTRIES = [(16.0, 0.0), (21.0, 180.0), (19.0, 90.0), (24.0, 270.0), (23.0, 45.0), (28.0, 135.0)]


def jitter(seed):
    r = np.random.default_rng(1000 + seed)
    return dict(solref_x=float(2 ** r.uniform(-1, 1)), friction=float(r.uniform(0.5, 1.1)), seed=seed)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job(a):
    arm, entry, J_r, gear, contact, gains, kind = a
    c = None if contact is None else {k: v for k, v in contact.items() if k != "seed"}
    r = T.run(arm, entry, J_r=J_r, gear=gear, gains=gains, contact=c, trace=True)
    r.update(kind=kind, passed=T.passed(r), jitter_seed=None if contact is None else contact["seed"],
             cls="updown" if tuple(entry) in UPDOWN else "across" if tuple(entry) in ACROSS else
             "oblique" if tuple(entry) in OBLIQUE else "tuning")
    return r


def tune(gains):
    jobs = [("Vs+", e, 2e-5, 24.0, None, gains, "tune") for e in T.TUNING]
    with mp.Pool(4) as p:
        for r in p.imap(job, jobs):
            with open(TUNE, "a") as fh:
                fh.write(json.dumps(dict(r, gains_given=gains)) + "\n")
            print(json.dumps(gains), T.summary_line(r), flush=True)


def rows(vs_gains):
    R = []
    for arm in ("V+", "P+", "Vs+", "N0", "N2", "VO"):
        R += [(arm, e, 2e-5, 24.0, None, vs_gains if arm == "Vs+" else None, "score") for e in FRESH]
    for J in (5e-6, 1e-6):
        R += [("V+", e, J, 32.0, None, None, "score") for e in UPDOWN]
    for s in (1, 2):
        for arm in ("V+", "P+", "Vs+"):
            R += [(arm, e, 2e-5, 24.0, jitter(s), vs_gains if arm == "Vs+" else None, "score") for e in JITTER_ENTRIES]
    return R


def score():
    assert sha(HERE / "m1_takeoff.py") == FROZEN_SHA, "m1_takeoff.py is not the frozen file"
    vs = json.loads(VS_GAINS.read_text())
    with open(OUT, "a") as fh:
        fh.write(json.dumps(dict(kind="freeze", m1_takeoff=sha(HERE / "m1_takeoff.py"), m1_lib=sha(HERE / "m1_lib.py"),
                                 mission=sha(HERE / "mission.py"), m1b_run=sha(__file__),
                                 vsplus_gains=sha(VS_GAINS), vs_gains=vs,
                                 jitter={s: jitter(s) for s in (1, 2)})) + "\n")
    R = rows(vs)
    with mp.Pool(4) as p:
        for r in p.imap_unordered(job, R):
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(T.summary_line(r), r["cls"], r["jitter_seed"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
    else:
        score()
