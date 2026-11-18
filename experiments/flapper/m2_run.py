"""Pass M2: the scored rows (PREREG_M2_water.md sections 2-4, addendum M2a).

    m2_run.py rows   [--only ARM,...]     # every scored row, 4 processes, appended to results/M2_rows.jsonl
    m2_run.py checks [--only C0,...]      # C0, C8, C9 rows, appended to results/M2_checks.jsonl

The laws are frozen in FROZEN below (tuned on seeds 100-103, results/M2_trials.jsonl);
each row carries the sha256 of m2_common.py, m2_laws.py, m2_fly.py and this file.
Scored seeds 0-5.

Rows whose two halves are by construction one rollout (an arm that ignores the
command: W0, S0, VB-W0, and the frozen arms WF, SF, whose command in half B is
half A's) are run once and that rollout stands for both halves; W0 seed 0 is run
for both halves as the check that the two are identical.
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROWS = HERE / "results" / "M2_rows.jsonl"
CHECKS = HERE / "results" / "M2_checks.jsonl"
SEEDS = list(range(6))

#: the tuned laws (filled from the tuning trials before any scored row).
FROZEN = {
    # literal tuning specs (results/M2_trials.jsonl), frozen 2026-11-18 after 61 rollouts / about 1100 s
    "WS": dict(law="WS", g=dict(kh=3.0, smin=-1.0, kd_deg=20)),                       # T54, T55
    "WV": dict(law="WV", g={}),                                                       # T56, T58
    "WK": dict(law="scull", g=dict(A_deg=0, b_deg=0, Psi_deg=40, f=0.45, kd_deg=15, kz_deg=-10, zlim_deg=10)),  # T32
    "VB": dict(law="WV", g={}),                                                       # WV on VB (T59, T60)
    "WF_of": "WV",                                                                    # the better as built (T56, T58 against T23)
    "XB": dict(law="exit", g=dict(mode="XB", b_up=0.65, fly=dict(kdy=0.25))),         # T50
    "XD": dict(law="exit", g=dict(mode="XD", pre="fly", h_sw=0.0, b_up=0.65, fly=dict(kdy=0.25))),  # T52
    # ADDED after data (T19, T20, T33): the vault from afloat, and from depth on surfacing
    "XV": dict(law="exit", g=dict(mode="XV", switch_clear=True, switch_s=2.0, q_strike=-1.0, fly=dict(kdy=0.25))),  # T42, T46-T48
    "XDV": dict(law="exit", g=dict(mode="XD", pre="vault", h_sw=0.0, switch_clear=True, switch_s=2.0,
                                   q_strike=-1.0, fly=dict(kdy=0.25))),               # T53
}

GW_ARMS = ["WV", "WK", "W0", "WF", "VB", "VB-W0"]
GS_ARMS = ["WS", "S0", "SF"]
GX_ARMS = ["XB", "XD", "X0f", "X0d", "X1f", "X1d", "XV", "XDV"]   # XV, XDV: added after data (trials T19, T20, T33)


def shas():
    from m2_common import sha
    return {f: sha(HERE / f) for f in ("m2_common.py", "m2_laws.py", "m2_fly.py", "m2_run.py")}


def spec_for(arm):
    """(law spec, start, gains, vb, T, settle)."""
    F = FROZEN
    if arm in ("W0", "S0", "X0f", "X0d", "VB-W0"):
        law = dict(law="still")
    elif arm == "WF":
        law = dict(F[F["WF_of"]], frozen=True)
    elif arm == "SF":
        law = dict(F["WS"], frozen=True)
    elif arm == "X1f":
        law = dict(F["WS"])
    elif arm == "X1d":
        law = dict(F[F["WF_of"]])
    elif arm == "VB":
        law = dict(F["VB"])
    else:
        law = dict(F[arm])
    SURF = ("WS", "S0", "SF")
    start = "float" if arm in SURF else ("depth" if arm[0] in "WV" else None)
    if arm in ("XB", "X0f", "X1f", "XV"):
        start = "float"
    if arm in ("XD", "X0d", "X1d", "XDV"):
        start = "depth"
    gains = "air" if arm.startswith("X") else "here"
    vb = arm.startswith("VB")
    if arm in SURF or arm.startswith(("W", "V")):
        T = 40.0
    elif arm.startswith("S"):
        T = 40.0
    else:
        T = 20.0 if start == "float" else 40.0
    return law, start, gains, vb, T


def command_for(arm, half, dr, law_spec):
    if arm in ("WS", "S0", "SF") or arm.startswith(("W", "V")):
        if half == "A" or law_spec.get("frozen"):
            # a frozen law keeps the first command it is given: half A's
            cmd = dict(h=dr["h"], z=3.0 + dr["delta"])
        else:
            cmd = dict(h=dr["h"] + math.pi, z=3.0 - dr["delta"])
        if arm in ("WS", "S0", "SF"):
            cmd.pop("z")
        return cmd
    if arm == "X1f":
        return dict(h=math.pi)
    if arm == "X1d":
        return dict(h=math.pi, z=3.0)
    return {}


def run_row(job):
    import numpy as np
    from m2_common import build, draw, run_law, settle
    from m2_laws import make
    arm, seed, half, extra = job
    t0 = time.time()
    law_spec, start, gains, vb, T = spec_for(arm)
    dr = draw(seed)
    kw = dict(extra or {})
    motors = kw.pop("motors", None)
    mc = build(start, dr, gains=gains, vb=vb, entrainment=kw.pop("entrainment", "wetting"),
               motors=motors, dt=kw.pop("dt", 1e-4))
    if start == "float":
        settle(mc, 3.0)
    law = make(law_spec)
    if hasattr(law, "phi"):
        law.phi = dr["phase"]
    if hasattr(law, "law"):
        law.law.phi = dr["phase"]
    cmd = command_for(arm, half, dr, law_spec)
    rec = run_law(mc, law, T, cmd)
    row = dict(kind="row", arm=arm, seed=seed, half=half, extra=extra, start=start, gains=gains, vb=vb, T=T,
               law=law_spec, cmd=cmd, draw=dr, summary=rec.summary(), wall_s=round(time.time() - t0, 1),
               exit_log=getattr(law, "log", None), trace=rec.trace)
    return row


def jobs_rows(only=None):
    J = []
    once = ("W0", "S0", "VB-W0", "WF", "SF")
    for arm in GW_ARMS + GS_ARMS:
        if only and arm not in only:
            continue
        for s in SEEDS:
            J.append((arm, s, "A", None))
            if arm not in once or (arm == "W0" and s == 0):
                J.append((arm, s, "B", None))
    for arm in GX_ARMS:
        if only and arm not in only:
            continue
        for s in SEEDS:
            J.append((arm, s, "X", None))
    return J


def main():
    what = sys.argv[1]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    sh = shas()
    if what == "rows":
        J = jobs_rows(only)
        out = ROWS
    else:
        from m2_checks import jobs_checks
        J = jobs_checks(only)
        out = CHECKS
    with open(out, "a") as fh:
        fh.write(json.dumps(dict(kind="hash", what=what, sha=sh, frozen=FROZEN, t=time.time())) + "\n")
    with Pool(4) as p:
        for row in p.imap_unordered(run_row, J):
            row["sha"] = sh
            with open(out, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            s = row["summary"]
            print("%s seed %d %s: P %.1f W peak %.0f W qacc %d wall %.0f s" % (
                row["arm"], row["seed"], row["half"], s["P_elec_mean_W"], s["P_peak_50ms_all_W"],
                s["bad_qacc"], row["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
