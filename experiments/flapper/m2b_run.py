"""Pass M2b: the scored rows (PREREG_M2b_water.md).

    m2b_run.py c0                         # C0 again on build2 (the gains at rest), before any scored row
    m2b_run.py rows [--only ARM,...]      # every scored row, both entrainment modes, 4 processes

Rows go to results/M2b_rows.jsonl; each run's first line carries the sha256 of
m2_common.py, m2_laws.py, m2b_laws.py, m2b_fly.py and this file.  Scored seeds
10-15; entrainment "wetting" and the solver's default (True), co-primary.
Arms that ignore the command (W0, VB-W0, S0) and the frozen arms (WTF,
WTF-VB, SF) are run once per seed and mode and stand for both halves.
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
ROWS = HERE / "results" / "M2b_rows.jsonl"
CHECKS = HERE / "results" / "M2b_checks.jsonl"
SEEDS = list(range(10, 16))
MODES = ("wetting", True)

#: the tuned laws (filled from results/M2b_trials.jsonl before any scored row)
FROZEN = {
    # literal tuning specs (results/M2b_trials.jsonl), frozen 2026-11-19 after 44 rollouts
    "XS": dict(law="stay", g={}),                                                       # U4, U6 (GX30 met), U2, U5 (lost)
    "WT": dict(law="WT", g=dict(sgn=1.0, gain=0.2, tailfirst=1, kz_deg=-15, kh=-1.5, dorsal=1)),     # as WT-VB (U27: as built rises)
    "WT-VB": dict(law="WT", g=dict(sgn=1.0, gain=0.2, tailfirst=1, kz_deg=-15, kh=-1.5, dorsal=1)),  # U37-U40
    "ST": dict(law="ST", g=dict(f=0.6, A_deg=80, b_deg=-40, down=0.25, Psi_r_deg=55, fwd_sign=-1.0, turn_sign=-1.0)),  # U30, U31
}
#: XSt: no setting tried beat XS on the tuning seeds (U3, U9-U22), so XSt = XS (section 2) and is not run twice.

GX_ARMS = ["XS", "X0f"]
GW_ARMS = ["WT", "WT-VB", "W0", "VB-W0", "WTF", "WTF-VB"]
GS_ARMS = ["ST", "S0", "SF"]
ONCE = ("W0", "VB-W0", "S0", "WTF", "WTF-VB", "SF")


def shas():
    from m2_common import sha
    return {f: sha(HERE / f) for f in ("m2_common.py", "m2_laws.py", "m2b_laws.py", "m2b_fly.py", "m2b_run.py")}


def spec_for(arm):
    F = FROZEN
    if arm in ("W0", "VB-W0", "S0", "X0f"):
        law = dict(law="still")
    elif arm == "WTF":
        law = dict(F["WT"], frozen=True)
    elif arm == "WTF-VB":
        law = dict(F["WT-VB"], frozen=True)
    elif arm == "SF":
        law = dict(F["ST"], frozen=True)
    else:
        law = dict(F[arm])
    if arm in GX_ARMS:
        return law, "float", "air", False, 40.0
    if arm in GS_ARMS:
        return law, "float", "here", False, 40.0
    return law, "depth", "here", arm.endswith("VB") or arm == "VB-W0", 40.0


def command_for(arm, half, dr, law_spec):
    if arm in GX_ARMS:
        return {}
    if half == "A" or law_spec.get("frozen"):
        cmd = dict(h=dr["h"], z=3.0 + dr["delta"])
    else:
        cmd = dict(h=dr["h"] + math.pi, z=3.0 - dr["delta"])
    if arm in GS_ARMS:
        cmd.pop("z")
    return cmd


def run_row(job):
    from m2_common import draw, run_law, settle
    from m2b_laws import build2, make2
    arm, seed, half, mode, extra = job
    t0 = time.time()
    law_spec, start, gains, vb, T = spec_for(arm)
    dr = draw(seed)
    mc = build2(start, dr, gains=gains, vb=vb, entrainment=mode)
    if start == "float":
        settle(mc, 3.0)
    law = make2(law_spec)
    tgt = getattr(law, "law", law)
    if hasattr(tgt, "phi"):
        tgt.phi = dr["phase"]
    cmd = command_for(arm, half, dr, law_spec)
    rec = run_law(mc, law, T, cmd)
    return dict(kind="row", arm=arm, seed=seed, half=half, mode=mode, extra=extra, start=start, gains=gains,
                vb=vb, T=T, law=law_spec, cmd=cmd, draw=dr, summary=rec.summary(),
                wall_s=round(time.time() - t0, 1), log=getattr(tgt, "log", None), trace=rec.trace)


def jobs_rows(only=None):
    J = []
    for arm in GX_ARMS + GS_ARMS + GW_ARMS:
        if only and arm not in only:
            continue
        for mode in MODES:
            for s in SEEDS:
                if arm in GX_ARMS:
                    J.append((arm, s, "X", mode, None))
                    continue
                J.append((arm, s, "A", mode, None))
                if arm not in ONCE:
                    J.append((arm, s, "B", mode, None))
    return J


def c0(mode="wetting"):
    """M2's C0 on build2 (in the given entrainment mode): a machine built for an exit arm, held at 90 deg in air,
    M0's take-off stroke for 6 beats; lift against 1.840 weights."""
    import numpy as np
    from m2_common import draw
    from m2b_laws import build2
    mc = build2("float", draw(10), gains="air", entrainment=mode)
    kp = mc.kp.tolist()
    mc.place(z=30.0, pitch_deg=90.0, vx=0.0)
    mc.d.qvel[:] = 0.0
    mc.env.solver.reset()
    mc.mj.mj_forward(mc.m, mc.d)
    f, Phi, Psi = 4.5, math.radians(85), math.radians(50)
    w = 2 * math.pi * f
    q0 = mc.d.qpos[:7].copy()
    nb = int(round(1 / (f * mc.dt)))
    st, fe = mc.role == "stroke", mc.role == "feather"
    az = []
    for k in range(6 * nb):
        t = mc.t
        q = np.zeros(len(mc.aid)); qd = np.zeros(len(mc.aid))
        q[st] = Phi / 2 * math.sin(w * t); qd[st] = Phi / 2 * w * math.cos(w * t)
        q[fe] = -Psi * math.cos(w * t); qd[fe] = Psi * w * math.sin(w * t)
        vb = mc.d.qvel[:3].copy()
        mc.step(q, qd)
        if k >= 2 * nb:
            az.append((mc.d.qvel[:3] - vb) / mc.dt)
        mc.d.qpos[:7] = q0
        mc.d.qvel[:6] = 0.0
    lift = (np.mean(az, axis=0)[2] + 9.80665) / 9.80665
    row = dict(kind="C0", lift_weights=float(lift), m0=1.840, rel=float(lift / 1.840 - 1),
               ok=bool(abs(lift / 1.840 - 1) <= 0.005), mode=mode, kp=kp, bad_qacc=mc.bad_qacc, sha=shas())
    with open(CHECKS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(row)


def main():
    what = sys.argv[1]
    if what == "c0":
        return c0(True if (len(sys.argv) > 2 and sys.argv[2] == "default") else "wetting")
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    J = jobs_rows(only)
    sh = shas()
    with open(ROWS, "a") as fh:
        fh.write(json.dumps(dict(kind="hash", sha=sh, frozen=FROZEN, n=len(J), t=time.time())) + "\n")
    with Pool(4) as p:
        for row in p.imap_unordered(run_row, J):
            row["sha"] = sh
            with open(ROWS, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            s = row["summary"]
            print("%s seed %d %s %s: P %.1f W peak %.0f W qacc %d wall %.0f s" % (
                row["arm"], row["seed"], row["half"], row["mode"], s["P_elec_mean_W"], s["P_peak_50ms_all_W"],
                s["bad_qacc"], row["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
