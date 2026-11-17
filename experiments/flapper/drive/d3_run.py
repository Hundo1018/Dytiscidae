"""Pass D3: what makes the spanwise twist at cruise (`PREREG_D3_tip_twist.md`).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d3_run.py grid OUT.jsonl
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/drive/d3_run.py trim GRID.jsonl OUT.jsonl

A trailing ``dt=5e-5`` runs the same rows at that step (loops at solref 2 dt):
added after the frozen grid was read, RESULT_D3 says why.

``grid``: the baselines, T0, TX, TP, TC at the root feather of -2.94 deg, and
the vacuum rows (H1).  ``trim``: the second read (H9), the root feather moved
until the lift is the weight, for T0, TX and any TP or TC row whose lift at
-2.94 was 0.7-1.3.  At most 3 processes.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
DT = 1e-4
V, F, PHI, A_TIP, PSI0, LEVER, GEAR = 8.0, 2.5, 40.0, 15.0, -2.94, 0.03, 48
CM = dict(camber_moment=True)


def one(cand, tw, chain, psi0, vacuum, cm=True, dt=DT):
    import l1a
    from freerig import run_free
    from dytiscidae.physics.medium import Fluid
    body = dict(l1a.BODY_S, wing_shape=dict(camber=0.04), wing_section="goe417a+lift")
    kw = dict(feather_axis=0.1, cop_law="plate", dt=dt)
    if cm:
        kw["solver_flags"] = CM
    if vacuum:
        kw["fluid"] = Fluid("vacuum", 1e-9, 1.81e-5)
    if cand in ("base", "B"):
        dv = None
        if cand == "B":
            from drive import Drive
            dv = Drive("FB", PHI, F, GEAR, a=LEVER, dt=dt, feather_axis=0.1)
        r = run_free(body, V, 0.0, F, PHI, A_TIP, psi0, drive=dv,
                     strip_twist=dict(tip_deg=0.0, share=1.0), **kw)
    else:
        from twist import TwistDrive
        body["wing_panels"] = 6
        dv = TwistDrive(PHI, F, GEAR, a=LEVER, dt=dt, feather_axis=0.1, twist=tw, psi_root_deg=psi0)
        r = run_free(body, V, 0.0, F, PHI, 0.0, psi0, drive=dv, chain=chain, **kw)
    # mechanical: motor shaft plus ideal servos (freerig's power_W); electrical:
    # motor shaft plus copper, plus the ideal servos' net power
    r["mech_W"] = r["power_W"]
    if dv is not None:
        r["elec_W"] = r["drive"]["elec_W"] + r["feather_power_W"] + (
            r.get("twist_power_W", 0.0) if r.get("twist_mode") == "servo" else 0.0)
    return r


def job(a):
    cand, tw, chain, vacuum, cm = a["cand"], a.get("tw"), a.get("chain"), a.get("vacuum", False), a.get("cm", True)
    dt = a.get("dt", DT)
    head = dict(cand=cand, tw=tw, chain=chain, vacuum=vacuum, cm=cm, label=a["label"], dt=dt)
    try:
        if not a.get("trim"):
            r = one(cand, tw, chain, PSI0, vacuum, cm, dt)
        else:
            # W7's secant on the root feather, lift = weight within 1%, at most 6 runs
            x0, x1 = PSI0, PSI0 - 0.6
            r0, r1 = one(cand, tw, chain, x0, False, True, dt), one(cand, tw, chain, x1, False, True, dt)
            r, evals = min((r0, r1), key=lambda q: abs(q["lift_over_weight"] - 1.0)), 2
            while abs(r["lift_over_weight"] - 1.0) > 0.01 and evals < 6:
                y0, y1 = r0["lift_over_weight"] - 1.0, r1["lift_over_weight"] - 1.0
                if abs(y1 - y0) < 1e-9:
                    break
                x2 = min(max(x1 - y1 * (x1 - x0) / (y1 - y0), -30.0), 25.0)
                r2 = one(cand, tw, chain, x2, False, True, dt)
                evals += 1
                x0, r0, x1, r1 = x1, r1, x2, r2
                if abs(r2["lift_over_weight"] - 1.0) < abs(r["lift_over_weight"] - 1.0):
                    r = r2
            r = dict(r, evals=evals, trimmed=bool(abs(r["lift_over_weight"] - 1.0) <= 0.03))
    except Exception as e:                       # a row, not a crash
        import traceback
        return dict(head, error=repr(e), tb=traceback.format_exc())
    r.update(head)
    r.pop("series", None)
    return r


def grid_rows():
    g = [dict(cand="base", label="prescribed, strips", cm=True), dict(cand="base", label="prescribed, strips, no Cm", cm=False),
         dict(cand="B", label="B strips", cm=True), dict(cand="B", label="B strips, no Cm", cm=False),
         dict(cand="T0", label="T0 servos", tw=dict(mode="servo"), chain=dict(mode="servo", tip_deg=A_TIP))]
    for k in (3.0, 10.0, 30.0):
        g.append(dict(cand="TX", label="TX k %g" % k, tw=dict(mode="crank", c_h=0.03), chain=dict(mode="passive", k=k)))
    for rest in (0.0, -15.0):
        for k in (1.0, 3.0, 10.0, 30.0):
            g.append(dict(cand="TP", label="TP k %g rest %g" % (k, rest), tw=dict(mode="spring", rest_deg=rest),
                          chain=dict(mode="passive", k=k)))
        for k in (1.0, 3.0):
            g.append(dict(cand="TC", label="TC k %g rest %g" % (k, rest),
                          tw=dict(mode="cable", post=0.08, rest_deg=rest), chain=dict(mode="passive", k=k)))
    vac = [x for x in g if x["label"] in ("T0 servos", "TX k 10", "TP k 3 rest 0", "TC k 3 rest 0", "TC k 1 rest 0")]
    return g + [dict(x, vacuum=True, label=x["label"] + ", vacuum") for x in vac]


def main() -> int:
    import multiprocessing as mp
    stage = sys.argv[1]
    dt = DT
    if sys.argv[-1].startswith("dt="):
        dt = float(sys.argv.pop()[3:])
    if stage == "grid":
        grid = grid_rows()
    else:
        rows = [json.loads(x) for x in open(sys.argv[2])]
        grid = []
        for r in rows:
            if r.get("vacuum") or "error" in r or r["cand"] in ("base", "B"):
                continue
            if r["cand"] in ("T0", "TX") or 0.7 <= r["lift_over_weight"] <= 1.3:
                tw = dict(r["tw"])
                if r["cand"] == "TC":       # the cable as built at -2.94: its anchor and length stay
                    tw.update(h=r["drive"]["twist"]["design"]["h"], L0=r["drive"]["twist"]["design"]["L0"])
                grid.append(dict(cand=r["cand"], label=r["label"] + ", trimmed", tw=tw, chain=r["chain"], trim=True))
    grid = [dict(x, dt=dt) for x in grid]
    out = Path(sys.argv[-1]).resolve()
    with mp.get_context("spawn").Pool(3) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(job, grid):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("done", r["label"], r.get("error", ""), flush=True)
    print("rows", len(grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
