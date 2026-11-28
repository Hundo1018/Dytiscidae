"""Pass M4, phase 2: the frozen gates and checks of PREREG_M4_cruise_entries.md
section 3 read from the rows and their 10 Hz traces.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m4_read.py cruise|entries|tune [FILE]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TR = HERE / "results" / "M4_traces"


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def rows(path):
    out = {}
    for x in open(path):
        if x.strip():
            r = json.loads(x)
            out[r["id"]] = r                    # the last row per id
    return out


def trace(rid):
    z = np.load(TR / f"{rid}.npz")
    return {k: z[k] for k in z.files}


def mean1s(a):
    """Trailing 1 s mean of a 10 Hz series."""
    a = np.asarray(a, float)
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.empty_like(a)
    for i in range(len(a)):
        j = max(0, i - 9)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


# ---------------------------------------------------------------- cruise
def cruise_half(r, T, cmd, kind):
    """kind 'H' or 'T'. Returns (pass, dict of quantities)."""
    t_c = r["t_c"]
    z0 = 30.0
    t = T["t"]
    trk_u = np.degrees(np.unwrap(np.radians(T["trk"])))
    trk1 = mean1s(trk_u)
    V1 = mean1s(T["V"])
    full = r["end"] == "time" and r["t_end"] >= t_c + 60.0 - 1e-3
    q = dict(full=full, end=r["end"], qacc=r["bad_qacc"])
    w = (t >= t_c) & (t <= t_c + 60.0 + 1e-6)
    i_c = int(np.argmin(np.abs(t - t_c)))
    late = t >= t_c + 50.0
    dpsi = float(trk_u[late].mean() - trk_u[i_c]) if late.any() else float(trk_u[-1] - trk_u[i_c])
    q["dpsi"] = dpsi
    if kind == "H":
        hold = w
        turn = np.zeros_like(w)
    else:
        turn = w & (t < t_c + 15.0)
        hold = w & (t >= t_c + 15.0)
    e = np.abs(wrap(trk1 - cmd))
    q.update(trk_err_max=float(e[hold].max()) if hold.any() else 999.0,
             roll_max_hold=float(np.abs(T["roll"][hold]).max()) if hold.any() else 999.0,
             pitch_max_hold=float(np.abs(T["pitch"][hold]).max()) if hold.any() else 999.0,
             dz_hold=[float((T["z"][hold] - z0).min()), float((T["z"][hold] - z0).max())] if hold.any() else [-99, 99],
             V_hold=[float(V1[hold].min()), float(V1[hold].max())] if hold.any() else [0, 0],
             beta_max_hold=float(np.abs(wrap(T["yaw"][hold] - T["trk"][hold])).max()) if hold.any() else 999.0)
    ok_hold = (hold.any() and q["trk_err_max"] <= 5.0 and q["roll_max_hold"] <= 10.0 and q["pitch_max_hold"] <= 10.0
               and -1.0 <= q["dz_hold"][0] and q["dz_hold"][1] <= 1.0 and 7.0 <= q["V_hold"][0] and q["V_hold"][1] <= 9.5)
    ok = full and r["bad_qacc"] == 0 and ok_hold
    if kind == "T":
        q.update(roll_max_turn=float(np.abs(T["roll"][turn]).max()), pitch_max_turn=float(np.abs(T["pitch"][turn]).max()),
                 dz_turn=[float((T["z"][turn] - z0).min()), float((T["z"][turn] - z0).max())],
                 V_turn=[float(V1[turn].min()), float(V1[turn].max())])
        sgn = np.sign(wrap(cmd - T["trk"][i_c]))
        q["sign_ok"] = bool(np.sign(dpsi) == sgn)
        ok_turn = (q["roll_max_turn"] <= 35.0 and q["pitch_max_turn"] <= 15.0 and -2.0 <= q["dz_turn"][0]
                   and q["dz_turn"][1] <= 2.0 and 6.5 <= q["V_turn"][0] and q["V_turn"][1] <= 10.0 and q["sign_ok"])
        q["ok_turn_phase"], q["ok_hold_phase"] = bool(ok_turn), bool(ok_hold)
        ok = ok and ok_turn
        # turn shape
        ww = w & np.isfinite(e)
        e_raw = np.abs(wrap(T["trk"] - cmd))
        idx = np.where(ww & (e_raw > 10.0))[0]
        q["t_in10"] = float(t[idx[-1]] - t_c) if len(idx) else 0.0
        over = (trk_u[w] - trk_u[i_c]) * sgn
        q["overshoot"] = float(over.max() - 90.0)
        q["bank_peak"] = float(np.abs(T["roll"][w]).max())
    else:
        q["ok_hold_phase"] = bool(ok_hold)
    return bool(ok), q


def cruise(path=HERE / "results" / "M4_cruise.jsonl"):
    R = rows(path)
    starts = ["K1", "K2", "K3", "K4", "K5", "K6"]
    psi0 = dict(K1=0.0, K2=60.0, K3=135.0, K4=-45.0, K5=-120.0, K6=180.0)
    res = {}
    for pt in ("P35", "P25"):
        for arm in ("L", "L0", "N"):
            for s in starts:
                if arm == "L":
                    ids = dict(H=f"{pt}_{s}_L_H", Tp=f"{pt}_{s}_L_Tp", Tm=f"{pt}_{s}_L_Tm")
                else:
                    i = f"{pt}_{s}_{arm}"
                    ids = dict(H=i, Tp=i, Tm=i)
                if not all(i in R for i in ids.values()):
                    continue
                rr = {k: R[v] for k, v in ids.items()}
                TT = {k: trace(v) for k, v in ids.items()}
                h_ok, hq = cruise_half(rr["H"], TT["H"], psi0[s], "H")
                a_ok, aq = cruise_half(rr["Tp"], TT["Tp"], psi0[s] + 90.0, "T")
                b_ok, bq = cruise_half(rr["Tm"], TT["Tm"], psi0[s] - 90.0, "T")
                P_t = 0.5 * (aq["dpsi"] - bq["dpsi"])
                hashes = len({rr[k]["hash"] for k in rr}) == 1
                t_ok = a_ok and b_ok and P_t >= 60.0 and hashes
                res[(pt, arm, s)] = dict(H=h_ok, T=t_ok, P_t=P_t, hash_eq=hashes, Hq=hq, Aq=aq, Bq=bq,
                                         P_H=rr["H"]["win"]["P_mean_W"], P_A=rr["Tp"]["win"]["P_mean_W"],
                                         pk50=max(rr[k]["win"]["P_peak50_W"] for k in rr),
                                         tq=rr["Tp"]["win"]["tq_peak"], end=rr["H"]["end"], t_end=rr["H"]["t_end"] - rr["H"]["t_c"])
    return R, res


def print_cruise():
    R, res = cruise()
    for (pt, arm, s), v in sorted(res.items()):
        h, a, b = v["Hq"], v["Aq"], v["Bq"]
        print(f"{pt} {arm:3s} {s} H {'PASS' if v['H'] else 'fail'} T {'PASS' if v['T'] else 'fail'} P_t {v['P_t']:7.2f} "
              f"hash= {v['hash_eq']} | H trk {h['trk_err_max']:.2f} roll {h['roll_max_hold']:.2f} pitch {h['pitch_max_hold']:.1f} "
              f"dz {h['dz_hold'][0]:+.2f}/{h['dz_hold'][1]:+.2f} V {h['V_hold'][0]:.2f}-{h['V_hold'][1]:.2f} beta {h['beta_max_hold']:.1f} "
              f"end {v['end']} {v['t_end']:.1f}s | T+ bank {a.get('bank_peak', 0):.1f} over {a.get('overshoot', 0):.1f} "
              f"t10 {a.get('t_in10', 0):.2f} dz {a.get('dz_turn', [0, 0])[0]:+.2f}/{a.get('dz_turn', [0, 0])[1]:+.2f} "
              f"trk {a['trk_err_max']:.2f} | T- bank {b.get('bank_peak', 0):.1f} t10 {b.get('t_in10', 0):.2f} trk {b['trk_err_max']:.2f} "
              f"| P {v['P_H']:.1f}/{v['P_A']:.1f} W pk50 {v['pk50']:.0f}")
    # pairing checks
    for pt in ("P35", "P25"):
        a, b = R.get(f"{pt}_K1_F_a"), R.get(f"{pt}_K1_F_b")
        if a and b:
            Ta, Tb = trace(a["id"]), trace(b["id"])
            print(pt, "F identical:", a["hash"] == b["hash"], "max |qpos diff| over trace", float(np.abs(Ta["qpos"] - Tb["qpos"]).max()))
        a, b = R.get(f"{pt}_K1_L0"), R.get(f"{pt}_K1_L0_b")
        if a and b:
            Ta, Tb = trace(a["id"]), trace(b["id"])
            print(pt, "L0 repeat identical: max |qpos diff|", float(np.abs(Ta["qpos"] - Tb["qpos"]).max()))
    for pt in ("P35", "P25"):
        for arm in ("L", "L0", "N"):
            sel = [v for (p, a, s), v in res.items() if p == pt and a == arm]
            if sel:
                print(f"{pt} {arm}: H {sum(v['H'] for v in sel)}/{len(sel)}  T {sum(v['T'] for v in sel)}/{len(sel)}")


# ---------------------------------------------------------------- entries
XE = dict(water=(-45.0, -33.0), land=(16.0, 28.0))


def entry_half(r, T, x_e, medium):
    q = dict(end=r["end"], qacc=r["bad_qacc"])
    c = r.get("contact") or {}
    if "x" not in c:
        q["fail"] = "no contact" if not c else c.get("medium", "?")
        return False, q
    tcn = c["t"]
    Tw = 10.0 if medium == "water" else 6.0
    q.update(x=c["x"], dx=c["x"] - x_e, vn=c["vn"], v_h=c["v_h"], roll=c["roll"], pitch_rel=c["pitch_rel"],
             phase=c.get("phase"))
    e1 = abs(c["x"] - x_e) <= 3.0
    e2 = c["vn"] >= -1.0 and abs(c["roll"]) <= 15.0 and -5.0 <= c["pitch_rel"] <= 30.0
    t = T["t"]
    complete = r["t_end"] >= tcn + Tw - 0.05
    after = (t >= tcn) & (t <= tcn + Tw + 1e-6)
    zbz = T["zbz"]
    e4 = bool((zbz[after] >= -0.25).all())
    com = T["com"]
    comv = np.linalg.norm(T["comv"], axis=1)
    if medium == "water":
        last = after & (t >= tcn + Tw - 2.0)
        v1 = mean1s(comv)
        q.update(com_z_max=float(com[last, 2].max()), v1_max=float(v1[last].max()), om_mean=float(T["om"][last].mean()),
                 zbz_min=float(zbz[last].min()))
        e3 = q["com_z_max"] <= 0.3 and q["v1_max"] <= 0.3 and q["om_mean"] <= 0.5 and q["zbz_min"] >= 0.5
    else:
        last = after & (t >= tcn + Tw - 1.0)
        gh, gw, gt = T["gz_hull"][last].sum(), T["gz_wing"][last].sum(), T["gz_tail"][last].sum()
        share = float((gh + gw) / max(gh + gw + gt, 1e-12))
        q.update(v_max=float(comv[last].max()), om_max=float(T["om"][last].max()), zbz_min=float(zbz[last].min()),
                 x_rest=float(com[last, 0].mean()), share_hull_wing=share, slide=float(np.linalg.norm(
                     com[last][-1, :2] - np.array(c["com"][:2]))))
        e3 = (q["v_max"] <= 0.05 and q["om_max"] <= 0.2 and q["zbz_min"] >= 0.5 and 8.0 <= q["x_rest"] <= 39.8
              and share >= 0.8)
    q.update(E1=bool(e1), E2=bool(e2), E3=bool(e3 and complete), E4=e4, E5=r["bad_qacc"] == 0)
    q["loads"] = r.get("loads")
    q["loads_10ms"] = r.get("loads_10ms")
    q["win"] = r["win"]
    return bool(e1 and e2 and e3 and complete and e4 and r["bad_qacc"] == 0), q


def entries(medium, path=None):
    R = rows(path or HERE / "results" / "M4_entries.jsonl")
    near, far = XE[medium]
    na = "NW" if medium == "water" else "NL"
    res = {}
    for s in ("S1", "S2", "S3", "S4"):
        for arm in ("E", na):
            ids = (f"{medium}_{s}_E_near", f"{medium}_{s}_E_far") if arm == "E" else (f"{medium}_{s}_{na}",) * 2
            if not all(i in R for i in ids):
                continue
            a_ok, aq = entry_half(R[ids[0]], trace(ids[0]), near, medium)
            b_ok, bq = entry_half(R[ids[1]], trace(ids[1]), far, medium)
            P_e = (bq["x"] - aq["x"]) / (far - near) if "x" in aq and "x" in bq else None
            hashes = R[ids[0]]["hash"] == R[ids[1]]["hash"]
            res[(arm, s)] = dict(ok=a_ok and b_ok and hashes, A=aq, B=bq, P_e=P_e, hash_eq=hashes)
    return R, res


def print_entries(medium):
    R, res = entries(medium)
    for (arm, s), v in sorted(res.items()):
        print(f"{medium} {arm} {s} {'PASS' if v['ok'] else 'fail'} P_e {v['P_e']} hash= {v['hash_eq']}")
        for h in ("A", "B"):
            q = v[h]
            print("   ", h, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in q.items()
                             if k not in ("loads", "loads_10ms", "win")})
            if q.get("loads"):
                print("      loads", {k: round(x, 2) for k, x in q["loads"].items()})
                print("      10ms ", {k: round(x, 2) for k, x in q["loads_10ms"].items()})
            if q.get("win"):
                w = q["win"]
                print("      power mean %.1f pk50 %.0f step %.0f tq %s" % (w["P_mean_W"], w["P_peak50_W"], w["P_peak_step_W"],
                                                                         {k: round(x, 2) for k, x in w["tq_peak"].items()}))
    for (a, b) in (("EF_a", "EF_b"),):
        ia, ib = f"{medium}_S1_{a}", f"{medium}_S1_{b}"
        if ia in R and ib in R:
            print("EF identical:", float(np.abs(trace(ia)["qpos"] - trace(ib)["qpos"]).max()))


def print_tune(path):
    R = rows(path)
    for rid, r in R.items():
        med = r["kind"]
        ok, q = entry_half(r, trace(rid), r["cmd"], med)
        print(rid, "PASS" if ok else "fail", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in q.items()
                                              if k not in ("loads", "loads_10ms", "win")},
              "P %.0f pk50 %.0f" % (r["win"]["P_mean_W"], r["win"]["P_peak50_W"]))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "cruise":
        print_cruise()
    elif what == "entries":
        print_entries(sys.argv[2])
    else:
        print_tune(sys.argv[2])
