"""Pass M4, phase 2: the scored and tuning rows (PREREG_M4_cruise_entries.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/m4_run.py GROUP OUT.jsonl WORKERS [ARG]

GROUP: cruise (ARG = P35 or P25), extra (LS and dt), entries (ARG = water or
land; law from results/M4_frozen.json), sens (entrainment sensitivity, ARG
water), tune (ARG = a json file: list of dict(medium, E, start, cmd, tag)).
Each row is one half; its 10 Hz trace goes to results/M4_traces/<id>.npz.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TR = HERE / "results" / "M4_traces"

K = dict(K1=(5.0, 0.0, 8.0), K2=(-5.0, 60.0, 8.0), K3=(3.0, 135.0, 8.4), K4=(-3.0, -45.0, 7.6),
         K5=(0.0, -120.0, 8.2), K6=(8.0, 180.0, 7.8))
S = dict(S1=(4.0, 0.0, 8.0, 0.0), S2=(-4.0, 3.0, 8.3, 0.3), S3=(2.0, -3.0, 7.7, -0.3), S4=(-2.0, 6.0, 8.0, 0.0))
U = dict(U1=(6.0, -5.0, 8.1, 0.2), U2=(-6.0, 5.0, 7.9, -0.2))
XE = dict(water=(-45.0, -33.0), land=(16.0, 28.0))


def _entry_start(st):
    roll0, psi0, V0, dz0 = st
    return (roll0, psi0, V0, dz0)


def cruise_specs(pt):
    out = []
    for name, (r0, p0, v0) in K.items():
        st = (r0, p0, v0)
        base = dict(kind="cruise", point=pt, start=st, start_name=name)
        out += [dict(base, id=f"{pt}_{name}_L_H", arm="L", half="H", cmd=p0),
                dict(base, id=f"{pt}_{name}_L_Tp", arm="L", half="T+", cmd=p0 + 90.0),
                dict(base, id=f"{pt}_{name}_L_Tm", arm="L", half="T-", cmd=p0 - 90.0),
                dict(base, id=f"{pt}_{name}_L0", arm="L0", half="all", cmd=p0),
                dict(base, id=f"{pt}_{name}_N", arm="N", half="all", cmd=p0)]
    r0, p0, v0 = K["K1"]
    base = dict(kind="cruise", point=pt, start=(r0, p0, v0), start_name="K1")
    out += [dict(base, id=f"{pt}_K1_F_a", arm="L", half="F", cmd=p0 + 90.0),
            dict(base, id=f"{pt}_K1_F_b", arm="L", half="F", cmd=p0 + 90.0),
            dict(base, id=f"{pt}_K1_L0_b", arm="L0", half="all", cmd=p0)]
    return out


def extra_specs():
    r0, p0, v0 = K["K1"]
    base = dict(kind="cruise", point="P35", start=(r0, p0, v0), start_name="K1")
    return [dict(base, id="P35_K1_LS_H", arm="LS", half="H", cmd=p0),
            dict(base, id="P35_K1_LS_Tp", arm="LS", half="T+", cmd=p0 + 90.0),
            dict(base, id="P35_K1_L_Tp_dt5e-5", arm="L", half="T+", cmd=p0 + 90.0, dt=5e-5)]


def frozen(medium):
    return json.loads((HERE / "results" / "M4_frozen.json").read_text())[medium]["E"]


def entry_specs(medium):
    E = frozen(medium)
    near, far = XE[medium]
    na = "NW" if medium == "water" else "NL"
    out = []
    for name, st in S.items():
        base = dict(kind=medium, point="P35", start=st, start_name=name)
        out += [dict(base, id=f"{medium}_{name}_E_near", arm="E", half="near", cmd=near, E=E),
                dict(base, id=f"{medium}_{name}_E_far", arm="E", half="far", cmd=far, E=E),
                dict(base, id=f"{medium}_{name}_{na}", arm=na, half="all", cmd=near, E=E)]
    base = dict(kind=medium, point="P35", start=S["S1"], start_name="S1")
    out += [dict(base, id=f"{medium}_S1_EF_a", arm="E", half="F", cmd=far, E=E),
            dict(base, id=f"{medium}_S1_EF_b", arm="E", half="F", cmd=far, E=E)]
    return out


def sens_specs(medium, start_name):
    E = frozen(medium)
    near, far = XE[medium]
    base = dict(kind=medium, point="P35", start=S[start_name], start_name=start_name, ent=True)
    return [dict(base, id=f"{medium}_{start_name}_E_near_entdefault", arm="E", half="near", cmd=near, E=E),
            dict(base, id=f"{medium}_{start_name}_E_far_entdefault", arm="E", half="far", cmd=far, E=E),
            dict(base, id=f"{medium}_S1_NW_entdefault", arm="NW", half="all", cmd=near, E=E, start=S["S1"],
                 start_name="S1")]


def tune_specs(path):
    out = []
    for i, t in enumerate(json.loads(Path(path).read_text())):
        st = U[t["start"]]
        out.append(dict(kind=t["medium"], point="P35", start=st, start_name=t["start"], arm="E",
                        half=t.get("half", ""), cmd=t["cmd"], E=t["E"], id=t["tag"], tune=True,
                        win=10.0))      # contact within 10 s of t_c: a rollout is at most 25 s simulated
    return out


def job(spec):
    import m4_lib
    t0 = time.time()
    try:
        out, T = m4_lib.run_half(spec)
    except Exception:
        import traceback
        return dict(id=spec["id"], error=traceback.format_exc()[-1500:], spec=spec)
    TR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TR / f"{spec['id']}.npz", **{k: (np.array(v) if k != "phase" else np.array(v, dtype="U10"))
                                                     for k, v in T.items()})
    out["half"] = spec.get("half")
    out["tune"] = spec.get("tune", False)
    out["wall_s"] = time.time() - t0
    return out


def main() -> int:
    group, outp, nw = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3])
    arg = sys.argv[4] if len(sys.argv) > 4 else None
    specs = dict(cruise=lambda: cruise_specs(arg), extra=extra_specs, entries=lambda: entry_specs(arg),
                 sens=lambda: sens_specs(arg, sys.argv[5]), tune=lambda: tune_specs(arg))[group]()
    done = set()
    if outp.exists():
        done = {json.loads(x)["id"] for x in open(outp) if x.strip() and "error" not in json.loads(x)}
    specs = [s for s in specs if s["id"] not in done]
    print(f"{group} {arg}: {len(specs)} halves to run", flush=True)
    with mp.get_context("spawn").Pool(nw) as pool, open(outp, "a") as fh:
        for r in pool.imap_unordered(job, specs):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if "error" in r:
                print(r["id"], "ERROR", r["error"][-300:], flush=True)
            else:
                c = r.get("contact") or {}
                print(r["id"], r["end"], "t_end %.1f" % r["t_end"], "qacc", r["bad_qacc"], "P %.1f W" % r["win"]["P_mean_W"],
                      ("contact x %.2f vn %.2f" % (c["x"], c["vn"])) if "x" in c else "", "wall %.0f s" % r["wall_s"],
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
