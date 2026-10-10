"""R3: why the stored mobility basis fails land but not air (arch49).

For a few R1 batched non-reproducers, four single-machine arms:
  A stored   identify_axes=False, controller bases = meta["mobility_basis"]
  B fresh    identify_axes=True (the batch identifies and overwrites)
  C id+stor  identify_axes=True, but identify_batch returns the stored bases
             (identification side effects happen, stored values are used)
  D inject   identify_axes=False, controller bases = B's fresh bases
  E          D with fresh modes copied to C order (values unchanged)
  F          A with stored modes copied to Fortran order (values unchanged)
Then the stored vs fresh arrays are compared, and the land rollout of A vs B is
traced (initial qpos/qvel, env.rng state, per-step commanded coefficients and
qpos) to the first divergence.

  flock /home/hundo/.cache/dytiscidae-gpu.lock env MUJOCO_GL=disable PYTHONPATH=$PWD \
     DYTISCIDAE_KERNEL_DIR=<main>/mojo/build <main>/.venv/bin/python -u \
     experiments/contact_model_dependence/r3.py --run <main>/runs/arch49 [--elites 18 23 25]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "experiments/shared_policy_value")
sys.path.insert(0, "experiments/contact_model_dependence")
D = Path("experiments/contact_model_dependence")
NTRACE = 400

TR = {"on": False, "dom": None, "cmds": [], "qpos": [], "entry": None}


def instrument():
    from dytiscidae.envs import batchroll as br
    from dytiscidae.control import cpg
    orig_roll, orig_step, orig_cmd = br.rollout_batch, br.step_batch, cpg.MobilityBasis.command_params

    def roll(envs, bf, seg, params_list, domain, *a, **k):
        TR["dom"] = getattr(domain, "value", str(domain))
        if TR["on"] and TR["dom"] == "land" and TR["entry"] is None:
            e = envs[0]
            TR["entry"] = dict(qpos=e.data.qpos.copy(), qvel=e.data.qvel.copy(),
                               time=float(e.data.time),
                               rng=json.dumps(e.rng.bit_generator.state, default=str),
                               params=params_list[0].flat().copy(),
                               cpg_base=e.cpg.base.flat().copy())
        try:
            return orig_roll(envs, bf, seg, params_list, domain, *a, **k)
        finally:
            TR["dom"] = None

    def step(envs, angles, bf, live, *a, **k):
        out = orig_step(envs, angles, bf, live, *a, **k)
        if TR["on"] and TR["dom"] == "land" and TR["entry"] is not None and len(TR["qpos"]) < NTRACE:
            TR["qpos"].append(np.concatenate([envs[0].data.qpos, envs[0].data.qvel]).copy())
        return out

    def cmd(self, base, coeffs, n, gain=1.0):
        out = orig_cmd(self, base, coeffs, n, gain)
        if TR["on"] and TR["dom"] == "land" and len(TR["cmds"]) < NTRACE:
            TR["cmds"].append(np.concatenate([np.asarray(coeffs, float), out.flat()]))
        return out

    br.rollout_batch, br.step_batch = roll, step
    cpg.MobilityBasis.command_params = cmd
    return br


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--elites", type=int, nargs="*", default=[18, 23, 25])
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    br = instrument()
    orig_ident = br.identify_batch
    from r2 import setup, mk
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.control.cpg import MobilityBasis
    elites, seg, nm, r1, pick, non, rep = setup(a.run)
    out = open(D / f"r3{a.tag}.rows.jsonl", "w")
    t0 = time.time()
    for idx in a.elites:
        x = mk(a.run, elites, idx)
        stored = MobilityBasis.bases_from_record((elites[idx].meta or {}).get("mobility_basis"))
        rec_land = (elites[idx].meta or {}).get("land")
        fresh_holder = {}
        traces = {}

        def run_arm(arm, bases, identify, ident_fn):
            br.identify_batch = ident_fn
            TR.update(on=True, dom=None, cmds=[], qpos=[], entry=None)
            c = Controller(params=x["c"].params, policy=x["c"].policy,
                           bases=None if bases is None else copy.deepcopy(bases))
            r = br.evaluate_tier1_batch([x["p"]], controllers=[c], segment_seconds=seg,
                                        identify_axes=identify, seed=[x["seed"]],
                                        shared=x["net"], n_modes=nm)[0]
            TR["on"] = False
            br.identify_batch = orig_ident
            traces[arm] = dict(entry=TR["entry"], cmds=np.array(TR["cmds"]), qpos=np.array(TR["qpos"]),
                               bases=c.bases)
            comp = {m: float(r.segments[m].competence) for m in ("air", "water", "land")}
            return comp

        def ident_fresh(envs, dom, **k):
            f = orig_ident(envs, dom, **k)
            fresh_holder[dom.value] = f[0]
            return f

        def ident_stored(envs, dom, **k):
            orig_ident(envs, dom, **k)          # side effects of identifying, value discarded
            return [copy.deepcopy(stored[dom.value])]

        res = {}
        res["A_stored"] = run_arm("A", stored, False, orig_ident)
        res["B_fresh"] = run_arm("B", None, True, ident_fresh)
        res["C_ident_then_stored"] = run_arm("C", None, True, ident_stored)
        res["D_inject_fresh"] = run_arm("D", fresh_holder, False, orig_ident)
        # layout arms: only the memory layout of `modes` changes, never a value
        def relayout(src, fn):
            out = {}
            for dom, b in src.items():
                nb = copy.deepcopy(b)
                nb.modes = fn(b.modes)
                out[dom] = nb
            return out
        res["E_fresh_modes_C_contiguous"] = run_arm(
            "E", relayout(fresh_holder, np.ascontiguousarray), False, orig_ident)
        res["F_stored_modes_F_order"] = run_arm(
            "F", relayout(stored, np.asfortranarray), False, orig_ident)
        layout = {dom: dict(fresh_strides=list(fresh_holder[dom].modes.strides),
                            fresh_C=bool(fresh_holder[dom].modes.flags.c_contiguous),
                            stored_strides=list(stored[dom].modes.strides),
                            stored_C=bool(stored[dom].modes.flags.c_contiguous))
                  for dom in fresh_holder if dom in stored}
        # stored vs fresh arrays
        cmp = {}
        for dom in ("air", "water"):
            s, f = stored.get(dom), fresh_holder.get(dom)
            if s is None or f is None:
                cmp[dom] = None
                continue
            d = {}
            for fld in ("modes", "effects", "authority"):
                sa, fa = np.asarray(getattr(s, fld)), np.asarray(getattr(f, fld))
                d[fld] = (dict(shape_s=list(sa.shape), shape_f=list(fa.shape))
                          if sa.shape != fa.shape else
                          dict(maxabs=float(np.max(np.abs(sa - fa))),
                               maxrel=float(np.max(np.abs(sa - fa) / (np.abs(fa) + 1e-300))),
                               bit_equal=bool(np.array_equal(sa, fa))))
            d["fresh_dropped_fields"] = dict(n_probes=f.n_probes, residual_fraction=f.residual_fraction,
                                             authority_threshold=f.authority_threshold,
                                             base_is_none=f.base is None,
                                             stored_n_probes=s.n_probes, stored_thr=s.authority_threshold)
            cmp[dom] = d
        # first divergence A vs B on land
        ea, eb = traces["A"]["entry"], traces["B"]["entry"]
        entry = {k: (float(np.max(np.abs(ea[k] - eb[k]))) if isinstance(ea[k], np.ndarray)
                     else (ea[k] == eb[k])) for k in ea}
        div = {}
        for key in ("cmds", "qpos"):
            A_, B_ = traces["A"][key], traces["B"][key]
            n = min(len(A_), len(B_))
            dd = np.max(np.abs(A_[:n] - B_[:n]), axis=1) if n else np.array([])
            nz = np.nonzero(dd > 0)[0]
            div[key] = dict(n=n, first_nonzero=int(nz[0]) if len(nz) else None,
                            first_delta=float(dd[nz[0]]) if len(nz) else 0.0,
                            delta_at_first_10=[float(v) for v in dd[:10]],
                            max=float(dd.max()) if n else None)
            if key == "cmds" and len(nz):
                j = nz[0]
                div[key]["coeff_A"] = A_[j][:8].tolist()
                div[key]["coeff_B"] = B_[j][:8].tolist()
        row = dict(index=idx, recorded_land=rec_land, recorded_air=(elites[idx].meta or {}).get("air"), recorded_water=(elites[idx].meta or {}).get("water"),
                   arms=res, modes_layout=layout, basis_cmp=cmp, land_entry_A_vs_B=entry, divergence_A_vs_B=div,
                   wall=time.time() - t0)
        out.write(json.dumps(row, default=str) + "\n"); out.flush()
        print(json.dumps(row, default=str, indent=1)[:6000], flush=True)
    out.close()


if __name__ == "__main__":
    main()
