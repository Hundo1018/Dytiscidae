"""N14 (ROADMAP 2026-10-10): the Tier-0 spar check reads the genome's frequency.

For every elite of a finished run (merged archive, as every other experiment
loads it), rebuild the phenotype with the project's own `build` and read the
`spar_inertial_reversal` checks it put in the structural report. Each check is
evaluated at `genome.flap_frequency` (phenotype.py:899); applied stress goes as
f^2 (structure.py:179), so the elite's own band top is, exactly as
derivations/flapping_power_bound.py::machine computes it,

    f_spar = f_genome * sqrt(allowable / applied)        (min over wings)

and the metric is the share of flapping elites with `flap_hz` > `f_spar`.
`flap_hz` in the archive meta is `pheno.genome.flap_frequency` (loop.py:1017),
the same number the check used.

A design that is not a `plausible flyer` (wing loading > ~450 N/m2 or AR < 1.5)
never reaches `flapping_inertial_check`: phenotype.py:842-857 `continue`s before
it. Those elites have no check, hence no band (read as unbounded, as the
derivation script's `default=inf`). The counterfactual column rebuilds them with
`is_plausible_flyer` forced True to see what the check would have said.

    PYTHONPATH=. MUJOCO_GL=disable DYTISCIDAE_KERNEL_DIR=<main>/mojo/build \
        .venv/bin/python experiments/spar_band_read/run.py --run runs/arch49
Read-only on the run directory.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

LIFT = ("wing", "membrane", "fin")   # genome.WING / MEMBRANE / FIN (phenotype.py:564)


def spar_band(pheno, f0):
    chk = [c for c in pheno.report.checks if c.name == "spar_inertial_reversal"]
    f = min((f0 * math.sqrt(c.allowable / c.applied) for c in chk if c.applied > 0),
            default=float("inf"))
    margin = min((c.margin for c in chk), default=None)
    return f, len(chk), margin


def powered_wing(pheno):
    """Actuated joint on a WING/MEMBRANE/FIN segment; and the same with a nonzero stroke gene."""
    any_p, any_stroke = False, False
    for s in pheno.segments:
        if s.kind in LIFT and s.part.joint != "none" and s.part.actuated:
            any_p = True
            if float(s.part.stroke_amplitude) > 0.0:
                any_stroke = True
    return any_p, any_stroke


def share(rows, key, pred=lambda r: True):
    sub = [r for r in rows if pred(r)]
    k = sum(1 for r in sub if r[key])
    n = len(sub)
    if n == 0:
        return dict(k=0, n=0, share=None, ci=[None, None])
    # Wilson 95% interval
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return dict(k=k, n=n, share=p, ci=[max(0.0, c - h), min(1.0, c + h)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch49")
    ap.add_argument("--out", default="experiments/spar_band_read/results_arch49.json")
    a = ap.parse_args()
    t0 = time.time()
    from dytiscidae.core import phenotype as ph
    from dytiscidae.core.phenotype import build
    from dytiscidae.ops.run import load_run_archive

    arch, names = load_run_archive(a.run)
    elites = list(arch.cells.values())
    print(f"{len(elites)} elites from islands {names}")
    real_prop = ph.Phenotype.is_plausible_flyer if hasattr(ph, "Phenotype") else None

    rows, dropped = [], []
    for i, e in enumerate(elites):
        g = e.genome
        f0 = float(g.flap_frequency)
        try:
            p = build(g)
        except Exception as ex:   # noqa: BLE001
            dropped.append((i, repr(ex)[:120]))
            continue
        fs, nchk, marg = spar_band(p, f0)
        pw, ps = powered_wing(p)
        n_rot = sum(1 for s in p.segments if getattr(s, "rotor", None) is not None)
        meta = e.meta or {}
        # counterfactual: the check as if the design were a plausible flyer
        fs_cf, nchk_cf = fs, nchk
        if nchk == 0 and pw:
            cls = type(p)
            old = cls.__dict__["is_plausible_flyer"]
            try:
                cls.is_plausible_flyer = property(lambda self: True)
                p2 = build(g)
                fs_cf, nchk_cf, _ = spar_band(p2, f0)
            finally:
                cls.is_plausible_flyer = old
        rows.append(dict(
            i=i, cell=list(e.cell) if hasattr(e, "cell") else None,
            island=meta.get("island"), body_plan=meta.get("body_plan"),
            flap_hz=f0, meta_flap_hz=meta.get("flap_hz"), n_rotors=n_rot,
            meta_n_rotors=meta.get("n_rotors"), powered_wing=pw, powered_stroke=ps,
            plausible_flyer=bool(p.is_plausible_flyer), n_spar_checks=nchk,
            spar_margin=marg, f_spar=None if math.isinf(fs) else fs,
            above_band=bool(f0 > fs), over_ratio=None if math.isinf(fs) else f0 / fs,
            f_spar_cf=None if math.isinf(fs_cf) else fs_cf, n_spar_checks_cf=nchk_cf,
            above_band_cf=bool(f0 > fs_cf),
            mass=meta.get("mass"), span=meta.get("span"), wing_area=meta.get("wing_area")))
    n = len(rows)
    for r in rows:
        r["flapper_union"] = bool(r["n_rotors"] == 0 or r["powered_wing"])
    defs = {
        "all elites": lambda r: True,
        "n_rotors == 0": lambda r: r["n_rotors"] == 0,
        "powered wing joint": lambda r: r["powered_wing"],
        "powered wing joint, stroke_amplitude > 0": lambda r: r["powered_stroke"],
        "n_rotors == 0 OR powered wing joint (literal, headline)": lambda r: r["flapper_union"],
        "n_rotors == 0 AND powered wing joint": lambda r: r["n_rotors"] == 0 and r["powered_wing"],
        "has a spar_inertial_reversal check": lambda r: r["n_spar_checks"] > 0,
    }
    out = {"run": a.run, "islands": names, "n_elites": n, "dropped": dropped,
           "mismatch_flap_hz": sum(1 for r in rows if abs(r["flap_hz"] - (r["meta_flap_hz"] or 0)) > 0.006),
           "mismatch_n_rotors": sum(1 for r in rows if r["n_rotors"] != r["meta_n_rotors"]),
           "shares": {}, "shares_counterfactual": {}, "wall_s": None}
    for name, pred in defs.items():
        out["shares"][name] = share(rows, "above_band", pred)
        out["shares_counterfactual"][name] = share(rows, "above_band_cf", pred)
    over = np.array([r["over_ratio"] for r in rows if r["above_band"]])
    out["over_ratio_of_above"] = (dict(n=int(over.size), min=float(over.min()), median=float(np.median(over)),
                                       max=float(over.max())) if over.size else None)
    out["gate_note"] = "Tier-0 rejects at gate_margin < -0.85 (loop.py:137), i.e. applied up to 6.67x allowable passes (f up to 2.58x the band)."
    out["min_spar_margin"] = min((r["spar_margin"] for r in rows if r["spar_margin"] is not None), default=None)
    out["rows"] = rows
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name in defs:
        s, c = out["shares"][name], out["shares_counterfactual"][name]
        f = lambda d: f"{d['k']}/{d['n']} = {100 * d['share']:.1f}% CI [{100 * d['ci'][0]:.1f}, {100 * d['ci'][1]:.1f}]" if d["n"] else "n/a"
        print(f"{name:58s} as built: {f(s):42s} | counterfactual (check forced): {f(c)}")
    print("dropped", len(dropped), "| flap_hz mismatches", out["mismatch_flap_hz"], "| n_rotors mismatches", out["mismatch_n_rotors"])
    print("over_ratio_of_above", out["over_ratio_of_above"], "| min spar margin", out["min_spar_margin"])
    print("wall", out["wall_s"], "s")


if __name__ == "__main__":
    main()
