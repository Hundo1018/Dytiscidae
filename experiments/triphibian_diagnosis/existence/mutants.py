#!/usr/bin/env python3
"""E0m: the unchanged hand law on one-step mutants of B1.  Spec: PREREG_nulls_mutants.md.

The sample.  ``np.random.default_rng(20261111)``; each mutant is one call of
`dytiscidae.core.genome.mutate` (genome.py:1234) with the arguments the search
passes at its one call site, `dytiscidae/evolution/loop.py:2128-2129`:

    child, operators = mutate(parent.genome, rng, operators=operators,
                              n_ops=regime.n_mutations)

where ``operators = curator.choose_operators()`` (curator.py:251-258: one
`OperatorBandit.select` per mutation, excluding what is already picked) and
``regime`` is what `Curator.update_regime` sets.  Here the bandit is a fresh
`OperatorBandit()` (as `Curator.__init__` builds it; no credit is ever paid, so
its untried arms are drawn uniformly) and the regime is the one `update_regime`
sets while an archive holds fewer than 20 elites (curator.py:472-476:
``structural_bias 1.6, n_mutations 2``).  One generator serves the plan and
the mutation.  No crossover (loop.py:2132 needs a second parent).

Kept: the first 32 draws whose genome differs from B1's, in order.  `triquad`
is imported, never edited; `Hand` and its gains are untouched.

    cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.
    E=experiments/triphibian_diagnosis/existence
    .venv/bin/python $E/mutants.py draw                 # the 32, their facts -> mutants.json
    .venv/bin/python $E/mutants.py run                  # rollouts -> results_mutants.jsonl
    .venv/bin/python $E/mutants.py run --only 7 --out /tmp/m07.jsonl     # one mutant
    .venv/bin/python $E/mutants.py summary
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import triquad as tq  # noqa: E402
from dytiscidae.core.genome import mutate  # noqa: E402
from dytiscidae.core.mjcf import compile_phenotype, scene_xml  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402
from dytiscidae.evolution.curator import OperatorBandit  # noqa: E402

SEED = 20261111
N_MUTANTS = 32
N_MUTATIONS = 2          # Regime.n_mutations while bootstrapping (curator.py:474)
STRUCTURAL_BIAS = 1.6    # Regime.structural_bias while bootstrapping
BUDGET_S = 3600.0        # spec: stop at 60 minutes of rollouts
BAND = (1.02, 1.10)
SEG = ("air", "water", "land")
MISS = ("mission_air", "mission_water", "mission_land")
OUT = HERE / "results_mutants.jsonl"
_BOOK = {"lineage", "body_plan", "generation", "parent_id", "genome_id"}


def triquad_sha() -> str:
    return hashlib.sha256((HERE / "triquad.py").read_bytes()).hexdigest()


def canon(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: canon(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, np.ndarray):
        return [canon(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [canon(x) for x in o]
    if isinstance(o, dict):
        return {str(k): canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    return o


def genome_sha(g) -> str:
    """Every gene; none of the bookkeeping (lineage, ids, generation)."""
    d = {f.name: canon(getattr(g, f.name)) for f in dataclasses.fields(g) if f.name not in _BOOK}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def body_sha(p) -> str:
    """The built machine as the simulator and the hand law read it: the compiled
    model's bodies, geoms, joints and actuators, the fluid panels, the rotors,
    the motors, the pack, and the mass and buoyant volume the law uses.  Gait
    genes (phase, stroke, rest throttle, flap frequency) are not in it unless
    they change one of those."""
    model, data, names, panels = compile_phenotype(p, scene=scene_xml(timestep=0.004))
    h = hashlib.sha256()
    for arr in (model.body_mass, model.body_inertia, model.body_pos, model.body_quat,
                model.body_ipos, model.geom_size, model.geom_pos, model.geom_quat,
                model.geom_type, model.jnt_type, model.jnt_axis, model.jnt_range,
                model.jnt_stiffness, model.dof_damping, model.dof_armature,
                model.actuator_gainprm, model.actuator_biasprm, model.actuator_forcerange,
                model.actuator_ctrlrange, model.mesh_vert):
        h.update(np.ascontiguousarray(arr).tobytes())
    for name in ("pos_local", "chord", "dr", "volume", "volume_buoyant", "kind", "cd_bluff",
                 "half_height"):
        h.update(np.ascontiguousarray(getattr(panels, name)).tobytes())
    rot = [(float(s.rotor.radius), float(s.rotor.pitch), int(s.rotor.handed))
           for s in p.segments if getattr(s, "rotor", None) is not None]
    mot = [(a.motor_class, float(a.mass), float(a.gear_ratio), bool(a.sealed)) for a in p.actuators]
    h.update(json.dumps([list(names), int(model.nq), int(model.nu), int(model.nbody), rot, mot,
                         p.genome.battery_chem, float(p.genome.battery_wh), float(p.mass),
                         float(p.buoyant_volume), float(p.gas_volume)]).encode())
    return h.hexdigest()


def facts_of(g, ref_body: str | None = None) -> dict:
    out: dict = {"genome_sha": genome_sha(g), "n_parts": len(g.parts), "n_edges": len(g.edges),
                 "battery": [float(g.battery_wh), g.battery_chem], "scale": float(g.scale)}
    try:
        p = build(g)
        out.update(buildable=True, mass=float(p.mass), n_segments=len(p.segments),
                   n_rotors=sum(getattr(s, "rotor", None) is not None for s in p.segments),
                   n_actuated=int(p.n_actuated), density_ratio=float(p.density_ratio),
                   buoyant_L=float(p.buoyant_volume) * 1e3, wing_area=float(p.wing_area),
                   report=p.report.summary())
        gate = tq.tier0_gate(p)
        out["tier0_pass"] = bool(gate["passes"])
        out["tier0"] = {"gate_margin": gate["gate_margin"],
                        "mission_fraction": gate["tier0_mission_fraction"],
                        "failing": {k: v[0] for k, v in gate["checks"].items() if v[0] < 0}}
    except Exception as exc:
        out.update(buildable=False, tier0_pass=False,
                   build_error=f"{type(exc).__name__}: {exc}")
        return out
    try:
        out["body_sha"] = body_sha(p)
        out["body_differs"] = None if ref_body is None else bool(out["body_sha"] != ref_body)
    except Exception as exc:
        out["body_sha"], out["body_differs"] = None, True
        out["compile_error"] = f"{type(exc).__name__}: {exc}"
    return out


def draw():
    """``(B1 facts, [mutant records], [genomes], n_draws, skipped)``, deterministic."""
    rng = np.random.default_rng(SEED)
    bandit = OperatorBandit()
    g0 = tq.b1_genome()
    f0 = facts_of(g0)
    assert f0["body_sha"] == body_sha(build(tq.b1_genome())), "body signature is not reproducible"
    recs, genomes, skipped, n = [], [], [], 0
    while len(recs) < N_MUTANTS:
        n += 1
        plan: list = []
        for _ in range(max(1, N_MUTATIONS)):            # Curator.choose_operators
            plan.append(bandit.select(rng, structural_bias=STRUCTURAL_BIAS, exclude=tuple(plan)))
        child, applied = mutate(g0.copy(), rng, operators=plan, n_ops=N_MUTATIONS)   # loop.py:2128
        if genome_sha(child) == f0["genome_sha"]:
            skipped.append({"draw": n, "plan": plan, "applied": applied})
            continue
        rec = {"index": len(recs), "name": f"M{len(recs):02d}", "draw": n, "plan": plan,
               "applied": applied, **facts_of(child, f0["body_sha"])}
        recs.append(rec)
        genomes.append(child)
    return f0, recs, genomes, n, skipped


def cmd_draw(args) -> int:
    f0, recs, genomes, n, skipped = draw()
    doc = {"seed": SEED, "n_mutations": N_MUTATIONS, "structural_bias": STRUCTURAL_BIAS,
           "call_site": "dytiscidae/evolution/loop.py:2128-2129", "draws": n, "skipped_identical": skipped,
           "B1": f0, "mutants": recs, "triquad_sha256": triquad_sha()}
    (HERE / "mutants.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{n} draws, {len(skipped)} left the genome unchanged, {len(recs)} kept")
    print(f"B1: mass {f0['mass']:.3f} rho {f0['density_ratio']:.3f} parts {f0['n_parts']} "
          f"segments {f0['n_segments']} rotors {f0['n_rotors']}")
    for r in recs:
        if not r["buildable"]:
            print(f"{r['name']} {'+'.join(r['applied']) or '-':34s} NOT BUILDABLE {r['build_error']}")
            continue
        print(f"{r['name']} {'+'.join(r['applied']) or '-':34s} tier0 {'pass' if r['tier0_pass'] else 'FAIL'}"
              f" ({r['tier0']['gate_margin']:+.2f} {list(r['tier0']['failing'])}) body "
              f"{'changed' if r['body_differs'] else 'same'} mass {r['mass']:.3f} rho "
              f"{r['density_ratio']:.3f} parts {r['n_parts']} seg {r['n_segments']} rotors "
              f"{r['n_rotors']} act {r['n_actuated']} E {r['battery'][0]:.0f}{r['battery'][1]}")
    return 0


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------


def run_item(env, name: str, arm: str, item: str, seed: int) -> dict:
    """One rollout, wired as `triquad.cmd_run` wires it.  An exception in
    `Hand.__init__` or in the rollout is the outcome "law not applicable"."""
    t0 = time.time()
    hand = None
    try:
        ctrl, hand = tq.controller_for(env, arm, False)
        if item in SEG:
            row = tq.run_seg_item(env, ctrl, Domain(item), seed)
        elif item in tq.CROSSINGS:
            row = tq.run_cross_item(env, ctrl, item)
        else:
            row = tq.run_mission_item(env, ctrl, Domain(item.split("_", 1)[1]))
        row["outcome"] = "ran"
    except Exception as exc:
        row = {"outcome": "law not applicable", "score": None,
               "failure": f"exception: {type(exc).__name__}: {str(exc)[:300]}",
               "traceback": traceback.format_exc()[-1500:]}
    finally:
        if hand is not None:
            env.cpg.command = hand._orig
    return {"mutant": name, "arm": arm, "item": item, "seed": int(seed), **row,
            "wall_s": round(time.time() - t0, 1)}


def cmd_run(args) -> int:
    out = Path(args.out) if args.out else OUT
    f0, recs, genomes, n, skipped = draw()
    stored = HERE / "mutants.json"
    if stored.exists():
        old = json.loads(stored.read_text())
        assert [m["genome_sha"] for m in old["mutants"]] == [m["genome_sha"] for m in recs], \
            "the draw no longer matches mutants.json"
    print(f"triquad.py sha256 {triquad_sha()}", flush=True)
    t_roll, done = 0.0, 0
    for rec, g in zip(recs, genomes):
        if args.only is not None and rec["index"] != args.only:
            continue
        if args.start is not None and rec["index"] < args.start:
            continue
        if t_roll >= BUDGET_S:
            print(f"STOP: {t_roll:.0f} s of rollouts; mutants 0..{rec['index'] - 1} are complete", flush=True)
            break
        name = rec["name"]
        head = {"mutant": name, "arm": "-", "item": "tier0", "seed": -1,
                "tier0_pass": rec["tier0_pass"], "buildable": rec["buildable"]}
        if not rec["tier0_pass"]:
            with open(out, "a") as f:
                f.write(json.dumps({**head, "outcome": "not buildable" if not rec["buildable"]
                                    else "tier0 rejected"}, default=float) + "\n")
            print(f"{name} {'not buildable' if not rec['buildable'] else 'tier0 rejected'}", flush=True)
            done += 1
            continue
        tq.BODIES[name] = (lambda _g=g: _g.copy())
        t_m = time.time()
        rows = []
        groups = [(0, [("hand", it) for it in SEG] + [("hand", k) for k in tq.CROSSINGS]
                   + [("hand", m) for m in MISS] + [("still", k) for k in tq.CROSSINGS]),
                  (1, [("hand", it) for it in SEG])]
        for seed, items in groups:
            try:
                p, env, facts = tq.make_env(name, seed)
            except Exception as exc:
                for arm, item in items:
                    rows.append({"mutant": name, "arm": arm, "item": item, "seed": seed,
                                 "outcome": "law not applicable", "score": None,
                                 "failure": f"exception in make_env: {type(exc).__name__}: {str(exc)[:300]}",
                                 "traceback": traceback.format_exc()[-1500:], "wall_s": 0.0})
                    with open(out, "a") as f:
                        f.write(json.dumps(rows[-1], default=float) + "\n")
                print(f"{name} seed {seed}: make_env raised {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            for arm, item in items:
                row = run_item(env, name, arm, item, seed)
                row["level_margin"] = facts.get("level_margin")
                rows.append(row)
                with open(out, "a") as f:
                    f.write(json.dumps(row, default=float) + "\n")
        dt = time.time() - t_m
        t_roll += dt
        done += 1
        print(f"{name} {'+'.join(rec['applied']):30s} " + digest(rows) + f" [{dt:.0f} s, total {t_roll:.0f} s]",
              flush=True)
    print(f"{done} mutants processed; rollout wall time {t_roll:.0f} s; triquad.py sha256 {triquad_sha()}",
          flush=True)
    return 0


def ok_cross(r) -> bool:
    return r.get("outcome") == "ran" and bool(r.get("crossed")) and int(r.get("bad_qacc", 0)) == 0


def ok_mission(r) -> bool:
    return (r.get("outcome") == "ran" and r.get("transitions") == [2, 2] and not r.get("failure")
            and int(r.get("bad_qacc", 0)) == 0 and r.get("survived", True))


def why(r) -> str:
    if r.get("outcome") != "ran":
        return "N/A"
    if int(r.get("bad_qacc", 0)) > 0:
        return "bad_qacc"
    return r.get("failure") or ""


def digest(rows) -> str:
    key = {(r["arm"], r["item"], r["seed"]): r for r in rows}
    cr = "".join("X" if ok_cross(key[("hand", k, 0)]) else ("!" if why(key[("hand", k, 0)]) in ("N/A", "bad_qacc", "diverged") else "-")
                 for k in tq.CROSSINGS if ("hand", k, 0) in key)
    ms = " ".join((f"{key[('hand', m, 0)]['transitions'][0]}/{key[('hand', m, 0)]['transitions'][1]}"
                   + ("!" if why(key[("hand", m, 0)]) else "")) if key[("hand", m, 0)].get("outcome") == "ran" else "N/A"
                  for m in MISS if ("hand", m, 0) in key)
    md = " ".join(f"{it[0]}:" + ",".join(("%.2f" % key[("hand", it, s)]["score"]) if key.get(("hand", it, s), {}).get("score") is not None
                                          and not why(key[("hand", it, s)]) else "F" for s in (0, 1) if ("hand", it, s) in key)
                  for it in SEG)
    stl = sum(ok_cross(key[("still", k, 0)]) for k in tq.CROSSINGS if ("still", k, 0) in key)
    return f"cross {cr} | missions {ms} | media {md} | still crossings {stl}"


def cmd_summary(args) -> int:
    doc = json.loads((HERE / "mutants.json").read_text())
    rows = defaultdict(dict)
    for line in open(Path(args.out) if args.out else OUT):
        r = json.loads(line)
        rows[r["mutant"]][(r["arm"], r["item"], r["seed"])] = r
    recs = [m for m in doc["mutants"] if m["name"] in rows]
    print(f"{len(recs)} of {len(doc['mutants'])} mutants have rows (in order: "
          f"{recs[0]['name']}..{recs[-1]['name']})" if recs else "no rows")
    print("| mutant | operators applied | Tier-0 | body | mass kg | rotors | parts | density ratio | crossings a2w w2a w2l l2a | missions air water land | media air, water, land (seeds 0,1) | still crossings | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    stats = []
    for m in recs:
        key = rows[m["name"]]
        if not m["buildable"]:
            print(f"| {m['name']} | {'+'.join(m['applied'])} | not buildable | - | - | - | {m['n_parts']} | - | - | - | - | - | {m.get('build_error', '')[:60]} |")
            stats.append({"m": m, "t0": False})
            continue
        base = (f"| {m['name']} | {'+'.join(m['applied'])} | {'pass' if m['tier0_pass'] else 'FAIL ' + ','.join(m['tier0']['failing'])} "
                f"| {'changed' if m['body_differs'] else 'same'} | {m['mass']:.3f} | {m['n_rotors']} | {m['n_parts']} | {m['density_ratio']:.3f} |")
        if not m["tier0_pass"]:
            print(base + " - | - | - | - | |")
            stats.append({"m": m, "t0": False})
            continue
        cross = [key.get(("hand", k, 0)) for k in tq.CROSSINGS]
        miss = [key.get(("hand", x, 0)) for x in MISS]
        still = [key.get(("still", k, 0)) for k in tq.CROSSINGS]
        c_ok = [bool(r) and ok_cross(r) for r in cross]
        m_ok = [bool(r) and ok_mission(r) for r in miss]
        s_ok = [bool(r) and ok_cross(r) for r in still]
        note = sorted({(r.get("failure") or "")[:70] for r in key.values() if r.get("outcome") != "ran"}
                      | {"bad_qacc" for r in key.values() if int(r.get("bad_qacc", 0) or 0) > 0 and r["arm"] == "hand"}
                      | {f"{r['item']}: {r['failure']}" for r in key.values()
                         if r["arm"] == "hand" and r.get("outcome") == "ran" and r.get("failure") in ("diverged", "unstable", "battery exhausted")})
        med = "; ".join(",".join(("%.2f" % key[("hand", it, s)]["score"]) if key.get(("hand", it, s), {}).get("score") is not None
                                 and not (key[("hand", it, s)].get("failure") or int(key[("hand", it, s)].get("bad_qacc", 0)) > 0) else "F"
                                 for s in (0, 1)) for it in SEG)
        print(base + f" {' '.join('X' if x else '-' for x in c_ok)} | "
              + " ".join((f"{r['transitions'][0]}/{r['transitions'][1]}" if r and r.get("outcome") == "ran" else "N/A") for r in miss)
              + f" | {med} | {sum(s_ok)} | {'; '.join(n for n in note if n)} |")
        stats.append({"m": m, "t0": True, "c_ok": c_ok, "m_ok": m_ok, "s_ok": s_ok,
                      "na": any(r.get("outcome") != "ran" for r in key.values() if r["arm"] == "hand")})
    n = len(stats)
    t0 = [s for s in stats if s["t0"]]
    ch = [s for s in t0 if s["m"]["body_differs"]]
    same = [s for s in t0 if not s["m"]["body_differs"]]
    print()
    print(f"M1: Tier-0 pass {len(t0)} of {n} (not buildable {sum(not s['m']['buildable'] for s in stats)})")
    for label, grp in (("body changed", ch), ("body unchanged", same)):
        if not grp:
            print(f"{label}: 0 mutants")
            continue
        allm = sum(all(s["m_ok"]) for s in grp)
        allc = sum(all(s["c_ok"]) for s in grp)
        print(f"{label}: {len(grp)} Tier-0-passing; mission from all three starts {allm} ({allm / len(grp):.3f}); "
              f"all four crossings {allc} ({allc / len(grp):.3f}); law not applicable on {sum(s['na'] for s in grp)}")
        print("   crossings lost by kind: " + ", ".join(
            f"{k} {sum(not s['c_ok'][i] for s in grp)}" for i, k in enumerate(tq.CROSSINGS))
            + "; missions lost by start: " + ", ".join(
            f"{x.split('_')[1]} {sum(not s['m_ok'][i] for s in grp)}" for i, x in enumerate(MISS)))
        inb = [s for s in grp if BAND[0] <= s["m"]["density_ratio"] <= BAND[1]]
        outb = [s for s in grp if not (BAND[0] <= s["m"]["density_ratio"] <= BAND[1])]
        for lab2, g2 in (("density ratio inside 1.02-1.10", inb), ("outside", outb)):
            if g2:
                print(f"   {lab2}: {len(g2)} mutants; mission from all three starts {sum(all(s['m_ok']) for s in g2)}; "
                      f"all four crossings {sum(all(s['c_ok']) for s in g2)}; water_to_air lost {sum(not s['c_ok'][1] for s in g2)}")
    print(f"M3: still-arm crossings credited: {sum(sum(s['s_ok']) for s in t0)} over {len(t0)} mutants x 4 "
          f"(mutants with any: {[s['m']['name'] for s in t0 if any(s['s_ok'])]})")
    if ch:
        share = sum(all(s["m_ok"]) for s in ch) / len(ch)
        band = "under 0.3" if share < 0.3 else ("0.3-0.7" if share < 0.7 else "0.7 or more")
        print(f"DECISION SHARE: {sum(all(s['m_ok']) for s in ch)} of {len(ch)} = {share:.3f} -> band {band}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draw")
    r = sub.add_parser("run")
    r.add_argument("--only", type=int, default=None)
    r.add_argument("--start", type=int, default=None)
    r.add_argument("--out", default="")
    s = sub.add_parser("summary")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    return {"draw": cmd_draw, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
