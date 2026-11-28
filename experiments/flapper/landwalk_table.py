"""Tables of F2 from results/landwalk.jsonl (reads only).

    .venv/bin/python experiments/flapper/landwalk_table.py [path]
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "landwalk.jsonl"
rs = [json.loads(ln) for ln in path.read_text().splitlines()]
f = lambda x, n=3: "None" if x is None else f"{x:.{n}f}"          # noqa: E731


def walks(r, key):
    w = r[key]
    return bool(r["score"] >= 0.15 and not r["failure"] and r["bad_qacc"] == 0
                and w["ground_support"] is not None and w["ground_support"] >= 0.9
                and w["wing_share"] is not None and w["wing_share"] >= 0.7
                and w["hull_contact"] is not None and w["hull_contact"] <= 0.2)


print("land rows.  window: late half of the walk phase, the worse half of the pair (whole walk phase in brackets)")
for L in "AB":
    for arm in ("walk", "propped", "frozen"):
        rr = sorted((r for r in rs if r["kind"] == "land" and r["layout"] == L and r["arm"] == arm), key=lambda r: r["seed"])
        for r in rr:
            w, ww = r["walk_late"], r["walk"]
            print(f"{L} {arm:7s} seed {r['seed']} score {f(r['score'])} failure {r['failure']!r} bad_qacc {r['bad_qacc']}"
                  f" | support {f(w['ground_support'])} [{f(ww['ground_support'])}] wing share {f(w['wing_share'])} [{f(ww['wing_share'])}]"
                  f" tail {f(w['tail_share'])} hull contact {f(w['hull_contact'])} [{f(ww['hull_contact'])}] airborne {f(w['airborne'], 2)}"
                  f" | torque N m late walk: stroke {f(w['torque_stroke'], 2)} feather {f(w['torque_feather'], 2)} tail {f(w['torque_tail'], 2)};"
                  f" whole rollout: {f(r['torque']['shoulder_stroke'], 2)} {f(r['torque']['feather'], 2)}"
                  f" | {'WALKS' if walks(r, 'walk_late') else 'no'}")
        if arm == "walk":
            print(f"  => {L}: walks on its wings on {sum(walks(r, 'walk_late') for r in rr)} of {len(rr)} seeds"
                  f" (whole walk phase as the window: {sum(walks(r, 'walk') for r in rr)});"
                  f" score {min(r['score'] for r in rr):.3f}-{max(r['score'] for r in rr):.3f},"
                  f" wing share {min(r['walk_late']['wing_share'] for r in rr):.3f}-{max(r['walk_late']['wing_share'] for r in rr):.3f},"
                  f" hull contact {min(r['walk_late']['hull_contact'] for r in rr):.3f}-{max(r['walk_late']['hull_contact'] for r in rr):.3f}")
        else:
            print(f"  => {L} {arm}: score under 0.012 on {sum(r['score'] < 0.012 for r in rr)} of {len(rr)}, largest {max(r['score'] for r in rr):.4f}")
print("top speed (each layout's FAST gait, straight along the shore, 8 s)")
for r in rs:
    if r["kind"] == "speed":
        w = r["window"]
        print(f"{r['layout']} seed {r['seed']} {f(r['speed_mps'], 4)} m/s = {f(r['hull_lengths_per_s'], 2)} hull lengths/s, across {f(r['across_m'])} m, bad_qacc {r['bad_qacc']},"
              f" wing share {f(w['wing_share'])} hull contact {f(w['hull_contact'])} airborne {f(w['airborne'], 2)},"
              f" torque stroke {f(w['torque_stroke'], 2)} feather {f(w['torque_feather'], 2)} tail {f(w['torque_tail'], 2)}")
print("air cost (rig.run)")
for r in rs:
    if r["kind"] == "air":
        c = r["cruise"]
        print(f"{r['name']:22s} {r['mass_kg'] * 1000:.2f} g | hover 12 Hz: lift {f(r['hover_lift_N'])} N, aero power {f(r['hover']['aero_power_W'], 2)} W"
              f" | cruise 8 m/s 8 Hz: feather offset {f(c['psi0_deg'], 2)} deg, lift/weight {f(c['lift_over_weight'])}, net force along {f(c['Fx'])} N, aero power {f(c['aero_power_W'], 2)} W, trimmed {c['trimmed']}")
