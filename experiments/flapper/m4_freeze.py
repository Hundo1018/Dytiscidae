"""Pass M4: the frozen rule of PREREG_M4 section 5 for the entry law: among the
in-box parameter sets tuned on U1/U2, the one with the largest smallest margin
over the tuning halves it was run on.  A half's margin is its smallest clause
slack, each slack scaled by its bound's own width:

    E1 (3 - |dx|) / 3;  E2 (vn + 1) / 1, (15 - |roll|) / 15, min(pitch + 5, 30 - pitch) / 17.5;
    E3 water (0.3 - z) / 0.3, (0.3 - v1) / 0.3, (0.5 - om) / 0.5, (zbz - 0.5) / 0.5;
       land (0.05 - v) / 0.05, (0.2 - om) / 0.2, (zbz - 0.5) / 0.5, (share - 0.8) / 0.2;
    no contact: -10.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m4_freeze.py water|land [write]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m4_read as R  # noqa: E402

BOX = dict(gamma=(2, 15), d_f=(0, 12), h_f=(0.2, 2.0), theta_f=(-5, 25), a_f=(0, 2.8), r_f=(0.8, 1.3),
           b_c=(-20, 60), e_c=(-30, 30))


def in_box(E):
    return all(BOX[k][0] - 1e-9 <= E[k] <= BOX[k][1] + 1e-9 for k in BOX)


def margin(q, medium):
    if "x" not in q:
        return -10.0
    s = [(3.0 - abs(q["dx"])) / 3.0, (q["vn"] + 1.0), (15.0 - abs(q["roll"])) / 15.0,
         min(q["pitch_rel"] + 5.0, 30.0 - q["pitch_rel"]) / 17.5]
    if medium == "water":
        s += [(0.3 - q["com_z_max"]) / 0.3, (0.3 - q["v1_max"]) / 0.3, (0.5 - q["om_mean"]) / 0.5, (q["zbz_min"] - 0.5) / 0.5]
    else:
        s += [(0.05 - q["v_max"]) / 0.05, (0.2 - q["om_max"]) / 0.2, (q["zbz_min"] - 0.5) / 0.5,
              (q["share_hull_wing"] - 0.8) / 0.2]
    return min(s)


def main():
    medium = sys.argv[1]
    rows = R.rows(HERE / "results" / "M4_tuning.jsonl")
    sets = {}
    for rid, r in rows.items():
        if r["kind"] != medium:
            continue
        ok, q = R.entry_half(r, R.trace(rid), r["cmd"], medium)
        key = json.dumps(r["E"], sort_keys=True)
        sets.setdefault(key, []).append((rid, r["start_name"], r["cmd"], ok, margin(q, medium), q.get("dx")))
    ranked = []
    for key, hs in sets.items():
        E = json.loads(key)
        m = min(h[4] for h in hs)
        ranked.append((in_box(E), m, key, hs))
        print("in box" if in_box(E) else "OUT OF BOX", "min margin %+.3f" % m, key,
              [(h[0], h[1], h[2], "pass" if h[3] else "fail", round(h[4], 3), None if h[5] is None else round(h[5], 2)) for h in hs])
    best = max((x for x in ranked if x[0]), key=lambda x: x[1])
    print("FROZEN:", best[2], "min margin %.3f" % best[1])
    if len(sys.argv) > 2 and sys.argv[2] == "write":
        p = HERE / "results" / "M4_frozen.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        d[medium] = dict(E=json.loads(best[2]), min_margin=best[1], halves=[h[0] for h in best[3]],
                         rule="largest smallest margin over the tuning halves run (m4_freeze.py)")
        p.write_text(json.dumps(d, indent=1, sort_keys=True))
        print("sha256", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
