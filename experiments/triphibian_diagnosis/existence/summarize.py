#!/usr/bin/env python3
"""E0: turn results_final.jsonl into the PREREG section 3 table (markdown on stdout).

    .venv/bin/python experiments/triphibian_diagnosis/existence/summarize.py [results_final.jsonl]

No simulation.  A rollout with a failure string or a bad_qacc is counted as
failed, never as a score of 0 (PREREG section 4).
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEG = ("air", "water", "land")
CROSS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")
MISS = ("mission_air", "mission_water", "mission_land")
FLOOR = 0.012


def load(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(dict)
    for r in rows:                       # the last row for a (body, arm, item, seed) wins
        by[(r["body"], r["arm"], r["item"])][r["seed"]] = r
    return by


def failed(r):
    return bool(r.get("failure")) and r["item"] in SEG or int(r.get("bad_qacc", 0)) > 0 \
        or (r["item"] in MISS and not r.get("survived", True))


def cell(by, body, arm, item):
    rs = by.get((body, arm, item), {})
    if not rs:
        return "not run"
    seeds = sorted(rs)
    n = len(seeds)
    sc = [rs[s]["score"] for s in seeds]
    bad = sum(int(rs[s].get("bad_qacc", 0)) > 0 for s in seeds)
    if item in SEG:
        ok = [rs[s] for s in seeds if not failed(rs[s])]
        vals = [r["score"] for r in ok]
        p = sum(v >= FLOOR for v in vals)
        txt = f"**{p}/{n}** >= {FLOOR}"
        if vals:
            txt += (f"; {sum(v >= 0.15 for v in vals)}/{n} >= 0.15; {sum(v >= 0.5 for v in vals)}/{n} >= 0.5; "
                    f"min {min(vals):.3f} med {st.median(vals):.3f} max {max(vals):.3f}")
        nf = n - len(ok)
        if nf:
            txt += f"; {nf} failed ({', '.join(sorted({rs[s]['failure'] or 'bad_qacc' for s in seeds if failed(rs[s])}))})"
        return txt
    if item in CROSS:
        p = sum(bool(rs[s]["crossed"]) and not failed(rs[s]) for s in seeds)
        holds = [rs[s]["hold"] for s in seeds]
        txt = f"**{p}/{n}** crossed; hold {min(holds):.2f}-{max(holds):.2f}; approach {min(sc):.2f}-{max(sc):.2f}"
        fl = sorted({rs[s]["failure"] for s in seeds if rs[s]["failure"]})
        if fl:
            txt += f"; ({'; '.join(fl)})"
    else:
        p = sum(rs[s]["transitions"] == [2, 2] and not failed(rs[s]) for s in seeds)
        tr = sorted({tuple(rs[s]["transitions"]) for s in seeds})
        txt = (f"**{p}/{n}** both transitions; transitions {', '.join(f'{a}/{b}' for a, b in tr)}; "
               f"continuous_score {min(sc):.3f}-{max(sc):.3f}")
        fl = sorted({rs[s]["failure"] for s in seeds if rs[s]["failure"]})
        if fl:
            txt += f"; ({'; '.join(fl)})"
    if bad:
        txt += f"; bad_qacc in {bad}"
    return txt


def identical(by, body, arm, item):
    rs = by.get((body, arm, item), {})
    keys = ("score", "booked_wh", "soc_end")
    vals = {tuple(rs[s][k] for k in keys) for s in rs}
    return len(vals) == 1 and len(rs) > 1


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else HERE / "results_final.jsonl"
    by = load(path)
    arms = [("B1", "hand"), ("B0", "hand"), ("B1", "still"), ("B0", "still")]
    print("| item | " + " | ".join(f"{b} {a}" for b, a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    for item in SEG + CROSS + MISS:
        print(f"| {item} | " + " | ".join(cell(by, b, a, item) for b, a in arms) + " |")
    print()
    print("Per-seed scores (hand arm):")
    for body in ("B1", "B0"):
        for item in SEG:
            rs = by.get((body, "hand", item), {})
            print(f"- {body} {item}: " + " ".join(f"{rs[s]['score']:.3f}" for s in sorted(rs)))
    print()
    print("Seed-independence of the crossing and mission probes (identical score, booked energy and end state of charge on every seed):")
    for body, arm in arms:
        same = [it for it in CROSS + MISS if identical(by, body, arm, it)]
        diff = [it for it in CROSS + MISS if (body, arm, it) in by and not identical(by, body, arm, it)]
        print(f"- {body} {arm}: identical on {len(same)} of {len(same) + len(diff)} items" + (f"; differ: {diff}" if diff else ""))
    print()
    print("Battery and booked energy (per rollout; a segment starts at the scatter's state of charge, 0.4-1.0):")
    print("| body arm | item | end state of charge min-max | booked Wh min-max | mean power W (mean) | peak draw W max | max actuator overload | failures | bad_qacc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for body, arm in arms:
        for item in SEG + CROSS + MISS:
            rs = by.get((body, arm, item), {})
            if not rs:
                continue
            soc = [r["soc_end"] for r in rs.values()]
            wh = [r["booked_wh"] for r in rs.values()]
            pw = [r["mean_power_w"] for r in rs.values()]
            fl = sorted({r["failure"] for r in rs.values() if r["failure"]})
            print(f"| {body} {arm} | {item} | {min(soc):.3f}-{max(soc):.3f} | {min(wh):.3f}-{max(wh):.3f} | "
                  f"{st.mean(pw):.0f} | {max(r['peak_draw_w'] for r in rs.values()):.0f} | "
                  f"{max(r['max_overload'] for r in rs.values()):.2f} | {'; '.join(fl) or '-'} | "
                  f"{sum(int(r['bad_qacc']) > 0 for r in rs.values())} |")
    print()
    # Tier-1's own extrapolation: segment mean power x 300 s x 3 cycles, plus transitions
    sys.path.insert(0, str(HERE.parents[2]))
    try:
        from dytiscidae.core.phenotype import build
        from dytiscidae.envs.triphibian import MissionSpec
        from dytiscidae.physics.energy import transition_energy
        sys.path.insert(0, str(HERE))
        import triquad as tq
        spec = MissionSpec()
        print("Energy against the pack, as `evaluate_tier1` extrapolates it (segment mean power x "
              f"{spec.seconds_per_domain:.0f} s x {spec.cycles} cycles + `transition_energy`; usable = 0.85 x pack):")
        print("| body | pack Wh | P air W | P water W | P land W | required Wh | usable Wh | usable/required | minutes of the 45 covered |")
        print("|---|---|---|---|---|---|---|---|---|")
        for body in ("B1", "B0"):
            p = build(tq.BODIES[body]())
            pw = {}
            for item in SEG:
                rs = by.get((body, "hand", item), {})
                pw[item] = st.mean(r["mean_power_w"] for r in rs.values()) if rs else float("nan")
            cruise = sum(pw.values()) * spec.seconds_per_domain * spec.cycles
            trans = sum(transition_energy(p.mass, k) for k in spec.transitions)
            req = (cruise + trans) / 3600.0
            use = 0.85 * p.genome.battery_wh
            print(f"| {body} | {p.genome.battery_wh:.0f} {p.genome.battery_chem} | {pw['air']:.0f} | {pw['water']:.0f} | "
                  f"{pw['land']:.0f} | {req:.1f} | {use:.1f} | {use / req:.2f} | {min(use / req, 1.0) * 45:.0f} |")
    except Exception as exc:  # the table above does not depend on this
        print(f"(energy extrapolation skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
