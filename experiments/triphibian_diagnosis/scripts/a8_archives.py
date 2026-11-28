"""A8 pass C: the final per-island archives and the 50-generation snapshots.

Reads runs/<run>/archive_<island>.json (the final export of each island, written by
evolution/loop.py:2512-2515 at the end of run_search, or the last checkpoint export
for a run that was killed) -- never runs/<run>/archive.json, which is the visited
island's dashboard export every 5 generations (ops/run.py:321-323).
Snapshots: runs/<run>/snapshots/gen####_<island>.json (loop.py:2464-2467).

Elite fields relied on: meta.air / meta.water / meta.land (recorded segment
competence, 3 dp; with N11 re-evaluation the median draw's), meta.body_plan,
meta.n_rotors (absent in arch43/44), meta.mass, meta.gen, meta.mission_fraction,
draws (N11 buffer: one entry per draw with air/water/land), improvements.

Output: results/archives.json.  Usage:
  nice -n 10 .venv/bin/python experiments/triphibian_diagnosis/scripts/a8_archives.py [run ...]
"""
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "archives.json"
DEFAULT = [f"arch{n}" for n in range(43, 61)]
FLOOR = 0.012
MEDIA = ("air", "water", "land")
ISLAND_MEDIA = {
    "air": ("air",), "water": ("water",), "land": ("land",),
    "amphibian": ("water", "land"), "aerial_diver": ("air", "water"),
    "land_air": ("land", "air"), "triphibian": ("air", "water", "land"),
    "generalist": ("air", "water", "land"),
}


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def rotor_class(nr):
    if nr is None:
        return "unknown"
    return "rotor" if nr > 0 else "no_rotor"


def summarize_archive(path, island):
    d = json.load(open(path))
    el = d.get("elites") or []
    media = ISLAND_MEDIA.get(island, MEDIA)
    s = {"generation": (d.get("summary") or {}).get("generation"), "n": len(el),
         "capacity": (d.get("summary") or {}).get("capacity")}
    c = collections.Counter()
    by_plan = collections.defaultdict(collections.Counter)
    by_rotor = collections.defaultdict(collections.Counter)
    rotors_hist = collections.Counter()
    draws_len = collections.Counter()
    improvements = []
    pair_x, pair_y = {m: [] for m in MEDIA}, {m: [] for m in MEDIA}
    retention = collections.Counter()
    top = []
    own_max = 0.0
    for e in el:
        m = e.get("meta") or {}
        comp = {k: f(m.get(k)) for k in MEDIA}
        flags = {k: comp[k] >= FLOOR for k in MEDIA}
        nfl = sum(flags.values())
        plan = m.get("body_plan") or "?"
        nr = m.get("n_rotors")
        rc = rotor_class(nr)
        rotors_hist["unknown" if nr is None else min(int(nr), 9)] += 1
        c["n"] += 1
        c[f"m{nfl}"] += 1
        for k in MEDIA:
            c[k] += int(flags[k])
        c["air_water"] += int(flags["air"] and flags["water"])
        c["air_land"] += int(flags["air"] and flags["land"])
        c["water_land"] += int(flags["water"] and flags["land"])
        c["floor_member"] += int(any(flags[k] for k in media))   # N5: clears >= 1 island medium
        c["all_island_media"] += int(all(flags[k] for k in media))
        c["mf_pos"] += int(f(m.get("mission_fraction")) > 0)
        c["all3_pos"] += int(min(comp.values()) > 0)
        own = min(comp[k] for k in media)
        own_max = max(own_max, own)
        for tab, key in ((by_plan, plan), (by_rotor, rc)):
            tab[key]["n"] += 1
            tab[key][f"m{nfl}"] += 1
            for k in MEDIA:
                tab[key][k] += int(flags[k])
        top.append({"min": min(comp.values()), **comp, "island": island, "plan": plan,
                    "mass": m.get("mass"), "n_rotors": nr, "gen": m.get("gen"),
                    "tier": e.get("tier"), "fitness": e.get("fitness"),
                    "mission_fraction": m.get("mission_fraction"),
                    "n_draws": len(e.get("draws") or [])})
        dr = e.get("draws") or []
        draws_len[len(dr)] += 1
        improvements.append(int(e.get("improvements") or 0))
        if len(dr) >= 2:
            # draw-to-draw (test-retest) pairs: first draw against each later one
            for k in MEDIA:
                x0 = f(dr[0].get(k))
                for later in dr[1:]:
                    pair_x[k].append(x0)
                    pair_y[k].append(f(later.get(k)))
            for k in media:
                vals = [f(x.get(k)) for x in dr]
                if comp[k] >= FLOOR:
                    retention[f"{k}_recorded_ge_floor"] += 1
                    retention[f"{k}_draws_ge_floor"] += sum(v >= FLOOR for v in vals)
                    retention[f"{k}_draws"] += len(vals)
    top.sort(key=lambda t: -t["min"])
    s["counts"] = dict(c)
    s["by_plan"] = {k: dict(v) for k, v in by_plan.items()}
    s["by_rotor"] = {k: dict(v) for k, v in by_rotor.items()}
    s["rotors_hist"] = {str(k): v for k, v in rotors_hist.items()}
    s["draws_len"] = {str(k): v for k, v in sorted(draws_len.items())}
    s["improvements_mean"] = round(sum(improvements) / len(improvements), 3) if improvements else None
    s["retest"] = {k: {"pairs": len(pair_x[k]), "r": pearson(pair_x[k], pair_y[k]),
                       "mean_first": (sum(pair_x[k]) / len(pair_x[k])) if pair_x[k] else None,
                       "mean_later": (sum(pair_y[k]) / len(pair_y[k])) if pair_y[k] else None,
                       "first_ge_floor": sum(x >= FLOOR for x in pair_x[k]),
                       "later_ge_floor_given_first": sum(1 for x, y in zip(pair_x[k], pair_y[k]) if x >= FLOOR and y >= FLOOR)}
                   for k in MEDIA}
    s["retention"] = dict(retention)
    s["own_max"] = own_max
    s["top"] = top[:10]
    return s


SNAP = re.compile(r"gen(\d+)_(\w+)\.json$")


def snapshot_series(run, islands):
    sd = RUNS / run / "snapshots"
    out = collections.defaultdict(list)
    if not sd.is_dir():
        return {}
    files = sorted(p for p in sd.iterdir() if SNAP.search(p.name))
    for p in files:
        m = SNAP.search(p.name)
        g, island = int(m.group(1)), m.group(2)
        if island not in ISLAND_MEDIA:
            continue
        d = json.load(open(p))
        el = d.get("elites") or []
        media = ISLAND_MEDIA[island]
        n_mem = n_all = 0
        own_max = 0.0
        med = {k: 0 for k in MEDIA}
        for e in el:
            mm = e.get("meta") or {}
            comp = {k: f(mm.get(k)) for k in MEDIA}
            n_mem += int(any(comp[k] >= FLOOR for k in media))
            n_all += int(all(comp[k] >= FLOOR for k in media))
            own_max = max(own_max, min(comp[k] for k in media))
            for k in MEDIA:
                med[k] += int(comp[k] >= FLOOR)
        out[island].append({"gen": g, "n": len(el), "floor_member": n_mem, "all_island_media": n_all,
                            "own_max": own_max, **{f"ge_{k}": v for k, v in med.items()}})
        del d
    return {k: sorted(v, key=lambda r: r["gen"]) for k, v in out.items()}


def main():
    runs = sys.argv[1:] or DEFAULT
    vocab = json.load(open(RES / "vocab.json"))
    out = json.load(open(OUT)) if OUT.exists() else {}
    for run in runs:
        t0 = time.time()
        cfg = (vocab.get(run, {}).get("generations") or {}).get("config") or {}
        islands = cfg.get("islands") or []
        rec = {"islands": {}, "archive_json": None}
        for isl in islands:
            p = RUNS / run / f"archive_{isl}.json"
            if p.exists():
                rec["islands"][isl] = summarize_archive(p, isl)
        aj = RUNS / run / "archive.json"
        if aj.exists():
            d = json.load(open(aj))
            rec["archive_json"] = {"generation": (d.get("summary") or {}).get("generation"),
                                   "n": len(d.get("elites") or [])}
            del d
            # Single-island runs killed after their last checkpoint (arch52, arch60): the
            # dashboard export is the newer copy of the same island; summarised as a check.
            if len(islands) == 1 and islands[0] in rec["islands"] and \
                    (rec["archive_json"]["generation"] or 0) > (rec["islands"][islands[0]]["generation"] or 0):
                rec["archive_json_summary"] = summarize_archive(aj, islands[0])
        rec["snapshots"] = snapshot_series(run, islands)
        out[run] = rec
        OUT.write_text(json.dumps(out, separators=(",", ":")))
        print(run, f"{time.time() - t0:.1f}s",
              {i: (s["generation"], s["n"], s["counts"].get("m2", 0), s["counts"].get("m3", 0)) for i, s in rec["islands"].items()},
              "snapshots", {k: len(v) for k, v in rec["snapshots"].items()}, flush=True)


if __name__ == "__main__":
    main()
