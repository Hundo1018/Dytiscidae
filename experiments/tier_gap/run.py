#!/usr/bin/env python3
"""How often is Tier-1 credit not reproduced by Tier-2, per medium?

Reads ``runs/<run>/events.jsonl`` only (no simulation).  For each promoted
design and each medium, compares the Tier-1 competence the search scored it with
against the Tier-2 leg, at a bar, as a 2x2:

                      Tier-2 leg >= 0.15     Tier-2 leg < 0.15
    Tier-1 >= bar            a                      b
    Tier-1 <  bar            c                      d

    OFR (overconfident-failure rate) = P(Tier-2 fail | Tier-1 pass) = b / (a+b)
    US  (underestimate share)        = P(Tier-2 pass | Tier-1 fail) = c / (c+d)

each with one-sided 95% Clopper-Pearson bounds (``domain.evidence``).  Used as a
diagnostic (arXiv 2610.02740, ``docs/papers/2610.02740.md``); nothing here gates
anything.

Where the data come from
------------------------

* **Tier-2** -- two sources, never mixed in one table:
  - ``values``: ``tier2_media`` on the promote event (arch47 onward).  Every
    promotion carries all three media (``tier2_label_all_media``: legs the chain
    never reached are probed from a fresh reset), so no leg "never ran".
  - ``chain``: the legs the *mission chain itself* ran, recovered from the
    promote event's ``notes`` ("leg i (medium) failed") -- the only Tier-2 record
    arch45 and arch46 have.  Tier-2 stops at the first failed leg, so the chain
    ran the legs before it (passes: the cyclic predecessors of the failing
    medium, air->water->land) and the failing leg, and nothing after.  Pass/fail
    only, no values.
* **Tier-1** -- ``air``/``water``/``land`` of the ``evaluate`` event that placed
  the elite.  The promote event has no design id and, before 2026-10-06, no
  Tier-1 per-medium competence, so the join is ``(island, cell)``: the latest
  non-rejected evaluate event in that cell.  ``descriptor_refit`` re-bins every
  cell, so the map is cleared at each refit (a cell number before and after one
  names different designs).  The join is *verified* when the evaluate event's
  ``mission_fraction`` (4 dp) equals the promote event's ``tier1_fraction``;
  unverified joins are dropped.  Promotions that carry ``tier1_media`` (written
  from 2026-10-06) use it directly, no join.
* **Bars** (Tier-1 pass): 0.15, Tier-2's own leg bar
  (``envs/evaluate.py:LEG_COMPETENCE_BAR``); 0.012, ``WEAKEST_BARS[1]``
  (``evolution/curriculum.py:106``), the triphibian island's "operates in all
  three media" bar, the lowest bar the search itself asks of a medium and the
  one ``three_media`` counts (``evolution/loop.py:1881``); 0.25,
  ``STAGES[0]`` (``evolution/curriculum.py:65``), the "operate in one medium"
  bar of every other island's ladder.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/tier_gap/run.py \\
        [--runs runs/arch45 ...] [--out experiments/tier_gap/results.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from dytiscidae.domain.evidence import clopper_pearson

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ("air", "water", "land")
CYCLE = ("air", "water", "land")           # envs/triphibian.py:DOMAIN_CYCLE
LEG_BAR = 0.15                             # envs/evaluate.py:LEG_COMPETENCE_BAR
BARS = {"leg_0.15": 0.15, "weakest_0.012": 0.012, "single_0.25": 0.25}
DEFAULT_RUNS = ("arch45", "arch46", "arch47", "arch48")


def _rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": k, "n": n, "rate": None, "lo95": None, "hi95": None}
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4),
            "lo95": round(lo, 4), "hi95": round(hi, 4)}


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None                       # constant: undefined, not zero
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 4)


def _spearman(x, y):
    return _pearson(_ranks(x), _ranks(y)) if len(x) >= 3 else None


def chain_legs(notes) -> dict | None:
    """Per-medium chain outcome ``{medium: passed}`` from a promote event's notes.

    ``None`` when no "leg i (medium) failed" note is present (a chain that ran to
    the end or to battery exhaustion, or a truncated note list).
    """
    for n in notes or []:
        if n.startswith("leg ") and " failed" in n:
            try:
                i = int(n.split()[1])
                med = n[n.index("(") + 1:n.index(")")]
            except (ValueError, IndexError):
                return None
            if med not in CYCLE:
                return None
            k = CYCLE.index(med)
            out = {}
            for back in range(1, min(i, 3) + 1):          # legs before the failure
                out.setdefault(CYCLE[(k - back) % 3], True)
            out[med] = False
            return out
    return None


def read_run(path: Path) -> dict:
    """Join promote events with the Tier-1 evaluate events that placed them."""
    placed: dict = {}                      # (island, cell) -> evaluate event
    promos, refits, join = [], 0, defaultdict(int)
    for line in open(path / "events.jsonl"):
        e = json.loads(line)
        k = e.get("kind")
        if k == "descriptor_refit":
            placed.clear()
            refits += 1
        elif k == "evaluate" and e.get("status") != "rejected":
            placed[(e.get("island"), tuple(e["cell"]))] = e
        elif k == "promote":
            p = dict(e)
            join["promotions"] += 1
            if e.get("exploit"):
                join["exploit_excluded"] += 1
                continue
            if e.get("tier1_media"):
                p["_t1"] = {m: float(e["tier1_media"][m]) for m in MEDIA}
                join["direct"] += 1
            else:
                ev = placed.get((e.get("island"), tuple(e["cell"])))
                if ev is None:
                    join["no_candidate"] += 1
                elif round(float(ev.get("mission_fraction", 0.0)), 4) != e["tier1_fraction"]:
                    join["mismatch"] += 1
                else:
                    p["_t1"] = {m: float(ev[m]) for m in MEDIA}
                    join["joined"] += 1
            promos.append(p)
    n = join["promotions"]
    ok = join["joined"] + join["direct"]
    return {"promos": promos, "refits": refits,
            "join": {**join, "joined_total": ok,
                     "join_rate": round(ok / n, 4) if n else None}}


def _ran(p, med, source):
    """The Tier-2 leg of ``med`` as ``(value|None, passed)``, or ``None`` if it never ran."""
    if source == "values":
        v = (p.get("tier2_media") or {}).get(med)
        return None if v is None else (float(v), v >= LEG_BAR)
    leg = chain_legs(p.get("notes"))
    return None if leg is None or med not in leg else (None, leg[med])


def _median(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 4) if xs else None


def tables(promos, source: str, everyone=None) -> dict:
    """2x2 per medium and bar, correlations, and the never-ran share.

    ``promos`` are the joined promotions (they have Tier-1); ``everyone`` is every
    scored promotion, the denominator of the never-ran share.
    """
    everyone = promos if everyone is None else everyone
    out = {}
    for med in MEDIA:
        rows = []                                   # (tier1, tier2 value|None, passed)
        for p in promos:
            r = _ran(p, med, source)
            if r is not None:
                rows.append((p.get("_t1", {}).get(med), r[0], r[1]))
        joined = [r for r in rows if r[0] is not None]
        cells = {}
        for name, bar in BARS.items():
            a = sum(1 for t1, _, t2 in joined if t1 >= bar and t2)
            b = sum(1 for t1, _, t2 in joined if t1 >= bar and not t2)
            c = sum(1 for t1, _, t2 in joined if t1 < bar and t2)
            d = sum(1 for t1, _, t2 in joined if t1 < bar and not t2)
            cells[name] = {"bar": bar, "t1pass_t2pass": a, "t1pass_t2fail": b,
                           "t1fail_t2pass": c, "t1fail_t2fail": d,
                           "OFR": _rate(b, a + b), "US": _rate(c, c + d)}
        xs = [r[0] for r in joined if r[1] is not None]
        ys = [r[1] for r in joined if r[1] is not None]
        out[med] = {
            "legs_ran": len(rows), "legs_ran_and_joined": len(joined),
            "never_ran_share": _rate(
                sum(_ran(p, med, source) is None for p in everyone), len(everyone)),
            "tier1_median": _median([r[0] for r in joined]),
            "tier2_median": _median(ys), "tier2_max": max(ys) if ys else None,
            "spearman": _spearman(xs, ys), "pearson": _pearson(xs, ys),
            "corr_n": len(xs), "bars": cells}
    return out


def analyse(run: str) -> dict:
    path = ROOT / "runs" / run if not Path(run).exists() else Path(run)
    d = read_run(path)
    promos = d["promos"]
    joined = [p for p in promos if "_t1" in p]
    res = {"run": path.name, **{k: d[k] for k in ("refits", "join")},
           "promotions_scored": len(promos), "joined": len(joined)}
    has_values = any(p.get("tier2_media") for p in promos)
    res["chain"] = tables(joined, "chain", promos)
    if has_values:
        res["values"] = tables(joined, "values", promos)
        # Consistency: a chain-failed medium should read below the bar in values.
        bad = tot = 0
        for p in promos:
            leg = chain_legs(p.get("notes"))
            v = p.get("tier2_media") or {}
            for med, passed in (leg or {}).items():
                if not passed and v.get(med) is not None:
                    tot += 1
                    bad += v[med] >= LEG_BAR
        res["chain_fail_but_value_ge_bar"] = {"k": bad, "n": tot}
    res["chain_parsed"] = sum(chain_legs(p.get("notes")) is not None for p in promos)
    # Join bias: does being joinable depend on how Tier-2 went?
    for name, grp in (("joined", joined), ("unjoined", [p for p in promos if "_t1" not in p])):
        res[f"tier2_fraction_pos_{name}"] = _rate(
            sum(p["tier2_fraction"] > 0 for p in grp), len(grp))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
    a = ap.parse_args()
    results = {"bars": BARS, "leg_bar": LEG_BAR, "runs": [analyse(r) for r in a.runs]}
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n")
    for r in results["runs"]:
        j = r["join"]
        print(f"\n== {r['run']}: {j['promotions']} promotions, joined "
              f"{j['joined_total']} ({j['join_rate']}), refits {r['refits']}, {dict(j)}")
        for src in ("values", "chain"):
            if src not in r:
                continue
            for med in MEDIA:
                t = r[src][med]
                for bn, b in t["bars"].items():
                    o, u = b["OFR"], b["US"]
                    print(f"  {src:6s} {med:5s} {bn:14s} ran {t['legs_ran_and_joined']:3d} "
                          f"2x2 {b['t1pass_t2pass']}/{b['t1pass_t2fail']}/"
                          f"{b['t1fail_t2pass']}/{b['t1fail_t2fail']}  "
                          f"OFR {o['k']}/{o['n']} [{o['lo95']},{o['hi95']}]  "
                          f"US {u['k']}/{u['n']} [{u['lo95']},{u['hi95']}]")
                print(f"  {src:6s} {med:5s} never-ran {t['never_ran_share']['k']}/{t['never_ran_share']['n']}  "
                      f"t1med {t['tier1_median']} t2med {t['tier2_median']} t2max {t['tier2_max']}  "
                      f"spearman {t['spearman']} pearson {t['pearson']} (n={t['corr_n']})")


if __name__ == "__main__":
    main()
