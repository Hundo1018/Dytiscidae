"""Pass M3: rank the tuning sets by the freeze rule and (with --write) freeze them.

The prereg (section 5) says "the frozen set is the one with the largest smallest
margin over the tuning starts' gate quantities" without a normalisation. Defined
here, after tuning began and before any scored row (disclosed in the result):

  walk pair (same set, same start, headings h and h+180, both run):
      pair mean / 0.020 - 1, each half's along / 0.06 - 1,
      each half with along > 0: 1 - |across| / (0.5 along)
  walk half without its partner: along / 0.06 - 1 (and the across term)
  turn half: 1 - err_mean / 15, 1 - err_max / 20, -1 if the yaw went the wrong way
  stand: 1 - drift / 0.02, 1 - |dyaw| / 2, 1 - body share / 0.02, 1 - body contact / 0.05
  margin of a set = the smallest of these over all its rollouts.
  A set is eligible only when it has been read on every tuning start: walk pairs
  at yaw 7, -25 and 97 and turns of +90 and -90 (a set read on one start cannot
  have a smallest margin "over the tuning starts"); S on both of its starts.

Only rows run under the prereg's backward law (time-reversed cycle, t033 on) are
candidates for G1/G2; rows t001-t032 used a sweep-sign reversal.

    PYTHONPATH=. .venv/bin/python experiments/flapper/m3_freeze.py [--write]
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST_TIME_REVERSED = 33


def key(r):
    return json.dumps(r.get("gait") or r.get("stand") or {}, sort_keys=True)


def margins(rows):
    ms, notes = [], []
    walks = defaultdict(list)
    for r in rows:
        if r["kind"] == "walk":
            walks[(r["yaw0_deg"], r["heading_deg"] % 180)].append(r)
        elif r["kind"] == "turn":
            em, ex = r["err_last2_mean_deg"], r["err_last2_max_deg"]
            ms += [1 - em / 15, 1 - ex / 20]
            if r["dyaw_deg"] * r["turn_deg"] <= 0:
                ms.append(-1.0)
            notes.append("turn%+d" % r["turn_deg"])
        elif r["kind"] == "stand":
            drift = (r["dx"] ** 2 + r["dy"] ** 2) ** 0.5
            ms += [1 - drift / 0.02, 1 - abs(r["dyaw_deg"]) / 2, 1 - r["share"]["body"] / 0.02,
                   1 - r["body_contact_frac"] / 0.05]
            notes.append("stand")
    for (y, h), hs in walks.items():
        for r in hs:
            ms.append(r["along"] / 0.06 - 1)
            if r["along"] > 0:
                ms.append(1 - abs(r["across"]) / (0.5 * r["along"]))
        if len(hs) == 2:
            ms.append((hs[0]["along"] + hs[1]["along"]) / (2 * hs[0]["window_s"]) / 0.020 - 1)
            notes.append("pair@%g" % y)
        else:
            notes.append("half@%g" % y)
    return (min(ms) if ms else float("nan")), notes


def main(write=False):
    rows = [json.loads(x) for x in open(HERE / "results" / "M3_tuning.jsonl")]
    sets = defaultdict(list)
    for r in rows:
        n = int(r["id"][1:])
        fam = r["arm"] if r["arm"] in ("G1", "G2", "S") else None
        if fam is None or (fam != "S" and n < FIRST_TIME_REVERSED):
            continue
        sets[(fam, key(r))].append(r)
    best = {}
    for (fam, k), rs in sorted(sets.items()):
        m, notes = margins(rs)
        print("%-3s margin %8.3f  n %2d  %s  %s" % (fam, m, len(rs), " ".join(sorted(notes)), k))
        need = {"pair@7", "pair@-25", "pair@97", "turn+90", "turn-90"} if fam != "S" else {"stand"}
        if not need <= set(notes):
            continue
        if fam not in best or m > best[fam][0]:
            best[fam] = (m, k, len(rs))
    print("\nbest:", {f: (round(v[0], 3), v[1]) for f, v in best.items()})
    if write:
        F = {f: json.loads(v[1]) for f, v in best.items()}
        F["margin"] = {f: v[0] for f, v in best.items()}
        F["rule"] = __doc__.split("\n\n")[1]
        out = HERE / "results" / "M3_frozen.json"
        out.write_text(json.dumps(F, indent=1, sort_keys=True))
        print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main("--write" in sys.argv)
