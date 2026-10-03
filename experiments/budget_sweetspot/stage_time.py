"""Where a generation's wall time goes, per 100-generation band, from the
`stages` events every search writes. residual = generation wall - stages."""
import json, sys
from statistics import mean

for run in sys.argv[1:]:
    st, el = {}, {}
    for l in open(f"{run}/events.jsonl"):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("kind") == "stages": st[r["gen"]] = r
    for l in open(f"{run}/generations.jsonl"):
        try: r = json.loads(l)
        except ValueError: continue
        if "generation" in r: el[r["generation"]] = r["elapsed"]
    print(f"== {run}  shards/gen={len(next(iter(st.values()))['shards'][0])}")
    print(" band   wall  main rescore refine  other | shard max/mean  refined/gen")
    for lo in range(0, max(el) + 1, 100):
        gs = [g for g in range(lo + 1, lo + 100) if g in st and g in el and g - 1 in el]
        if not gs: continue
        wall = mean(el[g] - el[g - 1] for g in gs)
        main = mean(st[g]["main"] for g in gs)
        res = mean(st[g]["rescore"] for g in gs)
        ref = mean(sum(st[g]["refine_steps"]) for g in gs)
        imb = mean(max(s) / mean(s) for g in gs for s in st[g]["shards"] if s)
        nref = mean(st[g]["refined"] for g in gs)
        print(f" {lo:4d} {wall:6.1f} {main:5.1f} {res:6.1f} {ref:6.1f} {wall-main-res-ref:6.1f} | {imb:9.2f} {nref:10.2f}")
