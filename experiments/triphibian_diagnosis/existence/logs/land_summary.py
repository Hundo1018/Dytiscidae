import json, sys, statistics as st
from collections import defaultdict
rows=[json.loads(l) for l in open("experiments/triphibian_diagnosis/existence/results_land.jsonl")]
fin={(r["body"],r["arm"],r["item"],r["seed"]):r for r in map(json.loads,open("experiments/triphibian_diagnosis/existence/results_final.jsonl"))}
by=defaultdict(dict)
for r in rows:
    if r.get("note"): 
        print("PROBE", r["note"], "| body", r["body"], "| land score", r.get("score"), "| bad_qacc", r.get("bad_qacc"), "| s halves", r.get("s_halves"), "| s stop", r.get("s_stop_halves"), "| Fz mean/mg over rollout", r.get("fz_mean_all_over_mg"), "| s", r.get("s"))
        continue
    by[(r["body"],r["arm"],r["item"])][r["seed"]]=r
def bad(r): return r.get("score") is None or r.get("bad_qacc",0)>0 or (r["item"] in ("air","water","land") and r.get("failure")) or (r["item"].startswith("mission") and r.get("failure"))
print("failed rows:", [(k, s, r.get("failure"), r.get("bad_qacc")) for k,d in by.items() for s,r in d.items() if bad(r)])
for arm in ("walk","still","frozen","shuffled"):
    d=by[("L1",arm,"land")]
    print(f"L1 {arm} land:", " ".join(f"{d[s]['score']:.3g}" for s in sorted(d)), "| s:", " ".join("None" if d[s]['s'] is None else f"{d[s]['s']:.3f}" for s in sorted(d)), "| rotor max rad/s:", f"{max(max(d[s]['rotor_max_walk']) for s in d):.2f}", "| >=0.15:", sum(d[s]['score']>=0.15 for s in d), ">=0.012:", sum(d[s]['score']>=0.012 for s in d), "max", f"{max(d[s]['score'] for s in d):.3g}")
d=by[("B1","hand","land")]
print("B1 hand land:", " ".join(f"{d[s]['score']:.3f}" for s in sorted(d)), "| s:", " ".join(f"{d[s]['s']:.3f}" for s in sorted(d)), "| rotor max:", f"{max(max(d[s]['rotor_max_walk']) for s in d):.0f}", "| E0 stored:", " ".join(f"{fin[('B1','hand','land',s)]['score']:.3f}" for s in range(8)))
w=by[("L1","walk","land")]
print("L1 walk land detail:", [(s, round(w[s]["m"]["walk_progress"],2), round(w[s]["m"]["pair_across"],2), round(w[s]["m"]["stop_score"],2), round(w[s]["m"]["contact_fraction"],2), round(w[s]["m"]["upright"],2), [round(x,3) for x in w[s]["s_halves"]]) for s in sorted(w)])
for it in ("air","water"):
    for b,a in (("L1","walk"),("B1","hand")):
        d=by[(b,a,it)]; print(b,a,it," ".join(f"{d[s]['score']:.3f}" for s in sorted(d)), "| bad", sum(bad(d[s]) for s in d))
for it in ("air_to_water","water_to_air","water_to_land","land_to_air"):
    d=by[("L1","walk",it)]; print("L1", it, "crossed", sum(bool(d[s]["crossed"]) and not d[s]["bad_qacc"] for s in d), "/", len(d), "hold", round(d[0]["hold"],3), "identical", len({d[s]["score"] for s in d})==1, "| B1 E0 hold", round(fin[("B1","hand",it,0)]["hold"],3))
for it in ("mission_air","mission_water","mission_land"):
    d=by[("L1","walk",it)]; r=d[0]; print("L1", it, r["transitions"], "score", round(r["score"],3), "legs", [(l["cmd"][0], l["entered"], l["entry_time"], l["on_task"]) for l in r["legs"]], "landed steps", r.get("landed_steps"), "s_landed", r.get("s_landed"), "rotor_max_landed_late", r.get("rotor_max_landed_late"), "walked_m", r.get("walked_m"), "identical", len({d[s]["score"] for s in d})==1, "| B1 E0", fin[("B1","hand",it,0)]["transitions"], round(fin[("B1","hand",it,0)]["score"],3))
print("energy L1 walk mean W air/water/land:", [round(st.mean(by[("L1","walk",it)][s]["mean_power_w"] for s in range(8))) for it in ("air","water","land")], "B1:", [round(st.mean(by[("B1","hand",it)][s]["mean_power_w"] for s in range(8))) for it in ("air","water","land")])
