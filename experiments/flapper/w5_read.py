"""Read pass W5's run against its data (PREREG_W5_separated.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w5_read.py results/W5_separated.jsonl
"""
import json
import math
import sys
from pathlib import Path

D = Path(__file__).resolve().parent / "sections" / "separated"
rows = [json.loads(x) for x in open(sys.argv[1])]


def csv(name):
    out = []
    for line in open(D / name):
        if line[0] in "#a":
            continue
        out.append([float(x) if x else None for x in line.strip().split(",")])
    return out


def rev(sec, re, rc):
    return {r["alpha"]: r for r in rows
            if r["rig"] == "revolve" and r["section"] == sec and r["re"] == re and r["r_over_c"] == rc}


def tr(sec, ar):
    return {r["alpha"]: r for r in rows if r["rig"] == "translate" and r["section"] == sec and r["ar"] == ar}


print("revolving, Re 54k, R/c 3.3 (Usherwood 2009: card Cv,max 1.3 near Ch 1.1-1.2; real wing Cv 1.49, Ch 0.86-1.02 at 40)")
f, g = rev(None, 54000.0, 3.3), rev("goe417a", 54000.0, 3.3)
print("  alpha   flat Cv   Ch     goe Cv   Ch")
for a in (0, 5, 10, 15, 20, 30, 35, 40, 45, 50, 60, 90):
    print("  %5d   %6.3f %6.3f   %6.3f %6.3f" % (a, f[a]["cv"], f[a]["ch"], g[a]["cv"], g[a]["ch"]))
pk = max(f.values(), key=lambda r: r["cv"])
print("  flat peak Cv %.3f at %d (Ch %.3f); largest |goe - flat| from 20 deg: Cv %.3f, Ch %.3f" % (
    pk["cv"], pk["alpha"], pk["ch"],
    max(abs(g[a]["cv"] - f[a]["cv"]) for a in f if a >= 20),
    max(abs(g[a]["ch"] - f[a]["ch"]) for a in f if a >= 20)))

print("\nrevolving, Re 14k (Lentink flat Ro 2.9: CL,max 1.85-1.9, CD ~1.9 near 45; Kruyt 2014 hummingbird fit, AR ~4)")
f29, f4, g4 = rev(None, 14000.0, 2.9), rev(None, 14000.0, 4.0), rev("goe417a", 14000.0, 4.0)
fit = {int(a): (cl, cd) for a, cl, cd in csv("kruyt2014_canna_revolving_fit.csv")}
print("  alpha   flat2.9 Cv Ch    flat4 Cv Ch     goe4 Cv Ch      fit CL CD")
for a in (5, 10, 20, 30, 35, 40, 45, 50, 60, 90):
    print("  %5d   %6.3f %6.3f   %6.3f %6.3f   %6.3f %6.3f   %6.3f %6.3f" % (
        a, f29[a]["cv"], f29[a]["ch"], f4[a]["cv"], f4[a]["ch"], g4[a]["cv"], g4[a]["ch"], *fit[a]))

print("\nstatic flat wings against Ananda 2012 (by eye, +-0.03)")
err = []
for ar, a, cl, cd in csv("ananda2012_flatplate_rect_wings.csv"):
    m = tr(None, ar)[int(a)]
    err.append(m["cl"] - cl)
    print("  AR %d  %4d  model CL %.3f  data %.2f  (%+.2f)%s" % (
        ar, a, m["cl"], cl, m["cl"] - cl, "   CD %.3f data %.2f" % (m["cd"], cd) if cd is not None else ""))
print("  rms CL error %.3f, range %+.2f..%+.2f" % (math.sqrt(sum(e * e for e in err) / len(err)), min(err), max(err)))

print("\nstatic AR 6 against NACA 4409 (Ostowari & Naik, by eye)")
m6, s6 = tr(None, 6.0), tr("goe417a", 6.0)
for a, cl, cd in csv("ostowari_naik_naca4409_AR6.csv"):
    a = int(a)
    if a in m6:
        s = "   goe417a model CL %.3f CD %.3f" % (s6[a]["cl"], s6[a]["cd"]) if a in s6 else ""
        print("  %4d  flat model CL %.3f CD %.3f   data %.2f %.2f%s" % (a, m6[a]["cl"], m6[a]["cd"], cl, cd, s))

print("\n2D plate against NACA 0012 (Sheldahl & Klimas, table)")
m2 = tr(None, "2D")
for a, cl, cd in csv("sheldahl_klimas_naca0012_poststall.csv"):
    a = int(a)
    if a in m2:
        print("  %4d  model CL %.3f CD %.3f   data %.3f %.3f  (%+.2f, %+.2f)" % (
            a, m2[a]["cl"], m2[a]["cd"], cl, cd, m2[a]["cl"] - cl, m2[a]["cd"] - cd))
