"""Pass M2b: the secondary reads of PREREG_M2b section 4 that m2b_read.py does not print.

C5: per ST/SF/S0 half, the largest 5 s change of nose heading (deg/s), and whether
the nose gets within 45 deg of the commanded heading or its reverse by t = 15 s.
C1: W0 / VB-W0 CoM depth at 40 s (at the surface: depth <= 0.5 m).
C3: WT half A (told deeper) depth at 40 s against its target.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m2b_read import load  # noqa: E402


def heading(r):
    t = np.array([x["t"] for x in r["trace"]])
    n = np.array([x["nose"] for x in r["trace"]])
    return t, np.unwrap(np.arctan2(n[:, 1], n[:, 0]))


NOSE_Z_MAX = 0.7      # added after the data: the nose heading is read only where the
                      # nose is within ~44 deg of level (ST stood on its nose or tail)


def turn_reads(r, h_cmd):
    """Returns (max 5 s rate over fully valid windows, deg/s; best distance to the
    heading or its reverse by t = 15 s over valid samples; start distance; share of
    samples with the nose steeper than NOSE_Z_MAX; min dorsal z)."""
    t, h = heading(r)
    nz = np.array([x["nose"][2] for x in r["trace"]])
    dz = np.array([x["dors"][2] for x in r["trace"]])
    ok = np.abs(nz) <= NOSE_Z_MAX
    k = int(round(5.0 / np.median(np.diff(t))))
    rates = [abs(h[i + k] - h[i]) for i in range(len(h) - k) if ok[i:i + k + 1].all()]
    rate = max(rates) / 5.0 if rates else float("nan")
    sel = (t <= 15.0 + 1e-6) & ok
    d = np.abs((h - h_cmd + math.pi) % (2 * math.pi) - math.pi)   # to the heading
    d = np.minimum(d, math.pi - d)                                  # or its reverse
    best = float(d[sel].min()) if sel.any() else float("nan")
    return (math.degrees(rate), math.degrees(best), math.degrees(float(d[0])),
            float(1.0 - ok.mean()), float(dz.min()))


def main():
    R = load()
    seeds = sorted({k[1] for k in R})
    for arm in ("ST", "SF", "S0"):
        for m in ("wetting", "default"):
            within = 0
            for s in seeds:
                got = False
                for half in ("A", "B"):
                    r = R.get((arm, s, half, m))
                    if r is None:
                        continue
                    rate, dmin, d0, steep, dmn = turn_reads(r, r["cmd"]["h"])
                    got |= dmin <= 45.0
                    print("C5 %s %s seed %d %s: max 5 s turn rate %.1f deg/s; to heading-or-reverse: start %.0f, best by 15 s %.0f deg"
                          " | nose steeper than 44 deg %.0f%% of samples, min dorsal z %+.2f" % (
                              arm, m, s, half, rate, d0, dmin, 100 * steep, dmn))
                within += got
            print("C5 %s %s: seeds with a half within 45 deg by 15 s: %d of %d\n" % (arm, m, within, len(seeds)))
    for arm in ("W0", "VB-W0", "WT", "WT-VB"):
        for m in ("wetting", "default"):
            for s in seeds:
                for half in ("A", "B"):
                    r = R.get((arm, s, half, m))
                    if r is None:
                        continue
                    z = -float(r["trace"][-1]["com"][2])
                    print("depth %s %s seed %d %s: CoM depth at %.1f s %.2f m (target %.2f)" % (
                        arm, m, s, half, r["trace"][-1]["t"], z, r["cmd"].get("z", float("nan"))))


if __name__ == "__main__":
    main()
