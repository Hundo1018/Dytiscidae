"""Pass S2's figure: the sections and plan of the structural models, read
from the CalculiX decks that were run (not from the design numbers).

    .venv/bin/python experiments/flapper/s2/draw_s2.py [CASES_DIR]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/hundo/Projects/Dytiscidae/.cfd/cases/S2")
OUT = Path(__file__).resolve().parent.parent / "figures" / "S2_sections.png"
SHOW = (("P500um_torque", "P: 0.5 mm carbon plate alone"),
        ("TP8x6_P300um_torque", "TP: 8x6 tube at 25% chord + 0.3 mm plate"),
        ("F8x6_depron6mm_torque", "F: 8x6 tube + 6 mm depron"))


def nodes(deck):
    out, on = [], False
    for line in open(deck):
        if line.startswith("*"):
            on = line.strip().upper() == "*NODE"
            continue
        if on:
            out.append([float(v) for v in line.split(",")[1:4]])
    return np.array(out)


def main():
    fig, ax = plt.subplots(len(SHOW), 2, figsize=(11, 7.5), gridspec_kw=dict(width_ratios=(1.3, 1)))
    for row, (case, title) in enumerate(SHOW):
        n = nodes(CASES / case / f"{case}.inp")
        root = n[np.abs(n[:, 0]) < 1e-9]
        a = ax[row, 0]
        a.plot(root[:, 1] * 1e3, root[:, 2] * 1e3, ".", ms=3)
        a.set_aspect("equal")
        chord = np.ptp(root[:, 1])
        a.set_title(f"{title}\nroot section: chord {chord * 1e3:.0f} mm, height {np.ptp(root[:, 2]) * 1e3:.1f} mm "
                    f"({np.ptp(root[:, 2]) / chord:.1%} of chord)", fontsize=9)
        a.set_xlabel("chordwise, mm")
        a.set_ylabel("mm")
        b = ax[row, 1]
        b.plot(n[:, 0], n[:, 1], ".", ms=1)
        b.set_aspect("equal")
        b.set_title(f"plan: span {np.ptp(n[:, 0]):.3f} m, {len(n)} nodes", fontsize=9)
        b.set_xlabel("spanwise, m (root clamped at 0)")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(OUT)


if __name__ == "__main__":
    main()
