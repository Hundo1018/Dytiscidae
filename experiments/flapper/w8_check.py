"""Pass W8, section 2: every drag run's last point against a straight line
through the three before it (PREREG_W8_section_to_stall.md).

    PYTHONPATH=. .venv/bin/python experiments/flapper/w8_check.py
"""
import glob
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sections  # noqa: E402

NAMES = ("goe417a", "bw3", "cr001", "e387a", "e387c", "e387e", "sd7003", "sd7037b", "s1223")
err = []
for n in NAMES:
    for re_, a, cl, cd in sections.tables(n):
        if len(a) < 5:
            continue
        line = np.polyfit(a[-4:-1], cd[-4:-1], 1)
        got = float(np.polyval(line, a[-1]))
        err.append(got - float(cd[-1]))
        print("%-8s Re %6.0f  last %5.2f deg  measured %.4f  line %.4f  (%+.4f)  step %.2f deg" % (
            n, re_, np.degrees(a[-1]), cd[-1], got, got - cd[-1], np.degrees(a[-1] - a[-2])))
print("runs %d; median error %+.4f; within 0.012: %d; range %+.4f..%+.4f" % (
    len(err), st.median(err), sum(abs(e) <= 0.012 for e in err), min(err), max(err)))
