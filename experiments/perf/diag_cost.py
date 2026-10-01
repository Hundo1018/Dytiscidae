"""What the batched path's per-step diagnostic reductions cost on the host.

    python experiments/perf/diag_cost.py [--panels 48] [--machines 4] [--reps 20000]

`BatchedFluid.finish` refreshes every field of each machine's `FluidDiagnostics`
from the downloaded panel arrays at every step.  Two of them are read on the
batched path (`mean_submerged` by the observation, `slam` by the transition
score); the other six are not.  This times the same numpy expressions on
arrays of the run's size, split into the read and the unread fields, so the
saving from dropping or deferring the unread ones is a measurement, not a guess.
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=48, help="panels per machine")
    ap.add_argument("--bodies", type=int, default=8, help="bodies per machine")
    ap.add_argument("--machines", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n = a.panels * a.machines
    o = {k: rng.normal(size=n) for k in ("subf", "alpha", "q", "lift", "drag",
                                         "d_bluff", "buoy", "vn")}
    mb = rng.normal(size=a.bodies)
    m_slam, prev = rng.normal(size=n), rng.normal(size=n)

    def unread(pa, pb):
        float(o["subf"][pa:pb].max())
        float(np.abs(o["alpha"][pa:pb]).max())
        float(o["q"][pa:pb].max())
        float(np.abs(o["lift"][pa:pb]).sum())
        float(np.abs(o["drag"][pa:pb] + o["d_bluff"][pa:pb]).sum())
        float(o["buoy"][pa:pb].sum())
        float(mb.sum())

    def read(pa, pb):
        float(o["subf"][pa:pb].mean())
        float(np.abs((m_slam[pa:pb] - prev[pa:pb]) / 0.004 * o["vn"][pa:pb]).max())

    for name, fn in (("unread (6 fields + added_mass)", unread),
                     ("read (mean_submerged, slam)", read)):
        t = time.perf_counter()
        for _ in range(a.reps):
            for m in range(a.machines):
                fn(m * a.panels, (m + 1) * a.panels)
        us = 1e6 * (time.perf_counter() - t) / a.reps
        print(f"{name:32s} {us:7.1f} us per batched step "
              f"({a.machines} machines x {a.panels} panels)")


if __name__ == "__main__":
    main()
