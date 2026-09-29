"""ROADMAP AK: the entry peaks over one full step of start height (0-7.5 cm,
so first contact falls on every phase of the step and of the damping refresh).
`python experiments/ray_entry/phase.py [hold] [every1] [stale] [noreact] [cycle]`:
`every1` refreshes the implicit damping every step, `stale` only on its 4-step
cadence (the defect), `cycle` runs nose-first at 8 m/s over one refresh cycle.
"""
import sys

import dytiscidae.physics.fluid as fl

if "stale" in sys.argv:
    # The defect: B refreshed on the cadence only, whatever crosses the surface.
    fl.ImplicitAeroDamping.due = lambda self: self._count % self.REFRESH_EVERY == 0
    print("B on the 4-step cadence only")
if "every1" in sys.argv:
    fl.ImplicitAeroDamping.REFRESH_EVERY = 1
    print("B refreshed every step")
FLAGS = set(sys.argv[1:])
sys.argv = ["probe"] + [a for a in sys.argv[1:] if a in ("hold", "noreact")]
sys.path.insert(0, "experiments/ray_entry")
import probe  # noqa: E402

CASES = ((0, 4.0), (80, 4.0), (0, 8.0), (80, 8.0), (80, 20.0))
HEIGHTS = (0.0, 0.015, 0.03, 0.045, 0.06, 0.075)
if "cycle" in FLAGS:
    # One full refresh cycle at 8 m/s nose-first: ~9.4 m/s at contact is
    # 3.8 cm a step, so four heights 4 cm apart put contact on each step.
    CASES, HEIGHTS = ((80, 8.0),), (0.0, 0.04, 0.08, 0.12)
for pit, v in CASES:
    for dz in HEIGHTS:
        probe.enter(pit, v, dz)
