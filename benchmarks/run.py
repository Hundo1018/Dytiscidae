#!/usr/bin/env python3
"""Run the ladder and report the first layer that departs.

    PYTHONPATH=. python benchmarks/run.py

Exit code is 0 whatever the outcome: a departing layer is a measurement, not a
broken run.  `tests/test_math.py` is where an invariant that must hold is
asserted; this is where the fidelity boundary is located.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

from benchmarks import layers as L  # noqa: E402
from benchmarks._report import write  # noqa: E402

LADDER = [
    L.layer1_rigid_body,
    L.layer2_added_mass,
    L.layer3_drag,
    L.layer4_buoyancy,
    L.layer5_jet,
    L.layer6_articulated,
    L.layer7_fluid_surrogate,
    L.layer8_controller,
]


def main() -> int:
    started = time.time()
    done = []
    for fn in LADDER:
        t0 = time.time()
        lay = fn()
        done.append(lay)
        head = f"layer {lay.number} -- {lay.title}"
        print(f"\n{head}\n{'-' * len(head)}  ({lay.adds})")
        for c in lay.checks:
            print(c.line())
            if c.note and not c.agrees:
                print(f"           {c.note}")
        print(f"  [{lay.number}] {'holds' if lay.agrees else 'DEPARTS'}, "
              f"worst error {lay.worst:.3g}, {time.time() - t0:.1f}s")

    print("\n" + "=" * 78)
    print("the ladder")
    for lay in done:
        mark = "holds  " if lay.agrees else "DEPARTS"
        bad = [c.name for c in lay.checks if not c.agrees]
        print(f"  {lay.number}. {lay.title:<20} {mark}  worst {lay.worst:.3g}"
              + (f"   <- {', '.join(bad)}" if bad else ""))
    first = next((lay.number for lay in done if not lay.agrees), None)
    if first is None:
        print("\n  Every layer reproduces its reference.  The simulator agrees "
              "with an answer that can be written down, all the way up to the "
              "controller.")
    else:
        print(f"\n  The simulator reproduces its reference up to layer "
              f"{first - 1} and departs at layer {first}.")
        print("  Everything above layer "
              f"{first} is built on a layer that does not agree with its own "
              "reference, so a result from it carries that error whatever else "
              "is true of it.")
    path = write(done, Path(__file__).parent / "results", started)
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
