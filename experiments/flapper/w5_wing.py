"""Pass W5, the machine: W4's two reference wings (Goe 417a rectangle, twist 0
and -8 deg) on W4's grids with the section's lift run carried past stall
(`goe417a+lift`, PREREG_W5_separated.md section 4).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/w5_wing.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from w4_wing import job  # noqa: E402


def main() -> int:
    cruise = [("cruise", V, f, Phi, Psi, 0.1, "plate") for V in (6.0, 8.0) for f in (2.0, 2.5, 3.0, 3.5)
              for Phi in (30.0, 40.0, 50.0, 60.0) for Psi in (0.0, 5.0, 10.0, 15.0, 20.0)]
    rest = [("rest", th, f, Psi, 0.1, "plate") for th in (45.0, 60.0, 75.0, 90.0)
            for f in (3.0, 3.5, 4.0, 4.5, 5.0) for Psi in (30.0, 40.0, 50.0, 60.0, 70.0)]
    grid = [("goe417a+lift", 1.0, tw, a) for tw in (0.0, -8.0) for a in cruise + rest]
    with mp.get_context("spawn").Pool(10) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
