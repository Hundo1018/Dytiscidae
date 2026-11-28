"""R0b, post hoc (not pre-registered): E-Flap's reported limit is 520 g of
payload at 5.6 Hz (1.03 kg all-up).  Does the model carry that, and how much
more, at the machine's own stroke of 40-50 deg?

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/reproduce_payload.py OUT.jsonl
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MASSES = (1.03, 1.3, 1.6, 2.0)
FREQ = 5.6


def job(a):
    import reproduce as R
    mass, V, Phi, Psi = a
    R.MACHINES["eflap"]["mass"] = mass
    r = R.point(("eflap", V, FREQ, Phi, Psi))
    r["load_mass"] = mass
    return r


def main() -> int:
    import reproduce as R
    grid = [(m, V, Phi, Psi) for m in MASSES for V in (4.0, 5.0, 6.0, 7.0)
            for Phi in (40.0, 50.0) for Psi in R.PSI]
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(job, grid, chunksize=1)
    with open(Path(sys.argv[1]).resolve(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for m in MASSES:
        for Phi in (40.0, 50.0):
            S = [r for r in rows if r["load_mass"] == m and r["Phi_deg"] == Phi]
            sus = [r for r in S if r["sustains"]]
            if sus:
                b = min(sus, key=lambda r: r["aero_power_W"])
                x = max(sus, key=lambda r: r["Fx"])
                print(f"{m} kg Phi {Phi:.0f}: sustain {len(sus)} of {len(S)}, lowest "
                      f"{b['aero_power_W']:.0f} W at {b['V']:.0f} m/s, most spare thrust {x['Fx']:.2f} N")
            else:
                print(f"{m} kg Phi {Phi:.0f}: none of {len(S)} (trimmed {sum(r['trimmed'] for r in S)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
