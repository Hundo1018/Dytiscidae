"""Render one still per point in the score distribution.

A histogram of `mission_fraction` says how many designs scored what.  It does
not say what a score buys, and every wrong call this project has made about a
score has been a call about what the machine was physically doing -- a wingless
design scoring 0.853 for flight, a tumble scoring as a turn, a bounce scoring as
a take-off.  Each of those was visible the moment somebody looked at the body.

So: pick the elites sitting at chosen percentiles of the distribution, render
each one, and hand `report.py` a still to put under the bar.  The report embeds
whatever is in `<run>/media/thumbs/manifest.json` and draws the plain histogram
if this was never run.

    .venv/bin/python .claude/skills/training-report/bodies.py runs/arch35

Minutes of compute, and it needs a GL context -- never run it against a machine
that is also running a search.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Where in the distribution to look.  The top of the distribution is where the
#: exploits live and the bottom is what the population actually looks like, so
#: both ends are worth a picture; the middle is there so the ends can be read
#: against something.
PERCENTILES = [("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99), ("max", 100)]


def _frame(video: str, out: Path, at: float = 0.30) -> bool:
    """One frame out of a turntable, as a PNG."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-ss", str(at), "-i", video,
           "-frames:v", "1", "-vf", "scale=300:-1", str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except Exception:
        return False
    return out.exists() and out.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--by", choices=("mission", "fitness"), default="mission")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=420)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dytiscidae.ops.run import load_run_archive
    from dytiscidae.core.phenotype import build
    from dytiscidae.viz.render import render_design, gl_available

    run = Path(a.run)
    archive, islands = load_run_archive(run)
    if archive is None:
        print(f"no archive in {run}")
        return 1
    # `gl_available` returns the backend *name* when it works and None when it
    # does not, so the test is `is None` -- reading a truthy backend name as a
    # failure is the obvious way to get this backwards.
    backend = gl_available()
    if backend is None:
        print("no GL context — cannot render bodies")
        return 1
    print(f"GL backend: {backend}")

    key = ((lambda e: (e.meta or {}).get("mission_fraction") or 0.0)
           if a.by == "mission" else (lambda e: e.fitness))
    pool = sorted(archive.cells.values(), key=key)
    if len(pool) < len(PERCENTILES):
        print(f"only {len(pool)} elites — not enough to sample a distribution")
        return 1

    out_dir = run / "media" / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, pct in PERCENTILES:
            i = min(int(round(pct / 100 * (len(pool) - 1))), len(pool) - 1)
            e = pool[i]
            m = e.meta or {}
            try:
                made = render_design(build(e.genome), tmp, stem=name, duration=2.0,
                                     turntable=True, width=a.width, height=a.height)
            except Exception as exc:                       # noqa: BLE001
                print(f"  {name}: render failed ({exc})")
                continue
            vid = next((v for v in made if v.endswith(".mp4")), None)
            if not vid:
                print(f"  {name}: nothing rendered")
                continue
            png = out_dir / f"{name}.png"
            if not _frame(vid, png):
                print(f"  {name}: could not extract a frame")
                continue
            rows.append({
                "file": png.name, "label": name, "pct": pct,
                "by": a.by,
                "mission_fraction": round(m.get("mission_fraction") or 0, 4),
                "fitness": round(float(e.fitness), 4),
                "body_plan": m.get("body_plan"), "n_parts": m.get("n_parts"),
                "dof": m.get("dof"), "mass": m.get("mass"), "span": m.get("span"),
                "wing_area": m.get("wing_area"),
                "energy_margin": m.get("energy_margin"),
                "air_gates": (m.get("air_gates") or [])[:2],
                "island": m.get("island"),
            })
            print(f"  {name} (pct {pct:3d}): mission {rows[-1]['mission_fraction']:.4f} "
                  f"fitness {rows[-1]['fitness']:.4f}  {png}")

    if not rows:
        print("nothing rendered")
        return 1
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"{out_dir / 'manifest.json'}  ({len(rows)} bodies)")
    print("re-run report.py to embed them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
