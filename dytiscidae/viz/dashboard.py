"""Generates a self-contained HTML dashboard for a run.

Everything is inlined -- data, styling, drawing code -- so the output is a single
file that opens from disk with no server, no network and no build step.  That
constraint is not aesthetic: these runs happen on remote machines and in
containers that get reclaimed, and a dashboard that needs a toolchain to view is
a dashboard nobody looks at.

It is regenerated from the telemetry each time it is called, so calling it
periodically during a run gives a live view, and calling it once at the end
gives the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..ops.telemetry import Telemetry


def _svg_line_chart(series: list[tuple[str, list[float], str]], *, width=520, height=170,
                    ylabel="") -> str:
    """Multi-series line chart as inline SVG, with an independent right axis for
    the second series (coverage and QD score differ by orders of magnitude)."""
    if not series or not any(len(s[1]) > 1 for s in series):
        return '<div class="empty">not enough data yet</div>'
    pad_l, pad_r, pad_t, pad_b = 44, 44, 10, 22
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = max(len(s[1]) for s in series)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" class="ax"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" class="ax"/>')
    for si, (name, vals, colour) in enumerate(series):
        if len(vals) < 2:
            continue
        v = np.array(vals, float)
        v = np.where(np.isfinite(v), v, 0.0)
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-12:
            hi = lo + 1.0
        pts = []
        for i, y in enumerate(v):
            px = pad_l + (i / max(n - 1, 1)) * w
            py = pad_t + h - (y - lo) / (hi - lo) * h
            pts.append(f"{px:.1f},{py:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.8"/>')
        side = pad_l - 6 if si == 0 else pad_l + w + 6
        anchor = "end" if si == 0 else "start"
        parts.append(f'<text x="{side}" y="{pad_t+8}" class="tick" text-anchor="{anchor}">'
                     f'{hi:.3g}</text>')
        parts.append(f'<text x="{side}" y="{pad_t+h}" class="tick" text-anchor="{anchor}">'
                     f'{lo:.3g}</text>')
    parts.append(f'<text x="{pad_l}" y="{height-4}" class="tick">gen 0</text>')
    parts.append(f'<text x="{pad_l+w}" y="{height-4}" class="tick" text-anchor="end">'
                 f'gen {n-1}</text>')
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:{c}"></i>{n_}</span>' for n_, _, c in series
    )
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def _heatmap(archive_json: dict, ax_x: int, ax_y: int, *, cell=26) -> str:
    """MAP-Elites projection: best fitness in each (x, y) bin."""
    axes = archive_json.get("axes", [])
    if len(axes) <= max(ax_x, ax_y):
        return '<div class="empty">no archive</div>'
    nx, ny = axes[ax_x]["bins"], axes[ax_y]["bins"]
    grid = np.full((nx, ny), np.nan)
    for e in archive_json.get("elites", []):
        c = e["cell"]
        i, j = c[ax_x], c[ax_y]
        f = e["fitness"]
        if np.isnan(grid[i, j]) or f > grid[i, j]:
            grid[i, j] = f
    finite = grid[np.isfinite(grid)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi - lo < 1e-9:
        hi = lo + 1e-9

    w, h = nx * cell, ny * cell
    out = [f'<svg viewBox="0 0 {w+60} {h+34}" class="heat">']
    for i in range(nx):
        for j in range(ny):
            v = grid[i, j]
            x, y = 46 + i * cell, (ny - 1 - j) * cell + 4
            if np.isnan(v):
                out.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                           f'class="cell-empty"/>')
            else:
                t = (v - lo) / (hi - lo)
                # Perceptually ordered dark-to-bright ramp; works in both themes.
                r = int(30 + 225 * t**0.8)
                g = int(45 + 150 * t)
                b = int(90 + 40 * (1 - t))
                out.append(
                    f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                    f'fill="rgb({r},{g},{b})"><title>{axes[ax_x]["name"]}[{i}] '
                    f'{axes[ax_y]["name"]}[{j}] fitness={v:.3f}</title></rect>'
                )
    ax_lo, ax_hi = axes[ax_x]["lo"], axes[ax_x]["hi"]
    ay_lo, ay_hi = axes[ax_y]["lo"], axes[ax_y]["hi"]
    out.append(f'<text x="46" y="{h+22}" class="tick">{ax_lo:.2g}</text>')
    out.append(f'<text x="{46+w}" y="{h+22}" class="tick" text-anchor="end">{ax_hi:.2g}'
               f' &#8594; {axes[ax_x]["name"]}</text>')
    out.append(f'<text x="42" y="{h-2}" class="tick" text-anchor="end">{ay_lo:.2g}</text>')
    out.append(f'<text x="42" y="14" class="tick" text-anchor="end">{ay_hi:.2g}</text>')
    out.append("</svg>")
    out.append(f'<div class="cap">{axes[ax_y]["name"]} (vertical) vs '
               f'{axes[ax_x]["name"]} (horizontal) &middot; brighter = fitter &middot; '
               f'grey = never reached</div>')
    return "".join(out)


def _bars(rows: list[tuple[str, float]], *, width=520) -> str:
    if not rows:
        return '<div class="empty">no operator statistics yet</div>'
    mx = max(abs(v) for _, v in rows) or 1.0
    out = ['<div class="bars">']
    for name, v in rows:
        pct = abs(v) / mx * 100.0
        colour = "var(--good)" if v >= 0 else "var(--bad)"
        out.append(
            f'<div class="bar"><span class="bn">{name}</span>'
            f'<span class="bt"><i style="width:{pct:.1f}%;background:{colour}"></i></span>'
            f'<span class="bv">{v:+.3f}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


CSS = """
:root{--bg:#ffffff;--fg:#16191d;--mut:#5d6570;--card:#f6f7f9;--line:#dfe3e8;
--good:#2f7d5b;--bad:#a8443a;--accent:#2a6f9e;--accent2:#8a5cb8}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e6e9ee;--mut:#98a2b0;
--card:#171b21;--line:#262c35;--good:#4fbf8b;--bad:#e0776a;--accent:#5aa9dd;--accent2:#b088e0}}
:root[data-theme="dark"]{--bg:#0f1216;--fg:#e6e9ee;--mut:#98a2b0;--card:#171b21;
--line:#262c35;--good:#4fbf8b;--bad:#e0776a;--accent:#5aa9dd;--accent2:#b088e0}
:root[data-theme="light"]{--bg:#ffffff;--fg:#16191d;--mut:#5d6570;--card:#f6f7f9;
--line:#dfe3e8;--good:#2f7d5b;--bad:#a8443a;--accent:#2a6f9e;--accent2:#8a5cb8}
*{box-sizing:border-box}
body{margin:0;padding:22px;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
h1{font-size:19px;margin:0 0 2px} h2{font-size:13px;text-transform:uppercase;
letter-spacing:.07em;color:var(--mut);margin:0 0 10px;font-weight:600}
.sub{color:var(--mut);margin-bottom:18px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px;
overflow-x:auto}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:10px;
margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.stat b{display:block;font-size:21px;font-variant-numeric:tabular-nums;line-height:1.2}
.stat span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.chart,.heat{width:100%;height:auto;display:block}
.ax{stroke:var(--line);stroke-width:1}
.tick{fill:var(--mut);font-size:9px}
.cell-empty{fill:var(--line);opacity:.4}
.cap{color:var(--mut);font-size:11px;margin-top:6px}
.legend{margin-top:5px;font-size:11px;color:var(--mut)}
.key{margin-right:12px} .key i{display:inline-block;width:10px;height:3px;
vertical-align:middle;margin-right:4px;border-radius:2px}
.bars{display:flex;flex-direction:column;gap:5px}
.bar{display:grid;grid-template-columns:130px 1fr 56px;align-items:center;gap:8px;font-size:12px}
.bn{color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bt{background:var(--line);border-radius:3px;height:9px;overflow:hidden}
.bt i{display:block;height:100%}
.bv{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.empty{color:var(--mut);font-style:italic;padding:14px 0;font-size:12px}
.axes li{margin-bottom:3px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
background:var(--line);color:var(--mut)}
.warn{color:var(--bad)}
"""


def build_dashboard(run_dir: str | Path, out_path: str | Path | None = None) -> str:
    """Read a run directory and write ``dashboard.html``.  Returns its path."""
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "dashboard.html"

    gens = [g for g in Telemetry.read(run_dir, "generations") if "generation" in g]
    exploits = Telemetry.read(run_dir, "exploits")
    events = Telemetry.read(run_dir, "events")
    archive_path = run_dir / "archive.json"
    archive = json.loads(archive_path.read_text()) if archive_path.exists() else {}

    last = gens[-1] if gens else {}
    best = last.get("best", {}) or {}

    cov = [g.get("coverage", 0.0) for g in gens]
    qd = [g.get("qd_score", 0.0) for g in gens]
    bf = [g.get("best_fitness", 0.0) for g in gens]
    filled = [g.get("filled", 0) for g in gens]

    ops = last.get("top_operators", [])
    op_rows = [(o["operator"], o["recent"]) for o in ops]

    # Regime timeline: how the curator's read of the run changed.
    regimes: list[tuple[int, str]] = []
    for g in gens:
        name = g.get("regime", "?")
        if not regimes or regimes[-1][1] != name:
            regimes.append((g.get("generation", 0), name))

    # Best design's discovered control axes.
    axes_html = '<div class="empty">no mobility identification recorded</div>'
    elites = archive.get("elites", [])
    if elites:
        top = max(elites, key=lambda e: e["fitness"])
        ma = (top.get("meta") or {}).get("mobility_axes")
        if isinstance(ma, dict) and ma:
            blocks = []
            for medium, lines in ma.items():
                items = "".join(f"<li>{ln}</li>" for ln in lines)
                blocks.append(f"<b>{medium}</b><ul class='axes'>{items}</ul>")
            axes_html = "".join(blocks)

    feasible = sum(1 for e in elites if (e.get("meta") or {}).get("feasible"))
    masses = [(e.get("meta") or {}).get("mass") for e in elites]
    masses = [m for m in masses if isinstance(m, (int, float))]

    stats = [
        ("generation", last.get("generation", 0)),
        ("elites", last.get("filled", 0)),
        ("coverage", f"{last.get('coverage', 0)*100:.2f}%"),
        ("QD score", f"{last.get('qd_score', 0):.2f}"),
        ("best fitness", f"{last.get('best_fitness', 0):.3f}"),
        ("evaluations", last.get("evaluated", 0)),
        ("tier-0 rejects", last.get("tier0_rejected", 0)),
        ("tier-2 verified", last.get("promotions", 0)),
        ("exploits caught", len(exploits)),
        ("feasible elites", f"{feasible}/{len(elites)}"),
    ]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for k, v in stats)

    best_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>'
        for k, v in best.items()
        if isinstance(v, (int, float, str))
    ) or '<tr><td colspan="2" class="empty">no elite yet</td></tr>'

    exploit_rows = "".join(
        f'<tr><td class="num">{e.get("gen","")}</td><td>{e.get("reason","")}</td></tr>'
        for e in exploits[-12:]
    ) or '<tr><td colspan="2" class="empty">none caught &mdash; good, but keep watching</td></tr>'

    regime_html = " &rarr; ".join(f'<span class="tag">g{g}: {n}</span>' for g, n in regimes[-8:])

    mass_note = ""
    if masses:
        mass_note = (
            f"Explored masses span {min(masses):.1f}&ndash;{max(masses):.1f} kg; "
            f"the 15 kg reference ambition is "
            f"{'inside' if min(masses) <= 15 <= max(masses) else 'outside'} that range."
        )

    html = f"""<title>Dytiscidae &mdash; run dashboard</title>
<style>{CSS}</style>
<h1>Dytiscidae &mdash; triphibian design search</h1>
<div class="sub">{run_dir} &middot; generation {last.get('generation', 0)} &middot;
regime <b>{last.get('regime','?')}</b> &mdash; {last.get('regime_note','')}</div>

<div class="stats">{stat_html}</div>

<div class="grid">
  <div class="card"><h2>Archive growth</h2>
    {_svg_line_chart([("coverage", cov, "var(--accent)"),
                      ("QD score", qd, "var(--accent2)")])}</div>
  <div class="card"><h2>Best fitness &amp; elite count</h2>
    {_svg_line_chart([("best fitness", bf, "var(--good)"),
                      ("elites", [float(x) for x in filled], "var(--mut)")])}</div>

  <div class="card"><h2>Map: mass &times; buoyancy</h2>{_heatmap(archive, 0, 1)}</div>
  <div class="card"><h2>Map: flight &times; dive competence</h2>{_heatmap(archive, 2, 3)}</div>

  <div class="card"><h2>Mutation operator value (windowed)</h2>{_bars(op_rows)}
    <div class="cap">Bandit reward per application. The curator spends its
    mutation budget in proportion to these, and they move as the run
    progresses.</div></div>

  <div class="card"><h2>Discovered control axes of the best design</h2>{axes_html}
    <div class="cap">Measured, not assumed. Each mode is a coordinated CPG
    perturbation and the label says what motion it actually produces. Air and
    water bases differ because the same surface is a wing in one and a paddle in
    the other.</div></div>

  <div class="card"><h2>Best design</h2>
    <table>{best_rows}</table>
    <div class="cap">{mass_note}</div></div>

  <div class="card"><h2>Simulator exploits quarantined</h2>
    <table><tr><th>gen</th><th>reason</th></tr>{exploit_rows}</table>
    <div class="cap">Candidates that scored by beating the model rather than the
    task. Their archive cells are tainted so the search does not rediscover the
    same trick from the same place.</div></div>

  <div class="card"><h2>Curator regime timeline</h2><div>{regime_html or '&mdash;'}</div>
    <div class="cap">The curator reclassifies the run each generation and sets
    structural mutation pressure, emitter mix and feasibility bias from what it
    sees. Search settings are an output here, not a configuration.</div></div>
</div>
<div class="sub" style="margin-top:16px">{len(events)} evaluation records &middot;
regenerate any time with <code>python -m dytiscidae.ops.run dashboard --run {run_dir}</code></div>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return str(out_path)
