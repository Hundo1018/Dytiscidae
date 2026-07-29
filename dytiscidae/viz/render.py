"""Watching the machine: turntables and episode video in each medium.

Telemetry will tell you a design scored 0.43 in water.  It will not tell you
that it is scoring 0.43 by sinking gracefully with its wings folded, which is
obvious in two seconds of video and can otherwise survive in an archive
indefinitely.  This is the cheapest available check on whether a number means
what it says.

Two products:

* **Turntable** -- the machine held still and rotated, so the morphology itself
  is legible.  Generated bodies are hard to read from a single angle: what looks
  like a wing from the front is often a paddle seen edge-on.
* **Episode** -- the machine actually moving, with the camera tracking it and
  the waterline in frame, so which medium it is in is visible rather than
  inferred.  A HUD strip carries depth, speed and power, because "it is flying"
  and "it is falling slowly" look identical without numbers.

Rendering is software-rasterised through OSMesa (see ``dytiscidae/__init__``),
so it needs no GPU and no display.  If no GL backend can be found at all, both
functions fall back to a matplotlib trajectory plot rather than raising -- a run
should never be lost to a missing driver.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Colours for the HUD, matched to the domain coding used in the dashboard.
_DOMAIN_RGB = {"air": (127, 168, 190), "water": (46, 110, 133), "land": (138, 116, 68)}


def gl_available() -> str | None:
    """Return the working MuJoCo GL backend name, or None."""
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><light pos='0 0 3'/>"
            "<geom type='sphere' size='.1'/></worldbody></mujoco>"
        )
        r = mujoco.Renderer(m, height=32, width=32)
        r.close()
        import os

        return os.environ.get("MUJOCO_GL", "default")
    except Exception:
        return None


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------


def _hud(frame: np.ndarray, *, domain: str, t: float, depth: float, speed: float,
         power: float, submerged: float, label: str) -> np.ndarray:
    """Overlay a compact instrument strip.

    Drawn with PIL if present, and with direct numpy block writes if not, so the
    HUD never becomes the reason a render fails.
    """
    h, w = frame.shape[:2]
    accent = _DOMAIN_RGB.get(domain, (200, 200, 200))
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        bar_h = 34
        d.rectangle([0, h - bar_h, w, h], fill=(8, 14, 18, 205))
        d.rectangle([0, h - bar_h, w, h - bar_h + 2], fill=(*accent, 255))
        txt = (
            f"{label}   {domain.upper()}   t={t:5.1f}s   "
            f"depth={depth:+6.2f}m   v={speed:5.2f}m/s   P={power:6.0f}W"
        )
        d.text((10, h - bar_h + 11), txt, fill=(226, 232, 224, 255))

        # Submersion gauge: a short vertical bar that fills as the machine wets.
        gx, gy, gw, gh = w - 26, 14, 12, 96
        d.rectangle([gx, gy, gx + gw, gy + gh], fill=(8, 14, 18, 170))
        fill_h = int(gh * float(np.clip(submerged, 0.0, 1.0)))
        if fill_h > 0:
            d.rectangle([gx, gy + gh - fill_h, gx + gw, gy + gh],
                        fill=(46, 110, 133, 235))
        d.rectangle([gx, gy, gx + gw, gy + gh], outline=(226, 232, 224, 120))
        return np.asarray(img)
    except Exception:
        out = frame.copy()
        out[h - 30 :, :, :] = (out[h - 30 :, :, :] * 0.25).astype(out.dtype)
        out[h - 30 : h - 28, :, :] = accent
        fill = int(90 * float(np.clip(submerged, 0.0, 1.0)))
        if fill > 0:
            out[110 - fill : 110, w - 24 : w - 12, :] = _DOMAIN_RGB["water"]
        return out


# --------------------------------------------------------------------------
# Turntable
# --------------------------------------------------------------------------


def render_turntable(
    env,
    *,
    out_path: str | Path,
    frames: int = 96,
    width: int = 720,
    height: int = 480,
    fps: int = 24,
    elevation: float = -14.0,
    label: str = "",
) -> str | None:
    """Rotate the camera once around the machine held in a neutral pose.

    The machine is posed mid-flap rather than at rest, because a flapping design
    at rest has its surfaces stacked flat and is nearly unreadable.
    """
    backend = gl_available()
    if backend is None:
        return None

    import imageio.v2 as imageio
    import mujoco

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ..envs.triphibian import Domain

    env.reset(Domain.AIR, randomise=False)
    # Pose it a quarter-cycle into the stroke so joints are visibly deflected.
    if len(env.act_names):
        env.data.ctrl[:] = env.cpg.command(env.cpg.base, 0.25 / max(env.cpg.base.frequency, 0.1))
        for _ in range(90):
            mujoco.mj_step(env.model, env.data)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    span = max(env.p.max_span, env.p.body_length, 0.4)
    cam.distance = span * 1.25
    cam.elevation = elevation
    cam.lookat[:] = env.root_pos()

    out = []
    for i in range(frames):
        cam.azimuth = 360.0 * i / frames
        renderer.update_scene(env.data, camera=cam)
        out.append(renderer.render())
    renderer.close()

    imageio.mimsave(str(out_path), out, fps=fps, macro_block_size=None)
    return str(out_path)


# --------------------------------------------------------------------------
# Episode
# --------------------------------------------------------------------------


def capture_episode(
    env,
    duration: float,
    *,
    domain,
    controller=None,
    fps: int = 25,
    width: int = 720,
    height: int = 480,
    label: str = "",
    follow: bool = True,
) -> list[np.ndarray]:
    """Roll out an episode and return the HUD-annotated frames.

    Separated from writing a file so the same rollout can be composited against
    another one -- which is the only honest way to show that a controller
    learned something, since "it moves" and "it moves better" are
    indistinguishable from a single clip.
    """
    import mujoco

    dom_name = domain.value if hasattr(domain, "value") else str(domain)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    span = max(env.p.max_span, env.p.body_length, 0.5)
    cam.distance = span * 2.1
    cam.elevation = -12.0
    # Slightly off the beam so the machine is seen three-quarter rather than
    # side-on, which hides all spanwise structure.
    cam.azimuth = 118.0

    env.reset(domain)
    params = controller.params if controller is not None else env.cpg.base
    basis = controller.basis_for(domain) if controller is not None else None
    cur = params

    n_steps = int(duration / env.timestep)
    every = max(1, int(1.0 / (fps * env.timestep)))
    ctrl_every = max(1, int(1.0 / (25.0 * env.timestep)))
    frames: list[np.ndarray] = []
    start = env.root_pos().copy()

    for i in range(n_steps):
        if controller is not None and controller.policy is not None and basis is not None \
                and i % ctrl_every == 0:
            cur = basis.command_params(params, controller.policy.act(env.observation()),
                                       env.cpg.n)
        alive = env.step(env.cpg.command(cur, env.data.time))
        pos = env.root_pos()
        if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 300:
            break
        if i % every == 0:
            if follow:
                cam.lookat[:] = pos
            renderer.update_scene(env.data, camera=cam)
            img = renderer.render()
            tw = env.body_twist()
            frames.append(
                _hud(
                    img,
                    domain=dom_name,
                    t=env.data.time,
                    depth=env.depth(),
                    speed=float(np.linalg.norm(tw[:3])),
                    power=env.budget.mean_power,
                    submerged=env.solver.diag.mean_submerged,
                    label=label,
                )
            )
        if not alive:
            break

    renderer.close()
    return frames


def render_episode(
    env,
    duration: float,
    *,
    domain,
    controller=None,
    out_path: str | Path = "episode.mp4",
    fps: int = 25,
    width: int = 720,
    height: int = 480,
    label: str = "",
    follow: bool = True,
) -> str | None:
    """Roll out an episode and write video with the waterline in frame."""
    if gl_available() is None:
        return _render_trajectory_fallback(env, duration, domain, controller, out_path)

    import imageio.v2 as imageio

    frames = capture_episode(env, duration, domain=domain, controller=controller,
                             fps=fps, width=width, height=height, label=label,
                             follow=follow)
    if not frames:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=None)
    return str(out_path)


def _side_by_side(left: list[np.ndarray], right: list[np.ndarray],
                  left_label: str, right_label: str) -> list[np.ndarray]:
    """Stack two frame sequences horizontally, padding the shorter with its
    last frame so an early failure on one side stays visible rather than
    truncating the comparison."""
    if not left or not right:
        return left or right
    n = max(len(left), len(right))
    h = max(left[0].shape[0], right[0].shape[0])
    out = []
    try:
        from PIL import Image, ImageDraw
        have_pil = True
    except Exception:
        have_pil = False

    for i in range(n):
        a = left[min(i, len(left) - 1)]
        b = right[min(i, len(right) - 1)]
        pad = np.zeros((h, 3, 3), dtype=a.dtype)
        pad[:, :, :] = (40, 48, 52)
        frame = np.concatenate([a, pad, b], axis=1)
        if have_pil:
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img, "RGBA")
            for x, text, colour in (
                (10, left_label, (200, 118, 100, 255)),
                (a.shape[1] + 13, right_label, (110, 200, 150, 255)),
            ):
                d.rectangle([x - 6, 6, x + 7 * len(text) + 8, 26], fill=(8, 14, 18, 210))
                d.text((x, 11), text, fill=colour)
            frame = np.asarray(img)
        out.append(frame)
    return out


def render_learning_comparison(
    phenotype,
    controller,
    out_dir: str | Path,
    *,
    stem: str = "learned",
    duration: float = 10.0,
    domains=None,
    width: int = 560,
    height: int = 400,
    fps: int = 25,
) -> list[str]:
    """Render open-loop against learned control, side by side, per domain.

    The left panel is the same body running its raw pattern generator: the
    rhythm it was born with, no feedback.  The right panel is the trained
    policy commanding coefficients in the body's own discovered mobility basis.
    Same morphology, same physics, same camera -- the only difference is whether
    anything learned to drive it.
    """
    from ..envs.triphibian import DOMAIN_CYCLE, TriphibianEnv

    if gl_available() is None:
        return []
    import imageio.v2 as imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = domains or DOMAIN_CYCLE
    label = f"{phenotype.mass:.1f}kg {phenotype.max_span:.2f}m"
    made: list[str] = []

    env = TriphibianEnv(phenotype, seed=0)
    for dom in domains:
        env.reset(dom, randomise=False)
        before = capture_episode(env, duration, domain=dom, controller=None,
                                 width=width, height=height, fps=fps, label=label)
        env.reset(dom, randomise=False)
        after = capture_episode(env, duration, domain=dom, controller=controller,
                                width=width, height=height, fps=fps, label=label)
        frames = _side_by_side(before, after, "OPEN LOOP  (as born)",
                               "LEARNED  (trained policy)")
        if not frames:
            continue
        path = out_dir / f"{stem}_{dom.value}_compare.mp4"
        imageio.mimsave(str(path), frames, fps=fps, macro_block_size=None)
        made.append(str(path))
    return made


def _render_trajectory_fallback(env, duration, domain, controller, out_path) -> str | None:
    """No GL: plot the trajectory against the waterline instead.

    Much less informative than video, but it still answers the question that
    matters most -- which side of the surface did it spend its time on.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    env.reset(domain)
    xs, zs, subs = [], [], []
    params = controller.params if controller is not None else env.cpg.base
    for _ in range(int(duration / env.timestep)):
        if not env.step(env.cpg.command(params, env.data.time)):
            break
        p = env.root_pos()
        if not np.all(np.isfinite(p)):
            break
        xs.append(p[0]); zs.append(p[2]); subs.append(env.solver.diag.mean_submerged)

    out = Path(str(out_path)).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
    ax.axhspan(-25, 0, color="#1d4e63", alpha=0.25, lw=0)
    ax.axhline(0, color="#2f7f9f", lw=1.2)
    if xs:
        sc = ax.scatter(xs, zs, c=subs, cmap="viridis", s=6, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label="submerged fraction")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m), 0 = waterline")
    ax.set_title(f"{domain.value if hasattr(domain, 'value') else domain} trajectory")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------


def render_design(
    phenotype,
    out_dir: str | Path,
    *,
    stem: str = "design",
    duration: float = 10.0,
    controller=None,
    turntable: bool = True,
    width: int = 720,
    height: int = 480,
) -> list[str]:
    """Turntable plus one episode per domain for a single design."""
    from ..envs.triphibian import Domain, TriphibianEnv

    out_dir = Path(out_dir)
    made: list[str] = []
    label = f"{stem}  {phenotype.mass:.1f}kg  {phenotype.max_span:.2f}m span"

    env = TriphibianEnv(phenotype, seed=0)
    if turntable:
        got = render_turntable(env, out_path=out_dir / f"{stem}_turntable.mp4",
                               width=width, height=height, label=label)
        if got:
            made.append(got)

    for dom in (Domain.AIR, Domain.WATER, Domain.LAND):
        got = render_episode(
            env, duration, domain=dom, controller=controller,
            out_path=out_dir / f"{stem}_{dom.value}.mp4",
            width=width, height=height, label=label,
        )
        if got:
            made.append(got)
    return made


def render_elites(archive, run_dir: str | Path, *, top: int = 3, duration: float = 10.0) -> list[str]:
    """Render the archive's best few designs."""
    from ..core.phenotype import build

    run_dir = Path(run_dir)
    made: list[str] = []
    elites = sorted(archive.cells.values(), key=lambda e: -e.fitness)[:top]
    for rank, e in enumerate(elites):
        try:
            p = build(e.genome)
            made += render_design(p, run_dir / "media", stem=f"elite{rank}", duration=duration)
        except Exception:
            continue
    return made
