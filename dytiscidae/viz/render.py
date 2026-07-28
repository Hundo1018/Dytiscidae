"""Offscreen rendering: watch a design actually fly, swim and walk.

Numbers in a telemetry file will tell you a machine scored 0.7 in water.  They
will not tell you that it is scoring 0.7 by sinking gracefully, which is the
sort of thing that is obvious in two seconds of video and can survive
indefinitely in an archive otherwise.  Rendering elites is not decoration; it is
the cheapest available check on whether the score means what it says.

Requires a working EGL or OSMesa context.  Where none exists -- plenty of
headless containers -- the functions degrade to writing a trajectory plot
instead of raising, so a run is never lost to a missing GL driver.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _try_gl() -> str | None:
    """Pick a usable headless GL backend, or None."""
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco

            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>"
            )
            r = mujoco.Renderer(m, height=64, width=64)
            r.close()
            return backend
        except Exception:
            continue
    return None


def render_episode(
    env,
    duration: float,
    *,
    domain,
    controller=None,
    out_path: str | Path = "episode.mp4",
    fps: int = 30,
    width: int = 640,
    height: int = 400,
    camera_distance: float = 6.0,
) -> str | None:
    """Roll out an episode and write a video.  Returns the path, or None.

    The camera tracks the machine and the scene includes the waterline, so the
    domain the machine is in is visible at a glance -- which is the whole point
    when the failure mode you are looking for is "flew into the sea".
    """
    backend = _try_gl()
    if backend is None:
        return _render_trajectory_fallback(env, duration, domain, controller, out_path)

    import imageio.v2 as imageio
    import mujoco

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = camera_distance
    cam.elevation = -18.0
    cam.azimuth = 135.0

    env.reset(domain)
    frames = []
    n_steps = int(duration / env.timestep)
    every = max(1, int(1.0 / (fps * env.timestep)))
    params = controller.params if controller is not None else env.cpg.base
    basis = controller.basis_for(domain) if controller is not None else None
    cur = params

    for i in range(n_steps):
        if controller is not None and controller.policy is not None and basis is not None \
                and i % max(1, int(1.0 / (25.0 * env.timestep))) == 0:
            cur = basis.command_params(params, controller.policy.act(env.observation()),
                                       env.cpg.n)
        if not env.step(env.cpg.command(cur, env.data.time)):
            break
        if not np.all(np.isfinite(env.root_pos())):
            break
        if i % every == 0:
            cam.lookat[:] = env.root_pos()
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())

    renderer.close()
    if not frames:
        return None
    imageio.mimsave(str(out_path), frames, fps=fps)
    return str(out_path)


def _render_trajectory_fallback(env, duration, domain, controller, out_path) -> str | None:
    """No GL available: plot the trajectory against the waterline instead.

    Less informative than video but it still answers the question that matters
    most -- which side of the surface did it spend its time on.
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
        xs.append(p[0])
        zs.append(p[2])
        subs.append(env.solver.diag.mean_submerged)

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
    ax.set_title(f"{domain.value if hasattr(domain,'value') else domain} trajectory")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def render_elites(archive, run_dir: str | Path, *, top: int = 3, duration: float = 8.0) -> list[str]:
    """Render the archive's best few designs in each domain."""
    from ..core.phenotype import build
    from ..envs.triphibian import Domain, TriphibianEnv

    run_dir = Path(run_dir)
    made = []
    elites = sorted(archive.cells.values(), key=lambda e: -e.fitness)[:top]
    for rank, e in enumerate(elites):
        try:
            p = build(e.genome)
            env = TriphibianEnv(p, seed=0)
            for dom in (Domain.AIR, Domain.WATER, Domain.LAND):
                path = run_dir / "media" / f"elite{rank}_{dom.value}.mp4"
                got = render_episode(env, duration, domain=dom, out_path=path)
                if got:
                    made.append(got)
        except Exception:
            continue
    return made
