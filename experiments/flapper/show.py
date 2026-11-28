"""Pictures and short films of the flapper as it is measured.

Each film is the same code path as a stored row (the walk of F2, the swim of
F3, the flight of F4), with a camera on it: `landwalk.make_env` is wrapped so
that every environment those row functions build is filmed.  Nothing is
re-tuned and nothing is scored here.

    MUJOCO_GL=osmesa FLAPPER_BODY_B='{"hull_radius": 0.0209}' DYTISCIDAE_DEPTH_PAIR=1 PYTHONPATH=. \
        .venv/bin/python experiments/flapper/show.py
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import landwalk as LW  # noqa: E402

OUT = HERE / "media"
W, H = 960, 720


class Camera:
    def __init__(self, env, distance, azimuth, elevation, every, label, keep):
        self.env, self.every, self.label, self.keep = env, every, label, keep
        # As `viz/film.py` films a machine: a free camera looking at the root,
        # and the near plane brought in (the scene is hundreds of metres wide and
        # the machine is 0.4 m, so the default near plane clips it away).
        from dytiscidae.viz.render import _geometry_bounds, _near_plane
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
        _, extent = _geometry_bounds(env)
        self._near = _near_plane(env.model, extent)
        self._near.__enter__()
        self.r = mujoco.Renderer(env.model, height=H, width=W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.frames, self.n = [], 0
        self._step = env.step
        env.step = self.step

    def shot(self):
        self.cam.lookat[:] = self.env.root_pos()
        self.r.update_scene(self.env.data, self.cam)
        img = Image.fromarray(self.r.render())
        d = ImageDraw.Draw(img)
        p = self.env.data.xpos[self.env.root_body]
        d.text((12, 10), f"{self.label}   t = {self.env.data.time:5.2f} s   "
               f"x = {p[0]:+.2f} m  z = {p[2]:+.2f} m", fill=(255, 255, 255))
        return np.asarray(img)

    def step(self, *a, **k):
        out = self._step(*a, **k)
        if self.n % self.every == 0 and len(self.frames) < self.keep:
            self.frames.append(self.shot())
        self.n += 1
        return out


def film(name, label, run, *, distance, azimuth, elevation, every, fps, keep):
    cams = []
    orig = LW.make_env

    def make_env(layout, seed, **over):
        env = orig(layout, seed, **over)
        cams.append(Camera(env, distance, azimuth, elevation, every, label, keep))
        return env

    LW.make_env = make_env
    try:
        row = run()
    finally:
        LW.make_env = orig
    frames = cams[0].frames
    imageio.mimsave(str(OUT / f"{name}.mp4"), frames, fps=fps, macro_block_size=None)
    for tag, i in (("a", len(frames) // 4), ("b", len(frames) // 2)):
        Image.fromarray(frames[i]).save(OUT / f"{name}_{tag}.png")
    print(name, len(frames), "frames", {k: row.get(k) for k in ("score", "met", "failure", "bad_qacc")
                                         if k in row}, flush=True)


def stills():
    env = LW.make_env("B", 0)
    from dytiscidae.envs.triphibian import Domain
    env.reset(Domain.AIR, randomise=False)
    env.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    c = Camera(env, 0.75, 135, -25, 1, "", 1)       # one renderer, turned for each view
    for tag, (dist, az, el) in {"three_quarter": (0.75, 135, -25), "top": (0.8, 90, -80),
                                "side": (0.7, 90, -5), "front": (0.7, 180, -8)}.items():
        c.cam.distance, c.cam.azimuth, c.cam.elevation = dist, az, el
        c.label = f"the two-pair flapper, {tag.replace('_', ' ')}"
        Image.fromarray(c.shot()).save(OUT / f"body_{tag}.png")
    print("stills written", flush=True)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    import fly
    import swim
    stills()
    if "--stills" in sys.argv:
        return 0
    film("land_walk", "land: walking on the wings (F2 gait), real time",
         lambda: LW.row_speed("B", 0), distance=0.9, azimuth=120, elevation=-18,
         every=33, fps=30, keep=240)
    film("water_swim", "water: the diagonal stroke at 1 Hz (F3), real time",
         lambda: swim.row_straight("S2x", 0), distance=1.0, azimuth=120, elevation=-20,
         every=33, fps=30, keep=240)
    film("water_dive", "water: told to go from 4 m to 5.5 m (F3e), real time",
         lambda: swim.row_step("S2d", 40, 5.5), distance=1.6, azimuth=90, elevation=-10,
         every=33, fps=30, keep=300)
    film("air_fly", "air: level flight at 12 Hz (F4), real time",
         lambda: fly.row_level("fly", 1), distance=1.4, azimuth=120, elevation=-15,
         every=33, fps=30, keep=240)
    film("air_fly_slow", "air: level flight at 12 Hz (F4), 8x slow motion",
         lambda: fly.row_level("fly", 1), distance=1.0, azimuth=135, elevation=-18,
         every=4, fps=30, keep=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
