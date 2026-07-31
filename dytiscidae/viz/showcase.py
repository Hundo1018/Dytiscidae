"""The showcase render: one continuous mission, with the flow and the loads on screen.

This is the artefact the whole pipeline exists to produce -- a trained controller
driving an evolved body through a complete air/water/land cycle in a single
unbroken simulation, with two overlays that make the physics visible:

**Wake.**  Vortex particles shed from the bound circulation the blade-element
solver is actually using (see ``physics/wake.py``).  Red and blue are the two
senses of vorticity; a flapping wing producing thrust lays down alternating
pairs, and a stalled one dumps a single shear layer.  This is a vortex method,
**not CFD** -- there is no Navier-Stokes solve anywhere in this project -- but it
is derived from the forces in use rather than drawn on top of them.

**Stress.**  Every structural member is recoloured each frame by its
instantaneous root-bending utilisation: the moment its own panels are applying,
divided by the material's allowable stress.  Green is comfortable, amber is
working, red is past the allowable.  This is the same calculation the Tier-0
feasibility check runs, evaluated against the loads actually being flown rather
than against a design case.

Both overlays cost real time and are never enabled during search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..physics.fluid import WING
from ..physics.structure import tube_section

# Stress ramp: green -> amber -> red, at utilisation 0, 0.6, 1.0.
_STRESS_STOPS = np.array([[0.30, 0.72, 0.45], [0.92, 0.72, 0.22], [0.86, 0.24, 0.18]])
_STRESS_LEVELS = np.array([0.0, 0.6, 1.0])


def stress_colour(util: float) -> tuple[float, float, float]:
    u = float(np.clip(util, 0.0, 1.0))
    r = np.interp(u, _STRESS_LEVELS, _STRESS_STOPS[:, 0])
    g = np.interp(u, _STRESS_LEVELS, _STRESS_STOPS[:, 1])
    b = np.interp(u, _STRESS_LEVELS, _STRESS_STOPS[:, 2])
    return float(r), float(g), float(b)


@dataclass
class StressProbe:
    """Instantaneous structural utilisation of every segment.

    Root bending only.  That is the load case that actually sizes a flapping
    wing and the one the feasibility report checks, and computing it from the
    panel forces already in hand costs nothing extra.  Torsion, buckling and
    joint reactions are not covered here -- they are checked statically at
    Tier 0, and a live version of them would need a structural model this
    project deliberately does not carry.
    """

    env: object

    def __post_init__(self) -> None:
        from ..core.phenotype import tube_wall

        model = self.env.model
        self.section_modulus = {}
        self.allowable = {}
        self.body_of_segment = {}
        self.geoms_of_segment: dict[int, list[int]] = {}

        import mujoco

        for seg in self.env.p.segments:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, seg.name)
            if bid < 0:
                continue
            wall = tube_wall(seg.radius)
            _, _, z = tube_section(2.0 * seg.radius, wall)
            self.section_modulus[seg.index] = max(z, 1e-12)
            self.allowable[seg.index] = seg.material.allowable_stress(cycles=1e5)
            self.body_of_segment[seg.index] = bid
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{seg.name}_g")
            if gid >= 0:
                self.geoms_of_segment.setdefault(seg.index, []).append(gid)

        self.body_to_segment = {b: s for s, b in self.body_of_segment.items()}

    def utilisation(self) -> dict[int, float]:
        """Per-segment stress utilisation, from the panel forces of this step."""
        state = getattr(self.env.solver, "last_state", None)
        if state is None:
            return {}
        pos = state["pos"]
        force = state["force"]
        body_id = state["body_id"]
        xpos = self.env.data.xpos

        out: dict[int, float] = {}
        for seg_index, bid in self.body_of_segment.items():
            sel = body_id == bid
            if not sel.any():
                continue
            arm = pos[sel] - xpos[bid]
            moment = np.cross(arm, force[sel]).sum(axis=0)
            sigma = float(np.linalg.norm(moment)) / self.section_modulus[seg_index]
            out[seg_index] = sigma / max(self.allowable[seg_index], 1.0)
        return out


def _add_wake_geoms(scene, wake, *, max_draw: int = 380, scale: float = 1.0) -> None:
    """Append vortex particles to a rendered scene as coloured spheres."""
    import mujoco

    pos, signed, fade = wake.render_data()
    if len(pos) == 0:
        return
    mag = np.abs(signed)
    peak = float(mag.max())
    if peak <= 1e-12:
        return

    order = np.argsort(-mag)[:max_draw]
    eye = np.eye(3).flatten()
    for i in order:
        if scene.ngeom >= scene.maxgeom:
            break
        strength = mag[i] / peak
        if strength < 0.05:
            continue
        radius = float(np.clip(0.012 + 0.05 * strength * scale, 0.008, 0.09))
        # Red for one sense of rotation, blue for the other: the alternating
        # pattern is the whole point of looking at a wake.
        if signed[i] > 0:
            rgba = np.array([0.90, 0.30, 0.24, 0.30 + 0.55 * fade[i] * strength])
        else:
            rgba = np.array([0.28, 0.55, 0.92, 0.30 + 0.55 * fade[i] * strength])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius]),
            np.asarray(pos[i], dtype=float),
            eye,
            rgba.astype(np.float32),
        )
        scene.ngeom += 1


def _recolour_stress(scene, probe: StressProbe, util: dict[int, float]) -> None:
    """Tint each structural geom by how hard it is working right now."""
    import mujoco

    geom_to_seg = {}
    for seg_index, gids in probe.geoms_of_segment.items():
        for g in gids:
            geom_to_seg[g] = seg_index

    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        seg = geom_to_seg.get(int(g.objid))
        if seg is None:
            continue
        r, gr, b = stress_colour(util.get(seg, 0.0))
        g.rgba[0], g.rgba[1], g.rgba[2] = r, gr, b


def _hud(frame, *, commanded, actual, t, leg, n_legs, depth, speed, power,
         submerged, max_util, label, wake_n):
    """Mission instrument strip: what was asked, what is happening, what it costs."""
    h, w = frame.shape[:2]
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame

    dom_rgb = {"air": (127, 168, 190), "water": (46, 110, 133), "land": (138, 116, 68)}
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    bar = 46
    d.rectangle([0, h - bar, w, h], fill=(8, 14, 18, 214))
    on_task = commanded == actual
    accent = dom_rgb.get(commanded, (200, 200, 200))
    d.rectangle([0, h - bar, w, h - bar + 3], fill=(*accent, 255))

    d.text((10, h - bar + 8),
           f"{label}   leg {leg+1}/{n_legs}   t={t:6.1f}s", fill=(226, 232, 224, 255))
    d.text((10, h - bar + 26),
           f"COMMANDED {commanded.upper():<6s}  ACTUAL {actual.upper():<6s}"
           f"  {'ON TASK' if on_task else 'OFF TASK'}",
           fill=(110, 200, 150, 255) if on_task else (224, 140, 110, 255))
    d.text((w - 300, h - bar + 8),
           f"depth {depth:+6.2f} m   v {speed:5.2f} m/s", fill=(226, 232, 224, 255))
    d.text((w - 300, h - bar + 26),
           f"P {power:6.0f} W   wake {wake_n:3d}   stress {max_util*100:3.0f}%",
           fill=(226, 232, 224, 255))

    # Submersion column.
    gx, gy, gw, gh = w - 26, 14, 12, 92
    d.rectangle([gx, gy, gx + gw, gy + gh], fill=(8, 14, 18, 170))
    fill_h = int(gh * float(np.clip(submerged, 0.0, 1.0)))
    if fill_h:
        d.rectangle([gx, gy + gh - fill_h, gx + gw, gy + gh], fill=(46, 110, 133, 235))
    d.rectangle([gx, gy, gx + gw, gy + gh], outline=(226, 232, 224, 120))

    # Stress bar, with the allowable marked so "past the line" is unambiguous.
    sx, sy, sw, sh = 14, 14, 12, 92
    d.rectangle([sx, sy, sx + sw, sy + sh], fill=(8, 14, 18, 170))
    lvl = int(sh * float(np.clip(max_util, 0.0, 1.0)))
    if lvl:
        r, g_, b = stress_colour(max_util)
        d.rectangle([sx, sy + sh - lvl, sx + sw, sy + sh],
                    fill=(int(r * 255), int(g_ * 255), int(b * 255), 240))
    d.rectangle([sx, sy, sx + sw, sy + sh], outline=(226, 232, 224, 120))
    d.line([sx - 3, sy, sx + sw + 3, sy], fill=(220, 90, 70, 220), width=1)

    # Legend for the wake colours, so red/blue is not a guess.
    d.rectangle([w - 168, h - bar - 26, w - 8, h - bar - 6], fill=(8, 14, 18, 190))
    d.ellipse([w - 162, h - bar - 21, w - 152, h - bar - 11], fill=(230, 77, 61, 255))
    d.text((w - 148, h - bar - 22), "vortex +", fill=(210, 216, 208, 255))
    d.ellipse([w - 86, h - bar - 21, w - 76, h - bar - 11], fill=(71, 140, 235, 255))
    d.text((w - 72, h - bar - 22), "-", fill=(210, 216, 208, 255))
    return np.asarray(img)


def render_mission(
    phenotype,
    controller,
    out_path: str | Path,
    *,
    spec=None,
    leg_seconds: float = 7.0,
    cycles: int = 1,
    seed: int = 0,
    width: int = 900,
    height: int = 560,
    fps: int = 25,
    show_wake: bool = True,
    show_stress: bool = True,
    label: str = "",
):
    """Render one continuous mission with flow and stress overlays.

    Returns ``(path, ContinuousResult)``.
    """
    from ..envs.mission import ContinuousResult, build_schedule, current_domain, run_continuous
    from ..envs.triphibian import Domain, MissionSpec, TriphibianEnv
    from ..physics.wake import VortexWake, attach_wake_probe
    from .render import gl_available

    if gl_available() is None:
        return None, ContinuousResult()

    import imageio.v2 as imageio
    import mujoco

    spec = spec or MissionSpec()
    spec_local = MissionSpec(cycles=cycles, seconds_per_domain=spec.seconds_per_domain,
                             target_depth=spec.target_depth)
    rng = np.random.default_rng(seed)

    env = TriphibianEnv(phenotype, seed=seed, detail=True)
    # Recording is needed for both overlays: the wake sheds from the recorded
    # circulation and the stress probe reads the recorded panel forces.
    attach_wake_probe(env.solver)
    wake = VortexWake() if show_wake else None
    probe = StressProbe(env) if show_stress else None

    # Frame on the geometry, and move the near clip plane in to match.  This
    # world is 186 m across, so MuJoCo's near plane defaults to 3.7 m and any
    # closer camera renders the machine entirely clipped -- an empty seascape
    # with a HUD on it.
    from .render import _geometry_bounds, _near_plane

    env.reset(Domain.AIR, randomise=False)
    _, extent = _geometry_bounds(env)
    _near_plane(env.model, extent).__enter__()
    renderer = mujoco.Renderer(env.model, height=height, width=width, max_geom=6000)
    cam = mujoco.MjvCamera()
    cam.distance = max(extent * 2.0, 0.6)
    cam.elevation = -13.0
    cam.azimuth = 116.0

    frames: list[np.ndarray] = []
    every = max(1, int(1.0 / (fps * env.timestep)))
    n_legs = len(DOMAINS := build_schedule(spec_local, rng, leg_seconds=leg_seconds))

    def on_step(e, commanded, actual, clock, step_i):
        if wake is not None:
            wake.update(e.solver, e.data, e.data.time, e.timestep)
        if step_i % every:
            return
        util = probe.utilisation() if probe is not None else {}
        max_util = max(util.values(), default=0.0)

        cam.lookat[:] = e.root_pos()
        renderer.update_scene(e.data, camera=cam)
        if probe is not None:
            _recolour_stress(renderer.scene, probe, util)
        if wake is not None:
            _add_wake_geoms(renderer.scene, wake)
        img = renderer.render()
        tw = e.body_twist()
        leg_i = min(int(clock // leg_seconds), n_legs - 1)
        frames.append(
            _hud(
                img,
                commanded=commanded.value,
                actual=actual.value,
                t=clock,
                leg=leg_i,
                n_legs=n_legs,
                depth=e.depth(),
                speed=float(np.linalg.norm(tw[:3])),
                power=e.budget.mean_power,
                submerged=e.solver.diag.mean_submerged,
                max_util=max_util,
                label=label or f"{phenotype.mass:.1f}kg {phenotype.max_span:.2f}m",
                wake_n=len(wake.pos) if wake is not None else 0,
            )
        )

    result = run_continuous(
        env, controller, DOMAINS, on_step=on_step,
        energy_scale=spec.seconds_per_domain / max(leg_seconds, 1e-6),
    )
    renderer.close()

    if not frames:
        return None, result
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=None)
    return str(out_path), result
