"""Film what was scored, beside what the mission does.

Why this exists
---------------

On 2026-09-21 the user watched arch40's film and saw a machine "only trembling
in place", and every judgement drawn from it was wrong in the same direction.
ROADMAP item Y has the measurement: the film was the *continuous mission*,
which starts on the beach, and the machine it showed was a water glider that
cannot leave a beach.  Its score was earned somewhere else entirely -- released
four metres under water, where it glides seven metres in eight seconds.

Nothing the project filmed was the experiment that produced a score:

* ``showcase`` films the continuous mission, which no score reads;
* ``render --top`` drove each elite open-loop, with no policy and no basis, at
  seed 0, without the evaluation's scatter, for 10 s instead of 8, and printed
  no number, so nobody could see that it was a different experiment.

Three rules, each enforced here rather than written down:

1. **The evaluated clips are the evaluation.**  They come from
   ``evaluate_tier1`` itself -- the function that produces a Tier-1 score --
   with a camera attached through ``on_step``: the same body, the same control
   law (``controller_for_elite``), the same seed, scatter, task and segment
   length.  Not a second rollout meant to resemble it.
2. **Every clip carries its own proof.**  The competence the search recorded
   and the competence this very rollout measured are both stamped on it.  If
   they differ by more than ``TOLERANCE`` the clip says, in red, that it is not
   the scored experiment.
3. **The continuous mission is never shown alone.**  Each of its legs is put
   beside the evaluated clip of the same medium and labelled as what it is.

``film_run`` is what a finished run calls on its own (``ops.run postrun``), so
none of this depends on anyone remembering to ask for it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

#: How far the filmed rollout's competence may sit from the recorded one before
#: the clip is declared a different experiment.  The recorded number comes from
#: the batched path and the film from the single-machine one; `test_search`
#: holds those two to floating point in land and air.  0.02 is a band that a
#: real divergence -- open-loop driving, a different seed, a re-identified
#: basis -- clears by an order of magnitude (measured on arch40's filmed elite:
#: open-loop water 0.090 against 0.771 recorded).
TOLERANCE = 0.02

MEDIA = ("air", "water", "land")
_RED = (214, 64, 52)
_GREEN = (46, 160, 90)
_AMBER = (224, 170, 40)


def run_provenance(run_dir: Path) -> dict:
    """The commit and configuration a run's checkpoint recorded, if any."""
    p = Path(run_dir) / "checkpoint.json"
    try:
        return json.loads(p.read_text()).get("provenance") or {}
    except Exception:
        return {}


_ROOT = Path(__file__).resolve().parents[2]


def current_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_ROOT, check=True).stdout.strip()
    except Exception:
        return ""


def code_changed_since(commit: str | None) -> list[str] | None:
    """Source files under ``dytiscidae/`` that differ from ``commit``.

    Compared against the working tree, so an uncommitted edit counts.  None when
    the commit is unknown or git cannot answer.  A docs-only commit after a run
    is not a reason to doubt its footage; a change to the package is -- though
    the per-medium score comparison is what decides, and this only says why a
    mismatch might have happened.
    """
    if not commit:
        return None
    try:
        out = subprocess.run(["git", "diff", "--name-only", commit, "--", "dytiscidae/"],
                             capture_output=True, text=True, cwd=_ROOT, check=True).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
    except Exception:
        return None


def pick_elite(run_dir, *, by: str = "mission", island: str | None = None):
    """The elite the film is of, chosen the way ``showcase`` chooses."""
    from ..ops.run import load_run_archive

    archive, _islands = load_run_archive(str(run_dir), island=island)
    if archive is None or not archive.cells:
        return None
    pool = list(archive.cells.values())
    if by == "mission":
        key = lambda e: (e.meta or {}).get("mission_fraction") or 0.0
    elif by == "island":
        from ..evolution.islands import own_domain_score

        def key(e):
            meta = e.meta or {}
            return own_domain_score(island or meta.get("island", ""), meta)
    else:
        key = lambda e: e.fitness
    return max(pool, key=key)


# --------------------------------------------------------------------------
# The evaluated clips
# --------------------------------------------------------------------------


class _Camera:
    """A follow camera attached to whichever env the evaluation builds."""

    def __init__(self, width: int, height: int, fps: int):
        self.width, self.height, self.fps = width, height, fps
        self.frames: dict[str, list] = {m: [] for m in MEDIA}
        self._renderer = None
        self._model = None
        self._cam = None
        self._near = None
        self.ok = True

    def _attach(self, env):
        import mujoco

        from .render import _geometry_bounds, _near_plane

        self.close()
        _, extent = _geometry_bounds(env)
        self._cam = mujoco.MjvCamera()
        self._cam.distance = max(extent * 1.8, 0.8)
        self._cam.elevation = -14.0
        self._cam.azimuth = 118.0
        self._near = _near_plane(env.model, extent)
        self._near.__enter__()
        self._renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)
        self._model = env.model

    def __call__(self, dom, env, i):
        if not self.ok:
            return
        every = max(1, int(round(1.0 / (self.fps * env.timestep))))
        if i % every:
            return
        try:
            if self._model is not env.model:
                self._attach(env)
            from .render import _hud

            self._cam.lookat[:] = env.root_pos()
            self._renderer.update_scene(env.data, camera=self._cam)
            img = self._renderer.render()
            tw = env.body_twist()
            self.frames[dom.value].append(_hud(
                img, domain=dom.value, t=float(env.data.time), depth=float(env.depth()),
                speed=float(np.linalg.norm(tw[:3])), power=float(env.budget.mean_power),
                submerged=float(env.solver.diag.mean_submerged), label="as evaluated"))
        except Exception as exc:                              # noqa: BLE001
            # A camera that cannot render must not change the measurement:
            # stop filming, keep evaluating.
            self.ok = False
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._near is not None:
            self._near.__exit__(None, None, None)
            self._near = None


def _load_network(path: Path, like):
    """A shared network with ``like``'s shape and the state stored at ``path``."""
    if like is None or not Path(path).exists():
        return None
    import torch

    hid = next((int(v.shape[0]) for v in like.state_dict().values() if v.ndim == 2), 64)
    net = type(like)(like.n_obs, like.n_modes, hidden=hid)
    net.load_state_dict({k: torch.as_tensor(v) for k, v in np.load(path).items()})
    net.eval()
    return net


def control_laws(elite, run_dir, shared):
    """The control laws the elite's score may have been earned under, best-known first.

    Yields ``(description, shared_network_or_None)``.  A record that says what
    scored it gets exactly that.  A legacy record does not, and for it the
    order encodes what is known about the legacy code: until 2026-09-21 the
    actor pool dropped the shared policy from every re-score on a generation
    split into more than one shard -- which on ``--workers 4 --min-shard 4`` is
    nearly every generation -- so "own policy alone" comes first.
    """
    meta = elite.meta or {}
    gen = int(meta.get("gen") or 0)
    flag = meta.get("scored_with_shared_policy")
    run_dir = Path(run_dir)
    if flag is False or shared is None:
        yield "own policy alone, as recorded", None
        return
    if flag is True:
        net = _load_network(run_dir / "scoring_networks" / f"gen{gen:05d}.npz", shared)
        if net is not None:
            yield f"own + shared policy, the network that scored gen {gen}", net
        else:
            yield (f"own + shared policy -- the network that scored gen {gen} was not "
                   f"kept; final network"), shared
        return
    yield ("own policy only (legacy run: its actor pool dropped the shared policy "
           "from re-scores)"), None
    snaps = sorted((int(q.stem[3:]), q) for q in (run_dir / "policy_snapshots").glob("gen*.npz"))
    before = [q for g, q in snaps if g < gen]
    if before:
        yield (f"own + shared policy, nearest earlier snapshot "
               f"({before[-1].stem})"), _load_network(before[-1], shared)
    yield "own + shared policy, the run's final network", shared


def evaluate_on_film(elite, run_dir, *, film: bool = True, width: int = 640,
                     height: int = 400, fps: int = 25, log=print) -> dict:
    """Re-run the elite's Tier-1 evaluation, filming it, and compare the scores.

    Tries the candidate control laws (``control_laws``) without a camera until
    one reproduces every medium's recorded competence within ``TOLERANCE``,
    then films that one.  If none does, it films the best-known law and every
    clip says it is not the scored experiment.  With ``film=False`` it is the
    same-experiment check alone, which is what the test suite runs.
    """
    from ..core.phenotype import build
    from ..envs.evaluate import evaluate_tier1
    from ..envs.triphibian import Domain
    from ..ops.run import controller_for_elite
    from .render import gl_available

    meta = elite.meta or {}
    p = build(elite.genome)
    seed = int(meta.get("eval_seed") or 0)
    prov = run_provenance(Path(run_dir))
    seg_s = float((prov.get("config") or {}).get("segment_seconds") or 8.0)
    ctrl = controller_for_elite(str(run_dir), elite, p, seed, log=lambda *a, **k: None)
    stored = sorted((meta.get("mobility_basis") or {}).keys())
    summed = getattr(ctrl, "policy", None) if ctrl is not None else None
    shared = getattr(summed, "shared", None)

    def record(result):
        out = {}
        for m in MEDIA:
            seg = result.segments.get(Domain(m))
            rec = meta.get(m)
            got = None if seg is None else float(seg.competence)
            out[m] = {"recorded": None if rec is None else float(rec), "reproduced": got,
                      "match": bool(rec is not None and got is not None
                                    and abs(float(rec) - got) <= TOLERANCE)}
        return out

    def run(net, cam=None):
        if summed is not None and hasattr(summed, "shared"):
            summed.shared = net
        return evaluate_tier1(p, controller=ctrl, segment_seconds=seg_s, seed=seed,
                              identify_axes=False, detail=cam is not None, on_step=cam)

    attempts, chosen = [], None
    for desc, net in control_laws(elite, run_dir, shared):
        res = run(net)
        rec = record(res)
        attempts.append({"control_law": desc,
                         "media": {m: dict(rec[m]) for m in MEDIA}})
        if all(rec[m]["match"] for m in MEDIA):
            chosen = (desc, net, rec, res)
            break
    if chosen is None:
        desc, net = next(iter(control_laws(elite, run_dir, shared)))
        res = run(net)
        chosen = (desc, net, record(res), res)
    desc, net, rec, result = chosen

    cam = _Camera(width, height, fps) if (film and gl_available() is not None) else None
    if cam is not None:
        try:
            result = run(net, cam)
        finally:
            cam.close()
        rec = record(result)          # the filmed rollout's own numbers

    media = {m: {**rec[m], "frames": (cam.frames[m] if cam is not None else [])}
             for m in MEDIA}
    log(f"  control law: {desc}")
    return {
        "result": result, "media": media, "seed": seed, "segment_seconds": seg_s,
        "bases_stored": stored, "controller": ctrl is not None,
        "control_law": desc, "attempts": attempts,
        "film_error": getattr(cam, "error", None) if cam is not None else
                      ("no GL context" if film else None),
    }


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def _banner(frame, lines, colour, width):
    """A title block above a frame: ``lines`` of text on a coloured rule."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(frame)).convert("RGB")
    if img.width != width:
        img = img.resize((width, int(round(img.height * width / img.width))))
    bar = 20 * len(lines) + 12
    out = Image.new("RGB", (width, img.height + bar), (12, 16, 20))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.rectangle([0, bar - 4, width, bar - 1], fill=colour)
    for k, text in enumerate(lines):
        d.text((10, 6 + 20 * k), text, fill=(232, 236, 230) if k else colour)
    return np.asarray(out)


def _pad(frames, n):
    if not frames:
        return []
    return list(frames[:n]) + [frames[-1]] * max(0, n - len(frames))


def _write(frames, path: Path, fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    imageio.mimsave(str(path), [f[:h, :w] for f in frames], fps=fps, macro_block_size=None)
    return str(path)


def _verdict(rec):
    if rec["reproduced"] is None or rec["recorded"] is None:
        return ("NOT MEASURED -- this medium produced no segment", _AMBER)
    if rec["match"]:
        return (f"recorded {rec['recorded']:.3f}  this clip {rec['reproduced']:.3f}  "
                f"= the scored experiment", _GREEN)
    return (f"recorded {rec['recorded']:.3f}  this clip {rec['reproduced']:.3f}  "
            f"-- NOT THE SCORED EXPERIMENT (diff {abs(rec['recorded'] - rec['reproduced']):.3f})",
            _RED)


def film_run(run_dir, *, by: str = "mission", island: str | None = None,
             out_dir=None, fps: int = 25, panel: int = 640, log=print) -> dict:
    """Film a finished run: evaluated clips, the continuous mission, and both side by side.

    Writes into ``<run>/media/``:

    * ``evaluated_{air,water,land}.mp4`` -- the Tier-1 evaluation, filmed;
    * ``mission.mp4`` -- the continuous mission, which starts on the beach;
    * ``evaluated_vs_mission.mp4`` -- the headline: each mission leg beside
      the evaluated clip of the same medium, both labelled;
    * ``film_manifest.json`` -- every number on the clips, and the verdicts.

    Returns the manifest.  Without a GL context the numbers are still measured
    and written, and the manifest says there is no footage.
    """
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "media"
    out.mkdir(parents=True, exist_ok=True)
    elite = pick_elite(run_dir, by=by, island=island)
    if elite is None:
        log(f"film: no archive in {run_dir}")
        return {}
    meta = elite.meta or {}
    prov = run_provenance(run_dir)
    head = current_commit()
    changed = code_changed_since(prov.get("git"))
    same_code = changed == []

    log(f"film: {run_dir.name}, the best elite by {by}: island {meta.get('island')}, "
        f"mission {meta.get('mission_fraction') or 0:.4f}, fitness {elite.fitness:.4f}")
    ev = evaluate_on_film(elite, run_dir, fps=fps, width=panel, height=int(panel * 0.625), log=log)
    for m in MEDIA:
        r = ev["media"][m]
        log(f"  {m:5s} recorded {r['recorded'] if r['recorded'] is not None else float('nan'):.3f}  "
            f"reproduced {r['reproduced'] if r['reproduced'] is not None else float('nan'):.3f}  "
            f"{'same experiment' if r['match'] else 'NOT THE SAME EXPERIMENT'}")

    who = f"{run_dir.name} | {meta.get('island', '?')} island | eval seed {ev['seed']}"
    law = f"control: {ev['control_law']}"
    code = [] if same_code else ["code: the package changed since the run's commit "
                                 "(the score check above decides)"]
    evaluated_paths = {}
    for m in MEDIA:
        r = ev["media"][m]
        text, colour = _verdict(r)
        frames = [_banner(f, [f"SCORED -- {m}, as evaluated ({ev['segment_seconds']:.0f} s "
                              f"from the evaluation's own start)", text, who, law, *code],
                          colour, panel) for f in r["frames"]]
        evaluated_paths[m] = _write(frames, out / f"evaluated_{m}.mp4", fps)

    # The continuous mission, from the showcase's own renderer, legs in the
    # order it flew them.
    mission_info, mission_path, legs = {}, None, []
    try:
        from ..core.phenotype import build
        from ..ops.run import controller_for_elite
        from .showcase import render_mission

        p = build(elite.genome)
        ctrl = controller_for_elite(str(run_dir), elite, p, ev["seed"], log=lambda *a, **k: None)
        # The same control law the evaluated clips reproduced the score with.
        want = ev["control_law"]
        for desc, net in control_laws(elite, run_dir, getattr(ctrl.policy, "shared", None)):
            if desc == want:
                if hasattr(ctrl.policy, "shared"):
                    ctrl.policy.shared = net
                break
        mission_path, mission = render_mission(p, ctrl, out / "mission.mp4",
                                               leg_seconds=ev["segment_seconds"], cycles=1,
                                               seed=0, fps=fps)
        legs = list(getattr(mission, "legs", []) or [])
        mission_info = {
            "on_task": float(getattr(mission, "on_task", 0.0) or 0.0),
            "transitions": f"{getattr(mission, 'transitions_completed', 0)}/"
                           f"{getattr(mission, 'transitions_commanded', 0)}",
            "max_depth": float(getattr(mission, "max_depth", 0.0) or 0.0),
            "legs": [{"medium": leg.commanded.value,
                      "on_task": float(leg.on_task_fraction)} for leg in legs],
        }
    except Exception as exc:                                  # noqa: BLE001
        mission_info = {"error": f"{type(exc).__name__}: {exc}"}

    composite = None
    if mission_path and legs and any(ev["media"][m]["frames"] for m in MEDIA):
        import imageio.v2 as imageio

        mframes = list(imageio.mimread(mission_path, memtest=False))
        per_leg = max(1, int(round(ev["segment_seconds"] * fps)))
        rows = []
        for k, leg in enumerate(legs):
            m = leg.commanded.value
            left = ev["media"][m]["frames"]
            right = mframes[k * per_leg:(k + 1) * per_leg]
            n = max(len(left), len(right))
            text, colour = _verdict(ev["media"][m])
            lt = [_banner(f, [f"SCORED -- {m}, as evaluated", text, who, law, *code],
                          colour, panel) for f in _pad(left, n)]
            rt = [_banner(f, [f"CONTINUOUS MISSION -- leg {k + 1}/{len(legs)}, commanded {m}",
                              f"on-task {100 * leg.on_task_fraction:.0f}% -- starts on the "
                              f"beach and is never re-placed",
                              "this is NOT what was scored: a machine that cannot reach",
                              "a medium is shown failing to reach it",
                              *([""] * len(code))],
                          _AMBER, panel) for f in _pad(right, n)]
            for a, b in zip(lt, rt):
                h = min(a.shape[0], b.shape[0])
                rows.append(np.concatenate([a[:h], b[:h]], axis=1))
        composite = _write(rows, out / "evaluated_vs_mission.mp4", fps)

    manifest = {
        "run": run_dir.name, "by": by, "island": meta.get("island"),
        "design": {k: meta.get(k) for k in ("body_plan", "n_parts", "dof", "mass", "span",
                                            "density_ratio", "mission_fraction")},
        "fitness": float(elite.fitness), "eval_seed": ev["seed"],
        "segment_seconds": ev["segment_seconds"], "bases_stored": ev["bases_stored"],
        "control_law": ev["control_law"], "attempts": ev["attempts"],
        "run_commit": prov.get("git"), "film_commit": head, "same_code": same_code,
        "code_changed_since_run": changed,
        "tolerance": TOLERANCE,
        "media": {m: {k: v for k, v in ev["media"][m].items() if k != "frames"} for m in MEDIA},
        "all_match": all(ev["media"][m]["match"] for m in MEDIA),
        "mission": mission_info, "film_error": ev["film_error"],
        "paths": {"evaluated": evaluated_paths, "mission": mission_path,
                  "composite": composite},
    }
    (out / "film_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log(f"  composite: {composite or 'not written (' + str(ev['film_error'] or 'no frames') + ')'}")
    if not manifest["all_match"]:
        log("  WARNING: at least one evaluated clip does not reproduce its recorded score -- "
            "it is marked on the clip; do not read the footage as evidence for that medium")
    return manifest
