"""A run's finished state as one portable artefact.

What a run already kept was the *machine* and not the *experiment*.  The genome,
each elite's own policy weights and the shared network's weights were all on
disk -- including the network's observation normaliser, which is a
``register_buffer`` and so travels in ``state_dict`` -- and four things were not:

* **the optimiser.**  ``save_state`` wrote the network and no Adam state, so a
  resumed run rebuilt the moments from scratch and its first updates behaved
  like the start of training.
* **the run's random stream.**  Every candidate's evaluation seed is drawn from
  it, so a resumed run draws a different one and no score survives the boundary.
  arch38's archived ``takeoff_height`` of 2.288 m re-measured as 0.000 for
  exactly this reason, while ``max_depth`` -- which barely depends on the initial
  condition -- came back at 9.51-9.69 against 9.60.
* **the mobility basis**, which every consumer re-identified instead, using a
  seed that had not been kept.
* **the code that wrote it.**  A checkpoint that cannot be matched to a commit
  cannot be trusted to mean what it meant.

And ``search_state.pkl`` pickles live ``Judge``, ``Curator``, ``Auditor`` and
``Curriculum`` objects, so it is hostage to this package's class layout -- not
only to the torch version its own comment worries about.  It stays, because
``--resume`` needs the learned bookkeeping; this is the artefact for everything
else.

Two files, neither of which needs to unpickle a project class:

``checkpoint.npz``
    every array -- network tensors, Adam moments, the chosen elites' policy
    weights and mobility bases -- under ``/``-joined names.
``checkpoint.json``
    every scalar and structure -- the genomes (plain nested dicts, because
    ``Genome``, ``Part``, ``Edge`` and ``CPPN`` are dataclasses of scalars), the
    RNG state, the optimiser's parameter groups, the config, the provenance.

Three uses, which is what "checkpoint" has to mean here:

* **use it directly** -- ``read()`` returns the network and the elites, and the
  showcase films from them without touching ``search_state.pkl``;
* **fine-tune** -- the network and its Adam moments come back together, so
  training continues rather than restarts;
* **continue the search** -- the RNG state comes back too, so the seed stream is
  the same one.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bumped whenever the layout changes in a way a reader has to know about.
#: `read` refuses a schema it was not written for rather than guessing, because
#: silently misreading a checkpoint is worse than not reading one.
SCHEMA = 1

ARRAYS = "checkpoint.npz"
META = "checkpoint.json"


def _plain(x):
    """Anything numpy, as something ``json`` will take."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _genome_dict(genome) -> dict | None:
    """A genome as a nested plain dict.

    ``Genome`` holds lists of ``Part``, ``Edge`` and ``CPPN``, all dataclasses of
    scalars and strings apart from one ``joint_axis`` array, so ``asdict`` plus
    `_plain` is a complete and portable representation.  Returns None rather than
    raising if that ever stops being true -- a checkpoint missing its genome is
    still worth having for the network.
    """
    try:
        return _plain(dataclasses.asdict(genome))
    except Exception:                                             # noqa: BLE001
        return None


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                             # noqa: BLE001
        return ""


def _hidden_of(state_dict) -> int:
    """The trunk width, from the stored tensors.

    Inferred rather than taken from ``--shared-hidden``, which is not an argument
    of every subcommand that needs to rebuild the network, and which would load
    the weights into the wrong shape if it disagreed.
    """
    for v in state_dict.values():
        a = np.asarray(v)
        if a.ndim == 2:
            return int(a.shape[0])
    return 64


# --------------------------------------------------------------------------
# writing


def _best(archipelago, key: str):
    """The elite the whole archipelago ranks first on ``key``.

    Both rankings are kept because they disagree: arch34 ended with
    corr(fitness, mission) = +0.70 and the fitness-best elite walked while the
    run's actual best flew, which is why `showcase` grew a `--by` at all.
    """
    best, best_v = None, None
    for a in archipelago.archives.values():
        for e in a.cells.values():
            v = (e.fitness if key == "fitness"
                 else (e.meta or {}).get("mission_fraction") or 0.0)
            if best_v is None or float(v) > float(best_v):
                best, best_v = e, v
    return best


def _elite_payload(elite, tag: str, arrays: dict) -> dict:
    """One elite's arrays into ``arrays``, and its scalars as the return."""
    meta = dict(elite.meta or {})
    w = meta.get("policy")
    if w is not None:
        arrays[f"elite/{tag}/policy"] = np.asarray(w, float)
    for dom, b in (meta.get("mobility_basis") or {}).items():
        for field in ("modes", "effects", "authority"):
            if b.get(field) is not None:
                arrays[f"elite/{tag}/basis/{dom}/{field}"] = np.asarray(
                    b[field], float)
    return {
        "genome": _genome_dict(elite.genome),
        "fitness": float(elite.fitness),
        "tier": int(elite.tier),
        "island": meta.get("island"),
        "mission_fraction": meta.get("mission_fraction"),
        "eval_seed": meta.get("eval_seed"),
        "gen": meta.get("gen"),
        "rungs": _plain(meta.get("rungs")),
        "ladder_measurements": _plain(meta.get("ladder_measurements")),
        "basis_media": sorted((meta.get("mobility_basis") or {}).keys()),
        "has_policy": w is not None,
    }


def write(state, gen: int) -> Path | None:
    """Write the portable checkpoint beside the archives.  Returns its path.

    Called from ``save_state``, so it lands wherever the rolling checkpoint
    does and at the same cadence.  Returns None when there is nothing portable
    to write.
    """
    run_dir = Path(state.config.run_dir)
    arrays: dict = {}
    meta: dict = {"schema": SCHEMA}

    if state.shared is not None:
        sd = {k: v.detach().cpu().numpy()
              for k, v in state.shared.state_dict().items()}
        for k, v in sd.items():
            arrays[f"net/{k}"] = v
        meta["net"] = {"n_obs": int(state.shared.n_obs),
                       "n_modes": int(state.shared.n_modes),
                       "hidden": _hidden_of(sd),
                       "keys": sorted(sd)}

    # Adam's moments, per parameter, plus the groups they belong to.  This is
    # the piece whose absence made "continue training" mean "start training with
    # a warm network", which is a different thing and was invisible.
    if state.shared_opt is not None:
        try:
            osd = state.shared_opt.state_dict()
            entries = []
            for pid, st in (osd.get("state") or {}).items():
                names = []
                for k, v in st.items():
                    a = (v.detach().cpu().numpy() if hasattr(v, "detach")
                         else np.asarray(v))
                    arrays[f"opt/{pid}/{k}"] = a
                    names.append(k)
                entries.append({"param": int(pid), "keys": sorted(names)})
            meta["opt"] = {"entries": entries,
                           "param_groups": _plain(osd.get("param_groups"))}
        except Exception as exc:                                  # noqa: BLE001
            meta["opt_error"] = f"{type(exc).__name__}: {exc}"

    # The stream every evaluation seed is drawn from.  `bit_generator.state` is
    # already a plain dict of ints and arrays.
    try:
        meta["rng"] = _plain(state.rng.bit_generator.state)
    except Exception as exc:                                      # noqa: BLE001
        meta["rng_error"] = f"{type(exc).__name__}: {exc}"

    meta["elites"] = {}
    if state.archipelago is not None:
        for tag, key in (("mission", "mission"), ("fitness", "fitness")):
            e = _best(state.archipelago, key)
            if e is not None:
                meta["elites"][tag] = _elite_payload(e, tag, arrays)

    import numpy as _np
    prov = {"generation": int(gen), "evaluated": int(state.evaluated),
            "git": _git_sha(), "numpy": _np.__version__}
    try:
        import torch as _torch
        prov["torch"] = _torch.__version__
    except Exception:                                             # noqa: BLE001
        prov["torch"] = ""
    try:
        prov["config"] = _plain(dict(state.config.__dict__))
    except Exception:                                             # noqa: BLE001
        prov["config"] = {}
    # The ladder this score was produced under, so a reader can tell whether a
    # stored number means what its name means now.  arch38's water ratchet
    # tracked `max_depth` and the ladder was rebuilt on `depth_gain`; both are
    # published, so nothing would have errored.
    try:
        from ..evolution.judge import LADDER
        prov["ladder"] = {d: [[n, m, float(t)] for n, m, t in rungs]
                          for d, rungs in LADDER.items()}
    except Exception:                                             # noqa: BLE001
        pass
    meta["provenance"] = prov

    if not arrays and not meta.get("elites"):
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    # Written through a handle, not a path: `savez_compressed` appends `.npz` to
    # any path that does not already end in it, so a `checkpoint.npz.tmp`
    # argument produced `checkpoint.npz.tmp.npz` and the atomic rename then had
    # nothing to rename.  A handle is taken as-is.
    a_tmp = run_dir / (ARRAYS + ".tmp")
    with open(a_tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    a_tmp.replace(run_dir / ARRAYS)
    m_tmp = run_dir / (META + ".tmp")
    m_tmp.write_text(json.dumps(meta, indent=1))
    m_tmp.replace(run_dir / META)
    return run_dir / ARRAYS


# --------------------------------------------------------------------------
# reading


# `eq=False`, as everything in this package that can reach an array is: a
# generated `__eq__` compares the arrays elementwise and raises on the truth
# value of the result.  `tests/test_search.py` scans for this.
@dataclass(eq=False)
class Elite:
    """One stored design, with everything needed to drive it."""

    genome: dict | None
    policy: np.ndarray | None
    bases: dict            # medium -> {"modes", "effects", "authority"}
    info: dict

    @property
    def eval_seed(self):
        return self.info.get("eval_seed")


@dataclass(eq=False)
class Checkpoint:
    """A run's finished state, read back without unpickling anything."""

    path: Path
    meta: dict
    arrays: dict

    @property
    def generation(self) -> int:
        return int((self.meta.get("provenance") or {}).get("generation", 0))

    def net_state(self) -> dict:
        """The shared network's ``state_dict``, as numpy arrays."""
        return {k[len("net/"):]: v for k, v in self.arrays.items()
                if k.startswith("net/")}

    def net_shape(self) -> tuple:
        n = self.meta.get("net") or {}
        return (int(n.get("n_obs", 0)), int(n.get("n_modes", 0)),
                int(n.get("hidden", 64)))

    def optimiser_state(self) -> dict | None:
        """Adam's state, in the shape ``load_state_dict`` wants."""
        spec = self.meta.get("opt")
        if not spec:
            return None
        state = {}
        for entry in spec.get("entries", []):
            pid = int(entry["param"])
            state[pid] = {k: self.arrays[f"opt/{pid}/{k}"]
                          for k in entry.get("keys", [])
                          if f"opt/{pid}/{k}" in self.arrays}
        return {"state": state, "param_groups": spec.get("param_groups") or []}

    def rng_state(self) -> dict | None:
        return self.meta.get("rng")

    def elite(self, by: str = "mission") -> Elite | None:
        info = (self.meta.get("elites") or {}).get(by)
        if info is None:
            return None
        bases = {}
        prefix = f"elite/{by}/basis/"
        for k, v in self.arrays.items():
            if not k.startswith(prefix):
                continue
            dom, field = k[len(prefix):].split("/", 1)
            bases.setdefault(dom, {})[field] = v
        return Elite(genome=info.get("genome"),
                     policy=self.arrays.get(f"elite/{by}/policy"),
                     bases=bases, info=info)


def read(where: str | Path) -> Checkpoint:
    """Load a checkpoint from a run directory or from the ``.npz`` itself."""
    p = Path(where)
    run_dir = p if p.is_dir() else p.parent
    a_path, m_path = run_dir / ARRAYS, run_dir / META
    if not a_path.exists() or not m_path.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    meta = json.loads(m_path.read_text())
    got = int(meta.get("schema", -1))
    if got != SCHEMA:
        raise ValueError(
            f"{m_path} is schema {got}, this reader is {SCHEMA}. Refusing "
            "rather than guessing at a layout it was not written for.")
    with np.load(a_path) as z:
        arrays = {k: z[k] for k in z.files}
    return Checkpoint(path=a_path, meta=meta, arrays=arrays)


def load_network(ck: Checkpoint):
    """Rebuild the shared policy from a checkpoint, weights loaded.

    Returns ``(net, missing, unexpected)``; raises if torch is unavailable or
    the checkpoint carries no network.
    """
    import torch

    from ..learning import ppo as _ppo

    sd = ck.net_state()
    if not sd:
        raise ValueError(f"{ck.path} carries no network")
    n_obs, n_modes, hidden = ck.net_shape()
    net = _ppo.SharedPolicy(n_obs, n_modes, hidden=hidden)
    missing, unexpected = net.load_state_dict(
        {k: torch.as_tensor(v) for k, v in sd.items()}, strict=False)
    net.eval()
    return net, list(missing), list(unexpected)
