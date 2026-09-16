"""The real trainer: this project's MAP-Elites design search, behind the port.

The search loop in ``evolution/loop.py`` is not rewritten and must not be.  Its
behaviour is measured -- pool shape, checkpoint cadence, island rotation, the
verification burst in generations 0-5 -- and every number in ``docs/ROADMAP.md``
was produced by it as it stands.  Re-implementing it inside a hexagon would
throw that away to satisfy a diagram.

So this adapter does four things and no arithmetic:

1. translates the plan's hyperparameters into a ``SearchConfig``;
2. hands ``run_search`` a ``should_stop`` that reads the job's stop flag, and an
   ``on_generation`` that turns a generation report into ``context.report``;
3. publishes the run directory's portable checkpoint into the ``CheckpointStore``
   -- only at the generations where the loop itself wrote one, thinned by the
   ``publish_every`` hyperparameter, which counts *checkpoints* and not
   generations -- and restores it on resume;
4. maps how the loop ended onto a ``StopReason``.

One thing is worth stating plainly, because it is the seam where this
architecture touches a measured system.  **The search owns its run directory.**
``context.workspace`` is that directory, the loop writes ``archive_*.pkl``,
``search_state.pkl``, ``events.jsonl`` and the portable checkpoint into it
exactly as it always has, and the report tooling, the dashboard and the showcase
keep working against it unchanged.  What this adapter adds is a *copy* of the
portable checkpoint in the store, digested and provenance-stamped, which is what
makes resume, retention and export answerable without knowing the layout.

A step here is a **generation**.  At this project's measured ~74 s per
generation in steady state, a budget of 900 steps is about 21 hours, and the
opening six-island verification burst costs ~300 s a generation for the first
six.  That is why ``TrainerCapabilities.step_unit`` exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...domain.checkpoint import CheckpointKind, CheckpointPayload, CheckpointRecord
from ...domain.job import StopReason
from ...ports.trainer import TrainerCapabilities, TrainingContext, TrainingOutcome

#: The files that make a search resumable.  ``archive_*.pkl`` is matched by glob
#: because the island set is configurable.  The ``.json`` exports are left out:
#: they are derived from the ``.pkl`` on every checkpoint, so storing them would
#: double the payload to hold a copy of something the loop regenerates.
RESUME_FILES = ("search_state.pkl", "checkpoint.npz", "checkpoint.json")
RESUME_GLOBS = ("archive_*.pkl",)

#: Hyperparameter names accepted from a plan, mapped onto ``SearchConfig``.
#: An explicit list rather than ``setattr`` over whatever arrives: a typo in a
#: plan would otherwise be accepted silently and the run would use the default,
#: which is the failure this project's own notes describe for ``--min-shard``
#: -- a default that was a trap and two generations in three ran under it.
_CONFIG_FIELDS = (
    "batch", "segment_seconds", "tier0_gate", "tier2_every", "tier1_5_seconds",
    "identify_axes_every", "controller_refine_steps", "controller_refine_sigma",
    "promotion_refine_steps", "mission_weight", "reward_shaping",
    "descriptor_bins", "descriptor_refit_every", "learned_axes",
    "migrate_every", "n_migrants", "use_critic", "critic_refit_every",
    "audit_every", "use_scout", "scout_horizon", "scout_reserve",
    "judge_quantile", "judge_update_every", "policy_hidden", "n_modes",
    "use_shared_policy", "shared_hidden", "shared_lr", "shared_epochs",
    "shared_target_kl", "shared_ent_coef", "shared_lr_anneal",
    "shared_minibatch", "audits_per_review", "checkpoint_every",
    "event_sample", "n_reference_seeds", "n_random_seeds", "islands",
)
#: Accepted from the job's ``resources`` rather than its plan, because they are
#: properties of the machine and not of the experiment.  This project measured
#: 4 shards of 4 at 31.7 s against 16x1 at 93.3 s on the same work: pool shape
#: changes wall time by 3x and the result by nothing, so it must not enter the
#: plan digest.
_RESOURCE_FIELDS = ("workers", "min_shard", "memory_ceiling_mb")

#: Mission settings, which are a property of the task rather than of the search.
_SPEC_FIELDS = ("cycles", "seconds_per_domain", "target_depth")


class SearchTrainer:
    """``ports.Trainer`` over ``evolution.loop.run_search``."""

    name = "search"

    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            step_unit="generation",
            cooperative_stop=True,
            checkpoints=True,
            resumes=True,
            exports=True,
            # Claimed, and the claim is narrow: the run's RNG stream, the
            # network and Adam's moments are all in the checkpoint, so a resume
            # continues the same sequence.  What it does not claim is
            # determinism across a different worker count -- the pool shards the
            # batch, and a different shard layout evaluates in a different order.
            deterministic=True,
            description="MAP-Elites over six islands with a shared PPO policy; "
                        "a step is one generation (~74 s measured in steady "
                        "state, ~300 s for generations 0-5)")

    # -- the contract -----------------------------------------------------

    def train(self, context: TrainingContext) -> TrainingOutcome:
        # Imported here, not at module scope: this is where MuJoCo, numpy and
        # -- with a shared policy -- torch are paid for, and naming this trainer
        # in a plan is what should cost that, not importing the package.
        from ...envs.triphibian import MissionSpec
        from ...evolution.loop import run_search

        plan = context.plan
        workspace = Path(context.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        resumed = self._restore(context, workspace)
        cfg = self._config(context, workspace, resume=resumed is not None)
        spec = MissionSpec(**{k: v for k, v in plan.hyperparameters.items()
                              if k in _SPEC_FIELDS})

        # Counted in *checkpoints*, not generations: see ``_Progress``.
        publish_every = int(plan.hyperparameter("publish_every", 1) or 0)
        tracker = _Progress(context, workspace, publish_every=publish_every,
                            checkpoint_every=max(int(cfg.checkpoint_every), 1))

        state = run_search(cfg, spec, on_generation=tracker.on_generation,
                           should_stop=tracker.should_stop)

        return self._outcome(context, state, cfg, tracker, workspace)

    def export(self, record: CheckpointRecord, payload: CheckpointPayload, *,
               destination: str, fmt: str) -> dict:
        """Write the portable checkpoint out as files.

        ``native`` writes ``checkpoint.npz`` and ``checkpoint.json`` into a
        directory, which is what ``ops/checkpoint.read`` takes and what the
        showcase films from.  ``state_dict`` additionally writes the shared
        network's tensors as a torch file.

        There is no "just the model" format, because in this project there is no
        such thing: a design's score was earned by a *body* and a control law
        together, and ``ops/checkpoint.py`` keeps them together for that reason.
        Exporting the network alone would produce a file that loads and means
        nothing.
        """
        if fmt not in ("native", "state_dict"):
            raise ValueError(
                f"{self.name} exports 'native' or 'state_dict', not {fmt!r}")
        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)

        written = 0
        for key in ("checkpoint.npz", "checkpoint.json"):
            blob = payload.get(key)
            if blob is None:
                raise KeyError(
                    f"checkpoint {record.checkpoint_id} has no {key!r}; it "
                    f"holds {list(payload)}")
            (out / key).write_bytes(blob)
            written += len(blob)

        detail: dict[str, Any] = {"bytes_written": written, "format": fmt,
                                  "path": str(out),
                                  "files": ["checkpoint.npz", "checkpoint.json"]}
        if fmt == "state_dict":
            detail.update(self._export_state_dict(out))
        return detail

    @staticmethod
    def _export_state_dict(out: Path) -> dict:
        import torch

        from ...ops import checkpoint as ck

        loaded = ck.read(out)
        net = ck.load_network(loaded)
        if net is None:
            return {"state_dict": None,
                    "note": "this checkpoint carries no shared policy; the run "
                            "was not started with --shared-policy"}
        path = out / "policy.pt"
        torch.save(net.state_dict(), path)
        detail = {"state_dict": str(path),
                  "bytes_written_state_dict": path.stat().st_size}
        return detail

    # -- internals --------------------------------------------------------

    def _config(self, context: TrainingContext, workspace: Path, *, resume: bool):
        from ...evolution.loop import SearchConfig

        plan = context.plan
        known = {f.name for f in SearchConfig.__dataclass_fields__.values()}
        unknown = [k for k in plan.hyperparameters
                   if k not in _CONFIG_FIELDS and k not in _SPEC_FIELDS
                   and k not in ("generations", "publish_every")]
        if unknown:
            raise ValueError(
                f"unknown hyperparameter(s) {sorted(unknown)} for trainer "
                f"{self.name!r}. Accepted: {sorted(set(_CONFIG_FIELDS) | set(_SPEC_FIELDS))}. "
                f"Refusing rather than ignoring them: a typo that falls back to "
                f"a default produces a run that is not the one that was asked "
                f"for and says nothing about it.")

        kwargs = {k: v for k, v in plan.hyperparameters.items()
                  if k in _CONFIG_FIELDS and k in known}
        if "islands" in kwargs and kwargs["islands"] is not None:
            kwargs["islands"] = tuple(kwargs["islands"])

        resources = getattr(context, "resources", None) or {}
        for key in _RESOURCE_FIELDS:
            if key in resources and key in known:
                kwargs[key] = resources[key]

        budget = plan.budget
        generations = plan.hyperparameter("generations")
        if budget.max_steps is not None:
            # The budget wins.  Two places saying how long a run is would
            # eventually disagree, and the budget is the one the job layer
            # enforces and the one a resume extends.
            generations = budget.max_steps
        if generations is None:
            raise ValueError(
                "a search needs a step budget: set TrainingBudget.max_steps "
                "(preferred) or the 'generations' hyperparameter")
        if budget.max_resident_mb is not None:
            kwargs.setdefault("memory_ceiling_mb", budget.max_resident_mb)

        return SearchConfig(generations=int(generations), seed=plan.seed,
                            run_dir=str(workspace), resume=resume, **kwargs)

    def _restore(self, context: TrainingContext, workspace: Path):
        """Unpack a stored checkpoint into the workspace, if there is one.

        Written into the workspace rather than handed to the loop in memory,
        because ``--resume`` reads files and rewriting that would be rewriting
        the measured part.  Anything already in the workspace is overwritten:
        a resume whose files half-came from the store and half from a previous
        attempt is a state nobody can reason about.
        """
        resumed = context.resume_from()
        if resumed is None:
            return None
        record, payload = resumed
        for key, blob in payload.items():
            target = workspace / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        context.note(f"restored {len(payload)} file(s) from "
                     f"{record.checkpoint_id} (step {record.step})",
                     files=sorted(payload), step=record.step)
        return record

    def _outcome(self, context: TrainingContext, state, cfg, tracker,
                 workspace: Path) -> TrainingOutcome:
        stopped_at = getattr(state, "stopped_at", None)
        final_step = stopped_at if stopped_at is not None else max(
            cfg.generations - 1, 0)

        record = tracker.publish(final_step, CheckpointKind.FINAL,
                                 force=True)
        reason = StopReason.COMPLETED
        if tracker.stop is not None:
            reason = tracker.stop.reason
        elif stopped_at is not None:
            # The loop broke on its own memory ceiling: the one stop it decides
            # for itself.  Recorded as a budget stop rather than a completion,
            # because a run that hit a ceiling has generations left to do.
            reason = StopReason.BUDGET_MEMORY

        summary = {
            "generations": final_step + 1,
            "evaluated": int(getattr(state, "evaluated", 0)),
            "tier0_rejected": int(getattr(state, "tier0_rejected", 0)),
            "run_dir": str(workspace),
        }
        if getattr(state, "stop_reason", None):
            summary["loop_stop_reason"] = state.stop_reason
        return TrainingOutcome(
            state=context.state, stop_reason=reason,
            final_checkpoint=record.checkpoint_id if record else None,
            summary=summary)


class _Progress:
    """Turns the loop's per-generation callback into job state.

    Separate from the trainer because it holds the mutable bits -- the pending
    stop, the last published step -- and keeping them off the trainer means the
    trainer instance is reusable and a test can inspect this on its own.
    """

    def __init__(self, context: TrainingContext, workspace: Path, *,
                 publish_every: int, checkpoint_every: int) -> None:
        self._context = context
        self._workspace = workspace
        # In checkpoints, not generations.  The loop writes ``search_state.pkl``
        # only on its own ``checkpoint_every`` cadence, so a publish at any
        # other generation would copy the *previous* checkpoint's files and
        # label them with this generation's step -- a stored checkpoint that
        # claims to be further along than it is, which a resume would believe.
        # Found by running a real four-generation search: generation 0's
        # publish fired before the loop had written anything, and generation
        # 1's would have shipped generation 0's state as step 1.
        self._publish_every = max(0, int(publish_every))
        self._checkpoint_every = max(1, int(checkpoint_every))
        self._checkpoints_seen = 0
        self._last_published = -1
        self.stop = None
        self.generations = 0

    # -- what run_search calls --------------------------------------------

    def on_generation(self, state, report: Mapping[str, Any]) -> None:
        """Progress only.  Called *before* the loop writes its checkpoint."""
        self.generations = int(report.get("generation", 0)) + 1
        step = int(report.get("generation", 0))
        training_state = self._context.state.advanced(
            step=step,
            samples_seen=int(report.get("evaluated", 0)),
            metrics=self._metrics(report),
            # Tagged with the island, because a generation's ``filled``,
            # ``coverage`` and ``qd_score`` belong to whichever island that
            # generation visited.  Consecutive lines are different archives, and
            # a metric without this tag gets plotted as a series -- which
            # CLAUDE.md names as a category error.
            detail={"island": str(report.get("island", "-")),
                    "regime": str(report.get("regime", ""))})
        self._context.report(training_state, metrics=training_state.metrics)

    def should_stop(self, generation: int, report: Mapping[str, Any]):
        """Publish if the loop has just checkpointed, then poll the stop flag.

        This runs *after* the loop's checkpoint block, which is the only point
        at which ``search_state.pkl`` and the archives on disk describe this
        generation.  Publishing anywhere else copies the previous checkpoint
        under this generation's number.
        """
        if self._publish_every and generation % self._checkpoint_every == 0:
            self._checkpoints_seen += 1
            if ((self._checkpoints_seen - 1) % self._publish_every == 0
                    and generation != self._last_published):
                self.publish(generation, CheckpointKind.PERIODIC)

        request = self._context.stop_requested()
        if request is None:
            return None
        self.stop = request
        return str(request)

    # -- publishing --------------------------------------------------------

    def publish(self, step: int, kind: CheckpointKind, *, force: bool = False):
        """Copy the run directory's portable checkpoint into the store.

        Returns None when there is nothing to copy, which is the honest answer
        for a run that has not reached its first checkpoint.  It does not
        fabricate an empty one: a checkpoint that exists and cannot be resumed
        from is worse than no checkpoint, because a resume will accept it.
        """
        payload = self._collect()
        if not payload:
            if force:
                self._context.note(
                    f"no checkpoint files in {self._workspace} at step {step}; "
                    f"this job has nothing to resume from")
            return None
        record = self._context.save_checkpoint(
            payload, step=step, kind=kind,
            provenance={"run_dir": str(self._workspace),
                        "files": sorted(payload)})
        self._last_published = step
        return record

    def _collect(self) -> dict[str, bytes]:
        payload: dict[str, bytes] = {}
        names = list(RESUME_FILES)
        for pattern in RESUME_GLOBS:
            names.extend(p.name for p in sorted(self._workspace.glob(pattern)))
        for name in names:
            path = self._workspace / name
            try:
                payload[name] = path.read_bytes()
            except FileNotFoundError:
                continue
            except OSError as exc:
                self._context.note(f"could not read {name}: {exc}")
        # ``search_state.pkl`` is what ``--resume`` actually reads.  Without it
        # the rest is an archive and not a resumable run, so publishing the rest
        # under a name that promises resumability would be a lie the resume
        # discovers hours later.
        if "search_state.pkl" not in payload:
            return {}
        return payload

    @staticmethod
    def _metrics(report: Mapping[str, Any]) -> dict[str, float]:
        """The scalars worth keeping as a series, and only those.

        Chosen rather than "everything numeric in the report", because the
        report carries per-island quantities whose meaning depends on which
        island the generation visited, and a sink that takes them all produces
        charts that interleave six archives.  The island is on the tag; these
        are the ones that are either global or safe to read per island.
        """
        out: dict[str, float] = {}
        for key in ("filled", "coverage", "qd_score", "best_fitness",
                    "mission_best", "mission_corr", "evaluated", "elapsed",
                    "rollouts", "diverged_rollouts"):
            value = report.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        for group, keys in (("curriculum", ("typical", "reached")),
                            ("critic", ("calibration",)),
                            ("scout", ("calibration", "protected", "depth_mean")),
                            ("auditor", ("invalidated",))):
            sub = report.get(group) or {}
            for key in keys:
                value = sub.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[f"{group}.{key}"] = float(value)
        return out
