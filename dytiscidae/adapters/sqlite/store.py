"""SQLite adapters for ``ExperimentStore`` and ``JobStore``.

The filesystem adapters are the default and this is the alternative, so the
honest question is what it buys, measured rather than assumed.

What it buys: **queries that the directory layout answers by scanning.**
"every failed job across every experiment", "experiments tagged `air` with a
job that got past step 500" -- those are one statement here and a walk over
every ``job.json`` there.  And a real transaction, so ``attach_job`` and the
job's creation either both happen or neither does.

What it costs: a file that one process writes at a time.  SQLite's WAL mode
allows concurrent readers with one writer, which is the shape this has -- one
worker writing, everything else reading -- but a second worker on the same
database will block, where two workers on the filesystem store only contend on
their own job's lock.

So the rule this file is built to: **SQLite holds the record and the
lifecycle; it never holds a checkpoint.**  Checkpoints are megabytes of array
data, and putting them in a row turns every backup of the metadata into a copy
of every network the run ever wrote.  ``FileCheckpointStore`` keeps them, and
this store keeps the rows that point at them.

The schema is versioned in ``schema_version`` and migrated forward on open.  A
database from a newer version is refused rather than read, because a reader
guessing at a column it does not know about is how a record comes to say
something its writer did not mean.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

from ...domain.errors import ExperimentNotFound, JobNotFound
from ...domain.experiment import Experiment
from ...domain.ids import ExperimentId, JobId
from ...domain.job import JobStatus, TrainingJob
from ...domain.state import TrainingState

#: Bumped whenever a migration is added.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    plan_digest   TEXT NOT NULL,
    trainer       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    tags          TEXT NOT NULL DEFAULT '',
    document      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    status        TEXT NOT NULL,
    plan_digest   TEXT NOT NULL,
    trainer       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    started_at    REAL,
    ended_at      REAL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    workspace     TEXT NOT NULL DEFAULT '',
    document      TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS jobs_by_experiment ON jobs(experiment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_by_status     ON jobs(status, created_at DESC);

-- Runtime state, one row per job, overwritten every step.  Separate from
-- ``jobs`` so that the hot write does not touch the row every listing reads.
CREATE TABLE IF NOT EXISTS job_state (
    job_id     TEXT PRIMARY KEY,
    step       INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    document   TEXT NOT NULL
);
"""

#: Columns lifted out of the JSON document so they can be indexed and queried.
#: The document remains the truth -- these are projections of it, rewritten
#: whenever it is, never edited on their own.  A projection that can be edited
#: separately is a second truth, and the two will disagree.
_PROJECTED = ("status", "plan_digest", "trainer", "created_at", "started_at",
              "ended_at", "attempts", "workspace")


class SqliteStore:
    """Connection and schema management, shared by the two stores below."""

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=timeout,
                                     isolation_level=None,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: readers do not block the writer, which is what a status query
        # against a running job needs.  ``synchronous=NORMAL`` is the WAL
        # pairing -- a commit survives a process crash, and only an OS crash can
        # lose the most recent ones, which for lifecycle rows that are re-derivable
        # from the worker is the right trade against an fsync per step.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        cur = self._conn.execute("PRAGMA user_version")
        version = int(cur.fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has schema version {version}, and this build "
                f"understands {SCHEMA_VERSION}. Refusing to read it: a reader "
                f"guessing at columns it does not know about is how a record "
                f"comes to say something its writer did not mean.")
        self._conn.executescript(_SCHEMA)
        if version < SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SqliteExperimentStore:
    """``ports.ExperimentStore`` over SQLite."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._store.connection

    def create(self, experiment: Experiment) -> Experiment:
        try:
            self._conn.execute(
                "INSERT INTO experiments (experiment_id, name, plan_digest, "
                "trainer, created_at, tags, document) VALUES (?,?,?,?,?,?,?)",
                (str(experiment.experiment_id), experiment.name,
                 experiment.plan.digest, experiment.plan.trainer,
                 experiment.created_at, ",".join(experiment.tags),
                 json.dumps(experiment.as_dict())))
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(
                f"experiment {experiment.experiment_id} / {experiment.name!r} "
                f"already exists: {exc}") from exc
        return experiment

    def get(self, experiment_id: ExperimentId) -> Experiment:
        row = self._conn.execute(
            "SELECT document FROM experiments WHERE experiment_id=?",
            (str(experiment_id),)).fetchone()
        if row is None:
            raise ExperimentNotFound(f"no experiment {experiment_id} in {self._store.path}")
        return Experiment.from_dict(json.loads(row["document"]))

    def find_by_name(self, name: str) -> Experiment | None:
        row = self._conn.execute(
            "SELECT document FROM experiments WHERE name=?", (name,)).fetchone()
        return Experiment.from_dict(json.loads(row["document"])) if row else None

    def update(self, experiment: Experiment) -> Experiment:
        cur = self._conn.execute(
            "UPDATE experiments SET name=?, plan_digest=?, trainer=?, tags=?, "
            "document=? WHERE experiment_id=?",
            (experiment.name, experiment.plan.digest, experiment.plan.trainer,
             ",".join(experiment.tags), json.dumps(experiment.as_dict()),
             str(experiment.experiment_id)))
        if cur.rowcount == 0:
            raise ExperimentNotFound(
                f"no experiment {experiment.experiment_id} to update")
        return experiment

    def attach_job(self, experiment_id: ExperimentId, job_id: JobId) -> Experiment:
        # One transaction, so a concurrent attach cannot read the same job list
        # and write back only its own addition.  IMMEDIATE takes the write lock
        # at BEGIN rather than at the first write, which is what stops two
        # readers from both deciding to upgrade.
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT document FROM experiments WHERE experiment_id=?",
                (str(experiment_id),)).fetchone()
            if row is None:
                raise ExperimentNotFound(f"no experiment {experiment_id}")
            updated = Experiment.from_dict(json.loads(row["document"])).with_job(
                JobId(job_id))
            conn.execute("UPDATE experiments SET document=? WHERE experiment_id=?",
                         (json.dumps(updated.as_dict()), str(experiment_id)))
            conn.execute("COMMIT")
            return updated
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def list(self, *, tag: str | None = None,
             limit: int | None = None) -> Sequence[Experiment]:
        sql = "SELECT document, tags FROM experiments ORDER BY created_at DESC"
        rows = self._conn.execute(sql).fetchall()
        out = []
        for row in rows:
            # Tag filtering in Python rather than SQL: a comma-joined column
            # matched with LIKE would match "air" inside "airframe", and the
            # correct SQL for it is a join table this does not otherwise need.
            if tag is not None and tag not in (row["tags"] or "").split(","):
                continue
            out.append(Experiment.from_dict(json.loads(row["document"])))
            if limit is not None and len(out) >= limit:
                break
        return out

    def delete(self, experiment_id: ExperimentId) -> None:
        self._conn.execute("DELETE FROM experiments WHERE experiment_id=?",
                           (str(experiment_id),))


class SqliteJobStore:
    """``ports.JobStore`` over SQLite."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._store.connection

    @staticmethod
    def _projection(job: TrainingJob) -> tuple:
        return (str(job.experiment_id), job.status.value, job.plan.digest,
                job.plan.trainer, job.created_at, job.started_at, job.ended_at,
                job.attempts, job.workspace, json.dumps(job.as_dict()))

    def create(self, job: TrainingJob) -> TrainingJob:
        try:
            self._conn.execute(
                "INSERT INTO jobs (job_id, experiment_id, status, plan_digest, "
                "trainer, created_at, started_at, ended_at, attempts, workspace, "
                "document) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(job.job_id), *self._projection(job)))
        except sqlite3.IntegrityError as exc:
            # Two different faults arrive through one exception class, and
            # reporting both as "already exists" sent a reader looking for a
            # duplicate job that was not there.  The foreign key is an integrity
            # guarantee the file store cannot offer -- a job whose experiment
            # does not exist is an orphan record, and the record is the point.
            if "FOREIGN KEY" in str(exc):
                raise ExperimentNotFound(
                    f"cannot create job {job.job_id}: experiment "
                    f"{job.experiment_id} does not exist. A job is one attempt "
                    f"at an experiment; create the experiment first.") from exc
            raise FileExistsError(f"job {job.job_id} already exists: {exc}") from exc
        return job

    def get(self, job_id: JobId) -> TrainingJob:
        row = self._conn.execute("SELECT document FROM jobs WHERE job_id=?",
                                 (str(job_id),)).fetchone()
        if row is None:
            raise JobNotFound(f"no job {job_id} in {self._store.path}")
        return TrainingJob.from_dict(json.loads(row["document"]))

    def update(self, job_id: JobId,
               change: Callable[[TrainingJob], TrainingJob]) -> TrainingJob:
        job_id = JobId(job_id)
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT document FROM jobs WHERE job_id=?",
                               (str(job_id),)).fetchone()
            if row is None:
                raise JobNotFound(f"no job {job_id} in {self._store.path}")
            updated = change(TrainingJob.from_dict(json.loads(row["document"])))
            if updated.job_id != job_id:
                raise ValueError(
                    f"the change returned job {updated.job_id}, not {job_id}")
            conn.execute(
                "UPDATE jobs SET experiment_id=?, status=?, plan_digest=?, "
                "trainer=?, created_at=?, started_at=?, ended_at=?, attempts=?, "
                "workspace=?, document=? WHERE job_id=?",
                (*self._projection(updated), str(job_id)))
            conn.execute("COMMIT")
            return updated
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def save_state(self, state: TrainingState) -> None:
        self._conn.execute(
            "INSERT INTO job_state (job_id, step, updated_at, document) "
            "VALUES (?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
            "step=excluded.step, updated_at=excluded.updated_at, "
            "document=excluded.document",
            (str(state.job_id), state.step, state.updated_at,
             json.dumps(state.as_dict())))

    def load_state(self, job_id: JobId) -> TrainingState | None:
        row = self._conn.execute("SELECT document FROM job_state WHERE job_id=?",
                                 (str(job_id),)).fetchone()
        return TrainingState.from_dict(json.loads(row["document"])) if row else None

    def list(self, *, experiment_id: ExperimentId | None = None,
             status: JobStatus | None = None,
             limit: int | None = None) -> Sequence[TrainingJob]:
        sql = "SELECT document FROM jobs"
        where, params = [], []
        if experiment_id is not None:
            where.append("experiment_id=?")
            params.append(str(experiment_id))
        if status is not None:
            where.append("status=?")
            params.append(JobStatus(status).value)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, job_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [TrainingJob.from_dict(json.loads(r["document"]))
                for r in self._conn.execute(sql, params).fetchall()]

    def delete(self, job_id: JobId) -> None:
        self._conn.execute("DELETE FROM job_state WHERE job_id=?", (str(job_id),))
        self._conn.execute("DELETE FROM jobs WHERE job_id=?", (str(job_id),))
