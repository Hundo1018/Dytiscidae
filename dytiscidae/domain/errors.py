"""Domain errors.

Every one of these is a condition the *domain* can recognise.  An adapter that
cannot reach its disk raises ``OSError``; it does not dress it up as a
``CheckpointNotFound``, because the two want different responses -- one is
retryable and the other is not, and a caller that cannot tell them apart will
retry forever or give up immediately.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base for everything the domain raises."""


class NotFound(DomainError):
    """An entity that was named does not exist."""


class JobNotFound(NotFound):
    pass


class ExperimentNotFound(NotFound):
    pass


class CheckpointNotFound(NotFound):
    pass


class DatasetNotFound(NotFound):
    pass


class IllegalTransition(DomainError):
    """A lifecycle move that the job's current state does not allow.

    Carries both ends so the message says what was attempted, not only that
    something was.  A cancel arriving twice, a resume of a job that succeeded,
    and a pause of a job that never started all land here, and telling them
    apart from the text is the point.
    """

    def __init__(self, job_id: str, frm: str, to: str, detail: str = "") -> None:
        self.job_id, self.frm, self.to = str(job_id), str(frm), str(to)
        msg = f"job {job_id}: cannot go {frm} -> {to}"
        super().__init__(f"{msg} ({detail})" if detail else msg)


class PlanRejected(DomainError):
    """A configuration that cannot describe a run.

    Raised at construction, not at the first generation.  A budget of zero
    steps, a negative seed, or a trainer name that is empty are all things a
    job should never reach RUNNING with -- the cost of finding out four hours
    in is the whole reason this check is here.
    """


class WorkerFailure(DomainError):
    """The training runtime stopped in a way the job has to record.

    Distinct from a Python exception escaping the trainer: that becomes a
    ``FailureInfo`` on the job.  This is for the launcher's own failures --
    a worker that never started, or exited without writing a terminal status.
    """
