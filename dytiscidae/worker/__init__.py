"""The training worker: a job's own process.

``runtime.TrainingWorker`` is the piece that turns a trainer's behaviour into
job state that outlives the process.  ``__main__`` is the entry point the
subprocess launcher starts.

Importing this package does not import a trainer, so it stays cheap on a
machine that has no MuJoCo: the trainer is resolved by name at run time.
"""

from .runtime import TrainingWorker, WorkerContext

__all__ = ["TrainingWorker", "WorkerContext"]
