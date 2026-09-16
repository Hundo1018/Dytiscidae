"""The trainer registry: names in, ``Trainer`` implementations out.

The application layer holds a ``Callable[[str], Trainer]`` and nothing more.
That indirection is what keeps ``import dytiscidae.application`` from pulling in
torch and MuJoCo -- a resolver that imports every trainer at module load would
put both of them behind every use case, including ``ListJobs``.

So resolution is lazy and per name.  ``resolve("synthetic")`` imports a module
with no dependencies; ``resolve("search")`` imports MuJoCo, numpy and possibly
torch, and only when a job actually names it.

Out-of-tree trainers register two ways:

``register("name", factory)``
    in process, for a test or an embedding application.

``DYTISCIDAE_TRAINERS="name=package.module:Attr,other=..."``
    by environment, which is how a worker in a *different process* gets a
    trainer the launcher's parent had registered in memory.  Without it, a test
    that registers a fake and then launches a real subprocess would find the
    subprocess had never heard of it.
"""

from __future__ import annotations

import os
from typing import Callable, Dict

from ...ports.trainer import Trainer

#: name -> a zero-argument factory.  Built-ins are imported inside their factory.
_REGISTRY: Dict[str, Callable[[], Trainer]] = {}


def register(name: str, factory: Callable[[], Trainer], *,
             replace: bool = False) -> None:
    """Register a trainer factory under ``name``.

    Refuses to shadow an existing name unless asked.  A silent replacement means
    a job recorded as running trainer X ran trainer Y, and the record gives no
    sign of it.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("trainer name must be non-empty")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"a trainer named {name!r} is already registered; pass replace=True "
            f"if shadowing it is what you mean")
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    _REGISTRY.pop(str(name), None)


def known() -> list[str]:
    """Every name that resolves, built-ins and environment entries included."""
    return sorted(set(_REGISTRY) | set(_BUILTINS) | set(_from_environment()))


def resolve(name: str) -> Trainer:
    """Build the trainer registered under ``name``.

    Order: explicit registrations, then the environment, then the built-ins.
    Explicit first so that a test's fake wins over a built-in of the same name
    inside one process, and the environment before the built-ins so that a
    worker process can be given one from outside.
    """
    name = str(name).strip()
    if name in _REGISTRY:
        return _REGISTRY[name]()

    env = _from_environment()
    if name in env:
        return _load(env[name])()

    if name in _BUILTINS:
        return _BUILTINS[name]()

    raise UnknownTrainer(
        f"no trainer named {name!r}. Known: {', '.join(known()) or '(none)'}. "
        f"Register one with adapters.trainers.register(), or name it in "
        f"DYTISCIDAE_TRAINERS as 'name=package.module:Attr'.")


class UnknownTrainer(KeyError):
    def __str__(self) -> str:                  # KeyError quotes its argument
        return self.args[0] if self.args else ""


# -- built-ins, each importing its dependencies only when called -------------

def _synthetic() -> Trainer:
    from .synthetic import SyntheticTrainer
    return SyntheticTrainer()


def _search() -> Trainer:
    # Imports numpy, MuJoCo and -- with a shared policy -- torch.  Behind a
    # function so that naming it in a plan is what costs, not importing this
    # package.
    from .search import SearchTrainer
    return SearchTrainer()


_BUILTINS: Dict[str, Callable[[], Trainer]] = {
    "synthetic": _synthetic,
    "search": _search,
}


def _from_environment() -> Dict[str, str]:
    spec = os.environ.get("DYTISCIDAE_TRAINERS", "")
    out: Dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, _, target = entry.partition("=")
        out[name.strip()] = target.strip()
    return out


def _load(target: str) -> Callable[[], Trainer]:
    module_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError(
            f"{target!r} must be 'package.module:Attr'; the colon is what "
            f"separates the module from the name inside it")
    import importlib

    def factory() -> Trainer:
        obj = getattr(importlib.import_module(module_name), attr)
        # Either a class to instantiate or an already-built instance.  Checked
        # rather than assumed, so a module exporting a singleton works too.
        return obj() if isinstance(obj, type) else obj

    return factory


__all__ = ["UnknownTrainer", "known", "register", "resolve", "unregister"]
