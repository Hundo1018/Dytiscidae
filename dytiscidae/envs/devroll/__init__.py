"""On-device evaluation path (IMPL_1010 A): MuJoCo Warp plus Warp fluid kernels.

Optional, like ``envs.batchroll``: nothing here is imported unless ``warp`` and
``mujoco_warp`` are present, and the rest of the project runs without them.
Pins live in ``requirements-device.txt``. Lanes A1a onward add the kernels
(``fluid_wp``, ``scatter_wp``, ``model_wp``, ``control_wp``, ``rollout_wp``).

The guard follows ``batchroll``'s: ``AVAILABLE`` is decided at import time and
says only that the packages import; :func:`usable` asks a subprocess whether a
CUDA device exists and whether the two MuJoCo versions agree, because an
import that succeeds with no device fails later and far from its cause.
"""

from __future__ import annotations

import os as _os
import subprocess as _subprocess
import sys as _sys

#: Set when ``warp`` and ``mujoco_warp`` import.  Says nothing about a GPU.
AVAILABLE = False
#: ``"MJWarp not importable"`` when the packages do not import, else empty.
UNAVAILABLE_REASON = ""
#: The exception text behind ``UNAVAILABLE_REASON``, for the reader's eye only.
IMPORT_ERROR = ""

try:  # pragma: no cover - depends on the device stack being installed
    import mujoco_warp as _mjw
    import warp as _wp
    AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - any import failure is "not importable"
    UNAVAILABLE_REASON = "MJWarp not importable"
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_PROBE = (
    "import sys, mujoco, mujoco_warp, warp\n"
    "n = warp.get_cuda_device_count()\n"
    "if n < 1:\n"
    "    sys.exit('no CUDA device visible to warp')\n"
    "if mujoco.__version__ != mujoco_warp.__version__:\n"
    "    sys.exit(f'mujoco {mujoco.__version__} != mujoco_warp {mujoco_warp.__version__}')\n"
)

#: Cached result of :func:`usable`; the probe starts an interpreter.
_USABLE: tuple | None = None


def usable(timeout: float = 180.0) -> tuple:
    """Return ``(ok, reason)``: can the device path run in this environment?

    ``reason`` is empty when ``ok``; otherwise it is ``"MJWarp not importable"``
    or ``"MJWarp not usable: <reason>"``.  The probe runs in a subprocess with
    the caller's environment (``PYTHONEXECUTABLE`` included, as in
    ``batchroll.usable``), so a driver that aborts the process cannot take the
    caller down with it.
    """
    global _USABLE
    if _USABLE is not None:
        return _USABLE
    if not AVAILABLE:
        _USABLE = (False, UNAVAILABLE_REASON)
        return _USABLE
    try:
        out = _subprocess.run([_sys.executable, "-c", _PROBE], capture_output=True,
                              text=True, timeout=timeout, env=dict(_os.environ))
    except (_subprocess.TimeoutExpired, OSError) as exc:
        _USABLE = (False, f"MJWarp not usable: {type(exc).__name__}: {exc}")
        return _USABLE
    if out.returncode == 0:
        _USABLE = (True, "")
    else:
        tail = (out.stderr or out.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit {out.returncode}"
        _USABLE = (False, f"MJWarp not usable: {detail}")
    return _USABLE
