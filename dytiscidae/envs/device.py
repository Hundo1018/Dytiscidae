"""Which device a search scores on, and the one condition that allows the CPU.

The search scores through the Mojo fluid kernel (``envs.batchroll``).  On the
numpy path it scores each candidate in the parent on the single-machine
evaluator, and the shared policy gets no transitions
(``experiments/cpu_path/run.py``).  Until 2026-10-12
``run_search`` fell back to the numpy path on *any* failure of
``batchroll.usable()``: a driver whose kernel module did not match its
userspace (2026-09-19, ``Failed to initialize NVML: 18``), a kernel older than
its source, an allocation the device could not make.  On a machine with a GPU
each of those is a fault to fix, and a run that carries on at CPU speed hides
it behind a log line: a generation measured ~110-225 s on the GPU path
(2026-10-03) and the CPU path is several times that.

The rule (the user, 2026-10-12): **fall back to the CPU only when no GPU
exists.**  A GPU that is present but cannot run the kernel -- broken driver,
stale build, too little memory, too slow -- does not trigger the fallback; the
run refuses at t=0 with the reason instead.

"Present" is read from the hardware, not from the driver, so a GPU whose driver
is broken still counts as present:

  * a PCI display or 3D controller (class ``0x0300``/``0x0302``) from NVIDIA
    (``0x10de``) or AMD (``0x1002``), the two vendors Mojo targets;
  * ``/proc/driver/nvidia/gpus/*``, ``/dev/nvidia<N>`` or ``/dev/kfd``, for
    containers that pass the device through without exposing ``/sys/bus/pci``.

Stdlib only, so the cheap suites (no third-party package) can test it.
"""
from __future__ import annotations

import re
from pathlib import Path

#: PCI vendor ids of the GPUs the Mojo kernel can target.
GPU_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD"}
#: PCI class prefixes: VGA-compatible controller, 3D controller.
GPU_CLASSES = ("0x0300", "0x0302")


class GpuUnusable(RuntimeError):
    """A GPU is present and the kernel cannot run on it.  Not a CPU fallback."""


def gpu_present(root: Path | str = "/") -> tuple:
    """``(present, evidence)``.  ``evidence`` names what was found, or what was looked at.

    ``root`` is the filesystem root, so a test can build a fake one.
    """
    root = Path(root)
    found = []

    pci = root / "sys" / "bus" / "pci" / "devices"
    if pci.is_dir():
        for dev in sorted(pci.iterdir()):
            try:
                vendor = (dev / "vendor").read_text().strip().lower()
                cls = (dev / "class").read_text().strip().lower()
            except OSError:
                continue
            if vendor in GPU_VENDORS and cls.startswith(GPU_CLASSES):
                found.append(f"PCI {dev.name} {GPU_VENDORS[vendor]} class {cls}")

    gpus = root / "proc" / "driver" / "nvidia" / "gpus"
    if gpus.is_dir():
        found += [f"{gpus / p.name}" for p in sorted(gpus.iterdir())]

    dev = root / "dev"
    if dev.is_dir():
        found += [str(dev / p.name) for p in sorted(dev.iterdir())
                  if re.fullmatch(r"nvidia\d+", p.name)]
        if (dev / "kfd").exists():
            found.append(str(dev / "kfd"))

    if found:
        return True, "; ".join(found)
    return False, (f"no NVIDIA/AMD controller under {pci}, nothing under "
                   f"{gpus}, no {dev}/nvidia<N> or {dev}/kfd")


def choose(usable, present=None) -> tuple:
    """``("gpu", "")`` or ``("cpu", reason)``; raises :class:`GpuUnusable`.

    ``usable`` is ``batchroll.usable``: ``() -> (ok, why)``.  ``present`` is
    :func:`gpu_present` with its default root.  Either is injectable for tests.
    """
    ok, why = usable()
    if ok:
        return "gpu", ""
    has_gpu, evidence = (present or gpu_present)()
    if has_gpu:
        raise GpuUnusable(
            f"a GPU is present ({evidence}) but the fluid kernel cannot run on "
            f"it: {why}\n"
            "The CPU fallback is for machines with no GPU; it is not taken for "
            "a GPU that is broken, stale or short of resources.  Fix the cause "
            "(`cd mojo && pixi run build-all`; a driver whose kernel module "
            "matches its userspace) and start the run again.")
    return "cpu", f"no GPU on this machine ({evidence}); {why}"
