"""The CPU fallback is taken only when no GPU exists.

``envs.device`` decides which device a search scores on.  The rule (the user,
2026-10-12): the numpy (CPU) path is allowed only on a machine with no GPU; a
GPU that is present but cannot run the kernel -- broken driver, stale build,
too little memory -- makes ``run_search`` refuse instead.

The detection and the decision need nothing installed and run everywhere.  The
last function checks that ``run_search`` actually routes through the decision;
it needs numpy, MuJoCo and torch and prints ``[skip]`` without them.

Run:  PYTHONPATH=. python tests/test_device.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dytiscidae.envs.device import GpuUnusable, choose, gpu_present  # noqa: E402

FAILURES: list[str] = []
SKIPS: list[str] = []
TEMPS: list[Path] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def skip(name: str, why: str) -> None:
    print(f"  [skip] {name}  ({why})")
    SKIPS.append(name)


def fake_root(files: dict) -> Path:
    """A filesystem root holding ``{relative path: text}``; ``None`` makes a directory."""
    root = Path(tempfile.mkdtemp(prefix="device-"))
    TEMPS.append(root)
    for rel, text in files.items():
        p = root / rel
        if text is None:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    return root


def pci(slot: str, vendor: str, cls: str) -> dict:
    d = f"sys/bus/pci/devices/{slot}"
    return {f"{d}/vendor": vendor + "\n", f"{d}/class": cls + "\n"}


def test_presence_is_read_from_the_hardware() -> None:
    print("\ndevice: a GPU is present when the hardware says so, driver or not")
    present, why = gpu_present(fake_root({}))
    check("an empty root has no GPU", present is False, why)

    # This container's own PCI bus, 2026-10-12: an Intel host bridge and virtio.
    root = fake_root({**pci("0000:00:00.0", "0x8086", "0x060000"),
                      **pci("0000:00:01.0", "0x1af4", "0x018000")})
    check("a host bridge and virtio devices are not a GPU", gpu_present(root)[0] is False)

    root = fake_root(pci("0000:00:02.0", "0x8086", "0x030000"))
    check("an Intel VGA controller is not a GPU the kernel targets",
          gpu_present(root)[0] is False)

    # An NVIDIA card's HDMI audio function shares the vendor, not the class.
    root = fake_root(pci("0000:01:00.1", "0x10de", "0x040300"))
    check("an NVIDIA audio function is not a GPU", gpu_present(root)[0] is False)

    for vendor, cls, name in (("0x10de", "0x030000", "NVIDIA VGA"),
                              ("0x10de", "0x030200", "NVIDIA 3D controller"),
                              ("0x1002", "0x038000", None),
                              ("0x1002", "0x030200", "AMD 3D controller")):
        root = fake_root(pci("0000:01:00.0", vendor, cls))
        present, why = gpu_present(root)
        if name is None:
            check(f"{vendor} class {cls} (display, other) is not counted", present is False)
        else:
            check(f"an {name} on the PCI bus is a GPU, with no driver loaded",
                  present is True and "0000:01:00.0" in why, why)

    present, why = gpu_present(fake_root({"proc/driver/nvidia/gpus/0000:01:00.0": None}))
    check("/proc/driver/nvidia/gpus/<bus> is a GPU", present is True, why)
    present, why = gpu_present(fake_root({"dev/nvidia0": ""}))
    check("/dev/nvidia0 is a GPU", present is True, why)
    present, _ = gpu_present(fake_root({"dev/nvidiactl": "", "dev/nvidia-uvm": ""}))
    check("/dev/nvidiactl and /dev/nvidia-uvm alone are not a GPU", present is False)
    present, why = gpu_present(fake_root({"dev/kfd": ""}))
    check("/dev/kfd (AMD ROCm) is a GPU", present is True, why)


def test_the_cpu_is_taken_only_without_a_gpu() -> None:
    print("\ndevice: the CPU path is taken only when no GPU exists")
    asked = []

    def present():
        asked.append(1)
        return True, "PCI 0000:01:00.0 NVIDIA"

    check("a usable kernel is the GPU path",
          choose(lambda: (True, ""), present) == ("gpu", ""))
    check("and the hardware is not even consulted", asked == [])

    backend, why = choose(lambda: (False, "ModuleNotFoundError: full_pipeline"),
                          lambda: (False, "nothing found"))
    check("no GPU and no kernel is the CPU path", backend == "cpu", why)
    check("and the reason keeps both halves",
          "nothing found" in why and "full_pipeline" in why, why)

    # The failures measured on this project's own GPU machine, each of which
    # used to send the run to the CPU.
    for reason in ("the extension imports but cannot be constructed: "
                   "ValueError: Failed to initialize NVML: 18",
                   "GPU kernel older than its source: fluid.mojo changed",
                   "the extension imports but cannot be constructed: "
                   "CUDA_ERROR_OUT_OF_MEMORY",
                   "ModuleNotFoundError: No module named 'full_pipeline'"):
        try:
            got = choose(lambda: (False, reason), present)
        except GpuUnusable as exc:
            check(f"GPU present, '{reason[-40:]}' refuses",
                  reason in str(exc) and "0000:01:00.0" in str(exc))
        else:
            check(f"GPU present, '{reason[-40:]}' refuses", False, f"returned {got}")


def test_run_search_routes_through_the_decision() -> None:
    print("\ndevice: run_search refuses on a present, unusable GPU, before any work")
    try:
        from dytiscidae.envs import batchroll, device
        from dytiscidae.evolution import loop
    except Exception as exc:                                  # noqa: BLE001
        skip("run_search routes through the decision", f"{type(exc).__name__}: {exc}")
        return

    class _Reached(Exception):
        pass

    def reached(state, spec):
        raise _Reached

    saved = (batchroll.usable, batchroll.AVAILABLE, batchroll.UNAVAILABLE_REASON,
             device.gpu_present, loop.seed_archipelago)
    tmp = tempfile.mkdtemp(prefix="device-run-")
    TEMPS.append(Path(tmp))
    cfg = loop.SearchConfig(generations=1, batch=1, seed=1, run_dir=tmp, workers=1)
    try:
        batchroll.usable = lambda: (False, "Failed to initialize NVML: 18")
        loop.seed_archipelago = reached

        device.gpu_present = lambda: (True, "PCI 0000:01:00.0 NVIDIA")
        try:
            loop.run_search(cfg)
            outcome = "returned"
        except GpuUnusable as exc:
            outcome = "refused" if "NVML: 18" in str(exc) else f"refused without reason: {exc}"
        except _Reached:
            outcome = "reached seeding on the CPU"
        check("GPU present and unusable: run_search refuses", outcome == "refused", outcome)

        device.gpu_present = lambda: (False, "nothing found")
        batchroll.AVAILABLE = True
        try:
            loop.run_search(cfg)
            outcome = "returned"
        except _Reached:
            outcome = "reached seeding"
        except GpuUnusable as exc:
            outcome = f"refused: {exc}"
        check("no GPU: run_search goes on, on the CPU path", outcome == "reached seeding", outcome)
        check("and the per-generation guard sees the CPU path, with the reason",
              batchroll.AVAILABLE is False and "nothing found" in batchroll.UNAVAILABLE_REASON,
              batchroll.UNAVAILABLE_REASON)
    finally:
        (batchroll.usable, batchroll.AVAILABLE, batchroll.UNAVAILABLE_REASON,
         device.gpu_present, loop.seed_archipelago) = saved


def main() -> int:
    print("=" * 68)
    print("device: CPU fallback only without a GPU")
    print("=" * 68)
    try:
        test_presence_is_read_from_the_hardware()
        test_the_cpu_is_taken_only_without_a_gpu()
        test_run_search_routes_through_the_decision()
    finally:
        for path in TEMPS:
            shutil.rmtree(path, ignore_errors=True)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    tail = f", {len(SKIPS)} skipped" if SKIPS else ""
    print(f"all device checks passed{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
