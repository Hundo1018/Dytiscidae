#!/usr/bin/env python3
"""Q6 -- Mojo -> Warp zero copy (ROADMAP 2026-10-10, "an evaluation that never
leaves the device", section 4, probe Q6).

Frozen text (verbatim, line breaks as in docs/ROADMAP.md):

    **Q6 (interop, hours): Mojo -> Warp zero copy.** A Mojo `DeviceBuffer`
    exposed to Python and wrapped by `wp.from_dlpack` or `wp.array(ptr=...)`,
    written by a Mojo kernel and read by a Warp kernel on one stream, 1e6 floats,
    checksum equal, no host copy (measure with `nvidia-smi`'s PCIe counters or a
    timing that does not scale with size). *Prediction:* it works or fails on the
    first try; there is no partial outcome. Decides C.

Readings applied (the request's, not changes to the text):
  * `wp.from_dlpack` is not attempted: Mojo's `DeviceBuffer` has no DLPack
    (F_on_device_tooling.md Q3); the route is `wp.array(ptr=<int>, copy=False)`.
  * "on one stream" is reported, not required: Mojo's `ctx.stream()` and Warp's
    stream are separate streams; ordering is by a host synchronize at each
    hand-off.  Whether the two handles are equal is recorded in the table.
  * "timing that does not scale with size": wrap time at 1e5 / 1e6 / 1e7 float32.
    No-copy is read as ratio(1e7 / 1e5) < 3 AND wrap(1e7) < 0.1 * host->device
    copy of the same 1e7 floats (measured in the same process).  The thresholds
    are mine, written before any number exists; the raw times are in the table.
  * Outcome words: WORKS (every check passes at every size), FAILS (nothing
    passes, or the child dies/raises before the first size completes), PARTIAL
    (anything else).  Against the frozen text WORKS and FAILS are CONFIRMED and
    PARTIAL is REFUTED ("there is no partial outcome").

Architecture: the parent (this script, no warp/mojo import) spawns a child
(`--child`) that does everything and appends one JSON line per stage to a log,
flushed at once.  A segfault or CUDA abort in the child therefore still leaves a
log saying how far it got; the parent reduces the log and writes the report.

Usage (GPU, ~30 s predicted, timeout 300 s):

    ../mjwarp-venv/bin/python experiments/mojo_warp_interop/run.py
    ../mjwarp-venv/bin/python experiments/mojo_warp_interop/run.py --selftest   # CPU only

Build the Mojo module first (see interop_probe.mojo; never touches mojo/build/).
"""
import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_REPORT = REPO / "runs" / "analysis_1010_failure_theory" / "Q6_result.md"
DEFAULT_RESULTS = HERE / "results_q6.json"
DEFAULT_BUILD = HERE / "build"
FROZEN_FALLBACK = "it works or fails on the first try; there is no partial outcome."

RATIO_MAX = 3.0        # wrap(1e7)/wrap(1e5) must stay below this to read as "no copy"
COPY_FRAC_MAX = 0.1    # wrap(1e7) must be < this fraction of a 1e7-float H2D copy


# --------------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU only: exercise parsing, reducer and report writer on synthetic events")
    ap.add_argument("--sizes", default="100000,1000000,10000000", help="float32 element counts, comma separated (1e6 is the headline)")
    ap.add_argument("--reps", type=int, default=300, help="wrap-timing repetitions per size")
    ap.add_argument("--copy-reps", type=int, default=5, help="host->device contrast repetitions per size")
    ap.add_argument("--order", choices=["mojo_first", "warp_first"], default="mojo_first",
                    help="which library creates its CUDA context first")
    ap.add_argument("--build-dir", default=str(DEFAULT_BUILD), help="directory holding interop_probe.so")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--timeout", type=float, default=300.0, help="child wall limit, seconds")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--log", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.sizes = [int(float(s)) for s in a.sizes.split(",") if s.strip()]
    if not a.sizes or any(n < 1 for n in a.sizes):
        ap.error("--sizes needs positive integers")
    return a


# ------------------------------------------------------------------- exact sums
def expected_sum(n, k=0):
    """Sum of (0.5*i + k) for i < n.  Exact in float64: n(n-1)/4 + k*n, n(n-1) < 2**53."""
    return n * (n - 1) / 4.0 + float(k) * n


def float32_exact(n, k=2):
    """True when every value 0.5*i + k, i < n, is exactly representable in float32."""
    return (0.5 * (n - 1) + k) < 2 ** 23


# ----------------------------------------------------------------- libcuda probe
def cu_probe(ptr):
    """Owner context / memory type of a device pointer, straight from the driver."""
    out = {}
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        cur = ctypes.c_void_p()
        out["cuCtxGetCurrent_rc"] = lib.cuCtxGetCurrent(ctypes.byref(cur))
        out["current_ctx"] = cur.value or 0
        lib.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        owner = ctypes.c_void_p()
        out["owner_ctx_rc"] = lib.cuPointerGetAttribute(ctypes.byref(owner), 1, ctypes.c_uint64(ptr))  # CU_POINTER_ATTRIBUTE_CONTEXT
        out["owner_ctx"] = owner.value or 0
        mt = ctypes.c_int(0)
        out["memtype_rc"] = lib.cuPointerGetAttribute(ctypes.byref(mt), 2, ctypes.c_uint64(ptr))  # MEMORY_TYPE; 2 = device
        out["memtype"] = mt.value
        dev = ctypes.c_int(-1)
        out["ordinal_rc"] = lib.cuPointerGetAttribute(ctypes.byref(dev), 9, ctypes.c_uint64(ptr))  # DEVICE_ORDINAL
        out["ordinal"] = dev.value
        pctx = ctypes.c_void_p()
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.c_void_p, ctypes.c_int]
        out["primary_rc"] = lib.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), 0)
        out["primary_ctx"] = pctx.value or 0
        if out["primary_rc"] == 0:
            lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
            lib.cuDevicePrimaryCtxRelease(0)
    except Exception as e:  # noqa: BLE001 -- a probe must report, not die
        out["error"] = repr(e)
    return out


# ------------------------------------------------------------------------ child
class EventLog:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, kind, **kw):
        kw["event"] = kind
        kw["t"] = time.time()
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def child_main(a):
    log = EventLog(a.log)
    log("start", python=sys.version.split()[0], pid=os.getpid(), order=a.order, sizes=a.sizes)
    sys.path.insert(0, a.build_dir)
    sys.path.insert(0, str(HERE))
    import numpy as np
    try:
        import interop_probe
    except Exception as e:  # noqa: BLE001
        log("fail", stage="import_mojo", error=repr(e))
        return 2
    log("stage", name="import_mojo", ok=True)
    try:
        import warp as wp
        wp.config.quiet = True
        import warp_kernels
    except Exception as e:  # noqa: BLE001
        log("fail", stage="import_warp", error=repr(e))
        return 2
    log("stage", name="import_warp", ok=True, warp=wp.config.version)
    dev = "cuda:0"

    def warp_ctx_info():
        d = wp.get_device(dev)
        info = {"name": d.name, "arch": getattr(d, "arch", None)}
        try:
            info["warp_ctx"] = int(d.context) if d.context is not None else 0
        except Exception as e:  # noqa: BLE001
            info["warp_ctx_error"] = repr(e)
        try:
            info["warp_stream"] = int(wp.get_stream(dev).cuda_stream)
        except Exception as e:  # noqa: BLE001
            info["warp_stream_error"] = repr(e)
        return info

    try:
        if a.order == "warp_first":
            wp.init()
            warm = wp.zeros(8, dtype=wp.float32, device=dev)
            wp.launch(warp_kernels.add_one, dim=8, inputs=[warm], device=dev)
            wp.synchronize_device(dev)
            log("stage", name="warp_ctx_created_first", ok=True, **warp_ctx_info())
        first = True
        for n in a.sizes:
            row = {"n": n, "float32_exact": float32_exact(n)}
            try:
                t0 = time.perf_counter()
                p = interop_probe.Probe(n)       # allocates on cuda:0, Mojo kernel writes 0.5*i
                row["mojo_alloc_fill_s"] = time.perf_counter() - t0
                ptr = p.ptr()
                row["ptr"] = ptr
                row["mojo_ctx"] = p.context_handle()
                row["mojo_stream"] = p.stream_handle()
                row["checksum_mojo"] = p.checksum()
                row["expect_mojo"] = expected_sum(n)
                row["ok_mojo_fill"] = row["checksum_mojo"] == row["expect_mojo"]
                log("size_mojo", **row)
                if first:
                    if a.order == "mojo_first":
                        wp.init()
                    first = False
                    info = warp_ctx_info()
                    row.update(info)
                    log("stage", name="warp_init", ok=True, **info)
                row["cu"] = cu_probe(ptr)
                # wrap: Warp aliases the Mojo allocation, no copy
                arr = wp.array(ptr=ptr, dtype=wp.float32, shape=(n,), device=dev, copy=False)
                row["wrap_ok"] = True
                row["warp_stream"] = warp_ctx_info().get("warp_stream")
                row["same_stream"] = bool(row["mojo_stream"] and row["mojo_stream"] == row["warp_stream"])
                # Warp reads what Mojo wrote (this download is a check, not part of the path)
                host = arr.numpy()
                row["ok_warp_reads_mojo"] = bool(np.array_equal(host, np.arange(n, dtype=np.float32) * np.float32(0.5)))
                del host
                # Warp writes, Mojo re-reads on device
                wp.launch(warp_kernels.add_one, dim=n, inputs=[arr], device=dev)
                wp.synchronize_device(dev)
                row["checksum_after_warp1"] = p.checksum()
                row["expect_after_warp1"] = expected_sum(n, 1)
                row["ok_mojo_sees_warp1"] = row["checksum_after_warp1"] == row["expect_after_warp1"]
                wp.launch(warp_kernels.add_one, dim=n, inputs=[arr], device=dev)
                wp.synchronize_device(dev)
                row["checksum_after_warp2"] = p.checksum()
                row["expect_after_warp2"] = expected_sum(n, 2)
                row["ok_mojo_sees_warp2"] = row["checksum_after_warp2"] == row["expect_after_warp2"]
                log("size_checks", **row)
                # wrap timing
                del arr
                for _ in range(10):
                    wp.array(ptr=ptr, dtype=wp.float32, shape=(n,), device=dev, copy=False)
                ts = []
                for _ in range(a.reps):
                    t = time.perf_counter_ns()
                    w = wp.array(ptr=ptr, dtype=wp.float32, shape=(n,), device=dev, copy=False)
                    ts.append((time.perf_counter_ns() - t) / 1e3)
                    del w
                row["wrap_us_median"] = statistics.median(ts)
                row["wrap_us_p10"], row["wrap_us_p90"] = pct(ts, 0.1), pct(ts, 0.9)
                # contrast: what a copy of the same floats costs
                hostbuf = np.zeros(n, dtype=np.float32)
                cs = []
                for _ in range(a.copy_reps):
                    wp.synchronize_device(dev)
                    t = time.perf_counter_ns()
                    c = wp.array(hostbuf, dtype=wp.float32, device=dev)
                    wp.synchronize_device(dev)
                    cs.append((time.perf_counter_ns() - t) / 1e3)
                    del c
                row["h2d_us_median"] = statistics.median(cs)
                row["h2d_us_min"], row["h2d_us_max"] = min(cs), max(cs)
                # the wrappers were created and destroyed hundreds of times: the Mojo buffer must be intact
                row["checksum_final"] = p.checksum()
                row["ok_intact_after_wrappers"] = row["checksum_final"] == row["expect_after_warp2"]
                log("size_done", **row)
                del p
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)
                log("size_error", **row)
                return 3
        log("done")
        return 0
    except Exception as e:  # noqa: BLE001
        log("fail", stage="warp_first_setup", error=repr(e))
        return 3


# ---------------------------------------------------------------------- reducer
def reduce_events(events, returncode, stderr_tail="", sizes=None):
    """Events (+ child exit status) -> result dict with the verdict.  Pure; used by --selftest."""
    rows = {}
    for e in events:
        if e.get("event") in ("size_mojo", "size_checks", "size_done", "size_error"):
            rows.setdefault(e["n"], {}).update({k: v for k, v in e.items() if k not in ("event", "t")})
            rows[e["n"]]["last_event"] = e["event"]
    notes, fails = [], []
    done = any(e.get("event") == "done" for e in events)
    for e in events:
        if e.get("event") == "fail":
            fails.append(f"{e.get('stage')}: {e.get('error')}")
    if returncode not in (0, None):
        name = f"signal {-returncode}" if returncode < 0 else f"exit {returncode}"
        fails.append(f"child terminated: {name}")
    for n, r in rows.items():
        if r.get("error"):
            fails.append(f"n={n}: {r['error']}")
    per = {}
    for n, r in rows.items():
        checks = {
            "mojo_fill": r.get("ok_mojo_fill"),
            "warp_reads_mojo": r.get("ok_warp_reads_mojo"),
            "mojo_sees_warp_x1": r.get("ok_mojo_sees_warp1"),
            "mojo_sees_warp_x2": r.get("ok_mojo_sees_warp2"),
            "intact_after_wrappers": r.get("ok_intact_after_wrappers"),
        }
        r["checks"] = checks
        r["all_ok"] = all(v is True for v in checks.values())
        cu = r.get("cu") or {}
        if cu.get("owner_ctx") and r.get("warp_ctx") is not None:
            r["ctx_owner_is_warp_ctx"] = cu["owner_ctx"] == r.get("warp_ctx")
        if r.get("mojo_ctx") and r.get("warp_ctx"):
            r["mojo_ctx_is_warp_ctx"] = r["mojo_ctx"] == r["warp_ctx"]
        per[n] = r
    ns = sorted(per)
    scaling = {}
    if len(ns) >= 2 and all("wrap_us_median" in per[n] for n in (ns[0], ns[-1])):
        lo, hi = per[ns[0]], per[ns[-1]]
        ratio = hi["wrap_us_median"] / lo["wrap_us_median"] if lo["wrap_us_median"] > 0 else float("inf")
        copy_frac = hi["wrap_us_median"] / hi["h2d_us_median"] if hi.get("h2d_us_median") else float("inf")
        scaling = {"n_lo": ns[0], "n_hi": ns[-1], "wrap_ratio": ratio, "wrap_over_h2d_at_hi": copy_frac,
                   "no_copy": bool(ratio < RATIO_MAX and copy_frac < COPY_FRAC_MAX),
                   "ratio_max": RATIO_MAX, "copy_frac_max": COPY_FRAC_MAX,
                   "size_ratio": ns[-1] / ns[0]}
    else:
        notes.append("wrap-scaling not computed (needs two sizes with timings)")
    n_pass = sum(1 for n in ns if per[n]["all_ok"])
    want = sizes if sizes else ns
    all_sizes_done = done and all(n in per and per[n].get("last_event") == "size_done" for n in want)
    setup_failed = any(e.get("event") == "fail" and e.get("stage") in ("import_mojo", "import_warp") for e in events)
    if setup_failed:
        verdict = "SETUP_FAILURE"      # the probe never ran: says nothing about interop
    elif (not ns) or n_pass == 0:
        verdict = "FAILS"
    elif all_sizes_done and n_pass == len(want) and not fails and scaling.get("no_copy", False):
        verdict = "WORKS"
    else:
        verdict = "PARTIAL"
    if verdict == "FAILS" and not fails:
        fails.append("no size passed every check")
    if verdict == "PARTIAL" and not fails:
        why = [f"n={n}: failed {[k for k, v in per[n]['checks'].items() if v is not True]}" for n in ns if not per[n]["all_ok"]]
        if scaling and not scaling.get("no_copy", False):
            why.append("wrap time scales with size (copy suspected)")
        fails.extend(why or ["incomplete"])
    streams = {n: {"mojo": per[n].get("mojo_stream"), "warp": per[n].get("warp_stream"), "same": per[n].get("same_stream")} for n in ns}
    ctxcheck = {n: {"mojo_ctx": per[n].get("mojo_ctx"), "warp_ctx": per[n].get("warp_ctx"),
                    "owner_ctx": (per[n].get("cu") or {}).get("owner_ctx"),
                    "primary_ctx": (per[n].get("cu") or {}).get("primary_ctx"),
                    "memtype": (per[n].get("cu") or {}).get("memtype")} for n in ns}
    cuda_errors = [f for f in fails if re.search(r"cuda|CUDA|illegal|invalid|context", f)]
    return {"verdict": verdict, "failures": fails, "notes": notes, "per_size": per, "scaling": scaling,
            "streams": streams, "contexts": ctxcheck, "cuda_error_lines": cuda_errors,
            "returncode": returncode, "stderr_tail": stderr_tail[-1500:]}


# ----------------------------------------------------------------------- report
def frozen_prediction():
    p = REPO / "docs" / "ROADMAP.md"
    try:
        text = p.read_text()
        m = re.search(r"\*\*Q6 \(interop.*?\*Prediction:\* (.*?) Decides C", text, re.S)
        if m:
            return " ".join(m.group(1).split())
    except OSError:
        pass
    return FROZEN_FALLBACK


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.6g}"
    return str(x)


def outcome_word(verdict):
    return {"WORKS": "CONFIRMED", "FAILS": "CONFIRMED", "PARTIAL": "REFUTED"}.get(verdict, "NOT COMPARABLE")


def render_report(res, args, wall, cmd, date, sha, selftest=False):
    per, sc = res["per_size"], res["scaling"]
    ns = sorted(per)
    outcome = outcome_word(res["verdict"])
    lines = []
    A = lines.append
    A(f"# Q6 — Mojo -> Warp zero copy, {date}, commit {sha}" + ("  [SELFTEST: synthetic numbers, not a measurement]" if selftest else ""))
    A("")
    A("## 6 What ran")
    A(f"command:        {cmd}")
    A(f"commit:         {sha} (probe files are new and uncommitted: experiments/mojo_warp_interop/)")
    A("seed(s):        n/a (deterministic data: element i = 0.5*i; no randomness)")
    A("inputs:         experiments/mojo_warp_interop/build/interop_probe.so (built from interop_probe.mojo); no stored data")
    A(f"wall:           {fmt(wall, 1)}")
    A(f"effective config: sizes={args.sizes}, wrap reps={args.reps}, H2D contrast reps={args.copy_reps}, context order={args.order}; "
      "wp.from_dlpack not attempted (Mojo DeviceBuffer has no DLPack, F_on_device_tooling.md Q3); 'one stream' reported, not enforced "
      "(Mojo and Warp each use their own stream, ordered by a host synchronize at each hand-off)")
    A("")
    A("## 7 Data quality")
    A(f"n:              {len(ns)} sizes x (1 Mojo fill check, 1 Warp read check, 2 Warp write -> Mojo checksum checks, 1 post-wrapper integrity check)")
    A("dropped:        none" if res["verdict"] != "FAILS" else f"dropped:        sizes without a completed row: {res['failures']}")
    A("artifacts:      checksum is a float64 sum over exactly-representable values (multiples of 0.5 below 2**23), so atomic-add order cannot change it; "
      "equality is exact, no tolerance. Wrap timing is host wall time of the Python call (perf_counter_ns), not a GPU time. "
      "First wrap and 10 warm-ups excluded. Mojo's CUstream/CUcontext come from the private std.gpu.host._nvidia_cuda.CUDA accessor.")
    if res["failures"]:
        A("failures:       " + " | ".join(res["failures"]))
    if res["stderr_tail"].strip():
        A("child stderr tail: " + res["stderr_tail"].strip().replace("\n", " // ")[-600:])
    A("")
    A("## 8 Numbers (the table the ROADMAP item asked for, nothing else)")
    A("| n (float32) | Mojo fill checksum == expected | Warp reads Mojo data | Mojo sees Warp x1 | Mojo sees Warp x2 | intact after wrappers | wrap median us | wrap p10-p90 us | H2D copy median us | wrap / H2D |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for n in ns:
        r = per[n]
        c = r["checks"]
        ratio = (r["wrap_us_median"] / r["h2d_us_median"]) if r.get("wrap_us_median") and r.get("h2d_us_median") else None
        A(f"| {n} | {fmt(c['mojo_fill'])} | {fmt(c['warp_reads_mojo'])} | {fmt(c['mojo_sees_warp_x1'])} | {fmt(c['mojo_sees_warp_x2'])} | "
          f"{fmt(c['intact_after_wrappers'])} | {fmt(r.get('wrap_us_median'), 1)} | {fmt(r.get('wrap_us_p10'), 1)}-{fmt(r.get('wrap_us_p90'), 1)} | "
          f"{fmt(r.get('h2d_us_median'), 1)} | {fmt(ratio, 4)} |")
    A("")
    if sc:
        A(f"wrap scaling:   wrap({sc['n_hi']}) / wrap({sc['n_lo']}) = {fmt(sc['wrap_ratio'], 2)} for a {sc['size_ratio']:.0f}x size increase "
          f"(threshold < {sc['ratio_max']}); wrap({sc['n_hi']}) / H2D({sc['n_hi']}) = {fmt(sc['wrap_over_h2d_at_hi'], 4)} (threshold < {sc['copy_frac_max']}); no copy: {fmt(sc['no_copy'])}")
    else:
        A("wrap scaling:   n/a (fewer than two sizes with timings)")
    A("checksums:      " + "; ".join(f"n={n}: mojo {fmt(per[n].get('checksum_mojo'), 1)} / after Warp x1 {fmt(per[n].get('checksum_after_warp1'), 1)} / after x2 {fmt(per[n].get('checksum_after_warp2'), 1)} (expected {fmt(per[n].get('expect_mojo'), 1)} / {fmt(per[n].get('expect_after_warp1'), 1)} / {fmt(per[n].get('expect_after_warp2'), 1)})" for n in ns))
    A("streams:        " + "; ".join(f"n={n}: Mojo CUstream {res['streams'][n]['mojo']}, Warp stream {res['streams'][n]['warp']}, same: {fmt(res['streams'][n]['same'])}" for n in ns)
      + ". Mojo's stream handle is obtainable (CUDA(ctx.stream()) in std.gpu.host._nvidia_cuda, a private module); DeviceStream has no public handle attribute.")
    A("contexts:       " + "; ".join(f"n={n}: Mojo CUcontext {c['mojo_ctx']}, Warp CUcontext {c['warp_ctx']}, pointer-owner CUcontext {c['owner_ctx']}, primary {c['primary_ctx']}, memtype {c['memtype']} (2 = device)" for n, c in res["contexts"].items()))
    A("CUDA errors / context mismatch: " + (("; ".join(res["cuda_error_lines"])) if res["cuda_error_lines"] else "none recorded") +
      "; child exit status " + str(res["returncode"]))
    A("")
    A("spread:         wrap time: median and p10-p90 over " + str(args.reps) + " calls per size (table); H2D contrast: min-max over " + str(args.copy_reps) + " calls: " +
      "; ".join(f"n={n}: {fmt(per[n].get('h2d_us_min'), 1)}-{fmt(per[n].get('h2d_us_max'), 1)} us" for n in ns) + ". Checksums are exact (no spread). One process, one run.")
    A("")
    A("## 9 Against the frozen prediction (quote it verbatim, then one word)")
    A(f'prediction:     "{frozen_prediction()}"')
    A(f"outcome:        {outcome}")
    A(f"by:             interop verdict {res['verdict']} (WORKS = every check at every size and wrap time does not scale; FAILS = nothing passes; PARTIAL = anything else, which the prediction excludes; SETUP_FAILURE = a module would not import, the probe never ran, NOT COMPARABLE). "
      + (f"Decisive: {sum(1 for n in ns if per[n]['all_ok'])}/{len(ns)} sizes passed all five checks; wrap ratio {fmt(sc.get('wrap_ratio'), 2) if sc else 'n/a'}." ))
    A("")
    A("## 12 Reproducer")
    A("```bash")
    A("cd /home/hundo/Projects/Dytiscidae/dytiscidae/mojo && pixi run mojo build --emit shared-lib -I src \\")
    A("    ../experiments/mojo_warp_interop/interop_probe.mojo \\")
    A("    -o ../experiments/mojo_warp_interop/build/.interop_probe.so.new \\")
    A("  && mv ../experiments/mojo_warp_interop/build/.interop_probe.so.new ../experiments/mojo_warp_interop/build/interop_probe.so")
    A("cd /home/hundo/Projects/Dytiscidae/dytiscidae")
    A("../mjwarp-venv/bin/python experiments/mojo_warp_interop/run.py")
    A("```")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ parent
def parent_main(a):
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="q6_")
    log = os.path.join(tmp, "events.jsonl")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--log", log, "--sizes", ",".join(map(str, a.sizes)),
           "--reps", str(a.reps), "--copy-reps", str(a.copy_reps), "--order", a.order, "--build-dir", a.build_dir]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        rc, err = cp.returncode, cp.stderr
    except subprocess.TimeoutExpired as e:
        rc, err = -9, f"TIMEOUT after {a.timeout}s\n" + ((e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
    events = []
    if os.path.exists(log):
        for line in open(log):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    res = reduce_events(events, rc, err, a.sizes)
    wall = time.time() - t0
    res["events"] = events
    res["args"] = vars(a)
    Path(a.results).write_text(json.dumps(res, indent=1, default=str))
    argv = " ".join(sys.argv)
    text = render_report(res, a, wall, f"{sys.executable} {argv}", time.strftime("%Y-%m-%d"), git_sha())
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(text)
    print(f"Q6 verdict={res['verdict']} outcome={outcome_word(res['verdict'])} "
          f"wrap_ratio={res['scaling'].get('wrap_ratio')} results={a.results} report={a.report}")
    return 0 if res["verdict"] == "WORKS" else 1


# ---------------------------------------------------------------------- selftest
def synth_row(n, ok=True, wrap_us=9.0, h2d_us=None):
    h2d_us = h2d_us if h2d_us is not None else n * 0.0004
    r = dict(n=n, ptr=0x7f0000000000, mojo_ctx=111, mojo_stream=222, warp_ctx=111, warp_stream=333,
             cu={"owner_ctx": 111, "primary_ctx": 111, "memtype": 2}, same_stream=False,
             checksum_mojo=expected_sum(n), expect_mojo=expected_sum(n), ok_mojo_fill=True,
             ok_warp_reads_mojo=True,
             checksum_after_warp1=expected_sum(n, 1) if ok else expected_sum(n), expect_after_warp1=expected_sum(n, 1), ok_mojo_sees_warp1=ok,
             checksum_after_warp2=expected_sum(n, 2), expect_after_warp2=expected_sum(n, 2), ok_mojo_sees_warp2=ok,
             ok_intact_after_wrappers=True, wrap_us_median=wrap_us, wrap_us_p10=wrap_us * 0.9, wrap_us_p90=wrap_us * 1.2,
             h2d_us_median=h2d_us, h2d_us_min=h2d_us * 0.9, h2d_us_max=h2d_us * 1.1)
    return r


def synth_events(rows, finished=True):
    ev = [{"event": "start"}]
    for r in rows:
        ev.append(dict(event="size_done", **r))
    if finished:
        ev.append({"event": "done"})
    return ev


def selftest():
    # 1. args
    a = parse_args(["--sizes", "1e5,1e6,1e7", "--reps", "7", "--order", "warp_first"])
    assert a.sizes == [100000, 1000000, 10000000] and a.reps == 7 and a.order == "warp_first"
    d = parse_args([])
    assert d.sizes == [100000, 1000000, 10000000] and not d.selftest and d.order == "mojo_first"
    import contextlib, io
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--sizes", "0"])
        raise AssertionError("bad sizes accepted")
    except SystemExit:
        pass
    # 2. exactness of the arithmetic the GPU check relies on (numpy float32 is the ground truth)
    import numpy as np
    for n in (100000, 1000000, 10000000):
        assert float32_exact(n)
        v = np.arange(n, dtype=np.float32) * np.float32(0.5) + np.float32(2.0)
        assert float(v.astype(np.float64).sum()) == expected_sum(n, 2), n
    # 3. reducer: WORKS / PARTIAL / FAILS
    sizes = [100000, 1000000, 10000000]
    good = [synth_row(n, wrap_us=9.0) for n in sizes]
    r1 = reduce_events(synth_events(good), 0, "", sizes)
    assert r1["verdict"] == "WORKS" and r1["scaling"]["no_copy"], r1["verdict"]
    scaling_bad = [synth_row(n, wrap_us=n * 0.001, h2d_us=n * 0.0015) for n in sizes]   # wrap tracks size -> copy
    r2 = reduce_events(synth_events(scaling_bad), 0, "", sizes)
    assert r2["verdict"] == "PARTIAL" and not r2["scaling"]["no_copy"], r2["verdict"]
    one_bad = [synth_row(100000), synth_row(1000000, ok=False), synth_row(10000000)]
    r3 = reduce_events(synth_events(one_bad), 0, "", sizes)
    assert r3["verdict"] == "PARTIAL", r3["verdict"]
    allbad = [synth_row(n, ok=False) for n in sizes]
    r4 = reduce_events(synth_events(allbad), 0, "", sizes)
    assert r4["verdict"] == "FAILS", r4["verdict"]
    r5 = reduce_events([{"event": "start"}, {"event": "fail", "stage": "import_mojo", "error": "ImportError('x')"}], 2, "tb", sizes)
    assert r5["verdict"] == "SETUP_FAILURE" and "import_mojo" in r5["failures"][0] and outcome_word(r5["verdict"]) == "NOT COMPARABLE"
    r6 = reduce_events(synth_events(good[:1], finished=False), -11, "Segmentation fault", sizes)   # died after the first size
    assert r6["verdict"] == "PARTIAL" and any("signal 11" in f for f in r6["failures"]), (r6["verdict"], r6["failures"])
    # 4. report writer, all three shapes
    with tempfile.TemporaryDirectory(prefix="q6_selftest_") as td:
        for name, res in (("works", r1), ("partial", r3), ("setup", r5), ("fails", r4)):
            res["events"] = []
            text = render_report(res, a, 12.3, "selftest", "2026-10-10", "abc1234", selftest=True)
            for hdr in ("## 6 What ran", "## 7 Data quality", "## 8 Numbers", "## 9 Against the frozen prediction", "## 12 Reproducer"):
                assert hdr in text, (name, hdr)
            for fld in ("command:", "commit:", "seed(s):", "inputs:", "wall:", "effective config:", "n:", "dropped:", "artifacts:", "spread:", "prediction:", "outcome:", "by:"):
                assert fld in text, (name, fld)
            assert ("outcome:        " + outcome_word(res["verdict"])) in text
            Path(td, f"{name}.md").write_text(text)
    pred = frozen_prediction()
    assert "first try" in pred, pred
    mod = "not built"
    bd = DEFAULT_BUILD / "interop_probe.so"
    if bd.exists():
        sys.path.insert(0, str(DEFAULT_BUILD))
        import interop_probe  # loads the module only; constructing a Probe would touch the GPU
        mod = "loads (" + ",".join(m for m in dir(interop_probe.Probe) if not m.startswith("__")) + ")"
    print(f"selftest ok: args, exact-sum arithmetic (3 sizes), reducer WORKS/PARTIAL/FAILS x6, report writer x4, prediction quoted, interop_probe.so {mod}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.child:
        sys.exit(child_main(args))
    sys.exit(parent_main(args))
