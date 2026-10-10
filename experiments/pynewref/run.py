"""D1 (IMPL_1010_SPEC §2): is the worker `Py_NewRef` abort a PATH-dependent libpython choice
inherited through the C environment?  Diagnosis only, no production change.

Driver:  flock /home/hundo/.cache/dytiscidae-gpu.lock python experiments/pynewref/run.py
Every arm runs in a fresh interpreter (the venv python) launched under one of two PATHs.
"""
import ctypes, json, multiprocessing as mp, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = Path("/home/hundo/Projects/Dytiscidae/dytiscidae")
BUILD = os.environ.get("DYTISCIDAE_KERNEL_DIR") or str(MAIN / "mojo" / "build")
PY = str(MAIN / ".venv" / "bin" / "python")
KEYS = ("PYTHONEXECUTABLE", "MOJO_PYTHON_LIBRARY")

CONSTRUCT = ("import sys;sys.path.insert(0,%r);import full_pipeline as f;"
             "f.FullPipeline(64,64,4);print('CONSTRUCT_OK')" % BUILD)


def cenv():
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    return {k: (libc.getenv(k.encode()) or b"").decode() or None for k in KEYS}


def construct_in_spawn_child():          # target of a spawn child (arms c, d)
    sys.path.insert(0, BUILD)
    import full_pipeline as f
    f.FullPipeline(64, 64, 4)
    print("CONSTRUCT_OK", flush=True)


def child_main(arm):
    sys.path.insert(0, BUILD)
    out = {"arm": arm}
    if arm in ("a", "b", "c"):
        import full_pipeline  # noqa: F401
        out["cenv_after_import"] = cenv()
    if arm in ("a", "b"):
        env = dict(os.environ) if arm == "b" else None
        r = subprocess.run([sys.executable, "-c", CONSTRUCT], capture_output=True, text=True,
                           timeout=240, env=env)
        out.update(rc=r.returncode, ok="CONSTRUCT_OK" in r.stdout, err=r.stderr.strip()[-300:])
    else:
        ctx = mp.get_context("spawn")
        if arm == "d":                   # spawn first, then import in the parent
            p = ctx.Process(target=construct_in_spawn_child)
            p.start(); p.join(240)
            import full_pipeline  # noqa: F401
            out["cenv_after_import"] = cenv()
        else:
            p = ctx.Process(target=construct_in_spawn_child)
            p.start(); p.join(240)
        out.update(rc=p.exitcode, ok=p.exitcode == 0)
    print("RESULT " + json.dumps(out), flush=True)


def run_arm(arm, path_mode):
    pyenv = os.path.expanduser("~/.pyenv/shims")
    venv = str(MAIN / ".venv" / "bin")
    base = os.environ["PATH"].split(":")
    base = [p for p in base if p not in (pyenv, venv)]
    path = ":".join(([pyenv] if path_mode == "pyenv" else [venv]) + base)
    env = dict(os.environ, PATH=path, MUJOCO_GL="disable", DYTISCIDAE_KERNEL_DIR=BUILD)
    env.pop("MOJO_PYTHON_LIBRARY", None); env.pop("PYTHONEXECUTABLE", None)
    t = time.time()
    r = subprocess.run([PY, str(HERE / "run.py"), "--child", arm], capture_output=True, text=True,
                       timeout=600, env=env, cwd=str(ROOT))
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    res = json.loads(line[-1][7:]) if line else {"arm": arm, "rc": r.returncode, "ok": False}
    res["path_mode"] = path_mode; res["wall_s"] = round(time.time() - t, 1)
    res["abort_py_newref"] = "Py_NewRef" in (r.stdout + r.stderr)
    res["stderr_tail"] = r.stderr.strip()[-200:]
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child_main(sys.argv[2]); sys.exit(0)
    results = []
    for mode in ("pyenv", "venv"):
        for arm in "abcd":
            r = run_arm(arm, mode); results.append(r)
            print(json.dumps(r), flush=True)
    (HERE / "results_arm_abcd.json").write_text(json.dumps(results, indent=1))
