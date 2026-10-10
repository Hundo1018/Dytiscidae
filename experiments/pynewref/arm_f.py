"""Arm f: run_search order without run_search.  Parent imports batchroll and calls usable(); a spawn
worker then imports batchroll and calls usable().  Prints the C environment at each step.
usage: arm_f.py [parent_usable=1] [worker_import_first=0]"""
import ctypes, multiprocessing as mp, os, sys, json
KEYS = ("PYTHONEXECUTABLE", "MOJO_PYTHON_LIBRARY", "PATH", "LD_LIBRARY_PATH", "PYTHONHOME")
def cenv():
    libc = ctypes.CDLL(None); libc.getenv.restype = ctypes.c_char_p
    return {k: ((libc.getenv(k.encode()) or b"").decode()[:90] or None) for k in KEYS}
def worker():
    print("WORKER before import", json.dumps(cenv()), "exe", sys.executable, flush=True)
    from dytiscidae.envs import batchroll as b
    print("WORKER after import ", json.dumps(cenv()), flush=True)
    print("WORKER usable", b.usable(), flush=True)
if __name__ == "__main__":
    pu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    from dytiscidae.envs import batchroll as b
    print("PARENT after import ", json.dumps(cenv()), "exe", sys.executable, flush=True)
    fix = os.environ.get("D1_FIX", "")
    if fix == "unsetenv": os.unsetenv("PYTHONEXECUTABLE")          # C level; os.environ never saw it
    if fix == "putenv": os.environ["PYTHONEXECUTABLE"] = sys.executable
    if pu: print("PARENT usable", b.usable(), flush=True)
    p = mp.get_context("spawn").Process(target=worker); p.start(); p.join(300)
