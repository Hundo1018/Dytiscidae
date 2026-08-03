"""Check the per-machine limiter and body scatter against numpy.

The reference runs the Python's own logic once per machine, on that machine's
panels only, which is what the unbatched solver does. If the batched kernel
agrees with that, the batching has not leaked one machine's limit into another.
"""
import numpy as np
import scatter_gpu

from dytiscidae.physics.medium import GRAVITY


def reference_one(F, pos, xipos, body_id, nbody, weight):
    """fluid.py:680-700 for a single machine."""
    F = F.copy()
    clamped = False
    fmag = np.linalg.norm(F, axis=1)
    limit = 60.0 * weight
    if np.any(fmag > limit):
        scale = np.minimum(1.0, limit / np.maximum(fmag, 1e-9))
        F = F * scale[:, None]
        clamped = True
    arm = pos - xipos[body_id]
    T = np.cross(arm, F)
    xfrc = np.zeros((nbody, 6))
    np.add.at(xfrc[:, :3], body_id, F)
    np.add.at(xfrc[:, 3:], body_id, T)
    return xfrc, clamped


def run_gpu(F, machine, limit, pos, xipos, body_id, nbody, nmach):
    n = len(machine)
    xfrc = np.zeros((nbody, 6)); clamped = np.zeros(nmach, dtype=np.int32)
    fmag = np.zeros(n); fmax = np.zeros(nmach)
    desc = np.array(
        [a.ctypes.data for a in (F, machine, limit, pos, xipos, body_id,
                                 xfrc, clamped, fmag, fmax)]
        + [n, nbody, nmach], dtype=np.int64)
    got = scatter_gpu.limit_and_scatter(desc)
    assert got == n, f"kernel reported {got}, expected {n}"
    return xfrc, clamped


def case(name, weights, per_machine_panels, force_scale, seed=0):
    """`weights` is one dry weight per machine; force_scale per machine sets
    whether that machine trips its own limiter."""
    rng = np.random.default_rng(seed)
    C = np.ascontiguousarray
    nmach = len(weights)
    Fs, poss, bids, machs, xips = [], [], [], [], []
    body_off = 0
    per_machine = []
    for m in range(nmach):
        npan = per_machine_panels[m]
        nb = max(3, npan // 8)
        F = rng.normal(scale=force_scale[m], size=(npan, 3))
        pos = rng.normal(scale=1.0, size=(npan, 3))
        xip = rng.normal(scale=0.5, size=(nb, 3))
        bid = rng.integers(0, nb, npan).astype(np.int32)
        per_machine.append((F, pos, xip, bid, nb))
        Fs.append(F); poss.append(pos); xips.append(xip)
        bids.append(bid + body_off); machs.append(np.full(npan, m, np.int32))
        body_off += nb

    F = C(np.concatenate(Fs)); pos = C(np.concatenate(poss))
    xipos = C(np.concatenate(xips)); body_id = C(np.concatenate(bids))
    machine = C(np.concatenate(machs))
    limit = C(np.array([60.0 * (w * GRAVITY + 1.0) for w in weights]))
    nbody = body_off

    g_xfrc, g_clamped = run_gpu(F, machine, limit, pos, xipos, body_id,
                                nbody, nmach)

    # reference: each machine alone, then placed at its own body offset
    r_xfrc = np.zeros((nbody, 6)); r_clamped = np.zeros(nmach, dtype=np.int32)
    off = 0
    for m, (Fm, pm, xm, bm, nb) in enumerate(per_machine):
        x, c = reference_one(Fm, pm, xm, bm, nb, weights[m] * GRAVITY + 1.0)
        r_xfrc[off:off + nb] = x
        r_clamped[m] = int(c)
        off += nb

    sc = max(float(np.abs(r_xfrc).max()), 1e-12)
    e = float(np.abs(r_xfrc - g_xfrc).max()) / sc
    flags_ok = bool((r_clamped == g_clamped).all())
    print(f"{name:34s} xfrc {e:10.3e}   clamped ref={list(r_clamped)} "
          f"gpu={list(g_clamped)} {'ok' if flags_ok else 'MISMATCH'}")
    return e, flags_ok


def main():
    worst, all_flags = 0.0, True

    # Nothing near the limit: forces must be untouched and no flag raised.
    e, f = case("no machine clamps", [5.0, 5.0, 5.0], [400, 400, 400],
                [1.0, 1.0, 1.0], seed=1)
    worst = max(worst, e); all_flags &= f

    # The case the per-machine design exists for: one light machine tumbling
    # among heavy calm ones. A global limit would use the heavy weight.
    e, f = case("one light machine clamps only", [0.5, 40.0, 40.0],
                [400, 400, 400], [1e4, 1.0, 1.0], seed=2)
    worst = max(worst, e); all_flags &= f

    # And the reverse: heavy one clamps, light ones must NOT be scaled.
    e, f = case("one heavy clamps, others clean", [40.0, 0.5, 0.5],
                [400, 400, 400], [1e5, 1.0, 1.0], seed=3)
    worst = max(worst, e); all_flags &= f

    e, f = case("every machine clamps", [1.0, 2.0, 3.0], [500, 300, 700],
                [1e5, 1e5, 1e5], seed=4)
    worst = max(worst, e); all_flags &= f

    e, f = case("uneven panel counts, 8 machines",
                list(np.linspace(0.5, 30.0, 8)),
                [137, 48, 400, 71, 900, 56, 220, 64],
                [1.0, 1e5, 1.0, 1.0, 1e4, 1.0, 1.0, 1.0], seed=5)
    worst = max(worst, e); all_flags &= f

    print()
    ok = worst < 1e-13 and all_flags
    print("GPU limiter/scatter checks passed" if ok
          else f"FAILED (worst {worst:.3e}, flags {all_flags})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
