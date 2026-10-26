# Replay of the first bad step's solver inputs (saved by probe_unstable_split.py with
# DUMP_NPZ=<file>) on a FRESH one-world MJWarp model/data: mjw.forward, is qacc finite?
# Then bisect: mass only / inertia only back to dry, and each body alone at its dumped
# added mass/inertia (others dry). Separates "these inputs break MJWarp's forward" from
# "state MJWarp carried from the previous step". 2026-10-26.
# Usage: probe_unstable_replay.py <plan> <medium> <seed> <npz>
import sys
import numpy as np, warp as wp, mujoco
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.devroll import rollout_wp as R, model_wp as Mw
from dytiscidae.envs.triphibian import Domain

plan, medium, seed, path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
z = np.load(path)
p = build(BODY_PLANS[plan]())
e = R.start_world(p, seed, Domain(medium), 0)
body = Mw.DeviceBody([e], dtype=wp.float32)
m, d, mjm = body.m, body.d, body.mjm
dry_m, dry_i = mjm.body_mass.copy(), mjm.body_inertia.copy()
names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}" for b in range(mjm.nbody)]
jnt = [int(mjm.body_jntnum[b]) for b in range(mjm.nbody)]


def forward(bm, bi, warm=True):
    d.qpos.assign(z["qpos"][None].astype(np.float32))
    d.qvel.assign(z["qvel"][None].astype(np.float32))
    d.ctrl.assign(z["ctrl"][None].astype(np.float32))
    d.xfrc_applied.assign(np.asarray(z["xfrc_applied"], np.float32).reshape(1, -1, 6))
    d.qfrc_applied.assign(z["qfrc_applied"][None].astype(np.float32))
    ws = z["qacc_warmstart"] if warm else np.zeros_like(z["qacc_warmstart"])
    d.qacc_warmstart.assign(ws[None].astype(np.float32))
    m.dof_damping.assign(z["dof_damping"][None].astype(np.float32))
    m.body_mass.assign(np.asarray(bm, np.float32)[None])
    m.body_inertia.assign(np.asarray(bi, np.float32).reshape(1, -1, 3))
    Mw._mjw.forward(m, d)
    q = d.qacc.numpy()[0]
    M = d.qM.numpy()[0] if hasattr(d, "qM") else None
    return bool(np.isfinite(q).all()), float(np.nanmax(np.abs(q))), M


bm, bi = z["body_mass"], z["body_inertia"]
print(f"REP {plan} {medium} dumped step={int(z['step'])} world={int(z['world'])} nbody={mjm.nbody} "
      f"nv={mjm.nv}", flush=True)
for b in range(mjm.nbody):
    print(f"REP   body {b:2d} {names[b]:14s} jnt={jnt[b]} dry_m={dry_m[b]:.4g} m={bm[b]:.4g} "
          f"dry_I={np.round(dry_i[b], 5).tolist()} I={np.round(bi[b], 5).tolist()}", flush=True)
for label, args in [("dumped", (bm, bi)), ("dumped, warmstart 0", (bm, bi, False)),
                    ("mass dry", (dry_m, bi)), ("inertia dry", (bm, dry_i)),
                    ("all dry", (dry_m, dry_i))]:
    ok, mx, _ = forward(*args)
    print(f"REP case {label:22s} finite={ok} max|qacc|={mx:.4g}", flush=True)
for b in range(1, mjm.nbody):
    m1, i1 = dry_m.copy(), dry_i.copy()
    m1[b], i1[b] = bm[b], bi[b]
    ok1, mx1, _ = forward(m1, i1)
    m2, i2 = bm.copy(), bi.copy()
    m2[b], i2[b] = dry_m[b], dry_i[b]
    ok2, mx2, _ = forward(m2, i2)
    print(f"REP body {b:2d} {names[b]:14s} jnt={jnt[b]} only-this-loaded finite={ok1} "
          f"({mx1:.3g})  all-but-this-loaded finite={ok2} ({mx2:.3g})", flush=True)

# Where in forward the NaN first appears: every Data array of world 0 after the dumped-case
# forward, finite or not, in MJWarp's pipeline order where the name says it; and the host's
# mj_forward on the same inputs for the smooth-dynamics quantities.
import copy
forward(bm, bi)
bad, good = [], []
for k in sorted(vars(d)):
    a = getattr(d, k)
    if isinstance(a, wp.array) and a.dtype in (wp.float32, wp.vec3, wp.mat33, wp.spatial_vector,
                                                wp.quat) and a.size > 0:
        x = a.numpy()
        (bad if not np.isfinite(x).all() else good).append(k)
print("REP nonfinite after forward:", bad, flush=True)
hm = copy.deepcopy(mjm); hm.body_mass[:] = bm; hm.body_inertia[:] = bi
hm.dof_damping[:] = z["dof_damping"]
hd = mujoco.MjData(hm)
for k in ("qpos", "qvel", "ctrl", "qfrc_applied", "qacc_warmstart"):
    getattr(hd, k)[:] = z[k]
hd.xfrc_applied[:] = np.asarray(z["xfrc_applied"], float).reshape(-1, 6)
mujoco.mj_forward(hm, hd)
for k in ("subtree_com", "cinert", "cdof", "qfrc_bias", "qfrc_passive", "qfrc_smooth", "qacc_smooth",
          "qacc", "ncon", "nefc"):
    hv = np.asarray(getattr(hd, k), float)
    dv = getattr(d, k, None)
    dv = None if dv is None else dv.numpy()
    if k in ("ncon", "nefc"):
        print(f"REP host {k}={hv}  device={None if dv is None else dv.ravel()[:2]}", flush=True)
        continue
    dvv = None if dv is None else np.asarray(dv[0], float).reshape(hv.shape) if dv[0].size == hv.size else None
    diff = None if dvv is None else float(np.nanmax(np.abs(dvv - hv)))
    print(f"REP {k:14s} host finite={np.isfinite(hv).all()} max={np.abs(hv).max():.4g}  device "
          f"finite={None if dv is None else bool(np.isfinite(dv).all())} max|dev-host|={diff}",
          flush=True)
print("REP device nacon", d.nacon.numpy(), "nefc", d.nefc.numpy() if hasattr(d, "nefc") else None,
      "host ncon", hd.ncon, "nefc", hd.nefc, flush=True)

# Confirmation: the same dumped inputs with body_subtreemass recomputed from the loaded
# body_mass (what mj_setConst would give) instead of the dry value put_model baked in.
def subtree(bm_):
    sub = np.asarray(bm_, float).copy()
    for b in range(mjm.nbody - 1, 0, -1):
        sub[mjm.body_parentid[b]] += sub[b]
    return sub
dry_sub = m.body_subtreemass.numpy().copy()
print(f"REP subtreemass root dry={dry_sub[0, 1]:.4g} consistent={subtree(bm)[1]:.4g}", flush=True)
for warm in (True, False):
    m.body_subtreemass.assign(subtree(bm)[None].astype(np.float32))
    ok, mx, _ = forward(bm, bi, warm)
    print(f"REP case dumped + consistent subtreemass warm={warm}: finite={ok} max|qacc|={mx:.4g} "
          f"max|subtree_com|={np.abs(d.subtree_com.numpy()).max():.4g} "
          f"max|cinert|={np.abs(d.cinert.numpy()).max():.4g} "
          f"max|qacc-host|={np.abs(d.qacc.numpy()[0] - hd.qacc).max():.3g}", flush=True)
m.body_subtreemass.assign(dry_sub)
