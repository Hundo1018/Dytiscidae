"""Hash of the T4, T4c and T5 builds (model arrays, panels, 200 steps) so that nothing in them moves.  Prints three digests."""
import hashlib, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def digest(mc, steps=200):
    h = hashlib.sha256()
    m = mc.m
    for nm in ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat", "jnt_range",
               "jnt_axis", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "geom_solref"):
        h.update(np.ascontiguousarray(getattr(m, nm)).tobytes())
    p = mc.env.solver.panels
    for nm in ("pos_local", "chord", "dr", "volume", "body_id"):
        h.update(np.ascontiguousarray(getattr(p, nm)).tobytes())
    q = np.zeros(len(mc.aid))
    for _ in range(steps):
        mc.step(q)
    h.update(np.ascontiguousarray(mc.d.qpos).tobytes())
    return h.hexdigest()[:16]


def main():
    import s0b_machine as T
    import s0b_t5_machine as T5
    pose = dict(x=-30.0, y=0.0, z=0.3, vx=0.0, vz=0.0)
    out = {}
    out["T4"] = digest(T.make("water", b=1.0, motor="x4", pose=pose, sweep_range_deg=75.0, feather_range_deg=(-150.0, 60.0)))
    out["T4c"] = digest(T.make("water", b=None, motor="x4", pose=pose, root_cut=0.11, sweep_range_deg=75.0, feather_range_deg=(-150.0, 60.0)))
    out["T5"] = digest(T5.make("water", motor="x4", fill=0.0, pose=pose))
    out["T5_fill95"] = digest(T5.make("water", motor="x4", fill=0.95, pose=pose))
    print(out)


if __name__ == "__main__":
    main()
