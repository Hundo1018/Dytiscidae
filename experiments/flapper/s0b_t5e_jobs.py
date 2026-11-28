"""Job lists for the T5e probes (s0b_t5e_probe.py).  `python s0b_t5e_jobs.py <name>` writes results/t5e/<name>.json."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = (("default", True), ("wetting", "wetting"))
ZF = 0.215                      # hull origin over the water at the settled float (S table, fill 0)
P0 = {"flip": "vert", "A": 0.9599310885968813, "dih": 0.6108652381980153, "Psi": 0.7853981633974483}
PITCH = dict(ch="tail", k=-1.0, kd=-0.3, lim=0.5, ff=0.0)
PFD = dict(k=0.2, kd=0.3, lim=0.15)
RECIPE = dict(kind="surface", T=150.0, blend_s=3.0, afloat_z=-9.0,
              pitch_loop=dict(sw0=0.45, kp=3.0, kd=4.0, lim=[-0.3, 0.75], rate=1.0),
              roll_loop=None,
              heave=dict(z_rest=-0.10, v_max=0.06, kz=0.5, kp=1.0, ki=0.2, u_max=0.6, fill_min=0.3, z_tol=0.03, v_tol=0.015, dwell_s=2.0, rate_float=0.1))


def fly(kick=False, hold=10.0, land=True, **kw):
    ph = dict(kind="fly", H=2.0, hold_s=hold, land=land, after=5.0, v_dn=0.15, out_s=1.5)
    if kick:
        ph["kick"] = dict(roll_rate=1.0, vy=0.3)
    ph.update(kw)
    return ph


def swim(T=60.0, f=0.7, v=0.45, a_max=1.0, a_min=0.3, **kw):
    ph = dict(kind="swim", dir=1.0, T=T, rev=False, P={}, v=v, a0=0.5, a_min=a_min, a_max=a_max, kp_v=0.5, ki_v=0.1, f=f, Phi_deg=20.0,
              yaw=dict(ch="Psi_lr", k=-0.3, kd=-0.45, lim=0.3), pitch=PITCH, pitch_from_depth=PFD)
    ph.update(kw)
    return ph


def uturn(T=80.0, a=0.6, f=0.7, gain=1.0 / 0.3, **kw):
    return swim(T=T, f=f, turn_deg=180.0, uturn=a, uturn_deg=25.0, yaw_amp_lr_gain=gain, until_facing_deg=15.0, until_w=0.15, **kw)


def surface(**kw):
    ph = json.loads(json.dumps(RECIPE))
    for k, v in kw.items():
        if isinstance(v, dict) and k in ph:
            ph[k].update(v)
        else:
            ph[k] = v
    return ph


def mission(id_, ent, phases, start="float", **kw):
    j = dict(fn="mission4", id=id_, entrainment=ent, start=start, phases=phases, fill_rate=0.1, T=kw.pop("T", 600.0),
             regain_at=["swim", "fly", "surface"], **kw)
    if start == "float":
        j.setdefault("z_float", ZF)
    return j


def deep(id_, ent, phases, depth=2.5, yaw0=30.0, pose0="flat", **kw):
    return mission(id_, ent, [dict(kind="ballast", depth=depth, T=1.0)] + phases, start="deep", depth=depth, yaw0_deg=yaw0, pose0=pose0, **kw)


JOBS = {}


def reg(name):
    def d(fn):
        JOBS[name] = fn
        return fn
    return d


@reg("b24_float")
def _b24_float():
    out = []
    for nm, ent in MODES:
        out.append(dict(fn="floatq", probe="B24", id=f"b24_float_{nm}", mode=nm, entrainment=ent, T=12.0, z0=ZF,
                        P={"dih": P0["dih"], "sw0": 0.0986},
                        push=[["roll+20", "roll", 20.0, 8.0], ["pitch_nose_down20", "pitch", 20.0, 8.0], ["pitch_nose_up20", "pitch", -20.0, 8.0]]))
        for tag, amp in (("hover", 0.761), ("max", 0.9)):
            out.append(dict(fn="held", probe="B24", id=f"b24_stroke_on_float_{tag}_{nm}", f=4.5, medium="water", z_hull=ZF, mode=nm, entrainment=ent,
                            P=dict(P0, sw0=0.0986, amp=amp)))
    return out


@reg("b24_u1")
def _b24_u1():
    return [mission(f"u01_kick_{nm}", ent, [dict(kind="settle", T=3.0), fly(kick=True)], z_float=ZF + 0.018) for nm, ent in MODES]


def after_surface(hold=5.0):
    return [dict(kind="pose", to="raised", T=2.0), dict(kind="settle", T=4.0), fly(hold=hold, land=False)]


@reg("b25_u1")
def _b25_u1():
    out = []
    for nm, ent in MODES:
        out.append(deep(f"u02_surf_{nm}", ent, [surface()] + after_surface(), depth=1.5, yaw0=0.0, tank_eps=0.02, roll0_deg=3.0))
    return out


@reg("b24_u2")
def _b24_u2():
    nk = [dict(kind="settle", T=3.0), fly(hold=4.0, land=False)]
    return [mission("u03_diag_wideclip_default", True, nk, z_float=ZF + 0.018, lim_hover=dict(sw0=(-0.25, 0.45), amp_lr_hi=0.3), diagnostic="clip widened beyond the clearance range: not evidence"),
            mission("u04_soft_default", True, nk, z_float=ZF + 0.018, law_gains=dict(wn=[6.0, 4.0, 3.0], k_v=0.3)),
            mission("u05_soft2_default", True, nk, z_float=ZF + 0.018, law_gains=dict(wn=[6.0, 3.0, 3.0], k_v=0.2, k_p=0.05))]


@reg("b24_u3")
def _b24_u3():
    nk = [dict(kind="settle", T=3.0), fly(hold=4.0, land=False)]
    kk = [dict(kind="settle", T=3.0), fly(kick=True)]
    return [mission("u06_nokick_default", True, nk, z_float=ZF + 0.018), mission("u07_kick_default", True, kk, z_float=ZF + 0.018),
            mission("u08_kick_wetting", "wetting", kk, z_float=ZF + 0.018)]


@reg("b24_u4")
def _b24_u4():
    nk = lambda **kw: [dict(kind="settle", T=3.0), fly(hold=4.0, land=False, **kw)]
    return [mission("u09_diag_wideclip_default", True, nk(), z_float=ZF + 0.018, lim_hover=dict(sw0=(-0.25, 0.45), sw0_lo=(-0.25, 0.45)), diagnostic="clip widened beyond the clearance range: not evidence"),
            mission("u10_slowclimb_default", True, nk(v_up=0.3), z_float=ZF + 0.018, law_gains=dict(v_c=0.3)),
            mission("u11_ramp8_default", True, nk(v_up=0.3, ramp_beats=8.0), z_float=ZF + 0.018, law_gains=dict(v_c=0.3))]


@reg("b24_u5")
def _b24_u5():
    kk = [dict(kind="settle", T=3.0), fly(kick=True)]
    return [mission("u12_kick_default", True, kk, z_float=ZF + 0.018), mission("u13_kick_wetting", "wetting", kk, z_float=ZF + 0.018)]


@reg("b24_rep")
def _b24_rep():
    out = []
    for nm, ent in MODES:
        out.append(mission(f"b24_kick_{nm}", ent, [dict(kind="settle", T=3.0), fly(kick=True)]))
        out.append(mission(f"b24_nokick_{nm}", ent, [dict(kind="settle", T=3.0), fly()]))
    return out


@reg("b25_u2")
def _b25_u2():
    return [deep("u14_surf_default", True, [surface()] + after_surface(), depth=1.5, yaw0=0.0, tank_eps=0.02, roll0_deg=3.0)]


@reg("b25_u3")
def _b25_u3():
    out = []
    for nm, ent in MODES:
        out.append(deep(f"u15_surf_noroll_{nm}", ent, [surface(roll_loop=None)] + [dict(kind="pose", to="raised", T=2.0), dict(kind="settle", T=4.0)], depth=1.5, yaw0=0.0, tank_eps=0.02, roll0_deg=3.0))
    return out


SLOW = dict(v_c=0.4)


@reg("b24_u6")
def _b24_u6():
    kk = lambda **kw: [dict(kind="settle", T=3.0), fly(kick=True, **kw)]
    return [mission("u16_kick_default", True, kk(), z_float=ZF + 0.018),
            mission("u17_kick_slow_default", True, kk(v_up=0.4), z_float=ZF + 0.018, law_gains=SLOW),
            mission("u18_kick_slow_wetting", "wetting", kk(v_up=0.4), z_float=ZF + 0.018, law_gains=SLOW)]


@reg("b25_u4")
def _b25_u4():
    kw = dict(depth=1.5, yaw0=0.0, tank_eps=0.02, roll0_deg=3.0)
    return [deep("u19_surf_fly_default", True, [surface()] + after_surface(), **kw),
            deep("u20_surf_fly_wetting", "wetting", [surface()] + after_surface(), **kw),
            deep("u21_surf_flyslow_default", True, [surface()] + [dict(kind="pose", to="raised", T=2.0), dict(kind="settle", T=4.0), fly(hold=5.0, land=False, v_up=0.4)], law_gains=SLOW, **kw)]


@reg("b27_u1")
def _b27_u1():
    return [deep(f"u22_swim_uturn_{nm}", ent, [swim(T=20.0), uturn(), swim(T=20.0)], depth=2.5, yaw0=0.0) for nm, ent in MODES]


STARTS = dict(S7=dict(depth=2.8, tank_eps=0.02, roll0_deg=3.0, yaw0=20.0), S8=dict(depth=2.0, tank_eps=0.03, pitch0_deg=4.0, yaw0=-40.0),
              S9=dict(depth=3.5, tank_eps=0.025, roll0_deg=-4.0, pitch0_deg=3.0, yaw0=100.0))


@reg("b25_rep")
def _b25_rep():
    return [deep(f"b25_{sn}_{nm}", ent, [surface()] + after_surface(), **st) for sn, st in STARTS.items() for nm, ent in MODES]


@reg("b24_final")
def _b24_final():
    out = []
    for nm, ent in MODES:
        out.append(mission(f"b24r_kick_{nm}", ent, [dict(kind="settle", T=5.0), fly(kick=True)], z_float=ZF + 0.005))
        out.append(mission(f"b24r_nokick_{nm}", ent, [dict(kind="settle", T=5.0), fly()], z_float=ZF + 0.005))
    return out


def uturn2(a=0.4, rate=5.0, T=80.0, gain=1.0 / 0.3, **kw):
    return swim(T=T, turn_deg=180.0, yaw_ramp_deg_s=rate, a_const=a, yaw_amp_lr_gain=gain, until_facing_deg=12.0, until_w=0.10, **kw)


@reg("b27_u2")
def _b27_u2():
    out = []
    for nm, ent in MODES:
        out.append(deep(f"u23_uturn_a04_{nm}", ent, [swim(T=8.0, v=0.36), uturn2(a=0.4)], depth=2.5, yaw0=0.0))
        out.append(deep(f"u24_uturn_a03_{nm}", ent, [swim(T=8.0, v=0.36), uturn2(a=0.3, pitch_from_depth=dict(k=0.2, kd=0.3, lim=0.08))], depth=2.5, yaw0=0.0))
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    jobs = JOBS[name]()
    (HERE / "results" / "t5e").mkdir(exist_ok=True)
    (HERE / "results" / "t5e" / f"{name}.json").write_text(json.dumps(jobs))
    print(name, len(jobs), [j["id"] for j in jobs])
