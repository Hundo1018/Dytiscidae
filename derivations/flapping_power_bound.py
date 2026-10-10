"""Every number in derivations/flapping_power_bound.md (ROADMAP N7).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python derivations/flapping_power_bound.py

Analytic. Each seed plan is built (`phenotype.build`) and compiled
(`mjcf.compile_phenotype`) only to read what the code says the machine is:
mass, wing area, span, each stroke joint's torque limit, damping and inertia
(one `mj_forward` at the build pose for the joint-space mass matrix; nothing is
stepped). No rollout, no GPU.

The constants the derivation restates are listed in SOURCES with the line they
come from. The script fails if a cited line no longer says what it is cited
for, so the document cannot drift from the code silently.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (path, line, substring that must be on that line, what the document uses it for)
SOURCES = [
    ("dytiscidae/core/genome.py", 384, "rng.uniform(1.5, 12.0)", "random_genome frequency band"),
    ("dytiscidae/core/genome.py", 620, "rng.uniform(1.5, 12.0)", "mut_gait frequency band"),
    ("dytiscidae/core/genome.py", 563, "0.0, 1.0", "stroke_amplitude clipped to [0, 1] of half-travel"),
    ("dytiscidae/core/genome.py", 461, "0.05, 2.6", "mut_joint: joint limit up to 2.6 rad"),
    ("dytiscidae/core/genome.py", 473, "0.001, 3.0", "mut_actuator: motor mass range"),
    ("dytiscidae/core/genome.py", 475, "1.0, 200.0", "mut_actuator: gear ratio range"),
    ("dytiscidae/core/genome.py", 441, "rng.normal(1.0, 0.25), 0.0, 3.0", "mut_drivetrain: tuned spring"),
    ("dytiscidae/control/cpg.py", 89, "amp = np.minimum(np.maximum(self.amplitude", "amplitude <= half-travel"),
    ("dytiscidae/control/cpg.py", 171, "p.offset + p.amplitude * np.sin(arg)", "sinusoidal stroke"),
    ("dytiscidae/control/cpg.py", 170, "self._rate = p.amplitude * (2.0 * np.pi * f)", "rate fed forward"),
    ("dytiscidae/physics/energy.py", 46, "self.p_cont = self.spec.specific_power * self.mass", "continuous rating"),
    ("dytiscidae/physics/energy.py", 49, "self.eta_gear = 0.97**stages", "gear efficiency"),
    ("dytiscidae/physics/energy.py", 53, "self.km = 0.05 * (self.mass / 0.1) ** 0.75", "motor constant"),
    ("dytiscidae/physics/energy.py", 61, "self.km * np.sqrt(self.p_cont) * self.gear_ratio * self.eta_gear", "stall torque"),
    ("dytiscidae/physics/energy.py", 66, "self.p_cont / max(self.stall_torque", "rated speed (not enforced)"),
    ("dytiscidae/core/mjcf.py", 686, "np.clip(frange, 0.02, 400.0)", "forcerange clip"),
    ("dytiscidae/core/mjcf.py", 715, '"kp": _fmt(max(2.0 * fr * gain', "kp = 2 tau g"),
    ("dytiscidae/core/mjcf.py", 716, '"kv": _fmt(max(0.15 * fr * gain', "kv = 0.15 tau g"),
    ("dytiscidae/core/mjcf.py", 717, '"forcerange"', "torque limit = stall torque"),
    ("dytiscidae/core/mjcf.py", 567, '"damping": _fmt(0.02 + 0.5 * s.mass)', "stroke-joint damping"),
    ("dytiscidae/core/mjcf.py", 568, '"armature": _fmt(max(1e-4, 0.01 * s.mass))', "stroke-joint armature"),
    ("dytiscidae/core/mjcf.py", 585, "k = ratio * s.joint_inertia * omega**2", "spring tuned on phenotype inertia"),
    ("dytiscidae/envs/triphibian.py", 659, "self.servo_lead = (np.where(bp[:, 1] < 0, bp[:, 2]", "lead = kv/kp"),
    ("dytiscidae/envs/triphibian.py", 1746, "return tgt + self.servo_lead[: len(tgt)] * rate", "feed-forward"),
    ("dytiscidae/physics/structure.py", 179, "ang_acc = flap_amplitude_rad * omega**2", "spar check scales f^2"),
    ("dytiscidae/physics/structure.py", 182, "i_root = wing_mass * (0.45 * semi_span) ** 2", "spar check inertia"),
    ("dytiscidae/core/phenotype.py", 899, "flap_freq=g.flap_frequency", "spar check at the genome frequency"),
    ("dytiscidae/core/phenotype.py", 900, "flap_amplitude_rad=abs(s.part.joint_range[1]", "spar check at half-travel"),
    ("dytiscidae/core/phenotype.py", 564, "if s.kind in (WING_KIND, MEMBRANE, FIN):", "what counts as wing area"),
    ("dytiscidae/core/phenotype.py", 312, "return self.max_span**2 / self.wing_area", "aspect ratio"),
    ("dytiscidae/core/phenotype.py", 774, "frontal_area = max(0.25 * math.pi", "frontal area"),
    ("dytiscidae/core/phenotype.py", 1128, "cd = {HULL: 0.20", "hull bluff CD"),
    ("dytiscidae/core/phenotype.py", 766, "m_j = d.mass + (d.actuator.mass", "phenotype joint inertia"),
    ("dytiscidae/physics/fluid.py", 345, "oswald = 0.75", "span efficiency"),
    ("dytiscidae/physics/fluid.py", 346, "cd_i = cl * cl / (np.pi * oswald", "induced drag"),
    ("dytiscidae/physics/fluid.py", 242, "def skin_friction_cd", "profile drag"),
    ("dytiscidae/physics/fluid.py", 274, "CN_LEV = 3.4", "separated-branch CN"),
    ("dytiscidae/physics/medium.py", 38, 'AIR = Fluid("air", 1.225, 1.81e-5)', "air"),
    ("dytiscidae/physics/medium.py", 20, "GRAVITY = 9.80665", "g"),
]


def check_sources() -> None:
    bad = []
    for path, line, needle, _ in SOURCES:
        text = (ROOT / path).read_text().splitlines()
        if line > len(text) or needle not in text[line - 1]:
            bad.append(f"{path}:{line} no longer contains {needle!r}")
    if bad:
        raise SystemExit("cited source lines moved:\n  " + "\n  ".join(bad))
    print(f"sources: {len(SOURCES)} cited lines checked")


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/flight_audit"))

import mujoco  # noqa: E402

from dytiscidae.core.bodyplans import BODY_PLANS  # noqa: E402
from dytiscidae.core.genome import FIN, MEMBRANE, WING  # noqa: E402
from dytiscidae.core.mjcf import compile_phenotype  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.physics.fluid import CN_LEV, skin_friction_cd  # noqa: E402
from dytiscidae.physics.medium import AIR, GRAVITY  # noqa: E402

RHO, NU = AIR.rho, AIR.mu / AIR.rho
OSWALD = 0.75                 # fluid.py:345
CD_HULL = 0.20                # phenotype.py:1128
F_LO, F_HI = 1.5, 12.0        # genome.py:384, 620
GEAR_MAX, FR_MAX = 200.0, 400.0   # genome.py:475, mjcf.py:686
LIFT_KINDS = (WING, MEMBRANE, FIN)  # phenotype.py:564: the surfaces S counts
FOUR_OVER_PI = 4.0 / math.pi  # largest fundamental of a function bounded by 1

# The teal level gait of 2026-09-26 (level_flight.py's best; rig_level.py flew it).
# A pickle this project wrote itself (experiments/flight_audit/level_flight.py),
# read the same way rig_level.py and probe_actuation.py read it.
GAIT_PKL = ROOT / "runs/_logs/level_best_teal_feathering.pkl"
RIG = {"margin_free": 0.424, "margin_prescribed": 1.276, "torque_limited": 0.60,
       "reach_lo": 0.19, "reach_hi": 0.39, "f": 11.33, "U": 6.8}   # plan_0923b.md:35-36


def feathered(name):
    """As experiments/flight_audit/feather_fly.py: every actuated surface hinge
    becomes a stroke + feathering pair."""
    g = BODY_PLANS[name]()
    for part in g.parts:
        if part.joint == "hinge" and part.actuated and part.is_surface:
            part.joint = "universal"
    return g


# ----------------------------------------------------------------- the machine

def machine(label, genome):
    p = build(genome)
    model, data, names, panels = compile_phenotype(p)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)

    segs = {s.index: s for s in p.segments}
    kids: dict[int, list[int]] = {}
    for s in p.segments:
        kids.setdefault(s.parent, []).append(s.index)

    def carries_lift(i):
        return segs[i].kind in LIFT_KINDS or any(carries_lift(c) for c in kids.get(i, ()))

    by_name = {s.name: s for s in p.segments}
    joints = []
    for k, an in enumerate(names):
        if not an.endswith("_a"):
            continue           # _f feathering motors, _r rotors: not stroke actuators
        s = by_name[an[:-2]]
        if not carries_lift(s.index):
            continue           # legs, feet, paddles: no lifting surface downstream
        j = model.actuator_trnid[k, 0]
        dof = model.jnt_dofadr[j]
        lo, hi = model.jnt_range[j]
        a = s.actuator
        # Lifting strips this joint moves, with their distance from its axis
        # (build pose): the load side of the bound, `load_power`.
        jb = int(model.jnt_bodyid[j])
        sub = set()
        for bid in range(model.nbody):
            x = bid
            while x > 0 and x != jb:
                x = int(model.body_parentid[x])
            if x == jb:
                sub.add(bid)
        ax, anc = data.xaxis[j].copy(), data.xanchor[j].copy()
        rs, cdr = [], []
        for n in range(panels.n):
            if panels.kind[n] != 0 or int(panels.body_id[n]) not in sub:
                continue
            bid = int(panels.body_id[n])
            xw = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ panels.pos_local[n]
            r = xw - anc
            rs.append(float(np.linalg.norm(r - (r @ ax) * ax)))
            cdr.append(float(panels.chord[n] * panels.dr[n]))
        joints.append(dict(
            r=np.array(rs), cdr=np.array(cdr), main=float(np.sum(cdr)) > 0.05,   # a tail or trim tab carries <= 0.04 m2
            name=an, tau=float(model.actuator_forcerange[k, 1]),
            I=float(M[dof, dof]), I_phen=float(s.joint_inertia),
            d=float(model.dof_damping[dof]), A_max=float(0.5 * (hi - lo)),
            gear=float(a.gear_ratio), p_cont=float(a.p_cont),
            w_rated=float(a.max_speed), act_index=k))

    # Spar check (Tier 0): applied stress goes as f^2 at fixed half-travel, so
    # the band's top is f_genome * sqrt(allowable / applied) for the worst wing.
    f0 = genome.flap_frequency
    spar = [c for c in p.report.checks if c.name == "spar_inertial_reversal"]
    f_spar = min((f0 * math.sqrt(c.allowable / c.applied) for c in spar if c.applied > 0),
                 default=float("inf"))
    return dict(label=label, m=float(p.mass), S=float(p.wing_area), b=float(p.max_span),
                AR=float(p.aspect_ratio), frontal=float(p.frontal_area),
                W=float(p.mass * GRAVITY), joints=joints, f_spar=f_spar,
                f_hi=min(F_HI, f_spar), names=list(names))


# ------------------------------------------------------------ power required

def p_required(mc, U, parasite=True):
    """Induced (momentum, k = 1/e) + profile (skin friction at U) [+ parasite]."""
    W, S, b = mc["W"], mc["S"], mc["b"]
    Sd = math.pi * b * b / 4.0
    T = W / (2.0 * RHO * Sd)
    vi = math.sqrt(0.5 * (-U * U + math.sqrt(U ** 4 + 4.0 * T * T)))
    p_ind = W * vi / OSWALD
    cbar = S / b
    cdf = float(skin_friction_cd(np.array([U * cbar / NU]))[0])
    p_pro = 0.5 * RHO * U ** 3 * S * cdf
    p_par = 0.5 * RHO * U ** 3 * CD_HULL * mc["frontal"] if parasite else 0.0
    return p_ind + p_pro + p_par, p_ind, p_pro, p_par


def p_required_min(mc, parasite, cl_max=None):
    Us = np.linspace(0.5, 40.0, 3951)
    if cl_max is not None:   # a wing that may not exceed the model's largest CL
        Us = Us[mc["W"] / (0.5 * RHO * Us * Us * mc["S"]) <= cl_max]
    P = np.array([p_required(mc, u, parasite)[0] for u in Us])
    i = int(np.argmin(P))
    U = float(Us[i])
    cl = mc["W"] / (0.5 * RHO * U * U * mc["S"])
    return float(P[i]), U, cl


# ----------------------------------------------------------- power available

def load_power(j, f, A, U):
    """Most power the air can take from this joint's strips: every strip at the
    largest normal-force coefficient the model has (CN_LEV, fluid.py:274), at
    its largest speed U + w A r, moving at its largest stroke speed w A r."""
    w = 2.0 * math.pi * f
    v = w * np.outer(A, j["r"])
    return 0.5 * RHO * CN_LEV * ((U + v) ** 2 * v) @ j["cdr"]


def joint_power(j, f, A_cap=None, I=None, tau=None, fundamental=FOUR_OVER_PI, U_load=None):
    """Largest mean power one stroke joint can put into its load at f.

    Sinusoidal stroke A sin(wt) (cpg.py:171), torque bounded by tau (mjcf.py:717).
    Only the torque's fundamental does work on a sinusoidal motion, and its
    amplitude is at most (4/pi) tau. It must carry inertia (quadrature, I w^2 A)
    and joint damping (d w A, in phase, lost) before what is left reaches the air.
    """
    w = 2.0 * math.pi * f
    I = j["I"] if I is None else I
    tau = j["tau"] if tau is None else tau
    T1 = fundamental * tau
    A_cap = j["A_max"] if A_cap is None else A_cap
    A = np.linspace(1e-5, A_cap, 4000)
    head = np.sqrt(np.maximum(T1 * T1 - (I * w * w * A) ** 2, 0.0)) - j["d"] * w * A
    P = 0.5 * w * A * np.maximum(head, 0.0)
    if U_load is not None:
        P = np.minimum(P, load_power(j, f, A, U_load))
    k = int(np.argmax(P))
    return float(P[k]), float(A[k])


def p_available(mc, f, variant="headline", A_caps=None):
    tot = 0.0
    for n, j in enumerate(mc["joints"]):
        cap = None if A_caps is None else A_caps[n]
        if variant == "headline":
            P, _ = joint_power(j, f, cap)
        elif variant == "main_only":       # main wings only, no tail or trim surface
            P = joint_power(j, f, cap)[0] if j["main"] else 0.0
        elif variant == "load_capped":     # what the air can take, at the min-power speed
            P, _ = joint_power(j, f, cap, U_load=mc["U2"])
        elif variant == "main_load_capped":
            P = joint_power(j, f, cap, U_load=mc["U2"])[0] if j["main"] else 0.0
        elif variant == "sine_torque":     # torque itself sinusoidal: fundamental = tau
            P, _ = joint_power(j, f, cap, fundamental=1.0)
        elif variant == "spring":          # series_stiffness 1 as mjcf builds it (k on I_phen)
            P, _ = joint_power(j, f, cap, I=abs(j["I"] - j["I_phen"]))
        elif variant == "spring_ideal":    # a spring that cancels the true inertia
            P, _ = joint_power(j, f, cap, I=0.0)
        elif variant == "gear_max":        # gear ratio at the genome's top, forcerange clip
            eta = lambda G: 0.97 ** max(0.0, math.log(max(G, 1.0)) / math.log(5.0))  # noqa: E731
            t = min(j["tau"] * GEAR_MAX * eta(GEAR_MAX) / (j["gear"] * eta(j["gear"])), FR_MAX)
            P, _ = joint_power(j, f, cap, tau=t)
        elif variant == "rated":           # if the motor's rated power were enforced
            P = min(joint_power(j, f, cap)[0], j["p_cont"])
        elif variant == "rated_speed":     # if the rated speed (energy.py:66) bounded the stroke rate
            a = j["A_max"] if cap is None else cap
            P, _ = joint_power(j, f, min(a, j["w_rated"] / (2.0 * math.pi * f)))
        else:
            raise ValueError(variant)
        tot += P
    return tot


def f_bind(j):
    """Frequency at which the full-stroke inertial torque alone reaches the limit."""
    return math.sqrt(j["tau"] / (j["I"] * j["A_max"])) / (2.0 * math.pi)


# ----------------------------------------------------------------------- main

def main():
    check_sources()
    plans = [("gannet", BODY_PLANS["gannet"]()), ("teal", BODY_PLANS["teal"]()),
             ("beetle", BODY_PLANS["beetle"]()), ("bat", BODY_PLANS["bat"]()),
             ("teal, feathering", feathered("teal"))]
    out = {}
    print("\n## inputs read from the built machines")
    for label, g in plans:
        mc = machine(label, g)
        out[label] = mc
        print(f"{label:17s} m {mc['m']:.3f} kg  S {mc['S']:.4f} m2  b {mc['b']:.3f} m  AR {mc['AR']:.2f}"
              f"  frontal {mc['frontal']:.4f} m2  spar f_max {mc['f_spar']:.2f} Hz")
        for j in mc["joints"]:
            print(f"    {j['name']:14s} tau {j['tau']:7.3f} N m  I {j['I']:.5f} (phenotype {j['I_phen']:.5f})"
                  f"  d {j['d']:.4f}  A_max {j['A_max']:.3f} rad  G {j['gear']:.0f}"
                  f"  p_cont {j['p_cont']:.0f} W  rated speed {j['w_rated']:.2f} rad/s  f_bind {f_bind(j):.2f} Hz"
                  f"  strips {len(j['r'])} area {j['cdr'].sum():.4f} m2 r_max {j['r'].max() if len(j['r']) else 0:.2f} m"
                  f"  {'main' if j['main'] else 'trim'}")

    print("\n## power required, minimum over trim speed")
    for label, mc in out.items():
        P2, U2, cl2 = p_required_min(mc, parasite=False)
        P3, U3, cl3 = p_required_min(mc, parasite=True)
        _, pi, pp, _ = p_required(mc, U2, parasite=False)
        P4, U4, _ = p_required_min(mc, parasite=True, cl_max=CN_LEV / 2.0)
        mc.update(Preq2=P2, U2=U2, Preq3=P3, U3=U3, Preq4=P4)
        print(f"{label:17s} induced+profile {P2:6.1f} W at {U2:4.1f} m/s (CL {cl2:.2f}; induced {pi:.1f}, profile {pp:.1f})"
              f" | +parasite {P3:6.1f} W at {U3:4.1f} m/s (CL {cl3:.2f})"
              f" | +parasite, CL <= {CN_LEV / 2:.1f}: {P4:6.1f} W at {U4:4.1f} m/s")

    print("\n## ratio P_avail / P_req(induced+profile) across the band [1.5, min(12, spar)]")
    print(f"{'plan':17s} {'band Hz':>12s} {'f_bind':>7s} {'at 1.5':>7s} {'max (f)':>14s} {'at top':>7s} {'min':>6s} | with parasite: max / min")
    rows = {}
    for label, mc in out.items():
        fs = np.linspace(F_LO, mc["f_hi"], 211)
        pa = np.array([p_available(mc, f) for f in fs])
        r2, r3 = pa / mc["Preq2"], pa / mc["Preq3"]
        k = int(np.argmax(r2))
        fb = min(f_bind(j) for j in mc["joints"]) if mc["joints"] else float("nan")
        rows[label] = dict(band=(F_LO, mc["f_hi"]), f_bind=fb, r_lo=float(r2[0]), r_max=float(r2[k]),
                           f_max=float(fs[k]), r_top=float(r2[-1]), r_min=float(r2.min()),
                           r3_max=float(r3.max()), r3_min=float(r3.min()),
                           Pavail_lo=float(pa[0]), Pavail_top=float(pa[-1]))
        print(f"{label:17s} {F_LO:4.1f}-{mc['f_hi']:5.2f}   {fb:6.2f} {r2[0]:7.2f} {r2[k]:7.2f} ({fs[k]:4.1f})"
              f" {r2[-1]:7.2f} {r2.min():6.2f} | {r3.max():.2f} / {r3.min():.2f}")

    print(f"\n    spar band top, gannet: {out['gannet']['f_spar']:.2f} Hz; ROADMAP 09-23 §5 (audit) had the check at"
          f" 1.99x allowable at 10.95 Hz, i.e. {10.95 / math.sqrt(1.99):.2f} Hz")
    print("\n## at the frequency of the largest ratio: stroke Strouhal number, and the conversion")
    print("## efficiency below which stroke power would stop covering P_req (1 / ratio)")
    for label, mc in out.items():
        r = rows[label]
        mains = [j for j in mc["joints"] if j["main"]] or mc["joints"]
        j = max(mains, key=lambda q: q["r"].max() if len(q["r"]) else 0)
        _, A_opt = joint_power(j, r["f_max"])
        St = r["f_max"] * 2.0 * j["r"].max() * math.sin(A_opt) / mc["U2"]
        rows[label].update(St=St, eta_break=1.0 / r["r_max"], eta_break3=1.0 / r["r3_max"])
        print(f"{label:17s} f {r['f_max']:4.1f} Hz  A {A_opt:.3f} rad  tip r {j['r'].max():.2f} m  U {mc['U2']:.1f} m/s"
              f"  St {St:.2f}  | efficiency break-even {1 / r['r_max']:.3f} (with parasite {1 / r['r3_max']:.3f})")

    print("\n## robustness: the same band, P_avail restricted (ratio to induced+profile; then +parasite)")
    print(f"{'plan':17s} {'main wings: max / min':>22s} {'load-capped: max / min':>23s} {'main+load-capped: max / min':>28s} {'(+parasite) max / min':>22s} {'rated speed: max / min':>22s} {'sine tau: max / min':>20s}")
    for label, mc in out.items():
        fs = np.linspace(F_LO, mc["f_hi"], 211)
        cols = []
        for v in ("main_only", "load_capped", "main_load_capped", "rated_speed", "sine_torque"):
            pa = np.array([p_available(mc, f, v) for f in fs])
            cols.append((pa.max() / mc["Preq2"], pa.min() / mc["Preq2"], pa))
        pa = cols[2][2]
        rows[label]["robust"] = [(float(a), float(b)) for a, b, _ in cols] + [
            (float(pa.max() / mc["Preq3"]), float(pa.min() / mc["Preq3"]))]
        print(f"{label:17s} {cols[0][0]:10.2f} / {cols[0][1]:<9.2f} {cols[1][0]:11.2f} / {cols[1][1]:<9.2f}"
              f" {cols[2][0]:14.2f} / {cols[2][1]:<11.2f} {pa.max() / mc['Preq3']:10.2f} / {pa.min() / mc['Preq3']:<9.2f}"
              f" {cols[3][0]:10.2f} / {cols[3][1]:<6.2f} {cols[4][0]:10.2f} / {cols[4][1]:<6.2f}")

    print("\n## sensitivity at the band top (ratio to induced+profile)")
    print(f"{'plan':17s} {'headline':>8s} {'sine tau':>8s} {'spring(code)':>12s} {'spring(ideal)':>13s} {'gear 200':>8s} {'rated cap':>9s}")
    for label, mc in out.items():
        f = mc["f_hi"]
        vals = [p_available(mc, f, v) / mc["Preq2"] for v in
                ("headline", "sine_torque", "spring", "spring_ideal", "gear_max", "rated")]
        rows[label]["sens_top"] = vals
        print(f"{label:17s} " + " ".join(f"{v:{w}.2f}" for v, w in zip(vals, (8, 8, 12, 13, 8, 9))))

    # ---------------------------------------------- the recorded rig of 09-26
    print("\n## the teal level gait on the rig, 2026-09-26 (feathering teal, 11.33 Hz, 6.8 m/s)")
    b = pickle.load(open(GAIT_PKL, "rb"))
    gp, U_rig = b["params"], float(b["v"])
    mc = out["teal, feathering"]
    f = float(gp.frequency)
    w = 2 * math.pi * f
    caps = []
    for j in mc["joints"]:
        A_cmd = float(gp.amplitude[j["act_index"]])
        caps.append(A_cmd)
        need = j["I"] * w * w * A_cmd
        reach = min(1.0, FOUR_OVER_PI * j["tau"] / (j["I"] * w * w * max(A_cmd, 1e-9)))
        P, A_opt = joint_power(j, f, A_cmd)
        print(f"    {j['name']:12s} A_cmd {A_cmd:.3f} rad  inertial torque needed {need:7.1f} N m"
              f" = {need / j['tau']:5.1f}x the limit  reach bound {reach:.3f}  P <= {P:5.1f} W at A {A_opt:.4f}")
    for j in mc["joints"][:1]:
        kp, kv = 2.0 * j["tau"], 0.15 * j["tau"]          # mjcf.py:715-716, drive_compliance 1
        G = (kp + 1j * w * kv) / (kp - j["I"] * w * w + 1j * w * (kv + j["d"]))
        fn = math.sqrt(kp / j["I"]) / (2 * math.pi)
        print(f"    servo, {j['name']}: kp {kp:.1f} N m/rad, closed-loop corner {fn:.2f} Hz; unsaturated response at"
              f" {f:.2f} Hz |q/q_ref| {abs(G):.3f}, lag {-math.degrees(np.angle(G)):.0f} deg")
    j0 = mc["joints"][0]
    print(f"    commanded tip stroke speed, {j0['name']}: w A r = {w * caps[0] * j0['r'].max():.1f} m/s")
    # The feathering motors the bound leaves out (Assumption 8), on its own terms:
    # same motor, MuJoCo inertia about the span axis, half-travel FEATHER_RANGE.
    g_f = feathered("teal")
    p_f = build(g_f)
    model_f, data_f, names_f, _ = compile_phenotype(p_f)
    mujoco.mj_forward(model_f, data_f)
    Mf = np.zeros((model_f.nv, model_f.nv))
    mujoco.mj_fullM(model_f, data_f, Mf)
    for k, an in enumerate(names_f):
        if an == "seg1_wing_f":
            jj = model_f.actuator_trnid[k, 0]
            dof = model_f.jnt_dofadr[jj]
            jf = dict(tau=float(model_f.actuator_forcerange[k, 1]), I=float(Mf[dof, dof]),
                      d=float(model_f.dof_damping[dof]), A_max=float(model_f.jnt_range[jj][1]))
            print(f"    feathering motor seg1_wing_f (excluded): tau {jf['tau']:.1f} N m, I {jf['I']:.5f};"
                  f" bound at 7.80 Hz {joint_power(jf, 7.80)[0]:.0f} W, at 11.33 Hz {joint_power(jf, f)[0]:.0f} W")
    pa_gait = p_available(mc, f, A_caps=caps)
    pa_gait_mlc = p_available(mc, f, "main_load_capped", A_caps=caps)
    print(f"    main wings, load-capped at the gait: {pa_gait_mlc:.1f} W"
          f" -> ratio {pa_gait_mlc / mc['Preq2']:.2f} (min-power P_req)")
    pr_rig = p_required(mc, U_rig, parasite=False)
    pr_rig3 = p_required(mc, U_rig, parasite=True)
    in_band = f <= mc["f_hi"]
    r_gait_min = pa_gait / mc["Preq2"]
    r_gait_rig = pa_gait / pr_rig[0]
    print(f"    P_avail at the gait {pa_gait:.1f} W | P_req min {mc['Preq2']:.1f} W -> ratio {r_gait_min:.2f}"
          f" | P_req at the rig's {U_rig:.2f} m/s {pr_rig[0]:.1f} W (+parasite {pr_rig3[0]:.1f}) -> ratio {r_gait_rig:.2f}")
    print(f"    11.33 Hz inside the feathering teal's spar band (<= {mc['f_hi']:.2f} Hz): {in_band}")
    for name, r in (("ratio, P_req at min-power speed", r_gait_min), ("ratio, P_req at the rig speed", r_gait_rig)):
        print(f"    {name}: {r:.2f} against 0.42 -> {r / RIG['margin_free'] - 1:+.0%}"
              f" (within 30%: {abs(r / RIG['margin_free'] - 1) <= 0.30});"
              f" as a force-equivalent r^(2/3) {r ** (2 / 3):.2f}")

    # Post hoc, not the prediction's metric: the lift the same wing makes held
    # still at the rig speed and the model's largest CL (CN_LEV / 2).
    fixed = 0.5 * RHO * U_rig ** 2 * mc["S"] * (CN_LEV / 2.0) / mc["W"]
    print(f"    post hoc: fixed-wing lift at {U_rig:.2f} m/s and CL {CN_LEV / 2:.1f} over W = {fixed:.3f}"
          f" (phenotype weight {mc['W']:.1f} N; the rig divided by its dry weight)")

    # ------------------------------------------------------------- verdict
    any_above = {k: v["r_max"] > 1.0 for k, v in rows.items()}
    print("\n## against the prediction")
    print("    any plan with ratio > 1 inside its band:", {k: round(v["r_max"], 2) for k, v in rows.items()})
    falsified = any(any_above.values())
    reproduced = abs(r_gait_min / RIG["margin_free"] - 1) <= 0.30
    print(f"    clause 1 (ratio < 1 everywhere): {'fails' if falsified else 'holds'};"
          f" clause 2 (0.42 within 30%): {'holds' if reproduced else 'fails'}")
    print("    OUTCOME:", "REFUTED" if falsified else ("CONFIRMED" if reproduced else "REFUTED"))

    res = {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}
    res["_gait"] = dict(f=f, U=U_rig, P_avail=pa_gait, ratio_min_power=r_gait_min,
                        ratio_rig_speed=r_gait_rig, P_req_rig=pr_rig[0])
    print("\nJSON " + json.dumps(res, default=float))


if __name__ == "__main__":
    main()
