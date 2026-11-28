"""Pass M2: M1's flying law, copied (not imported) so that the law M2 freezes
cannot change under it while M1 is still tuning.

Source: `m1_takeoff.py` (sha256 9ef9178465606daa..., uncommitted, 2026-11-18)
class `Fly`, `GAINS`, `PLUS`; `m1_lib.py` (sha256 37f3452a286f64d5...) `K1`,
`axes`, `k1_targets`.  Copied verbatim but for the module prefix `L.`.
M1's text: K1 (4.5 Hz, 85 deg, feather 50) modulated by a loop on beat means
of the true state: attitude to nose-up with a lean against horizontal
velocity, amplitude by a PI on climb rate toward a height H_t.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.80665
F_BEAT = 4.5
Q_STRIKE = -1.35


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


K1 = dict(f=4.5, Phi=math.radians(85.0), Psi=math.radians(50.0))


def axes(mc):
    """The hull's nose (x), left-wing (y) and back (z) axes in the world."""
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    return R[:, 0], R[:, 1], R[:, 2]


def k1_targets(mc, t, amp=1.0, bias=0.0, f=None, Phi=None, Psi=None, psi0=0.0,
               amp_lr=0.0, psi0_lr=0.0, phase0=0.0):
    """Joint targets and rates by role for the K1 stroke (both wings alike unless
    ``amp_lr`` / ``psi0_lr``: left gets +half, right -half).  Stroke positive is up
    for both wings (checked on the built model, M1 probe P0)."""
    f = K1["f"] if f is None else f
    Phi = K1["Phi"] if Phi is None else Phi
    Psi = K1["Psi"] if Psi is None else Psi
    w = 2 * math.pi * f
    ph = w * t + phase0
    q = np.zeros(len(mc.aid))
    qd = np.zeros(len(mc.aid))
    left = np.array([("seg1" in mc.mj.mj_id2name(mc.m, mc.mj.mjtObj.mjOBJ_ACTUATOR, int(a))) for a in mc.aid])
    for n, r in enumerate(mc.role):
        s = 1.0 if left[n] else -1.0
        if r == "stroke":
            A = Phi / 2 * amp * (1.0 + 0.5 * s * amp_lr)
            q[n] = bias + A * math.sin(ph)
            qd[n] = A * w * math.cos(ph)
        elif r == "feather":
            q[n] = -Psi * math.cos(ph) + psi0 + 0.5 * s * psi0_lr
            qd[n] = Psi * w * math.sin(ph)
    return q, qd


GAINS = dict(
    f=4.5,                      # Hz, base frequency
    k_v=0.15,                   # rad of lean per m/s of horizontal velocity
    lean_max=0.3,
    # attitude: direct PD from the error rotation (body frame) and the beat-mean body rates.
    # Signs measured on the free body (tuning rows "auth"): bias + turns about +y; the
    # left-right feather offset + turns about +z (and spins about -x); the left-right
    # amplitude difference + spins about +x, weakly.
    kpy=0.4, kdy=0.10,          # common stroke bias, rad per rad / per rad/s
    kpz=0.4, kdz=0.15,          # left-right feather offset
    kdx=0.10,                   # left-right amplitude difference, spin damping
    kpz_a=0.0, kdz_a=0.0,
    kdx_f=0.0,                  # left-right feather offset on the spin rate
    # ADDED after data (not in the frozen structure; RESULT section 0b): the left-right
    # feather amplitude difference on the back axis.  0 in the pre-registered arms.
    kpP=0.0, kdP=0.0,       # left-right amplitude difference on the back axis (left larger: -z)
    H_t=4.0, v_c=1.5,           # m, m/s
    k_h=1.0,                    # 1/s, height error to climb rate wanted
    a0=0.70, kp_vz=0.10, ki_vz=0.10,   # amplitude: hover guess, PI on climb rate
    tilt_ff=True,
    switch_q=-1.2,              # rad, vault: FLY starts when the stroke passes this
    switch_s=0.25,              # s after t_c, at the latest
    blend_s=0.1,
    p_switch_deg=35.0,          # P: FLY starts at this beat-mean nose elevation
    p_switch_s=3.0,
    lp_beats=0.5,               # the outputs' low-pass time constant, in beats
    tail_kd=0.0,                # tail: pitch damping (rad of tail per rad/s); 0 = not used
)

PLUS = dict(kpP=1.0, kdP=0.3)


class Fly:
    """K1 modulated by a loop on beat means of the true state (section 4)."""

    def __init__(self, mc, g, loop=True):
        self.mc, self.g, self.loop = mc, g, loop
        self.nb = int(round(1.0 / (F_BEAT * mc.dt) / 10))       # samples per beat (every 10 steps)
        self.buf = []
        self.out = dict(bias=0.0, amp=1.0 if not loop else g["a0"], amp_lr=0.0, psi0_lr=0.0, tail=0.0, Psi_lr=0.0)
        self.left = np.array([("seg1" in mc.mj.mj_id2name(mc.m, mc.mj.mjtObj.mjOBJ_ACTUATOR, int(a))) for a in mc.aid])
        self.integ = 0.0
        self.t0 = None
        self.k = 0

    def sample(self, H):
        """Every step: every 10th step store the state; update the outputs."""
        self.k += 1
        if self.k % 10:
            return
        mc = self.mc
        nose, left, back = axes(mc)
        self.buf.append((nose, left, back, mc.d.qvel[3:6].copy(), mc.com().copy(), H))
        if len(self.buf) > self.nb + 1:
            self.buf.pop(0)
        if not self.loop or len(self.buf) < 3:
            return
        g = self.g
        B = self.buf
        n = np.mean([b[0] for b in B], 0); n /= np.linalg.norm(n)
        l_ = np.mean([b[1] for b in B], 0)
        bk = np.mean([b[2] for b in B], 0)
        R = np.stack([n, l_ / np.linalg.norm(l_), bk / np.linalg.norm(bk)], 1)
        w = np.mean([b[3] for b in B], 0)                         # body-frame rates, beat mean
        T = (len(B) - 1) * 10 * mc.dt
        v = (B[-1][4] - B[0][4]) / T                              # centre-of-mass velocity, beat mean
        Hm = float(np.mean([b[5] for b in B]))
        lean = -g["k_v"] * v[:2]
        ln = np.linalg.norm(lean)
        if ln > g["lean_max"]:
            lean *= g["lean_max"] / ln
        nt = np.array([lean[0], lean[1], 1.0]); nt /= np.linalg.norm(nt)
        e = R.T @ np.cross(n, nt)
        ang = math.asin(clip(np.linalg.norm(np.cross(n, nt)), 0.0, 1.0))
        if n @ nt < 0:
            ang = math.pi - ang
        if np.linalg.norm(e) > 1e-9:                               # error as a rotation vector
            e = e / np.linalg.norm(e) * ang
        tgt = dict(bias=clip(g["kpy"] * e[1] - g["kdy"] * w[1], -0.3, 0.3),
                   psi0_lr=clip(g["kpz"] * e[2] - g["kdz"] * w[2] - g["kdx_f"] * w[0], -0.3, 0.3),
                   amp_lr=clip(-g["kdx"] * w[0] - (g["kpz_a"] * e[2] - g["kdz_a"] * w[2]), -0.3, 0.3))
        vzt = clip(g["k_h"] * (g["H_t"] - Hm), -g["v_c"], g["v_c"])
        err = vzt - v[2]
        dt_s = 10 * mc.dt
        self.integ = clip(self.integ + err * dt_s, -3.0, 3.0)
        cos_t = max(float(n[2]), 0.5)
        a = g["a0"] / math.sqrt(cos_t) if g["tilt_ff"] else g["a0"]
        tgt["Psi_lr"] = clip(-g["kpP"] * e[2] + g["kdP"] * w[2], -0.3, 0.3)
        tgt["amp"] = clip(a + g["kp_vz"] * err + g["ki_vz"] * self.integ, 0.60, 1.15)
        tgt["tail"] = clip(g["tail_kd"] * w[1], -0.5, 0.5)
        al = min(1.0, dt_s / (g["lp_beats"] / g["f"]))                                 # low-pass, half a beat
        for k_, val in tgt.items():
            self.out[k_] += al * (val - self.out[k_])

    def targets(self, t):
        g, o = self.g, self.out
        q, qd = k1_targets(self.mc, t - self.t0, amp=o["amp"], bias=o["bias"], f=g["f"],
                             amp_lr=o["amp_lr"], psi0_lr=o["psi0_lr"], phase0=-math.pi / 2)
        q[self.mc.role == "tail"] = o["tail"]
        if o["Psi_lr"]:                     # ADDED after data (M1 result section 0b): feather amplitude, left-right
            w = 2 * math.pi * g["f"]
            ph = w * (t - self.t0) - math.pi / 2
            fe = self.mc.role == "feather"
            s_ = np.where(self.left, 0.5, -0.5) * o["Psi_lr"]
            q[fe] += (-s_ * math.cos(ph))[fe]
            qd[fe] += (s_ * w * math.sin(ph))[fe]
        return q, qd


