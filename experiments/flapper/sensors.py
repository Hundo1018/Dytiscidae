"""Modelled sensors for the mission harness (`mission.Machine`), for pass M5
(`TRIPHIBIAN_SPEC.md` item 4: control laws read only these, not the true state).

The environment's own `TriphibianEnv.observation` reads the true body rates and
the true gravity direction; this module puts what a hull-mounted sensor set
would report in their place:

- an IMU at a mounting point on the hull: a rate gyro and an accelerometer
  sampled at ``rate_hz``, each with white noise, a bias that wanders, a range
  that saturates, and a delay. The accelerometer reads the specific force at
  its own mounting point, so it feels the hull's heave and pitch at the
  flapping frequency (PLAN, "Control and sensors": the hardest problem named);
- an attitude estimate from them (a complementary filter: the gyro integrated,
  pulled toward the accelerometer's tilt with a time constant), which is what
  a control law gets as "pitch" and "roll";
- a pressure sensor: depth in water, height over the start in air;
- a contact switch (ground load over a threshold), joint encoders, and a
  wetness estimate (the share of the wing strips the solver has wet, as a
  conductivity probe on each wing would give).

Every constant below is TYPED: noise figures are of the order of a small
MEMS IMU and a hobby pressure sensor, from memory, **not checked against a
datasheet**. A pass that leans on one names it and sweeps it.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

DEFAULTS = dict(
    rate_hz=1000.0,               # IMU sample rate
    delay_s=0.002,                # IMU to controller
    gyro_noise=0.014 * math.pi / 180 * math.sqrt(1000.0),     # rad/s rms per sample (0.014 deg/s/sqrt Hz)
    gyro_bias_walk=0.0005,        # rad/s per sqrt(s)
    gyro_range=2000 * math.pi / 180,
    acc_noise=175e-6 * 9.80665 * math.sqrt(1000.0),           # m/s2 rms per sample (175 ug/sqrt Hz)
    acc_bias=0.05,                # m/s2, fixed per run, random direction
    acc_range=16 * 9.80665,
    mount=(0.0, 0.0, 0.0),        # m, from the hull's centre of mass, hull frame
    tilt_tau_s=30.0,              # complementary filter: time constant of the accelerometer's pull (swept: RESULT_M0)
    p_noise_water=0.01,           # m of depth rms
    p_noise_air=0.10,             # m of height rms
    contact_threshold=0.05,       # of the weight
    init_true=False,              # True: the estimate starts at the true attitude (armed before a launch)
)


class Sensors:
    def __init__(self, machine, seed: int = 0, **over):
        self.mc = machine
        self.c = dict(DEFAULTS, **over)
        self.rng = np.random.default_rng(seed)
        self.every = max(int(round(1.0 / (self.c["rate_hz"] * machine.dt))), 1)
        self.lag = max(int(round(self.c["delay_s"] * self.c["rate_hz"])), 0)
        self.queue = deque(maxlen=self.lag + 1)
        self.gbias = np.zeros(3)
        b = self.rng.normal(size=3)
        self.abias = self.c["acc_bias"] * b / np.linalg.norm(b)
        self.k = 0
        self.est = None                 # attitude quaternion estimate (w, x, y, z), hull to world
        self.z0 = float(machine.d.qpos[2])
        self.last = None
        if self.c.get("init_true"):     # started before the launch (on the ground, in the hand): its true attitude
            R = machine.d.xmat[machine.hull].reshape(3, 3)
            self.est = _quat_from_R(R)

    # -- truth at the mount, hull frame
    def _truth(self):
        mc, m, d, mj = self.mc, self.mc.m, self.mc.d, self.mc.mj
        h = mc.hull
        R = d.xmat[h].reshape(3, 3)
        vel = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_XBODY, h, vel, 1)      # local: [ang, lin]
        acc = np.zeros(6)
        mj.mj_rnePostConstraint(m, d)
        mj.mj_objectAcceleration(m, d, mj.mjtObj.mjOBJ_XBODY, h, acc, 1)  # specific force at the body frame
        w, a0, al = vel[:3], acc[3:], acc[:3]
        r = np.asarray(self.c["mount"], float) + (d.xipos[h] - d.xpos[h]) @ R    # mount from the body frame
        a = a0 + np.cross(al, r) + np.cross(w, np.cross(w, r))
        return w, a, R

    def sample(self) -> dict | None:
        """Call once per physics step, after `Machine.step`; returns the reading
        the controller has now (the last delayed sample), or None before the first."""
        self.k += 1
        if self.k % self.every:
            return self.last
        c, rng = self.c, self.rng
        dt_s = self.every * self.mc.dt
        w, a, R = self._truth()
        self.gbias += rng.normal(0.0, c["gyro_bias_walk"] * math.sqrt(dt_s), 3)
        gyro = np.clip(w + self.gbias + rng.normal(0.0, c["gyro_noise"], 3), -c["gyro_range"], c["gyro_range"])
        acc = np.clip(a + self.abias + rng.normal(0.0, c["acc_noise"], 3), -c["acc_range"], c["acc_range"])
        self._filter(gyro, acc, dt_s, R)
        if self.est is None:             # not initialised yet: nothing to report
            return None
        mc, d = self.mc, self.mc.d
        wet = float(getattr(mc.env.solver.diag, "mean_submerged", 0.0))
        depth = mc.env.depth() if hasattr(mc.env, "depth") else 0.0
        reading = dict(
            gyro=gyro, acc=acc, quat=self.est.copy(), pitch_deg=self._pitch(self.est), roll_deg=self._roll(self.est),
            depth=float(depth + rng.normal(0.0, c["p_noise_water"])) if wet > 0.5 else None,
            height=float(d.qpos[2] - self.z0 + rng.normal(0.0, c["p_noise_air"])) if wet <= 0.5 else None,
            contact=bool(mc.contact_load() > c["contact_threshold"]),
            joints=d.qpos[mc.qad].copy(), joint_rates=d.qvel[mc.dof].copy(), wet=wet)
        self.queue.append(reading)
        self.last = self.queue[0] if len(self.queue) > self.lag else None
        return self.last

    # -- complementary filter on the rotation group
    def _filter(self, gyro, acc, dt, R_true):
        if self.est is None:             # initialised from the accelerometer's tilt and a zero heading,
            if abs(np.linalg.norm(acc) - 9.80665) > 0.1 * 9.80665:      # once it reads 1 g within 10%
                return
            g = acc / max(np.linalg.norm(acc), 1e-9)          # "up" in the hull frame
            ax = np.cross(g, [0.0, 0.0, 1.0])                  # the shortest turn taking it to the world's up
            s, cth = np.linalg.norm(ax), float(g[2])
            ang = math.atan2(s, cth)
            self.est = _quat_exp(ax / s * ang) if s > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
            return
        q = _quat_mul(self.est, _quat_exp(gyro * dt))
        # tilt correction: rotate so the estimated gravity in the hull frame moves toward the accelerometer's
        Rq = _quat_to_R(q)
        g_est = Rq.T @ np.array([0.0, 0.0, 1.0])
        n = np.linalg.norm(acc)
        if 0.5 * 9.80665 < n < 1.5 * 9.80665:          # trust the accelerometer only near 1 g
            g_meas = acc / n
            err = np.cross(g_meas, g_est)
            q = _quat_mul(q, _quat_exp(err * dt / self.c["tilt_tau_s"]))
        self.est = q / np.linalg.norm(q)

    @staticmethod
    def _pitch(q):
        return math.degrees(math.asin(max(-1.0, min(1.0, -2.0 * (q[0] * q[2] - q[3] * q[1])))))

    @staticmethod
    def _roll(q):
        return math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_exp(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    s = math.sin(0.5 * th) / th
    return np.array([math.cos(0.5 * th), s * v[0], s * v[1], s * v[2]])


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _quat_from_R(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q
