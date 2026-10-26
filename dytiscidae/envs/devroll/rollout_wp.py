"""Air and water segments on the device, scored on the host (IMPL_1010 A4a).

What ``batchroll.rollout_batch`` does around ``step_batch`` for one body's W
worlds, with every step on the device and one download per segment:

* **The loop.**  Per step, one capturable sequence: the controller
  (``DeviceControl.control``: decide on decision steps, the CPG every step), the
  decision record, the body's step (``DeviceBody.step``: servo, fluid, scatter,
  ``mjw.step``, power), the clock, the sensing of the new state, and the check
  and record kernel.  Each world's step count, inflow cadence, Wagner and slam
  priming are device counters (``DeviceBody.count``), so one captured graph per
  body covers both task phases and the decision and non-decision steps.
* **Divergence.**  CPU MuJoCo resets a world on a bad ``qacc``
  (``mjWARN_BADQACC``) and the host fails the segment as ``"unstable"`` at its
  end; MJWarp neither checks nor resets.  The check kernel flags a world whose
  ``qacc`` is not finite or exceeds ``mjMAXVAL`` (1e10), counts it in
  ``bad_qacc`` and fails it ``"unstable"``; a root position that is not finite or
  beyond 400 m fails it ``"diverged"`` (batchroll.py:936-941), as does a
  non-finite observation at a decision (``DeviceControl.observe``); a flat pack
  ends it ``"battery exhausted"`` with ``survived`` kept, as the host does; an
  MJWarp overflow flag (``d.overflow``: ``nefc > njmax`` per world, the
  broad/narrow-phase contact pools per launch) fails it ``"contact overflow"``.
* **Freezing.**  The host stops stepping a machine that failed, so its record
  and end position stop there.  On the device the world's record and end
  position stop at the same step; it is then held unloaded at its dry mass
  (``DeviceBody.hold``) and snapped back to its segment's start pose after every
  ``mjw.step``, so a NaN cannot reach MJWarp's solve or the contact pool the
  worlds share, and its battery stops charging.
* **The record.**  Per step and world the lists ``rollout_batch`` keeps (depth,
  clearance, altitude, ``R[2,2]``, ground contact, spin, ``vz``, ``xy``, the gait
  gain once one is set) and the running maximum of ``slam``; per decision the
  commanded and achieved angular rates; the start and end positions.  The
  arrays are sized to the segment: ``n_steps`` entries per world, and
  ``decisions_per_segment`` decisions, which ``DeviceControl.records`` refuses
  to exceed instead of dropping the rest.
* **Scoring** is the host's: ``batchroll.finish_segment`` on each world's own
  ``TriphibianEnv`` (its task still armed), with the device's power ledger.
* **Pairs.**  Worlds come in pairs from one initial state (``start_world``
  builds each world's machine through the host's reset, scatter and task
  sequence, so ``env.rng`` is where the host's would be): air and its mirror,
  water and its antipode (``tasks.pair_of``).  The second half is scored with
  ``pair_partner_of(first)``; a first half that failed is the result, as in
  ``evaluate.run_segment``.  The halves run side by side, not one after the
  other: the pairing is a function of the two records, so it is the same
  function of the draw as on the host.
* **Contacts.**  Air and water are expected to have none (T9).  Every world
  counts the contacts MJWarp reports for it while it is live; a non-zero count
  is reported in the body's notes ("contact in a contact-free segment").

Repeatability: with :func:`repeatable` called before the first launch (the
device tests call it), a segment repeats bit for bit, re-armed or fresh, direct
or from the graph.  Without it, float atomics in
the fluid's assembly and in MJWarp make two runs of one segment differ: an eel's
root spin by 6e-5 rad/s after 5 steps and 2e-3 m/s in ``vz`` after 200
(2026-10-26).

Not here: identification on the device (the host's ``identify_batch`` stays the
reference; ``evaluate_tier1_device`` refuses ``identify_axes``), feeding the
PPO buffer from the sampled worlds (A6), land and the crossings (A4b).
"""

# No ``from __future__ import annotations``: Warp reads the kernel annotations.
from dataclasses import dataclass, field

import numpy as _np

import warp as _wp

from dytiscidae.envs.devroll import control_wp as _control_wp
from dytiscidae.envs.devroll import model_wp as _model_wp

#: MuJoCo's ``mjMAXVAL``: ``mj_checkAcc`` resets a world whose |qacc| exceeds it.
MJ_MAXVAL = 1.0e10
#: The root-position bound of the host's divergence check (batchroll.py:937).
FAR = 400.0
#: Failure codes per world, and the host's words for them.
BATTERY, DIVERGED, UNSTABLE, OVERFLOW = 1, 2, 3, 4
FAILURES = {0: "", BATTERY: "battery exhausted", DIVERGED: "diverged", UNSTABLE: "unstable",
            OVERFLOW: "contact overflow"}
#: The note a body gets when a contact-free segment had a contact.
CONTACT_NOTE = "contact in a contact-free segment"
#: The media A4a runs on the device, in the host's order (DOMAIN_CYCLE).
MEDIA = ("air", "water")

_KERNELS: dict = {}


#: The device path's Warp modules, the ones :func:`repeatable` sets.
MODULES = ("fluid_wp", "scatter_wp", "model_wp", "control_wp", "rollout_wp")


def repeatable(on: bool = True) -> None:
    """Warp's ``RUN_TO_RUN`` determinism for the device path's own modules (their
    float atomics in a fixed order), before their kernels load.  MJWarp's
    modules are left as they are: a global ``wp.config.deterministic`` overflows
    the deterministic scatter buffer in MJWarp's ``crb`` at ``load_state``
    (2026-10-26), and with only these five modules a segment already repeats
    bit for bit."""
    import importlib

    mode = _wp.DeterministicMode.RUN_TO_RUN if on else _wp.DeterministicMode.NOT_GUARANTEED
    for name in MODULES:
        _wp.set_module_options({"deterministic": mode},
                               module=importlib.import_module("dytiscidae.envs.devroll." + name))


def decisions_per_segment(seconds: float, control_hz: float = 25.0,
                          timestep: float = 0.004) -> int:
    """Decisions one world makes in a segment: ``ceil(n_steps / control_every)``,
    with ``rollout_batch``'s ``n_steps`` and ``control_every``.  8 s at 25 Hz and
    dt 0.004 is 200; ``evaluate_tier1_batch``'s default 10 s is 250."""
    n_steps = int(seconds / timestep)
    every = max(1, int(1.0 / (control_hz * timestep)))
    return -(-n_steps // every)


def make_kernels(T):
    """The A4a record, check and freeze kernels for precision ``T``, cached."""
    if T in _KERNELS:
        return _KERNELS[T]
    V = _wp.vec3d if T == _wp.float64 else _wp.vec3
    M = _wp.mat33d if T == _wp.float64 else _wp.mat33
    f32 = _wp.float32
    F64 = _wp.float64

    @_wp.kernel
    def sync_dead(alive: _wp.array(dtype=int), power_active: _wp.array(dtype=int)):
        # A world that failed is not charged, as step_batch skips it.
        w = _wp.tid()
        if alive[w] == 0:
            power_active[w] = 0

    @_wp.kernel
    def record_decision(
        cap: int, due: _wp.array(dtype=int), ndec: _wp.array(dtype=int),
        cmd_twist: _wp.array(dtype=T), Ri: _wp.array(dtype=M), xipos: _wp.array(dtype=V),
        com: _wp.array(dtype=V), ang: _wp.array(dtype=V), lin: _wp.array(dtype=V),
        cmds: _wp.array(dtype=T), resp: _wp.array(dtype=T),
    ):
        # rollout_batch's cmds (twist_of(coeffs)[3:]) and resp (body_twist()[3:]
        # before the step), for each world that decided this step.
        w = _wp.tid()
        if due[w] == 0:
            return
        nd = ndec[w] - 1
        if nd < 0 or nd >= cap:
            return
        Rt = _wp.transpose(Ri[w])
        ta = Rt * ang[w]
        for k in range(3):
            cmds[(w * cap + nd) * 3 + k] = cmd_twist[w * 6 + 3 + k]
            resp[(w * cap + nd) * 3 + k] = ta[k]

    @_wp.kernel
    def count_contacts(
        nacon: _wp.array(dtype=int), cworld: _wp.array(dtype=int), alive: _wp.array(dtype=int),
        seen: _wp.array(dtype=int),
    ):
        # Every contact MJWarp made for a live world this step (the pool is
        # shared by all worlds and tagged with each contact's world).
        i = _wp.tid()
        if i >= nacon[0]:
            return
        w = cworld[i]
        if alive[w] != 0:
            _wp.atomic_add(seen, w, 1)

    @_wp.kernel
    def check_record(
        nv: int, cap: int, alive: _wp.array(dtype=int), diverged: _wp.array(dtype=int),
        power_active: _wp.array(dtype=int), overflow: _wp.array(dtype=int),
        qacc: _wp.array2d(dtype=f32),
        R: _wp.array(dtype=M), Ri: _wp.array(dtype=M), pos: _wp.array(dtype=V),
        xipos: _wp.array(dtype=V), com: _wp.array(dtype=V), ang: _wp.array(dtype=V),
        lin: _wp.array(dtype=V), clear: _wp.array(dtype=T), touching: _wp.array(dtype=int),
        slam: _wp.array(dtype=T), gain: _wp.array(dtype=T), has_gain: _wp.array(dtype=int),
        fail: _wp.array(dtype=int), survived: _wp.array(dtype=int),
        bad_qacc: _wp.array(dtype=int), end: _wp.array(dtype=F64), nrec: _wp.array(dtype=int),
        r_depth: _wp.array(dtype=T), r_clear: _wp.array(dtype=T), r_alt: _wp.array(dtype=T),
        r_up: _wp.array(dtype=T), r_contact: _wp.array(dtype=T), r_spin: _wp.array(dtype=T),
        r_vz: _wp.array(dtype=T), r_x: _wp.array(dtype=T), r_y: _wp.array(dtype=T),
        r_gain: _wp.array(dtype=T), r_has_gain: _wp.array(dtype=int),
        slam_max: _wp.array(dtype=T),
    ):
        # After a step and the sensing of its state: rollout_batch's checks in
        # its order, then its record (batchroll._record_step).
        w = _wp.tid()
        p = pos[w]
        if alive[w] == 0:
            # Failed at this step's decision (a non-finite observation): not
            # stepped on the host, so no record and no end position.
            if diverged[w] != 0 and fail[w] == 0:
                fail[w] = 2
                survived[w] = 0
            return
        if power_active[w] == 0:
            # The pack went flat on this step: stepped, not recorded, survived.
            fail[w] = 1
            alive[w] = 0
            for k in range(3):
                end[w * 3 + k] = F64(p[k])
            return
        bad = int(0)
        for j in range(nv):
            a = qacc[w, j]
            if not _wp.isfinite(a):
                bad = 1
            elif _wp.abs(a) > f32(1.0e10):
                bad = 1
        far = int(0)
        for k in range(3):
            if not _wp.isfinite(p[k]):
                far = 1
            elif _wp.abs(p[k]) > T(400.0):
                far = 1
        code = int(0)
        if bad != 0:
            bad_qacc[w] = bad_qacc[w] + 1
            code = 3
        elif far != 0:
            code = 2
        elif overflow[w] != 0:
            code = 4
        if code != 0:
            fail[w] = code
            survived[w] = 0
            alive[w] = 0
            power_active[w] = 0
            for k in range(3):
                end[w * 3 + k] = F64(p[k])
            return
        for k in range(3):
            end[w * 3 + k] = F64(p[k])
        n = nrec[w]
        if n < cap:
            i = w * cap + n
            r_depth[i] = -p[2]                  # the flat calm sea: depth is -z
            r_clear[i] = clear[w]
            r_alt[i] = p[2]
            r_up[i] = R[w][2, 2]
            r_contact[i] = T(0.0)
            if touching[w] != 0:
                r_contact[i] = T(1.0)
            # TriphibianEnv.body_twist ([lin, ang] at xipos, inertial frame).
            Rt = _wp.transpose(Ri[w])
            ta = Rt * ang[w]
            tl = Rt * (lin[w] - _wp.cross(xipos[w] - com[w], ang[w]))
            r_spin[i] = _wp.length(ta)
            r_vz[i] = tl[2]
            r_x[i] = p[0]
            r_y[i] = p[1]
            r_gain[i] = gain[w]
            r_has_gain[i] = has_gain[w]
            nrec[w] = n + 1
        slam_max[w] = _wp.max(slam_max[w], slam[w])

    @_wp.kernel
    def restore_dead(
        nq: int, nv: int, alive: _wp.array(dtype=int), q0: _wp.array2d(dtype=f32),
        qpos: _wp.array2d(dtype=f32), qvel: _wp.array2d(dtype=f32),
    ):
        # A frozen world back at its segment's start pose, at rest, after every
        # step: finite whatever it did, so MJWarp's next solve and the shared
        # contact pool see a sane state.
        w = _wp.tid()
        if alive[w] != 0:
            return
        for j in range(nq):
            qpos[w, j] = q0[w, j]
        for j in range(nv):
            qvel[w, j] = f32(0.0)

    _KERNELS[T] = {"sync_dead": sync_dead, "record_decision": record_decision,
                   "count_contacts": count_contacts, "check_record": check_record,
                   "restore_dead": restore_dead}
    return _KERNELS[T]


_CHANNELS = ("depth", "clear", "alt", "up", "contact", "spin", "vz", "x", "y", "gain")


class DeviceSegments:
    """One segment on each of W worlds of one body (see the module docstring).

    ``envs`` are the worlds' machines, each already at its segment's start
    (``start_world``).  ``policies`` and ``bases`` are per world, as
    ``rollout_batch`` takes them; ``shared`` the shared policy.  A segment:
    :meth:`arm` with each world's domain and base parameters, then :meth:`run`
    (or :meth:`step` ``n_steps`` times), then :meth:`results`.
    """

    def __init__(self, envs, *, seconds: float, dtype=_wp.float32, shared=None,
                 policies=None, bases=None, control_hz: float = 25.0,
                 njmax: int = _model_wp.NJMAX, nconmax: int = _model_wp.NCONMAX,
                 record=None) -> None:
        envs = list(envs)
        self.envs, self.W, self.dtype = envs, len(envs), dtype
        self.seconds = float(seconds)
        self.body = _model_wp.DeviceBody(envs, dtype=dtype, njmax=njmax, nconmax=nconmax)
        self.n_steps = int(self.seconds / self.body.timestep)
        need = decisions_per_segment(self.seconds, control_hz, self.body.timestep)
        cap = need if record is None else int(record)
        if cap < need:
            raise ValueError(f"a {self.seconds:g} s segment makes {need} decisions per world "
                             f"and the record holds {cap}")
        self.ctl = _control_wp.DeviceControl(self.body, shared=shared, policies=policies,
                                             bases=bases, control_hz=control_hz, record=cap)
        self.cap = cap
        W, n = self.W, self.n_steps
        T = dtype
        z = lambda k, t=T: _wp.zeros(max(int(k), 1), dtype=t)  # noqa: E731
        self.rec = {k: z(W * n) for k in _CHANNELS}
        self.rec["has_gain"] = z(W * n, int)
        self.cmds, self.resp = z(W * cap * 3), z(W * cap * 3)
        self.fail, self.survived, self.bad_qacc = z(W, int), z(W, int), z(W, int)
        self.nrec, self.contacts = z(W, int), z(W, int)
        self.end = z(W * 3, _wp.float64)
        self.slam_max = z(W)
        self.q0 = _wp.zeros((W, self.body.mjm.nq), dtype=_wp.float32)
        self.starts = _np.zeros((W, 3))
        self.domains = None
        self._graph = None

    # ------------------------------------------------------------------ arm

    def arm(self, domains, params_list, *, sampled=None, seeds=None) -> None:
        """Start the segment: the fluid reset, each machine's MuJoCo state and
        battery uploaded (``arm()`` with the host machines' start state, the A3
        contract), the overflow flags and the record cleared, the controller
        armed with each world's domain, and the start state sensed."""
        body, ctl = self.body, self.ctl
        doms = list(domains) if isinstance(domains, (list, tuple)) else [domains] * self.W
        if len(doms) != self.W:
            raise ValueError(f"arm takes one domain per world ({self.W})")
        for d in doms:
            if d.value not in MEDIA:
                raise NotImplementedError(f"{d.value} runs on the batched path until A4b")
        self.domains = doms
        body.reset()
        body.load_state([e.data for e in self.envs])
        body.power.load(self.envs)
        body.d.overflow.zero_()
        ctl.arm(doms, self.seconds, list(params_list), sampled=sampled, seeds=seeds)
        for a in (*self.rec.values(), self.cmds, self.resp, self.fail, self.bad_qacc,
                  self.nrec, self.contacts, self.slam_max):
            a.zero_()
        self.survived.fill_(1)
        self.starts = _np.stack([e.root_pos().copy() for e in self.envs])
        self.end.assign(self.starts.reshape(-1))
        _wp.copy(self.q0, body.d.qpos)
        ctl.sense()

    # ----------------------------------------------------------------- step

    def step(self) -> None:
        """One step for every world, no host transfer (see the module docstring)."""
        K = make_kernels(self.dtype)
        body, ctl, W, d = self.body, self.ctl, self.W, self.body.d
        self.decide()
        body.step(alive=ctl.active)
        _wp.launch(K["count_contacts"], dim=int(d.contact.worldid.shape[0]), inputs=[
            d.nacon, d.contact.worldid, ctl.active, self.contacts])
        ctl.tick()
        ctl.sense()
        self.check()
        _wp.launch(K["restore_dead"], dim=W, inputs=[
            body.mjm.nq, body.nv, ctl.active, self.q0, d.qpos, d.qvel])

    def decide(self) -> None:
        """Before the body's step, on the sensed state: the controller, a failed
        world's pack switched off, and the decision record."""
        K = make_kernels(self.dtype)
        body, ctl, b = self.body, self.ctl, self.ctl.sense_buf
        ctl.control()
        _wp.launch(K["sync_dead"], dim=self.W, inputs=[ctl.active, body.power.active])
        _wp.launch(K["record_decision"], dim=self.W, inputs=[
            self.cap, ctl.due, ctl.ndec, ctl.cmd_twist, b["Ri"], b["xipos"], b["com"],
            b["ang"], b["lin"], self.cmds, self.resp])

    def check(self) -> None:
        """After the step and the sensing of its state: the failure checks and
        the record (``check_record``)."""
        K = make_kernels(self.dtype)
        body, ctl, b, r = self.body, self.ctl, self.ctl.sense_buf, self.rec
        _wp.launch(K["check_record"], dim=self.W, inputs=[
            body.nv, self.n_steps, ctl.active, ctl.diverged, body.power.active,
            body.d.overflow, body.d.qacc, b["R"], b["Ri"], b["pos"], b["xipos"], b["com"],
            b["ang"], b["lin"], b["clear"], b["touching"], body.diag["slam"], ctl.gain,
            ctl.has_gain, self.fail, self.survived, self.bad_qacc, self.end, self.nrec,
            r["depth"], r["clear"], r["alt"], r["up"], r["contact"], r["spin"], r["vz"],
            r["x"], r["y"], r["gain"], r["has_gain"], self.slam_max])

    def capture(self):
        """The step as one CUDA graph (call after one direct :meth:`step`, which
        loads every module and allocates every buffer)."""
        with _wp.ScopedCapture() as cap:
            self.step()
        self._graph = cap.graph
        return cap.graph

    def run(self, *, graph: bool = True) -> None:
        """The whole segment: the first step direct, the rest from one captured
        graph (or direct, ``graph=False``)."""
        if self.n_steps <= 0:
            return
        self.step()
        if graph:
            g = self._graph if self._graph is not None else self.capture()
            for _ in range(self.n_steps - 1):
                _wp.capture_launch(g)
        else:
            for _ in range(self.n_steps - 1):
                self.step()

    # -------------------------------------------------------------- results

    def download(self) -> dict:
        """Everything the segment recorded, per world, in one transfer."""
        W, n, cap = self.W, self.n_steps, self.cap
        out = {k: v.numpy()[:W * n].reshape(W, n).astype(float) for k, v in self.rec.items()}
        out["has_gain"] = out["has_gain"].astype(bool)
        out["cmds"] = self.cmds.numpy()[:W * cap * 3].reshape(W, cap, 3).astype(float)
        out["resp"] = self.resp.numpy()[:W * cap * 3].reshape(W, cap, 3).astype(float)
        for k in ("fail", "survived", "bad_qacc", "nrec", "contacts"):
            out[k] = getattr(self, k).numpy()[:W].astype(int)
        out["end"] = self.end.numpy()[:W * 3].reshape(W, 3).astype(float)
        out["slam"] = self.slam_max.numpy()[:W].astype(float)
        out["ndec"] = self.ctl.ndec.numpy()[:W].astype(int)
        if int(out["ndec"].max(initial=0)) > cap:
            raise RuntimeError(f"a world made {int(out['ndec'].max())} decisions; the record "
                               f"holds {cap}")
        pw = self.body.power.state()
        out["mean_power"] = pw["total_j"] / _np.maximum(pw["t"], 1e-9)
        out["max_overload"] = pw["max_overload"]
        return out

    def recorded(self, w: int, data: dict) -> dict:
        """World ``w``'s record in ``rollout_batch``'s shape (``batchroll._record_step``)."""
        n, nd = int(data["nrec"][w]), int(data["ndec"][w])
        g = data["has_gain"][w, :n]
        return {"depths": list(data["depth"][w, :n]), "clears": list(data["clear"][w, :n]),
                "alts": list(data["alt"][w, :n]), "ups": list(data["up"][w, :n]),
                "contacts": list(data["contact"][w, :n]), "spins": list(data["spin"][w, :n]),
                "vzs": list(data["vz"][w, :n]),
                "xys": [_np.array([data["x"][w, i], data["y"][w, i]]) for i in range(n)],
                "gains": [float(x) for x in data["gain"][w, :n][g]],
                "cmds": [data["cmds"][w, i].copy() for i in range(nd)],
                "resp": [data["resp"][w, i].copy() for i in range(nd)],
                "slam": float(data["slam"][w]), "gain": None}

    def results(self, partners=None, data=None) -> list:
        """One ``SegmentResult`` per world, filled and scored by
        ``batchroll.finish_segment`` on the world's own machine.  ``partners``
        (one ``pair_partner_of`` dict or None per world) is set on the machine
        for its scoring, as ``evaluate.run_segment`` sets it for a second half."""
        from dytiscidae.envs.batchroll import finish_segment
        from dytiscidae.envs.triphibian import SegmentResult

        data = self.download() if data is None else data
        out = []
        for w, e in enumerate(self.envs):
            dom = self.domains[w]
            res = SegmentResult(domain=dom, duration=self.seconds)
            res.survived = bool(data["survived"][w])
            res.failure = FAILURES[int(data["fail"][w])]
            e.pair_partner = None if partners is None else partners[w]
            try:
                finish_segment(e, dom, res, self.recorded(w, data), self.starts[w],
                               data["end"][w], int(data["bad_qacc"][w]), self.seconds,
                               float(data["mean_power"][w]), float(data["max_overload"][w]))
            finally:
                e.pair_partner = None
            out.append(res)
        return out


# ------------------------------------------------------------------ the host side


def start_world(p, seed: int, dom, half: int, params=None, *, spec=None, sea_state=None,
                perturb=None):
    """A machine at the start of one half of ``dom``'s segment for eval seed
    ``seed``: built as ``evaluate_tier1_batch`` builds it, its random stream
    advanced by the resets the host makes before this medium (the host rewinds
    within a pair, so each earlier medium costs one), its own copy of the battery,
    then ``reset``, ``scatter``
    from ``_scatter_seed`` (posed by ``params``) and the task from ``task_seed``,
    ``pair_of`` it for the second half (``half`` 1).  ``params`` None is the
    body's own gait."""
    from dytiscidae.envs.evaluate import _scatter_seed
    from dytiscidae.envs.tasks import pair_of, schedule_for, task_seed
    from dytiscidae.envs.triphibian import DOMAIN_CYCLE, MissionSpec, TriphibianEnv

    import copy

    spec = spec or MissionSpec()
    e = TriphibianEnv(p, seed=int(seed), sea_state=sea_state, perturb=perturb)
    # Each world its own pack: ``PowerBudget`` holds the phenotype's ``Battery``
    # object, so every machine built from one phenotype shares it, and the last
    # world's ``scatter`` set every world's starting charge (2026-10-26: the air
    # halves started at the water draw's charge).
    e.budget.battery = copy.deepcopy(e.budget.battery)
    e.air_launch_height = spec.air_launch_height
    for earlier in DOMAIN_CYCLE[:DOMAIN_CYCLE.index(dom)]:
        e.reset(earlier)
    ss = _scatter_seed(int(seed), dom)
    task = schedule_for(dom, _np.random.default_rng(task_seed(ss)))
    if half:
        task = pair_of(task)
    e.reset(dom)
    e.scatter(_np.random.default_rng(ss), pose=params if params is not None else e.cpg.base)
    e.task = task
    return e


@dataclass
class DeviceDraws:
    """One body's device evaluation: per draw the air and water results (the
    paired score) and both halves, and the notes."""

    eval_seed: int
    draws: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    k_mean: int = 1
    #: ``(draw, medium, half)`` -> that world's record in ``rollout_batch``'s
    #: shape, and its step and decision counts and contact count.
    recorded: dict = field(default_factory=dict)

    def mean(self, medium: str) -> float:
        """The medium's competence, mean over the ``k_mean`` draws at the mean."""
        xs = [float(d[medium].competence) for d in self.draws[:self.k_mean]]
        return float(_np.mean(xs)) if xs else 0.0

    def stats(self, medium: str) -> dict:
        """The medium's ``loop.draw_stats`` over the ``k_mean`` draws at the mean
        (the sampled draws do not score; IMPL_1010 A6)."""
        from dytiscidae.evolution.loop import draw_stats
        return draw_stats([float(d[medium].competence) for d in self.draws[:self.k_mean]])


def body_worlds(p, ctrl, seeds, *, media=MEDIA, spec=None, still: bool = False,
                sea_state=None, perturb=None):
    """The worlds of one body: per draw seed, per medium, the two halves, with
    each world's controller pieces.  Returns ``(envs, domains, params, policies,
    bases, index)``, ``index[(draw, medium, half)]`` the world.  ``still``: every
    world drives ``held_still_params`` (rotors at zero speed) with no
    controller, the still machine of CLAUDE.md rules 6-7."""
    from dytiscidae.envs.triphibian import Domain

    envs, doms, params, pols, bases, index = [], [], [], [], [], {}
    for k, s in enumerate(seeds):
        for med in media:
            dom = Domain(med)
            for half in (0, 1):
                probe = None
                if still:
                    from dytiscidae.envs.triphibian import TriphibianEnv
                    probe = TriphibianEnv(p, seed=int(s), sea_state=sea_state, perturb=perturb)
                    par = probe.held_still_params()
                else:
                    par = ctrl.params if ctrl is not None and ctrl.params is not None else None
                e = start_world(p, s, dom, half, par, spec=spec, sea_state=sea_state,
                                perturb=perturb)
                if par is None:
                    par = e.cpg.base
                index[(k, med, half)] = len(envs)
                envs.append(e)
                doms.append(dom)
                params.append(par)
                pols.append(None if still or ctrl is None else ctrl.policy)
                bases.append(None if still or ctrl is None else ctrl.basis_for(dom))
                del probe
    return envs, doms, params, pols, bases, index


def evaluate_tier1_device(phenos, *, controllers=None, seeds=0, k_mean: int = 1,
                          k_sampled: int = 0, spec=None, segment_seconds: float = 8.0,
                          shared=None, buffer=None, identify_axes=False, dtype=_wp.float32,
                          still: bool = False, streams: int = 1, graph: bool = True,
                          sea_state=None, perturb=None) -> list:
    """Air and water for each body on the device, one ``DeviceDraws`` per body.

    Draw 0 is the body's ``eval_seed``; draw ``d >= 1`` is
    ``placement_draw_seed(eval_seed, d)``.  The first ``k_mean`` draws act at
    the policy mean and give the score; the next ``k_sampled`` sample (their
    observation and action records are what A6's learner reads; the buffer is
    not fed here).  Bodies are launched round-robin on ``streams`` CUDA
    streams, one captured graph per body.
    """
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import pair_partner_of
    from dytiscidae.evolution.loop import placement_draw_seed

    if identify_axes:
        raise NotImplementedError("identification on the device is not built (A4a); "
                                  "identify on the host and pass the bases")
    if buffer is not None:
        raise NotImplementedError("the sampled worlds do not feed the PPO buffer yet (A6)")
    seeds_l = ([int(seeds)] * len(phenos) if not isinstance(seeds, (list, tuple, _np.ndarray))
               else [int(s) for s in seeds])
    K = int(k_mean) + int(k_sampled)
    runs, plans = [], []
    for i, p in enumerate(phenos):
        ctrl = controllers[i] if controllers and controllers[i] is not None else None
        if ctrl is not None and ctrl.params is None:
            ctrl = Controller(params=None, policy=ctrl.policy, bases=ctrl.bases)
        draw_seeds = [seeds_l[i]] + [placement_draw_seed(seeds_l[i], d) for d in range(1, K)]
        envs, doms, params, pols, bases, index = body_worlds(
            p, ctrl, draw_seeds, spec=spec, still=still, sea_state=sea_state, perturb=perturb)
        seg = DeviceSegments(envs, seconds=segment_seconds, dtype=dtype,
                             shared=None if still else shared, policies=pols, bases=bases)
        sampled = [int(k >= k_mean) for k in range(K) for _ in MEDIA for _h in (0, 1)]
        noise = [placement_draw_seed(draw_seeds[k], 101 + j) for k in range(K)
                 for j in range(2 * len(MEDIA))]
        seg.arm(doms, params, sampled=sampled, seeds=noise)
        runs.append(seg)
        plans.append((draw_seeds, index))
    run_bodies(runs, streams=streams, graph=graph)
    out = []
    for seg, (draw_seeds, index) in zip(runs, plans):
        data = seg.download()
        allres = seg.results(data=data)
        partners = [None] * seg.W
        for (k, m, h), w in index.items():
            if h == 1:
                partners[w] = pair_partner_of(allres[index[(k, m, 0)]])
        second = seg.results(partners=partners, data=data)
        dd = DeviceDraws(eval_seed=draw_seeds[0], k_mean=int(k_mean))
        for key, w in index.items():
            dd.recorded[key] = dict(seg.recorded(w, data), steps=int(data["nrec"][w]),
                                    decisions=int(data["ndec"][w]),
                                    contacts=int(data["contacts"][w]))
        for k, s in enumerate(draw_seeds):
            row = {"seed": s}
            for m in MEDIA:
                a, b = allres[index[(k, m, 0)]], second[index[(k, m, 1)]]
                row[m] = b if a.survived else a
                row[m + "_halves"] = (a, b)
            dd.draws.append(row)
        if int(data["contacts"].sum()) > 0:
            dd.notes.append(f"{CONTACT_NOTE}: {int(data['contacts'].sum())} contacts in "
                            f"{int((data['contacts'] > 0).sum())} of {seg.W} worlds")
        out.append(dd)
    return out


def run_bodies(runs, *, streams: int = 1, graph: bool = True) -> None:
    """Every armed ``DeviceSegments`` to its segment's end, the bodies launched
    round-robin on ``streams`` CUDA streams (one captured graph per body)."""
    if not runs:
        return
    dev = _wp.get_device()
    pool = [_wp.Stream(dev) for _ in range(max(1, int(streams)))]
    graphs = []
    for j, seg in enumerate(runs):
        with _wp.ScopedStream(pool[j % len(pool)]):
            seg.step()
            graphs.append(seg.capture() if graph else None)
    n = max(seg.n_steps for seg in runs)
    for i in range(1, n):
        for j, seg in enumerate(runs):
            if i >= seg.n_steps:
                continue
            st = pool[j % len(pool)]
            if graphs[j] is not None:
                _wp.capture_launch(graphs[j], stream=st)
            else:
                with _wp.ScopedStream(st):
                    seg.step()
    for st in pool:
        _wp.synchronize_stream(st)
