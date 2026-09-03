"""Can one morphology-conditioned network represent the per-body optima?

Four architectures have now produced four shared policies that are
indistinguishable from zero on the mission (arch30, arch31, arch32, arch33; the
last measured -0.0014 +/- 0.0016 in-distribution over 48 bodies).  Each of them
answered the same question -- "does a shared controller trained this way help"
-- and each answer cost a full run.  None of them answered the prior question:

    Is there *any* single conditioned network that reproduces what the
    per-candidate optimisers already found?

That is a question about **representability**, not about training, and it can be
answered from data already on disk.  The search runs a per-candidate (1+1)-ES at
every promotion; each of those is a small policy fitted to one body, stored on
that body's archive entry, and demonstrably better on that body than the
untuned weights it started from.  They are free specialist teachers that have
never been used as anything but individual controllers.

So: train one conditioned student to match them by supervised regression (Rusu
et al. 2015 for the method; arXiv 2402.06570 and 2211.14296 for this exact
problem in morphology-conditioned control), hold out bodies it has never seen,
and report how much of the teachers' behaviour it recovers.

* If the student fits held-out bodies, then a shared controller is reachable and
  the four null results are about *training* -- reward, credit assignment,
  sample budget -- and PGA-MAP-Elites or a better learner is worth building.
* If it cannot fit them even with the teachers handed to it directly and no
  exploration problem in the way, then no shared controller of this class is
  reachable, and the capacity question is answered with a number rather than
  with a fifth run.

What this test does and does not establish
------------------------------------------

The teachers are functions; to compare two functions you have to choose where to
evaluate them.  There are no stored observation traces, so the states are drawn
from the observation's own envelope -- every channel over the range
``TriphibianEnv.observation`` can produce, with the domain one-hot and the
morphology block constructed exactly as the environment builds them
(``morphology_channels``, called rather than reimplemented).

That envelope is *wider* than the distribution a rollout visits, which occupies
a low-dimensional manifold inside it.  So the result is conservative in one
direction and only one: a student that fits here would certainly fit the real
distribution, while a student that fails here might still fit the real one.  A
high score is therefore evidence; a low score is a reason to repeat the
measurement on recorded traces before concluding anything.

The comparison that carries the meaning is not the student's error on its own.
It is the student against two baselines evaluated on the same held-out bodies:

* **constant** -- the mean teacher output, ignoring both state and body.  This is
  the bound a shared policy that cannot see the morphology is working against,
  and it is close to what "commanding nothing" achieves.
* **unconditioned** -- the same network with the eight morphology channels
  zeroed.  The gap between this and the conditioned student is exactly what
  conditioning buys, which is the quantity the four null results leave open.
* **shuffled** -- the conditioned network trained with each body's morphology
  paired to a *different* body's teacher.  This is the noise floor.  A
  (1+1)-ES given six steps on 168 weights is a weak optimiser, so some of what
  the teachers encode is optimiser noise rather than a per-body optimum, and a
  student cannot fit noise across bodies whatever the architecture.  If the
  honest pairing scores no better than the shuffled one, the morphology channels
  carry no information about the teacher and every number above is measuring
  the noise floor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..control.cpg import Policy
from ..envs.triphibian import MORPHOLOGY_DIM, TriphibianEnv, morphology_channels

#: Width of the state half of an observation: everything the environment senses
#: or is told, before the body-identity block is appended.
STATE_DIM = TriphibianEnv.OBS_DIM - MORPHOLOGY_DIM


@dataclass(eq=False)
class Teacher:
    """One body's fitted policy, plus who that body is.

    ``eq=False`` because two of its fields are arrays: a generated ``__eq__``
    would compare them elementwise and raise on the truth value of the result,
    from anything as innocent as ``teacher in list``.
    """

    name: str
    island: str
    weights: np.ndarray
    morph: np.ndarray
    fitness: float = 0.0
    mission_fraction: float = 0.0

    def policy(self, n_modes: int) -> Policy:
        return Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=n_modes, hidden=0,
                      weights=self.weights)


def refined_cells(run_dir) -> set:
    """``(island, cell)`` pairs whose policy was actually fitted to that body.

    Every archive entry carries policy weights, but most of them are a parent's
    weights inherited unchanged; only a promotion pays for the per-candidate
    (1+1)-ES.  Runs from 2026-09-03 onward record ``meta["policy_refined"]``
    directly, which is what ``load_teachers`` reads first.  For everything
    before that the promote events are the record, so this recovers it by
    streaming the event log.

    Returns an empty set when there is no event log, which the caller reads as
    "cannot tell" rather than as "none".
    """
    path = Path(run_dir) / "events.jsonl"
    out: set = set()
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            if '"promote"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "promote" and e.get("cell") is not None:
                out.add((str(e.get("island", "")), tuple(e["cell"])))
    return out


def load_teachers(run_dir, *, min_mission: float = 0.0,
                  refined_only: bool = False) -> tuple:
    """Every archive elite in ``run_dir`` that carries fitted policy weights.

    Returns ``(teachers, n_modes)``.  Weight vectors of a shape that does not
    match this build's observation width are skipped and counted rather than
    reinterpreted: a policy fitted against a 19-channel observation means
    nothing when read as a 27-channel one, which is the failure
    ``policy_shape_mismatch`` exists to report inside the search.
    """
    run = Path(run_dir)
    files = sorted(run.glob("archive_*.json")) or sorted(run.glob("archive.json"))
    teachers: list[Teacher] = []
    skipped = 0
    n_modes = 0
    seen: set[str] = set()
    promoted = refined_cells(run) if refined_only else set()
    for path in files:
        island = path.stem.replace("archive_", "") or "archive"
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for e in data.get("elites", []):
            meta = e.get("meta") or {}
            w = meta.get("policy")
            if not w:
                continue
            w = np.asarray(w, float)
            if w.size % (TriphibianEnv.OBS_DIM + 1) != 0:
                skipped += 1
                continue
            k = int(w.size // (TriphibianEnv.OBS_DIM + 1))
            if n_modes and k != n_modes:
                skipped += 1
                continue
            n_modes = k
            if float(meta.get("mission_fraction", 0.0)) < min_mission:
                continue
            if refined_only:
                marked = int(meta.get("policy_refined", 0) or 0) > 0
                if not marked and (island, tuple(e.get("cell", ()))) not in promoted:
                    skipped += 1
                    continue
            # A weight vector of all zeros is "command nothing" -- the state a
            # policy starts in when its parent's weights did not fit.  It is not
            # a teacher; including it would let the student earn credit for
            # reproducing the absence of a controller.
            if not np.any(np.abs(w) > 1e-9):
                skipped += 1
                continue
            name = f"{island}:{tuple(e.get('cell', ()))}"
            if name in seen:
                continue
            seen.add(name)
            teachers.append(Teacher(
                name=name,
                island=island,
                weights=w,
                morph=morphology_channels(
                    mass=float(meta.get("mass", 1.0)),
                    density_ratio=float(meta.get("density_ratio", 1.0)),
                    wing_area=float(meta.get("wing_area", 0.0)),
                    span=float(meta.get("span", 0.0)),
                    aspect_ratio=float(meta.get("aspect_ratio", 0.0)),
                    wing_loading=float(meta.get("wing_loading", 1.0)),
                    n_actuated=int(meta.get("dof", 0)),
                    battery_wh=float(meta.get("battery_wh", 0.0)),
                ),
                fitness=float(e.get("fitness", 0.0)),
                mission_fraction=float(meta.get("mission_fraction", 0.0)),
            ))
    return teachers, (n_modes or 0), skipped


def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws from the observation envelope, state channels only.

    Channel for channel, this is the range ``TriphibianEnv.observation``
    produces: body rates already divided by their scales and clipped to
    [-3, 3], a unit gravity direction in the body frame, bounded depth and
    wetness, a one-hot for the commanded domain, and the four sensed scalars
    added later (depth error, contact, battery, stroke phase and rate).
    """
    x = np.zeros((n, STATE_DIM))
    x[:, 0:3] = np.clip(rng.normal(0.0, 0.8, (n, 3)), -3, 3)   # linear twist / 5
    x[:, 3:6] = np.clip(rng.normal(0.0, 0.5, (n, 3)), -3, 3)   # angular twist / 4
    g = rng.normal(0.0, 1.0, (n, 3))
    x[:, 6:9] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    x[:, 9] = rng.uniform(-1.0, 1.0, n)      # tanh(depth / 5)
    x[:, 10] = rng.uniform(0.0, 1.0, n)      # submerged fraction
    dom = rng.integers(0, 3, n)              # commanded domain, one-hot
    x[np.arange(n), 11 + dom] = 1.0
    x[:, 14] = rng.uniform(-1.0, 1.0, n)     # tanh(depth error / 3)
    x[:, 15] = (rng.random(n) < 0.4).astype(float)   # ground contact
    x[:, 16] = rng.uniform(0.0, 1.0, n)      # battery fraction
    x[:, 17] = rng.uniform(-1.0, 1.0, n)     # stroke phase
    x[:, 18] = np.clip(rng.normal(0.0, 1.0, n), -3, 3)  # stroke rate
    return x


def teacher_targets(teachers, states: np.ndarray, n_modes: int) -> np.ndarray:
    """``(n_teachers, n_states, n_modes)`` of what each teacher commands."""
    out = np.zeros((len(teachers), len(states), n_modes))
    for i, t in enumerate(teachers):
        obs = np.concatenate(
            [states, np.repeat(t.morph[None, :], len(states), axis=0)], axis=1)
        out[i] = t.policy(n_modes).act(obs)
    return out


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of the teachers' output variance a prediction explains.

    Variance is taken about the *global* mean, so 0.0 is exactly the score of
    the constant baseline and a negative number means worse than commanding the
    population average.
    """
    err = float(np.mean((pred - target) ** 2))
    var = float(np.var(target))
    return 1.0 - err / max(var, 1e-12)


@dataclass
class DistillResult:
    n_teachers: int = 0
    n_modes: int = 0
    n_states: int = 0
    held_out: int = 0
    skipped: int = 0
    scores: dict = field(default_factory=dict)
    per_body: list = field(default_factory=list)
    note: str = ""

    def lines(self) -> list:
        out = [
            f"teachers            {self.n_teachers} "
            f"({self.held_out} held out, {self.skipped} unusable)",
            f"states per teacher  {self.n_states}",
            f"intent width        {self.n_modes}",
            "",
            f"{'baseline':<22}{'train R2':>10}{'held-out R2':>14}",
        ]
        for name in ("constant", "shuffled", "unconditioned", "conditioned"):
            s = self.scores.get(name)
            if s:
                out.append(f"{name:<22}{s['train']:>10.3f}{s['test']:>14.3f}")
        if self.note:
            out += ["", self.note]
        return out


def distil(run_dir, *, n_states: int = 256, hidden: int = 128,
           epochs: int = 400, lr: float = 3e-3, held_out: float = 0.3,
           seed: int = 0, min_mission: float = 0.0,
           refined_only: bool = False) -> DistillResult:
    """Fit a conditioned student to the stored per-body teachers.

    ``held_out`` is a fraction of *bodies*, not of samples: the question is
    whether the student generalises to a morphology it has never been fitted
    on, which is the only thing a shared controller in this search would ever be
    asked to do.
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    teachers, n_modes, skipped = load_teachers(
        run_dir, min_mission=min_mission, refined_only=refined_only)
    res = DistillResult(n_teachers=len(teachers), n_modes=n_modes,
                        n_states=n_states, skipped=skipped)
    if len(teachers) < 8 or n_modes == 0:
        res.note = "too few usable teachers to say anything"
        return res

    states = sample_states(n_states, rng)
    targets = teacher_targets(teachers, states, n_modes)

    order = rng.permutation(len(teachers))
    n_test = max(int(round(held_out * len(teachers))), 1)
    test_i, train_i = order[:n_test], order[n_test:]
    res.held_out = int(n_test)

    morph = np.stack([t.morph for t in teachers])
    # (teacher, state) pairs flattened, so a minibatch mixes bodies.
    def block(idx):
        s = np.repeat(states[None, :, :], len(idx), axis=0)
        m = np.repeat(morph[idx][:, None, :], len(states), axis=1)
        x = np.concatenate([s, m], axis=2).reshape(-1, TriphibianEnv.OBS_DIM)
        y = targets[idx].reshape(-1, n_modes)
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32))

    x_tr, y_tr = block(train_i)
    x_te, y_te = block(test_i)

    # Baseline 1: the mean teacher command, ignoring state and body alike.
    const = y_tr.mean(0, keepdim=True).numpy()
    res.scores["constant"] = {
        "train": _r2(np.repeat(const, len(y_tr), 0), y_tr.numpy()),
        "test": _r2(np.repeat(const, len(y_te), 0), y_te.numpy()),
    }

    def train(mask_morph: bool, y_train=None, y_test=None) -> dict:
        y_a = y_tr if y_train is None else y_train
        y_b = y_te if y_test is None else y_test
        torch.manual_seed(seed)
        net = nn.Sequential(
            nn.Linear(TriphibianEnv.OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_modes), nn.Tanh(),
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        keep = torch.ones(TriphibianEnv.OBS_DIM)
        if mask_morph:
            keep[STATE_DIM:] = 0.0
        a, b = x_tr * keep, x_te * keep
        n = a.shape[0]
        batch = min(4096, n)
        for _ in range(epochs):
            idx = torch.randperm(n)[:batch]
            loss = ((net(a[idx]) - y_a[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return {"train": _r2(net(a).numpy(), y_a.numpy()),
                    "test": _r2(net(b).numpy(), y_b.numpy()),
                    "net": net, "keep": keep}

    un = train(mask_morph=True)
    co = train(mask_morph=False)
    res.scores["unconditioned"] = {k: un[k] for k in ("train", "test")}
    res.scores["conditioned"] = {k: co[k] for k in ("train", "test")}

    # The noise floor: same network, same data, each body's morphology paired
    # with another body's teacher.  Whatever this scores is what the method
    # scores on morphology that means nothing.
    perm = rng.permutation(len(teachers))
    while np.any(perm == np.arange(len(teachers))) and len(teachers) > 1:
        perm = rng.permutation(len(teachers))
    shuffled = targets[perm]
    real, targets = targets, shuffled
    y_tr_s, y_te_s = block(train_i)[1], block(test_i)[1]
    targets = real
    sh = train(mask_morph=False, y_train=y_tr_s, y_test=y_te_s)
    res.scores["shuffled"] = {k: sh[k] for k in ("train", "test")}

    # Per held-out body, so a single unrepresentable outlier cannot hide inside
    # a pooled average.
    net, keep = co["net"], co["keep"]
    with torch.no_grad():
        for j in test_i:
            xj, yj = block([j])
            r2 = _r2(net(xj * keep).numpy(), yj.numpy())
            t = teachers[j]
            res.per_body.append((t.name, round(float(r2), 3),
                                 round(t.mission_fraction, 4)))
    res.per_body.sort(key=lambda r: r[1])

    cond = res.scores["conditioned"]["test"]
    gain = cond - res.scores["unconditioned"]["test"]
    floor = res.scores["shuffled"]["test"]
    if cond > 0.8:
        verdict = ("a conditioned network of this size represents the per-body "
                   "optima; the four null results are about training, not "
                   "capacity")
    elif cond > 0.4:
        verdict = ("partially representable: conditioning helps but a large "
                   "share of what the per-body optimisers found does not "
                   "survive being shared")
    elif cond - floor > 0.05:
        verdict = ("above the shuffled floor but far below useful: the "
                   "morphology channels carry a little information about the "
                   "per-body optimum and nothing like enough")
    else:
        verdict = ("at the shuffled floor: these morphology channels carry no "
                   "information about the per-body optimum, so no shared "
                   "controller conditioned on them is reachable at any "
                   "capacity or training budget")
    res.note = (
        f"conditioning is worth {gain:+.3f} R2 over the unconditioned student "
        f"and {cond - floor:+.3f} over the shuffled floor -- {verdict}\n"
        f"train R2 {res.scores['conditioned']['train']:.3f} against held-out "
        f"{cond:.3f}: the student fits the teachers it saw, so what fails is "
        f"generalisation across bodies, not capacity")
    return res
