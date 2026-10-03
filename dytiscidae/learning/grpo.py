"""GRPO for the shared policy (ROADMAP item N): group-relative advantages.

What it is, here
----------------

Each generation a few bodies (``grpo_bodies``) are rolled out ``grpo_group``
times each, with different exploration streams, on the same scatter and task
the generation's other machines faced.  Those rollouts are *learning-only*: they
are scored by nobody, filed nowhere, and exist to be read by one function.  A
trajectory's advantage is its terminal reward relative to the other rollouts of
the same body *in the same segment kind*::

    A = (R - mean_group(R)) / (std_group(R) + STD_EPS)

and the same number is given to every step of the trajectory.  It replaces GAE
for these rows.  Nothing else about the update changes: the surrogate is the
same clipped ratio, and the rows share minibatches with the ordinary PPO rows
(``ppo_update(..., group_buffer=...)``).

Why this and not a critic's baseline.  The reward is one terminal scalar spread
over ~2,000 steps, and most of its variance is *which body was rolled out*
(roadmap §2: 84.1% explained by the domain one-hot alone, and of the rest about
nine tenths by the body).  A group of rollouts of one body cancels the body
exactly and leaves "did this sampled action sequence beat this body's other
attempts", the only signal in the batch that is about the policy.

Decisions, and what each would have cost to get the other way
-------------------------------------------------------------

* **Group = (group id, segment kind).**  A body's mission is seven segments
  (three domains, four crossings), each with its own terminal reward.  Pooling
  them would compare a water competence to a crossing quality.
* **No potential shaping in a group advantage.**  Shaping adds
  ``gamma*Phi(s') - Phi(s)`` per step, which telescopes to ``-Phi(s_0)`` over
  the trajectory.  All rollouts of one group start from the same scattered
  state, so it is the same constant for every member and cancels in
  ``R - mean(R)``.  Adding it per step would only add the part that does not
  telescope (the gamma discount), i.e. noise.
* **No value loss on group rows; the entropy bonus stays.**  GRPO has no critic,
  and regressing it on a terminal reward the advantage has just subtracted the
  body from would pull the shared critic toward the wrong target.  The entropy
  term does not depend on the advantage at all, and it is what stops a
  standardised-advantage update from collapsing the exploration width.
* **One update, shared minibatches, not two updates.**  A second update after
  the first would see an observation normaliser that ``ppo_update`` has already
  moved (``SharedPolicy.observe`` runs at the end of an update), so its first
  importance ratio would already be off 1 before a gradient step.  Advantages
  are the exception: ordinary rows are standardised among themselves exactly
  as before and group rows are left as they are, because a batch-wide
  standardisation would add the mean and scale of a different kind of number
  back into them.
* **STD_EPS = 0.05, a soft floor.**  ``_terminal_scale`` already refuses a scale
  below 0.05 for the same reason: competences live on [0, 1], and a group whose
  rewards differ by 1e-9 (a body that does nothing whatever it explores) would
  otherwise have that float noise stretched to +-1.  With it, such a group
  gets an advantage of order 1e-8, and a group with identical returns gets
  exactly 0.  Population standard deviation (ddof 0), so the advantages of a
  group with std >> 0.05 have unit RMS whatever its size.
* **A group that cannot be normalised is dropped and counted, not zero-filled.**
  A group with fewer than two usable members, an ungrouped trajectory, or a
  non-finite reward does not carry "no signal"; it carries "unmeasured", and
  those must not share a value (the project's standing rule).  ``stats()``
  reports each kind.
"""

from __future__ import annotations

import numpy as np

from .ppo import RolloutBuffer

#: Which learners ``SearchConfig.shared_learner`` accepts.
LEARNERS = ("ppo", "grpo", "ppo+grpo")

#: Added to a group's standard deviation before dividing.  See the module doc.
STD_EPS = 0.05


def group_advantages(rewards, keys, eps: float = STD_EPS) -> np.ndarray:
    """``(R - mean_g) / (std_g + eps)`` for each element, ``g`` being its key.

    ``keys`` is any hashable per element.  The mean and the (population)
    standard deviation are taken over the elements sharing a key, and only
    over those: no statistic crosses a group boundary.  Pure arithmetic, so
    the test can pin it against the definition.
    """
    r = np.asarray(rewards, float)
    adv = np.zeros(r.shape, float)
    members: dict = {}
    for i, k in enumerate(keys):
        members.setdefault(k, []).append(i)
    for idx in members.values():
        v = r[idx]
        adv[idx] = (v - v.mean()) / (v.std() + eps)
    return adv


class GroupRolloutBuffer(RolloutBuffer):
    """Trajectories of repeated rollouts, flattened with group-relative advantages.

    Same ``add`` as the ordinary buffer.  ``build`` returns the same six
    arrays, with the advantage broadcast over each trajectory's steps, and
    ``RET`` equal to ``VAL`` (no value target exists; ``ppo_update`` ignores
    it for these rows).
    """

    def __init__(self):
        super().__init__(shaping=0.0)

    # ------------------------------------------------------------ the groups

    def _usable(self):
        """``(kept trajectories, their group keys, dropped-by-reason counts)``."""
        dropped = {"ungrouped": 0, "nonfinite": 0, "singleton": 0}
        by: dict = {}
        for t in self.trajectories:
            if t.group < 0:
                dropped["ungrouped"] += 1
                continue
            by.setdefault((t.group, t.tag), []).append(t)
        kept, keys = [], []
        for key, ts in by.items():
            if not all(np.isfinite(t.terminal_reward) for t in ts):
                dropped["nonfinite"] += len(ts)
                continue
            if len(ts) < 2:
                dropped["singleton"] += len(ts)
                continue
            for t in ts:
                kept.append(t)
                keys.append(key)
        return kept, keys, dropped

    @property
    def n_transitions(self) -> int:
        return sum(len(t) for t in self._usable()[0])

    def advantages(self):
        """``(kept trajectories, advantage per trajectory)``."""
        kept, keys, _ = self._usable()
        adv = group_advantages([t.terminal_reward for t in kept], keys)
        return kept, adv

    def build(self):
        kept, adv = self.advantages()
        if not kept:
            raise ValueError("no usable group: build() needs at least one "
                             "group of two or more rollouts")
        O, A, L, ADV, VAL = [], [], [], [], []
        for t, a in zip(kept, adv):
            n = len(t)
            O.append(np.asarray(t.obs, np.float32))
            A.append(np.asarray(t.act, np.float32))
            L.append(np.asarray(t.logp, np.float32))
            ADV.append(np.full(n, a))
            VAL.append(np.asarray(t.val, float))
        val = np.concatenate(VAL)
        return (np.concatenate(O), np.concatenate(A), np.concatenate(L),
                np.concatenate(ADV), val, val)

    def stats(self) -> dict:
        """What a later reading needs to tell "no signal" from "no data".

        ``grpo_group_std`` is the mean within-group standard deviation of the
        terminal reward, per segment kind: the size of the signal the update
        had.  ``grpo_zero_groups`` counts groups whose members all earned the
        same reward (an advantage of exactly 0).
        """
        kept, keys, dropped = self._usable()
        adv = group_advantages([t.terminal_reward for t in kept], keys)
        groups: dict = {}
        for t, k in zip(kept, keys):
            groups.setdefault(k, []).append(t.terminal_reward)
        std_by_tag: dict = {}
        zero = 0
        for (_g, tag), v in groups.items():
            std_by_tag.setdefault(tag, []).append(float(np.std(v)))
            zero += int(np.ptp(v) == 0.0)
        return {
            "grpo_groups": len(groups),
            "grpo_trajectories": len(kept),
            "grpo_transitions": sum(len(t) for t in kept),
            "grpo_bodies": len({g for g, _tag in groups}),
            "grpo_zero_groups": zero,
            "grpo_adv_abs": float(np.mean(np.abs(adv))) if len(adv) else 0.0,
            "grpo_group_std": {k: round(float(np.mean(v)), 4)
                               for k, v in sorted(std_by_tag.items())},
            "grpo_dropped": dropped,
        }


__all__ = ["LEARNERS", "STD_EPS", "GroupRolloutBuffer", "group_advantages"]
