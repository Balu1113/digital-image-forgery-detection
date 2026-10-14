"""Contextual bandit that learns which decision threshold to apply.

Reinforcement-learning view of the classifier decision:

* **State**   – the bucket of ``P(real)`` the model output falls into
                (e.g. bucket 4 of 10 covers ``0.4 <= P(real) < 0.5``).
* **Action**  – which decision threshold to use when mapping ``P(real)``
                onto the ``Real``/``Doctored`` verdict.
* **Reward**  – ``+1`` when the user confirms the verdict was correct,
                ``-1`` when they report it was wrong.

The learner is a per-state UCB1 bandit: every (state, action) pair tracks
pulls and cumulative reward; selection picks an untried action first and
otherwise maximises the optimistic index ``mean + c * sqrt(2 ln N / n)``.
Counts and sums are persisted as JSON so learning survives restarts.

Only Python's standard library is used, so this module stays import-light
(unlike the TensorFlow-dependent parts of the service).
"""

import json
import math
import os
import threading
from pathlib import Path

DEFAULT_ACTIONS = (0.35, 0.45, 0.55, 0.65)
DEFAULT_NUM_STATES = 10
DEFAULT_EXPLORATION = 2.0


class ContextualBandit:
    """UCB1 bandit over ``num_states`` states x ``len(actions)`` actions."""

    def __init__(self, num_states=DEFAULT_NUM_STATES, actions=DEFAULT_ACTIONS,
                 exploration=DEFAULT_EXPLORATION, path=None):
        num_states = int(num_states)
        if num_states < 1:
            raise ValueError("num_states must be >= 1")
        actions = tuple(float(a) for a in actions)
        if not actions:
            raise ValueError("at least one action is required")
        if any(not 0.0 < a < 1.0 for a in actions):
            raise ValueError("actions are thresholds and must lie in (0, 1)")

        self.num_states = num_states
        self.actions = actions
        self.exploration = float(exploration)
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._counts = [[0] * len(actions) for _ in range(num_states)]
        self._sums = [[0.0] * len(actions) for _ in range(num_states)]
        self._load()

    # ------------------------------------------------------------------ state
    def state_for(self, real_probability):
        """Bucket a ``P(real)`` value into one of ``num_states`` states."""
        p = min(max(float(real_probability), 0.0), 1.0)
        if p >= 1.0:
            p = 1.0 - 1e-9
        return int(p * self.num_states)

    def _check(self, state, action):
        if not isinstance(state, int) or not 0 <= state < self.num_states:
            raise ValueError(f"state must be an int in [0, {self.num_states})")
        if not isinstance(action, int) or not 0 <= action < len(self.actions):
            raise ValueError(f"action must be an int in [0, {len(self.actions)})")

    # ----------------------------------------------------------------- learning
    def select(self, state):
        """Choose an action for ``state`` using UCB1 (deterministic tie-breaks)."""
        if not isinstance(state, int) or not 0 <= state < self.num_states:
            raise ValueError(f"state must be an int in [0, {self.num_states})")
        with self._lock:
            counts = self._counts[state]
            untried = [i for i, count in enumerate(counts) if count == 0]
            if untried:
                return untried[0]
            total = sum(counts)
            bound = self.exploration * math.sqrt(2.0 * math.log(total))
            return max(
                range(len(counts)),
                key=lambda i: (self._sums[state][i] / counts[i]) + bound / math.sqrt(counts[i]),
            )

    def observe(self, state, action, reward, count=True):
        """Record ``reward`` for a (state, action) pair.

        ``reward`` is normally ``+1``/``-1``; a changed verdict sends the
        delta instead (``+2``/``-2``), which ``count=False`` applies without
        treating it as a fresh pull.
        """
        self._check(state, action)
        reward = float(reward)
        if not math.isfinite(reward) or abs(reward) > 2.0:
            raise ValueError("reward must be between -2 and 2")
        with self._lock:
            self._sums[state][action] += reward
            if count:
                self._counts[state][action] += 1
            self.save()

    def mean(self, state, action):
        """Average reward for a pair, or ``None`` when never pulled."""
        self._check(state, action)
        with self._lock:
            count = self._counts[state][action]
            if not count:
                return None
            return self._sums[state][action] / count

    def summary(self):
        """JSON-friendly snapshot used by ``GET /policy`` and tests."""
        with self._lock:
            states = []
            for state in range(self.num_states):
                counts = list(self._counts[state])
                means = [
                    (self._sums[state][a] / counts[a]) if counts[a] else None
                    for a in range(len(self.actions))
                ]
                best = None
                if any(counts):
                    best = max(
                        (a for a in range(len(self.actions)) if counts[a]),
                        key=lambda a: means[a],
                    )
                states.append({"counts": counts, "means": means, "best_action": best})
        return {
            "num_states": self.num_states,
            "actions": list(self.actions),
            "exploration": self.exploration,
            "pulls": sum(sum(s["counts"]) for s in states),
            "states": states,
        }

    # -------------------------------------------------------------- persistence
    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        counts = payload.get("counts", [])
        sums = payload.get("sums", [])
        if (
            len(counts) == self.num_states
            and len(sums) == self.num_states
            and all(len(row) == len(self.actions) for row in counts + sums)
        ):
            self._counts = [[int(v) for v in row] for row in counts]
            self._sums = [[float(v) for v in row] for row in sums]

    def save(self):
        """Atomically persist the current tables (no-op without a path)."""
        if not self.path:
            return
        with self._lock:
            payload = {"counts": self._counts, "sums": self._sums}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.path)
