"""High-level BeliefMemory API wrapping GaussianBelief."""

from __future__ import annotations

import json
import logging
import math
from typing import ClassVar

from vague.belief import GaussianBelief

logger = logging.getLogger(__name__)

_LAZY_INIT_THRESHOLD: int = 10


class BeliefMemory:
    """Persistent, queryable memory backed by a Gaussian Mixture Model."""

    _LAZY_THRESHOLD: ClassVar[int] = _LAZY_INIT_THRESHOLD

    def __init__(self, n_components: int = 32) -> None:
        self.n_components = n_components
        self.belief: GaussianBelief = GaussianBelief(n_components=n_components)
        self._pending: list[str] = []
        self._n_observations: int = 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def remember(self, text: str) -> None:
        """Incorporate a single text into memory."""
        self._n_observations += 1
        if self.belief._fitted:
            self.belief.update(text)
        else:
            self._pending.append(text)
            if len(self._pending) >= self._LAZY_THRESHOLD:
                logger.debug("Lazy-init: fitting belief on %d texts.", len(self._pending))
                self.belief.fit(self._pending)
                self._pending = []

    def remember_batch(self, texts: list[str]) -> None:
        """Incorporate multiple texts."""
        for text in texts:
            self.remember(text)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to k plain strings most relevant to query."""
        if not self.belief._fitted:
            # Not enough data yet — return whatever pending texts we have
            return self._pending[:k]
        results = self.belief.query(query, top_k=k)
        return [text for text, _ in results]

    # ------------------------------------------------------------------
    # Sharing
    # ------------------------------------------------------------------

    def share_with(self, other: "BeliefMemory") -> None:
        """Merge self's belief into other's belief in-place."""
        if not self.belief._fitted or not other.belief._fitted:
            raise ValueError("Both memories must be fitted before sharing.")
        merged = self.belief.merge(other.belief)
        other.belief = merged
        other._n_observations += self._n_observations

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about this memory."""
        n_obs = self._n_observations
        if self.belief._fitted:
            cr = self.belief.compression_ratio()
            weights = self.belief._gmm.weights_
            entropy = -sum(w * math.log(w) for w in weights if w > 0)
            n_comp = len(weights)
        else:
            cr = 0.0
            entropy = 0.0
            n_comp = self.n_components
        return {
            "n_components": n_comp,
            "n_observations": n_obs,
            "compression_ratio": cr,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize memory to a JSON file at path."""
        if not self.belief._fitted:
            raise ValueError("Memory must be fitted before saving.")
        data = self.belief.to_dict()
        data["_n_observations"] = self._n_observations
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.debug("Saved BeliefMemory to %s.", path)

    @classmethod
    def load(cls, path: str) -> "BeliefMemory":
        """Deserialize a BeliefMemory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_obs = data.pop("_n_observations", 0)
        belief = GaussianBelief.from_dict(data)
        mem = cls(n_components=belief.n_components)
        mem.belief = belief
        mem._n_observations = n_obs
        logger.debug("Loaded BeliefMemory from %s.", path)
        return mem
