"""SummaryBelief — Gaussian Splatting with compressed per-component payloads.

Each Gaussian component stores a LLM-generated summary of its assigned chunks
instead of raw text. At query time, the top-k component summaries are returned
rather than retrieved raw chunks — making the representation self-contained.

This is the key difference from GaussianBelief:

    GaussianBelief: GMM as retrieval index → returns raw chunks (large payload)
    SummaryBelief:  GMM as compressed representation → returns summaries (small payload)

Analogy to 3D Gaussian Splatting:
    3DGS:          each splat stores color/opacity (scalar payload)
    SummaryBelief: each splat stores a text summary (compressed semantic payload)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from vague.belief import GaussianBelief

logger = logging.getLogger(__name__)


class SummaryBelief(GaussianBelief):
    """GaussianBelief where each component carries a compressed text summary.

    Usage::

        from vague.summary_belief import SummaryBelief

        def llm_fn(prompt): ...  # any llm_fn-compatible callable

        mem = SummaryBelief(n_components=16)
        mem.fit_with_summaries(chunks, llm_fn=llm_fn)
        results = mem.query("your question", top_k=3)
        # returns component summaries, not raw chunks
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._summaries: dict[int, str] = {}   # component_idx → summary text
        self._component_sizes: dict[int, int] = {}  # component_idx → n chunks

    # ------------------------------------------------------------------
    # Extended fit
    # ------------------------------------------------------------------

    def fit_with_summaries(
        self,
        texts: list[str],
        llm_fn: Callable[[str], str],
        max_chunk_tokens: int = 1500,
    ) -> "SummaryBelief":
        """Fit GMM and generate one summary per component.

        Args:
            texts: List of text chunks to fit.
            llm_fn: LLM callable for summary generation.
            max_chunk_tokens: Truncate per-component context before summarizing
                to avoid exceeding model context limits.

        Returns:
            self
        """
        # 1. Fit the GMM as usual
        self.fit(texts)

        # 2. Hard-assign each chunk to its most likely component
        log_resp = self._component_log_prob(self._embeddings, self._gmm)  # (N, K)
        log_resp += np.log(self._gmm.weights_)
        assignments = np.argmax(log_resp, axis=1)  # (N,) — component index per chunk

        K = len(self._gmm.weights_)

        # 3. For each component, collect assigned chunks and summarize
        for k in range(K):
            chunk_indices = np.where(assignments == k)[0]
            if len(chunk_indices) == 0:
                self._summaries[k] = ""
                self._component_sizes[k] = 0
                continue

            component_chunks = [texts[i] for i in chunk_indices]
            self._component_sizes[k] = len(component_chunks)

            # Truncate to avoid LLM context overflow
            combined = "\n\n".join(component_chunks)
            combined = combined[: max_chunk_tokens * 4]  # rough char limit

            prompt = (
                "Summarize the following passages in 2-3 sentences, "
                "preserving all factual details and named entities:\n\n"
                f"{combined}\n\nSummary:"
            )
            self._summaries[k] = llm_fn(prompt)
            logger.debug(
                "Component %d: %d chunks → summary (%d chars)",
                k, len(component_chunks), len(self._summaries[k]),
            )

        logger.info(
            "SummaryBelief fitted: %d components, %d total chunks",
            K, len(texts),
        )
        return self

    # ------------------------------------------------------------------
    # Override query to return summaries
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-k component summaries ranked by relevance to query.

        Unlike GaussianBelief.query() which returns raw chunks, this returns
        the compressed per-component summaries — the true Gaussian payload.

        Args:
            query: Query string.
            top_k: Number of components to return.

        Returns:
            List of (summary, score) tuples sorted by score descending.
            Empty summaries (unassigned components) are excluded.
        """
        if not self._fitted:
            raise ValueError("SummaryBelief must be fitted before calling query().")
        if not self._summaries:
            raise ValueError("Call fit_with_summaries() before query().")

        q_vec = self._embedder.embed_single(query)

        # Score each component: log p(component | query)
        log_resp = self._component_log_prob(q_vec.reshape(1, -1), self._gmm)[0]  # (K,)
        log_resp += np.log(self._gmm.weights_)

        K = len(self._gmm.weights_)
        scored = [
            (self._summaries[k], float(log_resp[k]))
            for k in range(K)
            if self._summaries.get(k, "")
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """Return statistics about the per-component summaries."""
        sizes = list(self._component_sizes.values())
        lengths = [len(s) for s in self._summaries.values() if s]
        return {
            "n_components": len(self._gmm.weights_) if self._fitted else 0,
            "n_components_with_summary": len(lengths),
            "avg_chunks_per_component": float(np.mean(sizes)) if sizes else 0,
            "avg_summary_chars": float(np.mean(lengths)) if lengths else 0,
            "total_summary_chars": sum(lengths),
        }
