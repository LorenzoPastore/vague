"""Gaussian mixture belief representation for the Vague library."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from vague.embedder import Embedder

logger = logging.getLogger(__name__)


class GaussianBelief:
    """Represents a corpus of text as a Gaussian Mixture Model in embedding space.

    Supports online updates, belief merging, semantic querying, and
    serialization.
    """

    def __init__(
        self,
        n_components: int = 32,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize GaussianBelief.

        Args:
            n_components: Number of Gaussian components in the mixture.
            embedding_dim: Dimensionality of the embedding space.

        Raises:
            ValueError: If n_components < 1 or embedding_dim < 1.
        """
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")

        self.n_components = n_components
        self.embedding_dim = embedding_dim

        self._embedder = Embedder()
        self._gmm: GaussianMixture | None = None
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, D)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> "GaussianBelief":
        """Fit the GMM to a list of texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            self

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        n_components = min(self.n_components, len(texts))
        self._texts = list(texts)
        self._embeddings = self._embedder.embed(texts)

        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            max_iter=200,
            random_state=42,
        )
        self._gmm.fit(self._embeddings)
        self._fitted = True
        logger.debug("Fitted GMM with %d components on %d texts.", n_components, len(texts))
        return self

    def update(self, new_text: str, weight: float = 1.0) -> "GaussianBelief":
        """Incorporate a new text via online exponential moving average update.

        Args:
            new_text: Text to incorporate.
            weight: Relative weight of the new observation (>0).

        Returns:
            self

        Raises:
            ValueError: If not yet fitted or weight <= 0.
        """
        if not self._fitted:
            raise ValueError("GaussianBelief must be fitted before calling update().")
        if weight <= 0:
            raise ValueError("weight must be > 0.")

        x_new = self._embedder.embed_single(new_text)  # shape (D,)

        # Posterior responsibilities for x_new under current GMM
        log_resp = self._gmm._estimate_log_prob(x_new.reshape(1, -1))  # (1, K)
        log_resp += np.log(self._gmm.weights_)
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        k = int(np.argmax(log_resp[0]))

        n_components = len(self._gmm.weights_)
        lr = weight / (n_components + 1)

        self._gmm.means_[k] = (1 - lr) * self._gmm.means_[k] + lr * x_new

        # Update weight of nearest component proportionally
        self._gmm.weights_[k] += lr * (1.0 - self._gmm.weights_[k])
        self._gmm.weights_ /= self._gmm.weights_.sum()

        # Recompute precision cholesky for updated component to keep GMM consistent
        cov_k = self._gmm.covariances_[k]
        try:
            L = np.linalg.cholesky(cov_k)
            self._gmm.precisions_cholesky_[k] = np.linalg.solve(L, np.eye(len(L))).T
        except np.linalg.LinAlgError:
            pass  # keep existing precisions if covariance is not PD

        # Append text and embedding for future queries
        self._texts.append(new_text)
        self._embeddings = np.vstack([self._embeddings, x_new.reshape(1, -1)])

        logger.debug("Updated component %d with lr=%.4f.", k, lr)
        return self

    def merge(self, other: "GaussianBelief", alpha: float = 0.5) -> "GaussianBelief":
        """Merge another GaussianBelief into this one.

        Components from self are weighted by alpha; components from other by
        (1 - alpha).  Similar components (KL divergence < 0.1) are pruned by
        merging them.

        Args:
            other: Another fitted GaussianBelief with the same embedding_dim.
            alpha: Weight assigned to self's components (0 < alpha < 1).

        Returns:
            A new GaussianBelief containing the merged result.

        Raises:
            ValueError: If either belief is not fitted, dims mismatch, or
                alpha out of range.
        """
        if not self._fitted:
            raise ValueError("self must be fitted before merging.")
        if not other._fitted:
            raise ValueError("other must be fitted before merging.")
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Beliefs must have the same embedding_dim.")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")

        # Weighted concatenation of component parameters
        means = np.vstack([self._gmm.means_, other._gmm.means_])
        covs = np.concatenate([self._gmm.covariances_, other._gmm.covariances_], axis=0)

        w_self = self._gmm.weights_ * alpha
        w_other = other._gmm.weights_ * (1 - alpha)
        weights = np.concatenate([w_self, w_other])
        weights /= weights.sum()

        # Prune near-duplicate components (KL < 0.1 to nearest neighbour)
        keep_mask = self._prune_components(means, covs, weights, kl_threshold=0.1)
        means = means[keep_mask]
        covs = covs[keep_mask]
        weights = weights[keep_mask]
        weights /= weights.sum()

        # Build merged belief
        merged = GaussianBelief(
            n_components=len(weights),
            embedding_dim=self.embedding_dim,
        )
        merged._texts = self._texts + other._texts
        merged._embeddings = np.vstack([self._embeddings, other._embeddings])

        merged._gmm = self._build_gmm_from_params(means, covs, weights)
        merged._fitted = True
        logger.debug(
            "Merged beliefs: %d + %d → %d components.",
            len(self._gmm.weights_),
            len(other._gmm.weights_),
            len(weights),
        )
        return merged

    def query(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top_k most relevant texts for a query string.

        Scores are log-likelihoods under the GMM evaluated at each text's
        embedding.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (text, score) tuples sorted by score descending.

        Raises:
            ValueError: If not fitted or top_k < 1.
        """
        if not self._fitted:
            raise ValueError("GaussianBelief must be fitted before calling query().")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        q_vec = self._embedder.embed_single(query)  # shape (D,)

        # Posterior of query under each GMM component: r(k) = p(k | q)
        log_resp = self._gmm._estimate_log_prob(q_vec.reshape(1, -1))[0]  # (K,)
        log_resp += np.log(self._gmm.weights_)
        log_resp -= logsumexp(log_resp)
        resp = np.exp(log_resp)  # (K,) — soft assignment of query to components

        # Score each stored text: weighted cosine similarity to GMM component means
        # score(i) = sum_k r(k) * cos(x_i, mu_k)
        # Since embeddings are unit-normalized: cos(x_i, mu_k) = x_i @ mu_k / ||mu_k||
        mu_norms = np.linalg.norm(self._gmm.means_, axis=1, keepdims=True) + 1e-10
        mu_normalized = self._gmm.means_ / mu_norms  # (K, D)
        # self._embeddings: (N, D), mu_normalized: (K, D)
        cos_sims = self._embeddings @ mu_normalized.T  # (N, K)
        scores = cos_sims @ resp  # (N,) — GMM-weighted relevance score

        k = min(top_k, len(self._texts))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self._texts[i], float(scores[i])) for i in top_indices]

    def compression_ratio(self) -> float:
        """Ratio of approximate original token count to Gaussian parameter count.

        Returns:
            compression_ratio > 0.

        Raises:
            ValueError: If not yet fitted.
        """
        if not self._fitted:
            raise ValueError("GaussianBelief must be fitted before computing compression_ratio().")

        approx_tokens = sum(len(t.split()) * 1.3 for t in self._texts)
        D = self.embedding_dim
        K = len(self._gmm.weights_)
        gaussian_params = K * (D + D * D + 1)
        return float(approx_tokens / gaussian_params)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the belief to a plain dictionary.

        Returns:
            Dictionary containing GMM parameters and original texts.

        Raises:
            ValueError: If not fitted.
        """
        if not self._fitted:
            raise ValueError("GaussianBelief must be fitted before serialization.")

        return {
            "n_components": self.n_components,
            "embedding_dim": self.embedding_dim,
            "means_": self._gmm.means_.tolist(),
            "covariances_": self._gmm.covariances_.tolist(),
            "weights_": self._gmm.weights_.tolist(),
            "precisions_cholesky_": self._gmm.precisions_cholesky_.tolist(),
            "original_texts": self._texts,
            "embeddings_": self._embeddings.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GaussianBelief":
        """Deserialize a GaussianBelief from a dictionary produced by to_dict().

        Args:
            d: Dictionary as returned by to_dict().

        Returns:
            A fitted GaussianBelief instance.

        Raises:
            ValueError: If required keys are missing.
        """
        required = {"n_components", "embedding_dim", "means_", "covariances_",
                    "weights_", "precisions_cholesky_", "original_texts"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"Missing keys in dict: {missing}")

        means = np.array(d["means_"])
        covariances = np.array(d["covariances_"])
        weights = np.array(d["weights_"])
        precisions_chol = np.array(d["precisions_cholesky_"])
        texts = d["original_texts"]
        K = len(weights)

        belief = cls(n_components=d["n_components"], embedding_dim=d["embedding_dim"])
        belief._gmm = belief._build_gmm_from_params(means, covariances, weights, precisions_chol)
        belief._texts = list(texts)
        if "embeddings_" in d:
            belief._embeddings = np.array(d["embeddings_"])
        else:
            belief._embeddings = belief._embedder.embed(texts)
        belief._fitted = True
        logger.debug("Deserialized GaussianBelief with %d components.", K)
        return belief

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_gaussian(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """KL divergence KL(N1 || N2) for two full-covariance Gaussians."""
        D = len(mu1)
        try:
            sigma2_inv = np.linalg.inv(sigma2)
            diff = mu2 - mu1
            sign1, ld1 = np.linalg.slogdet(sigma1)
            sign2, ld2 = np.linalg.slogdet(sigma2)
            if sign1 <= 0 or sign2 <= 0:
                return float("inf")
            kl = 0.5 * (
                np.trace(sigma2_inv @ sigma1)
                + diff @ sigma2_inv @ diff
                - D
                + ld2 - ld1
            )
            return float(max(kl, 0.0))
        except np.linalg.LinAlgError:
            return float("inf")

    @staticmethod
    def _prune_components(
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
        kl_threshold: float = 0.1,
    ) -> np.ndarray:
        """Return a boolean mask of components to keep after pruning near-duplicates."""
        K = len(weights)
        removed = np.zeros(K, dtype=bool)

        for i in range(K):
            if removed[i]:
                continue
            for j in range(i + 1, K):
                if removed[j]:
                    continue
                kl = GaussianBelief._kl_gaussian(
                    means[i], covs[i], means[j], covs[j]
                )
                if kl < kl_threshold:
                    # Merge j into i (weighted average of means)
                    w_total = weights[i] + weights[j]
                    means[i] = (weights[i] * means[i] + weights[j] * means[j]) / w_total
                    weights[i] = w_total
                    removed[j] = True

        return ~removed

    @staticmethod
    def _build_gmm_from_params(
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        precisions_chol: np.ndarray | None = None,
    ) -> GaussianMixture:
        """Construct a GaussianMixture with pre-set parameters (no fitting)."""
        K, D = means.shape
        gmm = GaussianMixture(n_components=K, covariance_type="full")
        gmm.means_ = means.copy()
        gmm.covariances_ = covariances.copy()
        gmm.weights_ = weights.copy()

        if precisions_chol is not None:
            gmm.precisions_cholesky_ = precisions_chol.copy()
        else:
            # Recompute from covariances
            pchol = np.zeros_like(covariances)
            for k in range(K):
                try:
                    L = np.linalg.cholesky(covariances[k])
                    pchol[k] = np.linalg.solve(L, np.eye(D)).T
                except np.linalg.LinAlgError:
                    pchol[k] = np.eye(D)
            gmm.precisions_cholesky_ = pchol

        # Required by sklearn for score_samples to work without calling fit
        gmm.converged_ = True
        gmm.n_iter_ = 0
        gmm.lower_bound_ = -np.inf
        return gmm
