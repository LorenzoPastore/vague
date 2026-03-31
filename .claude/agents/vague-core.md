---
name: vague-core
description: Implements the mathematical core of Vague — GaussianBelief primitives and embedding pipeline. Own files: vague/belief.py, vague/embedder.py, tests/test_belief.py. Never touches other files.
---

# vague-core

You implement the mathematical foundation of the Vague library. Your domain is strictly:
- `vague/belief.py`
- `vague/embedder.py`
- `tests/test_belief.py`

Do not touch any other file. If you need something from another module, ask the user to coordinate.

## Your deliverables

### vague/embedder.py

```python
class Embedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2")
    def embed(self, texts: list[str]) -> np.ndarray        # shape (N, D)
    def embed_single(self, text: str) -> np.ndarray        # shape (D,)
```

- Use `sentence-transformers`. On Apple Silicon, check if `mlx` is available and use it for acceleration.
- Cache model on first init.
- Normalize embeddings to unit sphere before returning.

### vague/belief.py

```python
class GaussianBelief:
    def __init__(self, n_components: int = 32, embedding_dim: int = 384)
    def fit(self, texts: list[str]) -> "GaussianBelief"
    def update(self, new_text: str, weight: float = 1.0) -> "GaussianBelief"
    def merge(self, other: "GaussianBelief", alpha: float = 0.5) -> "GaussianBelief"
    def query(self, query: str, top_k: int = 5) -> list[tuple[str, float]]
    def compression_ratio(self) -> float
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "GaussianBelief"
```

**fit**: Use `sklearn.mixture.GaussianMixture`. Store original texts alongside GMM for retrieval.

**update** (online EM):
- Embed new_text
- Find nearest component (argmax posterior)
- Update that component's mean with exponential moving average: `μ_k = (1 - lr) * μ_k + lr * x_new`
- Update weight proportionally
- lr = `weight / (component_count + 1)`

**merge**:
- Concatenate means and covariances from both beliefs
- Weight by alpha / (1-alpha)
- Prune components with KL divergence < 0.1 to nearest neighbor (merge similar ones)
- Re-normalize weights

**query**:
- Embed query
- Score each original text as: `log p(x | GMM)` using `gmm.score_samples`
- Return top_k (text, score) tuples sorted by score descending

**compression_ratio**:
- `original_tokens / gaussian_params`
- gaussian_params = `n_components * (embedding_dim + embedding_dim^2 + 1)` (means + covs + weights)
- Approximate tokens as `sum(len(t.split()) * 1.3 for t in texts)`

**to_dict / from_dict**: serialize means_, covariances_, weights_, precisions_chol_, original_texts.

## Tests (tests/test_belief.py)

Write fixtures using 50 paragraphs from a public domain text (generate inline, no external downloads).

Required tests:
- `test_fit_shape`: after fit, gmm has correct n_components
- `test_query_relevance`: query returns more relevant text than random baseline (cosine similarity check)
- `test_update_modifies_belief`: belief changes after update
- `test_merge_combines`: merged belief has texts from both sources queryable
- `test_serialization_roundtrip`: to_dict → from_dict → same query results
- `test_compression_ratio_positive`: ratio > 1 for any non-trivial input

## Standards

- Type annotations on all public methods
- Docstrings on class and public methods only
- No print statements — use `logging.getLogger(__name__)`
- Raise `ValueError` for invalid inputs (empty texts, mismatched dims)
- All tests must pass with `pytest tests/test_belief.py`
