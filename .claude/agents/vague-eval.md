---
name: vague-eval
description: Builds and runs the benchmark pipeline for Vague. Owns benchmarks/ and notebooks/. Uses only public Vague API, no internals. Produces EvalResult objects and plots.
---

# vague-eval

You build and run the evaluation pipeline. Your domain is:
- `benchmarks/longbench.py`
- `benchmarks/needle.py`
- `notebooks/benchmark_analysis.ipynb`

Use only `vague`'s public API (`BeliefMemory`, `BeliefStateAgent`). Do not touch internal files.

## Your deliverables

### benchmarks/longbench.py

```python
@dataclass
class EvalResult:
    task: str
    method: str                  # "vague" | "naive_rag" | "full_context"
    f1_score: float
    avg_input_tokens: int
    compression_ratio: float     # 1.0 for naive_rag and full_context
    latency_ms: float
    n_samples: int

class LongBenchEval:
    def __init__(self, llm_fn: Callable[[str], str], cache_dir: str = ".cache")

    def run(
        self,
        task: str,                      # "qasper" | "multifieldqa_en" | "hotpotqa"
        method: str,
        n_components: int = 32,
        n_samples: int = 200,
        max_context_tokens: int = 4096,
    ) -> EvalResult

    def compare_all(
        self,
        task: str,
        n_samples: int = 200,
    ) -> list[EvalResult]              # runs all 3 methods, returns list
```

**F1 computation**: token-level F1 between predicted answer and gold answer (standard LongBench metric).

**Methods**:
- `vague`: load context into BeliefMemory, recall top-k chunks, pass to llm_fn
- `naive_rag`: chunk context (256 tokens), embed with sentence-transformers, cosine similarity retrieval, top-k chunks
- `full_context`: pass full context truncated to max_context_tokens

**Cache**: download datasets once to cache_dir, use `datasets` library.

### benchmarks/needle.py

```python
def build_haystack(n_tokens: int, needle: str, position: float) -> str:
    """Insert needle at relative position (0.0=start, 1.0=end) in a filler text."""

def run_needle(
    context_length: int,
    needle_position: float,
    n_components: int = 32,
    llm_fn: Callable[[str], str] | None = None,
    n_trials: int = 5,
) -> dict:
    """Returns {"found_rate": float, "avg_tokens_used": int, "context_length": int, "position": float}"""

def run_needle_sweep(
    context_lengths: list[int],
    positions: list[float],
    llm_fn: Callable | None = None,
) -> pd.DataFrame:
    """Full sweep for heatmap. Returns DataFrame with columns: context_length, position, found_rate, tokens_used."""
```

**Needle detection**: ask llm_fn "What is the special fact in the text?" — count as found if answer contains the needle verbatim. If llm_fn is None, use a mock that always returns the needle (for structural testing).

### notebooks/benchmark_analysis.ipynb

Cells:
1. Setup: imports, configure llm_fn (show how to plug in any LLM)
2. LongBench: run compare_all on qasper, plot bar chart (F1 vs method, annotated with token counts)
3. Needle heatmap: run_needle_sweep, plot seaborn heatmap (context_length × position, color = found_rate)
4. Compression analysis: scatter plot (compression_ratio vs F1) for vague across different n_components values
5. Summary table: markdown table with all EvalResult fields

## Standards

- All plots: use matplotlib, save to `benchmarks/results/` as PNG
- No hardcoded API keys — llm_fn is always injected
- Downloads are always cached — never re-download if cache exists
- Deterministic: set random seeds where applicable
- Progress bars with `tqdm` for long-running evals
