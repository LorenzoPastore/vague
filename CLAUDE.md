# Vague — Dev Context

Probabilistic belief-state memory for LLM agents using Gaussian Mixture Models.

## Environment

```bash
source .venv/bin/activate   # always activate before running anything
```

Package is installed editable — changes to `vague/` take effect immediately.

## Run tests

```bash
pytest                        # full suite (14 tests, ~40s)
pytest tests/test_belief.py   # core only (~25s)
```

## Key files

| File | Responsibility |
|------|---------------|
| `vague/belief.py` | GaussianBelief — fit, update, merge, query |
| `vague/embedder.py` | Embedder — sentence-transformers, MPS on Apple Silicon |
| `vague/memory.py` | BeliefMemory — high-level API, lazy init, save/load |
| `vague/agent.py` | BeliefStateAgent — recall + LLM prompt assembly |
| `vague/adapters/langgraph.py` | LangGraph node factory |
| `benchmarks/longbench.py` | LongBench eval pipeline |
| `benchmarks/needle.py` | Needle-in-a-haystack sweep |

## Quick smoke test

```python
from vague import BeliefMemory

mem = BeliefMemory(n_components=8)
mem.remember_batch(["Paris is the capital of France.", "The Eiffel Tower was built in 1889."] * 5)
print(mem.recall("Tell me about Paris", k=2))
print(mem.stats())
```

## Benchmark

```bash
cd notebooks
jupyter notebook benchmark_analysis.ipynb
# Cell 1: setup + llm_fn (Anthropic key from .env)
# Cell 2: LongBench qasper
# Cell 3: Needle sweep
```

## Known constraints

- numpy must be `<2.0` (torch 2.2 on Intel Mac incompatibility)
- sentence-transformers must be `<3.0` (requires torch >= 2.4 otherwise)
- torch 2.4+ has no wheel for macOS x86_64 — stuck on 2.2 on Intel Mac
- On Apple Silicon: MLX acceleration available via `pip install "vague[mlx]"`

## Current benchmark results (qasper, n=50, claude-3-haiku)

| Method | F1 | Tokens | Compression |
|--------|---:|-------:|------------:|
| full_context | 0.113 | 3748 | 1.0x |
| naive_rag | 0.124 | 1706 | 2.2x |
| vague | 0.121 | 1721 | 2.9x |
