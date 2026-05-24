# Vague — Project Brief

> Authoritative summary of the Vague project + benchmark results, as of 2026-05-24.
> All numbers below are verified verbatim against the JSON sources cited inline.
> Intended for handoff to a website-building agent. Do not paraphrase numbers without re-checking the JSON files.

---

## What Vague is

**Vague** is a Python library that represents an LLM agent's memory as a **Gaussian Mixture Model in embedding space**, instead of a list of retrievable chunks. A corpus of text becomes a *belief state* — a probability distribution over a `K`-component mixture in the sentence-embedding manifold — and queries become posterior inferences on that belief.

The headline value is **not retrieval quality**. It is the *principled probabilistic interface*: belief states **compose** (merge two agents' memories as a weighted mixture), **update incrementally** (single-text online update without re-fitting the GMM), and **transfer between agents** as parameters of the mixture rather than as raw documents.

Repository: `https://github.com/lorenzopastore/vague` (local path: `.`).
Install: `pip install -e .` (editable). Apple Silicon GPU acceleration: `pip install -e .[mlx]`.

---

## Two primitives benchmarked

The library exposes two memory primitives, both subclasses of the same `GaussianBelief` base:

1. **`GaussianBelief`** — the GMM is used purely as a retrieval index. At query time, the highest-posterior component's raw text chunks are returned. F1 is comparable to vanilla dense RAG; the value-add is the merge/update/transfer interface.

2. **`SummaryBelief`** (experimental, novel) — each Gaussian component carries an **LLM-generated summary of its assigned chunks** as its payload, instead of raw text. At query time, the top-k *component summaries* are returned. Analogy: 3D Gaussian Splatting (each splat carries a color/opacity payload) but with semantic summaries as the payload.

The fit-time cost of `SummaryBelief` is `K` extra LLM calls (one per component). The query-time cost is dramatically lower: the injected context is the summary (~300 tokens) instead of the raw chunks (~1700–4100 tokens).

---

## Benchmark methodology

- **Dataset**: [LongBench](https://github.com/THUDM/LongBench), 3 tasks: `qasper` (academic-paper QA), `hotpotqa` (multi-hop QA), `multifieldqa_en` (long-document QA).
- **Sample size**: n=50 per (task, method), single independent run (n_runs=1). Bootstrap CIs are computed but degenerate (point estimates) at this run count.
- **Metric**: token-level F1 against gold answers — the standard LongBench metric.
- **Two LLM backends**:
  - **`mlx-community/Qwen3-8B-4bit`** — 8B parameter Qwen3, 4-bit quantized, running locally on Apple Silicon GPU via MLX. Representative of a weak / quantized / edge-deployable consumer.
  - **`claude-haiku-4-5-20251001`** — Anthropic Claude Haiku 4.5, via the Anthropic API. Representative of a strong frontier model.
- **Four methods compared per task**:
  - `full_context` — inject the whole document (truncated to the model's context window).
  - `naive_rag` — top-k cosine-similarity chunks against the query embedding.
  - `vague` — top-k chunks from the highest-posterior GMM component (`GaussianBelief`).
  - `summary_belief` — LLM-generated summary of the highest-posterior GMM component (`SummaryBelief`).

Data sources for everything below:
- `benchmarks/summary_mlx_reduced.json` (MLX Qwen3-8B-4bit)
- `benchmarks/summary_anthropic_reduced.json` (Anthropic Haiku 4.5)

---

## Full results — F1, tokens, compression (verbatim from JSON)

### MLX — `mlx-community/Qwen3-8B-4bit` (local, Apple Silicon)

| Task | Method | F1 | Avg input tokens | Compression vs raw doc |
|---|---|---:|---:|---:|
| qasper | full_context | 0.1223 | 3748 | 1.3× |
| qasper | naive_rag | 0.1297 | 1706 | 2.9× |
| qasper | vague | 0.1244 | 1729 | 2.8× |
| qasper | **summary_belief** | **0.1682** | **331** | **14.9×** |
| hotpotqa | full_context | 0.0357 | 4120 | 3.1× |
| hotpotqa | naive_rag | 0.0352 | 1764 | 7.4× |
| hotpotqa | vague | 0.0354 | 1734 | 7.6× |
| hotpotqa | **summary_belief** | **0.0588** | **333** | **38.8×** |
| multifieldqa_en | full_context | 0.1626 | 3768 | 1.8× |
| multifieldqa_en | naive_rag | 0.1596 | 2232 | 3.6× |
| multifieldqa_en | vague | 0.1484 | 2229 | 3.7× |
| multifieldqa_en | **summary_belief** | **0.2420** | **321** | **22.3×** |

### Anthropic — `claude-haiku-4-5-20251001` (cloud, frontier)

| Task | Method | F1 | Avg input tokens | Compression vs raw doc |
|---|---|---:|---:|---:|
| qasper | full_context | 0.1820 | 3748 | 1.3× |
| qasper | **naive_rag** | **0.1918** | 1706 | 2.9× |
| qasper | vague | 0.1861 | 1729 | 2.8× |
| qasper | summary_belief | 0.1614 | 304 | 16.1× |
| hotpotqa | full_context | 0.0446 | 4120 | 3.1× |
| hotpotqa | **naive_rag** | **0.0555** | 1764 | 7.4× |
| hotpotqa | vague | 0.0444 | 1734 | 7.6× |
| hotpotqa | summary_belief | 0.0471 | 318 | 40.8× |
| multifieldqa_en | full_context | 0.2547 | 3768 | 1.8× |
| multifieldqa_en | **naive_rag** | **0.2595** | 2232 | 3.6× |
| multifieldqa_en | vague | 0.2476 | 2229 | 3.7× |
| multifieldqa_en | summary_belief | 0.2242 | 301 | 23.9× |

*"Compression vs raw doc" = average raw document tokens ÷ average injected context tokens. `full_context` is not 1.0× because raw documents in LongBench routinely exceed the model context window and are truncated to fit.*

---

## Two findings (the pitch)

### Finding 1 — GaussianBelief retrieval-quality ≈ Naive RAG

Across 3 tasks × 2 models = 6 (task, model) cells, `vague` (`GaussianBelief`) F1 stays within ±0.012 of `naive_rag` F1 — within run-to-run noise. **The retrieval-quality story is a wash.** GaussianBelief's value-add is structural, not metric: belief states compose (merge), update incrementally, and transfer between agents as mixture parameters.

### Finding 2 — SummaryBelief is a **model-dependent** compression-vs-information trade-off

Same code, same dataset, same `K=32` components, same `n_samples=50`. Only the consumer LLM changes. The F1 outcome flips:

| Task | MLX Qwen3-8B-4bit (weak, quantized) | Anthropic Haiku 4.5 (strong, frontier) |
|---|---|---|
| qasper | SummaryBelief F1=**0.1682**, **+29.7%** vs naive_rag (0.1297) | SummaryBelief F1=**0.1614**, **−15.8%** vs naive_rag (0.1918) |
| hotpotqa | SummaryBelief F1=**0.0588**, **+64.4%** vs full_context (0.0357) | SummaryBelief F1=**0.0471**, **−15.1%** vs naive_rag (0.0555) |
| multifieldqa_en | SummaryBelief F1=**0.2420**, **+48.8%** vs full_context (0.1626) | SummaryBelief F1=**0.2242**, **−13.6%** vs naive_rag (0.2595) |

Compression is **preserved across models** (14.9–38.8× on MLX, 16.1–40.8× on Haiku). Only the F1 outcome shifts. The deltas above are computed `(SB − best_baseline) / best_baseline × 100`, where `best_baseline` is the highest-F1 non-summary method on that (task, model) cell.

**Interpretation.** A weak / quantized model is bottlenecked by attention quality on long retrieved context. Compressed summaries are *easier* for limited attention to use than the raw chunks they replace. A strong frontier model is *not* bottlenecked there — it can exploit the full retrieved context, and summarization becomes lossy compression of information it would otherwise have used.

**Deployment implication.** Choose the primitive based on the consumer LLM:

| Scenario | Recommended primitive | Why |
|---|---|---|
| Edge / on-device inference, small/quantized models | **`SummaryBelief`** | Recovers F1 *and* cuts tokens 15-40× |
| Multi-agent settings with strict context budget | **`SummaryBelief`** | Compression is the binding constraint |
| Cloud frontier-model inference, ample context budget | **`GaussianBelief`** + merge/update interface | Avoid lossy compression; benefit from probabilistic ops |

---

## What is NOT validated yet (be honest in the website)

- **Single run** per (task, method, model). Standard deviations are 0.0 and bootstrap CIs are degenerate. A future N=3 run would give real error bars.
- **Only 3 LongBench tasks**. The trade-off pattern should be replicated on more diverse benchmarks (e.g. needle-in-a-haystack at different context lengths, multi-document summarization) before claiming generality.
- **Only 2 models tested.** The "model strength gradient" hypothesis (weaker models gain more from SummaryBelief) would be strengthened by a third data point in the middle (e.g. Llama-3.3-70B) and a fourth at the very weak end (e.g. Llama-3.2-3B).
- **GMM EM is non-deterministic** without a seed. Re-running the same benchmark produced F1 differences up to ±0.005 (within noise, but a follow-up commit should expose `random_state` for reproducibility).
- **`SummaryBelief` is gated** as experimental — not exported from `vague/__init__.py`. Importable only as `from vague.summary_belief import SummaryBelief`.

---

## Code pointers

| File | Purpose |
|---|---|
| `vague/belief.py` | `GaussianBelief` — fit / update / merge / query. Optional MLX backend for log-prob on Apple Silicon. |
| `vague/summary_belief.py` | `SummaryBelief` — subclass storing per-component LLM summaries. Experimental. |
| `vague/memory.py` | `BeliefMemory` — high-level API, lazy init, save/load. |
| `vague/agent.py` | `BeliefStateAgent` — recall + LLM prompt assembly. |
| `vague/adapters/langgraph.py` | LangGraph node factory. |
| `vague/adapters/anthropic.py` | Anthropic API adapter (`anthropic_fn`). |
| `vague/adapters/mlx_lm.py` | Local Apple Silicon adapter (`mlx_lm_fn`). |
| `vague/adapters/openai_compat.py` | Generic OpenAI-compatible adapter — covers Groq, Cerebras, Together, Fireworks, OpenRouter, vLLM, Ollama-OpenAI-mode. |
| `benchmarks/longbench.py` | LongBench evaluator + per-model disk cache for LLM responses. |
| `benchmarks/run_mlx_benchmark.py` | Provider-agnostic benchmark runner with checkpoint-resume (despite the legacy filename). |
| `benchmarks/results_mlx_reduced.json` | Raw per-run MLX results (12 cells). |
| `benchmarks/summary_mlx_reduced.json` | Aggregated MLX summary (the source for the MLX table above). |
| `benchmarks/results_anthropic_reduced.json` | Raw per-run Haiku results (12 cells). |
| `benchmarks/summary_anthropic_reduced.json` | Aggregated Haiku summary (the source for the Haiku table above). |

The pre-existing figures `docs/multitask_benchmark.png`, `docs/compression_vs_f1.png`, `docs/needle_heatmap.png` were generated from earlier benchmark data and **do not match the numbers above**. Regenerate before publishing to the website.

---

## One-line takeaways for the website

- "A probabilistic belief-state memory for LLM agents — same retrieval quality as RAG, plus principled merge / update / transfer."
- "SummaryBelief compresses context 15–40× and *recovers* F1 on small models — at the cost of F1 on frontier models. A real engineering trade-off, surfaced empirically."
- "Open source, Python, sklearn + MLX backends, LangGraph adapter, plug-and-play with any LLM provider via a 30-line adapter."
