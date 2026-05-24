# Vague

**Probabilistic belief-state memory for LLM agents.**

[![CI](https://img.shields.io/github/actions/workflow/status/lorenzopastore/vague/ci.yml?branch=main&label=CI)](https://github.com/lorenzopastore/vague/actions)
[![PyPI](https://img.shields.io/pypi/v/vague)](https://pypi.org/project/vague/)
[![Python](https://img.shields.io/pypi/pyversions/vague)](https://pypi.org/project/vague/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

Most agent memory systems store a list of text chunks and retrieve by cosine similarity. Vague stores a **Gaussian Mixture Model** over the embedding space instead. The result is a continuous, probabilistic representation that compresses naturally, updates incrementally, and — uniquely — **merges analytically across agents**.

```
Agent A observes docs → belief A (GMM)
Agent B observes docs → belief B (GMM)

belief_a.share_with(belief_b)   # closed-form merge, no re-indexing
belief_b.recall("Paris?")       # returns from both corpora
```

---

## Install

```bash
pip install vague
```

Optional extras:

```bash
pip install "vague[langgraph]"   # LangGraph adapter
pip install "vague[benchmark]"   # Evaluation tools (datasets, pandas, tiktoken)
pip install "vague[dev]"         # Testing + linting
```

---

## Quick start

```python
from vague import BeliefMemory

memory = BeliefMemory(n_components=32)
memory.remember_batch([
    "Paris is the capital of France.",
    "The Eiffel Tower was built in 1889.",
    "France has a population of 68 million.",
])

results = memory.recall("Tell me about Paris", k=3)
print(results)
# ['Paris is the capital of France.', 'The Eiffel Tower was built in 1889.', ...]

print(memory.stats())
# {'n_components': 32, 'n_observations': 3, 'compression_ratio': ..., 'entropy': ...}
```

---

## With a real LLM

```python
from vague import BeliefStateAgent, BeliefMemory
import anthropic

client = anthropic.Anthropic()

def llm_fn(prompt: str) -> str:
    r = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text

agent = BeliefStateAgent(llm_fn=llm_fn, memory=BeliefMemory(n_components=32))

for doc in my_documents:
    agent.observe(doc)

answer = agent.act("What are the key findings?")
print(agent.token_usage())
# {'total_input_tokens': 1842, 'total_output_tokens': 312, 'cache_hits': 0}
```

---

## Multi-agent belief sharing

Agents can merge their knowledge without exchanging raw text. The merge is a weighted combination of Gaussian components — O(K²) in the number of components, not in the number of documents.

```python
from vague import BeliefStateAgent, BeliefMemory

agent_science = BeliefStateAgent(llm_fn, memory=BeliefMemory())
agent_history = BeliefStateAgent(llm_fn, memory=BeliefMemory())

for doc in science_docs:
    agent_science.observe(doc)

for doc in history_docs:
    agent_history.observe(doc)

# Share science knowledge into history agent
agent_science.memory.share_with(agent_history.memory)

# history agent can now answer from both corpora
agent_history.act("How did scientific advances affect 19th century history?")
```

---

## LangGraph integration

```python
from vague import BeliefMemory
from vague.adapters.langgraph import gaussian_memory_node
from langgraph.graph import StateGraph

memory = BeliefMemory(n_components=32)
memory.remember_batch(documents)

recall_node = gaussian_memory_node(
    memory,
    input_key="input",
    context_key="context",
    k=5,
    also_remember=True,   # also stores new inputs into the belief
)

graph = StateGraph(dict)
graph.add_node("recall", recall_node)
graph.add_node("generate", your_llm_node)
graph.add_edge("recall", "generate")
```

---

## Benchmark

Evaluated on [LongBench](https://github.com/THUDM/LongBench) — 3 tasks, n=50 samples per (task, method), single run, `mlx-community/Qwen3-8B-4bit` (local, Apple Silicon via MLX).

| Task | Method | F1 | Avg tokens | Compression |
|---|---|---:|---:|---:|
| hotpotqa | Full context | 0.036 | 4120 | 3.1x |
| hotpotqa | Naive RAG | 0.035 | 1764 | 7.4x |
| hotpotqa | **Vague (GMM)** | 0.035 | 1734 | 7.6x |
| hotpotqa | **SummaryBelief** | 0.059 | 333 | 38.8x |
| multifieldqa_en | Full context | 0.163 | 3768 | 1.8x |
| multifieldqa_en | Naive RAG | 0.160 | 2232 | 3.6x |
| multifieldqa_en | **Vague (GMM)** | 0.148 | 2229 | 3.7x |
| multifieldqa_en | **SummaryBelief** | 0.242 | 321 | 22.3x |
| qasper | Full context | 0.122 | 3748 | 1.3x |
| qasper | Naive RAG | 0.130 | 1706 | 2.9x |
| qasper | **Vague (GMM)** | 0.124 | 1729 | 2.8x |
| qasper | **SummaryBelief** | 0.168 | 331 | 14.9x |

Source: `benchmarks/summary_mlx_reduced.json`.

**SummaryBelief is the key finding.** GaussianBelief alone matches Naive RAG within F1 noise (its value is the principled probabilistic interface — merge, update, transfer — not a retrieval-quality win). SummaryBelief, by storing one LLM-generated summary per Gaussian component instead of raw chunks, beats every baseline on every task while compressing the injected context by **15–40×**.

![Multi-task benchmark](docs/multitask_benchmark.png)

The cost is asymmetric: SummaryBelief makes K extra LLM calls during the fit phase (once), but every subsequent query injects far fewer tokens — favorable any time the corpus is queried more than ~K times.

### Needle-in-a-haystack

Recall rate of a planted fact at varying context lengths and positions:

![Needle heatmap](docs/needle_heatmap.png)

Vague retrieves reliably up to ~2k tokens. At 4k+ the GMM begins to saturate — increasing `n_components` recovers recall at the cost of more parameters.


---

## How it works

A `BeliefMemory` wraps a `GaussianBelief`, which maintains:

- **Means** μₖ — the centroid of each knowledge cluster in embedding space
- **Covariances** Σₖ — the spread (uncertainty) of each cluster
- **Weights** πₖ — the relative importance of each component

On `remember(text)`, the new embedding updates the nearest component via exponential moving average. On `recall(query)`, the query's posterior under the mixture weights each stored embedding — texts near high-probability components for that query surface first.

On `share_with(other)`, components from both mixtures are concatenated, weighted by `alpha`, then pruned where KL divergence between components is below a threshold. No text crosses the boundary.

```
GaussianBelief
├── fit(texts)            → initial GMM via sklearn EM
├── update(text)          → online EMA update on nearest component
├── merge(other)          → alpha-weighted mixture + KL pruning
├── query(q, top_k)       → GMM-weighted cosine scoring
└── to_dict() / from_dict → JSON serialization
```

---

## API reference

### `BeliefMemory`

| Method | Description |
|---|---|
| `remember(text)` | Add a single observation |
| `remember_batch(texts)` | Add multiple observations |
| `recall(query, k=5)` | Return top-k relevant strings |
| `share_with(other)` | Merge self's belief into other (in-place) |
| `stats()` | Returns `n_components`, `n_observations`, `compression_ratio`, `entropy` |
| `save(path)` | Serialize to JSON |
| `BeliefMemory.load(path)` | Deserialize from JSON |

### `BeliefStateAgent`

| Method | Description |
|---|---|
| `observe(text)` | Store text into memory |
| `act(task)` | Recall context → build prompt → call `llm_fn` → return response |
| `token_usage()` | Returns cumulative `total_input_tokens`, `total_output_tokens`, `cache_hits` |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
