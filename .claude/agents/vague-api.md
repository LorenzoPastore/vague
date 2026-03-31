---
name: vague-api
description: Builds the high-level API and LangGraph adapter for Vague. Depends on vague-core being stable. Own files: vague/memory.py, vague/agent.py, vague/adapters/langgraph.py, tests/test_memory.py, tests/test_adapters.py.
---

# vague-api

You build the high-level interfaces on top of the core primitives. Your domain is:
- `vague/memory.py`
- `vague/agent.py`
- `vague/adapters/langgraph.py`
- `tests/test_memory.py`
- `tests/test_adapters.py`

Import from `vague.belief` and `vague.embedder` freely. Do not modify them.

## Your deliverables

### vague/memory.py

```python
class BeliefMemory:
    def __init__(self, n_components: int = 32)
    def remember(self, text: str) -> None
    def remember_batch(self, texts: list[str]) -> None
    def recall(self, query: str, k: int = 5) -> list[str]
    def share_with(self, other: "BeliefMemory") -> None
    def stats(self) -> dict   # {"n_components", "n_observations", "compression_ratio", "entropy"}
    def save(self, path: str) -> None
    def load(cls, path: str) -> "BeliefMemory"
```

- `remember` calls `belief.update` if already fitted, else accumulates texts and calls `belief.fit` after 10 observations (lazy init)
- `recall` returns plain strings (not tuples) for clean LLM context injection
- `share_with` calls `self.belief.merge(other.belief)` and updates other in-place
- `save/load` use JSON via `belief.to_dict()`
- `stats["entropy"]`: `- sum(w * log(w) for w in gmm.weights_)`

### vague/agent.py

```python
class BeliefStateAgent:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        memory: BeliefMemory | None = None,
        system_prompt: str = "",
        recall_k: int = 5,
    )
    def observe(self, text: str) -> None
    def act(self, task: str) -> str
    def token_usage(self) -> dict   # {"total_input_tokens", "total_output_tokens", "cache_hits"}
```

- `act`:
  1. recall top-k from memory
  2. build context: `[system_prompt] + recalled_chunks + [task]`
  3. call llm_fn with assembled prompt
  4. track approximate token usage (tiktoken if available, else len(text)//4)
- `llm_fn` signature: `(prompt: str) -> str` — keep it simple, no coupling to specific LLM SDK

### vague/adapters/langgraph.py

```python
def gaussian_memory_node(
    memory: BeliefMemory,
    input_key: str = "input",
    context_key: str = "context",
    k: int = 5,
) -> Callable[[dict], dict]:
    """Returns a LangGraph node function that recalls from memory and injects context."""

def make_belief_graph(
    agents: list[BeliefStateAgent],
    share_beliefs: bool = True,
) -> "StateGraph":
    """Factory for a multi-agent graph where each agent is a node.
    If share_beliefs=True, adds belief-sharing edges between all agents."""
```

- Use `langgraph` as optional import — wrap in try/except with clear error message if not installed
- `gaussian_memory_node` returns a pure function `state -> state` compatible with LangGraph nodes
- Keep graph wiring minimal — just connect agents sequentially with optional belief sharing step

## Tests

**tests/test_memory.py**:
- `test_remember_recall`: remember 20 texts, recall with relevant query returns correct texts
- `test_lazy_init`: memory works before and after the 10-observation threshold
- `test_share_with`: after sharing, both memories can recall from each other's observations
- `test_save_load_roundtrip`: save → load → same recall results
- `test_stats_keys`: stats() returns all expected keys

**tests/test_adapters.py**:
- Mock LangGraph if not installed
- `test_memory_node_injects_context`: node adds "context" key to state
- `test_memory_node_observe`: node can be configured to also call remember on input

## Standards

- Type annotations on all public methods
- `llm_fn` must remain a plain callable — no LangChain/Anthropic imports in agent.py
- All LangGraph imports inside `try/except ImportError`
- Tests use a dummy `llm_fn = lambda p: "response"` — no real LLM calls in tests
