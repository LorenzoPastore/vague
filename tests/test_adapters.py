"""Tests for vague.adapters.langgraph.

LangGraph is optional. A minimal mock is injected via conftest.py so these
tests run even without the real package installed.
"""

from __future__ import annotations

import pytest

from vague.memory import BeliefMemory
from vague.agent import BeliefStateAgent
from vague.adapters.langgraph import gaussian_memory_node, make_belief_graph


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ML_TEXTS = [
    "Neural networks learn representations from data.",
    "Gradient descent minimizes the loss function.",
    "Backpropagation computes gradients efficiently.",
    "Convolutional networks excel at image recognition.",
    "Recurrent networks process sequential data.",
    "Attention mechanisms power transformer models.",
    "Transfer learning reuses pretrained weights.",
    "Regularization prevents overfitting in models.",
    "Batch normalization stabilizes training dynamics.",
    "Dropout randomly deactivates neurons during training.",
]


def _fitted_memory() -> BeliefMemory:
    mem = BeliefMemory(n_components=4)
    mem.remember_batch(_ML_TEXTS)
    return mem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memory_node_injects_context():
    """gaussian_memory_node adds a 'context' key to state."""
    mem = _fitted_memory()
    node = gaussian_memory_node(mem, input_key="input", context_key="context", k=3)

    state = {"input": "gradient descent optimization"}
    result = node(state)

    assert "context" in result
    assert isinstance(result["context"], list)
    assert len(result["context"]) <= 3
    # Original keys preserved
    assert result["input"] == state["input"]


def test_memory_node_observe():
    """gaussian_memory_node with also_remember=True calls memory.remember."""
    mem = _fitted_memory()
    original_n = mem._n_observations

    node = gaussian_memory_node(
        mem, input_key="input", context_key="context", k=3, also_remember=True
    )
    state = {"input": "new observation text about machine learning"}
    node(state)

    assert mem._n_observations == original_n + 1


def test_make_belief_graph_structure():
    """make_belief_graph returns a graph with the correct node and edge structure."""
    dummy_llm = lambda p: "response"
    agents = [
        BeliefStateAgent(dummy_llm, memory=_fitted_memory()),
        BeliefStateAgent(dummy_llm, memory=_fitted_memory()),
    ]

    graph = make_belief_graph(agents, share_beliefs=False)

    assert "agent_0" in graph.nodes
    assert "agent_1" in graph.nodes
