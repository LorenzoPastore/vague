"""
Conftest: pre-stub unimplemented modules so that importing vague.belief
(which triggers vague/__init__.py) does not fail while other agents'
modules are still TODO stubs.

Fast test mode: set VAGUE_FAST_TESTS=1 (or pass --fast to pytest) to replace
the sentence-transformer Embedder with a deterministic random-vector mock.
This cuts the test suite from ~40s to ~2s. The GMM logic is fully exercised;
only embedding quality is not tested (which is the Embedder's own concern).


Also injects a minimal LangGraph mock when langgraph is not installed so
that the adapter tests can run without the optional dependency.
"""

import importlib
import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=False,
                     help="Mock Embedder with random vectors (skips model load)")


def _make_mock_embedder(dim: int = 16):
    """Return an Embedder-compatible object using deterministic per-text vectors.

    Each text always maps to the same vector (hash-seeded RNG), so roundtrip
    serialization tests pass. Uses float64 and dim=16 for numerical stability.
    """
    def _vec(text: str) -> np.ndarray:
        seed = hash(text) & 0xFFFF_FFFF
        return np.random.default_rng(seed).standard_normal(dim)

    class _MockEmbedder:
        def embed(self, texts):
            return np.stack([_vec(t) for t in texts])

        def embed_single(self, text):
            return _vec(text)

    return _MockEmbedder()


_FAST_DIM = 16


@pytest.fixture(autouse=True, scope="session")
def maybe_mock_embedder(request):
    fast = request.config.getoption("--fast") or os.environ.get("VAGUE_FAST_TESTS") == "1"
    if fast:
        from vague.belief import GaussianBelief

        mock = _make_mock_embedder(_FAST_DIM)

        # Patch Embedder class so every GaussianBelief() gets the mock
        original_init = GaussianBelief.__init__

        def _fast_init(self, n_components=32, embedding_dim=_FAST_DIM, backend="sklearn"):
            original_init(self, n_components=n_components,
                          embedding_dim=embedding_dim, backend=backend)
            self._embedder = mock  # override after super().__init__

        with patch("vague.belief.GaussianBelief.__init__", _fast_init):
            yield
    else:
        yield


def pytest_collection_modifyitems(config, items):
    """Skip semantic tests in fast mode (random vectors have no semantic structure)."""
    fast = config.getoption("--fast", default=False) or os.environ.get("VAGUE_FAST_TESTS") == "1"
    if fast:
        skip = pytest.mark.skip(reason="semantic / numerical-precision test skipped in --fast mode")
        for item in items:
            # MLX parity test relies on real-embedding magnitudes; random
            # vectors in fast mode amplify fp32-vs-fp64 drift past tolerance.
            if "relevance" in item.name or "semantic" in item.name or "mlx" in item.name:
                item.add_marker(skip)


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub vague.memory if it has not been implemented yet
try:
    from vague.memory import BeliefMemory  # noqa: F401
except ImportError:
    _stub_module("vague.memory", BeliefMemory=object)

# Stub vague.agent if it has not been implemented yet
try:
    from vague.agent import BeliefStateAgent  # noqa: F401
except ImportError:
    _stub_module("vague.agent", BeliefStateAgent=object)

# Inject a minimal langgraph mock if not installed so adapter tests work
try:
    import langgraph  # noqa: F401
except ImportError:
    lg_mod = types.ModuleType("langgraph")
    lg_graph_mod = types.ModuleType("langgraph.graph")

    class _MockStateGraph:
        def __init__(self, schema):
            self._nodes = {}
            self._edges = []
            self._entry = None
            self._finish = None

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, src, dst):
            self._edges.append((src, dst))

        def set_entry_point(self, name):
            self._entry = name

        def set_finish_point(self, name):
            self._finish = name

    lg_graph_mod.StateGraph = _MockStateGraph
    lg_mod.graph = lg_graph_mod
    sys.modules["langgraph"] = lg_mod
    sys.modules["langgraph.graph"] = lg_graph_mod

    # Reload the adapter so it picks up the mock
    if "vague.adapters.langgraph" in sys.modules:
        importlib.reload(sys.modules["vague.adapters.langgraph"])
