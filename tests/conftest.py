"""
Conftest: pre-stub unimplemented modules so that importing vague.belief
(which triggers vague/__init__.py) does not fail while other agents'
modules are still TODO stubs.

Also injects a minimal LangGraph mock when langgraph is not installed so
that the adapter tests can run without the optional dependency.
"""

import importlib
import sys
import types


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
