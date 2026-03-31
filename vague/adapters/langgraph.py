"""LangGraph adapter for vague BeliefMemory and BeliefStateAgent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from vague.memory import BeliefMemory
from vague.agent import BeliefStateAgent

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from langgraph.graph import StateGraph as StateGraphType


def _require_langgraph() -> None:
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is not installed. Install it with: pip install langgraph"
        )


def gaussian_memory_node(
    memory: BeliefMemory,
    input_key: str = "input",
    context_key: str = "context",
    k: int = 5,
    also_remember: bool = False,
) -> Callable[[dict], dict]:
    """Return a LangGraph node function that recalls from memory and injects context.

    Args:
        memory: The BeliefMemory instance to query.
        input_key: State key containing the input text.
        context_key: State key where recalled context will be written.
        k: Number of chunks to recall.
        also_remember: If True, also call memory.remember on the input.

    Returns:
        A pure function ``state -> state`` compatible with LangGraph nodes.
    """
    def node(state: dict) -> dict:
        text = state.get(input_key, "")
        if also_remember and text:
            memory.remember(text)
        recalled = memory.recall(text, k=k) if text else []
        return {**state, context_key: recalled}

    return node


def make_belief_graph(
    agents: list[BeliefStateAgent],
    share_beliefs: bool = True,
) -> "StateGraphType":
    """Factory for a multi-agent StateGraph where each agent is a node.

    Agents are connected sequentially. If share_beliefs=True, a belief-sharing
    step is inserted between each adjacent pair of agents.

    Args:
        agents: List of BeliefStateAgent instances.
        share_beliefs: Whether to wire in belief-sharing edges between agents.

    Returns:
        A configured (but not compiled) LangGraph StateGraph.

    Raises:
        ImportError: If langgraph is not installed.
        ValueError: If agents list is empty.
    """
    _require_langgraph()

    if not agents:
        raise ValueError("agents list must not be empty.")

    graph = StateGraph(dict)

    # Register one node per agent.
    # agent_0 reads from "input"; subsequent agents read from "output"
    # (the previous agent's response).
    for i, agent in enumerate(agents):
        name = f"agent_{i}"
        input_key = "input" if i == 0 else "output"

        def make_agent_node(a: BeliefStateAgent, ik: str) -> Callable[[dict], dict]:
            def agent_node(state: dict) -> dict:
                task = state.get(ik, "")
                response = a.act(task)
                return {**state, "output": response}
            return agent_node

        graph.add_node(name, make_agent_node(agent, input_key))

    # Sequential edges with optional belief-sharing steps
    for i in range(len(agents) - 1):
        src = f"agent_{i}"
        dst = f"agent_{i + 1}"

        if share_beliefs:
            share_name = f"share_{i}_{i + 1}"

            def make_share_node(
                a: BeliefStateAgent, b: BeliefStateAgent
            ) -> Callable[[dict], dict]:
                def share_node(state: dict) -> dict:
                    try:
                        a.memory.share_with(b.memory)
                    except ValueError:
                        # One or both memories not yet fitted — skip
                        pass
                    return state
                return share_node

            graph.add_node(share_name, make_share_node(agents[i], agents[i + 1]))
            graph.add_edge(src, share_name)
            graph.add_edge(share_name, dst)
        else:
            graph.add_edge(src, dst)

    graph.set_entry_point("agent_0")
    graph.set_finish_point(f"agent_{len(agents) - 1}")

    return graph
