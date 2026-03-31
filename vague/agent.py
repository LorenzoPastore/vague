"""BeliefStateAgent: LLM agent with Gaussian belief-backed memory."""

from __future__ import annotations

import logging
from typing import Callable

from vague.memory import BeliefMemory

logger = logging.getLogger(__name__)

try:
    import tiktoken as _tiktoken
    _ENCODER = _tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None


def _count_tokens(text: str) -> int:
    """Approximate token count using tiktoken if available, else len//4."""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text) // 4


class BeliefStateAgent:
    """An LLM agent that uses BeliefMemory for context retrieval."""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        memory: BeliefMemory | None = None,
        system_prompt: str = "",
        recall_k: int = 5,
    ) -> None:
        self.llm_fn = llm_fn
        self.memory = memory if memory is not None else BeliefMemory()
        self.system_prompt = system_prompt
        self.recall_k = recall_k
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def observe(self, text: str) -> None:
        """Add text to agent memory."""
        self.memory.remember(text)

    def act(self, task: str) -> str:
        """Recall context, build prompt, call llm_fn, return response."""
        recalled = self.memory.recall(task, k=self.recall_k)

        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if recalled:
            parts.append("--- Context ---")
            parts.extend(recalled)
            parts.append("--- End Context ---")
        parts.append(task)
        prompt = "\n".join(parts)

        input_tokens = _count_tokens(prompt)
        self._total_input_tokens += input_tokens

        response = self.llm_fn(prompt)

        output_tokens = _count_tokens(response)
        self._total_output_tokens += output_tokens

        logger.debug(
            "act: input_tokens=%d output_tokens=%d", input_tokens, output_tokens
        )
        return response

    def token_usage(self) -> dict:
        """Return cumulative token usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
