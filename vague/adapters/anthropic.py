"""Anthropic adapter for vague — Claude API inference.

Factory function that returns an ``llm_fn``-compatible callable backed by
the Anthropic API.

Usage::

    from vague.adapters.anthropic import anthropic_fn

    llm = anthropic_fn(model="claude-haiku-4-5-20251001", max_tokens=256)
    answer = llm("What is the capital of France?")
"""

from __future__ import annotations

from typing import Callable


def anthropic_fn(
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 256,
    api_key: str | None = None,
) -> Callable[[str], str]:
    """Return an ``llm_fn``-compatible callable backed by Claude API.

    Args:
        model: Anthropic model ID. Defaults to Claude Haiku (fastest/cheapest).
        max_tokens: Maximum tokens to generate.
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Returns:
        A callable ``(prompt: str) -> str``.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def llm_fn(prompt: str) -> str:
        r = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    return llm_fn
