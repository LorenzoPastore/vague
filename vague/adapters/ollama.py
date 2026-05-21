"""Ollama adapter for vague — local inference via Ollama API.

Usage::

    from vague.adapters.ollama import ollama_fn

    llm = ollama_fn(model="llama3.2:3b")
    answer = llm("What is the capital of France?")
"""

from __future__ import annotations

from typing import Callable


def ollama_fn(
    model: str = "qwen3:8b",
    max_tokens: int = 256,
    base_url: str = "http://localhost:11434",
) -> Callable[[str], str]:
    """Return an ``llm_fn``-compatible callable backed by Ollama.

    Args:
        model: Ollama model name (must be already pulled).
        max_tokens: Maximum tokens to generate.
        base_url: Ollama server URL.

    Returns:
        A callable ``(prompt: str) -> str``.
    """
    import requests

    def llm_fn(prompt: str) -> str:
        r = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt,
                  "stream": False, "options": {"num_predict": max_tokens}},
            timeout=120,
        )
        r.raise_for_status()
        response = r.json()["response"]
        # Strip <think>...</think> blocks (Qwen3 thinking mode)
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response

    return llm_fn
