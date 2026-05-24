"""OpenAI-compatible adapter for vague — works with any provider that exposes
the OpenAI chat-completions API shape: Groq, Cerebras, Together, Fireworks,
OpenRouter, DeepInfra, vLLM (local), Ollama (in OpenAI-compat mode), etc.

A single factory function plus thin convenience wrappers that preset
``base_url`` and the environment variable that holds the API key.

Usage::

    # Generic — any provider you have a base_url + key for.
    from vague.adapters.openai_compat import openai_compatible_fn
    llm = openai_compatible_fn(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    # Or use the convenience wrappers (auto-reads <PROVIDER>_API_KEY env var).
    from vague.adapters.openai_compat import groq_fn, cerebras_fn, together_fn

    llm = groq_fn("llama-3.3-70b-versatile")       # 280 tok/s, ~$0.59/MTok in
    llm = cerebras_fn("llama-3.3-70b")             # ~2200 tok/s, free tier
    llm = together_fn("Qwen/Qwen2.5-72B-Instruct-Turbo")

Provider notes (as of 2026-05):
    Groq      — Custom LPU silicon. Free tier with rate limit + ~$5 credit on
                signup. https://console.groq.com/keys
    Cerebras  — Wafer-scale CS-3. Free tier with rate limit (10 RPM, 1M TPD on
                Llama 3.3 70B). https://cloud.cerebras.ai
    Together  — Standard managed inference, wide model selection.
                https://api.together.xyz/settings/api-keys
    Fireworks — Similar to Together. https://fireworks.ai/api-keys
    OpenRouter — Aggregator over many providers, pick cheapest per model.
                 https://openrouter.ai/keys
"""

from __future__ import annotations

import os
import re
from typing import Callable


def openai_compatible_fn(
    model: str,
    base_url: str,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    strip_thinking: bool = True,
    extra_headers: dict | None = None,
    max_retries: int = 10,
    timeout: float = 120.0,
) -> Callable[[str], str]:
    """Return an ``llm_fn``-compatible callable backed by an OpenAI-compatible API.

    Args:
        model: Model identifier as accepted by the provider.
        base_url: Endpoint URL (e.g. ``"https://api.groq.com/openai/v1"``).
        api_key: API key. If None, the caller is responsible for setting
            ``OPENAI_API_KEY`` in the environment (or pass via env to the
            convenience wrappers below).
        max_tokens: Max tokens to generate per call.
        temperature: Sampling temperature. Default ``0.0`` for reproducibility.
        strip_thinking: If True, strip ``<think>...</think>`` blocks from the
            response (Qwen3 / DeepSeek-R1 style reasoning models).
        extra_headers: Optional dict of headers to pass through (e.g. OpenRouter
            requires ``HTTP-Referer`` and ``X-Title`` for analytics).

    Returns:
        A callable ``(prompt: str) -> str`` compatible with ``llm_fn``.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
    _think_re = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

    def llm_fn(prompt: str) -> str:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        r = client.chat.completions.create(**kwargs)
        text = r.choices[0].message.content or ""
        if strip_thinking:
            text = _think_re.sub("", text).strip()
        return text

    return llm_fn


# ---------------------------------------------------------------------------
# Convenience wrappers — preset base_url + env var for popular providers
# ---------------------------------------------------------------------------


def _env_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise EnvironmentError(
            f"{name} environment variable is not set. "
            f"Get a key from the provider's console (see vague.adapters.openai_compat "
            f"docstring for sign-up URLs) and `export {name}=...`."
        )
    return key


def groq_fn(
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Groq LPU inference. ~200-700 tok/s. Free tier on https://console.groq.com.

    Popular models:
        - ``llama-3.3-70b-versatile``      (~280 tok/s, $0.59/$0.79 per MTok)
        - ``llama-3.1-8b-instant``         (~750 tok/s, $0.05/$0.08 per MTok)
        - ``qwen/qwen3-32b``               (~400 tok/s)
        - ``deepseek-r1-distill-llama-70b``
    """
    return openai_compatible_fn(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=_env_key("GROQ_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def cerebras_fn(
    model: str = "llama-3.3-70b",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Cerebras wafer-scale inference. ~2200 tok/s on Llama 3.3 70B. Free tier
    on https://cloud.cerebras.ai (rate limit: 10 RPM, 1M tokens/day).

    Popular models:
        - ``llama-3.3-70b``                 (the headliner)
        - ``llama3.1-8b``
        - ``qwen-3-32b``
    """
    return openai_compatible_fn(
        model=model,
        base_url="https://api.cerebras.ai/v1",
        api_key=_env_key("CEREBRAS_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def together_fn(
    model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Together AI managed inference. https://api.together.xyz."""
    return openai_compatible_fn(
        model=model,
        base_url="https://api.together.xyz/v1",
        api_key=_env_key("TOGETHER_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def fireworks_fn(
    model: str = "accounts/fireworks/models/qwen2p5-72b-instruct",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Fireworks managed inference. https://fireworks.ai."""
    return openai_compatible_fn(
        model=model,
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=_env_key("FIREWORKS_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def openrouter_fn(
    model: str = "qwen/qwen-2.5-72b-instruct",
    max_tokens: int = 256,
    temperature: float = 0.0,
    referer: str = "https://github.com/lorenzopastore/vague",
    title: str = "vague-benchmark",
) -> Callable[[str], str]:
    """OpenRouter aggregator. Auto-routes to the cheapest provider per model.
    https://openrouter.ai. Referer/title are recommended for analytics."""
    return openai_compatible_fn(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=_env_key("OPENROUTER_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={"HTTP-Referer": referer, "X-Title": title},
    )
