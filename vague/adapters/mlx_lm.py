"""MLX-LM adapter for vague — Apple Silicon GPU inference.

Factory function that returns an ``llm_fn``-compatible callable backed by
``mlx_lm`` (requires ``mlx-lm`` installed in the current environment;
Apple Silicon only).

Usage::

    from vague.adapters.mlx_lm import mlx_lm_fn

    llm = mlx_lm_fn("mlx-community/Qwen3-8B-4bit", max_tokens=256)
    answer = llm("What is the capital of France?")
"""

from __future__ import annotations

from typing import Callable


def mlx_lm_fn(
    model_path: str,
    max_tokens: int = 256,
) -> Callable[[str], str]:
    """Return an ``llm_fn``-compatible callable backed by mlx_lm.

    The model is loaded once on first call and cached for subsequent calls
    (lazy loading to avoid import overhead when the adapter is imported but
    not used).

    Args:
        model_path: HuggingFace repo id or local path to an MLX model.
            Aliases (``qwen3-8b``, ``qwen3-30b``, ``qwen-coder``,
            ``deepseek-r1``) are resolved to their canonical HF repos.
        max_tokens: Maximum number of tokens to generate per call.

    Returns:
        A callable ``(prompt: str) -> str`` compatible with
        ``BeliefStateAgent(llm_fn=...)``.

    Raises:
        ImportError: If ``mlx_lm`` is not installed.
    """
    _ALIASES: dict[str, str] = {
        "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
        "qwen3-30b": "mlx-community/Qwen3-30B-A3B-4bit",
        "qwen-coder": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "qwen-coder-32b": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "deepseek-r1": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "deepseek": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    }
    resolved = _ALIASES.get(model_path, model_path)

    # Lazy-loaded model state — captured in closure
    _state: dict = {}

    def _load() -> None:
        try:
            from mlx_lm import load, generate as _generate  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mlx_lm is not installed. Install it with: pip install mlx-lm "
                "(Apple Silicon only)."
            ) from e
        _state["generate"] = _generate
        _state["model"], _state["tokenizer"] = load(resolved)

    _is_qwen3 = "Qwen3" in resolved

    def _apply_template(tokenizer, prompt: str) -> str:
        """Apply chat template with thinking disabled for Qwen3."""
        messages = [{"role": "user", "content": prompt}]
        try:
            # enable_thinking=False disables <think> blocks (Qwen3 only)
            kwargs = {"enable_thinking": False} if _is_qwen3 else {}
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        except Exception:
            return prompt

    def llm_fn(prompt: str) -> str:
        if not _state:
            _load()
        p = _apply_template(_state["tokenizer"], prompt)
        return _state["generate"](
            _state["model"],
            _state["tokenizer"],
            prompt=p,
            max_tokens=max_tokens,
            verbose=False,
        )

    return llm_fn
