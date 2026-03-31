"""Needle-in-a-haystack evaluation for BeliefMemory retrieval."""

from __future__ import annotations

import random
import time
from typing import Callable

import pandas as pd
from tqdm import tqdm

from vague import BeliefMemory

# ---------------------------------------------------------------------------
# Token counting helper (matches agent.py)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Filler text corpus
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Scientists have discovered a new species of deep-sea fish near the Mariana Trench.",
    "The annual rainfall in the Amazon basin is approximately 2,300 millimeters.",
    "Philosophy of mind explores the relationship between mental states and the physical brain.",
    "Quantum computing leverages superposition to process information exponentially faster.",
    "The Renaissance period saw a flourishing of art, architecture, and scientific thought.",
    "Ocean currents play a critical role in regulating global climate patterns.",
    "Machine learning models require large, high-quality datasets to generalize effectively.",
    "The mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells.",
    "Linguistics studies the structure, evolution, and diversity of human language.",
    "Historical records show that ancient Mesopotamia developed one of the earliest writing systems.",
    "Renewable energy sources, such as solar and wind, are rapidly displacing fossil fuels.",
    "The Great Wall of China stretches over 21,000 kilometers across northern China.",
    "Behavioral economics examines how psychological factors influence economic decisions.",
    "A supernova occurs when a massive star exhausts its nuclear fuel and collapses.",
    "Modern cryptography relies on computationally hard mathematical problems for security.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Urban planning aims to create livable, sustainable, and equitable cities.",
    "The stock market reflects collective expectations about corporate earnings and macro conditions.",
    "Evolutionary biology explains how natural selection drives adaptation over generations.",
]


def _generate_filler(n_tokens: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    words: list[str] = []
    while _count_tokens(" ".join(words)) < n_tokens:
        words.extend(rng.choice(_FILLER_SENTENCES).split())
    # trim to exact token budget
    text = " ".join(words)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        return enc.decode(tokens[:n_tokens])
    except ImportError:
        return text[: n_tokens * 4]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

_DEFAULT_NEEDLE = "The secret code is ZEPHYR-9472."
_NEEDLE_QUESTION = "What is the special fact in the text?"


def build_haystack(n_tokens: int, needle: str, position: float, seed: int = 0) -> str:
    """Insert needle at relative position (0.0=start, 1.0=end) in filler text."""
    if not 0.0 <= position <= 1.0:
        raise ValueError("position must be in [0.0, 1.0]")

    # Allocate most tokens for filler; needle is small
    needle_tokens = _count_tokens(needle)
    filler_tokens = max(n_tokens - needle_tokens, 1)
    filler = _generate_filler(filler_tokens, seed=seed)

    filler_words = filler.split()
    insert_idx = int(len(filler_words) * position)
    needle_words = needle.split()

    combined = filler_words[:insert_idx] + needle_words + filler_words[insert_idx:]
    return " ".join(combined)


def run_needle(
    context_length: int,
    needle_position: float,
    n_components: int = 32,
    llm_fn: Callable[[str], str] | None = None,
    n_trials: int = 5,
) -> dict:
    """
    Returns {"found_rate": float, "avg_tokens_used": int,
             "context_length": int, "position": float}
    """
    needle = _DEFAULT_NEEDLE

    if llm_fn is None:
        # Mock: always returns the needle (structural testing)
        llm_fn = lambda _: needle  # noqa: E731

    found_count = 0
    tokens_used: list[int] = []

    for trial in range(n_trials):
        haystack = build_haystack(
            n_tokens=context_length,
            needle=needle,
            position=needle_position,
            seed=trial,
        )

        # Load into BeliefMemory
        mem = BeliefMemory(n_components=n_components)
        # chunk at 256-token boundaries
        words = haystack.split()
        chunk_size = 256  # words (approximate tokens)
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        mem.remember_batch(chunks)

        recalled = mem.recall(_NEEDLE_QUESTION, k=5)
        context = "\n".join(recalled)
        prompt = f"{context}\n\nQuestion: {_NEEDLE_QUESTION}\nAnswer:"

        n_tok = _count_tokens(prompt)
        tokens_used.append(n_tok)

        answer = llm_fn(prompt)
        if needle.lower() in answer.lower():
            found_count += 1

    return {
        "found_rate": found_count / n_trials,
        "avg_tokens_used": int(sum(tokens_used) / n_trials),
        "context_length": context_length,
        "position": needle_position,
    }


def run_needle_sweep(
    context_lengths: list[int],
    positions: list[float],
    llm_fn: Callable | None = None,
) -> pd.DataFrame:
    """
    Full sweep for heatmap.
    Returns DataFrame with columns: context_length, position, found_rate, tokens_used.
    """
    rows = []
    combos = [(cl, pos) for cl in context_lengths for pos in positions]

    for context_length, position in tqdm(combos, desc="needle sweep", unit="cell"):
        result = run_needle(
            context_length=context_length,
            needle_position=position,
            llm_fn=llm_fn,
        )
        rows.append(
            {
                "context_length": result["context_length"],
                "position": result["position"],
                "found_rate": result["found_rate"],
                "tokens_used": result["avg_tokens_used"],
            }
        )

    return pd.DataFrame(rows)
