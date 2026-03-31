"""LongBench evaluation pipeline for vague vs. naive_rag vs. full_context."""

from __future__ import annotations

import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm

from vague import BeliefMemory

# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])
    except ImportError:
        # fallback: character-level approximation (4 chars/token)
        return text[: max_tokens * 4]


# ---------------------------------------------------------------------------
# F1 score (token-level, standard LongBench metric)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_f1(prediction: str, answers: list[str]) -> float:
    """LongBench uses max F1 across all gold answers."""
    return max(_token_f1(prediction, a) for a in answers) if answers else 0.0


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_tokens: int = 256) -> list[str]:
    """Split text into roughly chunk_tokens-sized chunks by words."""
    words = text.split()
    # Approx 0.75 words per token on average — use word count as proxy
    words_per_chunk = chunk_tokens  # rough 1 word ~ 1 token (conservative)
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i : i + words_per_chunk]))
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Naive RAG retriever
# ---------------------------------------------------------------------------

class _NaiveRAG:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, context: str, query: str, k: int = 5) -> list[str]:
        import numpy as np

        chunks = _chunk_text(context, chunk_tokens=256)
        if len(chunks) <= k:
            return chunks

        chunk_embs = self._model.encode(chunks, show_progress_bar=False)
        query_emb = self._model.encode([query], show_progress_bar=False)[0]

        # Cosine similarity
        norms_c = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10
        norm_q = np.linalg.norm(query_emb) + 1e-10
        sims = (chunk_embs / norms_c) @ (query_emb / norm_q)
        top_idx = np.argsort(sims)[::-1][:k]
        return [chunks[i] for i in sorted(top_idx)]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

_TASK_FILES = {
    "qasper":          "data/qasper.jsonl",
    "multifieldqa_en": "data/multifieldqa_en.jsonl",
    "hotpotqa":        "data/hotpotqa.jsonl",
}


def _extract_longbench_zip(cache_dir: str) -> str:
    """Download data.zip from THUDM/LongBench and extract to cache_dir. Returns extract dir."""
    import zipfile
    from huggingface_hub import hf_hub_download

    extract_dir = os.path.join(cache_dir, "longbench")
    if os.path.isdir(extract_dir) and any(
        f.endswith(".jsonl") for f in os.listdir(os.path.join(extract_dir, "data")) if os.path.isdir(os.path.join(extract_dir, "data"))
    ):
        return extract_dir  # already extracted

    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _load_dataset(task: str, cache_dir: str, n_samples: int, seed: int = 42):
    import json

    if task not in _TASK_FILES:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(_TASK_FILES)}")

    extract_dir = _extract_longbench_zip(cache_dir)
    jsonl_path = os.path.join(extract_dir, _TASK_FILES[task])

    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[: min(n_samples, len(samples))]


def _get_context_and_answers(task: str, sample: dict) -> tuple[str, list[str]]:
    """Extract (context, list_of_gold_answers) from a raw sample."""
    context = sample.get("context", sample.get("input", ""))
    answers_raw = sample.get("answers", sample.get("answer", []))
    if isinstance(answers_raw, str):
        answers = [answers_raw]
    elif isinstance(answers_raw, list):
        answers = [a for a in answers_raw if a]
    else:
        answers = []
    return context, answers


def _get_question(sample: dict) -> str:
    return sample.get("input", sample.get("question", ""))


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    task: str
    method: str  # "vague" | "naive_rag" | "full_context"
    f1_score: float
    avg_input_tokens: int
    compression_ratio: float  # context_tokens / prompt_tokens; higher = more compression
    latency_ms: float
    n_samples: int


# ---------------------------------------------------------------------------
# LongBenchEval
# ---------------------------------------------------------------------------

class LongBenchEval:
    def __init__(self, llm_fn: Callable[[str], str], cache_dir: str = ".cache") -> None:
        self.llm_fn = llm_fn
        self.cache_dir = cache_dir
        self._naive_rag: _NaiveRAG | None = None

    # ------------------------------------------------------------------
    # Internal retrieval methods
    # ------------------------------------------------------------------

    def _run_vague(
        self,
        context: str,
        question: str,
        n_components: int,
        top_k: int = 5,
    ) -> tuple[str, int, float]:
        """Return (prompt, n_input_tokens, compression_ratio)."""
        mem = BeliefMemory(n_components=n_components)
        chunks = _chunk_text(context, chunk_tokens=256)
        mem.remember_batch(chunks)

        context_tokens = _count_tokens(context)
        recalled = mem.recall(question, k=top_k)
        prompt = "\n".join(recalled) + f"\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = _count_tokens(prompt)
        # compression_ratio = how many times smaller the prompt is vs full context
        cr = context_tokens / prompt_tokens if prompt_tokens > 0 else 1.0
        return prompt, prompt_tokens, cr

    def _run_naive_rag(
        self,
        context: str,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, int]:
        if self._naive_rag is None:
            self._naive_rag = _NaiveRAG()
        retrieved = self._naive_rag.retrieve(context, question, k=top_k)
        prompt = "\n".join(retrieved) + f"\n\nQuestion: {question}\nAnswer:"
        return prompt, _count_tokens(prompt)

    def _run_full_context(
        self,
        context: str,
        question: str,
        max_context_tokens: int,
    ) -> tuple[str, int]:
        truncated = _truncate_to_tokens(context, max_context_tokens)
        prompt = truncated + f"\n\nQuestion: {question}\nAnswer:"
        return prompt, _count_tokens(prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        method: str,
        n_components: int = 32,
        n_samples: int = 200,
        max_context_tokens: int = 4096,
    ) -> EvalResult:
        if method not in ("vague", "naive_rag", "full_context"):
            raise ValueError(f"Unknown method '{method}'")

        samples = _load_dataset(task, self.cache_dir, n_samples)

        f1_scores: list[float] = []
        token_counts: list[int] = []
        compression_ratios: list[float] = []
        latencies: list[float] = []

        for sample in tqdm(samples, desc=f"{task}/{method}", unit="sample"):
            context, answers = _get_context_and_answers(task, sample)
            question = _get_question(sample)

            if not context or not answers:
                continue

            t0 = time.perf_counter()

            if method == "vague":
                prompt, n_tok, cr = self._run_vague(context, question, n_components)
            elif method == "naive_rag":
                prompt, n_tok = self._run_naive_rag(context, question)
                cr = 1.0
            else:  # full_context
                prompt, n_tok = self._run_full_context(context, question, max_context_tokens)
                cr = 1.0

            prediction = self.llm_fn(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000

            f1 = _best_f1(prediction, answers)
            f1_scores.append(f1)
            token_counts.append(n_tok)
            compression_ratios.append(cr)
            latencies.append(latency_ms)

        n = len(f1_scores)
        return EvalResult(
            task=task,
            method=method,
            f1_score=sum(f1_scores) / n if n else 0.0,
            avg_input_tokens=int(sum(token_counts) / n) if n else 0,
            compression_ratio=sum(compression_ratios) / n if n else 1.0,
            latency_ms=sum(latencies) / n if n else 0.0,
            n_samples=n,
        )

    def compare_all(
        self,
        task: str,
        n_samples: int = 200,
    ) -> list[EvalResult]:
        results = []
        for method in ("vague", "naive_rag", "full_context"):
            results.append(self.run(task, method, n_samples=n_samples))
        return results
