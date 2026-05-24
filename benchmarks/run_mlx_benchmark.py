"""Provider-agnostic benchmark runner for LongBench.

Backends supported via ``--provider``:
    mlx        — local Apple Silicon (default; ``pip install -e .[mlx]``)
    groq       — Groq LPU cloud (needs GROQ_API_KEY)
    cerebras   — Cerebras wafer-scale (needs CEREBRAS_API_KEY)
    anthropic  — Claude API (needs ANTHROPIC_API_KEY)
    together   — Together AI (needs TOGETHER_API_KEY)
    fireworks  — Fireworks (needs FIREWORKS_API_KEY)
    openrouter — OpenRouter aggregator (needs OPENROUTER_API_KEY)
    ollama     — local Ollama server (no key needed)

Default config: 3 runs × 3 tasks × n=200, all 4 methods (vague, naive_rag,
full_context, summary_belief). Default model is provider-specific (see
PROVIDER_DEFAULT_MODELS in source).

Usage::

    # full default run on local MLX
    python benchmarks/run_mlx_benchmark.py

    # smoke test on Groq Llama 3.3 70B (cloud, ~280 tok/s)
    python benchmarks/run_mlx_benchmark.py --provider groq \\
        --tasks qasper --n-samples 10 --n-runs 1 --tag smoke

    # full benchmark on Cerebras Qwen-3 235B (~2200 tok/s)
    python benchmarks/run_mlx_benchmark.py --provider cerebras \\
        --n-samples 100 --n-runs 3

    # exclude experimental summary_belief
    python benchmarks/run_mlx_benchmark.py \\
        --methods vague naive_rag full_context

Outputs:
    benchmarks/results_<provider>[_<tag>].json  — raw per-run results
    benchmarks/summary_<provider>[_<tag>].json  — aggregated per (task, method)
    Prints summary table with mean±std and 95% bootstrap CI to stdout.
    Checkpoint-resumable: re-runs of the same command skip cells already done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(scores: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.array(scores)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_TASKS = ["qasper", "hotpotqa", "multifieldqa_en"]
DEFAULT_METHODS = ["vague", "naive_rag", "full_context", "summary_belief"]

# Default model per provider — sensible flagship choices.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "mlx":       "mlx-community/Qwen3-8B-4bit",
    "groq":      "llama-3.3-70b-versatile",
    "cerebras":  "qwen-3-235b-a22b-instruct-2507",
    "anthropic": "claude-haiku-4-5-20251001",
    "together":  "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/qwen2p5-72b-instruct",
    "openrouter":"qwen/qwen-2.5-72b-instruct",
    "ollama":    "qwen3:8b",
}


def _build_llm_fn(provider: str, model: str, max_tokens: int):
    """Resolve a llm_fn for the requested provider."""
    from vague.adapters import (
        anthropic_fn, cerebras_fn, fireworks_fn, groq_fn,
        mlx_lm_fn, ollama_fn, openrouter_fn, together_fn,
    )
    factories = {
        "mlx":        lambda: mlx_lm_fn(model, max_tokens=max_tokens),
        "groq":       lambda: groq_fn(model, max_tokens=max_tokens),
        "cerebras":   lambda: cerebras_fn(model, max_tokens=max_tokens),
        "anthropic":  lambda: anthropic_fn(model=model, max_tokens=max_tokens),
        "together":   lambda: together_fn(model, max_tokens=max_tokens),
        "fireworks":  lambda: fireworks_fn(model, max_tokens=max_tokens),
        "openrouter": lambda: openrouter_fn(model, max_tokens=max_tokens),
        "ollama":     lambda: ollama_fn(model, max_tokens=max_tokens),
    }
    if provider not in factories:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: {sorted(factories)}"
        )
    return factories[provider]()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--provider", default="mlx",
                   choices=sorted(PROVIDER_DEFAULT_MODELS.keys()),
                   help="LLM backend. Default 'mlx' (local Apple Silicon).")
    p.add_argument("--model", default=None,
                   help="Model identifier. Defaults to a sensible per-provider "
                        "choice (see PROVIDER_DEFAULT_MODELS in source).")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                   help="LongBench tasks to evaluate.")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   choices=["vague", "naive_rag", "full_context", "summary_belief"],
                   help="Methods to evaluate.")
    p.add_argument("--n-samples", type=int, default=200,
                   help="Samples per (task, method) per run.")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Independent runs (for std + bootstrap CI).")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="LLM max output tokens.")
    p.add_argument("--tag", default="",
                   help="Optional suffix for output filenames (e.g. 'smoke').")
    args = p.parse_args()
    if args.model is None:
        args.model = PROVIDER_DEFAULT_MODELS[args.provider]
    return args


def main() -> None:
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from benchmarks.longbench import LongBenchEval

    args = _parse_args()
    PROVIDER = args.provider
    MODEL = args.model
    TASKS = args.tasks
    METHODS = args.methods
    N_SAMPLES = args.n_samples
    N_RUNS = args.n_runs
    CACHE_DIR = str(repo_root / ".cache")
    # Output filename: results_<provider>[_<tag>].json — provider always
    # included so different backends don't overwrite each other. Tag is
    # appended only if explicitly supplied.
    # NOTE: this preserves the legacy MLX naming (results_mlx_<tag>.json)
    # so existing checkpoints from prior runs remain reusable.
    parts = [PROVIDER]
    if args.tag:
        parts.append(args.tag)
    suffix = "_" + "_".join(parts)

    print(f"[benchmark] Provider: {PROVIDER}")
    print(f"[benchmark] Model: {MODEL}")
    print(f"[benchmark] Tasks: {TASKS}")
    print(f"[benchmark] Methods: {METHODS}")
    print(f"[benchmark] N_SAMPLES={N_SAMPLES}, N_RUNS={N_RUNS}")
    print()

    # Resume support: load any existing checkpoint and skip completed cells.
    out_path = repo_root / "benchmarks" / f"results{suffix}.json"
    if out_path.exists():
        with open(out_path) as f:
            raw_results: dict[str, dict[str, list[dict]]] = json.load(f)
        # Ensure schema covers all current (task, method) keys.
        for t in TASKS:
            raw_results.setdefault(t, {})
            for m in METHODS:
                raw_results[t].setdefault(m, [])
        completed = sum(
            len(raw_results[t][m]) for t in TASKS for m in METHODS
        )
        print(f"[benchmark] Resuming from checkpoint → {out_path} "
              f"({completed} cells already done)\n")
    else:
        raw_results = {t: {m: [] for m in METHODS} for t in TASKS}

    def _save_checkpoint() -> None:
        with open(out_path, "w") as f:
            json.dump(raw_results, f, indent=2)

    # Build llm_fn once — provider-specific factory; MLX/Ollama load lazily.
    llm = _build_llm_fn(PROVIDER, MODEL, args.max_tokens)

    for run_idx in range(N_RUNS):
        print(f"=== Run {run_idx + 1}/{N_RUNS} ===")
        eval_ = LongBenchEval(
            llm_fn=llm,
            cache_dir=CACHE_DIR,
            model_id=f"{PROVIDER}-{MODEL}",
        )
        for task in TASKS:
            for method in METHODS:
                if len(raw_results[task][method]) > run_idx:
                    prev = raw_results[task][method][run_idx]
                    print(
                        f"  {task:20s} | {method:12s} | "
                        f"F1={prev['f1']:.4f} | tokens={prev['avg_tokens']:.0f}"
                        f" | compression={prev['compression']:.1f}x | (cached)"
                    )
                    continue
                t0 = time.perf_counter()
                result = eval_.run(task, method, n_samples=N_SAMPLES)
                elapsed = time.perf_counter() - t0
                raw_results[task][method].append({
                    "f1": result.f1_score,
                    "avg_tokens": result.avg_input_tokens,
                    "compression": result.compression_ratio,
                    "latency_ms": result.latency_ms,
                    "n_samples": result.n_samples,
                    "wall_s": elapsed,
                })
                _save_checkpoint()
                print(
                    f"  {task:20s} | {method:12s} | "
                    f"F1={result.f1_score:.4f} | "
                    f"tokens={result.avg_input_tokens} | "
                    f"compression={result.compression_ratio:.1f}x | "
                    f"wall={elapsed:.0f}s"
                )
        print()

    print(f"[benchmark] Raw results saved → {out_path}")
    print()

    # Summary table
    print("=" * 90)
    print(f"{'Task':20s} | {'Method':12s} | {'F1 mean':>8s} | {'±std':>6s} | {'95% CI':>16s} | {'Tokens':>7s} | {'Compr':>6s}")
    print("-" * 90)

    summary: dict[str, dict[str, dict]] = {}

    # Aggregate over every (task, method) present in the checkpoint, not just
    # the subset run in this invocation — otherwise running a single method on
    # a single task would wipe the previously computed summary for everything
    # else. The checkpoint is the source of truth.
    for task in sorted(raw_results.keys()):
        summary[task] = {}
        for method in sorted(raw_results[task].keys()):
            runs = raw_results[task][method]
            if not runs:
                continue
            f1_vals = [r["f1"] for r in runs]
            tok_vals = [r["avg_tokens"] for r in runs]
            cr_vals = [r["compression"] for r in runs]

            f1_mean = float(np.mean(f1_vals))
            f1_std = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0
            # Bootstrap CI is only meaningful when n_runs > 1.
            if len(f1_vals) > 1:
                ci_lo, ci_hi = bootstrap_ci(f1_vals)
            else:
                ci_lo, ci_hi = f1_mean, f1_mean
            tok_mean = float(np.mean(tok_vals))
            cr_mean = float(np.mean(cr_vals))

            summary[task][method] = {
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "ci_95": [ci_lo, ci_hi],
                "avg_tokens": tok_mean,
                "compression": cr_mean,
                "n_runs": len(f1_vals),
            }

            print(
                f"{task:20s} | {method:12s} | "
                f"{f1_mean:>8.4f} | "
                f"{f1_std:>6.4f} | "
                f"[{ci_lo:.4f}, {ci_hi:.4f}] | "
                f"{tok_mean:>7.0f} | "
                f"{cr_mean:>5.1f}x"
            )

    print("=" * 90)

    # Save summary
    summary_path = repo_root / "benchmarks" / f"summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[benchmark] Summary saved → {summary_path}")

    # Return summary for README update
    return summary


if __name__ == "__main__":
    main()
