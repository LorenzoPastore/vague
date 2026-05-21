"""Benchmark runner: 3 runs × 3 tasks × n=200, mlx_lm Qwen3-8B-4bit.

Apple Silicon only. Install the optional extra: ``pip install -e .[mlx]``.

Usage (from repo root):
    python benchmarks/run_mlx_benchmark.py

Outputs:
    benchmarks/results_mlx.json  — raw per-run results
    benchmarks/summary_mlx.json  — aggregated per (task, method)
    Prints summary table with mean±std and 95% bootstrap CI to stdout.

Note: METHODS deliberately excludes the experimental ``summary_belief``
to keep the established-methods comparison reproducible at this cost.
Run a separate pass to evaluate SummaryBelief, e.g. by calling
``LongBenchEval.compare_all`` directly (its DEFAULT_METHODS includes it).
"""

from __future__ import annotations

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

def main() -> None:
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from vague.adapters.mlx_lm import mlx_lm_fn
    from benchmarks.longbench import LongBenchEval

    MODEL = "mlx-community/Qwen3-8B-4bit"
    TASKS = ["qasper", "hotpotqa", "multifieldqa_en"]
    METHODS = ["vague", "naive_rag", "full_context"]
    N_SAMPLES = 200
    N_RUNS = 3
    CACHE_DIR = str(repo_root / ".cache")

    print(f"[benchmark] Model: {MODEL}")
    print(f"[benchmark] Tasks: {TASKS}")
    print(f"[benchmark] N_SAMPLES={N_SAMPLES}, N_RUNS={N_RUNS}")
    print()

    # Build llm_fn once — model loads lazily on first call
    llm = mlx_lm_fn(MODEL, max_tokens=64)

    # raw_results[task][method] = list of (f1, avg_tokens, compression) per run
    raw_results: dict[str, dict[str, list[dict]]] = {
        t: {m: [] for m in METHODS} for t in TASKS
    }

    for run_idx in range(N_RUNS):
        print(f"=== Run {run_idx + 1}/{N_RUNS} ===")
        eval_ = LongBenchEval(llm_fn=llm, cache_dir=CACHE_DIR)
        for task in TASKS:
            for method in METHODS:
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
                print(
                    f"  {task:20s} | {method:12s} | "
                    f"F1={result.f1_score:.4f} | "
                    f"tokens={result.avg_input_tokens} | "
                    f"compression={result.compression_ratio:.1f}x | "
                    f"wall={elapsed:.0f}s"
                )
        print()

    # Save raw results
    out_path = repo_root / "benchmarks" / "results_mlx.json"
    with open(out_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"[benchmark] Raw results saved → {out_path}")
    print()

    # Summary table
    print("=" * 90)
    print(f"{'Task':20s} | {'Method':12s} | {'F1 mean':>8s} | {'±std':>6s} | {'95% CI':>16s} | {'Tokens':>7s} | {'Compr':>6s}")
    print("-" * 90)

    summary: dict[str, dict[str, dict]] = {}

    for task in TASKS:
        summary[task] = {}
        for method in METHODS:
            runs = raw_results[task][method]
            f1_vals = [r["f1"] for r in runs]
            tok_vals = [r["avg_tokens"] for r in runs]
            cr_vals = [r["compression"] for r in runs]

            f1_mean = float(np.mean(f1_vals))
            f1_std = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(f1_vals * N_RUNS)  # replicate for bootstrap stability
            tok_mean = float(np.mean(tok_vals))
            cr_mean = float(np.mean(cr_vals))

            summary[task][method] = {
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "ci_95": [ci_lo, ci_hi],
                "avg_tokens": tok_mean,
                "compression": cr_mean,
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
    summary_path = repo_root / "benchmarks" / "summary_mlx.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[benchmark] Summary saved → {summary_path}")

    # Return summary for README update
    return summary


if __name__ == "__main__":
    main()
