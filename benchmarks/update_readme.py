"""Regenerate the README ``## Benchmark`` section from a summary JSON.

Usage::

    # use the default summary_mlx_reduced.json
    python benchmarks/update_readme.py

    # or point at any other summary file
    python benchmarks/update_readme.py --summary benchmarks/summary_anthropic_reduced.json

The script replaces everything between ``START_MARKER`` and ``END_MARKER`` in
README.md with a freshly generated table, then prints a diff-friendly summary.
Idempotent — re-running produces the same output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"

# Methods listed in display order (best baseline first, novel last for emphasis).
METHOD_ORDER = ["full_context", "naive_rag", "vague", "summary_belief"]
METHOD_LABELS = {
    "full_context":   "Full context",
    "naive_rag":      "Naive RAG",
    "vague":          "**Vague (GMM)**",
    "summary_belief": "**SummaryBelief**",
}

# Friendly per-summary-file metadata (model + #runs). Falls back to "<unknown>"
# when not found — keep these in sync as new benchmark configs land.
FILE_META: dict[str, dict] = {
    "summary_mlx_reduced.json": {
        "model": "`mlx-community/Qwen3-8B-4bit` (local, Apple Silicon via MLX)",
        "n_samples": 50,
        "n_runs": 1,
    },
    "summary_anthropic_reduced.json": {
        "model": "`claude-haiku-4-5` (Anthropic API)",
        "n_samples": 50,
        "n_runs": 1,
    },
    "summary_groq_reduced.json": {
        "model": "`llama-3.3-70b-versatile` (Groq LPU)",
        "n_samples": 50,
        "n_runs": 1,
    },
}

START_MARKER = "## Benchmark"
END_MARKER = "---\n\n## How it works"


def _has_ci(s: dict) -> bool:
    """True iff the bootstrap CI is non-degenerate (i.e. n_runs > 1)."""
    lo, hi = s.get("ci_95", [0.0, 0.0])
    return abs(hi - lo) > 1e-9


def build_table(summary: dict, meta: dict, summary_filename: str) -> str:
    n_samples = meta.get("n_samples", "?")
    n_runs = meta.get("n_runs", "?")
    model = meta.get("model", "<unknown>")

    # Pick any task to probe whether CIs are meaningful.
    any_task = next(iter(summary.values()))
    any_method = next(iter(any_task.values()))
    show_ci = _has_ci(any_method)

    header_runs = (
        f"{n_runs} independent runs"
        if isinstance(n_runs, int) and n_runs > 1
        else "single run"
    )

    lines = [
        "## Benchmark",
        "",
        f"Evaluated on [LongBench](https://github.com/THUDM/LongBench) "
        f"— 3 tasks, n={n_samples} samples per (task, method), "
        f"{header_runs}, {model}.",
        "",
    ]

    if show_ci:
        lines += [
            "| Task | Method | F1 mean | ±std | 95% CI | Avg tokens | Compression |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| Task | Method | F1 | Avg tokens | Compression |",
            "|---|---|---:|---:|---:|",
        ]

    for task in summary:
        for method in METHOD_ORDER:
            if method not in summary[task]:
                continue
            s = summary[task][method]
            label = METHOD_LABELS.get(method, method)
            if show_ci:
                ci_lo, ci_hi = s["ci_95"]
                lines.append(
                    f"| {task} | {label} | {s['f1_mean']:.3f} | "
                    f"{s['f1_std']:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | "
                    f"{s['avg_tokens']:.0f} | {s['compression']:.1f}x |"
                )
            else:
                lines.append(
                    f"| {task} | {label} | {s['f1_mean']:.3f} | "
                    f"{s['avg_tokens']:.0f} | {s['compression']:.1f}x |"
                )

    lines += [
        "",
        f"Source: `benchmarks/{summary_filename}`.",
        "",
        "**SummaryBelief is the key finding.** GaussianBelief alone matches Naive "
        "RAG within F1 noise (its value is the principled probabilistic interface "
        "— merge, update, transfer — not a retrieval-quality win). "
        "SummaryBelief, by storing one LLM-generated summary per Gaussian "
        "component instead of raw chunks, beats every baseline on every task "
        "while compressing the injected context by **15–40×**.",
        "",
        "![Multi-task benchmark](docs/multitask_benchmark.png)",
        "",
        "The cost is asymmetric: SummaryBelief makes K extra LLM calls during "
        "the fit phase (once), but every subsequent query injects far fewer "
        "tokens — favorable any time the corpus is queried more than ~K times.",
        "",
        "### Needle-in-a-haystack",
        "",
        "Recall rate of a planted fact at varying context lengths and positions:",
        "",
        "![Needle heatmap](docs/needle_heatmap.png)",
        "",
        "Vague retrieves reliably up to ~2k tokens. At 4k+ the GMM begins to "
        "saturate — increasing `n_components` recovers recall at the cost of "
        "more parameters.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--summary",
        default="benchmarks/summary_mlx_reduced.json",
        help="Path to summary JSON (relative to repo root).",
    )
    args = p.parse_args()

    summary_path = (REPO_ROOT / args.summary).resolve()
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    meta = FILE_META.get(summary_path.name, {})

    table = build_table(summary, meta, summary_path.name)

    with open(README_PATH) as f:
        content = f.read()
    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        print("[update_readme] START/END markers not found in README.md.")
        print("[update_readme] New table follows — paste it manually if needed:\n")
        print(table)
        return

    new_content = content[:start_idx] + table + "\n" + content[end_idx:]
    with open(README_PATH, "w") as f:
        f.write(new_content)
    print(f"[update_readme] README.md updated from {summary_path.name}.")


if __name__ == "__main__":
    main()
