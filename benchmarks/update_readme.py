"""Update README.md benchmark table from benchmarks/summary_mlx.json.

Usage:
    python benchmarks/update_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SUMMARY_PATH = REPO_ROOT / "benchmarks" / "summary_mlx.json"
README_PATH = REPO_ROOT / "README.md"

TASKS = ["qasper", "hotpotqa", "multifieldqa_en"]
METHODS = ["full_context", "naive_rag", "vague"]
METHOD_LABELS = {
    "full_context": "Full context",
    "naive_rag": "Naive RAG",
    "vague": "**Vague**",
}

START_MARKER = "## Benchmark"
END_MARKER = "---\n\n## How it works"

def build_table(summary: dict) -> str:
    lines = [
        "## Benchmark",
        "",
        "Evaluated on [LongBench](https://github.com/THUDM/LongBench) — 3 tasks, n=200 samples each,"
        " 3 independent runs, `mlx-community/Qwen3-8B-4bit` (Apple Silicon GPU via MLX).",
        "",
        "| Task | Method | F1 mean | ±std | 95% CI | Avg tokens | Compression |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for method in METHODS:
            s = summary[task][method]
            label = METHOD_LABELS[method]
            ci_lo, ci_hi = s["ci_95"]
            lines.append(
                f"| {task} | {label} | {s['f1_mean']:.3f} | {s['f1_std']:.3f}"
                f" | [{ci_lo:.3f}, {ci_hi:.3f}] | {s['avg_tokens']:.0f} | {s['compression']:.1f}x |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    with open(README_PATH) as f:
        content = f.read()

    table = build_table(summary)

    # Replace everything between START_MARKER and END_MARKER
    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        print(f"[update_readme] Markers not found. start={start_idx}, end={end_idx}")
        print("[update_readme] Printing new table only:")
        print(table)
        return

    new_content = content[:start_idx] + table + "\n" + content[end_idx:]
    with open(README_PATH, "w") as f:
        f.write(new_content)
    print(f"[update_readme] README updated: {README_PATH}")


if __name__ == "__main__":
    main()
