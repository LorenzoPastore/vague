"""Basic usage of BeliefMemory: remember, recall, stats, save, load.

Run with:
    python examples/basic_usage.py

No external downloads or real LLM required.
"""

import json
import os
import tempfile

from vague import BeliefMemory

# ---------------------------------------------------------------------------
# Inline corpus — 20 self-contained sentences
# ---------------------------------------------------------------------------

CORPUS = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower was constructed between 1887 and 1889.",
    "France is a republic located in Western Europe.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "Napoleon Bonaparte became Emperor of France in 1804.",
    "The Louvre in Paris is the world's most visited art museum.",
    "French is an official language of 29 countries worldwide.",
    "The Seine river flows through the heart of Paris.",
    "Versailles Palace was the principal residence of French kings from 1682 to 1789.",
    "France produces more than 1,000 distinct varieties of cheese.",
    "The Tour de France cycling race has been held annually since 1903.",
    "Mont Blanc, on the French-Italian border, is the highest peak in the Alps.",
    "The French Republic's motto is Liberté, Égalité, Fraternité.",
    "France has the largest land area of any country in the European Union.",
    "Claude Monet was a French painter and a founder of Impressionism.",
    "The Cannes Film Festival is held each May on the French Riviera.",
    "France was a founding member of the European Union in 1957.",
    "The TGV high-speed train network connects major French cities at up to 320 km/h.",
    "Victor Hugo wrote Les Misérables, published in 1862.",
    "France is the most visited country in the world by international tourists.",
]


def dummy_llm(prompt: str) -> str:
    """Echo the prompt back — stands in for a real LLM call."""
    lines = prompt.splitlines()
    # Return only the last line (the task) prefixed with "Answer: "
    task = lines[-1] if lines else prompt
    return f"Answer: {task}"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create memory
    # ------------------------------------------------------------------
    print("Creating BeliefMemory with 8 components...")
    memory = BeliefMemory(n_components=8)

    # ------------------------------------------------------------------
    # 2. Remember
    # ------------------------------------------------------------------
    print(f"Loading {len(CORPUS)} sentences into memory...")
    memory.remember_batch(CORPUS)
    print("Done.\n")

    # ------------------------------------------------------------------
    # 3. Recall
    # ------------------------------------------------------------------
    queries = [
        "What do you know about Paris?",
        "Tell me about French art and culture.",
        "When was the Eiffel Tower built?",
    ]

    for query in queries:
        results = memory.recall(query, k=3)
        print(f"Query: {query!r}")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r}")
        print()

    # ------------------------------------------------------------------
    # 4. Stats
    # ------------------------------------------------------------------
    s = memory.stats()
    print("Memory stats:")
    print(json.dumps(s, indent=2))
    print()

    # ------------------------------------------------------------------
    # 5. Save and load
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        memory.save(path)
        print(f"Saved memory to {path}")

        loaded = BeliefMemory.load(path)
        results = loaded.recall("Tell me about France.", k=2)
        print("Recall from loaded memory:")
        for r in results:
            print(f"  - {r}")
    finally:
        os.unlink(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
