"""Smoke test for the experimental SummaryBelief.

SummaryBelief is intentionally NOT exported from ``vague.__init__`` — it is
imported here from its module path to underscore its experimental status.
"""

from __future__ import annotations

from vague.summary_belief import SummaryBelief

CORPUS: list[str] = [
    "Paris is the capital of France and the largest city in the country.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Rome was the capital of the Roman Empire for over 500 years.",
    "The Colosseum in Rome could hold up to 80000 spectators.",
    "Tokyo is the capital of Japan and one of the most populous cities.",
    "Mount Fuji is the highest peak in Japan, an active stratovolcano.",
    "Berlin was divided by a wall from 1961 until its fall in 1989.",
    "The Brandenburg Gate is a neoclassical monument in central Berlin.",
] * 2  # 16 docs, enough to spread across multiple components


def _mock_llm_fn(prompt: str) -> str:
    """Returns a deterministic, length-bounded mock summary."""
    # First 60 chars of the prompt body as a stand-in summary.
    body = prompt.split("\n\nSummary:")[0]
    return f"[mock summary] {body[-80:].strip()}"


def test_summary_belief_fit_and_query_returns_summaries() -> None:
    """fit_with_summaries should generate per-component summaries; query
    should return summaries (strings), not raw chunks."""
    sb = SummaryBelief(n_components=4)
    sb.fit_with_summaries(CORPUS, llm_fn=_mock_llm_fn)

    # All non-empty components have a summary string.
    assert sb._summaries, "Expected non-empty per-component summaries."
    assert all(isinstance(s, str) for s in sb._summaries.values())

    results = sb.query("Tell me about Paris", top_k=2)
    assert 1 <= len(results) <= 2

    # Each result is (summary, score); summary must be a string and bear the
    # mock prefix — confirming it came from llm_fn, not from raw chunks.
    for summary, score in results:
        assert isinstance(summary, str) and summary
        assert isinstance(score, float)
        assert summary.startswith("[mock summary]"), (
            f"Expected summary from llm_fn, got raw chunk: {summary!r}"
        )

    stats = sb.summary_stats()
    assert stats["n_components"] == 4
    assert stats["n_components_with_summary"] >= 1
    assert stats["total_summary_chars"] > 0
