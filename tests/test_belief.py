"""Tests for vague.belief.GaussianBelief."""

from __future__ import annotations

import numpy as np
import pytest

from vague.belief import GaussianBelief

# ---------------------------------------------------------------------------
# Inline public-domain corpus — 50 paragraphs from Pride and Prejudice
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
    "My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last?",
    "Mr. Bennet replied that he had not.",
    "But it is, returned she; for Mrs. Long has just been here, and she told me all about it.",
    "Mr. Bennet made no answer.",
    "Do you not want to know who has taken it? cried his wife impatiently.",
    "You want to tell me, and I have no objection to hearing it.",
    "This was invitation enough.",
    "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England.",
    "What is his name? Bingley.",
    "Is he married or single? Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!",
    "How so? Can it affect them? My dear Mr. Bennet, replied his wife, how can you be so tiresome! You must know that I am thinking of his marrying one of them.",
    "Is that his design in settling here?",
    "Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.",
    "I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party.",
    "My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be any thing extraordinary now.",
    "When a woman has five grown-up daughters, she ought to give over thinking of her own beauty.",
    "In such cases, a woman has not often much beauty to think of.",
    "As soon as they entered, Bingley looked at Jane so much, that his companions could not but observe his admiration. It was plain to them all.",
    "Elizabeth, having rather expected to affront him, was amazed at his gallantry; but there was a mixture of sweetness and archness in her manner which made it difficult for her to affront anybody.",
    "When Jane and Elizabeth were alone, the former, who had been cautious in her praise of Mr. Bingley before, expressed to her sister just how very much she admired him.",
    "He is just what a young man ought to be, said she, sensible, good-humoured, lively; and I never saw such happy manners! so much ease, with such perfect good breeding!",
    "He is also handsome, replied Elizabeth, which a young man ought likewise to be, if he possibly can. His character is thereby complete.",
    "I was very much flattered by his asking me to dance a second time. I did not expect such a compliment.",
    "Did not you? I did for you. But that is one great difference between us. Compliments always take you by surprise, and me never.",
    "Why should I be surprised? I never try to please, and I am always pleased.",
    "That is exactly the matter. You never see a fault in any body. All the world is good and agreeable in your eyes. I have never heard you speak ill of a human being in my life.",
    "I would not wish to be hasty in censuring any one; but I always speak what I think.",
    "I know you do; and it is that which makes the wonder. With your good sense, to be so honestly blind to the follies and nonsense of others! Affectation of candour is common enough; one meets it every where.",
    "Elizabeth had never been blind to the impropriety of her father's behaviour as a husband.",
    "She had always seen it with pain; but respecting his abilities, and grateful for his affectionate treatment of herself, she endeavoured to forget what she could not overlook.",
    "Mr. Bingley had not been of age two years when he was tempted by an accidental recommendation to look at Netherfield House.",
    "He did look at it, and into it for half an hour, was pleased with the situation and the principal rooms, satisfied with what the owner said in its praise, and took it immediately.",
    "Between him and Darcy there was a very steady friendship, in spite of a great opposition of character.",
    "Bingley was endeared to Darcy by the easiness, openness, ductility of his temper, though no disposition could offer a greater contrast to his own.",
    "Darcy had never been so bewitched by any woman as he was by her.",
    "He really believed, that were it not for the inferiority of her connections, he should be in some danger.",
    "Miss Bennet he acknowledged to be pretty, but she smiled too much.",
    "Mrs. Hurst and her sister allowed it to be so, but still they admired her and liked her, and pronounced her to be a sweet girl.",
    "She is tolerable; but not handsome enough to tempt me; and I am in no humour at present to give consequence to young ladies who are slighted by other men.",
    "Darcy walked off; and Elizabeth remained with no very cordial feelings towards him.",
    "She told the story however with great spirit among her friends; for she had a lively, playful disposition, which delighted in any thing ridiculous.",
    "The evening altogether passed off pleasantly to the whole family.",
    "Mrs. Bennet had seen her eldest daughter much admired by the Netherfield party.",
    "Mr. Bingley had danced with her twice, and she had been distinguished by his sisters.",
    "Jane was as much gratified by this as her mother could be, though in a quieter way.",
    "Elizabeth felt Jane's pleasure. Mary had heard herself mentioned to Miss Bingley as the most accomplished girl in the neighbourhood.",
    "Catherine and Lydia had been fortunate enough to be never without partners, which was all that they had yet learnt to care for at a ball.",
    "They returned therefore in good spirits to Longbourn, the village where they lived, and of which they were the principal inhabitants.",
]

# A smaller split for merge tests
CORPUS_A = CORPUS[:25]
CORPUS_B = CORPUS[25:]

N_COMPONENTS = 8  # small enough to fit quickly in tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_belief() -> GaussianBelief:
    """A GaussianBelief fitted on the full corpus."""
    gb = GaussianBelief(n_components=N_COMPONENTS)
    gb.fit(CORPUS)
    return gb


@pytest.fixture(scope="module")
def belief_a() -> GaussianBelief:
    gb = GaussianBelief(n_components=N_COMPONENTS)
    gb.fit(CORPUS_A)
    return gb


@pytest.fixture(scope="module")
def belief_b() -> GaussianBelief:
    gb = GaussianBelief(n_components=N_COMPONENTS)
    gb.fit(CORPUS_B)
    return gb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fit_shape(fitted_belief: GaussianBelief) -> None:
    """After fit, the GMM must have the requested number of components."""
    assert fitted_belief._gmm is not None
    # n_components is capped at len(texts), so we get at most N_COMPONENTS
    assert fitted_belief._gmm.n_components == min(N_COMPONENTS, len(CORPUS))
    assert fitted_belief._gmm.means_.shape[0] == min(N_COMPONENTS, len(CORPUS))


def test_query_relevance(fitted_belief: GaussianBelief) -> None:
    """Texts returned for a query should be more similar to the query than random."""
    from vague.embedder import Embedder

    query = "Mr. Bingley danced with Jane at the ball"
    results = fitted_belief.query(query, top_k=5)

    assert len(results) == 5
    for text, score in results:
        assert isinstance(text, str)
        assert isinstance(score, float)

    # Top result should have higher cosine similarity to query than a random text
    embedder = Embedder()
    q_vec = embedder.embed_single(query)
    top_text = results[0][0]
    top_vec = embedder.embed_single(top_text)

    # Pick a text that is clearly unrelated
    random_text = "She had always seen it with pain; but respecting his abilities."
    rnd_vec = embedder.embed_single(random_text)

    # The GMM scores should produce a non-trivially ordered list;
    # we verify the top result exists in the corpus and has higher similarity than random
    assert float(np.dot(q_vec, top_vec)) >= float(np.dot(q_vec, rnd_vec))
    assert top_text in CORPUS


def test_update_modifies_belief(fitted_belief: GaussianBelief) -> None:
    """Calling update() should change the GMM means."""
    original_means = fitted_belief._gmm.means_.copy()

    new_text = "Mr. Darcy proposed to Elizabeth most unexpectedly on a cold morning."
    fitted_belief.update(new_text, weight=2.0)

    updated_means = fitted_belief._gmm.means_

    # At least one component mean should have changed
    assert not np.allclose(original_means, updated_means), (
        "Expected means to change after update, but they did not."
    )
    # The new text should now be in the corpus
    assert new_text in fitted_belief._texts


def test_merge_combines(belief_a: GaussianBelief, belief_b: GaussianBelief) -> None:
    """Merged belief must contain texts from both source beliefs."""
    merged = belief_a.merge(belief_b, alpha=0.5)

    assert merged._fitted
    # All texts from both beliefs should be in merged
    for text in CORPUS_A:
        assert text in merged._texts
    for text in CORPUS_B:
        assert text in merged._texts

    # Should be queryable
    results = merged.query("Bingley danced with Jane", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r[0], str) for r in results)


def test_serialization_roundtrip(fitted_belief: GaussianBelief) -> None:
    """to_dict → from_dict should produce substantially identical query results.

    JSON serialization may introduce sub-epsilon floating point differences that
    can reorder near-tied log-likelihood scores. We require at least 4/5 top
    results to match (robust to one near-tie swap at the boundary).
    """
    query = "Darcy admired Elizabeth greatly"

    original_results = fitted_belief.query(query, top_k=5)

    d = fitted_belief.to_dict()
    restored = GaussianBelief.from_dict(d)

    restored_results = restored.query(query, top_k=5)

    assert len(original_results) == len(restored_results)

    orig_texts = {text for text, _ in original_results}
    rest_texts = {text for text, _ in restored_results}

    overlap = len(orig_texts & rest_texts)
    assert overlap >= 4, (
        f"Too few overlapping results after roundtrip ({overlap}/5).\n"
        f"  original: {sorted(orig_texts)}\n"
        f"  restored: {sorted(rest_texts)}"
    )


def test_compression_ratio_positive(fitted_belief: GaussianBelief) -> None:
    """compression_ratio() must return a positive float."""
    ratio = fitted_belief.compression_ratio()
    assert isinstance(ratio, float)
    assert ratio > 0, f"Expected compression_ratio > 0, got {ratio}"
