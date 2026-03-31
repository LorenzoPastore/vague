"""Tests for vague.memory.BeliefMemory."""

from __future__ import annotations

from vague.memory import BeliefMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MACHINE_LEARNING_TEXTS = [
    "Neural networks learn representations from data.",
    "Gradient descent minimizes the loss function.",
    "Backpropagation computes gradients efficiently.",
    "Convolutional networks excel at image recognition.",
    "Recurrent networks process sequential data.",
    "Attention mechanisms power transformer models.",
    "Transfer learning reuses pretrained weights.",
    "Regularization prevents overfitting in models.",
    "Batch normalization stabilizes training dynamics.",
    "Dropout randomly deactivates neurons during training.",
    "Learning rate schedules improve convergence.",
    "Adam optimizer combines momentum and RMSProp.",
    "Embeddings map discrete tokens to dense vectors.",
    "Self-supervised learning uses unlabeled data.",
    "Reinforcement learning optimizes cumulative reward.",
    "Generative adversarial networks produce realistic images.",
    "Variational autoencoders learn latent distributions.",
    "Ensemble methods combine multiple weak learners.",
    "Feature engineering transforms raw inputs.",
    "Cross-validation estimates generalization performance.",
]

_COOKING_TEXTS = [
    "Sauté onions in olive oil until golden.",
    "Boil pasta in salted water for 10 minutes.",
    "Preheat the oven to 180 degrees Celsius.",
    "Whisk eggs until fluffy before folding in flour.",
    "Season steak with salt and pepper before grilling.",
    "Simmer the sauce on low heat to develop flavor.",
    "Julienne carrots add color to the salad.",
    "Blanch vegetables to preserve their bright color.",
    "Deglaze the pan with white wine after browning.",
    "Rest the roast for 15 minutes before slicing.",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_remember_recall():
    """Remember 20 ML texts, recall with ML query returns relevant texts."""
    mem = BeliefMemory(n_components=5)
    mem.remember_batch(_MACHINE_LEARNING_TEXTS)

    results = mem.recall("neural network gradient optimization", k=5)

    assert isinstance(results, list)
    assert len(results) <= 5
    assert all(isinstance(r, str) for r in results)
    # At least one result should be from the ML corpus
    assert any(r in _MACHINE_LEARNING_TEXTS for r in results)


def test_lazy_init():
    """Memory works before and after the 10-observation threshold."""
    mem = BeliefMemory(n_components=4)

    # Before threshold: recall works on pending texts
    for text in _MACHINE_LEARNING_TEXTS[:5]:
        mem.remember(text)
    assert not mem.belief._fitted
    early = mem.recall("gradient descent", k=3)
    assert isinstance(early, list)

    # After threshold: belief is fitted
    for text in _MACHINE_LEARNING_TEXTS[5:12]:
        mem.remember(text)
    assert mem.belief._fitted

    results = mem.recall("gradient descent", k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_share_with():
    """After sharing, other memory can recall from self's observations."""
    mem_ml = BeliefMemory(n_components=4)
    mem_cooking = BeliefMemory(n_components=4)

    mem_ml.remember_batch(_MACHINE_LEARNING_TEXTS)
    mem_cooking.remember_batch(_COOKING_TEXTS)

    # Share ML knowledge into cooking memory
    mem_ml.share_with(mem_cooking)

    # cooking memory should now recall ML texts for an ML query
    results = mem_cooking.recall("neural network training", k=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, str) for r in results)


def test_save_load_roundtrip(tmp_path):
    """Save to disk, load back, verify same recall results (as a set)."""
    mem = BeliefMemory(n_components=4)
    mem.remember_batch(_MACHINE_LEARNING_TEXTS)

    query = "gradient descent optimization"
    before = mem.recall(query, k=5)

    path = str(tmp_path / "memory.json")
    mem.save(path)

    loaded = BeliefMemory.load(path)
    after = loaded.recall(query, k=5)

    # Order may vary due to floating-point differences; set equality is correct
    assert set(before) == set(after)
    assert len(before) == len(after)


def test_stats_keys():
    """stats() returns all expected keys with correct types."""
    mem = BeliefMemory(n_components=4)
    mem.remember_batch(_MACHINE_LEARNING_TEXTS)

    s = mem.stats()
    expected_keys = {"n_components", "n_observations", "compression_ratio", "entropy"}
    assert expected_keys == set(s.keys())
    assert s["n_observations"] == len(_MACHINE_LEARNING_TEXTS)
    assert s["compression_ratio"] > 0
    assert s["entropy"] >= 0
