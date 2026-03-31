"""Multi-agent belief sharing demo.

Two BeliefStateAgent instances each observe different knowledge domains.
After share_with, both can answer questions from each other's corpus.

Run with:
    python examples/multiagent_demo.py

No external downloads or real LLM required.
"""

from vague import BeliefMemory, BeliefStateAgent

# ---------------------------------------------------------------------------
# Domain corpora — kept separate to simulate independent agents
# ---------------------------------------------------------------------------

SPACE_CORPUS = [
    "The Moon is Earth's only natural satellite.",
    "Mars has two small moons named Phobos and Deimos.",
    "The International Space Station orbits Earth at roughly 400 km altitude.",
    "Neil Armstrong became the first human to walk on the Moon on July 20, 1969.",
    "The James Webb Space Telescope was launched in December 2021.",
    "Saturn's rings are composed mainly of ice particles and rocky debris.",
    "A light-year is the distance light travels in one year, about 9.46 trillion km.",
    "The Milky Way galaxy contains an estimated 100–400 billion stars.",
    "Jupiter is the largest planet in the solar system.",
    "Voyager 1, launched in 1977, is the farthest human-made object from Earth.",
]

OCEAN_CORPUS = [
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "The Mariana Trench is the deepest known point in the ocean, at about 11 km.",
    "Oceans cover approximately 71% of Earth's surface.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Blue whales are the largest animals ever known to have lived on Earth.",
    "The Gulf Stream is a powerful Atlantic ocean current that influences European climate.",
    "Bioluminescence in the deep ocean is produced by organisms like dinoflagellates.",
    "Tidal forces are primarily caused by the Moon's gravitational pull.",
    "Deep-sea hydrothermal vents support life without sunlight through chemosynthesis.",
    "The Arctic Ocean is the smallest and shallowest of the world's oceans.",
]


def echo_llm(prompt: str) -> str:
    """Dummy LLM: returns the most relevant recalled line from the prompt context."""
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    # The last line is the task; return the first context line as a simulated answer
    if len(lines) > 1:
        return f"[Based on memory] {lines[0]}"
    return "[No context available]"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create two agents with separate memories
    # ------------------------------------------------------------------
    agent_space = BeliefStateAgent(
        llm_fn=echo_llm,
        memory=BeliefMemory(n_components=8),
        system_prompt="You are an astronomy assistant.",
        recall_k=3,
    )

    agent_ocean = BeliefStateAgent(
        llm_fn=echo_llm,
        memory=BeliefMemory(n_components=8),
        system_prompt="You are a marine science assistant.",
        recall_k=3,
    )

    # ------------------------------------------------------------------
    # 2. Each agent observes its own domain
    # ------------------------------------------------------------------
    print("Agent Space observing astronomy corpus...")
    for doc in SPACE_CORPUS:
        agent_space.observe(doc)

    print("Agent Ocean observing oceanography corpus...")
    for doc in OCEAN_CORPUS:
        agent_ocean.observe(doc)

    print()

    # ------------------------------------------------------------------
    # 3. Before sharing: each agent only knows its own domain
    # ------------------------------------------------------------------
    print("--- Before belief sharing ---")

    space_answer = agent_space.act("Tell me about the Moon.")
    print(f"Agent Space on Moon:  {space_answer}")

    ocean_answer = agent_ocean.act("Tell me about the Pacific Ocean.")
    print(f"Agent Ocean on Pacific: {ocean_answer}")

    # Agent Space has no ocean knowledge yet
    space_cross = agent_space.act("What is the deepest part of the ocean?")
    print(f"Agent Space on ocean (before share): {space_cross}")
    print()

    # ------------------------------------------------------------------
    # 4. Share beliefs: agent_ocean receives agent_space's beliefs
    # ------------------------------------------------------------------
    print("Sharing agent_space beliefs -> agent_ocean...")
    agent_space.memory.share_with(agent_ocean.memory)
    print("Sharing agent_ocean beliefs -> agent_space...")
    agent_ocean.memory.share_with(agent_space.memory)
    print()

    # ------------------------------------------------------------------
    # 5. After sharing: both agents can answer cross-domain questions
    # ------------------------------------------------------------------
    print("--- After belief sharing ---")

    space_cross_after = agent_space.act("What is the deepest part of the ocean?")
    print(f"Agent Space on ocean (after share):  {space_cross_after}")

    ocean_cross_after = agent_ocean.act("Who was the first human to walk on the Moon?")
    print(f"Agent Ocean on Moon (after share): {ocean_cross_after}")

    # ------------------------------------------------------------------
    # 6. Token usage
    # ------------------------------------------------------------------
    print()
    print("Token usage — Agent Space:", agent_space.token_usage())
    print("Token usage — Agent Ocean:", agent_ocean.token_usage())


if __name__ == "__main__":
    main()
