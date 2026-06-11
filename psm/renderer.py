"""
Personality renderer: converts a PersonalityState into a natural-language
psychological profile string that can be injected into the system prompt.

Bucket mapping  (score → bucket):
    bucket = clamp(int(score * 5), 0, 4)

    0 = strongly low
    1 = moderately low
    2 = neutral           ← no sentence emitted
    3 = moderately high
    4 = strongly high
"""
from __future__ import annotations

from psm.state import PersonalityState

# ── Rule tables ───────────────────────────────────────────────────────────────
# Each trait maps bucket index → description string (bucket 2 = None → omit).

_RULES: dict[str, dict[int, str | None]] = {
    "openness": {
        0: "The assistant prefers familiar and conventional approaches.",
        1: "The assistant tends to stick to established methods over novel ones.",
        2: None,
        3: "The assistant is generally curious and open to new ideas.",
        4: "The assistant actively explores novel possibilities and alternative perspectives.",
    },
    "conscientiousness": {
        0: "The assistant tends to be flexible and spontaneous rather than strictly organised.",
        1: "The assistant is somewhat relaxed about structure and detailed planning.",
        2: None,
        3: "The assistant is generally careful, organised, and goal-oriented.",
        4: "The assistant is highly methodical, precise, and thorough in all tasks.",
    },
    "extraversion": {
        0: "The assistant tends to be reserved rather than highly expressive.",
        1: "The assistant communicates in a calm and measured way.",
        2: None,
        3: "The assistant is fairly expressive and engaging in conversation.",
        4: "The assistant is enthusiastic, lively, and highly sociable in tone.",
    },
    "agreeableness": {
        0: "The assistant tends to prioritise objective analysis over emotional support.",
        1: "The assistant favours directness and may not emphasise consensus.",
        2: None,
        3: "The assistant is generally considerate, cooperative, and supportive.",
        4: "The assistant responds in a warm, cooperative, and empathetic manner.",
    },
    "neuroticism": {
        0: "The assistant remains emotionally stable and rarely focuses on worst-case outcomes.",
        1: "The assistant generally maintains a steady, grounded perspective.",
        2: None,
        3: "The assistant is attentive to potential risks and complications.",
        4: "The assistant tends to carefully consider risks and potential negative outcomes.",
    },
}

_TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# ── Public API ─────────────────────────────────────────────────────────────────

def score_to_bucket(score: float) -> int:
    """Convert a [0, 1] score to a bucket in [0, 4]."""
    return max(0, min(4, int(score * 5)))


def render_personality(state: PersonalityState) -> str:
    """
    Convert a PersonalityState to a natural-language profile block.

    Returns an empty string if all traits fall into the neutral bucket.

    Example output:
        Psychological Profile

        - The assistant actively explores novel possibilities and alternative perspectives.
        - The assistant responds in a warm, cooperative, and empathetic manner.
        - The assistant tends to be reserved rather than highly expressive.
    """
    lines: list[str] = []
    for trait in _TRAIT_ORDER:
        score = getattr(state, trait)
        bucket = score_to_bucket(score)
        description = _RULES[trait].get(bucket)
        if description:
            lines.append(f"- {description}")

    if not lines:
        return ""

    return "Psychological Profile\n\n" + "\n".join(lines)


def render_personality_verbose(state: PersonalityState) -> dict[str, dict]:
    """
    Return a verbose dict with bucket, score, and description per trait.
    Useful for debugging and UI display.
    """
    result = {}
    for trait in _TRAIT_ORDER:
        score = getattr(state, trait)
        bucket = score_to_bucket(score)
        result[trait] = {
            "score": round(score, 4),
            "bucket": bucket,
            "description": _RULES[trait].get(bucket, ""),
        }
    return result
