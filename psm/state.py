"""
Core state definitions.
PersonalityState  – the persistent Big Five vector.
AgentState        – the LangGraph conversation state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, TypedDict

from config import DEFAULT_PERSONALITY


# ── Big Five personality vector ───────────────────────────────────────────────

@dataclass
class PersonalityState:
    openness: float = DEFAULT_PERSONALITY["openness"]
    conscientiousness: float = DEFAULT_PERSONALITY["conscientiousness"]
    extraversion: float = DEFAULT_PERSONALITY["extraversion"]
    agreeableness: float = DEFAULT_PERSONALITY["agreeableness"]
    neuroticism: float = DEFAULT_PERSONALITY["neuroticism"]

    # ── helpers ───────────────────────────────────────────────────────────────

    def clamp(self) -> "PersonalityState":
        """Ensure all values are in [0, 1]."""
        return PersonalityState(
            openness=max(0.0, min(1.0, self.openness)),
            conscientiousness=max(0.0, min(1.0, self.conscientiousness)),
            extraversion=max(0.0, min(1.0, self.extraversion)),
            agreeableness=max(0.0, min(1.0, self.agreeableness)),
            neuroticism=max(0.0, min(1.0, self.neuroticism)),
        )

    def update(self, estimate: "PersonalityState", alpha: float = 0.99) -> "PersonalityState":
        """
        Exponential moving average:
            state = alpha * state + (1 - alpha) * estimate
        """
        return PersonalityState(
            openness=alpha * self.openness + (1 - alpha) * estimate.openness,
            conscientiousness=alpha * self.conscientiousness + (1 - alpha) * estimate.conscientiousness,
            extraversion=alpha * self.extraversion + (1 - alpha) * estimate.extraversion,
            agreeableness=alpha * self.agreeableness + (1 - alpha) * estimate.agreeableness,
            neuroticism=alpha * self.neuroticism + (1 - alpha) * estimate.neuroticism,
        ).clamp()

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "PersonalityState":
        return cls(**{k: float(v) for k, v in d.items()})

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v:.3f}" for k, v in self.to_dict().items())
        return f"PersonalityState({fields})"


# ── LangGraph agent state ─────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # conversation
    session_id: str
    messages: list[dict[str, str]]          # [{"role": "user"|"assistant", "content": "..."}]
    turn_count: int
    total_tokens: int

    # personality
    personality: PersonalityState
    personality_profile: str                # rendered natural-language description

    # trigger flags
    trigger_memory_update: bool

    # generation
    system_prompt: str
    response: str

    # misc
    metadata: dict[str, Any]
