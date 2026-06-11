"""
PersonalityEstimatorNode
Runs the Big Five estimator on the recent conversation window.
Only executes when trigger_memory_update is True.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config import BIG5_MODEL
from psm.personality_estimator import PersonalityEstimator
from psm.state import AgentState, PersonalityState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_estimator() -> PersonalityEstimator:
    return PersonalityEstimator(model_name=BIG5_MODEL, use_fallback=True)


def _build_text_window(messages: list[dict[str, str]], max_chars: int = 4000) -> str:
    """Concatenate recent messages for the estimator, newest last."""
    parts: list[str] = []
    total = 0
    for msg in reversed(messages):
        chunk = f"{msg['role'].upper()}: {msg['content']}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(reversed(parts))


def personality_estimator_node(state: AgentState) -> AgentState:
    if not state.get("trigger_memory_update", False):
        # Skip – not triggered
        return state

    messages = state.get("messages", [])
    if not messages:
        return state

    text = _build_text_window(messages)
    estimator = _get_estimator()
    estimate: PersonalityState = estimator.estimate(text)

    logger.debug(
        "PersonalityEstimatorNode | estimate=%s | session=%s",
        estimate,
        state.get("session_id"),
    )

    return {
        **state,
        "metadata": {
            **state.get("metadata", {}),
            "personality_estimate": estimate.to_dict(),
        },
    }
