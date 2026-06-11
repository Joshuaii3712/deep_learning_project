"""
PersonalityEstimatorNode

When a trigger fires, estimates personality from the *front 50%* of the
active message list — the portion that will subsequently be dropped by
ContextCompressionNode.  This ensures the dropped content is absorbed into
the personality state before it leaves the LLM context.

Only user-side messages are used for estimation to avoid diluting the
personality signal with neutral assistant responses.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache

from config import BIG5_MODEL
from psm.personality_estimator import PersonalityEstimator
from psm.state import AgentState, PersonalityState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_estimator() -> PersonalityEstimator:
    return PersonalityEstimator(model_name=BIG5_MODEL)


def _build_text_window(messages: list[dict[str, str]], max_chars: int = 4000) -> str:
    """
    Concatenate user utterances from *messages* (front portion to be compressed).
    Newest-first traversal, user-only, up to max_chars.
    """
    parts: list[str] = []
    total = 0
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        chunk = msg["content"] + "\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(reversed(parts))


def personality_estimator_node(state: AgentState) -> AgentState:
    if not state.get("trigger_memory_update", False):
        return state

    messages = state.get("messages", [])
    if not messages:
        return state

    # Estimate on the front 50% — the portion about to be compressed away
    cut = max(1, math.floor(len(messages) * 0.5))
    front_messages = messages[:cut]

    text = _build_text_window(front_messages)
    if not text.strip():
        logger.warning(
            "PersonalityEstimatorNode | no user text in front 50%%, skipping | session=%s",
            state.get("session_id"),
        )
        return state

    estimator = _get_estimator()
    estimate: PersonalityState = estimator.estimate(text)

    logger.info(
        "PersonalityEstimatorNode | front=%d/%d msgs | estimate=%s | session=%s",
        cut, len(messages), estimate, state.get("session_id"),
    )

    return {
        **state,
        "metadata": {
            **state.get("metadata", {}),
            "personality_estimate": estimate.to_dict(),
        },
    }