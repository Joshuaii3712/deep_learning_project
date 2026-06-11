"""
MemoryTriggerNode
Decides whether a personality state update should be triggered this turn.

Triggers when ANY of the following conditions are met:
  1. total_tokens  > TRIGGER_TOKEN_LIMIT
  2. context utilisation > TRIGGER_CONTEXT_RATIO  (tokens / model context window)
  3. turn_count is a multiple of TRIGGER_TURN_LIMIT
"""
from __future__ import annotations

import logging

from config import (
    MODEL_N_CTX,
    TRIGGER_CONTEXT_RATIO,
    TRIGGER_TOKEN_LIMIT,
    TRIGGER_TURN_LIMIT,
)
from psm.state import AgentState

logger = logging.getLogger(__name__)


def memory_trigger_node(state: AgentState) -> AgentState:
    total_tokens = state.get("total_tokens", 0)
    turn_count = state.get("turn_count", 0)

    reasons: list[str] = []

    if total_tokens > TRIGGER_TOKEN_LIMIT:
        reasons.append(f"tokens={total_tokens}>{TRIGGER_TOKEN_LIMIT}")

    ctx_ratio = total_tokens / MODEL_N_CTX if MODEL_N_CTX > 0 else 0.0
    if ctx_ratio > TRIGGER_CONTEXT_RATIO:
        reasons.append(f"ctx_ratio={ctx_ratio:.2f}>{TRIGGER_CONTEXT_RATIO}")

    if turn_count > 0 and turn_count % TRIGGER_TURN_LIMIT == 0:
        reasons.append(f"turn_count={turn_count} (multiple of {TRIGGER_TURN_LIMIT})")

    triggered = len(reasons) > 0
    if triggered:
        logger.info(
            "MemoryTriggerNode | TRIGGERED | session=%s | reasons=%s",
            state.get("session_id"),
            reasons,
        )

    return {
        **state,
        "trigger_memory_update": triggered,
        "metadata": {
            **state.get("metadata", {}),
            "trigger_reasons": reasons,
        },
    }
