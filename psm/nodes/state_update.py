"""
StateUpdateNode
Applies the EMA update to the persistent PersonalityState and persists it.

    state = alpha * state + (1 - alpha) * estimate

Only executes when trigger_memory_update is True and an estimate is available.
"""
from __future__ import annotations

import logging

from config import ALPHA
from psm.state import AgentState, PersonalityState

logger = logging.getLogger(__name__)


def state_update_node(state: AgentState) -> AgentState:
    if not state.get("trigger_memory_update", False):
        return state

    estimate_dict = state.get("metadata", {}).get("personality_estimate")
    if estimate_dict is None:
        logger.warning("StateUpdateNode | no estimate found, skipping update")
        return state

    current: PersonalityState = state.get("personality", PersonalityState())
    estimate = PersonalityState.from_dict(estimate_dict)

    updated = current.update(estimate, alpha=ALPHA)

    logger.info(
        "StateUpdateNode | updated personality | session=%s | %s → %s",
        state.get("session_id"),
        current,
        updated,
    )

    # Persist to database if db is available via metadata
    db = state.get("metadata", {}).get("db")
    if db is not None:
        session_id = state.get("session_id", "")
        if session_id:
            trigger_reasons = state.get("metadata", {}).get("trigger_reasons", [])
            db.save_personality(session_id, updated)
            db.record_personality_snapshot(
                session_id,
                updated,
                trigger_reason="; ".join(trigger_reasons),
            )

    return {
        **state,
        "personality": updated,
    }
