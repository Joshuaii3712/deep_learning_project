"""
ConversationNode
Appends the latest user message to the agent state and persists it.
"""
from __future__ import annotations

import logging

from psm.state import AgentState

logger = logging.getLogger(__name__)


def conversation_node(state: AgentState) -> AgentState:
    """
    Receives an AgentState that must already contain:
      - session_id
      - messages  (the full history including the new user turn)
      - turn_count
      - total_tokens

    Increments turn_count by 1 and returns the updated state.
    The caller is responsible for appending the new user message
    before invoking the graph.
    """
    turn = state.get("turn_count", 0) + 1
    logger.debug("ConversationNode | session=%s turn=%d", state.get("session_id"), turn)

    return {
        **state,
        "turn_count": turn,
        "trigger_memory_update": False,  # reset flag
    }
