"""
GenerationNode
Calls the LLM and appends the assistant response to the message history.
Updates total_tokens in state.
"""
from __future__ import annotations

import logging

from psm.llm import get_llm
from psm.state import AgentState

logger = logging.getLogger(__name__)


def generation_node(state: AgentState) -> AgentState:
    llm = get_llm()

    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "You are a helpful assistant.")

    response: str = llm.generate(
        messages=messages,
        system_prompt=system_prompt,
    )

    # Rough token count update
    response_tokens = llm.count_tokens(response)
    total_tokens = state.get("total_tokens", 0) + response_tokens

    # Append assistant turn to history
    updated_messages = list(messages) + [{"role": "assistant", "content": response}]

    # Persist to database if available
    db = state.get("metadata", {}).get("db")
    session_id = state.get("session_id", "")
    if db is not None and session_id:
        db.add_message(session_id, "assistant", response, token_count=response_tokens)
        db.upsert_session(session_id, total_tokens=total_tokens, turn_count=state.get("turn_count", 0))

    logger.debug(
        "GenerationNode | response_tokens=%d | total_tokens=%d | session=%s",
        response_tokens,
        total_tokens,
        session_id,
    )

    return {
        **state,
        "messages": updated_messages,
        "response": response,
        "total_tokens": total_tokens,
    }
