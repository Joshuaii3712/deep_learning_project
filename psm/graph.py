"""
PSM LangGraph workflow.

Graph topology:
    ConversationNode
        ↓
    MemoryTriggerNode
        ↓
    PersonalityEstimatorNode   ← runs on front 50% of messages when triggered
        ↓
    StateUpdateNode            ← EMA update
        ↓
    ContextCompressionNode     ← drops front 50% from active context (kept in DB)
        ↓
    PersonalityRendererNode
        ↓
    GenerationNode             ← uses recent 50% + personality profile
        ↓
    END

SQLite stores the full conversation history permanently.
The in-memory message list only holds the recent (uncompressed) half.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from langgraph.graph import END, StateGraph

from psm.database import PSMDatabase
from psm.nodes import (
    conversation_node,
    generation_node,
    memory_trigger_node,
    personality_estimator_node,
    personality_renderer_node,
    state_update_node,
)
from psm.state import AgentState, PersonalityState

logger = logging.getLogger(__name__)


# ── Context compression node ───────────────────────────────────────────────────

def context_compression_node(state: AgentState) -> AgentState:
    """
    When a memory trigger fires, drop the front 50% of the active message list
    from the LLM context.  The full history is already persisted in SQLite by
    the database layer; this only affects what is passed to the LLM.

    The personality state (updated by StateUpdateNode) absorbs the behavioral
    signal from the dropped messages, so behavioral consistency is maintained
    via the profile injection rather than raw history.
    """
    if not state.get("trigger_memory_update", False):
        return state

    messages = state.get("messages", [])
    if len(messages) < 4:
        # Too few messages to compress meaningfully
        return state

    cut = math.floor(len(messages) * 0.5)
    dropped = messages[:cut]
    retained = messages[cut:]

    logger.info(
        "ContextCompressionNode | dropped=%d retained=%d | session=%s",
        len(dropped), len(retained), state.get("session_id"),
    )

    return {
        **state,
        "messages": retained,
        "metadata": {
            **state.get("metadata", {}),
            "last_compression_dropped": len(dropped),
        },
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Compile and return the PSM LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("conversation",           conversation_node)
    graph.add_node("memory_trigger",         memory_trigger_node)
    graph.add_node("personality_estimator",  personality_estimator_node)
    graph.add_node("state_update",           state_update_node)
    graph.add_node("context_compression",    context_compression_node)
    graph.add_node("personality_renderer",   personality_renderer_node)
    graph.add_node("generation",             generation_node)

    graph.set_entry_point("conversation")
    graph.add_edge("conversation",          "memory_trigger")
    graph.add_edge("memory_trigger",        "personality_estimator")
    graph.add_edge("personality_estimator", "state_update")
    graph.add_edge("state_update",          "context_compression")
    graph.add_edge("context_compression",   "personality_renderer")
    graph.add_edge("personality_renderer",  "generation")
    graph.add_edge("generation",            END)

    return graph.compile()


# ── PSMAgent ───────────────────────────────────────────────────────────────────

class PSMAgent:
    """
    High-level agent wrapping the compiled graph.

    Memory model
    ------------
    - SQLite  : full conversation history (permanent archive)
    - self._state["messages"] : active LLM context (recent 50% after compression)
    - self._state["personality"] : persistent Big Five state vector

    At each trigger the front 50% of the active context is compressed into the
    personality state and removed from the LLM context window, while the full
    history is preserved in SQLite for auditing and replay.

    Parameters
    ----------
    session_id : str
    db : PSMDatabase | None
    system_prompt : str
        Base system prompt (personality profile is appended below this).
    """

    def __init__(
        self,
        session_id: str,
        db: PSMDatabase | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.session_id = session_id
        self.db = db
        self.system_prompt = system_prompt
        self.graph = build_graph()

        # Load or initialise state
        if db is not None:
            db.upsert_session(session_id)
            personality  = db.load_personality(session_id)
            # On resume, only load the most recent 50% of stored messages
            # so we don't re-inflate a previously compressed context.
            all_msgs     = db.get_message_history(session_id)
            cut          = math.floor(len(all_msgs) * 0.5)
            active_msgs  = all_msgs[cut:] if len(all_msgs) >= 4 else all_msgs
            session_row  = db.get_session(session_id) or {}
            total_tokens = session_row.get("total_tokens", 0)
            turn_count   = session_row.get("turn_count", 0)
        else:
            personality  = PersonalityState()
            active_msgs  = []
            total_tokens = 0
            turn_count   = 0

        self._state: AgentState = {
            "session_id":         session_id,
            "messages":           active_msgs,
            "turn_count":         turn_count,
            "total_tokens":       total_tokens,
            "personality":        personality,
            "personality_profile": "",
            "trigger_memory_update": False,
            "system_prompt":      system_prompt,
            "response":           "",
            "metadata":           {"db": db},
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant response.

        The user message is appended to both the active context and SQLite.
        After generation, if compression occurred, the active message list is
        already trimmed; SQLite always retains the full history.
        """
        # Append user turn to active context
        self._state["messages"] = list(self._state.get("messages", [])) + [
            {"role": "user", "content": user_message}
        ]

        # Persist user turn to DB immediately (before generation)
        if self.db is not None:
            user_tokens = len(user_message.split()) * 4 // 3
            self.db.add_message(
                self.session_id, "user", user_message, token_count=user_tokens
            )
            self._state["total_tokens"] = (
                self._state.get("total_tokens", 0) + user_tokens
            )

        # Reset system prompt so renderer always injects fresh profile
        self._state["system_prompt"] = self.system_prompt

        # Run graph (compression happens inside if triggered)
        result: AgentState = self.graph.invoke(self._state)
        self._state = result

        return result.get("response", "")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def personality(self) -> PersonalityState:
        return self._state.get("personality", PersonalityState())

    @property
    def personality_profile(self) -> str:
        return self._state.get("personality_profile", "")

    @property
    def turn_count(self) -> int:
        return self._state.get("turn_count", 0)

    @property
    def active_context_length(self) -> int:
        """Number of messages currently in the active LLM context."""
        return len(self._state.get("messages", []))