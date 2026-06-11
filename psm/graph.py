"""
PSM LangGraph workflow.

Graph topology:
    ConversationNode
        ↓
    MemoryTriggerNode
        ↓
    PersonalityEstimatorNode
        ↓
    StateUpdateNode
        ↓
    PersonalityRendererNode
        ↓
    GenerationNode
        ↓
    END
"""
from __future__ import annotations

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


def build_graph() -> StateGraph:
    """Compile and return the PSM LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("conversation", conversation_node)
    graph.add_node("memory_trigger", memory_trigger_node)
    graph.add_node("personality_estimator", personality_estimator_node)
    graph.add_node("state_update", state_update_node)
    graph.add_node("personality_renderer", personality_renderer_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("conversation")
    graph.add_edge("conversation", "memory_trigger")
    graph.add_edge("memory_trigger", "personality_estimator")
    graph.add_edge("personality_estimator", "state_update")
    graph.add_edge("state_update", "personality_renderer")
    graph.add_edge("personality_renderer", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


# ── Convenience runner ─────────────────────────────────────────────────────────

class PSMAgent:
    """
    High-level agent that wraps the compiled graph and manages session state.

    Parameters
    ----------
    session_id : str
        Unique identifier for this conversation.
    db : PSMDatabase | None
        If provided, personality and messages are persisted.
    system_prompt : str
        Base system prompt (before personality injection).
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
            personality = db.load_personality(session_id)
            raw_msgs = db.get_message_history(session_id)
            total_tokens = (db.get_session(session_id) or {}).get("total_tokens", 0)
            turn_count = (db.get_session(session_id) or {}).get("turn_count", 0)
        else:
            personality = PersonalityState()
            raw_msgs = []
            total_tokens = 0
            turn_count = 0

        self._state: AgentState = {
            "session_id": session_id,
            "messages": raw_msgs,
            "turn_count": turn_count,
            "total_tokens": total_tokens,
            "personality": personality,
            "personality_profile": "",
            "trigger_memory_update": False,
            "system_prompt": system_prompt,
            "response": "",
            "metadata": {"db": db},
        }

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        Updates internal state in-place.
        """
        # Add user message to history & persist
        self._state["messages"] = list(self._state.get("messages", [])) + [
            {"role": "user", "content": user_message}
        ]
        if self.db is not None:
            user_tokens = len(user_message.split()) * 4 // 3
            self.db.add_message(self.session_id, "user", user_message, token_count=user_tokens)
            self._state["total_tokens"] = (
                self._state.get("total_tokens", 0) + user_tokens
            )

        # Reset system prompt to base (renderer will re-inject personality)
        self._state["system_prompt"] = self.system_prompt

        # Run graph
        result: AgentState = self.graph.invoke(self._state)
        self._state = result

        return result.get("response", "")

    @property
    def personality(self) -> PersonalityState:
        return self._state.get("personality", PersonalityState())

    @property
    def personality_profile(self) -> str:
        return self._state.get("personality_profile", "")

    @property
    def turn_count(self) -> int:
        return self._state.get("turn_count", 0)
