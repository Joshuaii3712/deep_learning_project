"""
PersonalityRendererNode
Converts the current PersonalityState into a natural-language profile
and injects it directly below the system prompt.
"""
from __future__ import annotations

import logging

from psm.renderer import render_personality
from psm.state import AgentState, PersonalityState

logger = logging.getLogger(__name__)


def personality_renderer_node(state: AgentState) -> AgentState:
    personality: PersonalityState = state.get("personality", PersonalityState())
    profile = render_personality(personality)

    # Build the enriched system prompt
    base_system = state.get("system_prompt", "You are a helpful assistant.")
    if profile:
        enriched_system = f"{base_system}\n\n{profile}"
    else:
        enriched_system = base_system

    logger.debug(
        "PersonalityRendererNode | profile_lines=%d | session=%s",
        profile.count("\n"),
        state.get("session_id"),
    )

    return {
        **state,
        "personality_profile": profile,
        "system_prompt": enriched_system,
    }
