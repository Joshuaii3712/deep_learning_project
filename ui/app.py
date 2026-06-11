"""
Streamlit UI for the Psychological State Memory (PSM) agent.

Run:
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import streamlit as st
import pandas as pd

from config import DEFAULT_PERSONALITY
from psm import PSMAgent, PSMDatabase, PersonalityState
from psm.renderer import render_personality_verbose

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSM Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.trait-bar-container { margin: 4px 0; }
.trait-label { font-size: 0.85rem; color: #888; margin-bottom: 2px; }
.profile-box {
    background: #1e1e2e;
    border-left: 3px solid #7c6af5;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-top: 8px;
}
.trigger-badge {
    background: #2a2a3e;
    color: #a0a0ff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

TRAIT_COLORS = {
    "openness": "#7c6af5",
    "conscientiousness": "#4fc3f7",
    "extraversion": "#81c784",
    "agreeableness": "#ffb74d",
    "neuroticism": "#ef5350",
}

TRAIT_EMOJI = {
    "openness": "🔭",
    "conscientiousness": "📋",
    "extraversion": "💬",
    "agreeableness": "🤝",
    "neuroticism": "⚡",
}


# ── Session state init ─────────────────────────────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = PSMDatabase()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = PSMAgent(
        session_id=st.session_state.session_id,
        db=st.session_state.db,
        system_prompt="You are a helpful, thoughtful assistant.",
    )

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of (role, content)

if "personality_history_display" not in st.session_state:
    st.session_state.personality_history_display = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def reset_session():
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.agent = PSMAgent(
        session_id=new_id,
        db=st.session_state.db,
        system_prompt=st.session_state.get("system_prompt_input", "You are a helpful assistant."),
    )
    st.session_state.chat_display = []
    st.session_state.personality_history_display = []


def load_session(session_id: str):
    st.session_state.session_id = session_id
    st.session_state.agent = PSMAgent(
        session_id=session_id,
        db=st.session_state.db,
        system_prompt=st.session_state.get("system_prompt_input", "You are a helpful assistant."),
    )
    msgs = st.session_state.db.get_message_history(session_id)
    st.session_state.chat_display = [(m["role"], m["content"]) for m in msgs]
    st.session_state.personality_history_display = (
        st.session_state.db.get_personality_history(session_id)
    )


def render_trait_bar(trait: str, score: float, description: str):
    color = TRAIT_COLORS.get(trait, "#888")
    emoji = TRAIT_EMOJI.get(trait, "")
    pct = int(score * 100)
    label = trait.capitalize()
    desc_text = description if description else "—"
    st.markdown(
        f"""
        <div class="trait-bar-container">
            <div class="trait-label">{emoji} <b>{label}</b> &nbsp;<span style="color:{color}">{score:.2f}</span></div>
            <div style="background:#2a2a3e;border-radius:6px;height:8px;width:100%">
                <div style="background:{color};border-radius:6px;height:8px;width:{pct}%"></div>
            </div>
            <div style="font-size:0.75rem;color:#666;margin-top:2px">{desc_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 PSM")
    st.caption("Psychological State Memory")

    st.divider()

    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, thoughtful assistant.",
        height=80,
        key="system_prompt_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Session", use_container_width=True):
            reset_session()
            st.rerun()
    with col2:
        st.caption(f"`{st.session_state.session_id[:8]}…`")

    st.divider()

    # ── Personality panel ──────────────────────────────────────────────────────
    st.subheader("Personality State")
    agent: PSMAgent = st.session_state.agent
    verbose = render_personality_verbose(agent.personality)

    for trait, info in verbose.items():
        render_trait_bar(trait, info["score"], info["description"])

    st.divider()

    # ── Active profile ─────────────────────────────────────────────────────────
    if agent.personality_profile:
        st.subheader("Active Profile")
        st.markdown(
            f'<div class="profile-box">{agent.personality_profile.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Session browser ────────────────────────────────────────────────────────
    st.subheader("Sessions")
    sessions = st.session_state.db.list_sessions()
    if sessions:
        session_ids = [s["session_id"] for s in sessions[:10]]
        labels = [f"{s['session_id'][:8]}… (T{s['turn_count']})" for s in sessions[:10]]
        choice = st.selectbox("Load session", labels, index=0)
        if st.button("Load", use_container_width=True):
            idx = labels.index(choice)
            load_session(session_ids[idx])
            st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("PSM Conversational Agent")

# Display chat history
chat_container = st.container()
with chat_container:
    for role, content in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(content)

# Input
if prompt := st.chat_input("Type a message…"):
    # Show user message immediately
    st.session_state.chat_display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = agent.chat(prompt)
        st.markdown(response)

    st.session_state.chat_display.append(("assistant", response))

    # Update personality history for chart
    snap = agent.personality.to_dict()
    snap["turn"] = agent.turn_count
    st.session_state.personality_history_display.append(snap)

    st.rerun()

# ── Personality trajectory chart ───────────────────────────────────────────────
if len(st.session_state.personality_history_display) > 1:
    st.divider()
    st.subheader("Personality Trajectory")
    df = pd.DataFrame(st.session_state.personality_history_display)
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    chart_df = df[["turn"] + [t for t in traits if t in df.columns]].set_index("turn")
    st.line_chart(chart_df, height=200)
