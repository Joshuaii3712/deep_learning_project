"""
SQLite persistence for sessions, messages, and personality states.
Uses SQLAlchemy Core (no ORM) for minimal overhead.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, Text,
    create_engine, insert, select, update,
)

from config import DB_PATH
from psm.state import PersonalityState


# ── Schema ─────────────────────────────────────────────────────────────────────

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("turn_count", Integer, default=0),
    Column("total_tokens", Integer, default=0),
    # Big Five as individual columns for easy querying
    Column("openness", Float, default=0.5),
    Column("conscientiousness", Float, default=0.5),
    Column("extraversion", Float, default=0.5),
    Column("agreeableness", Float, default=0.5),
    Column("neuroticism", Float, default=0.5),
)

messages_table = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("timestamp", Float, nullable=False),
    Column("token_count", Integer, default=0),
)

personality_history_table = Table(
    "personality_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, nullable=False),
    Column("timestamp", Float, nullable=False),
    Column("openness", Float),
    Column("conscientiousness", Float),
    Column("extraversion", Float),
    Column("agreeableness", Float),
    Column("neuroticism", Float),
    Column("trigger_reason", String),
)


# ── Database class ─────────────────────────────────────────────────────────────

class PSMDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        metadata.create_all(self.engine)

    @contextmanager
    def _conn(self) -> Iterator:
        with self.engine.connect() as conn:
            yield conn
            conn.commit()

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                insert(sessions_table).values(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    turn_count=0,
                    total_tokens=0,
                    openness=0.5,
                    conscientiousness=0.5,
                    extraversion=0.5,
                    agreeableness=0.5,
                    neuroticism=0.5,
                )
            )

    def get_session(self, session_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                select(sessions_table).where(sessions_table.c.session_id == session_id)
            ).fetchone()
            return dict(row._mapping) if row else None

    def upsert_session(self, session_id: str, **kwargs) -> None:
        existing = self.get_session(session_id)
        kwargs["updated_at"] = time.time()
        with self._conn() as conn:
            if existing:
                conn.execute(
                    update(sessions_table)
                    .where(sessions_table.c.session_id == session_id)
                    .values(**kwargs)
                )
            else:
                now = kwargs.get("updated_at", time.time())
                conn.execute(
                    insert(sessions_table).values(
                        session_id=session_id,
                        created_at=now,
                        updated_at=now,
                        turn_count=kwargs.get("turn_count", 0),
                        total_tokens=kwargs.get("total_tokens", 0),
                        openness=kwargs.get("openness", 0.5),
                        conscientiousness=kwargs.get("conscientiousness", 0.5),
                        extraversion=kwargs.get("extraversion", 0.5),
                        agreeableness=kwargs.get("agreeableness", 0.5),
                        neuroticism=kwargs.get("neuroticism", 0.5),
                    )
                )

    def save_personality(self, session_id: str, personality: PersonalityState) -> None:
        d = personality.to_dict()
        self.upsert_session(session_id, **d)

    def load_personality(self, session_id: str) -> PersonalityState:
        row = self.get_session(session_id)
        if row is None:
            return PersonalityState()
        return PersonalityState(
            openness=row["openness"],
            conscientiousness=row["conscientiousness"],
            extraversion=row["extraversion"],
            agreeableness=row["agreeableness"],
            neuroticism=row["neuroticism"],
        )

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                insert(messages_table).values(
                    session_id=session_id,
                    role=role,
                    content=content,
                    timestamp=time.time(),
                    token_count=token_count,
                )
            )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        with self._conn() as conn:
            q = (
                select(messages_table)
                .where(messages_table.c.session_id == session_id)
                .order_by(messages_table.c.id)
            )
            if limit:
                q = q.limit(limit)
            rows = conn.execute(q).fetchall()
            return [dict(r._mapping) for r in rows]

    def get_message_history(self, session_id: str) -> list[dict[str, str]]:
        """Return list of {role, content} dicts for LLM consumption."""
        rows = self.get_messages(session_id)
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ── Personality history ───────────────────────────────────────────────────

    def record_personality_snapshot(
        self,
        session_id: str,
        personality: PersonalityState,
        trigger_reason: str = "",
    ) -> None:
        d = personality.to_dict()
        with self._conn() as conn:
            conn.execute(
                insert(personality_history_table).values(
                    session_id=session_id,
                    timestamp=time.time(),
                    trigger_reason=trigger_reason,
                    **d,
                )
            )

    def get_personality_history(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                select(personality_history_table)
                .where(personality_history_table.c.session_id == session_id)
                .order_by(personality_history_table.c.timestamp)
            ).fetchall()
            return [dict(r._mapping) for r in rows]

    # ── All sessions ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                select(sessions_table).order_by(sessions_table.c.updated_at.desc())
            ).fetchall()
            return [dict(r._mapping) for r in rows]
