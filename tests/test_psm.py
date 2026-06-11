"""
Unit tests for PSM core components.
Run: python -m pytest tests/test_psm.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from psm.state import PersonalityState
from psm.renderer import render_personality, score_to_bucket, render_personality_verbose
from psm.database import PSMDatabase


# ── PersonalityState ──────────────────────────────────────────────────────────

class TestPersonalityState:
    def test_default_values(self):
        p = PersonalityState()
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert getattr(p, trait) == 0.5

    def test_clamp(self):
        p = PersonalityState(openness=1.5, neuroticism=-0.2)
        clamped = p.clamp()
        assert clamped.openness == 1.0
        assert clamped.neuroticism == 0.0

    def test_ema_update(self):
        current = PersonalityState(openness=0.5)
        estimate = PersonalityState(openness=1.0)
        updated = current.update(estimate, alpha=0.99)
        expected = 0.99 * 0.5 + 0.01 * 1.0
        assert abs(updated.openness - expected) < 1e-9

    def test_update_no_change_at_alpha_1(self):
        current = PersonalityState(openness=0.3)
        estimate = PersonalityState(openness=0.9)
        updated = current.update(estimate, alpha=1.0)
        assert abs(updated.openness - 0.3) < 1e-9

    def test_to_from_dict_roundtrip(self):
        p = PersonalityState(openness=0.8, neuroticism=0.2)
        assert PersonalityState.from_dict(p.to_dict()) == p


# ── Renderer ──────────────────────────────────────────────────────────────────

class TestRenderer:
    def test_score_to_bucket(self):
        assert score_to_bucket(0.0) == 0
        assert score_to_bucket(0.2) == 1
        assert score_to_bucket(0.5) == 2
        assert score_to_bucket(0.7) == 3
        assert score_to_bucket(1.0) == 4

    def test_neutral_returns_empty(self):
        p = PersonalityState()  # all 0.5
        profile = render_personality(p)
        assert profile == ""

    def test_high_openness_renders(self):
        p = PersonalityState(openness=0.9)
        profile = render_personality(p)
        assert "novel" in profile or "alternative" in profile or "explores" in profile

    def test_low_agreeableness_renders(self):
        p = PersonalityState(agreeableness=0.0)
        profile = render_personality(p)
        assert "objective" in profile or "analysis" in profile

    def test_profile_header(self):
        p = PersonalityState(openness=0.9)
        profile = render_personality(p)
        assert profile.startswith("Psychological Profile")

    def test_verbose_has_all_traits(self):
        p = PersonalityState()
        verbose = render_personality_verbose(p)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in verbose
            assert "score" in verbose[trait]
            assert "bucket" in verbose[trait]


# ── Database ──────────────────────────────────────────────────────────────────

class TestDatabase:
    @pytest.fixture
    def db(self):
        return PSMDatabase(":memory:")

    def test_create_and_load_session(self, db):
        db.create_session("sess1")
        row = db.get_session("sess1")
        assert row is not None
        assert row["session_id"] == "sess1"

    def test_save_and_load_personality(self, db):
        db.create_session("sess2")
        p = PersonalityState(openness=0.8, neuroticism=0.1)
        db.save_personality("sess2", p)
        loaded = db.load_personality("sess2")
        assert abs(loaded.openness - 0.8) < 1e-6
        assert abs(loaded.neuroticism - 0.1) < 1e-6

    def test_add_and_retrieve_messages(self, db):
        db.create_session("sess3")
        db.add_message("sess3", "user", "Hello!")
        db.add_message("sess3", "assistant", "Hi there!")
        msgs = db.get_message_history("sess3")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there!"

    def test_personality_history_snapshot(self, db):
        db.create_session("sess4")
        p = PersonalityState(openness=0.7)
        db.record_personality_snapshot("sess4", p, trigger_reason="test")
        history = db.get_personality_history("sess4")
        assert len(history) == 1
        assert abs(history[0]["openness"] - 0.7) < 1e-6

    def test_list_sessions(self, db):
        db.create_session("sA")
        db.create_session("sB")
        sessions = db.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "sA" in ids and "sB" in ids
