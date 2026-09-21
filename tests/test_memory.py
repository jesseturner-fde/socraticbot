"""Unit tests for memory and session management."""

import json
from pathlib import Path
import pytest

from studyagent.memory import (
    SessionManager,
    format_profile_for_prompt,
    get_default_profile,
    load_user_profile,
    record_concept_mastery,
    save_user_profile,
    update_profile_trait,
)


def test_default_profile_structure():
    """Verify default profile has required persona and concept keys."""
    profile = get_default_profile()
    assert "user_persona" in profile
    assert "concept_mastery" in profile
    assert "last_updated" in profile
    assert len(profile["concept_mastery"]) == 5
    for concept in [
        "architecture_overview",
        "scaled_dot_product",
        "multi_head_attention",
        "positional_encoding",
        "computational_complexity",
    ]:
        assert concept in profile["concept_mastery"]
        assert profile["concept_mastery"][concept]["score"] == 0
        assert profile["concept_mastery"][concept]["status"] == "unseen"


def test_load_and_save_profile(tmp_path):
    """Verify loading, modifying, and saving user profile to custom path."""
    profile_path = tmp_path / "custom_profile.json"
    assert not profile_path.exists()

    # Automatically creates default on load if missing
    loaded = load_user_profile(path=profile_path)
    assert profile_path.exists()
    assert loaded["user_persona"]["background"] is not None

    # Modify and save
    loaded["user_persona"]["background"] = "Distributed Systems Researcher"
    save_user_profile(loaded, path=profile_path)

    # Re-read
    reloaded = load_user_profile(path=profile_path)
    assert reloaded["user_persona"]["background"] == "Distributed Systems Researcher"


def test_update_profile_trait(tmp_path):
    """Verify updating individual persona traits."""
    profile_path = tmp_path / "trait_profile.json"

    update_profile_trait("background", "ML Infrastructure Engineer", path=profile_path)
    prof = load_user_profile(profile_path)
    assert "ML Infrastructure Engineer" in prof["user_persona"]["background"]

    update_profile_trait("learning_style", "Prefers mathematical proofs", path=profile_path)
    prof = load_user_profile(profile_path)
    assert "Prefers mathematical proofs" in prof["user_persona"]["learning_style"]


def test_record_concept_mastery(tmp_path):
    """Verify scoring logic, status updates, and misconception recording."""
    profile_path = tmp_path / "mastery_profile.json"

    # Score 45 -> learning
    record_concept_mastery(
        concept_key="scaled_dot_product",
        score=45,
        notes="Confused dot product with element-wise product",
        path=profile_path,
    )
    prof = load_user_profile(profile_path)
    entry = prof["concept_mastery"]["scaled_dot_product"]
    assert entry["score"] == 45
    assert entry["status"] == "learning"
    assert "Confused dot product with element-wise product" in entry["misconceptions"]

    # Score 95 -> mastered
    record_concept_mastery(
        concept_key="scaled_dot_product",
        score=95,
        notes="Understands softmax variance normalization",
        path=profile_path,
    )
    prof = load_user_profile(profile_path)
    entry = prof["concept_mastery"]["scaled_dot_product"]
    assert entry["score"] == 95
    assert entry["status"] == "mastered"
    assert len(entry["misconceptions"]) == 2

    # Clamping test
    record_concept_mastery(concept_key="scaled_dot_product", score=150, path=profile_path)
    prof = load_user_profile(profile_path)
    assert prof["concept_mastery"]["scaled_dot_product"]["score"] == 100


def test_format_profile_for_prompt(tmp_path):
    """Verify format_profile_for_prompt produces markdown representation."""
    profile_path = tmp_path / "test_prof.json"
    record_concept_mastery("positional_encoding", 80, notes="Understands sinusoidal wavelengths", path=profile_path)
    prof = load_user_profile(profile_path)

    text = format_profile_for_prompt(prof)
    assert "### Learner Profile & Context (Long-Term Memory)" in text
    assert "`positional_encoding`: 80% [mastered]" in text


def test_session_manager_lifecycle(tmp_path):
    """Verify session creation, turn appending, and listing."""
    sm = SessionManager(sessions_dir=tmp_path)
    assert len(sm.list_sessions()) == 0

    # Create new session
    session = sm.new_session("test_session_1")
    assert session["session_id"] == "test_session_1"
    assert (tmp_path / "test_session_1.json").exists()

    # Add turns
    sm.add_turn(
        session_data=session,
        role="user",
        content="What is multi-head attention?",
    )
    sm.add_turn(
        session_data=session,
        role="assistant",
        content="Multi-head attention projects queries, keys, and values h times...",
        tools_invoked=["retrieve_paper_section"],
    )

    reloaded = sm.load_session("test_session_1")
    assert reloaded is not None
    assert len(reloaded["turns"]) == 2
    assert reloaded["turns"][0]["role"] == "user"
    assert reloaded["turns"][1]["tools_invoked"] == ["retrieve_paper_section"]
    assert "What is multi-head attention?" in reloaded["session_summary"]

    # List sessions
    sessions_list = sm.list_sessions()
    assert len(sessions_list) == 1
    assert sessions_list[0]["session_id"] == "test_session_1"
    assert sessions_list[0]["turns_count"] == 2
