"""Unit tests for ContextCompactor and non-blocking async session operations."""

import asyncio
from pathlib import Path
import pytest

from studyagent.memory import (
    ContextCompactor,
    SessionManager,
    load_user_profile_async,
    save_user_profile_async,
)


def test_context_compactor_threshold():
    """Verify compactor triggers only when turns exceed max_turns threshold."""
    compactor = ContextCompactor(max_turns=4, preserve_recent=2)

    few_turns = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert not compactor.is_compaction_needed(few_turns)

    many_turns = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "tools_invoked": ["retrieve_paper_section"]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]
    assert compactor.is_compaction_needed(many_turns)


def test_context_compactor_compaction_logic():
    """Verify compaction preserves latest N turns and synthesizes older turns."""
    compactor = ContextCompactor(max_turns=4, preserve_recent=2)
    turns = [
        {"role": "user", "content": "Inquiry 1: Scaled dot product"},
        {"role": "assistant", "content": "Ans 1", "tools_invoked": ["retrieve_paper_section"]},
        {"role": "user", "content": "Inquiry 2: Multi-head attention"},
        {"role": "assistant", "content": "Ans 2", "tools_invoked": ["web_search"]},
        {"role": "user", "content": "Inquiry 3: Positional encoding"},
        {"role": "assistant", "content": "Ans 3"},
    ]

    synopsis, recent = compactor.compact(turns, existing_synopsis="Previous study session summary.")

    # Must preserve exactly the last 2 turns
    assert len(recent) == 2
    assert recent[0]["content"] == "Inquiry 3: Positional encoding"
    assert recent[1]["content"] == "Ans 3"

    # Synopsis must contain references to compacted inquiries and tools
    assert "Inquiry 1" in synopsis
    assert "Inquiry 2" in synopsis
    assert "retrieve_paper_section" in synopsis or "web_search" in synopsis
    assert "Previous study session summary" in synopsis


@pytest.mark.asyncio
async def test_async_session_manager(tmp_path):
    """Verify asynchronous session creation, turn appending, and persistence."""
    sm = SessionManager(sessions_dir=tmp_path)
    session = await sm.new_session_async("async_test_session")
    assert session["session_id"] == "async_test_session"

    # Async add turns
    await sm.add_turn_async(
        session_data=session,
        role="user",
        content="What is FlashAttention?",
    )
    await sm.add_turn_async(
        session_data=session,
        role="assistant",
        content="FlashAttention tiles attention into GPU SRAM.",
        tools_invoked=["web_search"],
    )

    reloaded = await sm.load_session_async("async_test_session")
    assert reloaded is not None
    assert len(reloaded["turns"]) == 2
    assert reloaded["turns"][0]["role"] == "user"
    assert reloaded["turns"][1]["tools_invoked"] == ["web_search"]

    sessions_list = await sm.list_sessions_async()
    assert len(sessions_list) == 1
    assert sessions_list[0]["session_id"] == "async_test_session"


@pytest.mark.asyncio
async def test_async_user_profile_operations(tmp_path):
    """Verify asynchronous profile loading and saving in background thread pool."""
    test_file = tmp_path / "async_profile.json"
    prof = await load_user_profile_async(test_file)
    assert "user_persona" in prof

    prof["user_persona"]["background"] = "Async ML Engineer"
    await save_user_profile_async(prof, test_file)

    reloaded = await load_user_profile_async(test_file)
    assert reloaded["user_persona"]["background"] == "Async ML Engineer"
