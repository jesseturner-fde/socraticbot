"""Unit tests for studyagent tools."""

from pathlib import Path
from unittest.mock import patch
import pytest

from studyagent.tools import (
    retrieve_paper_section,
    web_search,
    update_user_profile,
    record_concept_progress,
)


def test_retrieve_paper_section_all_concepts():
    """Verify all 5 required paper concepts are retrievable with formulas and details."""
    concepts = [
        "architecture_overview",
        "scaled_dot_product",
        "multi_head_attention",
        "positional_encoding",
        "computational_complexity",
    ]
    for c in concepts:
        res = retrieve_paper_section(c)
        assert "error" not in res
        assert "paper_section" in res
        assert "summary" in res
        assert "key_points" in res
        assert len(res["key_points"]) > 0
        assert "formulas" in res
        assert len(res["formulas"]) > 0


def test_retrieve_paper_section_aliases():
    """Verify fuzzy matching and aliases for topics."""
    res_overview = retrieve_paper_section("architecture")
    assert res_overview["topic_key"] == "architecture_overview"

    res_dot = retrieve_paper_section("dot_product")
    assert res_dot["topic_key"] == "scaled_dot_product"

    res_mha = retrieve_paper_section("mha")
    assert res_mha["topic_key"] == "multi_head_attention"

    res_pe = retrieve_paper_section("sinusoidal")
    assert res_pe["topic_key"] == "positional_encoding"


def test_retrieve_paper_section_unknown_topic():
    """Verify helpful error handling for unknown topics."""
    res = retrieve_paper_section("quantum_computing_v1")
    assert "error" in res
    assert "available_topics" in res
    assert len(res["available_topics"]) == 5


def test_web_search_curated_topics():
    """Verify modern ML concepts return rich technical results."""
    queries = [
        "FlashAttention memory complexity",
        "RoPE rotary positional embeddings",
        "Grouped Query Attention GQA",
        "PyTorch native SDPA scaled dot product",
        "LLaMA architecture RMSNorm",
    ]
    for q in queries:
        result = web_search(q)
        assert len(result) > 50
        assert "Search Result" in result or "Web Search Results" in result


def test_web_search_general_query():
    """Verify general technical synthesis fallback."""
    result = web_search("general transformer training tips")
    assert "Technical Search Result" in result
    assert "FlashAttention" in result


def test_web_search_discovery_engine_mcp(monkeypatch):
    """Verify MCP endpoint handling with mock network response."""
    monkeypatch.setenv("DISCOVERY_ENGINE_API_KEY", "mock_key")

    mock_response = {
        "result": {
            "content": "Google Discovery Engine found 3 papers on modern attention optimizations."
        }
    }

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        res = web_search("contemporary attention scaling")
        assert "Google Discovery Engine MCP Result" in res
        assert "modern attention optimizations" in res


def test_update_user_profile_tool(tmp_path):
    """Verify tool properly updates long-term profile."""
    test_prof_file = tmp_path / "user_profile.json"
    with patch("studyagent.memory.DEFAULT_PROFILE_PATH", test_prof_file):
        msg = update_user_profile("background", "Robotics software engineer")
        assert "Robotics software engineer" in msg

        # Update learning style
        msg2 = update_user_profile("learning_style", "Prefers hands-on debugging")
        assert "Prefers hands-on debugging" in msg2


def test_record_concept_progress_tool(tmp_path):
    """Verify concept mastery tool scores and records misconceptions."""
    test_prof_file = tmp_path / "user_profile.json"
    with patch("studyagent.memory.DEFAULT_PROFILE_PATH", test_prof_file):
        msg = record_concept_progress(
            "multi_head_attention",
            90,
            "Mastered representation subspace projections"
        )
        assert "score=90" in msg
        assert "status=mastered" in msg
