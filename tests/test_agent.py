"""Unit tests for agent orchestration, system instruction, and telemetry."""

from unittest.mock import MagicMock

import pytest

from studyagent.agent import SocraticStudyAgent, build_system_instruction
from studyagent.memory import SessionManager
from studyagent.telemetry import TelemetryTracer, trace_tool


def test_build_system_instruction():
    """Verify system instruction contains required Socratic rules and tool guidance."""
    instr = build_system_instruction()
    assert "Socratic Technical Study Agent" in instr
    assert "active recall" in instr.lower()
    assert "Socratic Clarification Loop" in instr
    assert "retrieve_paper_section" in instr
    assert "web_search" in instr
    assert "update_user_profile" in instr
    assert "record_concept_progress" in instr


def test_agent_initialization(tmp_path):
    """Verify SocraticStudyAgent initializes ADK agent with correct tools and model."""
    sm = SessionManager(sessions_dir=tmp_path)
    agent = SocraticStudyAgent(model_name="gemini-3.5-flash", session_manager=sm)

    assert agent.adk_agent is not None
    assert agent.adk_agent.name == "socratic_study_agent"
    assert agent.adk_agent.model == "gemini-3.5-flash"
    assert len(agent.adk_agent.tools) == 4


@pytest.mark.asyncio
async def test_agent_mock_run(tmp_path):
    """Verify agent streaming and turn persistence using a mocked ADK runner."""
    sm = SessionManager(sessions_dir=tmp_path)
    sm.new_session("mock_session")
    agent = SocraticStudyAgent(session_manager=sm)

    # Mock runner's run_async generator
    mock_fc = MagicMock()
    mock_fc.name = "retrieve_paper_section"
    mock_fc.args = {"topic_key": "scaled_dot_product"}

    mock_event1 = MagicMock()
    mock_event1.get_function_calls.return_value = [mock_fc]
    mock_event1.get_function_responses.return_value = []
    mock_event1.content = None

    mock_part = MagicMock()
    mock_part.text = "Dividing by sqrt(d_k) stabilizes the variance. What happens to softmax without it?"

    mock_event2 = MagicMock()
    mock_event2.get_function_calls.return_value = []
    mock_event2.get_function_responses.return_value = []
    mock_event2.content = MagicMock(parts=[mock_part])

    async def mock_run_async(*args, **kwargs):
        yield mock_event1
        yield mock_event2

    agent.runner.run_async = mock_run_async

    events = []
    async for event in agent.send_message_stream(
        message="Why scale dot product?",
        session_id="mock_session",
    ):
        events.append(event)

    types_received = [e["type"] for e in events]
    assert "tool_call" in types_received
    assert "content_chunk" in types_received
    assert "final_response" in types_received

    final = [e for e in events if e["type"] == "final_response"][0]
    assert "Dividing by sqrt(d_k)" in final["text"]
    assert "retrieve_paper_section" in final["tools_invoked"]

    # Verify session turns were persisted
    sess_data = sm.load_session("mock_session")
    assert len(sess_data["turns"]) == 2
    assert sess_data["turns"][0]["role"] == "user"
    assert sess_data["turns"][1]["role"] == "assistant"
    assert sess_data["turns"][1]["tools_invoked"] == ["retrieve_paper_section"]


def test_telemetry_tracer():
    """Verify TelemetryTracer records tool calls and turns properly."""
    tracer = TelemetryTracer.get_instance(trace_enabled=False)
    tracer.clear()

    @trace_tool("sample_calc")
    def sample_calc(a: int, b: int) -> int:
        return a + b

    # Execute tool
    res = sample_calc(3, 4)
    assert res == 7
    assert len(tracer.active_tool_calls) == 1
    assert tracer.active_tool_calls[0].tool_name == "sample_calc"

    # Record turn
    turn_trace = tracer.record_turn(
        user_query="What is 3 + 4?",
        response_preview="The answer is 7.",
        duration_seconds=0.15,
    )
    assert turn_trace.turn_id == 1
    assert len(turn_trace.tool_calls) == 1
    assert len(tracer.active_tool_calls) == 0

    metrics = tracer.get_summary_metrics()
    assert metrics["total_turns"] == 1
    assert metrics["total_tool_calls"] == 1
    assert metrics["average_turn_latency_sec"] == 0.15
