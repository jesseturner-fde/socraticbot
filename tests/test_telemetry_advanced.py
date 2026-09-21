"""Unit tests for OpenTelemetry distributed tracing, JSON logging, intent logging, and PII scrubbing."""

import json
import logging
from unittest.mock import patch
import pytest

from studyagent.telemetry import (
    JSONLogFormatter,
    PIIScrubber,
    TelemetryTracer,
    trace_tool,
)


def test_pii_scrubber_emails_and_ips():
    """Verify scrubber removes emails and IPv4 addresses."""
    text = "Send reports to alice@domain.co.uk from server 10.0.4.15."
    clean = PIIScrubber.redact(text)
    assert "alice@domain.co.uk" not in clean
    assert "[REDACTED_EMAIL]" in clean
    assert "10.0.4.15" not in clean
    assert "[REDACTED_IP]" in clean


def test_pii_scrubber_api_keys():
    """Verify scrubber redacts Gemini and Google API keys."""
    mock_aiza = "AI" + "za" + "SyMockDummyTestKeyForPIITesting123456"
    mock_token = "AQ" + ".MockServiceAccountDummyKeyForTestingOnly1234"
    text = f"Authorization: {mock_aiza} and {mock_token}"
    clean = PIIScrubber.redact(text)
    assert "AI" + "za" not in clean
    assert "AQ" + ".Mock" not in clean
    assert "[REDACTED_API_KEY]" in clean


def test_pii_scrubber_nested_structures():
    """Verify scrubber works recursively across dictionaries and lists."""
    mock_key = "AI" + "za" + "SyNestedMockDummyKey12345678901234567"
    payload = {
        "user": {"email": "dev@company.com", "keys": [mock_key]},
        "records": ["Contact 555-123-4567"],
    }
    cleaned = PIIScrubber.redact(payload)
    assert cleaned["user"]["email"] == "[REDACTED_EMAIL]"
    assert cleaned["user"]["keys"][0] == "[REDACTED_API_KEY]"
    assert "[REDACTED_PHONE]" in cleaned["records"][0]


def test_structured_json_logging():
    """Verify JSONLogFormatter outputs parseable single-line JSON logs."""
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="agent.py",
        lineno=42,
        msg="Processing turn with user query containing email@test.com",
        args=(),
        exc_info=None,
    )
    record.event_type = "test_event"
    record.agent = "orchestrator"
    record.session_id = "sess_001"

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["event_type"] == "test_event"
    assert parsed["agent"] == "orchestrator"
    assert parsed["session_id"] == "sess_001"
    assert "[REDACTED_EMAIL]" in parsed["message"]
    assert "timestamp" in parsed


def test_pre_execution_intent_logging(caplog):
    """Verify log_pre_execution_intent logs structured event before execution."""
    tracer = TelemetryTracer.get_instance(trace_enabled=False)
    with caplog.at_level(logging.INFO):
        tracer.log_pre_execution_intent(
            tool_name="retrieve_paper_section",
            intent_description="Retrieve Section 3.2.2 for multi-head projections",
            arguments={"topic_key": "multi_head_attention"},
            agent_name="paper_specialist",
            session_id="session_intent_test",
        )
    assert any("retrieve_paper_section" in record.message for record in caplog.records)


def test_opentelemetry_span_creation():
    """Verify tool execution generates valid OpenTelemetry span without errors."""
    tracer = TelemetryTracer.get_instance(trace_enabled=False)
    tracer.clear()

    @trace_tool("test_otel_tool")
    def sample_func(x: int) -> int:
        return x * 2

    res = sample_func(21)
    assert res == 42
    assert len(tracer.active_tool_calls) == 1
    assert tracer.active_tool_calls[0].tool_name == "test_otel_tool"
    assert tracer.active_tool_calls[0].success
