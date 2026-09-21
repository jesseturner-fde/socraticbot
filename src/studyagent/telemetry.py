"""Telemetry, observability, OpenTelemetry distributed tracing, structured JSON logging,
pre-execution intent logging, and PII redaction for the Socratic Technical Study Agent.
"""

from __future__ import annotations

from datetime import datetime, timezone
import functools
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

# OpenTelemetry Tracing
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# =====================================================================
# PII Redaction Engine
# =====================================================================


class PIIScrubber:
    """Regex-based scrubber to redact sensitive PII and secrets."""

    PATTERNS = [
        # Google Gemini / Google Cloud API keys (AIza..., AQ....)
        (re.compile(r"AIza[0-9A-Za-z\-_]{25,}"), "[REDACTED_API_KEY]"),
        (re.compile(r"AQ\.[0-9A-Za-z\-_]{20,}"), "[REDACTED_API_KEY]"),
        (re.compile(r"(?i)(api[-_]?key|secret|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{15,})['\"]?"), r"\1=[REDACTED_SECRET]"),
        # Email addresses
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
        # Phone numbers
        (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
        # IP Addresses
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
        # SSN / Credit Cards
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CREDIT_CARD]"),
    ]

    @classmethod
    def redact(cls, text: Any) -> Any:
        """Recursively scrubs PII from strings, dictionaries, and lists."""
        if isinstance(text, str):
            scrubbed = text
            for pattern, replacement in cls.PATTERNS:
                scrubbed = pattern.sub(replacement, scrubbed)
            return scrubbed
        elif isinstance(text, dict):
            return {k: cls.redact(v) for k, v in text.items()}
        elif isinstance(text, list):
            return [cls.redact(item) for item in text]
        return text


# =====================================================================
# Structured JSON Logging
# =====================================================================


class JSONLogFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": PIIScrubber.redact(record.getMessage()),
        }
        # Add custom extra attributes if available
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type
        if hasattr(record, "agent"):
            log_entry["agent"] = record.agent
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "metadata"):
            log_entry["metadata"] = PIIScrubber.redact(record.metadata)

        # Inject OpenTelemetry trace IDs if an active span exists
        current_span = otel_trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            ctx = current_span.get_span_context()
            log_entry["trace_id"] = format(ctx.trace_id, "032x")
            log_entry["span_id"] = format(ctx.span_id, "016x")

        return json.dumps(log_entry)


def setup_structured_logger(name: str = "studyagent") -> logging.Logger:
    """Configures structured JSON logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONLogFormatter())
        logger.addHandler(handler)
    return logger


structured_logger = setup_structured_logger()


# =====================================================================
# OpenTelemetry Setup & Telemetry Tracer
# =====================================================================


def _init_tracer() -> otel_trace.Tracer:
    """Initializes OpenTelemetry TracerProvider."""
    provider = otel_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)
    return otel_trace.get_tracer("studyagent.telemetry", "0.1.0")


GLOBAL_TRACER = _init_tracer()


@dataclass
class ToolExecutionTrace:
    """Record of an individual tool call."""

    tool_name: str
    arguments: Dict[str, Any]
    duration_seconds: float
    output_summary: str
    success: bool
    error_message: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TurnTrace:
    """Record of an entire conversational turn."""

    turn_id: int
    user_query: str
    response_preview: str
    duration_seconds: float
    tool_calls: List[ToolExecutionTrace] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TelemetryTracer:
    """Singleton-friendly telemetry and tracing engine using OpenTelemetry."""

    _instance: Optional[TelemetryTracer] = None

    def __init__(self, trace_enabled: bool = False, console: Optional[Console] = None):
        self.trace_enabled = trace_enabled
        self.console = console or Console()
        self.turn_traces: List[TurnTrace] = []
        self.active_tool_calls: List[ToolExecutionTrace] = []
        self.otel_tracer = GLOBAL_TRACER
        self.structured_log = structured_logger

    @classmethod
    def get_instance(cls, trace_enabled: bool = False) -> TelemetryTracer:
        if cls._instance is None:
            cls._instance = cls(trace_enabled=trace_enabled)
        else:
            if trace_enabled:
                cls._instance.trace_enabled = True
        return cls._instance

    def enable_trace(self, enabled: bool = True) -> None:
        """Enables or disables live stdout/rich tracing."""
        self.trace_enabled = enabled

    def log_pre_execution_intent(
        self,
        tool_name: str,
        intent_description: str,
        arguments: Dict[str, Any],
        agent_name: str = "orchestrator",
        session_id: str = "unknown",
    ) -> None:
        """Logs explicit pre-execution intent before any tool or sub-agent execution."""
        safe_args = PIIScrubber.redact(arguments)
        safe_intent = PIIScrubber.redact(intent_description)

        # Structured JSON Log
        self.structured_log.info(
            f"Pre-execution intent for tool '{tool_name}': {safe_intent}",
            extra={
                "event_type": "pre_execution_intent",
                "agent": agent_name,
                "session_id": session_id,
                "metadata": {
                    "tool": tool_name,
                    "intent": safe_intent,
                    "arguments": safe_args,
                },
            },
        )

        # Real-time console panel if --trace is enabled
        if self.trace_enabled:
            panel_text = Text()
            panel_text.append(f"Agent: {agent_name} -> Tool: {tool_name}\n", style="bold cyan")
            panel_text.append(f"Intent: {safe_intent}\n", style="italic yellow")
            panel_text.append(f"Args: {safe_args}", style="dim")
            self.console.print(
                Panel(
                    panel_text,
                    title="[bold blue]🎯 Pre-Execution Intent[/bold blue]",
                    border_style="blue",
                )
            )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        duration_seconds: float,
        output_summary: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> ToolExecutionTrace:
        """Records an executed tool call with OpenTelemetry span and structured logging."""
        safe_args = PIIScrubber.redact(arguments)
        safe_summary = PIIScrubber.redact(output_summary)
        safe_err = PIIScrubber.redact(error_message) if error_message else None

        trace = ToolExecutionTrace(
            tool_name=tool_name,
            arguments=safe_args,
            duration_seconds=round(duration_seconds, 4),
            output_summary=safe_summary,
            success=success,
            error_message=safe_err,
        )
        self.active_tool_calls.append(trace)

        # OpenTelemetry span instrumentation
        with self.otel_tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.duration_seconds", duration_seconds)
            span.set_attribute("tool.success", success)
            if not success and error_message:
                span.set_status(Status(StatusCode.ERROR, safe_err))
                span.record_exception(Exception(safe_err))
            else:
                span.set_status(Status(StatusCode.OK))

        # Structured JSON Log
        self.structured_log.info(
            f"Tool '{tool_name}' executed in {round(duration_seconds, 4)}s (Success: {success})",
            extra={
                "event_type": "tool_execution",
                "metadata": {
                    "tool": tool_name,
                    "arguments": safe_args,
                    "duration_seconds": round(duration_seconds, 4),
                    "success": success,
                    "error": safe_err,
                },
            },
        )

        if self.trace_enabled:
            status_style = "bold green" if success else "bold red"
            status_text = "SUCCESS" if success else f"FAILED: {safe_err}"
            text = Text()
            text.append(f"Tool: {tool_name}\n", style="bold cyan")
            text.append(f"Args: {safe_args}\n", style="yellow")
            text.append(f"Status: {status_text} ({trace.duration_seconds}s)\n", style=status_style)
            text.append(f"Output: {safe_summary[:200]}...", style="dim")
            self.console.print(
                Panel(
                    text,
                    title="[bold magenta]🔍 OpenTelemetry Tool Trace[/bold magenta]",
                    border_style="magenta",
                )
            )

        return trace

    def record_turn(
        self,
        user_query: str,
        response_preview: str,
        duration_seconds: float,
        token_usage: Optional[Dict[str, int]] = None,
        session_id: str = "unknown",
    ) -> TurnTrace:
        """Records a completed conversational turn with OpenTelemetry span and structured logging."""
        turn_id = len(self.turn_traces) + 1
        safe_query = PIIScrubber.redact(user_query)
        safe_response = PIIScrubber.redact(response_preview)

        turn_trace = TurnTrace(
            turn_id=turn_id,
            user_query=safe_query,
            response_preview=safe_response,
            duration_seconds=round(duration_seconds, 3),
            tool_calls=list(self.active_tool_calls),
            token_usage=token_usage or {},
        )
        self.turn_traces.append(turn_trace)
        self.active_tool_calls.clear()

        # OpenTelemetry span
        with self.otel_tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("turn.id", turn_id)
            span.set_attribute("session.id", session_id)
            span.set_attribute("turn.duration_seconds", duration_seconds)
            span.set_attribute("tools.invoked_count", len(turn_trace.tool_calls))
            span.set_status(Status(StatusCode.OK))

        # Structured JSON Log
        self.structured_log.info(
            f"Turn #{turn_id} completed in {round(duration_seconds, 3)}s",
            extra={
                "event_type": "agent_turn",
                "session_id": session_id,
                "metadata": {
                    "turn_id": turn_id,
                    "query_preview": safe_query[:80],
                    "duration_seconds": round(duration_seconds, 3),
                    "tools_invoked": [t.tool_name for t in turn_trace.tool_calls],
                    "token_usage": token_usage or {},
                },
            },
        )

        if self.trace_enabled:
            tokens_str = (
                f"Tokens: {turn_trace.token_usage}"
                if turn_trace.token_usage
                else "Tokens: estimated"
            )
            self.console.print(
                f"[dim magenta][Trace] Turn #{turn_id} completed in {turn_trace.duration_seconds}s | "
                f"Tools invoked: {len(turn_trace.tool_calls)} | {tokens_str}[/dim magenta]"
            )

        return turn_trace

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates cumulative metrics across all turns."""
        total_turns = len(self.turn_traces)
        total_tool_calls = sum(len(t.tool_calls) for t in self.turn_traces)
        avg_latency = (
            round(sum(t.duration_seconds for t in self.turn_traces) / total_turns, 3)
            if total_turns > 0
            else 0.0
        )
        return {
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "average_turn_latency_sec": avg_latency,
            "trace_enabled": self.trace_enabled,
        }

    def clear(self) -> None:
        """Resets all recorded telemetry."""
        self.turn_traces.clear()
        self.active_tool_calls.clear()


def trace_tool(tool_name: str) -> Callable:
    """Decorator to trace tool execution duration, OpenTelemetry span, and results."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = TelemetryTracer.get_instance()

            # Pre-execution intent logging
            bound_args = {"args": [str(a) for a in args], **kwargs}
            tracer.log_pre_execution_intent(
                tool_name=tool_name,
                intent_description=f"Invoking {tool_name} to fulfill agent inquiry",
                arguments=bound_args,
            )

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                output_str = str(result)
                summary = output_str[:250] + ("..." if len(output_str) > 250 else "")
                tracer.record_tool_call(
                    tool_name=tool_name,
                    arguments=bound_args,
                    duration_seconds=duration,
                    output_summary=summary,
                    success=True,
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                tracer.record_tool_call(
                    tool_name=tool_name,
                    arguments=bound_args,
                    duration_seconds=duration,
                    output_summary="Tool execution raised an exception",
                    success=False,
                    error_message=str(e),
                )
                raise

        return wrapper

    return decorator
