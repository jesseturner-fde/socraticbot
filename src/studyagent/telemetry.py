"""Telemetry and observability module for the Socratic Technical Study Agent.

Provides centralized tracing, execution latency metrics, token estimation,
and real-time trace output via the --trace CLI flag.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


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
    """Singleton-friendly telemetry and tracing engine."""

    _instance: Optional[TelemetryTracer] = None

    def __init__(self, trace_enabled: bool = False, console: Optional[Console] = None):
        self.trace_enabled = trace_enabled
        self.console = console or Console()
        self.turn_traces: List[TurnTrace] = []
        self.active_tool_calls: List[ToolExecutionTrace] = []

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

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        duration_seconds: float,
        output_summary: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> ToolExecutionTrace:
        """Records an executed tool call and outputs live trace if enabled."""
        trace = ToolExecutionTrace(
            tool_name=tool_name,
            arguments=arguments,
            duration_seconds=round(duration_seconds, 4),
            output_summary=output_summary,
            success=success,
            error_message=error_message,
        )
        self.active_tool_calls.append(trace)

        if self.trace_enabled:
            status_style = "bold green" if success else "bold red"
            status_text = "SUCCESS" if success else f"FAILED: {error_message}"
            text = Text()
            text.append(f"Tool: {tool_name}\n", style="bold cyan")
            text.append(f"Args: {arguments}\n", style="yellow")
            text.append(f"Status: {status_text} ({trace.duration_seconds}s)\n", style=status_style)
            text.append(f"Output: {output_summary[:200]}...", style="dim")
            self.console.print(Panel(text, title="[bold magenta]🔍 Agent Tool Trace[/bold magenta]", border_style="magenta"))

        return trace

    def record_turn(
        self,
        user_query: str,
        response_preview: str,
        duration_seconds: float,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> TurnTrace:
        """Records a completed conversational turn."""
        turn_id = len(self.turn_traces) + 1
        turn_trace = TurnTrace(
            turn_id=turn_id,
            user_query=user_query,
            response_preview=response_preview,
            duration_seconds=round(duration_seconds, 3),
            tool_calls=list(self.active_tool_calls),
            token_usage=token_usage or {},
        )
        self.turn_traces.append(turn_trace)
        self.active_tool_calls.clear()

        if self.trace_enabled:
            tokens_str = f"Tokens: {turn_trace.token_usage}" if turn_trace.token_usage else "Tokens: estimated"
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
    """Decorator to trace tool execution duration and results automatically."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = TelemetryTracer.get_instance()
            start = time.perf_counter()
            # Combine args into a dictionary for logging
            bound_args = {"args": [str(a) for a in args], **kwargs}
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
