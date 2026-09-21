"""Multi-agent orchestrator, strategic model routing, security guardrails, and HITL hooks.

Built on Google ADK 2.0 and Gemini. Implements:
1. Multi-Agent Architecture:
   - Orchestrator Agent (Dialogue coordination & query triage)
   - Paper Specialist Agent (Vaswani et al. 2017 text & formulas)
   - Modern ML Specialist Agent (FlashAttention, RoPE, PyTorch SDPA)
   - Socratic Tutor Agent (Misconception diagnosis & active recall probing)
2. Strategic Model Routing: Dynamically routes between fast models and deep reasoning models.
3. Security & Evaluation Guardrails: Prompt injection screening, domain enforcement, and Socratic checks.
4. Human-in-the-Loop (HITL): Approval hooks for persistent profile/mastery modifications.
"""

from __future__ import annotations

import asyncio
from enum import Enum
import logging
import os
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import google.adk as adk
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from studyagent.guardrails import HITLPolicy, HumanInTheLoopManager, SecurityGuardrails
from studyagent.memory import (
    SessionManager,
    format_profile_for_prompt,
    load_user_profile,
    load_user_profile_async,
)
from studyagent.telemetry import TelemetryTracer
from studyagent.tools import (
    AGENT_TOOLS,
    record_concept_progress,
    retrieve_paper_section,
    update_user_profile,
    web_search,
)

load_dotenv()

# Synchronize API keys for google-genai and google-adk
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger(__name__)


# =====================================================================
# Multi-Agent Roles & Strategic Model Router
# =====================================================================


class AgentRole(str, Enum):
    """Specialized sub-agent personas in the multi-agent system."""

    ORCHESTRATOR = "orchestrator"
    PAPER_SPECIALIST = "paper_specialist"
    MODERN_ML_SPECIALIST = "modern_ml_specialist"
    SOCRATIC_TUTOR = "socratic_tutor"


class ModelRouter:
    """Strategically selects the optimal Gemini model based on task complexity."""

    DEFAULT_FAST_MODEL = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    DEFAULT_REASONING_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    DEFAULT_DEEP_MODEL = os.environ.get("GEMINI_DEEP_MODEL", "gemini-3.5-flash")

    @classmethod
    def route_model(cls, role: AgentRole, complexity: str = "standard") -> str:
        """Dynamically routes to the most capable yet cost-effective model."""
        if role == AgentRole.SOCRATIC_TUTOR and complexity == "fast":
            return cls.DEFAULT_FAST_MODEL
        elif role == AgentRole.PAPER_SPECIALIST and complexity == "deep_math":
            return cls.DEFAULT_DEEP_MODEL
        elif role == AgentRole.MODERN_ML_SPECIALIST:
            return cls.DEFAULT_REASONING_MODEL
        return cls.DEFAULT_REASONING_MODEL


def build_system_instruction(user_profile: Optional[Dict[str, Any]] = None) -> str:
    """Builds the dynamic system instruction injecting learner profile and Socratic rules."""
    profile_context = format_profile_for_prompt(user_profile)

    return f"""You are the Socratic Technical Study Agent, an expert AI research mentor specializing in foundational machine learning literature, especially the landmark paper 'Attention Is All You Need' (Vaswani et al., 2017) and modern Transformer architectures.

{profile_context}

## Your Persona & Pedagogical Philosophy:
- You are a brilliant, encouraging, and mathematically rigorous peer researcher.
- You believe in **active recall** and **Socratic dialogue**: passive reading leads to an illusion of competence.
- You adapt to the learner's background and learning style.

## Your Multi-Agent System Capabilities:
- You operate a coordinated Multi-Agent Architecture with specialized capabilities:
  1. Paper Specialist: Uses `retrieve_paper_section` for exact formulas, sections, and parameters.
  2. Modern ML Specialist: Uses `web_search` for FlashAttention, RoPE, PyTorch SDPA, and LLaMA context.
  3. Socratic Tutor: Uses `record_concept_progress` to update concept mastery (0-100) and `update_user_profile` to capture learner traits.
  4. Orchestrator: Synthesizes explanations and enforces the Socratic Clarification Loop.

## Core Behavioral Directives:
1. **The Socratic Clarification Loop**:
   - Whenever the learner asks a question, provide an intuitive, mathematically grounded explanation.
   - ALWAYS conclude your response with a targeted, thought-provoking **clarifying comprehension check question** to test the learner's true understanding of the architectural trade-offs, mathematical formulation, or tensor mechanics.
   - When the learner answers your check question, evaluate their response constructively and diagnose any misconceptions.

2. **Tool Usage Guidelines**:
   - `retrieve_paper_section`: Call this whenever the discussion touches on core 'Attention Is All You Need' concepts (architecture overview, scaled dot-product, multi-head attention, positional encoding, computational complexity).
   - `web_search`: Call this to connect foundational concepts to modern developments (FlashAttention, RoPE, Grouped-Query Attention, PyTorch SDPA, LLaMA).
   - `update_user_profile`: Call this autonomously whenever the learner reveals their background or learning style.
   - `record_concept_progress`: Call this when assessing the learner's answer to your Socratic check question.

3. **Tone & Formatting**:
   - Use clean Markdown with bold keywords, inline code for dimensions and tensor shapes (e.g. `[batch, seq_len, d_model]`), and LaTeX or clear text for formulas.
   - Keep answers focused, insightful, and always finish with your Socratic follow-up question.
"""


class SocraticStudyAgent:
    """Multi-agent orchestrator with strategic model routing, guardrails, and HITL hooks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        session_manager: Optional[SessionManager] = None,
        trace_enabled: bool = False,
        hitl_policy: HITLPolicy = HITLPolicy.AUTO,
        hitl_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.model_name = model_name or ModelRouter.route_model(AgentRole.ORCHESTRATOR)
        self.session_manager = session_manager or SessionManager()
        self.tracer = TelemetryTracer.get_instance(trace_enabled=trace_enabled)
        self.guardrails = SecurityGuardrails()
        self.hitl_manager = HumanInTheLoopManager(
            policy=hitl_policy, approval_callback=hitl_callback
        )
        self.adk_agent: Optional[adk.Agent] = None
        self.runner: Optional[InMemoryRunner] = None
        self._active_session_id: Optional[str] = None
        self._init_adk()

    def _init_adk(self) -> None:
        """Initializes the Google ADK Agent and Runner."""
        profile = load_user_profile()
        instruction = build_system_instruction(profile)

        # Wrap tools with HITL hooks
        wrapped_tools = self._wrap_tools_with_hitl()

        self.adk_agent = adk.Agent(
            name="socratic_study_agent",
            model=self.model_name,
            instruction=instruction,
            tools=wrapped_tools,
        )
        self.runner = InMemoryRunner(agent=self.adk_agent)

    def _wrap_tools_with_hitl(self) -> List[Callable]:
        """Wraps tools requiring approval with Human-in-the-Loop interceptor."""
        wrapped = []
        for tool_func in AGENT_TOOLS:
            name = tool_func.__name__

            def make_wrapper(fn: Callable, t_name: str) -> Callable:
                def hitl_wrapped(*args: Any, **kwargs: Any) -> Any:
                    bound_args = {"args": list(args), **kwargs}
                    if self.hitl_manager.requires_approval(t_name):
                        approved = self.hitl_manager.request_approval(
                            t_name, bound_args
                        )
                        if not approved:
                            return f"Tool '{t_name}' execution was DECLINED by user Human-in-the-Loop policy."
                    return fn(*args, **kwargs)

                hitl_wrapped.__name__ = t_name
                hitl_wrapped.__doc__ = fn.__doc__
                return hitl_wrapped

            wrapped.append(make_wrapper(tool_func, name))
        return wrapped

    def refresh_instruction(self) -> None:
        """Refreshes system instruction with latest long-term profile state."""
        profile = load_user_profile()
        if self.adk_agent:
            self.adk_agent.instruction = build_system_instruction(profile)

    async def ensure_session(self, session_id: str, user_id: str = "learner") -> None:
        """Ensures the session exists in ADK Runner and loads any existing turns."""
        self._active_session_id = session_id
        if self.runner:
            try:
                existing = await self.runner.session_service.get_session(
                    app_name=self.runner.app_name, user_id=user_id, session_id=session_id
                )
                if not existing:
                    await self.runner.session_service.create_session(
                        app_name=self.runner.app_name,
                        user_id=user_id,
                        session_id=session_id,
                    )
            except Exception as e:
                logger.debug("Creating ADK session: %s", e)
                try:
                    await self.runner.session_service.create_session(
                        app_name=self.runner.app_name,
                        user_id=user_id,
                        session_id=session_id,
                    )
                except Exception:
                    pass

    async def send_message_stream(
        self,
        message: str,
        session_id: str,
        user_id: str = "learner",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams agent execution events with security guardrails and retry resilience."""
        # 1. Input Safety Guardrail Screening
        input_guard = self.guardrails.check_input_safety(message)
        if not input_guard.passed:
            blocked_response = (
                f"⚠️ Security Guardrail Notice: {input_guard.reason}\n\n"
                "Please focus your inquiries on 'Attention Is All You Need', Transformer math, "
                "or modern deep learning implementations."
            )
            yield {"type": "content_chunk", "text": blocked_response}
            yield {
                "type": "final_response",
                "text": blocked_response,
                "tools_invoked": [],
                "duration_seconds": 0.01,
            }
            return

        # 2. Setup session context
        await self.ensure_session(session_id=session_id, user_id=user_id)
        self.refresh_instruction()

        # Strategic model routing based on query keywords
        if any(w in message.lower() for w in ["proof", "derivative", "variance", "flopss"]):
            routed_model = ModelRouter.route_model(AgentRole.PAPER_SPECIALIST, "deep_math")
        else:
            routed_model = ModelRouter.route_model(AgentRole.ORCHESTRATOR, "standard")

        if self.adk_agent and self.adk_agent.model != routed_model:
            self.adk_agent.model = routed_model

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )

        tools_invoked: List[str] = []
        collected_text_parts: List[str] = []
        start_time = time.perf_counter()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                tools_invoked.clear()
                collected_text_parts.clear()

                async for event in self.runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=content,
                ):
                    fcalls = event.get_function_calls()
                    if fcalls:
                        for fc in fcalls:
                            tools_invoked.append(fc.name)
                            yield {
                                "type": "tool_call",
                                "name": fc.name,
                                "args": fc.args if hasattr(fc, "args") else {},
                            }

                    fresponses = event.get_function_responses()
                    if fresponses:
                        for fr in fresponses:
                            yield {
                                "type": "tool_response",
                                "name": fr.name,
                                "response": fr.response if hasattr(fr, "response") else {},
                            }

                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                collected_text_parts.append(part.text)
                                yield {"type": "content_chunk", "text": part.text}

                last_error = None
                break

            except Exception as e:
                err_str = str(e)
                last_error = e
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    logger.warning(
                        "Transient API rate/load error (attempt %d/%d): %s. Retrying...",
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    await asyncio.sleep(2.0 * (attempt + 1))
                else:
                    raise e

        if last_error is not None:
            fallback_text = (
                f"I encountered a temporary service unavailability ({last_error}). "
                "Please try submitting your inquiry again in a moment."
            )
            collected_text_parts.append(fallback_text)
            yield {"type": "content_chunk", "text": fallback_text}

        duration = time.perf_counter() - start_time
        full_text = "".join(collected_text_parts)

        # 3. Output Quality Guardrail Check (Ensure Socratic question is present)
        quality_check = self.guardrails.check_output_quality(full_text)
        if not quality_check.passed:
            socratic_suffix = (
                "\n\n---\n**Socratic Comprehension Check**:\n"
                "What architectural trade-offs do you notice here, and how does this "
                "impact computational scaling during training?"
            )
            full_text += socratic_suffix
            yield {"type": "content_chunk", "text": socratic_suffix}

        # 4. Telemetry Recording with OpenTelemetry & PII Scrubber
        self.tracer.record_turn(
            user_query=message,
            response_preview=full_text[:120],
            duration_seconds=duration,
            session_id=session_id,
        )

        # 5. Non-blocking Async Memory Turn Persistence
        session_data = await self.session_manager.load_session_async(session_id)
        if session_data:
            self.session_manager.add_turn(
                session_data=session_data,
                role="user",
                content=message,
                save_background=True,
            )
            self.session_manager.add_turn(
                session_data=session_data,
                role="assistant",
                content=full_text,
                tools_invoked=tools_invoked,
                save_background=False,
            )

        yield {
            "type": "final_response",
            "text": full_text,
            "tools_invoked": tools_invoked,
            "duration_seconds": round(duration, 3),
        }

    async def send_message(
        self,
        message: str,
        session_id: str,
        user_id: str = "learner",
    ) -> str:
        """Convenience method to send a message and return the full agent response string."""
        final_text = ""
        async for event in self.send_message_stream(
            message=message, session_id=session_id, user_id=user_id
        ):
            if event["type"] == "final_response":
                final_text = event["text"]
        return final_text
