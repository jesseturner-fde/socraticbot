"""Agent orchestrator and Socratic conversational engine.

Built on Google ADK 2.0 and Gemini. Implements the Socratic Clarification Loop:
1. Grounded technical explanation using 'retrieve_paper_section' and 'web_search'.
2. Autonomous profile reflection via 'update_user_profile'.
3. Concept diagnosis via 'record_concept_progress'.
4. Targeted Socratic clarifying follow-up question to test active recall.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from dotenv import load_dotenv

import google.adk as adk
from google.adk.runners import InMemoryRunner
from google.genai import types

from studyagent.memory import (
    SessionManager,
    format_profile_for_prompt,
    load_user_profile,
)
from studyagent.telemetry import TelemetryTracer
from studyagent.tools import AGENT_TOOLS

# Ensure environment variables are loaded
load_dotenv()

# Synchronize API keys for google-genai and google-adk
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def build_system_instruction(user_profile: Optional[Dict[str, Any]] = None) -> str:
    """Builds the dynamic system instruction injecting learner profile and Socratic rules."""
    profile_context = format_profile_for_prompt(user_profile)

    return f"""You are the Socratic Technical Study Agent, an expert AI research mentor specializing in foundational machine learning literature, especially the landmark paper 'Attention Is All You Need' (Vaswani et al., 2017) and modern Transformer architectures.

{profile_context}

## Your Persona & Pedagogical Philosophy:
- You are a brilliant, encouraging, and mathematically rigorous peer researcher.
- You believe in **active recall** and **Socratic dialogue**: passive reading leads to an illusion of competence.
- You adapt to the learner's background and learning style indicated above.

## Core Behavioral Directives:
1. **The Socratic Clarification Loop**:
   - Whenever the learner asks a question, first provide a crystal-clear, intuitive, and mathematically grounded explanation.
   - ALWAYS conclude your response with a targeted, thought-provoking **clarifying comprehension check question** to test the learner's true understanding of the architectural trade-offs, mathematical formulation, or tensor mechanics.
   - When the learner answers your check question, evaluate their response constructively. If they possess a misconception, diagnose it kindly.

2. **Tool Usage Guidelines**:
   - `retrieve_paper_section`: Call this whenever the discussion touches on core 'Attention Is All You Need' concepts (architecture overview, scaled dot-product, multi-head attention, positional encoding, computational complexity) to cite exact formulas and paper parameters.
   - `web_search`: Call this to connect foundational concepts to modern developments (FlashAttention, RoPE, Grouped-Query Attention, PyTorch SDPA, LLaMA).
   - `update_user_profile`: Call this autonomously whenever the learner reveals their technical background, preferred learning style (e.g. 'I prefer PyTorch code over math'), or goals.
   - `record_concept_progress`: Call this when assessing the learner's answer to your Socratic check question to update their mastery score (0-100) and log any diagnosed misconceptions.

3. **Tone & Formatting**:
   - Use clean Markdown with bold keywords, inline code for dimensions and tensor shapes (e.g. `[batch, seq_len, d_model]`), and LaTeX or clear text for formulas.
   - Keep answers focused, insightful, and always finish with your Socratic follow-up question.
"""


class SocraticStudyAgent:
    """Orchestrator for the Socratic Study Agent conversational workflow."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        session_manager: Optional[SessionManager] = None,
        trace_enabled: bool = False,
    ):
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.session_manager = session_manager or SessionManager()
        self.tracer = TelemetryTracer.get_instance(trace_enabled=trace_enabled)
        self.adk_agent: Optional[adk.Agent] = None
        self.runner: Optional[InMemoryRunner] = None
        self._active_session_id: Optional[str] = None
        self._init_adk()

    def _init_adk(self) -> None:
        """Initializes the Google ADK Agent and Runner."""
        profile = load_user_profile()
        instruction = build_system_instruction(profile)

        self.adk_agent = adk.Agent(
            name="socratic_study_agent",
            model=self.model_name,
            instruction=instruction,
            tools=AGENT_TOOLS,
        )
        self.runner = InMemoryRunner(agent=self.adk_agent)

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
                # Check if session exists in runner's session service
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
        """Streams agent execution events for a user message with automatic retry for spikes.

        Yields dictionaries with event types:
        - {"type": "tool_call", "name": str, "args": dict}
        - {"type": "tool_response", "name": str, "response": any}
        - {"type": "content_chunk", "text": str}
        - {"type": "final_response", "text": str, "tools_invoked": list[str]}
        """
        await self.ensure_session(session_id=session_id, user_id=user_id)
        self.refresh_instruction()

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
                    # Check for tool function calls
                    fcalls = event.get_function_calls()
                    if fcalls:
                        for fc in fcalls:
                            tools_invoked.append(fc.name)
                            yield {
                                "type": "tool_call",
                                "name": fc.name,
                                "args": fc.args if hasattr(fc, "args") else {},
                            }

                    # Check for tool responses
                    fresponses = event.get_function_responses()
                    if fresponses:
                        for fr in fresponses:
                            yield {
                                "type": "tool_response",
                                "name": fr.name,
                                "response": fr.response if hasattr(fr, "response") else {},
                            }

                    # Check for text chunks
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                collected_text_parts.append(part.text)
                                yield {"type": "content_chunk", "text": part.text}

                # Successfully finished iteration
                last_error = None
                break

            except Exception as e:
                err_str = str(e)
                last_error = e
                # Check for transient 503 or 429 errors
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

        # Record to telemetry
        self.tracer.record_turn(
            user_query=message,
            response_preview=full_text[:120],
            duration_seconds=duration,
        )

        # Update session manager
        session_data = self.session_manager.load_session(session_id)
        if session_data:
            self.session_manager.add_turn(
                session_data=session_data,
                role="user",
                content=message,
            )
            self.session_manager.add_turn(
                session_data=session_data,
                role="assistant",
                content=full_text,
                tools_invoked=tools_invoked,
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
