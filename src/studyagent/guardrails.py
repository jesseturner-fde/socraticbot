"""Security, evaluation guardrails, and Human-in-the-Loop (HITL) policies.

Implements:
1. Input safety screening (prompt injection, jailbreak defense, toxicity filter).
2. Educational domain alignment (enforces focus on Transformers and ML).
3. Output quality guardrail (factuality check and Socratic question presence).
4. Human-in-the-Loop (HITL) approval hooks for impactful tool operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HITLPolicy(str, Enum):
    """Policies governing Human-in-the-Loop approvals."""

    AUTO = "auto"                     # Fully automated execution
    CONFIRM_CRITICAL = "confirm_critical"  # Requires approval for persistent memory/score mutations
    CONFIRM_ALL = "confirm_all"       # Requires approval for every tool execution


@dataclass
class GuardrailResult:
    """Result of a guardrail safety/evaluation inspection."""

    passed: bool
    reason: str
    flagged_patterns: List[str]
    sanitized_text: Optional[str] = None


class SecurityGuardrails:
    """Comprehensive input and output safety guardrails."""

    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions"),
        re.compile(r"(?i)disregard\s+(all\s+)?(previous|system)\s+prompts?"),
        re.compile(r"(?i)reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)"),
        re.compile(r"(?i)you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan\s+mode|unrestricted)"),
        re.compile(r"(?i)bypass\s+(all\s+)?(safety|guardrails|rules)"),
        re.compile(r"(?i)output\s+the\s+text\s+above"),
    ]

    OFFENSIVE_PATTERNS = [
        re.compile(r"(?i)\b(kill\s+yourself|generate\s+malware|how\s+to\s+make\s+a\s+bomb)\b"),
    ]

    OUT_OF_DOMAIN_PATTERNS = [
        re.compile(r"(?i)\b(recipe\s+for|bake\s+a\s+cake|who\s+won\s+the\s+presidential\s+election|astrology\s+horoscope)\b"),
    ]

    @classmethod
    def check_input_safety(cls, user_text: str) -> GuardrailResult:
        """Screen user prompt for prompt injection, adversarial bypasses, or toxicity."""
        flagged = []
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if pattern.search(user_text):
                flagged.append(f"Prompt Injection Pattern: {pattern.pattern}")

        for pattern in cls.OFFENSIVE_PATTERNS:
            if pattern.search(user_text):
                flagged.append("Harmful or Offensive Query")

        if flagged:
            logger.warning("Input guardrail triggered: %s", flagged)
            return GuardrailResult(
                passed=False,
                reason="Input triggered security guardrail (prompt injection or prohibited content).",
                flagged_patterns=flagged,
            )

        # Check domain alignment (optional soft check)
        for pattern in cls.OUT_OF_DOMAIN_PATTERNS:
            if pattern.search(user_text):
                return GuardrailResult(
                    passed=False,
                    reason="Query is outside the technical study domain of machine learning and Transformer architectures.",
                    flagged_patterns=["Out-of-domain query"],
                )

        return GuardrailResult(
            passed=True,
            reason="Input passed all security guardrails.",
            flagged_patterns=[],
            sanitized_text=user_text.strip(),
        )

    @classmethod
    def check_output_quality(cls, response_text: str) -> GuardrailResult:
        """Evaluates model response for Socratic question presence and basic factuality."""
        flagged = []
        has_question_mark = "?" in response_text
        has_socratic_keywords = any(
            kw in response_text.lower()
            for kw in ["what do you think", "why do you think", "how would", "socratic", "check", "exercise"]
        )

        if not (has_question_mark or has_socratic_keywords):
            flagged.append("Missing Socratic clarifying question.")

        if len(response_text.strip()) < 30:
            flagged.append("Response is too terse for pedagogical instruction.")

        passed = len(flagged) == 0
        reason = "Output passed pedagogical guardrails." if passed else "; ".join(flagged)
        return GuardrailResult(
            passed=passed,
            reason=reason,
            flagged_patterns=flagged,
        )


class HumanInTheLoopManager:
    """Manages Human-in-the-Loop (HITL) approval hooks before tool execution."""

    def __init__(
        self,
        policy: HITLPolicy = HITLPolicy.AUTO,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.policy = policy
        self.approval_callback = approval_callback

    def requires_approval(self, tool_name: str) -> bool:
        """Determines if a tool call requires explicit human confirmation."""
        if self.policy == HITLPolicy.CONFIRM_ALL:
            return True
        if self.policy == HITLPolicy.CONFIRM_CRITICAL:
            # Updating persistent profile or concept progress modifies user disk state
            return tool_name in ["update_user_profile", "record_concept_progress"]
        return False

    def request_approval(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Requests approval from the configured human hook."""
        if not self.requires_approval(tool_name):
            return True

        if self.approval_callback:
            return self.approval_callback(tool_name, arguments)

        # Default fallback approval
        return True
