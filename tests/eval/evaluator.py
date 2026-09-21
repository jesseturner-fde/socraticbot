"""Automated evaluation framework for measuring agent regressions against the golden dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from studyagent.guardrails import SecurityGuardrails
from studyagent.tools import retrieve_paper_section, web_search

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


class AgentEvaluator:
    """Evaluates agent responses against the curated Golden Benchmark."""

    def __init__(self, dataset_path: Optional[Path | str] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else GOLDEN_DATASET_PATH
        self.benchmark_data = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_response(
        self,
        test_case: Dict[str, Any],
        response_text: str,
        tools_invoked: List[str],
    ) -> Dict[str, Any]:
        """Evaluates a single response against test case criteria."""
        case_id = test_case["id"]
        is_blocked_expected = test_case["expected_security_blocked"]

        # 1. Guardrail / Blocking Check
        guardrail_passed = True
        if is_blocked_expected:
            is_blocked_actual = (
                "security guardrail" in response_text.lower()
                or "domain" in response_text.lower()
                or len(tools_invoked) == 0
            )
            guardrail_passed = is_blocked_actual
        else:
            is_blocked_actual = "security guardrail notice" in response_text.lower()
            guardrail_passed = not is_blocked_actual

        # 2. Tool Routing Precision
        expected_tools = set(test_case.get("expected_tools", []))
        actual_tools = set(tools_invoked)
        tool_match = (
            expected_tools.issubset(actual_tools)
            if expected_tools
            else len(actual_tools) == 0
        )

        # 3. Grounding & Keyword Recall
        expected_keywords = test_case.get("expected_keywords", [])
        found_keywords = [
            kw for kw in expected_keywords if kw.lower() in response_text.lower()
        ]
        keyword_recall = (
            len(found_keywords) / len(expected_keywords)
            if expected_keywords
            else 1.0
        )
        grounding_passed = keyword_recall >= 0.5

        # 4. Socratic Question Adherence
        expected_socratic = test_case.get("expected_socratic", True)
        if expected_socratic:
            has_question = "?" in response_text
            has_socratic_prompt = any(
                kw in response_text.lower()
                for kw in ["what", "how", "why", "socratic", "think", "trade-off"]
            )
            socratic_passed = has_question and has_socratic_prompt
        else:
            socratic_passed = True

        overall_passed = guardrail_passed and grounding_passed and socratic_passed

        return {
            "case_id": case_id,
            "category": test_case["category"],
            "overall_passed": overall_passed,
            "guardrail_passed": guardrail_passed,
            "grounding_passed": grounding_passed,
            "keyword_recall": round(keyword_recall, 2),
            "socratic_passed": socratic_passed,
            "tool_match": tool_match,
            "tools_invoked": tools_invoked,
        }

    def run_benchmark(
        self,
        predict_fn: Callable[[str], Tuple[str, List[str]]],
    ) -> Dict[str, Any]:
        """Runs the entire golden dataset through the agent prediction function.

        Args:
            predict_fn: Callable accepting a query string and returning (response_text, tools_invoked).

        Returns:
            Benchmark report with overall pass rate, category breakdown, and detailed results.
        """
        results = []
        cases = self.benchmark_data.get("test_cases", [])

        for case in cases:
            query = case["query"]
            response_text, tools_invoked = predict_fn(query)
            eval_res = self.evaluate_response(case, response_text, tools_invoked)
            results.append(eval_res)

        total = len(results)
        passed = sum(1 for r in results if r["overall_passed"])
        pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0

        avg_keyword_recall = (
            round(sum(r["keyword_recall"] for r in results) / total, 2)
            if total > 0
            else 0.0
        )
        socratic_rate = (
            round(sum(1 for r in results if r["socratic_passed"]) / total * 100, 1)
            if total > 0
            else 0.0
        )

        return {
            "benchmark_name": self.benchmark_data.get("benchmark_name"),
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate_percentage": pass_rate,
            "average_keyword_recall": avg_keyword_recall,
            "socratic_adherence_rate": socratic_rate,
            "case_results": results,
        }
