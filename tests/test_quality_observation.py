from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.observe_chat_quality import evaluate_case, load_cases, summarize_records


class QualityObservationTests(unittest.TestCase):
    def test_load_cases_returns_non_empty_list(self) -> None:
        cases = load_cases(PROJECT_ROOT / "tests" / "fixtures" / "quality_observation_cases.json")
        self.assertGreaterEqual(len(cases), 13)

    def test_evaluate_case_reports_issue_buckets(self) -> None:
        case = {
            "id": "demo",
            "category": "customer_service",
            "question": "demo question",
            "expect_contains": ["质量问题"],
            "expect_sources_contains": ["customer_service_policy"],
            "min_confidence": 0.7,
            "min_image_ids": 1,
        }
        response = {
            "_http_status": 200,
            "code": 0,
            "data": {
                "answer": "普通回答",
                "sources": [],
                "image_ids": [],
                "confidence": 0.55,
            },
        }

        record = evaluate_case(case, response)
        self.assertFalse(record["ok"])
        self.assertIn("answer_alignment", record["issues"])
        self.assertIn("source_routing", record["issues"])
        self.assertIn("image_binding", record["issues"])
        self.assertIn("low_confidence", record["issues"])

    def test_evaluate_case_supports_error_detail_expectations(self) -> None:
        case = {
            "id": "bad_request",
            "category": "api_error",
            "question": " ",
            "expect_http_status": 400,
            "expect_error_contains": ["question must not be empty"],
        }
        response = {
            "_http_status": 400,
            "detail": "question must not be empty",
        }

        record = evaluate_case(case, response)
        self.assertTrue(record["ok"])

    def test_summarize_records_groups_categories_and_issues(self) -> None:
        summary = summarize_records(
            [
                {"category": "manual_rag", "ok": True, "issues": []},
                {"category": "manual_rag", "ok": False, "issues": ["answer_alignment"]},
                {"category": "customer_service", "ok": False, "issues": ["source_routing", "low_confidence"]},
            ]
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["categories"]["manual_rag"]["total"], 2)
        self.assertEqual(summary["categories"]["manual_rag"]["passed"], 1)
        self.assertEqual(summary["issue_buckets"]["answer_alignment"], 1)
        self.assertEqual(summary["issue_buckets"]["source_routing"], 1)


if __name__ == "__main__":
    unittest.main()
