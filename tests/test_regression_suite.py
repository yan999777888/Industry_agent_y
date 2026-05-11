from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_regression_suite import build_payload, check_case, load_cases


class RegressionSuiteTests(unittest.TestCase):
    def test_load_cases_returns_list(self) -> None:
        cases = load_cases(PROJECT_ROOT / "tests" / "fixtures" / "regression_cases.json")
        self.assertGreaterEqual(len(cases), 30)

    def test_build_payload_supports_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "tiny.bin"
            image_path.write_bytes(b"abc")
            payload = build_payload(
                {
                    "question": "test",
                    "image_paths": [str(image_path)],
                }
            )
        self.assertEqual(payload["question"], "test")
        self.assertEqual(len(payload["images"]), 1)
        self.assertEqual(payload["images"][0], base64.b64encode(b"abc").decode("utf-8"))

    def test_check_case_reports_failures(self) -> None:
        case = {
            "question": "test",
            "expect_contains": ["结论"],
            "expect_sources_contains": ["customer_service_policy"],
            "min_image_ids": 1,
        }
        response = {
            "code": 0,
            "data": {
                "answer": "普通回答",
                "sources": [],
                "image_ids": [],
            },
        }
        ok, failures = check_case(case, response)
        self.assertFalse(ok)
        self.assertGreaterEqual(len(failures), 3)

    def test_check_case_supports_http_status_expectations(self) -> None:
        case = {
            "question": "",
            "expect_http_status": 400,
            "expect_error_contains": ["question must not be empty"],
        }
        response = {
            "_http_status": 400,
            "detail": "question must not be empty",
        }
        ok, failures = check_case(case, response)
        self.assertTrue(ok, msg=str(failures))

    def test_check_case_supports_retrieval_debug_expectations(self) -> None:
        case = {
            "question": "物流一直显示待揽收，是什么原因？",
            "expect_debug_equals": {
                "sub_results.0.retrieval_debug.route_decision.route": "customer_service",
            },
            "expect_debug_contains": {
                "sub_results.0.retrieval_debug.customer_service_kb.hit_source_types": ["data_file"],
            },
        }
        response = {
            "_http_status": 200,
            "code": 0,
            "data": {
                "answer": "物流显示待揽收，一般表示正在等待快递员揽件。",
                "sources": ["customer_service_policy", "customer_service_kb"],
                "image_ids": [],
                "retrieval_debug": {
                    "sub_results": [
                        {
                            "retrieval_debug": {
                                "route_decision": {"route": "customer_service"},
                                "customer_service_kb": {"hit_source_types": ["data_file"]},
                            }
                        }
                    ]
                },
            },
        }

        ok, failures = check_case(case, response)
        self.assertTrue(ok, msg=str(failures))


if __name__ == "__main__":
    unittest.main()
