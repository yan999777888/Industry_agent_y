from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.runtime_checks import StartupHealthReport, assert_startup_ready, run_startup_checks


class RuntimeChecksTests(unittest.TestCase):
    def test_startup_checks_detect_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_startup_checks(
                base_url="http://127.0.0.1:9",
                model="missing-model",
                vision_model="",
                processed_dir=Path(tmpdir),
            )
        self.assertEqual(report.status, "degraded")
        self.assertTrue(any(item.name == "index.sqlite" and not item.ok for item in report.components))

    def test_assert_startup_ready_raises_on_required_failure(self) -> None:
        report = StartupHealthReport(
            status="degraded",
            components=[],
        )
        report.components.append(
            type("ComponentStatusLike", (), {"name": "index.sqlite", "ok": False, "detail": "missing", "required": True})()
        )
        with self.assertRaises(RuntimeError):
            assert_startup_ready(report)

    def test_startup_checks_cloud_backend_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.sqlite"
            index_path.write_text("", encoding="utf-8")
            report = run_startup_checks(
                base_url="https://api.example.com/v1",
                model="demo-model",
                vision_model="",
                llm_backend="openai_compatible",
                api_key="",
                processed_dir=Path(tmpdir),
            )

        self.assertEqual(report.status, "degraded")
        self.assertTrue(any(item.name == "llm_api_key" and not item.ok for item in report.components))
