from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_submission_quality import analyze_submission


class SubmissionQualityAnalysisTests(unittest.TestCase):
    def test_analyze_submission_reports_format_risks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            submission = root / "submission.csv"
            debug = root / "debug.jsonl"
            submission.write_text(
                "id,ret\n"
                "1,根据现有资料无法回答此问题。\n"
                "2,请问如何安装？这是一个很长的回答。<PIC>\n",
                encoding="utf-8-sig",
            )
            debug.write_text(
                json.dumps({"id": "1", "question": "这个怎么安装？"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"id": "2", "question": '"能安装吗？","需要注意什么？"'}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            report = analyze_submission(submission, debug_path=debug)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["metrics"]["fallback"], 1)
        self.assertEqual(report["metrics"]["pic_marker"], 1)
        self.assertGreaterEqual(report["metrics"]["question_echo"], 1)
        self.assertTrue(report["risk_summary"])


if __name__ == "__main__":
    unittest.main()
