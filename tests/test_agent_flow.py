from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.service import ChatRequest, CustomerServiceAgent
from industry_agent.rag.retriever import SQLiteRetriever


class RetrieverFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = SQLiteRetriever()
        cls.agent = CustomerServiceAgent(cls.retriever)

    def test_drill_indicator_query_hits_drill_manual(self) -> None:
        response = self.retriever.retrieve("我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？", limit=3)
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].product_name, "电钻")
        self.assertIn("DCB107", response.results[0].title)

    def test_band_size_query_hits_fitness_tracker(self) -> None:
        response = self.retriever.retrieve("我想更换健身追踪器的表带，有其他尺寸可选吗？", limit=3)
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].product_name, "健身追踪器")
        joined_titles = " ".join(item.title for item in response.results)
        self.assertIn("表带", joined_titles)

    def test_agent_returns_grounded_answer(self) -> None:
        response = self.agent.chat(ChatRequest(question="我想更换健身追踪器的表带，有其他尺寸可选吗？"))
        self.assertTrue(response.answer)
        self.assertGreater(response.confidence, 0.3)
        self.assertTrue(response.sources)

    def test_agent_handles_unknown_query(self) -> None:
        response = self.agent.chat(ChatRequest(question="请问这台设备能不能直接连接火星基地网络？"))
        self.assertTrue(response.answer)
        self.assertLess(response.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
