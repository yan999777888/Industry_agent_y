"""LLM-based query expansion for improved retrieval recall.

Before retrieval, uses the LLM to generate related terms, synonyms,
and context-aware rewrites of the original query.
Controlled by INDUSTRY_AGENT_ENABLE_QUERY_EXPANSION.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from industry_agent.llm.client import LLMClient

logger = logging.getLogger(__name__)

QUERY_EXPANSION_PROMPT = """\
Add 3-5 relevant search keywords for this query. Return ONLY a JSON object:

{query}

Format: {{"expanded_terms":["kw1","kw2","kw3"],"rewritten_query":"more specific rewrite"}}

Keep same language as query. Do not fabricate terms.
"""


class QueryExpander:
    """Expands user queries using the LLM for better retrieval recall."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def expand(self, query: str) -> dict[str, Any]:
        """Expand a query using the LLM.

        Returns:
            dict with:
            - original_query: str
            - expanded_terms: list[str]
            - rewritten_query: str | None
            - queries: list[str] (up to 3 query variants to search with)
        """
        if len(query.strip()) < 4:
            return {"original_query": query, "expanded_terms": [], "rewritten_query": None, "queries": [query]}

        try:
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            parsed = self._parse_response(response)
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            parsed = {}

        expanded_terms = parsed.get("expanded_terms", [])
        rewritten_query = parsed.get("rewritten_query", "")

        queries = [query]
        if rewritten_query and rewritten_query != query:
            queries.append(rewritten_query)
        if expanded_terms:
            fused = f"{query} {' '.join(expanded_terms[:5])}"
            if fused != query:
                queries.append(fused)

        return {
            "original_query": query,
            "expanded_terms": expanded_terms[:10],
            "rewritten_query": rewritten_query,
            "queries": queries[:3],
        }

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling various formats."""
        if not response or not response.strip():
            return {}  # Empty response, silently ignore
        for pattern in [
            r"```(?:json)?\s*\n(.*?)\n```",
            r"```(?:json)?\s*(.*?)\s*```",
        ]:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try raw response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Fallback: extract expanded_terms and rewritten_query via regex
        result: dict[str, Any] = {}
        terms_match = re.search(r'"expanded_terms"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if terms_match:
            terms_text = terms_match.group(1)
            terms = re.findall(r'"([^"]+)"', terms_text)
            result["expanded_terms"] = terms
        rewritten_match = re.search(r'"rewritten_query"\s*:\s*"([^"]+)"', response)
        if rewritten_match:
            result["rewritten_query"] = rewritten_match.group(1)
        if result:
            return result

        logger.warning("Could not parse LLM query expansion response: %s", response[:200])
        return {}


def _suppress_qe_warning() -> None:
    """Silence noisy query expansion parse warnings in production."""
    logging.getLogger("industry_agent.rag.query_expansion").setLevel(logging.ERROR)
