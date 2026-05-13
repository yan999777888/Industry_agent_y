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
You are a product customer service search specialist. Expand the user's query to improve search recall.

Original query: {query}

Generate:
1. **expanded_terms**: Key product names, model numbers, issue keywords, synonyms, and related terms
2. **rewritten_query**: A search-oriented rewrite of the query with more technical/specific terms

Return ONLY JSON:
```json
{{
  "expanded_terms": ["term1", "term2", ...],
  "rewritten_query": "rewritten search query"
}}
```

Rules:
- Do NOT fabricate product models or terms not implied by the query
- Expanded terms must be highly relevant to the original query
- For English queries, keep expanded terms in English
- For Chinese queries, keep expanded terms in Chinese
- If the query is already clear and specific, minimal expansion is fine
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
        """Parse JSON from LLM response, handling markdown code blocks."""
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM query expansion response: %s", response[:200])
            return {}
