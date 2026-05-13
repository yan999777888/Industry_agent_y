# Industry Agent - Multi-Modal Customer Service Agent

## Project Overview

A multi-modal customer service intelligent agent for industrial products, built on RAG (Retrieval-Augmented Generation) architecture. The system answers questions based on product manuals (21 manuals, 4132 chunks) with image support, multi-turn dialogue, complex question decomposition, and customer service strategy routing.

## Sub-Agent Architecture

This project uses a **main agent + specialized sub-agents** architecture. The main agent (Claude) coordinates the following specialized agents:

### 1. RAG Agent (`rag_agent`)
**Responsibility**: Retrieval, search, and knowledge retrieval optimization
- Key files: `src/industry_agent/rag/retriever.py`, `src/industry_agent/rag/vector_store.py`, `src/industry_agent/rag/hybrid_retriever.py`, `src/industry_agent/rag/embedding.py`, `src/industry_agent/rag/index_builder.py`, `src/industry_agent/rag/factory.py`
- Scope:
  - SQLite FTS5 + LIKE keyword retrieval
  - Vector search with sentence-transformers embedding
  - Hybrid retrieval with RRF fusion
  - Query analysis, keyword extraction, synonym expansion
  - Evidence scoring and filtering
  - Domain-aware search for English manuals (汇总英文)
  - Product alias resolution

### 2. Agent/Orchestration Agent (`orchestration_agent`)
**Responsibility**: Main orchestration, routing, response generation, and dialogue management
- Key files: `src/industry_agent/agent/service.py`, `src/industry_agent/agent/orchestrator.py`, `src/industry_agent/agent/question_router.py`, `src/industry_agent/agent/question_splitter.py`, `src/industry_agent/agent/context_manager.py`, `src/industry_agent/agent/session_store.py`, `src/industry_agent/agent/prompts.py`, `src/industry_agent/agent/response_formatter.py`
- Scope:
  - Question routing (manual RAG vs customer service)
  - Complex question splitting
  - Multi-turn context inheritance
  - Evidence assembly and context building
  - LLM call orchestration
  - Extractive answer fallback
  - Answer formatting and style normalization
  - Session state management

### 3. Customer Service Agent (`cs_agent`)
**Responsibility**: Customer service policy, knowledge base, and strategy-driven responses
- Key files: `src/industry_agent/agent/customer_service_policy.py`, `src/industry_agent/agent/customer_service_kb.py`, `src/industry_agent/agent/customer_service_kb_data.json`
- Scope:
  - Customer service topic matching and scenario rules
  - Policy-based answer generation
  - Customer service knowledge retrieval
  - Multi-topic and multi-scenario coverage
  - Detail intent detection (materials, timeline, fees, eligibility, process, contact)

### 4. Image Understanding Agent (`image_agent`)
**Responsibility**: Multi-modal image analysis and visual grounding
- Key files: `src/industry_agent/agent/image_understanding.py`, `src/industry_agent/agent/skills/image_skill.py`
- Scope:
  - Base64 image parsing and metadata extraction
  - Visual description via vision models
  - Image-to-retrieval term mapping
  - Visual grounding for manual image selection
  - Component/status/issue term extraction from images

### 5. Knowledge Base Agent (`kb_agent`)
**Responsibility**: Knowledge base construction, parsing, chunking, and indexing
- Key files: `src/industry_agent/kb/parser.py`, `src/industry_agent/kb/chunker.py`, `src/industry_agent/kb/models.py`, `src/industry_agent/kb/build_index.py`, `src/industry_agent/kb/index_store.py`, `scripts/build_kb.py`
- Scope:
  - Manual document parsing (JSON/Python literal format)
  - Text normalization (Unicode, LaTeX, OCR fixes)
  - Image marker attachment (`<PIC>` -> `[[PIC:id]]`)
  - Semantic chunking by section and type
  - English domain detection and segment annotation
  - SQLite index building (FTS5)
  - Chunk quality metrics and warnings

### 6. Evaluation Agent (`eval_agent`)
**Responsibility**: Testing, quality observation, regression, and submission generation
- Key files: `scripts/evaluate_chat.py`, `scripts/run_regression_suite.py`, `scripts/observe_chat_quality.py`, `scripts/generate_submission.py`, `scripts/reprocess_submission.py`, `tests/fixtures/`
- Scope:
  - End-to-end evaluation against `/chat` endpoint
  - Fixed regression suite execution
  - Quality observation with category tags
  - Submission file generation with multi-question structure preservation
  - Answer reprocessing for unanswerable questions

### 7. API Agent (`api_agent`)
**Responsibility**: FastAPI service layer and health checks
- Key files: `src/industry_agent/api/app.py`, `src/industry_agent/agent/runtime_checks.py`, `src/industry_agent/config.py`
- Scope:
  - `/health` endpoint (index, LLM, vision model status)
  - `/chat` endpoint (request validation, response assembly)
  - Startup checks and configuration validation
  - Backend switching (service vs orchestrator)

## Task Dispatch Protocol

When the user issues a task, the main agent should:

1. **Analyze** which sub-agent(s) are relevant
2. **Dispatch** to the appropriate agent(s) with clear context
3. **Coordinate** when tasks span multiple agents
4. **Verify** results before reporting back

### Agent Selection Guide

| Task Type | Primary Agent | Supporting Agent(s) |
|-----------|--------------|---------------------|
| Retrieval quality, search scoring | `rag_agent` | `kb_agent` |
| Answer quality, routing errors | `orchestration_agent` | `cs_agent`, `rag_agent` |
| Customer service responses | `cs_agent` | `orchestration_agent` |
| Image handling, visual grounding | `image_agent` | `rag_agent` |
| Knowledge base rebuild/fix | `kb_agent` | `rag_agent` |
| Testing, evaluation, submission | `eval_agent` | `orchestration_agent` |
| API deployment, config | `api_agent` | `orchestration_agent` |

## Known Issues (from TODO.md)

1. Route over-filters to customer service branch, bypassing retrieval
2. Customer service policy only fetches one topic at a time, missing multi-intent
3. Customer service strategy is rule-based, not true RAG
4. Evidence filtering locks on top_product, may miss cross-product chunks
5. Manual QA answers are over-formatted vs official extractive style
6. Submission cleanup may second-damage answers
7. Image binding is "whatever retrieved" not "answer-grounded"
8. Customer service KB is derived from rules, not an independent data source
9. Retriever evidence filtering still has top_product lock
10. Submission cleanup is "rewrite" not pure "clean"
11. Image pipeline lacks sentence-level grounding
12. Response formatter is too template-like vs "short extract + precise images"
13. Missing comprehensive regression set for multi-intent + multimodal + mixed routing

## Environment

- Python 3.10+
- Default LLM: `mimo-v2.0-flash` (OpenAI-compatible via `api.xiaomimimo.com`)
- Embedding: `BAAI/bge-small-zh-v1.5`
- Retrieval: hybrid (LIKE + FTS5 + optional vector)
- Knowledge base: 21 manuals, 4132 chunks
