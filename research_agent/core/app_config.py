"""Application defaults shared by all Research Agent entry points.

Secrets, service URLs, model names and machine-specific paths remain deployment
environment variables.  Stable algorithm budgets and resilience settings live
here so a fresh checkout has one reproducible configuration baseline.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


CODE_DEFAULTS: Mapping[str, str] = {
    # Main/reviewer API behavior.
    "QWEN_ENABLE_THINKING": "true",
    "LLM_TIMEOUT": "300",
    "REVIEWER_TIMEOUT": "300",
    "LOCAL_LLM_TIMEOUT": "180",
    "SYNTHESIS_TIMEOUT": "300",
    "MAIN_API_MAX_CONCURRENCY": "2",
    "REVIEWER_API_MAX_CONCURRENCY": "1",
    "LOCAL_LLM_MAX_CONCURRENCY": "1",
    "LLM_MAX_CONCURRENCY": "1",
    "ASSISTANT_LLM_MAX_RETRIES": "3",
    "LLM_RETRY_BASE_SECONDS": "1",
    "LLM_CIRCUIT_FAILURE_THRESHOLD": "3",
    "LLM_CIRCUIT_COOLDOWN_SECONDS": "60",
    "REVIEWER_LLM_ENABLED": "true",
    # Context and memory budgets.
    "MAIN_API_CONTEXT_WINDOW": "262144",
    "CONTEXT_MAX_TOKENS": "196608",
    "CONTEXT_RESERVED_TOKENS": "16000",
    "MEMORY_CONTENT_LIMIT": "2000",
    "MEMORY_SUMMARY_LIMIT": "12000",
    "MEMORY_SUMMARY_INPUT_LIMIT": "120000",
    "CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS": "1600",
    "QWEN_MAX_OUTPUT_TOKENS": "1600",
    # LightRAG indexing/query defaults sized for the current 16K local model.
    "LIGHTRAG_INDEX_VERSION": "paper_graph_v1",
    "LIGHTRAG_EMBEDDING_DIM": "1024",
    "LIGHTRAG_EMBEDDING_MAX_TOKENS": "8192",
    "LIGHTRAG_LLM_MAX_ASYNC": "1",
    "LIGHTRAG_EMBEDDING_MAX_ASYNC": "1",
    "LIGHTRAG_EMBEDDING_BATCH_SIZE": "1",
    "LIGHTRAG_MAX_PARALLEL_INSERT": "1",
    "LIGHTRAG_LLM_MAX_RETRIES": "2",
    "LIGHTRAG_CHUNK_SIZE": "600",
    "LIGHTRAG_CHUNK_OVERLAP": "80",
    "LIGHTRAG_MAX_SOURCE_FILES": "8",
    "LIGHTRAG_LOCAL_RERANK_ENABLED": "false",
    "LIGHTRAG_QUERY_MODE": "auto",
    "LIGHTRAG_TOP_K": "30",
    "LIGHTRAG_MAX_GLEANING": "0",
    "LIGHTRAG_MAX_EXTRACTION_RECORDS": "30",
    "LIGHTRAG_MAX_EXTRACTION_ENTITIES": "20",
    "LIGHTRAG_SUMMARY_CONTEXT_SIZE": "6000",
    "LIGHTRAG_ENTITY_EXTRACTION_USE_JSON": "true",
    "LIGHTRAG_MAX_TOTAL_TOKENS": "5500",
    "LIGHTRAG_MAX_ENTITY_TOKENS": "1400",
    "LIGHTRAG_MAX_RELATION_TOKENS": "1600",
    "LIGHTRAG_CHUNK_TOP_K": "8",
    "LIGHTRAG_KEYWORD_MAX_OUTPUT_TOKENS": "512",
    "LIGHTRAG_LLM_MAX_OUTPUT_TOKENS": "3000",
    "LIGHTRAG_QUERY_TIMEOUT_SECONDS": "30",
    # Local evidence extraction and verification.
    "RAG_HEADER_CONTEXT_CHARS": "3000",
    "RAG_FULLDOC_SUMMARY_CHARS": "6000",
    "RAG_GRAPH_EVIDENCE_CHARS": "8000",
    "RAG_EXTRACTION_CONTEXT_CHARS": "24000",
    "RAG_VERIFY_METADATA_ONLINE": "true",
    "RAG_VERIFY_SUMMARY_METADATA_ONLINE": "false",
    "RAG_METADATA_VERIFY_TIMEOUT_SECONDS": "8",
    "RAG_EVIDENCE_SPANS_PER_SOURCE": "5",
    "RAG_DETAILED_EVIDENCE_SPANS_PER_SOURCE": "8",
    "RAG_COMPARISON_EVIDENCE_SPANS_PER_SOURCE": "6",
    "RAG_EVIDENCE_CANDIDATE_MULTIPLIER": "2",
    # Local model identifiers. Hugging Face resolves them through its standard cache.
    "EMBEDDING_MODEL_PATH": "BAAI/bge-large-zh-v1.5",
    "RERANKER_MODEL_PATH": "BAAI/bge-reranker-v2-m3",
    "RERANKER_DEVICE": "cpu",
    # Output budgets.
    "LOCAL_LLM_MAX_OUTPUT_TOKENS": "2048",
    "ASSISTANT_MAX_OUTPUT_TOKENS": "2048",
    "RAG_FINGERPRINT_MAX_OUTPUT_TOKENS": "256",
    "RAG_EXTRACTION_MAX_OUTPUT_TOKENS": "2800",
    "QUERY_REWRITE_MAX_OUTPUT_TOKENS": "700",
    "WEB_PAPER_SUMMARY_MAX_OUTPUT_TOKENS": "2400",
    "WEB_PAPER_SUMMARY_THINKING_BUDGET": "512",
    "SYNTHESIS_MAX_OUTPUT_TOKENS": "8192",
    "SYNTHESIS_BRIEF_CHAR_LIMIT": "300",
    "REVIEWER_MAX_OUTPUT_TOKENS": "4096",
    "REVIEW_PACKET_MAX_CHARS": "180000",
    # Web enrichment and startup behavior.
    "WEB_PAPER_LLM_ENRICHMENT": "true",
    "WEB_PAPER_LLM_MAX_RETRIES": "2",
    "WEB_SEARCH_DEFAULT_RESULT_LIMIT": "5",
    "S2_MIN_REQUEST_INTERVAL_SECONDS": "1.5",
    "AGENT_PREWARM_LIGHTRAG": "true",
    "AGENT_UPLOAD_MAX_BYTES": str(25 * 1024 * 1024),
    "AGENT_CHECKPOINT_BACKEND": "sqlite",
}


def apply_code_defaults() -> None:
    """Apply code defaults without overriding explicit deployment settings."""

    for name, value in CODE_DEFAULTS.items():
        os.environ.setdefault(name, value)


__all__ = ["CODE_DEFAULTS", "apply_code_defaults"]
