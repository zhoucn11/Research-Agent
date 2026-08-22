import os
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.core.llm_clients import get_qwen_llm, safe_llm_invoke


class RetrievalQueryPlan(BaseModel):
    original_query: str = Field(default="")
    semantic_queries: List[str] = Field(default_factory=list)
    keyword_queries: List[str] = Field(default_factory=list)
    hyde_query: str = Field(default="")
    query_intent: str = Field(default="general")


def _dedupe_keep_order(items: list[str], limit: int = 8) -> list[str]:
    seen = set()
    output = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item or "")).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def should_skip_query_rewrite(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    if q.upper() == "SUMMARY_ALL":
        return True
    if q.lower().endswith(".pdf"):
        return True
    return False


def fallback_query_plan(user_query: str, last_human_msg: str = "") -> RetrievalQueryPlan:
    base_queries = _dedupe_keep_order([user_query, last_human_msg], limit=3)
    return RetrievalQueryPlan(
        original_query=user_query or last_human_msg,
        semantic_queries=base_queries,
        keyword_queries=base_queries,
        hyde_query=user_query or last_human_msg,
        query_intent="fallback",
    )


async def rewrite_retrieval_query(
    user_query: str,
    last_human_msg: str = "",
    memory_context: str = "",
) -> RetrievalQueryPlan:
    if should_skip_query_rewrite(user_query):
        return fallback_query_plan(user_query, last_human_msg)

    llm = get_qwen_llm(
        temperature=0.0,
        max_tokens=int(os.environ.get("QUERY_REWRITE_MAX_OUTPUT_TOKENS", "700")),
    )
    structured_llm = llm.with_structured_output(RetrievalQueryPlan, method="json_mode")
    sys_msg = SystemMessage(content="You rewrite academic RAG search queries. Return strict JSON only.")
    user_msg = HumanMessage(
        content=f"""Rewrite the query for local academic paper retrieval.

Rules:
1. Preserve named methods, datasets, model names, file names, metrics, and years.
2. Do not answer the question.
3. Produce short semantic queries for dense retrieval.
4. Produce compact keyword queries for BM25 retrieval.
5. Produce one hypothetical evidence paragraph for HyDE-style dense retrieval.
6. Use Chinese if the user asks in Chinese, but keep technical terms in English.

User retrieval query:
{user_query}

Full latest user message:
{last_human_msg}

Persistent user/project memory:
{memory_context or "None"}

JSON schema:
{{
  "original_query": "...",
  "semantic_queries": ["...", "..."],
  "keyword_queries": ["...", "..."],
  "hyde_query": "...",
  "query_intent": "summary|method|experiment|limitation|comparison|metadata|general"
}}
"""
    )

    plan = await safe_llm_invoke(structured_llm, [sys_msg, user_msg], "Query_Rewrite", max_retries=2)
    if not plan:
        return fallback_query_plan(user_query, last_human_msg)

    plan.original_query = plan.original_query or user_query
    plan.semantic_queries = _dedupe_keep_order([user_query, *plan.semantic_queries, last_human_msg], limit=8)
    plan.keyword_queries = _dedupe_keep_order([user_query, *plan.keyword_queries], limit=8)
    plan.hyde_query = re.sub(r"\s+", " ", str(plan.hyde_query or user_query)).strip()
    return plan


def plan_to_retrieval_queries(plan: RetrievalQueryPlan) -> list[str]:
    return _dedupe_keep_order(
        [
            plan.original_query,
            *plan.semantic_queries,
            *plan.keyword_queries,
            plan.hyde_query,
            "Title Abstract Contribution Method",
            "Experiment Results Evaluation Conclusion",
            "Limitations Discussion Future Work",
        ],
        limit=12,
    )
