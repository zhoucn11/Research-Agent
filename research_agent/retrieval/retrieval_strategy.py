from dataclasses import dataclass
import re


VALID_LIGHTRAG_MODES = {"local", "global", "hybrid", "naive", "mix"}


@dataclass(frozen=True)
class RetrievalStrategy:
    mode: str
    reason: str
    enable_rerank: bool
    top_k: int
    chunk_top_k: int
    max_entity_tokens: int
    max_relation_tokens: int
    max_total_tokens: int


def select_retrieval_strategy(
    query: str,
    *,
    target_source_count: int,
    global_summary: bool,
    research_mode: str = "auto",
    configured_mode: str = "auto",
    rerank_available: bool = False,
) -> RetrievalStrategy:
    """用确定性规则选择 LightRAG 模式和预算，避免为路由再调用一次模型。"""
    text = str(query or "").casefold()
    configured_mode = str(configured_mode or "auto").lower()
    research_mode = str(research_mode or "auto").lower()
    is_deep = research_mode == "deep"
    comparison = bool(re.search(r"对比|比较|区别|差异|异同|compare|versus|\bvs\.?\b", text))
    relationship = bool(re.search(r"关系|演进|影响|关联|机制|路线|relationship|evolution", text))
    metadata = bool(re.search(r"作者|年份|doi|标题|期刊|会议|author|year|venue", text))

    if configured_mode in VALID_LIGHTRAG_MODES:
        mode = configured_mode
        reason = f"部署配置固定为 {mode}"
    elif global_summary:
        mode = "global"
        reason = "全库主题归纳使用全局图谱检索"
    elif target_source_count == 1 and metadata and not comparison:
        mode = "naive"
        reason = "单篇元数据问题优先原文向量块"
    elif target_source_count == 1 and not relationship and not comparison:
        mode = "naive"
        reason = "单篇内容问答优先原文向量块，深度模式只增加证据预算"
    elif relationship and target_source_count <= 1:
        mode = "local"
        reason = "实体关系问题使用局部图谱"
    else:
        mode = "mix"
        reason = "跨论文或开放主题同时使用图谱与向量证据"

    if is_deep:
        top_k = 30
        chunk_top_k = max(8, min(max(1, target_source_count) * 2, 12))
        max_entity_tokens = 1600
        max_relation_tokens = 1800
        max_total_tokens = 5200
    else:
        top_k = 20
        chunk_top_k = max(4, min(max(1, target_source_count), 8))
        max_entity_tokens = 1100
        max_relation_tokens = 1200
        max_total_tokens = 3400

    enable_rerank = bool(rerank_available and is_deep and mode == "mix")
    return RetrievalStrategy(
        mode=mode,
        reason=reason,
        enable_rerank=enable_rerank,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        max_entity_tokens=max_entity_tokens,
        max_relation_tokens=max_relation_tokens,
        max_total_tokens=max_total_tokens,
    )
