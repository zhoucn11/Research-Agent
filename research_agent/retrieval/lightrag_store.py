import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from research_agent.core.runtime_events import runtime_print as print

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIGHTRAG_INDEX_VERSION = os.environ.get("LIGHTRAG_INDEX_VERSION", "paper_graph_v1")
_TEXT_CHUNK_CACHE: tuple[tuple[str, int, int], list[dict]] | None = None


def _safe_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "default"


def get_lightrag_working_dir() -> Path:
    configured = os.environ.get("LIGHTRAG_WORKING_DIR")
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "lightrag_storage" / _safe_version(LIGHTRAG_INDEX_VERSION)


def get_manifest_path(working_dir: Path | None = None) -> Path:
    return (working_dir or get_lightrag_working_dir()) / "research_agent_manifest.json"


def load_index_manifest(path: Path | None = None) -> dict:
    path = path or get_manifest_path()
    if not path.exists():
        return {"index_version": LIGHTRAG_INDEX_VERSION, "documents": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"index_version": LIGHTRAG_INDEX_VERSION, "documents": {}}
    if payload.get("index_version") != LIGHTRAG_INDEX_VERSION:
        return {"index_version": LIGHTRAG_INDEX_VERSION, "documents": {}}
    payload.setdefault("documents", {})
    return payload


def list_indexed_sources() -> list[str]:
    return sorted(load_index_manifest().get("documents", {}).keys())


def _normalize_paper_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def find_indexed_source_by_title(title: str, manifest: dict | None = None) -> str:
    """按 manifest 中的正式标题或 PDF 文件名精确定位本地论文，禁止语义近似替代。"""
    target = _normalize_paper_identity(title)
    if not target:
        return ""
    records = (manifest or load_index_manifest()).get("documents", {})
    partial_matches = []
    for source, record in records.items():
        identities = {
            _normalize_paper_identity(Path(source).stem),
            _normalize_paper_identity((record or {}).get("title", "")),
        }
        identities.discard("")
        if target in identities:
            return source
        if len(target) >= 8 and any(target in identity or identity in target for identity in identities):
            partial_matches.append(source)
    return partial_matches[0] if len(partial_matches) == 1 else ""


def load_document_header_context(
    sources: list[str],
    working_dir: Path | None = None,
    per_source_chars: int = 1800,
    total_chars: int = 8000,
) -> str:
    """从 LightRAG 原文 KV 读取论文首页，作为标题、作者和年份的身份锚点。"""
    base_dir = working_dir or get_lightrag_working_dir()
    manifest = load_index_manifest(get_manifest_path(base_dir))
    full_docs_path = base_dir / "kv_store_full_docs.json"
    try:
        full_docs = json.loads(full_docs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    blocks = []
    used_chars = 0
    records = manifest.get("documents", {})
    for source in sources:
        doc_id = (records.get(source) or {}).get("doc_id")
        doc = full_docs.get(doc_id) if doc_id else None
        content = str((doc or {}).get("content", "")).strip()
        if not content:
            continue
        remaining = total_chars - used_chars
        if remaining <= 0:
            break
        header = content[: min(per_source_chars, remaining)].strip()
        block = f"【来源：{source}】\n{header}"
        blocks.append(block)
        used_chars += len(header)
    return "\n\n".join(blocks)


def _query_terms(text: str) -> set[str]:
    text = str(text or "").casefold()
    terms = {
        term for term in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", text)
        if term not in {"paper", "summary", "compare", "research"}
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return terms


def _normalized_term_position(text: str, term: str) -> int:
    """在忽略空格和标点后定位术语，并返回其在原文中的起始位置。"""
    normalized_term = re.sub(r"[^a-z0-9]+", "", str(term or "").casefold())
    if len(normalized_term) < 6:
        return -1
    normalized_chars = []
    original_positions = []
    for index, char in enumerate(str(text or "").casefold()):
        if char.isascii() and char.isalnum():
            normalized_chars.append(char)
            original_positions.append(index)
    position = "".join(normalized_chars).find(normalized_term)
    return original_positions[position] if position >= 0 else -1


def _load_text_chunk_records(working_dir: Path) -> list[dict]:
    global _TEXT_CHUNK_CACHE
    path = working_dir / "kv_store_text_chunks.json"
    try:
        stat = path.stat()
        signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return []
    if _TEXT_CHUNK_CACHE and _TEXT_CHUNK_CACHE[0] == signature:
        return _TEXT_CHUNK_CACHE[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = []
    for chunk_id, value in (payload or {}).items():
        if isinstance(value, dict):
            records.append({"chunk_id": chunk_id, **value})
    _TEXT_CHUNK_CACHE = (signature, records)
    return records


def _load_full_doc_contents(working_dir: Path, sources: list[str]) -> dict[str, str]:
    """读取与 source 对应的原文，仅用于给结构化表格 chunk 恢复真实页码。"""
    manifest = load_index_manifest(get_manifest_path(working_dir))
    try:
        payload = json.loads((working_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result = {}
    records = manifest.get("documents", {})
    for source in sources:
        doc_id = (records.get(source) or {}).get("doc_id")
        content = str((payload.get(doc_id) or {}).get("content", "")) if doc_id else ""
        if content:
            result[source] = content
    return result


def _infer_table_page(content: str, full_doc: str) -> int | None:
    """用原文中同一 Table 标题附近的真实 [page:N] 标记补齐表格派生 chunk。"""
    table_match = re.search(r"\bTable\s+(\d+)\b", str(content or ""), re.I)
    if not table_match or not full_doc:
        return None
    original_match = re.search(rf"\bTable\s+{re.escape(table_match.group(1))}\b", full_doc, re.I)
    if not original_match:
        return None
    page_matches = list(re.finditer(r"\[page:(\d+)\]", full_doc[:original_match.start()]))
    return int(page_matches[-1].group(1)) if page_matches else None


def _looks_like_reference_section(content: str) -> bool:
    lowered = str(content or "").casefold()
    if re.search(r"(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)", lowered):
        return True
    citation_count = len(re.findall(r"\[\d{1,3}\]", lowered))
    publication_markers = len(re.findall(
        r"\barxiv\b|\bproceedings\b|\bjournal\b|\btransactions\b|\bpress\b|\bconference\b",
        lowered,
    ))
    return citation_count >= 4 and publication_markers >= 2


def _infer_section(content: str) -> str:
    lowered = content.casefold()
    if _looks_like_reference_section(content):
        return "References"
    for marker, label in (
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("related work", "Related Work"),
        ("methodology", "Method"),
        ("method", "Method"),
        ("experiment", "Experiments"),
        ("conclusion", "Conclusion"),
    ):
        if marker in lowered:
            return label
    return "未知章节"


def _metric_anchor_position(text: str) -> int | None:
    """定位带单位的实验指标，避免证据窗口只保留 chunk 开头。"""
    match = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:%|ms|milliseconds?|bleu|map|ap(?:50|75)?|flops|fps)"
        r"(?![a-z0-9_])",
        text,
        re.I,
    )
    return match.start() if match else None


def _clean_evidence_quote(
    content: str,
    terms: set[str] | None = None,
    limit: int = 700,
    *,
    prefer_metric: bool = False,
) -> str:
    text = re.sub(r"(?s)\[PAPER_METADATA\].*?\[/PAPER_METADATA\]", " ", content or "")
    text = re.sub(r"\s*\[page:\d+\]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^:?\d+\]\s*", "", text)
    lowered = text.casefold()
    metric_position = _metric_anchor_position(text) if prefer_metric else None
    matched_terms = []
    for term in (terms or set()):
        position = lowered.find(term)
        if position < 0:
            position = _normalized_term_position(text, term)
        if position >= 0:
            matched_terms.append((len(term), position, term))
    positions = [position for _, position, _ in matched_terms]
    # 指标问题点名具体模型、模块或表项时，证据窗口必须围绕最长的身份词，
    # 不能永远截取表格第一行的第一个数字并把其他方法的指标带进答案。
    identity_match = max(matched_terms, default=None)
    identity_position = identity_match[1] if identity_match and identity_match[0] >= 6 else None
    if prefer_metric and identity_position is not None:
        anchor = identity_position
    else:
        anchor = metric_position if metric_position is not None else (min(positions) if positions else None)
    if anchor is not None and len(text) > limit:
        start = max(0, anchor - (300 if metric_position is not None else 180))
        text = text[start:start + limit]
        if start:
            text = "…" + text
    return text[:limit].rstrip()


def _is_broad_evidence_query(query: str) -> bool:
    normalized = str(query or "").casefold()
    return bool(re.search(
        r"总结|综述|概括|介绍|讲一下|详细|核心方法|主要结论|实验结果|"
        r"summari[sz]e|overview|review|method.*result|method.*conclusion",
        normalized,
    ))


def _metric_evidence_bonus(content: str, query: str, *, broad_query: bool) -> float:
    metric_query = bool(re.search(
        r"指标|性能|数值|实验|结果|速度|延迟|参数量|复杂度|计算瓶颈|"
        r"bleu|\bmap\b|\bap(?:50|75)?\b|flops|latency|throughput|accuracy|precision|recall|complexity",
        str(query or "").casefold(),
    ))
    if not broad_query and not metric_query:
        return 0.0
    lowered = str(content or "").casefold()
    marker_count = len(re.findall(
        r"bleu|\bmap\b|\bap(?:50|75)?\b|flops|latency|throughput|accuracy|precision|recall|"
        r"parameters?|\bms\b|complexity per layer|maximum path length|state[- ]of[- ]the[- ]art|"
        r"outperform|wmt|coco",
        lowered,
    ))
    numeric_count = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|ms|bleu)?\b", lowered))
    if marker_count == 0:
        return 0.0
    complexity_focus = bool(
        re.search(r"复杂度|计算瓶颈|complexity", str(query or "").casefold())
        and re.search(r"complexity per layer|maximum path length", lowered)
    )
    return min(8.0, 1.2 + marker_count * 0.35 + min(numeric_count, 6) * 0.15 + (4.0 if complexity_focus else 0.0))


def _select_diverse_evidence_spans(
    ranked: list[tuple[float, int, dict]],
    limit: int,
    *,
    broad_query: bool,
) -> list[dict]:
    """优先覆盖不同章节和页码，再按相关性补齐，避免同页近邻 chunk 挤占证据包。"""
    selected: list[dict] = []
    selected_chunks: set[str] = set()
    selected_quotes: set[str] = set()
    page_counts: dict[int | None, int] = {}

    def add(item: tuple[float, int, dict], *, max_per_page: int | None = None) -> bool:
        span = item[2]
        chunk_id = str(span.get("chunk_id") or "")
        quote_key = re.sub(r"\W+", "", str(span.get("quote") or "").casefold())[:240]
        page = span.get("page_start")
        if chunk_id in selected_chunks or (quote_key and quote_key in selected_quotes):
            return False
        if max_per_page is not None and page_counts.get(page, 0) >= max_per_page:
            return False
        selected.append(span)
        selected_chunks.add(chunk_id)
        if quote_key:
            selected_quotes.add(quote_key)
        page_counts[page] = page_counts.get(page, 0) + 1
        return True

    if broad_query:
        metric_candidate = next(
            (item for item in ranked if _metric_anchor_position(str(item[2].get("quote") or "")) is not None),
            None,
        )
        for section in ("Abstract", "Introduction", "Method", "Experiments", "Conclusion"):
            candidate = next(
                (item for item in ranked if item[2].get("section") == section),
                None,
            )
            if candidate is not None:
                add(candidate, max_per_page=1)
            if section == "Abstract" and metric_candidate is not None:
                add(metric_candidate)
            if len(selected) >= limit:
                return selected

    for max_per_page in (1, 2, None):
        for item in ranked:
            add(item, max_per_page=max_per_page)
            if len(selected) >= limit:
                return selected
    return selected


def load_relevant_evidence_spans(
    sources: list[str],
    query: str,
    working_dir: Path | None = None,
    per_source: int = 2,
) -> dict[str, list[dict]]:
    """从 LightRAG 的真实 text chunk KV 中选取页级证据，不触发重新建图。"""
    base_dir = working_dir or get_lightrag_working_dir()
    source_lookup = {os.path.basename(source).casefold(): source for source in sources}
    candidates: dict[str, list[tuple[float, int, dict]]] = {source: [] for source in sources}
    terms = _query_terms(query)
    broad_query = _is_broad_evidence_query(query)
    full_docs_by_source = _load_full_doc_contents(base_dir, sources)

    for record in _load_text_chunk_records(base_dir):
        raw_source = os.path.basename(str(record.get("file_path") or ""))
        source = source_lookup.get(raw_source.casefold())
        content = str(record.get("content") or "")
        if not source or not content:
            continue
        lowered = content.casefold()
        lexical_hits = sum(1 for term in terms if term in lowered)
        identity_terms = {
            term for term in terms
            if len(term) >= 6 and term not in {
                "params", "flops", "latency", "throughput", "accuracy", "precision",
                "recall", "markdown", "table", "result", "results",
            }
        }
        normalized_identity_hits = sum(
            1 for term in identity_terms
            if term not in lowered and _normalized_term_position(content, term) >= 0
        )
        order = int(record.get("chunk_order_index") or 0)
        section = _infer_section(content)
        if broad_query and section == "References":
            continue
        section_bonus = {
            "Abstract": 2.0,
            "Introduction": 1.2,
            "Method": 2.2,
            "Experiments": 2.2,
            "Conclusion": 2.0,
        }.get(section, 0.0) if broad_query else 0.0
        metric_bonus = _metric_evidence_bonus(content, query, broad_query=broad_query)
        score = float(
            lexical_hits * 3
            + normalized_identity_hits * 8
            + (2 if order == 0 else 1 if order == 1 else 0)
            + section_bonus
            + (0.25 if broad_query else 0.0)
            + metric_bonus
        )
        if score <= 0:
            continue
        pages = [int(value) for value in re.findall(r"\[page:(\d+)\]", content)]
        page_start = min(pages) if pages else None
        if page_start is None:
            page_start = _infer_table_page(content, full_docs_by_source.get(source, ""))
        page_end = max(pages) if pages else page_start
        quote_terms = set(terms)
        if re.search(r"复杂度|计算瓶颈|complexity", str(query or "").casefold()):
            quote_terms.update({"complexity per layer", "maximum path length"})
        quote = _clean_evidence_quote(content, quote_terms, prefer_metric=metric_bonus > 0)
        if not quote:
            continue
        span = {
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "section": section,
            "chunk_id": str(record.get("chunk_id") or record.get("_id") or ""),
            "quote": quote,
            "confidence": round(min(0.95, 0.45 + score / (score + 10) * 0.5), 3),
        }
        candidates[source].append((score, -order, span))

    result = {}
    for source, spans in candidates.items():
        ranked = sorted(spans, key=lambda item: (item[0], item[1]), reverse=True)
        result[source] = _select_diverse_evidence_spans(
            ranked,
            max(1, per_source),
            broad_query=broad_query,
        )
    return result


def _file_signature(path: str) -> dict:
    stat = os.stat(path)
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


@dataclass(frozen=True)
class IndexSyncPlan:
    upsert_paths: list[str]
    removed_sources: list[str]


def plan_index_updates(physical_paths: list[str], manifest: dict | None = None) -> IndexSyncPlan:
    if manifest is None:
        manifest = load_index_manifest()
    records = manifest.get("documents", {})
    physical = {os.path.basename(path): path for path in physical_paths}
    # 个人本地论文库以文件名作为稳定身份。代码/PDF 重新上传会改变 mtime，
    # 但不应因此重建昂贵的知识图谱；同名 PDF 默认视为同一篇论文。
    upsert_paths = [path for source, path in physical.items() if source not in records]

    removed = sorted(set(records) - set(physical))
    return IndexSyncPlan(upsert_paths=upsert_paths, removed_sources=removed)


@dataclass
class LightRAGDocument:
    source: str
    path: str
    text: str
    title: str
    year: str


class LightRAGStore:
    """LightRAG Core 的项目适配层：复用现有 vLLM 与本地 HuggingFace embedding。"""

    def __init__(self, working_dir: Path | None = None):
        self.working_dir = working_dir or get_lightrag_working_dir()
        self.manifest_path = get_manifest_path(self.working_dir)
        self.rag = None
        self._embedding_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()

    async def initialize(self):
        if self.rag is not None:
            return self

        try:
            import numpy as np
            from lightrag import LightRAG
            from lightrag.llm.openai import openai_complete_if_cache
            from lightrag.utils import EmbeddingFunc
        except ImportError as exc:
            raise RuntimeError(
                "缺少 LightRAG 依赖，请在服务器环境执行 `pip install -r requirements-lightrag.txt`。"
            ) from exc

        self.working_dir.mkdir(parents=True, exist_ok=True)
        from research_agent.retrieval.local_models import get_embeddings

        embeddings = get_embeddings()
        embedding_dim = int(os.environ.get("LIGHTRAG_EMBEDDING_DIM", "1024"))
        embedding_name = os.environ.get("LIGHTRAG_EMBEDDING_MODEL", "bge-large-zh-v1.5")
        embedding_max_tokens = int(os.environ.get("LIGHTRAG_EMBEDDING_MAX_TOKENS", "8192"))

        async def embedding_func(texts: list[str]) -> np.ndarray:
            async with self._embedding_lock:
                vectors = await asyncio.to_thread(embeddings.embed_documents, list(texts))
            array = np.asarray(vectors, dtype=np.float32)
            if array.ndim != 2 or (array.shape[1] if len(array) else embedding_dim) != embedding_dim:
                actual = array.shape[1] if array.ndim == 2 and len(array) else "unknown"
                raise ValueError(
                    f"LightRAG embedding 维度配置为 {embedding_dim}，实际为 {actual}；"
                    "请修正 LIGHTRAG_EMBEDDING_DIM 并提升 LIGHTRAG_INDEX_VERSION。"
                )
            return array

        wrapped_embedding = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_tokens,
            model_name=embedding_name,
            func=embedding_func,
        )

        model_name = os.environ.get("LIGHTRAG_LLM_MODEL", os.environ.get("LOCAL_LLM_MODEL", "qwen3"))
        base_url = os.environ.get(
            "LIGHTRAG_LLM_BASE_URL",
            os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6006/v1"),
        )
        api_key = os.environ.get("LIGHTRAG_LLM_API_KEY", os.environ.get("LOCAL_LLM_API_KEY", "sk-local"))
        max_gleaning = int(os.environ.get("LIGHTRAG_MAX_GLEANING", "0"))
        max_extraction_records = int(os.environ.get("LIGHTRAG_MAX_EXTRACTION_RECORDS", "30"))
        max_extraction_entities = int(os.environ.get("LIGHTRAG_MAX_EXTRACTION_ENTITIES", "20"))
        summary_context_size = int(os.environ.get("LIGHTRAG_SUMMARY_CONTEXT_SIZE", "6000"))
        extraction_use_json = os.environ.get(
            "LIGHTRAG_ENTITY_EXTRACTION_USE_JSON", "true"
        ).lower() in {"1", "true", "yes", "on"}

        raw_extra_body = os.environ.get("LIGHTRAG_LLM_EXTRA_BODY", "")
        try:
            configured_extra_body = json.loads(raw_extra_body) if raw_extra_body else None
        except json.JSONDecodeError as exc:
            raise ValueError("LIGHTRAG_LLM_EXTRA_BODY 必须是合法 JSON") from exc
        if configured_extra_body is None and "qwen3" in model_name.lower():
            configured_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        async def llm_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            kwargs.pop("model", None)
            if not kwargs.get("max_tokens"):
                budget_name = (
                    "LIGHTRAG_KEYWORD_MAX_OUTPUT_TOKENS"
                    if keyword_extraction
                    else "LIGHTRAG_LLM_MAX_OUTPUT_TOKENS"
                )
                default_budget = "512" if keyword_extraction else "3000"
                kwargs["max_tokens"] = int(os.environ.get(budget_name, default_budget))
            if configured_extra_body is not None:
                kwargs.setdefault("extra_body", configured_extra_body)

            class CompletionCall:
                async def ainvoke(self, _unused_prompt):
                    return await openai_complete_if_cache(
                        model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages or [],
                        api_key=api_key,
                        base_url=base_url,
                        **kwargs,
                    )

            try:
                from research_agent.core.llm_clients import LOCAL_ROLE, safe_llm_invoke
            except ModuleNotFoundError:
                # 允许不安装 LangChain 的 LightRAG 存储契约单测运行；正式服务依赖完整 requirements。
                return await CompletionCall().ainvoke("")

            result = await safe_llm_invoke(
                CompletionCall(),
                "",
                "LightRAG_Keyword" if keyword_extraction else "LightRAG_Graph_Extraction",
                max_retries=max(1, int(os.environ.get("LIGHTRAG_LLM_MAX_RETRIES", "2"))),
                role=LOCAL_ROLE,
            )
            if result is None:
                raise RuntimeError("LightRAG 本地模型调用在重试/熔断后仍不可用")
            return str(result)

        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            llm_model_name=model_name,
            llm_model_max_async=int(os.environ.get("LIGHTRAG_LLM_MAX_ASYNC", "1")),
            embedding_func=wrapped_embedding,
            embedding_batch_num=int(os.environ.get("LIGHTRAG_EMBEDDING_BATCH_SIZE", "1")),
            embedding_func_max_async=int(os.environ.get("LIGHTRAG_EMBEDDING_MAX_ASYNC", "1")),
            chunk_token_size=int(os.environ.get("LIGHTRAG_CHUNK_SIZE", "600")),
            chunk_overlap_token_size=int(os.environ.get("LIGHTRAG_CHUNK_OVERLAP", "80")),
            entity_extract_max_gleaning=max_gleaning,
            entity_extract_max_records=max_extraction_records,
            entity_extract_max_entities=max_extraction_entities,
            entity_extraction_use_json=extraction_use_json,
            summary_context_size=summary_context_size,
            max_parallel_insert=int(os.environ.get("LIGHTRAG_MAX_PARALLEL_INSERT", "1")),
            addon_params={
                "language": "Chinese",
                "entity_types_guidance": (
                    "- Paper: academic paper, report, or preprint\n"
                    "- Method: model, algorithm, framework, or experimental technique\n"
                    "- Task: research problem or benchmark task\n"
                    "- Dataset: dataset, corpus, benchmark, or experimental sample\n"
                    "- Metric: evaluation metric or measured indicator\n"
                    "- Finding: conclusion, limitation, comparison, or empirical claim\n"
                    "- Concept: domain concept, phenomenon, or theory"
                ),
            },
        )
        print(
            "[LightRAG build config] "
            f"max_gleaning={max_gleaning}, max_records={max_extraction_records}, "
            f"max_entities={max_extraction_entities}, json={extraction_use_json}, "
            f"summary_context={summary_context_size}"
        )
        await self.rag.initialize_storages()
        return self

    @staticmethod
    def _document_id(source: str, signature: dict) -> str:
        raw = f"{LIGHTRAG_INDEX_VERSION}:{source}:{signature['mtime_ns']}:{signature['size']}"
        return "paper-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _write_manifest(self, manifest: dict) -> None:
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    async def _delete_document_ids_best_effort(self, document_ids: list[str]) -> None:
        for doc_id in document_ids:
            try:
                await self.rag.adelete_by_doc_id(doc_id)
            except Exception as exc:
                print(f"  └─ ⚠️ LightRAG 失败文档残留清理失败: {doc_id}: {exc}")

    async def sync_documents(self, documents: list[LightRAGDocument], physical_paths: list[str]) -> dict:
        await self.initialize()
        async with self._sync_lock:
            return await self._sync_documents_unlocked(documents, physical_paths)

    async def _sync_documents_unlocked(
        self,
        documents: list[LightRAGDocument],
        physical_paths: list[str],
    ) -> dict:
        manifest = load_index_manifest(self.manifest_path)
        records = dict(manifest.get("documents", {}))
        physical = {os.path.basename(path): path for path in physical_paths}
        docs_by_source = {document.source: document for document in documents}
        inserted = []
        removed = []

        insert_payload = []
        insert_ids = []
        insert_paths = []
        new_records = {}
        old_ids = {}
        failed = {}

        for source, document in docs_by_source.items():
            path = physical.get(source)
            if not path or not document.text.strip():
                continue
            signature = _file_signature(path)
            doc_id = self._document_id(source, signature)
            insert_payload.append(document.text)
            insert_ids.append(doc_id)
            insert_paths.append(document.path)
            old_ids[source] = (records.get(source) or {}).get("doc_id")
            new_records[source] = {
                "doc_id": doc_id,
                "mtime_ns": signature["mtime_ns"],
                "size": signature["size"],
                "title": document.title,
                "year": document.year,
                "path": document.path,
            }

        if insert_payload:
            try:
                await self.rag.ainsert(insert_payload, ids=insert_ids, file_paths=insert_paths)
            except BaseException:
                # ainsert 可能已经写入部分 chunk/entity；异常或 Ctrl+C 时必须尽力回滚新 doc_id。
                await asyncio.shield(self._delete_document_ids_best_effort(insert_ids))
                raise
            statuses = await self.rag.aget_docs_by_ids(insert_ids)
            failed_ids = []
            for source, record in new_records.items():
                status_record = statuses.get(record["doc_id"])
                if isinstance(status_record, dict):
                    raw_status = status_record.get("status")
                    error_msg = status_record.get("error_msg") or ""
                else:
                    raw_status = getattr(status_record, "status", None)
                    error_msg = getattr(status_record, "error_msg", "") or ""
                status_value = getattr(raw_status, "value", raw_status)
                status_text = str(status_value or "missing").lower()
                if status_text != "processed":
                    failed[source] = f"{status_text}: {error_msg}".rstrip(": ")
                    failed_ids.append(record["doc_id"])
                    continue

                old_id = old_ids.get(source)
                if old_id and old_id != record["doc_id"]:
                    try:
                        await self.rag.adelete_by_doc_id(old_id)
                    except Exception as exc:
                        print(f"  └─ ⚠️ LightRAG 旧版本文档清理失败: {source}: {exc}")
                records[source] = record
                inserted.append(source)

            if failed_ids:
                await self._delete_document_ids_best_effort(failed_ids)

        for source in sorted(set(records) - set(physical)):
            doc_id = (records.get(source) or {}).get("doc_id")
            try:
                if doc_id:
                    await self.rag.adelete_by_doc_id(doc_id)
                records.pop(source, None)
                removed.append(source)
            except Exception as exc:
                print(f"  └─ ⚠️ LightRAG 已删除 PDF 的索引清理失败: {source}: {exc}")

        self._write_manifest({"index_version": LIGHTRAG_INDEX_VERSION, "documents": records})
        return {
            "inserted": inserted,
            "failed": failed,
            "removed": removed,
            "indexed": sorted(records),
        }

    async def query(
        self,
        query: str,
        mode: str = "mix",
        expected_source_count: int = 1,
        *,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        max_entity_tokens: int | None = None,
        max_relation_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ) -> str:
        await self.initialize()
        from lightrag import QueryParam

        configured_chunk_top_k = int(os.environ.get("LIGHTRAG_CHUNK_TOP_K", "6"))
        chunk_top_k = chunk_top_k or max(configured_chunk_top_k, min(max(1, expected_source_count), 12))

        result = await self.rag.aquery(
            query,
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=top_k or int(os.environ.get("LIGHTRAG_TOP_K", "30")),
                chunk_top_k=chunk_top_k,
                max_entity_tokens=max_entity_tokens or int(os.environ.get("LIGHTRAG_MAX_ENTITY_TOKENS", "1400")),
                max_relation_tokens=max_relation_tokens or int(os.environ.get("LIGHTRAG_MAX_RELATION_TOKENS", "1600")),
                max_total_tokens=max_total_tokens or int(os.environ.get("LIGHTRAG_MAX_TOTAL_TOKENS", "4500")),
                enable_rerank=False,
            ),
        )
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("context", "data", "result"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result or "")

    async def finalize(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None


_GLOBAL_LIGHTRAG_STORE = None
_STORE_INIT_LOCK = None


async def get_lightrag_store() -> LightRAGStore:
    global _GLOBAL_LIGHTRAG_STORE, _STORE_INIT_LOCK
    if _GLOBAL_LIGHTRAG_STORE is not None:
        return _GLOBAL_LIGHTRAG_STORE
    if _STORE_INIT_LOCK is None:
        _STORE_INIT_LOCK = asyncio.Lock()
    async with _STORE_INIT_LOCK:
        if _GLOBAL_LIGHTRAG_STORE is None:
            store = LightRAGStore()
            await store.initialize()
            _GLOBAL_LIGHTRAG_STORE = store
    return _GLOBAL_LIGHTRAG_STORE


async def finalize_lightrag_store() -> None:
    global _GLOBAL_LIGHTRAG_STORE
    if _GLOBAL_LIGHTRAG_STORE is not None:
        await _GLOBAL_LIGHTRAG_STORE.finalize()
        _GLOBAL_LIGHTRAG_STORE = None
