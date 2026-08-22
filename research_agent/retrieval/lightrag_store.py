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


def _infer_section(content: str) -> str:
    lowered = content.casefold()
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


def _clean_evidence_quote(content: str, terms: set[str] | None = None, limit: int = 700) -> str:
    text = re.sub(r"(?s)\[PAPER_METADATA\].*?\[/PAPER_METADATA\]", " ", content or "")
    text = re.sub(r"\s*\[page:\d+\]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^:?\d+\]\s*", "", text)
    positions = [text.casefold().find(term) for term in (terms or set())]
    positions = [position for position in positions if position >= 0]
    if positions and len(text) > limit:
        start = max(0, min(positions) - 180)
        text = text[start:start + limit]
        if start:
            text = "…" + text
    return text[:limit].rstrip()


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

    for record in _load_text_chunk_records(base_dir):
        raw_source = os.path.basename(str(record.get("file_path") or ""))
        source = source_lookup.get(raw_source.casefold())
        content = str(record.get("content") or "")
        if not source or not content:
            continue
        lowered = content.casefold()
        lexical_hits = sum(1 for term in terms if term in lowered)
        order = int(record.get("chunk_order_index") or 0)
        score = float(lexical_hits * 3 + (2 if order == 0 else 1 if order == 1 else 0))
        if score <= 0:
            continue
        pages = [int(value) for value in re.findall(r"\[page:(\d+)\]", content)]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else page_start
        quote = _clean_evidence_quote(content, terms)
        if not quote:
            continue
        span = {
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "section": _infer_section(content),
            "chunk_id": str(record.get("chunk_id") or record.get("_id") or ""),
            "quote": quote,
            "confidence": round(min(0.95, 0.45 + score / (score + 10) * 0.5), 3),
        }
        candidates[source].append((score, -order, span))

    result = {}
    for source, spans in candidates.items():
        ranked = sorted(spans, key=lambda item: (item[0], item[1]), reverse=True)
        result[source] = [span for _, _, span in ranked[:max(1, per_source)]]
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
