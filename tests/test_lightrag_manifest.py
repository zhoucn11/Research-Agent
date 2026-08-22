import asyncio
import json
import os
import sys
import types

import pytest

from research_agent.retrieval.lightrag_store import (
    LIGHTRAG_INDEX_VERSION,
    LightRAGDocument,
    LightRAGStore,
    find_indexed_source_by_title,
    load_document_header_context,
    plan_index_updates,
)


def test_load_document_header_context_uses_manifest_source_identity(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "research_agent_manifest.json").write_text(
        json.dumps(
            {
                "index_version": LIGHTRAG_INDEX_VERSION,
                "documents": {
                    "paper-a.pdf": {"doc_id": "doc-a"},
                    "paper-b.pdf": {"doc_id": "doc-b"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (graph_dir / "kv_store_full_docs.json").write_text(
        json.dumps(
            {
                "doc-a": {"content": "Paper A\nAlice, Bob\nAbstract"},
                "doc-b": {"content": "Paper B\nCarol\nAbstract"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    context = load_document_header_context(
        ["paper-b.pdf", "paper-a.pdf"],
        working_dir=graph_dir,
        per_source_chars=100,
        total_chars=200,
    )

    assert context.index("paper-b.pdf") < context.index("paper-a.pdf")
    assert "Carol" in context
    assert "Alice, Bob" in context


def test_manifest_plan_uses_filename_identity_and_detects_new_removed_files(tmp_path):
    kept = tmp_path / "kept.pdf"
    changed = tmp_path / "changed.pdf"
    added = tmp_path / "added.pdf"
    kept.write_bytes(b"same")
    changed.write_bytes(b"new-content")
    added.write_bytes(b"added")

    kept_stat = os.stat(kept)
    changed_stat = os.stat(changed)
    manifest = {
        "documents": {
            "kept.pdf": {"mtime_ns": kept_stat.st_mtime_ns, "size": kept_stat.st_size},
            "changed.pdf": {"mtime_ns": changed_stat.st_mtime_ns, "size": 1},
            "removed.pdf": {"mtime_ns": 1, "size": 1},
        }
    }

    plan = plan_index_updates([str(kept), str(changed), str(added)], manifest)

    assert {os.path.basename(path) for path in plan.upsert_paths} == {"added.pdf"}
    assert plan.removed_sources == ["removed.pdf"]


def test_manifest_title_match_never_substitutes_semantically_related_paper():
    manifest = {
        "documents": {
            "Attention is All You Need.pdf": {"title": "Attention Is All You Need"},
            "Mamba yolo.pdf": {
                "title": "Mamba YOLO: A Simple Baseline for Object Detection with State Space Model"
            },
        }
    }

    assert find_indexed_source_by_title("Attention Is All You Need", manifest) == (
        "Attention is All You Need.pdf"
    )
    assert find_indexed_source_by_title(
        "You Only Look Once: Unified, Real-Time Object Detection",
        manifest,
    ) == ""


def test_store_persists_manifest_and_deletes_missing_derived_document(tmp_path):
    class FakeRAG:
        def __init__(self):
            self.insert_calls = []
            self.deleted_ids = []

        async def ainsert(self, texts, ids, file_paths):
            self.insert_calls.append((texts, ids, file_paths))

        async def aget_docs_by_ids(self, ids):
            return {
                doc_id: types.SimpleNamespace(status="processed", error_msg="")
                for doc_id in ids
            }

        async def adelete_by_doc_id(self, doc_id):
            self.deleted_ids.append(doc_id)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf-v1")
    store = LightRAGStore(tmp_path / "graph")
    store.rag = FakeRAG()
    document = LightRAGDocument(
        source="paper.pdf",
        path=str(pdf_path),
        text="paper content",
        title="Paper",
        year="2026",
    )

    first = asyncio.run(store.sync_documents([document], [str(pdf_path)]))
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    doc_id = manifest["documents"]["paper.pdf"]["doc_id"]

    assert first["inserted"] == ["paper.pdf"]
    assert store.rag.insert_calls[0][2] == [str(pdf_path)]

    second = asyncio.run(store.sync_documents([], []))
    assert second["removed"] == ["paper.pdf"]
    assert store.rag.deleted_ids == [doc_id]


def test_store_does_not_manifest_failed_lightrag_document(tmp_path):
    class FakeRAG:
        def __init__(self):
            self.deleted_ids = []

        async def ainsert(self, texts, ids, file_paths):
            return None

        async def aget_docs_by_ids(self, ids):
            return {
                doc_id: types.SimpleNamespace(
                    status="failed",
                    error_msg="entity extraction exceeded model context",
                )
                for doc_id in ids
            }

        async def adelete_by_doc_id(self, doc_id):
            self.deleted_ids.append(doc_id)

    pdf_path = tmp_path / "failed.pdf"
    pdf_path.write_bytes(b"pdf")
    store = LightRAGStore(tmp_path / "graph")
    store.rag = FakeRAG()
    document = LightRAGDocument(
        source="failed.pdf",
        path=str(pdf_path),
        text="paper content",
        title="Failed Paper",
        year="2026",
    )

    result = asyncio.run(store.sync_documents([document], [str(pdf_path)]))
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    retry_plan = plan_index_updates([str(pdf_path)], manifest)

    assert result["inserted"] == []
    assert result["failed"] == {
        "failed.pdf": "failed: entity extraction exceeded model context"
    }
    assert manifest["documents"] == {}
    assert retry_plan.upsert_paths == [str(pdf_path)]
    assert len(store.rag.deleted_ids) == 1


def test_store_rolls_back_partial_insert_when_lightrag_raises(tmp_path):
    class FakeRAG:
        def __init__(self):
            self.deleted_ids = []

        async def ainsert(self, texts, ids, file_paths):
            raise RuntimeError("interrupted during entity extraction")

        async def adelete_by_doc_id(self, doc_id):
            self.deleted_ids.append(doc_id)

    pdf_path = tmp_path / "partial.pdf"
    pdf_path.write_bytes(b"pdf")
    store = LightRAGStore(tmp_path / "graph")
    store.rag = FakeRAG()
    document = LightRAGDocument(
        source="partial.pdf",
        path=str(pdf_path),
        text="paper content",
        title="Partial Paper",
        year="2026",
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(store.sync_documents([document], [str(pdf_path)]))

    assert len(store.rag.deleted_ids) == 1
    assert not store.manifest_path.exists()


def test_store_initialization_matches_lightrag_public_api(tmp_path, monkeypatch):
    captured = {}
    complete_calls = []

    class FakeEmbeddingFunc:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def initialize_storages(self):
            captured["initialized"] = True

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.0] * 4 for _ in texts]

    async def fake_complete(*args, **kwargs):
        complete_calls.append((args, kwargs))
        return "ok"

    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.LightRAG = FakeLightRAG
    llm_package = types.ModuleType("lightrag.llm")
    openai_module = types.ModuleType("lightrag.llm.openai")
    openai_module.openai_complete_if_cache = fake_complete
    utils_module = types.ModuleType("lightrag.utils")
    utils_module.EmbeddingFunc = FakeEmbeddingFunc
    local_models_module = types.ModuleType("research_agent.retrieval.local_models")
    local_models_module.get_embeddings = lambda: FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.llm", llm_package)
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", openai_module)
    monkeypatch.setitem(sys.modules, "lightrag.utils", utils_module)
    monkeypatch.setitem(sys.modules, "research_agent.retrieval.local_models", local_models_module)
    monkeypatch.setenv("LIGHTRAG_EMBEDDING_DIM", "4")
    monkeypatch.delenv("LIGHTRAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("LIGHTRAG_CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("LIGHTRAG_MAX_GLEANING", raising=False)
    monkeypatch.delenv("LIGHTRAG_MAX_EXTRACTION_RECORDS", raising=False)
    monkeypatch.delenv("LIGHTRAG_MAX_EXTRACTION_ENTITIES", raising=False)
    monkeypatch.delenv("LIGHTRAG_SUMMARY_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("LIGHTRAG_ENTITY_EXTRACTION_USE_JSON", raising=False)

    store = LightRAGStore(tmp_path / "graph")
    asyncio.run(store.initialize())

    assert captured["initialized"] is True
    assert captured["working_dir"] == str(tmp_path / "graph")
    assert captured["llm_model_name"] == "qwen3"
    assert "llm_model_max_token_size" not in captured
    assert captured["llm_model_max_async"] == 1
    assert captured["embedding_batch_num"] == 1
    assert captured["embedding_func_max_async"] == 1
    assert captured["max_parallel_insert"] == 1
    assert captured["chunk_token_size"] == 600
    assert captured["chunk_overlap_token_size"] == 80
    assert captured["entity_extract_max_gleaning"] == 0
    assert captured["entity_extract_max_records"] == 30
    assert captured["entity_extract_max_entities"] == 20
    assert captured["entity_extraction_use_json"] is True
    assert captured["summary_context_size"] == 6000
    assert captured["embedding_func"].embedding_dim == 4

    asyncio.run(captured["llm_model_func"]("keywords", keyword_extraction=True))
    asyncio.run(captured["llm_model_func"]("extract", keyword_extraction=False))
    assert complete_calls[0][1]["max_tokens"] == 512
    assert complete_calls[1][1]["max_tokens"] == 3000


def test_query_defaults_fit_8192_context(tmp_path, monkeypatch):
    captured = {}

    class FakeQueryParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeRAG:
        async def aquery(self, query, param):
            captured.update(param.__dict__)
            return "context"

    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    for name in (
        "LIGHTRAG_CHUNK_TOP_K",
        "LIGHTRAG_MAX_ENTITY_TOKENS",
        "LIGHTRAG_MAX_RELATION_TOKENS",
        "LIGHTRAG_MAX_TOTAL_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    store = LightRAGStore(tmp_path / "graph")
    store.rag = FakeRAG()

    assert asyncio.run(store.query("compare papers")) == "context"
    assert captured["chunk_top_k"] == 6
    assert captured["max_entity_tokens"] == 1400
    assert captured["max_relation_tokens"] == 1600
    assert captured["max_total_tokens"] == 4500

    assert asyncio.run(store.query("compare all papers", expected_source_count=7)) == "context"
    assert captured["chunk_top_k"] == 7


def test_query_budget_can_use_16k_model_without_consuming_output_reserve(tmp_path, monkeypatch):
    captured = {}

    class FakeQueryParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeRAG:
        async def aquery(self, query, param):
            captured.update(param.__dict__)
            return "context"

    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setenv("LIGHTRAG_CHUNK_TOP_K", "8")
    monkeypatch.setenv("LIGHTRAG_MAX_ENTITY_TOKENS", "1400")
    monkeypatch.setenv("LIGHTRAG_MAX_RELATION_TOKENS", "1600")
    monkeypatch.setenv("LIGHTRAG_MAX_TOTAL_TOKENS", "5500")

    store = LightRAGStore(tmp_path / "graph")
    store.rag = FakeRAG()

    assert asyncio.run(store.query("compare papers", expected_source_count=2)) == "context"
    assert captured["chunk_top_k"] == 8
    assert captured["max_entity_tokens"] == 1400
    assert captured["max_relation_tokens"] == 1600
    assert captured["max_total_tokens"] == 5500
