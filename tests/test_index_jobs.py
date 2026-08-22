import asyncio

from research_agent.retrieval import index_jobs
from research_agent.retrieval.lightrag_store import LightRAGDocument


def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(index_jobs, "INDEX_JOB_DB", tmp_path / "index_jobs.sqlite3")
    monkeypatch.setattr(index_jobs, "_PDF_DIR", tmp_path)
    index_jobs.init_index_job_store()


def test_index_job_is_persistent_deduplicated_and_cancellable(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")

    first = index_jobs.enqueue_index_job(str(pdf))
    second = index_jobs.enqueue_index_job(str(pdf))

    assert first["job_id"] == second["job_id"]
    assert first["status"] == "queued"
    assert index_jobs.cancel_index_job(first["job_id"]) is True
    assert index_jobs.get_index_job(first["job_id"])["status"] == "cancelled"
    assert index_jobs.retry_index_job(first["job_id"]) is True
    assert index_jobs.get_index_job(first["job_id"])["status"] == "queued"


def test_interrupted_index_job_recovers_to_queue(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    job = index_jobs.enqueue_index_job(str(pdf))
    claimed = index_jobs._claim_next_job()

    assert claimed["job_id"] == job["job_id"]
    assert claimed["status"] == "parsing"
    assert index_jobs.recover_interrupted_index_jobs() == 1
    assert index_jobs.get_index_job(job["job_id"])["status"] == "queued"


def test_completed_job_is_requeued_when_current_manifest_no_longer_contains_source(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    first = index_jobs.enqueue_index_job(str(pdf))
    index_jobs._update_job(first["job_id"], status="completed", progress=100)
    monkeypatch.setattr(index_jobs, "list_indexed_sources", lambda: [])

    second = index_jobs.enqueue_index_job(str(pdf))

    assert second["job_id"] != first["job_id"]
    assert second["status"] == "queued"


def test_index_job_completes_only_after_store_confirms_insert(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    job = index_jobs.enqueue_index_job(str(pdf))
    claimed = index_jobs._claim_next_job()

    async def fake_parse(paths):
        return [(paths[0], "paper text")]

    async def fake_build(parsed):
        return [LightRAGDocument(
            source="paper.pdf",
            path=str(pdf),
            text="paper text",
            title="Paper",
            year="2025",
        )]

    class FakeStore:
        async def sync_documents(self, documents, physical_paths):
            return {"inserted": ["paper.pdf"], "failed": {}, "removed": [], "indexed": ["paper.pdf"]}

    async def fake_store():
        return FakeStore()

    monkeypatch.setattr(index_jobs, "parse_pdfs", fake_parse)
    monkeypatch.setattr(index_jobs, "build_lightrag_documents", fake_build)
    monkeypatch.setattr(index_jobs, "get_lightrag_store", fake_store)
    monkeypatch.setattr(index_jobs, "list_indexed_sources", lambda: [])

    asyncio.run(index_jobs._process_job(claimed))

    completed = index_jobs.get_index_job(job["job_id"])
    assert completed["status"] == "completed"
    assert completed["progress"] == 100


def test_rag_node_no_longer_builds_index_in_chat_request():
    source = (index_jobs.PROJECT_ROOT / "research_agent" / "agents" / "rag_agent.py").read_text(encoding="utf-8")

    assert "sync_documents(" not in source
    assert "parse_pdfs(" not in source
