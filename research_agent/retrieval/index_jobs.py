import asyncio
from datetime import datetime, timezone
import glob
import os
from pathlib import Path
import sqlite3
import uuid

from research_agent.core.runtime_events import runtime_print as print
from research_agent.retrieval.lightrag_store import (
    get_lightrag_store,
    list_indexed_sources,
    plan_index_updates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_JOB_DB = Path(os.environ.get("AGENT_INDEX_JOB_DB", PROJECT_ROOT / "index_jobs.sqlite3"))
_WORKER_TASK = None
_WAKE_EVENT = None
_PDF_DIR = None


async def parse_pdfs(paths: list[str]):
    from research_agent.retrieval.pdf_indexing import parse_pdfs as parse
    return await parse(paths)


async def build_lightrag_documents(parsed_papers: list[tuple[str, str]]):
    from research_agent.retrieval.pdf_indexing import build_lightrag_documents as build
    return await build(parsed_papers)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    INDEX_JOB_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_JOB_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_index_job_store() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS index_jobs (
            job_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            file_path TEXT NOT NULL,
            operation TEXT NOT NULL DEFAULT 'upsert',
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_index_jobs_status ON index_jobs(status, created_at)")


def _row_dict(row) -> dict | None:
    return dict(row) if row is not None else None


def get_index_job(job_id: str) -> dict | None:
    init_index_job_store()
    with _connect() as conn:
        return _row_dict(conn.execute("SELECT * FROM index_jobs WHERE job_id = ?", (job_id,)).fetchone())


def list_index_jobs(limit: int = 100) -> list[dict]:
    init_index_job_store()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM index_jobs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_jobs_for_sources(sources: list[str]) -> list[dict]:
    wanted = {str(source).casefold() for source in sources if str(source).strip()}
    latest = []
    seen = set()
    for job in list_index_jobs(500):
        source = str(job["source"]).casefold()
        if source in wanted and source not in seen:
            latest.append(job)
            seen.add(source)
    return latest


def _notify_worker() -> None:
    if _WAKE_EVENT is not None:
        _WAKE_EVENT.set()


def enqueue_index_job(file_path: str) -> dict:
    init_index_job_store()
    path = Path(file_path).resolve()
    source = path.name
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM index_jobs WHERE source = ? ORDER BY created_at DESC LIMIT 1",
            (source,),
        ).fetchone()
        if existing:
            existing_job = dict(existing)
            if existing_job["status"] in {"queued", "parsing", "indexing", "failed", "cancelled"}:
                return existing_job
            # 索引目录/版本被更换或清理后，旧 completed 记录不能阻止当前版本重新建图。
            if existing_job["status"] == "completed" and source in list_indexed_sources():
                return existing_job
        job_id = "index_" + uuid.uuid4().hex[:16]
        now = _now()
        conn.execute(
            "INSERT INTO index_jobs(job_id, source, file_path, operation, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'upsert', 'queued', ?, ?)",
            (job_id, source, str(path), now, now),
        )
    _notify_worker()
    return get_index_job(job_id)


def _enqueue_reconcile_job(pdf_dir: Path) -> dict:
    init_index_job_store()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM index_jobs WHERE operation = 'reconcile' AND status IN ('queued','parsing','indexing') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if existing:
            return dict(existing)
        job_id = "index_" + uuid.uuid4().hex[:16]
        now = _now()
        conn.execute(
            "INSERT INTO index_jobs(job_id, source, file_path, operation, status, created_at, updated_at) "
            "VALUES (?, '__reconcile__', ?, 'reconcile', 'queued', ?, ?)",
            (job_id, str(pdf_dir.resolve()), now, now),
        )
    _notify_worker()
    return get_index_job(job_id)


def enqueue_pending_index_jobs(physical_paths: list[str]) -> list[dict]:
    plan = plan_index_updates(physical_paths)
    jobs = [enqueue_index_job(path) for path in plan.upsert_paths]
    if plan.removed_sources:
        pdf_dir = Path(physical_paths[0]).resolve().parent if physical_paths else Path(_PDF_DIR or "test_pdfs")
        jobs.append(_enqueue_reconcile_job(pdf_dir))
    return jobs


def cancel_index_job(job_id: str) -> bool:
    init_index_job_store()
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE index_jobs SET status = 'cancelled', updated_at = ? WHERE job_id = ? AND status = 'queued'",
            (_now(), job_id),
        )
    return cursor.rowcount == 1


def retry_index_job(job_id: str) -> bool:
    init_index_job_store()
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE index_jobs SET status = 'queued', progress = 0, error = '', updated_at = ? "
            "WHERE job_id = ? AND status IN ('failed','cancelled')",
            (_now(), job_id),
        )
    if cursor.rowcount == 1:
        _notify_worker()
        return True
    return False


def recover_interrupted_index_jobs() -> int:
    init_index_job_store()
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE index_jobs SET status = 'queued', progress = 0, "
            "error = '服务重启后自动恢复', updated_at = ? WHERE status IN ('parsing','indexing')",
            (_now(),),
        )
    return cursor.rowcount


def _claim_next_job() -> dict | None:
    init_index_job_store()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM index_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE index_jobs SET status = 'parsing', progress = 5, attempts = attempts + 1, "
            "error = '', updated_at = ? WHERE job_id = ?",
            (_now(), row["job_id"]),
        )
        conn.commit()
    return get_index_job(row["job_id"])


def _update_job(job_id: str, *, status: str, progress: int, error: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE index_jobs SET status = ?, progress = ?, error = ?, updated_at = ? WHERE job_id = ?",
            (status, max(0, min(progress, 100)), str(error or "")[:2000], _now(), job_id),
        )


async def _process_job(job: dict) -> None:
    job_id = job["job_id"]
    try:
        if job["operation"] == "reconcile":
            _update_job(job_id, status="indexing", progress=50)
            pdf_dir = Path(job["file_path"])
            physical_paths = [str(path) for path in pdf_dir.glob("*.pdf")]
            store = await get_lightrag_store()
            await store.sync_documents([], physical_paths)
            _update_job(job_id, status="completed", progress=100)
            return

        path = Path(job["file_path"])
        if not path.is_file():
            _update_job(job_id, status="failed", progress=100, error="PDF 文件不存在")
            return
        if job["source"] in list_indexed_sources():
            _update_job(job_id, status="completed", progress=100)
            return

        parsed = await parse_pdfs([str(path)])
        if not parsed:
            _update_job(job_id, status="failed", progress=100, error="PDF 解析未产生有效文本")
            return
        documents = await build_lightrag_documents(parsed)
        if not documents:
            _update_job(job_id, status="failed", progress=100, error="论文身份抽取失败")
            return

        _update_job(job_id, status="indexing", progress=55)
        pdf_dir = Path(_PDF_DIR) if _PDF_DIR else path.parent
        physical_paths = glob.glob(str(pdf_dir / "*.pdf"))
        store = await get_lightrag_store()
        result = await store.sync_documents(documents, physical_paths)
        if job["source"] in result.get("inserted", []) or job["source"] in list_indexed_sources():
            _update_job(job_id, status="completed", progress=100)
        else:
            reason = (result.get("failed", {}) or {}).get(job["source"], "LightRAG 未确认 processed 状态")
            _update_job(job_id, status="failed", progress=100, error=reason)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _update_job(job_id, status="failed", progress=100, error=f"{type(exc).__name__}: {exc}")
        print(f"[INDEX JOB] {job['source']} 失败: {exc}")


async def _worker_loop() -> None:
    while True:
        job = _claim_next_job()
        if job is not None:
            await _process_job(job)
            continue
        _WAKE_EVENT.clear()
        try:
            await asyncio.wait_for(_WAKE_EVENT.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def start_index_worker(pdf_dir: str | Path) -> None:
    global _WORKER_TASK, _WAKE_EVENT, _PDF_DIR
    _PDF_DIR = Path(pdf_dir).resolve()
    init_index_job_store()
    recovered = recover_interrupted_index_jobs()
    physical_paths = glob.glob(str(_PDF_DIR / "*.pdf"))
    jobs = enqueue_pending_index_jobs(physical_paths)
    queued_count = sum(job["status"] == "queued" for job in jobs)
    if recovered or queued_count:
        print(f"[INDEX JOB] 恢复 {recovered} 个任务，新排队 {queued_count} 个任务。")
    if _WAKE_EVENT is None:
        _WAKE_EVENT = asyncio.Event()
    if _WORKER_TASK is None or _WORKER_TASK.done():
        _WORKER_TASK = asyncio.create_task(_worker_loop(), name="lightrag-index-worker")
    _notify_worker()


async def stop_index_worker() -> None:
    global _WORKER_TASK, _WAKE_EVENT
    task = _WORKER_TASK
    _WORKER_TASK = None
    _WAKE_EVENT = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
