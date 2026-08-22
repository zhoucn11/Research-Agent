import asyncio
import concurrent.futures
import multiprocessing
import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as tqdm_asyncio

from research_agent.core.llm_clients import LOCAL_ROLE, get_local_llm, safe_llm_invoke
from research_agent.core.runtime_events import runtime_print as print
from research_agent.retrieval.lightrag_store import LightRAGDocument


try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except ImportError:
    RAGFlowPdfParser = None


class PaperFingerprint(BaseModel):
    official_title: str = Field(default="未知", description="论文正文中的大标题")
    official_year: str = Field(default="未知", description="发表年份")


def _deepdoc_node_to_text(node) -> str:
    if node is None or isinstance(node, bytes):
        return ""
    if isinstance(node, str):
        return node
    if hasattr(node, "text"):
        return str(node.text or "")
    if hasattr(node, "mode") and hasattr(node, "size"):
        return ""
    if isinstance(node, dict):
        preferred_keys = ("text", "html", "caption", "table", "content", "description")
        preferred = [_deepdoc_node_to_text(node.get(key)) for key in preferred_keys if key in node]
        if preferred:
            return "\n".join(part for part in preferred if part)
        return "\n".join(_deepdoc_node_to_text(value) for value in node.values())
    if isinstance(node, (list, tuple, set)):
        return "\n".join(filter(None, (_deepdoc_node_to_text(item) for item in node)))
    text = str(node)
    return "" if text.startswith("<") and "object at 0x" in text else text


def _deepdoc_position_tag_to_page_marker(match: re.Match) -> str:
    return f" [page:{match.group(1).split('-')[0]}] "


def _normalize_paper_text(text: str) -> str:
    text = re.sub(r"@@([0-9-]+)\t[0-9.\t-]+?##", _deepdoc_position_tag_to_page_marker, text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text_with_deepdoc(pdf_path: str) -> str:
    if RAGFlowPdfParser is None:
        print("  └─ ⚠️ DeepDoc parser 未成功导入。")
        return ""
    try:
        parsed = RAGFlowPdfParser()(pdf_path)
        if isinstance(parsed, tuple):
            main_text = _deepdoc_node_to_text(parsed[0])
            auxiliary_text = _deepdoc_node_to_text(parsed[1:])
            parsed = "\n\n".join(part for part in (main_text, auxiliary_text) if part)
        return _normalize_paper_text(_deepdoc_node_to_text(parsed))
    except Exception as exc:
        print(f"  └─ ⚠️ DeepDoc 解析失败: {os.path.basename(pdf_path)}: {exc}")
        return ""


def parse_pdf_sync(pdf_path: str) -> str:
    return extract_text_with_deepdoc(pdf_path)


async def parse_pdfs(pdf_paths: list[str]) -> list[tuple[str, str]]:
    if not pdf_paths:
        return []
    loop = asyncio.get_running_loop()
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp_context) as pool:
        tasks = [loop.run_in_executor(pool, parse_pdf_sync, path) for path in pdf_paths]
        texts = await tqdm_asyncio.gather(*tasks, desc="📄 解析增量 PDF", colour="blue")
    return [(path, text) for path, text in zip(pdf_paths, texts) if text]


async def _fingerprint_paper(path: str, text: str, structured_llm) -> LightRAGDocument:
    file_name = os.path.basename(path)
    messages = [
        SystemMessage(content="你是学术文献指纹提取器，必须严格输出合法 JSON。"),
        HumanMessage(content=f"""从论文首页片段提取标题和发表年份。
文件名：{file_name}
首页片段：
{text[:2500]}

规则：official_title 必须来自正文标题，找不到用文件名；official_year 找不到填“未知”。"""),
    ]
    fingerprint = await safe_llm_invoke(
        structured_llm,
        messages,
        f"LightRAG指纹-{file_name}",
        max_retries=1,
        role=LOCAL_ROLE,
    )
    title = (
        fingerprint.official_title
        if fingerprint and fingerprint.official_title and fingerprint.official_title != "未知"
        else Path(file_name).stem
    )
    year = fingerprint.official_year if fingerprint and fingerprint.official_year else "未知"
    index_text = f"""[PAPER_METADATA]
source: {file_name}
title: {title}
year: {year}
[/PAPER_METADATA]

{text}"""
    return LightRAGDocument(
        source=file_name,
        path=str(Path(path).resolve()),
        text=index_text,
        title=title,
        year=year,
    )


async def build_lightrag_documents(parsed_papers: list[tuple[str, str]]) -> list[LightRAGDocument]:
    if not parsed_papers:
        return []
    fingerprint_llm = get_local_llm(
        temperature=0,
        max_tokens=int(os.environ.get("RAG_FINGERPRINT_MAX_OUTPUT_TOKENS", "256")),
    ).with_structured_output(PaperFingerprint, method="json_mode")
    tasks = [_fingerprint_paper(path, text, fingerprint_llm) for path, text in parsed_papers]
    return [
        document
        for document in await tqdm_asyncio.gather(*tasks, desc="🧬 论文身份抽取", colour="cyan")
        if document.text.strip()
    ]
