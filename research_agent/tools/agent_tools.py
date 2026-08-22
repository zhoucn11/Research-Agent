# agent_tools.py
from langchain_core.tools import tool
from research_agent.core.tool_validation import (
    LocalRetrievalInput,
    PdfUploadInput,
    WebSearchInput,
)


@tool(args_schema=WebSearchInput)
def trigger_web_search(rationale: str, user_core_topic: str, keyword: str, year_range: str = "") -> str:
    """【联网检索必备工具】"""
    pass


@tool(args_schema=PdfUploadInput)
def trigger_pdf_upload() -> str:
    """【仅当】用户明确表示上传了新 PDF，或者要求解析、处理新本地文件时调用。"""
    pass


@tool(args_schema=LocalRetrievalInput)
def trigger_local_retrieval(rationale: str, query: str) -> str:
    """【仅当】用户针对本地已有的文献进行提问时调用。"""
    pass
