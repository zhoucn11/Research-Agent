from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WebSearchInput(StrictToolInput):
    rationale: str = Field(
        min_length=8,
        max_length=600,
        description="说明为何必须联网，以及重试时如何缩短、放宽或拆分了关键词。",
    )
    user_core_topic: str = Field(
        min_length=2,
        max_length=1000,
        description="保留用户完整研究主题；复杂意图放这里，不要塞进 keyword。",
    )
    keyword: str = Field(
        min_length=2,
        max_length=240,
        description="通常是 2-5 个英文关键词；精确标题检索可使用 user_core_topic 中《》包裹的完整标题。",
    )
    year_range: str = Field(default="", max_length=9, description="空字符串、YYYY 或 YYYY-YYYY。")

    @model_validator(mode="after")
    def validate_keyword(self):
        value = " ".join(self.keyword.split())
        explicit_titles = re.findall(r"《([^》]{4,200})》", self.user_core_topic)
        normalize = lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())
        if any(normalize(value) == normalize(title) for title in explicit_titles):
            self.keyword = value
            return self

        words = value.split()
        if not 2 <= len(words) <= 5:
            raise ValueError("keyword 必须包含 2-5 个英文关键词")
        if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", word) for word in words):
            raise ValueError("keyword 只能包含英文、数字、点和连字符，不允许引号、括号或运算符")
        if any(word.casefold() in {"and", "or"} for word in words):
            raise ValueError("keyword 不允许 AND/OR 布尔运算符")
        self.keyword = " ".join(words)
        return self

    @field_validator("year_range")
    @classmethod
    def validate_year_range(cls, value: str) -> str:
        if not value:
            return ""
        if not re.fullmatch(r"\d{4}(?:-\d{4})?", value):
            raise ValueError("year_range 只能是 YYYY 或 YYYY-YYYY")
        years = [int(item) for item in value.split("-")]
        max_year = datetime.now(timezone.utc).year + 1
        if any(year < 1900 or year > max_year for year in years):
            raise ValueError(f"year_range 必须位于 1900-{max_year}")
        if len(years) == 2 and years[0] > years[1]:
            raise ValueError("year_range 起始年份不能晚于结束年份")
        return value


class LocalRetrievalInput(StrictToolInput):
    rationale: str = Field(
        min_length=8,
        max_length=600,
        description="说明为什么查本地文献，以及需要解决的具体问题。",
    )
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="核心问题；总结整个本地库时必须严格传 SUMMARY_ALL。",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in {"\n", "\t"} for char in value):
            raise ValueError("query 含有非法控制字符")
        if "SUMMARY_ALL" in value and value != "SUMMARY_ALL":
            raise ValueError("全库总结时 query 必须严格等于 SUMMARY_ALL")
        return value


class PdfUploadInput(StrictToolInput):
    pass


TOOL_INPUT_SCHEMAS = {
    "trigger_web_search": WebSearchInput,
    "trigger_local_retrieval": LocalRetrievalInput,
    "trigger_pdf_upload": PdfUploadInput,
}


class ToolValidationError(ValueError):
    pass


def validate_tool_call(tool_call: dict) -> dict:
    if not isinstance(tool_call, dict):
        raise ToolValidationError("tool_call 必须是对象")
    name = str(tool_call.get("name") or "")
    schema = TOOL_INPUT_SCHEMAS.get(name)
    if schema is None:
        raise ToolValidationError(f"未知工具: {name or '<empty>'}")
    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        raise ToolValidationError(f"{name}.args 必须是对象")
    try:
        normalized_args = schema.model_validate(args).model_dump()
    except ValueError as exc:
        error_items = getattr(exc, "errors", lambda **_: [])(include_url=False, include_input=False)
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg', 'invalid')}"
            for item in error_items[:5]
        ) or type(exc).__name__
        raise ToolValidationError(f"{name} 参数不合法: {detail}") from exc
    normalized = dict(tool_call)
    normalized["name"] = name
    normalized["args"] = normalized_args
    if not normalized.get("id"):
        raise ToolValidationError(f"{name} 缺少 tool_call id")
    normalized.setdefault("type", "tool_call")
    return normalized


def validate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    calls = list(tool_calls or [])
    if len(calls) > 1:
        raise ToolValidationError("每轮最多允许调用一个工具")
    return [validate_tool_call(call) for call in calls]


def summarize_tool_call_for_trace(tool_call: dict) -> dict:
    """保留可诊断字段，不把用户完整问题或 rationale 写入轨迹库。"""
    name = str((tool_call or {}).get("name") or "")
    args = (tool_call or {}).get("args") or {}
    if name == "trigger_web_search":
        return {
            "keyword": str(args.get("keyword") or "")[:120],
            "year_range": str(args.get("year_range") or "")[:9],
            "topic_chars": len(str(args.get("user_core_topic") or "")),
        }
    if name == "trigger_local_retrieval":
        query = str(args.get("query") or "")
        return {
            "query_kind": "summary_all" if query == "SUMMARY_ALL" else "scoped_query",
            "query_chars": len(query),
        }
    return {}
