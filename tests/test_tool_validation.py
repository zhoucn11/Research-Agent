import pytest

from research_agent.core.tool_validation import (
    ToolValidationError,
    validate_tool_call,
    validate_tool_calls,
)


def _web_call(**overrides):
    args = {
        "rationale": "用户明确要求联网检索最新论文。",
        "user_core_topic": "检索 Transformer 高效架构",
        "keyword": "efficient transformer architecture",
        "year_range": "2024-2026",
    }
    args.update(overrides)
    return {"name": "trigger_web_search", "args": args, "id": "call-1", "type": "tool_call"}


def test_web_tool_normalizes_and_accepts_strict_input():
    validated = validate_tool_call(_web_call(keyword=" efficient   transformer architecture "))
    assert validated["args"]["keyword"] == "efficient transformer architecture"


@pytest.mark.parametrize("keyword", [
    "transformer",
    "this is a complete sentence search",
    "transformer AND mamba",
    '"transformer" comparison',
])
def test_web_tool_rejects_invalid_keyword(keyword):
    with pytest.raises(ToolValidationError):
        validate_tool_call(_web_call(keyword=keyword))


def test_web_tool_rejects_unknown_fields_and_invalid_year_order():
    call = _web_call(year_range="2026-2020")
    call["args"]["unexpected"] = True
    with pytest.raises(ToolValidationError):
        validate_tool_call(call)


def test_local_summary_all_must_be_exact_and_only_one_tool_per_turn():
    local_call = {
        "name": "trigger_local_retrieval",
        "args": {"rationale": "用户要求总结整个本地论文知识库。", "query": "SUMMARY_ALL plus"},
        "id": "call-local",
    }
    with pytest.raises(ToolValidationError):
        validate_tool_call(local_call)
    with pytest.raises(ToolValidationError):
        validate_tool_calls([_web_call(), _web_call()])
