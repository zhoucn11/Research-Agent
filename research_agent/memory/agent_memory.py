import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from research_agent.core.runtime_events import runtime_print as print

RECENT_DIALOGUE_ROUNDS = 5  # 兼容旧配置；实际压缩由 token 阈值触发。
MEMORY_CONTENT_LIMIT = int(os.environ.get("MEMORY_CONTENT_LIMIT", "2000"))
MEMORY_SUMMARY_LIMIT = int(os.environ.get("MEMORY_SUMMARY_LIMIT", "12000"))
SUMMARY_INPUT_LIMIT = int(os.environ.get("MEMORY_SUMMARY_INPUT_LIMIT", "120000"))


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content or "")


def _message_type(message) -> str:
    return str(getattr(message, "type", "") or getattr(message, "role", ""))


def _message_token_text(message) -> str:
    parts = [_message_type(message), _content_to_text(getattr(message, "content", ""))]
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        parts.append(_content_to_text(tool_calls))
    return "\n".join(parts)


def approximate_tokens(text: str) -> int:
    """中英混合文本的低成本估算；压缩阈值只需要量级，不需要精确计费。"""
    return max(0, (len(text or "") + 2) // 3)


def estimate_tokens(messages: list, summary: str = "", reserved_tokens: int = 0) -> int:
    message_text = "\n".join(_message_token_text(message) for message in messages)
    return approximate_tokens(message_text) + approximate_tokens(summary) + max(0, reserved_tokens)


def clip_memory_text(text: str, limit: int = MEMORY_CONTENT_LIMIT) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.7))
    tail = max(1, limit - head)
    return f"{text[:head]} ...[中间已截断]... {text[-tail:]}"


def format_message_for_memory(message) -> str:
    """把旧消息变成低噪声文本，避免摘要模型再次吞入整段工具日志。"""
    content = clip_memory_text(_content_to_text(getattr(message, "content", "")))
    message_type = _message_type(message)

    if message_type in {"human", "user"} and content:
        return f"用户: {content}"
    if message_type in {"ai", "assistant"}:
        tool_calls = getattr(message, "tool_calls", []) or []
        if tool_calls:
            tool_names = ", ".join(call.get("name", "unknown_tool") for call in tool_calls)
            return f"AI工具决策: 调用 {tool_names}"
        if content:
            return f"AI: {content}"
    if message_type == "tool" and content:
        return f"工具Observation摘要: {content}"
    return ""


def find_recent_window_start(messages: list, rounds: int = RECENT_DIALOGUE_ROUNDS) -> int:
    """兼容旧调用：返回最近 N 个 human 回合的起始位置。"""
    human_seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        if _message_type(messages[idx]) in {"human", "user"}:
            human_seen += 1
            if human_seen == rounds:
                return idx
    return 0


def safe_split_index(messages: list, keep_recent: int) -> int:
    """切分边界不能落在 tool 回复上，否则会产生无对应 tool_calls 的孤儿消息。"""
    split = max(0, len(messages) - max(1, keep_recent))
    while split > 0 and _message_type(messages[split]) == "tool":
        split -= 1
    return split


def _extract_key_info(old_messages: list) -> str:
    """LLM 摘要不可用时的确定性降级，优先留下路径、错误、目标和待办。"""
    lines = [format_message_for_memory(message) for message in old_messages]
    lines = [line for line in lines if line]
    joined = "\n".join(lines)

    path_pattern = r"(?:[A-Za-z]:\\[^\s,;，；]+|/(?:[^\s/]+/)+[^\s,;，；]+|[\w./\\-]+\.(?:py|md|json|ya?ml|toml|pdf|docx))"
    paths = list(dict.fromkeys(re.findall(path_pattern, joined, flags=re.IGNORECASE)))[:20]
    important = [
        line for line in lines
        if re.search(r"error|exception|失败|报错|决定|目标|约束|待办|完成|已确认|下一步", line, re.IGNORECASE)
    ][:24]
    recent_user = [line for line in lines if line.startswith("用户:")][-4:]

    parts = ["### 自动降级记忆"]
    if recent_user:
        parts.append("\n".join(f"- {line}" for line in recent_user))
    if important:
        parts.append("\n".join(f"- {line}" for line in important))
    if paths:
        parts.append("### 涉及文件\n" + "\n".join(f"- {path}" for path in paths))
    if len(parts) == 1:
        parts.append("- 旧对话已压缩，未提取到稳定事实。")
    return "\n\n".join(parts)


async def summarize_memory(current_summary: str, old_messages: list, llm=None) -> str:
    """把窗口外历史合并为结构化长期记忆；失败时退化为规则抽取。"""
    from research_agent.core.llm_clients import get_qwen_llm, safe_llm_invoke

    memory_lines = [format_message_for_memory(message) for message in old_messages]
    memory_lines = [line for line in memory_lines if line]
    if not memory_lines:
        return current_summary

    chat_history_str = "\n".join(memory_lines)
    if len(chat_history_str) > SUMMARY_INPUT_LIMIT:
        chat_history_str = (
            chat_history_str[: int(SUMMARY_INPUT_LIMIT * 0.7)]
            + "\n...[摘要输入中段已截断]...\n"
            + chat_history_str[-int(SUMMARY_INPUT_LIMIT * 0.3):]
        )

    prompt = f"""你是 Agent 的上下文记忆压缩器。请把【已有记忆】与【即将移出窗口的旧消息】合并为新的结构化记忆。

只保留后续任务仍需要的事实：用户目标与硬约束、关键决定及原因、已完成步骤、涉及文件/论文/工具、已确认的证据、出现过的错误及解决状态、未完成待办。删除寒暄、重复讨论、原始日志、整段代码和已失效的中间结果。不要虚构。

严格使用以下结构；没有内容写“无”：
### 用户目标与约束
### 关键决定
### 已完成与证据
### 错误与风险
### 待办事项

【已有记忆】
{current_summary or "无"}

【旧消息】
{chat_history_str}
"""

    print(f"  └─ 🗜️ 上下文达到摘要阈值，折叠 {len(old_messages)} 条旧消息...")
    response = await safe_llm_invoke(
        llm or get_qwen_llm(
            temperature=0.1,
            max_tokens=int(os.environ.get("CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS", "1200")),
        ),
        prompt,
        "Context_Compressor",
        max_retries=1,
    )
    if response and _content_to_text(response.content).strip():
        summary = _content_to_text(response.content).strip()
    else:
        fallback = _extract_key_info(old_messages)
        summary = "\n\n".join(part for part in [current_summary.strip(), fallback] if part)

    if len(summary) > MEMORY_SUMMARY_LIMIT:
        summary = summary[:MEMORY_SUMMARY_LIMIT] + "\n...[长期记忆已截断]"
    return summary


def _copy_message_with_content(message, content: str):
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    cloned = copy.copy(message)
    cloned.content = content
    return cloned


SummaryFunc = Callable[[str, list, object], Awaitable[str]]


@dataclass
class ContextCompressionResult:
    messages: list
    summary: str
    removed_messages: list = field(default_factory=list)
    updated_messages: list = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    estimated_tokens: int = 0


class ContextWindowManager:
    """按截断、摘要、硬折叠三层策略惰性治理 LangGraph 消息历史。"""

    def __init__(
        self,
        max_tokens: int | None = None,
        snip_ratio: float = 0.60,
        summarize_ratio: float = 0.78,
        collapse_ratio: float = 0.90,
        reserved_tokens: int | None = None,
        keep_recent: int = 8,
        collapse_keep_recent: int = 4,
        summarizer: SummaryFunc = summarize_memory,
    ):
        api_context_window = int(os.environ.get("MAIN_API_CONTEXT_WINDOW", "262144"))
        default_max_tokens = max(16384, int(api_context_window * 0.75))
        self.max_tokens = max_tokens or int(os.environ.get("CONTEXT_MAX_TOKENS", str(default_max_tokens)))
        self.snip_at = int(self.max_tokens * snip_ratio)
        self.summarize_at = int(self.max_tokens * summarize_ratio)
        self.collapse_at = int(self.max_tokens * collapse_ratio)
        self.reserved_tokens = (
            reserved_tokens
            if reserved_tokens is not None
            else int(
                os.environ.get(
                    "CONTEXT_RESERVED_TOKENS",
                    str(min(16000, max(4096, self.max_tokens // 12))),
                )
            )
        )
        self.keep_recent = keep_recent
        self.collapse_keep_recent = collapse_keep_recent
        self.summarizer = summarizer

    def _snip_tool_outputs(self, messages: list) -> tuple[list, list]:
        protected_from = max(0, len(messages) - self.keep_recent)
        prepared = list(messages)
        updates = []

        for index, message in enumerate(messages[:protected_from]):
            if _message_type(message) != "tool":
                continue
            content = _content_to_text(getattr(message, "content", ""))
            if len(content) <= 1500:
                continue

            lines = content.splitlines()
            if len(lines) > 6:
                clipped = (
                    "\n".join(lines[:3])
                    + f"\n... ({len(lines)} lines, 历史工具输出已截断) ...\n"
                    + "\n".join(lines[-3:])
                )
            else:
                clipped = content[:700] + "\n...[历史工具输出中段已截断]...\n" + content[-700:]

            cloned = _copy_message_with_content(message, clipped)
            prepared[index] = cloned
            if getattr(cloned, "id", None):
                updates.append(cloned)
        return prepared, updates

    async def prepare(self, messages: list, current_summary: str = "", llm=None) -> ContextCompressionResult:
        prepared_messages = list(messages)
        active_originals = list(messages)
        summary = current_summary or ""
        removed = []
        updated = []
        actions = []

        current = estimate_tokens(prepared_messages, summary, self.reserved_tokens)

        if current > self.snip_at:
            prepared_messages, updated = self._snip_tool_outputs(prepared_messages)
            if updated:
                actions.append("snip")
            current = estimate_tokens(prepared_messages, summary, self.reserved_tokens)

        if current > self.summarize_at and len(prepared_messages) > self.keep_recent + 2:
            split = safe_split_index(prepared_messages, self.keep_recent)
            if split > 0:
                summary = await self.summarizer(summary, prepared_messages[:split], llm)
                removed.extend(active_originals[:split])
                prepared_messages = prepared_messages[split:]
                active_originals = active_originals[split:]
                actions.append("summarize")
                current = estimate_tokens(prepared_messages, summary, self.reserved_tokens)

        if current > self.collapse_at and len(prepared_messages) > self.collapse_keep_recent + 1:
            split = safe_split_index(prepared_messages, self.collapse_keep_recent)
            if split > 0:
                summary = await self.summarizer(summary, prepared_messages[:split], llm)
                removed.extend(active_originals[:split])
                prepared_messages = prepared_messages[split:]
                active_originals = active_originals[split:]
                actions.append("collapse")
                current = estimate_tokens(prepared_messages, summary, self.reserved_tokens)

        removed_ids = {getattr(message, "id", None) for message in removed}
        updated = [message for message in updated if getattr(message, "id", None) not in removed_ids]

        return ContextCompressionResult(
            messages=prepared_messages,
            summary=summary,
            removed_messages=removed,
            updated_messages=updated,
            actions=actions,
            estimated_tokens=current,
        )


context_window_manager = ContextWindowManager()
