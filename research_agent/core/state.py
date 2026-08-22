# state.py
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def replaceable_paper_reducer(left: list, right):
    """候选/最终证据默认覆盖，避免多轮 ReAct 搜索结果无限累加。"""
    if right == "CLEAR":
        return []
    if right is None:
        return left or []
    return right


def replaceable_text_reducer(left: str, right):
    if right == "CLEAR":
        return ""
    if right is None:
        return left or ""
    return str(right)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # 工具节点产出的当前候选结果：Search/RAG 只写这里，默认覆盖，不参与最终生成
    candidate_papers: Annotated[list, replaceable_paper_reducer]

    # Assistant 确认后的最终证据：Synthesizer/Reviewer 只读这里
    selected_papers: Annotated[list, replaceable_paper_reducer]

    # 候选证据升级为 selected_papers 前的确定性门控摘要。
    evidence_gate: dict

    # LightRAG 单次查询返回的实体、关系和原文上下文，供 Synthesizer 做跨论文分析。
    graph_evidence: Annotated[str, replaceable_text_reducer]

    draft_review: str
    review_feedback: str
    review_status: str
    review_round: int
    pdf_file_paths: list
    # 结构化长期记忆摘要：由滑动窗口外的历史消息压缩得到
    summary: str
    # 仅在服务重启、LangGraph 内存 checkpoint 尚未恢复时注入最近 SQLite 对话。
    conversation_bootstrap: str
    indexed_files: list


    step_count: int
    research_goal: str
    collected_evidence: str
    pending_questions: str
    user_profile: str
    # 当前 HTTP 请求是否为上一轮语义跟进；用于隔离历史 Tool 协议与标注证据复用。
    is_follow_up: bool
    # auto/quick/deep；只影响检索与生成预算，不改变证据边界。
    research_mode: str
