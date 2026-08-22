import asyncio
import os
from pathlib import Path

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from research_agent.core.runtime_events import emit_runtime_event, runtime_print as print
from research_agent.core.tool_validation import ToolValidationError, validate_tool_calls
from research_agent.core.state import AgentState
from research_agent.core.tools import (
    assistant_node,
    search_map_node,
    rag_map_node,
    synthesizer_node,
    reviewer_node,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("search_map", search_map_node)
    workflow.add_node("rag_map", rag_map_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("reviewer", reviewer_node)

    def route_assistant(state: AgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            try:
                validated_calls = validate_tool_calls(last_msg.tool_calls)
            except ToolValidationError as exc:
                emit_runtime_event(
                    "tool_validation_rejected",
                    "Graph rejected invalid tool call",
                    validation_error=str(exc),
                )
                return END
            tool_name = validated_calls[0]["name"]
            if tool_name == "trigger_web_search":
                return "search_map"
            if tool_name in ["trigger_pdf_upload", "trigger_local_retrieval"]:
                return "rag_map"

        content = last_msg.content if isinstance(last_msg.content, str) else ""
        if state.get("selected_papers") and "[APPROVE_SYNTHESIS]" in content:
            return "synthesizer"
        return END

    def route_reviewer(state: AgentState):
        if state.get("review_status") == "revise":
            return "synthesizer"
        return END

    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges("assistant", route_assistant)
    workflow.add_edge("search_map", "assistant")
    workflow.add_edge("rag_map", "assistant")
    workflow.add_edge("synthesizer", "reviewer")
    workflow.add_conditional_edges("reviewer", route_reviewer)
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


# CLI 和没有显式启动生命周期的测试继续使用内存版本；FastAPI startup 会切换到持久版本。
agent_app = build_graph()
_PERSISTENT_AGENT_APP = None
_PERSISTENT_SAVER = None
_PERSISTENT_SAVER_CONTEXT = None
_PERSISTENT_INIT_LOCK = None


async def initialize_persistent_agent_app():
    global _PERSISTENT_AGENT_APP, _PERSISTENT_SAVER, _PERSISTENT_SAVER_CONTEXT, _PERSISTENT_INIT_LOCK
    if _PERSISTENT_AGENT_APP is not None:
        return _PERSISTENT_AGENT_APP
    if os.environ.get("AGENT_CHECKPOINT_BACKEND", "sqlite").lower() == "memory":
        _PERSISTENT_AGENT_APP = agent_app
        return agent_app
    if _PERSISTENT_INIT_LOCK is None:
        _PERSISTENT_INIT_LOCK = asyncio.Lock()

    async with _PERSISTENT_INIT_LOCK:
        if _PERSISTENT_AGENT_APP is not None:
            return _PERSISTENT_AGENT_APP
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            print(
                "[SYSTEM] 未安装 langgraph-checkpoint-sqlite，LangGraph 暂时使用内存 checkpoint；"
                "请执行 pip install -r requirements-lightrag.txt。"
            )
            _PERSISTENT_AGENT_APP = agent_app
            return agent_app

        checkpoint_path = Path(
            os.environ.get("AGENT_CHECKPOINT_DB", PROJECT_ROOT / "agent_checkpoints.sqlite3")
        ).expanduser()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        context = AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        saver = await context.__aenter__()
        try:
            await saver.setup()
        except Exception:
            await context.__aexit__(None, None, None)
            raise
        _PERSISTENT_SAVER_CONTEXT = context
        _PERSISTENT_SAVER = saver
        _PERSISTENT_AGENT_APP = build_graph(saver)
        print(f"[SYSTEM] LangGraph checkpoint 已持久化到: {checkpoint_path}")
        return _PERSISTENT_AGENT_APP


async def get_agent_app():
    return _PERSISTENT_AGENT_APP or await initialize_persistent_agent_app()


async def delete_agent_thread(thread_id: str) -> None:
    if not thread_id:
        return
    await initialize_persistent_agent_app()
    active_app = _PERSISTENT_AGENT_APP or agent_app
    saver = _PERSISTENT_SAVER or getattr(active_app, "checkpointer", None)
    delete_method = getattr(saver, "adelete_thread", None) if saver is not None else None
    if delete_method is not None:
        await delete_method(thread_id)


async def finalize_agent_app() -> None:
    global _PERSISTENT_AGENT_APP, _PERSISTENT_SAVER, _PERSISTENT_SAVER_CONTEXT
    context = _PERSISTENT_SAVER_CONTEXT
    _PERSISTENT_AGENT_APP = None
    _PERSISTENT_SAVER = None
    _PERSISTENT_SAVER_CONTEXT = None
    if context is not None:
        await context.__aexit__(None, None, None)
