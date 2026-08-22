import builtins
import asyncio
import contextvars
import functools
import inspect
import time
from contextlib import contextmanager
from typing import Callable


_EVENT_SINK = contextvars.ContextVar("research_agent_event_sink", default=None)
_TRACE_ID = contextvars.ContextVar("research_agent_trace_id", default="")
_SESSION_ID = contextvars.ContextVar("research_agent_session_id", default="")
_NODE_NAME = contextvars.ContextVar("research_agent_node_name", default="")


def emit_runtime_event(event: str, content: str, **fields) -> None:
    sink = _EVENT_SINK.get()
    if sink is None:
        return
    payload = {
        "event": event,
        "content": str(content or ""),
        "trace_id": _TRACE_ID.get(),
        "session_id": _SESSION_ID.get(),
        "node": _NODE_NAME.get(),
        "timestamp": time.time(),
        **fields,
    }
    sink(payload)


def runtime_print(*values, sep=" ", end="\n", file=None, flush=False) -> None:
    """保留控制台输出，同时把当前请求日志发送到请求级事件队列。"""
    builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
    message = sep.join(str(value) for value in values).strip("\r\n")
    if message:
        emit_runtime_event("log", message)


async def emit_visible_text(text: str, *, node: str, chunk_size: int = 32) -> None:
    """把已经通过确定性/结构化审阅的最终文本按块发送，保证 token 拼接等于 final。"""
    content = str(text or "")
    if not content:
        return
    stream_id = f"{node}:approved"
    for offset in range(0, len(content), max(1, chunk_size)):
        emit_runtime_event(
            "visible_token",
            content[offset:offset + chunk_size],
            stream_id=stream_id,
        )
        await asyncio.sleep(0)


@contextmanager
def bind_runtime_events(trace_id: str, session_id: str, sink: Callable[[dict], None]):
    sink_token = _EVENT_SINK.set(sink)
    trace_token = _TRACE_ID.set(trace_id)
    session_token = _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.reset(session_token)
        _TRACE_ID.reset(trace_token)
        _EVENT_SINK.reset(sink_token)


@contextmanager
def runtime_node_scope(node_name: str):
    token = _NODE_NAME.set(node_name)
    try:
        yield
    finally:
        _NODE_NAME.reset(token)


def instrument_node(node_name: str):
    """为 LangGraph 节点增加结构化开始/结束/失败事件，不改变原函数签名。"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            with runtime_node_scope(node_name):
                emit_runtime_event("node_start", f"{node_name} started")
                try:
                    result = func(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    emit_runtime_event(
                        "node_end",
                        f"{node_name} completed in {elapsed_ms / 1000:.2f}s",
                        latency_ms=elapsed_ms,
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    emit_runtime_event(
                        "node_error",
                        f"{node_name} failed: {type(exc).__name__}: {exc}",
                        latency_ms=elapsed_ms,
                    )
                    raise
        return wrapper
    return decorator
