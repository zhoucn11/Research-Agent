import asyncio


_BACKGROUND_CHAT_TASKS: set[asyncio.Task] = set()


def track_background_task(task: asyncio.Task) -> asyncio.Task:
    """保留后台聊天任务的强引用，使 SSE 断开不会连带取消 Agent。"""
    _BACKGROUND_CHAT_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_CHAT_TASKS.discard)
    return task


def approved_token_events(text: str, trace_id: str, chunk_size: int = 48) -> list[dict]:
    """只把已审阅且已落库的最终正文切成 token 事件。"""
    content = str(text or "")
    if not content:
        return []
    if len(content) > 1:
        chunk_size = min(max(1, chunk_size), max(1, len(content) // 2))
    else:
        chunk_size = 1
    return [
        {
            "type": "token",
            "content": content[offset:offset + chunk_size],
            "node": "final",
            "stream_id": "final:approved",
            "trace_id": trace_id,
        }
        for offset in range(0, len(content), chunk_size)
    ]
