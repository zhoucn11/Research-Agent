VISIBLE_STREAM_NODES = frozenset({"assistant", "synthesizer", "reviewer"})
VISIBLE_STREAM_TAGS = {
    "assistant": "assistant_visible",
    "synthesizer": "synthesizer_visible",
    "reviewer": "reviewer_visible",
}


def message_chunk_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def token_event_from_chunk(message, metadata: dict) -> dict | None:
    node = str((metadata or {}).get("langgraph_node") or "")
    if node not in VISIBLE_STREAM_NODES:
        return None
    tags = set((metadata or {}).get("tags") or [])
    if VISIBLE_STREAM_TAGS[node] not in tags:
        return None

    content = message_chunk_text(message)
    if not content:
        return None

    step = (metadata or {}).get("langgraph_step", "0")
    return {
        "type": "token",
        "content": content,
        "node": node,
        "stream_id": f"{node}:{step}",
    }
