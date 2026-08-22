from types import SimpleNamespace

from research_agent.api.streaming_utils import message_chunk_text, token_event_from_chunk


def test_message_chunk_text_supports_text_blocks():
    message = SimpleNamespace(
        content=[
            {"type": "text", "text": "你"},
            {"type": "output_text", "text": "好"},
            {"type": "tool_call_chunk", "args": "{}"},
        ]
    )

    assert message_chunk_text(message) == "你好"


def test_token_event_only_exposes_tagged_visible_nodes():
    message = SimpleNamespace(content="你好")

    assert token_event_from_chunk(
        message,
        {"langgraph_node": "assistant", "langgraph_step": 1, "tags": ["context_summary"]},
    ) is None
    assert token_event_from_chunk(
        message,
        {"langgraph_node": "synthesizer", "langgraph_step": 2, "tags": ["assistant_visible"]},
    ) is None
    assert token_event_from_chunk(
        message,
        {"langgraph_node": "assistant", "langgraph_step": 3, "tags": ["assistant_visible"]},
    ) == {
        "type": "token",
        "content": "你好",
        "node": "assistant",
        "stream_id": "assistant:3",
    }
    assert token_event_from_chunk(
        message,
        {"langgraph_node": "synthesizer", "langgraph_step": 4, "tags": ["synthesizer_visible"]},
    ) == {
        "type": "token",
        "content": "你好",
        "node": "synthesizer",
        "stream_id": "synthesizer:4",
    }
