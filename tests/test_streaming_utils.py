import asyncio
from pathlib import Path

from research_agent.api.streaming_utils import approved_token_events, track_background_task


def test_approved_tokens_reconstruct_final_text():
    final_text = "## 核心方法\n\n这是已经通过审阅并完成落库的最终正文。"
    events = approved_token_events(final_text, "trace-test", chunk_size=8)

    assert len(events) >= 2
    assert "".join(event["content"] for event in events) == final_text
    assert {event["stream_id"] for event in events} == {"final:approved"}
    assert {event["trace_id"] for event in events} == {"trace-test"}


def test_background_agent_survives_unrelated_consumer_cancellation():
    async def scenario():
        release = asyncio.Event()
        persisted = []

        async def background_agent():
            await release.wait()
            persisted.append("assistant final")

        task = track_background_task(asyncio.create_task(background_agent()))
        consumer = asyncio.create_task(asyncio.sleep(3600))
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        release.set()
        await task
        return persisted

    assert asyncio.run(scenario()) == ["assistant final"]


def test_server_does_not_cancel_background_agent_on_sse_disconnect():
    root = Path(__file__).resolve().parents[1]
    source = (root / "research_agent" / "api" / "server.py").read_text(encoding="utf-8")

    assert "agent_task.cancel()" not in source
    assert "track_background_task(asyncio.create_task(complete_agent_run()))" in source
