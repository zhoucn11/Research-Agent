import asyncio
import importlib.util
import sys
import types
from pathlib import Path


class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Message:
    def __init__(self, content="ok"):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


def _load_llm_clients(monkeypatch):
    fake_provider = types.ModuleType("langchain_openai")
    fake_provider.ChatOpenAI = _FakeChatOpenAI
    fake_core = types.ModuleType("langchain_core")
    fake_messages = types.ModuleType("langchain_core.messages")
    fake_messages.HumanMessage = _Message
    fake_messages.SystemMessage = _Message
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_provider)
    monkeypatch.setitem(sys.modules, "langchain_core", fake_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", fake_messages)
    path = Path(__file__).parents[1] / "research_agent" / "core" / "llm_clients.py"
    spec = importlib.util.spec_from_file_location("llm_clients_resilience_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retryable_429_is_retried_once(monkeypatch):
    module = _load_llm_clients(monkeypatch)
    sleeps = []

    class RateLimitError(Exception):
        status_code = 429
        response = types.SimpleNamespace(headers={"Retry-After": "0"})

    class FlakyLLM:
        calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("rate limited")
            return _Message()

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    llm = FlakyLLM()
    result = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=2))
    assert result.content == "ok"
    assert llm.calls == 2
    assert sleeps == [0.0]


def test_non_retryable_error_does_not_open_role_circuit(monkeypatch):
    module = _load_llm_clients(monkeypatch)
    monkeypatch.setenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "1")

    class InvalidLLM:
        calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            raise ValueError("invalid request")

    llm = InvalidLLM()
    first = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=3))
    second = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=3))
    assert first is None and second is None
    assert llm.calls == 2


def test_output_truncation_is_not_retried_or_counted_by_circuit(monkeypatch):
    module = _load_llm_clients(monkeypatch)
    monkeypatch.setenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "1")

    class LengthLimitError(Exception):
        status_code = 200

    class TruncatedLLM:
        calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            raise LengthLimitError("Could not parse response content as the length limit was reached")

    llm = TruncatedLLM()
    first = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=2))
    second = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=2))

    assert first is None and second is None
    assert llm.calls == 2


def test_terminal_server_error_opens_role_circuit(monkeypatch):
    module = _load_llm_clients(monkeypatch)
    monkeypatch.setenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "1")

    class ServerError(Exception):
        status_code = 503

    class UnavailableLLM:
        calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            raise ServerError("service unavailable")

    llm = UnavailableLLM()
    first = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=1))
    second = asyncio.run(module.safe_llm_invoke(llm, "prompt", "task", max_retries=1))

    assert first is None and second is None
    assert llm.calls == 1
