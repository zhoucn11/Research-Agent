import asyncio
import os
import re
import time
from typing import Any, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.core.runtime_events import emit_runtime_event, runtime_print


MAIN_API_ROLE = "main_api"
REVIEWER_API_ROLE = "reviewer_api"
LOCAL_ROLE = "local"

_ROLE_SEMAPHORES: dict[tuple[str, int], asyncio.Semaphore] = {}
_CIRCUIT_STATE: dict[str, dict[str, float]] = {}


def _qwen_extra_body(model_name: str, thinking_budget: int | None = None) -> dict | None:
    if "qwen" not in model_name.lower():
        return None
    thinking_enabled = os.environ.get("QWEN_ENABLE_THINKING", "false").lower() in {
        "1", "true", "yes", "on",
    }
    extra_body = {"enable_thinking": thinking_enabled}
    if thinking_enabled and thinking_budget is not None:
        extra_body["thinking_budget"] = max(1, int(thinking_budget))
    return extra_body


def _main_llm(
    temperature: float,
    streaming: bool,
    max_tokens: int,
    timeout: int,
    thinking_budget: int | None = None,
) -> ChatOpenAI:
    model_name = os.environ.get("OPENAI_MODEL", "qwen-plus")
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=temperature,
        timeout=timeout,
        streaming=streaming,
        max_tokens=max_tokens,
        extra_body=_qwen_extra_body(model_name, thinking_budget),
    )


def get_qwen_llm(temperature=0.0, streaming=False, max_tokens: int | None = None):
    """主 Agent、上下文压缩和证据提炼共用的 Qwen 远程 API。"""
    timeout = int(os.environ.get("LLM_TIMEOUT", 300))
    output_limit = max_tokens or int(os.environ.get("QWEN_MAX_OUTPUT_TOKENS", "1600"))
    return _main_llm(temperature, streaming, output_limit, timeout)


def get_synthesis_llm(
    temperature=0.1,
    streaming=False,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
):
    """综述 Agent 沿用主 Qwen API，只单独配置超时和输出预算。"""
    timeout = int(os.environ.get("SYNTHESIS_TIMEOUT", os.environ.get("LLM_TIMEOUT", 300)))
    output_limit = max_tokens or int(os.environ.get("SYNTHESIS_MAX_OUTPUT_TOKENS", "8192"))
    return _main_llm(temperature, streaming, output_limit, timeout, thinking_budget)


def get_reviewer_llm(temperature=0.0, streaming=False, max_tokens: int | None = None):
    """Reviewer 独占 Kimi API；不静默回退到主模型，避免审阅失去独立性。"""
    model_name = (os.environ.get("REVIEWER_MODEL") or "").strip()
    api_key = (os.environ.get("REVIEWER_API_KEY") or "").strip()
    base_url = (os.environ.get("REVIEWER_BASE_URL") or "").strip()
    missing = [
        name for name, value in (
            ("REVIEWER_MODEL", model_name),
            ("REVIEWER_API_KEY", api_key),
            ("REVIEWER_BASE_URL", base_url),
        )
        if not value or "PLACEHOLDER" in value.upper()
    ]
    if missing:
        raise RuntimeError("Reviewer 已启用但缺少独立配置: " + ", ".join(missing))

    timeout = int(os.environ.get("REVIEWER_TIMEOUT", os.environ.get("LLM_TIMEOUT", 300)))
    output_limit = max_tokens or int(os.environ.get("REVIEWER_MAX_OUTPUT_TOKENS", "4096"))
    is_kimi_k2 = model_name.lower() in {"kimi-k2.5", "kimi-k2.6"}
    client_options = dict(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        streaming=streaming,
        max_tokens=output_limit,
        max_retries=0,
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if is_kimi_k2
            else _qwen_extra_body(model_name)
        ),
    )
    client_options["temperature"] = 0.6 if is_kimi_k2 else temperature
    return ChatOpenAI(**client_options)


def get_local_llm(temperature=0.0, streaming=False, max_tokens: int | None = None):
    """本地 vLLM 只承担批量、低风险、可确定性兜底的辅助任务。"""
    output_limit = max_tokens or int(os.environ.get("LOCAL_LLM_MAX_OUTPUT_TOKENS", "2048"))
    return ChatOpenAI(
        model=os.environ.get("LOCAL_LLM_MODEL", "qwen3"),
        api_key=os.environ.get("LOCAL_LLM_API_KEY", "sk-local"),
        base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6006/v1"),
        temperature=temperature,
        streaming=streaming,
        max_tokens=output_limit,
        timeout=int(os.environ.get("LOCAL_LLM_TIMEOUT", 180)),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        model_kwargs={"frequency_penalty": 1.2, "presence_penalty": 1.2},
    )


def _role_concurrency(role: str) -> int:
    env_name = {
        MAIN_API_ROLE: "MAIN_API_MAX_CONCURRENCY",
        REVIEWER_API_ROLE: "REVIEWER_API_MAX_CONCURRENCY",
        LOCAL_ROLE: "LOCAL_LLM_MAX_CONCURRENCY",
    }.get(role, "LLM_MAX_CONCURRENCY")
    fallback = "2" if role == MAIN_API_ROLE else "1"
    return max(1, int(os.environ.get(env_name, os.environ.get("LLM_MAX_CONCURRENCY", fallback))))


def _role_semaphore(role: str) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = (role, id(loop))
    semaphore = _ROLE_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_role_concurrency(role))
        _ROLE_SEMAPHORES[key] = semaphore
    return semaphore


def _status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    value = direct or getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"(?:status|error code)\D{0,8}(\d{3})", str(exc), re.I)
    return int(match.group(1)) if match else None


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in {408, 409, 425, 429} or status >= 500
    return not isinstance(exc, (ValueError, TypeError, PermissionError))


def _is_output_truncation(exc: Exception) -> bool:
    text = str(exc).casefold()
    return (
        "length limit was reached" in text
        or "finish_reason=length" in text
        or "lengthfinishreasonerror" in type(exc).__name__.casefold()
    )


def _circuit(role: str) -> dict[str, float]:
    return _CIRCUIT_STATE.setdefault(role, {"failures": 0.0, "open_until": 0.0})


def _circuit_is_open(role: str) -> bool:
    state = _circuit(role)
    if state["open_until"] <= time.monotonic():
        if state["open_until"]:
            state.update(failures=0.0, open_until=0.0)
        return False
    return True


def _record_success(role: str) -> None:
    _circuit(role).update(failures=0.0, open_until=0.0)


def _record_failure(role: str) -> None:
    state = _circuit(role)
    state["failures"] += 1
    threshold = max(1, int(os.environ.get("LLM_CIRCUIT_FAILURE_THRESHOLD", "3")))
    if state["failures"] >= threshold:
        cooldown = max(1.0, float(os.environ.get("LLM_CIRCUIT_COOLDOWN_SECONDS", "60")))
        state["open_until"] = time.monotonic() + cooldown


def _usage_fields(response: Any) -> dict:
    if isinstance(response, dict) and response.get("raw") is not None:
        response = response["raw"]
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    fields = {}
    for source_key, target_key in (
        ("input_tokens", "input_tokens"),
        ("prompt_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source_key) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)) and target_key not in fields:
            fields[target_key] = int(value)
    return fields


def response_finish_reason(response: Any) -> str:
    """兼容 LangChain/OpenAI 响应结构，提取服务端的停止原因。"""
    if isinstance(response, dict) and response.get("raw") is not None:
        response = response["raw"]
    metadata = getattr(response, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    if not reason and isinstance(metadata, dict):
        choices = metadata.get("choices") or []
        if choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
    if not reason:
        additional = getattr(response, "additional_kwargs", None) or {}
        reason = additional.get("finish_reason") if isinstance(additional, dict) else None
    return str(reason or "").strip().casefold()


def response_was_truncated(response: Any) -> bool:
    """HTTP 200 不代表生成完整；length/max_tokens 都视为明确截断。"""
    return response_finish_reason(response) in {"length", "max_tokens"}


async def safe_llm_invoke(
    structured_llm,
    prompt: Union[str, List[Any]],
    task_name: str,
    max_retries: int = 3,
    *,
    role: str = MAIN_API_ROLE,
    invoke_config: dict | None = None,
    error_sink: list[Exception] | None = None,
):
    """统一模型调用策略：角色级限流、分类重试、指数退避和轻量熔断。"""
    if error_sink is not None:
        error_sink.clear()
    if _circuit_is_open(role):
        runtime_print(f"  [LLM熔断] {task_name} 跳过调用：{role} 仍在冷却期。")
        emit_runtime_event("llm_circuit_open", f"{task_name} circuit open", role=role)
        return None

    attempts = max(1, int(max_retries))
    async with _role_semaphore(role):
        started_at = time.perf_counter()
        emit_runtime_event("llm_call_start", f"{task_name} started", role=role, max_attempts=attempts)
        for attempt in range(attempts):
            try:
                if invoke_config is None:
                    response = await structured_llm.ainvoke(prompt)
                else:
                    response = await structured_llm.ainvoke(prompt, invoke_config)
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                _record_success(role)
                finish_reason = response_finish_reason(response)
                emit_runtime_event(
                    "llm_call_end",
                    f"{task_name} completed",
                    role=role,
                    attempts=attempt + 1,
                    latency_ms=elapsed_ms,
                    **({"finish_reason": finish_reason} if finish_reason else {}),
                    **_usage_fields(response),
                )
                return response
            except Exception as exc:
                output_truncated = _is_output_truncation(exc)
                retryable = _is_retryable(exc) and not output_truncated
                status = _status_code(exc)
                error_snippet = str(exc)[:180].replace("\n", " ")
                runtime_print(
                    f"  [LLM调用] {task_name} 失败 (第 {attempt + 1}/{attempts} 次, "
                    f"role={role}, status={status or 'n/a'}): {error_snippet}"
                )
                if output_truncated:
                    emit_runtime_event(
                        "llm_output_truncated",
                        f"{task_name} output truncated",
                        role=role,
                        attempt=attempt + 1,
                        status_code=status,
                    )
                should_retry = retryable and attempt < attempts - 1
                if should_retry:
                    base = max(0.1, float(os.environ.get("LLM_RETRY_BASE_SECONDS", "1")))
                    delay = _retry_after_seconds(exc)
                    delay = delay if delay is not None else min(12.0, base * (2 ** attempt))
                    emit_runtime_event(
                        "llm_retry",
                        f"{task_name} retry scheduled",
                        role=role,
                        attempt=attempt + 1,
                        status_code=status,
                        retry_after_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 参数、结构化解析和长度截断属于单次请求问题，不能熔断整个模型角色。
                if retryable:
                    _record_failure(role)
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                emit_runtime_event(
                    "llm_call_error",
                    f"{task_name} failed: {type(exc).__name__}",
                    role=role,
                    attempts=attempt + 1,
                    status_code=status,
                    retryable=retryable,
                    latency_ms=elapsed_ms,
                )
                if error_sink is not None:
                    error_sink.append(exc)
                return None
    return None


def reset_llm_runtime_state() -> None:
    """测试与运维探针使用，不触碰任何模型服务端状态。"""
    _ROLE_SEMAPHORES.clear()
    _CIRCUIT_STATE.clear()


async def prewarm_local_prefix_cache(stable_system_prompt: str) -> None:
    """仅供明确的本地批处理前缀预热；主路由已迁移到远程 Qwen。"""
    llm = get_local_llm(temperature=0, streaming=False, max_tokens=1)
    await safe_llm_invoke(
        llm,
        [SystemMessage(content=stable_system_prompt), HumanMessage(content="健康检查：只回复 OK。")],
        "Local_Prefix_Prewarm",
        max_retries=1,
        role=LOCAL_ROLE,
    )
