import asyncio
import base64
import mimetypes
import os
from pathlib import Path

import requests


IMAGE_PROMPT = (
    "你是研究助手的多模态预处理器。"
    "请把图片中可用于后续任务执行的信息转换成文本，重点输出："
    "1. 图片中的可见文字；"
    "2. 图表、界面、公式、场景的关键信息；"
    "3. 可能影响后续检索和总结的线索。"
    "请使用简洁中文输出。"
)


def _api_base_url() -> str:
    base_url = os.environ.get("MULTIMODAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    return base_url.rstrip("/")


def _api_key() -> str:
    return os.environ.get("MULTIMODAL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def _image_model() -> str:
    return os.environ.get("MULTIMODAL_IMAGE_MODEL") or os.environ.get("OPENAI_MODEL", "")


def _auth_headers() -> dict:
    api_key = _api_key()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _extract_message_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")).strip())
        return "\n".join(text for text in texts if text)
    return str(content).strip()


def _image_to_text(file_path: str) -> str:
    base_url = _api_base_url()
    if not base_url:
        raise RuntimeError("MULTIMODAL_BASE_URL or OPENAI_BASE_URL is not configured")

    model = _image_model()
    if not model:
        raise RuntimeError("MULTIMODAL_IMAGE_MODEL or OPENAI_MODEL is not configured")

    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
    image_bytes = Path(file_path).read_bytes()
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
        timeout=(20, 180),
    )
    response.raise_for_status()
    content = _extract_message_text(response.json())
    if not content:
        raise RuntimeError("vision model returned empty content")
    return content


def _prepare_single_asset(file_info: dict) -> tuple[str, str]:
    name = file_info["name"]
    path = file_info["path"]
    content_type = (file_info.get("content_type") or "").lower()

    if content_type.startswith("image/"):
        text = _image_to_text(path)
        return f"[图片识别结果] {name}\n{text}", f"[VISION] 已完成图片理解: {name}"

    return (
        f"[附件说明] {name}\n当前仅自动识别图片和 PDF，其他附件类型会被忽略。",
        f"[SKIP] 未解析的附件类型: {name}",
    )


async def build_multimodal_context(file_infos: list[dict]) -> tuple[str, list[str]]:
    if not file_infos:
        return "", []

    tasks = [asyncio.to_thread(_prepare_single_asset, file_info) for file_info in file_infos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    context_blocks = []
    logs = []
    for file_info, result in zip(file_infos, results):
        if isinstance(result, Exception):
            logs.append(f"[WARN] {file_info['name']} 预处理失败: {result}")
            context_blocks.append(
                f"[附件处理失败] {file_info['name']}\n"
                f"无法自动识别该附件，原因: {result}"
            )
            continue

        context_text, log_text = result
        context_blocks.append(context_text)
        logs.append(log_text)

    return "\n\n".join(context_blocks), logs
