import re

from research_agent.core.paper_evidence import normalize_paper_summaries


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def finalize_incomplete_search_response(text: str, *, was_web_tool_result: bool) -> str:
    """阻止任务在没有实际工具调用时以“下一轮再搜”的空承诺结束。"""
    content = str(text or "").strip()
    if not was_web_tool_result:
        return content
    future_search = bool(re.search(r"下一轮|后续.{0,8}(?:检索|搜索)|将(?:采用|使用|继续).{0,20}(?:检索|搜索)", content))
    if not future_search:
        return content
    return (
        "本轮联网检索未能找到满足全部条件的有效论文，因此没有把相关性不足的结果冒充答案。"
        "请放宽年份或关键词后重新发起检索。"
    )


def requested_result_limit(user_core_topic: str) -> int | None:
    text = str(user_core_topic or "")
    if "一篇" in text or re.search(r"\b1\s*篇", text):
        return 1
    match = re.search(r"([2-9]|10)\s*篇", text)
    return int(match.group(1)) if match else None


def effective_result_limit(user_core_topic: str, default_limit: int = 5) -> int:
    return requested_result_limit(user_core_topic) or max(1, default_limit)


def extract_explicit_paper_titles(user_text: str) -> list[str]:
    return [title.strip() for title in re.findall(r"《([^》]{4,200})》", str(user_text or "")) if title.strip()]


def is_exact_title_lookup(user_text: str) -> bool:
    """点名论文的解读请求只需要目标论文；相关文献和对比请求仍保留多篇召回。"""
    text = str(user_text or "")
    if not extract_explicit_paper_titles(text):
        return False
    return not any(marker in text for marker in ("相关", "相似", "类似", "对比", "比较", "区别", "差异", "异同"))


def select_exact_title_record(papers: list[dict], title: str) -> dict | None:
    target = normalize_title(title)
    for paper in papers or []:
        if normalize_title(paper.get("title", "")) == target:
            return paper
    return None


def rank_papers_by_query(papers: list[dict], keyword: str) -> list[dict]:
    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(keyword or "").casefold())
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "comparison", "paper"}
    }
    if not query_tokens:
        return papers
    ranked = []
    for index, paper in enumerate(papers):
        title_tokens = set(re.findall(r"[a-z0-9]+", str(paper.get("title") or "").casefold()))
        ranked.append((len(query_tokens & title_tokens), -index, paper))
    return [paper for _, _, paper in sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)]


def select_named_anchor_papers(state: dict, user_core_topic: str) -> list:
    normalized_topic = normalize_title(user_core_topic)
    if not normalized_topic:
        return []
    available = [
        *(state.get("candidate_papers") or []),
        *(state.get("selected_papers") or []),
    ]
    anchors = []
    for paper in normalize_paper_summaries(available):
        title = normalize_title(getattr(paper, "title", ""))
        if len(title) >= 8 and title in normalized_topic:
            anchors.append(paper)
    return normalize_paper_summaries(anchors)


def explicitly_requests_web_search(user_text: str) -> bool:
    """显式联网要求优先于已有证据门控，避免拿上一轮本地论文直接作答。"""
    text = str(user_text or "").casefold()
    return any(
        marker in text
        for marker in ("网上", "联网", "全网", "外网", "网络文献", "web search", "online search")
    )
