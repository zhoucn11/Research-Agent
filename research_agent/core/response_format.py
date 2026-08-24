import re
import urllib.parse

from research_agent.core.paper_evidence import is_local_catalog_request


_NUMBER_WORDS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def detect_response_format(user_text: str, memory_text: str = "") -> str:
    """把用户明确的展示约束转换为可由代码保证的输出模式。"""
    text = str(user_text or "")
    compact = re.sub(r"\s+", "", text)

    if is_local_catalog_request(text):
        return "catalog"

    table_markers = (
        "只给一个对比表", "只给对比表", "只返回表格", "仅返回表格",
        "只要表格", "仅要表格", "只用表格", "仅用表格", "一行Markdown表格",
        "用表格回答", "使用表格", "Markdown表", "同一张表", "放在同一张表",
    )
    if any(marker in compact for marker in table_markers):
        return "table_only"

    catalog_action = any(marker in compact for marker in ("列出", "返回", "有哪些", "汇总"))
    catalog_fields = "标题" in compact and "作者" in compact
    catalog_limit = any(marker in compact for marker in ("只", "仅", "不要展开", "不展开"))
    if catalog_action and catalog_fields and catalog_limit:
        return "catalog"

    metadata_action = any(marker in compact for marker in ("只列出", "只返回", "仅列出", "仅返回"))
    metadata_fields = any(marker in compact for marker in ("作者", "年份", "标题", "DOI", "doi"))
    if metadata_action and metadata_fields:
        return "metadata_only"

    if any(marker in compact for marker in (
        "一句话", "简要", "简短", "不要展开", "不展开", "只说", "只根据论文回答",
    )):
        return "brief"

    memory = re.sub(r"\s+", "", str(memory_text or ""))
    if any(marker in memory for marker in ("回答只用一句话", "回答用一句话", "只用一句话回答", "最多三句话", "不超过三句话")):
        return "brief"
    return "default"


def _requested_sentence_limit(user_text: str, memory_text: str) -> int | None:
    compact = re.sub(r"\s+", "", f"{user_text}\n{memory_text}")
    if "一句话" in compact:
        return 1
    match = re.search(r"(?:最多|不超过|限制在)([一二三四五六七八九十]|\d{1,2})句(?:话)?", compact)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)


def _requested_char_limit(user_text: str, memory_text: str) -> int | None:
    compact = re.sub(r"\s+", "", f"{user_text}\n{memory_text}")
    match = re.search(r"(?:压缩到|限制在|不超过|最多)?(\d{1,4})个?(?:汉字|字|字符)(?:以内)?", compact)
    return int(match.group(1)) if match else None


def _take_sentences(text: str, limit: int) -> str:
    plain = compact_brief(text, limit=max(1000, len(str(text or "")) + 1))
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+", plain) if part.strip()]
    return " ".join(parts[:limit])


def apply_response_constraints(text: str, user_text: str = "", memory_text: str = "") -> str:
    """在最终落库前执行用户可验证的句数和字数约束。"""
    result = str(text or "").strip()
    sentence_limit = _requested_sentence_limit(user_text, memory_text)
    if sentence_limit:
        result = _take_sentences(result, sentence_limit)
    char_limit = _requested_char_limit(user_text, memory_text)
    if char_limit and len(result) > char_limit:
        result = result[:char_limit].rstrip("，,；;：: ")
    return result


def remove_unsupported_quantitative_claims(text: str, papers: list) -> tuple[str, list[str]]:
    """移除页级证据中不存在的指标数值，宁可少答也不保留猜测。"""
    span_quotes = [
        str(getattr(span, "quote", "") or "")
        for paper in papers or []
        for span in (getattr(paper, "evidence_spans", []) or [])
    ]
    if span_quotes:
        evidence = " ".join(span_quotes)
    else:
        evidence = " ".join(
            f"{getattr(paper, 'core_method', '')} {getattr(paper, 'key_findings', '')}"
            for paper in papers or []
        )
    supported_numbers = set(re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", evidence))
    claim_pattern = re.compile(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|BLEU|mAP|AP(?:50|75)?|ms|毫秒)",
        re.I,
    )
    removed = []

    def replace(match: re.Match) -> str:
        claim = match.group(0)
        if match.group(1) in supported_numbers:
            return claim
        removed.append(claim)
        return "（数值缺少页级证据，已省略）"

    return claim_pattern.sub(replace, str(text or "")), removed


def should_append_source_table(user_text: str) -> bool:
    text = str(user_text or "")
    return any(
        marker in text
        for marker in ("总结", "综述", "概括", "归纳", "对比", "比较", "清单", "列出所有", "全部", "整个本地")
    )


def stable_reference_index(paper, fallback: int) -> int:
    value = getattr(paper, "reference_index", None)
    return value if isinstance(value, int) and value >= 1 else fallback


def primary_evidence_citation(paper, fallback: int) -> str:
    reference_index = stable_reference_index(paper, fallback)
    spans = getattr(paper, "evidence_spans", []) or []
    if not spans:
        return f"[{reference_index}]"
    page = getattr(spans[0], "page_start", None)
    return f"[{reference_index}:p{page}]" if page else f"[{reference_index}:摘要]"


def _cell(value, limit: int = 350) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit] + "...[已截断]"
    return text.replace("|", "\\|")


def render_catalog(papers: list) -> str:
    return "\n".join(
        f"{stable_reference_index(paper, position)}. **{_cell(getattr(paper, 'title', '未知标题'), 180)}** — "
        f"{_cell(getattr(paper, 'authors', '未知作者'), 220)}"
        for position, paper in enumerate(papers, start=1)
    )


def render_comparison_table(papers: list) -> str:
    lines = [
        "| 文献编号 | 标题 | 作者 | 核心方法 | 关键结论 |",
        "|:---:|:---|:---|:---|:---|",
    ]
    for position, paper in enumerate(papers, start=1):
        ref = stable_reference_index(paper, position)
        lines.append(
            f"| [{ref}] | {_cell(getattr(paper, 'title', '未知标题'), 180)} | "
            f"{_cell(getattr(paper, 'authors', '未知作者'), 180)} | "
            f"{_cell(getattr(paper, 'core_method', ''), 260)} | "
            f"{_cell(getattr(paper, 'key_findings', ''), 240)} {primary_evidence_citation(paper, position)} |"
        )
    return "\n".join(lines)


def missing_page_citations(text: str, papers: list) -> list[int]:
    missing = []
    for position, paper in enumerate(papers, start=1):
        if not getattr(paper, "evidence_spans", []):
            continue
        reference_index = stable_reference_index(paper, position)
        if not re.search(rf"\[{reference_index}:(?:p\d+|摘要)\]", str(text or "")):
            missing.append(reference_index)
    return missing


def render_evidence_appendix(papers: list) -> str:
    lines = ["## 证据定位"]
    for position, paper in enumerate(papers, start=1):
        reference_index = stable_reference_index(paper, position)
        for span in (getattr(paper, "evidence_spans", []) or [])[:2]:
            page = getattr(span, "page_start", None)
            label = f"[{reference_index}:p{page}]" if page else f"[{reference_index}:摘要]"
            source = str(getattr(span, "source", "") or getattr(paper, "source", ""))
            quote = _cell(getattr(span, "quote", ""), 320)
            if source.startswith("http"):
                link = source
            else:
                link = f"/pdfs/{urllib.parse.quote(source)}" + (f"#page={page}" if page else "")
            lines.append(f"- {label} [{_cell(getattr(paper, 'title', '未知标题'), 120)} · 原文]({link})：{quote}")
    return "\n".join(lines) if len(lines) > 1 else ""


def extract_first_markdown_table(text: str) -> str:
    """只保留模型输出中的第一张合法 Markdown 表。"""
    blocks = []
    current = []
    for line in str(text or "").splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in blocks:
        if len(block) >= 2 and re.search(r"\|\s*:?-{3,}", block[1]):
            return "\n".join(block)
    return ""


def compact_brief(text: str, limit: int = 420) -> str:
    """移除标题和表格，并在完整句边界内限制简答长度。"""
    kept = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        kept.append(re.sub(r"^[-*+]\s+", "", stripped))
    result = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if len(result) <= limit:
        return result
    clipped = result[:limit]
    sentence_end = max(clipped.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?"))
    if sentence_end >= max(40, limit // 2):
        return clipped[:sentence_end + 1]
    return clipped.rstrip("，,；;：: ") + "……"
