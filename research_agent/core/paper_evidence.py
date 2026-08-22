import asyncio
import difflib
import os
import re

from research_agent.schemas.models import EvidenceGateResult


_PLACEHOLDER_TEXTS = {
    "",
    ".",
    "..",
    "...",
    "…",
    "****",
    "unknown",
    "none",
    "null",
    "未知",
    "未提及",
    "格式错误",
}

_SUMMARY_MARKERS = (
    "总结", "综述", "概括", "归纳", "相关工作", "整理成", "写成", "写进论文",
    "可直接写", "段落", "小节框架", "论文表述", "文献综述", "怎么讲", "讲了什么",
    "主要讲", "讲一下", "说一下", "展开讲", "详细讲", "介绍", "解读", "分析",
    "列出", "清单", "有哪些",
)
_COMPARISON_MARKERS = ("对比", "比较", "区别", "差异", "异同")
_EVIDENCE_ANSWER_MARKERS = (
    "为什么", "是否", "是多少", "什么", "如何", "作者", "年份", "结果", "指标",
    "性能", "复杂度", "预训练", "瓶颈", "证据", "页码", "核心", "实验",
)


def is_local_catalog_request(user_text: str) -> bool:
    """识别只需要本地文献目录的请求，避免为列清单启动图查询和证据抽取。"""
    compact = re.sub(r"\s+", "", str(user_text or ""))
    local_scope = any(marker in compact for marker in ("本地", "知识库", "文献库", "论文库", "PDF库"))
    catalog_intent = any(marker in compact for marker in (
        "有哪些文献", "有哪些论文", "有什么文献", "有什么论文",
        "列出文献", "列出论文", "文献清单", "论文清单", "文献列表", "论文列表",
    ))
    return local_scope and catalog_intent


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_authors(value) -> str:
    text = _clean_text(value)
    if not text:
        return "未知作者"
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])[\d.*†‡]+", "", text)
    text = re.sub(r"[*†‡]+", "", text)
    text = re.sub(r"(?:(?<=^)|(?<=[,，;；\s]))\d+(?:\.\d+)*(?=\s|[,，;；]|$)", "", text)
    text = re.sub(r"\s*[,，;；]\s*", ", ", text)
    text = re.sub(r"(?:,\s*){2,}", ", ", text)
    return text.strip(" ,") or "未知作者"


def _is_meaningful(value, min_length: int = 4) -> bool:
    text = _clean_text(value)
    lowered = text.casefold()
    if lowered in _PLACEHOLDER_TEXTS:
        return False
    if set(text) <= {"*", ".", "-", "_", "…", " "}:
        return False
    if any(marker in text for marker in ("未提取到", "未获取到", "请查阅原文")):
        return False
    return len(text) >= min_length


def _paper_quality_score(paper) -> int:
    return sum(
        min(len(_clean_text(value)), limit)
        for value, limit in (
            (getattr(paper, "title", ""), 120),
            (getattr(paper, "core_method", ""), 350),
            (getattr(paper, "key_findings", ""), 350),
        )
        if _is_meaningful(value)
    )


def normalize_paper_summaries(papers: list, allowed_sources: list[str] | None = None) -> list:
    """过滤空证据，并让同一个本地来源只保留质量最高的一条摘要。"""
    allowed_sources = allowed_sources or []
    allowed_lookup = {source.casefold(): source for source in allowed_sources}
    best_by_key = {}

    for paper in papers or []:
        if paper is None:
            continue

        title = _clean_text(getattr(paper, "title", ""))
        raw_source = _clean_text(getattr(paper, "source", ""))
        source = os.path.basename(raw_source) if allowed_sources else raw_source
        core_method = _clean_text(getattr(paper, "core_method", ""))
        key_findings = _clean_text(getattr(paper, "key_findings", ""))

        if not _is_meaningful(title):
            continue
        if not (_is_meaningful(core_method, 8) or _is_meaningful(key_findings, 8)):
            continue

        if allowed_sources:
            matched_source = allowed_lookup.get(source.casefold())
            if not matched_source:
                continue
            source = matched_source
            paper.source = matched_source
        elif not _is_meaningful(source):
            continue

        paper.title = title
        paper.authors = _clean_authors(getattr(paper, "authors", ""))
        paper.core_method = core_method
        paper.key_findings = key_findings

        # 本地 PDF 和网络 URL 都以真实来源作为稳定身份；标题只是展示字段。
        identity = source.casefold()
        current = best_by_key.get(identity)
        if current is None or _paper_quality_score(paper) > _paper_quality_score(current):
            best_by_key[identity] = paper

    return list(best_by_key.values())


def select_referenced_papers(user_text: str, papers: list, limit: int = 20) -> list:
    """按上一轮稳定顺序解析“第 N 篇/最后一篇”，其余复数指代保留整组。"""
    available = normalize_paper_summaries(papers)[:limit]
    if not available:
        return []

    # 编号属于候选列表身份，选中子集后不能再从 1 开始编号。
    for position, paper in enumerate(available, start=1):
        current = getattr(paper, "reference_index", None)
        if not isinstance(current, int) or current < 1:
            paper.reference_index = position

    ordinal = re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*[篇个]", user_text or "")
    if ordinal:
        raw_number = ordinal.group(1)
        if raw_number.isdigit():
            number = int(raw_number)
        else:
            digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
            if raw_number == "十":
                number = 10
            elif "十" in raw_number:
                left, right = raw_number.split("十", 1)
                number = digits.get(left, 1) * 10 + digits.get(right, 0)
            else:
                number = digits.get(raw_number, 0)
        index = number - 1
        return [available[index]] if 0 <= index < len(available) else []
    if "最后一篇" in (user_text or ""):
        return [available[-1]]
    return available


def select_papers_for_synthesis(user_text: str, papers: list, limit: int = 20) -> tuple[list, str]:
    """确定性判断总结/对比意图，避免把证据充分性完全交给路由模型。"""
    text = user_text or ""
    wants_summary = any(marker in text for marker in _SUMMARY_MARKERS)
    wants_comparison = any(marker in text for marker in _COMPARISON_MARKERS)
    wants_evidence_answer = any(marker in text for marker in _EVIDENCE_ANSWER_MARKERS)
    if not (wants_summary or wants_comparison or wants_evidence_answer):
        return [], ""

    selected = select_referenced_papers(text, papers, limit)
    if not selected:
        return [], ""
    if wants_comparison and len(selected) >= 2:
        return selected, "comparison"
    if wants_summary:
        return selected, "partial_summary" if wants_comparison else "summary"
    if wants_evidence_answer:
        return selected, "answer"
    return [], ""


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def _explicit_titles(user_text: str) -> list[str]:
    text = str(user_text or "")
    titles = re.findall(r"《([^》]{4,200})》", text)
    titles.extend(re.findall(r'[“\"]([^“”\"\n]{4,200})[”\"]', text))
    return list(dict.fromkeys(title.strip() for title in titles if title.strip()))


def _is_web_source(source: str) -> bool:
    return bool(re.search(r"https?://", str(source or ""), flags=re.I))


def _field_is_available(paper, field_name: str) -> bool:
    value = getattr(paper, field_name, "")
    if _clean_text(value).startswith("未知"):
        return False
    min_length = 4
    if field_name == "year":
        return bool(re.fullmatch(r"(?:19|20)\d{2}", _clean_text(value)))
    if field_name == "doi":
        min_length = 6
    return _is_meaningful(value, min_length=min_length)


def evaluate_evidence_gate(user_text: str, papers: list, limit: int = 20) -> EvidenceGateResult:
    """在候选论文升级前确定性检查目标命中、字段、来源和证据覆盖。"""
    selected, mode = select_papers_for_synthesis(user_text, papers, limit)
    if not mode and any(marker in str(user_text or "") for marker in _COMPARISON_MARKERS):
        selected = select_referenced_papers(user_text, papers, limit)
        mode = "comparison"
    if not mode:
        return EvidenceGateResult()
    if not selected:
        return EvidenceGateResult(mode=mode, reasons=["没有可用于当前任务的候选论文。"])

    text = str(user_text or "")
    reasons: list[str] = []
    warnings: list[str] = []
    catalog_request = is_local_catalog_request(text)
    wants_comparison = any(marker in text for marker in _COMPARISON_MARKERS)

    available_titles = {_normalized_title(getattr(paper, "title", "")) for paper in selected}
    for title in _explicit_titles(text):
        if _normalized_title(title) not in available_titles:
            reasons.append(f"点名论文《{title}》未被精确命中，不能用相关论文替代。")

    if wants_comparison:
        unique_titles = {_normalized_title(getattr(paper, "title", "")) for paper in selected}
        unique_sources = {_clean_text(getattr(paper, "source", "")).casefold() for paper in selected}
        unique_titles.discard("")
        unique_sources.discard("")
        if len(unique_titles) < 2:
            reasons.append("对比任务至少需要两篇不同论文，当前证据不足。")
        if len(unique_sources) < 2:
            reasons.append("对比任务至少需要两个可区分的论文来源。")

    required_fields = {"title", "source"}
    if re.search(r"作者|谁写|author", text, flags=re.I):
        required_fields.add("authors")
    if re.search(r"年份|哪一年|year|发表时间", text, flags=re.I):
        required_fields.add("year")
    if re.search(r"\bdoi\b", text, flags=re.I):
        required_fields.add("doi")
    content_question = bool(re.search(
        r"总结|综述|相关工作|对比|比较|区别|差异|方法|架构|核心|为什么|如何|"
        r"结论|结果|指标|性能|实验|复杂度|概括|介绍|解读|分析|讲了什么|怎么讲|主要讲|"
        r"讲一下|说一下|展开讲|详细讲|"
        r"summary|review|compare|method|result",
        text,
        flags=re.I,
    ))
    if content_question and not catalog_request:
        required_fields.update({"core_method", "key_findings"})

    evidence_supported = 0
    for position, paper in enumerate(selected, start=1):
        label = f"[{getattr(paper, 'reference_index', None) or position}]《{getattr(paper, 'title', '未知标题')}》"
        missing = [field for field in sorted(required_fields) if not _field_is_available(paper, field)]
        if missing:
            field_labels = {
                "title": "标题", "source": "来源", "authors": "作者", "year": "年份",
                "doi": "DOI", "core_method": "核心方法", "key_findings": "关键结论",
            }
            reasons.append(f"{label} 缺少必要字段：{'、'.join(field_labels[field] for field in missing)}。")

        source = str(getattr(paper, "source", "") or "")
        spans = [span for span in (getattr(paper, "evidence_spans", []) or []) if str(getattr(span, "quote", "") or "").strip()]
        if catalog_request or not content_question:
            evidence_supported += 1
        elif _is_web_source(source):
            evidence_supported += 1
            if not spans:
                warnings.append(f"{label} 当前只有联网摘要级证据，没有页级定位。")
        elif spans:
            evidence_supported += 1
        else:
            reasons.append(f"{label} 缺少可回链的本地 EvidenceSpan，不能据此生成内容性结论。")

    coverage = evidence_supported / len(selected) if selected else 0.0
    return EvidenceGateResult(
        passed=not reasons,
        mode=mode,
        selected_papers=selected,
        reasons=list(dict.fromkeys(reasons)),
        warnings=list(dict.fromkeys(warnings)),
        evidence_coverage=coverage,
    )


def paper_fields_from_document_header(header_context: str, source: str) -> dict:
    """从 full_docs 首页和摘要构造可用的论文级兜底证据，不依赖图查询命中 chunk。"""
    text = str(header_context or "")
    title_match = re.search(r"(?im)^title:\s*(.+)$", text)
    year_match = re.search(r"(?im)^year:\s*(.+)$", text)
    title = _clean_text(title_match.group(1) if title_match else os.path.splitext(source)[0])
    year = _clean_text(year_match.group(1) if year_match else "未知") or "未知"

    clean_text = re.sub(r"\s*\[page:\d+\]\s*", "\n", text)
    clean_text = re.sub(r"\n{2,}", "\n", clean_text)
    pre_abstract = re.split(r"(?im)^\s*Abstract\s*$", clean_text, maxsplit=1)[0]
    if "[/PAPER_METADATA]" in pre_abstract:
        pre_abstract = pre_abstract.split("[/PAPER_METADATA]", 1)[1]

    # 旧索引的 manifest 可能把年份写成“未知”；优先从论文首页的明确年份补回。
    if year in {"未知", "未知年份", "未提及"}:
        header_years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", pre_abstract[:3500])
        if header_years:
            year = header_years[0]

    affiliation_markers = (
        "university", "college", "institute", "research", "laboratory", "lab ",
        "department", "school", "hospital", "technology", "google brain", "china",
        "usa", "corresponding", "email", "@",
    )
    authors = []
    for raw_line in pre_abstract.splitlines():
        line = _clean_text(raw_line)
        if not line or line.casefold() == title.casefold():
            continue
        lowered = line.casefold()
        if any(marker in lowered for marker in affiliation_markers):
            continue
        for candidate in re.split(r",|\band\b", line):
            candidate = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])[\d.*†‡]+", "", candidate)
            candidate = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]", " ", candidate)
            candidate = _clean_text(candidate)
            words = candidate.split()
            if 2 <= len(words) <= 5 and all(word[0].isupper() for word in words if word):
                if candidate not in authors:
                    authors.append(candidate)

    abstract_match = re.search(
        r"(?is)\bAbstract\b\s*(.+?)(?=\n\s*(?:\d+\s*)?Introduction\b|\Z)",
        clean_text,
    )
    abstract = _clean_text(abstract_match.group(1) if abstract_match else "")
    if not abstract:
        abstract = "论文首页已入库，但摘要区未被解析器完整识别。"
    finding_markers = (
        "achieve", "improv", "outperform", "result", "experiment", "accuracy",
        "bleu", "map", "average precision", "state-of-the-art", "performance",
    )
    finding_sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", abstract)
        if any(marker in sentence.casefold() for marker in finding_markers)
    ]
    key_findings = _clean_text(" ".join(finding_sentences)) or abstract

    return {
        "title": title,
        "authors": ", ".join(authors) if authors else "未知作者",
        "year": year,
        "source": source,
        "core_method": abstract[:1200],
        "key_findings": key_findings[:1800],
    }


def should_replace_author_metadata(local_authors: str, online_authors: str) -> bool:
    """在线标题已核准后，判断作者信息是否足以安全修正本地 OCR。"""
    local = _clean_authors(local_authors)
    online = _clean_authors(online_authors)
    if online in {"", "未知", "未知作者", "未提及"}:
        return False
    if local in {"", "未知", "未知作者", "未提及"}:
        return True

    local_items = [item.strip() for item in re.split(r"[,;，；]", local) if item.strip()]
    online_items = [item.strip() for item in re.split(r"[,;，；]", online) if item.strip()]
    if len(online_items) > len(local_items):
        return True
    if len(online_items) != len(local_items) or local.casefold() == online.casefold():
        return False

    # 等长作者表常见的是单字符 OCR 错误；高相似度时采用权威在线元数据。
    similarity = difflib.SequenceMatcher(None, local.casefold(), online.casefold()).ratio()
    return similarity >= 0.85


async def recover_missing_paper_summaries(
    initial_papers: list,
    target_sources: list[str],
    recover_one,
    concurrency: int = 1,
) -> tuple[list, list[str], dict[str, str]]:
    """调用来源级补召回函数，返回统一去重结果、仍缺失来源和错误信息。"""
    normalized = normalize_paper_summaries(initial_papers, allowed_sources=target_sources)
    covered = {str(paper.source).casefold() for paper in normalized}
    missing = [source for source in target_sources if source.casefold() not in covered]
    if not missing:
        return normalized, [], {}

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(source: str):
        async with semaphore:
            try:
                return source, await recover_one(source), ""
            except Exception as exc:
                return source, [], str(exc)

    batches = await asyncio.gather(*(guarded(source) for source in missing))
    recovered = []
    errors = {}
    for source, papers, error in batches:
        if error:
            errors[source] = error
        recovered.extend(papers or [])

    merged = normalize_paper_summaries(
        [*normalized, *recovered],
        allowed_sources=target_sources,
    )
    covered = {str(paper.source).casefold() for paper in merged}
    remaining = [source for source in target_sources if source.casefold() not in covered]
    return merged, remaining, errors
