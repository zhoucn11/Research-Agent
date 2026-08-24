# search_agent.py
import asyncio
import os
import time

from langchain_core.messages import HumanMessage, ToolMessage
from tqdm.asyncio import tqdm as tqdm_asyncio

from research_agent.retrieval.academic_search import (
    process_single_paper_summary,
    search_academic_papers_detailed,
)
from research_agent.retrieval.academic_providers import paper_summary_fields
from research_agent.core.agent_state_helpers import compact_state_value
from research_agent.core.llm_clients import get_synthesis_llm
from research_agent.core.paper_evidence import normalize_paper_summaries
from research_agent.core.state import AgentState
from research_agent.core.runtime_events import emit_runtime_event, instrument_node, runtime_print as print
from research_agent.core.tool_validation import ToolValidationError, validate_tool_call
from research_agent.core.web_search_helpers import (
    effective_result_limit,
    extract_explicit_paper_titles,
    is_exact_title_lookup,
    normalize_title,
    rank_papers_by_query,
    select_exact_title_record,
    select_named_anchor_papers,
)
from research_agent.schemas.models import PaperSummary


WEB_SEARCH_RECALL_LIMIT = 20


@instrument_node("search_map")
async def search_map_node(state: AgentState):
    node_start = time.time()
    print("\n[⚙️ Search Map Agent] 正在启动全球检索...")
    raw_tool_call = state["messages"][-1].tool_calls[0]
    try:
        tool_call = validate_tool_call(raw_tool_call)
    except ToolValidationError as exc:
        message = f"联网检索参数未通过确定性校验，已拒绝执行：{exc}"
        emit_runtime_event("tool_validation_rejected", message, tool_name=raw_tool_call.get("name", ""))
        return {
            "candidate_papers": [],
            "pending_questions": "需要 Assistant 生成合法的联网检索参数。",
            "messages": [ToolMessage(tool_call_id=raw_tool_call.get("id", "invalid-tool"), content=message)],
        }

    rationale = tool_call["args"].get("rationale", "未提供理由")
    print(f"  └─ 🧠 行动理由: {rationale}")

    keyword = tool_call["args"].get("keyword", "Agent")
    user_core_topic = tool_call["args"].get("user_core_topic", keyword)
    year_range = tool_call["args"].get("year_range", "")
    print(f"  └─ 🎯 意图转化: 检索词 '{keyword}', 时间范围 '{year_range}'")
    api_start = time.time()
    search_result = await asyncio.to_thread(
        search_academic_papers_detailed,
        keyword,
        year_range,
        WEB_SEARCH_RECALL_LIMIT,
    )
    raw_papers = search_result.papers
    print(
        f"  └─ API 召回耗时: {time.time() - api_start:.2f}s "
        f"(来源: {search_result.provider}, 共 {len(raw_papers)} 篇)"
    )
    emit_runtime_event(
        "retrieval_result",
        "Academic web search completed",
        provider=search_result.provider,
        raw_count=len(raw_papers),
        keyword=keyword,
        latency_ms=round((time.time() - api_start) * 1000, 2),
    )
    if search_result.errors:
        print(f"  └─ ⚠️ 联网诊断: {'; '.join(search_result.errors)}")

    latest_user_text = next(
        (str(message.content or "") for message in reversed(state.get("messages") or []) if isinstance(message, HumanMessage)),
        "",
    )
    request_context = f"{latest_user_text}\n{user_core_topic}".strip()
    anchor_papers = select_named_anchor_papers(state, request_context)
    explicit_title_values = extract_explicit_paper_titles(request_context)
    explicit_titles = {normalize_title(title) for title in explicit_title_values}
    for paper in raw_papers:
        fields = paper_summary_fields(paper)
        if fields and normalize_title(fields["title"]) in explicit_titles:
            anchor_papers.append(PaperSummary(**fields))
    covered_anchor_titles = {normalize_title(paper.title) for paper in anchor_papers}
    for title in explicit_title_values:
        if normalize_title(title) in covered_anchor_titles:
            continue
        anchor_started = time.time()
        anchor_search = await asyncio.to_thread(
            search_academic_papers_detailed,
            title,
            "",
            5,
            1,
        )
        exact_record = select_exact_title_record(anchor_search.papers, title)
        fields = paper_summary_fields(exact_record) if exact_record else None
        if fields:
            anchor_papers.append(PaperSummary(**fields))
            covered_anchor_titles.add(normalize_title(fields["title"]))
            print(
                f"  └─ ⚓ 精确补查对比锚点: 《{fields['title']}》"
                f" ({anchor_search.provider}, {time.time() - anchor_started:.2f}s)"
            )
    anchor_papers = normalize_paper_summaries(anchor_papers)
    anchor_titles = {normalize_title(paper.title) for paper in anchor_papers}
    if is_exact_title_lookup(request_context):
        raw_papers = []
    else:
        raw_papers = [
            paper for paper in raw_papers
            if normalize_title(paper.get("title", "")) not in anchor_titles
        ]
    raw_papers = rank_papers_by_query(raw_papers, keyword)
    result_limit = effective_result_limit(
        latest_user_text or user_core_topic,
        int(os.environ.get("WEB_SEARCH_DEFAULT_RESULT_LIMIT", "5")),
    )
    raw_papers = [paper for paper in raw_papers if paper_summary_fields(paper)]
    raw_papers = raw_papers[:result_limit]

    use_llm_enrichment = os.environ.get("WEB_PAPER_LLM_ENRICHMENT", "false").lower() in {
        "1", "true", "yes", "on",
    }
    llm = None
    if use_llm_enrichment:
        llm = get_synthesis_llm(
            temperature=0,
            max_tokens=int(os.environ.get("WEB_PAPER_SUMMARY_MAX_OUTPUT_TOKENS", "2400")),
            thinking_budget=int(os.environ.get("WEB_PAPER_SUMMARY_THINKING_BUDGET", "512")),
        )
        print(f"  └─ 🌐 使用 Synthesizer 同源远程 API 提炼 {len(raw_papers)} 篇网络论文。")
    else:
        print("  └─ ⚡ 使用 API 摘要确定性构造网络证据，跳过逐篇本地 LLM 提炼。")
    tasks = [process_single_paper_summary(p, llm) for p in raw_papers]
    results = []
    if tasks:
        results = [
            res for res in await tqdm_asyncio.gather(*tasks, desc="🌐 提炼网络文献", colour="green")
            if res is not None
        ]
    combined_results = normalize_paper_summaries([*anchor_papers, *results])

    print(f"  [⏱️ Search Node 总耗时] {time.time() - node_start:.2f}s")
    if not combined_results and search_result.provider == "none" and search_result.errors:
        obs_msg = (
            "联网学术服务当前不可用，Semantic Scholar 与 OpenAlex 均未成功返回数据。"
            "这是服务或网络错误，不是关键词不匹配；禁止继续改写关键词重试，请直接向用户说明并建议检查代理/API 配置。"
        )
    elif not combined_results:
        obs_msg = (
            "联网服务正常，但本轮未检索到有效文献。请直接如实报告并停止本轮搜索；"
            "如需放宽主题或年份，等待用户下一轮明确提出。"
        )
    elif anchor_papers and not results:
        titles = [f"《{paper.title}》" for paper in anchor_papers]
        obs_msg = (
            f"已通过联网学术服务精确找到用户点名论文：{', '.join(titles)}。"
            "请直接基于该论文证据回答，禁止继续搜索或替换为相似论文。"
        )
    else:
        titles = [f"《{p.title}》" for p in combined_results]
        obs_msg = (
            f"本轮联网检索共获得 {len(combined_results)} 篇可追踪文献：{', '.join(titles)}。"
            "请直接使用当前排序结果回答、列出候选或进入证据门控；同一用户轮次禁止继续调用联网搜索。"
            "若结果相关性不足，应明确说明缺口，等待用户下一轮调整条件。"
        )

    evidence_update = (
        f"联网检索 keyword='{keyword}', year_range='{year_range or '不限'}'，"
        f"由 {search_result.provider} 获得 {len(results)} 篇可结构化提取的网络文献。"
    )
    if anchor_papers:
        evidence_update += f" 保留 {len(anchor_papers)} 篇用户点名的既有论文作为对比锚点。"
    if combined_results:
        evidence_update += " 代表结果：" + "；".join([p.title for p in combined_results[:5]])

    return {
        "candidate_papers": combined_results,
        "collected_evidence": compact_state_value(evidence_update),
        "pending_questions": (
            "等待 Assistant 基于联网结果与对比锚点生成回答。"
            if combined_results else "等待 Assistant 直接报告无结果或联网服务异常。"
        ),
        "messages": [ToolMessage(tool_call_id=tool_call["id"], content=obs_msg)],
    }
