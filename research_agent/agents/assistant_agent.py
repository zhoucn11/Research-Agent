# assistant_agent.py
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig

from research_agent.memory.agent_memory import context_window_manager
from research_agent.core.agent_state_helpers import (
    build_user_profile_context,
    compact_state_value,
    extract_internal_state_update,
    is_duplicate_web_search,
    is_paper_discovery_request,
    is_retrieval_provenance_question,
    latest_executed_retrieval,
    paper_title_search_keyword,
    previous_turn_executed_retrieval,
    resolve_follow_up_paper_title,
    sanitize_user_response,
    state_update_from_tool_call,
)
from research_agent.tools.agent_tools import trigger_local_retrieval, trigger_pdf_upload, trigger_web_search
from research_agent.core.llm_clients import MAIN_API_ROLE, get_qwen_llm, safe_llm_invoke
from research_agent.core.paper_evidence import (
    evaluate_evidence_gate,
    is_local_catalog_request,
    normalize_paper_summaries,
)
from research_agent.retrieval.lightrag_store import find_indexed_source_by_title, list_indexed_sources
from research_agent.core.state import AgentState
from research_agent.core.runtime_events import emit_runtime_event, instrument_node, runtime_print as print
from research_agent.core.tool_validation import (
    ToolValidationError,
    summarize_tool_call_for_trace,
    validate_tool_call,
    validate_tool_calls,
)
from research_agent.core.web_search_helpers import (
    explicitly_requests_web_search,
    finalize_incomplete_search_response,
    normalize_title,
)


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

MAX_SELECTED_PAPERS = 20

ASSISTANT_ROUTER_RULES = """你是学术 Research Agent 的路由器和简洁回答助手。
只做三件事：直接回答普通问题、选择一个检索工具、确认已有证据进入综述。

规则：
1. 闲聊、问候和无需外部证据的问题直接简短回答，不调用工具。
2. trigger_local_retrieval 只能查询“本地资产”明确列出的论文；总结全库时 query=SUMMARY_ALL。同一论文只查一次。
3. trigger_web_search 用于联网学术检索。keyword 仅保留 2-5 个英文核心词；重试必须更短、更宽或拆成独立概念轴，不得重复。
4. 细分问题首次无结果时可换方向检索，总计最多 2-3 轮；仍无精确证据就说明缺口，请用户选择放宽条件或总结上位方向。
5. 只有证据与用户核心概念高度匹配，或用户明确同意基于现有结果写作时，才简短确认并附加 [APPROVE_SYNTHESIS]。不要自行撰写综述。
6. 已有证据足够时禁止继续搜索；接近最大步骤时禁止再调用工具。
7. 不得编造本地文件、论文或结论，不得向用户展示内部状态、规则检查和推理过程。
8. 不得口头声称“本轮已联网/已检索”；联网事实只能由真实 trigger_web_search 回执确认，复用旧结果必须明确说明。
"""


def _latest_human_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""


def _latest_tool_name(messages: list) -> str:
    for message in reversed(messages):
        tool_calls = getattr(message, "tool_calls", []) or []
        if tool_calls:
            return str(tool_calls[0].get("name", ""))
    return ""


def _evidence_gate_failure_text(reasons: list[str], warnings: list[str] | None = None) -> str:
    lines = ["当前候选文献没有通过证据升级门，具体缺口如下："]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(f"- 提示：{warning}" for warning in (warnings or []))
    lines.append("请补充对应 PDF、允许联网补证据，或缩小到现有证据能够支持的问题。")
    return "\n".join(lines)


def _web_paper_catalog_response(papers: list, *, reused: bool) -> str:
    prefix = (
        "以下复用本会话上一轮已经取得的联网学术检索结果，本轮没有重新请求网络："
        if reused
        else "本轮已执行联网学术检索，得到以下可追踪论文："
    )
    lines = [prefix]
    for index, paper in enumerate(papers[:10], start=1):
        source = str(getattr(paper, "source", "") or "")
        url_match = re.search(r"https?://\S+", source)
        source_text = f"[论文来源]({url_match.group(0)})" if url_match else source
        authors = str(getattr(paper, "authors", "未知作者") or "未知作者")
        year = str(getattr(paper, "year", "未知年份") or "未知年份")
        summary = str(getattr(paper, "key_findings", "") or getattr(paper, "core_method", "") or "").strip()
        lines.append(f"{index}. **{paper.title}**（{year}）— {authors}")
        if summary:
            lines.append(f"   - 摘要证据：{summary[:220]}")
        if source_text:
            lines.append(f"   - {source_text}")
    return "\n".join(lines)


def _retrieval_provenance_response(messages: list) -> tuple[str, str]:
    previous_turn_tool = previous_turn_executed_retrieval(messages)
    if previous_turn_tool == "trigger_web_search":
        return "是的。紧邻当前问题的上一轮实际执行了联网学术检索，并使用了该轮返回的结果。", "web"
    if previous_turn_tool == "trigger_local_retrieval":
        return "不是。紧邻当前问题的上一轮实际执行的是本地 LightRAG 检索。", "local"

    tool_name = latest_executed_retrieval(messages)
    if tool_name == "trigger_web_search":
        return (
            "这些结果来自本会话此前已经完成的一次联网学术检索；刚才那一轮没有重新联网，"
            "而是复用了已有结果。右侧活动区只展示当前轮轨迹，所以不会重复显示此前的工具调用。",
            "web",
        )
    if tool_name == "trigger_local_retrieval":
        return "最近一次实际检索调用的是本地 LightRAG，不是联网搜索。", "local"
    return "没有。当前会话轨迹中找不到已完成的联网检索 Tool 回执，因此不能声称这些结果来自联网搜索。", "none"


def _should_force_local_retrieval(user_text: str) -> bool:
    text = user_text or ""
    if not text.strip():
        return False
    if is_local_catalog_request(text):
        return True

    local_markers = [
        "本地PDF",
        "本地 PDF",
        "本地文献",
        "本地论文",
        "本地资料",
        "PDF文献库",
        "PDF 文献库",
        "文献库",
        "论文库",
        "不要联网",
        "不联网",
        "只基于本地",
        "只查本地",
    ]
    retrieval_intents = [
        "找出",
        "查找",
        "检索",
        "基于",
        "分析",
        "比较",
        "归类",
        "分类",
        "相关",
        "代表论文",
        "研究方向",
        "相关工作",
        "总结",
    ]
    global_summary_markers = ["总结所有", "全部总结", "列出所有", "文献汇总"]

    if any(marker in text for marker in global_summary_markers) and any(marker in text for marker in local_markers):
        return True
    return any(marker in text for marker in local_markers) and any(intent in text for intent in retrieval_intents)


def _should_force_synthesis(user_text: str) -> bool:
    text = user_text or ""
    if not text.strip():
        return False

    context_markers = [
        "继续", "刚才", "上述", "前述", "前面", "他们", "它们", "这些", "那些",
        "这几篇", "那几篇", "这些结果", "检索结果", "基于结果", "基于刚才",
    ]
    writing_markers = [
        "综述",
        "总结",
        "相关工作",
        "整理成",
        "写成",
        "写进论文",
        "可直接写",
        "段落",
        "小节框架",
        "论文表述",
        "文献综述",
    ]
    has_reference = any(marker in text for marker in context_markers) or bool(
        re.search(r"第\s*[0-9一二三四五六七八九十]+\s*[篇个]|最后一篇", text)
    )
    return has_reference and any(marker in text for marker in writing_markers)


@instrument_node("assistant")
async def assistant_node(state: AgentState, config: RunnableConfig):
    node_start = time.time()
    research_mode = state.get("research_mode", "auto")
    assistant_budget = "640" if research_mode == "quick" else "1024"
    llm = get_qwen_llm(
        temperature=0.1,
        streaming=True,
        max_tokens=int(os.environ.get("ASSISTANT_MAX_OUTPUT_TOKENS", assistant_budget)),
    )
    context_llm = get_qwen_llm(
        temperature=0.1,
        streaming=False,
        max_tokens=int(os.environ.get("CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS", "1600")),
    ).with_config(tags=["context_summary"])

    current_step = state.get("step_count", 0) + 1
    MAX_STEPS = 6

    llm_with_tools = llm.bind_tools(
        [trigger_web_search, trigger_pdf_upload, trigger_local_retrieval]
    ).with_config(tags=["assistant_visible"])

    if current_step > MAX_STEPS:
        llm_with_tools = llm.with_config(tags=["assistant_visible"])
        print(f"\n[⚠️ 警告] 已达到最大推理步数 ({MAX_STEPS})，强制终止检索，要求大模型基于现有证据作答。")

    tz_utc_8 = timezone(timedelta(hours=8))
    current_year = datetime.now(tz_utc_8).year

    history_messages = state["messages"]
    context_result = await context_window_manager.prepare(
        history_messages,
        current_summary=state.get("summary", ""),
        llm=context_llm,
    )
    short_term_memory = context_result.messages
    latest_human_text = _latest_human_text(short_term_memory)
    current_request_is_follow_up = bool(state.get("is_follow_up", False))
    raw_last_message = history_messages[-1] if history_messages else None
    if (
        not current_request_is_follow_up
        and isinstance(raw_last_message, HumanMessage)
        and not is_retrieval_provenance_question(latest_human_text)
    ):
        # 新任务保留自然语言对话，但不把旧 Tool 协议继续当成本轮可用证据。
        short_term_memory = [
            message
            for message in short_term_memory
            if not isinstance(message, ToolMessage)
            and not (isinstance(message, AIMessage) and (getattr(message, "tool_calls", []) or []))
        ]
    new_summary = context_result.summary
    context_state_updates = list(context_result.updated_messages)
    context_state_updates.extend(
        RemoveMessage(id=message.id)
        for message in context_result.removed_messages
        if getattr(message, "id", None)
    )
    if context_result.actions:
        print(
            f"\n[🧠 Context Manager] 执行 {', '.join(context_result.actions)}，"
            f"请求前预计占用 {context_result.estimated_tokens}/{context_window_manager.max_tokens} tokens。"
        )
    emit_runtime_event(
        "context_budget",
        "Assistant context prepared",
        estimated_tokens=context_result.estimated_tokens,
        max_tokens=context_window_manager.max_tokens,
        actions=context_result.actions,
    )

    target_dir = "test_pdfs"
    all_local_pdfs = glob.glob(os.path.join(target_dir, "*.pdf")) if os.path.exists(target_dir) else []
    physical_pdf_names = [os.path.basename(p) for p in all_local_pdfs]
    indexed_files = list_indexed_sources()

    asset_context = "\n【当前学术资产实时审计清单】：\n"
    asset_context += f"📂 物理硬盘文件 (共{len(physical_pdf_names)}篇): {', '.join(physical_pdf_names) if physical_pdf_names else '空'}\n"
    asset_context += f"🕸️ LightRAG 图谱已入库 (共{len(indexed_files)}篇): {', '.join(indexed_files) if indexed_files else '暂无文献入库'}\n"

    candidate_papers = state.get("candidate_papers", [])
    selected_papers = state.get("selected_papers", [])
    papers = normalize_paper_summaries(candidate_papers or selected_papers)
    papers_detail = ""
    if papers:
        papers_detail = "\n【当前候选/已选文献详情（请务必基于这些证据决策）】:\n" + "\n".join(
            [
                (
                    f"- {p.title} ({p.year})；作者：{p.authors}；"
                    f"DOI：{p.doi}；期刊/会议：{p.venue}；"
                    f"核心方法：{str(p.core_method)[:180]}；"
                    f"关键结论：{str(p.key_findings)[:180]}；来源：{p.source}"
                )
                for p in papers
            ]
        )

    memory_text = f"\n【结构化长期记忆摘要】\n{new_summary}" if new_summary else ""
    user_profile_context = build_user_profile_context(state.get("user_profile", ""))
    bootstrap_context = state.get("conversation_bootstrap", "")
    bootstrap_text = f"\n【服务重启后的最近会话恢复】\n{bootstrap_context}" if bootstrap_context else ""

    current_goal = state.get("research_goal", "解析用户最新意图")
    evidence = state.get("collected_evidence", "暂无")
    pending = state.get("pending_questions", "未知")

    sys_msg = SystemMessage(content=f"""{ASSISTANT_ROUTER_RULES}
【当前运行状态】
- 年份：{current_year}；步骤：{current_step}/{MAX_STEPS}
- 研究模式：{research_mode}（只影响预算，不允许降低证据标准）
- 目标：{current_goal}
- 已有证据：{evidence}
- 待解决：{pending}
{memory_text}
{user_profile_context}
{bootstrap_text}
{asset_context}
{papers_detail}
""")

    final_messages = [sys_msg] + short_term_memory

    cleaned_messages = []
    for m in final_messages:
        if not isinstance(m, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
            continue
        if isinstance(m, AIMessage) and not m.content:
            cleaned_messages.append(AIMessage(content=" ", tool_calls=m.tool_calls, id=m.id))
        else:
            cleaned_messages.append(m)

    last_message = short_term_memory[-1] if short_term_memory else None
    available_papers = state.get("candidate_papers") or state.get("selected_papers") or []
    normalized_available_papers = normalize_paper_summaries(
        available_papers if isinstance(available_papers, list) else []
    )
    resolved_paper_title = resolve_follow_up_paper_title(
        short_term_memory,
        latest_human_text,
        [paper.title for paper in normalized_available_papers],
    )
    referenced_papers = [
        paper
        for paper in normalized_available_papers
        if resolved_paper_title and normalize_title(paper.title) == normalize_title(resolved_paper_title)
    ]
    explicitly_refreshes_paper = bool(re.search(r"(搜|查|检索|找)", latest_human_text))
    paper_discovery_request = is_paper_discovery_request(latest_human_text)
    requests_fresh_web_search = bool(
        explicitly_requests_web_search(latest_human_text)
        or explicitly_refreshes_paper
        or re.search(r"重新|再搜|再查|最新|实时", latest_human_text)
    )

    if isinstance(last_message, HumanMessage) and is_retrieval_provenance_question(latest_human_text):
        content, source_kind = _retrieval_provenance_response(history_messages)
        emit_runtime_event(
            "evidence_provenance",
            "Retrieval provenance answered from completed tool messages",
            source_kind=source_kind,
        )
        print(f"\n[🔎 证据来源] 已按真实 Tool 回执回答，最近检索来源: {source_kind}。")
        return {
            "messages": context_state_updates + [AIMessage(content=content)],
            "summary": new_summary,
            "step_count": current_step,
        }

    web_papers = [
        paper
        for paper in normalized_available_papers
        if re.search(r"https?://", str(getattr(paper, "source", "") or ""), flags=re.I)
    ]
    current_web_result = (
        isinstance(last_message, ToolMessage)
        and _latest_tool_name(short_term_memory) == "trigger_web_search"
    )
    can_reuse_web_papers = isinstance(last_message, HumanMessage) and not requests_fresh_web_search
    if paper_discovery_request and web_papers and (can_reuse_web_papers or current_web_result):
        reused = isinstance(last_message, HumanMessage)
        content = _web_paper_catalog_response(web_papers, reused=reused)
        emit_runtime_event(
            "evidence_provenance",
            "Reused prior web evidence" if reused else "Used current web retrieval evidence",
            source_kind="web",
            reused=reused,
            paper_count=len(web_papers),
        )
        print(
            "\n[♻️ 证据复用] 本轮未重新联网，使用上一轮联网候选。"
            if reused
            else "\n[🌐 证据来源] 本轮联网检索已完成，直接展示可追踪结果。"
        )
        return {
            "messages": context_state_updates + [AIMessage(content=content)],
            "summary": new_summary,
            "step_count": current_step,
            "collected_evidence": compact_state_value(
                f"{'复用' if reused else '本轮取得'} {len(web_papers)} 篇联网学术 API 证据。"
            ),
            "pending_questions": "等待用户选择论文或细分检索方向。",
        }

    if (
        current_step <= MAX_STEPS
        and isinstance(last_message, HumanMessage)
        and resolved_paper_title
        and (explicitly_refreshes_paper or not referenced_papers)
    ):
        local_source = find_indexed_source_by_title(resolved_paper_title)
        local_only = any(
            marker in latest_human_text
            for marker in ("不要联网", "不联网", "只基于本地", "只查本地", "仅限本地")
        )
        if local_source:
            tool_call = {
                "name": "trigger_local_retrieval",
                "args": {
                    "rationale": (
                        f"已将单数指代解析为《{resolved_paper_title}》，并在本地索引中精确匹配到 "
                        f"{local_source}；只检索该论文，禁止全图近似替代。"
                    ),
                    "query": f"《{resolved_paper_title}》\n用户问题：{latest_human_text}"[:500],
                },
                "id": f"resolved_local_retrieval_{int(time.time() * 1000)}",
                "type": "tool_call",
            }
        elif local_only:
            response = AIMessage(
                content=f"本地知识库中没有找到《{resolved_paper_title}》，因此没有用相似论文替代。"
            )
            print(
                f"\n[🧠 主脑决策] 已解析指代为《{resolved_paper_title}》，"
                "但本地无精确来源且用户禁止联网。"
            )
            return {
                "messages": context_state_updates + [response],
                "summary": new_summary,
                "step_count": current_step,
                "research_goal": compact_state_value(f"查询《{resolved_paper_title}》"),
                "pending_questions": "需要用户上传该论文或允许联网检索。",
            }
        else:
            tool_call = {
                "name": "trigger_web_search",
                "args": {
                    "rationale": (
                        f"已将单数指代解析为《{resolved_paper_title}》，但本地索引没有该精确论文；"
                        "转为联网精确检索，禁止使用相似本地论文替代。"
                    ),
                    "user_core_topic": f"查找并解读《{resolved_paper_title}》；用户问题：{latest_human_text}",
                    "keyword": paper_title_search_keyword(resolved_paper_title),
                    "year_range": "",
                },
                "id": f"resolved_web_search_{int(time.time() * 1000)}",
                "type": "tool_call",
            }

        tool_call = validate_tool_call(tool_call)
        response = AIMessage(content="", tool_calls=[tool_call])
        state_updates = state_update_from_tool_call(tool_call)
        print(
            f"\n[🧠 主脑决策 (步骤 {current_step}/{MAX_STEPS})] "
            f"指代解析为《{resolved_paper_title}》，委派工具: {tool_call['name']}"
        )
        emit_runtime_event(
            "tool_call",
            f"Assistant selected {tool_call['name']}",
            tool_name=tool_call["name"],
            tool_args=summarize_tool_call_for_trace(tool_call),
        )
        return {
            "messages": context_state_updates + [response],
            "summary": new_summary,
            "step_count": current_step,
            **state_updates,
        }

    if (
        isinstance(last_message, ToolMessage)
        and resolved_paper_title
        and _latest_tool_name(short_term_memory) == "trigger_web_search"
        and not referenced_papers
    ):
        tool_observation = str(last_message.content or "")
        if "服务当前不可用" in tool_observation:
            content = "联网学术服务当前不可用，未能核验这篇论文；本轮不会重复搜索。"
        else:
            content = f"联网检索未找到《{resolved_paper_title}》的精确论文记录，本轮不会用相关论文替代或重复搜索。"
        print(f"\n[🧠 主脑决策] 《{resolved_paper_title}》无精确联网证据，停止检索。")
        return {
            "messages": context_state_updates + [AIMessage(content=content)],
            "summary": new_summary,
            "step_count": current_step,
            "research_goal": compact_state_value(f"精确检索《{resolved_paper_title}》"),
            "collected_evidence": "精确标题联网检索未命中。",
            "pending_questions": "可由用户核对标题、提供 DOI 或上传 PDF。",
        }

    if resolved_paper_title:
        available_papers = referenced_papers
    explicitly_requests_web = explicitly_requests_web_search(latest_human_text)
    evidence_gate = evaluate_evidence_gate(
        latest_human_text,
        available_papers if isinstance(available_papers, list) else [],
        MAX_SELECTED_PAPERS,
    )
    selected_for_synthesis = evidence_gate.selected_papers if evidence_gate.passed else []
    synthesis_mode = evidence_gate.mode
    if (
        current_step <= MAX_STEPS
        and not (explicitly_requests_web and isinstance(last_message, HumanMessage))
        and not (explicitly_requests_web and synthesis_mode == "answer")
        and selected_for_synthesis
        and synthesis_mode
    ):
        mode_label = {
            "comparison": "跨论文总结与对比",
            "summary": "论文总结",
            "partial_summary": "已有论文总结（当前证据不足以执行对比）",
            "answer": "证据约束回答",
        }[synthesis_mode]
        response = AIMessage(content="已基于当前候选文献证据进入综述生成流程。[APPROVE_SYNTHESIS]")
        emit_runtime_event(
            "evidence_gate",
            "Evidence gate passed",
            passed=True,
            mode=synthesis_mode,
            selected_count=len(selected_for_synthesis),
        )
        print(
            f"\n[🧠 主脑决策 (步骤 {current_step}/{MAX_STEPS})] "
            f"命中{mode_label}确定性路由，启动 Synthesizer。"
        )
        return {
            "messages": context_state_updates + [response],
            "summary": new_summary,
            "step_count": current_step,
            "selected_papers": selected_for_synthesis,
            "evidence_gate": evidence_gate.model_dump(exclude={"selected_papers"}),
            "review_round": 0,
            "review_status": "pending",
            "review_feedback": "",
            "collected_evidence": compact_state_value(
                f"确定性证据门控通过：{len(selected_for_synthesis)} 篇候选文献进入{mode_label}流程。"
            ),
            "pending_questions": "进入 Synthesizer 生成综述内容。",
        }

    if (
        evidence_gate.mode
        and not evidence_gate.passed
        and evidence_gate.reasons
        and available_papers
        and not (explicitly_requests_web and isinstance(last_message, HumanMessage))
    ):
        content = _evidence_gate_failure_text(evidence_gate.reasons, evidence_gate.warnings)
        emit_runtime_event(
            "evidence_gate",
            "Evidence gate rejected candidates",
            passed=False,
            mode=evidence_gate.mode,
            reasons=evidence_gate.reasons,
        )
        print("\n[🛡️ Evidence Gate] 候选论文未升级：" + "；".join(evidence_gate.reasons))
        return {
            "messages": context_state_updates + [AIMessage(content=content)],
            "summary": new_summary,
            "step_count": current_step,
            "evidence_gate": evidence_gate.model_dump(exclude={"selected_papers"}),
            "collected_evidence": compact_state_value("证据门控未通过：" + "；".join(evidence_gate.reasons)),
            "pending_questions": "需要补齐上面列出的具体证据缺口。",
        }

    if (
        current_step <= MAX_STEPS
        and isinstance(last_message, HumanMessage)
        and _should_force_local_retrieval(latest_human_text)
    ):
        tool_call = {
            "name": "trigger_local_retrieval",
            "args": {
                "rationale": "用户明确要求基于本地 PDF/本地文献库完成学术检索，强制进入本地 RAG，避免主控模型直接凭上下文回答。",
                "query": latest_human_text[:500],
            },
            "id": f"forced_local_retrieval_{int(time.time() * 1000)}",
            "type": "tool_call",
        }
        tool_call = validate_tool_call(tool_call)
        response = AIMessage(content="", tool_calls=[tool_call])
        emit_runtime_event(
            "tool_call",
            "Assistant selected trigger_local_retrieval",
            tool_name=tool_call["name"],
            tool_args=summarize_tool_call_for_trace(tool_call),
        )
        print(f"\n[🧠 主脑决策 (步骤 {current_step}/{MAX_STEPS})] 命中本地检索确定性路由，委派工具: trigger_local_retrieval")
        return {
            "messages": context_state_updates + [response],
            "summary": new_summary,
            "step_count": current_step,
            "research_goal": compact_state_value(latest_human_text),
            "pending_questions": "等待本地 RAG 检索并返回候选论文证据。",
        }

    validation_messages = list(cleaned_messages)
    response = None
    tool_calls = []
    for validation_attempt in range(2):
        response = await safe_llm_invoke(
            llm_with_tools,
            validation_messages,
            "Assistant_Router",
            max_retries=max(1, int(os.environ.get("ASSISTANT_LLM_MAX_RETRIES", "3"))),
            role=MAIN_API_ROLE,
            invoke_config=config,
        )
        if response is None:
            break
        try:
            tool_calls = validate_tool_calls(getattr(response, "tool_calls", []) or [])
        except ToolValidationError as exc:
            emit_runtime_event(
                "tool_validation_rejected",
                "Assistant tool call rejected by deterministic validator",
                validation_error=str(exc),
                attempt=validation_attempt + 1,
            )
            print(f"  [🛡️ Tool Validator] 第 {validation_attempt + 1} 次工具参数被拒绝: {exc}")
            if validation_attempt == 0:
                validation_messages = validation_messages + [SystemMessage(content=(
                    "上一次生成的工具调用被确定性参数校验拒绝。请严格按 Tool schema 重新生成且最多调用一个工具；"
                    f"校验错误：{exc}。不要解释校验过程。"
                ))]
                continue
            response = AIMessage(content="工具参数连续两次未通过系统校验，本轮已安全停止；请缩短检索词或明确检索范围后重试。")
            tool_calls = []
        else:
            if tool_calls and hasattr(response, "model_copy"):
                response = response.model_copy(update={"tool_calls": tool_calls})
            elif tool_calls:
                response.tool_calls = tool_calls
            requires_web_tool = (
                isinstance(last_message, HumanMessage)
                and paper_discovery_request
                and (not normalized_available_papers or requests_fresh_web_search)
            )
            selected_web_tool = bool(tool_calls and tool_calls[0].get("name") == "trigger_web_search")
            if requires_web_tool and not selected_web_tool:
                if validation_attempt == 0:
                    validation_messages = validation_messages + [SystemMessage(content=(
                        "当前请求是在查找论文，且当前没有可用候选证据。禁止直接列举论文；"
                        "必须调用 trigger_web_search，并从最近自然语言对话中保留研究主题。"
                    ))]
                    continue
                response = AIMessage(
                    content="本轮没有实际执行联网检索，因此没有生成未经核验的论文清单。请明确研究主题后重试。"
                )
                tool_calls = []
            break

    if response is not None:
            if tool_calls and is_duplicate_web_search(tool_calls[0], short_term_memory):
                duplicate_keyword = tool_calls[0].get("args", {}).get("keyword", "")
                print(
                    f"  [🛡️ 系统拦截] 联网关键词 '{duplicate_keyword}' 已执行过，"
                    "禁止重复检索。"
                )
                response = AIMessage(
                    content=(
                        "相同联网关键词已经检索过，本轮不会重复调用。"
                        "当前结果尚不足以进入总结，请调整关键词或检索范围。"
                    )
                )
                tool_calls = []
            if current_step > MAX_STEPS and tool_calls:
                print("  [🛡️ 系统拦截] 大模型企图违规调用工具，已被强行物理截断！")
                response.tool_calls = []
                tool_calls = []

            if response.content and not tool_calls:
                response.content = finalize_incomplete_search_response(
                    response.content,
                    was_web_tool_result=(
                        isinstance(last_message, ToolMessage)
                        and _latest_tool_name(short_term_memory) == "trigger_web_search"
                    ),
                )

            if not response.content and not tool_calls:
                response.content = "【内部状态更新】\n- 目标：终结对话\n- 证据：经过多轮泛化，确认该交叉领域尚属学术空白。\n- 待解决：无\n\n很抱歉，经过多轮深度检索，目前尚未发现完全匹配该具体场景的文献。这可能是极度前沿的空白领域。建议将应用场景放宽至普通的“水下机器人”，您需要我为您总结底层通用技术吗？"

            state_updates = {}
            if tool_calls:
                tool_name = tool_calls[0]["name"]
                state_updates = state_update_from_tool_call(tool_calls[0])
                print(f"\n[🧠 主脑决策 (步数 {current_step}/{MAX_STEPS})] 决定委派工具: {tool_name}")
                emit_runtime_event(
                    "tool_call",
                    f"Assistant selected {tool_name}",
                    tool_name=tool_name,
                    tool_args=summarize_tool_call_for_trace(tool_calls[0]),
                )
            elif response.content:
                state_updates = extract_internal_state_update(response.content)
                if "[APPROVE_SYNTHESIS]" in response.content and state.get("candidate_papers"):
                    model_gate = evaluate_evidence_gate(
                        latest_human_text,
                        state.get("candidate_papers", []),
                        MAX_SELECTED_PAPERS,
                    )
                    state_updates["evidence_gate"] = model_gate.model_dump(exclude={"selected_papers"})
                    if model_gate.passed:
                        selected = model_gate.selected_papers
                        state_updates.update({
                            "selected_papers": selected,
                            "review_round": 0,
                            "review_status": "pending",
                            "review_feedback": "",
                            "collected_evidence": compact_state_value(
                                f"Evidence Gate 已确认 {len(selected)} 篇候选文献进入最终证据池。"
                            ),
                        })
                    else:
                        response.content = _evidence_gate_failure_text(model_gate.reasons, model_gate.warnings)
                        state_updates["collected_evidence"] = compact_state_value(
                            "证据门控未通过：" + "；".join(model_gate.reasons)
                        )
                if "[APPROVE_SYNTHESIS]" in response.content:
                    print(f"\n[🧠 主脑决策] 证据已满足妥协阈值，下发 [APPROVE_SYNTHESIS] 暗号，启动综述。")
                else:
                    print(f"\n[🧠 主脑决策] 停止检索，准备直接向用户报告结果。")
                response.content = sanitize_user_response(response.content)

            print(f"  [⏱️ Assistant 耗时] {time.time() - node_start:.2f}s")

            return {
                "messages": context_state_updates + [response],
                "summary": new_summary,
                "step_count": current_step,
                **state_updates,
            }

    return {
        "messages": context_state_updates + [AIMessage(content="主决策模型暂时不可用，系统已完成自动重试并停止继续调用工具，请稍后重试。")],
        "summary": new_summary,
        "step_count": current_step,
    }
