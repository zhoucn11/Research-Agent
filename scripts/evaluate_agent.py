"""对正在运行的 Research Agent 做可复现的端到端 SSE 评测。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "docs" / "AGENT_EVAL_DATASET.jsonl"
DEFAULT_RAW_REPORT = ROOT / "AGENT_EVAL_RESULTS.json"
DEFAULT_MD_REPORT = ROOT / "AGENT_EVAL_REPORT.md"
TOOL_PATTERN = re.compile(r"委派工具\s*[:：]\s*(trigger_[a-z_]+)", re.I)
FALLBACK_TOOL_PATTERN = re.compile(r"接收到调度指令\s*[:：]\s*(trigger_[a-z_]+)", re.I)
KEYWORD_PATTERN = re.compile(r"检索词\s*['‘“\"]([^'’”\"]+)", re.I)
CITATION_PATTERN = re.compile(r"\[(\d+):(p(\d+)|摘要)\]")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSONL: {exc}") from exc
    ids = [case["id"] for case in cases]
    if len(cases) < 20 or len(set(ids)) != len(cases):
        raise ValueError(f"测试集必须至少包含 20 个唯一题目，当前为 {len(cases)}")
    for case in cases:
        evidence_ids = [item["id"] for item in case.get("gold_evidence", [])]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"{case['id']} 的 gold_evidence.id 必须唯一")
        unknown_ids = {
            evidence_id
            for claim in case.get("gold_claims", [])
            for evidence_id in claim.get("evidence_ids", [])
            if evidence_id not in evidence_ids
        }
        if unknown_ids:
            raise ValueError(f"{case['id']} 的 gold_claims 引用了未知证据: {sorted(unknown_ids)}")
    return cases


def request_json(method: str, url: str, user_id: str, timeout: float, **kwargs) -> dict:
    response = requests.request(
        method,
        url,
        headers={"X-User-ID": user_id},
        timeout=(10, timeout),
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def create_session(base_url: str, user_id: str, scenario_id: str, timeout: float) -> str:
    payload = request_json(
        "POST",
        f"{base_url}/api/sessions",
        user_id,
        timeout,
        json={"title": f"eval-{scenario_id}"},
    )
    return str(payload["session_id"])


def delete_session(base_url: str, user_id: str, session_id: str, timeout: float) -> None:
    try:
        request_json("DELETE", f"{base_url}/api/sessions/{session_id}", user_id, timeout)
    except requests.RequestException:
        pass


def stream_chat(
    base_url: str,
    user_id: str,
    session_id: str,
    message: str,
    mode: str,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    first_token_at = None
    final_content = ""
    status_code = None
    error = ""
    try:
        with requests.post(
            f"{base_url}/api/chat",
            headers={"X-User-ID": user_id, "X-Eval-Mode": "1", "Accept": "text/event-stream"},
            json={"message": message, "session_id": session_id, "mode": mode},
            stream=True,
            timeout=(10, timeout),
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload_text = raw_line[5:].strip()
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError:
                    events.append({"type": "invalid_sse", "content": payload_text})
                    continue
                events.append(event)
                if event.get("type") == "token" and first_token_at is None:
                    first_token_at = time.perf_counter()
                if event.get("type") == "final":
                    final_content = str(event.get("content") or "")
    except requests.RequestException as exc:
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    log_events = [event for event in events if event.get("type") == "log"]
    log_text = "\n".join(str(event.get("content") or "") for event in log_events)
    tools = TOOL_PATTERN.findall(log_text)
    if not tools:
        tools = FALLBACK_TOOL_PATTERN.findall(log_text)
    keywords = [re.sub(r"\s+", " ", item.strip().lower()) for item in KEYWORD_PATTERN.findall(log_text)]
    server_ttft = next(
        (float(event["latency_ms"]) for event in log_events if event.get("event") == "ttft" and event.get("latency_ms") is not None),
        None,
    )
    retrieval_events = [event for event in events if event.get("type") == "retrieval"]
    retrieved_papers = retrieval_events[-1].get("papers", []) if retrieval_events else []
    retrieved_evidence = []
    for paper in retrieved_papers:
        reference_index = paper.get("reference_index")
        for span in paper.get("evidence_spans", []) or []:
            retrieved_evidence.append({**span, "reference_index": reference_index})
    return {
        "status_code": status_code,
        "error": error,
        "answer": final_content,
        "events": events,
        "logs": log_text,
        "tools": tools,
        "search_keywords": keywords,
        "retrieval_event_observed": bool(retrieval_events),
        "retrieved_papers": retrieved_papers,
        "retrieved_evidence": retrieved_evidence,
        "ttft_client_ms": round((first_token_at - started) * 1000, 2) if first_token_at else None,
        "ttft_server_ms": server_ttft,
        "e2e_ms": round((ended - started) * 1000, 2),
        "completed": bool(final_content) and not error,
    }


def pattern_matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.I | re.S) is not None


def check_fact_groups(case: dict[str, Any], answer: str) -> tuple[float, list[dict[str, Any]]]:
    checks = case.get("checks") or []
    if not checks:
        return 1.0, []
    details = []
    for check in checks:
        if "all" in check:
            passed = all(pattern_matches(pattern, answer) for pattern in check["all"])
        else:
            passed = any(pattern_matches(pattern, answer) for pattern in check.get("any", []))
        details.append({"name": check["name"], "passed": passed})
    return sum(item["passed"] for item in details) / len(details), details


def check_format(case: dict[str, Any], answer: str) -> tuple[float, list[str]]:
    config = case.get("format") or {}
    failures = []
    compact_answer = answer.strip()
    if config.get("max_chars") is not None and len(compact_answer) > int(config["max_chars"]):
        failures.append(f"长度 {len(compact_answer)} > {config['max_chars']}")
    if config.get("max_sentences") is not None:
        sentence_count = len(re.findall(r"[。！？.!?]+(?:\s|$)", compact_answer))
        sentence_count = max(sentence_count, 1 if compact_answer else 0)
        if sentence_count > int(config["max_sentences"]):
            failures.append(f"句数 {sentence_count} > {config['max_sentences']}")
    has_table = bool(re.search(r"^\s*\|.+\|\s*$", answer, flags=re.M))
    if config.get("must_markdown_table") and not has_table:
        failures.append("缺少 Markdown 表格")
    if config.get("forbid_markdown_table") and has_table:
        failures.append("不应包含 Markdown 表格")
    return (0.0 if failures else 1.0), failures


def tool_scores(expected: list[str], actual: list[str]) -> dict[str, float]:
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    true_positive = sum((expected_counter & actual_counter).values())
    if not expected and not actual:
        precision = recall = f1 = exact = 1.0
    else:
        precision = true_positive / len(actual) if actual else 0.0
        recall = true_positive / len(expected) if expected else (1.0 if not actual else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = float(expected_counter == actual_counter)
    return {"tool_precision": precision, "tool_recall": recall, "tool_f1": f1, "tool_exact_match": exact}


def _source_identity(value: str) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _rule_matches(rule: dict, text: str) -> bool:
    if "all" in rule:
        return all(pattern_matches(pattern, text) for pattern in rule["all"])
    return any(pattern_matches(pattern, text) for pattern in rule.get("any", []))


def _page_intersects(span: dict, pages: list[int]) -> bool:
    if not pages:
        return True
    start = span.get("page_start")
    end = span.get("page_end") or start
    if not isinstance(start, int):
        return False
    return any(start <= page <= int(end) for page in pages)


def _evidence_matches(gold: dict, span: dict) -> bool:
    if _source_identity(gold.get("source")) != _source_identity(span.get("source")):
        return False
    if not _page_intersects(span, [int(page) for page in gold.get("pages", [])]):
        return False
    anchors = gold.get("anchors", [])
    return not anchors or any(pattern_matches(pattern, str(span.get("quote") or "")) for pattern in anchors)


def _citation_supports(answer: str, papers: list[dict], gold: dict) -> bool:
    references = {
        int(paper.get("reference_index")): _source_identity(paper.get("source"))
        for paper in papers
        if isinstance(paper.get("reference_index"), int)
    }
    gold_source = _source_identity(gold.get("source"))
    gold_pages = {int(page) for page in gold.get("pages", [])}
    for match in CITATION_PATTERN.finditer(answer):
        reference_index = int(match.group(1))
        page = int(match.group(3)) if match.group(3) else None
        if references.get(reference_index) != gold_source:
            continue
        if not gold_pages or page in gold_pages:
            return True
    return False


def evidence_scores(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    papers = run.get("retrieved_papers") or []
    spans = run.get("retrieved_evidence") or []
    gold_evidence = case.get("gold_evidence") or []
    gold_sources = case.get("gold_sources") or case.get("expected_sources") or []

    source_recall = None
    if gold_sources:
        actual_sources = {_source_identity(paper.get("source")) for paper in papers}
        source_recall = sum(_source_identity(source) in actual_sources for source in gold_sources) / len(gold_sources)

    matched_gold_ids = {
        gold["id"]
        for gold in gold_evidence
        if any(_evidence_matches(gold, span) for span in spans)
    }
    evidence_recall = len(matched_gold_ids) / len(gold_evidence) if gold_evidence else None
    evidence_precision = None
    if gold_evidence:
        evidence_precision = (
            sum(any(_evidence_matches(gold, span) for gold in gold_evidence) for span in spans) / len(spans)
            if spans else 0.0
        )

    citations = list(CITATION_PATTERN.finditer(run.get("answer", "")))
    citation_correctness = None
    if citations:
        references = {
            int(paper.get("reference_index")): _source_identity(paper.get("source"))
            for paper in papers
            if isinstance(paper.get("reference_index"), int)
        }
        correct = 0
        for citation in citations:
            reference_index = int(citation.group(1))
            page = int(citation.group(3)) if citation.group(3) else None
            source = references.get(reference_index)
            if source and any(
                _source_identity(span.get("source")) == source
                and (page is None or _page_intersects(span, [page]))
                for span in spans
            ):
                correct += 1
        citation_correctness = correct / len(citations)
    elif case.get("citation_required"):
        citation_correctness = 0.0

    evidence_by_id = {item["id"]: item for item in gold_evidence}
    claim_details = []
    for claim in case.get("gold_claims", []) or []:
        answer_present = _rule_matches(claim.get("answer", {}), run.get("answer", ""))
        supporting_ids = claim.get("evidence_ids", [])
        evidence_checks = [evidence_id in matched_gold_ids for evidence_id in supporting_ids]
        match_any = claim.get("evidence_match") == "any"
        evidence_present = (
            any(evidence_checks) if match_any else bool(evidence_checks) and all(evidence_checks)
        )
        citation_supported = True
        if claim.get("citation_required", case.get("citation_required", False)):
            citation_checks = [
                _citation_supports(run.get("answer", ""), papers, evidence_by_id[evidence_id])
                for evidence_id in supporting_ids
                if evidence_id in evidence_by_id
            ]
            citation_supported = (
                any(citation_checks) if match_any else bool(citation_checks) and all(citation_checks)
            )
        claim_details.append({
            "name": claim["name"],
            "answer_present": answer_present,
            "evidence_present": evidence_present,
            "citation_supported": citation_supported,
            "passed": answer_present and evidence_present and citation_supported,
        })
    claim_support = (
        sum(detail["passed"] for detail in claim_details) / len(claim_details)
        if claim_details else None
    )
    return {
        "source_recall": source_recall,
        "evidence_recall": evidence_recall,
        "evidence_precision": evidence_precision,
        "citation_correctness": citation_correctness,
        "claim_support": claim_support,
        "matched_gold_evidence": sorted(matched_gold_ids),
        "claim_details": claim_details,
    }


def score_case(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    answer = run["answer"]
    fact_coverage, fact_details = check_fact_groups(case, answer)
    forbidden_hits = [pattern for pattern in case.get("forbidden", []) if pattern_matches(pattern, answer)]
    forbidden_compliance = 0.0 if forbidden_hits else 1.0
    format_score, format_failures = check_format(case, answer)
    expected_options = case.get("acceptable_tool_sequences") or [case.get("expected_tools", [])]
    route_scores = max(
        (tool_scores(expected, run["tools"]) for expected in expected_options),
        key=lambda scores: (scores["tool_exact_match"], scores["tool_f1"]),
    )
    within_tool_budget = float(len(run["tools"]) <= int(case.get("max_tool_calls", 99)))
    unique_keywords = float(len(run["search_keywords"]) == len(set(run["search_keywords"])))
    citation_score = None
    if case.get("citation_required"):
        citation_score = float(bool(CITATION_PATTERN.search(answer)))
    evidence = evidence_scores(case, run)
    goal_accuracy = statistics.fmean([fact_coverage, forbidden_compliance, format_score])
    return {
        "completion": float(run["completed"]),
        "goal_accuracy": goal_accuracy,
        "factual_keypoint_coverage": fact_coverage,
        "forbidden_claim_compliance": forbidden_compliance,
        "format_compliance": format_score,
        "citation_presence": citation_score,
        "within_tool_budget": within_tool_budget,
        "unique_web_search_keywords": unique_keywords,
        **route_scores,
        "fact_details": fact_details,
        "forbidden_hits": forbidden_hits,
        "format_failures": format_failures,
        **evidence,
    }


def percentile(values: list[float], percent_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mean_present(results: list[dict], metric: str) -> float | None:
    values = [result["scores"].get(metric) for result in results]
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_metrics = [
        "completion", "goal_accuracy", "factual_keypoint_coverage", "forbidden_claim_compliance",
        "format_compliance", "citation_presence", "citation_correctness", "source_recall",
        "evidence_recall", "evidence_precision", "claim_support",
        "tool_precision", "tool_recall", "tool_f1", "tool_exact_match", "within_tool_budget",
        "unique_web_search_keywords",
    ]
    metrics = {name: mean_present(results, name) for name in score_metrics}
    multi_turn = [result for result in results if int(result["case"].get("turn", 1)) > 1]
    metrics["multi_turn_goal_accuracy"] = mean_present(multi_turn, "goal_accuracy")
    ttft_values = [result["run"]["ttft_client_ms"] for result in results if result["run"]["ttft_client_ms"] is not None]
    e2e_values = [result["run"]["e2e_ms"] for result in results]
    latency = {
        "ttft_p50_ms": percentile(ttft_values, 0.50), "ttft_p95_ms": percentile(ttft_values, 0.95),
        "e2e_p50_ms": percentile(e2e_values, 0.50), "e2e_p95_ms": percentile(e2e_values, 0.95),
    }
    latency_by_mode = {}
    for mode in ("quick", "deep"):
        mode_results = [result for result in results if result["case"].get("mode") == mode]
        mode_ttft = [result["run"]["ttft_client_ms"] for result in mode_results if result["run"]["ttft_client_ms"] is not None]
        mode_e2e = [result["run"]["e2e_ms"] for result in mode_results]
        latency_by_mode[mode] = {
            "count": len(mode_results),
            "ttft_p50_ms": percentile(mode_ttft, 0.50),
            "ttft_p95_ms": percentile(mode_ttft, 0.95),
            "e2e_p50_ms": percentile(mode_e2e, 0.50),
            "e2e_p95_ms": percentile(mode_e2e, 0.95),
        }
    weighted_parts = [
        (metrics["goal_accuracy"], 0.25), (metrics["claim_support"], 0.20),
        (metrics["evidence_recall"], 0.15), (metrics["evidence_precision"], 0.10),
        (metrics["citation_correctness"], 0.10), (metrics["tool_f1"], 0.10),
        (metrics["multi_turn_goal_accuracy"], 0.05), (metrics["completion"], 0.05),
    ]
    weighted_total = sum(value * weight for value, weight in weighted_parts if value is not None)
    used_weight = sum(weight for value, weight in weighted_parts if value is not None)
    return {
        "case_count": len(results), "metrics": metrics, "latency": latency, "latency_by_mode": latency_by_mode,
        "composite_score": weighted_total / used_weight if used_weight else 0.0,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def milliseconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f} ms"


def _key_findings(report: dict[str, Any]) -> list[str]:
    summary = report["summary"]
    metrics = summary["metrics"]
    findings = [
        f"本轮完成 {summary['case_count']} 个用户回合，完成率 {percent(metrics['completion'])}，"
        f"工具调用 F1 为 {percent(metrics['tool_f1'])}。",
        f"真实证据指标：Source Recall {percent(metrics['source_recall'])}，"
        f"Evidence Recall/Precision {percent(metrics['evidence_recall'])}/{percent(metrics['evidence_precision'])}，"
        f"Claim Support {percent(metrics['claim_support'])}，Citation Correctness {percent(metrics['citation_correctness'])}。",
    ]
    protocol_missing = [
        result["case"]["id"] for result in report["results"]
        if result["case"].get("gold_evidence") and not result["run"].get("retrieval_event_observed")
    ]
    if protocol_missing:
        findings.append("评测证据事件缺失：" + "、".join(protocol_missing) + "；这些题的真实证据指标按 0 计。")
    failures = sorted(
        report["results"],
        key=lambda item: (
            item["scores"]["claim_support"] if item["scores"]["claim_support"] is not None else 1.0,
            item["scores"]["goal_accuracy"],
        ),
    )[:3]
    failed_labels = [
        f"{item['case']['id']}(Goal {percent(item['scores']['goal_accuracy'])}, "
        f"Claim {percent(item['scores']['claim_support'])})"
        for item in failures
        if item["scores"]["goal_accuracy"] < 1.0
        or (
            item["scores"]["claim_support"] is not None
            and item["scores"]["claim_support"] < 1.0
        )
    ]
    if failed_labels:
        findings.append("优先复查低分题：" + "、".join(failed_labels) + "。")
    slowest = max(report["results"], key=lambda item: item["run"].get("e2e_ms") or 0, default=None)
    if slowest:
        findings.append(
            f"最慢题为 {slowest['case']['id']}：TTFT {milliseconds(slowest['run'].get('ttft_client_ms'))}，"
            f"E2E {milliseconds(slowest['run'].get('e2e_ms'))}。"
        )
    return findings


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    metrics = summary["metrics"]
    latency = summary["latency"]
    latency_by_mode = summary["latency_by_mode"]
    lines = [
        "# Research Agent 端到端评测报告", "",
        f"- 评测时间：{report['evaluated_at']}", f"- 服务地址：`{report['base_url']}`",
        f"- 测试题数：{summary['case_count']}（含多轮会话）",
        f"- 综合分：**{summary['composite_score'] * 100:.1f}/100**", "",
        "## 汇总指标", "", "| 指标 | 结果 | 说明 |", "|:---|---:|:---|",
        f"| Agent Goal Accuracy | {percent(metrics['goal_accuracy'])} | 关键点、禁编造和格式约束的平均完成度 |",
        f"| Factual Keypoint Coverage | {percent(metrics['factual_keypoint_coverage'])} | 参考答案关键事实召回率 |",
        f"| Source Recall | {percent(metrics['source_recall'])} | 最终状态覆盖标准论文来源的比例 |",
        f"| Evidence Recall | {percent(metrics['evidence_recall'])} | 命中人工标注 source/page/anchor 证据单元的比例 |",
        f"| Evidence Precision | {percent(metrics['evidence_precision'])} | 返回 EvidenceSpan 中与本题标准证据匹配的比例 |",
        f"| Claim Support | {percent(metrics['claim_support'])} | 答案关键声明同时具备标准证据和正确页码引用的比例 |",
        f"| Citation Correctness | {percent(metrics['citation_correctness'])} | 页码引用可回链到本轮实际 EvidenceSpan 的比例 |",
        f"| Tool Call F1 | {percent(metrics['tool_f1'])} | 期望工具与实际工具调用的 F1 |",
        f"| Tool Exact Match | {percent(metrics['tool_exact_match'])} | 工具调用及次数完全匹配率 |",
        f"| Citation Presence | {percent(metrics['citation_presence'])} | 要求引用时是否至少出现一个页码引用，仅作格式诊断 |",
        f"| Multi-turn Goal Accuracy | {percent(metrics['multi_turn_goal_accuracy'])} | 第 2 轮及以后问题的目标完成度 |",
        f"| Format Compliance | {percent(metrics['format_compliance'])} | 表格、字数、句数等约束遵循率 |",
        f"| Completion Rate | {percent(metrics['completion'])} | HTTP/SSE 正常收到 final 的比例 |",
        f"| Tool Budget Compliance | {percent(metrics['within_tool_budget'])} | 未超过允许的工具调用次数 |", "",
        "## 延迟", "",
        f"- TTFT：P50 {milliseconds(latency['ttft_p50_ms'])}，P95 {milliseconds(latency['ttft_p95_ms'])}",
        f"- 端到端：P50 {milliseconds(latency['e2e_p50_ms'])}，P95 {milliseconds(latency['e2e_p95_ms'])}", "",
        "| 模式 | 题数 | TTFT P50 / P95 | E2E P50 / P95 |",
        "|:---|---:|---:|---:|",
        f"| quick | {latency_by_mode['quick']['count']} | {milliseconds(latency_by_mode['quick']['ttft_p50_ms'])} / {milliseconds(latency_by_mode['quick']['ttft_p95_ms'])} | {milliseconds(latency_by_mode['quick']['e2e_p50_ms'])} / {milliseconds(latency_by_mode['quick']['e2e_p95_ms'])} |",
        f"| deep | {latency_by_mode['deep']['count']} | {milliseconds(latency_by_mode['deep']['ttft_p50_ms'])} / {milliseconds(latency_by_mode['deep']['ttft_p95_ms'])} | {milliseconds(latency_by_mode['deep']['e2e_p50_ms'])} / {milliseconds(latency_by_mode['deep']['e2e_p95_ms'])} |", "",
        "## 关键发现", "",
        *[f"- {finding}" for finding in _key_findings(report)], "",
        "## 逐题结果", "",
        "| ID | 场景 | Goal | Claim | E-R/P | Citation | Tool F1 | TTFT | E2E | 主要失败 |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for result in report["results"]:
        case, scores, run = result["case"], result["scores"], result["run"]
        failures = [item["name"] for item in scores["fact_details"] if not item["passed"]]
        failures.extend(scores["format_failures"])
        failures.extend(f"禁用声明:{item}" for item in scores["forbidden_hits"])
        if run["error"]:
            failures.append(run["error"][:80])
        failures.extend(detail["name"] for detail in scores["claim_details"] if not detail["passed"])
        if scores["citation_presence"] == 0:
            failures.append("缺少页级引用")
        failure_text = "；".join(failures) or "—"
        lines.append(
            f"| {case['id']} | {case['category']} | {percent(scores['goal_accuracy'])} | "
            f"{percent(scores['claim_support'])} | {percent(scores['evidence_recall'])}/{percent(scores['evidence_precision'])} | "
            f"{percent(scores['citation_correctness'])} | {percent(scores['tool_f1'])} | "
            f"{milliseconds(run['ttft_client_ms'])} | "
            f"{milliseconds(run['e2e_ms'])} | {failure_text} |"
        )
    lines.extend([
        "", "## 口径说明", "",
        "本报告使用人工标注的 source/page/anchor 与评测专用 `retrieval` SSE 事件做确定性核对。"
        "`Claim Support` 要求关键声明、实际 EvidenceSpan 和正确页码引用同时成立；它不依赖另一个大模型裁判，"
        "但也不等同于开放域语义 Faithfulness。联网动态结果只评路由、来源和格式，不纳入本地金标证据指标。", "",
        "综合分权重：Goal Accuracy 25%、Claim Support 20%、Evidence Recall 15%、Evidence Precision 10%、"
        "Citation Correctness 10%、Tool F1 10%、Multi-turn Goal Accuracy 5%、Completion Rate 5%。"
        "原始答案、评测证据事件和逐项判定见 `AGENT_EVAL_RESULTS.json`。", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the live Research Agent SSE API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-report", type=Path, default=DEFAULT_RAW_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--timeout", type=float, default=360.0, help="每道题最大等待秒数")
    parser.add_argument("--keep-sessions", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="仅调试前 N 题；正式评测应保持 0")
    parser.add_argument("--rescore-existing", type=Path, help="不请求服务，按当前数据集重新评分已有原始结果")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")

    cases = load_dataset(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.rescore_existing:
        previous = json.loads(args.rescore_existing.read_text(encoding="utf-8"))
        cases_by_id = {case["id"]: case for case in cases}
        results = []
        for old_result in previous.get("results", []):
            case_id = old_result["case"]["id"]
            case = cases_by_id[case_id]
            results.append(
                {
                    "case": case,
                    "session_id": old_result.get("session_id", ""),
                    "run": old_result["run"],
                    "scores": score_case(case, old_result["run"]),
                }
            )
        report = {
            **{key: value for key, value in previous.items() if key not in {"summary", "results"}},
            "status": "completed",
            "rescored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": aggregate_results(results),
            "results": results,
        }
        args.raw_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        args.md_report.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"重新评分完成: {report['summary']['composite_score'] * 100:.1f}/100")
        return 0
    user_id = "agent_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions: dict[str, str] = {}
    results = []
    try:
        health = requests.get(f"{args.base_url}/", timeout=(5, 10))
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"服务不可用: {exc}", file=sys.stderr)
        return 2

    try:
        for index, case in enumerate(cases, start=1):
            scenario_id = case["scenario_id"]
            if scenario_id not in sessions:
                sessions[scenario_id] = create_session(args.base_url, user_id, scenario_id, args.timeout)
            print(f"[{index:02d}/{len(cases):02d}] {case['id']} {case['category']} ...", flush=True)
            run = stream_chat(
                args.base_url, user_id, sessions[scenario_id], case["prompt"], case.get("mode", "auto"), args.timeout,
            )
            scores = score_case(case, run)
            results.append({"case": case, "session_id": sessions[scenario_id], "run": run, "scores": scores})
            args.raw_report.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "in_progress",
                        "base_url": args.base_url,
                        "user_id": user_id,
                        "completed_cases": len(results),
                        "results": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"    goal={scores['goal_accuracy'] * 100:.0f}% tool_f1={scores['tool_f1'] * 100:.0f}% "
                f"ttft={run['ttft_client_ms']}ms e2e={run['e2e_ms']}ms", flush=True,
            )
    finally:
        if not args.keep_sessions:
            for session_id in sessions.values():
                delete_session(args.base_url, user_id, session_id, args.timeout)

    report = {
        "schema_version": 2, "status": "completed",
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": args.base_url, "user_id": user_id, "dataset": str(args.dataset),
        "summary": aggregate_results(results), "results": results,
    }
    args.raw_report.parent.mkdir(parents=True, exist_ok=True)
    args.md_report.parent.mkdir(parents=True, exist_ok=True)
    args.raw_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md_report.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"\n综合分: {report['summary']['composite_score'] * 100:.1f}/100")
    print(f"Markdown 报告: {args.md_report}")
    print(f"原始结果: {args.raw_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
