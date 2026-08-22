import asyncio
import json

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.agents import synthesis_agent
from research_agent.schemas.models import EvidenceSpan, PaperSummary


class _FakeReviewerLLM:
    def with_config(self, **kwargs):
        return self


def _paper() -> PaperSummary:
    return PaperSummary(
        title="Paper A",
        authors="Alice Smith",
        year="2025",
        source="Paper A.pdf",
        core_method="A grounded method.",
        key_findings="A grounded finding.",
        evidence_spans=[EvidenceSpan(
            source="Paper A.pdf",
            page_start=3,
            section="Method",
            chunk_id="chunk-3",
            quote="The paper introduces a grounded method.",
            confidence=0.9,
        )],
    )


def _state(review_round: int = 0) -> dict:
    return {
        "messages": [HumanMessage(content="总结这篇论文")],
        "selected_papers": [_paper()],
        "draft_review": "该论文提出一种方法 [1:p3]。",
        "summary": "",
        "review_round": review_round,
    }


def _patch_reviewer(monkeypatch, payload: dict | None):
    monkeypatch.setenv("REVIEWER_LLM_ENABLED", "true")
    monkeypatch.setattr(synthesis_agent, "get_reviewer_llm", lambda **kwargs: _FakeReviewerLLM())
    calls = []

    async def fake_invoke(*args, **kwargs):
        calls.append((args, kwargs))
        if payload is None:
            return None
        return AIMessage(content=json.dumps(payload, ensure_ascii=False))

    async def fake_emit(*args, **kwargs):
        return None

    monkeypatch.setattr(synthesis_agent, "safe_llm_invoke", fake_invoke)
    monkeypatch.setattr(synthesis_agent, "emit_visible_text", fake_emit)
    return calls


def test_reviewer_rejects_with_structured_feedback_and_requests_one_revision(monkeypatch):
    calls = _patch_reviewer(monkeypatch, {
        "passed": False,
        "summary": "存在无证据指标。",
        "issues": [{
            "claim": "mAP 提升 10%",
            "citation": "[1:p3]",
            "verdict": "unsupported",
            "severity": "high",
            "reason": "证据未出现该数值",
            "suggested_fix": "删除数值",
        }],
        "revised_text": "",
    })

    result = asyncio.run(synthesis_agent.reviewer_node(_state(review_round=0)))

    assert result["review_status"] == "revise"
    assert result["review_round"] == 1
    assert "mAP 提升 10%" in result["review_feedback"]
    assert "messages" not in result
    assert len(calls) == 1


def test_reviewer_second_rejection_returns_safe_evidence_fallback(monkeypatch):
    calls = _patch_reviewer(monkeypatch, {
        "passed": False,
        "summary": "仍有问题。",
        "issues": [{
            "claim": "unsupported claim",
            "citation": "",
            "verdict": "unclear",
            "severity": "high",
            "reason": "no evidence",
            "suggested_fix": "remove",
        }],
        "revised_text": "",
    })

    result = asyncio.run(synthesis_agent.reviewer_node(_state(review_round=1)))

    assert result["review_status"] == "failed_safe"
    assert "证据审阅未通过" in result["messages"][0].content
    assert "The paper introduces a grounded method" in result["messages"][0].content
    assert len(calls) == 1


def test_reviewer_unavailable_keeps_draft_without_revision(monkeypatch):
    calls = _patch_reviewer(monkeypatch, None)
    state = _state()

    result = asyncio.run(synthesis_agent.reviewer_node(state))

    assert result["review_status"] == "review_unavailable"
    assert state["draft_review"] in result["messages"][0].content
    assert len(calls) == 1


def test_reviewer_passes_grounded_draft_without_rewriting(monkeypatch):
    calls = _patch_reviewer(monkeypatch, {
        "passed": True,
        "summary": "全部声明有证据。",
        "issues": [],
        "revised_text": "",
    })
    state = _state()

    result = asyncio.run(synthesis_agent.reviewer_node(state))

    assert result["review_status"] == "passed"
    assert state["draft_review"] in result["messages"][0].content
    assert len(calls) == 1
    assert calls[0][1]["max_retries"] == 1
    assert "只审查来源归属和语义幻觉" in calls[0][0][1]


def test_review_packet_caps_evidence_spans_per_paper():
    spans = [
        EvidenceSpan(
            source="Paper A.pdf",
            page_start=index + 1,
            section="Method",
            chunk_id=f"chunk-{index}",
            quote=f"Grounded evidence {index}.",
            confidence=0.9,
        )
        for index in range(7)
    ]
    paper = _paper().model_copy(update={"evidence_spans": spans})

    packet = synthesis_agent._build_review_packet("总结论文", "初稿", [paper])

    assert packet.count("chunk=") == 5
    assert "LightRAG 关系证据" not in packet
