import json
from pathlib import Path

from scripts.evaluate_agent import load_dataset, score_case


ROOT = Path(__file__).resolve().parents[1]


def _run(answer: str, tools=None, logs: str = "") -> dict:
    return {
        "answer": answer,
        "tools": tools or [],
        "logs": logs,
        "search_keywords": [],
        "completed": True,
        "retrieval_event_observed": False,
        "retrieved_papers": [],
        "retrieved_evidence": [],
    }


def test_eval_dataset_has_thirty_unique_cases_and_real_evidence_labels():
    cases = load_dataset(ROOT / "docs" / "AGENT_EVAL_DATASET.jsonl")
    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == 30
    assert sum(bool(case.get("gold_evidence")) for case in cases) >= 20


def test_followup_accepts_cache_reuse_or_one_retrieval():
    case = next(
        case for case in load_dataset(ROOT / "docs" / "AGENT_EVAL_DATASET.jsonl")
        if case["id"] == "EVAL_004"
    )
    cache_score = score_case(case, _run("避免递归依赖，可同时计算所有位置并行训练。[1:p1]"))
    retrieval_score = score_case(
        case,
        _run(
            "避免递归依赖，可同时计算所有位置并行训练。[1:p1]",
            tools=["trigger_local_retrieval"],
        ),
    )
    assert cache_score["tool_f1"] == 1.0
    assert retrieval_score["tool_f1"] == 1.0


def test_source_evidence_claim_and_citation_are_scored_together():
    case = {
        "checks": [{"name": "fact", "any": ["28[.]4"]}],
        "forbidden": [],
        "expected_tools": [],
        "citation_required": True,
        "gold_sources": ["Attention is All You Need.pdf"],
        "gold_evidence": [{
            "id": "bleu",
            "source": "Attention is All You Need.pdf",
            "pages": [8],
            "anchors": ["28[.]4"],
        }],
        "gold_claims": [{
            "name": "bleu_claim",
            "answer": {"any": ["28[.]4"]},
            "evidence_ids": ["bleu"],
        }],
        "format": {},
    }
    run = _run("28.4 BLEU [1:p8]")
    run.update({
        "retrieval_event_observed": True,
        "retrieved_papers": [{
            "reference_index": 1,
            "source": "Attention is All You Need.pdf",
            "evidence_spans": [],
        }],
        "retrieved_evidence": [{
            "reference_index": 1,
            "source": "Attention is All You Need.pdf",
            "page_start": 8,
            "page_end": 8,
            "quote": "Transformer big obtains 28.4 BLEU.",
        }],
    })
    score = score_case(case, run)
    assert score["source_recall"] == 1.0
    assert score["evidence_recall"] == 1.0
    assert score["evidence_precision"] == 1.0
    assert score["citation_correctness"] == 1.0
    assert score["claim_support"] == 1.0


def test_wrong_page_citation_fails_claim_support_even_when_answer_fact_is_right():
    case = {
        "checks": [{"name": "fact", "any": ["28[.]4"]}],
        "forbidden": [],
        "expected_tools": [],
        "citation_required": True,
        "gold_evidence": [{
            "id": "bleu", "source": "paper.pdf", "pages": [8], "anchors": ["28[.]4"],
        }],
        "gold_claims": [{
            "name": "bleu", "answer": {"any": ["28[.]4"]}, "evidence_ids": ["bleu"],
        }],
        "format": {},
    }
    run = _run("28.4 BLEU [1:p3]")
    run["retrieved_papers"] = [{"reference_index": 1, "source": "paper.pdf"}]
    run["retrieved_evidence"] = [{
        "reference_index": 1, "source": "paper.pdf", "page_start": 8, "page_end": 8,
        "quote": "28.4 BLEU",
    }]
    score = score_case(case, run)
    assert score["factual_keypoint_coverage"] == 1.0
    assert score["evidence_recall"] == 1.0
    assert score["citation_correctness"] == 0.0
    assert score["claim_support"] == 0.0


def test_attention_bleu_gold_is_41_point_0_not_41_point_8():
    case = next(
        case for case in load_dataset(ROOT / "docs" / "AGENT_EVAL_DATASET.jsonl")
        if case["id"] == "EVAL_005"
    )
    serialized = json.dumps(case, ensure_ascii=False)
    assert "41[.]0" in serialized
    assert "41[.]8" in serialized
    assert "41[.]8" in case["forbidden"]


def test_dataset_lines_are_valid_jsonl():
    path = ROOT / "docs" / "AGENT_EVAL_DATASET.jsonl"
    assert all(json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines())
