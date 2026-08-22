# models.py
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceSpan(BaseModel):
    """能够回到真实来源的最小证据单元。"""

    source: str = Field(default="", description="本地 PDF 文件名或网络来源 URL")
    page_start: int | None = Field(default=None, description="PDF 起始页码")
    page_end: int | None = Field(default=None, description="PDF 结束页码")
    section: str = Field(default="未知章节", description="论文中的章节")
    chunk_id: str = Field(default="", description="LightRAG 原文 chunk ID")
    quote: str = Field(default="", description="支持结论的原文片段")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="证据匹配置信度")

class PaperSummary(BaseModel):
    """单篇论文的核心信息提取 - 工业级防爆版"""
    title: str = Field(default="未知标题", description="论文标题")
    authors: str = Field(default="未知作者", description="作者信息")
    year: str = Field(default="未知年份", description="发表年份")
    source: str = Field(default="本地文档", description="来源")
    core_method: str = Field(default="未提取到核心方法，请查阅原文", description="核心研究方法")
    key_findings: str = Field(default="未提取到关键结论，请查阅原文", description="关键结论")
    doi: str = Field(default="未知", description="数字对象标识符")
    venue: str = Field(default="未知", description="期刊或会议")
    reference_index: int | None = Field(default=None, description="候选文献列表中的稳定编号")
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list, description="页级或摘要级证据")

    @field_validator('title', 'authors', 'year', 'source', 'core_method', 'key_findings', 'doi', 'venue', mode='before')
    @classmethod
    def force_to_string(cls, v) -> str:
        if v is None: return "未提及"
        if isinstance(v, list): return ", ".join([str(i) for i in v])
        if isinstance(v, dict): return str(next(iter(v.values()))) if v else "格式错误"
        return str(v).strip()


class WebPaperEnrichment(BaseModel):
    """联网摘要的最小 LLM 提炼结果；权威元数据始终由学术 API 提供。"""

    core_method: str = Field(min_length=4, max_length=600, description="核心研究方法的中文总结")
    key_findings: str = Field(min_length=4, max_length=600, description="关键结论的中文总结")


class EvidenceGateResult(BaseModel):
    passed: bool = False
    mode: str = ""
    selected_papers: list[PaperSummary] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class ReviewIssue(BaseModel):
    claim: str = ""
    citation: str = ""
    verdict: Literal["supported", "unsupported", "unclear", "citation_error"] = "unclear"
    severity: Literal["low", "medium", "high"] = "medium"
    reason: str = ""
    suggested_fix: str = ""


class ReviewResult(BaseModel):
    passed: bool = False
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    revised_text: str = ""
