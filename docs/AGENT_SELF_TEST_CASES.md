# Research Agent 评测说明

当前本地知识库只包含 `test_pdfs/Attention is All You Need.pdf` 与 `test_pdfs/Mamba yolo.pdf`。旧版 MobileMamba、BiFormer、YOLO-AES 等测试题已经删除，唯一机器可读测试集为 `docs/AGENT_EVAL_DATASET.jsonl`。

## 测试集

测试集固定为 30 个用户回合，其中 16 个是同一会话中的后续追问。覆盖范围如下：

| 范围 | 题数 | 主要验证内容 |
|:---|---:|:---|
| 普通直答与本地目录 | 2 | 不误用工具、只列出真实存在的两篇论文 |
| Attention 单篇精读 | 7 | 架构、并行性、真实 BLEU、复杂度、作者、位置编码、任务边界 |
| Mamba YOLO 单篇精读 | 7 | SSM/ODSSBlock/RG Block、指标、预训练、作者、消融和表格数值 |
| 跨论文对比与压缩追问 | 4 | 同表对比、综合归纳、100 字约束、错误演进关系纠正 |
| 全库综述 | 1 | 两篇论文必须放进同一条技术演进线 |
| 抗幻觉与证据不足 | 4 | 不存在论文、未报告指标、错误作者和错误架构前提 |
| 联网检索 | 2 | 主题检索与精确标题检索，禁止本地结果冒充外网来源 |
| 会话记忆 | 3 | 偏好写入、三句话约束和带证据的跨论文追问 |

每题都定义可接受工具轨迹、最大工具调用次数、关键事实、禁止声明、引用和格式约束。本地证据题还标注 `gold_sources`、`gold_evidence(source/pages/anchors)` 与 `gold_claims`。多轮题允许复用证据或最多补做一次本地检索，不惩罚缓存命中。原文核对已修正旧数据错误：《Attention Is All You Need》的 WMT 2014 英法结果是 **41.0 BLEU**，不是 41.8。

## 指标口径

本项目参考 RAGAS 的 Faithfulness、Context Precision/Recall、Response Relevancy、Factual Correctness、Tool Call Accuracy/F1、Agent Goal Accuracy，以及 LangSmith 的 correctness/relevance/groundedness/retrieval relevance。工程性能参考 vLLM 的 TTFT、端到端延迟和完成请求数。

评测请求发送 `X-Eval-Mode: 1`，服务只在该请求中增加 `retrieval` SSE 事件，返回最终状态实际携带的论文、引用编号和 `EvidenceSpan`；普通前端协议不变。评测器据此确定性计算 Source Recall、Evidence Recall/Precision、Citation Correctness 与 Claim Support。`Claim Support` 要求答案声明、人工金标证据、实际 EvidenceSpan 和正确页码同时成立；它比旧 `Grounding Proxy` 严格，但仍不冒充开放域语义裁判。

官方指标资料：

- RAGAS Faithfulness：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- RAGAS Context Precision：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
- RAGAS Context Recall：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
- RAGAS Agent/Tool Use：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/
- LangSmith RAG Evaluation：https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- vLLM Metrics：https://docs.vllm.ai/en/latest/design/metrics/

## 运行方式

先重启 8080 服务，再从项目根目录串行运行：

```bash
python scripts/evaluate_agent.py --base-url http://127.0.0.1:8080 --timeout 360
```

串行是为了避免同一个 vLLM/GPU 上的并发争用污染延迟。脚本为每个场景创建独立会话，通过 `X-User-ID` 隔离数据，解析 `log/token/retrieval/final`，每完成一题增量写入原始结果，结束后生成报告。报告“关键发现”完全根据本轮结果生成，不再含旧基线硬编码文案。默认会删除服务端评测会话。

只验证前两题可用：

```bash
python scripts/evaluate_agent.py --limit 2
```

修改评分规则后可基于已有事件流离线重算，不再次请求模型：

```bash
python scripts/evaluate_agent.py --rescore-existing AGENT_EVAL_RESULTS.json
```

综合分权重为 Goal Accuracy 25%、Claim Support 20%、Evidence Recall 15%、Evidence Precision 10%、Citation Correctness 10%、Tool F1 10%、Multi-turn Goal Accuracy 5%、Completion Rate 5%。联网动态结果没有固定内容金标，只评路由、来源和格式。

## 修复后必须回归的断言

本轮代码已经增加确定性单测；服务重启后的 8080 实测还要重点确认以下行为：

| 回归点 | 输入示例 | 通过标准 |
|:---|:---|:---|
| 证据追问 | `它为什么更适合并行训练？只基于刚才论文回答` | 复用目标论文并给出 `[n:pN]`，不由主脑裸答 |
| 指标保护 | `英德和英法 BLEU 分别是多少` | 只能复制页级证据中的 28.4/41.8，不出现 41.0 或“推测页码” |
| 元数据精确匹配 | `这篇论文作者和年份是什么` | Attention 显示 2017，近似标题的 2025 结果不得覆盖 |
| 单篇深度延迟 | 深度模式精读 Mamba YOLO | LightRAG 使用 `naive`，不因深度模式强制走 `mix` |
| 联网终止 | 联网搜索无结果 | 返回明确零结果或真实结果，不能只承诺“下一轮检索” |
| 会话格式记忆 | 先记住`最多三句话`再追问 | 最终 SSE `final` 和数据库消息均不超过三句话 |
| 本地清单快路径 | `我本地有哪些文献` | 读取 manifest/full_docs，不出现 LightRAG 图查询和证据抽取日志 |
| 证据升级门 | 点名不存在标题，或只有一篇来源却要求对比 | 不进入 Synthesizer；明确列出标题、来源、字段或 EvidenceSpan 缺口 |
| Reviewer 驳回与返修 | 构造包含无证据指标的综述 | 未审初稿不产生 token；最多返修一次，第二次失败输出“证据审阅未通过”的安全摘要 |
| 后台建图恢复 | 上传新 PDF 后立即提问，再重启服务 | 上传接口立即返回排队日志；查询只显示 queued/parsing/indexing；重启后任务恢复，只有 completed 后才可检索 |
| 后台任务接口 | 查询、取消或重试索引任务 | `GET /api/index-jobs` 可见状态；queued 可取消；failed/cancelled 可 retry；失败原因保留 |
| 跨会话长期画像 | A 会话记住“以后最多三句话”，B 会话追问 | Assistant/Synthesizer 遵守三句话；画像不得改变工具路由或绕过证据门；删除画像后不再生效 |

离线回归命令为 `python -m pytest -q`。端到端测试必须先重启 API 进程再运行 `python scripts/evaluate_agent.py --base-url http://127.0.0.1:8080 --timeout 360`；已有图谱无需重建，新上传 PDF 由后台任务增量入库。
