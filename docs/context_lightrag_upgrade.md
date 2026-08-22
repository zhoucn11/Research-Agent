# 上下文记忆与 LightRAG 升级说明

这次升级同时覆盖上下文、图谱检索、页级证据、持久执行和并发日志。现在上下文由 token 水位驱动，论文库由 LightRAG 维护文档、向量、实体关系图和处理状态；回答可以回到真实 `source/page/chunk`，LangGraph 任务也能在服务重启后恢复状态。

## 上下文治理

`research_agent/memory/agent_memory.py` 采用三层惰性策略：预计占用达到窗口 60% 时，只裁剪旧工具输出并保留头尾；达到 78% 时，用主 Qwen API 把旧消息合并进结构化会话摘要，最近 8 条消息保持原样；达到 90% 时再做硬折叠。切分位置会主动退回到 `assistant.tool_calls` 之前，保证后续 `ToolMessage` 不会变成孤儿消息。摘要调用失败时，会用规则提取目标、文件、错误和待办，不让记忆模块拖垮主流程。

默认配置集中维护在 `research_agent/core/app_config.py`，以下是当前基线；部署环境只有在调试或容量调整时才需要用同名环境变量覆盖：

```bash
MAIN_API_CONTEXT_WINDOW=262144
CONTEXT_MAX_TOKENS=196608
CONTEXT_RESERVED_TOKENS=16000
MEMORY_CONTENT_LIMIT=2000
MEMORY_SUMMARY_LIMIT=12000
MEMORY_SUMMARY_INPUT_LIMIT=120000
```

`CONTEXT_MAX_TOKENS` 默认取远程主 API 窗口的 75%，给系统提示、Tool schema、证据和输出留出安全空间；若供应商实际窗口不是 256K，必须同步修改 `MAIN_API_CONTEXT_WINDOW`。SQLite 读取返回最近 N 条消息；服务重启后只在 LangGraph checkpoint 为空时恢复最近对话，避免重复注入。

## LightRAG 数据流

入口仍然是 `rag_map_node`，DeepDoc PDF 解析、标题/年份指纹和 `PaperSummary` 结构化输出继续保留。变化发生在中间检索层：

```text
PDF -> DeepDoc 全文 -> 论文元数据头 -> LightRAG 实体/关系抽取与切块
用户问题 -> 确定性查询策略 -> LightRAG global/local/naive/mix -> 批量证据抽取 -> PaperSummary[] + EvidenceSpan[]
```

每篇 PDF 以文件名作为本地文献库的稳定身份，`research_agent_manifest.json` 保存 LightRAG doc ID、标题、年份和建图时的文件信息。只有新文件名才增量建图，文件名消失后清理派生索引；重新上传代码或 PDF 导致的 mtime 变化不会触发重建。该策略适合当前个人论文库，但同名 PDF 覆盖不会自动更新图谱：确需替换内容时，应先删除旧 PDF 并执行一次同步清理，再放入新文件，或直接使用新文件名。原 `local_faiss_db` 不再读取，但不会自动删除，确认新图谱稳定后可手动归档。

安装：

```bash
pip install -r requirements-lightrag.txt
```

关键环境变量：

```bash
# 主 Qwen API：Assistant 决策、上下文压缩、Query Rewrite、联网摘要和 Synthesizer 共用
OPENAI_BASE_URL=https://your-main-api.example/v1
OPENAI_API_KEY=your-main-api-key
OPENAI_MODEL=your-main-model

# Reviewer Agent：独立使用 Kimi K2.6
REVIEWER_BASE_URL=https://api.moonshot.cn/v1
REVIEWER_API_KEY=your-moonshot-api-key
REVIEWER_MODEL=kimi-k2.6
REVIEWER_LLM_ENABLED=true
# Reviewer 显式开启 Kimi K2.6 思考，并按官方约束发送 temperature=1.0
# 两类长文本任务分别控制超时和最大输出
SYNTHESIS_TIMEOUT=300
SYNTHESIS_MAX_OUTPUT_TOKENS=4096
REVIEWER_TIMEOUT=300
REVIEWER_MAX_OUTPUT_TOKENS=8192
REVIEW_PACKET_MAX_CHARS=180000
RAG_EVIDENCE_SPANS_PER_SOURCE=3

# provider 级并发、重试与熔断
MAIN_API_MAX_CONCURRENCY=2
REVIEWER_API_MAX_CONCURRENCY=1
LOCAL_LLM_MAX_CONCURRENCY=1
ASSISTANT_LLM_MAX_RETRIES=3
LLM_RETRY_BASE_SECONDS=1
LLM_CIRCUIT_FAILURE_THRESHOLD=3
LLM_CIRCUIT_COOLDOWN_SECONDS=60

LIGHTRAG_INDEX_VERSION=paper_graph_v1
LIGHTRAG_WORKING_DIR=/path/to/lightrag_storage/paper_graph_v1

LIGHTRAG_LLM_MODEL=qwen3
LIGHTRAG_LLM_BASE_URL=http://127.0.0.1:6006/v1
LIGHTRAG_LLM_API_KEY=sk-local
LIGHTRAG_LLM_MAX_ASYNC=1
# Qwen3 默认关闭 thinking，亦可自行覆盖：
# LIGHTRAG_LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}

LIGHTRAG_EMBEDDING_MODEL=bge-large-zh-v1.5
LIGHTRAG_EMBEDDING_DIM=1024
LIGHTRAG_EMBEDDING_BATCH_SIZE=1
LIGHTRAG_EMBEDDING_MAX_ASYNC=1
LIGHTRAG_MAX_PARALLEL_INSERT=1

# 默认 auto；显式设置 mix/local/global 等会固定使用该模式
LIGHTRAG_QUERY_MODE=auto
LIGHTRAG_TOP_K=30
LIGHTRAG_CHUNK_SIZE=600
LIGHTRAG_CHUNK_OVERLAP=80
LIGHTRAG_CHUNK_TOP_K=6
LIGHTRAG_MAX_ENTITY_TOKENS=1400
LIGHTRAG_MAX_RELATION_TOKENS=1600
LIGHTRAG_MAX_TOTAL_TOKENS=4500
RAG_EXTRACTION_CONTEXT_CHARS=24000
# 普通总结默认不联网补元数据；只有显式询问作者/年份/DOI 时才核验
RAG_VERIFY_SUMMARY_METADATA_ONLINE=false

# Semantic Scholar 所有端点合计上限为 1 RPS；默认 1.1 秒一次，给边界抖动留余量
S2_MIN_REQUEST_INTERVAL_SECONDS=1.1

# 仅深度 mix 查询可选启用本机 CrossEncoder；默认 CPU，论文内容不会外传
LIGHTRAG_LOCAL_RERANK_ENABLED=false
RERANKER_DEVICE=cpu
RERANKER_MODEL_PATH=/root/autodl-tmp/financial-report-rag/models/bge-reranker-v2-m3

# LangGraph 持久 checkpoint
AGENT_CHECKPOINT_BACKEND=sqlite
AGENT_CHECKPOINT_DB=/path/to/Research-Agent/agent_checkpoints.sqlite3

# PDF/LightRAG 后台任务状态库
AGENT_INDEX_JOB_DB=/path/to/Research-Agent/index_jobs.sqlite3

# 脱敏执行轨迹；不保存 prompt、最终正文和 visible token
AGENT_TRACE_DB=/path/to/Research-Agent/agent_traces.sqlite3

# 可选启动预热
AGENT_PREWARM_LIGHTRAG=false

# 单文件上传默认上限 25 MiB
AGENT_UPLOAD_MAX_BYTES=26214400
```

自适应策略不额外调用模型：全库归纳走 `global`，单篇事实问答（包括深度模式）优先走 `naive`，实体关系问题走 `local`，跨论文比较和开放主题走 `mix`。深度单篇问答仍会提高 chunk 与 token 预算，但不再为了“深度”强制承担完整图关系检索和本地 rerank。前端“自动/快速/深度”只改变召回和生成预算，不降低证据标准；自动模式遇到综述、比较、全部论文等请求时进入深度预算。

## 页级证据、持久执行与安全边界

DeepDoc 入库时保留 `[page:N]`，LightRAG 的 `kv_store_text_chunks.json` 同时保存 `file_path`、chunk ID 和正文。查询结束后系统从这些真实 chunk 中选择与问题相关的片段，写入 `PaperSummary.evidence_spans`，字段包括 `source`、`page_start/page_end`、`section`、`chunk_id`、`quote` 和 `confidence`。Synthesizer 使用 `[文献号:p页码]`，Reviewer 的 Citation Guard 检查缺失标记并附加“证据定位”，本地链接直接打开 `/pdfs/<文件>#page=N`。网络摘要没有页码时使用 `[文献号:摘要]`，不得伪造页码。当前索引已经保存页码和 chunk，因此本次代码升级不要求重新建图。

FastAPI 启动时默认使用 `AsyncSqliteSaver`，checkpoint 写入独立的 `agent_checkpoints.sqlite3`；原 `agent_memory.sqlite3` 继续负责会话、消息、摘要和画像。删除会话时同步删除对应 LangGraph thread。若缺少 `langgraph-checkpoint-sqlite`，服务会告警并回退到内存；执行 `pip install -r requirements-lightrag.txt` 后重启即可启用持久模式。

PDF 上传后的 DeepDoc 解析和 LightRAG 建图已移出聊天请求链路。`index_jobs.sqlite3` 保存 `queued/parsing/indexing/completed/failed/cancelled` 状态、进度、失败原因和尝试次数；服务重启会把中断任务恢复为 `queued`，失败或取消任务通过 `POST /api/index-jobs/{job_id}/retry` 显式重试。`GET /api/index-jobs` 和 `GET /api/index-jobs/{job_id}` 查询进度，`DELETE` 只允许取消尚未执行的任务。聊天只使用 manifest 中真正完成且物理文件仍存在的论文，上传后未完成时返回明确状态，不再一边聊天一边建图，也不写假成功。

候选论文升级现在由 `EvidenceGateResult` 确定性控制，而不是只看“总结/对比”意图：点名标题必须精确命中；对比至少覆盖两篇不同论文和两个来源；作者、年份、DOI 等按问题要求检查；本地内容性结论必须带真实 `EvidenceSpan`。失败会返回逐项缺口，模型输出 `[APPROVE_SYNTHESIS]` 也不能绕过。Reviewer 接收用户问题、初稿、论文元数据、全部页级 span 和图谱证据组成的只读包，输出结构化 `ReviewResult`；存在 unsupported、unclear 或 citation_error 时驳回，最多隐藏返修一次，仍不通过就降级为可回链证据摘要。前端只流式展示最终通过或安全降级的文本，不提前泄露未审初稿。

“讲一下第一篇”“说一下第二篇”“展开讲”等口语化单篇跟进由代码直接归一为内容解读意图：先按上一轮稳定编号锁定论文，再检查核心字段和真实 `EvidenceSpan`，通过后进入 Synthesizer/Reviewer；不再把检索后的二次决策交给 Assistant 模型。`retrieval_result` 轨迹同时记录 `evidence_span_count`，用于区分“意图未命中”和“页级证据确实缺失”。

长期画像已经闭环到 Assistant 和 Synthesizer，但按“数据而非指令”处理：只适配语言、篇幅、排版和研究兴趣，不能影响检索路由、工具调用、证据标准、引用或用户本轮显式要求；会话级约定仍只作用于当前 session。

SSE 不再全局替换 `sys.stdout/sys.stderr`，而是通过 `ContextVar` 给每个请求绑定独立 `trace_id/session_id/node`，保留原 `log/token/final` 协议。非 token 事件同步脱敏写入 `agent_traces.sqlite3`，包括节点耗时、模型角色、重试/熔断、Tool、检索、证据门和 Reviewer 裁决；prompt、最终正文和 `visible_token` 不入库。`GET /api/traces` 可按 session 查询，`GET /api/traces/{trace_id}` 返回单次轨迹，均按 `X-User-ID` 隔离。该请求头仍只是命名空间，不是完整登录鉴权；当前本地 PDF/LightRAG 仍是共享研究工作区。

联网来源声明同样由代码控制：只有存在与 Assistant tool call 配对的真实 `ToolMessage`，系统才允许确认执行过联网检索；若当前轮复用上一轮网络候选，正文和活动轨迹都会明确标注“本轮未重新联网”。同一问题的轻微错别字重试通过字符相似度保留候选；非跟进新任务会从路由 prompt 中同时移除旧 `assistant.tool_calls` 与对应 `ToolMessage`，防止历史工具结果冒充本轮检索。“查找相关论文”在没有候选时必须实际生成 `trigger_web_search`，直接列举会被确定性拦截。

模型职责按风险拆分：Assistant 路由/工具决策、上下文压缩、联网摘要和 Synthesizer 使用主 Qwen API；Reviewer 独占 Kimi，缺配置时安全降级为可回链证据摘要，不允许静默换回主模型；PDF 指纹、LightRAG 建图和本地论文证据抽取仍使用本地 vLLM，避免默认外传本地论文正文。前置文件路由已经改为确定性文件名/唯一特征词匹配，不再额外调用小模型。

联网逐篇中文提炼继续复用主 Qwen API，但使用独立预算：`WEB_PAPER_SUMMARY_MAX_OUTPUT_TOKENS=2400`、`WEB_PAPER_SUMMARY_THINKING_BUDGET=512`。结构化响应只生成 `core_method/key_findings`，标题、作者、年份、DOI、来源与摘要证据仍以学术 API 字段为准。HTTP 200 但 `finish_reason=length` 的截断不会用相同预算盲目重试，也不会累计主 API 熔断；该篇会确定性回退到 Semantic Scholar/OpenAlex 原摘要。熔断只统计可重试的网络、限流和服务端故障。

本地 vLLM 的具体权重仍由启动命令的 `--model` 决定，示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/financial-report-rag/models/qwen/Qwen3-8B \
  --served-model-name qwen3 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.6 \
  --dtype half \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 6006
```

本地 prefix cache 只服务建图等稳定批处理前缀，不缓存回答。主决策已走外部 API，因此不再执行 `AGENT_PREWARM_LOCAL_LLM` 路由提示预热。

embedding 模型或维度变化时，必须同时提升 `LIGHTRAG_INDEX_VERSION`，让系统使用新的存储目录重新建图，不能让不同维度的向量共用旧索引。LightRAG 建图对 LLM 能力要求明显高于普通 RAG；8B 模型可以做开发验证，但正式论文库建议用至少 32B、32K 上下文的非推理模型离线建图，查询阶段再按成本选择模型。

默认存储是 LightRAG 自带的 JSON/NanoVectorDB/NetworkX，适合单机作品和单进程 Uvicorn。若要开多个 API worker 或扩到大语料，应把 KV、向量和图存储迁移到 PostgreSQL/Neo4j 等共享后端，不能让多个进程同时写同一组本地 JSON 文件。

## 项目级 Skill

项目采用 `.agent/skills/<skill-name>/SKILL.md` 保存人工审核、可版本化的 Skill，不恢复旧的对话自生成 Skill。Registry 使用固定 allowlist，只有 Synthesizer 收到综述、总结、相关工作或跨论文对比意图时才加载写作规则；单篇作者/指标追问不加载，Assistant、RAG、Search 和 Reviewer 也不读取 Skill 正文，避免路由提示词膨胀和职责串线。

当前只保留一个 Skill：

- `literature-review-writing`：只消费已经批准的 `selected_papers`、`graph_evidence` 和页级证据，先构造跨论文比较轴，再按主题综合共同点、方法差异、边界和研究空白；禁止自行检索、补论文、改编号或猜测元数据。

原 `literature-search` 与 `citation-verifier` 已移除：精确标题停止、关键词去重、来源隔离和元数据核验都是安全边界，继续由确定性代码和现有 Tool 链路实现，不依赖模型是否正确命中 Skill。后续只有出现稳定、可复用且不属于安全边界的写作流程时才新增 Skill，避免把基础设施和原子工具包装成提示词。

## 延迟治理

代码已经去掉或压缩多项固定开销：记忆不再每 5 轮必调模型；Assistant 和上下文压缩迁移到主 API，并把压缩水位推迟到 78%；前置文件路由改为确定性匹配；LightRAG 使用一次批量抽取；查询模式和 token budget 按快速/深度任务动态选择；快速模式未询问作者、年份、DOI 时跳过联网元数据核验。远程 Qwen、Kimi 和本地 vLLM 分别限流、分类重试和熔断，避免一个 provider 拖垮全部模型任务。

后续正式评测统一放在功能稳定后完成：基于现有 40 条数据补充真实 source/page 标注，比较普通向量 RAG、固定 mix、自适应 LightRAG、自适应 + 本地 reranker，记录 source/chunk recall、引用准确率、路由准确率、TTFT 和端到端 p95。当前新增单测只验证确定性行为，不冒充真实模型质量评测。

仍可继续优化的关键路径是把深度模式的在线元数据核验移到后台补全，并给后台索引增加前端进度展示、任务级 deadline 和索引质量门。vLLM 继续开启 prefix caching 和 chunked prefill，并根据 `/metrics` 实测调整并发与显存，不盲目扩大上下文。

本轮完成标准是确定性测试通过，并在服务器重启后验证 checkpoint、三种研究模式、页级链接、并发 SSE 和真实 LightRAG 查询；正式质量指标按计划最后实施。

## 20 题评测后的可靠性修复（2026-08-12）

本轮把评测中暴露的错误改成了可测试的代码约束，而不是继续加长提示词。具体行为如下：

- “为什么、如何、指标、实验结果、作者、年份”等证据追问会进入 Synthesizer/Reviewer，不再由 Assistant 凭会话印象直接回答；简答缺少引用时会补真实 `[文献号:p页码]` 或 `[文献号:摘要]`。
- 指标数值必须存在于已附加的页级证据中。模型若生成证据中不存在的 BLEU、mAP、AP、百分比或延迟数值，Numeric Guard 会在落库前移除并标明证据不足，禁止推测页码。
- 在线元数据核验会请求多个候选并要求规范化标题完全一致；不能再把 `Tensor Product Attention Is All You Need` 的 2025 年元数据写到 `Attention Is All You Need`。普通深度问答不再自动联网补元数据，显式作者/年份问题才执行核验。
- “最多 N 句话”和“N 个字符以内”在 API 最终消息落库前强制执行，并同时读取本轮前后的会话摘要，避免记忆已写入但回答不遵守。
- 联网工具返回后若模型只说“下一轮继续检索”却没有发起工具调用，系统会改成明确的零结果说明；相同关键词仍禁止重复调用。
- 大型“核心情报表”只用于总结、综述、对比和全库清单；单点事实问答仅保留直接答案与证据定位，减少冗余输出和生成延迟。
- 本地索引同步以 PDF 文件名判断新增和删除，不再用 mtime/size 判断内容变化；代码重新上传后会直接复用已有图谱。
- “本地有哪些文献/论文清单”走目录快路径，直接读取 manifest 和 full_docs，不执行图查询或结构化证据抽取。其他查询的抽取 schema 只保留六个必要字段并限制摘要长度；若模型仍因输出截断而无法解析，则使用已入库首页和摘要兜底，不再把一次 JSON 截断变成空结果。
- Semantic Scholar 的所有调用在单进程内共用线程安全限速器，默认请求起始时间至少间隔 1.1 秒；429 重试仍遵循同一节拍，OpenAlex 兜底不占用该额度。多 Uvicorn worker 部署需要 Redis 等跨进程限速器，当前单进程部署无需额外组件。

这些调整只改变查询路由、运行时核验和输出收口，不修改 `lightrag_storage` 中的 chunk、向量或图结构，因此不需要重新构建图谱。离线回归结果为 `66 passed, 1 skipped`；8080 端到端指标必须在服务器加载新代码后重新运行，旧 `AGENT_EVAL_REPORT.md` 保留为修复前基线。

2026-08-21 的轨迹、模型容错、远程上下文和 Tool 校验升级同样不修改现有图谱；当前全量确定性回归为 `87 passed, 2 skipped`。跳过项来自本机缺少服务器侧 LangChain/vLLM 依赖，真实模型、联网 API、SSE 和 LightRAG 查询仍需在服务器重启后验收。
