# 基于《AI Agent 开发指南》的项目补强清单

对照基线为李沐、李博杰《AI Agent 开发指南》主仓库及其 Context、Tool、异步系统、评测、持续演进和多 Agent 章节：

- 总览：https://github.com/bojieli/ai-agent-book
- Context 与 Skill：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter2.md
- Tool 与 MCP：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter4.md
- 异步与事件驱动：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter6.md
- 评测：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter7.md
- 持续演进：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter9.md
- 多 Agent：https://github.com/bojieli/ai-agent-book/blob/main/book/chapter10.md

## 已经具备的底座

- Harness 已成形：LangGraph 固定控制流、显式 `AgentState`、最大步数、重复检索停止、候选证据到确认证据的升级门。
- Context 已分层：60%/78%/90% token 水位、协议安全裁剪、会话摘要、用户画像、SQLite checkpoint；Skill 正文仅在 Synthesizer 运行时加载。
- Memory/RAG 已超出普通向量问答：LightRAG 图谱、结构化论文身份、页级 `EvidenceSpan`、本地与联网证据隔离。
- Tool 边界清晰：模型只看到原子 schema，业务执行在节点中；请求级 SSE 已包含 trace、node、TTFT 和节点耗时。
- 已有 30 个用户回合的端到端数据集；22 题带人工 source/page/anchor 和 claim 金标，评测专用 retrieval 事件可核对最终状态实际 EvidenceSpan。

## 2026-08-20 已完成的闭环

- `EvidenceGateResult` 已在候选证据升级前检查精确标题、来源数量、必要字段和页级证据覆盖，并返回具体缺口。
- Reviewer 已使用只读 `ReviewPacket` 和结构化 `ReviewResult`；不通过时最多返修一次，再失败降级为安全证据摘要，未审初稿不对外流式展示。
- PDF 解析与 LightRAG 建图已迁移为 SQLite 持久化后台任务，支持状态查询、中断恢复、失败原因、取消和显式重试；聊天只读 completed 文档。
- `user_profile` 已作为低优先级偏好数据注入 Assistant/Synthesizer，并防止画像内容改变路由、工具和证据边界。

## 2026-08-21 已完成的工程闭环

- 非 token 运行轨迹已脱敏持久化到独立 SQLite，可按用户/session 列表查询并按 trace 查看；记录节点、模型、Tool、检索、证据门、Reviewer、耗时、token usage 和错误，不保存 prompt 与正文。
- 主 Qwen、Kimi Reviewer、本地 vLLM 已拆分并发槽，按 429/超时/5xx 分类重试并支持 `Retry-After`、指数退避和角色级熔断；Reviewer 禁止静默回退主模型。
- Assistant 上下文窗口按远程 256K 配置放宽到 75%，摘要阈值后移到 78%；Synthesizer/Reviewer 增加证据和输出预算，并把实际上下文字符数写入 trace。
- Tool 参数由严格 Pydantic schema 校验：禁止多 Tool、额外字段、非法年份、超长或带 AND/OR 的联网关键词；模型只允许一次纠正，节点和 Graph 再做防御性校验。

## 补强项

| 优先级 | 缺口 | 当前证据 | 建议完成标准 |
|:---:|:---|:---|:---|
| P1 | 重复运行与统计阈值 | 已完成 retrieval 事件、30 回合数据集、Source/Evidence/Citation/Claim 指标和动态报告；当前仍是单次端到端运行 | 服务器资源允许后对关键题重复 3 次，报告 Pass^k、均值、方差和 P95，并为核心指标设置版本回归阈值 |
| P0（公网）/P1（自用） | 身份认证与资产隔离 | 会话按可伪造的 `X-User-ID` 隔离，CORS 为 `*`，上传 PDF、LightRAG workspace 和 `/pdfs` 下载链路仍共享 | 增加真实 token/session 认证、上传大小/MIME/文件名限制、per-user/project workspace；收紧 CORS，日志和下载接口做归属校验，密钥只驻留服务端 |
| P1 | Tool ACI 与统一注册 | 现有 3 个 Tool 适合当前规模，但 Assistant 仍硬编码 `bind_tools` 和路由名 | 当工具继续增长时引入 typed registry：目标语义、输入输出 schema、权限、超时、重试、幂等性；再按需适配 MCP Tools/Resources，当前无需先做“工具市场” |
| P1 | 检索置信度与停止依据可解释 | 已有自适应 global/local/naive/mix 和重复停止，但“证据足够”主要仍由 Assistant 判断 | 计算标题匹配、来源覆盖、摘要/页级证据覆盖和 rerank margin；把 score 与缺口写入状态，代码据此决定总结、转联网或明确拒答 |
| P1 | LightRAG 索引质量门 | manifest 能避免建图失败假成功，但没有实体/关系来源率、孤立节点率、chunk 覆盖率和证据回链抽检 | 新索引先写 staging；抽检每篇 chunk 覆盖、实体/关系来源、孤立节点和随机 citation 回链，达到阈值后原子切换为 active 版本 |
| P1 | 会话并发、幂等与总 deadline | 同一 session 没有请求锁和 request id；SSE 重试可能重复检索/写历史，同一 LangGraph thread 的并发请求可能相互覆盖 | 增加 session 级互斥、幂等键和整任务 deadline；SQLite 开启 WAL/busy timeout 或改异步访问；同一请求重放只能得到同一个已提交结果 |
| P2 | 显式 Agent handoff 契约 | 中心化图共享 State，当前节点少且可控；但没有独立 handoff packet、预算和访问记录 | 只有未来并行/外置 Agent 时再增加 `goal/constraints/accepted_facts/artifact_refs/remaining_budget/visited_agents`，并校验环路与预算；现在不必为概念而拆节点 |
| P2 | 生产反馈驱动演进 | 有静态测试集和运行日志，但失败案例不会自动进入可审核语料，prompt/skill 也没有版本效果对比 | 将低分、超时、人工差评 trace 脱敏入 failure corpus；人工标注后再加入回归集；记录 prompt/skill/index 版本并做离线 A/B，禁止运行时自修改 |
| P2 | 可治理长期记忆 | 用户画像已按 user 隔离且可删，但内容主要由正则抽取并追加，缺少来源、置信度、时效和冲突覆盖关系 | 把偏好改为 Memory Card：`source_turn/time/confidence/scope/supersedes`；提供查看、纠正、过期和删除接口，低置信信息不得自动进入全局画像 |
| P2 | 多模态证据回链 | 图片和表格可以转文本进入流程，但最终声明还不能稳定回到图片区域、表格单元格和论文页码 | 扩展 Figure/Table `EvidenceSpan`，保存页码、bbox、caption/row/column，并让 Reviewer 校验视觉证据引用 |
| P2 | 文档与死代码治理 | 架构文档仍保留已停用的 FAISS/BM25/HyDE 叙述，`query_rewriter.py` 当前也没有调用方 | 将旧实现移入明确的历史章节或删除；为无调用模块增加去留说明，保证简历、面试文档、代码和部署行为一致 |
| P2 | 多进程存储扩展 | SQLite、NanoVectorDB/JSON、NetworkX 适合单机单 worker，多个 Uvicorn worker 共享写入风险高 | 真正扩容时迁移共享 checkpoint/KV/向量/图存储并增加并发写测试；作品集阶段保持单 worker，避免过早引入分布式复杂度 |
| P2 | 规模上限与覆盖披露 | 最终证据最多取前 20 篇、普通本地检索最多选有限来源，当前没有 corpus-selection 解释和覆盖报告 | 增加可解释的论文选择阶段，返回总候选数、纳入/排除原因和覆盖范围；截断必须向用户披露，不能静默当作“全库” |
| P2 | 可复现部署 | 增量依赖文件不能完整锁定环境，也没有 CI、readiness/liveness 和模型/索引启动自检 | 提供完整 lock、最小 CPU 单测 CI、`/health/live` 与 `/health/ready`；启动时校验模型端点、Reviewer 配置、索引版本和数据库写权限 |

## 推荐实施顺序

1. 下一步优先补后台索引质量门、会话并发/幂等和完整部署依赖。
2. 服务器资源允许后再做评测重复运行、方差统计与回归阈值。
3. 最后按真实部署范围补认证和资产隔离；只有工具或 Agent 数量明显增加时再做 MCP 与 handoff 协议。

暂不建议投入训练/RL、自主改写 Prompt、通用 Computer Use、动态工具市场或去中心化 Agent 群。这些方向与当前“可追踪的学术研究助手”主线关系弱，投入大，也会稀释面试时最有价值的证据治理、检索质量和工程闭环。
