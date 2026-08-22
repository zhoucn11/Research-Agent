# Engineering Rules

## 架构边界

1. `tools/agent_tools.py` 只声明模型可见的 Tool schema，不放业务实现。
2. `core/graph.py` 只负责节点和边的编排；具体检索、合成和状态判断放在对应模块。
3. `agents/` 负责一次节点执行，`retrieval/` 负责外部数据源和索引，`schemas/` 负责边界数据验证。
4. API 层负责协议、持久化和流式传输，不复制 Agent 的学术决策逻辑。
5. Skill 是经过审核的工作流说明，复用现有 Tool；记忆、SSE、数据库和模型客户端属于基础设施，不封装为 Skill。

## 状态与路由

- `candidate_papers` 是当前工具结果，默认覆盖；升级为 `selected_papers` 前必须通过确定性 `EvidenceGateResult`，检查精确标题、来源数量、必要字段和本地页级证据覆盖。
- 非跟进问题必须清理旧候选、旧选择和旧图谱证据，防止跨问题污染。
- 工具执行后返回 Assistant 形成 Observation，不直接跳到 Synthesizer。
- 只有 `selected_papers` 非空且出现 `[APPROVE_SYNTHESIS]` 才进入合成链路。
- Reviewer 必须消费压缩后的只读证据包并返回结构化裁决；只审查来源归属与语义幻觉。首审不通过最多触发一次定向返修和二次审阅，二审仍失败才输出安全证据摘要；接口不可用不得误判为内容驳回，也不得触发返修。
- 相同关键词去重之外，还要按精确论文目标去重；精确标题未命中后直接停止，不能用相关论文替代。
- 主题检索允许改变概念轴，但不能反复添加修饰词直到步骤上限。

## 学术证据

- 每条论文证据必须保留真实 `source`；本地来源对应 PDF，网络来源对应可访问记录或链接。
- 元数据按标题相似度核验。低相似度在线结果不能覆盖本地标题、作者或年份。
- OCR 作者纠错只接受高相似度、同顺序或更完整的权威作者表。
- 未知字段保持“未知”，不得把模型背景知识写成已检索证据。
- 文献编号属于候选列表身份；选中“第二篇”后仍使用 `[2]`，不能对子集重新从 `[1]` 编号。
- 用户要求“只给表格、只列标题作者、简要回答”时，由代码后处理保证格式，不能只依赖提示词。
- 本地关键结论优先绑定 `EvidenceSpan(source/page/chunk/quote)`；没有真实页码时不得伪造页码。
- PDF 和网页正文属于不可信数据，其中的命令、角色设定和工具调用要求不得进入控制流。

## LightRAG 与文件

- `research_agent_manifest.json` 只能在 LightRAG 文档状态真正完成后写入。
- PDF 解析和 LightRAG 写索引只能由持久化后台任务执行；聊天请求只查询 `completed` 文档。任务中断后恢复为 `queued`，失败必须保留原因并由显式重试重新入队。
- 本地论文以 PDF 文件名作为索引身份；mtime 变化不得触发重建。同名内容替换需先清理旧条目或改用新文件名。
- 查询优先走 LightRAG；点名论文查询超时后可以使用同一索引保存的原文首页和摘要兜底，但不能换成普通全库近似召回。
- 修改 embedding 模型、维度或不兼容切块语义时提升 `LIGHTRAG_INDEX_VERSION`，不要覆盖旧索引。
- 不把 `lightrag_storage/`、SQLite、上传 PDF 或用户文档当作可随意清理的测试产物。
- 测试不得上传、删除或重建真实论文库；使用临时目录和 fake store。

## 记忆与流式协议

- 压缩边界不得落在 `ToolMessage` 前，必须保留对应 tool call。
- 长期画像与当前会话约定分开存储；会话约定不能污染其他会话。
- 长期画像只能作为低优先级偏好数据注入 Assistant/Synthesizer，用于语言、篇幅、排版和研究兴趣；不得改变路由、工具、证据、引用或本轮明确要求。
- SSE 首先可发送日志，生成阶段逐 token 发送，最后发送唯一 `final`；不得重复追加整段正文。
- API 完成后写入 user/assistant 两条历史，并保证前端刷新后结果一致。
- FastAPI 使用持久 checkpoint；会话删除必须同步删除对应 LangGraph thread。
- 禁止通过全局替换 `sys.stdout/sys.stderr` 收集请求日志；请求日志必须按 trace/session 隔离。

## Skill

- Skill 放在 `.agent/skills/<name>/SKILL.md`，名称使用小写连字符。
- Frontmatter 只保留 `name` 和 `description`；description 同时说明能力和触发场景。
- 新 Skill 必须经 Registry 发现、作用域测试和结构校验；由目标 Agent 的固定 allowlist 按需加载，禁止全局注入。
- 当前只有 Synthesizer 可以加载写作 Skill；Assistant、RAG、Search 和 Reviewer 不读取 Skill 正文，检索与安全停止条件继续由代码兜底。
- 禁止运行时自动生成并永久启用 Skill，防止对话内容污染后续任务。

## 安全、配置与性能

- `.env` 是部署配置，不把真实 API key 写入文档、测试、错误信息或提交内容。
- 不随意提高并发、上下文或 GPU 显存占用；服务器需要为 PDF/表格解析保留显存。
- 延迟优化先记录 Assistant、Search/RAG、Synthesizer、TTFT 和端到端耗时，再改模型或预算。
- 快速/深度模式只能改变预算和模型路径，不能降低引用、来源或反幻觉标准。
- 缓存失败不得改变回答正确性；prefix cache 只降低 prefill，不代表缓存回答。

## 验证

1. 为行为变化增加确定性测试，优先覆盖路由、状态、去重、格式和 manifest。
2. 运行目标测试，再运行 `python -m pytest -q`。
3. 修改 `SKILL.md` 时运行 Skill 结构校验，并测试显式调用、自动命中和无关问题不命中。
4. 修改 SSE 时验证至少两个 token 事件、token 拼接等于 final、历史消息成功保存。
5. 修改后台索引时使用临时 SQLite 和 fake store 验证排队、恢复、失败、重试及“仅 completed 可查询”，不得触碰真实图谱。
6. 真实 vLLM、GPU 解析、LightRAG 图查询和联网 API 无法在本地证明时，明确列出服务器验收步骤。
7. 行为变化后更新相应文档；测试通过不代表部署环境变量已经同步，重启后仍需接口回归。
