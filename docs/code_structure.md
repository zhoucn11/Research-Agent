# Code Structure

项目代码按职责拆到 `research_agent/` 包内，根目录只保留启动入口和资源目录。

```text
research_agent/
  api/        FastAPI 服务、SSE 流式输出、前端接口
  agents/     LangGraph 节点：主控、RAG、联网检索、合成、审阅
  core/       图编排、共享状态、模型容错、Tool 校验、页级证据、运行事件与持久轨迹
  memory/     上下文窗口治理、SQLite 会话记忆和长期用户画像
  retrieval/  LightRAG 图谱存储、自适应查询、PDF 解析、持久化后台索引任务、学术 API 与图文预处理
  schemas/    Pydantic 数据结构
  tools/      LLM 工具 schema
```

索引链路的职责进一步拆分为：`pdf_indexing.py` 只负责 DeepDoc 解析和 LightRAG 文档构造，`index_jobs.py` 负责 SQLite 任务状态机、恢复、重试和单 worker 执行，`lightrag_store.py` 负责索引事务及 manifest 真完成校验。聊天节点 `rag_agent.py` 只读已完成索引，不再同步建图。

可观测性链路由 `runtime_events.py` 生成请求级事件，`trace_store.py` 将非 token 事件脱敏写入 SQLite；`llm_clients.py` 按主 Qwen、Kimi Reviewer、本地 vLLM 三个角色隔离并发、重试和熔断；`app_config.py` 集中维护非敏感算法默认值，并允许部署环境用同名变量覆盖；`tool_validation.py` 是 Tool 执行前的确定性参数边界。

根目录入口：

```text
api_server.py  Web 服务启动入口，兼容原来的运行方式
main.py        命令行交互入口，兼容原来的运行方式
```
