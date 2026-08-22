# Research Agent Repository Guide

本文件作用于整个仓库。它是编码 Agent 的入口地图，不是完整设计文档；开始修改前，先根据任务读取下方对应资料。更深目录若新增 `AGENTS.md`，其规则只覆盖该目录并优先于本文件。

## 项目目标

这是一个面向学术论文的 LangGraph Research Agent，提供本地 LightRAG、联网论文检索、引用核验、多轮记忆、综述合成、SSE 流式输出和项目级 Skill。优先保证证据可追踪、路由可解释、失败可停止，禁止用“看起来合理”的内容替代真实论文证据。

## 先读索引

- 任务与规则导航：[docs/agent-rules/index.md](docs/agent-rules/index.md)
- 强制工程规则：[docs/agent-rules/engineering-rules.md](docs/agent-rules/engineering-rules.md)
- 代码目录：[docs/code_structure.md](docs/code_structure.md)
- 上下文、LightRAG、Skill 与部署参数：[docs/context_lightrag_upgrade.md](docs/context_lightrag_upgrade.md)
- 完整架构与面试说明：[docs/research_agent_project_review_v2.md](docs/research_agent_project_review_v2.md)
- 基于《AI Agent 开发指南》的补强清单：[docs/AI_AGENT_BOOK_GAP_CHECKLIST.md](docs/AI_AGENT_BOOK_GAP_CHECKLIST.md)
- 人工回归清单：[docs/AGENT_SELF_TEST_CASES.md](docs/AGENT_SELF_TEST_CASES.md)
- 机器评测数据：[docs/AGENT_EVAL_DATASET.jsonl](docs/AGENT_EVAL_DATASET.jsonl)

## 仓库地图

```text
api_server.py                 FastAPI 服务入口，固定监听 8080
main.py                       CLI 入口
research_agent/api/           HTTP、会话、上传、SSE
research_agent/agents/        Assistant、RAG、Search、Synthesizer、Reviewer
research_agent/core/          LangGraph、状态、LLM、证据与 Skill Registry
research_agent/retrieval/     LightRAG、学术 API、本地模型、图文预处理
research_agent/memory/        上下文压缩、SQLite 会话和用户画像
research_agent/tools/         暴露给模型的原子 Tool schema
research_agent/schemas/       Pydantic 边界模型
.agent/skills/                Synthesizer 按需加载的综述写作 Skill
frontend/                     原生 HTML/CSS/JS 前端
tests/                        确定性单元与路由回归
lightrag_storage/             昂贵的派生索引，不是临时目录
```

## 按任务读取

| 任务 | 先读文件 |
|---|---|
| 修改路由、工具调用、停止条件 | `research_agent/agents/assistant_agent.py`、`research_agent/core/graph.py`、`research_agent/core/agent_state_helpers.py` |
| 修改本地论文检索或建图 | `research_agent/agents/rag_agent.py`、`research_agent/retrieval/lightrag_store.py`、`research_agent/core/paper_evidence.py` |
| 修改联网论文检索 | `research_agent/agents/search_agent.py`、`research_agent/retrieval/academic_providers.py`、`research_agent/core/web_search_helpers.py` |
| 修改综述、格式或编号 | `research_agent/agents/synthesis_agent.py`、`research_agent/core/response_format.py`、`research_agent/schemas/models.py` |
| 修改记忆 | `research_agent/memory/agent_memory.py`、`research_agent/memory/memory_store.py` |
| 修改 Skill | `research_agent/core/skill_registry.py`、目标 `.agent/skills/*/SKILL.md` |
| 修改 API、流式或前端 | `research_agent/api/server.py`、`research_agent/api/streaming_utils.py`、`frontend/` |

## 常用命令

```bash
python -m pytest -q
python api_server.py
python main.py
pip install -r requirements-lightrag.txt
```

本地缺少服务器 vLLM、GPU 解析模型或论文文件时，不把无法启动判定为代码失败；先跑确定性测试，再明确说明哪些链路只能在服务器验证。

## 不可破坏的约束

- 不删除、清空或重建 `lightrag_storage/`，除非用户明确要求；embedding 模型或维度变化必须新建索引版本。
- 不把 `ToolMessage` 与对应的 `assistant.tool_calls` 拆散；记忆裁剪必须保留消息协议完整性。
- 不改变 SSE 的 `log`、`token`、`final` 语义；token 拼接必须等于 final。
- 不恢复全局 stdout/stderr 劫持；运行事件必须按 trace/session 隔离。
- 页级引用只能来自真实 LightRAG chunk 的 source/page/chunk，不得由模型补造。
- 不凭常识补论文、作者、年份、DOI、指标或实验结论；证据不足时明确标注。
- Tool 保持原子动作，Skill 保存跨步骤工作流；禁止复制一套新的检索后端到 Skill。
- 精确论文目标只搜索一次，不能通过改写关键词绕过去；主题检索才允许有限的概念轴重试。
- 不在代码、日志、文档或回答中暴露 `.env` 密钥。
- 行为、配置或目录变化时，同步更新对应 `docs/`；不要让文档继续描述已删除链路。

## 完成标准

改动应尽量小，并配套复现问题的测试。至少运行受影响测试；跨模块修改运行 `python -m pytest -q`。涉及服务器模型、真实 LightRAG 或联网 API 时，在本地测试通过后列出服务器验证用例，不伪造运行结果。
