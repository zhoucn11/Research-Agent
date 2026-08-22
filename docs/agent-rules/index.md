# Agent Rules Index

根目录 `AGENTS.md` 是入口，本目录存放详细工程规则。按任务只读取必要文档，避免把所有知识一次塞进上下文。

## 必读规则

- [engineering-rules.md](engineering-rules.md)：架构边界、证据安全、状态协议、验证与文档同步规则。

## 现有知识索引

| 主题 | 来源 | 何时读取 |
|---|---|---|
| 包结构与入口 | [../code_structure.md](../code_structure.md) | 新增模块、移动文件或定位入口 |
| 上下文记忆 | [../context_lightrag_upgrade.md](../context_lightrag_upgrade.md) | 修改 token 水位、摘要、会话恢复 |
| LightRAG 与索引 | [../context_lightrag_upgrade.md](../context_lightrag_upgrade.md) | 修改切块、embedding、建图、manifest、查询预算 |
| 项目级 Skill | [../context_lightrag_upgrade.md](../context_lightrag_upgrade.md) | 新增或修改 `.agent/skills`、Registry、触发规则 |
| 完整系统设计 | [../research_agent_project_review_v2.md](../research_agent_project_review_v2.md) | 理解端到端数据流、准备架构说明 |
| Harness 补强清单 | [../AI_AGENT_BOOK_GAP_CHECKLIST.md](../AI_AGENT_BOOK_GAP_CHECKLIST.md) | 规划评测、轨迹、异步任务、安全和扩展性 |
| 人工端到端验收 | [../AGENT_SELF_TEST_CASES.md](../AGENT_SELF_TEST_CASES.md) | 服务器重启后的综合测试 |
| 机器评测样例 | [../AGENT_EVAL_DATASET.jsonl](../AGENT_EVAL_DATASET.jsonl) | 扩展路由、记忆、RAG、抗幻觉评测 |

## 维护约定

新增知识时优先更新已有文档；只有主题具有独立所有权和更新节奏时才新建文件。根 `AGENTS.md` 只增加入口链接和全局硬约束，不复制详细正文。发现文档与代码冲突时，以实际代码和测试为证据修正文档，并记录无法在本地验证的服务器条件。
