# Research Agent

面向学术论文调研的多 Agent 研究助手。系统基于 LangGraph 编排本地论文检索、联网学术搜索、证据门控、综述生成和独立 Reviewer，支持 PDF/图片输入、LightRAG 图谱检索、页级证据、长期会话记忆、SSE 流式输出与后台可恢复建图任务。

## 核心能力

- 本地论文：DeepDoc 解析 PDF，LightRAG 联合实体、关系和原文 chunk 检索。
- 联网检索：接入 Semantic Scholar，支持精确标题和主题检索、限速与有限重试。
- 证据治理：候选论文只有通过标题、来源、字段和页级证据检查后才能进入综述。
- 多 Agent：Assistant 负责路由，RAG/Search 负责取证，Synthesizer 负责写作，独立 Reviewer 检查来源归属与语义幻觉，并允许一次定向返修与二次审阅。
- 工程闭环：FastAPI + SSE + SQLite 会话/检查点/轨迹，PDF 建图任务支持排队、恢复、失败记录和显式重试。

主要目录：

```text
research_agent/agents/       Assistant、RAG、Search、Synthesizer、Reviewer
research_agent/core/         LangGraph、状态、模型容错、证据门、运行轨迹
research_agent/retrieval/    LightRAG、学术 API、PDF 解析与后台索引
research_agent/memory/       上下文压缩、会话记忆和用户画像
research_agent/api/          FastAPI 与 SSE
frontend/                    原生 Web 前端
deepdoc/                     RAGFlow/DeepDoc 风格 PDF 解析适配
.agent/skills/               Synthesizer 使用的综述写作 Skill
```

## 运行条件

- Python 3.11 或 3.12
- Linux + NVIDIA GPU（PDF 解析和本地模型推荐）
- 一个 OpenAI Chat Completions 兼容的主模型 API
- 一个本地 vLLM 服务，用于 LightRAG 建图和低风险批处理
- DeepDoc 模型权重（默认放在仓库内的 `models/deepdoc/`）

本仓库不提交 API Key、模型权重、论文原文、SQLite 数据库或已生成的 LightRAG 图谱。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell 使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填写主模型、Reviewer、Semantic Scholar 和本地模型连接。算法预算、并发、重试和 LightRAG 参数统一维护在 `research_agent/core/app_config.py`，不需要写入 `.env`；部署环境仍可用同名环境变量临时覆盖代码默认值。

## 启动本地 vLLM

下面是 16K 上下文、低显存占用的参考命令，模型路径按服务器实际位置修改：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80 \
  --dtype half \
  --enforce-eager \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 6006
```

## 启动 Web 服务

```bash
python api_server.py
```

浏览器访问 `http://127.0.0.1:8080`。首次上传 PDF 后，服务会创建后台索引任务；只有 LightRAG 文档状态真正进入 `processed` 才会写入 manifest。图谱和数据库均为本地运行产物，不需要提交 GitHub。

CLI 模式：

```bash
python main.py
```

## 安全说明

- 不要提交 `.env`，公开仓库只保留 `.env.example`。
- 不要提交 `lightrag_storage/`、`test_pdfs/`、`uploaded_assets/` 和模型权重。
- PDF 与网页正文均视为不可信数据，不允许其中的指令改变 Agent 控制流。
- 页级引用只能来自真实 LightRAG chunk，证据不足时必须明确停止或降级。
