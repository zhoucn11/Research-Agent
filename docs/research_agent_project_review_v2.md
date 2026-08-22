# Academic Research Agent 项目梳理升级版

> 2026-08 更新：本地检索已由 FAISS/BM25 普通 RAG 迁移为 LightRAG 图谱检索，上下文记忆采用 60%/78%/90% token 水位的三层惰性压缩；自生成 Skill 功能已移除，只保留会话记忆和长期用户画像。本文后续对旧 FAISS/Skill 链路的描述仅用于说明演进过程，当前实现与部署参数以 `docs/context_lightrag_upgrade.md` 为准。

> 2026-08-12 更新：LightRAG 查询已从固定 `mix` 改为 `auto` 策略（global/local/naive/mix），新增 `EvidenceSpan` 页级证据和 PDF 定位链接；LangGraph 改用异步 SQLite checkpoint；SSE 日志改为请求级结构化事件，支持 trace/node/TTFT/总耗时；会话与长期画像按用户命名空间隔离；前端提供自动/快速/深度三档预算。本轮没有实施完整系统综述平台，正式质量评测按计划最后完成。

> 2026-08-20 更新：项目级 Skill 已收敛为 Synthesizer 专用的 `literature-review-writing`。检索和引用核验不再通过 Skill 注入 Assistant，而由现有路由、停止条件和元数据校验代码负责；写作 Skill 只组织已确认的论文证据，不具备检索或改写证据的权限。

> 2026-08-20 可靠性更新：`candidate_papers` 通过确定性 Evidence Gate 后才能升级；Reviewer 改为来源归属与语义幻觉的结构化裁决，并保留一次定向返修与二次审阅；PDF 解析和 LightRAG 建图迁移为可恢复后台任务；长期用户画像以低优先级偏好数据参与 Assistant/Synthesizer。

> 2026-08-21 工程更新：Assistant 与上下文压缩迁移到主 Qwen API，Reviewer 继续独占 Kimi；三类 provider 使用独立并发、分类重试和熔断。Tool 参数改由 Pydantic 确定性裁决，执行轨迹脱敏持久化到 SQLite，并提供 trace 查询接口。

生成时间：2026-04-26

本文档基于当前本地代码、前端实现和原始 `Research Agent.docx` 重新整理，目标不是替换原文档，而是形成一份更贴近当前版本的项目讲解材料。当前项目已经从最初的“文本问答 + 本地 RAG + 联网检索”升级为“图文输入预处理、会话隔离、长期记忆、LightRAG 图谱检索、前端会话管理”的学术研究助手，因此面试讲解时不能只讲 RAG，也要讲清楚 Agent 编排、状态治理、证据治理和工程化闭环。

## 一、项目核心定位

这个项目本质上是一个面向科研文献调研与综述生成的多 Agent 学术工作流系统。它不是简单的“把 PDF 丢给大模型问答”，而是把一次科研任务拆成几个稳定阶段：用户输入进入后端，后端先做会话、记忆、附件和多模态预处理，然后交给 LangGraph 中心化工作流，由 Assistant 主控节点判断任务应该走本地 PDF RAG、联网学术搜索，还是直接基于已有证据进入综述生成；工具节点把检索结果结构化成候选论文，Assistant 再判断证据是否足够，最后由 Synthesizer 生成综述、Reviewer 做复核。这样做的原因是学术场景最怕两个问题：一是长链路任务里状态混乱，二是综述生成时证据来源不清。项目用 AgentState 把候选证据、最终证据、推理步数、会话摘要、用户偏好等都显式放进状态里，避免模型在多轮对话中凭记忆乱编。

面试里可以这样概括：我做的不是一个单点 RAG demo，而是一个以 LangGraph 为执行内核的科研任务自动化框架。RAG 只是其中一个工具节点，真正的重点是“任务路由、工具调用、状态隔离、证据升级、综述生成和复核”的完整闭环。

## 二、当前代码结构

当前代码已经按职责拆成相对清晰的包结构。根目录的 `api_server.py` 和 `main.py` 只是兼容入口，真正的服务代码在 `research_agent/` 下。`research_agent/api/server.py` 负责 FastAPI 接口、SSE 流式返回、文件上传、会话 API 和前端静态资源挂载；`research_agent/core/graph.py` 负责 LangGraph 工作流搭建；`research_agent/core/state.py` 定义 AgentState；`research_agent/core/tools.py` 把具体 Agent 函数封装成图节点；`research_agent/agents/` 下是 Assistant、RAG、Search、Synthesis 等 Agent 节点实现；`research_agent/retrieval/` 放 LightRAG 图谱存储、本地模型、学术搜索和多模态预处理；`research_agent/memory/` 放上下文治理、SQLite 会话记忆和长期用户画像；`research_agent/schemas/models.py` 定义 PaperSummary 等结构化输出；`frontend/` 是浏览器端界面，包括左侧会话栏、聊天区、日志区、附件上传和 SSE 消费逻辑。

这个结构面试时可以强调一个点：项目没有把所有逻辑堆在一个文件里，而是按“接口层、工作流层、Agent 节点层、检索层、记忆层、前端层”分层。这样后续加 MCP、加更多工具、换模型、换向量库时不需要重写整体架构。

## 三、端到端执行流程

一次完整请求从前端开始。用户在前端输入文本，也可以上传 PDF 或图片。前端通过 `/api/chat` 发送请求，后端 `server.py` 根据 Content-Type 判断是 JSON 文本还是 multipart 表单。如果是 PDF，保存到 `test_pdfs`，后续交给本地 RAG 入库；如果是图片，保存到 `uploaded_assets/<session_id>/`，然后调用 `multimodal_preprocessor.py` 中的视觉模型接口，把图片内容先转成文本。这里的设计是“稳一点”的方案：不把图片直接塞进整个 Agent 工作流，而是先转成可解释的文本上下文，再拼回用户问题。这样后面的 Assistant、RAG、Search、Synthesizer 都还是处理文本，工程风险更低，也更容易 debug。

后端会读取当前会话摘要、长期用户画像，并在 LangGraph checkpoint 为空时注入最近消息作为恢复上下文，随后调用 `agent_app.ainvoke` 进入 LangGraph。Graph 的入口是 Assistant。Assistant 先看用户意图，如果需要本地检索就生成 `trigger_local_retrieval` 工具调用，如果需要联网搜索就生成 `trigger_web_search`，如果已有候选论文并且用户要求“继续整理成综述”，则把候选论文升级为 `selected_papers` 并输出 `[APPROVE_SYNTHESIS]` 进入写作链路。工具节点执行完以后不会直接结束，而是把 ToolMessage 返回给 Assistant，让 Assistant 基于 Observation 再决策。这就是一个简化版 ReAct 循环：Assistant 负责 Thought 和 Action，RAG/Search 节点负责 Observation。

最终结果通过 SSE 一边把日志推给右侧 Runtime Log，一边在结束时推送已审阅 token 和唯一 final。user 消息在后台任务启动前写入 SQLite，assistant 正文在 Guard、Reviewer 和输出约束完成后先落库再发送；浏览器刷新只会断开 SSE，后台任务继续运行并完成会话持久化，同时更新会话摘要和长期用户画像。

## 四、LangGraph 工作流设计

`core/graph.py` 里定义了项目的中心化多 Agent 图。图节点包括 `assistant`、`search_map`、`rag_map`、`synthesizer`、`reviewer`。入口固定为 Assistant，Assistant 的路由函数读取最后一条消息：如果最后消息带 tool_calls，就根据工具名路由到 RAG 或 Search；如果消息内容包含 `[APPROVE_SYNTHESIS]` 且 state 里有 `selected_papers`，就进入 Synthesizer；否则直接 END。

这个设计的重点是“中心化调度”。所有任务不是让多个 Agent 自由互相调用，而是由 Assistant 统一判断下一步。好处是可控、容易调试、状态边界清楚；缺点是 Assistant prompt 会变长，路由稳定性依赖提示词和少量确定性规则。你后面加入了两个确定性补丁：一个是用户明确说“基于本地 PDF/本地文献库”时强制进入本地 RAG，避免大模型直接回答；另一个是用户说“继续基于刚才结果写综述”时强制进入 Synthesizer，避免主脑把写作任务自己消化掉。这两个修改说明项目不是完全相信 LLM 自主路由，而是在关键业务路径上加规则兜底。

面试常问“为什么用 LangGraph 而不是普通 LangChain Chain”。可以回答：普通 Chain 更像线性流水线，适合固定步骤；我的任务存在多轮检索、工具观察、证据是否足够的分支判断，LangGraph 可以把状态、节点和条件边显式建模，既能保留 ReAct 式动态决策，又能限制最大步数和关键状态，工程上更可控。

## 五、AgentState 与状态治理

`core/state.py` 是项目里非常关键的一层。AgentState 里不仅有 messages，还包括 candidate_papers、selected_papers、draft_review、pdf_file_paths、summary、indexed_files、step_count、research_goal、collected_evidence、pending_questions、user_profile 等字段。candidate_papers 表示检索节点刚产出的候选论文，selected_papers 表示 Assistant 确认后可以进入最终综述的证据。两者分开是为了避免“检索到什么就立刻写什么”，必须经过 Assistant 的证据门控。

这里还有一个 reducer 设计：candidate_papers 和 selected_papers 使用可替换 reducer，支持 `CLEAR` 清空。这样新任务开始时可以清掉旧证据，防止上一轮文献混进下一轮任务。多轮 follow-up 则可以保留候选论文，让用户说“继续基于刚才结果写综述”时还能进入写作阶段。面试如果问“如何避免状态污染”，核心答案就是三层：会话级 thread_id 隔离 LangGraph checkpoint；新任务时 CLEAR 候选证据；候选证据和最终证据分开，只有 Assistant 明确批准后才进入 Synthesizer。

## 六、工具协议与 ReAct 解耦

`tools/agent_tools.py` 中定义了三个工具：`trigger_web_search`、`trigger_pdf_upload`、`trigger_local_retrieval`。这些工具本身不是直接执行函数，而是给 LLM 暴露的 schema。Assistant 生成工具调用后，Graph 根据 tool name 路由到对应节点，节点执行完后返回 ToolMessage。这个设计和 MCP 的思想有相似之处：能力通过 schema 暴露，模型只负责决定是否调用和传参，真正执行逻辑在工具节点内部。

面试里可以把这个讲成“工具声明和工具执行分离”。工具声明层负责告诉模型有哪些动作可选，执行层负责真实检索、入库、搜索和结构化抽取。这样以后如果接 MCP，就可以把现在的本地工具包装成 MCP tools，或者让 MCP server 暴露外部数据库、论文库、文件系统等资源，而不需要重写 Assistant 的核心决策逻辑。

## 七、Assistant 主控节点

`agents/assistant_agent.py` 是主脑。它做了几类事情：加载模型并 bind_tools；读取历史消息并触发记忆压缩；扫描本地 PDF 和向量库构造资产清单；把 candidate_papers/selected_papers 拼进系统提示词；维护 step_count 和最大推理步数；根据工具调用、暗号和确定性规则更新状态。

Assistant 的 prompt 很长，是因为它承担了路由、证据审查和行为约束。里面明确规定本地和联网隔离，禁止把联网搜到的论文当成本地库里的论文；本地 RAG 只能查真实存在的 PDF；全库总结要用 `SUMMARY_ALL`；同一 PDF 不要反复精读；检索无结果时要关键词降维而不是越搜越复杂；证据满足条件时才输出 `[APPROVE_SYNTHESIS]`。这部分在面试中不要只说“写了 prompt”，要说“我把 prompt 当成一种运行时策略层，用来约束 Agent 的工具边界、停止条件、证据升级条件和失败兜底”。

如果面试官问“为什么还要确定性规则，不能完全让大模型判断吗”，可以回答：生产系统不能完全依赖模型遵守提示词。像本地检索、继续综述这类高频关键路径，如果模型偶尔偷懒直接回答，用户体验会很差，所以我在 Assistant 前置了轻量规则，命中明确意图时直接构造 tool_call 或 APPROVE_SYNTHESIS。这是用规则兜住业务确定性，用 LLM 处理语义不确定性。

## 八、本地 PDF RAG 模块

`agents/rag_agent.py` 是本地 RAG 的核心。它先检测 `test_pdfs` 下的 PDF 和已有 FAISS 索引是否同步。索引版本由 `RAG_INDEX_VERSION` 控制，如果 PDF 变化、索引版本变化或向量库内容和硬盘文件不一致，就触发重建或增量入库。PDF 解析优先使用 DeepDoc/RAGFlow 风格解析器，尝试保留版面、表格、图注和页面位置信息。解析后的文本会归一化，包括去除断词、替换页码 marker、压缩空白。

切块不是纯固定长度，而是先做章节识别。代码里定义了 abstract、introduction、related_work、method、experiment、discussion、conclusion、references 等 section pattern，然后按章节切分，再用 RecursiveCharacterTextSplitter 做 chunk。每个 chunk 的 metadata 里保存 source、fixed_title、fixed_year、section、chunk_index、page_start、page_end、source_mtime 和索引版本。这个设计比普通切块更适合论文，因为用户问“方法”“实验”“局限”时，可以优先找对应章节，后续也方便做引用定位。

检索阶段分两层。第一层是 source selection：如果用户指定 PDF 文件名，就直接精确锁定；如果是全库总结，就调全库；否则先通过查询重写生成多组 recall query，在 FAISS 里海选相关 source，限制最多 `RAG_MAX_SOURCE_FILES`。第二层是 per-file retrieval：对每篇候选 PDF 构建 BM25Retriever 和 FAISS retriever，再用 EnsembleRetriever 融合，最后用 CrossEncoder reranker 做压缩重排。这样同时利用关键词精确匹配和向量语义召回，适合学术论文里大量术语、缩写和方法名共存的场景。

最后，RAG 节点把检索上下文交给本地 LLM 做结构化抽取，输出 PaperSummary，包括 title、authors、year、source、core_method、key_findings。随后调用 Semantic Scholar 风格的联网元数据校验逻辑，对标题、年份、作者做交叉修正，但代码里有相似度门槛，避免联网标题错误覆盖本地论文身份。

面试高频问题是“为什么 BM25 + Dense + Reranker 要三段做”。可以回答：Dense 适合语义相似，但容易漏掉精确方法名、模型名；BM25 对关键词和术语更稳，但不理解同义表达；Reranker 在候选集较小后做精排，能用更强的交叉编码器判断 query 和 chunk 的真实相关性。三者组合比单纯向量库更适合论文检索。

## 九、查询重写模块

`retrieval/query_rewriter.py` 是你这次新增的检索优化重点。它定义了 `RetrievalQueryPlan`，包含 original_query、semantic_queries、keyword_queries、hyde_query 和 query_intent。触发本地 RAG 后，如果 query 不是空、不是 `SUMMARY_ALL`、也不是精确 PDF 文件名，就会调用强模型做查询重写。重写规则要求保留方法名、数据集、模型名、指标和年份，不回答问题，只生成适合 Dense 的语义查询、适合 BM25 的短关键词查询，以及一段 HyDE 风格的假想证据段落。

这个模块应该放在“路由之后、进入具体检索之前”，而不是用户刚输入就全局重写。原因是不同路由的重写目标不同：本地 RAG 要优化 chunk/source 召回，联网搜索要优化搜索关键词，普通聊天不需要重写。如果用户一输入就重写，可能把原始意图改坏，还会影响 Assistant 判断到底该走哪个工具。现在的做法是 Assistant 先判定任务类型，RAG 节点再对检索 query 做局部重写，边界更清楚。

面试里可以这样讲：查询重写不是为了“让问题更好看”，而是为不同检索器生成不同视角的召回入口。semantic query 让向量召回覆盖同义表达，keyword query 保住术语和缩写，HyDE query 提供一个理想答案形态帮助 Dense retrieval 拉近相关段落。最终这些 query 去重后用于 source selection 和 per-file retrieval。

## 十、联网学术搜索模块

`agents/search_agent.py` 和 `retrieval/academic_search.py` 负责联网学术检索。它的定位不是替代本地 RAG，而是在本地没有资料、用户要求最新论文、或者需要外部元数据校验时补充外部证据。联网侧接入 Semantic Scholar/OpenAlex 获取标题、作者、年份和摘要，默认直接用 API 字段构造结构化证据；需要中文深度提炼时可开启主 Qwen API enrichment，不再占用本地模型。

面试里要强调本地和联网的证据边界。联网搜索得到的是外部候选，不代表本地 PDF 已经存在；本地 RAG 得到的是用户上传或本地库里的全文证据。Assistant prompt 里明确禁止把联网搜到的论文名拿去本地 RAG 精读，除非物理文件确实存在。这是防止“工具幻觉”的关键设计。

## 十一、Synthesizer 与 Reviewer

`agents/synthesis_agent.py` 负责把 selected_papers 写成综述，Reviewer 负责证据裁决而不是润色。写作节点只读取 selected_papers；候选论文必须先通过精确标题、来源数量、必要字段和 EvidenceSpan 覆盖检查。编号、字段、引用格式和无证据数值由代码 Guard 判断；Reviewer 使用 Kimi K2.6 非思考模式，只接收压缩后的初稿、论文元数据和每篇最多 5 条 EvidenceSpan，检查来源错配与语义幻觉。首审不通过只把结构化问题交回 Synthesizer 做一次定向返修，二审仍不通过才输出可回链证据摘要；接口不可用不会被误判成内容驳回，也不会触发返修。

Assistant、上下文压缩和 Synthesizer 复用 `OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL` 对应的主 Qwen；Reviewer 通过 `REVIEWER_BASE_URL/REVIEWER_API_KEY/REVIEWER_MODEL` 独占 Kimi。Reviewer 配置缺失时不再回退主模型，而是输出可回链证据摘要。主 API、Reviewer 和本地 vLLM 分别配置并发、超时、分类重试和熔断，避免模型故障相互传染。

这部分面试可以回答两个问题。第一，为什么不让 Assistant 直接写综述？因为 Assistant 已经承担路由和工具决策，如果让它同时写长文，会导致职责过重，也容易在工具调用和最终回答之间混淆。第二，为什么 Reviewer 不能只是润色？因为学术回答最重要的是 claim 是否被正确来源支持；因此 Reviewer 只做证据裁决，不能扩展证据池，输出通过、驳回或安全降级的可测试状态。

## 十二、多模态输入模块

当前多模态只支持图文，不做语音。实现位置在 `retrieval/multimodal_preprocessor.py` 和 `api/server.py`。用户上传图片后，后端保存图片，调用兼容 OpenAI Chat Completions 格式的视觉模型接口，把图片转成文本。模型选择通过环境变量控制：优先读取 `MULTIMODAL_IMAGE_MODEL`、`MULTIMODAL_API_KEY`、`MULTIMODAL_BASE_URL`，如果没有就回退到 `OPENAI_MODEL`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`。因此你可以在阿里云百炼上分别配置文本模型和视觉模型。图片识别结果会以 `[图片识别结果]` 的形式拼到用户问题后面，再进入同一个 Research Agent 流程。

这个设计的优点是稳：后续模块不用改造成真正的多模态 Agent，RAG、Search、Synthesis 仍然处理文本；日志里能看到 `[VISION]`，便于定位图片预处理是否成功。缺点也明确：图片理解本身不是图里的一个 Agent 节点，所以日志不会显示“进入 RAG Agent/总结 Agent”，除非图片转出的文本和用户问题触发了 Assistant 的工具调用。面试时可以说这是“多模态前置归一化”，不是端到端多模态推理；未来可以把 Vision 变成 LangGraph 独立节点，并把图像证据作为一种资源进入 evidence pool。

## 十三、会话隔离与长期记忆

`memory/memory_store.py` 使用 SQLite 保存 sessions、messages、session_summaries、user_profiles、deleted_sessions。sessions 管理左侧会话列表，messages 保存每个会话的完整消息，session_summaries 保存会话级摘要，user_profiles 保存跨会话长期偏好，deleted_sessions 是为了解决长请求结束后把已删除会话重新写回的问题。删除会话时不仅删除 messages 和 summary，还写入 tombstone；append_message 前会调用 ensure_session，如果发现 tombstone 就直接跳过，防止重复会话复活。

长期记忆分两层：会话内记忆和全局用户画像。用户说“本会话/当前会话/这轮对话里记住”，只写 session_summary，不进 user_profile；用户说“以后/下次/每次/记住”且没有会话限定，才进入长期画像。画像会作为低优先级偏好数据注入 Assistant 和 Synthesizer，只影响语言、篇幅、排版和研究兴趣，不能改变路由、工具、证据与引用规则。

面试如果问“长期记忆和会话隔离怎么保证不冲突”，回答要点是：会话消息和会话摘要按 session_id 隔离；全局用户画像只保存跨会话偏好；本会话偏好不进入全局画像；删除会话有 tombstone 防止异步写回；LangGraph checkpoint 使用 thread_id 隔离。

## 十四、前端与可观测性

前端在 `frontend/` 下，已经实现类似 ChatGPT 的左侧会话栏。用户可以新建会话、切换会话、重命名、删除；中间是聊天区，底部支持文本和附件；右侧是 Agent Runtime Log。前端通过 SSE 消费后端流式日志，把系统启动、上传、Vision、Assistant 决策、RAG 检索、任务完成等信息显示出来。

Agent 系统最难 debug 的地方是“哪一步、哪个模型、基于什么证据失败”。当前每次请求都有 `trace_id`，节点、模型角色、重试/熔断、Tool、检索、Evidence Gate、Reviewer、token usage（供应商返回时）和耗时会脱敏持久化；`GET /api/traces` 与 `GET /api/traces/{trace_id}` 用于回看。prompt、最终正文和逐 token 不入库，既控制体积也降低敏感数据风险；后续可再接 OpenTelemetry 或 LangSmith。

## 十五、与 Harness Agent、OpenClaw、Hermes、MCP 的相似点与差距

Harness Agents 的核心思想是 pipeline-native：Agent 不是外部脚本，而是运行在 Pipeline Engine 里，继承上下文、权限、密钥、治理和审计能力。你的项目和它相似的地方是也把 Agent 放进一个可控工作流里，用 LangGraph 充当执行编排层，而不是让模型无限自由行动。不同点是 Harness 更偏 DevSecOps 生产流水线，强调 RBAC、审批、回滚、审计和模板版本化；你的项目更偏科研文献工作流，目前权限治理和审计还比较轻。

OpenClaw 的 ACP/外部 harness 思路强调让不同 CLI Agent 或外部运行时通过统一协议接入，并支持后台任务、会话绑定和 runtime 控制。你的项目目前也有“工具 schema + 图节点执行”的雏形，但还没有统一 adapter 协议。如果要借鉴 OpenClaw，可以把本地 RAG、Semantic Scholar、PDF 解析、Vision、文件系统都封装成标准工具或 MCP server，再让 Assistant 只依赖工具注册表，而不是硬编码工具列表。

Hermes Agent 值得借鉴的是长期运行、Memory、MCP 和 CLI/Gateway 组合。当前项目保留可解释的会话记忆与用户画像，没有把对话自动转成可执行 Skill，避免未经审核的规则持续污染后续任务。

MCP 的参考价值最大。MCP 把能力分成 Tools、Resources、Prompts：Tools 是模型可调用动作，Resources 是应用可读取上下文，Prompts 是用户可选择的模板。你的项目现在已经有 Tools 形态，但 Resources 和 Prompts 还没标准化。未来可以把本地 PDF 库暴露为 resources，把综述写作模板、论文对比模板、实验表格提取模板暴露为 prompts，把 Semantic Scholar、Vision、RAG 检索暴露为 tools。这样项目会从“自定义 Agent 系统”升级成“兼容协议的科研 Agent 平台”。

## 十六、先进 RAG 方法对当前项目的启发

当前项目的 RAG 主要是结构化切块 + BM25/Dense 混合召回 + CrossEncoder 重排 + 元数据校验，这已经比普通向量库问答强很多。但如果继续升级，可以重点参考四类方法。

第一是 GraphRAG。GraphRAG 解决的是“全局问题”，例如“这批论文主要研究主题是什么”“不同方向之间有什么关系”。普通 RAG 擅长找局部片段，但很难回答跨全库的主题归纳。GraphRAG 的思路是先从文档中抽取实体、关系和 claim，构建知识图谱，再对社区做摘要，查询时检索社区摘要并 map-reduce 成最终答案。你的项目里全库综述、研究方向归类、技术路线梳理都很适合引入轻量 GraphRAG。

第二是 RAPTOR。RAPTOR 通过递归聚类和摘要构建树状索引，让系统既能召回底层 chunk，也能召回高层摘要。它适合长论文、多章节材料和综述类问题。你现在已经有章节化 chunk，下一步可以在每篇 PDF 内部生成 section summary，再在多篇论文之间生成 topic summary，形成“chunk -> section -> paper -> topic”的层次索引。

第三是 Self-RAG/CRAG 类自反思检索。Self-RAG 强调模型在生成过程中判断是否需要检索、检索内容是否有用、回答是否被证据支持；CRAG 强调对检索质量做评估，如果静态库检索不好，就触发外部搜索或过滤重组。你的项目已经有 Assistant 证据门控，但还没有独立的 retrieval evaluator。未来可以在 RAG 后加一个 EvidenceJudge 节点，判断召回是否足够、是否需要改写 query、是否需要联网补充、是否应拒答。

第四是多向量和 late interaction。学术论文常见表格、公式、图注、方法名、实验结果分别承载不同信息，单一 chunk embedding 不一定够。后续可以把 title/abstract/method/experiment/table/caption 分成不同向量字段，或对每篇论文建立多粒度 embedding；也可以引入 ColBERT/late interaction 检索增强术语匹配能力。

## 十七、当前不足与未来规划

当前项目的第一类不足是路由和 prompt 仍然偏重。Assistant prompt 承担了很多规则，长期看会导致维护成本高、上下文占用大。改进方向是把路由拆成轻量 Router/EvidenceJudge/Planner 节点，让每个节点职责更单一，并把规则从 prompt 迁移为可测试的策略函数。

第二类不足是 RAG 缺少系统评测。现在能跑通，但还需要建立论文检索评测集，统计 source recall、chunk recall、rerank NDCG、answer faithfulness、citation accuracy。生产级 RAG 不能只靠人工感觉，需要 badcase 池和指标闭环。

第三类不足是图像输入只是前处理。它能稳定转文本，但没有把图片当成可追踪证据源。未来可以把图片识别结果保存为 multimodal resource，记录文件名、OCR 文本、视觉描述、模型版本和置信度，并允许后续引用图片证据。

第四类不足是长期记忆还没有权限和可编辑界面。现在用户可以通过接口删除关键词，但更合理的是在前端提供“记忆管理”页面，用户能查看、删除、禁用某条偏好。长期记忆必须用户可控，否则容易产生隐私和污染问题。

第五类不足是工具治理还没有覆盖授权与副作用。当前已有固定 allowlist、严格输入 schema、单步单 Tool、节点二次校验和一次纠错；未来接 MCP 或写操作工具时，仍需要 per-server filtering、权限声明、危险操作审批、幂等键和 human-in-the-loop。

第六类不足是生成内容缺少引用链。PaperSummary 里有 source，但最终综述如果要更学术，应该把每个结论绑定到具体论文、章节、页码或 chunk id，做到可追溯引用。这一点可以结合现有 page_start/page_end metadata 实现。

## 十八、面试高频问题与回答要点

问题：你的项目和普通 RAG 问答有什么区别？
回答：普通 RAG 多数是“检索 chunk -> 拼 prompt -> 生成答案”，我的项目把 RAG 放进了多 Agent 工作流里。Assistant 先判断任务意图，再调用本地 RAG 或联网搜索，检索结果只进入 candidate_papers，必须经过 Assistant 确认才升级为 selected_papers，然后由 Synthesizer 和 Reviewer 完成综述生成。重点不是单次问答，而是科研调研任务的状态治理、证据治理和多阶段协同。

问题：为什么要把 candidate_papers 和 selected_papers 分开？
回答：candidate_papers 是工具节点返回的候选结果，可能不完整、不相关或只是中间观察；selected_papers 是通过 Evidence Gate 的最终证据池。门控会检查点名标题、对比来源数、问题要求的字段和本地页级证据覆盖，因此模型即使输出批准暗号也不能绕过。

问题：查询重写放在哪里，为什么？
回答：放在路由之后、本地 RAG 执行之前。因为查询重写服务于具体检索器，不同路由目标不同。用户刚输入就重写容易影响意图判断，甚至把普通聊天改成检索任务。现在是 Assistant 先确定走本地检索，RAG 节点再生成 semantic query、keyword query 和 HyDE query，用于 source selection 和 per-file retrieval。

问题：为什么要做长期记忆？怎么避免污染？
回答：长期记忆用于保存稳定偏好，比如用户常研究的方向、输出格式和回答风格，这样下一次打开系统仍能延续个性化体验。为了避免污染，我把记忆分成会话级摘要和全局用户画像；带“本会话/当前会话”的偏好只写 session_summary，不进入 user_profile；删除会话时写 tombstone，防止异步请求结束后把已删除会话复活。

问题：图片输入为什么没有显示进入 Agent 流程？
回答：当前图片是多模态前置归一化，先由视觉模型转成文本，然后把文本并入用户问题，再交给原有 Agent 流程。所以日志会显示 VISION 完成，但不一定进入 RAG 或 Synthesizer。只有图片转出的文本和用户指令触发了本地检索、联网搜索或综述写作，才会进入对应 Agent 节点。这个方案牺牲了一点端到端多模态能力，但稳定、易调试、和现有文本 Agent 兼容。

问题：你的系统如何降低幻觉？
回答：主要有几层：本地 RAG 只查 manifest 真完成的 PDF；联网结果和本地库严格隔离；PDF 入库在可恢复后台任务中完成；元数据交叉校验有标题相似度门槛；Evidence Gate 阻止不完整候选进入写作；Synthesizer 只基于结构化证据生成；代码 Guard 检查编号、字段和数值，Reviewer 检查来源错配与语义幻觉，首审失败只允许一次定向返修，二审仍失败才安全降级。

问题：如果让你继续优化，你优先做什么？
回答：我会先做评测和证据追踪，因为这是生产级 RAG 的底座。具体是建立查询-相关论文-相关 chunk 的评测集，统计 recall 和 citation accuracy；然后把最终综述每句话绑定 source/chunk/page。第二步做 GraphRAG/RAPTOR，解决全库主题归纳和跨论文综述。第三步做 MCP 化，把 RAG、搜索、Vision、文件资源标准化为工具和资源，提升可扩展性。

问题：你如何把项目往 Harness/Hermes/MCP 方向讲？
回答：可以说我的项目底层思想和这些新 Agent 架构一致：不是做单轮聊天，而是把 Agent 放进可控运行时。LangGraph 对应 Harness 的 pipeline runtime，AgentState 对应共享上下文和执行状态，tool schema 对应 MCP tools，会话摘要和用户画像对应可治理的 Memory。不同点是我的项目聚焦学术研究，暂时没有企业级权限、审批和协议化工具市场；未来会把工具注册、资源读取和提示模板进一步标准化。

## 十九、可落地的升级路线

短期路线：完善 query rewrite 日志和评测，把每次重写出的 semantic/keyword/HyDE query、命中的 source、rerank 后 chunk 保存下来，方便 badcase 分析；前端增加记忆管理页，让用户查看和删除长期偏好；最终综述增加引用来源展示。

中期路线：加入 EvidenceJudge 节点，对 RAG/Search 结果做充分性评估；加入 RAPTOR 式层次摘要索引，先做 paper summary 和 topic summary；把图片识别结果保存为可追踪 multimodal resource。

长期路线：继续完善 LightRAG 的论文实体、方法、数据集、任务和指标图谱；把本地工具改造成 MCP server，支持 tools/resources/prompts 三类能力；最终形成一个可插拔、可观测、可治理的科研 Agent 框架。

## 二十、参考资料

1. Harness Agents 官方文档：Harness Agents 是 pipeline-native AI workers，运行在 Harness pipelines 中并继承上下文、权限、密钥和治理能力。https://developer.harness.io/docs/platform/harness-ai/harness-agents/
2. MCP 官方文档：MCP 以 Host/Client/Server 架构提供标准化上下文交换，核心能力包括 Tools、Resources、Prompts。https://modelcontextprotocol.io/docs/learn
3. MCP Tools：工具是模型可调用的 schema-defined interfaces，并建议 human-in-the-loop。https://modelcontextprotocol.io/docs/concepts/tools
4. MCP Resources：资源是应用驱动的上下文数据，可用 URI 标识并由客户端读取。https://modelcontextprotocol.io/docs/concepts/resources
5. MCP Prompts：Prompts 是服务器暴露给客户端的可复用提示模板，通常由用户显式选择。https://modelcontextprotocol.io/docs/concepts/prompts
6. Hermes Agent MCP 文档：Hermes 通过 MCP 接入外部工具服务器，支持自动发现、资源和 prompts 包装、per-server filtering。https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
7. Hermes Skills 指南：Hermes 使用 Skills 作为可安装扩展机制，可沉淀可复用能力。https://openclawlaunch.com/guides/hermes-agent-skills
8. OpenClaw ACP Agents 文档：OpenClaw 支持通过 ACP runtime 调用外部 coding harness，并以后台任务形式跟踪会话。https://docs.openclaw.ai/tools/acp-agents
9. Microsoft GraphRAG 论文：GraphRAG 通过实体图谱和社区摘要解决全局语料问题。https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/
10. RAPTOR 论文：RAPTOR 通过递归嵌入、聚类和摘要构建树状检索结构。https://arxiv.org/abs/2401.18059
11. Self-RAG 论文：Self-RAG 通过检索与生成过程中的自反思提升事实性。https://arxiv.org/abs/2310.11511
12. CRAG 论文：CRAG 通过检索质量评估、外部搜索补充和去噪重组提升 RAG 鲁棒性。https://huggingface.co/papers/2401.15884
