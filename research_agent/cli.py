# main.py
import os
import re

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import asyncio
import glob
from langchain_core.messages import HumanMessage
from research_agent.core.graph import agent_app


async def intelligent_file_router(user_input: str, folder_name: str) -> list:
    """
    【前置小脑：语义路由器】
    加入硬拦截机制，防大模型智障、防耗时、防幻觉。
    """
    if not os.path.exists(folder_name): return []
    all_paths = glob.glob(os.path.join(folder_name, "*.pdf"))
    if not all_paths: return []

    file_map = {os.path.basename(p): p for p in all_paths}
    normalized_input = user_input.lower()
    exact_matches = [
        path
        for name, path in file_map.items()
        if name.lower() in normalized_input or os.path.splitext(name)[0].lower() in normalized_input
    ]
    if exact_matches:
        return exact_matches[:3]

    # ========================================================
    # 🚨 终极物理短路：防 LLM 智障与耗时的关键！
    # 拦截所有“代词”和“序数词”，防止小脑拿本地文件的顺序去瞎凑数！
    # ========================================================
    import re
    fast_skip_keywords = ["哪些", "所有", "全部", "总结", "列出", "有什么", "多少"]

    # 暴力拦截：第X篇、最后一篇、上一篇、刚才、这个、那个...
    is_pronoun_or_index = re.search(r'(第\d+[篇个]|最后|上一|下一|刚才|那个|这篇|那篇|这些|那些)', user_input)

    if (is_pronoun_or_index or any(kw in user_input for kw in fast_skip_keywords)) and len(user_input) < 30:
        return []

    local_intent_markers = ["本地", "文件", "PDF", "pdf", "论文", "文献", "上传", "阅读", "解析"]
    if not any(marker in user_input for marker in local_intent_markers):
        return []

    # 文件选择不再调用模型：只有文件名中的唯一特征词被用户明确提到才命中。
    # 其余歧义交给主 Qwen Agent 根据会话和真实索引决定，避免本地小模型误绑论文。
    token_to_paths: dict[str, list[str]] = {}
    for name, path in file_map.items():
        for token in set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{3,}", os.path.splitext(name)[0].lower())):
            token_to_paths.setdefault(token, []).append(path)
    unique_matches = {
        paths[0]
        for token, paths in token_to_paths.items()
        if len(paths) == 1 and re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", normalized_input)
    }
    return sorted(unique_matches)[:3]


async def interactive_chat():
    print("\n" + "=" * 50)
    print("[SYSTEM] Academic Copilot (Multi-Agent Hybrid Model) Started")
    print("[HINT] 你可以尝试闲聊，也可以下达搜论文指令。")
    print("[HINT] 若要测试本地解析，请确保 'test_pdfs' 文件夹下有 PDF。")
    print("=" * 50 + "\n")

    config = {"configurable": {"thread_id": "user_session_web_002"}, "recursion_limit": 25}
    LOCAL_FOLDER_NAME = "test_pdfs"

    while True:
        raw_input = input("\n[USER]: ")

        if raw_input.lower() in ['quit', 'exit']:
            print("[SYSTEM] Goodbye!")
            break

        if not raw_input.strip(): continue

        clean_input = raw_input.encode('utf-8', 'ignore').decode('utf-8')
        print("\n[ROUTER] 正在分析您的文件意图...", flush=True)

        pdf_paths = await intelligent_file_router(clean_input, LOCAL_FOLDER_NAME)

        if pdf_paths:
            print(f"[🎯 意图命中] 智能路由为您锁定了 {len(pdf_paths)} 份目标文献：")
            for p in pdf_paths[:3]: print(f"  - {os.path.basename(p)}")
            if len(pdf_paths) > 3: print("  - ... (省略显示)")
        else:
            print("[🔍 意图识别] 宽泛检索或闲聊模式，交由后端 RAG 引擎判断。")

        # ========================================================
        # 🌟 核心修复：每次提问前，强行重置所有的“临时状态”，打断宿醉！
        # ========================================================
        state_input = {
            "messages": [HumanMessage(content=clean_input)],
            "pdf_file_paths": pdf_paths,
            "step_count": 0,                        # 步数归零，给足充足的思考机会
            "candidate_papers": "CLEAR",            # 清空本轮候选论文，避免旧工具结果污染
            "selected_papers": "CLEAR",             # 清空最终证据池，等待本轮重新确认
            "research_goal": "解析用户最新意图",       # 清空上一题的目标
            "collected_evidence": "暂无",           # 清空上一题的证据
            "pending_questions": "未知"             # 清空上一题的疑问
        }

        print("\n[AGENT] Thinking... (如果涉及查文献或深度解析，可能需要较长时间)", flush=True)

        try:
            result = await agent_app.ainvoke(state_input, config=config)
            all_messages = result.get("messages", [])

            print("\n" + "=" * 30 + " 最终结论 " + "=" * 30)
            if all_messages and all_messages[-1].content.strip():
                print(f"\n[COPILOT]:\n{all_messages[-1].content}")
            else:
                print(
                    "\n[COPILOT]: 经过多轮检索，未能获取有效信息。这可能是由于您查询的领域属于前沿空白，或者大模型推理中断。请尝试缩减关键词。")

        except Exception as e:
            print(f"\n[ERROR] 运行中断: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_chat())
