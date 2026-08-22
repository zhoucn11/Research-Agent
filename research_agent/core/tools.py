# tools.py
import logging
import os
import warnings

from dotenv import load_dotenv

from research_agent.core.app_config import apply_code_defaults


load_dotenv()
apply_code_defaults()
os.environ["NO_PROXY"] = "aliyuncs.com,localhost,127.0.0.1"

from research_agent.tools.agent_tools import (
    LocalRetrievalInput,
    WebSearchInput,
    trigger_local_retrieval,
    trigger_pdf_upload,
    trigger_web_search,
)
from research_agent.agents.assistant_agent import assistant_node
from research_agent.agents.rag_agent import rag_map_node
from research_agent.agents.search_agent import search_map_node
from research_agent.agents.synthesis_agent import reviewer_node, synthesizer_node


warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


__all__ = [
    "WebSearchInput",
    "LocalRetrievalInput",
    "trigger_web_search",
    "trigger_pdf_upload",
    "trigger_local_retrieval",
    "assistant_node",
    "search_map_node",
    "rag_map_node",
    "synthesizer_node",
    "reviewer_node",
]
