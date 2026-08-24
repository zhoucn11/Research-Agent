# local_models.py
import os
import time

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

from research_agent.core.runtime_events import runtime_print as print


_GLOBAL_EMBEDDINGS = None
_GLOBAL_RERANKER = None


def get_embeddings():
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        print("🚀 [首次加载] 正在将 Embedding 模型载入显存...")
        load_start = time.time()
        try:
            _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=os.environ.get(
                    "EMBEDDING_MODEL_PATH",
                    "BAAI/bge-large-zh-v1.5",
                ),
                model_kwargs={"device": "cuda"},
                encode_kwargs={"batch_size": 1, "normalize_embeddings": True},
            )
            print(f"✅ Embedding 模型加载成功 (耗时: {time.time() - load_start:.2f}s)。")
        except Exception as e:
            print(f"❌ Embedding 模型加载失败: {e}")
            raise e
    return _GLOBAL_EMBEDDINGS


def get_reranker():
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        print("🚀 [首次加载] 正在将 Reranker 重排模型载入显存...")
        load_start = time.time()
        try:
            model = HuggingFaceCrossEncoder(
                model_name=os.environ.get(
                    "RERANKER_MODEL_PATH",
                    "BAAI/bge-reranker-v2-m3",
                ),
                model_kwargs={"device": os.environ.get("RERANKER_DEVICE", "cpu")},
            )
            _GLOBAL_RERANKER = CrossEncoderReranker(model=model, top_n=10)
            print(f"✅ Reranker 模型加载成功 (耗时: {time.time() - load_start:.2f}s)。")
        except Exception as e:
            print(f"❌ Reranker 模型加载失败: {e}")
            raise e
    return _GLOBAL_RERANKER
