import argparse
import asyncio

from .config import (
    DEFAULT_COLLECTION,
    DEFAULT_TOPK,
    MILVUS_DB_PATH,
    DASHSCOPE_API_KEY,
    EMBEDDING_DIM,
    IS_LOCAL_LLM,
)
from .embedding import BaiLianEmbedding
from .pipeline import answer_question
from .vector_store import build_index, build_vector_store


async def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus-lite + LlamaIndex 向量检索 + RAG 问答（模块化版）")
    parser.add_argument("--question", "-q", required=True, help="你的问题")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="召回条数")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection 名")
    args = parser.parse_args()

    if not DASHSCOPE_API_KEY and not IS_LOCAL_LLM:
        raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY）")

    vector_store = build_vector_store(db_path=MILVUS_DB_PATH, collection_name=args.collection, dim=EMBEDDING_DIM)
    embed_model = BaiLianEmbedding(api_key=DASHSCOPE_API_KEY)

    index = build_index(vector_store=vector_store, embed_model=embed_model)

    answer, nodes = answer_question(index=index, question=args.question, topk=args.topk)

    if not nodes:
        print("未召回到任何内容。你可以检查 collection 是否存在、dim 是否匹配、或先完成向量入库流程。")
        return 1

    print("\n=== TopK 检索结果（作为上下文）===")
    for i, node in enumerate(nodes, start=1):
        text = (node.get_content() or "").strip()
        text_preview = text[:800]
        print(f"\n[{i}] score={getattr(node, 'score', None)}")
        print(text_preview)

    print("\n=== RAG 回答 ===")
    print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
