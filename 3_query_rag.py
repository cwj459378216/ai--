import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import requests
from pymilvus import MilvusClient

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding


class BaiLianEmbedding(BaseEmbedding):
    """DashScope/百炼 Embedding：用于向量检索（RAG 的检索阶段）。
    RAG 是 Retrieval-Augmented Generation 的缩写，中文一般叫：检索增强生成

    说明：这里用的是 DashScope 的 embedding REST API 形态。
    如果你账号/区域/模型名不同，改 `endpoint` / `model` 即可。
    """

    model_name: str = "bailian"
    api_key: str
    endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    model: str = "text-embedding-v2"

    def _get_embedding_with_type(self, text: str, text_type: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": {
                "texts": [text],
                "text_type": text_type,
            },
        }
        resp = requests.post(self.endpoint, json=data, headers=headers, timeout=60)
        resp.raise_for_status()

        try:
            payload = resp.json()
        except ValueError as exc:
            body_preview = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"Embedding API returned non-JSON response (status={resp.status_code}): {body_preview}"
            ) from exc

        try:
            return payload["output"]["embeddings"][0]["embedding"]
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected embedding response shape: keys={list(payload.keys())} "
                f"payload_preview={str(payload)[:500]}"
            ) from exc

    def _get_text_embedding(self, text: str):
        return self._get_embedding_with_type(text, "document")

    def _get_query_embedding(self, query: str):
        return self._get_embedding_with_type(query, "query")

    async def _aget_text_embedding(self, text: str):
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str):
        return await self._aget_text_embedding(query)


def dashscope_generate_answer(*, api_key: str, prompt: str, model: str) -> str:
    """用 DashScope/百炼做 RAG 的“生成”调用。

    说明：优先走 DashScope 的 OpenAI 兼容 Chat Completions（messages 形态），
    对 `qwen-plus` 这类 chat 模型更稳定，能避免 legacy prompt 接口的 400。
    """

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的助手。请只基于给定上下文回答问题；如果资料不足请直说。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(url, json=data, headers=headers, timeout=120)
    if resp.status_code >= 400:
        body_preview = (resp.text or "").strip()[:1500]
        raise RuntimeError(
            f"LLM API error status={resp.status_code}: {body_preview}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        body_preview = (resp.text or "").strip()[:800]
        raise RuntimeError(
            f"LLM API returned non-JSON response (status={resp.status_code}): {body_preview}"
        ) from exc

    try:
        return payload["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected LLM response shape: keys={list(payload.keys())} "
            f"payload_preview={str(payload)[:800]}"
        ) from exc


async def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus-lite + LlamaIndex 向量检索 + RAG 问答示例")
    parser.add_argument("--question", "-q", required=True, help="你的问题")
    parser.add_argument("--topk", type=int, default=3, help="召回条数")
    parser.add_argument(
        "--llm-model",
        default=os.getenv("DASHSCOPE_LLM_MODEL", "qwen-plus"),
        help="DashScope 生成模型名（默认读 DASHSCOPE_LLM_MODEL，否则 qwen-plus）",
    )
    parser.add_argument(
        "--collection",
        default="rag_demo",
        help="Milvus collection 名（默认 rag_demo）",
    )
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise SystemExit(
            "请先设置环境变量 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY），再运行脚本。"
        )

    llm_model = args.llm_model

    milvus_path = Path("./milvus_data/milvus.db")
    milvus_path.parent.mkdir(parents=True, exist_ok=True)

    # 连接 milvus-lite
    milvus_client = MilvusClient(uri=str(milvus_path.resolve()))

    # 绑定到已写入的 collection
    vector_store = MilvusVectorStore(
        client=milvus_client,
        collection_name=args.collection,
        dim=1536,
        overwrite=False,
    )

    embed_model = BaiLianEmbedding(api_key=api_key)

    # 从已有向量库恢复索引
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 1) 向量检索（topK）
    retriever = index.as_retriever(similarity_top_k=args.topk)
    nodes = retriever.retrieve(args.question)

    if not nodes:
        print("未召回到任何内容。你可以检查 collection 是否存在、dim 是否匹配、或先运行 2_vector_to_milvus.py")
        return 1

    print("\n=== TopK 检索结果（作为上下文）===")
    context_blocks: list[str] = []
    for i, node in enumerate(nodes, start=1):
        text = (node.get_content() or "").strip()
        text_preview = text[:800]
        print(f"\n[{i}] score={getattr(node, 'score', None)}")
        print(text_preview)
        context_blocks.append(f"[{i}] {text}")

    # 2) RAG：把检索到的上下文塞进提示词，再调用 LLM 生成答案
    context = "\n\n".join(context_blocks)
    prompt = (
        "你是一个严谨的助手。请只基于【上下文】回答【问题】。\n"
        "如果上下文不足以回答，请明确说“不确定/资料不足”。\n\n"
        f"【上下文】\n{context}\n\n"
        f"【问题】\n{args.question}\n\n"
        "【回答】\n"
    )

    try:
        answer = dashscope_generate_answer(api_key=api_key, prompt=prompt, model=llm_model)
    except RuntimeError as exc:
        msg = str(exc)
        if "model_not_found" in msg or "does not exist" in msg:
            raise RuntimeError(
                "DashScope 模型不可用：你当前配置的 DASHSCOPE_LLM_MODEL 可能是错误的或无权限。\n"
                "建议改成你账号有权限的模型名，例如：qwen-plus / qwen-max / qwen-turbo。\n"
                "你也可以临时用参数覆盖：--llm-model qwen-plus"
            ) from exc
        raise

    print("\n=== RAG 回答 ===")
    print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
