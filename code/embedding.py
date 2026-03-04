import asyncio
from typing import Any

import requests
from llama_index.core.base.embeddings.base import BaseEmbedding

from .config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


class BaiLianEmbedding(BaseEmbedding):
    """Embedding：支持 DashScope 与本地 OpenAI 兼容接口。"""

    model_name: str = "bailian"
    api_key: str
    endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    model: str = "text-embedding-v2"

    def _get_embedding_with_type(self, text: str, text_type: str):
        if EMBEDDING_BASE_URL:
            return self._get_embedding_openai(text)
        return self._get_embedding_dashscope(text, text_type)

    def _get_embedding_openai(self, text: str):
        base = EMBEDDING_BASE_URL.rstrip("/")
        is_ollama_api = base.endswith("/api")
        url = f"{base}/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

        if is_ollama_api:
            data: dict[str, Any] = {
                "model": EMBEDDING_MODEL,
                "prompt": text,
            }
        else:
            data = {
                "model": EMBEDDING_MODEL,
                "input": text,
            }

        resp = requests.post(url, json=data, headers=headers, timeout=60)
        resp.raise_for_status()

        try:
            payload = resp.json()
        except ValueError as exc:
            body_preview = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"Embedding API returned non-JSON response (status={resp.status_code}): {body_preview}"
            ) from exc

        try:
            if "data" in payload:
                return payload["data"][0]["embedding"]
            if "embedding" in payload:
                return payload["embedding"]
            raise KeyError("data/embedding not found")
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected embedding response shape: keys={list(payload.keys())} "
                f"payload_preview={str(payload)[:500]}"
            ) from exc

    def _get_embedding_dashscope(self, text: str, text_type: str):
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
