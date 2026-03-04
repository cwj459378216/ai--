from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from llama_index.core.schema import NodeWithScore

from .embedding import BaiLianEmbedding


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class EmbeddingReranker:
    """使用同一 embedding 模型进行轻量重排序。"""

    embed_model: BaiLianEmbedding

    def rerank(self, *, query: str, nodes: Iterable[NodeWithScore]) -> list[NodeWithScore]:
        query_vec = self.embed_model._get_query_embedding(query)
        rescored: list[NodeWithScore] = []
        for node in nodes:
            text = (node.get_content() or "").strip()
            if not text:
                continue
            doc_vec = self.embed_model._get_text_embedding(text)
            node.score = _dot(query_vec, doc_vec)
            rescored.append(node)
        rescored.sort(key=lambda n: n.score if n.score is not None else -1.0, reverse=True)
        return rescored
