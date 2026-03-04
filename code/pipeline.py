from __future__ import annotations

from llama_index.core import VectorStoreIndex

from .config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_LLM_MODEL,
    DASHSCOPE_SMALL_LLM_MODEL,
    ENABLE_QUERY_REWRITE,
    ENABLE_RERANK,
    DEFAULT_TOPK,
    IS_LOCAL_LLM,
)
from .embedding import BaiLianEmbedding
from .llm import generate_answer, rewrite_question
from .rerank import EmbeddingReranker
from .retrieval import format_context, merge_nodes, retrieve_nodes


def answer_question(
    *,
    index: VectorStoreIndex | None,
    question: str,
    topk: int = DEFAULT_TOPK,
    enable_query_rewrite: bool = ENABLE_QUERY_REWRITE,
    enable_rerank: bool = ENABLE_RERANK,
    llm_model: str = DASHSCOPE_LLM_MODEL,
    small_llm_model: str = DASHSCOPE_SMALL_LLM_MODEL,
) -> tuple[str, list]:
    if not DASHSCOPE_API_KEY and not IS_LOCAL_LLM:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY）")

    if index is None:
        prompt = (
            "你是一个严谨的助手。\n"
            "请尽量给出直接答案；如果不确定请说明依据不足。\n\n"
            f"【问题】\n{question}\n\n"
            "【回答】\n"
        )
        answer = generate_answer(api_key=DASHSCOPE_API_KEY, prompt=prompt, model=llm_model)
        return answer, []

    embed_model = index._embed_model

    rewritten = question
    if enable_query_rewrite:
        try:
            rewritten = rewrite_question(
                api_key=DASHSCOPE_API_KEY, question=question, model=small_llm_model
            )
        except Exception:
            rewritten = question

    candidate_k = max(topk * 2, topk)
    retriever = index.as_retriever(similarity_top_k=candidate_k)

    nodes_rewrite = retrieve_nodes(retriever=retriever, question=rewritten)
    nodes_raw = retrieve_nodes(retriever=retriever, question=question)
    nodes = merge_nodes(primary=nodes_raw, secondary=nodes_rewrite)

    if not nodes:
        prompt = (
            "你是一个严谨的助手。\n"
            "请尽量给出直接答案；如果不确定请说明依据不足。\n\n"
            f"【问题】\n{question}\n\n"
            "【回答】\n"
        )
        answer = generate_answer(api_key=DASHSCOPE_API_KEY, prompt=prompt, model=llm_model)
        return answer, []

    if enable_rerank and isinstance(embed_model, BaiLianEmbedding) and nodes:
        reranker = EmbeddingReranker(embed_model=embed_model)
        nodes = reranker.rerank(query=question, nodes=nodes)

    if nodes:
        nodes = nodes[:topk]

    context = format_context(nodes)
    prompt = (
        "你是一个严谨的助手。请只基于【上下文】回答【问题】。\n"
        "如果上下文不足以回答，请明确说“不确定/资料不足”。\n\n"
        f"【上下文】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        "【回答】\n"
    )

    answer = generate_answer(api_key=DASHSCOPE_API_KEY, prompt=prompt, model=llm_model)
    return answer, nodes
