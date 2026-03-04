import os
import sys
from pathlib import Path

import streamlit as st

try:
    from .config import (
        DEFAULT_COLLECTION,
        DEFAULT_TOPK,
        DASHSCOPE_API_KEY,
        DASHSCOPE_LLM_MODEL,
        DASHSCOPE_SMALL_LLM_MODEL,
        DOC1_DIR,
        DOC2_DIR,
        ENABLE_QUERY_REWRITE,
        ENABLE_RERANK,
        MILVUS_DB_PATH,
        EMBEDDING_DIM,
        IS_LOCAL_LLM,
        IS_LOCAL_EMBEDDING,
    )
    from .embedding import BaiLianEmbedding
    from .pipeline import answer_question
    from .vector_store import build_index, build_vector_store, has_collection
    from .vectorize import reset_vector_store, vectorize_dir
except ImportError:
    base_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base_dir))
    from code.config import (
        DEFAULT_COLLECTION,
        DEFAULT_TOPK,
        DASHSCOPE_API_KEY,
        DASHSCOPE_LLM_MODEL,
        DASHSCOPE_SMALL_LLM_MODEL,
        DOC1_DIR,
        DOC2_DIR,
        ENABLE_QUERY_REWRITE,
        ENABLE_RERANK,
        MILVUS_DB_PATH,
        EMBEDDING_DIM,
        IS_LOCAL_LLM,
        IS_LOCAL_EMBEDDING,
    )
    from code.embedding import BaiLianEmbedding
    from code.pipeline import answer_question
    from code.vector_store import build_index, build_vector_store, has_collection
    from code.vectorize import reset_vector_store, vectorize_dir


st.set_page_config(page_title="RAG GUI", layout="wide")

st.title("RAG 图形化界面")

with st.sidebar:
    st.header("基础配置")
    env_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
    env_llm_model = os.getenv("DASHSCOPE_LLM_MODEL")
    env_small_llm_model = os.getenv("DASHSCOPE_SMALL_LLM_MODEL")
    api_key = st.text_input(
        "DASHSCOPE_API_KEY",
        value=env_api_key or DASHSCOPE_API_KEY or "",
        type="password",
        help="也可在环境变量中设置 DASHSCOPE_API_KEY 或 BAILIAN_API_KEY",
    )
    collection = st.text_input("Collection", value=DEFAULT_COLLECTION)
    topk = st.number_input("TopK", min_value=1, max_value=20, value=DEFAULT_TOPK)
    llm_model = st.text_input("生成模型", value=env_llm_model or DASHSCOPE_LLM_MODEL)
    small_llm_model = st.text_input("小模型（改写）", value=env_small_llm_model or DASHSCOPE_SMALL_LLM_MODEL)
    enable_rewrite = st.checkbox("启用问题改写", value=ENABLE_QUERY_REWRITE)
    enable_rerank = st.checkbox("启用重排序", value=ENABLE_RERANK)

st.subheader("向量化")
col1, col2 = st.columns(2)
with col1:
    st.write(f"doc1: {DOC1_DIR}")
    doc1_files = [p.name for p in DOC1_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    st.write(doc1_files or ["(空)"])
with col2:
    st.write(f"doc2: {DOC2_DIR}")
    doc2_files = [p.name for p in DOC2_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    st.write(doc2_files or ["(空)"])

chunk_size = st.number_input("分块大小", min_value=100, max_value=2000, value=500)
overlap = st.number_input("分块重叠", min_value=0, max_value=500, value=50)
move_to_doc2 = st.checkbox("向量化后移动到 doc2", value=True)

if st.button("开始向量化"):
    if not api_key and not IS_LOCAL_EMBEDDING:
        st.error("请先设置 DASHSCOPE_API_KEY")
    else:
        try:
            count = vectorize_dir(
                input_dir=Path(DOC1_DIR),
                collection_name=collection,
                api_key=api_key or "",
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                move_to_doc2=move_to_doc2,
            )
            if count > 0:
                st.success(f"向量化完成，写入 {count} 个 chunk，DB: {MILVUS_DB_PATH}")
            else:
                st.warning("未发现可向量化的文件")
        except Exception as exc:
            st.error(f"向量化失败：{exc}")

st.markdown("### 清空向量库")
confirm_clear = st.checkbox("确认清空当前 collection", value=False)
if st.button("清空向量库"):
    if not confirm_clear:
        st.warning("请先勾选确认清空")
    else:
        try:
            cleared = reset_vector_store(collection_name=collection, delete_db=True, move_doc2=True)
            if cleared:
                st.success(f"已清空 collection: {collection}，并重置本地 DB 与 doc 目录")
            else:
                st.warning("collection 不存在，但已重置本地 DB 与 doc 目录")
        except Exception as exc:
            st.error(f"清空失败：{exc}")

st.divider()

st.subheader("检索问答")
question = st.text_area("你的问题", height=120)

if st.button("生成答案"):
    if not api_key and not IS_LOCAL_LLM:
        st.error("请先设置 DASHSCOPE_API_KEY")
    elif not question.strip():
        st.warning("请输入问题")
    else:
        try:
            if api_key:
                os.environ["DASHSCOPE_API_KEY"] = api_key

            if has_collection(db_path=MILVUS_DB_PATH, collection_name=collection):
                vector_store = build_vector_store(
                    db_path=MILVUS_DB_PATH,
                    collection_name=collection,
                    dim=EMBEDDING_DIM,
                )
                embed_model = BaiLianEmbedding(api_key=api_key or "")
                index = build_index(vector_store=vector_store, embed_model=embed_model)

                answer, nodes = answer_question(
                    index=index,
                    question=question,
                    topk=int(topk),
                    enable_query_rewrite=enable_rewrite,
                    enable_rerank=enable_rerank,
                    llm_model=llm_model,
                    small_llm_model=small_llm_model,
                )
            else:
                st.info("向量库为空或 collection 不存在，将直接调用模型回答。")
                answer, nodes = answer_question(
                    index=None,
                    question=question,
                    topk=int(topk),
                    enable_query_rewrite=enable_rewrite,
                    enable_rerank=enable_rerank,
                    llm_model=llm_model,
                    small_llm_model=small_llm_model,
                )

            st.markdown("### 回答")
            st.write(answer)

            if nodes:
                st.markdown("### TopK 上下文")
                for i, node in enumerate(nodes, start=1):
                    text = (node.get_content() or "").strip()
                    st.markdown(f"**[{i}] score={getattr(node, 'score', None)}**")
                    st.write(text[:1200])
        except Exception as exc:
            st.error(f"生成失败：{exc}")
