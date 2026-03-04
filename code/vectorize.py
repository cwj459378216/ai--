from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from pymilvus import MilvusClient

from .config import DOC1_DIR, DOC2_DIR, MILVUS_DB_PATH, EMBEDDING_DIM
from .embedding import BaiLianEmbedding
from .loader import load_documents_from_dir, list_files
from .vector_store import build_vector_store, clear_collection, has_collection


def vectorize_dir(
    *,
    input_dir: Path = DOC1_DIR,
    collection_name: str,
    api_key: str,
    chunk_size: int = 500,
    overlap: int = 50,
    move_to_doc2: bool = True,
    auto_fix_dim: bool = True,
) -> int:
    documents = load_documents_from_dir(
        input_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not documents:
        print("未发现可向量化的文件。")
        return 0

    embed_model = BaiLianEmbedding(api_key=api_key)
    dim = _detect_embedding_dim(embed_model, documents, fallback=EMBEDDING_DIM)
    if dim != EMBEDDING_DIM:
        print(f"⚠️ 检测到 embedding 维度为 {dim}，将覆盖配置中的 EMBEDDING_DIM={EMBEDDING_DIM}")

    existing_dim = _get_existing_collection_dim(db_path=MILVUS_DB_PATH, collection_name=collection_name)
    if existing_dim is not None and int(existing_dim) != int(dim):
        if auto_fix_dim:
            print(f"⚠️ 发现已有 collection 维度={existing_dim}，与当前 {dim} 不一致，已重置本地 DB")
            _reset_local_db(MILVUS_DB_PATH)
        else:
            raise RuntimeError(
                f"collection 维度({existing_dim})与当前 embedding 维度({dim})不一致，请先清空向量库或启用 auto_fix_dim"
            )

    vector_store = build_vector_store(db_path=MILVUS_DB_PATH, collection_name=collection_name, dim=dim)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        _ = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    except Exception as exc:
        msg = str(exc)
        if "readonly database" in msg or "opened by another program" in msg:
            print("⚠️ 检测到只读或占用的本地 DB，已自动重置并重试")
            _reset_local_db(MILVUS_DB_PATH)
            vector_store = build_vector_store(
                db_path=MILVUS_DB_PATH,
                collection_name=collection_name,
                dim=dim,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            _ = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
            )
        else:
            raise

    if move_to_doc2:
        DOC2_DIR.mkdir(parents=True, exist_ok=True)
        for file in list_files(input_dir):
            target = DOC2_DIR / file.name
            file.replace(target)

    print("✅ 文档向量化完成并写入 Milvus-lite")
    return len(documents)


def _detect_embedding_dim(embed_model: BaiLianEmbedding, documents, fallback: int) -> int:
    try:
        sample = next((d for d in documents if (d.get_content() or "").strip()), None)
        if sample is None:
            return fallback
        embedding = embed_model.get_text_embedding(sample.get_content())
        if embedding:
            return len(embedding)
    except Exception:
        return fallback
    return fallback


def _get_existing_collection_dim(*, db_path: Path, collection_name: str) -> int | None:
    try:
        client = MilvusClient(uri=str(db_path.resolve()))
    except Exception:
        return None
    try:
        if collection_name not in client.list_collections():
            return None
        desc = client.describe_collection(collection_name)
    except Exception:
        return None

    fields = desc.get("fields")
    if not fields:
        fields = (desc.get("schema") or {}).get("fields") or []

    for field in fields:
        field_type = str(field.get("type", "")).upper()
        if "VECTOR" in field_type:
            params = field.get("params") or field.get("type_params") or {}
            dim_val = params.get("dim") or params.get("dimension")
            if dim_val is not None:
                try:
                    return int(dim_val)
                except Exception:
                    return None
    return None


def _reset_local_db(db_path: Path) -> None:
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception:
        pass
    try:
        lock_path = db_path.parent / f".{db_path.name}.lock"
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def reset_vector_store(*, collection_name: str, delete_db: bool = True, move_doc2: bool = True) -> bool:
    cleared = clear_collection(db_path=MILVUS_DB_PATH, collection_name=collection_name)

    if delete_db and MILVUS_DB_PATH.exists():
        try:
            MILVUS_DB_PATH.unlink()
        except Exception:
            pass
        try:
            lock_path = MILVUS_DB_PATH.parent / f".{MILVUS_DB_PATH.name}.lock"
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
        try:
            os.chmod(MILVUS_DB_PATH.parent, 0o755)
        except Exception:
            pass

    if move_doc2:
        DOC1_DIR.mkdir(parents=True, exist_ok=True)
        DOC2_DIR.mkdir(parents=True, exist_ok=True)
        for file in list_files(DOC2_DIR):
            target = DOC1_DIR / file.name
            file.replace(target)

    return cleared
