import asyncio
import os
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient

from .embedding import BaiLianEmbedding


def build_vector_store(*, db_path: Path, collection_name: str, dim: int) -> MilvusVectorStore:
    _ensure_db_ready(db_path)
    _ensure_collection_dim(db_path=db_path, collection_name=collection_name, dim=dim)

    def _create() -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=str(db_path.resolve()),
            collection_name=collection_name,
            dim=dim,
            overwrite=False,
        )

    try:
        asyncio.get_running_loop()
        return _create()
    except RuntimeError:
        return asyncio.run(_create_async(_create))


async def _create_async(factory):
    return factory()


def clear_collection(*, db_path: Path, collection_name: str) -> bool:
    client = MilvusClient(uri=str(db_path.resolve()))
    collections = client.list_collections()
    if collection_name not in collections:
        return False
    client.drop_collection(collection_name)
    return True


def _ensure_collection_dim(*, db_path: Path, collection_name: str, dim: int) -> None:
    try:
        client = MilvusClient(uri=str(db_path.resolve()))
    except Exception:
        return

    try:
        if collection_name not in client.list_collections():
            return
        existing_dim = _get_collection_dim(client, collection_name)
        if existing_dim is None or int(existing_dim) == int(dim):
            return
        client.drop_collection(collection_name)
    except Exception:
        return


def _get_collection_dim(client: MilvusClient, collection_name: str) -> int | None:
    try:
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


def has_collection(*, db_path: Path, collection_name: str) -> bool:
    try:
        client = MilvusClient(uri=str(db_path.resolve()))
        return collection_name in client.list_collections()
    except Exception:
        return False


def _ensure_db_ready(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(db_path.parent, 0o755)
    except Exception:
        pass

    if not os.access(db_path.parent, os.W_OK):
        raise RuntimeError(f"Milvus DB directory is not writable: {db_path.parent}")

    if db_path.exists():
        try:
            if not os.access(db_path, os.W_OK):
                os.chmod(db_path, 0o644)
            if not os.access(db_path, os.W_OK):
                for item in db_path.parent.glob(f"{db_path.name}*"):
                    try:
                        item.unlink()
                    except Exception:
                        pass
        except Exception:
            pass


def build_index(*, vector_store: MilvusVectorStore, embed_model: BaiLianEmbedding) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
