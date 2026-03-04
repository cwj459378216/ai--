import argparse

from .config import DEFAULT_COLLECTION
from .vectorize import reset_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="清空向量库 Collection")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection 名")
    args = parser.parse_args()

    cleared = reset_vector_store(collection_name=args.collection, delete_db=True, move_doc2=True)
    if cleared:
        print(f"✅ 已清空 collection: {args.collection}，并重置本地 DB 与 doc 目录")
        return 0

    print(f"⚠️ collection 不存在：{args.collection}，已重置本地 DB 与 doc 目录")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
