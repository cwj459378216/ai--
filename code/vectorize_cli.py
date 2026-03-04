import argparse
from pathlib import Path

from .config import DEFAULT_COLLECTION, DASHSCOPE_API_KEY, DOC1_DIR, IS_LOCAL_EMBEDDING
from .vectorize import vectorize_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="文档向量化（模块化版）")
    parser.add_argument("--input-dir", default=str(DOC1_DIR), help="待向量化目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection 名")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小")
    parser.add_argument("--overlap", type=int, default=50, help="分块重叠")
    parser.add_argument("--no-move", action="store_true", help="不移动文件到 doc2")
    parser.add_argument("--no-auto-fix-dim", action="store_true", help="禁用维度不一致自动修复")
    args = parser.parse_args()

    if not DASHSCOPE_API_KEY and not IS_LOCAL_EMBEDDING:
        raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY）")

    count = vectorize_dir(
        input_dir=Path(args.input_dir),
        collection_name=args.collection,
        api_key=DASHSCOPE_API_KEY or "",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        move_to_doc2=not args.no_move,
        auto_fix_dim=not args.no_auto_fix_dim,
    )

    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
