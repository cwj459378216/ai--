from typing import Iterable

from llama_index.core.schema import NodeWithScore


def retrieve_nodes(*, retriever, question: str) -> list[NodeWithScore]:
    return list(retriever.retrieve(question))


def format_context(nodes: Iterable[NodeWithScore]) -> str:
    blocks: list[str] = []
    for i, node in enumerate(nodes, start=1):
        text = (node.get_content() or "").strip()
        blocks.append(f"[{i}] {text}")
    return "\n\n".join(blocks)


def merge_nodes(*, primary: Iterable[NodeWithScore], secondary: Iterable[NodeWithScore]) -> list[NodeWithScore]:
    merged: list[NodeWithScore] = []
    seen: set[str] = set()

    def _key(node: NodeWithScore) -> str:
        inner = getattr(node, "node", None)
        for attr in ("node_id", "id_", "doc_id"):
            if inner is not None and hasattr(inner, attr):
                return str(getattr(inner, attr))
        return (node.get_content() or "")[:200]

    for item in list(primary) + list(secondary):
        key = _key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return merged
