from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text_by_chars(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[TextChunk]:
    """Chunk text by character count with optional overlap.

    Returns chunks carrying original [start, end) character offsets.
    """

    if text is None:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # Ensure progress even with extreme inputs
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    length = len(normalized)
    if length == 0:
        return []

    step = max(1, chunk_size - overlap)

    # Prefer splitting on natural boundaries near the end of a chunk.
    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", "；", ",", "，", " ", "\t"]
    soft_min = int(chunk_size * 0.6)

    chunks: list[TextChunk] = []
    start = 0

    while start < length:
        tentative_end = min(start + chunk_size, length)
        end = tentative_end

        # Try to find a good split point, but don't create tiny chunks.
        window = normalized[start:tentative_end]
        if len(window) > soft_min:
            search_from = soft_min
            best = -1
            for sep in separators:
                idx = window.rfind(sep, search_from)
                if idx > best:
                    best = idx
                    best_sep = sep
            if best != -1:
                end = start + best + len(best_sep)

        piece = normalized[start:end]
        if piece.strip():
            chunks.append(TextChunk(text=piece, start=start, end=end))

        if end >= length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks
