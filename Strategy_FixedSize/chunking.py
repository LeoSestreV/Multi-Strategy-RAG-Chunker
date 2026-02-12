from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import FixedSizeChunkingConfig
from preprocessing import Sentence


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    char_count: int = 0


class FixedSizeChunker:
    """Divides text mathematically by character count, always breaking
    at the last space boundary to avoid splitting words."""

    def __init__(self, config: FixedSizeChunkingConfig) -> None:
        self.config = config

    def _split_at_space(self, text: str, max_len: int) -> tuple[str, str]:
        """Split text at the last space before max_len."""
        if len(text) <= max_len:
            return text, ""

        split_pos = text.rfind(" ", 0, max_len)
        if split_pos == -1:
            split_pos = max_len

        return text[:split_pos].rstrip(), text[split_pos:].lstrip()

    def _compute_coherence(self, embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return self.config.default_coherence_score
        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            sims.append(sim)
        return float(np.mean(sims))

    def chunk(self, sentences: list[Sentence],
              embeddings: list[np.ndarray]) -> list[Chunk]:
        if not sentences:
            return []

        full_text = " ".join(s.text for s in sentences)

        # Split text into fixed-size pieces at word boundaries
        raw_chunks: list[tuple[str, int, int]] = []
        remaining = full_text
        offset = 0

        while remaining:
            piece, remaining = self._split_at_space(remaining, self.config.chunk_size)
            if piece:
                raw_chunks.append((piece, offset, offset + len(piece)))
                offset += len(piece) + 1

        # Merge small trailing chunk
        if len(raw_chunks) > 1:
            last_text, last_start, last_end = raw_chunks[-1]
            if len(last_text) < self.config.min_chunk_size:
                prev_text, prev_start, prev_end = raw_chunks[-2]
                raw_chunks[-2] = (prev_text + " " + last_text, prev_start, last_end)
                raw_chunks.pop()

        # Map sentences to chunks and compute coherence
        emb_map: dict[str, np.ndarray] = {
            s.text: e for s, e in zip(sentences, embeddings) if e is not None
        }
        chunks: list[Chunk] = []

        for chunk_text, start, end in raw_chunks:
            # Find sentences contained in this chunk
            chunk_sents: list[str] = []
            chunk_embs: list[np.ndarray] = []
            for sent in sentences:
                if sent.text in chunk_text:
                    chunk_sents.append(sent.text)
                    if sent.text in emb_map:
                        chunk_embs.append(emb_map[sent.text])

            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                sentences=chunk_sents,
                coherence_score=self._compute_coherence(chunk_embs) if chunk_embs else 0.0,
                char_count=len(chunk_text)
            ))

        return chunks
