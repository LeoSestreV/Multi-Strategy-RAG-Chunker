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
    """Divides text by character count, breaking at sentence boundaries
    (last strong punctuation . ! ? before the size limit) and applying
    overlap to preserve context between consecutive chunks."""

    OVERLAP_RATIO = 0.1  # 10% of chunk_size used as overlap

    def __init__(self, config: FixedSizeChunkingConfig) -> None:
        self.config = config

    def _find_sentence_break(self, text: str, max_len: int) -> int:
        """Find the best split position at a sentence boundary before max_len."""
        if len(text) <= max_len:
            return len(text)

        # Search backwards from max_len for the last sentence-ending punctuation
        search_region = text[:max_len]
        best_pos = -1
        for punct in '.!?':
            pos = search_region.rfind(punct)
            if pos > best_pos:
                best_pos = pos

        # If found a sentence boundary, split right after the punctuation
        if best_pos > 0:
            return best_pos + 1

        # Fallback: split at last space to avoid breaking words
        space_pos = text.rfind(" ", 0, max_len)
        if space_pos > 0:
            return space_pos

        return max_len

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
        overlap_size = int(self.config.chunk_size * self.OVERLAP_RATIO)

        # Split text into pieces at sentence boundaries with overlap
        raw_chunks: list[tuple[str, int, int]] = []
        pos = 0

        while pos < len(full_text):
            remaining = full_text[pos:]
            if len(remaining) <= self.config.chunk_size:
                raw_chunks.append((remaining.strip(), pos, pos + len(remaining)))
                break

            split_at = self._find_sentence_break(remaining, self.config.chunk_size)
            piece = remaining[:split_at].strip()
            if piece:
                raw_chunks.append((piece, pos, pos + split_at))

            # Move forward, but step back by overlap_size for context continuity
            next_pos = pos + split_at
            if overlap_size > 0 and next_pos < len(full_text):
                next_pos = max(pos + 1, next_pos - overlap_size)
                # Adjust to avoid splitting mid-word
                space = full_text.rfind(" ", pos + 1, next_pos + 1)
                if space > pos:
                    next_pos = space + 1
            pos = next_pos

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
