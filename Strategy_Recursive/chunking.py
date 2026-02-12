from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import RecursiveChunkingConfig
from preprocessing import Sentence


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    split_depth: int = 0


class RecursiveChunker:
    """Splits text hierarchically using a cascade of separators:
    paragraphs -> sentences -> words, until each chunk fits the target size."""

    def __init__(self, config: RecursiveChunkingConfig) -> None:
        self.config = config

    def _recursive_split(self, text: str, depth: int = 0) -> list[tuple[str, int]]:
        if len(text) <= self.config.target_chunk_size:
            return [(text, depth)]

        if depth >= len(self.config.separators):
            return [(text, depth)]

        sep = self.config.separators[depth]
        parts = text.split(sep)

        if len(parts) == 1:
            return self._recursive_split(text, depth + 1)

        results: list[tuple[str, int]] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.config.target_chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) <= self.config.target_chunk_size:
                        results.append((current, depth))
                    else:
                        results.extend(self._recursive_split(current, depth + 1))
                current = part

        if current:
            if len(current) <= self.config.target_chunk_size:
                results.append((current, depth))
            else:
                results.extend(self._recursive_split(current, depth + 1))

        return results

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
        splits = self._recursive_split(full_text)

        # Merge small fragments
        merged: list[tuple[str, int]] = []
        buffer_text = ""
        buffer_depth = 0
        for text, depth in splits:
            if buffer_text:
                candidate = buffer_text + " " + text
            else:
                candidate = text
                buffer_depth = depth

            if len(candidate) < self.config.min_chunk_size:
                buffer_text = candidate
            else:
                if buffer_text and len(buffer_text) >= self.config.min_chunk_size:
                    merged.append((buffer_text, buffer_depth))
                    buffer_text = text
                    buffer_depth = depth
                else:
                    merged.append((candidate, buffer_depth))
                    buffer_text = ""
        if buffer_text:
            if merged:
                prev_text, prev_depth = merged[-1]
                merged[-1] = (prev_text + " " + buffer_text, prev_depth)
            else:
                merged.append((buffer_text, buffer_depth))

        # Build chunk objects with position tracking
        all_embeddings: dict[str, np.ndarray] = {
            s.text: e for s, e in zip(sentences, embeddings) if e is not None
        }
        chunks: list[Chunk] = []
        char_offset = 0

        for chunk_text, depth in merged:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            start = char_offset
            end = start + len(chunk_text)

            # Find matching sentence embeddings for coherence
            chunk_embs: list[np.ndarray] = []
            chunk_sents: list[str] = []
            for sent in sentences:
                if sent.text in chunk_text:
                    chunk_sents.append(sent.text)
                    if sent.text in all_embeddings:
                        chunk_embs.append(all_embeddings[sent.text])

            coherence = self._compute_coherence(chunk_embs) if chunk_embs else 0.0

            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                sentences=chunk_sents,
                coherence_score=coherence,
                split_depth=depth
            ))
            char_offset = end + 1

        return chunks
