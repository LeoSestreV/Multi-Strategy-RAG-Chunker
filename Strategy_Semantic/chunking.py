from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import SemanticChunkingConfig
from preprocessing import Sentence


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: list[str] = field(default_factory=list)
    coherence_score: float = 0.0


class SemanticChunker:
    """Groups sentences into semantically coherent chunks by detecting
    similarity drops between consecutive sentence embeddings."""

    def __init__(self, config: SemanticChunkingConfig) -> None:
        self.config = config

    def _compute_similarities(self, embeddings: list[np.ndarray]) -> list[float]:
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        breakpoints: list[int] = []
        for i, sim in enumerate(similarities):
            if sim < self.config.similarity_threshold:
                breakpoints.append(i + 1)
        return breakpoints

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

        valid = [(s, e) for s, e in zip(sentences, embeddings) if e is not None]
        if not valid:
            return []

        sents, embs = zip(*valid)
        sents, embs = list(sents), list(embs)

        similarities = self._compute_similarities(embs)
        breakpoints = self._find_breakpoints(similarities)

        groups: list[tuple[list[Sentence], list[np.ndarray]]] = []
        start = 0
        for bp in breakpoints:
            groups.append((sents[start:bp], embs[start:bp]))
            start = bp
        groups.append((sents[start:], embs[start:]))

        # Merge small chunks
        merged: list[tuple[list[Sentence], list[np.ndarray]]] = []
        buffer_sents: list[Sentence] = []
        buffer_embs: list[np.ndarray] = []
        for grp_sents, grp_embs in groups:
            buffer_sents.extend(grp_sents)
            buffer_embs.extend(grp_embs)
            total_len = sum(len(s.text) for s in buffer_sents)
            if total_len >= self.config.min_chunk_size:
                merged.append((list(buffer_sents), list(buffer_embs)))
                buffer_sents, buffer_embs = [], []
        if buffer_sents:
            if merged:
                prev_s, prev_e = merged[-1]
                prev_s.extend(buffer_sents)
                prev_e.extend(buffer_embs)
                merged[-1] = (prev_s, prev_e)
            else:
                merged.append((buffer_sents, buffer_embs))

        # Split oversized chunks
        final_groups: list[tuple[list[Sentence], list[np.ndarray]]] = []
        for grp_sents, grp_embs in merged:
            total_len = sum(len(s.text) for s in grp_sents)
            if total_len > self.config.max_chunk_size and len(grp_sents) > 1:
                mid = int(len(grp_sents) * self.config.split_ratio)
                mid = max(1, min(mid, len(grp_sents) - 1))
                final_groups.append((grp_sents[:mid], grp_embs[:mid]))
                final_groups.append((grp_sents[mid:], grp_embs[mid:]))
            else:
                final_groups.append((grp_sents, grp_embs))

        chunks: list[Chunk] = []
        for grp_sents, grp_embs in final_groups:
            text = " ".join(s.text for s in grp_sents)
            chunks.append(Chunk(
                text=text,
                start_char=grp_sents[0].start_char,
                end_char=grp_sents[-1].end_char,
                sentences=[s.text for s in grp_sents],
                coherence_score=self._compute_coherence(grp_embs)
            ))
        return chunks
