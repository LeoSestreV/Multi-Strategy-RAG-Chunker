from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

    def __init__(self, similarity_threshold: float = 0.5,
                 max_chunk_size: int = 1500,
                 min_chunk_size: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _compute_similarities(self, embeddings: list[np.ndarray]) -> list[float]:
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def _compute_coherence(self, embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return 1.0
        sims = []
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

        groups = []
        start = 0
        for bp in breakpoints:
            groups.append((sents[start:bp], embs[start:bp]))
            start = bp
        groups.append((sents[start:], embs[start:]))

        # Merge small chunks
        merged = []
        buffer_sents, buffer_embs = [], []
        for grp_sents, grp_embs in groups:
            buffer_sents.extend(grp_sents)
            buffer_embs.extend(grp_embs)
            total_len = sum(len(s.text) for s in buffer_sents)
            if total_len >= self.min_chunk_size:
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
        final_groups = []
        for grp_sents, grp_embs in merged:
            total_len = sum(len(s.text) for s in grp_sents)
            if total_len > self.max_chunk_size and len(grp_sents) > 1:
                mid = len(grp_sents) // 2
                final_groups.append((grp_sents[:mid], grp_embs[:mid]))
                final_groups.append((grp_sents[mid:], grp_embs[mid:]))
            else:
                final_groups.append((grp_sents, grp_embs))

        chunks = []
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
