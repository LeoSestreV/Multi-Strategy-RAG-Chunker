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
    sentence_count: int = 0


class SentenceChunker:
    """Groups complete sentences into chunks, never splitting mid-sentence.
    Accumulates sentences until reaching the size or count limit."""

    def __init__(self, max_sentences_per_chunk: int = 8,
                 max_chunk_size: int = 1500,
                 min_chunk_size: int = 100):
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

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

        groups = []
        current_sents = []
        current_embs = []
        current_len = 0

        for sent, emb in zip(sents, embs):
            candidate_len = current_len + len(sent.text) + (1 if current_sents else 0)

            if current_sents and (
                candidate_len > self.max_chunk_size or
                len(current_sents) >= self.max_sentences_per_chunk
            ):
                groups.append((list(current_sents), list(current_embs)))
                current_sents = []
                current_embs = []
                current_len = 0

            current_sents.append(sent)
            current_embs.append(emb)
            current_len += len(sent.text) + (1 if len(current_sents) > 1 else 0)

        if current_sents:
            groups.append((current_sents, current_embs))

        # Merge small trailing chunks
        if len(groups) > 1:
            last_text = " ".join(s.text for s in groups[-1][0])
            if len(last_text) < self.min_chunk_size:
                prev_s, prev_e = groups[-2]
                prev_s.extend(groups[-1][0])
                prev_e.extend(groups[-1][1])
                groups[-2] = (prev_s, prev_e)
                groups.pop()

        chunks = []
        for grp_sents, grp_embs in groups:
            text = " ".join(s.text for s in grp_sents)
            chunks.append(Chunk(
                text=text,
                start_char=grp_sents[0].start_char,
                end_char=grp_sents[-1].end_char,
                sentences=[s.text for s in grp_sents],
                coherence_score=self._compute_coherence(grp_embs),
                sentence_count=len(grp_sents)
            ))
        return chunks
