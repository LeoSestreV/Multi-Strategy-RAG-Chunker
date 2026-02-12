from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import TokenChunkingConfig
from preprocessing import Sentence


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    token_count: int = 0


class TokenChunker:
    """Splits text into chunks that respect token limits using tiktoken.
    Sentences are accumulated until adding the next would exceed the token budget."""

    def __init__(self, config: TokenChunkingConfig) -> None:
        self.config = config
        self.encoder = tiktoken.get_encoding(config.encoding_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

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

        groups: list[tuple[list[Sentence], list[np.ndarray], int]] = []
        current_sents: list[Sentence] = []
        current_embs: list[np.ndarray] = []
        current_tokens = 0

        for sent, emb in zip(sents, embs):
            sent_tokens = self._count_tokens(sent.text)

            if current_sents and current_tokens + sent_tokens > self.config.max_tokens_per_chunk:
                groups.append((list(current_sents), list(current_embs), current_tokens))

                # Calculate overlap: take trailing sentences that fit in overlap budget
                overlap_sents: list[Sentence] = []
                overlap_embs: list[np.ndarray] = []
                overlap_tok = 0
                for s, e in reversed(list(zip(current_sents, current_embs))):
                    s_tok = self._count_tokens(s.text)
                    if overlap_tok + s_tok > self.config.overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_embs.insert(0, e)
                    overlap_tok += s_tok

                current_sents = list(overlap_sents)
                current_embs = list(overlap_embs)
                current_tokens = overlap_tok

            current_sents.append(sent)
            current_embs.append(emb)
            current_tokens += sent_tokens

        if current_sents:
            groups.append((current_sents, current_embs, current_tokens))

        # Merge small trailing chunks
        if len(groups) > 1:
            last_text = " ".join(s.text for s in groups[-1][0])
            if len(last_text) < self.config.min_chunk_size:
                prev_s, prev_e, prev_t = groups[-2]
                last_s, last_e, last_t = groups[-1]
                prev_s.extend(last_s)
                prev_e.extend(last_e)
                groups[-2] = (prev_s, prev_e, prev_t + last_t)
                groups.pop()

        chunks: list[Chunk] = []
        for grp_sents, grp_embs, tok_count in groups:
            text = " ".join(s.text for s in grp_sents)
            chunks.append(Chunk(
                text=text,
                start_char=grp_sents[0].start_char,
                end_char=grp_sents[-1].end_char,
                sentences=[s.text for s in grp_sents],
                coherence_score=self._compute_coherence(grp_embs),
                token_count=self._count_tokens(text)
            ))
        return chunks
