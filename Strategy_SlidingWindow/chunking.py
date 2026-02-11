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
    overlap_with_previous: int = 0
    window_index: int = 0


class SlidingWindowChunker:
    """Creates chunks using a fixed-size sliding window with configurable
    overlap (10-20%) for contextual continuity between consecutive chunks."""

    def __init__(self, window_size: int = 1000,
                 overlap_ratio: float = 0.15,
                 min_chunk_size: int = 100):
        self.window_size = window_size
        self.overlap_ratio = max(0.1, min(0.2, overlap_ratio))
        self.min_chunk_size = min_chunk_size
        self.step_size = int(window_size * (1 - self.overlap_ratio))

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

        # Build sentence-level windows based on character count
        chunks = []
        window_idx = 0
        start_sent = 0

        while start_sent < len(sents):
            current_sents = []
            current_embs = []
            current_len = 0

            i = start_sent
            while i < len(sents) and current_len + len(sents[i].text) <= self.window_size:
                current_sents.append(sents[i])
                current_embs.append(embs[i])
                current_len += len(sents[i].text) + 1
                i += 1

            # Ensure at least one sentence per window
            if not current_sents and start_sent < len(sents):
                current_sents.append(sents[start_sent])
                current_embs.append(embs[start_sent])
                i = start_sent + 1

            if not current_sents:
                break

            text = " ".join(s.text for s in current_sents)

            # Calculate actual overlap characters with previous chunk
            overlap_chars = 0
            if chunks:
                prev_text = chunks[-1].text
                # Find common suffix/prefix
                for s in current_sents:
                    if s.text in prev_text:
                        overlap_chars += len(s.text)
                    else:
                        break

            chunks.append(Chunk(
                text=text,
                start_char=current_sents[0].start_char,
                end_char=current_sents[-1].end_char,
                sentences=[s.text for s in current_sents],
                coherence_score=self._compute_coherence(current_embs),
                overlap_with_previous=overlap_chars,
                window_index=window_idx
            ))

            # Advance by step_size worth of characters
            step_chars = 0
            next_start = start_sent
            while next_start < len(sents) and step_chars < self.step_size:
                step_chars += len(sents[next_start].text) + 1
                next_start += 1

            if next_start == start_sent:
                next_start += 1

            start_sent = next_start
            window_idx += 1

        # Merge small trailing chunk
        if len(chunks) > 1 and len(chunks[-1].text) < self.min_chunk_size:
            last = chunks.pop()
            prev = chunks[-1]
            # Add only new sentences
            new_sents = [s for s in last.sentences if s not in prev.sentences]
            if new_sents:
                chunks[-1] = Chunk(
                    text=prev.text + " " + " ".join(new_sents),
                    start_char=prev.start_char,
                    end_char=last.end_char,
                    sentences=prev.sentences + new_sents,
                    coherence_score=prev.coherence_score,
                    overlap_with_previous=prev.overlap_with_previous,
                    window_index=prev.window_index
                )

        return chunks
