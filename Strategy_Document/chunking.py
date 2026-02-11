import re
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
    section_title: str = ""
    is_end_section: bool = False


class DocumentChunker:
    """Identifies logical document boundaries (titles, section breaks,
    end signatures) and groups content into structural sections."""

    def __init__(self, max_chunk_size: int = 2000,
                 min_chunk_size: int = 100,
                 section_patterns: list = None,
                 end_signature_patterns: list = None):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.section_patterns = section_patterns or [
            r'^[A-ZÀÂÉÈÊËÏÎÔÙÛÜÇ][A-ZÀÂÉÈÊËÏÎÔÙÛÜÇa-zàâéèêëïîôùûüç\s\-\':]+$',
            r'^#{1,3}\s+.+',
            r'^\d+[\.\)]\s+[A-ZÀÂÉÈÊËÏÎÔÙÛÜÇ]',
            r'^[IVXLCDM]+[\.\)]\s+',
        ]
        self.end_signature_patterns = end_signature_patterns or [
            r'^(Bibliographie|Références|Publications|Notes)\s*$',
            r'^(Signé|Fait à|Date\s*:)',
        ]

    def _is_section_title(self, text: str) -> bool:
        text = text.strip()
        if len(text) > 80 or len(text) < 3:
            return False
        for pattern in self.section_patterns:
            if re.match(pattern, text):
                return True
        # Heuristic: short line, no ending punctuation, mostly capitalized words
        if not text.endswith(('.', ',', ';', ':', '!', '?')) and len(text.split()) <= 6:
            words = text.split()
            cap_count = sum(1 for w in words if w[0].isupper())
            if cap_count >= len(words) * 0.6:
                return True
        return False

    def _is_end_signature(self, text: str) -> bool:
        text = text.strip()
        for pattern in self.end_signature_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

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

        # Detect section boundaries
        sections = []
        current_sents = []
        current_embs = []
        current_title = ""
        is_end = False

        for sent, emb in zip(sentences, embeddings):
            if self._is_end_signature(sent.text):
                # Flush current section
                if current_sents:
                    sections.append((current_title, current_sents, current_embs, False))
                current_title = sent.text
                current_sents = [sent]
                current_embs = [emb]
                is_end = True
                continue

            if self._is_section_title(sent.text) and current_sents:
                sections.append((current_title, current_sents, current_embs, is_end))
                current_title = sent.text
                current_sents = [sent]
                current_embs = [emb]
                is_end = False
                continue

            if not current_title and self._is_section_title(sent.text):
                current_title = sent.text

            current_sents.append(sent)
            current_embs.append(emb)

        if current_sents:
            sections.append((current_title, current_sents, current_embs, is_end))

        # Build chunks from sections
        chunks = []
        for title, sec_sents, sec_embs, end_flag in sections:
            text = " ".join(s.text for s in sec_sents)

            # Split oversized sections
            if len(text) > self.max_chunk_size and len(sec_sents) > 1:
                mid = len(sec_sents) // 2
                for sub_sents, sub_embs in [(sec_sents[:mid], sec_embs[:mid]),
                                             (sec_sents[mid:], sec_embs[mid:])]:
                    valid_embs = [e for e in sub_embs if e is not None]
                    sub_text = " ".join(s.text for s in sub_sents)
                    chunks.append(Chunk(
                        text=sub_text,
                        start_char=sub_sents[0].start_char,
                        end_char=sub_sents[-1].end_char,
                        sentences=[s.text for s in sub_sents],
                        coherence_score=self._compute_coherence(valid_embs),
                        section_title=title,
                        is_end_section=end_flag
                    ))
            else:
                valid_embs = [e for e in sec_embs if e is not None]
                chunks.append(Chunk(
                    text=text,
                    start_char=sec_sents[0].start_char,
                    end_char=sec_sents[-1].end_char,
                    sentences=[s.text for s in sec_sents],
                    coherence_score=self._compute_coherence(valid_embs),
                    section_title=title,
                    is_end_section=end_flag
                ))

        # Merge small chunks
        if len(chunks) > 1:
            merged = [chunks[0]]
            for c in chunks[1:]:
                if len(c.text) < self.min_chunk_size:
                    prev = merged[-1]
                    merged[-1] = Chunk(
                        text=prev.text + " " + c.text,
                        start_char=prev.start_char,
                        end_char=c.end_char,
                        sentences=prev.sentences + c.sentences,
                        coherence_score=(prev.coherence_score + c.coherence_score) / 2,
                        section_title=prev.section_title,
                        is_end_section=c.is_end_section
                    )
                else:
                    merged.append(c)
            chunks = merged

        return chunks
