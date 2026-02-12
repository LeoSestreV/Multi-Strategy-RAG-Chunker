from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import PropositionChunkingConfig
from preprocessing import Sentence


@dataclass
class Proposition:
    text: str
    source_sentence_index: int
    embedding: np.ndarray = field(default=None, repr=False)


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    atomic_facts: list[str] = field(default_factory=list)
    proposition_count: int = 0


class PropositionChunker:
    """Decomposes sentences into atomic propositions using an LLM,
    then groups semantically similar propositions into chunks."""

    def __init__(self, config: PropositionChunkingConfig) -> None:
        self.config = config

    @staticmethod
    def _is_valid_proposition(text: str) -> bool:
        """Filter out incomplete propositions (isolated words, punctuation, etc.)."""
        text = text.strip()
        if len(text) < 10:
            return False
        # Must contain at least one verb-like structure (space implies multi-word)
        if ' ' not in text:
            return False
        # Reject strings that are only punctuation/stopwords
        stripped = text.strip('.,;:!?-—–()[]"\'« » ')
        if len(stripped) < 5:
            return False
        return True

    def _extract_propositions(self, sentence: Sentence) -> list[Proposition]:
        """Use Ollama LLM to decompose a sentence into atomic facts in French."""
        prompt = (
            "Décompose la phrase suivante en une liste de propositions atomiques "
            "(des faits simples et autonomes). Chaque proposition doit être une "
            "phrase complète en français, avec sujet et verbe. "
            "Ne traduis PAS en anglais : la sortie doit être exclusivement en français. "
            "Retourne UNIQUEMENT un tableau JSON de chaînes de caractères, rien d'autre.\n\n"
            f"Phrase : \"{sentence.text}\"\n\n"
            "Tableau JSON :"
        )

        ollama_base_url = self.config.ollama_base_url.rstrip("/")

        payload = json.dumps({
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "[]")

                # Extract JSON array from response
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end > start:
                    facts = json.loads(response_text[start:end])
                    return [
                        Proposition(
                            text=f.strip(),
                            source_sentence_index=sentence.index
                        )
                        for f in facts
                        if isinstance(f, str) and f.strip() and self._is_valid_proposition(f)
                    ]
        except Exception as e:
            print(f"LLM proposition extraction error: {e}")

        # Fallback: use original sentence as a single proposition
        return [Proposition(text=sentence.text, source_sentence_index=sentence.index)]

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

        # Extract propositions from all sentences in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            prop_lists = list(executor.map(self._extract_propositions, sentences))

        all_propositions: list[Proposition] = []
        for props in prop_lists:
            all_propositions.extend(props)

        if not all_propositions:
            return []

        # Get embeddings for each proposition (already threaded via EmbeddingClient)
        from embeddings import EmbeddingClient
        ollama_base_url = self.config.ollama_base_url.rstrip("/")
        emb_client = EmbeddingClient(
            base_url=ollama_base_url,
            max_workers=self.config.max_workers
        )
        prop_embeddings = emb_client.get_embeddings_batch(
            [p.text for p in all_propositions]
        )

        for prop, emb in zip(all_propositions, prop_embeddings):
            prop.embedding = emb

        valid_props = [p for p in all_propositions if p.embedding is not None]
        if not valid_props:
            return []

        # Group propositions by semantic similarity (greedy clustering)
        groups: list[list[Proposition]] = []
        used: set[int] = set()

        for i, prop in enumerate(valid_props):
            if i in used:
                continue

            group: list[Proposition] = [prop]
            used.add(i)

            for j in range(i + 1, len(valid_props)):
                if j in used:
                    continue
                if len(group) >= self.config.max_propositions_per_chunk:
                    break

                # Compare with group centroid
                group_embs = np.array([p.embedding for p in group])
                centroid = group_embs.mean(axis=0).reshape(1, -1)
                sim = cosine_similarity(
                    centroid,
                    valid_props[j].embedding.reshape(1, -1)
                )[0][0]

                if sim >= self.config.similarity_threshold:
                    group.append(valid_props[j])
                    used.add(j)

            groups.append(group)

        # Build chunks from proposition groups
        chunks: list[Chunk] = []
        sent_map: dict[int, Sentence] = {s.index: s for s in sentences}
        emb_map: dict[int, np.ndarray] = {
            s.index: e for s, e in zip(sentences, embeddings) if e is not None
        }

        for group in groups:
            facts = [p.text for p in group]
            text = " ".join(facts)

            # Find source sentence range
            source_indices = sorted(set(p.source_sentence_index for p in group))
            source_sents = [sent_map[idx] for idx in source_indices if idx in sent_map]
            source_embs = [emb_map[idx] for idx in source_indices if idx in emb_map]

            if source_sents:
                start_char = source_sents[0].start_char
                end_char = source_sents[-1].end_char
            else:
                start_char = 0
                end_char = len(text)

            chunks.append(Chunk(
                text=text,
                start_char=start_char,
                end_char=end_char,
                sentences=[s.text for s in source_sents],
                coherence_score=self._compute_coherence(source_embs) if source_embs else 0.0,
                atomic_facts=facts,
                proposition_count=len(facts)
            ))

        # Merge small chunks
        if len(chunks) > 1:
            merged: list[Chunk] = [chunks[0]]
            for c in chunks[1:]:
                if len(c.text) < self.config.min_chunk_size:
                    prev = merged[-1]
                    merged[-1] = Chunk(
                        text=prev.text + " " + c.text,
                        start_char=prev.start_char,
                        end_char=c.end_char,
                        sentences=prev.sentences + c.sentences,
                        coherence_score=(prev.coherence_score + c.coherence_score) / 2,
                        atomic_facts=prev.atomic_facts + c.atomic_facts,
                        proposition_count=prev.proposition_count + c.proposition_count
                    )
                else:
                    merged.append(c)
            chunks = merged

        return chunks
