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
    """Décompose les phrases en propositions atomiques via LLM par lots, 
    puis les regroupe par similarité sémantique."""

    def __init__(self, config: PropositionChunkingConfig) -> None:
        self.config = config

    @staticmethod
    def _is_valid_proposition(text: str) -> bool:
        """Filtre les propositions incomplètes ou triviales."""
        text = text.strip()
        if len(text) < 10:
            return False
        if ' ' not in text:
            return False
        stripped = text.strip('.,;:!?-—–()[]"\'« » ')
        if len(stripped) < 5:
            return False
        return True

    def _extract_propositions_batch(self, sentences: list[Sentence]) -> list[Proposition]:
        """Extrait les propositions pour un lot de phrases avec un prompt renforcé."""
        if not sentences:
            return []

        input_data = {str(s.index): s.text for s in sentences}
        
        # Prompt détaillé pour réduire les incohérences
        prompt = (
            "Tu es un expert en linguistique computationnelle spécialisé dans l'analyse de textes historiques et biographiques.\n"
            "Ta tâche est de décomposer les phrases suivantes en propositions atomiques indépendantes (faits unitaires).\n\n"
            "DIRECTIVES DE TRANSFORMATION :\n"
            "1. INDÉPENDANCE : Chaque proposition doit être autonome. Si tu lis la proposition seule, on doit comprendre de qui et de quoi on parle.\n"
            "2. RÉSOLUTION NOMINALE : Remplace TOUS les pronoms (il, elle, celui-ci, le, la, leur, etc.) par le nom du sujet ou de l'objet qu'ils remplacent dans le contexte de la phrase.\n"
            "3. STRUCTURE : Utilise impérativement une structure [Sujet] + [Verbe] + [Complément]. Ne crée pas de fragments.\n"
            "4. DÉCONSTRUCTION : Si une phrase contient plusieurs actions ou attributs (ex: 'A fit X et Y'), crée deux propositions : '[Sujet] fit X' et '[Sujet] fit Y'.\n"
            "5. FIDÉLITÉ : Ne résume pas. Conserve le vocabulaire précis de l'auteur.\n"
            "6. LANGUE : Réponds exclusivement en français.\n\n"
            "CONTRAINTE DE FORMAT :\n"
            "Réponds UNIQUEMENT avec un objet JSON strict. Pas de texte avant ou après.\n"
            "Format : {\"ID_PHRASE\": [\"Proposition 1\", \"Proposition 2\"]}\n\n"
            f"DONNÉES À TRAITER :\n{json.dumps(input_data, ensure_ascii=False)}\n\n"
            "Réponse JSON :"
        )

        ollama_base_url = self.config.ollama_base_url.rstrip("/")
        payload = json.dumps({
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0} # Température à 0 pour plus de cohérence
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        all_props = []
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "{}")
                
                # Nettoyage de la réponse pour isoler le JSON
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    batch_results = json.loads(response_text[start:end])
                    
                    for s_idx_str, facts in batch_results.items():
                        try:
                            s_idx = int(s_idx_str)
                            if isinstance(facts, list):
                                for f in facts:
                                    if isinstance(f, str) and self._is_valid_proposition(f):
                                        all_props.append(Proposition(
                                            text=f.strip(), 
                                            source_sentence_index=s_idx
                                        ))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Erreur lors de l'extraction par lot : {e}")
            # Fallback automatique sur la phrase originale
            for s in sentences:
                all_props.append(Proposition(text=s.text, source_sentence_index=s.index))

        return all_props

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

    def chunk(self, sentences: list[Sentence], embeddings: list[np.ndarray]) -> list[Chunk]:
        if not sentences:
            return []

        # Découpage en lots pour le LLM (3 à 5 phrases est un bon compromis)
        batch_size = 4 
        sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

        all_propositions: list[Proposition] = []
        
        # Parallélisation des appels par lots
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(self._extract_propositions_batch, sentence_batches))
            for res in results:
                all_propositions.extend(res)

        if not all_propositions:
            return []

        # Calcul des embeddings pour les propositions extraites
        from embeddings import EmbeddingClient
        emb_client = EmbeddingClient(
            base_url=self.config.ollama_base_url,
            max_workers=self.config.max_workers
        )
        prop_texts = [p.text for p in all_propositions]
        prop_embs = emb_client.get_embeddings_batch(prop_texts)

        for prop, emb in zip(all_propositions, prop_embs):
            prop.embedding = emb

        valid_props = [p for p in all_propositions if p.embedding is not None]
        
        # Groupement sémantique
        groups: list[list[Proposition]] = []
        used = set()

        for i, prop in enumerate(valid_props):
            if i in used: continue
            group = [prop]
            used.add(i)

            for j in range(i + 1, len(valid_props)):
                if j in used or len(group) >= self.config.max_propositions_per_chunk:
                    continue

                group_embs = np.array([p.embedding for p in group])
                centroid = group_embs.mean(axis=0).reshape(1, -1)
                sim = cosine_similarity(centroid, valid_props[j].embedding.reshape(1, -1))[0][0]

                if sim >= self.config.similarity_threshold:
                    group.append(valid_props[j])
                    used.add(j)
            groups.append(group)

        # Création des Chunks
        final_chunks = []
        sent_map = {s.index: s for s in sentences}
        emb_map = {s.index: e for s, e in zip(sentences, embeddings) if e is not None}

        for group in groups:
            facts = [p.text for p in group]
            source_indices = sorted(set(p.source_sentence_index for p in group))
            source_sents = [sent_map[idx] for idx in source_indices if idx in sent_map]
            source_embs = [emb_map[idx] for idx in source_indices if idx in emb_map]

            final_chunks.append(Chunk(
                text=" ".join(facts),
                start_char=source_sents[0].start_char if source_sents else 0,
                end_char=source_sents[-1].end_char if source_sents else 0,
                sentences=[s.text for s in source_sents],
                coherence_score=self._compute_coherence(source_embs) if source_embs else 0.0,
                atomic_facts=facts,
                proposition_count=len(facts)
            ))

        return final_chunks