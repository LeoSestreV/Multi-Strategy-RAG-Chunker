from dataclasses import asdict

from config import PropositionChunkingConfig
from preprocessing import Preprocessor
from embeddings import EmbeddingClient
from chunking import PropositionChunker, Chunk


class Pipeline:
    """Orchestrates: Raw Text -> Preprocessor -> Embeddings -> Chunker -> Result."""

    def __init__(self, config: PropositionChunkingConfig = None):
        self.config = config or PropositionChunkingConfig()
        self.preprocessor = Preprocessor(self.config.spacy_model)
        self.embedder = EmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
            max_workers=self.config.max_workers
        )
        self.chunker = PropositionChunker(self.config)

    def run(self, raw_text: str) -> list[Chunk]:
        sentences = self.preprocessor.process(raw_text)
        texts = [s.text for s in sentences]
        embeddings = self.embedder.get_embeddings_batch(texts)
        chunks = self.chunker.chunk(sentences, embeddings)
        return chunks

    @staticmethod
    def _clean_chunk_dict(d: dict) -> dict:
        """Remove empty or trivial entries from the sentences and atomic_facts lists."""
        if "sentences" in d:
            d["sentences"] = [
                s for s in d["sentences"]
                if isinstance(s, str) and len(s.strip()) > 1
                and len(s.strip().strip('.,;:!?-\u2014\u2013()[]"\' ')) > 1
            ]
        if "atomic_facts" in d:
            d["atomic_facts"] = [
                f for f in d["atomic_facts"]
                if isinstance(f, str) and len(f.strip()) > 10
            ]
            d["proposition_count"] = len(d["atomic_facts"])
        return d

    def run_to_dict(self, raw_text: str) -> list[dict]:
        chunks = self.run(raw_text)
        return [self._clean_chunk_dict(asdict(c)) for c in chunks]
