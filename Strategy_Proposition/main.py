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
        self.chunker = PropositionChunker(
            llm_model=self.config.llm_model,
            ollama_base_url=self.config.ollama_base_url,
            max_propositions_per_chunk=self.config.max_propositions_per_chunk,
            similarity_threshold=self.config.similarity_threshold,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            max_workers=self.config.max_workers
        )

    def run(self, raw_text: str) -> list[Chunk]:
        sentences = self.preprocessor.process(raw_text)
        texts = [s.text for s in sentences]
        embeddings = self.embedder.get_embeddings_batch(texts)
        chunks = self.chunker.chunk(sentences, embeddings)
        return chunks

    def run_to_dict(self, raw_text: str) -> list[dict]:
        chunks = self.run(raw_text)
        return [asdict(c) for c in chunks]
