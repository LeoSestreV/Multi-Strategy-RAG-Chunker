from dataclasses import asdict

from config import RecursiveChunkingConfig
from preprocessing import Preprocessor
from embeddings import EmbeddingClient
from chunking import RecursiveChunker, Chunk


class Pipeline:
    """Orchestrates: Raw Text -> Preprocessor -> Embeddings -> Chunker -> Result."""

    def __init__(self, config: RecursiveChunkingConfig = None):
        self.config = config or RecursiveChunkingConfig()
        self.preprocessor = Preprocessor(self.config.spacy_model)
        self.embedder = EmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
            max_workers=self.config.max_workers
        )
        self.chunker = RecursiveChunker(
            target_chunk_size=self.config.target_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            separators=self.config.separators
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
