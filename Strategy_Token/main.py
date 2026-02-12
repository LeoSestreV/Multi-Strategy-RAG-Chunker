from dataclasses import asdict

from config import TokenChunkingConfig
from preprocessing import Preprocessor
from embeddings import EmbeddingClient
from chunking import TokenChunker, Chunk


class Pipeline:
    """Orchestrates: Raw Text -> Preprocessor -> Embeddings -> Chunker -> Result."""

    def __init__(self, config: TokenChunkingConfig = None):
        self.config = config or TokenChunkingConfig()
        self.preprocessor = Preprocessor(self.config.spacy_model)
        self.embedder = EmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
            max_workers=self.config.max_workers
        )
        self.chunker = TokenChunker(self.config)

    def run(self, raw_text: str) -> list[Chunk]:
        sentences = self.preprocessor.process(raw_text)
        texts = [s.text for s in sentences]
        embeddings = self.embedder.get_embeddings_batch(texts)
        chunks = self.chunker.chunk(sentences, embeddings)
        return chunks

    def run_to_dict(self, raw_text: str) -> list[dict]:
        chunks = self.run(raw_text)
        return [asdict(c) for c in chunks]
