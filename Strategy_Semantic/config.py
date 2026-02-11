from dataclasses import dataclass, field


@dataclass
class SemanticChunkingConfig:
    """Configuration for the semantic chunking pipeline."""
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    spacy_model: str = "fr_core_news_lg"
    similarity_threshold: float = 0.5
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    input_dir: str = "../BioTxt/"
    output_dir: str = "./output/"
    max_workers: int = 4
