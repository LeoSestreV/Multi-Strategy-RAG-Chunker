from dataclasses import dataclass


@dataclass
class TokenChunkingConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    spacy_model: str = "fr_core_news_lg"
    max_tokens_per_chunk: int = 256
    overlap_tokens: int = 20
    encoding_name: str = "cl100k_base"
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    default_coherence_score: float = 1.0
    input_dir: str = "../BioTxt/"
    output_dir: str = "./output/"
    max_workers: int = 4
