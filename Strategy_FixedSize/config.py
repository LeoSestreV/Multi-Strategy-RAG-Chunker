from dataclasses import dataclass


@dataclass
class FixedSizeChunkingConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    spacy_model: str = "fr_core_news_lg"
    chunk_size: int = 800
    min_chunk_size: int = 100
    input_dir: str = "../BioTxt/"
    output_dir: str = "./output/"
    max_workers: int = 4
