from dataclasses import dataclass


@dataclass
class PropositionChunkingConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    spacy_model: str = "fr_core_news_lg"
    llm_model: str = "mistral"
    max_propositions_per_chunk: int = 10
    similarity_threshold: float = 0.6
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    default_coherence_score: float = 1.0
    input_dir: str = "../BioTxt/"
    output_dir: str = "./output/"
    max_workers: int = 1
