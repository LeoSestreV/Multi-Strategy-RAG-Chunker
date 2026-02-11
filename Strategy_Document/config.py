from dataclasses import dataclass, field


@dataclass
class DocumentChunkingConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    spacy_model: str = "fr_core_news_lg"
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    section_patterns: list = field(default_factory=lambda: [
        r'^[A-ZÀÂÉÈÊËÏÎÔÙÛÜÇ][A-ZÀÂÉÈÊËÏÎÔÙÛÜÇa-zàâéèêëïîôùûüç\s\-\':]+$',
        r'^#{1,3}\s+.+',
        r'^\d+[\.\)]\s+[A-ZÀÂÉÈÊËÏÎÔÙÛÜÇ]',
        r'^[IVXLCDM]+[\.\)]\s+',
    ])
    end_signature_patterns: list = field(default_factory=lambda: [
        r'^(Bibliographie|Références|Publications|Notes)\s*$',
        r'^(Signé|Fait à|Date\s*:)',
    ])
    input_dir: str = "../BioTxt/"
    output_dir: str = "./output/"
    max_workers: int = 4
