import re
from dataclasses import dataclass

import spacy


@dataclass
class Sentence:
    text: str
    start_char: int
    end_char: int
    index: int


class Preprocessor:
    """Cleans raw OCR text and segments it into sentences using SpaCy."""

    def __init__(self, spacy_model: str = "fr_core_news_lg"):
        self.nlp = spacy.load(spacy_model)

    def clean_text(self, raw_text: str) -> str:
        # Remove hyphenation at line breaks
        text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', raw_text)
        # Normalize line breaks to spaces
        text = re.sub(r'\n(?!\n)', ' ', text)
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Preserve paragraph breaks
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()

    def segment_sentences(self, clean_text: str) -> list[Sentence]:
        doc = self.nlp(clean_text)
        sentences = []
        for i, sent in enumerate(doc.sents):
            sentences.append(Sentence(
                text=sent.text.strip(),
                start_char=sent.start_char,
                end_char=sent.end_char,
                index=i
            ))
        return sentences

    def process(self, raw_text: str) -> list[Sentence]:
        clean = self.clean_text(raw_text)
        return self.segment_sentences(clean)
