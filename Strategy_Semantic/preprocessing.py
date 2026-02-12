import re
import unicodedata
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
        # Normalize Unicode (NFC) to merge composed characters
        text = unicodedata.normalize("NFC", raw_text)
        # Remove soft hyphens
        text = text.replace("\u00ad", "")
        # OCR hyphenation: rejoin words split across lines (e.g., "mathé-\nmaticien")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # OCR hyphenation: rejoin words split mid-line (e.g., "mathé- maticien")
        # Only match lowercase continuation to avoid breaking real hyphens
        text = re.sub(r'(\w{2,})-\s{1,3}([a-z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ef\u00ee\u00f4\u00f9\u00fb\u00fc\u00ff\u00e7\u0153\u00e6]\w+)', r'\1\2', text)
        # Normalize single line breaks to spaces
        text = re.sub(r'\n(?!\n)', ' ', text)
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Preserve paragraph breaks
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()

    def segment_sentences(self, clean_text: str) -> list[Sentence]:
        doc = self.nlp(clean_text)
        sentences = []
        idx = 0
        for sent in doc.sents:
            text = sent.text.strip()
            # Filter out empty or trivial sentences (punctuation-only, single chars)
            if len(text) < 2:
                continue
            stripped = text.strip('.,;:!?-\u2014\u2013()[]"\' ')
            if len(stripped) < 2:
                continue
            sentences.append(Sentence(
                text=text,
                start_char=sent.start_char,
                end_char=sent.end_char,
                index=idx
            ))
            idx += 1
        return sentences

    def process(self, raw_text: str) -> list[Sentence]:
        clean = self.clean_text(raw_text)
        return self.segment_sentences(clean)
