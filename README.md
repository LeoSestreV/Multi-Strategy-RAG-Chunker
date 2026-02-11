# Multi-Strategy RAG Chunker

A benchmark framework for comparing text chunking strategies applied to French academic biographies. Each strategy is a standalone project with an identical pipeline architecture, enabling direct comparison of chunking approaches for Retrieval-Augmented Generation (RAG).

## Architecture

Every strategy follows the same data flow:

```
Raw Text (.txt)
     │
     ▼
┌─────────────┐
│ Preprocessor │  → OCR cleanup (regex) + sentence segmentation (SpaCy fr_core_news_lg)
└──────┬──────┘
       ▼
┌─────────────┐
│  Embeddings  │  → Sentence vectorization via Ollama API (nomic-embed-text)
└──────┬──────┘
       ▼
┌─────────────┐
│   Chunker    │  → Strategy-specific grouping logic
└──────┬──────┘
       ▼
  JSON output
```

Each strategy folder contains 6 files:

| File | Role |
|---|---|
| `config.py` | Dataclass holding all tunable parameters |
| `preprocessing.py` | Text cleaning and SpaCy sentence segmentation |
| `embeddings.py` | Ollama embedding client (`/api/embeddings`) |
| `chunking.py` | Strategy-specific chunker class |
| `main.py` | `Pipeline` class orchestrating the full flow |
| `run_pipeline.py` | Entry point — scans `../BioTxt/`, writes JSON to `./output/` |

## Strategies

### 1. Semantic Chunking (Reference)

**Folder:** `Strategy_Semantic/`

Detects **cosine similarity drops** between consecutive sentence embeddings to find natural topic boundaries. When similarity falls below a threshold, a new chunk begins.

**How it works:**
1. Compute embeddings for every sentence
2. Calculate pairwise cosine similarity between consecutive sentences
3. Mark breakpoints where similarity < `similarity_threshold`
4. Group sentences between breakpoints into chunks
5. Merge chunks smaller than `min_chunk_size`, split those exceeding `max_chunk_size`

**Key parameters:**
- `similarity_threshold` (default: 0.5) — similarity below this triggers a split
- `max_chunk_size` (default: 1500 chars)
- `min_chunk_size` (default: 100 chars)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`

---

### 2. Recursive Character Chunking

**Folder:** `Strategy_Recursive/`

Splits text **hierarchically** using a cascade of separators, from coarse to fine: paragraphs → line breaks → sentences → words. Only descends to the next separator level when a chunk still exceeds the target size.

**How it works:**
1. Attempt to split on `\n\n` (paragraphs)
2. If any piece is still too large, split on `\n` (lines)
3. Then on `. ` (sentences), then on ` ` (words)
4. Merge fragments smaller than `min_chunk_size`
5. Compute coherence score via sentence embeddings

**Key parameters:**
- `target_chunk_size` (default: 1000 chars)
- `separators` (default: `("\n\n", "\n", ". ", " ")`)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `split_depth`

The `split_depth` field (0–3) indicates which separator level was needed, giving insight into the text's structural granularity.

---

### 3. Sentence-Based Chunking

**Folder:** `Strategy_Sentence/`

Groups **complete sentences** without ever splitting mid-sentence. Accumulates sentences until hitting either a sentence count limit or a character size limit.

**How it works:**
1. Iterate through SpaCy-segmented sentences
2. Add sentences to the current chunk until:
   - `max_sentences_per_chunk` is reached, or
   - adding the next sentence would exceed `max_chunk_size`
3. Start a new chunk and continue
4. Merge small trailing chunks into the previous one

**Key parameters:**
- `max_sentences_per_chunk` (default: 8)
- `max_chunk_size` (default: 1500 chars)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `sentence_count`

---

### 4. Document-Based Chunking

**Folder:** `Strategy_Document/`

Detects **logical document boundaries** using layout heuristics: section titles, numbered headings, bibliographic signatures, and paragraph breaks. Chunks correspond to structural sections of the document.

**How it works:**
1. Scan each sentence against regex patterns for:
   - **Section titles:** capitalized phrases, numbered headings (`1.`, `I.`), markdown headers
   - **End signatures:** "Bibliographie", "Références", "Publications", etc.
2. When a title or signature is detected, flush the current section as a chunk
3. Split oversized sections at the midpoint
4. Merge undersized sections with their neighbors

**Key parameters:**
- `section_patterns` — list of regex patterns for title detection
- `end_signature_patterns` — list of regex patterns for bibliography/end markers
- `max_chunk_size` (default: 2000 chars)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `section_title`, `is_end_section`

---

### 5. Token-Based Chunking

**Folder:** `Strategy_Token/`

Uses the **tiktoken** tokenizer (`cl100k_base` encoding) to ensure each chunk stays within a strict **token budget**. Includes token-level overlap between consecutive chunks for contextual continuity.

**How it works:**
1. Count tokens for each sentence using tiktoken
2. Accumulate sentences until the next one would exceed `max_tokens_per_chunk`
3. When a chunk is full, carry over trailing sentences that fit within `overlap_tokens`
4. The final `token_count` in the output is the actual token count of the assembled chunk

**Key parameters:**
- `max_tokens_per_chunk` (default: 256)
- `overlap_tokens` (default: 20)
- `encoding_name` (default: `cl100k_base`)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `token_count`

---

### 6. Sliding Window Chunking

**Folder:** `Strategy_SlidingWindow/`

A **fixed-size window** slides across the text with a configurable **overlap ratio** (10–20%). Each chunk shares overlapping sentences with the previous one, ensuring no context is lost at boundaries.

**How it works:**
1. Define `window_size` (characters) and `overlap_ratio`
2. Compute `step_size = window_size × (1 - overlap_ratio)`
3. Fill each window with sentences until reaching `window_size`
4. Advance the start position by `step_size` characters (sentence-aligned)
5. Track actual overlap characters between consecutive windows

**Key parameters:**
- `window_size` (default: 1000 chars)
- `overlap_ratio` (default: 0.15, clamped to [0.1, 0.2])

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `overlap_with_previous`, `window_index`

---

### 7. Proposition-Based Chunking

**Folder:** `Strategy_Proposition/`

The most complex strategy. Uses an **LLM (Mistral via Ollama)** to decompose each sentence into **atomic propositions** (standalone facts), then clusters these propositions by semantic similarity.

**How it works:**
1. For each sentence, prompt the LLM to extract atomic facts as a JSON array
2. Compute embeddings for every proposition
3. Greedy clustering: for each unassigned proposition, compute the cosine similarity between it and the group's centroid
4. Add to the group if similarity ≥ `similarity_threshold`, up to `max_propositions_per_chunk`
5. Build chunks from proposition groups, tracking source sentences

**Key parameters:**
- `llm_model` (default: `mistral`) — Ollama model for fact extraction
- `max_propositions_per_chunk` (default: 10)
- `similarity_threshold` (default: 0.6)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `atomic_facts`, `proposition_count`

The `atomic_facts` list contains each individual extracted proposition, enabling fine-grained analysis of information density per chunk.

---

### 8. Fixed-Size Chunking

**Folder:** `Strategy_FixedSize/`

The simplest approach: **mathematical division** by character count. The only intelligence is breaking at the last space before the limit to avoid splitting words.

**How it works:**
1. Join all sentences into a single text
2. Take the first `chunk_size` characters
3. Find the last space before that limit and split there
4. Repeat on the remainder until the text is exhausted
5. Merge a small trailing chunk into the previous one

**Key parameters:**
- `chunk_size` (default: 800 chars)
- `min_chunk_size` (default: 100 chars)

**Output fields:** `text`, `start_char`, `end_char`, `sentences`, `coherence_score`, `char_count`

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_lg
```

Ollama must be running locally with `nomic-embed-text` pulled (and `mistral` for the Proposition strategy):

```bash
ollama pull nomic-embed-text
ollama pull mistral          # only needed for Strategy_Proposition
```

### Running a Strategy

```bash
cd Strategy_Semantic/
python run_pipeline.py
```

Each pipeline scans `../BioTxt/*.txt` and writes results to `./output/` as JSON files.

### JSON Output Format

Every strategy produces a JSON file with this structure:

```json
{
  "source_file": "bio_sample_01.txt",
  "strategy": "semantic_chunking",
  "timestamp": "2026-02-10T12:00:00",
  "config": { ... },
  "total_chunks": 5,
  "chunks": [
    {
      "text": "...",
      "start_char": 0,
      "end_char": 312,
      "sentences": ["...", "..."],
      "coherence_score": 0.87
    }
  ]
}
```

The `coherence_score` (0.0–1.0) is computed identically across all strategies as the mean cosine similarity between consecutive sentence embeddings within a chunk, making it the primary metric for cross-strategy comparison.
