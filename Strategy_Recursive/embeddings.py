import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np


class EmbeddingClient:
    """Handles vectorization via the Ollama /api/embeddings endpoint."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434",
                 max_workers: int = 4):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_workers = max_workers

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        payload = json.dumps({
            "model": self.model,
            "prompt": text
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.get_embedding, texts))
