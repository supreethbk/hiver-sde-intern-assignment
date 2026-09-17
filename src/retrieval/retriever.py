"""Retrieve similar, historical AmazonHelp conversations."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = ROOT / "results" / "amazonhelp_embeddings_3k.pkl"
REFERENCE_FILE = ROOT / "results" / "amazonhelp_reference_3k.csv"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


class HistoricalRetriever:
    """Semantic retriever with a TF-IDF fallback for a fresh checkout."""

    def __init__(self) -> None:
        self.method = "tfidf"
        self.embeddings: np.ndarray | None = None
        self.messages: list[str]
        self.responses: list[str]
        self.vectorizer: TfidfVectorizer | None = None
        self.vectors = None

        if CACHE_FILE.exists():
            with CACHE_FILE.open("rb") as handle:
                cache = pickle.load(handle)
            self.embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
            self.messages = list(cache["customer_messages"])
            self.responses = list(cache["support_responses"])
            self.method = "semantic"
            return

        reference = pd.read_csv(REFERENCE_FILE)
        self.messages = reference["customer_message"].fillna("").astype(str).tolist()
        self.responses = reference["support_response"].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )
        self.vectors = self.vectorizer.fit_transform(self.messages)

    def _semantic_scores(self, query: str) -> np.ndarray:
        response = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": [query]},
            timeout=60,
        )
        response.raise_for_status()
        query_embedding = np.asarray(response.json()["embeddings"][0], dtype=np.float32)
        assert self.embeddings is not None
        denominator = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        return np.dot(self.embeddings, query_embedding) / np.maximum(denominator, 1e-12)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        """Return the nearest historical customer/support pairs."""
        if self.method == "semantic":
            try:
                scores = self._semantic_scores(query)
            except requests.RequestException:
                # A cache alone is not enough if the embed server is unavailable.
                self.method = "tfidf"
                self.vectorizer = TfidfVectorizer(
                    lowercase=True, stop_words="english", ngram_range=(1, 2)
                )
                self.vectors = self.vectorizer.fit_transform(self.messages)

        if self.method == "tfidf":
            assert self.vectorizer is not None and self.vectors is not None
            scores = (self.vectorizer.transform([query]) @ self.vectors.T).toarray()[0]

        indices = np.argsort(scores)[-top_k:][::-1]
        return [
            {
                "customer_message": self.messages[index],
                "support_response": self.responses[index],
                "similarity": round(float(scores[index]), 4),
            }
            for index in indices
        ]
