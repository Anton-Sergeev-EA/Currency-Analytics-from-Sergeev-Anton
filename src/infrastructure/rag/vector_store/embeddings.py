"""
VectorStore — lightweight TF-IDF + cosine-similarity retriever.

This used to be a chromadb + sentence-transformers (torch) stack. Two
problems with that, found during a review of the codebase:

1. It was never actually wired into RAGService (see
   src/application/services/rag_service.py's history) - the "RAG"
   assistant built its context by hand from forecast/data services and
   never called VectorStore.search() at all. Real retrieval simply
   wasn't happening.
2. sentence-transformers pulls in torch and transformers as transitive
   dependencies - well over 1GB of disk and several hundred MB of RAM
   just to embed a knowledge base of a few dozen short sentences. On a
   4GB-RAM VDS with limited disk, that cost buys nothing a much cheaper
   technique can't also deliver for a corpus this small.

TF-IDF + cosine similarity (via scikit-learn, already a project
dependency for the forecasting models) needs no model download, no GPU,
negligible RAM, and is a completely standard, well-understood retrieval
method for small, mostly-keyword-driven corpora like this one (currency
stats, investment tips) - a legitimate "real" retriever, not a
downgrade dressed up as one. The public interface (index_documents,
search, count, clear) is unchanged so callers don't need to know which
implementation is behind it.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._documents: List[str] = []
        self._sources: List[str] = []
        self._index_path = os.path.join(settings.DATA_DIR, "vector_db", "tfidf_index.joblib")
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._index_path):
            return
        try:
            bundle = joblib.load(self._index_path)
            self._vectorizer = bundle["vectorizer"]
            self._matrix = bundle["matrix"]
            self._documents = bundle["documents"]
            self._sources = bundle["sources"]
            logger.info("VectorStore: loaded %d documents from disk", len(self._documents))
        except Exception as exc:
            logger.warning("VectorStore: failed to load persisted index (%s), starting empty", exc)

    def index_documents(self, documents: List[Dict]) -> bool:
        if not documents:
            logger.warning("VectorStore.index_documents called with no documents")
            return False

        try:
            texts = [doc["text"] for doc in documents]
            sources = [doc.get("source", "unknown") for doc in documents]

            # Russian + English stopwords aren't bundled with scikit-learn,
            # and this corpus is small and domain-specific enough that a
            # plain word/character n-gram TF-IDF works well without one.
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
                max_features=4096,
            )
            matrix = vectorizer.fit_transform(texts)

            self._vectorizer = vectorizer
            self._matrix = matrix
            self._documents = texts
            self._sources = sources

            joblib.dump(
                {
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                    "documents": texts,
                    "sources": sources,
                },
                self._index_path,
            )
            logger.info("VectorStore: indexed %d documents", len(documents))
            return True
        except Exception as exc:
            logger.error("VectorStore.index_documents failed: %s", exc, exc_info=True)
            return False

    def search(self, query: str, n_results: int = 3) -> List[str]:
        results = self.search_with_scores(query, n_results)
        return [text for text, _source, _score in results]

    def search_with_scores(self, query: str, n_results: int = 3) -> List[tuple]:
        """Returns [(text, source, similarity_score), ...], best first."""
        if self._vectorizer is None or self._matrix is None or not self._documents:
            logger.debug("VectorStore.search called before indexing - no documents available")
            return []

        try:
            query_vec = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self._matrix)[0]
            ranked = similarities.argsort()[::-1][:n_results]
            return [
                (self._documents[i], self._sources[i], float(similarities[i]))
                for i in ranked
                if similarities[i] > 0
            ]
        except Exception as exc:
            logger.error("VectorStore.search failed: %s", exc, exc_info=True)
            return []

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> bool:
        self._vectorizer = None
        self._matrix = None
        self._documents = []
        self._sources = []
        try:
            if os.path.exists(self._index_path):
                os.remove(self._index_path)
        except OSError as exc:
            logger.warning("VectorStore.clear: could not remove index file: %s", exc)
        return True
