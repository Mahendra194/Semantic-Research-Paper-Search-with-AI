"""
Retriever Module
================
High-level semantic search interface that orchestrates the embedder and vector store.
Supports optional metadata filtering (by year, category) on top of vector similarity.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .embedder import EmbeddingModel
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Semantic search retriever combining embedding model and FAISS index.

    Usage:
        retriever = SemanticRetriever(embedder, vector_store)
        results = retriever.search("attention mechanisms in NLP", top_k=5)
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: FAISSVectorStore,
    ):
        """
        Initialize the retriever.

        Args:
            embedder: EmbeddingModel instance for query encoding.
            vector_store: FAISSVectorStore instance with indexed papers.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        year_filter: Optional[int] = None,
        category_filter: Optional[str] = None,
        author_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional metadata filters.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            year_filter: If set, only return papers from this year or later.
            category_filter: If set, only return papers matching this category prefix.
            author_filter: If set, only return papers whose authors contain this string.
            min_score: Minimum similarity score threshold.

        Returns:
            List of result dicts with 'metadata', 'score', 'rank'.
        """
        # Encode query
        query_embedding = self.embedder.encode_query(query)

        # Over-fetch if filters are applied (to ensure enough results post-filter)
        fetch_k = top_k * 5 if (year_filter or category_filter or author_filter) else top_k

        # Search vector store
        raw_results = self.vector_store.search(query_embedding, top_k=fetch_k)

        # Apply filters
        filtered = []
        for result in raw_results:
            meta = result["metadata"]

            # Score threshold
            if result["score"] < min_score:
                continue

            # Year filter
            if year_filter is not None:
                paper_year = meta.get("year")
                if paper_year is not None and paper_year < year_filter:
                    continue

            # Category filter (smart matching: "csai" matches "cs.AI")
            if category_filter is not None:
                paper_cats = meta.get("categories", "")
                if not _category_matches(category_filter, paper_cats):
                    continue

            # Author filter
            if author_filter is not None:
                paper_authors = meta.get("authors", "")
                if author_filter.lower() not in paper_authors.lower():
                    continue

            filtered.append(result)

        # Re-rank and trim to top_k
        results = filtered[:top_k]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        logger.info(
            f"Query: '{query[:50]}...' → {len(results)} results "
            f"(from {len(raw_results)} candidates)"
        )

        return results

    def search_with_comparison(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search and return both semantic and keyword-based results for comparison.

        Args:
            query: Search query.
            top_k: Number of results per method.

        Returns:
            Dict with 'semantic' and 'keyword' result lists.
        """
        semantic_results = self.search(query, top_k=top_k)

        # Simple keyword matching as baseline
        keyword_results = self._keyword_search(query, top_k=top_k)

        return {
            "semantic": semantic_results,
            "keyword": keyword_results,
        }

    def _keyword_search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search (checks if query terms appear in text).
        Used as a baseline comparison to demonstrate embedding superiority.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts.
        """
        query_terms = query.lower().split()
        scored = []

        for meta in self.vector_store.metadata:
            text = meta.get("text", meta.get("title", "") + " " + meta.get("abstract", ""))
            text_lower = text.lower()

            # Count term matches
            matches = sum(1 for term in query_terms if term in text_lower)
            if matches > 0:
                score = matches / len(query_terms)
                scored.append({"metadata": meta, "score": score, "rank": 0})

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        results = scored[:top_k]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific paper by its arXiv ID.

        Args:
            paper_id: The arXiv paper ID.

        Returns:
            Paper metadata dict or None if not found.
        """
        for meta in self.vector_store.metadata:
            if meta.get("id") == paper_id:
                return meta
        return None

    @property
    def num_papers(self) -> int:
        """Number of indexed papers."""
        return self.vector_store.size



def _category_matches(user_input: str, paper_categories: str) -> bool:
    """
    Smart category matching that handles common shorthand.
    e.g., 'csai' matches 'cs.AI', 'statML' matches 'stat.ML', etc.
    """
    # Normalize: strip dots, dashes, spaces and lowercase
    def normalize(s: str) -> str:
        return re.sub(r'[.\-\s]', '', s).lower()

    user_norm = normalize(user_input)

    # Check each category in the paper's category list
    for cat in paper_categories.split():
        if user_norm in normalize(cat) or normalize(cat) in user_norm:
            return True

    # Also try direct substring match on original (for exact matches like "cs.AI")
    if user_input.lower() in paper_categories.lower():
        return True

    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Retriever module loaded. Use SemanticRetriever class for searching.")
