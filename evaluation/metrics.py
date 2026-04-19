"""
Evaluation Metrics Module
=========================
Provides retrieval quality metrics (Precision@K, Recall@K, NDCG, MRR)
and a TF-IDF baseline for comparing against semantic search.
"""

import logging
from typing import List, Dict, Any, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Core Retrieval Metrics
# ──────────────────────────────────────────────

def precision_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int
) -> float:
    """
    Precision@K: fraction of top-K retrieved docs that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved paper IDs.
        relevant_ids: Set of ground-truth relevant paper IDs.
        k: Cutoff rank.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


def recall_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int
) -> float:
    """
    Recall@K: fraction of relevant docs found in top-K.

    Args:
        retrieved_ids: Ordered list of retrieved paper IDs.
        relevant_ids: Set of ground-truth relevant paper IDs.
        k: Cutoff rank.

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved paper IDs.
        relevant_ids: Set of ground-truth relevant paper IDs.
        k: Cutoff rank.

    Returns:
        NDCG@K score (0.0 to 1.0).
    """
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (all relevant docs at the top)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved paper IDs.
        relevant_ids: Set of ground-truth relevant paper IDs.

    Returns:
        MRR score (0.0 to 1.0).
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved_ids: List[str], relevant_ids: Set[str]
) -> float:
    """
    Average Precision for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved paper IDs.
        relevant_ids: Set of ground-truth relevant paper IDs.

    Returns:
        AP score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(relevant_ids)


# ──────────────────────────────────────────────
#  TF-IDF Baseline
# ──────────────────────────────────────────────

class TFIDFBaseline:
    """
    TF-IDF + cosine similarity baseline for comparison with semantic search.

    Usage:
        baseline = TFIDFBaseline()
        baseline.fit(documents, doc_ids)
        results = baseline.search("attention mechanism", top_k=5)
    """

    def __init__(self, max_features: int = 50000):
        """
        Args:
            max_features: Maximum vocabulary size for TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.doc_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def fit(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Fit the TF-IDF vectorizer on documents.

        Args:
            documents: List of document texts.
            doc_ids: Corresponding document IDs.
            metadata: Optional list of metadata dicts.
        """
        logger.info(f"Fitting TF-IDF on {len(documents)} documents...")
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.doc_ids = doc_ids
        self.metadata = metadata or [{} for _ in doc_ids]
        logger.info(
            f"TF-IDF matrix shape: {self.tfidf_matrix.shape}, "
            f"vocab size: {len(self.vectorizer.vocabulary_)}"
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using TF-IDF + cosine similarity.

        Args:
            query: Search query text.
            top_k: Number of results.

        Returns:
            List of result dicts with 'metadata', 'score', 'rank'.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "metadata": self.metadata[idx],
                "score": float(scores[idx]),
                "rank": rank + 1,
            })
        return results


# ──────────────────────────────────────────────
#  Full Evaluation Pipeline
# ──────────────────────────────────────────────

def evaluate_retriever(
    retriever,
    test_queries: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """
    Run full evaluation of a retriever against a test set.

    Args:
        retriever: Object with a search(query, top_k) method that returns
                   results with 'metadata' containing 'id'.
        test_queries: List of dicts with 'query' and 'relevant_ids' (set of IDs).
        k_values: List of K values to evaluate.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    all_results = {
        "per_query": [],
        "aggregate": {},
    }

    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        mrrs = []
        aps = []

        for test_item in test_queries:
            query = test_item["query"]
            relevant = set(test_item["relevant_ids"])

            # Run search
            results = retriever.search(query, top_k=max(k_values))
            retrieved_ids = [r["metadata"]["id"] for r in results]

            p = precision_at_k(retrieved_ids, relevant, k)
            r = recall_at_k(retrieved_ids, relevant, k)
            n = ndcg_at_k(retrieved_ids, relevant, k)
            m = mrr(retrieved_ids, relevant)
            ap = average_precision(retrieved_ids, relevant)

            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)
            mrrs.append(m)
            aps.append(ap)

        all_results["aggregate"][f"P@{k}"] = np.mean(precisions)
        all_results["aggregate"][f"R@{k}"] = np.mean(recalls)
        all_results["aggregate"][f"NDCG@{k}"] = np.mean(ndcgs)

    all_results["aggregate"]["MRR"] = np.mean(mrrs)
    all_results["aggregate"]["MAP"] = np.mean(aps)

    return all_results


def compare_with_baseline(
    semantic_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare semantic search metrics against TF-IDF baseline.

    Returns:
        Dict showing improvement (or regression) for each metric.
    """
    comparison = {}
    for metric in semantic_results["aggregate"]:
        sem_val = semantic_results["aggregate"][metric]
        base_val = baseline_results["aggregate"].get(metric, 0.0)
        improvement = sem_val - base_val
        pct_improvement = (
            (improvement / base_val * 100) if base_val > 0 else float("inf")
        )
        comparison[metric] = {
            "semantic": round(sem_val, 4),
            "tfidf_baseline": round(base_val, 4),
            "improvement": round(improvement, 4),
            "pct_improvement": round(pct_improvement, 2),
        }
    return comparison


# Sample test queries for evaluation
SAMPLE_TEST_QUERIES = [
    {
        "query": "attention mechanisms in neural networks",
        "relevant_ids": set(),  # To be filled with actual paper IDs
        "description": "Papers about attention in NNs",
    },
    {
        "query": "generative adversarial networks for image synthesis",
        "relevant_ids": set(),
        "description": "GAN papers for image generation",
    },
    {
        "query": "reinforcement learning for robotics",
        "relevant_ids": set(),
        "description": "RL applied to robotics tasks",
    },
    {
        "query": "natural language processing with transformers",
        "relevant_ids": set(),
        "description": "NLP transformer papers",
    },
    {
        "query": "federated learning privacy preserving",
        "relevant_ids": set(),
        "description": "Federated learning and privacy",
    },
    {
        "query": "graph neural networks for molecular property prediction",
        "relevant_ids": set(),
        "description": "GNN for molecular/chemistry tasks",
    },
    {
        "query": "few-shot learning meta-learning",
        "relevant_ids": set(),
        "description": "Few-shot and meta-learning approaches",
    },
    {
        "query": "object detection in autonomous driving",
        "relevant_ids": set(),
        "description": "Object detection for self-driving cars",
    },
    {
        "query": "knowledge distillation model compression",
        "relevant_ids": set(),
        "description": "Model compression techniques",
    },
    {
        "query": "contrastive learning self-supervised representations",
        "relevant_ids": set(),
        "description": "Self-supervised contrastive learning",
    },
    {
        "query": "neural architecture search automated machine learning",
        "relevant_ids": set(),
        "description": "NAS and AutoML",
    },
    {
        "query": "diffusion models for text to image generation",
        "relevant_ids": set(),
        "description": "Diffusion-based image generation",
    },
    {
        "query": "multi-task learning shared representations",
        "relevant_ids": set(),
        "description": "Multi-task learning approaches",
    },
    {
        "query": "continual learning catastrophic forgetting",
        "relevant_ids": set(),
        "description": "Continual/lifelong learning",
    },
    {
        "query": "vision transformers for image classification",
        "relevant_ids": set(),
        "description": "ViT and vision transformers",
    },
    {
        "query": "large language models alignment safety",
        "relevant_ids": set(),
        "description": "LLM alignment and safety",
    },
    {
        "query": "point cloud processing 3D deep learning",
        "relevant_ids": set(),
        "description": "3D point cloud networks",
    },
    {
        "query": "time series forecasting deep learning",
        "relevant_ids": set(),
        "description": "Deep learning for time series",
    },
    {
        "query": "explainable AI interpretable machine learning",
        "relevant_ids": set(),
        "description": "XAI and interpretability",
    },
    {
        "query": "speech recognition end to end models",
        "relevant_ids": set(),
        "description": "E2E speech recognition",
    },
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick metric test
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "f"}

    print(f"P@5 = {precision_at_k(retrieved, relevant, 5):.4f}")
    print(f"R@5 = {recall_at_k(retrieved, relevant, 5):.4f}")
    print(f"NDCG@5 = {ndcg_at_k(retrieved, relevant, 5):.4f}")
    print(f"MRR = {mrr(retrieved, relevant):.4f}")
    print(f"AP = {average_precision(retrieved, relevant):.4f}")
