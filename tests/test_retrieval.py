"""
Test Suite for Retrieval System
================================
Unit tests using synthetic data — no real arXiv dataset needed.
Tests data loading utilities, embedding, vector store, and retriever.
"""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_papers():
    """Create a small set of synthetic papers."""
    return [
        {
            "id": "2301.00001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new network architecture the Transformer based solely on attention mechanisms.",
            "authors": "Vaswani et al.",
            "categories": "cs.CL cs.AI",
            "update_date": "2023-01-15",
            "text": "Attention Is All You Need We propose a new network architecture the Transformer based solely on attention mechanisms.",
            "primary_category": "cs.CL",
            "year": 2023,
        },
        {
            "id": "2301.00002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT a new language representation model designed to pre-train deep bidirectional transformers.",
            "authors": "Devlin et al.",
            "categories": "cs.CL",
            "update_date": "2023-02-20",
            "text": "BERT: Pre-training of Deep Bidirectional Transformers We introduce BERT a new language representation model designed to pre-train deep bidirectional transformers.",
            "primary_category": "cs.CL",
            "year": 2023,
        },
        {
            "id": "2302.00003",
            "title": "Convolutional Neural Networks for Image Classification",
            "abstract": "Deep convolutional neural networks have achieved remarkable success in visual recognition tasks.",
            "authors": "LeCun et al.",
            "categories": "cs.CV",
            "update_date": "2023-03-10",
            "text": "Convolutional Neural Networks for Image Classification Deep convolutional neural networks have achieved remarkable success in visual recognition tasks.",
            "primary_category": "cs.CV",
            "year": 2023,
        },
        {
            "id": "2303.00004",
            "title": "Reinforcement Learning for Game Playing",
            "abstract": "We present a reinforcement learning agent that achieves superhuman performance in complex games.",
            "authors": "Silver et al.",
            "categories": "cs.AI cs.LG",
            "update_date": "2023-04-05",
            "text": "Reinforcement Learning for Game Playing We present a reinforcement learning agent that achieves superhuman performance in complex games.",
            "primary_category": "cs.AI",
            "year": 2023,
        },
        {
            "id": "2304.00005",
            "title": "Generative Adversarial Networks",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process.",
            "authors": "Goodfellow et al.",
            "categories": "cs.LG cs.AI",
            "update_date": "2023-05-12",
            "text": "Generative Adversarial Networks We propose a new framework for estimating generative models via an adversarial process.",
            "primary_category": "cs.LG",
            "year": 2023,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Create synthetic embeddings (384-dim to match MiniLM)."""
    np.random.seed(42)
    embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def temp_json_file(sample_papers):
    """Create a temporary JSON file with sample papers."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        for paper in sample_papers:
            # Convert to raw format (without derived fields)
            raw = {
                "id": paper["id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "update_date": paper["update_date"],
            }
            f.write(json.dumps(raw) + "\n")
        filepath = f.name
    yield filepath
    os.unlink(filepath)


# ──────────────────────────────────────────────
#  Data Loader Tests
# ──────────────────────────────────────────────

class TestDataLoader:
    """Tests for the data_loader module."""

    def test_load_arxiv_data(self, temp_json_file):
        from src.data_loader import load_arxiv_data

        df = load_arxiv_data(temp_json_file, cs_only=True)
        assert len(df) == 5
        assert "text" in df.columns
        assert "primary_category" in df.columns
        assert "year" in df.columns

    def test_load_with_max_records(self, temp_json_file):
        from src.data_loader import load_arxiv_data

        df = load_arxiv_data(temp_json_file, max_records=2)
        assert len(df) == 2

    def test_load_with_category_filter(self, temp_json_file):
        from src.data_loader import load_arxiv_data

        df = load_arxiv_data(
            temp_json_file, cs_only=False, categories=["cs.CV"]
        )
        assert len(df) >= 1
        assert all("cs.CV" in cat for cat in df["categories"])

    def test_text_field_combined(self, temp_json_file):
        from src.data_loader import load_arxiv_data

        df = load_arxiv_data(temp_json_file, max_records=1)
        row = df.iloc[0]
        assert row["title"] in row["text"]
        assert row["abstract"] in row["text"]

    def test_file_not_found(self):
        from src.data_loader import load_arxiv_data

        with pytest.raises(FileNotFoundError):
            load_arxiv_data("/nonexistent/path.json")

    def test_save_and_load_processed(self, temp_json_file):
        from src.data_loader import (
            load_arxiv_data,
            save_processed_data,
            load_processed_data,
        )

        df = load_arxiv_data(temp_json_file, max_records=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "processed.parquet")
            save_processed_data(df, out_path)
            df_loaded = load_processed_data(out_path)
            assert len(df_loaded) == len(df)
            assert list(df_loaded.columns) == list(df.columns)

    def test_dataset_stats(self, temp_json_file):
        from src.data_loader import load_arxiv_data, get_dataset_stats

        df = load_arxiv_data(temp_json_file)
        stats = get_dataset_stats(df)
        assert stats["total_papers"] == 5
        assert stats["unique_categories"] > 0
        assert stats["avg_abstract_length"] > 0


# ──────────────────────────────────────────────
#  Vector Store Tests
# ──────────────────────────────────────────────

class TestVectorStore:
    """Tests for the vector_store module."""

    def test_build_flat_index(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers, index_type="flat")
        assert store.size == 5

    def test_build_ivf_index(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers, index_type="ivf")
        assert store.size == 5

    def test_search_returns_results(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers)

        query = sample_embeddings[0:1]
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_search_scores_ordered(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers)

        query = sample_embeddings[0:1]
        results = store.search(query, top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_self_search_top_hit(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers)

        # Searching with a paper's own embedding should return itself as top hit
        query = sample_embeddings[2:3]
        results = store.search(query, top_k=1)
        assert results[0]["metadata"]["id"] == sample_papers[2]["id"]

    def test_save_and_load_index(self, sample_papers, sample_embeddings):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        store.build_index(sample_embeddings, sample_papers)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save_index(tmpdir, name="test_index")
            loaded = FAISSVectorStore.load_index(tmpdir, name="test_index")
            assert loaded.size == store.size

            # Verify search works after loading
            query = sample_embeddings[0:1]
            results = loaded.search(query, top_k=3)
            assert len(results) == 3

    def test_dimension_mismatch_raises(self, sample_papers):
        from src.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=384)
        wrong_dim = np.random.randn(5, 128).astype(np.float32)
        with pytest.raises(AssertionError):
            store.build_index(wrong_dim, sample_papers)


# ──────────────────────────────────────────────
#  Evaluation Metrics Tests
# ──────────────────────────────────────────────

class TestMetrics:
    """Tests for the evaluation metrics."""

    def test_precision_at_k(self):
        from evaluation.metrics import precision_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}

        assert precision_at_k(retrieved, relevant, 5) == 2 / 5
        assert precision_at_k(retrieved, relevant, 1) == 1.0
        assert precision_at_k(retrieved, relevant, 2) == 0.5

    def test_recall_at_k(self):
        from evaluation.metrics import recall_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}

        assert recall_at_k(retrieved, relevant, 5) == 2 / 3
        assert recall_at_k(retrieved, relevant, 1) == 1 / 3

    def test_ndcg_at_k(self):
        from evaluation.metrics import ndcg_at_k

        # Perfect ranking: all relevant at top
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert abs(ndcg_at_k(retrieved, relevant, 3) - 1.0) < 1e-6

    def test_mrr(self):
        from evaluation.metrics import mrr

        assert mrr(["a", "b", "c"], {"a"}) == 1.0
        assert mrr(["x", "a", "b"], {"a"}) == 0.5
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_average_precision(self):
        from evaluation.metrics import average_precision

        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        ap = average_precision(retrieved, relevant)
        # AP = (1/1 + 2/3) / 2 = 0.8333...
        assert abs(ap - 0.8333) < 0.01


class TestTFIDFBaseline:
    """Tests for the TF-IDF baseline searcher."""

    def test_fit_and_search(self):
        from evaluation.metrics import TFIDFBaseline

        docs = [
            "attention mechanism in neural networks",
            "convolutional neural networks for images",
            "reinforcement learning for game playing",
        ]
        ids = ["p1", "p2", "p3"]

        baseline = TFIDFBaseline()
        baseline.fit(docs, ids)

        results = baseline.search("attention mechanism", top_k=2)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        # "attention mechanism" should match first doc best
        assert results[0]["score"] > results[1]["score"]


# ──────────────────────────────────────────────
#  Utils Tests
# ──────────────────────────────────────────────

class TestUtils:
    """Tests for utility functions."""

    def test_clean_text(self):
        from src.utils import clean_text

        assert clean_text("hello   world\n\n  test") == "hello world test"
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_get_arxiv_url(self):
        from src.utils import get_arxiv_url

        assert get_arxiv_url("2301.12345") == "https://arxiv.org/abs/2301.12345"

    def test_truncate_text(self):
        from src.utils import truncate_text

        assert truncate_text("hello", 10) == "hello"
        assert truncate_text("hello world this is long", 10).endswith("...")

    def test_format_authors_short(self):
        from src.utils import format_authors_short

        assert format_authors_short("Alice, Bob, Charlie, Dave", max_authors=2) == "Alice, Bob et al."
        assert format_authors_short("Alice, Bob", max_authors=3) == "Alice, Bob"
        assert format_authors_short("") == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
