"""
Vector Store Module
===================
Manages FAISS-based vector indices for fast approximate nearest neighbor search.
Supports multiple index types, metadata mapping, and persistence.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-backed vector store with metadata mapping.

    Supports:
      - IndexFlatIP (exact inner product — best for small/medium datasets)
      - IndexIVFFlat (approximate — faster for large datasets)

    Usage:
        store = FAISSVectorStore(dimension=384)
        store.build_index(embeddings, metadata_list)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, dimension: int):
        """
        Initialize the vector store.

        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_type: str = ""

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        index_type: str = "flat",
        nlist: int = 100,
    ) -> None:
        """
        Build a FAISS index from embeddings.

        Args:
            embeddings: Numpy array of shape (n, dimension).
            metadata: List of metadata dicts, one per embedding.
            index_type: 'flat' for IndexFlatIP, 'ivf' for IndexIVFFlat.
            nlist: Number of Voronoi cells for IVF index.
        """
        n, d = embeddings.shape
        assert d == self.dimension, (
            f"Embedding dim {d} doesn't match store dim {self.dimension}"
        )
        assert len(metadata) == n, (
            f"Metadata length {len(metadata)} doesn't match embeddings length {n}"
        )

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index_type = "flat"
            logger.info("Building IndexFlatIP (exact search)...")
        elif index_type == "ivf":
            # For IVF, nlist should be < n
            actual_nlist = min(nlist, n // 10, n)
            actual_nlist = max(actual_nlist, 1)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index_type = "ivf"
            logger.info(f"Building IndexIVFFlat with nlist={actual_nlist}...")
            # IVF requires training
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {index_type}. Use 'flat' or 'ivf'.")

        self.index.add(embeddings)
        self.metadata = metadata

        logger.info(
            f"Index built: {self.index.ntotal} vectors, type={self.index_type}"
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the index for nearest neighbors.

        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: 'metadata', 'score', 'rank'.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Set nprobe for IVF index (search more cells for better recall)
        if self.index_type == "ivf":
            self.index.nprobe = min(10, self.index.ntotal)

        # Clamp top_k to available vectors
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for unfilled spots
                continue
            result = {
                "metadata": self.metadata[idx],
                "score": float(score),
                "rank": rank + 1,
            }
            results.append(result)

        return results

    def save_index(self, directory: str, name: str = "faiss_index") -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            directory: Directory to save files in.
            name: Base name for the saved files.
        """
        if self.index is None:
            raise RuntimeError("No index to save.")

        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        faiss.write_index(self.index, index_path)

        # Save metadata
        meta_path = os.path.join(directory, f"{name}_metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        # Save config
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_vectors": self.index.ntotal,
        }
        config_path = os.path.join(directory, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Index saved to {directory}/{name}*")

    @classmethod
    def load_index(cls, directory: str, name: str = "faiss_index") -> "FAISSVectorStore":
        """
        Load a FAISS index and metadata from disk.

        Args:
            directory: Directory containing saved files.
            name: Base name of the saved files.

        Returns:
            FAISSVectorStore instance.
        """
        # Load config
        config_path = os.path.join(directory, f"{name}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        store = cls(dimension=config["dimension"])
        store.index_type = config["index_type"]

        # Load FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        store.index = faiss.read_index(index_path)

        # Load metadata
        meta_path = os.path.join(directory, f"{name}_metadata.pkl")
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)

        logger.info(
            f"Index loaded: {store.index.ntotal} vectors, type={store.index_type}"
        )
        return store

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick sanity check
    dim = 384
    n = 100
    np.random.seed(42)
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize for inner product similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = [{"id": str(i), "title": f"Paper {i}"} for i in range(n)]

    store = FAISSVectorStore(dimension=dim)
    store.build_index(embeddings, metadata)

    query = embeddings[0:1]
    results = store.search(query, top_k=3)
    print("Search results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['metadata']['title']} (score: {r['score']:.4f})")
