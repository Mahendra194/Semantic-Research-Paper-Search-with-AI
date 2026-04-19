"""
Embedder Module
===============
Generates dense vector embeddings for paper texts using SentenceTransformers.
Supports multiple models and efficient batch processing with progress tracking.
"""

import os
import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Supported embedding models
MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Fast, lightweight model — great balance of speed and quality",
    },
    "nomic-embed-text-v1": {
        "name": "nomic-ai/nomic-embed-text-v1",
        "dim": 768,
        "description": "Higher quality embeddings, larger model",
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "description": "High quality general-purpose embeddings",
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating text embeddings.

    Usage:
        model = EmbeddingModel("all-MiniLM-L6-v2")
        embeddings = model.encode_texts(["hello world", "research paper"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Key from MODELS dict or a full HuggingFace model path.
        """
        if model_name in MODELS:
            self.model_info = MODELS[model_name]
            self.model_path = self.model_info["name"]
            self.dimension = self.model_info["dim"]
        else:
            self.model_path = model_name
            self.model_info = {"name": model_name, "description": "Custom model"}
            self.dimension = None  # Will be determined after loading

        logger.info(f"Loading embedding model: {self.model_path}")
        self.model = SentenceTransformer(self.model_path, trust_remote_code=True)

        # Determine dimension if not known
        if self.dimension is None:
            self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Model loaded. Embedding dimension: {self.dimension}"
        )

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense embeddings.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per batch (tune for GPU memory).
            show_progress: Show a tqdm progress bar.
            normalize: L2-normalize embeddings (required for cosine similarity via dot product).

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        logger.info(
            f"Encoding {len(texts)} texts with batch_size={batch_size}..."
        )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: Search query text.
            normalize: L2-normalize the embedding.

        Returns:
            numpy array of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save embeddings to a .npy file and optional metadata.

    Args:
        embeddings: Embedding array to save.
        output_path: Path for the .npy file.
        metadata: Optional metadata dict (model name, params, etc.).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(f"Saved embeddings to {output_path} (shape: {embeddings.shape})")

    if metadata:
        import json

        meta_path = output_path.replace(".npy", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved embedding metadata to {meta_path}")


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load embeddings from a .npy file.

    Args:
        filepath: Path to the .npy file.

    Returns:
        numpy array of embeddings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    embeddings = np.load(filepath)
    logger.info(f"Loaded embeddings from {filepath} (shape: {embeddings.shape})")
    return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick demo
    model = EmbeddingModel("all-MiniLM-L6-v2")
    sample_texts = [
        "Attention is all you need transformer architecture",
        "Convolutional neural networks for image classification",
        "Reinforcement learning for game playing",
    ]
    embeddings = model.encode_texts(sample_texts, show_progress=False)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Similarity (0,1): {np.dot(embeddings[0], embeddings[1]):.4f}")
    print(f"Similarity (0,2): {np.dot(embeddings[0], embeddings[2]):.4f}")
