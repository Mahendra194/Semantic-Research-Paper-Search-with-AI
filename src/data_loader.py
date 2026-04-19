"""
Data Loader Module
==================
Efficiently loads and processes the arXiv metadata JSON dataset.
Handles the large file (5+ GB) using streaming with ijson for memory efficiency.
"""

import json
import os
import logging
from typing import Optional, List

import ijson
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_arxiv_data(
    filepath: str,
    max_records: Optional[int] = None,
    categories: Optional[List[str]] = None,
    cs_only: bool = True,
) -> pd.DataFrame:
    """
    Load arXiv metadata from JSON file using streaming for memory efficiency.

    Args:
        filepath: Path to the arxiv-metadata-oai-snapshot.json file.
        max_records: Maximum number of records to load (None = all).
        categories: Specific category prefixes to filter (e.g., ['cs.AI', 'cs.CL']).
                    Overrides cs_only if provided.
        cs_only: If True and categories is None, filter to Computer Science papers.

    Returns:
        DataFrame with columns: id, title, abstract, authors, categories, update_date, text
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv"
        )

    records = []
    count = 0

    logger.info(f"Loading arXiv data from {filepath}...")

    # The arxiv JSON file has one JSON object per line
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading papers", unit=" papers"):
            if max_records and count >= max_records:
                break

            try:
                paper = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            paper_categories = paper.get("categories", "")

            # Apply category filter
            if categories:
                if not any(cat in paper_categories for cat in categories):
                    continue
            elif cs_only:
                if not any(
                    paper_categories.startswith(prefix)
                    or f" {prefix}" in paper_categories
                    for prefix in ["cs."]
                ):
                    continue

            # Extract relevant fields
            record = {
                "id": paper.get("id", ""),
                "title": _clean_text(paper.get("title", "")),
                "abstract": _clean_text(paper.get("abstract", "")),
                "authors": paper.get("authors", ""),
                "categories": paper_categories,
                "update_date": paper.get("update_date", ""),
            }

            records.append(record)
            count += 1

    df = pd.DataFrame(records)

    # Combine title + abstract for embedding
    df["text"] = df["title"] + " " + df["abstract"]

    # Extract primary category
    df["primary_category"] = df["categories"].apply(
        lambda x: x.split()[0] if isinstance(x, str) and x else ""
    )

    # Extract year from update_date
    df["year"] = pd.to_datetime(df["update_date"], errors="coerce").dt.year

    logger.info(f"Loaded {len(df)} papers.")
    return df


def _clean_text(text: str) -> str:
    """Remove extra whitespace and newlines from text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed DataFrame to a Parquet file for fast reloading.

    Args:
        df: Processed DataFrame.
        output_path: Path to save the Parquet file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path} ({len(df)} records)")


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load a previously processed DataFrame from Parquet.

    Args:
        filepath: Path to the Parquet file.

    Returns:
        Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data not found at {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded processed data from {filepath} ({len(df)} records)")
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics about the loaded dataset.

    Args:
        df: The papers DataFrame.

    Returns:
        Dictionary with dataset statistics.
    """
    stats = {
        "total_papers": len(df),
        "unique_categories": df["primary_category"].nunique(),
        "top_categories": df["primary_category"].value_counts().head(10).to_dict(),
        "year_range": (
            int(df["year"].min()) if df["year"].notna().any() else None,
            int(df["year"].max()) if df["year"].notna().any() else None,
        ),
        "avg_abstract_length": df["abstract"].str.len().mean(),
        "avg_title_length": df["title"].str.len().mean(),
    }
    return stats


if __name__ == "__main__":
    # Quick test / standalone run
    import sys

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/arxiv_metadata.json"
    df = load_arxiv_data(path, max_records=1000)
    print(f"\nLoaded {len(df)} papers")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample paper:\n{df.iloc[0].to_dict()}")
    print(f"\nStats: {get_dataset_stats(df)}")
