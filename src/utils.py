"""
Utility Functions
=================
Common helpers for text processing, formatting, configuration, and logging.
"""

import os
import re
import logging
from typing import Dict, Any, Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the project.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_env() -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logging.warning("python-dotenv not installed. Skipping .env loading.")


def get_hf_token() -> Optional[str]:
    """Get the Hugging Face token from environment."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        logging.warning(
            "HF_TOKEN not set. Some models may not be accessible. "
            "Set it in .env or as an environment variable."
        )
    return token


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, newlines, and special chars.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Remove LaTeX commands (common in arXiv abstracts)
    text = re.sub(r"\$.*?\$", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def get_arxiv_url(paper_id: str) -> str:
    """
    Generate the arXiv URL for a given paper ID.

    Args:
        paper_id: arXiv paper ID (e.g., '2301.12345').

    Returns:
        Full arXiv abstract URL.
    """
    return f"https://arxiv.org/abs/{paper_id}"


def get_arxiv_pdf_url(paper_id: str) -> str:
    """
    Generate the arXiv PDF URL for a given paper ID.

    Args:
        paper_id: arXiv paper ID.

    Returns:
        Full arXiv PDF URL.
    """
    return f"https://arxiv.org/pdf/{paper_id}"


def format_paper_card(paper: Dict[str, Any], score: Optional[float] = None) -> str:
    """
    Format a paper metadata dict into a human-readable card.

    Args:
        paper: Paper metadata dict.
        score: Optional relevance score.

    Returns:
        Formatted string.
    """
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown")
    categories = paper.get("categories", "")
    abstract = paper.get("abstract", "")
    paper_id = paper.get("id", "")
    year = paper.get("year", "")

    # Truncate long abstracts
    if len(abstract) > 300:
        abstract = abstract[:300] + "..."

    # Truncate long author lists
    if len(authors) > 100:
        authors = authors[:100] + "..."

    card = f"""📄 **{title}**
👤 {authors}
🏷️ {categories}  |  📅 {year}
🔗 {get_arxiv_url(paper_id)}"""

    if score is not None:
        card += f"\n📊 Similarity: {score:.4f}"

    card += f"\n\n{abstract}"

    return card


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def format_authors_short(authors: str, max_authors: int = 3) -> str:
    """
    Format author string to show first N authors + 'et al.'

    Args:
        authors: Full author string (comma-separated).
        max_authors: Maximum authors to show.

    Returns:
        Shortened author string.
    """
    if not authors:
        return "Unknown"
    
    # Try to split by common separators
    author_list = [a.strip() for a in re.split(r",\s*and\s*|,\s*|\s+and\s+", authors)]
    author_list = [a for a in author_list if a]

    if len(author_list) <= max_authors:
        return ", ".join(author_list)
    return ", ".join(author_list[:max_authors]) + " et al."


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
