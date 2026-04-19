"""
Streamlit App — Semantic Research Paper Search & Q&A
=====================================================
Multi-tab interface:
  1. Semantic Search: query → ranked paper cards with similarity scores + paper selection
  2. Ask Questions (RAG): natural language Q&A with source citations
  3. Live arXiv Search: real-time search across all arXiv papers
"""

import os
import sys
import logging

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Auto-load .env file (picks up GEMINI_API_KEY, OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed, keys must be set in environment

# Load Streamlit Cloud secrets into environment variables
# (This is how secrets work on Streamlit Cloud — they appear in st.secrets)
try:
    if hasattr(st, "secrets"):
        for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "HF_REPO_ID"]:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = st.secrets[key]
except Exception:
    pass

from src.utils import setup_logging, get_arxiv_url, format_authors_short, truncate_text
from src.embedder import EmbeddingModel, DEFAULT_MODEL
from src.vector_store import FAISSVectorStore
from src.retriever import SemanticRetriever
from src.rag_pipeline import RAGPipeline
from src.arxiv_search import search_arxiv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Auto-download index from Hugging Face Hub
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def download_index_from_hf():
    """Download the FAISS index and data from Hugging Face Hub if not present locally."""
    index_dir = os.path.join(PROJECT_ROOT, "data", "index")
    index_file = os.path.join(index_dir, "faiss_index.index")

    if os.path.exists(index_file):
        return True  # Already downloaded

    repo_id = os.environ.get("HF_REPO_ID", "")
    token = os.environ.get("HF_TOKEN", "")

    if not repo_id:
        return False  # No repo configured

    try:
        from huggingface_hub import hf_hub_download

        os.makedirs(index_dir, exist_ok=True)
        data_dir = os.path.join(PROJECT_ROOT, "data")

        files_to_download = [
            ("index/faiss_index.index", os.path.join(index_dir, "faiss_index.index")),
            ("index/faiss_index_config.json", os.path.join(index_dir, "faiss_index_config.json")),
            ("index/faiss_index_metadata.pkl", os.path.join(index_dir, "faiss_index_metadata.pkl")),
            ("papers_processed.parquet", os.path.join(data_dir, "papers_processed.parquet")),
        ]

        progress = st.progress(0, text="📥 Downloading search index from cloud...")
        for i, (repo_path, local_path) in enumerate(files_to_download):
            if not os.path.exists(local_path):
                progress.progress(
                    (i + 1) / len(files_to_download),
                    text=f"📥 Downloading {repo_path}...",
                )
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    token=token if token else None,
                    local_dir=data_dir,
                )
                # hf_hub_download may place file in a subfolder; move if needed
                if downloaded != local_path and os.path.exists(downloaded):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    if not os.path.exists(local_path):
                        os.rename(downloaded, local_path)

        progress.progress(1.0, text="✅ Index downloaded successfully!")
        import time
        time.sleep(1)
        progress.empty()
        return True

    except Exception as e:
        logger.error(f"Failed to download index from HF: {e}")
        st.warning(f"⚠️ Could not download index: {e}")
        return False

# ──────────────────────────────────────────────
#  Page Config & Styling
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="📚 Research Paper Search & Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .paper-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .paper-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .paper-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e0e7ff;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .paper-title a {
        color: #e0e7ff;
        text-decoration: none;
    }
    .paper-title a:hover {
        color: #a5b4fc;
        text-decoration: underline;
    }
    .paper-authors {
        color: #a5b4fc;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .paper-abstract {
        color: #c7d2fe;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 12px 0;
    }
    .category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .live-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .score-bar-container {
        margin-top: 8px;
    }
    .score-text {
        color: #94a3b8;
        font-size: 0.8rem;
    }
    .score-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin-top: 4px;
    }
    .score-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
    }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 12px;
        flex-shrink: 0;
    }
    .paper-links {
        display: flex;
        gap: 16px;
        align-items: center;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .paper-links a {
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .link-arxiv {
        color: #818cf8;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .link-arxiv:hover {
        background: rgba(99, 102, 241, 0.2);
    }
    .link-pdf {
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .link-pdf:hover {
        background: rgba(16, 185, 129, 0.2);
    }
    .hero-title {
        text-align: center;
        background: linear-gradient(135deg, #c7d2fe, #a5b4fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 32px;
    }
    .answer-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        line-height: 1.7;
        color: #e0e7ff;
    }
    .stats-strip {
        display: flex;
        gap: 24px;
        justify-content: center;
        margin: 16px 0;
    }
    .stat-item { text-align: center; }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #818cf8;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .llm-status-ok {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 8px 0 16px 0;
        color: #6ee7b7;
        font-size: 0.85rem;
    }
    .llm-status-warn {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 8px 0 16px 0;
        color: #fcd34d;
        font-size: 0.85rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 24px;
        color: #c7d2fe;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def render_paper_card(paper: dict, score: float, rank: int, show_score: bool = True):
    """Render a paper result as a styled card using clean HTML."""
    title = paper.get("title", "Untitled")
    authors = format_authors_short(paper.get("authors", ""), max_authors=4)
    abstract = paper.get("abstract", "")
    categories = paper.get("categories", "").split()
    paper_id = paper.get("id", "")
    year = paper.get("year", "")
    source = paper.get("source", "local")
    arxiv_url = paper.get("arxiv_url", get_arxiv_url(paper_id))
    pdf_url = paper.get("pdf_url", f"https://arxiv.org/pdf/{paper_id}")

    badges = "".join(f'<span class="category-badge">{cat}</span>' for cat in categories[:4])

    source_badge = ""
    if source == "arxiv_live":
        source_badge = '<span class="live-badge">🌐 Live</span> '

    score_html = ""
    if show_score and score > 0:
        score_pct = max(0, min(100, score * 100))
        score_html = (
            '<div class="score-bar-container">'
            f'<div class="score-text">Similarity: {score:.4f}</div>'
            f'<div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>'
            '</div>'
        )

    abstract_display = truncate_text(abstract, 400)

    html = (
        '<div class="paper-card">'
        '<div style="display:flex;align-items:flex-start;">'
        f'<span class="rank-badge">{rank}</span>'
        '<div style="flex:1;">'
        f'<div class="paper-title"><a href="{arxiv_url}" target="_blank">{title}</a></div>'
        f'<div class="paper-authors">👤 {authors} &nbsp;|&nbsp; 📅 {year}</div>'
        f'<div>{source_badge}{badges}</div>'
        f'<div class="paper-abstract">{abstract_display}</div>'
        f'{score_html}'
        '<div class="paper-links">'
        f'<a href="{arxiv_url}" target="_blank" class="link-arxiv">📄 View on arXiv</a>'
        f'<a href="{pdf_url}" target="_blank" class="link-pdf">📥 Download PDF</a>'
        '</div>'
        '</div></div></div>'
    )

    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str) -> EmbeddingModel:
    """Load and cache the embedding model."""
    return EmbeddingModel(model_name)


@st.cache_resource(show_spinner="Loading vector index...")
def load_vector_store(index_dir: str) -> FAISSVectorStore:
    """Load and cache the FAISS vector store."""
    return FAISSVectorStore.load_index(index_dir)


def get_index_dir() -> str:
    """Get the default index directory."""
    return os.path.join(PROJECT_ROOT, "data", "index")


def check_index_exists() -> bool:
    """Check if a pre-built index exists. If not, try downloading from HF Hub."""
    index_dir = get_index_dir()
    if os.path.exists(os.path.join(index_dir, "faiss_index.index")):
        return True
    # Try auto-downloading from Hugging Face
    return download_index_from_hf()


def check_llm_available() -> tuple:
    """Check if any LLM backend is available. Returns (is_available, backend_name)."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai
            return True, "Google Gemini"
        except ImportError:
            return False, "google-genai not installed"
    if openai_key:
        return True, "OpenAI"
    return False, "No API key configured"


# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Search parameters
    top_k = st.slider("Number of Results (K)", min_value=1, max_value=20, value=5)

    # Optional filters
    st.markdown("### 🔍 Filters")
    year_filter = st.number_input(
        "Papers from year ≥",
        min_value=1990,
        max_value=2026,
        value=2000,
        step=1,
        help="Only show papers updated from this year onwards",
    )
    year_filter_enabled = st.checkbox("Enable year filter", value=False)

    category_filter = st.text_input(
        "Category filter (e.g., cs.AI)",
        value="",
        help="Filter results by arXiv category",
    )

    st.divider()

    # RAG settings
    st.markdown("### 🤖 RAG Settings")

    llm_available, llm_backend = check_llm_available()
    if llm_available:
        st.markdown(
            f'<div class="llm-status-ok">✅ LLM Ready — {llm_backend}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="llm-status-warn">⚠️ {llm_backend}. '
            f'Add GEMINI_API_KEY to .env for AI answers.</div>',
            unsafe_allow_html=True,
        )

    temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        help="Lower = more focused answers",
    )
    max_tokens = st.slider("Max Response Tokens", 256, 4096, 2048, 256)

    st.divider()

    # Live search settings
    st.markdown("### 🌐 Live Search Settings")
    live_max_results = st.slider(
        "Max live results", min_value=5, max_value=30, value=10,
        help="Max papers to fetch from arXiv API",
    )
    live_sort_by = st.selectbox(
        "Sort live results by",
        options=["relevance", "lastUpdatedDate", "submittedDate"],
        index=0,
    )

    st.divider()

    # Comparison toggle
    show_keyword_comparison = st.checkbox(
        "🔄 Compare with Keyword Search", value=False,
        help="Show keyword-based results side by side",
    )

    enable_csv_export = st.checkbox("📥 Enable CSV Export", value=False)

    st.divider()
    st.markdown("### 📊 Index Status")
    if check_index_exists():
        st.success("✅ Index loaded and ready!")
    else:
        st.warning(
            "⚠️ No index found. Run notebook `02_embeddings_and_index.ipynb` "
            "to build the index first."
        )


# ──────────────────────────────────────────────
#  Main Content
# ──────────────────────────────────────────────

st.markdown(
    '<div class="hero-title">🔬 Research Paper Search & Q&A</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    "Semantic search over arXiv papers powered by embeddings & RAG"
    "</div>",
    unsafe_allow_html=True,
)

# Tabs (3 tabs — no About)
tab_search, tab_rag, tab_live = st.tabs(
    ["🔍 Semantic Search", "💬 Ask Questions (RAG)", "🌐 Live arXiv Search"]
)


# ── Tab 1: Semantic Search ──────────────────

with tab_search:
    query = st.text_input(
        "Search for research papers",
        placeholder="e.g., attention mechanisms in transformer architectures",
        key="search_query",
    )

    if query:
        if not check_index_exists():
            st.error(
                "⚠️ Vector index not found. Please build the index first by running "
                "the `02_embeddings_and_index.ipynb` notebook."
            )
        else:
            with st.spinner("🔍 Searching..."):
                try:
                    embedder = load_embedder(DEFAULT_MODEL)
                    store = load_vector_store(get_index_dir())
                    retriever = SemanticRetriever(embedder, store)

                    yr = year_filter if year_filter_enabled else None
                    cat = category_filter if category_filter else None

                    if show_keyword_comparison:
                        comparison = retriever.search_with_comparison(query, top_k=top_k)
                        semantic_results = comparison["semantic"]
                        keyword_results = comparison["keyword"]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 🧠 Semantic Results")
                            for r in semantic_results:
                                render_paper_card(r["metadata"], r["score"], r["rank"])
                        with col2:
                            st.markdown("### 🔤 Keyword Results")
                            if keyword_results:
                                for r in keyword_results:
                                    render_paper_card(r["metadata"], r["score"], r["rank"])
                            else:
                                st.info("No keyword matches found.")
                    else:
                        results = retriever.search(
                            query, top_k=top_k, year_filter=yr, category_filter=cat
                        )

                        if results:
                            # Stats strip
                            avg_score = np.mean([r["score"] for r in results])
                            st.markdown(f"""
                            <div class="stats-strip">
                                <div class="stat-item">
                                    <div class="stat-value">{len(results)}</div>
                                    <div class="stat-label">Papers Found</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{avg_score:.3f}</div>
                                    <div class="stat-label">Avg Similarity</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{results[0]['score']:.3f}</div>
                                    <div class="stat-label">Top Score</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Paper selection with checkboxes
                            st.markdown("#### Select papers to ask AI about specific ones:")
                            selected_indices = []
                            for r in results:
                                rank = r["rank"]
                                title_short = r["metadata"].get("title", "Untitled")[:80]
                                is_selected = st.checkbox(
                                    f"#{rank} — {title_short}",
                                    value=False,
                                    key=f"select_paper_{rank}",
                                )
                                if is_selected:
                                    selected_indices.append(rank - 1)
                                render_paper_card(r["metadata"], r["score"], r["rank"])

                            # Ask AI about selected papers
                            st.divider()
                            st.markdown("### 💬 Ask AI About Selected Papers")
                            if selected_indices:
                                st.info(f"📌 {len(selected_indices)} paper(s) selected")
                            else:
                                st.caption("Select papers above, or AI will use all results.")

                            selected_question = st.text_area(
                                "Ask a question about your selected papers",
                                placeholder="e.g., Compare the methods used in these papers",
                                height=80,
                                key="selected_papers_question",
                            )

                            if st.button("🚀 Ask AI", type="primary", key="selected_rag_btn"):
                                if not selected_question:
                                    st.warning("Please enter a question.")
                                elif not llm_available:
                                    st.error(
                                        "⚠️ No LLM configured. Add `GEMINI_API_KEY` "
                                        "to your `.env` file."
                                    )
                                else:
                                    # Pick selected papers or all
                                    if selected_indices:
                                        chosen = [results[i] for i in selected_indices]
                                    else:
                                        chosen = results

                                    papers_for_rag = [r["metadata"] for r in chosen]

                                    with st.spinner("🤔 Generating answer..."):
                                        try:
                                            rag = RAGPipeline(
                                                retriever,
                                                max_new_tokens=max_tokens,
                                                temperature=temperature,
                                            )
                                            answer_result = rag.answer_from_papers(
                                                selected_question, papers_for_rag
                                            )

                                            st.markdown("### 💡 Answer")
                                            st.markdown(
                                                f'<div class="answer-card">{answer_result["answer"]}</div>',
                                                unsafe_allow_html=True,
                                            )

                                            if answer_result["sources"]:
                                                st.markdown("### 📚 Papers Used")
                                                for j, src in enumerate(answer_result["sources"], 1):
                                                    with st.expander(
                                                        f"#{j} — {src['title'][:80]}..."
                                                    ):
                                                        st.markdown(f"**Authors:** {src['authors']}")
                                                        st.markdown(f"**Categories:** {src['categories']}")
                                                        st.markdown(
                                                            f"**arXiv:** [{src['arxiv_id']}]({src['arxiv_url']})"
                                                        )
                                        except Exception as e:
                                            st.error(f"AI answer error: {e}")
                                            logger.exception("Selected RAG failed")

                            # CSV Export
                            if enable_csv_export:
                                export_data = []
                                for r in results:
                                    m = r["metadata"]
                                    export_data.append({
                                        "rank": r["rank"],
                                        "score": r["score"],
                                        "id": m.get("id", ""),
                                        "title": m.get("title", ""),
                                        "authors": m.get("authors", ""),
                                        "categories": m.get("categories", ""),
                                        "year": m.get("year", ""),
                                        "abstract": m.get("abstract", ""),
                                        "url": get_arxiv_url(m.get("id", "")),
                                    })
                                csv_df = pd.DataFrame(export_data)
                                st.download_button(
                                    "📥 Download Results as CSV",
                                    csv_df.to_csv(index=False),
                                    file_name="search_results.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.info("No results found. Try a different query or adjust filters.")

                except Exception as e:
                    st.error(f"Search error: {e}")
                    logger.exception("Search failed")


# ── Tab 2: RAG Q&A ──────────────────────────

with tab_rag:
    question = st.text_area(
        "Ask a question about research papers",
        placeholder="e.g., What are the main approaches to few-shot learning in NLP?",
        height=100,
        key="rag_question",
    )

    if st.button("🚀 Get Answer", type="primary", key="rag_btn"):
        if not question:
            st.warning("Please enter a question.")
        elif not check_index_exists():
            st.error("⚠️ Vector index not found. Build it first using notebook 02.")
        else:
            with st.spinner("🤔 Retrieving papers and generating answer..."):
                try:
                    embedder = load_embedder(DEFAULT_MODEL)
                    store = load_vector_store(get_index_dir())
                    retriever = SemanticRetriever(embedder, store)
                    rag = RAGPipeline(
                        retriever,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )

                    yr = year_filter if year_filter_enabled else None
                    cat = category_filter if category_filter else None

                    result = rag.answer(
                        question,
                        top_k=top_k,
                        year_filter=yr,
                        category_filter=cat,
                    )

                    st.markdown("### 💡 Answer")
                    st.markdown(
                        f'<div class="answer-card">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    if result["sources"]:
                        st.markdown("### 📚 Source Papers")
                        for i, src in enumerate(result["sources"], 1):
                            with st.expander(
                                f"#{i} — {src['title'][:80]}... (score: {src['score']:.4f})"
                            ):
                                st.markdown(f"**Authors:** {src['authors']}")
                                st.markdown(f"**Categories:** {src['categories']}")
                                st.markdown(f"**arXiv:** [{src['arxiv_id']}]({src['arxiv_url']})")

                except Exception as e:
                    st.error(f"RAG error: {e}")
                    logger.exception("RAG failed")


# ── Tab 3: Live arXiv Search ──────────────────

with tab_live:
    st.markdown("""
    <div style="color: #a5b4fc; font-size: 0.95rem; margin-bottom: 16px;">
        🌐 Search <strong>all of arXiv</strong> in real-time — not limited to your local index.
        Perfect for finding the latest papers or topics outside your indexed dataset.
    </div>
    """, unsafe_allow_html=True)

    col_search, col_cat = st.columns([3, 1])
    with col_search:
        live_query = st.text_input(
            "Search arXiv papers",
            placeholder="e.g., large language model alignment RLHF",
            key="live_search_query",
        )
    with col_cat:
        live_category = st.text_input(
            "Category (optional)",
            placeholder="e.g., cs.CL",
            key="live_category",
            help="arXiv category like cs.AI, cs.CL, stat.ML, etc.",
        )

    if live_query:
        with st.spinner("🌐 Searching arXiv..."):
            try:
                cat_filter = live_category.strip() if live_category else None
                result = search_arxiv(
                    query=live_query,
                    max_results=live_max_results,
                    sort_by=live_sort_by,
                    category=cat_filter,
                )

                if "error" in result:
                    st.error(f"arXiv API error: {result['error']}")
                elif result["papers"]:
                    st.markdown(f"""
                    <div class="stats-strip">
                        <div class="stat-item">
                            <div class="stat-value">{len(result['papers'])}</div>
                            <div class="stat-label">Papers Shown</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{result['total_results']:,}</div>
                            <div class="stat-label">Total on arXiv</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">Live</div>
                            <div class="stat-label">Source</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Paper selection for live results
                    st.markdown("#### Select papers to ask AI about specific ones:")
                    live_selected_indices = []
                    for i, paper in enumerate(result["papers"]):
                        title_short = paper.get("title", "Untitled")[:80]
                        is_selected = st.checkbox(
                            f"#{i+1} — {title_short}",
                            value=False,
                            key=f"live_select_paper_{i}",
                        )
                        if is_selected:
                            live_selected_indices.append(i)
                        render_paper_card(paper, score=0.0, rank=i + 1, show_score=False)

                    # Ask AI about live results
                    st.divider()
                    st.markdown("### 💬 Ask AI About These Papers")
                    if live_selected_indices:
                        st.info(f"📌 {len(live_selected_indices)} paper(s) selected")
                    else:
                        st.caption("Select papers above, or AI will use all results.")

                    live_question = st.text_area(
                        "Ask a question about the papers found above",
                        placeholder="e.g., What are the key techniques discussed in these papers?",
                        height=80,
                        key="live_rag_question",
                    )

                    if st.button("🚀 Get AI Answer", type="primary", key="live_rag_btn"):
                        if not live_question:
                            st.warning("Please enter a question.")
                        elif not llm_available:
                            st.error(
                                "⚠️ No LLM configured. Add `GEMINI_API_KEY` to your "
                                "`.env` file to enable AI answers."
                            )
                        else:
                            # Pick selected papers or all
                            if live_selected_indices:
                                chosen_papers = [result["papers"][i] for i in live_selected_indices]
                            else:
                                chosen_papers = result["papers"]

                            with st.spinner("🤔 Generating answer from live papers..."):
                                try:
                                    embedder = load_embedder(DEFAULT_MODEL)
                                    store = load_vector_store(get_index_dir())
                                    retriever = SemanticRetriever(embedder, store)
                                    rag = RAGPipeline(
                                        retriever,
                                        max_new_tokens=max_tokens,
                                        temperature=temperature,
                                    )

                                    answer_result = rag.answer_from_papers(
                                        live_question, chosen_papers,
                                    )

                                    st.markdown("### 💡 Answer")
                                    st.markdown(
                                        f'<div class="answer-card">{answer_result["answer"]}</div>',
                                        unsafe_allow_html=True,
                                    )

                                    if answer_result["sources"]:
                                        st.markdown("### 📚 Referenced Papers")
                                        for j, src in enumerate(answer_result["sources"], 1):
                                            with st.expander(
                                                f"#{j} — {src['title'][:80]}..."
                                            ):
                                                st.markdown(f"**Authors:** {src['authors']}")
                                                st.markdown(f"**Categories:** {src['categories']}")
                                                st.markdown(
                                                    f"**arXiv:** [{src['arxiv_id']}]({src['arxiv_url']})"
                                                )

                                except Exception as e:
                                    st.error(f"AI answer error: {e}")
                                    logger.exception("Live RAG failed")

                    # CSV Export for live results
                    if enable_csv_export:
                        export_data = []
                        for i, p in enumerate(result["papers"], 1):
                            export_data.append({
                                "rank": i,
                                "id": p.get("id", ""),
                                "title": p.get("title", ""),
                                "authors": p.get("authors", ""),
                                "categories": p.get("categories", ""),
                                "year": p.get("year", ""),
                                "published": p.get("published", ""),
                                "abstract": p.get("abstract", ""),
                                "url": p.get("arxiv_url", ""),
                                "pdf": p.get("pdf_url", ""),
                            })
                        csv_df = pd.DataFrame(export_data)
                        st.download_button(
                            "📥 Download Live Results as CSV",
                            csv_df.to_csv(index=False),
                            file_name="arxiv_live_results.csv",
                            mime="text/csv",
                            key="live_csv_download",
                        )

                else:
                    st.info(
                        "No results found on arXiv. Try different keywords or "
                        "remove the category filter."
                    )

            except Exception as e:
                st.error(f"Live search error: {e}")
                logger.exception("Live arXiv search failed")
