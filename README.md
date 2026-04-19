# 🔬 Semantic Research Paper Search & Q&A System

> **ML Class Project 2026** — Semantic search + RAG over arXiv papers using embeddings, FAISS, and LLMs.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4285F4)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

A web application that enables:
- **Semantic Search** — Find research papers by meaning, not just keywords
- **RAG Q&A** — Ask natural language questions and get answers grounded in papers
- **Explainability** — See similarity scores explaining why each paper was retrieved
- **Comparison** — Side-by-side semantic vs. keyword search to demonstrate embedding superiority

### ML Concepts Demonstrated
| Concept | Implementation |
|---------|---------------|
| Dense Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector Search | FAISS IndexFlatIP / IndexIVFFlat |
| Retrieval Evaluation | Precision@K, Recall@K, NDCG, MRR, MAP |
| RAG Pipeline | Retrieved context + LLM (Phi-3-mini / Mistral-7B) |
| Baseline Comparison | TF-IDF + cosine similarity |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│   Embedding Model    │  sentence-transformers
│   (all-MiniLM-L6-v2) │  384-dim vectors
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    FAISS Index        │  Inner product search
│    (50k+ vectors)     │  < 10ms query time
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Retriever        │  Top-K + metadata filters
│  (year, category)    │  (year, category)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌──────────┐
│ Search  │  │ RAG Q&A  │
│  Tab    │  │   Tab    │
│(papers) │  │(answers) │
└─────────┘  └──────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd semantic-research-paper-rag
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [arXiv Metadata Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle.

Place `arxiv-metadata-oai-snapshot.json` in the **parent directory** of this project (or update the path in the notebooks).

### 3. Run Notebooks (in order)

```bash
# Step 1: EDA and data preparation
jupyter notebook notebooks/01_eda_and_data_prep.ipynb

# Step 2: Generate embeddings and build FAISS index
jupyter notebook notebooks/02_embeddings_and_index.ipynb

# Step 3: Evaluate retrieval quality (optional but recommended)
jupyter notebook notebooks/03_evaluation.ipynb
```

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

### 5. (Optional) Set up RAG with LLM

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
# HF_TOKEN=hf_xxxxxxxxxxxxx
```

---

## 📁 Project Structure

```
semantic-research-paper-rag/
├── data/                          # (gitignored) data & indices
│   ├── papers_processed.parquet   # Processed dataset
│   ├── embeddings.npy             # Pre-computed embeddings
│   └── index/                     # FAISS index files
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb # Exploratory Data Analysis
│   ├── 02_embeddings_and_index.ipynb  # Build embeddings & index
│   └── 03_evaluation.ipynb       # Retrieval evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Efficient JSON loading & filtering
│   ├── embedder.py                # SentenceTransformer wrapper
│   ├── vector_store.py            # FAISS index management
│   ├── retriever.py               # Semantic search with filters
│   ├── rag_pipeline.py            # RAG: retrieval + LLM generation
│   └── utils.py                   # Helpers & formatting
├── app/
│   └── streamlit_app.py           # Web demo (Search + Q&A)
├── evaluation/
│   └── metrics.py                 # P@K, R@K, NDCG, TF-IDF baseline
├── tests/
│   └── test_retrieval.py          # Unit tests (synthetic data)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Results & Evaluation

### Metrics Computed
| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of all relevant docs found in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain |
| **MRR** | Mean Reciprocal Rank |
| **MAP** | Mean Average Precision |

### Semantic Search vs TF-IDF

The evaluation notebook (03) demonstrates that semantic embeddings outperform TF-IDF:

- **Handles synonyms**: "make models smaller" → finds model compression papers
- **Conceptual matching**: "teaching computers language" → finds NLP papers
- **Paraphrase understanding**: Different wordings find the same relevant research

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Embeddings | sentence-transformers | ≥2.7.0 |
| Vector DB | FAISS (faiss-cpu) | ≥1.8.0 |
| RAG LLM | Phi-3-mini / Mistral-7B | via HuggingFace |
| Frontend | Streamlit | ≥1.35.0 |
| Data | pandas + ijson | streaming |
| Evaluation | scikit-learn + custom | P@K, NDCG |

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_retrieval.py -v
```

Tests use synthetic data — no dataset download required.

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Uses only abstracts + titles (no full paper text)
- RAG requires a GPU for reasonable speed with larger models
- Category-based evaluation ground truth is a proxy

### Future Improvements
- **Full-text indexing** with chunking strategies
- **Hybrid search** (combining semantic + keyword)
- **Re-ranking** with cross-encoder models
- **Citation graph** integration for better recommendations
- **Multi-modal** search (figures, tables, equations)
- **Fine-tuned embeddings** on scientific text (e.g., SciBERT)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for ML Class 2026
</p>
