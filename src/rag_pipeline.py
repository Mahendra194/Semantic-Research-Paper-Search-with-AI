"""
RAG Pipeline Module
===================
Retrieval-Augmented Generation pipeline for answering questions
about research papers using retrieved context + LLM.

Supports:
  1. Google Gemini API (FREE — recommended, no GPU needed)
  2. OpenAI API (GPT-3.5 / GPT-4)
  3. Local HuggingFace model (Phi-3-mini, needs ~8GB RAM)
  4. Extractive fallback (no LLM — just shows abstract snippets)
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .retriever import SemanticRetriever

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are an expert research assistant specializing in scientific literature analysis. Your task is to synthesize information from the provided research paper abstracts to answer the user's question comprehensively.

### Retrieved Research Papers:
{context}

### User Question:
{question}

### Instructions:
1. **Synthesize, don't summarize** — Combine insights from multiple papers into a coherent, detailed answer. Do NOT just list what each paper says individually.
2. **Cite your sources** — When referencing a finding or claim, mention the paper title in parentheses, e.g., "... as demonstrated in (Attention Is All You Need)."
3. **Be thorough** — Provide a detailed answer with at minimum 3-5 sentences. Cover:
   - The main approaches/methods relevant to the question
   - Key findings or results
   - How different papers relate to or build upon each other
   - Any trade-offs, limitations, or open challenges mentioned
4. **Structure your answer** — Use clear paragraphs. If the question covers multiple aspects, address each one.
5. **If information is insufficient** — Say clearly what the retrieved papers do cover and what gaps remain. Suggest what search terms might help find more relevant papers.

### Detailed Answer:"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Retrieves relevant papers via semantic search, then uses an LLM
    to generate a grounded answer with citations.

    Usage:
        rag = RAGPipeline(retriever)
        result = rag.answer("What are the latest approaches to few-shot learning?")
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        use_quantization: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.llm_pipeline = None
        self.prompt_template = RAG_PROMPT_TEMPLATE

        # API keys
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # ──────────────────────────────────────────────
    # LLM Selection Logic
    # ──────────────────────────────────────────────

    def _generate_answer(self, prompt: str) -> str:
        """
        Try LLM backends in order of preference:
        1. Google Gemini API
        2. OpenAI API
        3. Local HuggingFace model (only if no API keys configured)
        4. Extractive fallback
        """
        # 1. Try Gemini (free, fast, no GPU)
        if self.gemini_api_key:
            result = self._gemini_generate(prompt)
            if result:
                return result

        # 2. Try OpenAI
        if self.openai_api_key:
            result = self._openai_generate(prompt)
            if result:
                return result

        # 3. Try local HuggingFace model (only if no API keys set)
        if not self.gemini_api_key and not self.openai_api_key:
            result = self._local_llm_generate(prompt)
            if result:
                return result

        # 4. Extractive fallback
        return self._extractive_fallback(prompt)

    # ──────────────────────────────────────────────
    # Backend 1: Google Gemini API (FREE)
    # ──────────────────────────────────────────────

    def _gemini_generate(self, prompt: str) -> Optional[str]:
        """Generate using Google Gemini API (free tier available)."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    # No max_output_tokens cap — let Gemini write a complete answer
                ),
            )
            logger.info("Answer generated via Google Gemini API ✅")
            return response.text.strip()

        except ImportError:
            logger.warning("google-genai not installed. Run: pip3 install google-genai")
            return None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    # ──────────────────────────────────────────────
    # Backend 2: OpenAI API
    # ──────────────────────────────────────────────

    def _openai_generate(self, prompt: str) -> Optional[str]:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            logger.info("Answer generated via OpenAI API ✅")
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    # ──────────────────────────────────────────────
    # Backend 3: Local HuggingFace Model
    # ──────────────────────────────────────────────

    def _load_llm(self):
        """Lazily load the local HuggingFace LLM pipeline."""
        if self.llm_pipeline is not None:
            return

        logger.info(f"Loading local LLM: {self.model_name}...")
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            model_kwargs = {"trust_remote_code": True, "device_map": "auto"}

            if self.use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    import torch

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available. Loading without quantization.")

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            logger.info("Local LLM loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm_pipeline = None

    def _local_llm_generate(self, prompt: str) -> Optional[str]:
        """Generate using local HuggingFace pipeline."""
        self._load_llm()
        if self.llm_pipeline is None:
            return None
        try:
            output = self.llm_pipeline(prompt)
            return output[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return None

    # ──────────────────────────────────────────────
    # Backend 4: Extractive Fallback
    # ──────────────────────────────────────────────

    def _extractive_fallback(self, prompt: str) -> str:
        """
        Last resort when no LLM is available.
        Extracts key sentences from paper abstracts.
        """
        logger.info("Using extractive fallback (no LLM configured)")

        try:
            context_start = prompt.index("### Retrieved Research Papers:") + len(
                "### Retrieved Research Papers:"
            )
            context_end = prompt.index("### Question:")
            context = prompt[context_start:context_end].strip()
        except ValueError:
            return "Unable to generate an answer. Please configure a GEMINI_API_KEY in your .env file."

        summaries = []
        for block in context.split("---"):
            lines = block.strip().split("\n")
            for line in lines:
                if line.startswith("Abstract:"):
                    abstract_text = line.replace("Abstract:", "").strip()
                    sentences = abstract_text.split(". ")[:2]
                    summaries.append(". ".join(sentences) + ".")
                    break

        if summaries:
            return (
                "⚠️ **No LLM configured** — showing raw abstract excerpts from retrieved papers.\n\n"
                "**To get real AI answers, add your `GEMINI_API_KEY` to the `.env` file (it's free!).**\n\n"
                "---\n\n"
                + "\n\n".join(f"• {s}" for s in summaries)
            )
        return "Unable to generate an answer. Please add GEMINI_API_KEY to your .env file."

    # ──────────────────────────────────────────────
    # Context Formatting
    # ──────────────────────────────────────────────

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved papers into a context string for the prompt."""
        context_parts = []
        for r in results:
            meta = r["metadata"]
            title = meta.get("title", "Untitled")
            abstract = meta.get("abstract", "No abstract available.")
            authors = meta.get("authors", "Unknown")
            score = r.get("score", 0.0)

            context_parts.append(
                f"**Paper: {title}**\n"
                f"Authors: {authors}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Abstract: {abstract}\n"
            )

        return "\n---\n".join(context_parts)

    # ──────────────────────────────────────────────
    # Main Answer Method
    # ──────────────────────────────────────────────

    def answer(
        self,
        question: str,
        top_k: int = 5,
        year_filter: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG (retrieve → format context → generate).

        Args:
            question: Natural language question.
            top_k: Number of papers to retrieve.
            year_filter: Optional year filter.
            category_filter: Optional category filter.

        Returns:
            Dict with 'answer', 'sources', 'query', 'context_used'.
        """
        # Step 1: Retrieve relevant papers
        results = self.retriever.search(
            query=question,
            top_k=top_k,
            year_filter=year_filter,
            category_filter=category_filter,
        )

        if not results:
            return {
                "answer": "No relevant papers found for your question.",
                "sources": [],
                "query": question,
                "context_used": "",
            }

        # Step 2: Format context
        context = self._format_context(results)

        # Step 3: Build prompt and generate answer
        prompt = self.prompt_template.format(context=context, question=question)
        answer_text = self._generate_answer(prompt)

        # Step 4: Format sources
        sources = []
        for r in results:
            meta = r["metadata"]
            sources.append({
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "arxiv_id": meta.get("id", ""),
                "arxiv_url": f"https://arxiv.org/abs/{meta.get('id', '')}",
                "score": r["score"],
                "categories": meta.get("categories", ""),
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "query": question,
            "context_used": context,
        }

    def answer_from_papers(
        self,
        question: str,
        papers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Answer a question using externally provided papers (e.g., from live arXiv search).

        Args:
            question: Natural language question.
            papers: List of paper dicts with 'title', 'abstract', 'authors', etc.

        Returns:
            Dict with 'answer', 'sources', 'query', 'context_used'.
        """
        if not papers:
            return {
                "answer": "No papers provided to answer from.",
                "sources": [],
                "query": question,
                "context_used": "",
            }

        # Convert live papers to the same format as retriever results
        results = []
        for p in papers:
            results.append({
                "metadata": p,
                "score": 0.0,  # No embedding score for live papers
            })

        # Format context
        context = self._format_context(results)

        # Build prompt and generate answer
        prompt = self.prompt_template.format(context=context, question=question)
        answer_text = self._generate_answer(prompt)

        # Format sources
        sources = []
        for p in papers:
            sources.append({
                "title": p.get("title", ""),
                "authors": p.get("authors", ""),
                "arxiv_id": p.get("id", ""),
                "arxiv_url": p.get("arxiv_url", f"https://arxiv.org/abs/{p.get('id', '')}"),
                "score": 0.0,
                "categories": p.get("categories", ""),
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "query": question,
            "context_used": context,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RAG Pipeline module loaded. Use RAGPipeline class for Q&A.")
