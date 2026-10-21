"""
Shared retrieval module for the agent in graph.py: loads the embedding
model, FAISS index and chunk metadata once, and exposes retrieve() and
format_context(). Also owns the final-answer `llm` that graph.py imports.

Answer generation itself (prompt + chain) lives in graph.py's answer_prompt
/ generate_node; the original single-shot answer() pipeline that used to
live here was removed since nothing called it.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

load_dotenv()

INDEX_FILE = Path("data/faiss_index/index.faiss")
METADATA_FILE = Path("data/faiss_index/metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

LLM_MODEL = "gpt-4o"
TOP_K = 5

# Loaded once at module level, not per-call — avoids reloading the
# embedding model and re-reading the index/metadata on every query
embed_model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(INDEX_FILE))
with open(METADATA_FILE) as f:
    chunks = json.load(f)

assert index.ntotal == len(chunks), (
    f"Index/metadata mismatch: index has {index.ntotal} vectors, "
    f"metadata has {len(chunks)} entries. Rebuild the index."
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(dist),
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
        })
    return results


def format_context(retrieved: list[dict]) -> str:
    """Format retrieved chunks with clear source labels so the LLM can
    cite them accurately rather than inventing citation text."""
    blocks = []
    for r in retrieved:
        blocks.append(f"[Source: {r['source']}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)
