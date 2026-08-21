import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant answering questions about a corpus of "
     "battery state-of-charge (SOC) estimation papers. Answer ONLY using "
     "the provided context — if the context doesn't contain the answer, "
     "say so explicitly rather than guessing.\n\n"
     "Every claim you make must be followed by a citation in the form "
     "[Source: <paper title>]. If a sentence draws on multiple sources, "
     "cite all of them. Do not fabricate citations for anything not in "
     "the context."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

chain = PROMPT | llm | StrOutputParser()


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


def answer(query: str, k: int = TOP_K) -> dict:
    retrieved = retrieve(query, k=k)
    context = format_context(retrieved)

    response = chain.invoke({"context": context, "question": query})

    return {
        "query": query,
        "answer": response,
        "sources": sorted(set(r["source"] for r in retrieved)),
        "retrieved_chunks": retrieved,
    }


if __name__ == "__main__":
    query = "How does Machine Learning estimate battery state of charge?"
    result = answer(query)

    print(f"Q: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources used in retrieval:")
    for s in result["sources"]:
        print(f"  - {s}")