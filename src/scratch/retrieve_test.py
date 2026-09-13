import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

INDEX_FILE = Path("data/faiss_index/index.faiss")
METADATA_FILE = Path("data/faiss_index/metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

def retrieve(query: str, k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE) as f:
        chunks = json.load(f)

    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(dist),
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"]
        })
    return results

if __name__ == "__main__":
    query = "How does Machine Learning estimate battery state of charge?"
    for r in retrieve(query):
        print(f"\n[{r['source']}] score={r['score']:.3f}")
        print(r["text"])