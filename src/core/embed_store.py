import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_FILE = Path("data/chunks.json")
INDEX_FILE = Path("data/faiss_index/index.faiss")
METADATA_FILE = Path("data/faiss_index/metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"   # fast, 384-dim, good enough for this scale

def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {INDEX_FILE}")

if __name__ == "__main__":
    main()