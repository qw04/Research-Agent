# Research Agent

A RAG + multi-agent research assistant over a corpus of academic papers. It answers questions with citations to the papers it retrieved from. A critic agent checks each answer against the sources, and the agent falls back to live arXiv search when the local corpus doesn't cover a question.

> **Status: in transition.** The project started as a fixed-corpus assistant for **battery state-of-charge (SOC) estimation** papers (10 PDFs in `papers/`). The goal now is a **generic, topic-driven research agent**: give it a topic, it finds the most influential papers on that topic, and you can then ask questions about them to learn the key techniques. Parts of the code are still SOC-specific; see [Roadmap](#roadmap).

---

## How it works

```
papers/*.pdf
   │  src/core/ingest.py        PDF → cleaned text → 800-char chunks (150 overlap)
   ▼
data/chunks.json
   │  src/core/embed_store.py   all-MiniLM-L6-v2 embeddings → FAISS IndexFlatL2
   ▼
data/faiss_index/
   │
   ▼  src/core/graph.py  (LangGraph)
retrieve → grade ─[sufficient]──────────────────────────→ generate → critic → answer
              ├─[insufficient, attempts < MAX]→ reformulate → retrieve (loop)
              └─[insufficient, attempts ≥ MAX]→ arxiv_search → generate → critic → answer
```

- **Ingest** ([src/core/ingest.py](src/core/ingest.py)): parses PDFs with `unstructured`, drops headers, footers, page numbers and tables, and stops at the References section, which it detects by the heading or by a run of citation-shaped lines.
- **Embed** ([src/core/embed_store.py](src/core/embed_store.py)): encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) into an exact FAISS L2 index.
- **Retrieve + generate** ([src/core/generate.py](src/core/generate.py)): retrieves the top-k chunks, then GPT-4o answers using only that context, with `[Source: <title>]` citations.
- **Agent graph** ([src/core/graph.py](src/core/graph.py)): a LangGraph state machine. GPT-4o-mini grades whether the retrieved chunks are sufficient. If they aren't, the query is reformulated and retrieval runs again. After repeated failures it falls back to a keyword search on arXiv (AND-joined terms, retried with OR).
- **Critic** ([src/core/critic.py](src/core/critic.py)): a CrewAI fact-checking critic reviews the draft claim by claim against the sources. If it rejects the draft, a reviser agent does exactly one rewrite. The critic is told the research topic up front instead of having it hardcoded.
- **Extra tools in `graph.py`:** `compare_papers()` contrasts two named papers. `paper_relationships()` looks a paper up in the offline paper graph.

### Other modules

| Path | What it does |
|---|---|
| [src/paper_graph/](src/paper_graph/) | Offline pass that extracts a profile per paper (techniques used, baselines, novelty, headline result) and builds a paper→technique / paper→baseline graph (`data/paper_graph.json`). Also includes a script that finds baselines several papers compare against but that have no paper of their own in the corpus (`find_common_dependents.py`), a graph visualizer, and a results infographic (`infographic.py`). |
| [src/discovery/topic_discovery.py](src/discovery/topic_discovery.py) | Stage 1 of the topic-driven mode: given a topic, it searches Wikipedia, DuckDuckGo, arXiv surveys and Semantic Scholar in parallel, then an LLM shortlists intro sources. Stages 2 (finding influential papers) and 3 (ingesting them) are not built yet. |
| [src/eval/evaluate.py](src/eval/evaluate.py) | Runs a hand-built eval set of 22 questions (`data/eval_set.json`) through the agent and scores faithfulness and relevance with an LLM judge. Results are logged to MLflow across configurations (top_k 3/5, critic on/off). |
| [src/api/main.py](src/api/main.py) | FastAPI wrapper over the core graph: `GET /health`, `GET /papers`, `POST /ask`, `POST /compare`, `GET /papers/{paper}/relationships`. |
| `Dockerfile`, [k8s/](k8s/) | Container image with the embedding model baked in, so it runs with no network access to Hugging Face. Also a single-replica Kubernetes Deployment + NodePort Service. |
| `src/scratch/` | Throwaway experiments. |

---

## Setup

Requires Python 3.11+ and an OpenAI API key.

```bash
python -m venv venv
venv/Scripts/activate        # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Create a `.env` at the repo root:

```
OPENAI_API_KEY=sk-...
```

`requirements.lock.txt` pins exact versions for the Docker build. `topic_discovery.py` also needs `ddgs`, which is not in `requirements.txt` yet.

## Usage

Run everything **from the repo root**: data paths such as `data/` and `papers/` are relative to it.

Build the corpus and index (re-run whenever `papers/` changes):

```bash
python src/core/ingest.py
```

```bash
python src/core/embed_store.py
```

Ask a question from the CLI:

```bash
python src/core/graph.py
```

Run the API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Run the evaluation (logs to MLflow):

```bash
python src/eval/evaluate.py
```

Topic discovery, stage 1:

```bash
python src/discovery/topic_discovery.py "your topic here"
```

---

## Roadmap

Moving from the battery SOC corpus to a generic, topic-driven agent:

1. **Core: remove the hardcoded domain**
   - [x] `critic.py`: the critic and reviser agents take a `topic` parameter, with a generic fallback.
   - [ ] `graph.py`: add `topic` to `GraphState` and `ask()`, pass it to the critic, and make the reformulate, answer, arXiv-keyword and compare prompts generic.
   - [ ] `generate.py`: make the system prompt use the topic.
2. **Topic discovery:** build stage 2 (pick the *influential* papers for a topic using arXiv plus Semantic Scholar citation counts and an LLM shortlist) and stage 3 (ingest the papers you approve).
3. **Downstream cleanup:** update the SOC wording in `src/api/main.py`, `src/eval/evaluate.py` (judge prompt + eval set), and `src/paper_graph/` (`DOMAIN_CONTEXT`, plot titles).
4. **Embed store (last)**
   - [ ] Keep a separate index per topic (e.g. `data/indexes/<topic_slug>/`) instead of one static index built from `papers/`.
   - [ ] Define the embedding model name in one place. Right now it's set separately in `embed_store.py` and `generate.py`, so they can drift apart without any error.
   - [ ] Consider an embedding model trained on scientific text (e.g. SPECTER2) or a stronger general one (bge / e5). Measure the effect with the eval harness before switching.
   - [ ] Remove the dependence on running from the repo root (resolve paths from the file location or from config).
