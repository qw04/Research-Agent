# Research Agent

A RAG + multi-agent research assistant over a corpus of academic papers. It answers questions with citations to the papers it retrieved from. A critic agent checks each answer against the sources, and the agent falls back to live arXiv search when the local corpus doesn't cover a question.

> **Status: in transition.** The project started as a fixed-corpus assistant for **battery state-of-charge (SOC) estimation** papers (10 PDFs in `papers/`). The goal now is a **generic, topic-driven research agent**: give it a topic, it finds the most influential papers on that topic, and you can then ask questions about them to learn the key techniques. Every prompt now takes a `topic` (defaulting to a generic "academic research"). The corpus and eval set are still the battery SOC ones; see [Roadmap](#roadmap).

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
- **Retrieval** ([src/core/generate.py](src/core/generate.py)): the shared retrieval module. It loads the embedding model, FAISS index and chunk metadata once, and provides `retrieve()` (top-k chunks) and `format_context()` (chunks labelled with their source). It also defines the GPT-4o `llm` used for final answers.
- **Agent graph** ([src/core/graph.py](src/core/graph.py)): a LangGraph state machine. Its `generate` node has GPT-4o answer using only the retrieved context, with `[Source: <title>]` citations. GPT-4o-mini grades whether the retrieved chunks are sufficient. If they aren't, the query is reformulated and retrieval runs again. After repeated failures it falls back to a keyword search on arXiv ([src/discovery/arxiv_search.py](src/discovery/arxiv_search.py)).
- **Critic** ([src/core/critic.py](src/core/critic.py)): a CrewAI fact-checking critic reviews the draft claim by claim against the sources. If it rejects the draft, a reviser agent does exactly one rewrite. The critic is told the research topic up front instead of having it hardcoded.
- **Standalone tools** (not part of the graph loop): `compare_papers()` ([src/core/compare.py](src/core/compare.py)) contrasts two named papers using their own chunks. `paper_relationships()` ([src/paper_graph/paper_relationships.py](src/paper_graph/paper_relationships.py)) looks a paper up in the precomputed paper graph.

### Other modules

| Path | What it does |
|---|---|
| [src/paper_graph/](src/paper_graph/) | Offline pass that extracts a profile per paper (techniques used, baselines, novelty, headline result) and builds a paper→technique / paper→baseline graph (`data/paper_graph.json`). Also includes a script that finds baselines several papers compare against but that have no paper of their own in the corpus (`find_common_dependents.py`), a graph visualizer, and a results infographic (`infographic.py`). |
| [src/discovery/arxiv_search.py](src/discovery/arxiv_search.py) | Live arXiv search behind the agent's fallback, also callable on its own as `search_arxiv(question)`. GPT-4o-mini turns the question into a few keywords, which are searched AND-joined, then OR-joined if that finds nothing. Returns abstracts in the same `{score, source, text}` shape as local retrieval. |
| [src/discovery/topic_discovery.py](src/discovery/topic_discovery.py) | Stage 1 of the topic-driven mode: given a topic, it searches Wikipedia, DuckDuckGo, arXiv surveys and Semantic Scholar in parallel, then an LLM shortlists intro sources. Stages 2 (finding influential papers) and 3 (ingesting them) are not built yet. |
| [src/eval/](src/eval/) | The evaluation harness. It runs a hand-built eval set of 22 questions (`data/eval_set.json`) through the agent 3 times per configuration, has an LLM judge score each answer, checks the judge itself, and logs everything to MLflow. The modules are listed below, and the results are in [Evaluation](#evaluation). |
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

Imports are package-style, rooted at `src/` (e.g. `from core.graph import ask`, `from discovery.arxiv_search import search_arxiv`). Each script you run directly adds `src/` to `sys.path` itself, so no `PYTHONPATH` setup is needed.

Build the corpus and index (re-run whenever `papers/` changes):

```bash
python src/core/ingest.py
```

```bash
python src/core/embed_store.py
```

Ask a question from the CLI (`--topic` is optional):

```bash
python src/core/graph.py "Which methods give the lowest error?" --topic "battery state-of-charge estimation"
```

Run the API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Run the evaluation (logs to MLflow; about 20 minutes and roughly 200 agent runs, all billed OpenAI calls):

```bash
python src/eval/evaluate.py
```

Quick check on the first N questions only (runs are suffixed `_limitN`):

```bash
python src/eval/evaluate.py --limit 2
```

Regenerate the judge-validation holdout (refuses to overwrite a file that already has human grades):

```bash
python src/eval/evaluate.py --build-validation
```

Audit the judge's per-question reliability (re-judges stored answers; doesn't re-run the agent; about 200 cheap judge calls):

```bash
python src/eval/evaluate.py --judge-audit
```

Topic discovery, stage 1:

```bash
python src/discovery/topic_discovery.py "your topic here"
```

---

## Evaluation

The eval set has 22 hand-written questions (method, result and comparison questions) with reference answers, all about the battery SOC corpus. Each question is run **3 times** per configuration. A GPT-4o-mini judge scores each answer 1–5 on:

- **faithfulness:** is every claim backed by the retrieved excerpts?
- **relevance:** does the answer address the question and cover the reference facts?

An attempt counts as a **success** only if it scores 5/5 on both.

### Code layout

`src/eval/evaluate.py` is only the command-line entry point. The logic lives in:

| Module | What it does |
|---|---|
| [settings.py](src/eval/settings.py) | File paths, the eval topic, the pass^k settings (3 attempts, success = 5/5) |
| [judge.py](src/eval/judge.py) | The judge: prompt v2, the answer-first variant, `score_answer()` (including the empty-answer rule) |
| [pass_k.py](src/eval/pass_k.py) | The agent runs, per-config metrics, MLflow logging, `data/eval_runs/<run>.json` |
| [validation.py](src/eval/validation.py) | The human-graded holdout and the agreement report |
| [robustness.py](src/eval/robustness.py) | The degenerate-answer test |
| [audit.py](src/eval/audit.py) | The reproducibility and order-invariance audit, and the trustworthy verdict rate |

All judging goes through `judge.py`, so the eval and every judge check use the same prompt, model and rules.

### Results (judge v2, 2026-11-14)

| Config | pass^1 | ≥1 of 3 | pass^3 | Avg faithfulness | Avg relevance | Runtime |
|---|---|---|---|---|---|---|
| top_k=3, critic off | 44% | 45% | 41% | 4.18 | 3.74 | 5 min |
| top_k=5, critic off | 55% | 55% | 55% | 4.39 | 3.95 | 5 min |
| top_k=5, critic on | 59% | 68% | 50% | 4.35 | 4.14 | 10 min |

- **pass^1:** mean per-attempt success rate.
- **≥1 of 3:** the question succeeded at least once.
- **pass^3:** the question succeeded on all 3 attempts. This is the consistency measure from τ-bench ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)).

What the numbers say:

With 22 questions, one question is worth about 4.5 percentage points on pass^3, and no confidence intervals have been computed yet. Read every comparison below as tentative.

- **top_k=5 beats top_k=3** on every metric. On pass^1, the gap is about 2.5 questions' worth (55% vs 44%). That's the clearest difference in the table, but it still has no interval.
- **The critic does not clearly change consistency.** Its pass^3 is 50% against 55% without it, which is 11 questions vs 12. An earlier version of this README said the critic was "the main source of run-to-run variation". The evidence doesn't support that:
  - **Answers vary without the critic too.** Even with the critic off, the answer text changes across the 3 attempts on 10 of 22 questions at top_k=5 (18 of 22 at top_k=3). The verdicts just happened to stay the same.
  - **The critic's pass^1-to-pass^3 gap comes from 4 questions, and none is clearly the agent varying:**
    - q05: identical answer text all 3 times but different verdicts, so this is pure judge variation.
    - q16, q20: the [judge audit](#judge-audit) marks both untrusted, because putting the answer first flips their verdict.
    - q03: went through the arXiv fallback and couldn't be audited.

  Separating agent variation from judge variation properly is planned (see [documentation.md](documentation.md), Open items).
- **The critic may help relevance (3.95 → 4.14), but not faithfulness (4.39 vs 4.35).** Both differences are small. An earlier version of the judge suggested a faithfulness gain, but that judge was flawed (see below), so that result doesn't stand.
- **Some questions fail in every configuration:**
  - q03: "LRLSTM" never matches locally, so the agent falls back to arXiv.
  - q21, q22: comparison questions that need information from several papers.
  - q07, q08, q12, q14: partial answers.

  These point to retrieval problems rather than generation problems.

### Checking the judge

Before the main eval, the script checks whether its own judge is trustworthy:

- **Robustness test** ([data/judge_robustness.json](data/judge_robustness.json)): the judge scores an empty answer, a one-word "unknown", and a confident made-up answer. All three must score low; any score above 3 triggers a warning. The current result is 3/3 as expected.
- **Human validation** ([data/judge_validation.json](data/judge_validation.json)): 3 holdout questions (one each of method, result and comparison), graded by a person and re-judged on every run. The current result is 4/6 scores matching exactly and 6/6 within ±1. This is a spot-check, not an accuracy estimate. The judge prompt was revised after seeing these three items, so the agreement is optimistic.

Both summaries are printed and attached to each MLflow run as its note.

### Judge audit

`python src/eval/evaluate.py --judge-audit` checks whether each **individual verdict** can be trusted. It re-judges the stored attempt-1 answers from `topk5_critic_on_x3` without re-running the agent:

- 5 times with the normal prompt
- 5 times with an answer-first variant: identical wording, with the answer placed before the reference facts and excerpts

A question's verdict (success = 5/5) is **trustworthy** only if it is:

- **reproducible:** all 5 normal judgements agree
- **order-invariant:** the answer-first judgements give the same verdict
- **accurate:** for the human-graded holdout items, within ±1 of the human score

This follows Kumar, Rathore & Moitra, *All Verdicts are Not Equal* ([arXiv:2610.12083](https://arxiv.org/abs/2610.12083)). Results are in [data/judge_audit.json](data/judge_audit.json).

| | Result |
|---|---|
| Trustworthy verdict rate | **14/19 auditable (74%)**; 14/22 (64%) if the unverifiable items count as untrusted |
| Reproducible | 95% (19/19 in the first audit run, 18/19 in the second) |
| Order-invariant | 74%: the answer-first order flips the verdict on q11, q14, q16, q18 and q20 |
| Unverifiable | q03, q04, q17 went through the arXiv fallback, whose context the eval log didn't store at the time. Eval logs now store context. |
| pass^1 / pass^3, critic on | 59% / 50% on all 22 questions, vs 60% / 57% on the 14 trustworthy ones |

Order effects can be large. With the answer first, q16 went from faithfulness 5 to 1, and q14 from 2 to 5. Three questions (q05, q12, q16) also scored differently from the same judge on the same input during the earlier eval run. So the judge is not perfectly reproducible across sessions, even at temperature 0.

### Limitations

- **Small eval set.** 22 questions from one corpus (battery SOC), written from excerpts of the papers. One question changes pass^3 by about 4.5 percentage points.
- **No uncertainty intervals yet**, so no comparison above has been shown to be outside the noise.
- **The judge is order-sensitive.** About a quarter of its verdicts are not trustworthy (see [Judge audit](#judge-audit)). The judge also isn't fully reproducible across sessions.
- **Human validation is 3 questions**, and the judge prompt was tuned on them.
- **Paper figures aren't checked.** Figures quoted from the cited papers in [documentation.md](documentation.md) were taken as given and haven't been independently verified.

The full history of why the eval changed, including the judge bug these checks caught, is in [documentation.md](documentation.md).

---

## Roadmap

Moving from the battery SOC corpus to a generic, topic-driven agent:

1. **Core: remove the hardcoded domain** ✅
   - [x] `critic.py`: the critic and reviser agents take a `topic` parameter, with a generic fallback.
   - [x] `graph.py`: `topic` is part of `GraphState` and `ask()`, and drives the reformulate, answer, arXiv and critic prompts. The default lives in `src/core/config.py`.
   - [x] `compare.py`: `compare_papers(..., topic=)`.
   - [x] `arxiv_search.py`: `search_arxiv(question, topic=)`. The topic keeps the arXiv keywords on-domain.
   - [x] `generate.py`: removed the unused SOC-specific prompt, chain and `answer()`. It's now a retrieval-only module.
2. **Topic discovery:** build stage 2 (pick the *influential* papers for a topic using arXiv plus Semantic Scholar citation counts and an LLM shortlist) and stage 3 (ingest the papers you approve).
3. **Downstream cleanup**
   - [x] API: `/ask` and `/compare` accept an optional `topic`.
   - [x] Eval: the judge prompt takes the topic, and the eval runs with `EVAL_TOPIC` (battery SOC, to match the current eval set).
   - [x] `paper_graph/`: the arXiv domain term and the plot title are command-line arguments.
   - [ ] Write an eval set that isn't tied to the battery SOC corpus, or generate one per topic.
   - [ ] Fix the retrieval failures the eval exposes (q03 acronym miss, multi-paper comparison questions q21/q22).
   - [ ] Grade a few fresh questions by hand, to test judge v2 on items it wasn't tuned against.
4. **Ingest: richer PDF extraction (later)**
   - [ ] **References:** `ingest.py` currently stops reading at the References section and throws it away. Parse it into structured citations (title, authors, year) and store them per paper. This could feed a citation graph within the corpus and help with ranking influential papers in topic discovery stage 2.
   - [ ] **Tables:** `ingest.py` currently drops `Table` elements (they're in `SKIP_CATEGORIES`), because bare table fragments ranked well in search without adding useful content. Extract them properly instead (e.g. `unstructured` with `strategy="hi_res"` and `infer_table_structure=True`), store them as their own chunks with the table's caption, and possibly reuse them for the results infographic.
5. **Embed store (last)**
   - [ ] Keep a separate index per topic (e.g. `data/indexes/<topic_slug>/`) instead of one static index built from `papers/`.
   - [ ] Define the embedding model name in one place. Right now it's set separately in `embed_store.py` and `generate.py`, so they can drift apart without any error.
   - [ ] Consider an embedding model trained on scientific text (e.g. SPECTER2) or a stronger general one (bge / e5). Measure the effect with the eval harness before switching.
   - [ ] Remove the dependence on running from the repo root (resolve paths from the file location or from config).
