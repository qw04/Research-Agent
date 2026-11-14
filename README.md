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

Audit the judge's per-question reliability on a stored run (normal, answer-first and paraphrased prompts, 5 times each; about 350 cheap judge calls; doesn't re-run the agent):

```bash
python src/eval/evaluate.py --judge-audit --suffix _ctx
```

Bootstrap 95% intervals for a set of runs (no API calls). The suffix picks the runs, e.g. `_ctx` for `*_x3_ctx`:

```bash
python src/eval/evaluate.py --stats --suffix _ctx_v3ens9
```

Re-judge every stored attempt with the aggregated judge (majority of 9 calls; about 1,800 cheap judge calls; no agent calls):

```bash
python src/eval/evaluate.py --rejudge --suffix _ctx --judge-n 9 --judge-mode ensemble
```

Run the full eval with the aggregated judge from the start:

```bash
python src/eval/evaluate.py --judge-n 9 --suffix _v3
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

The current judge is **v3-ensemble9**. Each answer is judged 9 times: 3 times each with the normal prompt, an answer-first version, and a reworded (paraphrased) version. The success verdict is the majority of the 9; faithfulness and relevance are their means. The single-call judge (v2) is still the default; pass `--judge-n 9` to use v3.

### Code layout

`src/eval/evaluate.py` is only the command-line entry point. The logic lives in:

| Module | What it does |
|---|---|
| [settings.py](src/eval/settings.py) | File paths, the eval topic, the pass^k settings (3 attempts, success = 5/5) |
| [judge.py](src/eval/judge.py) | The judge: prompt v2, answer-first and paraphrase variants, `score_answer()` (including the empty-answer rule), and the majority-of-N judge `score_answer_aggregated()` |
| [pass_k.py](src/eval/pass_k.py) | The agent runs, per-config metrics, MLflow logging, `data/eval_runs/<run>.json` (each attempt stores its retrieved excerpts) |
| [validation.py](src/eval/validation.py) | The human-graded holdout and the agreement report |
| [robustness.py](src/eval/robustness.py) | The degenerate-answer test |
| [audit.py](src/eval/audit.py) | The per-question audit: reproducibility, order and wording sensitivity, trustworthy verdict rate |
| [stats.py](src/eval/stats.py) | Bootstrap intervals, paired config differences, identical-text breakdown |
| [rejudge.py](src/eval/rejudge.py) | Re-scores stored attempts with the aggregated judge, without re-running the agent |

All judging goes through `judge.py`, so the eval and every judge check use the same prompts, model and rules.

### Results

The table shows run 2 (agent answers with context logged), re-judged with v3-ensemble9. Brackets are bootstrap 95% intervals: 22 questions resampled 10,000 times, seed 0.

| Config | pass^1 | ≥1 of 3 | pass^3 | Avg faithfulness | Avg relevance |
|---|---|---|---|---|---|
| top_k=3, critic off | 47% [27, 68] | 50% [27, 73] | 45% [23, 68] | 4.30 [3.80, 4.73] | 3.75 [3.32, 4.17] |
| top_k=5, critic off | 55% [32, 73] | 55% [32, 73] | 55% [32, 73] | 4.50 [4.06, 4.85] | 3.96 [3.48, 4.42] |
| top_k=5, critic on | 52% [33, 70] | 64% [45, 82] | 41% [23, 64] | 4.26 [3.83, 4.64] | 3.98 [3.58, 4.37] |

- **pass^1:** mean per-attempt success rate.
- **≥1 of 3:** the question succeeded at least once.
- **pass^3:** the question succeeded on all 3 attempts. This is the consistency measure from τ-bench ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)).

**Paired differences.** The same resampled questions are used for both configs. Only one difference is distinguishable from question sampling:

| Comparison | pass^1 | pass^3 | Avg faithfulness | Avg relevance |
|---|---|---|---|---|
| Critic on − off (top_k=5) | −3 pp [−15, +6] | −14 pp [−27, 0] | **−0.24 [−0.49, −0.04]** | +0.02 [−0.21, +0.28] |
| top_k 5 − 3 (critic off) | +8 pp [−9, +24] | +9 pp [−9, +27] | +0.20 [−0.12, +0.61] | +0.21 [−0.09, +0.56] |

**How stable is this across runs?** pass^1 / pass^3 per config:

| Config | Run 1, v2 judge | Run 2, v2 judge | Run 2, v3-ensemble9 |
|---|---|---|---|
| top_k=3, critic off | 44% / 41% | 44% / 41% | 47% / 45% |
| top_k=5, critic off | 55% / 55% | 55% / 55% | 55% / 55% |
| top_k=5, critic on | 59% / 50% | 53% / 45% | 52% / 41% |

What the numbers say. Read all of it as tentative: one question is worth about 4.5 points on pass^3.

- **top_k=5 vs 3:** the point estimates favour top_k=5 in every run, but no interval excludes zero.
- **The critic seems to lower faithfulness.** −0.24 [−0.49, −0.04] in run 2 with v3, and −0.14 [−0.32, −0.02] with v2. In run 1 the difference was −0.05 [−0.23, +0.08]. So this replicated in one of two agent runs.
- **The critic adds run-to-run variation at the question level.** Re-judging the same stored answers with v3 removes most judge noise. Without the critic, the gap between pass^1 and pass^3 disappears (55% / 55%); with it, the gap remains (52% / 41%). It comes from five questions: q03, q04, q17, q18 and q20.
  - In each, the critic rejected the draft in every attempt, and the reviser's rewrites differed from run to run.
  - With the critic off, q18 and q20 produced identical text and verdicts all 3 times.

  The pass^3 difference (−14 pp [−27, 0]) still doesn't exclude zero, and several of these attempts are borderline: 3 to 6 of 9 judge votes for success. This reverses the earlier withdrawal of the claim. The history is in [documentation.md](documentation.md).
- **Some questions fail in every configuration:**
  - q03: "LRLSTM" never matches locally, so the agent falls back to arXiv.
  - q21, q22: comparison questions that need information from several papers.
  - q07, q08, q12, q14: partial answers.

  These point to retrieval problems rather than generation problems.

### Checking the judge

| Check | v2 (single call) | v3-ensemble9 |
|---|---|---|
| **Robustness** ([data/judge_robustness.json](data/judge_robustness.json), [v3](data/judge_robustness_v3-ensemble9.json)): an empty answer, a one-word "unknown" and a confident made-up answer must all score ≤ 2 | 3/3 | 3/3 |
| **Human validation** ([data/judge_validation.json](data/judge_validation.json)): 3 holdout questions graded by a person and re-judged on every run | 4/6 exact, 6/6 within ±1 | 3/6 exact, 5/6 within ±1 |
| **Changed verdicts**, re-judging the 198 stored run-2 attempts | — | 9/198 changed vs v2 (top_k=3: 4, top_k=5 off: 0, critic on: 5) |

Aggregating did **not** improve agreement with the human grades on these 3 items. The biggest gap is q21's faithfulness: the human gave 4, v3 gives 2, because the answer-first and paraphrased prompts score it lower. The 3 items were also used to tune the v2 prompt, and no fresh items have been graded, so this check is a spot-check, not an accuracy estimate. Both summaries are printed and attached to each MLflow run as its note.

### Judge audit

`--judge-audit` checks whether each **individual verdict** can be trusted. It takes the stored attempt-1 answer of each question and re-judges it 5 times under each of three conditions:
- **normal:** the v2 prompt
- **answer-first:** identical wording, with the answer moved before the reference facts and excerpts
- **paraphrase:** same rubric and order, reworded

A verdict is **trustworthy** only if it is:
- **reproducible:** the normal calls agree
- **order-invariant** and **paraphrase-invariant:** each variant's calls agree with the normal verdict
- **accurate:** for human-graded items, within ±1 of the human score

This follows Kumar, Rathore & Moitra, *All Verdicts are Not Equal* ([arXiv:2610.12083](https://arxiv.org/abs/2610.12083)), and Yagubyan, *The Coin Flip Judge?* ([arXiv:2606.13685](https://arxiv.org/abs/2606.13685)) for the paraphrase condition. Both are adapted here from pairwise to pointwise judging.

| Audit | First audit, run 1, 2 conditions ([file](data/judge_audit.json)) | Current audit, run 2, 3 conditions ([file](data/judge_audit_topk5_critic_on_x3_ctx.json)) |
|---|---|---|
| Trustworthy verdict rate | 14/19 auditable (74%); 3 unverifiable | **17/22 (77%)**, all auditable |
| Reproducible (normal prompt) | 95–100% | 91% |
| Majority verdict flips vs normal | answer-first 26% | answer-first 18%, paraphrase 9% |
| Adapted Kumar T, human-graded items | — | 0.87 (3 items) |
| Untrustworthy | q11, q14, q16, q18, q20 | q03, q11, q14, q16, q18 |

**What the audit says:**
- Within one prompt, repeated calls almost always agree. Changing the prompt's order or wording is what flips verdicts. That's why v3 varies the prompt rather than just repeating it.
- Order effects can be large. With the answer first, q16's faithfulness went from 5 to 1, and q14's from 2 to 5.
- Some questions (q03, q08, q18) also scored differently from what the same judge gave the same input during the eval run.

### Limitations

- **Small eval set.** 22 questions from one corpus (battery SOC), written from excerpts of the papers. One question changes pass^3 by about 4.5 percentage points, and intervals are wide.
- **What the intervals cover.** They reflect which questions are in the set, not every source of noise. v3 reduces judge noise but doesn't remove it: some attempts sit at 3 to 6 of 9 votes.
- **Human validation** is only 3 questions, and the v2 prompt was tuned on them. No fresh questions have been graded, and no second judge model has been compared.
- **The paraphrased prompt** was written for this project and not independently reviewed. Kumar et al.'s T is computed with a pointwise adaptation on only 3 items.
- **Paper figures** quoted in [documentation.md](documentation.md) were taken as given and haven't been independently verified.

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
