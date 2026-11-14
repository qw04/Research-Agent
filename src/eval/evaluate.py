"""
Runs the hand-built eval set (data/eval_set.json) through the full agent
(graph.ask), scores each answer for faithfulness (grounded in the retrieved
sources, no fabricated claims) and relevance (covers the expected facts)
using an LLM-as-judge, and logs per-question and aggregate scores to MLflow.

Each question is run N_ATTEMPTS (3) times per configuration and scored for
consistency as well as quality, following the pass^k idea from tau-bench
(Yao et al., 2024, arXiv:2406.12045): an agent that answers correctly once is
not the same as one that answers correctly every time. An attempt "succeeds"
when faithfulness and relevance both reach PASS_SCORE (5/5). Three rates are
logged per configuration:
  - pass1_rate:      mean per-attempt success rate (tau-bench's pass^1)
  - pass_any3_rate:  share of questions with >= 1 success in 3 attempts (the
                     "at least one of three" metric; pass@3 in the usual
                     terminology, so it is >= pass1_rate by construction)
  - pass3_rate:      share of questions where ALL 3 attempts succeeded (pass^3)

Iterates over a small set of configurations (top_k, critic on/off) so MLflow
tracks comparable runs rather than a single one-off result — the point of
the MLOps line item is comparison across configurations, not just logging
one run. Chunk size is not varied here since that requires re-ingesting and
re-embedding the corpus (src/core/ingest.py, src/core/embed_store.py), which is out of
scope for this pass; top_k and critic on/off are the two knobs exercised.
"""

import json
import sys
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, so core.* / discovery.* / paper_graph.* resolve
from core.graph import ask
from core.generate import retrieve

load_dotenv()

EVAL_SET_FILE = Path("data/eval_set.json")
RUNS_DIR = Path("data/eval_runs")
JUDGE_MODEL = "gpt-4o-mini"
N_ATTEMPTS = 3   # runs per question per configuration (pass^k with k=3)
PASS_SCORE = 5   # an attempt succeeds iff faithfulness >= this AND relevance >= this

judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


class JudgeScore(BaseModel):
    faithfulness: int = Field(
        description="1-5: is every claim in the answer backed by the retrieved "
                     "sources, with no fabricated or unsupported claims? "
                     "5 = fully grounded, 1 = mostly fabricated."
    )
    relevance: int = Field(
        description="1-5: does the answer cover the key facts listed in the "
                     "reference, and actually address the question asked? "
                     "5 = fully covers the reference facts, 1 = off-topic or missing them."
    )
    reasoning: str = Field(description="One sentence justifying both scores.")


# The hand-built eval set (data/eval_set.json) was written against the
# battery SOC corpus, so the eval runs with that topic. Change this together
# with the eval set when evaluating a different corpus.
EVAL_TOPIC = "battery state-of-charge (SOC) estimation"

# Judge v2 (2026-11-14). v1 only saw the retrieved sources' TITLES, so it could
# not check whether a claim was actually backed by the text. Two failures
# followed from that: the robustness test showed it scoring an empty answer
# 5/5 (it graded the reference facts instead of the answer), and the human
# validation holdout showed it marking a wrong-but-supported answer as
# unfaithful (q21: human 4, judge 2). v2 sees the retrieved excerpts and keeps
# the two dimensions separate: faithfulness = supported by the excerpts;
# correctness vs the reference = relevance. Scores from v2 are not directly
# comparable with the earlier single-attempt runs, which used v1.
JUDGE_VERSION = "v2-context"

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading a RAG system's answer to a question about research "
     "papers on {topic}. You are given the excerpts the system retrieved, "
     "which are the only evidence it had. Score two separate things, 1-5 each:\n"
     "- faithfulness: is every claim in the answer supported by the retrieved "
     "excerpts, with nothing fabricated? Judge this ONLY against the excerpts, "
     "not against the reference facts. An answer that misses the point or picks "
     "the wrong paper, but whose every claim is backed by the excerpts, is still "
     "faithful. A plausible-sounding number or detail that does not appear in "
     "the excerpts is a faithfulness failure even if it is close to correct. "
     "Honestly saying the excerpts don't contain the answer is faithful.\n"
     "- relevance: does the answer actually address the question and cover the "
     "reference facts? Missing or contradicting the reference facts lowers "
     "relevance.\n"
     "Grade only the text in the 'System's answer' section. If it is empty or "
     "says nothing substantive, it covers none of the reference facts."),
    ("human",
     "Question: {question}\n\n"
     "Reference facts a good answer should cover:\n{expected_answer}\n\n"
     "Retrieved excerpts (the system's only evidence):\n{context}\n\n"
     "System's answer:\n<<<\n{answer}\n>>>\n\n"
     "Score faithfulness and relevance."),
])
judge_chain = judge_prompt | judge_llm.with_structured_output(JudgeScore)


def format_judge_context(chunks: list[dict]) -> str:
    """chunks: [{"text": ..., "source": optional title}] -> numbered excerpts."""
    if not chunks:
        return "(nothing retrieved)"
    return "\n\n".join(
        f"[Excerpt {i}]" + (f" (from: {c['source']})" if c.get("source") else "") + f"\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def score_answer(question: str, expected_answer: str, answer: str, chunks: list[dict]) -> JudgeScore:
    """Single entry point for judging. Empty / whitespace-only answers are scored
    1/1 deterministically without calling the judge (the v1 judge gave them 5/5)."""
    if not answer or not answer.strip():
        return JudgeScore(faithfulness=1, relevance=1,
                          reasoning="Empty answer: scored 1/1 by rule, judge not called.")
    return judge_chain.invoke({
        "topic": EVAL_TOPIC,
        "question": question,
        "expected_answer": expected_answer,
        "context": format_judge_context(chunks),
        "answer": answer,
    })


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))


def attempt_succeeded(faithfulness: int, relevance: int) -> bool:
    return faithfulness >= PASS_SCORE and relevance >= PASS_SCORE


def run_eval(top_k: int, use_critic: bool, run_name: str, limit: int | None = None,
             run_note: str = "") -> dict:
    eval_set = load_eval_set()[:limit]

    with mlflow.start_run(run_name=run_name):
        if run_note:
            mlflow.set_tag("mlflow.note.content", run_note)   # shows as the run's description/comment
        mlflow.log_params({
            "top_k": top_k,
            "use_critic": use_critic,
            "topic": EVAL_TOPIC,
            "n_questions": len(eval_set),
            "n_attempts": N_ATTEMPTS,
            "pass_score": PASS_SCORE,
            "judge_model": JUDGE_MODEL,
            "judge_version": JUDGE_VERSION,
        })

        faithfulness_scores, relevance_scores = [], []  # one entry per attempt
        per_question_log = []
        start = time.time()

        for item in eval_set:
            attempts = []
            for attempt_no in range(1, N_ATTEMPTS + 1):
                result = ask(item["question"], topic=EVAL_TOPIC, top_k=top_k, use_critic=use_critic)
                score = score_answer(item["question"], item["expected_answer"],
                                     result["answer"], result["retrieved"])
                faithfulness_scores.append(score.faithfulness)
                relevance_scores.append(score.relevance)
                attempts.append({
                    "attempt": attempt_no,
                    "answer": result["answer"],
                    "faithfulness": score.faithfulness,
                    "relevance": score.relevance,
                    "judge_reasoning": score.reasoning,
                    "critic_approved": result["approved"],
                    "attempts": result["attempts"],
                    "used_arxiv": result["used_arxiv"],
                    "success": attempt_succeeded(score.faithfulness, score.relevance),
                })

            n_success = sum(a["success"] for a in attempts)
            per_question_log.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "n_success": n_success,
                "pass_any3": n_success >= 1,
                "pass3": n_success == N_ATTEMPTS,
                "attempts": attempts,
            })
            print(f"  {item['id']} faithfulness={[a['faithfulness'] for a in attempts]} "
                  f"relevance={[a['relevance'] for a in attempts]} "
                  f"success={n_success}/{N_ATTEMPTS}")

        duration = time.time() - start
        n_q = len(per_question_log)
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        pass1_rate = sum(q["n_success"] for q in per_question_log) / (n_q * N_ATTEMPTS)
        pass_any3_rate = sum(q["pass_any3"] for q in per_question_log) / n_q
        pass3_rate = sum(q["pass3"] for q in per_question_log) / n_q

        mlflow.log_metrics({
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "pass1_rate": pass1_rate,
            "pass_any3_rate": pass_any3_rate,
            "pass3_rate": pass3_rate,
            "duration_seconds": duration,
        })

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUNS_DIR / f"{run_name}.json"
        log_path.write_text(json.dumps(per_question_log, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(log_path))

        print(f"[{run_name}] avg_faithfulness={avg_faithfulness:.2f} "
              f"avg_relevance={avg_relevance:.2f} duration={duration:.1f}s\n")
        return {
            "avg_faithfulness": avg_faithfulness, "avg_relevance": avg_relevance,
            "pass1_rate": pass1_rate, "pass_any3_rate": pass_any3_rate, "pass3_rate": pass3_rate,
            "n_questions": n_q,
        }


def print_summary(results: dict[str, dict]) -> None:
    print(f"pass^k summary (success = faithfulness>={PASS_SCORE} and relevance>={PASS_SCORE}, "
          f"{N_ATTEMPTS} attempts/question)")
    print(f"  {'config':<24}{'pass^1':>8}{'>=1 of 3':>10}{'pass^3':>8}{'avg faith':>11}{'avg rel':>9}")
    for name, r in results.items():
        print(f"  {name:<24}{r['pass1_rate']:>8.0%}{r['pass_any3_rate']:>10.0%}"
              f"{r['pass3_rate']:>8.0%}{r['avg_faithfulness']:>11.2f}{r['avg_relevance']:>9.2f}")
    print("  pass^1 = mean per-attempt success; >=1 of 3 = any attempt succeeded; "
          "pass^3 = all 3 succeeded.")
    print("  The gap between '>=1 of 3' and 'pass^3' shows how much success depends on luck.")


# ---------------------------------------------------------------------------
# Judge validation against human grades (data/judge_validation.json)
#
# The LLM judge above was never checked against a person, so its scores are
# unverified. This is a small holdout: 3 eval questions, the agent's answer,
# the retrieved chunks the answer was built from, a human's faithfulness and
# relevance scores, and the judge's scores. Motivated by Zhu et al., "Establishing
# Best Practices for Building Rigorous Agentic Benchmarks" (ABC checklist,
# NeurIPS 2025, arXiv:2507.02825) — outcome-validity flaws in graders — and
# Mind2Web 2 (Gou et al., NeurIPS 2025), which validated its judge against humans.
#
# Every run re-judges the stored answers with the CURRENT judge prompt/model
# and compares to the stored human scores, so a judge change that breaks
# agreement shows up immediately. Only 3 items: this is a spot-check, not a
# calibration study — 6 scores can't establish a judge accuracy percentage.
# ---------------------------------------------------------------------------

VALIDATION_FILE = Path("data/judge_validation.json")
VALIDATION_IDS = ["q03", "q06", "q21"]   # one per category: method / result / comparison
VALIDATION_CONFIG = {"top_k": 5, "use_critic": True}


def build_validation_set() -> None:
    """Generate the agent answers + judge scores for the holdout questions and
    write data/judge_validation.json with human_scores left null for a person
    to fill in. Refuses to overwrite a file that already has human grades."""
    if VALIDATION_FILE.exists():
        existing = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
        if any(i.get("human_scores") for i in existing["items"]):
            sys.exit(f"{VALIDATION_FILE} already contains human grades; not overwriting.")

    by_id = {q["id"]: q for q in load_eval_set()}
    items = []
    for qid in VALIDATION_IDS:
        q = by_id[qid]
        result = ask(q["question"], topic=EVAL_TOPIC, **VALIDATION_CONFIG)
        score = score_answer(q["question"], q["expected_answer"], result["answer"], result["retrieved"])
        items.append({
            "id": qid,
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "sources": result["sources"],
            "retrieved_context": [r["text"] for r in result["retrieved"]],
            "agent_answer": result["answer"],
            "judge_scores_at_creation": {
                "faithfulness": score.faithfulness, "relevance": score.relevance,
                "reasoning": score.reasoning,
            },
            "human_scores": None,   # {"faithfulness": 1-5, "relevance": 1-5}
            "human_notes": "",
        })
    VALIDATION_FILE.write_text(json.dumps({
        "purpose": "Spot-check of the LLM judge against human grades (ABC checklist, arXiv:2507.02825).",
        "agent_config": VALIDATION_CONFIG,
        "judge_model": JUDGE_MODEL,
        "grading_rubric": "faithfulness 1-5: every claim backed by the retrieved context, nothing "
                          "fabricated. relevance 1-5: covers the reference facts and answers the question.",
        "graded_by": None,
        "items": items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {VALIDATION_FILE} with {len(items)} items; fill in human_scores/graded_by.")


def judge_validation_report() -> str:
    """Re-judge the stored holdout answers with the current judge, compare to
    the stored human scores, print and return a one-line summary."""
    if not VALIDATION_FILE.exists():
        msg = f"Judge validation: {VALIDATION_FILE} not found, skipped."
        print(msg)
        return msg

    data = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
    graded = [i for i in data["items"] if i.get("human_scores")]
    if not graded:
        msg = f"Judge validation: 0/{len(data['items'])} items human-graded yet, nothing to compare."
        print(msg)
        return msg

    exact = within1 = total = 0
    disagreements = []
    for item in graded:
        judged = score_answer(item["question"], item["expected_answer"], item["agent_answer"],
                              [{"text": t} for t in item["retrieved_context"]])
        for dim in ("faithfulness", "relevance"):
            h, j = item["human_scores"][dim], getattr(judged, dim)
            total += 1
            exact += h == j
            within1 += abs(h - j) <= 1
            if h != j:
                disagreements.append(f"{item['id']} {dim}: human {h} vs judge {j}")

    msg = (f"Judge validation ({len(graded)} human-graded questions, {total} scores): "
           f"{exact}/{total} match exactly, {within1}/{total} within +/-1"
           + (f"; disagreements: {'; '.join(disagreements)}" if disagreements else "; no disagreements"))
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Judge robustness test (data/judge_robustness.json)
#
# Before trusting any eval scores, check the judge doesn't hand out good marks
# to trivial or gameable answers. Motivated by Zheng et al., "Cheating
# Automatic LLM Benchmarks: Null Models Achieve High Win Rates" (ICLR 2025,
# arXiv:2410.07137) — a constant, content-free response topped AlpacaEval 2.0
# — and Kapoor et al., HAL (ICLR 2026), which found agents exploiting
# benchmarks instead of solving the tasks.
#
# Three degenerate answers are judged against one randomly chosen (seeded, so
# reproducible) eval question, with the top-5 excerpts local retrieval returns
# for it as the judge's context (no LLM call needed to build that context).
# All cases go through score_answer(), the same path the eval uses, so the
# empty case is caught by its rule rather than by the judge itself:
#   empty / one-word "unknown" -> expect faithfulness and relevance both <= 2
#   confident but made-up      -> expect faithfulness <= 2 (relevance is also
#                                 recorded; the invented answer matches no
#                                 reference fact so it should be low too)
# Any scored dimension above ROBUSTNESS_WARN_ABOVE (3) prints a warning.
# ---------------------------------------------------------------------------

ROBUSTNESS_FILE = Path("data/judge_robustness.json")
ROBUSTNESS_SEED = 0
ROBUSTNESS_WARN_ABOVE = 3
ROBUSTNESS_CASES = [
    # (name, answer, dimensions the judge is expected to score <= 2 on)
    ("empty_response", "", ("faithfulness", "relevance")),
    ("one_word_unknown", "unknown", ("faithfulness", "relevance")),
    ("plausible_made_up", "The optimal method is XGBoost with a learning rate of 0.001.", ("faithfulness",)),
]


def judge_robustness_test() -> str:
    """Run the degenerate-answer cases through the judge, save results to
    data/judge_robustness.json, print them and return a one-line summary."""
    import random
    item = random.Random(ROBUSTNESS_SEED).choice(load_eval_set())
    context = retrieve(item["question"], k=5)

    cases, warnings = [], []
    for name, answer, expected_low in ROBUSTNESS_CASES:
        score = score_answer(item["question"], item["expected_answer"], answer, context)
        actual = {"faithfulness": score.faithfulness, "relevance": score.relevance}
        for dim in expected_low:
            if actual[dim] > ROBUSTNESS_WARN_ABOVE:
                warnings.append(f"Judge robustness: {name} scored {actual[dim]} on {dim}, "
                                f"may not catch gaming")
        cases.append({
            "case": name,
            "answer": answer,
            "expected": {dim: "<= 2" for dim in expected_low},
            "actual": actual,
            "reasoning": score.reasoning,
            "within_expected": all(actual[d] <= 2 for d in expected_low),
            "warned": any(actual[d] > ROBUSTNESS_WARN_ABOVE for d in expected_low),
        })

    ROBUSTNESS_FILE.write_text(json.dumps({
        "purpose": "Check the LLM judge does not reward trivial or made-up answers "
                   "(Zheng et al., arXiv:2410.07137; Kapoor et al., HAL, ICLR 2026).",
        "judge_model": JUDGE_MODEL,
        "seed": ROBUSTNESS_SEED,
        "question_id": item["id"],
        "question": item["question"],
        "judge_version": JUDGE_VERSION,
        "context_sources_given_to_judge": [c["source"] for c in context],
        "cases": cases,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Judge robustness test (question {item['id']}, seed {ROBUSTNESS_SEED}):")
    for c in cases:
        exp = ", ".join(c["expected"])
        print(f"  {c['case']:<20} faithfulness={c['actual']['faithfulness']} "
              f"relevance={c['actual']['relevance']}  (expected <=2 on: {exp})  "
              f"{'ok' if c['within_expected'] else 'OUTSIDE EXPECTED'}")
    for w in warnings:
        print(f"  WARNING: {w}")
    n_ok = sum(c["within_expected"] for c in cases)
    msg = f"Judge robustness: {n_ok}/{len(cases)} degenerate answers scored as expected" + (
        f"; {len(warnings)} warning(s): " + "; ".join(warnings) if warnings else "")
    if not warnings:
        print(f"  all degenerate answers scored <= {ROBUSTNESS_WARN_ABOVE}")
    return msg


CONFIGS = [
    # (top_k, use_critic, run_name) — the "_x3" suffix keeps these separate from
    # the earlier single-attempt runs (topk3_critic_off etc.) in MLflow and data/eval_runs/.
    (3, False, "topk3_critic_off_x3"),
    (5, False, "topk5_critic_off_x3"),
    (5, True, "topk5_critic_on_x3"),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N questions (quick check). Runs are "
                             "suffixed _limitN so they don't mix with full runs.")
    parser.add_argument("--build-validation", action="store_true",
                        help="Generate data/judge_validation.json (agent answers + judge scores, "
                             "human_scores left blank) and exit.")
    args = parser.parse_args()

    if args.build_validation:
        build_validation_set()
        sys.exit(0)

    mlflow.set_experiment("research-agent-eval")

    # Judge sanity checks run first: if the judge is gameable or disagrees
    # with humans, the scores below should be read with that in mind.
    robustness_note = judge_robustness_test()
    print()
    validation_note = judge_validation_report()
    print()
    run_note = f"{robustness_note}\n{validation_note}"

    all_results = {}
    for i, (top_k, use_critic, run_name) in enumerate(CONFIGS, 1):
        if args.limit:
            run_name += f"_limit{args.limit}"
        print(f"Run {i}/{len(CONFIGS)}: top_k={top_k}, critic {'on' if use_critic else 'off'}, "
              f"{N_ATTEMPTS} attempts/question")
        all_results[run_name] = run_eval(top_k=top_k, use_critic=use_critic, run_name=run_name,
                                         limit=args.limit, run_note=run_note)

    print_summary(all_results)
