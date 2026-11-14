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

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading a RAG system's answer to a question about research "
     "papers on {topic}. Score faithfulness "
     "(is every claim backed by the cited sources, with nothing fabricated) "
     "and relevance (does the answer cover the reference facts) on a 1-5 "
     "scale each. Be strict — a plausible-sounding but unsupported number "
     "is a faithfulness failure even if it's close to correct."),
    ("human",
     "Question: {question}\n\n"
     "Reference facts a good answer should cover:\n{expected_answer}\n\n"
     "Sources the system retrieved:\n{sources}\n\n"
     "System's answer:\n{answer}\n\n"
     "Score faithfulness and relevance."),
])
judge_chain = judge_prompt | judge_llm.with_structured_output(JudgeScore)


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))


def attempt_succeeded(faithfulness: int, relevance: int) -> bool:
    return faithfulness >= PASS_SCORE and relevance >= PASS_SCORE


def run_eval(top_k: int, use_critic: bool, run_name: str, limit: int | None = None) -> dict:
    eval_set = load_eval_set()[:limit]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "top_k": top_k,
            "use_critic": use_critic,
            "topic": EVAL_TOPIC,
            "n_questions": len(eval_set),
            "n_attempts": N_ATTEMPTS,
            "pass_score": PASS_SCORE,
            "judge_model": JUDGE_MODEL,
        })

        faithfulness_scores, relevance_scores = [], []  # one entry per attempt
        per_question_log = []
        start = time.time()

        for item in eval_set:
            attempts = []
            for attempt_no in range(1, N_ATTEMPTS + 1):
                result = ask(item["question"], topic=EVAL_TOPIC, top_k=top_k, use_critic=use_critic)
                score: JudgeScore = judge_chain.invoke({
                    "topic": EVAL_TOPIC,
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "sources": ", ".join(result["sources"]) or "(none retrieved)",
                    "answer": result["answer"],
                })
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
    args = parser.parse_args()

    mlflow.set_experiment("research-agent-eval")

    all_results = {}
    for i, (top_k, use_critic, run_name) in enumerate(CONFIGS, 1):
        if args.limit:
            run_name += f"_limit{args.limit}"
        print(f"Run {i}/{len(CONFIGS)}: top_k={top_k}, critic {'on' if use_critic else 'off'}, "
              f"{N_ATTEMPTS} attempts/question")
        all_results[run_name] = run_eval(top_k=top_k, use_critic=use_critic, run_name=run_name, limit=args.limit)

    print_summary(all_results)
