"""
The main eval: runs every question in data/eval_set.json through the agent
N_ATTEMPTS (3) times per configuration, judges each answer, and logs
per-question and aggregate scores to MLflow and data/eval_runs/<run>.json.

Consistency follows pass^k from tau-bench (Yao et al., 2024, arXiv:2406.12045):
an agent that answers correctly once is not the same as one that answers
correctly every time. An attempt "succeeds" when faithfulness and relevance
both reach PASS_SCORE (5/5). Three rates are logged per configuration:
  - pass1_rate:      mean per-attempt success rate (tau-bench's pass^1)
  - pass_any3_rate:  share of questions with >= 1 success in 3 attempts (the
                     "at least one of three" metric; pass@3 in the usual
                     terminology, so it is >= pass1_rate by construction)
  - pass3_rate:      share of questions where ALL 3 attempts succeeded (pass^3)

Iterates over a small set of configurations (top_k, critic on/off) so MLflow
tracks comparable runs rather than a single one-off result. Chunk size is not
varied, since that requires re-ingesting and re-embedding the corpus
(src/core/ingest.py, src/core/embed_store.py).
"""

import json
import time

import mlflow

from core.graph import ask
from eval.judge import JUDGE_MODEL, judge_version_for, score_answer_aggregated
from eval.settings import EVAL_TOPIC, N_ATTEMPTS, PASS_SCORE, RUNS_DIR, load_eval_set

CONFIGS = [
    # (top_k, use_critic, run_name) — the "_x3" suffix keeps these separate from
    # the earlier single-attempt runs (topk3_critic_off etc.) in MLflow and data/eval_runs/.
    (3, False, "topk3_critic_off_x3"),
    (5, False, "topk5_critic_off_x3"),
    (5, True, "topk5_critic_on_x3"),
]


def run_eval(top_k: int, use_critic: bool, run_name: str, limit: int | None = None,
             run_note: str = "", judge_n: int = 1, judge_mode: str = "ensemble") -> dict:
    """judge_n / judge_mode: see score_answer_aggregated() in judge.py. The
    default (N=1) is the single-call v2 judge."""
    eval_set = load_eval_set()[:limit]
    judge_version = judge_version_for(judge_n, judge_mode)

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
            "judge_version": judge_version,
            "judge_n": judge_n,
            "judge_mode": judge_mode if judge_n > 1 else "single",
        })

        faithfulness_scores, relevance_scores = [], []  # one entry per attempt
        per_question_log = []
        start = time.time()

        for item in eval_set:
            attempts = []
            for attempt_no in range(1, N_ATTEMPTS + 1):
                result = ask(item["question"], topic=EVAL_TOPIC, top_k=top_k, use_critic=use_critic)
                score = score_answer_aggregated(item["question"], item["expected_answer"],
                                                result["answer"], result["retrieved"], judge_n, judge_mode)
                faithfulness_scores.append(score["faithfulness"])
                relevance_scores.append(score["relevance"])
                attempts.append({
                    "attempt": attempt_no,
                    "answer": result["answer"],
                    "faithfulness": score["faithfulness"],
                    "relevance": score["relevance"],
                    "judge_reasoning": score["reasoning"],
                    "critic_approved": result["approved"],
                    "attempts": result["attempts"],
                    "used_arxiv": result["used_arxiv"],
                    "success": score["success"],
                    "judge_calls": score["calls"],
                    "judge_version": judge_version,
                    # stored so the judge audit can re-judge this exact answer
                    # against the exact evidence without re-running the agent
                    "retrieved": [{"source": r["source"], "text": r["text"]} for r in result["retrieved"]],
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
