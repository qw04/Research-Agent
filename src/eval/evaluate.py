"""
Runs the hand-built eval set (data/eval_set.json) through the full agent
(graph.ask), scores each answer for faithfulness (grounded in the retrieved
sources, no fabricated claims) and relevance (covers the expected facts)
using an LLM-as-judge, and logs per-question and aggregate scores to MLflow.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from core.graph import ask

load_dotenv()

EVAL_SET_FILE = Path("data/eval_set.json")
RUNS_DIR = Path("data/eval_runs")
JUDGE_MODEL = "gpt-4o-mini"

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


judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading a RAG system's answer to a question about battery "
     "state-of-charge (SOC) estimation research papers. Score faithfulness "
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


def run_eval(top_k: int, use_critic: bool, run_name: str) -> dict:
    eval_set = load_eval_set()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "top_k": top_k,
            "use_critic": use_critic,
            "n_questions": len(eval_set),
            "judge_model": JUDGE_MODEL,
        })

        faithfulness_scores, relevance_scores = [], []
        per_question_log = []
        start = time.time()

        for item in eval_set:
            result = ask(item["question"], top_k=top_k, use_critic=use_critic)
            score: JudgeScore = judge_chain.invoke({
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "sources": ", ".join(result["sources"]) or "(none retrieved)",
                "answer": result["answer"],
            })
            faithfulness_scores.append(score.faithfulness)
            relevance_scores.append(score.relevance)
            per_question_log.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": result["answer"],
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "judge_reasoning": score.reasoning,
                "critic_approved": result["approved"],
                "attempts": result["attempts"],
                "used_arxiv": result["used_arxiv"],
            })
            print(f"  {item['id']} faithfulness={score.faithfulness} relevance={score.relevance}")

        duration = time.time() - start
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        mlflow.log_metrics({
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "duration_seconds": duration,
        })

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUNS_DIR / f"{run_name}.json"
        log_path.write_text(json.dumps(per_question_log, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(log_path))

        print(f"[{run_name}] avg_faithfulness={avg_faithfulness:.2f} "
              f"avg_relevance={avg_relevance:.2f} duration={duration:.1f}s\n")
        return {"avg_faithfulness": avg_faithfulness, "avg_relevance": avg_relevance}


if __name__ == "__main__":
    mlflow.set_experiment("research-agent-eval")

    print("Run 1/3: top_k=3, critic off")
    run_eval(top_k=3, use_critic=False, run_name="topk3_critic_off")

    print("Run 2/3: top_k=5, critic off")
    run_eval(top_k=5, use_critic=False, run_name="topk5_critic_off")

    print("Run 3/3: top_k=5, critic on")
    run_eval(top_k=5, use_critic=True, run_name="topk5_critic_on")
