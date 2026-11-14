"""
Judge validation against human grades (data/judge_validation.json).

The LLM judge was never checked against a person, so its scores were
unverified. This is a small holdout: 3 eval questions, the agent's answer,
the retrieved chunks the answer was built from, a human's faithfulness and
relevance scores, and the judge's scores. Motivated by Zhu et al.,
"Establishing Best Practices for Building Rigorous Agentic Benchmarks" (ABC
checklist, NeurIPS 2025, arXiv:2507.02825) — outcome-validity flaws in
graders — and Mind2Web 2 (Gou et al., NeurIPS 2025), which validated its
judge against humans.

Every run re-judges the stored answers with the CURRENT judge prompt/model and
compares to the stored human scores, so a judge change that breaks agreement
shows up immediately. Only 3 items: this is a spot-check, not a calibration
study — 6 scores can't establish a judge accuracy percentage.

Human scores must come from a person. Never fill them with model output.
"""

import json
import sys

from core.graph import ask
from eval.judge import JUDGE_MODEL, score_answer
from eval.settings import EVAL_TOPIC, VALIDATION_FILE, load_eval_set

VALIDATION_IDS = ["q03", "q06", "q21"]   # one per category: method / result / comparison
VALIDATION_CONFIG = {"top_k": 5, "use_critic": True}


def load_graded_items() -> dict[str, dict]:
    """Holdout items that have human scores, keyed by question id."""
    if not VALIDATION_FILE.exists():
        return {}
    items = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))["items"]
    return {i["id"]: i for i in items if i.get("human_scores")}


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
