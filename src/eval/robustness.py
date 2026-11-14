"""
Judge robustness test (data/judge_robustness.json).

Before trusting any eval scores, check the judge doesn't hand out good marks
to trivial or gameable answers. Motivated by Zheng et al., "Cheating
Automatic LLM Benchmarks: Null Models Achieve High Win Rates" (ICLR 2025,
arXiv:2410.07137) — a constant, content-free response topped AlpacaEval 2.0 —
and Kapoor et al., HAL (ICLR 2026), which found agents exploiting benchmarks
instead of solving the tasks.

Three degenerate answers are judged against one randomly chosen (seeded, so
reproducible) eval question, with the top-5 excerpts local retrieval returns
for it as the judge's context (no LLM call needed to build that context).
All cases go through score_answer(), the same path the eval uses, so the
empty case is caught by its rule rather than by the judge itself:
  empty / one-word "unknown" -> expect faithfulness and relevance both <= 2
  confident but made-up      -> expect faithfulness <= 2 (relevance is also
                                recorded; the invented answer matches no
                                reference fact so it should be low too)
Any scored dimension above ROBUSTNESS_WARN_ABOVE (3) prints a warning.
"""

import json
import random

from core.generate import retrieve
from eval.judge import JUDGE_MODEL, JUDGE_VERSION, score_answer
from eval.settings import ROBUSTNESS_FILE, load_eval_set

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
