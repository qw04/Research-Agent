"""
Command-line entry point for the eval. Run from the repo root.

    python src/eval/evaluate.py                     full eval: judge checks, then pass^k over CONFIGS
    python src/eval/evaluate.py --limit 2           quick check on the first 2 questions
    python src/eval/evaluate.py --build-validation  regenerate the human-validation holdout
    python src/eval/evaluate.py --judge-audit       per-item judge reliability audit

The code lives in the other modules of this package:
    settings.py     paths, eval topic, pass^k parameters, success rule
    judge.py        LLM judge: prompt (v2-context), answer-first variant, score_answer()
    pass_k.py       agent runs x3 per question, per-config metrics, MLflow logging
    validation.py   human-graded holdout and agreement report
    robustness.py   degenerate-answer test (empty / "unknown" / made-up)
    audit.py        reproducibility + order-invariance audit, trustworthy verdict rate
See documentation.md for why each piece exists.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, so core.* / eval.* resolve

import mlflow

from eval.audit import AUDIT_REPS, AUDIT_SOURCE_RUN, judge_audit
from eval.pass_k import CONFIGS, print_summary, run_eval
from eval.robustness import judge_robustness_test
from eval.settings import MLFLOW_EXPERIMENT, N_ATTEMPTS
from eval.validation import build_validation_set, judge_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N questions (quick check). Runs are "
                             "suffixed _limitN so they don't mix with full runs.")
    parser.add_argument("--build-validation", action="store_true",
                        help="Generate data/judge_validation.json (agent answers + judge scores, "
                             "human_scores left blank) and exit.")
    parser.add_argument("--judge-audit", action="store_true",
                        help=f"Re-judge stored answers from {AUDIT_SOURCE_RUN} (attempt 1) {AUDIT_REPS}x normal "
                             f"+ {AUDIT_REPS}x answer-first, report the trustworthy verdict rate, and exit. "
                             "Does not re-run the agent.")
    args = parser.parse_args()

    if args.judge_audit:
        judge_audit()
        return

    if args.build_validation:
        build_validation_set()
        return

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

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


if __name__ == "__main__":
    main()
