"""
Command-line entry point for the eval. Run from the repo root.

    python src/eval/evaluate.py                     full eval: judge checks, then pass^k over CONFIGS
    python src/eval/evaluate.py --limit 2           quick check on the first 2 questions
    python src/eval/evaluate.py --build-validation  regenerate the human-validation holdout
    python src/eval/evaluate.py --judge-audit       per-item judge reliability audit
    python src/eval/evaluate.py --stats             bootstrap 95% intervals for the *_x3 runs (no API calls)
    python src/eval/evaluate.py --suffix _ctx       name new runs *_x3_ctx (also selects runs for --stats/--rejudge/--judge-audit)
    python src/eval/evaluate.py --suffix _ctx --rejudge --judge-n 9
                                                    re-judge stored *_x3_ctx attempts with a majority-of-9 judge

The code lives in the other modules of this package:
    settings.py     paths, eval topic, pass^k parameters, success rule
    judge.py        LLM judge: prompt v2, answer-first + paraphrase variants, score_answer(),
                    aggregated majority-of-N judge (v3)
    pass_k.py       agent runs x3 per question, per-config metrics, MLflow logging
    validation.py   human-graded holdout and agreement report
    robustness.py   degenerate-answer test (empty / "unknown" / made-up)
    audit.py        reproducibility + order-invariance audit, trustworthy verdict rate
    stats.py        bootstrap intervals, paired differences, identical-text breakdown
    rejudge.py      re-score stored attempts with the aggregated judge (no agent calls)
See documentation.md for why each piece exists.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, so core.* / eval.* resolve

import mlflow

from eval.audit import AUDIT_REPS, AUDIT_SOURCE_RUN, judge_audit
from eval.judge import JUDGE_MODES, check_judge_n
from eval.rejudge import rejudge_run, rejudge_tag
from eval.pass_k import CONFIGS, print_summary, run_eval
from eval.robustness import judge_robustness_test
from eval.settings import MLFLOW_EXPERIMENT, N_ATTEMPTS
from eval.stats import run_stats
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
                        help=f"Re-judge stored answers from {AUDIT_SOURCE_RUN}<suffix> (attempt 1) {AUDIT_REPS}x "
                             "each with the normal, answer-first and paraphrased judge prompts, report the "
                             "trustworthy verdict rate, and exit. Does not re-run the agent.")
    parser.add_argument("--suffix", default="",
                        help="Extra suffix for run names, e.g. _ctx -> topk5_critic_on_x3_ctx. Used by the "
                             "eval and by --stats, so new runs never overwrite earlier ones.")
    parser.add_argument("--stats", action="store_true",
                        help="Bootstrap 95%% intervals (and paired config differences) for the runs "
                             "data/eval_runs/<config>_x3<suffix>.json, then exit. No API calls.")
    parser.add_argument("--judge-n", type=int, default=1,
                        help="Judge each answer N times and take the majority success verdict (odd N; "
                             "N=1, the default, is the single-call v2 judge). Applies to the eval, "
                             "--rejudge and the judge checks.")
    parser.add_argument("--judge-mode", choices=JUDGE_MODES, default="ensemble",
                        help="With --judge-n > 1: 'ensemble' cycles the normal, answer-first and "
                             "paraphrased prompts (N a multiple of 3); 'repeat' reuses the normal prompt.")
    parser.add_argument("--rejudge", action="store_true",
                        help="Re-judge every stored attempt of the three <config>_x3<suffix> runs with "
                             "the --judge-n/--judge-mode judge (no agent calls), run the judge checks and "
                             "--stats on the result, then exit.")
    args = parser.parse_args()
    try:
        check_judge_n(args.judge_n, args.judge_mode)
    except ValueError as e:
        parser.error(str(e))

    if args.stats:
        run_stats("_x3" + args.suffix)
        return

    if args.rejudge:
        if args.judge_n == 1:
            parser.error("--rejudge needs --judge-n > 1 (N=1 would just repeat the original judging)")
        judge_robustness_test(args.judge_n, args.judge_mode)
        print()
        judge_validation_report(args.judge_n, args.judge_mode)
        print()
        for _, _, run_name in CONFIGS:
            rejudge_run(run_name + args.suffix, args.judge_n, args.judge_mode)
        print()
        run_stats("_x3" + args.suffix + "_" + rejudge_tag(args.judge_n, args.judge_mode))
        return

    if args.judge_audit:
        judge_audit(AUDIT_SOURCE_RUN + args.suffix)
        return

    if args.build_validation:
        build_validation_set()
        return

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Judge sanity checks run first: if the judge is gameable or disagrees
    # with humans, the scores below should be read with that in mind.
    robustness_note = judge_robustness_test(args.judge_n, args.judge_mode)
    print()
    validation_note = judge_validation_report(args.judge_n, args.judge_mode)
    print()
    run_note = f"{robustness_note}\n{validation_note}"

    all_results = {}
    for i, (top_k, use_critic, run_name) in enumerate(CONFIGS, 1):
        run_name += args.suffix
        if args.limit:
            run_name += f"_limit{args.limit}"
        print(f"Run {i}/{len(CONFIGS)}: top_k={top_k}, critic {'on' if use_critic else 'off'}, "
              f"{N_ATTEMPTS} attempts/question")
        all_results[run_name] = run_eval(top_k=top_k, use_critic=use_critic, run_name=run_name,
                                         limit=args.limit, run_note=run_note,
                                         judge_n=args.judge_n, judge_mode=args.judge_mode)

    print_summary(all_results)


if __name__ == "__main__":
    main()
