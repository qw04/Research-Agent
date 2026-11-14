"""
Re-judge stored eval runs with the aggregated (v3) judge, without re-running
the agent.

Why: pass^k repeats the agent but, in v2 runs, judges each attempt once, so a
gap between pass^1 and pass^3 mixes two sources of variation — the agent
giving different answers, and the judge giving different verdicts. Kumar,
Rathore & Moitra (arXiv:2610.12083) show verdict reliability is item-specific
and Haldar & Hockenmaier ("Rating Roulette", arXiv:2510.27106) show low
intra-rater reliability across runs, so the judge's share can't be assumed
to be zero. Re-scoring the SAME stored answers with a majority-of-N judge
(score_answer_aggregated) removes most of the judge's share, so whatever gap
remains is mostly the agent.

Needs runs that stored each attempt's retrieved excerpts ("retrieved"), i.e.
runs made after the judge audit was added (the *_ctx runs).
Writes data/eval_runs/<source>_<tag>.json in the same format as pass_k.py, so
--stats works on it unchanged, and logs an MLflow run with the same metrics.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow

from eval.judge import JUDGE_MODEL, judge_version_for, score_answer_aggregated
from eval.settings import (EVAL_TOPIC, MLFLOW_EXPERIMENT, N_ATTEMPTS, PASS_SCORE, RUNS_DIR,
                           load_eval_set)


def rejudge_tag(n: int, mode: str) -> str:
    return f"v3{mode[:3]}{n}"          # e.g. v3ens9, v3rep5


def rejudge_run(source_run: str, n: int, mode: str) -> dict:
    run = json.loads((RUNS_DIR / f"{source_run}.json").read_text(encoding="utf-8"))
    missing = [q["id"] for q in run for a in q["attempts"] if "retrieved" not in a]
    if missing:
        raise ValueError(f"{source_run} has attempts without stored context ({sorted(set(missing))}); "
                         "re-judging needs a run that logged 'retrieved' (e.g. a *_ctx run).")
    ref = {q["id"]: q for q in load_eval_set()}
    version = judge_version_for(n, mode)
    out_name = f"{source_run}_{rejudge_tag(n, mode)}"

    jobs = [(qi, ai) for qi, q in enumerate(run) for ai in range(len(q["attempts"]))]
    start = time.time()
    print(f"Re-judging {source_run}: {len(jobs)} attempts x {n} calls ({mode}) = {len(jobs) * n} judge calls")

    def work(job):
        qi, ai = job
        q, a = run[qi], run[qi]["attempts"][ai]
        return score_answer_aggregated(q["question"], ref[q["id"]]["expected_answer"], a["answer"],
                                       a["retrieved"], n, mode)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, jobs))

    new_run = []
    for qi, q in enumerate(run):
        attempts = []
        for ai, a in enumerate(q["attempts"]):
            s = results[jobs.index((qi, ai))]
            attempts.append({**a,
                             "faithfulness": s["faithfulness"], "relevance": s["relevance"],
                             "success": s["success"], "judge_calls": s["calls"],
                             "judge_reasoning": s["reasoning"], "judge_version": version,
                             "v2_single_call": {"faithfulness": a["faithfulness"], "relevance": a["relevance"],
                                                "success": a["success"]}})
        n_success = sum(x["success"] for x in attempts)
        new_run.append({**{k: v for k, v in q.items() if k != "attempts"},
                        "n_success": n_success, "pass_any3": n_success >= 1,
                        "pass3": n_success == N_ATTEMPTS, "attempts": attempts})

    n_q = len(new_run)
    all_attempts = [a for q in new_run for a in q["attempts"]]
    metrics = {
        "avg_faithfulness": sum(a["faithfulness"] for a in all_attempts) / len(all_attempts),
        "avg_relevance": sum(a["relevance"] for a in all_attempts) / len(all_attempts),
        "pass1_rate": sum(q["n_success"] for q in new_run) / (n_q * N_ATTEMPTS),
        "pass_any3_rate": sum(q["pass_any3"] for q in new_run) / n_q,
        "pass3_rate": sum(q["pass3"] for q in new_run) / n_q,
        "verdicts_changed_vs_v2": sum(a["success"] != a["v2_single_call"]["success"] for a in all_attempts),
        "duration_seconds": time.time() - start,
    }

    log_path = RUNS_DIR / f"{out_name}.json"
    log_path.write_text(json.dumps(new_run, indent=2), encoding="utf-8")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=out_name):
        mlflow.log_params({"source_run": source_run, "topic": EVAL_TOPIC, "n_questions": n_q,
                           "n_attempts": N_ATTEMPTS, "pass_score": PASS_SCORE, "judge_model": JUDGE_MODEL,
                           "judge_version": version, "judge_n": n, "judge_mode": mode,
                           "agent_rerun": False})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(log_path))

    print(f"[{out_name}] pass^1 {metrics['pass1_rate']:.0%}  >=1 of 3 {metrics['pass_any3_rate']:.0%}  "
          f"pass^3 {metrics['pass3_rate']:.0%}  avg faith {metrics['avg_faithfulness']:.2f}  "
          f"avg rel {metrics['avg_relevance']:.2f}  verdicts changed vs single-call v2: "
          f"{metrics['verdicts_changed_vs_v2']}/{len(all_attempts)}")
    return {"name": out_name, **metrics}
