"""
Uncertainty for the eval metrics: bootstrap 95% intervals.

With 22 questions, one question moves pass^3 by ~4.5 percentage points, so
point estimates alone can't say whether one configuration beats another.
Following the uncertainty-reporting recommendation in Yagubyan, "The Coin Flip
Judge?" (arXiv:2606.13685, preprint), every metric gets a percentile bootstrap
interval: resample the 22 QUESTIONS with replacement (BOOTSTRAP_RESAMPLES
times, fixed seed) and recompute. Comparisons between configurations use a
PAIRED bootstrap (the same resampled questions for both runs), since both
runs answer the same questions.

What the interval does and doesn't cover: it reflects which questions happen
to be in the eval set. It does not include judge noise (each attempt is
judged once in v2 runs) or the agent's own randomness beyond the 3 attempts.
Conformal / interval evaluation (Sheng et al., EMNLP 2025) would need a much
larger human-labelled calibration set and is left as future work.

Also reports, per configuration, whether each question's 3 answer texts were
identical: if the texts are identical but the verdicts differ, that variation
can only come from the judge (Haldar & Hockenmaier, "Rating Roulette",
arXiv:2510.27106, on intra-rater reliability of LLM judges).
"""

import json

import mlflow
import numpy as np

from eval.settings import MLFLOW_EXPERIMENT, N_ATTEMPTS, RUNS_DIR

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
METRICS = ["pass1_rate", "pass_any3_rate", "pass3_rate", "avg_faithfulness", "avg_relevance"]

# (a, b, label): report metric(a) - metric(b), paired over questions.
COMPARISONS = [
    ("topk5_critic_on", "topk5_critic_off", "critic on - off (top_k=5)"),
    ("topk5_critic_off", "topk3_critic_off", "top_k 5 - 3 (critic off)"),
]
BASE_RUNS = ["topk3_critic_off", "topk5_critic_off", "topk5_critic_on"]


def load_run(name: str) -> list[dict]:
    return json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def per_question_arrays(run: list[dict]) -> dict[str, np.ndarray]:
    """One value per question for each metric; each metric is the mean of
    these over questions (equal attempts per question, so the per-attempt
    averages logged by pass_k.py are the same numbers)."""
    run = sorted(run, key=lambda q: q["id"])
    return {
        "pass1_rate": np.array([q["n_success"] / N_ATTEMPTS for q in run]),
        "pass_any3_rate": np.array([float(q["pass_any3"]) for q in run]),
        "pass3_rate": np.array([float(q["pass3"]) for q in run]),
        "avg_faithfulness": np.array([np.mean([a["faithfulness"] for a in q["attempts"]]) for q in run]),
        "avg_relevance": np.array([np.mean([a["relevance"] for a in q["attempts"]]) for q in run]),
    }


def _ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap(runs: dict[str, list[dict]], comparisons=COMPARISONS) -> dict:
    """runs: {base_name: run_log}. Returns point estimates, 95% intervals and
    paired differences. All runs must cover the same question ids."""
    ids = None
    arrays = {}
    for name, run in runs.items():
        run_ids = sorted(q["id"] for q in run)
        if ids is None:
            ids = run_ids
        elif run_ids != ids:
            raise ValueError(f"{name} covers different questions; paired bootstrap needs identical sets")
        arrays[name] = per_question_arrays(run)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(ids), size=(BOOTSTRAP_RESAMPLES, len(ids)))   # shared -> paired

    out = {"n_questions": len(ids), "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
           "configs": {}, "comparisons": []}
    for name, arr in arrays.items():
        out["configs"][name] = {
            m: {"point": float(arr[m].mean()), "ci95": _ci(arr[m][idx].mean(axis=1))} for m in METRICS
        }
    for a, b, label in comparisons:
        if a not in arrays or b not in arrays:
            continue
        comp = {"label": label, "a": a, "b": b, "metrics": {}}
        for m in METRICS:
            diff = arrays[a][m] - arrays[b][m]
            lo, hi = _ci(diff[idx].mean(axis=1))
            comp["metrics"][m] = {"point": float(diff.mean()), "ci95": (lo, hi),
                                  "excludes_zero": bool(lo > 0 or hi < 0)}
        out["comparisons"].append(comp)
    return out


def variance_breakdown(run: list[dict]) -> dict:
    """Split questions by whether their 3 answer texts were identical, and
    list the questions whose success verdict varied across attempts."""
    identical, varied_identical, varied_different = [], [], []
    for q in sorted(run, key=lambda q: q["id"]):
        same_text = len({a["answer"] for a in q["attempts"]}) == 1
        verdict_varied = 0 < q["n_success"] < len(q["attempts"])
        if same_text:
            identical.append(q["id"])
            if verdict_varied:
                varied_identical.append(q["id"])
        elif verdict_varied:
            varied_different.append(q["id"])
    return {
        "identical_text_questions": identical,
        "verdict_varied_identical_text": varied_identical,      # can only be judge variation
        "verdict_varied_different_text": varied_different,      # agent and/or judge variation
    }


def _fmt(m: str, v: float) -> str:
    return f"{v:.0%}" if m.endswith("_rate") else f"{v:.2f}"


def run_stats(suffix: str = "_x3") -> dict:
    """Bootstrap the three configurations' logs data/eval_runs/<base><suffix>.json,
    save data/eval_stats<suffix>.json, print, and log to MLflow."""
    runs = {b: load_run(b + suffix) for b in BASE_RUNS if (RUNS_DIR / f"{b + suffix}.json").exists()}
    if not runs:
        raise FileNotFoundError(f"no runs found for suffix {suffix!r} in {RUNS_DIR}")
    result = bootstrap(runs)
    result["suffix"] = suffix
    result["judge_versions"] = sorted({a.get("judge_version", "v2-context")
                                       for run in runs.values() for q in run for a in q["attempts"]})
    result["variance_breakdown"] = {name: variance_breakdown(run) for name, run in runs.items()}

    out_file = RUNS_DIR.parent / f"eval_stats{suffix}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Bootstrap 95% intervals ({result['n_questions']} questions resampled, "
          f"{BOOTSTRAP_RESAMPLES:,} resamples, seed {BOOTSTRAP_SEED}; runs *{suffix}; "
          f"judge {', '.join(result['judge_versions'])})")
    for name, ms in result["configs"].items():
        cells = "  ".join(f"{m.replace('_rate', '').replace('avg_', '')}={_fmt(m, v['point'])} "
                          f"[{_fmt(m, v['ci95'][0])}, {_fmt(m, v['ci95'][1])}]" for m, v in ms.items())
        print(f"  {name:<18} {cells}")
    print("Paired differences (interval excludes zero = difference distinguishable from question sampling):")
    for c in result["comparisons"]:
        print(f"  {c['label']}")
        for m, v in c["metrics"].items():
            sign = "excludes 0" if v["excludes_zero"] else "includes 0"
            print(f"    {m:<17} {v['point']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {sign}")
    print("Identical answer text across attempts / verdict varied anyway (judge-only variation):")
    for name, vb in result["variance_breakdown"].items():
        print(f"  {name:<18} identical text {len(vb['identical_text_questions'])}/{result['n_questions']}; "
              f"verdict varied with identical text: {vb['verdict_varied_identical_text'] or 'none'}; "
              f"with different text: {vb['verdict_varied_different_text'] or 'none'}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"stats{suffix}"):
        mlflow.log_params({"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                           "runs": ",".join(b + suffix for b in runs),
                           "judge_version": ",".join(result["judge_versions"])})
        metrics = {}
        for name, ms in result["configs"].items():
            for m, v in ms.items():
                metrics[f"{name}.{m}"] = v["point"]
                metrics[f"{name}.{m}.ci_low"], metrics[f"{name}.{m}.ci_high"] = v["ci95"]
        for c in result["comparisons"]:
            key = f"diff.{c['a']}-{c['b']}"
            for m, v in c["metrics"].items():
                metrics[f"{key}.{m}"] = v["point"]
                metrics[f"{key}.{m}.ci_low"], metrics[f"{key}.{m}.ci_high"] = v["ci95"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_file))
    return result
