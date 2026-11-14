"""
Judge audit: per-item verdict reliability (data/judge_audit_<run>.json).

pass^k repeats the AGENT but judges each answer once. Kumar, Rathore & Moitra,
"All Verdicts are Not Equal: Rethinking LLM Judge Reliability"
(arXiv:2610.12083, AACL-IJCNLP 2026) report that judge verdicts can change
across identical reruns even at temperature 0, that presentation order can
flip verdicts, and that reliability is item-specific rather than a property
of the judge model. Their "trustworthy verdict rate" is the share of verdicts
that are reproducible, order-invariant and accurate. Yagubyan, "The Coin Flip
Judge?" (arXiv:2606.13685) adds that meaning-preserving paraphrases of the
judge prompt change a quarter of gpt-4o-mini's majority outcomes, so wording
is audited too. (Kumar et al. study pairwise judging; this is a pointwise
adaptation.)

For each question, attempt 1's stored answer from the source run is judged
AUDIT_REPS times under each condition:
  normal         the v2 judge prompt
  answer_first   same wording, the answer moved before the reference/excerpts
  paraphrase     same rubric and section order, reworded (judge.py)
Verdict = success (5/5 on both). A question is trustworthy iff:
  reproducible:         all normal verdicts agree
  order-invariant:      all answer-first verdicts agree with each other and
                        with the normal majority verdict
  paraphrase-invariant: likewise for the paraphrase verdicts
  accurate:             (only for questions in data/judge_validation.json)
                        every normal judgement of the HUMAN-GRADED answer is
                        within +/-1 of the human score on both dimensions

Also reported, on the human-graded questions only, an adapted version of
Kumar et al.'s T = C * (A - flip/2): C = share of normal verdicts equal to
the majority, flip = share of answer-first verdicts differing from it,
A = share of judgements of the human-graded answer within +/-1 on both
dimensions. It needs human labels, so it can't be computed for all questions.

Context: the audit needs the excerpts the answer was built from. Eval logs
store them ("retrieved" per attempt) since the first audit. For older logs,
the context is rebuilt with retrieve(question, k=top_k), which is exact when
the agent did not reformulate (attempts == 0, no arXiv); questions that went
through the arXiv fallback can't be rebuilt and are reported as
"unverifiable" instead of being judged against the wrong evidence.

The first audit (2 conditions, source run topk5_critic_on_x3) is kept as
data/judge_audit.json; audits from this version write
data/judge_audit_<source run>.json.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import mlflow

from core.generate import retrieve
from eval.judge import (JUDGE_MODEL, JUDGE_VERSION, judge_chain, judge_chain_answer_first,
                        judge_chain_paraphrase, judge_once)
from eval.settings import (MLFLOW_EXPERIMENT, N_ATTEMPTS, PASS_SCORE, RUNS_DIR,
                           attempt_succeeded, load_eval_set)
from eval.validation import load_graded_items

AUDIT_SOURCE_RUN = "topk5_critic_on_x3"   # + --suffix
AUDIT_SOURCE_TOP_K = 5
AUDIT_REPS = 5
CONDITIONS = {"normal": judge_chain, "answer_first": judge_chain_answer_first,
              "paraphrase": judge_chain_paraphrase}


def audit_file(source_run: str):
    return RUNS_DIR.parent / f"judge_audit_{source_run}.json"


def _distribution(scores: list[tuple[int, int]]) -> dict:
    return {
        "faithfulness": dict(sorted(Counter(f for f, _ in scores).items())),
        "relevance": dict(sorted(Counter(r for _, r in scores).items())),
    }


def _spread(scores: list[tuple[int, int]], dim: int) -> int:
    vals = [s[dim] for s in scores]
    return max(vals) - min(vals)


def _majority(verdicts: list[bool]) -> bool:
    return sum(verdicts) * 2 > len(verdicts)


def _agreement(verdicts: list[bool], target: bool) -> float:
    return sum(v == target for v in verdicts) / len(verdicts)


def _pass_rates(rows: list[dict]) -> dict:
    if not rows:
        return {"pass1": None, "pass3": None, "n": 0}
    return {"pass1": sum(q["n_success"] for q in rows) / (len(rows) * N_ATTEMPTS),
            "pass3": sum(q["pass3"] for q in rows) / len(rows), "n": len(rows)}


def judge_audit(source_run: str = AUDIT_SOURCE_RUN) -> dict:
    run = json.loads((RUNS_DIR / f"{source_run}.json").read_text(encoding="utf-8"))
    eval_by_id = {q["id"]: q for q in load_eval_set()}
    validation = load_graded_items()

    # Build the work list: (question id, condition, chain, answer, chunks)
    items, jobs = {}, []
    for q in run:
        a = q["attempts"][0]
        ref = eval_by_id[q["id"]]
        if "retrieved" in a:
            chunks, ctx_source = a["retrieved"], "stored"
        elif a["attempts"] == 0 and not a["used_arxiv"]:
            chunks, ctx_source = retrieve(ref["question"], k=AUDIT_SOURCE_TOP_K), "rebuilt (exact: no reformulation)"
        else:
            chunks, ctx_source = None, "unavailable (arXiv fallback context was not logged)"
        items[q["id"]] = {"ref": ref, "answer": a["answer"], "chunks": chunks, "ctx_source": ctx_source,
                          "logged_score": (a["faithfulness"], a["relevance"])}
        if chunks is None:
            continue
        for _ in range(AUDIT_REPS):
            for cond, chain in CONDITIONS.items():
                jobs.append((q["id"], cond, chain, a["answer"], chunks))
        if q["id"] in validation:   # accuracy: judge the answer the human actually graded
            v = validation[q["id"]]
            for _ in range(AUDIT_REPS):
                jobs.append((q["id"], "human_graded_answer", judge_chain, v["agent_answer"],
                             [{"text": t} for t in v["retrieved_context"]]))

    n_auditable = sum(i["chunks"] is not None for i in items.values())
    print(f"Judge audit: {len(jobs)} judge calls over {n_auditable} auditable questions "
          f"(source run {source_run}, attempt 1; conditions: {', '.join(CONDITIONS)})...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(lambda j: judge_once(j[2], items[j[0]]["ref"]["question"],
                                                  items[j[0]]["ref"]["expected_answer"], j[3], j[4]), jobs))
    scores: dict[str, dict[str, list]] = {}
    for (qid, cond, *_), out in zip(jobs, outs):
        scores.setdefault(qid, {}).setdefault(cond, []).append(out)

    per_question = []
    for qid, it in items.items():
        row = {"id": qid, "category": it["ref"]["category"], "context": it["ctx_source"],
               "logged_eval_score": {"faithfulness": it["logged_score"][0], "relevance": it["logged_score"][1]}}
        if it["chunks"] is None:
            row.update({"auditable": False, "trustworthy": False})
            per_question.append(row)
            continue
        verdicts = {c: [attempt_succeeded(f, r) for f, r in scores[qid][c]] for c in CONDITIONS}
        normal_majority = _majority(verdicts["normal"])
        invariant = {c: len(set(verdicts[c])) == 1 and _majority(verdicts[c]) == normal_majority
                     for c in ("answer_first", "paraphrase")}
        reproducible = len(set(verdicts["normal"])) == 1
        all_scores = [s for c in CONDITIONS for s in scores[qid][c]]
        row.update({
            "auditable": True,
            **{c: {"scores": scores[qid][c], "distribution": _distribution(scores[qid][c]),
                   "consistent": len(set(verdicts[c])) == 1,
                   "agreement_with_own_majority": _agreement(verdicts[c], _majority(verdicts[c]))}
               for c in CONDITIONS},
            "verdict": ("success" if normal_majority else "fail") if reproducible else "mixed",
            "verdict_identical_all": len({v for c in CONDITIONS for v in verdicts[c]}) == 1,
            "faithfulness_spread_gt1": _spread(all_scores, 0) > 1,
            "relevance_spread_gt1": _spread(all_scores, 1) > 1,
            "reproducible": reproducible,
            "order_invariant": invariant["answer_first"],
            "paraphrase_invariant": invariant["paraphrase"],
            "paraphrase_majority_differs": _majority(verdicts["paraphrase"]) != normal_majority,
            "answer_first_majority_differs": _majority(verdicts["answer_first"]) != normal_majority,
            # cross-session check (informational, not part of "trustworthy"):
            # does today's normal judgement match the score logged during the eval run?
            "matches_logged_eval_score": all(s == tuple(it["logged_score"]) for s in scores[qid]["normal"]),
        })
        accurate = None
        if qid in validation:
            h = validation[qid]["human_scores"]
            hg = scores[qid]["human_graded_answer"]
            within = [abs(f - h["faithfulness"]) <= 1 and abs(r - h["relevance"]) <= 1 for f, r in hg]
            accurate = all(within)
            C = _agreement(verdicts["normal"], normal_majority)
            flip = 1 - _agreement(verdicts["answer_first"], normal_majority)
            A = sum(within) / len(within)
            row["accuracy"] = {"human": h, "judge_on_human_graded_answer": hg, "within_1_every_run": accurate,
                               "kumar_T_adapted": {"C": C, "flip": flip, "A": A, "T": C * (A - flip / 2)}}
        row["trustworthy"] = (reproducible and invariant["answer_first"] and invariant["paraphrase"]
                              and accurate is not False)
        per_question.append(row)

    audited = [r for r in per_question if r["auditable"]]
    trusted_ids = {r["id"] for r in per_question if r["trustworthy"]}
    n_trust = len(trusted_ids)
    t_rows = [r["accuracy"]["kumar_T_adapted"]["T"] for r in audited if "accuracy" in r]
    rate = lambda key: sum(r[key] for r in audited) / len(audited)

    summary = {
        "n_questions": len(per_question),
        "n_auditable": len(audited),
        "n_trustworthy": n_trust,
        "trustworthy_verdict_rate": n_trust / len(audited),
        "trustworthy_verdict_rate_all": n_trust / len(per_question),
        "reproducible_rate": rate("reproducible"),
        "order_invariant_rate": rate("order_invariant"),
        "paraphrase_invariant_rate": rate("paraphrase_invariant"),
        "consistency_by_condition": {c: sum(r[c]["consistent"] for r in audited) / len(audited) for c in CONDITIONS},
        "paraphrase_flip_rate": rate("paraphrase_majority_differs"),
        "answer_first_flip_rate": rate("answer_first_majority_differs"),
        "kumar_T_adapted_holdout": {"mean": sum(t_rows) / len(t_rows) if t_rows else None, "n": len(t_rows)},
        "untrustworthy": [r["id"] for r in audited if not r["trustworthy"]],
        "unverifiable": [r["id"] for r in per_question if not r["auditable"]],
        "score_spread_gt1": [r["id"] for r in audited if r["faithfulness_spread_gt1"] or r["relevance_spread_gt1"]],
        "differs_from_logged_eval_score": [r["id"] for r in audited if not r["matches_logged_eval_score"]],
        "pass_rates_full": _pass_rates(run),
        "pass_rates_trustworthy_only": _pass_rates([q for q in run if q["id"] in trusted_ids]),
    }

    out_file = audit_file(source_run)
    out_file.write_text(json.dumps({
        "purpose": "Per-item judge reliability (Kumar, Rathore & Moitra, arXiv:2610.12083; "
                   "Yagubyan, arXiv:2606.13685 for the paraphrase condition).",
        "judge_model": JUDGE_MODEL, "judge_version": JUDGE_VERSION, "reps_per_condition": AUDIT_REPS,
        "conditions": list(CONDITIONS), "source_run": source_run, "source_attempt": 1,
        "verdict_definition": f"success = faithfulness >= {PASS_SCORE} and relevance >= {PASS_SCORE}",
        "summary": summary, "questions": per_question,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_report(per_question, summary, source_run)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    tr = summary["pass_rates_trustworthy_only"]
    with mlflow.start_run(run_name=f"judge_audit_{source_run}"):
        mlflow.log_params({"judge_version": JUDGE_VERSION, "judge_model": JUDGE_MODEL,
                           "reps_per_condition": AUDIT_REPS, "source_run": source_run,
                           "conditions": ",".join(CONDITIONS)})
        metrics = {
            "trustworthy_verdict_rate": summary["trustworthy_verdict_rate"],
            "trustworthy_verdict_rate_all": summary["trustworthy_verdict_rate_all"],
            "reproducible_rate": summary["reproducible_rate"],
            "order_invariant_rate": summary["order_invariant_rate"],
            "paraphrase_invariant_rate": summary["paraphrase_invariant_rate"],
            "paraphrase_flip_rate": summary["paraphrase_flip_rate"],
            "answer_first_flip_rate": summary["answer_first_flip_rate"],
            "pass1_rate_trustworthy": tr["pass1"] or 0.0,
            "pass3_rate_trustworthy": tr["pass3"] or 0.0,
            **{f"consistency_{c}": v for c, v in summary["consistency_by_condition"].items()},
        }
        if summary["kumar_T_adapted_holdout"]["mean"] is not None:
            metrics["kumar_T_adapted_holdout"] = summary["kumar_T_adapted_holdout"]["mean"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_file))
    return summary


def _print_report(per_question: list[dict], s: dict, source_run: str) -> None:
    print(f"\nJudge audit ({AUDIT_REPS} judgements per condition: normal / answer-first / paraphrase):")
    dist = lambda r, c, d: r[c]["distribution"][d]
    for r in per_question:
        if not r["auditable"]:
            print(f"  {r['id']}  UNVERIFIABLE  ({r['context']})")
            continue
        flags = []
        if not r["reproducible"]:
            flags.append("not reproducible")
        if not r["order_invariant"]:
            flags.append("order flips")
        if not r["paraphrase_invariant"]:
            flags.append("paraphrase flips")
        if r.get("accuracy") and not r["accuracy"]["within_1_every_run"]:
            flags.append("disagrees with human")
        print(f"  {r['id']}  {'trusted  ' if r['trustworthy'] else 'UNTRUSTED'} verdict={r['verdict']:<7} "
              f"F {dist(r,'normal','faithfulness')}/{dist(r,'answer_first','faithfulness')}/{dist(r,'paraphrase','faithfulness')}  "
              f"R {dist(r,'normal','relevance')}/{dist(r,'answer_first','relevance')}/{dist(r,'paraphrase','relevance')}"
              + (f"  [{', '.join(flags)}]" if flags else ""))
    print(f"\n  trustworthy verdict rate: {s['n_trustworthy']}/{s['n_auditable']} auditable = "
          f"{s['trustworthy_verdict_rate']:.0%}  ({s['n_trustworthy']}/{s['n_questions']} = "
          f"{s['trustworthy_verdict_rate_all']:.0%} counting unverifiable as untrusted)")
    print("  consistency (all 5 verdicts agree) by condition: "
          + ", ".join(f"{c} {v:.0%}" for c, v in s["consistency_by_condition"].items()))
    print(f"  majority verdict differs from normal prompt: answer-first {s['answer_first_flip_rate']:.0%}, "
          f"paraphrase {s['paraphrase_flip_rate']:.0%}")
    print(f"  reproducible {s['reproducible_rate']:.0%}, order-invariant {s['order_invariant_rate']:.0%}, "
          f"paraphrase-invariant {s['paraphrase_invariant_rate']:.0%}")
    kt = s["kumar_T_adapted_holdout"]
    if kt["mean"] is not None:
        print(f"  adapted Kumar T on {kt['n']} human-graded questions: {kt['mean']:.2f} (max 1)")
    print(f"  untrustworthy: {s['untrustworthy'] or 'none'}   unverifiable: {s['unverifiable'] or 'none'}")
    print(f"  normal-prompt score differs from the one logged in the eval run: "
          f"{s['differs_from_logged_eval_score'] or 'none'}")
    full, tr = s["pass_rates_full"], s["pass_rates_trustworthy_only"]
    print(f"  {source_run}: pass^1 {full['pass1']:.0%} / pass^3 {full['pass3']:.0%} on all {full['n']} questions"
          f"  vs  pass^1 {tr['pass1']:.0%} / pass^3 {tr['pass3']:.0%} on the {tr['n']} trustworthy ones")
