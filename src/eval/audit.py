"""
Judge audit: per-item verdict reliability (data/judge_audit.json).

pass^k repeats the AGENT but judges each answer once. Kumar, Rathore & Moitra,
"All Verdicts are Not Equal: Rethinking LLM Judge Reliability"
(arXiv:2610.12083, AACL-IJCNLP 2026) report that judge verdicts can change
across identical reruns even at temperature 0, that presentation order can
flip verdicts, and that reliability is item-specific rather than a property
of the judge model. Their "trustworthy verdict rate" is the share of verdicts
that are reproducible, order-invariant and accurate. (The paper is about
pairwise judging; this is a pointwise adaptation.)

For each question, attempt 1's stored answer from AUDIT_SOURCE_RUN is judged
AUDIT_REPS times with the normal prompt and AUDIT_REPS times with an
answer-first variant (identical wording, sections reordered). Verdict =
success (5/5 on both). A question is trustworthy iff:
  reproducible:     all normal-prompt verdicts agree
  order-invariant:  all answer-first verdicts agree with each other and with
                    the normal prompt's majority verdict
  accurate:         (only for questions in data/judge_validation.json) every
                    normal-prompt judgement of the HUMAN-GRADED answer is
                    within +/-1 of the human score on both dimensions

Context: the audit needs the excerpts the answer was built from. Eval logs
store them ("retrieved" per attempt) since the audit was added. For older
logs, the context is rebuilt with retrieve(question, k=top_k), which is exact
when the agent did not reformulate (attempts == 0, no arXiv); questions that
went through the arXiv fallback can't be rebuilt and are reported as
"unverifiable" instead of being judged against the wrong evidence.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import mlflow

from core.generate import retrieve
from eval.judge import (JUDGE_MODEL, JUDGE_VERSION, judge_chain,
                        judge_chain_answer_first, judge_once)
from eval.settings import (AUDIT_FILE, MLFLOW_EXPERIMENT, N_ATTEMPTS, PASS_SCORE,
                           RUNS_DIR, attempt_succeeded, load_eval_set)
from eval.validation import load_graded_items

AUDIT_SOURCE_RUN = "topk5_critic_on_x3"
AUDIT_SOURCE_TOP_K = 5
AUDIT_REPS = 5


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


def _pass_rates(rows: list[dict]) -> dict:
    if not rows:
        return {"pass1": None, "pass3": None, "n": 0}
    return {"pass1": sum(q["n_success"] for q in rows) / (len(rows) * N_ATTEMPTS),
            "pass3": sum(q["pass3"] for q in rows) / len(rows), "n": len(rows)}


def judge_audit() -> dict:
    run = json.loads((RUNS_DIR / f"{AUDIT_SOURCE_RUN}.json").read_text(encoding="utf-8"))
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
            jobs.append((q["id"], "normal", judge_chain, a["answer"], chunks))
            jobs.append((q["id"], "answer_first", judge_chain_answer_first, a["answer"], chunks))
        if q["id"] in validation:   # accuracy: judge the answer the human actually graded
            v = validation[q["id"]]
            for _ in range(AUDIT_REPS):
                jobs.append((q["id"], "human_graded_answer", judge_chain, v["agent_answer"],
                             [{"text": t} for t in v["retrieved_context"]]))

    print(f"Judge audit: {len(jobs)} judge calls over {sum(i['chunks'] is not None for i in items.values())} "
          f"auditable questions (source run {AUDIT_SOURCE_RUN}, attempt 1)...")
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
        normal, flipped = scores[qid]["normal"], scores[qid]["answer_first"]
        v_normal = [attempt_succeeded(f, r) for f, r in normal]
        v_flipped = [attempt_succeeded(f, r) for f, r in flipped]
        reproducible = len(set(v_normal)) == 1
        order_invariant = len(set(v_flipped)) == 1 and _majority(v_flipped) == _majority(v_normal)
        all10 = normal + flipped
        row.update({
            "auditable": True,
            "normal": {"scores": normal, "distribution": _distribution(normal)},
            "answer_first": {"scores": flipped, "distribution": _distribution(flipped)},
            "verdict": ("success" if v_normal[0] else "fail") if reproducible else "mixed",
            "verdict_identical_all_10": len(set(v_normal + v_flipped)) == 1,
            "faithfulness_spread_gt1": _spread(all10, 0) > 1,
            "relevance_spread_gt1": _spread(all10, 1) > 1,
            "reproducible": reproducible,
            "order_invariant": order_invariant,
            # cross-session check (informational, not part of "trustworthy"):
            # does today's judgement match the score logged during the eval run?
            "matches_logged_eval_score": all(s == tuple(it["logged_score"]) for s in normal),
        })
        accurate = None
        if qid in validation:
            h = validation[qid]["human_scores"]
            hg = scores[qid]["human_graded_answer"]
            accurate = all(abs(f - h["faithfulness"]) <= 1 and abs(r - h["relevance"]) <= 1 for f, r in hg)
            row["accuracy"] = {"human": h, "judge_on_human_graded_answer": hg, "within_1_every_run": accurate}
        row["trustworthy"] = reproducible and order_invariant and accurate is not False
        per_question.append(row)

    audited = [r for r in per_question if r["auditable"]]
    trusted_ids = {r["id"] for r in per_question if r["trustworthy"]}
    n_trust = len(trusted_ids)

    summary = {
        "n_questions": len(per_question),
        "n_auditable": len(audited),
        "n_trustworthy": n_trust,
        "trustworthy_verdict_rate": n_trust / len(audited),
        "trustworthy_verdict_rate_all": n_trust / len(per_question),
        "reproducible_rate": sum(r["reproducible"] for r in audited) / len(audited),
        "order_invariant_rate": sum(r["order_invariant"] for r in audited) / len(audited),
        "untrustworthy": [r["id"] for r in audited if not r["trustworthy"]],
        "unverifiable": [r["id"] for r in per_question if not r["auditable"]],
        "score_spread_gt1": [r["id"] for r in audited if r["faithfulness_spread_gt1"] or r["relevance_spread_gt1"]],
        "differs_from_logged_eval_score": [r["id"] for r in audited if not r["matches_logged_eval_score"]],
        "pass_rates_full": _pass_rates(run),
        "pass_rates_trustworthy_only": _pass_rates([q for q in run if q["id"] in trusted_ids]),
    }

    AUDIT_FILE.write_text(json.dumps({
        "purpose": "Per-item judge reliability (Kumar, Rathore & Moitra, arXiv:2610.12083).",
        "judge_model": JUDGE_MODEL, "judge_version": JUDGE_VERSION, "reps_per_condition": AUDIT_REPS,
        "source_run": AUDIT_SOURCE_RUN, "source_attempt": 1,
        "verdict_definition": f"success = faithfulness >= {PASS_SCORE} and relevance >= {PASS_SCORE}",
        "summary": summary, "questions": per_question,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_report(per_question, summary)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    tr = summary["pass_rates_trustworthy_only"]
    with mlflow.start_run(run_name=f"judge_audit_{AUDIT_SOURCE_RUN}"):
        mlflow.log_params({"judge_version": JUDGE_VERSION, "judge_model": JUDGE_MODEL,
                           "reps_per_condition": AUDIT_REPS, "source_run": AUDIT_SOURCE_RUN})
        mlflow.log_metrics({
            "trustworthy_verdict_rate": summary["trustworthy_verdict_rate"],
            "trustworthy_verdict_rate_all": summary["trustworthy_verdict_rate_all"],
            "reproducible_rate": summary["reproducible_rate"],
            "order_invariant_rate": summary["order_invariant_rate"],
            "pass1_rate_trustworthy": tr["pass1"] or 0.0,
            "pass3_rate_trustworthy": tr["pass3"] or 0.0,
        })
        mlflow.log_artifact(str(AUDIT_FILE))
    return summary


def _print_report(per_question: list[dict], s: dict) -> None:
    print(f"\nJudge audit ({AUDIT_REPS} normal + {AUDIT_REPS} answer-first judgements per answer):")
    for r in per_question:
        if not r["auditable"]:
            print(f"  {r['id']}  UNVERIFIABLE  ({r['context']})")
            continue
        flags = []
        if not r["reproducible"]:
            flags.append("not reproducible")
        elif not r["order_invariant"]:
            flags.append("order flips verdict")
        if r.get("accuracy") and not r["accuracy"]["within_1_every_run"]:
            flags.append("disagrees with human")
        if r["faithfulness_spread_gt1"] or r["relevance_spread_gt1"]:
            flags.append("score spread >1")
        print(f"  {r['id']}  {'trusted  ' if r['trustworthy'] else 'UNTRUSTED'} verdict={r['verdict']:<7} "
              f"F normal={r['normal']['distribution']['faithfulness']} first={r['answer_first']['distribution']['faithfulness']}  "
              f"R normal={r['normal']['distribution']['relevance']} first={r['answer_first']['distribution']['relevance']}"
              + (f"  [{', '.join(flags)}]" if flags else ""))
    print(f"\n  trustworthy verdict rate: {s['n_trustworthy']}/{s['n_auditable']} auditable = "
          f"{s['trustworthy_verdict_rate']:.0%}  ({s['n_trustworthy']}/{s['n_questions']} = "
          f"{s['trustworthy_verdict_rate_all']:.0%} counting unverifiable as untrusted)")
    print(f"  reproducible: {s['reproducible_rate']:.0%}   order-invariant: {s['order_invariant_rate']:.0%}")
    print(f"  untrustworthy: {s['untrustworthy'] or 'none'}   unverifiable: {s['unverifiable'] or 'none'}")
    print(f"  differs from the score logged in the original eval run (cross-session): "
          f"{s['differs_from_logged_eval_score'] or 'none'}")
    full, tr = s["pass_rates_full"], s["pass_rates_trustworthy_only"]
    print(f"  {AUDIT_SOURCE_RUN}: pass^1 {full['pass1']:.0%} / pass^3 {full['pass3']:.0%} on all {full['n']} questions"
          f"  vs  pass^1 {tr['pass1']:.0%} / pass^3 {tr['pass3']:.0%} on the {tr['n']} trustworthy ones")
