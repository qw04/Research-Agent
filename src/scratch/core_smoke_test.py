"""
Smoke test for src/core: exercises retrieval, the full ask() graph (local
answer, default topic, and the reformulate -> arXiv fallback path), the
critic, and compare_papers() against the real OpenAI and arXiv APIs.

Not a unit test suite — it makes live, billed LLM calls (a few cents per
run) and the arXiv check depends on arXiv being up. Prints PASS/FAIL per
check and a summary at the end.

Run from the repo root:
    python src/scratch/core_smoke_test.py
"""

import sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, so core.* / discovery.* / paper_graph.* resolve
T = "battery state-of-charge estimation"
results = {}
def check(name, fn):
    t = time.time()
    try:
        out = fn()
        results[name] = ("PASS", time.time() - t, out)
    except Exception as e:
        traceback.print_exc()
        results[name] = ("FAIL", time.time() - t, f"{type(e).__name__}: {e}"[:300])
    print(f"\n=== {name}: {results[name][0]} ({results[name][1]:.0f}s) {results[name][2]}\n", flush=True)

from core.generate import retrieve, format_context, chunks
from core.graph import ask
from core.critic import review_answer
from core.compare import compare_papers

def t_retrieve():
    r = retrieve("LSTM state of charge", k=3)
    assert len(r) == 3 and all({"score","source","text"} <= x.keys() for x in r)
    assert r[0]["score"] <= r[-1]["score"]
    ctx = format_context(r); assert ctx.count("[Source:") == 3
    return [x["source"][:50] for x in r]
check("retrieve + format_context", t_retrieve)

def t_ask_local():
    r = ask("How is an LSTM combined with a UKF to estimate SOC?", topic=T, verbose=True)
    assert r["answer"] and r["sources"] and r["topic"] == T
    return dict(attempts=r["attempts"], arxiv=r["used_arxiv"], approved=r["approved"], answer=r["answer"][:250])
check("ask: in-corpus question, critic on", t_ask_local)

def t_ask_default():
    r = ask("What datasets are used to train the models?", verbose=True, use_critic=False)
    assert r["topic"] == "academic research" and r["answer"]
    return dict(attempts=r["attempts"], arxiv=r["used_arxiv"], answer=r["answer"][:150])
check("ask: no topic (default), critic off", t_ask_default)

def t_ask_arxiv():
    r = ask("How do diffusion models generate images?", topic="generative image models", verbose=True, use_critic=False)
    return dict(attempts=r["attempts"], arxiv=r["used_arxiv"], sources=r["sources"], answer=r["answer"][:150])
check("ask: out-of-corpus question (reformulate -> arXiv path)", t_ask_arxiv)

def t_critic():
    ctx = "[Source: Paper X]\nSVM regression achieved an RMSE of 2.3% on the test set."
    bad = "SVM achieves 0% error [Source: Paper X]."
    r = review_answer("How accurate is SVM?", bad, ctx, topic=T)
    assert r["revised"] and not r["approved"], r
    return dict(approved=r["approved"], feedback=r["feedback"][:150], revised=r["answer"][:150])
check("critic catches fabricated claim", t_critic)

def t_compare():
    papers = sorted({c["source"] for c in chunks})
    r = compare_papers(papers[0], papers[1], "Which reports lower error?", topic=T)
    try:
        compare_papers(papers[0], "nonexistent paper", "x"); raise AssertionError("no ValueError")
    except ValueError: pass
    return r["answer"][:150]
check("compare_papers (+ bad title raises ValueError)", t_compare)

print("\n\n######## SUMMARY")
for k, (s, d, _) in results.items(): print(f"{s}  {d:5.0f}s  {k}")

sys.exit(0 if all(s == "PASS" for s, _, _ in results.values()) else 1)
