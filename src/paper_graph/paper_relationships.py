"""
Paper-relationship tool — standalone, not part of the main graph loop in
src/core/graph.py.

Reads the precomputed dependency/technique graph built offline by
src/paper_graph/build_paper_graph.py (data/paper_graph.json) — no live LLM
extraction happens here, so this tool is cheap and deterministic to call at
query time.
"""

import json
from pathlib import Path

PAPER_GRAPH_FILE = Path("data/paper_graph.json")


def _load_paper_graph() -> dict:
    if not PAPER_GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"{PAPER_GRAPH_FILE} not found — run `python src/paper_graph/build_paper_graph.py` first."
        )
    return json.loads(PAPER_GRAPH_FILE.read_text(encoding="utf-8"))


def paper_relationships(paper: str) -> dict:
    """Look up what's novel about a paper, what techniques it uses, what
    baselines it compares itself against, and which other papers in this
    corpus it relates to — either because it explicitly benchmarks against
    another corpus paper's method, or because the two papers share a
    technique. Source must match the 'source' field in metadata.json exactly."""
    graph = _load_paper_graph()
    if paper not in graph["papers"]:
        raise ValueError(f"No profile found for source: {paper!r}. Check the exact title.")

    profile = graph["papers"][paper]

    compares_against_papers = sorted({
        e["to"] for e in graph["edges"]
        if e["from"] == paper and e["type"] == "compares_against_paper"
    } | {
        e["from"] for e in graph["edges"]
        if e["to"] == paper and e["type"] == "compares_against_paper"
    })

    shares_technique_with = sorted({
        other for other, other_profile in graph["papers"].items()
        if other != paper and set(other_profile["techniques"]) & set(profile["techniques"])
    })

    return {
        "paper": paper,
        "techniques": profile["techniques"],
        "novelty": profile["novelty"],
        "baselines": profile["baselines"],
        "key_result": profile["key_result"],
        "compares_against_papers": compares_against_papers,
        "shares_technique_with": shares_technique_with,
    }
