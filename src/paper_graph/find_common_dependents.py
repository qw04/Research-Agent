"""
Corpus-gap analysis: finds baseline techniques that multiple papers in this
corpus compare themselves against ("compares_against_technique" edges in
data/paper_graph.json) but that aren't backed by a dedicated paper in the
corpus, then searches arXiv for candidate papers that could fill that gap.

This is a discovery/reporting tool, not an auto-ingestion pipeline —
actually expanding the corpus (re-running src/core/ingest.py and
src/core/embed_store.py on any paper found here) is a separate, deliberate
decision this script does not make on its own.

Reuses the same arxiv.Client()/arxiv.Search() pattern as
src/discovery/arxiv_search.py, including its own AND-then-OR fallback when a strict
query returns nothing (arXiv search is literal keyword matching, not
semantic).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import arxiv

GRAPH_FILE = Path("data/paper_graph.json")
OUTPUT_FILE = Path("data/common_dependents.json")
MIN_IN_DEGREE = 2  # "common" = a baseline for at least this many corpus papers
MAX_CANDIDATES = 3
# arXiv's AND matching is literal keyword matching, not semantic — a long
# quoted topic phrase almost never appears verbatim in an abstract, so it
# returns nothing. A single broad domain term (e.g. "battery" for a battery
# SOC corpus) actually narrows to on-domain results. Passed on the command
# line; if omitted, searches use the technique alone.

# arxiv.Client() rate-limits itself (default delay_seconds=3.0 between
# requests, per arXiv's API terms), so no manual sleep is needed here.
client = arxiv.Client()


def load_graph() -> dict:
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


def find_gaps(graph: dict) -> list[dict]:
    dependents = Counter()
    depending_papers: dict[str, list[str]] = {}
    for e in graph["edges"]:
        if e["type"] == "compares_against_technique":
            dependents[e["to"]] += 1
            depending_papers.setdefault(e["to"], []).append(e["from"])

    gaps = [
        {"technique": tech, "in_degree": n, "depended_on_by": depending_papers[tech]}
        for tech, n in dependents.items()
        if n >= MIN_IN_DEGREE
    ]
    gaps.sort(key=lambda g: -g["in_degree"])
    return gaps


def _format_results(results) -> list[dict]:
    return [
        {
            "title": r.title,
            "arxiv_id": r.get_short_id(),
            "url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d"),
            "summary": r.summary.replace("\n", " ")[:300],
        }
        for r in results
    ]


def search_arxiv(technique: str, domain: str | None = None) -> list[dict]:
    results = []
    if domain:
        query = f'all:"{technique}" AND all:"{domain}"'
        search = arxiv.Search(query=query, max_results=MAX_CANDIDATES, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))

    if not results:
        # no domain given, or the domain-scoped query came up empty — use the technique alone
        search = arxiv.Search(query=f'all:"{technique}"', max_results=MAX_CANDIDATES,
                               sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))

    return _format_results(results)


if __name__ == "__main__":
    # Optional single broad domain term, e.g. `python src/paper_graph/find_common_dependents.py battery`
    domain = sys.argv[1] if len(sys.argv) > 1 else None

    graph = load_graph()
    gaps = find_gaps(graph)

    print(f"Found {len(gaps)} technique(s) used as a baseline by >= {MIN_IN_DEGREE} "
          f"corpus papers, with no dedicated paper of their own in the corpus:\n")

    report = []
    for gap in gaps:
        print(f"- {gap['technique']} (baseline for {gap['in_degree']} papers)")
        candidates = search_arxiv(gap["technique"], domain)
        for c in candidates:
            print(f"    candidate: {c['title']} ({c['arxiv_id']}, {c['published']})")
        if not candidates:
            print("    (no arXiv results found)")
        report.append({**gap, "candidate_papers": candidates})

    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_FILE}")
