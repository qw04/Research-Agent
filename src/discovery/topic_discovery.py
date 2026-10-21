"""
Topic-driven literature discovery — stage 1 of a new "research any topic"
mode, separate from the fixed-corpus SOC pipeline in graph.py.

Full intended pipeline (this module builds stage 1 only):

    topic -> [wikipedia, web search, arxiv surveys, semantic scholar]  (parallel)
           -> shortlist_intro_sources (LLM judgment, gpt-4o-mini)
           -> printed for you to review here in chat

Stage 2 (arXiv primary-paper discovery filtered down to the *influential*
few via an LLM shortlist + Semantic Scholar citation counts, the AlexNet-
for-CV idea) and stage 3 (ingesting your approved sources into a corpus for
the existing retrieve/generate/critic graph) are not built yet.

No LangGraph interrupt/checkpointer is used here on purpose — the shortlist
is the end of the graph for now; you review and approve sources in
conversation rather than the graph pausing mid-run.

Source types and why each one is here:
  - Wikipedia: hit the MediaWiki API directly (the `wikipedia` PyPI package
    is unmaintained and breaks against the current API — it sends no
    User-Agent and gets an HTML error page back instead of JSON). Pulls the
    page intro extract plus its External links section, since Wikipedia's
    own references are a natural bridge to better sources.
  - DuckDuckGo web search (via `ddgs`, no API key): broadest net, catches
    Medium articles, blog posts, course notes, tutorials in one query.
  - arXiv survey/review papers: literal keyword search (arXiv is not
    semantic) for "survey" or "review" alongside the topic terms.
  - Semantic Scholar: paper search with citation counts, used here as a
    lightweight "well-cited intro-adjacent paper" signal. The unauthenticated
    tier is aggressively rate-limited (shared global pool) — this node fails
    soft (returns an empty list with a warning) rather than blocking the
    whole pipeline. Get a free key at
    https://www.semanticscholar.org/product/api#api-key-form
    for reliable use, and to reuse in the later influential-paper stage.

Papers With Code was considered as a 4th source but its public REST API
(/api/v1/papers/, /methods/, /search/) now returns the HTML site shell
instead of JSON for all endpoints — it appears to have been retired —
so it was dropped in favor of Semantic Scholar.

Run: python src/discovery/topic_discovery.py "battery SOC modelling"
"""

import sys
from pathlib import Path
from typing import Literal, TypedDict

import arxiv
import requests
from ddgs import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from core.graph import utility_llm

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "research-agent-topic-discovery/0.1 (educational portfolio project; contact: shahsaumya2004@gmail.com)"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

WIKI_MAX_PAGES = 5
WIKI_MAX_EXTLINKS = 15
WEB_MAX_RESULTS = 8
SURVEY_MAX_RESULTS = 5
S2_MAX_RESULTS = 5
SHORTLIST_SIZE = "8-12"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TopicState(TypedDict):
    topic: str
    wiki_sources: list[dict]
    web_sources: list[dict]
    survey_sources: list[dict]
    s2_sources: list[dict]
    shortlist: list[dict]
    verbose: bool


# ---------------------------------------------------------------------------
# Search nodes (run in parallel — each writes a distinct state key)
# ---------------------------------------------------------------------------

def wiki_search_node(state: TopicState) -> TopicState:
    topic = state["topic"]
    headers = {"User-Agent": USER_AGENT}

    search_resp = requests.get(WIKI_API, params={
        "action": "query", "list": "search", "srsearch": topic,
        "format": "json", "srlimit": WIKI_MAX_PAGES,
    }, headers=headers, timeout=10)
    titles = [r["title"] for r in search_resp.json().get("query", {}).get("search", [])]

    sources = []
    for title in titles:
        resp = requests.get(WIKI_API, params={
            "action": "query", "titles": title,
            "prop": "extracts|extlinks", "exintro": True, "explaintext": True,
            "ellimit": WIKI_MAX_EXTLINKS, "format": "json",
        }, headers=headers, timeout=10)
        page = next(iter(resp.json()["query"]["pages"].values()))
        if "extract" not in page:
            continue
        sources.append({
            "title": page["title"],
            "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            "summary": page["extract"][:600],
            "references": [l["*"] for l in page.get("extlinks", [])],
        })

    if state["verbose"]:
        print(f"[wiki_search] {len(sources)} pages for {topic!r}")
    return {"wiki_sources": sources}


def web_search_node(state: TopicState) -> TopicState:
    topic = state["topic"]
    query = f"{topic} introduction explained"
    try:
        raw = list(DDGS().text(query, max_results=WEB_MAX_RESULTS))
    except Exception as e:
        if state["verbose"]:
            print(f"[web_search] failed: {e}")
        return {"web_sources": []}

    sources = [
        {"title": r["title"], "url": r["href"], "summary": r["body"][:400]}
        for r in raw
    ]
    if state["verbose"]:
        print(f"[web_search] {len(sources)} results for {query!r}")
    return {"web_sources": sources}


def survey_search_node(state: TopicState) -> TopicState:
    # arXiv search is literal keyword matching, not semantic — quoting the
    # whole multi-word topic as one phrase almost never appears verbatim in
    # an abstract and returns nothing (same failure mode handled in
    # arxiv_search.py). AND-join individual topic words instead, and fall
    # back to OR-joining if the strict query comes up empty.
    topic = state["topic"]
    terms = topic.split()
    client = arxiv.Client()

    def _search(term_join: str) -> list[dict]:
        term_query = f" {term_join} ".join(f'all:"{t}"' for t in terms)
        query = f'(all:"survey" OR all:"review") AND ({term_query})'
        search = arxiv.Search(query=query, max_results=SURVEY_MAX_RESULTS, sort_by=arxiv.SortCriterion.Relevance)
        return [
            {"title": r.title, "url": r.entry_id, "summary": r.summary[:500]}
            for r in client.results(search)
        ]

    sources = _search("AND")
    if not sources:
        sources = _search("OR")

    if state["verbose"]:
        print(f"[survey_search] {len(sources)} arXiv surveys for {topic!r}")
    return {"survey_sources": sources}


def s2_search_node(state: TopicState) -> TopicState:
    topic = state["topic"]
    try:
        resp = requests.get(S2_API, params={
            "query": topic,
            "fields": "title,url,abstract,citationCount,year",
            "limit": S2_MAX_RESULTS,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        if state["verbose"]:
            print(f"[s2_search] skipped (rate-limited or unavailable): {e}")
        return {"s2_sources": []}

    sources = [
        {
            "title": p["title"],
            "url": p.get("url") or "",
            "summary": (p.get("abstract") or "")[:500],
            "citation_count": p.get("citationCount", 0),
            "year": p.get("year"),
        }
        for p in data.get("data", [])
        if p.get("title")
    ]
    if state["verbose"]:
        print(f"[s2_search] {len(sources)} papers for {topic!r}")
    return {"s2_sources": sources}


# ---------------------------------------------------------------------------
# Shortlist node — LLM judgment over all candidates
# ---------------------------------------------------------------------------

class IntroSourcePick(BaseModel):
    title: str
    url: str
    source_type: Literal["wikipedia", "wikipedia_reference", "web", "arxiv_survey", "semantic_scholar"]
    reason: str = Field(description="One sentence: why this is a good introductory source for someone starting this topic.")


class IntroShortlist(BaseModel):
    sources: list[IntroSourcePick] = Field(
        description=f"{SHORTLIST_SIZE} non-redundant introductory sources, ordered from "
                     "broadest/most foundational to most specific."
    )


shortlist_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are curating an introductory reading list for someone about to research a topic "
     "in depth, before they move on to primary research literature (arXiv papers). From the "
     "candidate sources below, select the best {shortlist_size} non-redundant sources. "
     "Prioritize sources that build genuine understanding — foundational explainers, "
     "well-regarded surveys, reputable technical articles — over shallow, off-topic, or "
     "duplicate content. Drop anything that's clearly irrelevant to the topic. Order the "
     "result from broadest/most foundational to most specific."),
    ("human", "Topic: {topic}\n\nCandidate sources:\n{candidates}"),
])
shortlister = shortlist_prompt | utility_llm.with_structured_output(IntroShortlist)


def _format_candidates(state: TopicState) -> str:
    lines = []
    for s in state["wiki_sources"]:
        lines.append(f"[wikipedia] {s['title']} — {s['url']}\n  {s['summary'][:200]}")
        for ref in s["references"][:8]:
            lines.append(f"[wikipedia_reference] {ref}")
    for s in state["web_sources"]:
        lines.append(f"[web] {s['title']} — {s['url']}\n  {s['summary'][:200]}")
    for s in state["survey_sources"]:
        lines.append(f"[arxiv_survey] {s['title']} — {s['url']}\n  {s['summary'][:200]}")
    for s in state["s2_sources"]:
        lines.append(f"[semantic_scholar] {s['title']} ({s.get('citation_count', 0)} citations) — {s['url']}\n  {s['summary'][:200]}")
    return "\n".join(lines)


def shortlist_node(state: TopicState) -> TopicState:
    candidates = _format_candidates(state)
    result: IntroShortlist = shortlister.invoke({
        "topic": state["topic"],
        "candidates": candidates,
        "shortlist_size": SHORTLIST_SIZE,
    })
    picks = [p.model_dump() for p in result.sources]
    if state["verbose"]:
        print(f"[shortlist] {len(picks)} sources selected")
    return {**state, "shortlist": picks}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(TopicState)

    graph.add_node("wiki_search", wiki_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("survey_search", survey_search_node)
    graph.add_node("s2_search", s2_search_node)
    graph.add_node("shortlist", shortlist_node)

    for n in ("wiki_search", "web_search", "survey_search", "s2_search"):
        graph.add_edge(START, n)
        graph.add_edge(n, "shortlist")
    graph.add_edge("shortlist", END)

    return graph.compile()


app = build_graph()


def discover(topic: str, verbose: bool = True) -> dict:
    return app.invoke({
        "topic": topic,
        "wiki_sources": [],
        "web_sources": [],
        "survey_sources": [],
        "s2_sources": [],
        "shortlist": [],
        "verbose": verbose,
    })


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "battery state of charge modelling"
    result = discover(topic)
    print(f"\nProposed introductory sources for {topic!r}:\n")
    for i, s in enumerate(result["shortlist"], 1):
        print(f"  [{i}] ({s['source_type']}) {s['title']}")
        print(f"      {s['url']}")
        print(f"      {s['reason']}\n")
