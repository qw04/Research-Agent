"""
Live arXiv search, used by graph.py's arxiv_search node as the fallback when
local retrieval fails after MAX_LOCAL_ATTEMPTS reformulations.

arXiv's API does literal keyword matching, not semantic search, so the raw
question isn't passed through directly — an LLM first turns it into a few
specific search terms, which are AND-joined into a field-scoped query, with
an OR-joined retry if the strict query returns nothing.

Results are returned in the same {score, source, text} shape as
generate.retrieve() so the rest of the graph treats them identically.
"""

import arxiv
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

ARXIV_MAX_RESULTS = 3

# Query-building is a simple rewriting task — same cheap-model rationale as
# graph.py's utility_llm. Defined here rather than imported from graph.py to
# avoid a circular import (graph.py imports this module).
query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ArxivQuery(BaseModel):
    terms: list[str] = Field(
        description="2-4 specific keywords/phrases (not a full sentence) that capture "
                     "the core topic, e.g. ['graph neural network', 'molecular property prediction']. "
                     "Prefer specific technical terms over generic ones — arXiv's search "
                     "does literal keyword matching, so a broad term like 'machine learning' "
                     "alone will dominate and drown out the more specific terms."
    )


arxiv_query_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Convert the question into a small set of specific search terms suitable for arXiv's "
     "keyword-based search API (not semantic search — literal term matching). Favor precise, "
     "narrow technical terms over broad category words. Use the research topic to keep "
     "the terms on-domain when the question alone is ambiguous."),
    ("human", "Research topic: {topic}\nQuestion: {question}"),
])
arxiv_query_builder = arxiv_query_prompt | query_llm.with_structured_output(ArxivQuery)


def _run_search(client: arxiv.Client, query: str, max_results: int) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        {
            "score": 0.0,  # not comparable to FAISS distance, kept for shape consistency
            "source": f"arXiv - {result.title}",
            "text": result.summary,
        }
        for result in client.results(search)
    ]


def search_arxiv(
    question: str,
    topic: str = "academic research",
    max_results: int = ARXIV_MAX_RESULTS,
    verbose: bool = False,
) -> list[dict]:
    """Search arXiv for papers relevant to `question` within `topic`,
    returning abstracts in the same shape as generate.retrieve()."""
    query_terms: ArxivQuery = arxiv_query_builder.invoke({"topic": topic, "question": question})
    and_query = " AND ".join(f'all:"{term}"' for term in query_terms.terms)
    if verbose:
        print(f"[arxiv_search] terms={query_terms.terms} query={and_query!r}")

    client = arxiv.Client()
    results = _run_search(client, and_query, max_results)

    # AND-joining every term can be too strict and return nothing — fall back
    # to OR-joining if the strict query came up empty
    if not results:
        or_query = " OR ".join(f'all:"{term}"' for term in query_terms.terms)
        if verbose:
            print(f"[arxiv_search] AND query returned nothing, retrying OR: {or_query!r}")
        results = _run_search(client, or_query, max_results)

    if verbose:
        for r in results:
            print(f"    {r['source']}")

    return results
