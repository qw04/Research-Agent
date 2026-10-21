"""
Graph shape:

    retrieve -> grade -> [sufficient]   -> generate -> critic -> END
                       -> [insufficient, attempts < MAX] -> reformulate -> retrieve (loop)
                       -> [insufficient, attempts >= MAX] -> arxiv_search -> generate -> critic -> END

The critic node (src/core/critic.py) is a separate, role-based multi-agent
framework (CrewAI) rather than another LangGraph node's worth of logic —
it reviews the generated answer against the retrieved sources and, if it
finds an unsupported claim, replaces the answer with one revision pass.

Also exposes compare_papers(paper_a, paper_b, question) as a standalone tool,
not part of the main graph loop — call it directly when the user wants a
side-by-side comparison of two specific papers.
"""

import json
import sys
from pathlib import Path
from typing import TypedDict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from generate import retrieve, format_context, llm, chunks
from critic import review_answer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "discovery"))
from discovery.arxiv_search import search_arxiv

MAX_LOCAL_ATTEMPTS = 2  # how many times to reformulate + re-search locally before falling back to arXiv
TOP_K = 5

# Grading and reformulation are simple classification/rewriting tasks —
# a cheaper model handles them fine and cuts cost, since
# these nodes can fire multiple times per query in the reformulation loop.
# `llm` (imported from generate.py, gpt-4o) is reserved for final answer
# generation, where quality matters most.
utility_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Grading, reformulation, and arXiv query-building are simpler classification/
# rewriting tasks that don't need the full model — and they run multiple times
# per query inside the reformulation loop, so a cheaper model here meaningfully
# cuts cost. Final answer generation still uses the stronger `llm` from generate.py.
control_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    original_question: str
    question: str            # current (possibly reformulated) query
    retrieved: list[dict]     # chunks from local FAISS or arXiv, same shape
    grade: str                # "sufficient" | "insufficient"
    attempts: int
    used_arxiv: bool
    answer: str
    sources: list[str]
    approved: bool
    critic_feedback: str
    top_k: int
    use_critic: bool
    verbose: bool


# ---------------------------------------------------------------------------
# Structured outputs for LLM grading / reformulation
# ---------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved chunks contain enough information "
                     "to actually answer the question, not just topically related."
    )
    reasoning: str = Field(description="One sentence explaining the judgment.")


class ReformulatedQuery(BaseModel):
    query: str = Field(description="A rewritten search query, more likely to retrieve relevant chunks.")


grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether retrieved document chunks are sufficient to answer a question. "
     "Be strict: topically related is not the same as sufficient. The chunks must contain "
     "enough specific information to actually construct an answer."),
    ("human",
     "Question: {question}\n\nRetrieved chunks:\n{context}\n\nGrade these chunks."),
])
grader = grade_prompt | utility_llm.with_structured_output(RelevanceGrade)

reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries to improve retrieval from a corpus of battery "
     "state-of-charge (SOC) estimation research papers. The previous query did not "
     "retrieve sufficient results. Rewrite it to use different terminology, be more "
     "specific, or approach the topic from a different angle — while preserving the "
     "original intent."),
    ("human",
     "Original question: {original_question}\nPrevious query: {question}\n\n"
     "Write a better search query."),
])
reformulator = reformulate_prompt | utility_llm.with_structured_output(ReformulatedQuery)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant answering questions about battery state-of-charge "
     "(SOC) estimation. Answer ONLY using the provided context — if the context doesn't "
     "contain the answer, say so explicitly rather than guessing.\n\n"
     "Every claim must be followed by a citation in the form [Source: <paper title>] for "
     "local papers, or [Source: arXiv - <paper title>] for arXiv results. Do not fabricate "
     "citations for anything not in the context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])
answer_chain = answer_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    results = retrieve(state["question"], k=state["top_k"])
    if state["verbose"]:
        print(f"[retrieve] query={state['question']!r}")
        for r in results:
            print(f"    score={r['score']:.3f}  {r['source']}")
    return {**state, "retrieved": results}


def grade_node(state: GraphState) -> GraphState:
    context = format_context(state["retrieved"])
    result: RelevanceGrade = grader.invoke({
        "question": state["original_question"],
        "context": context,
    })
    grade = "sufficient" if result.sufficient else "insufficient"
    if state["verbose"]:
        print(f"[grade] {grade} — {result.reasoning}")
    return {**state, "grade": grade}


def route_after_grade(state: GraphState) -> Literal["generate", "reformulate", "arxiv_search"]:
    if state["grade"] == "sufficient":
        return "generate"
    if state["attempts"] < MAX_LOCAL_ATTEMPTS:
        return "reformulate"
    return "arxiv_search"


def reformulate_node(state: GraphState) -> GraphState:
    result: ReformulatedQuery = reformulator.invoke({
        "original_question": state["original_question"],
        "question": state["question"],
    })
    if state["verbose"]:
        print(f"[reformulate] {state['question']!r} -> {result.query!r}")
    return {**state, "question": result.query, "attempts": state["attempts"] + 1}


def arxiv_search_node(state: GraphState) -> GraphState:
    """Local retrieval failed after MAX_LOCAL_ATTEMPTS reformulations —
    fall back to live arXiv search (src/discovery/arxiv_search.py) for
    papers the local corpus doesn't cover."""
    arxiv_results = search_arxiv(state["original_question"], verbose=state["verbose"])
    return {**state, "retrieved": arxiv_results, "used_arxiv": True}


def generate_node(state: GraphState) -> GraphState:
    context = format_context(state["retrieved"])
    answer = answer_chain.invoke({
        "context": context,
        "question": state["original_question"],
    })
    sources = sorted(set(r["source"] for r in state["retrieved"]))
    return {**state, "answer": answer, "sources": sources}


def critic_node(state: GraphState) -> GraphState:
    if not state["use_critic"]:
        return {**state, "approved": True, "critic_feedback": ""}

    context = format_context(state["retrieved"])
    result = review_answer(
        question=state["original_question"],
        answer=state["answer"],
        context=context,
        verbose=state["verbose"],
    )
    return {
        **state,
        "answer": result["answer"],
        "approved": result["approved"],
        "critic_feedback": result["feedback"],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("arxiv_search", arxiv_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "reformulate": "reformulate",
        "arxiv_search": "arxiv_search",
    })
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("arxiv_search", "generate")
    graph.add_edge("generate", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


app = build_graph()


def ask(question: str, verbose: bool = False, top_k: int = TOP_K, use_critic: bool = True) -> dict:
    result = app.invoke({
        "original_question": question,
        "question": question,
        "retrieved": [],
        "grade": "",
        "attempts": 0,
        "used_arxiv": False,
        "answer": "",
        "sources": [],
        "approved": False,
        "critic_feedback": "",
        "top_k": top_k,
        "use_critic": use_critic,
        "verbose": verbose,
    })
    return result


# ---------------------------------------------------------------------------
# Compare-two-papers tool (standalone, not part of the main graph)
# ---------------------------------------------------------------------------

compare_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You compare two research papers on battery state-of-charge estimation. "
     "Answer the user's question by contrasting the two papers directly — their "
     "methods, assumptions, and results. Cite claims as [Source: <paper title>]. "
     "If one paper doesn't address something the other does, say so explicitly."),
    ("human",
     "Paper A: {paper_a}\nPaper A excerpts:\n{context_a}\n\n"
     "Paper B: {paper_b}\nPaper B excerpts:\n{context_b}\n\n"
     "Question: {question}\n\nAnswer:"),
])
compare_chain = compare_prompt | llm | StrOutputParser()


def _chunks_for_source(source: str, max_chunks: int = 6) -> str:
    matches = [c for c in chunks if c["source"] == source][:max_chunks]
    return "\n\n---\n\n".join(c["text"] for c in matches)


def compare_papers(paper_a: str, paper_b: str, question: str) -> dict:
    """Compare two papers by source title (must match the 'source' field in
    metadata.json exactly). Pulls all available chunks for each paper up to
    a cap, rather than doing a similarity search, since we want full coverage
    of both specific papers rather than the top-k most similar to the query."""
    context_a = _chunks_for_source(paper_a)
    context_b = _chunks_for_source(paper_b)

    if not context_a or not context_b:
        missing = paper_a if not context_a else paper_b
        raise ValueError(f"No chunks found for source: {missing!r}. Check the exact title.")

    answer = compare_chain.invoke({
        "paper_a": paper_a,
        "context_a": context_a,
        "paper_b": paper_b,
        "context_b": context_b,
        "question": question,
    })
    return {"paper_a": paper_a, "paper_b": paper_b, "question": question, "answer": answer}


# ---------------------------------------------------------------------------
# Paper-relationship tool (standalone, not part of the main graph loop)
# ---------------------------------------------------------------------------
# Reads the precomputed dependency/technique graph built offline by
# src/paper_graph/build_paper_graph.py (data/paper_graph.json) — no live LLM extraction
# happens here, so this tool is cheap and deterministic to call at query time.

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


if __name__ == "__main__":
    result = ask("What is State of Health for a Battery?", verbose=True)
    print(f"Q: {result['original_question']}")
    print(f"Attempts: {result['attempts']} | Used arXiv: {result['used_arxiv']}")
    print(f"Critic approved: {result['approved']}" + (f" | feedback: {result['critic_feedback']}" if not result["approved"] else ""))
    print(f"\nA: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s}")