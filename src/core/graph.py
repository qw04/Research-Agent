"""
Graph shape:

    retrieve -> grade -> [sufficient]   -> generate -> critic -> END
                       -> [insufficient, attempts < MAX] -> reformulate -> retrieve (loop)
                       -> [insufficient, attempts >= MAX] -> arxiv_search -> generate -> critic -> END

The critic node (src/core/critic.py) is a separate, role-based multi-agent
framework (CrewAI) rather than another LangGraph node's worth of logic —
it reviews the generated answer against the retrieved sources and, if it
finds an unsupported claim, replaces the answer with one revision pass.
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from generate import retrieve, format_context, llm
from critic import review_answer
from config import DEFAULT_TOPIC

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
    topic: str                # research topic the corpus/question is about
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
     "You rewrite search queries to improve retrieval from a corpus of research "
     "papers on {topic}. The previous query did not "
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
     "You are a research assistant answering questions about {topic}. "
     "Answer ONLY using the provided context — if the context doesn't "
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
        "topic": state["topic"],
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
    arxiv_results = search_arxiv(state["original_question"], topic=state["topic"], verbose=state["verbose"])
    return {**state, "retrieved": arxiv_results, "used_arxiv": True}


def generate_node(state: GraphState) -> GraphState:
    context = format_context(state["retrieved"])
    answer = answer_chain.invoke({
        "topic": state["topic"],
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
        topic=state["topic"],
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


def ask(
    question: str,
    topic: str | None = None,
    verbose: bool = False,
    top_k: int = TOP_K,
    use_critic: bool = True,
) -> dict:
    """Run the agent on `question`. `topic` is the research topic the corpus
    is about; it frames the reformulation, arXiv-keyword, answer and critic
    prompts. Falls back to DEFAULT_TOPIC when not given."""
    result = app.invoke({
        "topic": topic or DEFAULT_TOPIC,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the research agent a question.")
    parser.add_argument("question")
    parser.add_argument("--topic", default=None, help="Research topic the corpus is about.")
    args = parser.parse_args()

    result = ask(args.question, topic=args.topic, verbose=True)
    print(f"Q: {result['original_question']}")
    print(f"Attempts: {result['attempts']} | Used arXiv: {result['used_arxiv']}")
    print(f"Critic approved: {result['approved']}" + (f" | feedback: {result['critic_feedback']}" if not result["approved"] else ""))
    print(f"\nA: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s}")