"""
Compare-two-papers tool — standalone, not part of the main graph loop in
graph.py. Call it directly when the user wants a side-by-side comparison of
two specific papers.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import DEFAULT_TOPIC
from core.generate import llm, chunks

compare_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You compare two research papers on {topic}. "
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


def compare_papers(paper_a: str, paper_b: str, question: str, topic: str | None = None) -> dict:
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
        "topic": topic or DEFAULT_TOPIC,
        "paper_a": paper_a,
        "context_a": context_a,
        "paper_b": paper_b,
        "context_b": context_b,
        "question": question,
    })
    return {"paper_a": paper_a, "paper_b": paper_b, "question": question, "answer": answer}
