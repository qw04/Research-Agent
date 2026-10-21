"""
FastAPI service wrapping the LangGraph agent (src/core/graph.py) over
HTTP. This is a thin transport layer only: no retrieval/generation/critic
logic lives here, all of it stays in core so the CLI (`python -m
src.core.graph`) and the API share the exact same behavior.

Run locally (from repo root, with the venv active):
    uvicorn src.api.main:app --reload --port 8000

Run in Docker: see Dockerfile at repo root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, so core.* / discovery.* / paper_graph.* resolve

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.graph import ask
from core.generate import chunks
from core.compare import compare_papers
from paper_graph.paper_relationships import paper_relationships

app = FastAPI(
    title="Research Paper Agent API",
    description="RAG + multi-agent Q&A over a corpus of research papers on a given topic.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    topic: str | None = None  # research topic the corpus is about; generic default if omitted
    top_k: int = Field(default=5, ge=1, le=20)
    use_critic: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    used_arxiv: bool
    attempts: int
    critic_approved: bool
    critic_feedback: str


class CompareRequest(BaseModel):
    paper_a: str
    paper_b: str
    question: str
    topic: str | None = None


class CompareResponse(BaseModel):
    paper_a: str
    paper_b: str
    answer: str


class PaperRelationships(BaseModel):
    paper: str
    techniques: list[str]
    novelty: str
    baselines: list[str]
    key_result: str
    compares_against_papers: list[str]
    shares_technique_with: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/papers")
def list_papers() -> list[str]:
    return sorted({c["source"] for c in chunks})


@app.post("/ask", response_model=AskResponse)
def ask_route(req: AskRequest) -> AskResponse:
    result = ask(req.question, topic=req.topic, top_k=req.top_k, use_critic=req.use_critic)
    return AskResponse(
        question=result["original_question"],
        answer=result["answer"],
        sources=result["sources"],
        used_arxiv=result["used_arxiv"],
        attempts=result["attempts"],
        critic_approved=result["approved"],
        critic_feedback=result["critic_feedback"],
    )


@app.post("/compare", response_model=CompareResponse)
def compare_route(req: CompareRequest) -> CompareResponse:
    try:
        result = compare_papers(req.paper_a, req.paper_b, req.question, topic=req.topic)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return CompareResponse(paper_a=result["paper_a"], paper_b=result["paper_b"], answer=result["answer"])


@app.get("/papers/{paper}/relationships", response_model=PaperRelationships)
def paper_relationships_route(paper: str) -> PaperRelationships:
    try:
        result = paper_relationships(paper)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PaperRelationships(**result)
